import ReconnectingWebSocket from 'reconnecting-websocket'
import Player from './player'
import { vec3 } from 'gl-matrix'
import jwt_decode from 'jwt-decode'
import Controller from './controller'
import Density from './density'
//import ProtoSerializer from './network/proto-serializer'

const apiUrl = 'wss://api.new-world.james-parker.dev'
// const apiUrl = 'ws://localhost:3000'

interface State {
  position: { x: number; y: number; z: number }
}

export default class Network {
  private constructor(
    //private readonly serializer: ProtoSerializer,
    private readonly clientId: string,
    private readonly socket: ReconnectingWebSocket,
    private readonly players: Map<string, Player>,
    private readonly createPlayer: (string) => Player,
    private readonly density: Density,
    private readonly device: GPUDevice
  ) {}

  static async init(
    controller: Controller,
    players: Map<string, Player>,
    createPlayer: (string) => Player,
    density: Density,
    device: GPUDevice
  ): Promise<Network> {
    const socket = new ReconnectingWebSocket(`${apiUrl}/client`)

    //const serializer = await ProtoSerializer.init()

    return new Promise((resolve) => {
      socket.onmessage = (e: MessageEvent): void => {
        const message = JSON.parse(e.data)
        if (message.type === 'connected') {
          if (message.state) {
            const decoded = jwt_decode(message.state) as State
            vec3.set(
              controller.position,
              decoded.position.x,
              decoded.position.y,
              decoded.position.z
            )
          }

          const network = new Network(
            //serializer,
            message.clientId,
            socket,
            players,
            createPlayer,
            density,
            device
          )
          socket.onmessage = network.processMessage.bind(network)
          resolve(network)
        }
      }
    })
  }

  async sendData(data: object): Promise<void> {
    //this.socket.send(this.serializer.serialize({ ...data, clientId: this.clientId }))
    this.socket.send(JSON.stringify({ ...data, clientId: this.clientId }))
  }

  private async processMessage(e: MessageEvent): Promise<void> {
    const message = JSON.parse(e.data)

    if (message.state) {
      const decoded = jwt_decode(message.state)
      console.log(decoded)
    }

    switch (message.type) {
      case 'client_connected':
        if (message.clientId === this.clientId) {
          break
        }
        this.createPlayer(message.clientId)
        break
      case 'position': {
        if (message.clientId === this.clientId) {
          break
        }

        let player = this.players[message.clientId]
        if (!player) {
          player = this.createPlayer(message.clientId)
        }
        vec3.set(player.position, message.position.x, message.position.y, message.position.z)
        break
      }
      case 'build': {
        const data = JSON.parse(message.data)
        this.density.modify(this.device, {
          x: data.position.x,
          y: data.position.y,
          z: data.position.z,
          size: data.size,
          type: data.tool,
          shape: data.shape,
          material: data.material
        })
      }
    }
  }
}
