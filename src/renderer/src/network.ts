import ReconnectingWebSocket from 'reconnecting-websocket'
import Player from './player'
import { vec3 } from 'gl-matrix'

const apiUrl = 'wss://api.new-world.james-parker.dev'
//let apiUrl = 'ws://localhost:3000';

export default class Network {
  private clientId: string
  private socket: ReconnectingWebSocket
  private readonly players: Map<string, Player>
  private readonly createPlayer: (string) => Player

  private constructor(
    clientId: string,
    socket: ReconnectingWebSocket,
    players: Map<string, Player>,
    createPlayer: (string) => Player
  ) {
    this.clientId = clientId
    this.socket = socket
    this.players = players
    this.createPlayer = createPlayer
  }

  static init(players: Map<string, Player>, createPlayer: (string) => Player): Promise<Network> {
    const socket = new ReconnectingWebSocket(`${apiUrl}/client`)


    return new Promise((resolve) => {
      socket.onmessage = (e: MessageEvent): void => {
        const message = JSON.parse(e.data)
        if (message.type === 'connected') {
          console.log(message)
          const network = new Network(message.clientId, socket, players, createPlayer)
          socket.onmessage = network.processMessage.bind(network)
          resolve(network)
        }
      }
    })
  }

  async sendData(data: object): Promise<void> {
    this.socket.send(JSON.stringify({ ...data, clientId: this.clientId }))
  }

  private async processMessage(e: MessageEvent): Promise<void> {
    const message = JSON.parse(e.data)
    console.log(message)

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
    }
  }
}
