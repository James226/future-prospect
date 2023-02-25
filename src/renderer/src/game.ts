import ContouringWorker from './contouring.worker?worker'
import Controller from './controller'
import Keyboard from './keyboard'
import VoxelCollection from './voxel-collection'
import Physics from './physics'
import { mat4, vec3, vec4 } from 'gl-matrix'
import Mouse from './mouse'
import { QueueItem } from './queueItem'
import Raycast from './raycast'
import Network from './network'
import Player from './player'
import Density from './density'
import WorldGenerator from './world-generator'

class Game {
  private loaded = false
  private generating = false
  private lastUpdate = 0
  private lastTimestamp = 0
  private stride = 16

  private voxelWorker: ContouringWorker
  private keyboard: Keyboard
  private mouse: Mouse
  private physics: Physics
  private controller: Controller
  private collection: VoxelCollection
  private raycast: Raycast
  private network: Network
  private players: Map<string, Player>
  private density: Density

  private constructor(
    voxelWorker: ContouringWorker,
    keyboard: Keyboard,
    mouse: Mouse,
    physics: Physics,
    controller: Controller,
    collection: VoxelCollection,
    raycast: Raycast,
    network: Network,
    players: Map<string, Player>,
    density: Density
  ) {
    this.voxelWorker = voxelWorker
    this.keyboard = keyboard
    this.mouse = mouse
    this.physics = physics
    this.controller = controller
    this.collection = collection
    this.raycast = raycast
    this.network = network
    this.players = players
    this.density = density
  }

  static async init(device: GPUDevice): Promise<Game> {
    const keyboard = new Keyboard()
    keyboard.init()

    const mouse = new Mouse()
    mouse.init()

    const controller = new Controller(keyboard, mouse)
    controller.init()

    const players = new Map<string, Player>()
    const network = await Network.init(controller, players, (id) => {
      const player = new Player(device, vec3.fromValues(2000000.0, 100.0, 100.0))
      players[id] = player
      return player
    })

    await network.sendData({ type: 'move', position: { x: 0, y: 0, z: 0 } })

    //this.player = new Player(vec3.fromValues(2000000.0, 100.0, 100.0));
    //this.player.init(device);

    const density = await Density.init(device)
    density.update(device, [
      { x: 2007000, y: 0, z: 1000 },
      { x: 2007000, y: 0, z: 0 }
    ])

    const physics = await Physics.init(
      device,
      vec4.fromValues(controller.position[0], controller.position[1], controller.position[2], 0),
      density
    )

    const collection = await VoxelCollection.init(device, density)

    const raycast = await Raycast.init(device, density)

    if (import.meta.hot) {
      import.meta.hot.accept('./voxel-collection.ts', async (module) => {
        console.log('new module', module)
        //this.collection = await VoxelCollection.init(device, this.collection.objects)
      })
    }

    const voxelWorker = new ContouringWorker()
    const game = new Game(
      voxelWorker,
      keyboard,
      mouse,
      physics,
      controller,
      collection,
      raycast,
      network,
      players,
      density
    )

    voxelWorker.onmessage = ({ data }): void => {
      if (data.type === 'init_complete') {
        console.log('Received Voxel engine init complete')

        document.getElementById('loading')!.style.display = 'none'
        game.loaded = true

        const stride = 32
        const chunkSize = 31
        const worldGenerator = new WorldGenerator(stride)

        let info = worldGenerator.init(
          controller.position[0] / chunkSize,
          controller.position[1] / chunkSize,
          controller.position[2] / chunkSize
        )

        const t0 = performance.now()

        voxelWorker.onmessage = ({ data }): void => {
          const { type, vertices, normals, indices, corners, stride } = data
          switch (type) {
            case 'clear':
              collection.freeAll()
              break
            case 'update': {
              if (vertices.byteLength) {
                collection.set(
                  device,
                  `${data.ix}x${data.iy}x${data.iz}`,
                  { x: data.x, y: data.y, z: data.z },
                  stride,
                  new Float32Array(vertices),
                  new Float32Array(normals),
                  new Uint16Array(indices),
                  new Uint32Array(corners)
                )
              } else {
                collection.free(`${data.ix}x${data.iy}x${data.iz}`)
              }
              break
            }
          }

          if (info.stride > 2 << 14) {
            console.log(`Generation complete in ${performance.now() - t0} milliseconds`)
            return
          }

          const r = worldGenerator.next(info)
          const result = r[0]
          voxelWorker.postMessage({
            stride: stride,
            position: controller.position,
            detail: {
              x: result.x,
              y: result.y,
              z: result.z,
              s: result.stride
            },
            density: density.augmentations
          })
          info = r[1]
        }

        const r = worldGenerator.next(info)
        const result = r[0]
        voxelWorker.postMessage({
          stride: stride,
          position: controller.position,
          detail: {
            x: result.x,
            y: result.y,
            z: result.z,
            s: result.stride
          },
          density: density.augmentations
        })
        info = r[1]
      }
    }

    return game
  }

  destroy(): void {
    this.voxelWorker.terminate()
  }

  generate(device: GPUDevice, data: unknown): void {
    if (this.generating || !this.loaded) return

    const t0 = performance.now()

    this.voxelWorker.onmessage = ({ data }): void => {
      const { type, vertices, normals, indices, corners, stride } = data
      switch (type) {
        case 'clear':
          this.collection.freeAll()
          break
        case 'update': {
          if (vertices.byteLength) {
            this.collection.set(
              device,
              `${data.ix}x${data.iy}x${data.iz}`,
              { x: data.x, y: data.y, z: data.z },
              stride,
              new Float32Array(vertices),
              new Float32Array(normals),
              new Uint16Array(indices),
              new Uint32Array(corners)
            )
          } else {
            this.collection.free(`${data.ix}x${data.iy}x${data.iz}`)
          }
          break
        }
      }
    }

    this.collection.freeAll()

    this.voxelWorker.postMessage({
      stride: this.stride,
      position: this.controller.position,
      detail: data,
      density: this.density.augmentations
    })

    this.stride /= 2
    if (this.stride < 1) this.stride = 32

    this.generating = false

    console.log(`Generation complete in ${performance.now() - t0} milliseconds`)
  }

  async update(device: GPUDevice, projectionMatrix: mat4, timestamp: number): Promise<void> {
    const deltaTime = timestamp - this.lastTimestamp

    if (this.keyboard.keypress('g')) {
      this.generate(device, null)
    }

    // Disable regeneration of world
    if (timestamp - this.lastUpdate > 10000) {
      //this.voxelWorker.postMessage({stride: this.stride, position: this.controller.position});
      this.network.sendData({
        type: 'position',
        position: {
          x: this.controller.position[0],
          y: this.controller.position[1],
          z: this.controller.position[2]
        }
      })

      this.lastUpdate = timestamp
    }

    const queue = (item: QueueItem): void => {
      device.queue.onSubmittedWorkDone().then((_) => {
        item.callback()
      })

      device.queue.submit(item.items)
    }

    this.physics.velocity = this.controller.velocity
    await this.physics.update(device, (q: QueueItem) => queue(q))

    this.controller.position = this.physics.position as vec3
    this.controller.update(device, projectionMatrix, queue, this.raycast, deltaTime)

    const viewMatrix = this.controller.viewMatrix

    this.collection.update(device, viewMatrix, timestamp)

    for (const id in this.players) {
      this.players[id].update(device, viewMatrix, timestamp)
    }

    this.keyboard.update()
    this.mouse.update()
    this.lastTimestamp = timestamp;
  }

  draw(passEncoder: GPURenderPassEncoder): void {
    for (const id in this.players) {
      this.players[id].draw(passEncoder)
    }
    this.collection.draw(passEncoder)
  }
}

export default Game
