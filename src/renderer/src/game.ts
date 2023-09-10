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
import Density, { DensityMaterial, DensityShape, DensityType } from './density'
import WorldGenerator, { WorldGeneratorInfo } from './world-generator'
import { Camera } from './camera'
import Pointer from './pointer'
import TouchController from './touch-controller'

class Game {
  private lastUpdate = 0
  private lastTimestamp = 0
  private tool = DensityType.Add
  private shape = DensityShape.Sphere
  private material = DensityMaterial.Rock

  private toolSelector = document.getElementById('add')!
  private shapeSelector = document.getElementById('shape')!
  private materialSelector = document.getElementById('material')!
  private sizeSelector = document.getElementById('size')!

  private constructor(
    private voxelWorker: ContouringWorker,
    private keyboard: Keyboard,
    private mouse: Mouse,
    private physics: Physics,
    private controller: Controller,
    private touchController: TouchController,
    private camera: Camera,
    private collection: VoxelCollection,
    private network: Network,
    private players: Map<string, Player>,
    private pointer: Pointer,
    private generate: () => void
  ) {
    const maxTool = DensityType[Object.keys(DensityType).sort((k) => -DensityType[k])[0]]
    const maxShape = DensityShape[Object.keys(DensityShape).sort((k) => -DensityShape[k])[0]]
    const maxMaterial =
      DensityMaterial[Object.keys(DensityMaterial).sort((k) => -DensityMaterial[k])[0]]

    this.toolSelector?.addEventListener('touchstart', (e) => {
      e.stopPropagation()
    })
    this.toolSelector?.addEventListener('touchend', (e) => {
      e.stopPropagation()
      this.tool = this.tool + 1
      if (this.tool > maxTool) this.tool = 0
      this.toolSelector!.innerText = DensityType[this.tool]
    })

    this.shapeSelector?.addEventListener('touchstart', (e) => {
      e.stopPropagation()
    })
    this.shapeSelector?.addEventListener('touchend', (e) => {
      e.stopPropagation()
      this.shape = this.shape + 1
      if (this.shape > maxShape) this.shape = 0
      this.shapeSelector!.innerText = DensityShape[this.shape]
    })

    this.materialSelector?.addEventListener('touchstart', (e) => {
      e.stopPropagation()
    })
    this.materialSelector?.addEventListener('touchend', (e) => {
      e.stopPropagation()
      this.material = this.material + 1
      if (this.material > maxMaterial) this.material = 0
      this.materialSelector!.innerText = DensityMaterial[this.material]
    })

    this.sizeSelector?.addEventListener('touchstart', (e) => {
      e.stopPropagation()
    })
    this.sizeSelector?.addEventListener('touchend', (e) => {
      e.stopPropagation()
      this.pointer.size = this.pointer.size = this.pointer.size * 2
      if (this.pointer.size > 1024) this.pointer.size = 4
      this.sizeSelector!.innerText = this.pointer.size.toString()
    })
  }

  static async init(device: GPUDevice): Promise<Game> {
    const keyboard = new Keyboard()
    keyboard.init()

    const mouse = new Mouse()
    mouse.init()

    const touchController = new TouchController()
    touchController.init()

    const controller = new Controller(keyboard, mouse, touchController)
    controller.init()

    const camera = new Camera(controller, mouse, touchController)

    const density = await Density.init(device)

    const players = new Map<string, Player>()
    const network = await Network.init(
      controller,
      players,
      (id) => {
        const player = new Player(device, vec3.fromValues(2000000.0, 100.0, 100.0))
        players[id] = player
        return player
      },
      density,
      device
    )

    await network.sendData({ type: 'move', position: { x: 0, y: 0, z: 0 } })

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

    const voxelWorker = await new Promise<ContouringWorker>((resolve) => {
      const voxelWorker = new ContouringWorker()

      voxelWorker.onmessage = ({ data }): void => {
        if (data.type === 'init_complete') {
          console.log('Received Voxel engine init complete')

          resolve(voxelWorker)
        }
      }
    })

    const stride = 2
    const chunkSize = 31
    const worldGenerator = new WorldGenerator(stride)

    let t0 = performance.now()

    let info: WorldGeneratorInfo
    let generating = false

    voxelWorker.onmessage = ({ data }): void => {
      const { type, vertices, consistency, normals, indices, corners, stride } = data
      switch (type) {
        case 'clear':
          collection.freeAll()
          break
        case 'update': {
          collection.set(
            device,
            `${data.ix}x${data.iy}x${data.iz}`,
            { x: data.x, y: data.y, z: data.z },
            { x: data.ix, y: data.iy, z: data.iz },
            stride,
            new Float32Array(vertices),
            new Float32Array(normals),
            new Uint16Array(indices),
            new Uint32Array(corners),
            consistency
          )
          break
        }
      }

      if (info.stride > 2 << 14) {
        generating = false
        console.log(
          `Generation complete in ${performance.now() - t0} milliseconds with ${
            collection.objects.size
          } objects`
        )
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

    const generate = (): void => {
      t0 = performance.now()
      info = worldGenerator.init(
        controller.position[0] / chunkSize,
        controller.position[1] / chunkSize,
        controller.position[2] / chunkSize
      )
      console.log(generating)
      if (generating) return

      generating = true
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
    generate()
    density.onModified = generate

    const pointer = new Pointer(device, controller, camera, raycast)

    const game = new Game(
      voxelWorker,
      keyboard,
      mouse,
      physics,
      controller,
      touchController,
      camera,
      collection,
      network,
      players,
      pointer,
      generate
    )

    document.getElementById('loading')!.style.display = 'none'

    return game
  }

  destroy(): void {
    this.voxelWorker.terminate()
  }

  async update(device: GPUDevice, projectionMatrix: mat4, timestamp: number): Promise<void> {
    const deltaTime = timestamp - this.lastTimestamp

    const queue = (item: QueueItem): void => {
      device.queue.onSubmittedWorkDone().then(() => {
        item.callback()
      })

      device.queue.submit(item.items)
    }

    const maxShape = DensityShape[Object.keys(DensityShape).sort((k) => -DensityShape[k])[0]]
    if (this.keyboard.keydown('1')) {
      this.tool = DensityType.Add
      this.toolSelector!.innerText = DensityType[this.tool]
    }
    if (this.keyboard.keydown('2')) {
      this.tool = DensityType.Subtract
      this.toolSelector!.innerText = DensityType[this.tool]
    }
    if (this.keyboard.keypress('3')) {
      this.shape = Math.max(0, this.shape - 1)
      this.shapeSelector!.innerText = DensityShape[this.shape]
    }
    if (this.keyboard.keypress('4')) {
      this.shape = Math.min(maxShape, this.shape + 1)
      this.shapeSelector!.innerText = DensityShape[this.shape]
    }
    if (this.keyboard.keydown('5')) {
      this.material = DensityMaterial.Rock
      this.materialSelector!.innerText = DensityMaterial[this.material]
    }
    if (this.keyboard.keydown('6')) {
      this.material = DensityMaterial.Wood
      this.materialSelector!.innerText = DensityMaterial[this.material]
    }
    if (this.keyboard.keydown('7')) {
      this.material = DensityMaterial.Fire
      this.materialSelector!.innerText = DensityMaterial[this.material]
    }
    if (this.keyboard.keypress('t')) this.pointer.snapToGrid = !this.pointer.snapToGrid
    if (this.keyboard.keypress('=')) {
      this.pointer.size = this.pointer.size * 2
      this.sizeSelector!.innerText = this.pointer.size.toString()
    }
    if (this.keyboard.keypress('-')) {
      this.pointer.size = Math.max(4, this.pointer.size / 2)
      this.sizeSelector!.innerText = this.pointer.size.toString()
    }

    // const tool = document.getElementById('tool')
    // if (tool) {
    //   tool.innerText = `${DensityType[this.tool]} - ${DensityShape[this.shape]} - ${
    //     DensityMaterial[this.material]
    //   } - ${this.pointer.size} - ${this.pointer.snapToGrid}`
    // }

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

    if (this.keyboard.keypress('g')) {
      this.generate()
    }

    if (this.keyboard.keypress(' ') || this.touchController.trigger() || this.keyboard.keypress('mouse-0')) {
      const gravityDirection = vec3.create()
      vec3.scale(gravityDirection, this.controller.up, 100)
      vec3.add(gravityDirection, this.controller.position, gravityDirection)

      this.network.sendData({
        type: 'build',
        data: JSON.stringify({
          position: {
            x: this.pointer.position[0],
            y: this.pointer.position[1],
            z: this.pointer.position[2]
          },
          shape: this.shape,
          material: this.material,
          tool: this.tool,
          size: this.pointer.size
        })
      })
    }

    this.physics.velocity = this.controller.velocity
    await this.physics.update(device, (q: QueueItem) => queue(q))

    this.controller.position = this.physics.position as vec3
    this.controller.update(deltaTime)
    this.camera.update(projectionMatrix)

    const viewMatrix = this.camera.viewMatrix

    this.pointer.update(device, queue, viewMatrix)

    this.collection.update(device, viewMatrix, timestamp)

    for (const id in this.players) {
      this.players[id].update(device, viewMatrix, timestamp)
    }

    this.keyboard.update()
    this.mouse.update()
    this.touchController.update()
    this.lastTimestamp = timestamp
  }

  draw(passEncoder: GPURenderPassEncoder): void {
    this.pointer.draw(passEncoder)
    for (const id in this.players) {
      this.players[id].draw(passEncoder)
    }
    this.collection.draw(passEncoder)
  }
}

export default Game
