import { vec3 } from 'gl-matrix'
import Voxel from './voxel'
import { QueueItem } from './queueItem'
import WorldGenerator, { WorldGeneratorInfo, WorldPosition } from './world-generator'
import { CountDownLatch } from './countDownLatch'

const ctx = self as unknown as Worker

function equal(a: WorldPosition, b: WorldPosition): boolean {
  return a.stride == b.stride && a.x == b.x && a.y == b.y && a.z == b.z
}

;(async function (): Promise<void> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('Unable to acquire GPU adapter, is WebGPU enabled?')
  const device = await adapter.requestDevice()

  const voxel = await Voxel.init(device)

  const generators: Voxel[] = []
  const threadCount = 1
  for (let i = 0; i < threadCount; i++) {
    generators[i] = await Voxel.init(device)
  }

  console.log('Voxel engine init complete')
  postMessage({ type: 'init_complete' })

  let gpuQueue: GPUCommandBuffer[] = []
  const callbacks: (() => Promise<void>)[] = []
  const queue = (item: QueueItem): void => {
    callbacks.push(item.callback)
    gpuQueue.push(...item.items)
    //device.queue.submit(item.items);
  }

  let lastInfo: null | WorldGeneratorInfo = null

  let generating = false
  const generatedPositions: WorldPosition[] = []

  onmessage = async function (e): Promise<void> {
    if (generating) {
      return
    }

    generating = true
    const { position, detail } = e.data

    if (detail) {
      const { x, y, z, s } = detail
      const { vertices, normals, indices } = await voxel.generate(
        device,
        queue,
        vec3.fromValues(x - 31 * s * 0.5, y - 31 * s * 0.5, z - 31 * s * 0.5),
        s
      )

      ctx.postMessage(
        {
          type: 'update',
          i: `${x}:${y}:${z}`,
          ix: x,
          iy: y,
          iz: z,
          x: 0,
          y: 0,
          z: 0,
          vertices: vertices.buffer,
          normals: normals.buffer,
          indices: indices.buffer,
          stride: s
        },
        [vertices.buffer, normals.buffer, indices.buffer]
      )
      generating = false
      return
    }

    const stride = 32
    // const size = 128
    const chunkSize = 31
    // const worldSize = Math.ceil(size / stride)

    const t0 = performance.now()

    const worldGenerator = new WorldGenerator(stride)

    let info = worldGenerator.init(
      position[0] / chunkSize,
      position[1] / chunkSize,
      position[2] / chunkSize
    )

    if (
      lastInfo !== null &&
      info.x === lastInfo.x &&
      info.y === lastInfo.y &&
      info.z === lastInfo.z
    ) {
      generating = false
      return
    }

    console.log(`Init world at ${info.x}:${info.y}:${info.z} for stride ${stride}`)

    lastInfo = info
    //ctx.postMessage({ type: 'clear'});

    const times = {
      32: [],
      64: [],
      128: [],
      256: [],
      512: [],
      1024: []
    }

    do {
      for (let i = 0; i < threadCount; i++) {
        //let timer = performance.now();
        const r = worldGenerator.next(info)
        const result = r[0]
        info = r[1]

        const matchingChunk = generatedPositions.filter((p) => equal(p, result))
        if (matchingChunk.length > 0) {
          continue
        }

        let generator: Voxel
        for (let j = 0; j < generators.length; j++) {
          if (!generators[i].running) {
            generator = generators[i]
            generator.running = true
            break
          }
        }

        const { x, y, z } = result
        const halfChunk = result.stride * chunkSize * 0.5

        //const { vertices, normals, indices, corners } = await voxel.generate(device, queue, vec3.fromValues(x * chunkSize -halfChunk, y * chunkSize -halfChunk, z * chunkSize -halfChunk), result.stride);

        generator!
          .generate(
            device,
            queue,
            vec3.fromValues(
              x * chunkSize - halfChunk,
              y * chunkSize - halfChunk,
              z * chunkSize - halfChunk
            ),
            result.stride
          )
          .then(({ vertices, normals, indices }) => {
            //console.log(`Block generated with ${vertices.length} in ${performance.now() - timer}`);
            ctx.postMessage(
              {
                type: 'update',
                i: `${x}:${y}:${z}`,
                ix: x,
                iy: y,
                iz: z,
                x: 0,
                y: 0,
                z: 0,
                vertices: vertices.buffer,
                normals: normals.buffer,
                indices: indices.buffer,
                stride: result.stride
              },
              [vertices.buffer, normals.buffer, indices.buffer]
            )
            generatedPositions.push(result)
            generator.running = false
            //times[info.stride].push(performance.now() - timer);
          })
        // if (vertices.length > 0) {
        //   console.log(`Generating ${x}:${y}:${z} (${x * chunkSize -halfChunk}:${y * chunkSize -halfChunk}:${z * chunkSize -halfChunk}) (${result.stride} / ${halfChunk} / ${info.previousOffset})`)
        // }

        // if (vertices.length > 0) {
        //   console.log(`Block generated with ${vertices.length} in ${performance.now() - timer}`);
        // }
        //ctx.postMessage(({ type: 'update', i: `${x}:${y}:${z}`, ix: x, iy: y, iz: z, x: 0, y: 0, z: 0, vertices: vertices.buffer, normals: normals.buffer, indices: indices.buffer, corners: corners.buffer, stride: result.stride }), [vertices.buffer, normals.buffer, indices.buffer, corners.buffer])
        while (gpuQueue.length > 0) {
          const promise = device.queue.onSubmittedWorkDone()
          device.queue.submit(gpuQueue)
          await promise
          gpuQueue = []
          const countdown = new CountDownLatch(callbacks.length)

          for (let j = callbacks.length - 1; j >= 0; j--) {
            callbacks[j]().then(() => {
              countdown.countDown()
            })
            callbacks.splice(j, 1)
          }
          await countdown.awaitZero()
        }
      }
    } while (info.stride <= 2 << 14)

    for (const k in times) {
      console.log(
        `${k}: ${times[k].reduce((partialSum, a) => partialSum + a, 0) / times[k].length}`
      )
    }
    generating = false

    console.log(`Generation complete in ${performance.now() - t0} milliseconds`)
  }
})()
