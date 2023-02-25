import { vec3 } from 'gl-matrix'
import Voxel from './voxel'
import { QueueItem } from './queueItem'

const ctx = self as unknown as Worker

;(async function (): Promise<void> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('Unable to acquire GPU adapter, is WebGPU enabled?')
  const device = await adapter.requestDevice()

  const voxel = await Voxel.init(device)

  console.log('Voxel engine init complete')
  postMessage({ type: 'init_complete' })

  const queue = (item: QueueItem): void => {
    device.queue.onSubmittedWorkDone().then(item.callback)
    device.queue.submit(item.items)
  }

  onmessage = async function (e): Promise<void> {
    const { detail, density } = e.data
    const chunkSize = 31

    const { x, y, z, s } = detail

    const halfChunk = s * chunkSize * 0.5
    const { vertices, normals, indices, consistency } = await voxel.generate(
      device,
      queue,
      vec3.fromValues(
        x * chunkSize - halfChunk,
        y * chunkSize - halfChunk,
        z * chunkSize - halfChunk
      ),
      s,
      density
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
        stride: s,
        consistency: consistency
      },
      [vertices.buffer, normals.buffer, indices.buffer]
    )
  }
})()
