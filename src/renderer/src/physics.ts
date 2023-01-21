import { vec3, vec4 } from 'gl-matrix'
import Physics from './physics.wgsl?raw'
import Density from './density.wgsl?raw'

import Random from 'seedrandom'

export default class Voxel {
  private running = false

  velocity: vec3
  position: vec4
  private readonly computePipeline: GPUComputePipeline
  private readonly actorsBuffer: GPUBuffer
  private readonly computeBindGroup: GPUBindGroup
  private readonly actorsReadBuffer: GPUBuffer
  private readonly densityBindGroup: GPUBindGroup

  private constructor(
    velocity: vec3,
    position: vec4,
    computePipeline: GPUComputePipeline,
    actorsBuffer: GPUBuffer,
    computeBindGroup: GPUBindGroup,
    actorsReadBuffer: GPUBuffer,
    densityBindGroup: GPUBindGroup
  ) {
    this.velocity = velocity
    this.position = position
    this.computePipeline = computePipeline
    this.actorsBuffer = actorsBuffer
    this.computeBindGroup = computeBindGroup
    this.actorsReadBuffer = actorsReadBuffer
    this.densityBindGroup = densityBindGroup
  }
  static async init(device: GPUDevice): Promise<Voxel> {
    const physics = Physics.replace('#import density', Density)

    const velocity = vec3.fromValues(0, 0, 0)
    const position = vec4.fromValues(2005000, 0, 0, 0)
    const start = performance.now()
    console.log('Loading physics engine')
    const computePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: physics
        }),
        entryPoint: 'computePhysics'
      }
    })

    const permutations = new Int32Array(512)

    const random = new Random('James')
    for (let i = 0; i < 256; i++) permutations[i] = 256 * random()

    for (let i = 256; i < 512; i++) permutations[i] = permutations[i - 256]

    const permutationsBuffer = device.createBuffer({
      size: permutations.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    })

    new Int32Array(permutationsBuffer.getMappedRange()).set(permutations)
    permutationsBuffer.unmap()

    const actorsBuffer = device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    })

    const actors = new Float32Array(actorsBuffer.getMappedRange())
    actors.set(position)
    actorsBuffer.unmap()

    const computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 6,
          resource: {
            buffer: actorsBuffer
          }
        }
      ]
    })

    const actorsReadBuffer = device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const augmentationSize = 4 * 4 + 4 * 4
    const augmentationBuffer = device.createBuffer({
      size: augmentationSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })

    const densityBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: augmentationBuffer
          }
        }
      ]
    })

    console.log('Physics engine loaded', performance.now() - start)

    return new Voxel(
      velocity,
      position,
      computePipeline,
      actorsBuffer,
      computeBindGroup,
      actorsReadBuffer,
      densityBindGroup
    )
  }

  generate(device, queue): Promise<void> {
    return new Promise((resolve) => {
      //const uniform = vec4.fromValues(position[0], position[1], position[2], stride);
      device.queue.writeBuffer(
        this.actorsBuffer,
        Float32Array.BYTES_PER_ELEMENT * 4,
        (<Float32Array>this.velocity).buffer
      )

      const computeEncoder = device.createCommandEncoder()
      const computePassEncoder = computeEncoder.beginComputePass()
      computePassEncoder.setPipeline(this.computePipeline)
      computePassEncoder.setBindGroup(0, this.computeBindGroup)
      computePassEncoder.setBindGroup(1, this.densityBindGroup)
      computePassEncoder.dispatchWorkgroups(1)
      computePassEncoder.end()

      const copyEncoder = device.createCommandEncoder()
      copyEncoder.copyBufferToBuffer(
        this.actorsBuffer,
        0,
        this.actorsReadBuffer,
        0,
        Float32Array.BYTES_PER_ELEMENT * 8
      )

      queue({
        items: [computeEncoder.finish(), copyEncoder.finish()],
        callback: () => {
          this.actorsReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const buffer = this.actorsReadBuffer.getMappedRange()
            const result = new Float32Array(buffer)
            ;(<Float32Array>this.position).set([result[0], result[1], result[2]])
            //this.position.set(new Float32Array(buffer, 0, 3));
            this.actorsReadBuffer.unmap()
            resolve()
          })
        }
      })
    })
  }

  async update(device, queue): Promise<void> {
    if (this.running) return Promise.resolve()

    this.running = true
    await this.generate(device, queue)
    this.running = false
  }
}
