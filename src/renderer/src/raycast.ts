import { vec3 } from 'gl-matrix'
import Ray from './ray.wgsl?raw'
import Density from './density'

class Intersection {
  position: vec3
  normal: vec3
  distance: number

  constructor(position, normal, distance) {
    this.position = position
    this.normal = normal
    this.distance = distance
  }
}

export default class Raycast {
  private readonly computePipeline: GPUComputePipeline
  private readonly uniformBuffer: GPUBuffer
  private readonly intersectionsBuffer: GPUBuffer
  private readonly computeBindGroup: GPUBindGroup
  private readonly intersectionsReadBuffer: GPUBuffer
  private readonly density: Density

  private constructor(
    computePipeline: GPUComputePipeline,
    uniformBuffer: GPUBuffer,
    intersectionsBuffer: GPUBuffer,
    computeBindGroup: GPUBindGroup,
    intersectionsReadBuffer: GPUBuffer,
    density: Density
  ) {
    this.computePipeline = computePipeline
    this.uniformBuffer = uniformBuffer
    this.intersectionsBuffer = intersectionsBuffer
    this.computeBindGroup = computeBindGroup
    this.intersectionsReadBuffer = intersectionsReadBuffer
    this.density = density
  }

  static async init(device: GPUDevice): Promise<Raycast> {
    const computePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: Density.patch(Ray)
        }),
        entryPoint: 'main'
      }
    })

    const density = await Density.init(device, computePipeline)

    const uniformBufferSize = Float32Array.BYTES_PER_ELEMENT * 8
    const uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    const intersectionsBuffer = device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    })

    //const actors = new Float32Array(this.intersectionsBuffer.getMappedRange())
    intersectionsBuffer.unmap()

    const computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: uniformBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: intersectionsBuffer
          }
        }
      ]
    })

    const intersectionsReadBuffer = device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    return new Raycast(
      computePipeline,
      uniformBuffer,
      intersectionsBuffer,
      computeBindGroup,
      intersectionsReadBuffer,
      density
    )
  }

  cast(device: GPUDevice, queue, position: vec3, direction: vec3): Promise<Intersection | null> {
    return new Promise((resolve) => {
      device.queue.writeBuffer(this.uniformBuffer, 0, (<Float32Array>position).buffer)
      device.queue.writeBuffer(
        this.uniformBuffer,
        Float32Array.BYTES_PER_ELEMENT * 4,
        (<Float32Array>direction).buffer
      )

      const computeEncoder = device.createCommandEncoder()
      const computePassEncoder = computeEncoder.beginComputePass()
      computePassEncoder.setPipeline(this.computePipeline)
      computePassEncoder.setBindGroup(0, this.computeBindGroup)
      this.density.apply(computePassEncoder)
      computePassEncoder.dispatchWorkgroups(1)
      computePassEncoder.end()

      const copyEncoder = device.createCommandEncoder()
      copyEncoder.copyBufferToBuffer(
        this.intersectionsBuffer,
        0,
        this.intersectionsReadBuffer,
        0,
        Float32Array.BYTES_PER_ELEMENT * 8
      )

      queue({
        items: [computeEncoder.finish(), copyEncoder.finish()],
        callback: () => {
          this.intersectionsReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const buffer = this.intersectionsReadBuffer.getMappedRange()
            const found = new Uint32Array(buffer, Float32Array.BYTES_PER_ELEMENT * 3, 1)
            if (found[0] === 0) resolve(null)

            const result = new Float32Array(buffer)
            const intersection = new Intersection(
              vec3.fromValues(result[0], result[1], result[2]),
              vec3.fromValues(result[4], result[5], result[6]),
              result[7]
            )
            this.intersectionsReadBuffer.unmap()
            resolve(intersection)
          })
        }
      })
    })
  }
}
