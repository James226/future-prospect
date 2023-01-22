import DensityShader from './density.wgsl?raw'

interface DensityModel {
  x: number
  y: number
  z: number
}

export default class Density {
  private readonly densityBindGroup: GPUBindGroup
  private readonly augmentationBuffer: GPUBuffer

  private constructor(augmentationBuffer: GPUBuffer, densityBindGroup: GPUBindGroup) {
    this.augmentationBuffer = augmentationBuffer
    this.densityBindGroup = densityBindGroup
  }

  static async init(device: GPUDevice, pipeline: GPUComputePipeline): Promise<Density> {
    const augmentationSize = 4 * 12
    const augmentationBuffer = device.createBuffer({
      size: augmentationSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    })
    const densityBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: augmentationBuffer
          }
        }
      ]
    })

    const density = new Density(augmentationBuffer, densityBindGroup)
    density.update(device, [{ x: 1995000, y: 0, z: 0 }])
    return density
  }

  update(device: GPUDevice, densityArray: DensityModel[]): void {
    const augmentations = new Float32Array(densityArray.length * 12)
    for (let i = 0; i < densityArray.length; i++) {
      augmentations[i * 12] = densityArray[i].x
      augmentations[i * 12 + 1] = densityArray[i].y
      augmentations[i * 12 + 2] = densityArray[i].z
    }

    device.queue.writeBuffer(
      this.augmentationBuffer,
      0,
      augmentations.buffer,
      augmentations.byteOffset,
      augmentations.byteLength
    )
  }

  static patch(shader: string): string {
    return shader.replace('#import density', DensityShader)
  }

  apply(encoder: GPUComputePassEncoder): void {
    encoder.setBindGroup(1, this.densityBindGroup)
  }
}
