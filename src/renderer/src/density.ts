import DensityShader from './density.wgsl?raw'

export interface DensityModel {
  x: number
  y: number
  z: number
}

export class DensityInstance {
  private readonly bindGroup: GPUBindGroup

  constructor(bindGroup: GPUBindGroup) {
    this.bindGroup = bindGroup
  }

  apply(encoder: GPUBindingCommandsMixin): void {
    encoder.setBindGroup(1, this.bindGroup)
  }
}

export default class Density {
  private readonly augmentationBuffer: GPUBuffer
  public augmentations: Float32Array

  private constructor(augmentationBuffer: GPUBuffer) {
    this.augmentationBuffer = augmentationBuffer
    this.augmentations = new Float32Array()
  }

  static async init(device: GPUDevice): Promise<Density> {
    const augmentationSize = 8 * Float32Array.BYTES_PER_ELEMENT * 2
    const augmentationBuffer = device.createBuffer({
      size: augmentationSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    })

    return new Density(augmentationBuffer)
  }

  async apply(device: GPUDevice, pipeline: GPUPipelineBase): Promise<DensityInstance> {
    const densityBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.augmentationBuffer
          }
        }
      ]
    })

    return new DensityInstance(densityBindGroup)
  }

  update(device: GPUDevice, densityArray: DensityModel[]): void {
    this.augmentations = new Float32Array(densityArray.length * 8)
    for (let i = 0; i < densityArray.length; i++) {
      this.augmentations[i * 8] = densityArray[i].x
      this.augmentations[i * 8 + 1] = densityArray[i].y
      this.augmentations[i * 8 + 2] = densityArray[i].z
    }

    device.queue.writeBuffer(
      this.augmentationBuffer,
      0,
      this.augmentations.buffer,
      this.augmentations.byteOffset,
      this.augmentations.byteLength
    )
  }

  updateRaw(device: GPUDevice, densityArray: Float32Array): void {
    this.augmentations = densityArray

    device.queue.writeBuffer(
      this.augmentationBuffer,
      0,
      this.augmentations.buffer,
      this.augmentations.byteOffset,
      this.augmentations.byteLength
    )
  }

  static patch(shader: string): string {
    return shader.replace('#import density', DensityShader)
  }
}
