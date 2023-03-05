import DensityShader from './density.wgsl?raw'

export enum DensityType {
  Subtract,
  Add
}

export enum DensityShape {
  Sphere,
  Box
}

export interface DensityModel {
  x: number
  y: number
  z: number
  type: DensityType
  shape: DensityShape
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
  public augmentationArray: DensityModel[] = []
  public augmentations: ArrayBuffer

  private constructor(augmentationBuffer: GPUBuffer) {
    this.augmentationBuffer = augmentationBuffer
    this.augmentations = new ArrayBuffer(Uint32Array.BYTES_PER_ELEMENT * 4)
  }

  static async init(device: GPUDevice): Promise<Density> {
    const augmentationSize = 8 * Float32Array.BYTES_PER_ELEMENT * 8
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

  modify(device: GPUDevice, augmentation: DensityModel): void {
    this.augmentationArray.push(augmentation)
    this.update(device, this.augmentationArray)
  }

  update(device: GPUDevice, densityArray: DensityModel[]): void {
    this.augmentations = new ArrayBuffer(
      Uint32Array.BYTES_PER_ELEMENT * 4 + Uint32Array.BYTES_PER_ELEMENT * densityArray.length * 8
    )

    const header = new Uint32Array(this.augmentations, 0, 4)
    header[0] = densityArray.length

    const augmentations = new Float32Array(this.augmentations, Uint32Array.BYTES_PER_ELEMENT * 4)
    const intAugmentations = new Uint32Array(this.augmentations, Uint32Array.BYTES_PER_ELEMENT * 4)
    for (let i = 0; i < densityArray.length; i++) {
      augmentations[i * 8] = densityArray[i].x
      augmentations[i * 8 + 1] = densityArray[i].y
      augmentations[i * 8 + 2] = densityArray[i].z
      augmentations[i * 8 + 3] = 100.0
      intAugmentations[i * 8 + 4] = densityArray[i].type | (densityArray[i].shape << 1)
    }

    device.queue.writeBuffer(
      this.augmentationBuffer,
      0,
      this.augmentations,
      0,
      this.augmentations.byteLength
    )
  }

  updateRaw(device: GPUDevice, densityArray: Float32Array): void {
    this.augmentations = densityArray

    device.queue.writeBuffer(
      this.augmentationBuffer,
      0,
      this.augmentations,
      0,
      this.augmentations.byteLength
    )
  }

  static patch(shader: string): string {
    return shader.replace('#import density', DensityShader)
  }
}
