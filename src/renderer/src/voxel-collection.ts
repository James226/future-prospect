import VertexShader from './vertex.wgsl?raw'
import FragmentShader from './fragment.wgsl?raw'
import { vec3 } from 'gl-matrix'
import VoxelObject from './voxel-object'
import Density, { DensityInstance } from './density'

const swapChainFormat = 'bgra8unorm'

export default class VoxelCollection {
  public readonly objects: Map<object, VoxelObject>
  private readonly pool: VoxelObject[]
  private readonly pipeline: GPURenderPipeline
  private density: DensityInstance

  private constructor(pipeline, densityInstance: DensityInstance) {
    this.pipeline = pipeline
    this.objects = new Map<object, VoxelObject>()
    this.density = densityInstance
    this.pool = []
  }

  static async init(device: GPUDevice, density: Density): Promise<VoxelCollection> {
    const uniformLayout = device.createBindGroupLayout({
      entries: [
        {
          // Transform
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'uniform'
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'read-only-storage'
          }
        }
      ]
    })

    const densityLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'read-only-storage'
          }
        }
      ]
    })

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [uniformLayout, densityLayout]
    })

    const pipeline = device.createRenderPipeline({
      layout: pipelineLayout,

      vertex: {
        module: device.createShaderModule({
          code: VertexShader
        }),
        buffers: [
          {
            arrayStride: 4 * 10,
            attributes: [
              {
                // position
                shaderLocation: 0,
                offset: 0,
                format: 'float32x4'
              },
              {
                // color
                shaderLocation: 1,
                offset: 4 * 4,
                format: 'float32x4'
              }
            ]
          }
        ],
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({
          code: Density.patch(FragmentShader)
        }),
        entryPoint: 'main',
        targets: [
          {
            format: swapChainFormat
          }
        ]
      },

      primitive: {
        topology: 'triangle-list',
        cullMode: 'back'
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus-stencil8'
      }
    })

    const densityInstance = await density.apply(device, pipeline)

    // if (module.hot) {
    //   module.hot.accept(['./fragment.wgsl'], (a) => {
    //     buildPipeline()
    //   })
    // }

    return new VoxelCollection(pipeline, densityInstance)
  }

  set(device, key, position, stride, vertices, normals, indices, corners): void {
    let obj: VoxelObject | undefined = this.objects.get(key)
    if (!obj) {
      obj = this.pool.pop()
      if (!obj) {
        obj = new VoxelObject(
          vec3.fromValues(position.x, position.y, position.z),
          stride,
          device,
          this.pipeline
        )
      } else {
        obj.position = vec3.fromValues(position.x, position.y, position.z)
      }
      this.objects.set(key, obj)
    }

    obj.setCorners(device, corners)
    obj.stride = stride
    obj.setVertexBuffer(device, vertices, normals)
    obj.setIndexBuffer(device, indices)
  }

  free(key): void {
    const obj = this.objects.get(key)

    if (obj) {
      this.pool.push(obj)
      this.objects.delete(key)
    }
  }

  freeAll(): void {
    for (const [key, value] of this.objects) {
      this.pool.push(value)
      this.objects.delete(key)
    }
  }

  update(device, projectionMatrix, timestamp: number): void {
    for (const value of this.objects.values()) {
      value.update(device, projectionMatrix, timestamp)
    }
  }

  draw(passEncoder: GPURenderPassEncoder): void {
    passEncoder.setPipeline(this.pipeline)
    this.density.apply(passEncoder)
    for (const value of this.objects.values()) {
      value.draw(passEncoder)
    }
  }
}
