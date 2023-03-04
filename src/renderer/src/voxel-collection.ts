import VertexShader from './vertex.wgsl?raw'
import FragmentShader from './fragment.wgsl?raw'
import { vec3 } from 'gl-matrix'
import VoxelObject from './voxel-object'
import Density, { DensityInstance } from './density'

const swapChainFormat = 'bgra8unorm'

class Ray {
  public sign: number[] = []
  private invdir: vec3

  constructor(public source: vec3, public dir: vec3) {
    this.invdir = vec3.create()
    this.setDirection(dir)
  }

  public setDirection(dir: vec3): void {
    this.dir = dir
    vec3.scale(this.invdir, dir, -1)
    this.sign[0] = this.invdir[0] < 0 ? 1 : 0
    this.sign[1] = this.invdir[1] < 0 ? 1 : 0
    this.sign[2] = this.invdir[2] < 0 ? 1 : 0
  }
}

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

  intersects(s, i, a): boolean {
    const u = [
      {
        x: i.x - a / 2,
        y: i.y - a / 2,
        z: i.z - a / 2
      },
      {
        x: i.x + a / 2,
        y: i.y + a / 2,
        z: i.z + a / 2
      }
    ]
    let l = (u[s.sign[0]].x - s.orig[0]) * s.invdir[0],
      f = (u[1 - s.sign[0]].x - s.orig[0]) * s.invdir[0]
    const h = (u[s.sign[1]].y - s.orig[1]) * s.invdir[1],
      d = (u[1 - s.sign[1]].y - s.orig[1]) * s.invdir[1]
    if (l > d || h > f) return !1
    h > l && (l = h), d < f && (f = d)
    const p = (u[s.sign[2]].z - s.orig[2]) * s.invdir[2],
      g = (u[1 - s.sign[2]].z - s.orig[2]) * s.invdir[2]
    return !(l > g || p > f)
  }

  getAdjacent(s, i, a): VoxelObject[] {
    const u: VoxelObject[] = []
    const l = i / 2
    const f = new Ray([a.x / 31, a.y / 31, a.z / 31], [s.x - a.x, s.y - a.y, s.z - a.z])
    for (const h of this.objects.values()) {
      const d = h.stride / 2
      h.index[0] + d >= s.x - l &&
        s.x + l >= h.index[0] - d &&
        h.index[1] + d >= s.y - l &&
        s.y + l >= h.index[1] - d &&
        h.index[2] + d >= s.z - l &&
        s.z + l >= h.index[2] - d &&
        (f.setDirection([a.x / 31 - h.index[0], a.y / 31 - h.index[1], a.z / 31 - h.index[2]]),
        this.intersects(f, s, i) && u.push(h))
    }
    return u
  }

  set(
    device,
    key,
    position,
    index,
    stride,
    vertices,
    normals,
    indices,
    corners,
    consistency
  ): void {
    let obj: VoxelObject | undefined = this.objects.get(key)
    if (!obj) {
      obj = this.pool.pop()
      if (!obj) {
        obj = new VoxelObject(
          vec3.fromValues(position.x, position.y, position.z),
          vec3.fromValues(index.x, index.y, index.z),
          stride,
          device,
          this.pipeline,
          consistency
        )
      } else {
        obj.index = vec3.fromValues(index.x, index.y, index.z)
        obj.position = vec3.fromValues(position.x, position.y, position.z)
      }
      this.objects.set(key, obj)
    }

    obj.setCorners(device, corners)
    obj.stride = stride
    obj.consistency = consistency
    obj.setVertexBuffer(device, vertices, normals)
    obj.setIndexBuffer(device, indices)

    const halfStride = stride / 2

    for (const [key, value] of this.objects) {
      if (value === obj) continue

      const halfObjStride = value.stride / 2

      if (
        value.index[0] + halfObjStride > obj.index[0] - halfStride &&
        obj.index[0] + halfStride > value.index[0] - halfObjStride &&
        value.index[1] + halfObjStride > obj.index[1] - halfStride &&
        obj.index[1] + halfStride > value.index[1] - halfObjStride &&
        value.index[2] + halfObjStride > obj.index[2] - halfStride &&
        obj.index[2] + halfStride > value.index[2] - halfObjStride
      ) {
        this.free(key)
      }
    }
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
      if (value.consistency) {
        value.draw(passEncoder)
      }
    }
  }
}
