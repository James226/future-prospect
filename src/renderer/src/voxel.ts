import { vec3 } from 'gl-matrix'
import ComputeCorners from './compute-corners.wgsl?raw'
import ComputePositions from './compute-positions.wgsl?raw'
import ComputeVoxels from './compute-voxels.wgsl?raw'
import Density, { DensityInstance } from './density'

import Random from 'seedrandom'
import ContourCells from './contouring'
import ConstructOctree from './octree'

interface DrawInfo {
  position: vec3
  averageNormal: vec3
  corners: number
  index: number
}

export interface Node {
  type: string
  size: number
  min: vec3
  drawInfo: DrawInfo
  children: Node[]
}

const generateVertexIndices = (node, vertices, normals, nodeSize): void => {
  if (node == null) return

  if (node.size > nodeSize) {
    if (node.type !== 'leaf') {
      for (let i = 0; i < 8; i++) {
        generateVertexIndices(node.children[i], vertices, normals, nodeSize)
      }
    }
  }

  if (node.type !== 'internal') {
    const d = node.drawInfo
    if (d == null) {
      throw 'Error! Could not add vertex!'
    }

    d.index = vertices.length / 3
    vertices.push(d.position[0], d.position[1], d.position[2])
    normals.push(d.averageNormal[0], d.averageNormal[1], d.averageNormal[2])
  }
}

const computeVoxels = (
  position,
  stride,
  voxelCount,
  computedVoxelsData
): {
  vertices: Float32Array
  normals: Float32Array
  indices: Uint16Array
  corners: Uint32Array
} => {
  const computedVoxels: unknown[] = []

  if (voxelCount === 0) {
    return {
      vertices: new Float32Array(),
      normals: new Float32Array(),
      indices: new Uint16Array(),
      corners: new Uint32Array()
    }
  }

  for (let i = 0; i < voxelCount * 12; i += 12) {
    if (computedVoxelsData[i + 11] !== 0) {
      const leaf = {
        type: 'leaf',
        size: stride,
        min: vec3.fromValues(
          computedVoxelsData[i],
          computedVoxelsData[i + 1],
          computedVoxelsData[i + 2]
        ),
        drawInfo: {
          position: vec3.fromValues(
            computedVoxelsData[i + 4],
            computedVoxelsData[i + 5],
            computedVoxelsData[i + 6]
          ),
          averageNormal: vec3.fromValues(
            computedVoxelsData[i + 8],
            computedVoxelsData[i + 9],
            computedVoxelsData[i + 10]
          ),
          corners: computedVoxelsData[i + 3]
        }
      }
      computedVoxels.push(leaf)
    }
  }

  const tree = ConstructOctree(computedVoxels, position, 32 * stride)

  const vertices = []
  const normals = []

  generateVertexIndices(tree, vertices, normals, 1)

  const indices = []
  ContourCells(tree!, indices)

  return {
    vertices: new Float32Array(vertices),
    normals: new Float32Array(normals),
    indices: new Uint16Array(indices),
    corners: new Uint32Array()
  }
}

export default class Voxel {
  public running = false

  private readonly computePipeline: GPUComputePipeline
  private readonly computeCornersPipeline: GPUComputePipeline
  private readonly uniformBuffer: GPUBuffer
  private readonly cornerMaterials: GPUBuffer
  private readonly cornerMaterialsRead: GPUBuffer
  private readonly voxelMaterialsBuffer: GPUBuffer
  private readonly voxelMaterialsBufferRead: GPUBuffer
  private readonly cornerIndexBuffer: GPUBuffer
  private readonly gpuReadBuffer: GPUBuffer
  private readonly permutationsBuffer: GPUBuffer
  private readonly voxelsBuffer: GPUBuffer
  private readonly computeBindGroup: GPUBindGroup
  private readonly computeCornersBindGroup: GPUBindGroup
  private readonly computePositionsPipeline: GPUComputePipeline
  private readonly computePositionsBindGroup: GPUBindGroup
  private readonly computeVoxelsPipeline: GPUComputePipeline
  private readonly computeVoxelsBindGroup: GPUBindGroup
  private readonly voxelReadBuffer: GPUBuffer

  private density: Density
  private densityBindGroup: DensityInstance
  private mainDensityBindGroup: DensityInstance

  private constructor(
    computePipeline: GPUComputePipeline,
    computeCornersPipeline: GPUComputePipeline,
    uniformBuffer: GPUBuffer,
    cornerMaterials: GPUBuffer,
    cornerMaterialsRead: GPUBuffer,
    voxelMaterialsBuffer: GPUBuffer,
    voxelMaterialsBufferRead: GPUBuffer,
    cornerIndexBuffer: GPUBuffer,
    gpuReadBuffer: GPUBuffer,
    permutationsBuffer: GPUBuffer,
    voxelsBuffer: GPUBuffer,
    computeBindGroup: GPUBindGroup,
    computeCornersBindGroup: GPUBindGroup,
    computePositionsPipeline: GPUComputePipeline,
    computePositionsBindGroup: GPUBindGroup,
    computeVoxelsPipeline: GPUComputePipeline,
    computeVoxelsBindGroup: GPUBindGroup,
    voxelReadBuffer: GPUBuffer,
    density: Density,
    densityBindGroup: DensityInstance,
    mainDensityBindGroup: DensityInstance
  ) {
    this.computePipeline = computePipeline
    this.computeCornersPipeline = computeCornersPipeline
    this.uniformBuffer = uniformBuffer
    this.cornerMaterials = cornerMaterials
    this.cornerMaterialsRead = cornerMaterialsRead
    this.voxelMaterialsBuffer = voxelMaterialsBuffer
    this.voxelMaterialsBufferRead = voxelMaterialsBufferRead
    this.cornerIndexBuffer = cornerIndexBuffer
    this.gpuReadBuffer = gpuReadBuffer
    this.permutationsBuffer = permutationsBuffer
    this.voxelsBuffer = voxelsBuffer
    this.computeBindGroup = computeBindGroup
    this.computeCornersBindGroup = computeCornersBindGroup
    this.computePositionsPipeline = computePositionsPipeline
    this.computePositionsBindGroup = computePositionsBindGroup
    this.computeVoxelsPipeline = computeVoxelsPipeline
    this.computeVoxelsBindGroup = computeVoxelsBindGroup
    this.voxelReadBuffer = voxelReadBuffer
    this.density = density
    this.densityBindGroup = densityBindGroup
    this.mainDensityBindGroup = mainDensityBindGroup
  }

  static async init(device: GPUDevice): Promise<Voxel> {
    const computeVoxelsCode = Density.patch(ComputeVoxels)
    const start = performance.now()
    console.log('Start loading voxel engine', performance.now() - start)

    const module = device.createShaderModule({
      code: computeVoxelsCode
    })

    const computePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'computeMaterials'
      }
    })

    //console.log('10', performance.now() - start);

    const computeCornersPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: ComputeCorners
        }),
        entryPoint: 'main'
      }
    })

    const uniformBufferSize = Math.max(4 * 5, 32)
    const uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    const cornerMaterials = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT * 33 * 33 * 33,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false
    })

    const cornerMaterialsRead = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT * 33 * 33 * 33,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const voxelMaterialsBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false
    })

    const voxelMaterialsBufferRead = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const cornerIndexBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT + Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false
    })

    const gpuReadBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const permutations = new Int32Array(512)

    const random = new Random(6452)
    for (let i = 0; i < 256; i++) permutations[i] = 256 * random()

    for (let i = 256; i < 512; i++) permutations[i] = permutations[i - 256]

    const permutationsBuffer = device.createBuffer({
      size: permutations.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    })

    new Int32Array(permutationsBuffer.getMappedRange()).set(permutations)
    permutationsBuffer.unmap()

    const voxelsBuffer = device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * 12 * 32 * 32 * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false
    })

    //console.log('20', performance.now() - start);

    const computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 1,
          resource: {
            buffer: cornerMaterials
          }
        },

        {
          binding: 5,
          resource: {
            buffer: uniformBuffer
          }
        }
      ]
    })

    //console.log('21', performance.now() - start);

    const computeCornersBindGroup = device.createBindGroup({
      layout: computeCornersPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 1,
          resource: {
            buffer: cornerMaterials
          }
        },
        {
          binding: 2,
          resource: {
            buffer: voxelMaterialsBuffer
          }
        }
      ]
    })

    const computePositionsPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: ComputePositions
        }),
        entryPoint: 'main'
      }
    })

    //console.log('30', performance.now() - start);

    const computePositionsBindGroup = device.createBindGroup({
      layout: computePositionsPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 2,
          resource: {
            buffer: voxelMaterialsBuffer
          }
        },
        {
          binding: 3,
          resource: {
            buffer: cornerIndexBuffer
          }
        }
      ]
    })

    //console.log('31', performance.now() - start);

    //console.log('31.5', performance.now() - start);
    const computeVoxelsPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main'
      }
    })

    //console.log('32', performance.now() - start);

    const computeVoxelsBindGroup = device.createBindGroup({
      layout: computeVoxelsPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 2,
          resource: {
            buffer: voxelMaterialsBuffer
          }
        },
        {
          binding: 3,
          resource: {
            buffer: cornerIndexBuffer
          }
        },
        {
          binding: 4,
          resource: {
            buffer: voxelsBuffer
          }
        },
        {
          binding: 5,
          resource: {
            buffer: uniformBuffer
          }
        }
      ]
    })

    //console.log('40', performance.now() - start);

    const voxelReadBuffer = device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * 12 * 32 * 32 * 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const density = await Density.init(device)
    const densityBindGroup = await density.apply(device, computePipeline)
    const mainDensityBindGroup = await density.apply(device, computeVoxelsPipeline)

    console.log('Done', performance.now() - start)

    return new Voxel(
      computePipeline,
      computeCornersPipeline,
      uniformBuffer,
      cornerMaterials,
      cornerMaterialsRead,
      voxelMaterialsBuffer,
      voxelMaterialsBufferRead,
      cornerIndexBuffer,
      gpuReadBuffer,
      permutationsBuffer,
      voxelsBuffer,
      computeBindGroup,
      computeCornersBindGroup,
      computePositionsPipeline,
      computePositionsBindGroup,
      computeVoxelsPipeline,
      computeVoxelsBindGroup,
      voxelReadBuffer,
      density,
      densityBindGroup,
      mainDensityBindGroup
    )
  }

  generate(
    device,
    queue,
    position,
    stride,
    density
  ): Promise<{
    vertices: Float32Array
    normals: Float32Array
    indices: Uint16Array
    corners: Uint32Array
    consistency: number
  }> {
    if (!stride) stride = 1

    return new Promise((resolve) => {
      this.density.updateRaw(device, density)

      const permutations = new Int32Array(512)

      const random = new Random('James')
      for (let i = 0; i < 256; i++) permutations[i] = 256 * random()

      for (let i = 256; i < 512; i++) permutations[i] = permutations[i - 256]

      device.queue.writeBuffer(
        this.permutationsBuffer,
        0,
        permutations.buffer,
        permutations.byteOffset,
        permutations.byteLength
      )

      const buffer = new ArrayBuffer(4 * 5)
      const uniform = new Float32Array(buffer, 0, 4)
      uniform.set(position, 0)
      uniform[3] = stride

      new Uint32Array(buffer, 16, 1)[0] = 33

      device.queue.writeBuffer(this.uniformBuffer, 0, buffer, 0, buffer.byteLength)

      const computeEncoder = device.createCommandEncoder()
      const octreeSize = 32
      const computePassEncoder = computeEncoder.beginComputePass()
      computePassEncoder.setPipeline(this.computePipeline)
      computePassEncoder.setBindGroup(0, this.computeBindGroup)
      this.densityBindGroup.apply(computePassEncoder)
      computePassEncoder.dispatchWorkgroups(octreeSize + 1, octreeSize + 1, octreeSize + 1)
      computePassEncoder.end()

      const computeCornersPass = computeEncoder.beginComputePass()
      computeCornersPass.setPipeline(this.computeCornersPipeline)
      computeCornersPass.setBindGroup(0, this.computeCornersBindGroup)
      computeCornersPass.dispatchWorkgroups(octreeSize, octreeSize, octreeSize)
      computeCornersPass.end()

      const computePositionsPass = computeEncoder.beginComputePass()
      computePositionsPass.setPipeline(this.computePositionsPipeline)
      computePositionsPass.setBindGroup(0, this.computePositionsBindGroup)
      computePositionsPass.dispatchWorkgroups(1)
      computePositionsPass.end()

      const copyEncoder = device.createCommandEncoder()
      copyEncoder.copyBufferToBuffer(
        this.cornerIndexBuffer,
        0,
        this.gpuReadBuffer,
        0,
        Uint32Array.BYTES_PER_ELEMENT
      )

      copyEncoder.copyBufferToBuffer(
        this.cornerMaterials,
        0,
        this.cornerMaterialsRead,
        0,
        Uint32Array.BYTES_PER_ELEMENT * 33 * 33 * 33
      )

      copyEncoder.copyBufferToBuffer(
        this.voxelMaterialsBuffer,
        0,
        this.voxelMaterialsBufferRead,
        0,
        Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32
      )
      queue({
        items: [computeEncoder.finish(), copyEncoder.finish()],
        callback: async () => {
          await this.cornerMaterialsRead.mapAsync(GPUMapMode.READ)
          const corners = new Uint32Array(this.cornerMaterialsRead.getMappedRange()).slice()
          this.cornerMaterialsRead.unmap()

          await this.gpuReadBuffer.mapAsync(GPUMapMode.READ)

          const arrayBuffer = this.gpuReadBuffer.getMappedRange()
          const voxelCount = new Uint32Array(arrayBuffer)[0]
          this.gpuReadBuffer.unmap()

          if (voxelCount === 0) {
            resolve({
              vertices: new Float32Array(),
              normals: new Float32Array(),
              indices: new Uint16Array(),
              corners: corners,
              consistency: corners[0]
            })
            return
          }

          const dispatchCount = Math.ceil(voxelCount / 128)
          const computeEncoder = device.createCommandEncoder()
          const computePassEncoder = computeEncoder.beginComputePass()
          computePassEncoder.setPipeline(this.computeVoxelsPipeline)
          computePassEncoder.setBindGroup(0, this.computeVoxelsBindGroup)
          this.mainDensityBindGroup.apply(computePassEncoder)
          computePassEncoder.dispatchWorkgroups(dispatchCount)
          computePassEncoder.end()

          const copyEncoder = device.createCommandEncoder()
          copyEncoder.copyBufferToBuffer(
            this.voxelsBuffer,
            0,
            this.voxelReadBuffer,
            0,
            Float32Array.BYTES_PER_ELEMENT * voxelCount * 12
          )

          queue({
            items: [computeEncoder.finish(), copyEncoder.finish()],
            callback: async () => {
              await this.voxelReadBuffer.mapAsync(GPUMapMode.READ)

              const arrayBuffer = this.voxelReadBuffer.getMappedRange()
              const computedVoxelsData = new Float32Array(arrayBuffer)
              const result = computeVoxels(position, stride, voxelCount, computedVoxelsData)

              this.voxelReadBuffer.unmap()

              resolve({ ...result, corners, consistency: -1 })
            }
          })
        }
      })
    })
  }
}
