import { describe, expect, it, beforeAll } from 'vitest'

import ComputeVoxels from './compute-voxels.wgsl?raw'
// import Random from 'seedrandom'

describe('compute-voxels', () => {
  describe('computeMaterials', () => {
    let materials: Uint32Array

    beforeAll(async () => {
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter) throw new Error('Failed to request GPU adapter, is WebGPU enabled?')
      const device = await adapter.requestDevice()

      const density = `fn getDensity(worldPosition: vec3<f32>) -> f32 {
        if (worldPosition.y < 3.0) {
          return -1.0;
        } else {
          return 1.0;
        }
       }`

      const computeVoxels = ComputeVoxels.replace('#import density', density)
      const computePipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({
            code: computeVoxels
          }),
          entryPoint: 'computeMaterials'
        }
      })

      const permutations = new Int32Array(512)

      // const random = new Random(6452)
      // for (let i = 0; i < 256; i++) permutations[i] = 256 * random()
      //
      // for (let i = 256; i < 512; i++) permutations[i] = permutations[i - 256]

      const permutationsBuffer = device.createBuffer({
        size: permutations.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      })

      new Int32Array(permutationsBuffer.getMappedRange()).set(permutations)
      permutationsBuffer.unmap()

      const cornerMaterials = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT * 5 * 5 * 5,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      })

      const cornerMaterialsRead = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT * 5 * 5 * 5,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      })

      const uniformBufferSize = 32
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      })


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

      const computeEncoder = device.createCommandEncoder()
      const computePassEncoder = computeEncoder.beginComputePass()
      computePassEncoder.setPipeline(computePipeline)
      computePassEncoder.setBindGroup(0, computeBindGroup)
      computePassEncoder.dispatchWorkgroups(5, 5, 5)
      computePassEncoder.end()

      const copyEncoder = device.createCommandEncoder()
      copyEncoder.copyBufferToBuffer(
        cornerMaterials,
        0,
        cornerMaterialsRead,
        0,
        Uint32Array.BYTES_PER_ELEMENT * 5 * 5 * 5
      )

      const buffer = new ArrayBuffer(4 * 5)
      const uniform = new Float32Array(buffer, 0, 4)
      uniform[3] = 1

      new Uint32Array(buffer, 16, 1)[0] = 5

      device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength)

      const queue = device.queue.onSubmittedWorkDone()

      device.queue.submit([computeEncoder.finish(), copyEncoder.finish()])

      await queue

      await cornerMaterialsRead.mapAsync(GPUMapMode.READ)
      const arrayBuffer = cornerMaterialsRead.getMappedRange()
      materials = new Uint32Array(arrayBuffer).slice()
      cornerMaterialsRead.unmap()
    })

    it('then materials is correct', () => {
      const expectedResult = new Uint32Array(5 * 5 * 5)

      for (let x = 0; x < 5; x++)
        for (let y = 0; y < 5; y++)
          for (let z = 0; z < 5; z++) {
            expectedResult[z * 5 * 5 + y * 5 + x] = y < 3 ? 256 : 0
          }

      expect(materials).toEqual(expectedResult)
    })
  })
})
