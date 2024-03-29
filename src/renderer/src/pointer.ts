import { mat4, vec3 } from 'gl-matrix'
import {
  cubePositionOffset,
  cubeUVOffset,
  cubeVertexArray,
  cubeVertexCount,
  cubeVertexSize
} from './cube'
import vertex from './playerVertex.wgsl?raw'
import fragment from './playerFragment.wgsl?raw'
import { Camera } from './camera'
import Controller from './controller'
import Raycast from './raycast'

export default class Pointer {
  public position: vec3
  private readonly vertexBuffer: GPUBuffer
  private readonly uniformBuffer: GPUBuffer
  private readonly uniformBindGroup: GPUBindGroup
  private readonly pipeline: GPURenderPipeline
  private updating: boolean
  public size = 32
  public snapToGrid = true

  constructor(
    device: GPUDevice,
    private controller: Controller,
    private camera: Camera,
    private raycast: Raycast
  ) {
    this.position = vec3.create()
    this.updating = false

    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: device.createShaderModule({
          code: vertex
        }),
        entryPoint: 'main',
        buffers: [
          {
            arrayStride: cubeVertexSize,
            attributes: [
              {
                // position
                shaderLocation: 0,
                offset: cubePositionOffset,
                format: 'float32x4'
              },
              {
                // uv
                shaderLocation: 1,
                offset: cubeUVOffset,
                format: 'float32x2'
              }
            ]
          }
        ]
      },
      fragment: {
        module: device.createShaderModule({
          code: fragment
        }),
        entryPoint: 'main',
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat()
          }
        ]
      },
      primitive: {
        topology: 'triangle-list',

        // Backface culling since the cube is solid piece of geometry.
        // Faces pointing away from the camera will be occluded by faces
        // pointing toward the camera.
        cullMode: 'back'
      },

      // Enable depth testing so that the fragment closest to the camera
      // is rendered in front.
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus-stencil8'
      }
    })

    this.vertexBuffer = device.createBuffer({
      size: cubeVertexArray.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true
    })
    new Float32Array(this.vertexBuffer.getMappedRange()).set(cubeVertexArray)
    this.vertexBuffer.unmap()

    const uniformBufferSize = 4 * 16
    this.uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    this.uniformBindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer
          }
        }
      ]
    })
  }

  getTransformationMatrix(projectionMatrix): mat4 {
    const modelMatrix = mat4.create()
    mat4.translate(modelMatrix, modelMatrix, this.position)
    mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(10.0, 10.0, 10.0))

    const modelViewProjectionMatrix = mat4.create()
    mat4.multiply(modelViewProjectionMatrix, projectionMatrix, modelMatrix)

    return modelViewProjectionMatrix
  }

  update(device: GPUDevice, queue, projectionMatrix: mat4): void {
    if (!this.updating) {
      this.updating = true
      const gravityDirection = vec3.create()
      vec3.scale(gravityDirection, this.controller.up, 100)
      vec3.add(gravityDirection, this.controller.position, gravityDirection)

      this.raycast
        .cast(device, queue, gravityDirection, vec3.scale(vec3.create(), this.camera.forward, -1))
        .then((r) => {
          this.updating = false
          if (r === null) {
            return
          }
          vec3.copy(this.position, r.position)
          if (this.snapToGrid) {
            const augmentationSize = this.size * 2
            this.position[0] = Math.round(this.position[0] / augmentationSize) * augmentationSize
            this.position[1] = Math.round(this.position[1] / augmentationSize) * augmentationSize
            this.position[2] = Math.round(this.position[2] / augmentationSize) * augmentationSize
          }
          //this.position = vec3.add(r.position, gravityDirection, r.position)
        })
    }

    const transformationMatrix = this.getTransformationMatrix(projectionMatrix)

    device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      (<Float32Array>transformationMatrix).buffer,
      (<Float32Array>transformationMatrix).byteOffset,
      (<Float32Array>transformationMatrix).byteLength
    )
  }

  draw(passEncoder: GPURenderPassEncoder): void {
    passEncoder.setPipeline(this.pipeline)
    passEncoder.setBindGroup(0, this.uniformBindGroup)
    passEncoder.setVertexBuffer(0, this.vertexBuffer)
    passEncoder.draw(cubeVertexCount, 1, 0, 0)
  }
}
