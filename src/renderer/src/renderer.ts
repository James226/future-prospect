export default class Renderer {
  private context: GPUCanvasContext
  device: GPUDevice
  private readonly presentationFormat: GPUTextureFormat
  private depthTexture: GPUTexture | null = null

  private constructor(device, context, presentationFormat) {
    this.device = device
    this.context = context
    this.presentationFormat = presentationFormat
  }

  static async init(canvas: HTMLCanvasElement): Promise<Renderer> {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('Unable to acquire GPU adapter, is WebGPU enabled?')
    const device = await adapter.requestDevice()

    const context = canvas.getContext('webgpu')
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat()

    return new Renderer(device, context, presentationFormat);
  }

  configure(width: number, height: number): void {
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: 'opaque'
    })

    this.depthTexture = this.device.createTexture({
      size: { width, height },
      format: 'depth24plus-stencil8',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    })
  }

  render(callback: (GPURenderPassDescriptor) => void): void {
    const commandEncoder = this.device.createCommandEncoder()
    const textureView = this.context.getCurrentTexture().createView()

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 75 / 255, g: 0 / 255, b: 130 / 255, a: 1.0 },
          loadOp: 'clear' as const,
          storeOp: 'store' as const
        }
      ],
      depthStencilAttachment: {
        view: this.depthTexture!.createView(),

        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',

        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: 'store'
      }
    }

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor)
    callback(passEncoder)
    passEncoder.end()

    this.device.queue.submit([commandEncoder.finish()])
  }
}
