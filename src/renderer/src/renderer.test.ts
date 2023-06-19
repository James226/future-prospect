import { describe, expect, it, beforeEach } from 'vitest'

import Renderer from './renderer'

describe('Renderer', () => {
  let canvas

  beforeEach(() => {
    canvas = document.createElement('canvas')
  })

  it('should not throw when initialized', async () => {
    const renderer = await Renderer.init(canvas)
    expect(() => renderer.configure(canvas.width, canvas.height)).not.toThrow()
    expect(() => renderer.render(() => {})).not.toThrow()
  })
})
