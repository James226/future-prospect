import { describe, expect, it, beforeEach } from 'vitest'
import WorldGenerator, { generatePoints } from './world-generator'

describe('World Generator', () => {
  let worldGenerator: WorldGenerator

  beforeEach(() => {
    worldGenerator = new WorldGenerator(8)
  })

  describe('when init is called', () => {
    let info
    beforeEach(() => {
      info = worldGenerator.init(3, 20, 43)
    })

    it('position is rounded to nearest stride', async () => {
      expect(info.x).toEqual(0)
      expect(info.y).toEqual(24)
      expect(info.z).toEqual(40)
    })

    it('current stride defaults to minimum stride', () => {
      expect(info.stride).toEqual(8)
    })

    it('previousOffset is correct', () => {
      expect(info.previousOffset).toEqual(-4)
    })

    it('layer is correct', () => {
      expect(info.layer).toEqual(1)
    })
  })

  describe('when next is called', () => {
    it('returns origin position', () => {
      const [result, info] = worldGenerator.next({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 0,
        layer: 1,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: -4,
        y: -4,
        z: -4,
        stride: 8
      })

      expect(info).toEqual({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 1,
        layer: 1,
        previousOffset: -4
      })
    })

    it('returns origin position for iteration 1', () => {
      const [result, info] = worldGenerator.next({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 1,
        layer: 1,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: -4,
        y: -4,
        z: 4,
        stride: 8
      })

      expect(info).toEqual({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 2,
        layer: 1,
        previousOffset: -4
      })
    })

    it('returns origin position for second layer', () => {
      const [result, info] = worldGenerator.next({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 0,
        layer: 2,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: -12,
        y: -12,
        z: -12,
        stride: 8
      })

      expect(info).toEqual({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 1,
        layer: 2,
        previousOffset: -4
      })
    })

    it('returns origin position for second layer second iteration', () => {
      const [result, info] = worldGenerator.next({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 6,
        layer: 2,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: -12,
        y: -4,
        z: 4,
        stride: 8
      })

      expect(info).toEqual({
        x: 0,
        y: 0,
        z: 0,
        stride: 8,
        iteration: 7,
        layer: 2,
        previousOffset: -4
      })
    })

    it('offsets position by root', () => {
      const [result, info] = worldGenerator.next({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 0,
        layer: 1,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: 36,
        y: 20,
        z: 28,
        stride: 8
      })

      expect(info).toEqual({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 1,
        layer: 1,
        previousOffset: -4
      })
    })

    it('offsets position by iteration', () => {
      const [result, info] = worldGenerator.next({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 1,
        layer: 1,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: 36,
        y: 20,
        z: 36,
        stride: 8
      })

      expect(info).toEqual({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 2,
        layer: 1,
        previousOffset: -4
      })
    })

    it('offsets position by last iteration', () => {
      const [result, info] = worldGenerator.next({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 7,
        layer: 1,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: 44,
        y: 28,
        z: 36,
        stride: 8
      })

      expect(info).toEqual({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 0,
        layer: 2,
        previousOffset: -4
      })
    })

    it('offsets position by last layer', () => {
      const [result, info] = worldGenerator.next({
        x: 40,
        y: 24,
        z: 32,
        stride: 8,
        iteration: 55,
        layer: 2,
        previousOffset: -4
      })

      expect(result).toEqual({
        x: 52,
        y: 36,
        z: 44,
        stride: 8
      })

      expect(info).toEqual({
        x: 40,
        y: 24,
        z: 32,
        stride: 16,
        iteration: 0,
        layer: 1,
        previousOffset: 8
      })
    })

    it('returns correct position for next stride', () => {
      const [result, info] = worldGenerator.next({
        x: 0,
        y: 0,
        z: 0,
        stride: 16,
        iteration: 0,
        layer: 1,
        previousOffset: 8
      })

      expect(result).toEqual({
        x: -24,
        y: -24,
        z: -24,
        stride: 16
      })

      expect(info).toEqual({
        x: 0,
        y: 0,
        z: 0,
        stride: 16,
        iteration: 1,
        layer: 1,
        previousOffset: 8
      })
    })

    it('returns correct position 32/0/1', () => {
      const [result, info] = worldGenerator.next({
        x: 0,
        y: 0,
        z: 0,
        stride: 32,
        iteration: 0,
        layer: 1,
        previousOffset: 48
      })

      expect(result).toEqual({
        x: -48,
        y: -48,
        z: -48,
        stride: 32
      })

      expect(info).toEqual({
        x: 0,
        y: 0,
        z: 0,
        stride: 32,
        iteration: 1,
        layer: 1,
        previousOffset: 48
      })
    })
  })

  describe('when radius is called', () => {
    const expectedResults = [
      { layer: 1, stride: 8, previousOffset: -4, result: 4 },
      { layer: 2, stride: 8, previousOffset: -4, result: 12 },
      { layer: 1, stride: 16, previousOffset: 8, result: 24 },
      { layer: 2, stride: 16, previousOffset: 8, result: 40 },
      { layer: -2, stride: 16, previousOffset: 8, result: -40 }
    ]

    for (let i = 0; i < expectedResults.length; i++) {
      const { layer, stride, previousOffset, result } = expectedResults[i]
      it(`then ring ${i + 1} is correct`, () => {
        expect(worldGenerator.radius(layer, stride, previousOffset)).toEqual(result)
      })
    }
  })

  describe('when layerRadius is called', () => {
    const expectedResults = [
      { layer: 1, stride: 8, result: 1 },
      { layer: 2, stride: 8, result: 2 },
      { layer: 1, stride: 16, result: 2 },
      { layer: 2, stride: 16, result: 2 },
      { layer: 3, stride: 16, result: 2 },
      { layer: 1, stride: 32, result: 2 },
      { layer: 2, stride: 32, result: 2 }
    ]

    for (let i = 0; i < expectedResults.length; i++) {
      const { layer, stride, result } = expectedResults[i]
      it(`then ring ${i + 1} is correct`, () => {
        expect(worldGenerator.layerRadius(layer, stride)).toEqual(result)
      })
    }
  })

  describe('when layerCount is called', () => {
    it('returns 2 for first stride', () => {
      expect(worldGenerator.layerCount(8)).toEqual(2)
    })

    it('returns correct for second stride', () => {
      expect(worldGenerator.layerCount(16)).toEqual(1)
    })

    it('returns correct for subsequent strides', () => {
      for (let i = 32; i < 1024; i += i) {
        expect(worldGenerator.layerCount(i)).toEqual(1)
      }
    })
  })

  describe('generate points', () => {
    it('returns the correct values', () => {
      expect(generatePoints(1)).toEqual([
        { x: -1, y: -1, z: -1 },
        { x: -1, y: -1, z: 1 },
        { x: -1, y: 1, z: -1 },
        { x: -1, y: 1, z: 1 },
        { x: 1, y: -1, z: -1 },
        { x: 1, y: -1, z: 1 },
        { x: 1, y: 1, z: -1 },
        { x: 1, y: 1, z: 1 }
      ])
    })

    it('returns the correct values 2', () => {
      expect(generatePoints(2).length).toEqual(56)
    })
  })
})
