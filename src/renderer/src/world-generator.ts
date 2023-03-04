export interface WorldGeneratorInfo {
  x: number
  y: number
  z: number
  stride: number
  iteration: number
  layer: number
  previousOffset: number
}

export interface WorldPosition extends Point {
  stride: number
}

interface Point {
  x: number
  y: number
  z: number
}

function fixPoint(i: number): number {
  if (i >= 0) return i + 1
  return i
}

export function generatePoints(radius: number): Point[] {
  const points: Point[] = []

  const diameter = radius * 2

  for (let x = 0; x < diameter; x++)
    for (let y = 0; y < diameter; y++)
      for (let z = 0; z < diameter; z++) {
        if (
          !(
            x === 0 ||
            x === diameter - 1 ||
            y === 0 ||
            y === diameter - 1 ||
            z === 0 ||
            z === diameter - 1
          )
        ) {
          continue
        }
        const point = {
          x: fixPoint(x - radius),
          y: fixPoint(y - radius),
          z: fixPoint(z - radius)
        }

        points.push(point)
      }

  return points
}

const points: Point[][] = []

for (let i = 0; i <= 6; i++) {
  points[i] = generatePoints(i)
}

export default class WorldGenerator {
  private readonly minStride: number

  constructor(minStride: number) {
    this.minStride = minStride
  }

  init(x: number, y: number, z: number): WorldGeneratorInfo {
    return {
      x: Math.round(x / this.minStride) * this.minStride,
      y: Math.round(y / this.minStride) * this.minStride,
      z: Math.round(z / this.minStride) * this.minStride,
      stride: this.minStride,
      iteration: 0,
      layer: 1,
      previousOffset: (-1 * this.minStride) / 2
    }
  }

  layerRadius(layer: number, stride: number): number {
    if (stride === this.minStride) {
      return layer
    }
    return 2
    // if (stride === this.minStride * 2) {
    //   return layer + 1
    // }
    // return layer + 2
  }

  radius(layer: number, stride: number, previousOffset: number): number {
    return Math.sign(layer) * (previousOffset + Math.abs(layer) * stride)
  }

  layerCount(stride: number): number {
    //if (stride == this.minStride * 2) return 3
    if (stride == this.minStride) return 2
    return 1
  }

  next(info: WorldGeneratorInfo): [WorldPosition, WorldGeneratorInfo] {
    const offsets = points[this.layerRadius(info.layer, info.stride)]

    const offset = offsets[info.iteration]
    const halfStride = info.stride / 2

    let iteration = info.iteration + 1
    let layer = info.layer
    let stride = info.stride
    let previousOffset = info.previousOffset

    if (iteration >= offsets.length) {
      iteration = 0
      layer++

      const layerCount = this.layerCount(info.stride)
      if (layer > layerCount) {
        layer = 1
        stride *= 2
        previousOffset = this.radius(info.layer, info.stride, info.previousOffset) - info.stride / 2
      }
    }
    return [
      {
        x: info.stride * offset.x + info.x - Math.sign(offset.x) * halfStride,
        y: info.stride * offset.y + info.y - Math.sign(offset.y) * halfStride,
        z: info.stride * offset.z + info.z - Math.sign(offset.z) * halfStride,
        stride: info.stride
      },
      { ...info, iteration, layer, stride, previousOffset }
    ]
  }
}
