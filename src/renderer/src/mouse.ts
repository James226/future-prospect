interface Point {
  x: number
  y: number
}

export default class Mouse {
  public position: Point

  constructor() {
    this.position = { x: 0, y: 0 }
  }

  init(): void {
    const mousemove = (e: MouseEvent): void => {
      this.position.x += e.movementX
      this.position.y += e.movementY
    }

    document.addEventListener('mousemove', mousemove)
  }

  update(): void {
    this.position.x = 0
    this.position.y = 0
  }
}
