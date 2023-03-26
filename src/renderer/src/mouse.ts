interface Point {
  x: number
  y: number
}

export default class Mouse {
  public position: Point
  private previousTouch: Touch | null = null

  constructor() {
    this.position = { x: 0, y: 0 }
  }

  init(): void {
    const mousemove = (e: MouseEvent): void => {
      this.position.x += e.movementX
      this.position.y += e.movementY
    }

    const touchMove = (e: TouchEvent): void => {
      e.preventDefault()
      const touch = e.touches[0]

      if (this.previousTouch) {
        this.position.x += this.previousTouch.screenX - touch.screenX
        this.position.y += this.previousTouch.screenY - touch.screenY
      }
      this.previousTouch = touch
    }

    document.addEventListener('mousemove', mousemove)
    document.addEventListener('touchmove', touchMove)
    document.addEventListener('touchend', () => {
      this.previousTouch = null
    })
  }

  update(): void {
    this.position.x = 0
    this.position.y = 0
  }
}
