import { Point } from './mouse'

export default class TouchController {
  public position: Point
  public primaryTrigger = false
  private leftMoved = false
  private previousTouch: Touch | null = null

  constructor() {
    this.position = { x: 0, y: 0 }
  }

  init(): void {
    const touchMove = (e: TouchEvent): void => {
      const touch = e.touches[0]
      console.log('Move')

      this.leftMoved = true

      if (this.previousTouch) {
        this.position.x += this.previousTouch.screenX - touch.screenX
        this.position.y += this.previousTouch.screenY - touch.screenY
      }
      this.previousTouch = touch
    }

    document.addEventListener('touchmove', touchMove)
    document.addEventListener('touchstart', (e) => {
      console.log(e)
    })
    document.addEventListener('touchend', () => {
      this.previousTouch = null
      if (!this.leftMoved) {
        this.primaryTrigger = true
      }
      this.leftMoved = false
    })
  }

  update(): void {
    this.position.x = 0
    this.position.y = 0
    this.primaryTrigger = false
  }
}
