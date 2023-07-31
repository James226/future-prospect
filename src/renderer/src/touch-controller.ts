import { Point } from './mouse'

class Joystick {
  public position: Point = { x: 0, y: 0 }
  public moved = false
  public startTouch: Touch | null = null
  public previousTrigger = false
  public trigger = false
}

export default class TouchController {
  public left: Joystick = new Joystick()
  public right: Joystick = new Joystick()
  public joystickMappings = new Map<number, Joystick>()

  init(): void {
    const touchMove = (e: TouchEvent): void => {
      for (let i = 0; i < e.touches.length; i++) {
        const touch = e.touches[i]
        const joystick = this.joystickMappings.get(touch.identifier)
        if (!joystick) continue

        if (joystick.startTouch) {
          joystick.position.x = touch.screenX - joystick.startTouch.screenX
          joystick.position.x -= Math.sign(joystick.position.x) * 10
          if (Math.abs(joystick.position.x) < 10) joystick.position.x = 0
          joystick.position.y = touch.screenY - joystick.startTouch.screenY
          joystick.position.y -= Math.sign(joystick.position.y) * 10
          if (Math.abs(joystick.position.y) < 10) joystick.position.y = 0

          if (joystick.position.x !== 0 || joystick.position.y !== 0) {
            joystick.moved = true
          }
        }
      }
    }

    document.addEventListener('touchmove', touchMove)
    document.addEventListener('touchstart', (e) => {

      const halfWidth = window.screen.width / 2
      for (let i = 0; i < e.changedTouches.length; i++) {
        const touch = e.changedTouches[i]

        if (this.joystickMappings.has(touch.identifier)) continue

        let joystick: Joystick | null = null
        if (touch.screenX < halfWidth) {
          joystick = this.left
        } else {
          joystick = this.right
        }
        this.joystickMappings.set(touch.identifier, joystick)
        joystick.startTouch = touch
      }
    })
    document.addEventListener('touchend', (e) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const touch = e.changedTouches[i]
        const joystick = this.joystickMappings.get(touch.identifier)
        this.joystickMappings.delete(touch.identifier)
        if (!joystick) continue

        if (!joystick.moved) {
          joystick.trigger = true
        }
        joystick.startTouch = null
        joystick.moved = false
        joystick.position.x = 0
        joystick.position.y = 0
      }
    })
  }

  update(): void {
    this.left.previousTrigger = this.left.trigger
    this.left.trigger = false

    this.right.previousTrigger = this.right.trigger
    this.right.trigger = false
  }

  trigger(): boolean {
    return this.right.previousTrigger && !this.right.trigger
  }
}
