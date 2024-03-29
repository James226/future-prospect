export default class Keyboard {
  private keys: Map<string, boolean>
  private readonly bufferKeys: Map<string, boolean>
  private lastKeys: Map<string, boolean>

  constructor() {
    this.keys = new Map<string, boolean>()
    this.bufferKeys = new Map<string, boolean>()
    this.lastKeys = new Map<string, boolean>()
  }

  init(): void {
    const keydown = (e: KeyboardEvent): void => {
      e.preventDefault()
      this.bufferKeys.set(e.key.toLowerCase(), true)
    }

    const keyup = ({ key }: KeyboardEvent): void => {
      this.bufferKeys.delete(key.toLowerCase())
    }


    const mousedown = (e: MouseEvent): void => {
      e.preventDefault()
      this.bufferKeys.set(`mouse-${e.button}`, true)
    }

    const mouseup = ({ button }: MouseEvent): void => {
      this.bufferKeys.delete(`mouse-${button}`)
    }

    document.addEventListener('keydown', keydown)
    document.addEventListener('keyup', keyup)
    document.addEventListener('mousedown', mousedown)
    document.addEventListener('mouseup', mouseup)
  }

  update(): void {
    this.lastKeys = new Map(this.keys)
    this.keys = new Map(this.bufferKeys)
  }

  keydown(key: string): boolean {
    return this.keys.get(key) === true
  }

  keyup(key: string): boolean {
    return this.keys.get(key) !== true
  }

  keypress(key: string): boolean {
    return this.keys.get(key) === true && this.lastKeys.get(key) !== true
  }
}
