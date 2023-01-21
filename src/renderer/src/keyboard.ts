export default class Keyboard {
  private readonly keys: Map<string, boolean>
  private lastKeys: Map<string, boolean>

  constructor() {
    this.keys = new Map<string, boolean>()
    this.lastKeys = new Map<string, boolean>()
  }

  init(): void {
    const keydown = (e: KeyboardEvent): void => {
      e.preventDefault()
      this.keys.set(e.key.toLowerCase(), true)
    }

    const keyup = ({ key }: KeyboardEvent): void => {
      this.keys.delete(key.toLowerCase())
    }

    document.addEventListener('keydown', keydown)
    document.addEventListener('keyup', keyup)
  }

  update(): void {
    this.lastKeys = new Map(this.keys)
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
