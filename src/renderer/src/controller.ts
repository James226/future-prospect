import { glMatrix, mat4, quat, vec3 } from 'gl-matrix'
import Keyboard from './keyboard'
import Mouse from './mouse'
import * as Tone from 'tone'
import Raycast from './raycast'

export default class Controller {
  public viewMatrix: mat4
  public position: vec3
  public velocity: vec3

  private keyboard: Keyboard
  private mouse: Mouse

  private readonly forward: vec3
  private readonly up: vec3
  private readonly right: vec3
  private readonly rotation: quat

  private noise: Tone.Noise | null

  constructor(keyboard: Keyboard, mouse: Mouse) {
    this.keyboard = keyboard
    this.mouse = mouse

    this.viewMatrix = mat4.create()
    this.position = vec3.fromValues(0, 0.0, -300.0)
    this.velocity = vec3.fromValues(0, 0, 0)
    this.rotation = quat.create()

    this.forward = vec3.create()
    this.right = vec3.create()
    this.up = vec3.create()

    this.noise = null
  }

  init(): void {
    this.noise = new Tone.Noise('pink')

    const dist = new Tone.AutoFilter({
      frequency: '4n',
      baseFrequency: 200,
      depth: 0
    })
      .toDestination()
      .start()
    this.noise.connect(dist)
  }

  update(device: GPUDevice, projectionMatrix: mat4, queue, raycast: Raycast): void {
    const distance = this.keyboard.keydown('shift') ? 20 : 5
    vec3.zero(this.velocity)
    if (this.keyboard.keydown('w')) {
      vec3.sub(this.velocity, this.velocity, this.forward)
    }

    if (this.keyboard.keydown('s')) {
      vec3.add(this.velocity, this.velocity, this.forward)
    }

    if (this.keyboard.keydown('a')) {
      vec3.sub(this.velocity, this.velocity, this.right)
    }

    if (this.keyboard.keydown('d')) {
      vec3.add(this.velocity, this.velocity, this.right)
    }

    if (this.keyboard.keydown('r')) {
      vec3.add(this.velocity, this.velocity, this.up)
    }

    if (this.keyboard.keydown('f')) {
      vec3.sub(this.velocity, this.velocity, this.up)
    }

    vec3.scale(this.velocity, this.velocity, distance)

    if (vec3.length(this.velocity) > 0) {
      if (this.noise !== null && this.noise.state === 'stopped') {
        //this.noise.start();
      }
    } else {
      if (this.noise != null && this.noise.state === 'started') {
        //this.noise.stop();
      }
    }

    if (this.keyboard.keydown('q')) {
      quat.rotateZ(this.rotation, this.rotation, glMatrix.toRadian(1.0))
    }

    if (this.keyboard.keydown('e')) {
      quat.rotateZ(this.rotation, this.rotation, glMatrix.toRadian(-1.0))
    }

    if (this.keyboard.keydown('1')) {
      document.getElementById('tool')!.innerText = 'true'
    }

    if (this.keyboard.keydown('`')) {
      document.getElementById('tool')!.innerText = 'false'
    }

    if (this.keyboard.keypress(' ')) {
      raycast
        .cast(device, queue, this.position, vec3.scale(vec3.create(), this.forward, -1))
        .then((r) =>
          r !== null ? console.log(r.position, r.distance) : console.log('No intersection found')
        )
    }

    quat.rotateX(this.rotation, this.rotation, glMatrix.toRadian(-this.mouse.position.y * 0.08))
    quat.rotateY(this.rotation, this.rotation, glMatrix.toRadian(-this.mouse.position.x * 0.08))

    mat4.identity(this.viewMatrix)
    const rotMat = mat4.fromQuat(mat4.create(), this.rotation)
    mat4.translate(this.viewMatrix, this.viewMatrix, this.position)

    const gravity = vec3.fromValues(2000000.0, 0.0, 0.0)
    const gravityDirection = vec3.sub(vec3.create(), this.position, gravity)
    vec3.normalize(gravityDirection, gravityDirection)

    mat4.translate(
      this.viewMatrix,
      this.viewMatrix,
      vec3.scale(gravityDirection, gravityDirection, 100)
    )

    mat4.mul(this.viewMatrix, this.viewMatrix, rotMat)
    mat4.invert(this.viewMatrix, this.viewMatrix)

    const inverted = mat4.invert(mat4.create(), this.viewMatrix)
    vec3.normalize(this.right, <vec3>inverted.slice(0, 3))
    vec3.normalize(this.up, <vec3>inverted.slice(4, 7))
    vec3.normalize(this.forward, <vec3>inverted.slice(8, 11))

    mat4.multiply(this.viewMatrix, projectionMatrix, this.viewMatrix)
  }
}
