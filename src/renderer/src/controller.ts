import { glMatrix, mat4, quat, vec3 } from 'gl-matrix'
import Keyboard from './keyboard'
import Mouse from './mouse'
import * as Tone from 'tone'
import Raycast from './raycast'

export default class Controller {
  public transformMatrix: mat4
  public position: vec3
  public velocity: vec3

  private readonly forward: vec3
  private readonly up: vec3
  private readonly right: vec3
  private readonly rotation: quat

  private noise: Tone.Noise | null

  constructor(private keyboard: Keyboard, private mouse: Mouse) {
    this.transformMatrix = mat4.create()
    this.position = vec3.fromValues(0, 0.0, -300.0)
    this.velocity = vec3.fromValues(0, 0, 0)
    this.rotation = quat.create()

    this.forward = vec3.create()
    this.right = vec3.create()
    this.up = vec3.fromValues(0, 1, 0)

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

  update(device: GPUDevice, queue, raycast: Raycast, deltaTime: number): void {
    const distance = this.keyboard.keydown('shift') ? 20 : 5
    vec3.zero(this.velocity)
    if (this.keyboard.keydown('w')) {
      vec3.sub(this.velocity, this.velocity, this.forward)
      vec3.add(this.velocity, this.velocity, this.up)
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

    const gravity = vec3.fromValues(2000000.0, 100.0, 100.0)
    const gravityDirection = vec3.sub(vec3.create(), this.position, gravity)
    vec3.normalize(gravityDirection, gravityDirection)
    const orientationDirection = quat.rotationTo(
      quat.create(),
      vec3.scale(vec3.create(), this.up, 1),
      vec3.scale(vec3.create(), gravityDirection, 1)
    )
    quat.multiply(orientationDirection, orientationDirection, this.rotation)
    quat.slerp(this.rotation, this.rotation, orientationDirection, 0.01 * deltaTime)

    quat.rotateY(this.rotation, this.rotation, glMatrix.toRadian(-this.mouse.position.x * 0.08))

    mat4.identity(this.transformMatrix)
    const rotMat = mat4.fromQuat(mat4.create(), this.rotation)
    mat4.translate(this.transformMatrix, this.transformMatrix, this.position)

    mat4.translate(
      this.transformMatrix,
      this.transformMatrix,
      vec3.scale(gravityDirection, gravityDirection, 100)
    )

    mat4.mul(this.transformMatrix, this.transformMatrix, rotMat)

    vec3.normalize(this.right, <vec3>this.transformMatrix.slice(0, 3))
    vec3.normalize(this.up, <vec3>this.transformMatrix.slice(4, 7))
    vec3.normalize(this.forward, <vec3>this.transformMatrix.slice(8, 11))
  }

  getTransformMatrix(): mat4 {
    return this.transformMatrix
  }
}
