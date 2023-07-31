import { glMatrix, mat4, vec3 } from 'gl-matrix'
import Controller from './controller'
import Mouse from './mouse'
import TouchController from './touch-controller'

export class Camera {
  public forward: vec3
  public viewMatrix: mat4
  private rotation: number

  constructor(
    private controller: Controller,
    private mouse: Mouse,
    private touchController: TouchController
  ) {
    this.viewMatrix = mat4.create()
    this.rotation = 0
    this.forward = vec3.create()
  }

  update(projectionMatrix: mat4): void {
    const y = this.mouse.position.y + this.touchController.right.position.y
    this.rotation -= glMatrix.toRadian(y * 0.08)
    this.rotation = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation))

    const transformMatrix = this.controller.getTransformMatrix()

    mat4.copy(this.viewMatrix, transformMatrix)
    mat4.rotateX(this.viewMatrix, this.viewMatrix, this.rotation)
    vec3.normalize(this.forward, <vec3>this.viewMatrix.slice(8, 11))

    mat4.invert(this.viewMatrix, this.viewMatrix)

    mat4.multiply(this.viewMatrix, projectionMatrix, this.viewMatrix)
  }
}
