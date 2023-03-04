import { glMatrix, mat4 } from 'gl-matrix'
import Controller from './controller'
import Mouse from './mouse'

export class Camera {
  public viewMatrix: mat4
  private rotation: number

  constructor(private controller: Controller, private mouse: Mouse) {
    this.viewMatrix = mat4.create()
    this.rotation = 0
  }

  update(projectionMatrix: mat4): void {
    this.rotation -= glMatrix.toRadian(this.mouse.position.y * 0.08)
    this.rotation = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation))

    const transformMatrix = this.controller.getTransformMatrix()

    mat4.copy(this.viewMatrix, transformMatrix)
    mat4.rotateX(this.viewMatrix, this.viewMatrix, this.rotation)
    mat4.invert(this.viewMatrix, this.viewMatrix)
    mat4.multiply(this.viewMatrix, projectionMatrix, this.viewMatrix)
  }
}
