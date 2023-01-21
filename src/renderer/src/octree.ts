import { vec3 } from 'gl-matrix'
import { Node } from './voxel'

const CHILD_MIN_OFFSETS = [
  vec3.fromValues(0, 0, 0),
  vec3.fromValues(0, 0, 1),
  vec3.fromValues(0, 1, 0),
  vec3.fromValues(0, 1, 1),
  vec3.fromValues(1, 0, 0),
  vec3.fromValues(1, 0, 1),
  vec3.fromValues(1, 1, 0),
  vec3.fromValues(1, 1, 1)
]

const constructParents = (children, position, parentSize): Node[] => {
  const parentsHash: Map<string, Node> = new Map<string, Node>()
  for (let i = 0; i < children.length; i++) {
    const node = children[i]
    const parentPos = vec3.sub(
      vec3.create(),
      node.min,
      vec3.fromValues(
        (node.min[0] - position[0]) % parentSize,
        (node.min[1] - position[1]) % parentSize,
        (node.min[2] - position[2]) % parentSize
      )
    )

    let parent = parentsHash[`${parentPos[0]},${parentPos[1]},${parentPos[2]}`]
    if (!parent) {
      parent = {
        min: parentPos,
        size: parentSize,
        type: 'internal',
        children: []
      }
      parentsHash[`${parent.min[0]},${parent.min[1]},${parent.min[2]}`] = parent
    }

    for (let j = 0; j < 8; j++) {
      const childMin = vec3.add(
        vec3.create(),
        parentPos,
        vec3.fromValues(
          CHILD_MIN_OFFSETS[j][0] * node.size,
          CHILD_MIN_OFFSETS[j][1] * node.size,
          CHILD_MIN_OFFSETS[j][2] * node.size
        )
      )

      if (vec3.equals(childMin, node.min)) {
        parent.children[j] = node
        break
      }
    }
  }

  children.length = 0

  return Object.values(parentsHash)
}

const constructTreeUpwards = (nodes, rootMin, rootNodeSize): Node | null => {
  if (nodes.length == 0) {
    return null
  }

  nodes.sort((lhs, rhs) => lhs.size - rhs.size)

  // the input nodes may be different sizes if a seam octree is being constructed
  // in that case we need to process the input nodes in stages along with the newly
  // constructed parent nodes until all the nodes have the same size
  while (nodes[0].size != nodes[nodes.length - 1].size) {
    // find the end of this run
    let iter = 0
    const size = nodes[iter].size
    do {
      ++iter
    } while (nodes[iter].size == size)

    // construct the new parent nodes for this run
    let newNodes: unknown[] = []
    for (let i = 0; i < iter; i++) newNodes.push(nodes[i])
    newNodes = constructParents(newNodes, rootMin, size * 2)

    // set up for the next iteration: the parents produced plus any remaining input nodes
    for (let i = iter; i < nodes.Count; i++) newNodes.push(nodes[i])

    nodes.length = 0
    for (let i = 0; i < newNodes.length; i++) nodes.push(newNodes[i])
  }

  let parentSize = nodes[0].size * 2
  while (parentSize <= rootNodeSize) {
    nodes = constructParents(nodes, rootMin, parentSize)
    parentSize *= 2
  }

  if (nodes.length != 1) {
    console.log(nodes)
    console.error('There can only be one root node!')
    return null
  }

  return nodes[0]
}

export default constructTreeUpwards
