import { Node } from './voxel'

const MATERIAL_AIR = 0

const cellProcFaceMask = [
  [0, 4, 0],
  [1, 5, 0],
  [2, 6, 0],
  [3, 7, 0],
  [0, 2, 1],
  [4, 6, 1],
  [1, 3, 1],
  [5, 7, 1],
  [0, 1, 2],
  [2, 3, 2],
  [4, 5, 2],
  [6, 7, 2]
]
const cellProcEdgeMask = [
  [0, 1, 2, 3, 0],
  [4, 5, 6, 7, 0],
  [0, 4, 1, 5, 1],
  [2, 6, 3, 7, 1],
  [0, 2, 4, 6, 2],
  [1, 3, 5, 7, 2]
]

const faceProcFaceMask = [
  [
    [4, 0, 0],
    [5, 1, 0],
    [6, 2, 0],
    [7, 3, 0]
  ],
  [
    [2, 0, 1],
    [6, 4, 1],
    [3, 1, 1],
    [7, 5, 1]
  ],
  [
    [1, 0, 2],
    [3, 2, 2],
    [5, 4, 2],
    [7, 6, 2]
  ]
]

const faceProcEdgeMask = [
  [
    [1, 4, 0, 5, 1, 1],
    [1, 6, 2, 7, 3, 1],
    [0, 4, 6, 0, 2, 2],
    [0, 5, 7, 1, 3, 2]
  ],
  [
    [0, 2, 3, 0, 1, 0],
    [0, 6, 7, 4, 5, 0],
    [1, 2, 0, 6, 4, 2],
    [1, 3, 1, 7, 5, 2]
  ],
  [
    [1, 1, 0, 3, 2, 0],
    [1, 5, 4, 7, 6, 0],
    [0, 1, 5, 0, 4, 1],
    [0, 3, 7, 2, 6, 1]
  ]
]

const edgeProcEdgeMask = [
  [
    [3, 2, 1, 0, 0],
    [7, 6, 5, 4, 0]
  ],
  [
    [5, 1, 4, 0, 1],
    [7, 3, 6, 2, 1]
  ],
  [
    [6, 4, 2, 0, 2],
    [7, 5, 3, 1, 2]
  ]
]

const processEdgeMask = [
  [3, 2, 1, 0],
  [7, 5, 6, 4],
  [11, 10, 9, 8]
]

const edgevmap = [
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7], // x-axis
  [0, 2],
  [1, 3],
  [4, 6],
  [5, 7], // y-axis
  [0, 1],
  [2, 3],
  [4, 5],
  [6, 7] // z-axis
]

const ContourProcessEdge = (node: Node[], dir: number, indices: number[]): void => {
  let minSize = 1000000 // arbitrary big number
  let minIndex = 0
  const indexes = [-1, -1, -1, -1]
  let flip = false
  const signChange = [false, false, false, false]

  for (let i = 0; i < 4; i++) {
    const edge = processEdgeMask[dir][i]
    const c1 = edgevmap[edge][0]
    const c2 = edgevmap[edge][1]

    const m1 = (node[i].drawInfo.corners >> c1) & 1
    const m2 = (node[i].drawInfo.corners >> c2) & 1

    if (node[i].size < minSize) {
      minSize = node[i].size
      minIndex = i
      flip = m1 !== MATERIAL_AIR
    }

    indexes[i] = node[i].drawInfo.index

    signChange[i] =
      (m1 === MATERIAL_AIR && m2 !== MATERIAL_AIR) || (m1 !== MATERIAL_AIR && m2 === MATERIAL_AIR)
  }

  if (signChange[minIndex]) {
    if (!flip) {
      indices.push(indexes[0])
      indices.push(indexes[1])
      indices.push(indexes[3])

      indices.push(indexes[0])
      indices.push(indexes[3])
      indices.push(indexes[2])
    } else {
      indices.push(indexes[0])
      indices.push(indexes[3])
      indices.push(indexes[1])

      indices.push(indexes[0])
      indices.push(indexes[2])
      indices.push(indexes[3])
    }
  }
}

const ContourEdgeProc = (node: Node[], dir: number, indices: number[]): void => {
  if (node[0] == null || node[1] == null || node[2] == null || node[3] == null) return

  if (
    node[0].type !== 'internal' &&
    node[1].type !== 'internal' &&
    node[2].type !== 'internal' &&
    node[3].type !== 'internal'
  ) {
    ContourProcessEdge(node, dir, indices)
  } else {
    for (let i = 0; i < 2; i++) {
      const edgeNodes: Node[] = []
      const c = [
        edgeProcEdgeMask[dir][i][0],
        edgeProcEdgeMask[dir][i][1],
        edgeProcEdgeMask[dir][i][2],
        edgeProcEdgeMask[dir][i][3]
      ]

      for (let j = 0; j < 4; j++) {
        if (node[j].type === 'leaf' || node[j].type === 'pseudo') {
          edgeNodes[j] = node[j]
        } else {
          edgeNodes[j] = node[j].children[c[j]]
        }
      }

      ContourEdgeProc(edgeNodes, edgeProcEdgeMask[dir][i][4], indices)
    }
  }
}

const ContourFaceProc = (node: Node[], dir: number, indices: number[]): void => {
  if (node[0] == null || node[1] == null) return

  if (node[0].type === 'internal' || node[1].type === 'internal') {
    for (let i = 0; i < 4; i++) {
      const faceNodes: Node[] = []

      const c = [faceProcFaceMask[dir][i][0], faceProcFaceMask[dir][i][1]]

      for (let j = 0; j < 2; j++) {
        if (node[j].type !== 'internal') {
          faceNodes[j] = node[j]
        } else {
          faceNodes[j] = node[j].children[c[j]]
        }
      }

      ContourFaceProc(faceNodes, faceProcFaceMask[dir][i][2], indices)
    }

    const orders = [
      [0, 0, 1, 1],
      [0, 1, 0, 1]
    ]

    for (let i = 0; i < 4; i++) {
      const edgeNodes: Node[] = []
      const c = [
        faceProcEdgeMask[dir][i][1],
        faceProcEdgeMask[dir][i][2],
        faceProcEdgeMask[dir][i][3],
        faceProcEdgeMask[dir][i][4]
      ]

      const order = [
        orders[faceProcEdgeMask[dir][i][0]][0],
        orders[faceProcEdgeMask[dir][i][0]][1],
        orders[faceProcEdgeMask[dir][i][0]][2],
        orders[faceProcEdgeMask[dir][i][0]][3]
      ]

      for (let j = 0; j < 4; j++) {
        if (node[order[j]].type === 'leaf' || node[order[j]].type === 'pseudo') {
          edgeNodes[j] = node[order[j]]
        } else {
          edgeNodes[j] = node[order[j]].children[c[j]]
        }
      }

      ContourEdgeProc(edgeNodes, faceProcEdgeMask[dir][i][5], indices)
    }
  }
}

const ContourCellProc = (node: Node, indices: number[]): void => {
  if (node == null) return

  if (node.type === 'internal') {
    for (let i = 0; i < 8; i++) {
      ContourCellProc(node.children[i], indices)
    }

    for (let i = 0; i < 12; i++) {
      const faceNodes: Node[] = []
      const c = [cellProcFaceMask[i][0], cellProcFaceMask[i][1]]

      faceNodes[0] = node.children[c[0]]
      faceNodes[1] = node.children[c[1]]

      ContourFaceProc(faceNodes, cellProcFaceMask[i][2], indices)
    }

    for (let i = 0; i < 6; i++) {
      const edgeNodes: Node[] = []
      const c = [
        cellProcEdgeMask[i][0],
        cellProcEdgeMask[i][1],
        cellProcEdgeMask[i][2],
        cellProcEdgeMask[i][3]
      ]

      for (let j = 0; j < 4; j++) {
        edgeNodes[j] = node.children[c[j]]
      }

      ContourEdgeProc(edgeNodes, cellProcEdgeMask[i][4], indices)
    }
  }
}

export default ContourCellProc
