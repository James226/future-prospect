const OctreeSize = 32u;

struct CornerMaterials {
  cornerMaterials : array<u32>,
};
@binding(1) @group(0) var<storage, read> cornerMaterials: CornerMaterials;

struct VoxelMaterials {
  voxelMaterials : array<u32>,
};
@binding(2) @group(0) var<storage, read_write> voxelMaterials: VoxelMaterials;

const CHILD_MIN_OFFSETS = array<vec3<u32>, 8>
(
  vec3<u32>(0u, 0u, 0u),
  vec3<u32>(0u, 0u, 1u),
  vec3<u32>(0u, 1u, 0u),
  vec3<u32>(0u, 1u, 1u),
  vec3<u32>(1u, 0u, 0u),
  vec3<u32>(1u, 0u, 1u),
  vec3<u32>(1u, 1u, 0u),
  vec3<u32>(1u, 1u, 1u)
);

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let index: u32 = GlobalInvocationID.z * 32u * 32u + GlobalInvocationID.y * 32u + GlobalInvocationID.x;

  let nodePos: vec3<u32> = vec3<u32>(GlobalInvocationID.x, GlobalInvocationID.y, GlobalInvocationID.z);
  var corners: u32 = 0u;

  var j: u32 = 0u;
  loop {
    if (j >= 8u) { break; }

    let cornerPos: vec3<u32> = vec3<u32>(GlobalInvocationID.x + CHILD_MIN_OFFSETS[j].x, GlobalInvocationID.y + CHILD_MIN_OFFSETS[j].y, GlobalInvocationID.z + CHILD_MIN_OFFSETS[j].z);
    let material: u32 = min(1, cornerMaterials.cornerMaterials[cornerPos.z * 33u * 33u + cornerPos.y * 33u + cornerPos.x]);
    corners = corners | (material << j);

    continuing {
      j = j + 1u;
    }
  }
  
  voxelMaterials.voxelMaterials[index] = corners;
}