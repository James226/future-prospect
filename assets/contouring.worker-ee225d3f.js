(function() {
  "use strict";
  var EPSILON = 1e-6;
  var ARRAY_TYPE = typeof Float32Array !== "undefined" ? Float32Array : Array;
  if (!Math.hypot)
    Math.hypot = function() {
      var y = 0, i = arguments.length;
      while (i--) {
        y += arguments[i] * arguments[i];
      }
      return Math.sqrt(y);
    };
  function create() {
    var out = new ARRAY_TYPE(3);
    if (ARRAY_TYPE != Float32Array) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
    }
    return out;
  }
  function fromValues(x, y, z) {
    var out = new ARRAY_TYPE(3);
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
  }
  function add(out, a, b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
  }
  function subtract(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
  }
  function equals(a, b) {
    var a0 = a[0], a1 = a[1], a2 = a[2];
    var b0 = b[0], b1 = b[1], b2 = b[2];
    return Math.abs(a0 - b0) <= EPSILON * Math.max(1, Math.abs(a0), Math.abs(b0)) && Math.abs(a1 - b1) <= EPSILON * Math.max(1, Math.abs(a1), Math.abs(b1)) && Math.abs(a2 - b2) <= EPSILON * Math.max(1, Math.abs(a2), Math.abs(b2));
  }
  var sub = subtract;
  (function() {
    var vec = create();
    return function(a, stride, offset, count, fn, arg) {
      var i, l;
      if (!stride) {
        stride = 3;
      }
      if (!offset) {
        offset = 0;
      }
      if (count) {
        l = Math.min(count * stride + offset, a.length);
      } else {
        l = a.length;
      }
      for (i = offset; i < l; i += stride) {
        vec[0] = a[i];
        vec[1] = a[i + 1];
        vec[2] = a[i + 2];
        fn(vec, vec, arg);
        a[i] = vec[0];
        a[i + 1] = vec[1];
        a[i + 2] = vec[2];
      }
      return a;
    };
  })();
  var ComputeCorners = "const OctreeSize = 32u;\r\n\r\nstruct CornerMaterials {\r\n  cornerMaterials : array<u32>,\r\n};\r\n@binding(1) @group(0) var<storage, read> cornerMaterials: CornerMaterials;\r\n\r\nstruct VoxelMaterials {\r\n  voxelMaterials : array<u32>,\r\n};\r\n@binding(2) @group(0) var<storage, read_write> voxelMaterials: VoxelMaterials;\r\n\r\nconst CHILD_MIN_OFFSETS = array<vec3<u32>, 8>\r\n(\r\n  vec3<u32>(0u, 0u, 0u),\r\n  vec3<u32>(0u, 0u, 1u),\r\n  vec3<u32>(0u, 1u, 0u),\r\n  vec3<u32>(0u, 1u, 1u),\r\n  vec3<u32>(1u, 0u, 0u),\r\n  vec3<u32>(1u, 0u, 1u),\r\n  vec3<u32>(1u, 1u, 0u),\r\n  vec3<u32>(1u, 1u, 1u)\r\n);\r\n\r\n@compute @workgroup_size(1)\r\nfn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {\r\n  let index: u32 = GlobalInvocationID.z * 32u * 32u + GlobalInvocationID.y * 32u + GlobalInvocationID.x;\r\n\r\n  let nodePos: vec3<u32> = vec3<u32>(GlobalInvocationID.x, GlobalInvocationID.y, GlobalInvocationID.z);\r\n  var corners: u32 = 0u;\r\n\r\n  var j: u32 = 0u;\r\n  loop {\r\n    if (j >= 8u) { break; }\r\n\r\n    let cornerPos: vec3<u32> = vec3<u32>(GlobalInvocationID.x + CHILD_MIN_OFFSETS[j].x, GlobalInvocationID.y + CHILD_MIN_OFFSETS[j].y, GlobalInvocationID.z + CHILD_MIN_OFFSETS[j].z);\r\n    let material: u32 = min(1, cornerMaterials.cornerMaterials[cornerPos.z * 33u * 33u + cornerPos.y * 33u + cornerPos.x]);\r\n    corners = corners | (material << j);\r\n\r\n    continuing {\r\n      j = j + 1u;\r\n    }\r\n  }\r\n  \r\n  voxelMaterials.voxelMaterials[index] = corners;\r\n}";
  var ComputePositions = "struct VoxelMaterials {\r\n  voxelMaterials : array<u32>,\r\n};\r\n@binding(2) @group(0) var<storage, read> voxelMaterials: VoxelMaterials;\r\n\r\nstruct CornerIndex {\r\n  cornerCount : u32,\r\n  cornerIndexes : array<u32>,\r\n};\r\n@binding(3) @group(0) var<storage, read_write> cornerIndex: CornerIndex;\r\n\r\n\r\n@compute @workgroup_size(1)\r\nfn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {\r\n	var position: u32 = 0u;\r\n\r\n	var i : u32 = 0u;\r\n	loop {\r\n		if (i >= 32u * 32u * 32u) { break; }\r\n		\r\n		if (voxelMaterials.voxelMaterials[i] != 0u && voxelMaterials.voxelMaterials[i] != 255u) {\r\n			cornerIndex.cornerIndexes[position] = i;\r\n			position = position + 1u;  \r\n		}\r\n			\r\n		continuing {  \r\n			i = i + 1u;\r\n		}\r\n	}\r\n\r\n	cornerIndex.cornerCount = position;\r\n}";
  var ComputeVoxels = "struct Permutations {\n  Perm : array<i32, 512>,\n};\n\n@binding(0) @group(0)\nvar<storage, read> perm : Permutations;\n\nstruct CornerMaterials {\n  cornerMaterials : array<u32>,\n};\n\n@binding(1) @group(0)\nvar<storage, read_write> cornerMaterials: CornerMaterials;\n\nstruct VoxelMaterials {\n  voxelMaterials : array<u32>,\n};\n\n@binding(2) @group(0)\nvar<storage, read_write> voxelMaterials: VoxelMaterials;\n\nstruct CornerIndex {\n  cornerCount : u32,\n  cornerIndexes : array<u32>\n};\n\n@binding(3) @group(0)\nvar<storage, read_write> cornerIndex: CornerIndex;\n\nstruct GPUVOX\n{\n	voxMin: vec3<f32>,\n	corners: f32,\n	vertPoint: vec3<f32>,\n	avgNormal: vec3<f32>,\n	numPoints: f32\n};\nstruct GPUVOXS {\n  voxels : array<GPUVOX>,\n};\n\n@binding(4) @group(0)\nvar<storage, read_write> voxels: GPUVOXS;\n\nstruct UniformBufferObject {\n  chunkPosition : vec3<f32>,\n  stride : f32,\n	width: u32\n};\n\n@binding(5) @group(0)\nvar<uniform> uniforms : UniformBufferObject;\n\nconst CHILD_MIN_OFFSETS: array<vec3<u32>, 8> = array<vec3<u32>, 8>\n(\n  vec3<u32>(0u, 0u, 0u),\n  vec3<u32>(0u, 0u, 1u),\n  vec3<u32>(0u, 1u, 0u),\n  vec3<u32>(0u, 1u, 1u),\n  vec3<u32>(1u, 0u, 0u),\n  vec3<u32>(1u, 0u, 1u),\n  vec3<u32>(1u, 1u, 0u),\n  vec3<u32>(1u, 1u, 1u)\n);\n\nconst edgevmap: array<vec2<i32>, 12> = array<vec2<i32>, 12>\n(\n	vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7),\n	vec2<i32>(0,2), vec2<i32>(1,3), vec2<i32>(4,6), vec2<i32>(5,7),\n	vec2<i32>(0,1), vec2<i32>(2,3), vec2<i32>(4,5), vec2<i32>(6,7)\n);\n\nfn random(i: vec2<f32>) -> f32 {\n  return fract(sin(dot(i,vec2(12.9898,78.233)))*43758.5453123);\n}\n\nfn Vec3Dot(a: vec3<f32>, b: vec3<f32>) -> f32\n{\n	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);\n}\n\nconst Grad3: array<vec3<f32>, 12> = array<vec3<f32>, 12>(\n	vec3<f32>(1.0,1.0,0.0), vec3<f32>(-1.0,1.0,0.0), vec3<f32>(1.0,-1.0,0.0), vec3<f32>(-1.0,-1.0,0.0),\n	vec3<f32>(1.0,0.0,1.0), vec3<f32>(-1.0,0.0,1.0), vec3<f32>(1.0,0.0,-1.0), vec3<f32>(-1.0,0.0,-1.0),\n	vec3<f32>(0.0,1.0,1.0), vec3<f32>(0.0,-1.0,1.0), vec3<f32>(0.0,1.0,-1.0), vec3<f32>(0.0,-1.0,-1.0)\n);\n\nfn Perlin(x1: f32, y1: f32, z1: f32) -> f32\n{\n	var X: i32 = 0;\n	if (x1 > 0.0) {\n		X = i32(x1);\n	} else {\n		X = i32(x1) - 1;\n	}\n\n	var Y: i32 = 0;\n	if (y1 > 0.0) {\n		Y = i32(y1);\n	} else {\n		Y = i32(y1) - 1;\n	}\n\n	var Z: i32 = 0;\n	if (z1 > 0.0) {\n		Z = i32(z1);\n	} else {\n		Z = i32(z1) - 1;\n	}\n\n	let x: f32 = x1 - f32(X);\n	let y: f32 = y1 - f32(Y);\n	let z: f32 = z1 - f32(Z);\n\n	X = X & 255;\n	Y = Y & 255;\n	Z = Z & 255;\n\n	let gi000: i32 = (perm.Perm[X + perm.Perm[Y + perm.Perm[Z] ] ] % 12);\n	let gi001: i32 = (perm.Perm[X + perm.Perm[Y + perm.Perm[Z + 1] ] ] % 12);\n	let gi010: i32 = (perm.Perm[X + perm.Perm[Y + 1 + perm.Perm[Z] ] ] % 12);\n	let gi011: i32 = (perm.Perm[X + perm.Perm[Y + 1 + perm.Perm[Z + 1] ] ] % 12);\n	let gi100: i32 = (perm.Perm[X + 1 + perm.Perm[Y + perm.Perm[Z] ] ] % 12);\n	let gi101: i32 = (perm.Perm[X + 1 + perm.Perm[Y + perm.Perm[Z + 1] ] ] % 12);\n	let gi110: i32 = (perm.Perm[X + 1 + perm.Perm[Y + 1 + perm.Perm[Z] ] ] % 12);\n	let gi111: i32 = (perm.Perm[X + 1 + perm.Perm[Y + 1 + perm.Perm[Z + 1] ] ] % 12);\n\n	let n000: f32 = dot(Grad3[gi000], vec3<f32>(x, y, z));\n	let n100: f32 = dot(Grad3[gi100], vec3<f32>(x - 1.0, y, z));\n	let n010: f32 = dot(Grad3[gi010], vec3<f32>(x, y - 1.0, z));\n	let n110: f32 = dot(Grad3[gi110], vec3<f32>(x - 1.0, y - 1.0, z));\n	let n001: f32 = dot(Grad3[gi001], vec3<f32>(x, y, z - 1.0));\n	let n101: f32 = dot(Grad3[gi101], vec3<f32>(x - 1.0, y, z - 1.0));\n	let n011: f32 = dot(Grad3[gi011], vec3<f32>(x, y - 1.0, z - 1.0));\n	let n111: f32 = dot(Grad3[gi111], vec3<f32>(x - 1.0, y - 1.0, z - 1.0));\n\n	let u: f32 = f32(x * x * x * (x * (x * 6.0 - 15.0) + 10.0));\n	let v: f32 = f32(y * y * y * (y * (y * 6.0 - 15.0) + 10.0));\n	let w: f32 = f32(z * z * z * (z * (z * 6.0 - 15.0) + 10.0));\n	let nx00: f32 = mix(n000, n100, u);\n	let nx01: f32 = mix(n001, n101, u);\n	let nx10: f32 = mix(n010, n110, u);\n	let nx11: f32 = mix(n011, n111, u);\n	let nxy0: f32 = mix(nx00, nx10, v);\n	let nxy1: f32 = mix(nx01, nx11, v);\n	let nxyz: f32 = mix(nxy0, nxy1, w);\n\n	return nxyz;\n}\n\nfn FractalNoise(octaves: i32, frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32\n{\n	let SCALE: f32 = 1.0 / 128.0;\n	var p: vec3<f32> = position * SCALE;\n	var nois: f32 = 0.0;\n\n	var amplitude: f32 = 1.0;\n	p = p * frequency;\n\n	var i: i32 = 0;\n	loop {\n		if (i >= octaves) { break; }\n\n		nois = nois + Perlin(p.x, p.y, p.z) * amplitude;\n		p = p * lacunarity;\n		amplitude = amplitude * persistence;\n\n		continuing {\n			i = i + 1;\n		}\n	}\n\n	return nois;\n}\n\nfn FractalNoise1(frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32\n{\n	let SCALE: f32 = 1.0 / 128.0;\n	var p: vec3<f32> = position * SCALE;\n	var nois: f32 = 0.0;\n\n	var amplitude: f32 = 1.0;\n	p = p * frequency;\n\n	nois = nois + Perlin(p.x, p.y, p.z) * amplitude;\n	p = p * lacunarity;\n	amplitude = amplitude * persistence;\n\n	return nois;\n}\n\nfn CalculateNoiseValue(pos: vec3<f32>, scale: f32) -> f32\n{\n	return FractalNoise(4, 0.5343, 2.2324, 0.68324, pos * scale);\n}\n\nfn CLerp(a: f32, b: f32, t: f32) -> f32\n{\n	return (1.0 - t) * a + t * b;\n}\n\n// SVD\n\nconst SVD_NUM_SWEEPS: i32 = 4;\nconst PSUEDO_INVERSE_THRESHOLD: f32 = 0.00000001;\n\nfn svd_mul_matrix_vec(m: mat3x3<f32>, b: vec4<f32>) -> vec4<f32>\n{\n	var a: mat3x3<f32> = m;\n\n	return vec4<f32>(\n		dot(vec4<f32>(a[0][0], a[0][1], a[0][2], 0.0), b),\n		dot(vec4<f32>(a[1][0], a[1][1], a[1][2], 0.0), b),\n		dot(vec4<f32>(a[2][0], a[2][1], a[2][2], 0.0), b),\n		0.0\n	);\n}\n\nfn givens_coeffs_sym(a_pp: f32, a_pq: f32, a_qq: f32) -> vec2<f32>\n{\n	if (a_pq == 0.0) {\n		return vec2<f32>(1.0, 0.0);\n	}\n\n	let tau: f32 = (a_qq - a_pp) / (2.0 * a_pq);\n	let stt: f32 = sqrt(1.0 + tau * tau);\n	var tan: f32;\n	if ((tau >= 0.0)) {\n		tan = (tau + stt);\n	} else {\n		tan = (tau - stt);\n	}\n	tan = 1.0 / tan;\n\n	let c: f32 = inverseSqrt(1.0 + tan * tan);\n	let s: f32 = tan * c;\n\n	return vec2<f32>(c, s);\n}\n\nfn svd_rotate_xy(x: f32, y: f32, c: f32, s: f32) -> vec2<f32>\n{\n	return vec2<f32>(c * x - s * y, s * x + c * y);\n}\n\nfn svd_rotateq_xy(x: f32, y: f32, z: f32, c: f32, s: f32) -> vec2<f32>\n{\n	let cc: f32 = c * c;\n	let ss: f32 = s * s;\n	let mx: f32 = 2.0 * c * s * z;\n\n	return vec2<f32>(\n		cc * x - mx + ss * y,\n		ss * x + mx + cc * z\n	);\n}\n\nvar<private> vtav: mat3x3<f32>;\nvar<private> v: mat3x3<f32>;\nvar<private> ATA: array<f32, 6>;\nvar<private> Atb: vec4<f32>;\nvar<private> pointaccum: vec4<f32>;\nvar<private> btb: f32;\n\nfn svd_rotate(a: i32, b: i32)\n{\n	if (vtav[a][b] == 0.0) { return; }\n\n\n\n	let coeffs: vec2<f32> = givens_coeffs_sym(vtav[a][a], vtav[a][b], vtav[b][b]);\n	let c: f32 = coeffs.x;\n	let s: f32 = coeffs.y;\n\n	let rot1: vec2<f32> = svd_rotateq_xy(vtav[a][a], vtav[b][b], vtav[a][b], c, s);\n	vtav[a][a] = rot1.x;\n	vtav[b][b] = rot1.y;\n\n	let rot2: vec2<f32> = svd_rotate_xy(vtav[0][3-b], vtav[1-a][2], c, s);\n	vtav[0][3-b] = rot2.x;\n	vtav[1-a][2] = rot2.y;\n\n	vtav[a][b] = 0.0;\n\n	let rot3: vec2<f32> = svd_rotate_xy(v[0][a], v[0][b], c, s);\n	v[0][a] = rot3.x; v[0][b] = rot3.y;\n\n	let rot4: vec2<f32> = svd_rotate_xy(v[1][a], v[1][b], c, s);\n	v[1][a] = rot4.x; v[1][b] = rot4.y;\n\n	let rot5: vec2<f32> = svd_rotate_xy(v[2][a], v[2][b], c, s);\n	v[2][a] = rot5.x; v[2][b] = rot5.y;\n}\n\nfn svd_solve_sym(b: array<f32, 6>) -> vec4<f32>\n{\n	var a: array<f32, 6> = b;\n\n	vtav = mat3x3<f32>(\n		vec3<f32>(a[0], a[1], a[2]),\n		vec3<f32>(0.0, a[3], a[4]),\n		vec3<f32>(0.0, 0.0, a[5])\n	);\n\n	var i: i32;\n	loop {\n		if (i >= SVD_NUM_SWEEPS) { break; }\n\n		svd_rotate(0, 1);\n		svd_rotate(0, 2);\n		svd_rotate(1, 2);\n\n		continuing {\n			i = i + 1;\n		}\n	}\n\n	var copy: mat3x3<f32> = vtav;\n	return vec4<f32>(copy[0][0], copy[1][1], copy[2][2], 0.0);\n}\n\n\nfn svd_invdet(x: f32, tol: f32) -> f32\n{\n	if (abs(x) < tol || abs(1.0 / x) < tol) {\n		return 0.0;\n	}\n	return (1.0 / x);\n}\n\nfn svd_pseudoinverse(sigma: vec4<f32>, c: mat3x3<f32>) -> mat3x3<f32>\n{\n	let d0: f32 = svd_invdet(sigma.x, PSUEDO_INVERSE_THRESHOLD);\n	let d1: f32 = svd_invdet(sigma.y, PSUEDO_INVERSE_THRESHOLD);\n	let d2: f32 = svd_invdet(sigma.z, PSUEDO_INVERSE_THRESHOLD);\n\n	var copy: mat3x3<f32> = c;\n\n	return mat3x3<f32> (\n		vec3<f32>(\n			copy[0][0] * d0 * copy[0][0] + copy[0][1] * d1 * copy[0][1] + copy[0][2] * d2 * copy[0][2],\n			copy[0][0] * d0 * copy[1][0] + copy[0][1] * d1 * copy[1][1] + copy[0][2] * d2 * copy[1][2],\n			copy[0][0] * d0 * copy[2][0] + copy[0][1] * d1 * copy[2][1] + copy[0][2] * d2 * copy[2][2]\n		),\n		vec3<f32>(\n			copy[1][0] * d0 * copy[0][0] + copy[1][1] * d1 * copy[0][1] + copy[1][2] * d2 * copy[0][2],\n			copy[1][0] * d0 * copy[1][0] + copy[1][1] * d1 * copy[1][1] + copy[1][2] * d2 * copy[1][2],\n			copy[1][0] * d0 * copy[2][0] + copy[1][1] * d1 * copy[2][1] + copy[1][2] * d2 * copy[2][2]\n		),\n		vec3<f32>(\n			copy[2][0] * d0 * copy[0][0] + copy[2][1] * d1 * copy[0][1] + copy[2][2] * d2 * copy[0][2],\n			copy[2][0] * d0 * copy[1][0] + copy[2][1] * d1 * copy[1][1] + copy[2][2] * d2 * copy[1][2],\n			copy[2][0] * d0 * copy[2][0] + copy[2][1] * d1 * copy[2][1] + copy[2][2] * d2 * copy[2][2]\n		),\n	);\n}\n\nfn svd_solve_ATA_Atb(a: vec4<f32>) -> vec4<f32>\n{\n	v = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));\n\n	let sigma: vec4<f32> = svd_solve_sym(ATA);\n\n	let Vinv: mat3x3<f32> = svd_pseudoinverse(sigma, v);\n	return svd_mul_matrix_vec(Vinv, a);\n}\n\nfn svd_vmul_sym(v: vec4<f32>) -> vec4<f32>\n{\n	let A_row_x: vec4<f32> = vec4<f32>(ATA[0], ATA[1], ATA[2], 0.0);\n	return vec4<f32> (\n		dot(A_row_x, v),\n		ATA[1] * v.x + ATA[3] * v.y + ATA[4] * v.z,\n		ATA[2] * v.x + ATA[4] * v.y + ATA[5] * v.z,\n		0.0\n	);\n}\n\n\n// // QEF\n\nfn qef_add(n: vec4<f32>, p: vec4<f32>)\n{\n	ATA[0] = ATA[0] + n.x * n.x;\n	ATA[1] = ATA[1] + n.x * n.y;\n	ATA[2] = ATA[2] + n.x * n.z;\n	ATA[3] = ATA[3] + n.y * n.y;\n	ATA[4] = ATA[4] + n.y * n.z;\n	ATA[5] = ATA[5] + n.z * n.z;\n\n	let b: f32 = dot(p, n);\n	Atb.x = Atb.x +n.x * b;\n	Atb.y = Atb.y +n.y * b;\n	Atb.z = Atb.z +n.z * b;\n	btb = btb + b * b;\n\n	pointaccum.x = pointaccum.x +p.x;\n	pointaccum.y = pointaccum.y +p.y;\n	pointaccum.z = pointaccum.z +p.z;\n	pointaccum.w = pointaccum.w +1.0;\n}\n\nfn qef_calc_error(x: vec4<f32>) -> f32\n{\n	var tmp: vec4<f32> = svd_vmul_sym(x);\n	tmp = Atb - tmp;\n\n	return dot(tmp, tmp);\n}\n\nfn qef_solve() -> vec4<f32>\n{\n	let masspoint: vec4<f32> = vec4<f32>(pointaccum.x / pointaccum.w, pointaccum.y / pointaccum.w, pointaccum.z / pointaccum.w, pointaccum.w / pointaccum.w);\n\n	var A_mp: vec4<f32> = svd_vmul_sym(masspoint);\n	A_mp = Atb - A_mp;\n\n	let x: vec4<f32> = svd_solve_ATA_Atb(A_mp);\n\n	let error: f32 = qef_calc_error(x);\n	let r: vec4<f32> = x + masspoint;\n\n	return vec4<f32>(r.x, r.y, r.z, error);\n}\n\n#import density\n\nfn ApproximateZeroCrossingPosition(p0: vec3<f32>, p1: vec3<f32>) -> vec3<f32>\n{\n	var minValue: f32 = 100000.0;\n	var t: f32 = 0.0;\n	var currentT: f32 = 0.0;\n	let steps: f32 = 8.0;\n	let increment: f32 = 1.0 / steps;\n	loop {\n		if (currentT > 1.0) { break; }\n\n		let p: vec3<f32> = p0 + ((p1 - p0) * currentT);\n		let density: f32 = abs(getDensity(p));\n		if (density < minValue)\n		{\n			minValue = density;\n			t = currentT;\n		}\n\n		continuing {\n			currentT = currentT + increment;\n		}\n	}\n\n	return p0 + ((p1 - p0) * t);\n}\n\nfn CalculateSurfaceNormal(p: vec3<f32>) -> vec3<f32>\n{\n	let H: f32 = uniforms.stride; // This needs to scale based on something...\n	let dx: f32 = getDensity(p + vec3<f32>(H, 0.0, 0.0)) - getDensity(p - vec3<f32>(H, 0.0, 0.0));\n	let dy: f32 = getDensity(p + vec3<f32>(0.0, H, 0.0)) - getDensity(p - vec3<f32>(0.0, H, 0.0));\n	let dz: f32 = getDensity(p + vec3<f32>(0.0, 0.0, H)) - getDensity(p - vec3<f32>(0.0, 0.0, H));\n\n	return normalize(vec3<f32>(dx, dy, dz));\n}\n\n@compute @workgroup_size(128)\nfn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {\n	let trueIndex: u32 = GlobalInvocationID.x;\n\n	if (trueIndex < cornerIndex.cornerCount)\n	{\n		let ures: u32 = 32u;\n\n		let nodeSize: u32 = u32(uniforms.stride);\n\n		let voxelIndex: u32 = cornerIndex.cornerIndexes[trueIndex];\n		let z: u32 = voxelIndex / (ures * ures);\n		let y: u32 = (voxelIndex - (z * ures * ures)) / ures;\n		let x: u32 = voxelIndex - (z * ures * ures) - (y * ures);\n\n		let corners: u32 = voxelMaterials.voxelMaterials[voxelIndex];\n\n		let nodePos: vec3<f32> = (vec3<f32>(f32(x), f32(y), f32 (z)) * uniforms.stride) + uniforms.chunkPosition;\n		voxels.voxels[trueIndex].voxMin = nodePos;\n		let MAX_CROSSINGS: i32 = 6;\n		var edgeCount: i32 = 0;\n\n		pointaccum = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n		ATA = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);\n		Atb = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n		var averageNormal: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);\n		btb = 0.0;\n\n		var j: i32 = 0;\n		loop {\n			if (!(j < 12 && edgeCount <= MAX_CROSSINGS)) {\n				break;\n			}\n\n			let c1: i32 = edgevmap[j].x;\n			let c2: i32 = edgevmap[j].y;\n\n			let m1: u32 = (corners >> u32(c1)) & 1u;\n			let m2: u32 = (corners >> u32(c2)) & 1u;\n\n			if (!((m1 == 0u && m2 == 0u) || (m1 == 1u && m2 == 1u)))\n			{\n				let p1: vec3<f32> = nodePos + vec3<f32>(f32(CHILD_MIN_OFFSETS[c1].x * nodeSize), f32(CHILD_MIN_OFFSETS[c1].y * nodeSize), f32(CHILD_MIN_OFFSETS[c1].z * nodeSize));\n				let p2: vec3<f32> = nodePos + vec3<f32>(f32(CHILD_MIN_OFFSETS[c2].x * nodeSize), f32(CHILD_MIN_OFFSETS[c2].y * nodeSize), f32(CHILD_MIN_OFFSETS[c2].z * nodeSize));\n				let p: vec3<f32> = ApproximateZeroCrossingPosition(p1, p2);\n				let n: vec3<f32> = CalculateSurfaceNormal(p);\n\n				qef_add(vec4<f32>(n.x, n.y, n.z, 0.0), vec4<f32>(p.x, p.y, p.z, 0.0));\n\n				averageNormal = averageNormal + n;\n\n				edgeCount = edgeCount + 1;\n			}\n\n			continuing {\n				j = j + 1;\n			}\n		}\n\n\n		averageNormal = normalize(averageNormal / vec3<f32>(f32(edgeCount), f32(edgeCount), f32(edgeCount)));\n\n		let com: vec3<f32> = vec3<f32>(pointaccum.x / pointaccum.w, pointaccum.y / pointaccum.w, pointaccum.z / pointaccum.w);\n\n		let result: vec4<f32> = qef_solve();\n		var solved_position: vec3<f32> = result.xyz;\n		let error: f32 = result.w;\n\n\n		let Min: vec3<f32> = nodePos;\n		let Max: vec3<f32> = nodePos + vec3<f32>(1.0, 1.0, 1.0);\n		if (solved_position.x < Min.x || solved_position.x > Max.x ||\n				solved_position.y < Min.y || solved_position.y > Max.y ||\n				solved_position.z < Min.z || solved_position.z > Max.z)\n		{\n			solved_position = com;\n		}\n\n		voxels.voxels[trueIndex].vertPoint = solved_position;\n		voxels.voxels[trueIndex].avgNormal = averageNormal;\n		voxels.voxels[trueIndex].numPoints = f32(edgeCount);\n		voxels.voxels[trueIndex].corners = f32(voxelMaterials.voxelMaterials[voxelIndex]);\n	}\n}\n\n@compute @workgroup_size(1)\nfn computeMaterials(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {\n		let width = uniforms.width;\n    let index: u32 = GlobalInvocationID.z * width * width + GlobalInvocationID.y * width + GlobalInvocationID.x;\n    let cornerPos: vec3<f32> = vec3<f32>(f32(GlobalInvocationID.x) * uniforms.stride, f32(GlobalInvocationID.y) * uniforms.stride, f32(GlobalInvocationID.z) * uniforms.stride);\n\n    let density: f32 = getDensity(cornerPos + uniforms.chunkPosition);\n\n		if (density < 0.0) {\n			if (true || length(cornerPos + uniforms.chunkPosition) < 2000.0) {\n        //cornerMaterials.cornerMaterials[index] = u32(random(vec2(f32(index))) * 255.0) + 1;\n			  cornerMaterials.cornerMaterials[index] = 256u;\n			} else {\n        cornerMaterials.cornerMaterials[index] = u32(length(cornerPos) / uniforms.stride * 256.0);\n			}\n		} else {\n			cornerMaterials.cornerMaterials[index] = 0u;\n		}\n}\n";
  var DensityShader = "const freq = 0.001;\n\nconst MATERIAL_AIR = 0u;\nconst MATERIAL_ROCK = 1u;\nconst MATERIAL_WOOD = 2u;\nconst MATERIAL_FIRE = 3u;\n\nstruct Density {\n  density: f32,\n  material: u32\n}\n\nstruct Augmentations {\n  count: u32,\n  augmentations: array<Augmentation>\n}\n\nstruct Augmentation {\n  position: vec3<f32>,\n  size: f32,\n  attributes: u32\n}\n\n@binding(0) @group(1) var<storage, read> augmentations: Augmentations;\n\nfn subtract(base: Density, sub: f32) -> Density {\n  return Density(max(base.density, sub), base.material);\n}\n\nfn add(base: Density, add: f32, material: u32) -> Density {\n  if (add <= 0) {\n    return Density(add, material);\n  }\n  return base;\n}\n\nfn Box(worldPosition: vec3<f32>, origin: vec3<f32>, halfDimensions: vec3<f32>) -> f32\n{\n	let local_pos: vec3<f32> = worldPosition - origin;\n	let pos: vec3<f32> = local_pos;\n\n	let d: vec3<f32> = vec3<f32>(abs(pos.x), abs(pos.y), abs(pos.z)) - halfDimensions;\n	let m: f32 = max(d.x, max(d.y, d.z));\n	return clamp(min(m, length(max(d, vec3<f32>(0.0, 0.0, 0.0)))), -100.0, 100.0);\n}\n\nfn Torus(worldPosition: vec3<f32>, origin: vec3<f32>, t: vec3<f32>) -> f32\n{\n	let p: vec3<f32> = worldPosition - origin;\n\n  let q: vec2<f32> = vec2<f32>(length(p.xz)-t.x,p.y);\n  return length(q)-t.y;\n}\n\nfn Sphere(worldPosition: vec3<f32>, origin: vec3<f32>, radius: f32) -> f32\n{\n	return clamp(length(worldPosition - origin) - radius, -100.0, 100.0);\n}\n\nfn FractalNoise21(octaves: i32, frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32\n{\n	let SCALE: f32 = 1.0 / 128.0;\n	var p: vec3<f32> = position * SCALE;\n	var nois: f32 = 0.0;\n\n	var amplitude: f32 = 1.0;\n	p = p * frequency;\n\n	var i: i32 = 0;\n	loop {\n		if (i >= octaves) { break; }\n\n		nois = nois + perlinNoise3(p) * amplitude;\n		p = p * lacunarity;\n		amplitude = amplitude * persistence;\n\n		continuing {\n			i = i + 1;\n		}\n	}\n\n	return nois;\n}\n\nfn FractalNoise2(frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32\n{\n	let SCALE: f32 = 1.0 / 128.0;\n	var p: vec3<f32> = position * SCALE;\n	var nois: f32 = 0.0;\n\n	var amplitude: f32 = 1.0;\n	p = p * frequency;\n\n	nois = nois + perlinNoise3(p) * amplitude;\n	p = p * lacunarity;\n	amplitude = amplitude * persistence;\n\n	return nois;\n}\n\nfn permute41(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }\nfn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }\nfn fade3(t: vec3<f32>) -> vec3<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }\n\nfn perlinNoise3(P: vec3<f32>) -> f32 {\n  var Pi0 : vec3<f32> = floor(P); // Integer part for indexing\n  var Pi1 : vec3<f32> = Pi0 + vec3<f32>(1.); // Integer part + 1\n  Pi0 = Pi0 % vec3<f32>(289.);\n  Pi1 = Pi1 % vec3<f32>(289.);\n  let Pf0 = fract(P); // Fractional part for interpolation\n  let Pf1 = Pf0 - vec3<f32>(1.); // Fractional part - 1.\n  let ix = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);\n  let iy = vec4<f32>(Pi0.yy, Pi1.yy);\n  let iz0 = Pi0.zzzz;\n  let iz1 = Pi1.zzzz;\n\n  let ixy = permute41(permute41(ix) + iy);\n  let ixy0 = permute41(ixy + iz0);\n  let ixy1 = permute41(ixy + iz1);\n\n  var gx0: vec4<f32> = ixy0 / 7.;\n  var gy0: vec4<f32> = fract(floor(gx0) / 7.) - 0.5;\n  gx0 = fract(gx0);\n  var gz0: vec4<f32> = vec4<f32>(0.5) - abs(gx0) - abs(gy0);\n  var sz0: vec4<f32> = step(gz0, vec4<f32>(0.));\n  gx0 = gx0 + sz0 * (step(vec4<f32>(0.), gx0) - 0.5);\n  gy0 = gy0 + sz0 * (step(vec4<f32>(0.), gy0) - 0.5);\n\n  var gx1: vec4<f32> = ixy1 / 7.;\n  var gy1: vec4<f32> = fract(floor(gx1) / 7.) - 0.5;\n  gx1 = fract(gx1);\n  var gz1: vec4<f32> = vec4<f32>(0.5) - abs(gx1) - abs(gy1);\n  var sz1: vec4<f32> = step(gz1, vec4<f32>(0.));\n  gx1 = gx1 - sz1 * (step(vec4<f32>(0.), gx1) - 0.5);\n  gy1 = gy1 - sz1 * (step(vec4<f32>(0.), gy1) - 0.5);\n\n  var g000: vec3<f32> = vec3<f32>(gx0.x, gy0.x, gz0.x);\n  var g100: vec3<f32> = vec3<f32>(gx0.y, gy0.y, gz0.y);\n  var g010: vec3<f32> = vec3<f32>(gx0.z, gy0.z, gz0.z);\n  var g110: vec3<f32> = vec3<f32>(gx0.w, gy0.w, gz0.w);\n  var g001: vec3<f32> = vec3<f32>(gx1.x, gy1.x, gz1.x);\n  var g101: vec3<f32> = vec3<f32>(gx1.y, gy1.y, gz1.y);\n  var g011: vec3<f32> = vec3<f32>(gx1.z, gy1.z, gz1.z);\n  var g111: vec3<f32> = vec3<f32>(gx1.w, gy1.w, gz1.w);\n\n  let norm0 = taylorInvSqrt4(\n      vec4<f32>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));\n  g000 = g000 * norm0.x;\n  g010 = g010 * norm0.y;\n  g100 = g100 * norm0.z;\n  g110 = g110 * norm0.w;\n  let norm1 = taylorInvSqrt4(\n      vec4<f32>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));\n  g001 = g001 * norm1.x;\n  g011 = g011 * norm1.y;\n  g101 = g101 * norm1.z;\n  g111 = g111 * norm1.w;\n\n  let n000 = dot(g000, Pf0);\n  let n100 = dot(g100, vec3<f32>(Pf1.x, Pf0.yz));\n  let n010 = dot(g010, vec3<f32>(Pf0.x, Pf1.y, Pf0.z));\n  let n110 = dot(g110, vec3<f32>(Pf1.xy, Pf0.z));\n  let n001 = dot(g001, vec3<f32>(Pf0.xy, Pf1.z));\n  let n101 = dot(g101, vec3<f32>(Pf1.x, Pf0.y, Pf1.z));\n  let n011 = dot(g011, vec3<f32>(Pf0.x, Pf1.yz));\n  let n111 = dot(g111, Pf1);\n\n  var fade_xyz: vec3<f32> = fade3(Pf0);\n  let temp = vec4<f32>(f32(fade_xyz.z)); // simplify after chrome bug fix\n  let n_z = mix(vec4<f32>(n000, n100, n010, n110), vec4<f32>(n001, n101, n011, n111), temp);\n  let n_yz = mix(n_z.xy, n_z.zw, vec2<f32>(f32(fade_xyz.y))); // simplify after chrome bug fix\n  let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);\n  return 2.2 * n_xyz;\n}\n\nfn CalculateNoiseValue2(pos: vec3<f32>, scale: f32) -> f32\n{\n	return FractalNoise21(4, 0.5343, 2.2324, 0.68324, pos * scale);\n}\n\nfn CLerp2(a: f32, b: f32, t: f32) -> f32\n{\n	return (1.0 - t) * a + t * b;\n}\n\nfn rotateAlign(v1: vec3<f32>, v2: vec3<f32>) -> mat3x3<f32>\n{\n    let axis = cross( v1, v2 );\n\n    let cosA = dot( v1, v2 );\n    let k = 1.0 / (1.0 + cosA);\n\n    return mat3x3<f32>( (axis.x * axis.x * k) + cosA,\n                 (axis.y * axis.x * k) - axis.z,\n                 (axis.z * axis.x * k) + axis.y,\n                 (axis.x * axis.y * k) + axis.z,\n                 (axis.y * axis.y * k) + cosA,\n                 (axis.z * axis.y * k) - axis.x,\n                 (axis.x * axis.z * k) - axis.y,\n                 (axis.y * axis.z * k) + axis.x,\n                 (axis.z * axis.z * k) + cosA\n                 );\n}\n\n\nfn AngleAxis3x3(angle: f32, axis: vec3<f32>) -> mat3x3<f32>\n{\n    let s = sin(angle);\n    let c = cos(angle);\n\n    let t = 1 - c;\n    let x = axis.x;\n    let y = axis.y;\n    let z = axis.z;\n\n    return mat3x3<f32>(\n        t * x * x + c,      t * x * y - s * z,  t * x * z + s * y,\n        t * x * y + s * z,  t * y * y + c,      t * y * z - s * x,\n        t * x * z - s * y,  t * y * z + s * x,  t * z * z + c\n    );\n}\n\nfn blockSize(blockType: u32) -> f32 {\n  if (blockType == 2 || blockType == 3) {\n    return 0.5;\n  }\n  return 1.0;\n}\n\nfn calculateDensity(worldPosition: vec3<f32>) -> Density {\n	var worldRadius: f32 = 5000.0;\n	var world: vec3<f32> = worldPosition - vec3<f32>(2000000.0, 100.0, 100.0);\n	var worldDist: f32 = -worldRadius + length(world);\n	let up = vec3<f32>(0.0, 1.0, 0.0);\n\n\n	let flatlandNoiseScale: f32 = 1.0;\n	let flatlandLerpAmount: f32 = 0.07;\n	let flatlandYPercent: f32 = 1.2;\n\n	let rockyNoiseScale: f32 = 1.5;\n	let rockyLerpAmount: f32 = 0.05;\n	let rockyYPercent: f32 = 0.7;\n\n	let maxMountainMixLerpAmount: f32 = 0.075;\n	let minMountainMixLerpAmount: f32 = 1.0;\n\n	let rockyBlend: f32 = 0.0;\n\n	let mountainBlend: f32 = clamp(abs(FractalNoise2(0.5343, 2.2324, 0.68324, world * 0.11)) * 4.0, 0.0, 1.0);\n	//let mountainBlend: f32 = 1.0;\n\n	//let mountain: f32 = CalculateNoiseValue2(world, 0.07);\n	let mountain: f32 = 0.0;\n\n//	var blob: f32 = CalculateNoiseValue2(world, flatlandNoiseScale + ((rockyNoiseScale - flatlandNoiseScale) * rockyBlend));\n//	blob = CLerp2(blob, (worldDist) * (flatlandYPercent + ((rockyYPercent - flatlandYPercent) * rockyBlend)),\n//				flatlandLerpAmount + ((rockyLerpAmount - flatlandLerpAmount) * rockyBlend))\n//				+ CLerp2(mountain, blob, minMountainMixLerpAmount + ((maxMountainMixLerpAmount - minMountainMixLerpAmount) * mountainBlend));\n\n  var result = Density(1.0, MATERIAL_AIR);\n\n	//result = add(result, blob, MATERIAL_WOOD);\n\n  result = add(result, Box(worldPosition, vec3<f32>(2000000.0, 150.0, 5000.0), vec3<f32>(5000.0, 1000.0, 5000.0)), MATERIAL_WOOD);\n  result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0, 100.0, 100.0), 5000.0), MATERIAL_ROCK);\n\n  //result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0 - 1000000.0, 0.0, 0.0), 1000000.0), MATERIAL_ROCK);\n\n  result = add(result, Sphere(worldPosition, vec3<f32>(0.0, 0.0, 0.0), 200000.0), MATERIAL_FIRE);\n\n  //result = subtract(result, -Sphere(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), 1000.0));\n  //result = subtract(result, -Box(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), vec3<f32>(6000.0, 500.0, 500.0)));\n  //result = subtract(result, -Box(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), vec3<f32>(500.0, 500.0, 5000.0)));\n\n  //result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), 1000.0), MATERIAL_FIRE);\n\n\n  let count = augmentations.count;\n\n  var i: u32 = 0u;\n  loop {\n    if (i >= count) { break; }\n\n    let augmentation = augmentations.augmentations[i];\n\n    let minBounds = augmentation.position - augmentation.size * 2;\n    let maxBounds = augmentation.position + augmentation.size * 2;\n    if (minBounds.x > worldPosition.x || minBounds.y > worldPosition.y || minBounds.z > worldPosition.z\n      || maxBounds.x < worldPosition.x || maxBounds.y < worldPosition.y || maxBounds.z < worldPosition.z) { continue; }\n\n    let shape = (augmentation.attributes & 0xFE) >> 1;\n    var density: f32 = 0.0;\n\n    let down = normalize(augmentation.position - vec3<f32>(2000000.0, 100.0, 100.0));\n    let rotation = rotateAlign(down, up);\n    let position = ((worldPosition - augmentation.position) * rotation - vec3<f32>(0.0, augmentation.size * blockSize(shape), 0.0)) + augmentation.position;\n\n    switch(shape) {\n      case 0: {\n        density = Sphere(position, vec3<f32>(augmentation.position.x, augmentation.position.y, augmentation.position.z), augmentation.size);\n      }\n      case 1: {\n        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y, augmentation.position.z), vec3<f32>(augmentation.size));\n      }\n      case 2: {\n        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y + augmentation.size / 2, augmentation.position.z), vec3<f32>(augmentation.size, 5.0, augmentation.size));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n      }\n      case 3: {\n        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y + augmentation.size / 2, augmentation.position.z), vec3<f32>(augmentation.size, 5.0, augmentation.size));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size, augmentation.position.y + augmentation.size, augmentation.position.z), vec3<f32>(5.0, augmentation.size / 2, augmentation.size)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));\n      }\n      case 4: {\n        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y - augmentation.size + 16, augmentation.position.z), vec3<f32>(augmentation.size, 16, augmentation.size));\n      }\n      default: {\n        continue;\n      }\n    }\n\n    if ((augmentation.attributes & 0x1) == 0x1) {\n      let material = (augmentation.attributes & 0x1FF00) >> 8;\n      result = add(result, density, material);\n    } else {\n      result = subtract(result, -density);\n    }\n\n\n    continuing {\n      i = i + 1u;\n    }\n  }\n\n  return result;\n}\n\nfn getDensity(worldPosition: vec3<f32>) -> f32 {\n	return calculateDensity(worldPosition).density;\n}\n";
  class DensityInstance {
    bindGroup;
    constructor(bindGroup) {
      this.bindGroup = bindGroup;
    }
    apply(encoder) {
      encoder.setBindGroup(1, this.bindGroup);
    }
  }
  class Density {
    augmentationBuffer;
    augmentationArray = [];
    augmentations;
    onModified = () => {
    };
    constructor(augmentationBuffer) {
      this.augmentationBuffer = augmentationBuffer;
      this.augmentations = new ArrayBuffer(Uint32Array.BYTES_PER_ELEMENT * 4);
    }
    static async init(device) {
      const augmentationSize = 64 * Float32Array.BYTES_PER_ELEMENT * 8 + 8;
      const augmentationBuffer = device.createBuffer({
        size: augmentationSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false
      });
      return new Density(augmentationBuffer);
    }
    async apply(device, pipeline) {
      const densityBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.augmentationBuffer
            }
          }
        ]
      });
      return new DensityInstance(densityBindGroup);
    }
    modify(device, augmentation) {
      this.augmentationArray.push(augmentation);
      this.update(device, this.augmentationArray);
      this.onModified();
    }
    update(device, densityArray) {
      this.augmentations = new ArrayBuffer(
        Uint32Array.BYTES_PER_ELEMENT * 4 + Uint32Array.BYTES_PER_ELEMENT * densityArray.length * 8
      );
      const header = new Uint32Array(this.augmentations, 0, 4);
      header[0] = densityArray.length;
      const augmentations = new Float32Array(this.augmentations, Uint32Array.BYTES_PER_ELEMENT * 4);
      const intAugmentations = new Uint32Array(this.augmentations, Uint32Array.BYTES_PER_ELEMENT * 4);
      for (let i = 0; i < densityArray.length; i++) {
        augmentations[i * 8] = densityArray[i].x;
        augmentations[i * 8 + 1] = densityArray[i].y;
        augmentations[i * 8 + 2] = densityArray[i].z;
        augmentations[i * 8 + 3] = densityArray[i].size;
        intAugmentations[i * 8 + 4] = densityArray[i].type | densityArray[i].shape << 1 | densityArray[i].material << 8;
      }
      device.queue.writeBuffer(
        this.augmentationBuffer,
        0,
        this.augmentations,
        0,
        this.augmentations.byteLength
      );
    }
    updateRaw(device, densityArray) {
      this.augmentations = densityArray;
      device.queue.writeBuffer(
        this.augmentationBuffer,
        0,
        this.augmentations,
        0,
        this.augmentations.byteLength
      );
    }
    static patch(shader) {
      return shader.replace("#import density", DensityShader);
    }
  }
  var commonjsGlobal = typeof globalThis !== "undefined" ? globalThis : typeof window !== "undefined" ? window : typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : {};
  function getDefaultExportFromCjs(x) {
    return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, "default") ? x["default"] : x;
  }
  function getAugmentedNamespace(n) {
    if (n.__esModule)
      return n;
    var f = n.default;
    if (typeof f == "function") {
      var a = function a2() {
        if (this instanceof a2) {
          var args = [null];
          args.push.apply(args, arguments);
          var Ctor = Function.bind.apply(f, args);
          return new Ctor();
        }
        return f.apply(this, arguments);
      };
      a.prototype = f.prototype;
    } else
      a = {};
    Object.defineProperty(a, "__esModule", { value: true });
    Object.keys(n).forEach(function(k) {
      var d = Object.getOwnPropertyDescriptor(n, k);
      Object.defineProperty(a, k, d.get ? d : {
        enumerable: true,
        get: function() {
          return n[k];
        }
      });
    });
    return a;
  }
  var alea$1 = { exports: {} };
  alea$1.exports;
  (function(module) {
    (function(global2, module2, define) {
      function Alea(seed) {
        var me = this, mash = Mash();
        me.next = function() {
          var t = 2091639 * me.s0 + me.c * 23283064365386963e-26;
          me.s0 = me.s1;
          me.s1 = me.s2;
          return me.s2 = t - (me.c = t | 0);
        };
        me.c = 1;
        me.s0 = mash(" ");
        me.s1 = mash(" ");
        me.s2 = mash(" ");
        me.s0 -= mash(seed);
        if (me.s0 < 0) {
          me.s0 += 1;
        }
        me.s1 -= mash(seed);
        if (me.s1 < 0) {
          me.s1 += 1;
        }
        me.s2 -= mash(seed);
        if (me.s2 < 0) {
          me.s2 += 1;
        }
        mash = null;
      }
      function copy(f, t) {
        t.c = f.c;
        t.s0 = f.s0;
        t.s1 = f.s1;
        t.s2 = f.s2;
        return t;
      }
      function impl(seed, opts) {
        var xg = new Alea(seed), state = opts && opts.state, prng = xg.next;
        prng.int32 = function() {
          return xg.next() * 4294967296 | 0;
        };
        prng.double = function() {
          return prng() + (prng() * 2097152 | 0) * 11102230246251565e-32;
        };
        prng.quick = prng;
        if (state) {
          if (typeof state == "object")
            copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      function Mash() {
        var n = 4022871197;
        var mash = function(data) {
          data = String(data);
          for (var i = 0; i < data.length; i++) {
            n += data.charCodeAt(i);
            var h = 0.02519603282416938 * n;
            n = h >>> 0;
            h -= n;
            h *= n;
            n = h >>> 0;
            h -= n;
            n += h * 4294967296;
          }
          return (n >>> 0) * 23283064365386963e-26;
        };
        return mash;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define && define.amd) {
        define(function() {
          return impl;
        });
      } else {
        this.alea = impl;
      }
    })(
      commonjsGlobal,
      module,
      // present in node.js
      false
      // present with an AMD loader
    );
  })(alea$1);
  var aleaExports = alea$1.exports;
  var xor128$1 = { exports: {} };
  xor128$1.exports;
  (function(module) {
    (function(global2, module2, define) {
      function XorGen(seed) {
        var me = this, strseed = "";
        me.x = 0;
        me.y = 0;
        me.z = 0;
        me.w = 0;
        me.next = function() {
          var t = me.x ^ me.x << 11;
          me.x = me.y;
          me.y = me.z;
          me.z = me.w;
          return me.w ^= me.w >>> 19 ^ t ^ t >>> 8;
        };
        if (seed === (seed | 0)) {
          me.x = seed;
        } else {
          strseed += seed;
        }
        for (var k = 0; k < strseed.length + 64; k++) {
          me.x ^= strseed.charCodeAt(k) | 0;
          me.next();
        }
      }
      function copy(f, t) {
        t.x = f.x;
        t.y = f.y;
        t.z = f.z;
        t.w = f.w;
        return t;
      }
      function impl(seed, opts) {
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result = (top + bot) / (1 << 21);
          } while (result === 0);
          return result;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (typeof state == "object")
            copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define && define.amd) {
        define(function() {
          return impl;
        });
      } else {
        this.xor128 = impl;
      }
    })(
      commonjsGlobal,
      module,
      // present in node.js
      false
      // present with an AMD loader
    );
  })(xor128$1);
  var xor128Exports = xor128$1.exports;
  var xorwow$1 = { exports: {} };
  xorwow$1.exports;
  (function(module) {
    (function(global2, module2, define) {
      function XorGen(seed) {
        var me = this, strseed = "";
        me.next = function() {
          var t = me.x ^ me.x >>> 2;
          me.x = me.y;
          me.y = me.z;
          me.z = me.w;
          me.w = me.v;
          return (me.d = me.d + 362437 | 0) + (me.v = me.v ^ me.v << 4 ^ (t ^ t << 1)) | 0;
        };
        me.x = 0;
        me.y = 0;
        me.z = 0;
        me.w = 0;
        me.v = 0;
        if (seed === (seed | 0)) {
          me.x = seed;
        } else {
          strseed += seed;
        }
        for (var k = 0; k < strseed.length + 64; k++) {
          me.x ^= strseed.charCodeAt(k) | 0;
          if (k == strseed.length) {
            me.d = me.x << 10 ^ me.x >>> 4;
          }
          me.next();
        }
      }
      function copy(f, t) {
        t.x = f.x;
        t.y = f.y;
        t.z = f.z;
        t.w = f.w;
        t.v = f.v;
        t.d = f.d;
        return t;
      }
      function impl(seed, opts) {
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result = (top + bot) / (1 << 21);
          } while (result === 0);
          return result;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (typeof state == "object")
            copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define && define.amd) {
        define(function() {
          return impl;
        });
      } else {
        this.xorwow = impl;
      }
    })(
      commonjsGlobal,
      module,
      // present in node.js
      false
      // present with an AMD loader
    );
  })(xorwow$1);
  var xorwowExports = xorwow$1.exports;
  var xorshift7$1 = { exports: {} };
  xorshift7$1.exports;
  (function(module) {
    (function(global2, module2, define) {
      function XorGen(seed) {
        var me = this;
        me.next = function() {
          var X = me.x, i = me.i, t, v;
          t = X[i];
          t ^= t >>> 7;
          v = t ^ t << 24;
          t = X[i + 1 & 7];
          v ^= t ^ t >>> 10;
          t = X[i + 3 & 7];
          v ^= t ^ t >>> 3;
          t = X[i + 4 & 7];
          v ^= t ^ t << 7;
          t = X[i + 7 & 7];
          t = t ^ t << 13;
          v ^= t ^ t << 9;
          X[i] = v;
          me.i = i + 1 & 7;
          return v;
        };
        function init(me2, seed2) {
          var j, X = [];
          if (seed2 === (seed2 | 0)) {
            X[0] = seed2;
          } else {
            seed2 = "" + seed2;
            for (j = 0; j < seed2.length; ++j) {
              X[j & 7] = X[j & 7] << 15 ^ seed2.charCodeAt(j) + X[j + 1 & 7] << 13;
            }
          }
          while (X.length < 8)
            X.push(0);
          for (j = 0; j < 8 && X[j] === 0; ++j)
            ;
          if (j == 8)
            X[7] = -1;
          else
            X[j];
          me2.x = X;
          me2.i = 0;
          for (j = 256; j > 0; --j) {
            me2.next();
          }
        }
        init(me, seed);
      }
      function copy(f, t) {
        t.x = f.x.slice();
        t.i = f.i;
        return t;
      }
      function impl(seed, opts) {
        if (seed == null)
          seed = +/* @__PURE__ */ new Date();
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result = (top + bot) / (1 << 21);
          } while (result === 0);
          return result;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (state.x)
            copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define && define.amd) {
        define(function() {
          return impl;
        });
      } else {
        this.xorshift7 = impl;
      }
    })(
      commonjsGlobal,
      module,
      // present in node.js
      false
      // present with an AMD loader
    );
  })(xorshift7$1);
  var xorshift7Exports = xorshift7$1.exports;
  var xor4096$1 = { exports: {} };
  xor4096$1.exports;
  (function(module) {
    (function(global2, module2, define) {
      function XorGen(seed) {
        var me = this;
        me.next = function() {
          var w = me.w, X = me.X, i = me.i, t, v;
          me.w = w = w + 1640531527 | 0;
          v = X[i + 34 & 127];
          t = X[i = i + 1 & 127];
          v ^= v << 13;
          t ^= t << 17;
          v ^= v >>> 15;
          t ^= t >>> 12;
          v = X[i] = v ^ t;
          me.i = i;
          return v + (w ^ w >>> 16) | 0;
        };
        function init(me2, seed2) {
          var t, v, i, j, w, X = [], limit = 128;
          if (seed2 === (seed2 | 0)) {
            v = seed2;
            seed2 = null;
          } else {
            seed2 = seed2 + "\0";
            v = 0;
            limit = Math.max(limit, seed2.length);
          }
          for (i = 0, j = -32; j < limit; ++j) {
            if (seed2)
              v ^= seed2.charCodeAt((j + 32) % seed2.length);
            if (j === 0)
              w = v;
            v ^= v << 10;
            v ^= v >>> 15;
            v ^= v << 4;
            v ^= v >>> 13;
            if (j >= 0) {
              w = w + 1640531527 | 0;
              t = X[j & 127] ^= v + w;
              i = 0 == t ? i + 1 : 0;
            }
          }
          if (i >= 128) {
            X[(seed2 && seed2.length || 0) & 127] = -1;
          }
          i = 127;
          for (j = 4 * 128; j > 0; --j) {
            v = X[i + 34 & 127];
            t = X[i = i + 1 & 127];
            v ^= v << 13;
            t ^= t << 17;
            v ^= v >>> 15;
            t ^= t >>> 12;
            X[i] = v ^ t;
          }
          me2.w = w;
          me2.X = X;
          me2.i = i;
        }
        init(me, seed);
      }
      function copy(f, t) {
        t.i = f.i;
        t.w = f.w;
        t.X = f.X.slice();
        return t;
      }
      function impl(seed, opts) {
        if (seed == null)
          seed = +/* @__PURE__ */ new Date();
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result = (top + bot) / (1 << 21);
          } while (result === 0);
          return result;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (state.X)
            copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define && define.amd) {
        define(function() {
          return impl;
        });
      } else {
        this.xor4096 = impl;
      }
    })(
      commonjsGlobal,
      // window object or global
      module,
      // present in node.js
      false
      // present with an AMD loader
    );
  })(xor4096$1);
  var xor4096Exports = xor4096$1.exports;
  var tychei$1 = { exports: {} };
  tychei$1.exports;
  (function(module) {
    (function(global2, module2, define) {
      function XorGen(seed) {
        var me = this, strseed = "";
        me.next = function() {
          var b = me.b, c = me.c, d = me.d, a = me.a;
          b = b << 25 ^ b >>> 7 ^ c;
          c = c - d | 0;
          d = d << 24 ^ d >>> 8 ^ a;
          a = a - b | 0;
          me.b = b = b << 20 ^ b >>> 12 ^ c;
          me.c = c = c - d | 0;
          me.d = d << 16 ^ c >>> 16 ^ a;
          return me.a = a - b | 0;
        };
        me.a = 0;
        me.b = 0;
        me.c = 2654435769 | 0;
        me.d = 1367130551;
        if (seed === Math.floor(seed)) {
          me.a = seed / 4294967296 | 0;
          me.b = seed | 0;
        } else {
          strseed += seed;
        }
        for (var k = 0; k < strseed.length + 20; k++) {
          me.b ^= strseed.charCodeAt(k) | 0;
          me.next();
        }
      }
      function copy(f, t) {
        t.a = f.a;
        t.b = f.b;
        t.c = f.c;
        t.d = f.d;
        return t;
      }
      function impl(seed, opts) {
        var xg = new XorGen(seed), state = opts && opts.state, prng = function() {
          return (xg.next() >>> 0) / 4294967296;
        };
        prng.double = function() {
          do {
            var top = xg.next() >>> 11, bot = (xg.next() >>> 0) / 4294967296, result = (top + bot) / (1 << 21);
          } while (result === 0);
          return result;
        };
        prng.int32 = xg.next;
        prng.quick = prng;
        if (state) {
          if (typeof state == "object")
            copy(state, xg);
          prng.state = function() {
            return copy(xg, {});
          };
        }
        return prng;
      }
      if (module2 && module2.exports) {
        module2.exports = impl;
      } else if (define && define.amd) {
        define(function() {
          return impl;
        });
      } else {
        this.tychei = impl;
      }
    })(
      commonjsGlobal,
      module,
      // present in node.js
      false
      // present with an AMD loader
    );
  })(tychei$1);
  var tycheiExports = tychei$1.exports;
  var seedrandom$1 = { exports: {} };
  var __viteBrowserExternal = {};
  var __viteBrowserExternal$1 = /* @__PURE__ */ Object.freeze({
    __proto__: null,
    default: __viteBrowserExternal
  });
  var require$$0 = /* @__PURE__ */ getAugmentedNamespace(__viteBrowserExternal$1);
  (function(module) {
    (function(global2, pool, math) {
      var width = 256, chunks = 6, digits = 52, rngname = "random", startdenom = math.pow(width, chunks), significance = math.pow(2, digits), overflow = significance * 2, mask = width - 1, nodecrypto;
      function seedrandom2(seed, options, callback) {
        var key = [];
        options = options == true ? { entropy: true } : options || {};
        var shortseed = mixkey(flatten(
          options.entropy ? [seed, tostring(pool)] : seed == null ? autoseed() : seed,
          3
        ), key);
        var arc4 = new ARC4(key);
        var prng = function() {
          var n = arc4.g(chunks), d = startdenom, x = 0;
          while (n < significance) {
            n = (n + x) * width;
            d *= width;
            x = arc4.g(1);
          }
          while (n >= overflow) {
            n /= 2;
            d /= 2;
            x >>>= 1;
          }
          return (n + x) / d;
        };
        prng.int32 = function() {
          return arc4.g(4) | 0;
        };
        prng.quick = function() {
          return arc4.g(4) / 4294967296;
        };
        prng.double = prng;
        mixkey(tostring(arc4.S), pool);
        return (options.pass || callback || function(prng2, seed2, is_math_call, state) {
          if (state) {
            if (state.S) {
              copy(state, arc4);
            }
            prng2.state = function() {
              return copy(arc4, {});
            };
          }
          if (is_math_call) {
            math[rngname] = prng2;
            return seed2;
          } else
            return prng2;
        })(
          prng,
          shortseed,
          "global" in options ? options.global : this == math,
          options.state
        );
      }
      function ARC4(key) {
        var t, keylen = key.length, me = this, i = 0, j = me.i = me.j = 0, s = me.S = [];
        if (!keylen) {
          key = [keylen++];
        }
        while (i < width) {
          s[i] = i++;
        }
        for (i = 0; i < width; i++) {
          s[i] = s[j = mask & j + key[i % keylen] + (t = s[i])];
          s[j] = t;
        }
        (me.g = function(count) {
          var t2, r = 0, i2 = me.i, j2 = me.j, s2 = me.S;
          while (count--) {
            t2 = s2[i2 = mask & i2 + 1];
            r = r * width + s2[mask & (s2[i2] = s2[j2 = mask & j2 + t2]) + (s2[j2] = t2)];
          }
          me.i = i2;
          me.j = j2;
          return r;
        })(width);
      }
      function copy(f, t) {
        t.i = f.i;
        t.j = f.j;
        t.S = f.S.slice();
        return t;
      }
      function flatten(obj, depth) {
        var result = [], typ = typeof obj, prop;
        if (depth && typ == "object") {
          for (prop in obj) {
            try {
              result.push(flatten(obj[prop], depth - 1));
            } catch (e) {
            }
          }
        }
        return result.length ? result : typ == "string" ? obj : obj + "\0";
      }
      function mixkey(seed, key) {
        var stringseed = seed + "", smear, j = 0;
        while (j < stringseed.length) {
          key[mask & j] = mask & (smear ^= key[mask & j] * 19) + stringseed.charCodeAt(j++);
        }
        return tostring(key);
      }
      function autoseed() {
        try {
          var out;
          if (nodecrypto && (out = nodecrypto.randomBytes)) {
            out = out(width);
          } else {
            out = new Uint8Array(width);
            (global2.crypto || global2.msCrypto).getRandomValues(out);
          }
          return tostring(out);
        } catch (e) {
          var browser = global2.navigator, plugins = browser && browser.plugins;
          return [+/* @__PURE__ */ new Date(), global2, plugins, global2.screen, tostring(pool)];
        }
      }
      function tostring(a) {
        return String.fromCharCode.apply(0, a);
      }
      mixkey(math.random(), pool);
      if (module.exports) {
        module.exports = seedrandom2;
        try {
          nodecrypto = require$$0;
        } catch (ex) {
        }
      } else {
        math["seed" + rngname] = seedrandom2;
      }
    })(
      // global: `self` in browsers (including strict mode and web workers),
      // otherwise `this` in Node and other environments
      typeof self !== "undefined" ? self : commonjsGlobal,
      [],
      // pool: entropy pool starts empty
      Math
      // math: package containing random, pow, and seedrandom
    );
  })(seedrandom$1);
  var seedrandomExports = seedrandom$1.exports;
  var alea = aleaExports;
  var xor128 = xor128Exports;
  var xorwow = xorwowExports;
  var xorshift7 = xorshift7Exports;
  var xor4096 = xor4096Exports;
  var tychei = tycheiExports;
  var sr = seedrandomExports;
  sr.alea = alea;
  sr.xor128 = xor128;
  sr.xorwow = xorwow;
  sr.xorshift7 = xorshift7;
  sr.xor4096 = xor4096;
  sr.tychei = tychei;
  var seedrandom = sr;
  var Random = /* @__PURE__ */ getDefaultExportFromCjs(seedrandom);
  const MATERIAL_AIR = 0;
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
  ];
  const cellProcEdgeMask = [
    [0, 1, 2, 3, 0],
    [4, 5, 6, 7, 0],
    [0, 4, 1, 5, 1],
    [2, 6, 3, 7, 1],
    [0, 2, 4, 6, 2],
    [1, 3, 5, 7, 2]
  ];
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
  ];
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
  ];
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
  ];
  const processEdgeMask = [
    [3, 2, 1, 0],
    [7, 5, 6, 4],
    [11, 10, 9, 8]
  ];
  const edgevmap = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    // x-axis
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    // y-axis
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7]
    // z-axis
  ];
  const ContourProcessEdge = (node, dir, indices) => {
    let minSize = 1e6;
    let minIndex = 0;
    const indexes = [-1, -1, -1, -1];
    let flip = false;
    const signChange = [false, false, false, false];
    for (let i = 0; i < 4; i++) {
      const edge = processEdgeMask[dir][i];
      const c1 = edgevmap[edge][0];
      const c2 = edgevmap[edge][1];
      const m1 = node[i].drawInfo.corners >> c1 & 1;
      const m2 = node[i].drawInfo.corners >> c2 & 1;
      if (node[i].size < minSize) {
        minSize = node[i].size;
        minIndex = i;
        flip = m1 !== MATERIAL_AIR;
      }
      indexes[i] = node[i].drawInfo.index;
      signChange[i] = m1 === MATERIAL_AIR && m2 !== MATERIAL_AIR || m1 !== MATERIAL_AIR && m2 === MATERIAL_AIR;
    }
    if (signChange[minIndex]) {
      if (!flip) {
        indices.push(indexes[0]);
        indices.push(indexes[1]);
        indices.push(indexes[3]);
        indices.push(indexes[0]);
        indices.push(indexes[3]);
        indices.push(indexes[2]);
      } else {
        indices.push(indexes[0]);
        indices.push(indexes[3]);
        indices.push(indexes[1]);
        indices.push(indexes[0]);
        indices.push(indexes[2]);
        indices.push(indexes[3]);
      }
    }
  };
  const ContourEdgeProc = (node, dir, indices) => {
    if (node[0] == null || node[1] == null || node[2] == null || node[3] == null)
      return;
    if (node[0].type !== "internal" && node[1].type !== "internal" && node[2].type !== "internal" && node[3].type !== "internal") {
      ContourProcessEdge(node, dir, indices);
    } else {
      for (let i = 0; i < 2; i++) {
        const edgeNodes = [];
        const c = [
          edgeProcEdgeMask[dir][i][0],
          edgeProcEdgeMask[dir][i][1],
          edgeProcEdgeMask[dir][i][2],
          edgeProcEdgeMask[dir][i][3]
        ];
        for (let j = 0; j < 4; j++) {
          if (node[j].type === "leaf" || node[j].type === "pseudo") {
            edgeNodes[j] = node[j];
          } else {
            edgeNodes[j] = node[j].children[c[j]];
          }
        }
        ContourEdgeProc(edgeNodes, edgeProcEdgeMask[dir][i][4], indices);
      }
    }
  };
  const ContourFaceProc = (node, dir, indices) => {
    if (node[0] == null || node[1] == null)
      return;
    if (node[0].type === "internal" || node[1].type === "internal") {
      for (let i = 0; i < 4; i++) {
        const faceNodes = [];
        const c = [faceProcFaceMask[dir][i][0], faceProcFaceMask[dir][i][1]];
        for (let j = 0; j < 2; j++) {
          if (node[j].type !== "internal") {
            faceNodes[j] = node[j];
          } else {
            faceNodes[j] = node[j].children[c[j]];
          }
        }
        ContourFaceProc(faceNodes, faceProcFaceMask[dir][i][2], indices);
      }
      const orders = [
        [0, 0, 1, 1],
        [0, 1, 0, 1]
      ];
      for (let i = 0; i < 4; i++) {
        const edgeNodes = [];
        const c = [
          faceProcEdgeMask[dir][i][1],
          faceProcEdgeMask[dir][i][2],
          faceProcEdgeMask[dir][i][3],
          faceProcEdgeMask[dir][i][4]
        ];
        const order = [
          orders[faceProcEdgeMask[dir][i][0]][0],
          orders[faceProcEdgeMask[dir][i][0]][1],
          orders[faceProcEdgeMask[dir][i][0]][2],
          orders[faceProcEdgeMask[dir][i][0]][3]
        ];
        for (let j = 0; j < 4; j++) {
          if (node[order[j]].type === "leaf" || node[order[j]].type === "pseudo") {
            edgeNodes[j] = node[order[j]];
          } else {
            edgeNodes[j] = node[order[j]].children[c[j]];
          }
        }
        ContourEdgeProc(edgeNodes, faceProcEdgeMask[dir][i][5], indices);
      }
    }
  };
  const ContourCellProc = (node, indices) => {
    if (node == null)
      return;
    if (node.type === "internal") {
      for (let i = 0; i < 8; i++) {
        ContourCellProc(node.children[i], indices);
      }
      for (let i = 0; i < 12; i++) {
        const faceNodes = [];
        const c = [cellProcFaceMask[i][0], cellProcFaceMask[i][1]];
        faceNodes[0] = node.children[c[0]];
        faceNodes[1] = node.children[c[1]];
        ContourFaceProc(faceNodes, cellProcFaceMask[i][2], indices);
      }
      for (let i = 0; i < 6; i++) {
        const edgeNodes = [];
        const c = [
          cellProcEdgeMask[i][0],
          cellProcEdgeMask[i][1],
          cellProcEdgeMask[i][2],
          cellProcEdgeMask[i][3]
        ];
        for (let j = 0; j < 4; j++) {
          edgeNodes[j] = node.children[c[j]];
        }
        ContourEdgeProc(edgeNodes, cellProcEdgeMask[i][4], indices);
      }
    }
  };
  const CHILD_MIN_OFFSETS = [
    fromValues(0, 0, 0),
    fromValues(0, 0, 1),
    fromValues(0, 1, 0),
    fromValues(0, 1, 1),
    fromValues(1, 0, 0),
    fromValues(1, 0, 1),
    fromValues(1, 1, 0),
    fromValues(1, 1, 1)
  ];
  const constructParents = (children, position, parentSize) => {
    const parentsHash = /* @__PURE__ */ new Map();
    for (let i = 0; i < children.length; i++) {
      const node = children[i];
      const parentPos = sub(
        create(),
        node.min,
        fromValues(
          (node.min[0] - position[0]) % parentSize,
          (node.min[1] - position[1]) % parentSize,
          (node.min[2] - position[2]) % parentSize
        )
      );
      let parent = parentsHash[`${parentPos[0]},${parentPos[1]},${parentPos[2]}`];
      if (!parent) {
        parent = {
          min: parentPos,
          size: parentSize,
          type: "internal",
          children: []
        };
        parentsHash[`${parent.min[0]},${parent.min[1]},${parent.min[2]}`] = parent;
      }
      for (let j = 0; j < 8; j++) {
        const childMin = add(
          create(),
          parentPos,
          fromValues(
            CHILD_MIN_OFFSETS[j][0] * node.size,
            CHILD_MIN_OFFSETS[j][1] * node.size,
            CHILD_MIN_OFFSETS[j][2] * node.size
          )
        );
        if (equals(childMin, node.min)) {
          parent.children[j] = node;
          break;
        }
      }
    }
    children.length = 0;
    return Object.values(parentsHash);
  };
  const constructTreeUpwards = (nodes, rootMin, rootNodeSize) => {
    if (nodes.length == 0) {
      return null;
    }
    nodes.sort((lhs, rhs) => lhs.size - rhs.size);
    while (nodes[0].size != nodes[nodes.length - 1].size) {
      let iter = 0;
      const size = nodes[iter].size;
      do {
        ++iter;
      } while (nodes[iter].size == size);
      let newNodes = [];
      for (let i = 0; i < iter; i++)
        newNodes.push(nodes[i]);
      newNodes = constructParents(newNodes, rootMin, size * 2);
      for (let i = iter; i < nodes.Count; i++)
        newNodes.push(nodes[i]);
      nodes.length = 0;
      for (let i = 0; i < newNodes.length; i++)
        nodes.push(newNodes[i]);
    }
    let parentSize = nodes[0].size * 2;
    while (parentSize <= rootNodeSize) {
      nodes = constructParents(nodes, rootMin, parentSize);
      parentSize *= 2;
    }
    if (nodes.length != 1) {
      console.log(nodes);
      console.error("There can only be one root node!");
      return null;
    }
    return nodes[0];
  };
  const generateVertexIndices = (node, vertices, normals, nodeSize) => {
    if (node == null)
      return;
    if (node.size > nodeSize) {
      if (node.type !== "leaf") {
        for (let i = 0; i < 8; i++) {
          generateVertexIndices(node.children[i], vertices, normals, nodeSize);
        }
      }
    }
    if (node.type !== "internal") {
      const d = node.drawInfo;
      if (d == null) {
        throw "Error! Could not add vertex!";
      }
      d.index = vertices.length / 3;
      vertices.push(d.position[0], d.position[1], d.position[2]);
      normals.push(d.averageNormal[0], d.averageNormal[1], d.averageNormal[2]);
    }
  };
  const computeVoxels = (position, stride, voxelCount, computedVoxelsData) => {
    const computedVoxels = [];
    if (voxelCount === 0) {
      return {
        vertices: new Float32Array(),
        normals: new Float32Array(),
        indices: new Uint16Array(),
        corners: new Uint32Array()
      };
    }
    for (let i = 0; i < voxelCount * 12; i += 12) {
      if (computedVoxelsData[i + 11] !== 0) {
        const leaf = {
          type: "leaf",
          size: stride,
          min: fromValues(
            computedVoxelsData[i],
            computedVoxelsData[i + 1],
            computedVoxelsData[i + 2]
          ),
          drawInfo: {
            position: fromValues(
              computedVoxelsData[i + 4],
              computedVoxelsData[i + 5],
              computedVoxelsData[i + 6]
            ),
            averageNormal: fromValues(
              computedVoxelsData[i + 8],
              computedVoxelsData[i + 9],
              computedVoxelsData[i + 10]
            ),
            corners: computedVoxelsData[i + 3]
          }
        };
        computedVoxels.push(leaf);
      }
    }
    const tree = constructTreeUpwards(computedVoxels, position, 32 * stride);
    const vertices = [];
    const normals = [];
    generateVertexIndices(tree, vertices, normals, 1);
    const indices = [];
    ContourCellProc(tree, indices);
    return {
      vertices: new Float32Array(vertices),
      normals: new Float32Array(normals),
      indices: new Uint16Array(indices),
      corners: new Uint32Array()
    };
  };
  class Voxel {
    running = false;
    computePipeline;
    computeCornersPipeline;
    uniformBuffer;
    cornerMaterials;
    cornerMaterialsRead;
    voxelMaterialsBuffer;
    voxelMaterialsBufferRead;
    cornerIndexBuffer;
    gpuReadBuffer;
    permutationsBuffer;
    voxelsBuffer;
    computeBindGroup;
    computeCornersBindGroup;
    computePositionsPipeline;
    computePositionsBindGroup;
    computeVoxelsPipeline;
    computeVoxelsBindGroup;
    voxelReadBuffer;
    density;
    densityBindGroup;
    mainDensityBindGroup;
    constructor(computePipeline, computeCornersPipeline, uniformBuffer, cornerMaterials, cornerMaterialsRead, voxelMaterialsBuffer, voxelMaterialsBufferRead, cornerIndexBuffer, gpuReadBuffer, permutationsBuffer, voxelsBuffer, computeBindGroup, computeCornersBindGroup, computePositionsPipeline, computePositionsBindGroup, computeVoxelsPipeline, computeVoxelsBindGroup, voxelReadBuffer, density, densityBindGroup, mainDensityBindGroup) {
      this.computePipeline = computePipeline;
      this.computeCornersPipeline = computeCornersPipeline;
      this.uniformBuffer = uniformBuffer;
      this.cornerMaterials = cornerMaterials;
      this.cornerMaterialsRead = cornerMaterialsRead;
      this.voxelMaterialsBuffer = voxelMaterialsBuffer;
      this.voxelMaterialsBufferRead = voxelMaterialsBufferRead;
      this.cornerIndexBuffer = cornerIndexBuffer;
      this.gpuReadBuffer = gpuReadBuffer;
      this.permutationsBuffer = permutationsBuffer;
      this.voxelsBuffer = voxelsBuffer;
      this.computeBindGroup = computeBindGroup;
      this.computeCornersBindGroup = computeCornersBindGroup;
      this.computePositionsPipeline = computePositionsPipeline;
      this.computePositionsBindGroup = computePositionsBindGroup;
      this.computeVoxelsPipeline = computeVoxelsPipeline;
      this.computeVoxelsBindGroup = computeVoxelsBindGroup;
      this.voxelReadBuffer = voxelReadBuffer;
      this.density = density;
      this.densityBindGroup = densityBindGroup;
      this.mainDensityBindGroup = mainDensityBindGroup;
    }
    static async init(device) {
      const computeVoxelsCode = Density.patch(ComputeVoxels);
      const start = performance.now();
      console.log("Start loading voxel engine", performance.now() - start);
      const module = device.createShaderModule({
        code: computeVoxelsCode
      });
      const computePipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module,
          entryPoint: "computeMaterials"
        }
      });
      const computeCornersPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: device.createShaderModule({
            code: ComputeCorners
          }),
          entryPoint: "main"
        }
      });
      const uniformBufferSize = Math.max(4 * 5, 32);
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      const cornerMaterials = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT * 33 * 33 * 33,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      });
      const cornerMaterialsRead = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT * 33 * 33 * 33,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const voxelMaterialsBuffer = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      });
      const voxelMaterialsBufferRead = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const cornerIndexBuffer = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT + Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      });
      const gpuReadBuffer = device.createBuffer({
        size: Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const permutations = new Int32Array(512);
      const random = new Random(6452);
      for (let i = 0; i < 256; i++)
        permutations[i] = 256 * random();
      for (let i = 256; i < 512; i++)
        permutations[i] = permutations[i - 256];
      const permutationsBuffer = device.createBuffer({
        size: permutations.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Int32Array(permutationsBuffer.getMappedRange()).set(permutations);
      permutationsBuffer.unmap();
      const voxelsBuffer = device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 12 * 32 * 32 * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
      });
      const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 1,
            resource: {
              buffer: cornerMaterials
            }
          },
          {
            binding: 5,
            resource: {
              buffer: uniformBuffer
            }
          }
        ]
      });
      const computeCornersBindGroup = device.createBindGroup({
        layout: computeCornersPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 1,
            resource: {
              buffer: cornerMaterials
            }
          },
          {
            binding: 2,
            resource: {
              buffer: voxelMaterialsBuffer
            }
          }
        ]
      });
      const computePositionsPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: device.createShaderModule({
            code: ComputePositions
          }),
          entryPoint: "main"
        }
      });
      const computePositionsBindGroup = device.createBindGroup({
        layout: computePositionsPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 2,
            resource: {
              buffer: voxelMaterialsBuffer
            }
          },
          {
            binding: 3,
            resource: {
              buffer: cornerIndexBuffer
            }
          }
        ]
      });
      const computeVoxelsPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module,
          entryPoint: "main"
        }
      });
      const computeVoxelsBindGroup = device.createBindGroup({
        layout: computeVoxelsPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 2,
            resource: {
              buffer: voxelMaterialsBuffer
            }
          },
          {
            binding: 3,
            resource: {
              buffer: cornerIndexBuffer
            }
          },
          {
            binding: 4,
            resource: {
              buffer: voxelsBuffer
            }
          },
          {
            binding: 5,
            resource: {
              buffer: uniformBuffer
            }
          }
        ]
      });
      const voxelReadBuffer = device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 12 * 32 * 32 * 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const density = await Density.init(device);
      const densityBindGroup = await density.apply(device, computePipeline);
      const mainDensityBindGroup = await density.apply(device, computeVoxelsPipeline);
      console.log("Done", performance.now() - start);
      return new Voxel(
        computePipeline,
        computeCornersPipeline,
        uniformBuffer,
        cornerMaterials,
        cornerMaterialsRead,
        voxelMaterialsBuffer,
        voxelMaterialsBufferRead,
        cornerIndexBuffer,
        gpuReadBuffer,
        permutationsBuffer,
        voxelsBuffer,
        computeBindGroup,
        computeCornersBindGroup,
        computePositionsPipeline,
        computePositionsBindGroup,
        computeVoxelsPipeline,
        computeVoxelsBindGroup,
        voxelReadBuffer,
        density,
        densityBindGroup,
        mainDensityBindGroup
      );
    }
    generate(device, queue, position, stride, density) {
      if (!stride)
        stride = 1;
      return new Promise((resolve) => {
        this.density.updateRaw(device, density);
        const permutations = new Int32Array(512);
        const random = new Random("James");
        for (let i = 0; i < 256; i++)
          permutations[i] = 256 * random();
        for (let i = 256; i < 512; i++)
          permutations[i] = permutations[i - 256];
        device.queue.writeBuffer(
          this.permutationsBuffer,
          0,
          permutations.buffer,
          permutations.byteOffset,
          permutations.byteLength
        );
        const buffer = new ArrayBuffer(4 * 5);
        const uniform = new Float32Array(buffer, 0, 4);
        uniform.set(position, 0);
        uniform[3] = stride;
        new Uint32Array(buffer, 16, 1)[0] = 33;
        device.queue.writeBuffer(this.uniformBuffer, 0, buffer, 0, buffer.byteLength);
        const computeEncoder = device.createCommandEncoder();
        const octreeSize = 32;
        const computePassEncoder = computeEncoder.beginComputePass();
        computePassEncoder.setPipeline(this.computePipeline);
        computePassEncoder.setBindGroup(0, this.computeBindGroup);
        this.densityBindGroup.apply(computePassEncoder);
        computePassEncoder.dispatchWorkgroups(octreeSize + 1, octreeSize + 1, octreeSize + 1);
        computePassEncoder.end();
        const computeCornersPass = computeEncoder.beginComputePass();
        computeCornersPass.setPipeline(this.computeCornersPipeline);
        computeCornersPass.setBindGroup(0, this.computeCornersBindGroup);
        computeCornersPass.dispatchWorkgroups(octreeSize, octreeSize, octreeSize);
        computeCornersPass.end();
        const computePositionsPass = computeEncoder.beginComputePass();
        computePositionsPass.setPipeline(this.computePositionsPipeline);
        computePositionsPass.setBindGroup(0, this.computePositionsBindGroup);
        computePositionsPass.dispatchWorkgroups(1);
        computePositionsPass.end();
        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
          this.cornerIndexBuffer,
          0,
          this.gpuReadBuffer,
          0,
          Uint32Array.BYTES_PER_ELEMENT
        );
        copyEncoder.copyBufferToBuffer(
          this.cornerMaterials,
          0,
          this.cornerMaterialsRead,
          0,
          Uint32Array.BYTES_PER_ELEMENT * 33 * 33 * 33
        );
        copyEncoder.copyBufferToBuffer(
          this.voxelMaterialsBuffer,
          0,
          this.voxelMaterialsBufferRead,
          0,
          Uint32Array.BYTES_PER_ELEMENT * 32 * 32 * 32
        );
        queue({
          items: [computeEncoder.finish(), copyEncoder.finish()],
          callback: async () => {
            await this.cornerMaterialsRead.mapAsync(GPUMapMode.READ);
            const corners = new Uint32Array(this.cornerMaterialsRead.getMappedRange()).slice();
            this.cornerMaterialsRead.unmap();
            await this.gpuReadBuffer.mapAsync(GPUMapMode.READ);
            const arrayBuffer = this.gpuReadBuffer.getMappedRange();
            const voxelCount = new Uint32Array(arrayBuffer)[0];
            this.gpuReadBuffer.unmap();
            if (voxelCount === 0) {
              resolve({
                vertices: new Float32Array(),
                normals: new Float32Array(),
                indices: new Uint16Array(),
                corners,
                consistency: corners[0]
              });
              return;
            }
            const dispatchCount = Math.ceil(voxelCount / 128);
            const computeEncoder2 = device.createCommandEncoder();
            const computePassEncoder2 = computeEncoder2.beginComputePass();
            computePassEncoder2.setPipeline(this.computeVoxelsPipeline);
            computePassEncoder2.setBindGroup(0, this.computeVoxelsBindGroup);
            this.mainDensityBindGroup.apply(computePassEncoder2);
            computePassEncoder2.dispatchWorkgroups(dispatchCount);
            computePassEncoder2.end();
            const copyEncoder2 = device.createCommandEncoder();
            copyEncoder2.copyBufferToBuffer(
              this.voxelsBuffer,
              0,
              this.voxelReadBuffer,
              0,
              Float32Array.BYTES_PER_ELEMENT * voxelCount * 12
            );
            queue({
              items: [computeEncoder2.finish(), copyEncoder2.finish()],
              callback: async () => {
                await this.voxelReadBuffer.mapAsync(GPUMapMode.READ);
                const arrayBuffer2 = this.voxelReadBuffer.getMappedRange();
                const computedVoxelsData = new Float32Array(arrayBuffer2);
                const result = computeVoxels(position, stride, voxelCount, computedVoxelsData);
                this.voxelReadBuffer.unmap();
                resolve({ ...result, corners, consistency: -1 });
              }
            });
          }
        });
      });
    }
  }
  const ctx = self;
  (async function() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter)
      throw new Error("Unable to acquire GPU adapter, is WebGPU enabled?");
    const device = await adapter.requestDevice();
    const voxel = await Voxel.init(device);
    console.log("Voxel engine init complete");
    postMessage({ type: "init_complete" });
    const queue = (item) => {
      device.queue.onSubmittedWorkDone().then(item.callback);
      device.queue.submit(item.items);
    };
    onmessage = async function(e) {
      const { detail, density } = e.data;
      const chunkSize = 31;
      const { x, y, z, s } = detail;
      const halfChunk = s * chunkSize * 0.5;
      const { vertices, normals, indices, consistency } = await voxel.generate(
        device,
        queue,
        fromValues(
          x * chunkSize - halfChunk,
          y * chunkSize - halfChunk,
          z * chunkSize - halfChunk
        ),
        s,
        density
      );
      ctx.postMessage(
        {
          type: "update",
          i: `${x}:${y}:${z}`,
          ix: x,
          iy: y,
          iz: z,
          x: 0,
          y: 0,
          z: 0,
          vertices: vertices.buffer,
          normals: normals.buffer,
          indices: indices.buffer,
          stride: s,
          consistency
        },
        [vertices.buffer, normals.buffer, indices.buffer]
      );
    };
  })();
})();
