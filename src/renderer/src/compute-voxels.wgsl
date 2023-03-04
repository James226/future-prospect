struct Permutations {
  Perm : array<i32, 512>,
};

@binding(0) @group(0)
var<storage, read> perm : Permutations;

struct CornerMaterials {
  cornerMaterials : array<u32>,
};

@binding(1) @group(0)
var<storage, read_write> cornerMaterials: CornerMaterials;

struct VoxelMaterials {
  voxelMaterials : array<u32>,
};

@binding(2) @group(0)
var<storage, read_write> voxelMaterials: VoxelMaterials;

struct CornerIndex {
  cornerCount : u32,
  cornerIndexes : array<u32>
};

@binding(3) @group(0)
var<storage, read_write> cornerIndex: CornerIndex;

struct GPUVOX
{
	voxMin: vec3<f32>,
	corners: f32,
	vertPoint: vec3<f32>,
	avgNormal: vec3<f32>,
	numPoints: f32
};
struct GPUVOXS {
  voxels : array<GPUVOX>,
};

@binding(4) @group(0)
var<storage, read_write> voxels: GPUVOXS;

struct UniformBufferObject {
  chunkPosition : vec3<f32>,
  stride : f32,
	width: u32
};

@binding(5) @group(0)
var<uniform> uniforms : UniformBufferObject;

const CHILD_MIN_OFFSETS: array<vec3<u32>, 8> = array<vec3<u32>, 8>
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

const edgevmap: array<vec2<i32>, 12> = array<vec2<i32>, 12>
(
	vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7),
	vec2<i32>(0,2), vec2<i32>(1,3), vec2<i32>(4,6), vec2<i32>(5,7),
	vec2<i32>(0,1), vec2<i32>(2,3), vec2<i32>(4,5), vec2<i32>(6,7)
);

fn random(i: vec2<f32>) -> f32 {
  return fract(sin(dot(i,vec2(12.9898,78.233)))*43758.5453123);
}

fn Vec3Dot(a: vec3<f32>, b: vec3<f32>) -> f32
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

const Grad3: array<vec3<f32>, 12> = array<vec3<f32>, 12>(
	vec3<f32>(1.0,1.0,0.0), vec3<f32>(-1.0,1.0,0.0), vec3<f32>(1.0,-1.0,0.0), vec3<f32>(-1.0,-1.0,0.0),
	vec3<f32>(1.0,0.0,1.0), vec3<f32>(-1.0,0.0,1.0), vec3<f32>(1.0,0.0,-1.0), vec3<f32>(-1.0,0.0,-1.0),
	vec3<f32>(0.0,1.0,1.0), vec3<f32>(0.0,-1.0,1.0), vec3<f32>(0.0,1.0,-1.0), vec3<f32>(0.0,-1.0,-1.0)
);

fn Perlin(x1: f32, y1: f32, z1: f32) -> f32
{
	var X: i32 = 0;
	if (x1 > 0.0) {
		X = i32(x1);
	} else {
		X = i32(x1) - 1;
	}

	var Y: i32 = 0;
	if (y1 > 0.0) {
		Y = i32(y1);
	} else {
		Y = i32(y1) - 1;
	}

	var Z: i32 = 0;
	if (z1 > 0.0) {
		Z = i32(z1);
	} else {
		Z = i32(z1) - 1;
	}

	let x: f32 = x1 - f32(X);
	let y: f32 = y1 - f32(Y);
	let z: f32 = z1 - f32(Z);

	X = X & 255;
	Y = Y & 255;
	Z = Z & 255;

	let gi000: i32 = (perm.Perm[X + perm.Perm[Y + perm.Perm[Z] ] ] % 12);
	let gi001: i32 = (perm.Perm[X + perm.Perm[Y + perm.Perm[Z + 1] ] ] % 12);
	let gi010: i32 = (perm.Perm[X + perm.Perm[Y + 1 + perm.Perm[Z] ] ] % 12);
	let gi011: i32 = (perm.Perm[X + perm.Perm[Y + 1 + perm.Perm[Z + 1] ] ] % 12);
	let gi100: i32 = (perm.Perm[X + 1 + perm.Perm[Y + perm.Perm[Z] ] ] % 12);
	let gi101: i32 = (perm.Perm[X + 1 + perm.Perm[Y + perm.Perm[Z + 1] ] ] % 12);
	let gi110: i32 = (perm.Perm[X + 1 + perm.Perm[Y + 1 + perm.Perm[Z] ] ] % 12);
	let gi111: i32 = (perm.Perm[X + 1 + perm.Perm[Y + 1 + perm.Perm[Z + 1] ] ] % 12);

	let n000: f32 = dot(Grad3[gi000], vec3<f32>(x, y, z));
	let n100: f32 = dot(Grad3[gi100], vec3<f32>(x - 1.0, y, z));
	let n010: f32 = dot(Grad3[gi010], vec3<f32>(x, y - 1.0, z));
	let n110: f32 = dot(Grad3[gi110], vec3<f32>(x - 1.0, y - 1.0, z));
	let n001: f32 = dot(Grad3[gi001], vec3<f32>(x, y, z - 1.0));
	let n101: f32 = dot(Grad3[gi101], vec3<f32>(x - 1.0, y, z - 1.0));
	let n011: f32 = dot(Grad3[gi011], vec3<f32>(x, y - 1.0, z - 1.0));
	let n111: f32 = dot(Grad3[gi111], vec3<f32>(x - 1.0, y - 1.0, z - 1.0));

	let u: f32 = f32(x * x * x * (x * (x * 6.0 - 15.0) + 10.0));
	let v: f32 = f32(y * y * y * (y * (y * 6.0 - 15.0) + 10.0));
	let w: f32 = f32(z * z * z * (z * (z * 6.0 - 15.0) + 10.0));
	let nx00: f32 = mix(n000, n100, u);
	let nx01: f32 = mix(n001, n101, u);
	let nx10: f32 = mix(n010, n110, u);
	let nx11: f32 = mix(n011, n111, u);
	let nxy0: f32 = mix(nx00, nx10, v);
	let nxy1: f32 = mix(nx01, nx11, v);
	let nxyz: f32 = mix(nxy0, nxy1, w);

	return nxyz;
}

fn FractalNoise(octaves: i32, frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32
{
	let SCALE: f32 = 1.0 / 128.0;
	var p: vec3<f32> = position * SCALE;
	var nois: f32 = 0.0;

	var amplitude: f32 = 1.0;
	p = p * frequency;

	var i: i32 = 0;
	loop {
		if (i >= octaves) { break; }

		nois = nois + Perlin(p.x, p.y, p.z) * amplitude;
		p = p * lacunarity;
		amplitude = amplitude * persistence;

		continuing {
			i = i + 1;
		}
	}

	return nois;
}

fn FractalNoise1(frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32
{
	let SCALE: f32 = 1.0 / 128.0;
	var p: vec3<f32> = position * SCALE;
	var nois: f32 = 0.0;

	var amplitude: f32 = 1.0;
	p = p * frequency;

	nois = nois + Perlin(p.x, p.y, p.z) * amplitude;
	p = p * lacunarity;
	amplitude = amplitude * persistence;

	return nois;
}

fn CalculateNoiseValue(pos: vec3<f32>, scale: f32) -> f32
{
	return FractalNoise(4, 0.5343, 2.2324, 0.68324, pos * scale);
}

fn CLerp(a: f32, b: f32, t: f32) -> f32
{
	return (1.0 - t) * a + t * b;
}

// SVD

const SVD_NUM_SWEEPS: i32 = 4;
const PSUEDO_INVERSE_THRESHOLD: f32 = 0.00000001;

fn svd_mul_matrix_vec(m: mat3x3<f32>, b: vec4<f32>) -> vec4<f32>
{
	var a: mat3x3<f32> = m;

	return vec4<f32>(
		dot(vec4<f32>(a[0][0], a[0][1], a[0][2], 0.0), b),
		dot(vec4<f32>(a[1][0], a[1][1], a[1][2], 0.0), b),
		dot(vec4<f32>(a[2][0], a[2][1], a[2][2], 0.0), b),
		0.0
	);
}

fn givens_coeffs_sym(a_pp: f32, a_pq: f32, a_qq: f32) -> vec2<f32>
{
	if (a_pq == 0.0) {
		return vec2<f32>(1.0, 0.0);
	}

	let tau: f32 = (a_qq - a_pp) / (2.0 * a_pq);
	let stt: f32 = sqrt(1.0 + tau * tau);
	var tan: f32;
	if ((tau >= 0.0)) {
		tan = (tau + stt);
	} else {
		tan = (tau - stt);
	}
	tan = 1.0 / tan;

	let c: f32 = inverseSqrt(1.0 + tan * tan);
	let s: f32 = tan * c;

	return vec2<f32>(c, s);
}

fn svd_rotate_xy(x: f32, y: f32, c: f32, s: f32) -> vec2<f32>
{
	return vec2<f32>(c * x - s * y, s * x + c * y);
}

fn svd_rotateq_xy(x: f32, y: f32, z: f32, c: f32, s: f32) -> vec2<f32>
{
	let cc: f32 = c * c;
	let ss: f32 = s * s;
	let mx: f32 = 2.0 * c * s * z;

	return vec2<f32>(
		cc * x - mx + ss * y,
		ss * x + mx + cc * z
	);
}

var<private> vtav: mat3x3<f32>;
var<private> v: mat3x3<f32>;
var<private> ATA: array<f32, 6>;
var<private> Atb: vec4<f32>;
var<private> pointaccum: vec4<f32>;
var<private> btb: f32;

fn svd_rotate(a: i32, b: i32)
{
	if (vtav[a][b] == 0.0) { return; }



	let coeffs: vec2<f32> = givens_coeffs_sym(vtav[a][a], vtav[a][b], vtav[b][b]);
	let c: f32 = coeffs.x;
	let s: f32 = coeffs.y;

	let rot1: vec2<f32> = svd_rotateq_xy(vtav[a][a], vtav[b][b], vtav[a][b], c, s);
	vtav[a][a] = rot1.x;
	vtav[b][b] = rot1.y;

	let rot2: vec2<f32> = svd_rotate_xy(vtav[0][3-b], vtav[1-a][2], c, s);
	vtav[0][3-b] = rot2.x;
	vtav[1-a][2] = rot2.y;

	vtav[a][b] = 0.0;

	let rot3: vec2<f32> = svd_rotate_xy(v[0][a], v[0][b], c, s);
	v[0][a] = rot3.x; v[0][b] = rot3.y;

	let rot4: vec2<f32> = svd_rotate_xy(v[1][a], v[1][b], c, s);
	v[1][a] = rot4.x; v[1][b] = rot4.y;

	let rot5: vec2<f32> = svd_rotate_xy(v[2][a], v[2][b], c, s);
	v[2][a] = rot5.x; v[2][b] = rot5.y;
}

fn svd_solve_sym(b: array<f32, 6>) -> vec4<f32>
{
	var a: array<f32, 6> = b;

	vtav = mat3x3<f32>(
		vec3<f32>(a[0], a[1], a[2]),
		vec3<f32>(0.0, a[3], a[4]),
		vec3<f32>(0.0, 0.0, a[5])
	);

	var i: i32;
	loop {
		if (i >= SVD_NUM_SWEEPS) { break; }

		svd_rotate(0, 1);
		svd_rotate(0, 2);
		svd_rotate(1, 2);

		continuing {
			i = i + 1;
		}
	}

	var copy: mat3x3<f32> = vtav;
	return vec4<f32>(copy[0][0], copy[1][1], copy[2][2], 0.0);
}


fn svd_invdet(x: f32, tol: f32) -> f32
{
	if (abs(x) < tol || abs(1.0 / x) < tol) {
		return 0.0;
	}
	return (1.0 / x);
}

fn svd_pseudoinverse(sigma: vec4<f32>, c: mat3x3<f32>) -> mat3x3<f32>
{
	let d0: f32 = svd_invdet(sigma.x, PSUEDO_INVERSE_THRESHOLD);
	let d1: f32 = svd_invdet(sigma.y, PSUEDO_INVERSE_THRESHOLD);
	let d2: f32 = svd_invdet(sigma.z, PSUEDO_INVERSE_THRESHOLD);

	var copy: mat3x3<f32> = c;

	return mat3x3<f32> (
		vec3<f32>(
			copy[0][0] * d0 * copy[0][0] + copy[0][1] * d1 * copy[0][1] + copy[0][2] * d2 * copy[0][2],
			copy[0][0] * d0 * copy[1][0] + copy[0][1] * d1 * copy[1][1] + copy[0][2] * d2 * copy[1][2],
			copy[0][0] * d0 * copy[2][0] + copy[0][1] * d1 * copy[2][1] + copy[0][2] * d2 * copy[2][2]
		),
		vec3<f32>(
			copy[1][0] * d0 * copy[0][0] + copy[1][1] * d1 * copy[0][1] + copy[1][2] * d2 * copy[0][2],
			copy[1][0] * d0 * copy[1][0] + copy[1][1] * d1 * copy[1][1] + copy[1][2] * d2 * copy[1][2],
			copy[1][0] * d0 * copy[2][0] + copy[1][1] * d1 * copy[2][1] + copy[1][2] * d2 * copy[2][2]
		),
		vec3<f32>(
			copy[2][0] * d0 * copy[0][0] + copy[2][1] * d1 * copy[0][1] + copy[2][2] * d2 * copy[0][2],
			copy[2][0] * d0 * copy[1][0] + copy[2][1] * d1 * copy[1][1] + copy[2][2] * d2 * copy[1][2],
			copy[2][0] * d0 * copy[2][0] + copy[2][1] * d1 * copy[2][1] + copy[2][2] * d2 * copy[2][2]
		),
	);
}

fn svd_solve_ATA_Atb(a: vec4<f32>) -> vec4<f32>
{
	v = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));

	let sigma: vec4<f32> = svd_solve_sym(ATA);

	let Vinv: mat3x3<f32> = svd_pseudoinverse(sigma, v);
	return svd_mul_matrix_vec(Vinv, a);
}

fn svd_vmul_sym(v: vec4<f32>) -> vec4<f32>
{
	let A_row_x: vec4<f32> = vec4<f32>(ATA[0], ATA[1], ATA[2], 0.0);
	return vec4<f32> (
		dot(A_row_x, v),
		ATA[1] * v.x + ATA[3] * v.y + ATA[4] * v.z,
		ATA[2] * v.x + ATA[4] * v.y + ATA[5] * v.z,
		0.0
	);
}


// // QEF

fn qef_add(n: vec4<f32>, p: vec4<f32>)
{
	ATA[0] = ATA[0] + n.x * n.x;
	ATA[1] = ATA[1] + n.x * n.y;
	ATA[2] = ATA[2] + n.x * n.z;
	ATA[3] = ATA[3] + n.y * n.y;
	ATA[4] = ATA[4] + n.y * n.z;
	ATA[5] = ATA[5] + n.z * n.z;

	let b: f32 = dot(p, n);
	Atb.x = Atb.x +n.x * b;
	Atb.y = Atb.y +n.y * b;
	Atb.z = Atb.z +n.z * b;
	btb = btb + b * b;

	pointaccum.x = pointaccum.x +p.x;
	pointaccum.y = pointaccum.y +p.y;
	pointaccum.z = pointaccum.z +p.z;
	pointaccum.w = pointaccum.w +1.0;
}

fn qef_calc_error(x: vec4<f32>) -> f32
{
	var tmp: vec4<f32> = svd_vmul_sym(x);
	tmp = Atb - tmp;

	return dot(tmp, tmp);
}

fn qef_solve() -> vec4<f32>
{
	let masspoint: vec4<f32> = vec4<f32>(pointaccum.x / pointaccum.w, pointaccum.y / pointaccum.w, pointaccum.z / pointaccum.w, pointaccum.w / pointaccum.w);

	var A_mp: vec4<f32> = svd_vmul_sym(masspoint);
	A_mp = Atb - A_mp;

	let x: vec4<f32> = svd_solve_ATA_Atb(A_mp);

	let error: f32 = qef_calc_error(x);
	let r: vec4<f32> = x + masspoint;

	return vec4<f32>(r.x, r.y, r.z, error);
}

#import density

fn ApproximateZeroCrossingPosition(p0: vec3<f32>, p1: vec3<f32>) -> vec3<f32>
{
	var minValue: f32 = 100000.0;
	var t: f32 = 0.0;
	var currentT: f32 = 0.0;
	let steps: f32 = 8.0;
	let increment: f32 = 1.0 / steps;
	loop {
		if (currentT > 1.0) { break; }

		let p: vec3<f32> = p0 + ((p1 - p0) * currentT);
		let density: f32 = abs(getDensity(p));
		if (density < minValue)
		{
			minValue = density;
			t = currentT;
		}

		continuing {
			currentT = currentT + increment;
		}
	}

	return p0 + ((p1 - p0) * t);
}

fn CalculateSurfaceNormal(p: vec3<f32>) -> vec3<f32>
{
	let H: f32 = uniforms.stride; // This needs to scale based on something...
	let dx: f32 = getDensity(p + vec3<f32>(H, 0.0, 0.0)) - getDensity(p - vec3<f32>(H, 0.0, 0.0));
	let dy: f32 = getDensity(p + vec3<f32>(0.0, H, 0.0)) - getDensity(p - vec3<f32>(0.0, H, 0.0));
	let dz: f32 = getDensity(p + vec3<f32>(0.0, 0.0, H)) - getDensity(p - vec3<f32>(0.0, 0.0, H));

	return normalize(vec3<f32>(dx, dy, dz));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let trueIndex: u32 = GlobalInvocationID.x;

	if (trueIndex < cornerIndex.cornerCount)
	{
		let ures: u32 = 32u;

		let nodeSize: u32 = u32(uniforms.stride);

		let voxelIndex: u32 = cornerIndex.cornerIndexes[trueIndex];
		let z: u32 = voxelIndex / (ures * ures);
		let y: u32 = (voxelIndex - (z * ures * ures)) / ures;
		let x: u32 = voxelIndex - (z * ures * ures) - (y * ures);

		let corners: u32 = voxelMaterials.voxelMaterials[voxelIndex];

		let nodePos: vec3<f32> = (vec3<f32>(f32(x), f32(y), f32 (z)) * uniforms.stride) + uniforms.chunkPosition;
		voxels.voxels[trueIndex].voxMin = nodePos;
		let MAX_CROSSINGS: i32 = 6;
		var edgeCount: i32 = 0;

		pointaccum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
		ATA = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		Atb = vec4<f32>(0.0, 0.0, 0.0, 0.0);
		var averageNormal: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
		btb = 0.0;

		var j: i32 = 0;
		loop {
			if (!(j < 12 && edgeCount <= MAX_CROSSINGS)) {
				break;
			}

			let c1: i32 = edgevmap[j].x;
			let c2: i32 = edgevmap[j].y;

			let m1: u32 = (corners >> u32(c1)) & 1u;
			let m2: u32 = (corners >> u32(c2)) & 1u;

			if (!((m1 == 0u && m2 == 0u) || (m1 == 1u && m2 == 1u)))
			{
				let p1: vec3<f32> = nodePos + vec3<f32>(f32(CHILD_MIN_OFFSETS[c1].x * nodeSize), f32(CHILD_MIN_OFFSETS[c1].y * nodeSize), f32(CHILD_MIN_OFFSETS[c1].z * nodeSize));
				let p2: vec3<f32> = nodePos + vec3<f32>(f32(CHILD_MIN_OFFSETS[c2].x * nodeSize), f32(CHILD_MIN_OFFSETS[c2].y * nodeSize), f32(CHILD_MIN_OFFSETS[c2].z * nodeSize));
				let p: vec3<f32> = ApproximateZeroCrossingPosition(p1, p2);
				let n: vec3<f32> = CalculateSurfaceNormal(p);

				qef_add(vec4<f32>(n.x, n.y, n.z, 0.0), vec4<f32>(p.x, p.y, p.z, 0.0));

				averageNormal = averageNormal + n;

				edgeCount = edgeCount + 1;
			}

			continuing {
				j = j + 1;
			}
		}


		averageNormal = normalize(averageNormal / vec3<f32>(f32(edgeCount), f32(edgeCount), f32(edgeCount)));

		let com: vec3<f32> = vec3<f32>(pointaccum.x / pointaccum.w, pointaccum.y / pointaccum.w, pointaccum.z / pointaccum.w);

		let result: vec4<f32> = qef_solve();
		var solved_position: vec3<f32> = result.xyz;
		let error: f32 = result.w;


		let Min: vec3<f32> = nodePos;
		let Max: vec3<f32> = nodePos + vec3<f32>(1.0, 1.0, 1.0);
		if (solved_position.x < Min.x || solved_position.x > Max.x ||
				solved_position.y < Min.y || solved_position.y > Max.y ||
				solved_position.z < Min.z || solved_position.z > Max.z)
		{
			solved_position = com;
		}

		voxels.voxels[trueIndex].vertPoint = solved_position;
		voxels.voxels[trueIndex].avgNormal = averageNormal;
		voxels.voxels[trueIndex].numPoints = f32(edgeCount);
		voxels.voxels[trueIndex].corners = f32(voxelMaterials.voxelMaterials[voxelIndex]);
	}
}

@compute @workgroup_size(1)
fn computeMaterials(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
		let width = uniforms.width;
    let index: u32 = GlobalInvocationID.z * width * width + GlobalInvocationID.y * width + GlobalInvocationID.x;
    let cornerPos: vec3<f32> = vec3<f32>(f32(GlobalInvocationID.x) * uniforms.stride, f32(GlobalInvocationID.y) * uniforms.stride, f32(GlobalInvocationID.z) * uniforms.stride);

    let density: f32 = getDensity(cornerPos + uniforms.chunkPosition);

		if (density < 0.0) {
			if (true || length(cornerPos + uniforms.chunkPosition) < 2000.0) {
        //cornerMaterials.cornerMaterials[index] = u32(random(vec2(f32(index))) * 255.0) + 1;
			  cornerMaterials.cornerMaterials[index] = 256u;
			} else {
        cornerMaterials.cornerMaterials[index] = u32(length(cornerPos) / uniforms.stride * 256.0);
			}
		} else {
			cornerMaterials.cornerMaterials[index] = 0u;
		}
}
