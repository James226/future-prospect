var qn=Object.defineProperty;var jn=(G,C,R)=>C in G?qn(G,C,{enumerable:!0,configurable:!0,writable:!0,value:R}):G[C]=R;var h=(G,C,R)=>(jn(G,typeof C!="symbol"?C+"":C,R),R);(function(){"use strict";var G=1e-6,C=typeof Float32Array<"u"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var e=0,t=arguments.length;t--;)e+=arguments[t]*arguments[t];return Math.sqrt(e)});function R(){var e=new C(3);return C!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e}function B(e,t,o){var u=new C(3);return u[0]=e,u[1]=t,u[2]=o,u}function fn(e,t,o){return e[0]=t[0]+o[0],e[1]=t[1]+o[1],e[2]=t[2]+o[2],e}function ln(e,t,o){return e[0]=t[0]-o[0],e[1]=t[1]-o[1],e[2]=t[2]-o[2],e}function pn(e,t){var o=e[0],u=e[1],s=e[2],p=t[0],f=t[1],r=t[2];return Math.abs(o-p)<=G*Math.max(1,Math.abs(o),Math.abs(p))&&Math.abs(u-f)<=G*Math.max(1,Math.abs(u),Math.abs(f))&&Math.abs(s-r)<=G*Math.max(1,Math.abs(s),Math.abs(r))}var vn=ln;(function(){var e=R();return function(t,o,u,s,p,f){var r,n;for(o||(o=3),u||(u=0),s?n=Math.min(s*o+u,t.length):n=t.length,r=u;r<n;r+=o)e[0]=t[r],e[1]=t[r+1],e[2]=t[r+2],p(e,e,f),t[r]=e[0],t[r+1]=e[1],t[r+2]=e[2];return t}})();var mn=`const OctreeSize = 32u;\r
\r
struct CornerMaterials {\r
  cornerMaterials : array<u32>,\r
};\r
@binding(1) @group(0) var<storage, read> cornerMaterials: CornerMaterials;\r
\r
struct VoxelMaterials {\r
  voxelMaterials : array<u32>,\r
};\r
@binding(2) @group(0) var<storage, read_write> voxelMaterials: VoxelMaterials;\r
\r
const CHILD_MIN_OFFSETS = array<vec3<u32>, 8>\r
(\r
  vec3<u32>(0u, 0u, 0u),\r
  vec3<u32>(0u, 0u, 1u),\r
  vec3<u32>(0u, 1u, 0u),\r
  vec3<u32>(0u, 1u, 1u),\r
  vec3<u32>(1u, 0u, 0u),\r
  vec3<u32>(1u, 0u, 1u),\r
  vec3<u32>(1u, 1u, 0u),\r
  vec3<u32>(1u, 1u, 1u)\r
);\r
\r
@compute @workgroup_size(1)\r
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {\r
  let index: u32 = GlobalInvocationID.z * 32u * 32u + GlobalInvocationID.y * 32u + GlobalInvocationID.x;\r
\r
  let nodePos: vec3<u32> = vec3<u32>(GlobalInvocationID.x, GlobalInvocationID.y, GlobalInvocationID.z);\r
  var corners: u32 = 0u;\r
\r
  var j: u32 = 0u;\r
  loop {\r
    if (j >= 8u) { break; }\r
\r
    let cornerPos: vec3<u32> = vec3<u32>(GlobalInvocationID.x + CHILD_MIN_OFFSETS[j].x, GlobalInvocationID.y + CHILD_MIN_OFFSETS[j].y, GlobalInvocationID.z + CHILD_MIN_OFFSETS[j].z);\r
    let material: u32 = min(1, cornerMaterials.cornerMaterials[cornerPos.z * 33u * 33u + cornerPos.y * 33u + cornerPos.x]);\r
    corners = corners | (material << j);\r
\r
    continuing {\r
      j = j + 1u;\r
    }\r
  }\r
  \r
  voxelMaterials.voxelMaterials[index] = corners;\r
}`,xn=`struct VoxelMaterials {\r
  voxelMaterials : array<u32>,\r
};\r
@binding(2) @group(0) var<storage, read> voxelMaterials: VoxelMaterials;\r
\r
struct CornerIndex {\r
  cornerCount : u32,\r
  cornerIndexes : array<u32>,\r
};\r
@binding(3) @group(0) var<storage, read_write> cornerIndex: CornerIndex;\r
\r
\r
@compute @workgroup_size(1)\r
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {\r
	var position: u32 = 0u;\r
\r
	var i : u32 = 0u;\r
	loop {\r
		if (i >= 32u * 32u * 32u) { break; }\r
		\r
		if (voxelMaterials.voxelMaterials[i] != 0u && voxelMaterials.voxelMaterials[i] != 255u) {\r
			cornerIndex.cornerIndexes[position] = i;\r
			position = position + 1u;  \r
		}\r
			\r
		continuing {  \r
			i = i + 1u;\r
		}\r
	}\r
\r
	cornerIndex.cornerCount = position;\r
}`,yn=`struct Permutations {
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
`,gn=`const freq = 0.001;

const MATERIAL_AIR = 0u;
const MATERIAL_ROCK = 1u;
const MATERIAL_WOOD = 2u;
const MATERIAL_FIRE = 3u;

struct Density {
  density: f32,
  material: u32
}

struct Augmentations {
  count: u32,
  augmentations: array<Augmentation>
}

struct Augmentation {
  position: vec3<f32>,
  size: f32,
  attributes: u32
}

@binding(0) @group(1) var<storage, read> augmentations: Augmentations;

fn subtract(base: Density, sub: f32) -> Density {
  return Density(max(base.density, sub), base.material);
}

fn add(base: Density, add: f32, material: u32) -> Density {
  if (add <= 0) {
    return Density(add, material);
  }
  return base;
}

fn Box(worldPosition: vec3<f32>, origin: vec3<f32>, halfDimensions: vec3<f32>) -> f32
{
	let local_pos: vec3<f32> = worldPosition - origin;
	let pos: vec3<f32> = local_pos;

	let d: vec3<f32> = vec3<f32>(abs(pos.x), abs(pos.y), abs(pos.z)) - halfDimensions;
	let m: f32 = max(d.x, max(d.y, d.z));
	return clamp(min(m, length(max(d, vec3<f32>(0.0, 0.0, 0.0)))), -100.0, 100.0);
}

fn Torus(worldPosition: vec3<f32>, origin: vec3<f32>, t: vec3<f32>) -> f32
{
	let p: vec3<f32> = worldPosition - origin;

  let q: vec2<f32> = vec2<f32>(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

fn Sphere(worldPosition: vec3<f32>, origin: vec3<f32>, radius: f32) -> f32
{
	return clamp(length(worldPosition - origin) - radius, -100.0, 100.0);
}

fn FractalNoise21(octaves: i32, frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32
{
	let SCALE: f32 = 1.0 / 128.0;
	var p: vec3<f32> = position * SCALE;
	var nois: f32 = 0.0;

	var amplitude: f32 = 1.0;
	p = p * frequency;

	var i: i32 = 0;
	loop {
		if (i >= octaves) { break; }

		nois = nois + perlinNoise3(p) * amplitude;
		p = p * lacunarity;
		amplitude = amplitude * persistence;

		continuing {
			i = i + 1;
		}
	}

	return nois;
}

fn FractalNoise2(frequency: f32, lacunarity: f32, persistence: f32, position: vec3<f32>) -> f32
{
	let SCALE: f32 = 1.0 / 128.0;
	var p: vec3<f32> = position * SCALE;
	var nois: f32 = 0.0;

	var amplitude: f32 = 1.0;
	p = p * frequency;

	nois = nois + perlinNoise3(p) * amplitude;
	p = p * lacunarity;
	amplitude = amplitude * persistence;

	return nois;
}

fn permute41(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }
fn fade3(t: vec3<f32>) -> vec3<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise3(P: vec3<f32>) -> f32 {
  var Pi0 : vec3<f32> = floor(P); // Integer part for indexing
  var Pi1 : vec3<f32> = Pi0 + vec3<f32>(1.); // Integer part + 1
  Pi0 = Pi0 % vec3<f32>(289.);
  Pi1 = Pi1 % vec3<f32>(289.);
  let Pf0 = fract(P); // Fractional part for interpolation
  let Pf1 = Pf0 - vec3<f32>(1.); // Fractional part - 1.
  let ix = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy = vec4<f32>(Pi0.yy, Pi1.yy);
  let iz0 = Pi0.zzzz;
  let iz1 = Pi1.zzzz;

  let ixy = permute41(permute41(ix) + iy);
  let ixy0 = permute41(ixy + iz0);
  let ixy1 = permute41(ixy + iz1);

  var gx0: vec4<f32> = ixy0 / 7.;
  var gy0: vec4<f32> = fract(floor(gx0) / 7.) - 0.5;
  gx0 = fract(gx0);
  var gz0: vec4<f32> = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
  var sz0: vec4<f32> = step(gz0, vec4<f32>(0.));
  gx0 = gx0 + sz0 * (step(vec4<f32>(0.), gx0) - 0.5);
  gy0 = gy0 + sz0 * (step(vec4<f32>(0.), gy0) - 0.5);

  var gx1: vec4<f32> = ixy1 / 7.;
  var gy1: vec4<f32> = fract(floor(gx1) / 7.) - 0.5;
  gx1 = fract(gx1);
  var gz1: vec4<f32> = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
  var sz1: vec4<f32> = step(gz1, vec4<f32>(0.));
  gx1 = gx1 - sz1 * (step(vec4<f32>(0.), gx1) - 0.5);
  gy1 = gy1 - sz1 * (step(vec4<f32>(0.), gy1) - 0.5);

  var g000: vec3<f32> = vec3<f32>(gx0.x, gy0.x, gz0.x);
  var g100: vec3<f32> = vec3<f32>(gx0.y, gy0.y, gz0.y);
  var g010: vec3<f32> = vec3<f32>(gx0.z, gy0.z, gz0.z);
  var g110: vec3<f32> = vec3<f32>(gx0.w, gy0.w, gz0.w);
  var g001: vec3<f32> = vec3<f32>(gx1.x, gy1.x, gz1.x);
  var g101: vec3<f32> = vec3<f32>(gx1.y, gy1.y, gz1.y);
  var g011: vec3<f32> = vec3<f32>(gx1.z, gy1.z, gz1.z);
  var g111: vec3<f32> = vec3<f32>(gx1.w, gy1.w, gz1.w);

  let norm0 = taylorInvSqrt4(
      vec4<f32>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 = g000 * norm0.x;
  g010 = g010 * norm0.y;
  g100 = g100 * norm0.z;
  g110 = g110 * norm0.w;
  let norm1 = taylorInvSqrt4(
      vec4<f32>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 = g001 * norm1.x;
  g011 = g011 * norm1.y;
  g101 = g101 * norm1.z;
  g111 = g111 * norm1.w;

  let n000 = dot(g000, Pf0);
  let n100 = dot(g100, vec3<f32>(Pf1.x, Pf0.yz));
  let n010 = dot(g010, vec3<f32>(Pf0.x, Pf1.y, Pf0.z));
  let n110 = dot(g110, vec3<f32>(Pf1.xy, Pf0.z));
  let n001 = dot(g001, vec3<f32>(Pf0.xy, Pf1.z));
  let n101 = dot(g101, vec3<f32>(Pf1.x, Pf0.y, Pf1.z));
  let n011 = dot(g011, vec3<f32>(Pf0.x, Pf1.yz));
  let n111 = dot(g111, Pf1);

  var fade_xyz: vec3<f32> = fade3(Pf0);
  let temp = vec4<f32>(f32(fade_xyz.z)); // simplify after chrome bug fix
  let n_z = mix(vec4<f32>(n000, n100, n010, n110), vec4<f32>(n001, n101, n011, n111), temp);
  let n_yz = mix(n_z.xy, n_z.zw, vec2<f32>(f32(fade_xyz.y))); // simplify after chrome bug fix
  let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

fn CalculateNoiseValue2(pos: vec3<f32>, scale: f32) -> f32
{
	return FractalNoise21(4, 0.5343, 2.2324, 0.68324, pos * scale);
}

fn CLerp2(a: f32, b: f32, t: f32) -> f32
{
	return (1.0 - t) * a + t * b;
}

fn rotateAlign(v1: vec3<f32>, v2: vec3<f32>) -> mat3x3<f32>
{
    let axis = cross( v1, v2 );

    let cosA = dot( v1, v2 );
    let k = 1.0 / (1.0 + cosA);

    return mat3x3<f32>( (axis.x * axis.x * k) + cosA,
                 (axis.y * axis.x * k) - axis.z,
                 (axis.z * axis.x * k) + axis.y,
                 (axis.x * axis.y * k) + axis.z,
                 (axis.y * axis.y * k) + cosA,
                 (axis.z * axis.y * k) - axis.x,
                 (axis.x * axis.z * k) - axis.y,
                 (axis.y * axis.z * k) + axis.x,
                 (axis.z * axis.z * k) + cosA
                 );
}


fn AngleAxis3x3(angle: f32, axis: vec3<f32>) -> mat3x3<f32>
{
    let s = sin(angle);
    let c = cos(angle);

    let t = 1 - c;
    let x = axis.x;
    let y = axis.y;
    let z = axis.z;

    return mat3x3<f32>(
        t * x * x + c,      t * x * y - s * z,  t * x * z + s * y,
        t * x * y + s * z,  t * y * y + c,      t * y * z - s * x,
        t * x * z - s * y,  t * y * z + s * x,  t * z * z + c
    );
}

fn blockSize(blockType: u32) -> f32 {
  if (blockType == 2 || blockType == 3) {
    return 0.5;
  }
  return 1.0;
}

fn calculateDensity(worldPosition: vec3<f32>) -> Density {
	var worldRadius: f32 = 5000.0;
	var world: vec3<f32> = worldPosition - vec3<f32>(2000000.0, 100.0, 100.0);
	var worldDist: f32 = -worldRadius + length(world);
	let up = vec3<f32>(0.0, 1.0, 0.0);


	let flatlandNoiseScale: f32 = 1.0;
	let flatlandLerpAmount: f32 = 0.07;
	let flatlandYPercent: f32 = 1.2;

	let rockyNoiseScale: f32 = 1.5;
	let rockyLerpAmount: f32 = 0.05;
	let rockyYPercent: f32 = 0.7;

	let maxMountainMixLerpAmount: f32 = 0.075;
	let minMountainMixLerpAmount: f32 = 1.0;

	let rockyBlend: f32 = 0.0;

	let mountainBlend: f32 = clamp(abs(FractalNoise2(0.5343, 2.2324, 0.68324, world * 0.11)) * 4.0, 0.0, 1.0);
	//let mountainBlend: f32 = 1.0;

	//let mountain: f32 = CalculateNoiseValue2(world, 0.07);
	let mountain: f32 = 0.0;

//	var blob: f32 = CalculateNoiseValue2(world, flatlandNoiseScale + ((rockyNoiseScale - flatlandNoiseScale) * rockyBlend));
//	blob = CLerp2(blob, (worldDist) * (flatlandYPercent + ((rockyYPercent - flatlandYPercent) * rockyBlend)),
//				flatlandLerpAmount + ((rockyLerpAmount - flatlandLerpAmount) * rockyBlend))
//				+ CLerp2(mountain, blob, minMountainMixLerpAmount + ((maxMountainMixLerpAmount - minMountainMixLerpAmount) * mountainBlend));

  var result = Density(1.0, MATERIAL_AIR);

	//result = add(result, blob, MATERIAL_WOOD);

  result = add(result, Box(worldPosition, vec3<f32>(2000000.0, 150.0, 5000.0), vec3<f32>(5000.0, 1000.0, 5000.0)), MATERIAL_WOOD);
  result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0, 100.0, 100.0), 5000.0), MATERIAL_ROCK);

  //result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0 - 1000000.0, 0.0, 0.0), 1000000.0), MATERIAL_ROCK);

  result = add(result, Sphere(worldPosition, vec3<f32>(0.0, 0.0, 0.0), 200000.0), MATERIAL_FIRE);

  //result = subtract(result, -Sphere(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), 1000.0));
  //result = subtract(result, -Box(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), vec3<f32>(6000.0, 500.0, 500.0)));
  //result = subtract(result, -Box(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), vec3<f32>(500.0, 500.0, 5000.0)));

  //result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0, 0.0, 0.0), 1000.0), MATERIAL_FIRE);


  let count = augmentations.count;

  var i: u32 = 0u;
  loop {
    if (i >= count) { break; }

    let augmentation = augmentations.augmentations[i];

    let minBounds = augmentation.position - augmentation.size * 2;
    let maxBounds = augmentation.position + augmentation.size * 2;
    if (minBounds.x > worldPosition.x || minBounds.y > worldPosition.y || minBounds.z > worldPosition.z
      || maxBounds.x < worldPosition.x || maxBounds.y < worldPosition.y || maxBounds.z < worldPosition.z) { continue; }

    let shape = (augmentation.attributes & 0xFE) >> 1;
    var density: f32 = 0.0;

    let down = normalize(augmentation.position - vec3<f32>(2000000.0, 100.0, 100.0));
    let rotation = rotateAlign(down, up);
    let position = ((worldPosition - augmentation.position) * rotation - vec3<f32>(0.0, augmentation.size * blockSize(shape), 0.0)) + augmentation.position;

    switch(shape) {
      case 0: {
        density = Sphere(position, vec3<f32>(augmentation.position.x, augmentation.position.y, augmentation.position.z), augmentation.size);
      }
      case 1: {
        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y, augmentation.position.z), vec3<f32>(augmentation.size));
      }
      case 2: {
        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y + augmentation.size / 2, augmentation.position.z), vec3<f32>(augmentation.size, 5.0, augmentation.size));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
      }
      case 3: {
        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y + augmentation.size / 2, augmentation.position.z), vec3<f32>(augmentation.size, 5.0, augmentation.size));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size, augmentation.position.y + augmentation.size, augmentation.position.z), vec3<f32>(5.0, augmentation.size / 2, augmentation.size)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z - augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x - augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
        density = min(density, Box(position, vec3<f32>(augmentation.position.x + augmentation.size / 2, augmentation.position.y, augmentation.position.z + augmentation.size / 2), vec3<f32>(5.0, augmentation.size / 2, 5.0)));
      }
      case 4: {
        density = Box(position, vec3<f32>(augmentation.position.x, augmentation.position.y - augmentation.size + 16, augmentation.position.z), vec3<f32>(augmentation.size, 16, augmentation.size));
      }
      default: {
        continue;
      }
    }

    if ((augmentation.attributes & 0x1) == 0x1) {
      let material = (augmentation.attributes & 0x1FF00) >> 8;
      result = add(result, density, material);
    } else {
      result = subtract(result, -density);
    }


    continuing {
      i = i + 1u;
    }
  }

  return result;
}

fn getDensity(worldPosition: vec3<f32>) -> f32 {
	return calculateDensity(worldPosition).density;
}
`;class dn{constructor(t){h(this,"bindGroup");this.bindGroup=t}apply(t){t.setBindGroup(1,this.bindGroup)}}class q{constructor(t){h(this,"augmentationBuffer");h(this,"augmentationArray",[]);h(this,"augmentations");h(this,"onModified",()=>{});this.augmentationBuffer=t,this.augmentations=new ArrayBuffer(Uint32Array.BYTES_PER_ELEMENT*4)}static async init(t){const o=64*Float32Array.BYTES_PER_ELEMENT*8+8,u=t.createBuffer({size:o,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!1});return new q(u)}async apply(t,o){const u=t.createBindGroup({layout:o.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:this.augmentationBuffer}}]});return new dn(u)}modify(t,o){this.augmentationArray.push(o),this.update(t,this.augmentationArray),this.onModified()}update(t,o){this.augmentations=new ArrayBuffer(Uint32Array.BYTES_PER_ELEMENT*4+Uint32Array.BYTES_PER_ELEMENT*o.length*8);const u=new Uint32Array(this.augmentations,0,4);u[0]=o.length;const s=new Float32Array(this.augmentations,Uint32Array.BYTES_PER_ELEMENT*4),p=new Uint32Array(this.augmentations,Uint32Array.BYTES_PER_ELEMENT*4);for(let f=0;f<o.length;f++)s[f*8]=o[f].x,s[f*8+1]=o[f].y,s[f*8+2]=o[f].z,s[f*8+3]=o[f].size,p[f*8+4]=o[f].type|o[f].shape<<1|o[f].material<<8;t.queue.writeBuffer(this.augmentationBuffer,0,this.augmentations,0,this.augmentations.byteLength)}updateRaw(t,o){this.augmentations=o,t.queue.writeBuffer(this.augmentationBuffer,0,this.augmentations,0,this.augmentations.byteLength)}static patch(t){return t.replace("#import density",gn)}}var N=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function hn(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function zn(e){if(e.__esModule)return e;var t=e.default;if(typeof t=="function"){var o=function u(){if(this instanceof u){var s=[null];s.push.apply(s,arguments);var p=Function.bind.apply(t,s);return new p}return t.apply(this,arguments)};o.prototype=t.prototype}else o={};return Object.defineProperty(o,"__esModule",{value:!0}),Object.keys(e).forEach(function(u){var s=Object.getOwnPropertyDescriptor(e,u);Object.defineProperty(o,u,s.get?s:{enumerable:!0,get:function(){return e[u]}})}),o}var V={exports:{}};V.exports,function(e){(function(t,o,u){function s(n){var i=this,l=r();i.next=function(){var a=2091639*i.s0+i.c*23283064365386963e-26;return i.s0=i.s1,i.s1=i.s2,i.s2=a-(i.c=a|0)},i.c=1,i.s0=l(" "),i.s1=l(" "),i.s2=l(" "),i.s0-=l(n),i.s0<0&&(i.s0+=1),i.s1-=l(n),i.s1<0&&(i.s1+=1),i.s2-=l(n),i.s2<0&&(i.s2+=1),l=null}function p(n,i){return i.c=n.c,i.s0=n.s0,i.s1=n.s1,i.s2=n.s2,i}function f(n,i){var l=new s(n),a=i&&i.state,c=l.next;return c.int32=function(){return l.next()*4294967296|0},c.double=function(){return c()+(c()*2097152|0)*11102230246251565e-32},c.quick=c,a&&(typeof a=="object"&&p(a,l),c.state=function(){return p(l,{})}),c}function r(){var n=4022871197,i=function(l){l=String(l);for(var a=0;a<l.length;a++){n+=l.charCodeAt(a);var c=.02519603282416938*n;n=c>>>0,c-=n,c*=n,n=c>>>0,c-=n,n+=c*4294967296}return(n>>>0)*23283064365386963e-26};return i}o&&o.exports?o.exports=f:u&&u.amd?u(function(){return f}):this.alea=f})(N,e,!1)}(V);var Pn=V.exports,X={exports:{}};X.exports,function(e){(function(t,o,u){function s(r){var n=this,i="";n.x=0,n.y=0,n.z=0,n.w=0,n.next=function(){var a=n.x^n.x<<11;return n.x=n.y,n.y=n.z,n.z=n.w,n.w^=n.w>>>19^a^a>>>8},r===(r|0)?n.x=r:i+=r;for(var l=0;l<i.length+64;l++)n.x^=i.charCodeAt(l)|0,n.next()}function p(r,n){return n.x=r.x,n.y=r.y,n.z=r.z,n.w=r.w,n}function f(r,n){var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var c=i.next()>>>11,v=(i.next()>>>0)/4294967296,m=(c+v)/(1<<21);while(m===0);return m},a.int32=i.next,a.quick=a,l&&(typeof l=="object"&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:u&&u.amd?u(function(){return f}):this.xor128=f})(N,e,!1)}(X);var bn=X.exports,H={exports:{}};H.exports,function(e){(function(t,o,u){function s(r){var n=this,i="";n.next=function(){var a=n.x^n.x>>>2;return n.x=n.y,n.y=n.z,n.z=n.w,n.w=n.v,(n.d=n.d+362437|0)+(n.v=n.v^n.v<<4^(a^a<<1))|0},n.x=0,n.y=0,n.z=0,n.w=0,n.v=0,r===(r|0)?n.x=r:i+=r;for(var l=0;l<i.length+64;l++)n.x^=i.charCodeAt(l)|0,l==i.length&&(n.d=n.x<<10^n.x>>>4),n.next()}function p(r,n){return n.x=r.x,n.y=r.y,n.z=r.z,n.w=r.w,n.v=r.v,n.d=r.d,n}function f(r,n){var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var c=i.next()>>>11,v=(i.next()>>>0)/4294967296,m=(c+v)/(1<<21);while(m===0);return m},a.int32=i.next,a.quick=a,l&&(typeof l=="object"&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:u&&u.amd?u(function(){return f}):this.xorwow=f})(N,e,!1)}(H);var _n=H.exports,$={exports:{}};$.exports,function(e){(function(t,o,u){function s(r){var n=this;n.next=function(){var l=n.x,a=n.i,c,v;return c=l[a],c^=c>>>7,v=c^c<<24,c=l[a+1&7],v^=c^c>>>10,c=l[a+3&7],v^=c^c>>>3,c=l[a+4&7],v^=c^c<<7,c=l[a+7&7],c=c^c<<13,v^=c^c<<9,l[a]=v,n.i=a+1&7,v};function i(l,a){var c,v=[];if(a===(a|0))v[0]=a;else for(a=""+a,c=0;c<a.length;++c)v[c&7]=v[c&7]<<15^a.charCodeAt(c)+v[c+1&7]<<13;for(;v.length<8;)v.push(0);for(c=0;c<8&&v[c]===0;++c);for(c==8?v[7]=-1:v[c],l.x=v,l.i=0,c=256;c>0;--c)l.next()}i(n,r)}function p(r,n){return n.x=r.x.slice(),n.i=r.i,n}function f(r,n){r==null&&(r=+new Date);var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var c=i.next()>>>11,v=(i.next()>>>0)/4294967296,m=(c+v)/(1<<21);while(m===0);return m},a.int32=i.next,a.quick=a,l&&(l.x&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:u&&u.amd?u(function(){return f}):this.xorshift7=f})(N,e,!1)}($);var An=$.exports,Z={exports:{}};Z.exports,function(e){(function(t,o,u){function s(r){var n=this;n.next=function(){var l=n.w,a=n.X,c=n.i,v,m;return n.w=l=l+1640531527|0,m=a[c+34&127],v=a[c=c+1&127],m^=m<<13,v^=v<<17,m^=m>>>15,v^=v>>>12,m=a[c]=m^v,n.i=c,m+(l^l>>>16)|0};function i(l,a){var c,v,m,g,_,d=[],I=128;for(a===(a|0)?(v=a,a=null):(a=a+"\0",v=0,I=Math.max(I,a.length)),m=0,g=-32;g<I;++g)a&&(v^=a.charCodeAt((g+32)%a.length)),g===0&&(_=v),v^=v<<10,v^=v>>>15,v^=v<<4,v^=v>>>13,g>=0&&(_=_+1640531527|0,c=d[g&127]^=v+_,m=c==0?m+1:0);for(m>=128&&(d[(a&&a.length||0)&127]=-1),m=127,g=4*128;g>0;--g)v=d[m+34&127],c=d[m=m+1&127],v^=v<<13,c^=c<<17,v^=v>>>15,c^=c>>>12,d[m]=v^c;l.w=_,l.X=d,l.i=m}i(n,r)}function p(r,n){return n.i=r.i,n.w=r.w,n.X=r.X.slice(),n}function f(r,n){r==null&&(r=+new Date);var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var c=i.next()>>>11,v=(i.next()>>>0)/4294967296,m=(c+v)/(1<<21);while(m===0);return m},a.int32=i.next,a.quick=a,l&&(l.X&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:u&&u.amd?u(function(){return f}):this.xor4096=f})(N,e,!1)}(Z);var wn=Z.exports,W={exports:{}};W.exports,function(e){(function(t,o,u){function s(r){var n=this,i="";n.next=function(){var a=n.b,c=n.c,v=n.d,m=n.a;return a=a<<25^a>>>7^c,c=c-v|0,v=v<<24^v>>>8^m,m=m-a|0,n.b=a=a<<20^a>>>12^c,n.c=c=c-v|0,n.d=v<<16^c>>>16^m,n.a=m-a|0},n.a=0,n.b=0,n.c=-1640531527,n.d=1367130551,r===Math.floor(r)?(n.a=r/4294967296|0,n.b=r|0):i+=r;for(var l=0;l<i.length+20;l++)n.b^=i.charCodeAt(l)|0,n.next()}function p(r,n){return n.a=r.a,n.b=r.b,n.c=r.c,n.d=r.d,n}function f(r,n){var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var c=i.next()>>>11,v=(i.next()>>>0)/4294967296,m=(c+v)/(1<<21);while(m===0);return m},a.int32=i.next,a.quick=a,l&&(typeof l=="object"&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:u&&u.amd?u(function(){return f}):this.tychei=f})(N,e,!1)}(W);var En=W.exports,en={exports:{}},Bn={},Mn=Object.freeze({__proto__:null,default:Bn}),Sn=zn(Mn);(function(e){(function(t,o,u){var s=256,p=6,f=52,r="random",n=u.pow(s,p),i=u.pow(2,f),l=i*2,a=s-1,c;function v(x,y,b){var z=[];y=y==!0?{entropy:!0}:y||{};var P=d(_(y.entropy?[x,M(o)]:x??I(),3),z),E=new m(z),w=function(){for(var S=E.g(p),T=n,A=0;S<i;)S=(S+A)*s,T*=s,A=E.g(1);for(;S>=l;)S/=2,T/=2,A>>>=1;return(S+A)/T};return w.int32=function(){return E.g(4)|0},w.quick=function(){return E.g(4)/4294967296},w.double=w,d(M(E.S),o),(y.pass||b||function(S,T,A,U){return U&&(U.S&&g(U,E),S.state=function(){return g(E,{})}),A?(u[r]=S,T):S})(w,P,"global"in y?y.global:this==u,y.state)}function m(x){var y,b=x.length,z=this,P=0,E=z.i=z.j=0,w=z.S=[];for(b||(x=[b++]);P<s;)w[P]=P++;for(P=0;P<s;P++)w[P]=w[E=a&E+x[P%b]+(y=w[P])],w[E]=y;(z.g=function(S){for(var T,A=0,U=z.i,j=z.j,Y=z.S;S--;)T=Y[U=a&U+1],A=A*s+Y[a&(Y[U]=Y[j=a&j+T])+(Y[j]=T)];return z.i=U,z.j=j,A})(s)}function g(x,y){return y.i=x.i,y.j=x.j,y.S=x.S.slice(),y}function _(x,y){var b=[],z=typeof x,P;if(y&&z=="object")for(P in x)try{b.push(_(x[P],y-1))}catch{}return b.length?b:z=="string"?x:x+"\0"}function d(x,y){for(var b=x+"",z,P=0;P<b.length;)y[a&P]=a&(z^=y[a&P]*19)+b.charCodeAt(P++);return M(y)}function I(){try{var x;return c&&(x=c.randomBytes)?x=x(s):(x=new Uint8Array(s),(t.crypto||t.msCrypto).getRandomValues(x)),M(x)}catch{var y=t.navigator,b=y&&y.plugins;return[+new Date,t,b,t.screen,M(o)]}}function M(x){return String.fromCharCode.apply(0,x)}if(d(u.random(),o),e.exports){e.exports=v;try{c=Sn}catch{}}else u["seed"+r]=v})(typeof self<"u"?self:N,[],Math)})(en);var In=en.exports,Cn=Pn,Tn=bn,Gn=_n,Rn=An,Dn=wn,Un=En,L=In;L.alea=Cn,L.xor128=Tn,L.xorwow=Gn,L.xorshift7=Rn,L.xor4096=Dn,L.tychei=Un;var Nn=L,rn=hn(Nn);const O=0,K=[[0,4,0],[1,5,0],[2,6,0],[3,7,0],[0,2,1],[4,6,1],[1,3,1],[5,7,1],[0,1,2],[2,3,2],[4,5,2],[6,7,2]],F=[[0,1,2,3,0],[4,5,6,7,0],[0,4,1,5,1],[2,6,3,7,1],[0,2,4,6,2],[1,3,5,7,2]],J=[[[4,0,0],[5,1,0],[6,2,0],[7,3,0]],[[2,0,1],[6,4,1],[3,1,1],[7,5,1]],[[1,0,2],[3,2,2],[5,4,2],[7,6,2]]],D=[[[1,4,0,5,1,1],[1,6,2,7,3,1],[0,4,6,0,2,2],[0,5,7,1,3,2]],[[0,2,3,0,1,0],[0,6,7,4,5,0],[1,2,0,6,4,2],[1,3,1,7,5,2]],[[1,1,0,3,2,0],[1,5,4,7,6,0],[0,1,5,0,4,1],[0,3,7,2,6,1]]],k=[[[3,2,1,0,0],[7,6,5,4,0]],[[5,1,4,0,1],[7,3,6,2,1]],[[6,4,2,0,2],[7,5,3,1,2]]],Ln=[[3,2,1,0],[7,5,6,4],[11,10,9,8]],on=[[0,4],[1,5],[2,6],[3,7],[0,2],[1,3],[4,6],[5,7],[0,1],[2,3],[4,5],[6,7]],On=(e,t,o)=>{let u=1e6,s=0;const p=[-1,-1,-1,-1];let f=!1;const r=[!1,!1,!1,!1];for(let n=0;n<4;n++){const i=Ln[t][n],l=on[i][0],a=on[i][1],c=e[n].drawInfo.corners>>l&1,v=e[n].drawInfo.corners>>a&1;e[n].size<u&&(u=e[n].size,s=n,f=c!==O),p[n]=e[n].drawInfo.index,r[n]=c===O&&v!==O||c!==O&&v===O}r[s]&&(f?(o.push(p[0]),o.push(p[3]),o.push(p[1]),o.push(p[0]),o.push(p[2]),o.push(p[3])):(o.push(p[0]),o.push(p[1]),o.push(p[3]),o.push(p[0]),o.push(p[3]),o.push(p[2])))},Q=(e,t,o)=>{if(!(e[0]==null||e[1]==null||e[2]==null||e[3]==null))if(e[0].type!=="internal"&&e[1].type!=="internal"&&e[2].type!=="internal"&&e[3].type!=="internal")On(e,t,o);else for(let u=0;u<2;u++){const s=[],p=[k[t][u][0],k[t][u][1],k[t][u][2],k[t][u][3]];for(let f=0;f<4;f++)e[f].type==="leaf"||e[f].type==="pseudo"?s[f]=e[f]:s[f]=e[f].children[p[f]];Q(s,k[t][u][4],o)}},an=(e,t,o)=>{if(!(e[0]==null||e[1]==null)&&(e[0].type==="internal"||e[1].type==="internal")){for(let s=0;s<4;s++){const p=[],f=[J[t][s][0],J[t][s][1]];for(let r=0;r<2;r++)e[r].type!=="internal"?p[r]=e[r]:p[r]=e[r].children[f[r]];an(p,J[t][s][2],o)}const u=[[0,0,1,1],[0,1,0,1]];for(let s=0;s<4;s++){const p=[],f=[D[t][s][1],D[t][s][2],D[t][s][3],D[t][s][4]],r=[u[D[t][s][0]][0],u[D[t][s][0]][1],u[D[t][s][0]][2],u[D[t][s][0]][3]];for(let n=0;n<4;n++)e[r[n]].type==="leaf"||e[r[n]].type==="pseudo"?p[n]=e[r[n]]:p[n]=e[r[n]].children[f[n]];Q(p,D[t][s][5],o)}}},sn=(e,t)=>{if(e!=null&&e.type==="internal"){for(let o=0;o<8;o++)sn(e.children[o],t);for(let o=0;o<12;o++){const u=[],s=[K[o][0],K[o][1]];u[0]=e.children[s[0]],u[1]=e.children[s[1]],an(u,K[o][2],t)}for(let o=0;o<6;o++){const u=[],s=[F[o][0],F[o][1],F[o][2],F[o][3]];for(let p=0;p<4;p++)u[p]=e.children[s[p]];Q(u,F[o][4],t)}}},nn=[B(0,0,0),B(0,0,1),B(0,1,0),B(0,1,1),B(1,0,0),B(1,0,1),B(1,1,0),B(1,1,1)],un=(e,t,o)=>{const u=new Map;for(let s=0;s<e.length;s++){const p=e[s],f=vn(R(),p.min,B((p.min[0]-t[0])%o,(p.min[1]-t[1])%o,(p.min[2]-t[2])%o));let r=u[`${f[0]},${f[1]},${f[2]}`];r||(r={min:f,size:o,type:"internal",children:[]},u[`${r.min[0]},${r.min[1]},${r.min[2]}`]=r);for(let n=0;n<8;n++){const i=fn(R(),f,B(nn[n][0]*p.size,nn[n][1]*p.size,nn[n][2]*p.size));if(pn(i,p.min)){r.children[n]=p;break}}}return e.length=0,Object.values(u)},Fn=(e,t,o)=>{if(e.length==0)return null;for(e.sort((s,p)=>s.size-p.size);e[0].size!=e[e.length-1].size;){let s=0;const p=e[s].size;do++s;while(e[s].size==p);let f=[];for(let r=0;r<s;r++)f.push(e[r]);f=un(f,t,p*2);for(let r=s;r<e.Count;r++)f.push(e[r]);e.length=0;for(let r=0;r<f.length;r++)e.push(f[r])}let u=e[0].size*2;for(;u<=o;)e=un(e,t,u),u*=2;return e.length!=1?(console.log(e),console.error("There can only be one root node!"),null):e[0]},cn=(e,t,o,u)=>{if(e!=null){if(e.size>u&&e.type!=="leaf")for(let s=0;s<8;s++)cn(e.children[s],t,o,u);if(e.type!=="internal"){const s=e.drawInfo;if(s==null)throw"Error! Could not add vertex!";s.index=t.length/3,t.push(s.position[0],s.position[1],s.position[2]),o.push(s.averageNormal[0],s.averageNormal[1],s.averageNormal[2])}}},kn=(e,t,o,u)=>{const s=[];if(o===0)return{vertices:new Float32Array,normals:new Float32Array,indices:new Uint16Array,corners:new Uint32Array};for(let i=0;i<o*12;i+=12)if(u[i+11]!==0){const l={type:"leaf",size:t,min:B(u[i],u[i+1],u[i+2]),drawInfo:{position:B(u[i+4],u[i+5],u[i+6]),averageNormal:B(u[i+8],u[i+9],u[i+10]),corners:u[i+3]}};s.push(l)}const p=Fn(s,e,32*t),f=[],r=[];cn(p,f,r,1);const n=[];return sn(p,n),{vertices:new Float32Array(f),normals:new Float32Array(r),indices:new Uint16Array(n),corners:new Uint32Array}};class tn{constructor(t,o,u,s,p,f,r,n,i,l,a,c,v,m,g,_,d,I,M,x,y){h(this,"running",!1);h(this,"computePipeline");h(this,"computeCornersPipeline");h(this,"uniformBuffer");h(this,"cornerMaterials");h(this,"cornerMaterialsRead");h(this,"voxelMaterialsBuffer");h(this,"voxelMaterialsBufferRead");h(this,"cornerIndexBuffer");h(this,"gpuReadBuffer");h(this,"permutationsBuffer");h(this,"voxelsBuffer");h(this,"computeBindGroup");h(this,"computeCornersBindGroup");h(this,"computePositionsPipeline");h(this,"computePositionsBindGroup");h(this,"computeVoxelsPipeline");h(this,"computeVoxelsBindGroup");h(this,"voxelReadBuffer");h(this,"density");h(this,"densityBindGroup");h(this,"mainDensityBindGroup");this.computePipeline=t,this.computeCornersPipeline=o,this.uniformBuffer=u,this.cornerMaterials=s,this.cornerMaterialsRead=p,this.voxelMaterialsBuffer=f,this.voxelMaterialsBufferRead=r,this.cornerIndexBuffer=n,this.gpuReadBuffer=i,this.permutationsBuffer=l,this.voxelsBuffer=a,this.computeBindGroup=c,this.computeCornersBindGroup=v,this.computePositionsPipeline=m,this.computePositionsBindGroup=g,this.computeVoxelsPipeline=_,this.computeVoxelsBindGroup=d,this.voxelReadBuffer=I,this.density=M,this.densityBindGroup=x,this.mainDensityBindGroup=y}static async init(t){const o=q.patch(yn),u=performance.now();console.log("Start loading voxel engine",performance.now()-u);const s=t.createShaderModule({code:o}),p=await t.createComputePipelineAsync({layout:"auto",compute:{module:s,entryPoint:"computeMaterials"}}),f=await t.createComputePipelineAsync({layout:"auto",compute:{module:t.createShaderModule({code:mn}),entryPoint:"main"}}),r=Math.max(4*5,32),n=t.createBuffer({size:r,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=t.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*33*33*33,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),l=t.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*33*33*33,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),a=t.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*32*32*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),c=t.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*32*32*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),v=t.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT+Uint32Array.BYTES_PER_ELEMENT*32*32*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),m=t.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),g=new Int32Array(512),_=new rn(6452);for(let A=0;A<256;A++)g[A]=256*_();for(let A=256;A<512;A++)g[A]=g[A-256];const d=t.createBuffer({size:g.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});new Int32Array(d.getMappedRange()).set(g),d.unmap();const I=t.createBuffer({size:Float32Array.BYTES_PER_ELEMENT*12*32*32*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),M=t.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:i}},{binding:5,resource:{buffer:n}}]}),x=t.createBindGroup({layout:f.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:a}}]}),y=await t.createComputePipelineAsync({layout:"auto",compute:{module:t.createShaderModule({code:xn}),entryPoint:"main"}}),b=t.createBindGroup({layout:y.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:v}}]}),z=await t.createComputePipelineAsync({layout:"auto",compute:{module:s,entryPoint:"main"}}),P=t.createBindGroup({layout:z.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:v}},{binding:4,resource:{buffer:I}},{binding:5,resource:{buffer:n}}]}),E=t.createBuffer({size:Float32Array.BYTES_PER_ELEMENT*12*32*32*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),w=await q.init(t),S=await w.apply(t,p),T=await w.apply(t,z);return console.log("Done",performance.now()-u),new tn(p,f,n,i,l,a,c,v,m,d,I,M,x,y,b,z,P,E,w,S,T)}generate(t,o,u,s,p){return s||(s=1),new Promise(f=>{this.density.updateRaw(t,p);const r=new Int32Array(512),n=new rn("James");for(let d=0;d<256;d++)r[d]=256*n();for(let d=256;d<512;d++)r[d]=r[d-256];t.queue.writeBuffer(this.permutationsBuffer,0,r.buffer,r.byteOffset,r.byteLength);const i=new ArrayBuffer(4*5),l=new Float32Array(i,0,4);l.set(u,0),l[3]=s,new Uint32Array(i,16,1)[0]=33,t.queue.writeBuffer(this.uniformBuffer,0,i,0,i.byteLength);const a=t.createCommandEncoder(),c=32,v=a.beginComputePass();v.setPipeline(this.computePipeline),v.setBindGroup(0,this.computeBindGroup),this.densityBindGroup.apply(v),v.dispatchWorkgroups(c+1,c+1,c+1),v.end();const m=a.beginComputePass();m.setPipeline(this.computeCornersPipeline),m.setBindGroup(0,this.computeCornersBindGroup),m.dispatchWorkgroups(c,c,c),m.end();const g=a.beginComputePass();g.setPipeline(this.computePositionsPipeline),g.setBindGroup(0,this.computePositionsBindGroup),g.dispatchWorkgroups(1),g.end();const _=t.createCommandEncoder();_.copyBufferToBuffer(this.cornerIndexBuffer,0,this.gpuReadBuffer,0,Uint32Array.BYTES_PER_ELEMENT),_.copyBufferToBuffer(this.cornerMaterials,0,this.cornerMaterialsRead,0,Uint32Array.BYTES_PER_ELEMENT*33*33*33),_.copyBufferToBuffer(this.voxelMaterialsBuffer,0,this.voxelMaterialsBufferRead,0,Uint32Array.BYTES_PER_ELEMENT*32*32*32),o({items:[a.finish(),_.finish()],callback:async()=>{await this.cornerMaterialsRead.mapAsync(GPUMapMode.READ);const d=new Uint32Array(this.cornerMaterialsRead.getMappedRange()).slice();this.cornerMaterialsRead.unmap(),await this.gpuReadBuffer.mapAsync(GPUMapMode.READ);const I=this.gpuReadBuffer.getMappedRange(),M=new Uint32Array(I)[0];if(this.gpuReadBuffer.unmap(),M===0){f({vertices:new Float32Array,normals:new Float32Array,indices:new Uint16Array,corners:d,consistency:d[0]});return}const x=Math.ceil(M/128),y=t.createCommandEncoder(),b=y.beginComputePass();b.setPipeline(this.computeVoxelsPipeline),b.setBindGroup(0,this.computeVoxelsBindGroup),this.mainDensityBindGroup.apply(b),b.dispatchWorkgroups(x),b.end();const z=t.createCommandEncoder();z.copyBufferToBuffer(this.voxelsBuffer,0,this.voxelReadBuffer,0,Float32Array.BYTES_PER_ELEMENT*M*12),o({items:[y.finish(),z.finish()],callback:async()=>{await this.voxelReadBuffer.mapAsync(GPUMapMode.READ);const P=this.voxelReadBuffer.getMappedRange(),E=new Float32Array(P),w=kn(u,s,M,E);this.voxelReadBuffer.unmap(),f({...w,corners:d,consistency:-1})}})}})})}}const Yn=self;(async function(){const e=await navigator.gpu.requestAdapter();if(!e)throw new Error("Unable to acquire GPU adapter, is WebGPU enabled?");const t=await e.requestDevice(),o=await tn.init(t);console.log("Voxel engine init complete"),postMessage({type:"init_complete"});const u=s=>{t.queue.onSubmittedWorkDone().then(s.callback),t.queue.submit(s.items)};onmessage=async function(s){const{detail:p,density:f}=s.data,r=31,{x:n,y:i,z:l,s:a}=p,c=a*r*.5,{vertices:v,normals:m,indices:g,consistency:_}=await o.generate(t,u,B(n*r-c,i*r-c,l*r-c),a,f);Yn.postMessage({type:"update",i:`${n}:${i}:${l}`,ix:n,iy:i,iz:l,x:0,y:0,z:0,vertices:v.buffer,normals:m.buffer,indices:g.buffer,stride:a,consistency:_},[v.buffer,m.buffer,g.buffer])}})()})();
