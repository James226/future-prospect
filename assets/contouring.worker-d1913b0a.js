var Yn=Object.defineProperty;var qn=(G,C,R)=>C in G?Yn(G,C,{enumerable:!0,configurable:!0,writable:!0,value:R}):G[C]=R;var h=(G,C,R)=>(qn(G,typeof C!="symbol"?C+"":C,R),R);(function(){"use strict";var G=1e-6,C=typeof Float32Array<"u"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var t=0,e=arguments.length;e--;)t+=arguments[e]*arguments[e];return Math.sqrt(t)});function R(){var t=new C(3);return C!=Float32Array&&(t[0]=0,t[1]=0,t[2]=0),t}function M(t,e,o){var c=new C(3);return c[0]=t,c[1]=e,c[2]=o,c}function fn(t,e,o){return t[0]=e[0]+o[0],t[1]=e[1]+o[1],t[2]=e[2]+o[2],t}function ln(t,e,o){return t[0]=e[0]-o[0],t[1]=e[1]-o[1],t[2]=e[2]-o[2],t}function pn(t,e){var o=t[0],c=t[1],s=t[2],p=e[0],f=e[1],r=e[2];return Math.abs(o-p)<=G*Math.max(1,Math.abs(o),Math.abs(p))&&Math.abs(c-f)<=G*Math.max(1,Math.abs(c),Math.abs(f))&&Math.abs(s-r)<=G*Math.max(1,Math.abs(s),Math.abs(r))}var vn=ln;(function(){var t=R();return function(e,o,c,s,p,f){var r,n;for(o||(o=3),c||(c=0),s?n=Math.min(s*o+c,e.length):n=e.length,r=c;r<n;r+=o)t[0]=e[r],t[1]=e[r+1],t[2]=e[r+2],p(t,t,f),e[r]=t[0],e[r+1]=t[1],e[r+2]=t[2];return e}})();var yn=`const OctreeSize = 32u;\r
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
}`,mn=`struct Permutations {
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
`,dn=`const freq = 0.001;

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

fn calculateDensity(worldPosition: vec3<f32>) -> Density {
	var worldRadius: f32 = 5000.0;
	var world: vec3<f32> = worldPosition - vec3<f32>(2000000.0, 100.0, 100.0);
	var worldDist: f32 = -worldRadius + length(world);

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

	let mountain: f32 = CalculateNoiseValue2(world, 0.07);
	//let mountain: f32 = 0.0;

	var blob: f32 = CalculateNoiseValue2(world, flatlandNoiseScale + ((rockyNoiseScale - flatlandNoiseScale) * rockyBlend));
	blob = CLerp2(blob, (worldDist) * (flatlandYPercent + ((rockyYPercent - flatlandYPercent) * rockyBlend)),
				flatlandLerpAmount + ((rockyLerpAmount - flatlandLerpAmount) * rockyBlend))
				+ CLerp2(mountain, blob, minMountainMixLerpAmount + ((maxMountainMixLerpAmount - minMountainMixLerpAmount) * mountainBlend));

  var result = Density(1.0, MATERIAL_AIR);

	result = add(result, blob, MATERIAL_WOOD);

  result = add(result, Box(worldPosition, vec3<f32>(2000000.0, 150.0, 5000.0), vec3<f32>(5000.0, 1000.0, 5000.0)), MATERIAL_ROCK);
  //result = add(result, Sphere(worldPosition, vec3<f32>(2000000.0, 100.0, 100.0), 5000.0), MATERIAL_ROCK);

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
    let shape = (augmentation.attributes & 0xFE) >> 1;
    var density: f32 = 0.0;
    switch(shape) {
      case 0: {
        density = Sphere(worldPosition, vec3<f32>(augmentation.position.x, augmentation.position.y, augmentation.position.z), augmentation.size);
      }
      case 1: {
        density = Box(worldPosition, vec3<f32>(augmentation.position.x, augmentation.position.y, augmentation.position.z), vec3<f32>(augmentation.size));
      }
      default: {
        density = 0.0;
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
`;class gn{constructor(e){h(this,"bindGroup");this.bindGroup=e}apply(e){e.setBindGroup(1,this.bindGroup)}}class k{constructor(e){h(this,"augmentationBuffer");h(this,"augmentationArray",[]);h(this,"augmentations");this.augmentationBuffer=e,this.augmentations=new ArrayBuffer(Uint32Array.BYTES_PER_ELEMENT*4)}static async init(e){const o=64*Float32Array.BYTES_PER_ELEMENT*8+8,c=e.createBuffer({size:o,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!1});return new k(c)}async apply(e,o){const c=e.createBindGroup({layout:o.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:this.augmentationBuffer}}]});return new gn(c)}modify(e,o){this.augmentationArray.push(o),this.update(e,this.augmentationArray)}update(e,o){this.augmentations=new ArrayBuffer(Uint32Array.BYTES_PER_ELEMENT*4+Uint32Array.BYTES_PER_ELEMENT*o.length*8);const c=new Uint32Array(this.augmentations,0,4);c[0]=o.length;const s=new Float32Array(this.augmentations,Uint32Array.BYTES_PER_ELEMENT*4),p=new Uint32Array(this.augmentations,Uint32Array.BYTES_PER_ELEMENT*4);for(let f=0;f<o.length;f++)s[f*8]=o[f].x,s[f*8+1]=o[f].y,s[f*8+2]=o[f].z,s[f*8+3]=o[f].size,p[f*8+4]=o[f].type|o[f].shape<<1|o[f].material<<8;e.queue.writeBuffer(this.augmentationBuffer,0,this.augmentations,0,this.augmentations.byteLength)}updateRaw(e,o){this.augmentations=o,e.queue.writeBuffer(this.augmentationBuffer,0,this.augmentations,0,this.augmentations.byteLength)}static patch(e){return e.replace("#import density",dn)}}var N=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function hn(t){if(t.__esModule)return t;var e=t.default;if(typeof e=="function"){var o=function c(){if(this instanceof c){var s=[null];s.push.apply(s,arguments);var p=Function.bind.apply(e,s);return new p}return e.apply(this,arguments)};o.prototype=e.prototype}else o={};return Object.defineProperty(o,"__esModule",{value:!0}),Object.keys(t).forEach(function(c){var s=Object.getOwnPropertyDescriptor(t,c);Object.defineProperty(o,c,s.get?s:{enumerable:!0,get:function(){return t[c]}})}),o}var V={},Pn={get exports(){return V},set exports(t){V=t}};(function(t){(function(e,o,c){function s(n){var i=this,l=r();i.next=function(){var a=2091639*i.s0+i.c*23283064365386963e-26;return i.s0=i.s1,i.s1=i.s2,i.s2=a-(i.c=a|0)},i.c=1,i.s0=l(" "),i.s1=l(" "),i.s2=l(" "),i.s0-=l(n),i.s0<0&&(i.s0+=1),i.s1-=l(n),i.s1<0&&(i.s1+=1),i.s2-=l(n),i.s2<0&&(i.s2+=1),l=null}function p(n,i){return i.c=n.c,i.s0=n.s0,i.s1=n.s1,i.s2=n.s2,i}function f(n,i){var l=new s(n),a=i&&i.state,u=l.next;return u.int32=function(){return l.next()*4294967296|0},u.double=function(){return u()+(u()*2097152|0)*11102230246251565e-32},u.quick=u,a&&(typeof a=="object"&&p(a,l),u.state=function(){return p(l,{})}),u}function r(){var n=4022871197,i=function(l){l=String(l);for(var a=0;a<l.length;a++){n+=l.charCodeAt(a);var u=.02519603282416938*n;n=u>>>0,u-=n,u*=n,n=u>>>0,u-=n,n+=u*4294967296}return(n>>>0)*23283064365386963e-26};return i}o&&o.exports?o.exports=f:c&&c.amd?c(function(){return f}):this.alea=f})(N,t,!1)})(Pn);var X={},bn={get exports(){return X},set exports(t){X=t}};(function(t){(function(e,o,c){function s(r){var n=this,i="";n.x=0,n.y=0,n.z=0,n.w=0,n.next=function(){var a=n.x^n.x<<11;return n.x=n.y,n.y=n.z,n.z=n.w,n.w^=n.w>>>19^a^a>>>8},r===(r|0)?n.x=r:i+=r;for(var l=0;l<i.length+64;l++)n.x^=i.charCodeAt(l)|0,n.next()}function p(r,n){return n.x=r.x,n.y=r.y,n.z=r.z,n.w=r.w,n}function f(r,n){var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var u=i.next()>>>11,v=(i.next()>>>0)/4294967296,y=(u+v)/(1<<21);while(y===0);return y},a.int32=i.next,a.quick=a,l&&(typeof l=="object"&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:c&&c.amd?c(function(){return f}):this.xor128=f})(N,t,!1)})(bn);var H={},_n={get exports(){return H},set exports(t){H=t}};(function(t){(function(e,o,c){function s(r){var n=this,i="";n.next=function(){var a=n.x^n.x>>>2;return n.x=n.y,n.y=n.z,n.z=n.w,n.w=n.v,(n.d=n.d+362437|0)+(n.v=n.v^n.v<<4^(a^a<<1))|0},n.x=0,n.y=0,n.z=0,n.w=0,n.v=0,r===(r|0)?n.x=r:i+=r;for(var l=0;l<i.length+64;l++)n.x^=i.charCodeAt(l)|0,l==i.length&&(n.d=n.x<<10^n.x>>>4),n.next()}function p(r,n){return n.x=r.x,n.y=r.y,n.z=r.z,n.w=r.w,n.v=r.v,n.d=r.d,n}function f(r,n){var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var u=i.next()>>>11,v=(i.next()>>>0)/4294967296,y=(u+v)/(1<<21);while(y===0);return y},a.int32=i.next,a.quick=a,l&&(typeof l=="object"&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:c&&c.amd?c(function(){return f}):this.xorwow=f})(N,t,!1)})(_n);var $={},An={get exports(){return $},set exports(t){$=t}};(function(t){(function(e,o,c){function s(r){var n=this;n.next=function(){var l=n.x,a=n.i,u,v;return u=l[a],u^=u>>>7,v=u^u<<24,u=l[a+1&7],v^=u^u>>>10,u=l[a+3&7],v^=u^u>>>3,u=l[a+4&7],v^=u^u<<7,u=l[a+7&7],u=u^u<<13,v^=u^u<<9,l[a]=v,n.i=a+1&7,v};function i(l,a){var u,v=[];if(a===(a|0))v[0]=a;else for(a=""+a,u=0;u<a.length;++u)v[u&7]=v[u&7]<<15^a.charCodeAt(u)+v[u+1&7]<<13;for(;v.length<8;)v.push(0);for(u=0;u<8&&v[u]===0;++u);for(u==8?v[7]=-1:v[u],l.x=v,l.i=0,u=256;u>0;--u)l.next()}i(n,r)}function p(r,n){return n.x=r.x.slice(),n.i=r.i,n}function f(r,n){r==null&&(r=+new Date);var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var u=i.next()>>>11,v=(i.next()>>>0)/4294967296,y=(u+v)/(1<<21);while(y===0);return y},a.int32=i.next,a.quick=a,l&&(l.x&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:c&&c.amd?c(function(){return f}):this.xorshift7=f})(N,t,!1)})(An);var Z={},wn={get exports(){return Z},set exports(t){Z=t}};(function(t){(function(e,o,c){function s(r){var n=this;n.next=function(){var l=n.w,a=n.X,u=n.i,v,y;return n.w=l=l+1640531527|0,y=a[u+34&127],v=a[u=u+1&127],y^=y<<13,v^=v<<17,y^=y>>>15,v^=v>>>12,y=a[u]=y^v,n.i=u,y+(l^l>>>16)|0};function i(l,a){var u,v,y,d,A,g=[],S=128;for(a===(a|0)?(v=a,a=null):(a=a+"\0",v=0,S=Math.max(S,a.length)),y=0,d=-32;d<S;++d)a&&(v^=a.charCodeAt((d+32)%a.length)),d===0&&(A=v),v^=v<<10,v^=v>>>15,v^=v<<4,v^=v>>>13,d>=0&&(A=A+1640531527|0,u=g[d&127]^=v+A,y=u==0?y+1:0);for(y>=128&&(g[(a&&a.length||0)&127]=-1),y=127,d=4*128;d>0;--d)v=g[y+34&127],u=g[y=y+1&127],v^=v<<13,u^=u<<17,v^=v>>>15,u^=u>>>12,g[y]=v^u;l.w=A,l.X=g,l.i=y}i(n,r)}function p(r,n){return n.i=r.i,n.w=r.w,n.X=r.X.slice(),n}function f(r,n){r==null&&(r=+new Date);var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var u=i.next()>>>11,v=(i.next()>>>0)/4294967296,y=(u+v)/(1<<21);while(y===0);return y},a.int32=i.next,a.quick=a,l&&(l.X&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:c&&c.amd?c(function(){return f}):this.xor4096=f})(N,t,!1)})(wn);var W={},En={get exports(){return W},set exports(t){W=t}};(function(t){(function(e,o,c){function s(r){var n=this,i="";n.next=function(){var a=n.b,u=n.c,v=n.d,y=n.a;return a=a<<25^a>>>7^u,u=u-v|0,v=v<<24^v>>>8^y,y=y-a|0,n.b=a=a<<20^a>>>12^u,n.c=u=u-v|0,n.d=v<<16^u>>>16^y,n.a=y-a|0},n.a=0,n.b=0,n.c=-1640531527,n.d=1367130551,r===Math.floor(r)?(n.a=r/4294967296|0,n.b=r|0):i+=r;for(var l=0;l<i.length+20;l++)n.b^=i.charCodeAt(l)|0,n.next()}function p(r,n){return n.a=r.a,n.b=r.b,n.c=r.c,n.d=r.d,n}function f(r,n){var i=new s(r),l=n&&n.state,a=function(){return(i.next()>>>0)/4294967296};return a.double=function(){do var u=i.next()>>>11,v=(i.next()>>>0)/4294967296,y=(u+v)/(1<<21);while(y===0);return y},a.int32=i.next,a.quick=a,l&&(typeof l=="object"&&p(l,i),a.state=function(){return p(i,{})}),a}o&&o.exports?o.exports=f:c&&c.amd?c(function(){return f}):this.tychei=f})(N,t,!1)})(En);var K={},zn={get exports(){return K},set exports(t){K=t}},Mn={},Bn=Object.freeze({__proto__:null,default:Mn}),In=hn(Bn);(function(t){(function(e,o,c){var s=256,p=6,f=52,r="random",n=c.pow(s,p),i=c.pow(2,f),l=i*2,a=s-1,u;function v(x,m,_){var P=[];m=m==!0?{entropy:!0}:m||{};var b=g(A(m.entropy?[x,B(o)]:x??S(),3),P),z=new y(P),E=function(){for(var I=z.g(p),T=n,w=0;I<i;)I=(I+w)*s,T*=s,w=z.g(1);for(;I>=l;)I/=2,T/=2,w>>>=1;return(I+w)/T};return E.int32=function(){return z.g(4)|0},E.quick=function(){return z.g(4)/4294967296},E.double=E,g(B(z.S),o),(m.pass||_||function(I,T,w,D){return D&&(D.S&&d(D,z),I.state=function(){return d(z,{})}),w?(c[r]=I,T):I})(E,b,"global"in m?m.global:this==c,m.state)}function y(x){var m,_=x.length,P=this,b=0,z=P.i=P.j=0,E=P.S=[];for(_||(x=[_++]);b<s;)E[b]=b++;for(b=0;b<s;b++)E[b]=E[z=a&z+x[b%_]+(m=E[b])],E[z]=m;(P.g=function(I){for(var T,w=0,D=P.i,j=P.j,q=P.S;I--;)T=q[D=a&D+1],w=w*s+q[a&(q[D]=q[j=a&j+T])+(q[j]=T)];return P.i=D,P.j=j,w})(s)}function d(x,m){return m.i=x.i,m.j=x.j,m.S=x.S.slice(),m}function A(x,m){var _=[],P=typeof x,b;if(m&&P=="object")for(b in x)try{_.push(A(x[b],m-1))}catch{}return _.length?_:P=="string"?x:x+"\0"}function g(x,m){for(var _=x+"",P,b=0;b<_.length;)m[a&b]=a&(P^=m[a&b]*19)+_.charCodeAt(b++);return B(m)}function S(){try{var x;return u&&(x=u.randomBytes)?x=x(s):(x=new Uint8Array(s),(e.crypto||e.msCrypto).getRandomValues(x)),B(x)}catch{var m=e.navigator,_=m&&m.plugins;return[+new Date,e,_,e.screen,B(o)]}}function B(x){return String.fromCharCode.apply(0,x)}if(g(c.random(),o),t.exports){t.exports=v;try{u=In}catch{}}else c["seed"+r]=v})(typeof self<"u"?self:N,[],Math)})(zn);var Sn=V,Cn=X,Tn=H,Gn=$,Rn=Z,Un=W,L=K;L.alea=Sn,L.xor128=Cn,L.xorwow=Tn,L.xorshift7=Gn,L.xor4096=Rn,L.tychei=Un;var rn=L;const O=0,J=[[0,4,0],[1,5,0],[2,6,0],[3,7,0],[0,2,1],[4,6,1],[1,3,1],[5,7,1],[0,1,2],[2,3,2],[4,5,2],[6,7,2]],F=[[0,1,2,3,0],[4,5,6,7,0],[0,4,1,5,1],[2,6,3,7,1],[0,2,4,6,2],[1,3,5,7,2]],Q=[[[4,0,0],[5,1,0],[6,2,0],[7,3,0]],[[2,0,1],[6,4,1],[3,1,1],[7,5,1]],[[1,0,2],[3,2,2],[5,4,2],[7,6,2]]],U=[[[1,4,0,5,1,1],[1,6,2,7,3,1],[0,4,6,0,2,2],[0,5,7,1,3,2]],[[0,2,3,0,1,0],[0,6,7,4,5,0],[1,2,0,6,4,2],[1,3,1,7,5,2]],[[1,1,0,3,2,0],[1,5,4,7,6,0],[0,1,5,0,4,1],[0,3,7,2,6,1]]],Y=[[[3,2,1,0,0],[7,6,5,4,0]],[[5,1,4,0,1],[7,3,6,2,1]],[[6,4,2,0,2],[7,5,3,1,2]]],Dn=[[3,2,1,0],[7,5,6,4],[11,10,9,8]],on=[[0,4],[1,5],[2,6],[3,7],[0,2],[1,3],[4,6],[5,7],[0,1],[2,3],[4,5],[6,7]],Nn=(t,e,o)=>{let c=1e6,s=0;const p=[-1,-1,-1,-1];let f=!1;const r=[!1,!1,!1,!1];for(let n=0;n<4;n++){const i=Dn[e][n],l=on[i][0],a=on[i][1],u=t[n].drawInfo.corners>>l&1,v=t[n].drawInfo.corners>>a&1;t[n].size<c&&(c=t[n].size,s=n,f=u!==O),p[n]=t[n].drawInfo.index,r[n]=u===O&&v!==O||u!==O&&v===O}r[s]&&(f?(o.push(p[0]),o.push(p[3]),o.push(p[1]),o.push(p[0]),o.push(p[2]),o.push(p[3])):(o.push(p[0]),o.push(p[1]),o.push(p[3]),o.push(p[0]),o.push(p[3]),o.push(p[2])))},nn=(t,e,o)=>{if(!(t[0]==null||t[1]==null||t[2]==null||t[3]==null))if(t[0].type!=="internal"&&t[1].type!=="internal"&&t[2].type!=="internal"&&t[3].type!=="internal")Nn(t,e,o);else for(let c=0;c<2;c++){const s=[],p=[Y[e][c][0],Y[e][c][1],Y[e][c][2],Y[e][c][3]];for(let f=0;f<4;f++)t[f].type==="leaf"||t[f].type==="pseudo"?s[f]=t[f]:s[f]=t[f].children[p[f]];nn(s,Y[e][c][4],o)}},an=(t,e,o)=>{if(!(t[0]==null||t[1]==null)&&(t[0].type==="internal"||t[1].type==="internal")){for(let s=0;s<4;s++){const p=[],f=[Q[e][s][0],Q[e][s][1]];for(let r=0;r<2;r++)t[r].type!=="internal"?p[r]=t[r]:p[r]=t[r].children[f[r]];an(p,Q[e][s][2],o)}const c=[[0,0,1,1],[0,1,0,1]];for(let s=0;s<4;s++){const p=[],f=[U[e][s][1],U[e][s][2],U[e][s][3],U[e][s][4]],r=[c[U[e][s][0]][0],c[U[e][s][0]][1],c[U[e][s][0]][2],c[U[e][s][0]][3]];for(let n=0;n<4;n++)t[r[n]].type==="leaf"||t[r[n]].type==="pseudo"?p[n]=t[r[n]]:p[n]=t[r[n]].children[f[n]];nn(p,U[e][s][5],o)}}},sn=(t,e)=>{if(t!=null&&t.type==="internal"){for(let o=0;o<8;o++)sn(t.children[o],e);for(let o=0;o<12;o++){const c=[],s=[J[o][0],J[o][1]];c[0]=t.children[s[0]],c[1]=t.children[s[1]],an(c,J[o][2],e)}for(let o=0;o<6;o++){const c=[],s=[F[o][0],F[o][1],F[o][2],F[o][3]];for(let p=0;p<4;p++)c[p]=t.children[s[p]];nn(c,F[o][4],e)}}},tn=[M(0,0,0),M(0,0,1),M(0,1,0),M(0,1,1),M(1,0,0),M(1,0,1),M(1,1,0),M(1,1,1)],cn=(t,e,o)=>{const c=new Map;for(let s=0;s<t.length;s++){const p=t[s],f=vn(R(),p.min,M((p.min[0]-e[0])%o,(p.min[1]-e[1])%o,(p.min[2]-e[2])%o));let r=c[`${f[0]},${f[1]},${f[2]}`];r||(r={min:f,size:o,type:"internal",children:[]},c[`${r.min[0]},${r.min[1]},${r.min[2]}`]=r);for(let n=0;n<8;n++){const i=fn(R(),f,M(tn[n][0]*p.size,tn[n][1]*p.size,tn[n][2]*p.size));if(pn(i,p.min)){r.children[n]=p;break}}}return t.length=0,Object.values(c)},Ln=(t,e,o)=>{if(t.length==0)return null;for(t.sort((s,p)=>s.size-p.size);t[0].size!=t[t.length-1].size;){let s=0;const p=t[s].size;do++s;while(t[s].size==p);let f=[];for(let r=0;r<s;r++)f.push(t[r]);f=cn(f,e,p*2);for(let r=s;r<t.Count;r++)f.push(t[r]);t.length=0;for(let r=0;r<f.length;r++)t.push(f[r])}let c=t[0].size*2;for(;c<=o;)t=cn(t,e,c),c*=2;return t.length!=1?(console.log(t),console.error("There can only be one root node!"),null):t[0]},un=(t,e,o,c)=>{if(t!=null){if(t.size>c&&t.type!=="leaf")for(let s=0;s<8;s++)un(t.children[s],e,o,c);if(t.type!=="internal"){const s=t.drawInfo;if(s==null)throw"Error! Could not add vertex!";s.index=e.length/3,e.push(s.position[0],s.position[1],s.position[2]),o.push(s.averageNormal[0],s.averageNormal[1],s.averageNormal[2])}}},On=(t,e,o,c)=>{const s=[];if(o===0)return{vertices:new Float32Array,normals:new Float32Array,indices:new Uint16Array,corners:new Uint32Array};for(let i=0;i<o*12;i+=12)if(c[i+11]!==0){const l={type:"leaf",size:e,min:M(c[i],c[i+1],c[i+2]),drawInfo:{position:M(c[i+4],c[i+5],c[i+6]),averageNormal:M(c[i+8],c[i+9],c[i+10]),corners:c[i+3]}};s.push(l)}const p=Ln(s,t,32*e),f=[],r=[];un(p,f,r,1);const n=[];return sn(p,n),{vertices:new Float32Array(f),normals:new Float32Array(r),indices:new Uint16Array(n),corners:new Uint32Array}};class en{constructor(e,o,c,s,p,f,r,n,i,l,a,u,v,y,d,A,g,S,B,x,m){h(this,"running",!1);h(this,"computePipeline");h(this,"computeCornersPipeline");h(this,"uniformBuffer");h(this,"cornerMaterials");h(this,"cornerMaterialsRead");h(this,"voxelMaterialsBuffer");h(this,"voxelMaterialsBufferRead");h(this,"cornerIndexBuffer");h(this,"gpuReadBuffer");h(this,"permutationsBuffer");h(this,"voxelsBuffer");h(this,"computeBindGroup");h(this,"computeCornersBindGroup");h(this,"computePositionsPipeline");h(this,"computePositionsBindGroup");h(this,"computeVoxelsPipeline");h(this,"computeVoxelsBindGroup");h(this,"voxelReadBuffer");h(this,"density");h(this,"densityBindGroup");h(this,"mainDensityBindGroup");this.computePipeline=e,this.computeCornersPipeline=o,this.uniformBuffer=c,this.cornerMaterials=s,this.cornerMaterialsRead=p,this.voxelMaterialsBuffer=f,this.voxelMaterialsBufferRead=r,this.cornerIndexBuffer=n,this.gpuReadBuffer=i,this.permutationsBuffer=l,this.voxelsBuffer=a,this.computeBindGroup=u,this.computeCornersBindGroup=v,this.computePositionsPipeline=y,this.computePositionsBindGroup=d,this.computeVoxelsPipeline=A,this.computeVoxelsBindGroup=g,this.voxelReadBuffer=S,this.density=B,this.densityBindGroup=x,this.mainDensityBindGroup=m}static async init(e){const o=k.patch(mn),c=performance.now();console.log("Start loading voxel engine",performance.now()-c);const s=e.createShaderModule({code:o}),p=await e.createComputePipelineAsync({layout:"auto",compute:{module:s,entryPoint:"computeMaterials"}}),f=await e.createComputePipelineAsync({layout:"auto",compute:{module:e.createShaderModule({code:yn}),entryPoint:"main"}}),r=Math.max(4*5,32),n=e.createBuffer({size:r,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=e.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*33*33*33,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),l=e.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*33*33*33,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),a=e.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*32*32*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),u=e.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT*32*32*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),v=e.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT+Uint32Array.BYTES_PER_ELEMENT*32*32*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),y=e.createBuffer({size:Uint32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),d=new Int32Array(512),A=new rn(6452);for(let w=0;w<256;w++)d[w]=256*A();for(let w=256;w<512;w++)d[w]=d[w-256];const g=e.createBuffer({size:d.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});new Int32Array(g.getMappedRange()).set(d),g.unmap();const S=e.createBuffer({size:Float32Array.BYTES_PER_ELEMENT*12*32*32*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!1}),B=e.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:i}},{binding:5,resource:{buffer:n}}]}),x=e.createBindGroup({layout:f.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:a}}]}),m=await e.createComputePipelineAsync({layout:"auto",compute:{module:e.createShaderModule({code:xn}),entryPoint:"main"}}),_=e.createBindGroup({layout:m.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:v}}]}),P=await e.createComputePipelineAsync({layout:"auto",compute:{module:s,entryPoint:"main"}}),b=e.createBindGroup({layout:P.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:v}},{binding:4,resource:{buffer:S}},{binding:5,resource:{buffer:n}}]}),z=e.createBuffer({size:Float32Array.BYTES_PER_ELEMENT*12*32*32*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),E=await k.init(e),I=await E.apply(e,p),T=await E.apply(e,P);return console.log("Done",performance.now()-c),new en(p,f,n,i,l,a,u,v,y,g,S,B,x,m,_,P,b,z,E,I,T)}generate(e,o,c,s,p){return s||(s=1),new Promise(f=>{this.density.updateRaw(e,p);const r=new Int32Array(512),n=new rn("James");for(let g=0;g<256;g++)r[g]=256*n();for(let g=256;g<512;g++)r[g]=r[g-256];e.queue.writeBuffer(this.permutationsBuffer,0,r.buffer,r.byteOffset,r.byteLength);const i=new ArrayBuffer(4*5),l=new Float32Array(i,0,4);l.set(c,0),l[3]=s,new Uint32Array(i,16,1)[0]=33,e.queue.writeBuffer(this.uniformBuffer,0,i,0,i.byteLength);const a=e.createCommandEncoder(),u=32,v=a.beginComputePass();v.setPipeline(this.computePipeline),v.setBindGroup(0,this.computeBindGroup),this.densityBindGroup.apply(v),v.dispatchWorkgroups(u+1,u+1,u+1),v.end();const y=a.beginComputePass();y.setPipeline(this.computeCornersPipeline),y.setBindGroup(0,this.computeCornersBindGroup),y.dispatchWorkgroups(u,u,u),y.end();const d=a.beginComputePass();d.setPipeline(this.computePositionsPipeline),d.setBindGroup(0,this.computePositionsBindGroup),d.dispatchWorkgroups(1),d.end();const A=e.createCommandEncoder();A.copyBufferToBuffer(this.cornerIndexBuffer,0,this.gpuReadBuffer,0,Uint32Array.BYTES_PER_ELEMENT),A.copyBufferToBuffer(this.cornerMaterials,0,this.cornerMaterialsRead,0,Uint32Array.BYTES_PER_ELEMENT*33*33*33),A.copyBufferToBuffer(this.voxelMaterialsBuffer,0,this.voxelMaterialsBufferRead,0,Uint32Array.BYTES_PER_ELEMENT*32*32*32),o({items:[a.finish(),A.finish()],callback:async()=>{await this.cornerMaterialsRead.mapAsync(GPUMapMode.READ);const g=new Uint32Array(this.cornerMaterialsRead.getMappedRange()).slice();this.cornerMaterialsRead.unmap(),await this.gpuReadBuffer.mapAsync(GPUMapMode.READ);const S=this.gpuReadBuffer.getMappedRange(),B=new Uint32Array(S)[0];if(this.gpuReadBuffer.unmap(),B===0){f({vertices:new Float32Array,normals:new Float32Array,indices:new Uint16Array,corners:g,consistency:g[0]});return}const x=Math.ceil(B/128),m=e.createCommandEncoder(),_=m.beginComputePass();_.setPipeline(this.computeVoxelsPipeline),_.setBindGroup(0,this.computeVoxelsBindGroup),this.mainDensityBindGroup.apply(_),_.dispatchWorkgroups(x),_.end();const P=e.createCommandEncoder();P.copyBufferToBuffer(this.voxelsBuffer,0,this.voxelReadBuffer,0,Float32Array.BYTES_PER_ELEMENT*B*12),o({items:[m.finish(),P.finish()],callback:async()=>{await this.voxelReadBuffer.mapAsync(GPUMapMode.READ);const b=this.voxelReadBuffer.getMappedRange(),z=new Float32Array(b),E=On(c,s,B,z);this.voxelReadBuffer.unmap(),f({...E,corners:g,consistency:-1})}})}})})}}const Fn=self;(async function(){const t=await navigator.gpu.requestAdapter();if(!t)throw new Error("Unable to acquire GPU adapter, is WebGPU enabled?");const e=await t.requestDevice(),o=await en.init(e);console.log("Voxel engine init complete"),postMessage({type:"init_complete"});const c=s=>{e.queue.onSubmittedWorkDone().then(s.callback),e.queue.submit(s.items)};onmessage=async function(s){const{detail:p,density:f}=s.data,r=31,{x:n,y:i,z:l,s:a}=p,u=a*r*.5,{vertices:v,normals:y,indices:d,consistency:A}=await o.generate(e,c,M(n*r-u,i*r-u,l*r-u),a,f);Fn.postMessage({type:"update",i:`${n}:${i}:${l}`,ix:n,iy:i,iz:l,x:0,y:0,z:0,vertices:v.buffer,normals:y.buffer,indices:d.buffer,stride:a,consistency:A},[v.buffer,y.buffer,d.buffer])}})()})();
