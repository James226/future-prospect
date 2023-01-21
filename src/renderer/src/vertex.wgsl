struct Uniforms {
  modelViewProjectionMatrix : mat4x4<f32>,
  stride: i32,
  position: vec3<f32>
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct Lights
{
  position: vec3<f32>,
  diffuse: vec3<f32>
};

fn lambert(N: vec3<f32>, L: vec3<f32>) -> f32
{
  var nrmN = normalize(N);
  var nrmL = normalize(L);
  var result = dot(nrmN, nrmL);
  return max(result, 0.0);
}

struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) fragPosition: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
};

@vertex
fn main(@location(0) position : vec4<f32>,
        @location(1) normal : vec4<f32>) -> VertexOutput {

  var light: Lights;
  light.position = normalize(-position.xyz);
  light.diffuse = vec3<f32>(1.0, 1.0, 1.0);

  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix * position;
  output.normal = normal.xyz;

  var result = vec3<f32>(0.4, 0.4, 0.4) + light.diffuse * lambert(normal.xyz, light.position);

  output.fragPosition = position;
  output.color = result;
  return output;
}
