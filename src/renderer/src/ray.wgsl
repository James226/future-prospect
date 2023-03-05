#import density

struct Uniforms {
  position: vec3<f32>,
  direction: vec3<f32>
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct Intersection {
  position: vec3<f32>,
  found: u32,
  normal: vec3<f32>,
  distance: f32
}
@binding(1) @group(0) var <storage, read_write> intersection: Intersection;


@compute @workgroup_size(1)
fn main() {

  let direction = normalize(uniforms.direction);
  var i = 1.0;

  intersection.found = 0;

  loop {
    let position = uniforms.position + direction * i;
    let density = getDensity(position);

    if (i > 10000) {
      return;
    }

    if (density < 0) {
      intersection.found = 1;
      intersection.position = position;
      intersection.normal = vec3<f32>(3.0, 5.0, 1.0);
      intersection.distance = length(uniforms.position - position);
      return;
    }

    continuing {
      i = i + 1;
    }
  }
}
