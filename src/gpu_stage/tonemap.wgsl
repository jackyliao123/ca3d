struct VertexOut {
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    @size(64) linear_transform: mat4x4<f32>,
    @size(16) tonemapping_target_color_space: vec4<u32>,
    @size(16) output_scale: f32,
};

@group(0) @binding(0)
var linear_buffer_sampler: sampler;

@group(0) @binding(1)
var linear_buffer_texture: texture_2d<f32>;

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

var<private> v_positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
);

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOut {
    var out: VertexOut;

    out.position = vec4<f32>(v_positions[v_idx], 0.0, 1.0);

    return out;
}

fn aces_luminance(x: vec3<f32>) -> vec3<f32> {
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn aces_full(x: vec3<f32>) -> vec3<f32> {
    // Based on http://www.oscars.org/science-technology/sci-tech-projects/aces
	let m1 = mat3x3<f32>(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	let m2 = mat3x3<f32>(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);
	let v = m1 * x;
	let a = v * (v + 0.0245786) - 0.000090537;
	let b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return clamp(m2 * (a / b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(x: vec3<f32>) -> vec3<f32> {
    return pow(clamp(x, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tonemapping = uniforms.tonemapping_target_color_space.x;
    let target_color_space = uniforms.tonemapping_target_color_space.y;
    let output_scale = uniforms.output_scale;

    var color: vec3<f32> = textureSample(
        linear_buffer_texture,
        linear_buffer_sampler,
        in.position.xy / vec2<f32>(textureDimensions(linear_buffer_texture))
    ).xyz;

    color = (vec4<f32>(color, 1.0) * uniforms.linear_transform).xyz;
    if(tonemapping == 1u) {
        color = aces_luminance(color.xyz);
    } else if(tonemapping == 2u) {
        color = aces_full(color.xyz);
    }

    if(target_color_space == 1u) {
        color = linear_to_srgb(color);
    }
    color *= output_scale;
    return vec4<f32>(color, 1.0);
}