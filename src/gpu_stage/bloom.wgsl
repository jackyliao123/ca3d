struct Uniforms {
    @size(16) scale_fact: vec2<f32>,
};

@group(0) @binding(0)
var input_sampler: sampler;

@group(1) @binding(0)
var input: texture_2d<f32>;

@group(1) @binding(1)
var output_store: texture_storage_2d<rgba16float, write>;

@group(1) @binding(2)
var<uniform> upscale_uniform: Uniforms;

@group(1) @binding(3)
var upscale_output_load: texture_2d<f32>;

// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
@compute
@workgroup_size(8, 8)
fn cs_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let write_pos = gid.xy;
    if(any(write_pos >= textureDimensions(output_store))) {
        return;
    }

    let read_center = (vec2<f32>(write_pos) * 2.0 + 1.0) / vec2<f32>(textureDimensions(input));
    var accum = vec4<f32>(0.0);

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-2, -2)) / 32.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-2,  0)) / 16.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-2,  2)) / 32.0;

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-1, -1)) / 8.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-1,  1)) / 8.0;

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 0, -2)) / 16.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 0,  0)) / 8.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 0,  2)) / 16.0;

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 1, -1)) / 8.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 1,  1)) / 8.0;

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 2, -2)) / 32.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 2,  0)) / 16.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 2,  2)) / 32.0;

    textureStore(output_store, write_pos, accum);
}

@compute
@workgroup_size(8, 8)
fn cs_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let write_pos = gid.xy;
    if(any(write_pos >= textureDimensions(output_store))) {
        return;
    }
    let scale_fact = upscale_uniform.scale_fact;

    let read_center = (vec2<f32>(write_pos) + 0.5) / 2.0 / vec2<f32>(textureDimensions(input));
    var accum = vec4<f32>(0.0);

    let load = textureLoad(upscale_output_load, write_pos, 0);

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-1, -1)) / 16.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-1,  0)) / 8.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>(-1,  1)) / 16.0;

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 0, -1)) / 8.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 0,  0)) / 4.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 0,  1)) / 8.0;

    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 1, -1)) / 16.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 1,  0)) / 8.0;
    accum += textureSampleLevel(input, input_sampler, read_center, 0.0, vec2<i32>( 1,  1)) / 16.0;

    textureStore(output_store, write_pos, load * scale_fact.x + accum * scale_fact.y);
}