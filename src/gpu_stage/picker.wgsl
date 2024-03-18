struct PushConstants {
    @size(64) view_proj: mat4x4<f32>,
    @size(16) translate: vec3<f32>,
};

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> data: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(texture, 0);
    if(gid.x >= dimensions.x || gid.y >= dimensions.y) {
        return;
    }
    data[dimensions.x * gid.y + gid.x] = textureLoad(texture, gid.xy, 0);
}