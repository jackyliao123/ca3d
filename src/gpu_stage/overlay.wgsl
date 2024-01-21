struct InstanceInput {
    @location(1) color: vec4<f32>,
    @location(2) offset1: vec4<f32>,
    @location(3) offset2: vec4<f32>,
}

struct PushConstants {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
};

var<push_constant> consts: PushConstants;

struct VertexOutput {
    @location(0) vert_color: vec4<f32>,
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_wireframe(@location(0) pos: vec4<f32>, in: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let screen_offset1 = (consts.view * vec4<f32>(in.offset1.xyz, 1.0)).xyz;
    let screen_offset2 = (consts.view * vec4<f32>(in.offset2.xyz, 1.0)).xyz;

    var t: f32 = pos.w;

    if (t > 0.5 && screen_offset2.z < 0.0 || t < 0.5 && screen_offset1.z < 0.0) {
        t = -screen_offset1.z / (screen_offset2.z - screen_offset1.z);
    }

    let depth = mix(screen_offset1, screen_offset2, t).z;

    let offset_scale = mix(in.offset1, in.offset2, t);
    let offset = offset_scale.xyz;
    let scale = offset_scale.w;

    var dir: vec3<f32> = in.offset2.xyz - in.offset1.xyz;
    if (length(dir) < 0.0001) {
        dir = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        dir = normalize(dir);
    }

    let tangent = normalize(cross(dir, vec3<f32>(1.0, 2.35, -3.1415))); // Some random values to make it not parallel to dir
    let bitangent = normalize(cross(dir, tangent));

    let model = mat4x4<f32>(
        vec4<f32>(scale * tangent, 0.0),
        vec4<f32>(scale * bitangent, 0.0),
        vec4<f32>(scale * dir, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    out.vert_color = in.color;
    out.pos = consts.proj * consts.view * vec4<f32>(((model * vec4<f32>(pos.xyz * depth / 500.0, 1.0)).xyz + offset).xyz, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.vert_color;
}