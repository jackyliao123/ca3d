struct FaceInstance {
    @location(0) color: u32,
    @location(1) info: u32,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) ao: u32,
};

struct PushConstants {
    @size(64) view_proj: mat4x4<f32>,
    @size(16) translate: vec3<f32>,
};

var<push_constant> consts: PushConstants;

var<private> which_vertex: array<u32, 6> = array<u32, 6>(
    0u, 1u, 2u, 2u, 1u, 3u
);

var<private> normal: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(-1.0,  0.0,  0.0),
    vec3<f32>( 1.0,  0.0,  0.0),
    vec3<f32>( 0.0, -1.0,  0.0),
    vec3<f32>( 0.0,  1.0,  0.0),
    vec3<f32>( 0.0,  0.0, -1.0),
    vec3<f32>( 0.0,  0.0,  1.0),
);

var<private> pos: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>( 0.0,  0.0,  0.0),
    vec3<f32>( 0.0,  0.0,  1.0),
    vec3<f32>( 0.0,  1.0,  0.0),
    vec3<f32>( 0.0,  1.0,  1.0),
    vec3<f32>( 1.0,  0.0,  0.0),
    vec3<f32>( 1.0,  0.0,  1.0),
    vec3<f32>( 1.0,  1.0,  0.0),
    vec3<f32>( 1.0,  1.0,  1.0),
);

var<private> indices: array<u32, 24> = array<u32, 24>(
    0u, 1u, 2u, 3u,
    5u, 4u, 7u, 6u,
    1u, 0u, 5u, 4u,
    2u, 3u, 6u, 7u,
    0u, 2u, 4u, 6u,
    3u, 1u, 7u, 5u,
);

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, face: FaceInstance) -> VertexOut {
    let info = face.info;
    let offset = vec3<u32>(info & 0x3Fu, (info >> 6u) & 0x3Fu, (info >> 12u) & 0x3Fu);
    let side = (info >> 18u) & 0x7u;

    let back_side = (side & 1u);

    let which = which_vertex[v_idx];
    let ao = (info >> (21u + which * 2u)) & 0x3u;
    let world_pos = vec3<f32>(offset) + pos[indices[side * 4u + which]] + consts.translate;
    let world_normal = normal[side];
    let color = unpack4x8unorm(face.color);

    var out: VertexOut;

    out.position = consts.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = world_normal;
    out.color = color;
    out.ao = ao;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let emission = 1.0; // step(in.color.a, 0.05) * 25.0
    return vec4<f32>(in.color.rgb * (dot(in.world_normal, vec3<f32>(0.8, 1.0, 0.2)) * 0.25 + 0.75) * (1.0 + emission), 1.0);
}