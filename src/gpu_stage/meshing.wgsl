struct DrawIndirect {
    @size(4) vertex_count: u32,
    @size(4) instance_count: atomic<u32>,
    @size(4) first_vertex: u32,
    @size(4) first_instance: u32,
};

struct FaceInstance {
    @size(4) color: u32,
    @size(4) info: u32,
}

struct PushConstants {
    @size(4) max_faces: u32,
    @size(4) group: u32,
    @size(4) origin_x: u32,
    @size(4) which: u32,
};

var<push_constant> consts: PushConstants;

@group(0) @binding(0)
var<storage, read_write> indirect: DrawIndirect;

@group(0) @binding(1)
var<storage, read_write> faces: array<FaceInstance>;

@group(1) @binding(0)
var atlas: texture_storage_3d<r32uint, read>;

@group(1) @binding(1)
var chunk_groups: binding_array<texture_storage_3d<r32uint, read>, 8>;

fn load(pos: vec3<i32>) -> u32 {
    if(any(pos >= vec3<i32>(64, 64, 64))) {
        return 0u;
    }
    if(any(pos < vec3<i32>(0, 0, 0))) {
        return 0u;
    }
    return textureLoad(chunk_groups[consts.group], pos + vec3<i32>(vec3<u32>(consts.origin_x, 0u, consts.which)) * 64).r;
}

fn append_face(color: u32, side: u32, pos: vec3<i32>) {
    let index = atomicAdd(&indirect.instance_count, 1u);
    if(index > consts.max_faces) {
        atomicStore(&indirect.instance_count, consts.max_faces);
        return;
    } else if(index == consts.max_faces) {
        return;
    }
    faces[index].color = color;
    faces[index].info = u32((pos.x << 0u) | (pos.y << 6u) | (pos.z << 12u)) | (side << 18u);
}

@compute
@workgroup_size(4, 4, 4)
fn cs_generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = vec3<i32>(gid);
    let cur = load(pos);
    if(cur == 0u) {
        return;
    }
    if(load(pos + vec3<i32>(-1, 0, 0)) == 0u) {
        append_face(cur, 0u, pos);
    }
    if(load(pos + vec3<i32>(1, 0, 0)) == 0u) {
        append_face(cur, 1u, pos);
    }
    if(load(pos + vec3<i32>(0, -1, 0)) == 0u) {
        append_face(cur, 2u, pos);
    }
    if(load(pos + vec3<i32>(0, 1, 0)) == 0u) {
        append_face(cur, 3u, pos);
    }
    if(load(pos + vec3<i32>(0, 0, -1)) == 0u) {
        append_face(cur, 4u, pos);
    }
    if(load(pos + vec3<i32>(0, 0, 1)) == 0u) {
        append_face(cur, 5u, pos);
    }
}