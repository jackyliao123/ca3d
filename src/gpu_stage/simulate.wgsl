struct PushConstants {
    @size(4) rng: u32,
    @size(4) chunks_per_buffer_shift: u32,
    @size(4) starting_which: u32,
    @size(4) num_chunks: u32,
}

struct ChunkInfoEntry {
    @size(16) chunk_pos: vec3<i32>,
}

var<push_constant> consts: PushConstants;

@group(0) @binding(0)
var<storage, read_write> chunks: array<ChunkInfoEntry>;

@group(1) @binding(0)
var atlas: texture_storage_3d<r32uint, read>;

@group(1) @binding(1)
var grids: binding_array<texture_storage_3d<r32uint, read_write>, 8>;

fn hash(in: u32) -> u32 {
    var x = in;
    x += x << 10u;
    x ^= x >>  6u;
    x += x <<  3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

var<private> dirs: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
    vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 0, 0),
    vec3<i32>(0, 1, 0),
    vec3<i32>(0, -1, 0),
    vec3<i32>(0, 0, 1),
    vec3<i32>(0, 0, -1)
);

struct Shared {
    loaded: array<u32, 1000>,
    neighbor: array<u32, 27>,
}

var<workgroup> workgroup_shared: Shared;

@compute
@workgroup_size(8, 8, 8)
fn cs_simulate(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    ) {
    let wg = (wid.z * num_wg.y + wid.y) * num_wg.x + wid.x;
    let chunk_idx = wg / 512u;
    if(chunk_idx >= consts.num_chunks) {
        return;
    }
    let current_chunk = chunks[chunk_idx];
    let current_wg = wg % 512;
    let wg_pos = vec3<u32>(current_wg % 8, (current_wg / 8) % 8, current_wg / 64) * 8u;

    if(all(lid <= vec3<u32>(2u))) {
        workgroup_shared.neighbor[dot(vec3<u32>(1u, 3u, 9u), lid)] =
            textureLoad(atlas, current_chunk.chunk_pos + vec3<i32>(lid) - vec3<i32>(1) + vec3<i32>(32)).r;
    }

    workgroupBarrier();

    if(lidx < 500u) {
        for(var i = 0u; i < 2u; i += 1u) {
            let pos = vec3<i32>(vec3<u32>((lidx % 5u) * 2u + i, (lidx / 5u) % 10u, lidx / 50u) + wg_pos) - vec3<i32>(1, 1, 1);
            let neighbor = workgroup_shared.neighbor[
                dot(vec3<i32>(1, 3, 9), extractBits(pos, 6u, 26u) + vec3<i32>(1, 1, 1))
            ];
            var loaded = 0u;
            if(neighbor != 0u) {
                let chunk_idx = neighbor - 1u;
                let buffer_idx = chunk_idx >> consts.chunks_per_buffer_shift;
                let offset_x = chunk_idx & ((1u << consts.chunks_per_buffer_shift) - 1u);
                loaded = textureLoad(grids[buffer_idx], vec3<u32>(pos & vec3(63)) + vec3<u32>(offset_x, 0u, consts.starting_which) * 64u).r;
            }
            workgroup_shared.loaded[lidx * 2 + i] = loaded;
        }
    }

    workgroupBarrier();

    let rng = hash(consts.rng + chunk_idx * 262144u + dot(wg_pos + lid, vec3<u32>(1u, 64u, 4096u)));
    var cur = workgroup_shared.loaded[dot(lid + vec3<u32>(1), vec3<u32>(1u, 10u, 100u))];

    for(var i = 0u; i < 6u; i += 1u) {
        let neighbor = workgroup_shared.loaded[dot(vec3<u32>(vec3<i32>(lid) + vec3<i32>(1) + dirs[i]), vec3<u32>(1u, 10u, 100u))];
        if(neighbor != 0u) {
            cur = max(cur, neighbor);
            if (f32(rng) / 4294967295.0 < 0.01) {
                cur = hash(rng);
            }
        }
    }

    let buffer_idx = chunk_idx >> consts.chunks_per_buffer_shift;
    let offset_x = chunk_idx & ((1u << consts.chunks_per_buffer_shift) - 1u);
    textureStore(grids[buffer_idx], wg_pos + lid + vec3<u32>(offset_x, 0u, consts.starting_which ^ 1u) * 64u, vec4<u32>(cur, 0u, 0u, 0u));
}
