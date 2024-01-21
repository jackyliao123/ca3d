use std::collections::{HashMap, HashSet};

use nalgebra_glm as glm;

use crate::chunk::{Chunk, ResidencyOffset};
use crate::chunk_datastore::ChunkDatastore;
use crate::wgpu_context::WgpuContext;

#[derive(Default)]
struct SharedBufferOffsetTracker {
    index_to_offset: HashMap<u64, u32>,
    offset_to_index: Vec<u64>,
    max_index: u64,
}

impl SharedBufferOffsetTracker {
    fn new() -> Self {
        Default::default()
    }

    fn add_and_get_index(&mut self) -> u64 {
        let index = self.max_index;
        let offset = self.offset_to_index.len() as u32;
        self.max_index += 1;
        self.index_to_offset.insert(index, offset);
        self.offset_to_index.push(index);
        index
    }

    fn remove_index(&mut self, index: u64) {
        let removed_offset = self.index_to_offset.remove(&index).unwrap();
        if removed_offset == self.offset_to_index.len() as u32 - 1 {
            self.offset_to_index.pop();
        } else {
            self.offset_to_index[removed_offset as usize] = self.offset_to_index.pop().unwrap();
            self.index_to_offset.insert(
                self.offset_to_index[removed_offset as usize],
                removed_offset,
            );
        }
    }

    fn get_offset(&self, index: u64) -> u32 {
        self.index_to_offset
            .get(&index)
            .cloned()
            .unwrap_or_else(|| panic!("index {} not found", index))
    }
}

pub struct ChunkManager {
    chunks: HashMap<glm::IVec3, Chunk>,
    shared_buffer_offset_tracker: SharedBufferOffsetTracker,
    atlas_updates: HashSet<glm::IVec3>,
    datastore: ChunkDatastore,
    modified_this_frame: bool,
    which: u32,
}
impl ChunkManager {
    pub fn new(ctx: &WgpuContext) -> Self {
        Self {
            chunks: HashMap::new(),
            shared_buffer_offset_tracker: SharedBufferOffsetTracker::new(),
            atlas_updates: HashSet::new(),
            datastore: ChunkDatastore::new(ctx, 32),
            modified_this_frame: false,
            which: 0,
        }
    }

    pub fn add_chunk(&mut self, mut chunk: Chunk) {
        if self.chunks.contains_key(&chunk.pos) {
            panic!("chunk {:?} already exists", chunk.pos);
        }
        self.modified_this_frame = true;
        let mut neighbors = 0u32;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let neighbor_pos = &chunk.pos + glm::vec3(dx, dy, dz);
                    let neighbor = self.chunks.get_mut(&neighbor_pos);
                    if let Some(neighbor) = neighbor {
                        neighbor.neighbors += 1;
                        neighbors += 1;
                    }
                }
            }
        }
        self.atlas_updates.insert(chunk.pos);
        chunk.neighbors = neighbors;
        self.chunks.insert(chunk.pos, chunk);
    }

    pub fn remove_chunk(&mut self, pos: &glm::IVec3) -> Chunk {
        self.modified_this_frame = true;
        let mut chunk = self
            .chunks
            .remove(pos)
            .unwrap_or_else(|| panic!("chunk {:?} not found", pos));
        self.shared_buffer_offset_tracker.remove_index(
            chunk
                .residency
                .as_ref()
                .unwrap_or_else(|| panic!("chunk {:?} offset not tracked", pos))
                .index,
        );
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let neighbor_pos = pos + glm::vec3(dx, dy, dz);
                    let neighbor = self.chunks.get_mut(&neighbor_pos);
                    if let Some(neighbor) = neighbor {
                        neighbor.neighbors -= 1;
                    }
                }
            }
        }
        self.atlas_updates.insert(*pos);
        chunk.neighbors = 0;
        chunk
    }

    pub fn chunks(&self) -> &HashMap<glm::IVec3, Chunk> {
        &self.chunks
    }

    pub fn chunks_mut(&mut self) -> &mut HashMap<glm::IVec3, Chunk> {
        &mut self.chunks
    }

    pub fn num_offsets(&self) -> u32 {
        if self.modified_this_frame {
            panic!("total_offsets called before finalize_changes_and_start_frame");
        }
        self.shared_buffer_offset_tracker.offset_to_index.len() as u32
    }

    pub fn upload_chunk_data(&self, ctx: &WgpuContext, pos: glm::IVec3, data: &[u32]) {
        if self.modified_this_frame {
            panic!("upload_chunk_data called before finalize_changes_and_start_frame");
        }
        let chunk = self
            .chunks
            .get(&pos)
            .unwrap_or_else(|| panic!("chunk {:?} not found", pos));
        self.datastore
            .upload_chunk_data(ctx, (chunk.offset(), self.which), data);
    }

    pub fn finalize_changes_and_start_frame(&mut self, ctx: &WgpuContext) {
        if !self.modified_this_frame {
            return;
        }

        // Process the copies incurred by chunk removals first
        let mut copies = Vec::new();
        for chunk in self.chunks.values_mut() {
            if let Some(residency) = &mut chunk.residency {
                let offset = self
                    .shared_buffer_offset_tracker
                    .get_offset(residency.index);
                copies.push((0, offset));
                residency.offset = offset;
                self.atlas_updates.insert(chunk.pos);
            }
        }

        if !copies.is_empty() {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("chunk_manager finalize_changes_and_start_frame"),
                });
            for (old_offset, offset) in copies {
                self.datastore
                    .copy(&mut encoder, (old_offset, self.which), (offset, self.which));
            }
            ctx.queue.submit([encoder.finish()]);
        }

        for chunk in self.chunks.values_mut() {
            if chunk.residency.is_none() {
                let index = self.shared_buffer_offset_tracker.add_and_get_index();
                let offset = self.shared_buffer_offset_tracker.get_offset(index);
                chunk.residency = Some(ResidencyOffset::new(index, offset));
            }
        }

        self.datastore.ensure_size(
            ctx,
            self.shared_buffer_offset_tracker.offset_to_index.len() as u32,
        );

        for pos in self.atlas_updates.drain() {
            match self.chunks.get(&pos) {
                Some(chunk) => {
                    self.datastore.update_atlas(ctx, pos, chunk.offset() + 1);
                }
                None => {
                    self.datastore.update_atlas(ctx, pos, 0);
                }
            }
        }

        self.modified_this_frame = false;
    }

    pub fn offset_to_group_and_origin_x(&self, offset: u32) -> (u32, u32) {
        (
            offset / self.datastore.chunks_per_group(),
            offset % self.datastore.chunks_per_group(),
        )
    }

    pub fn bind_group_layout(&self, read_write: bool) -> &wgpu::BindGroupLayout {
        self.datastore.bind_group_layout(read_write)
    }

    pub fn bind_group(&self, read_write: bool) -> &wgpu::BindGroup {
        self.datastore.bind_group(read_write)
    }

    pub fn chunks_per_group(&self) -> u32 {
        self.datastore.chunks_per_group()
    }

    pub fn which(&self) -> u32 {
        self.which
    }

    pub fn advance_which(&mut self, amount: u32) {
        self.which = (self.which + amount) % 2;
    }
}
