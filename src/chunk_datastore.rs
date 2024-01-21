use crate::util::TextureAndView;
use crate::wgpu_context::WgpuContext;
use nalgebra_glm as glm;
use std::mem::size_of;
use std::num::NonZeroU32;
use wgpu::*;

pub struct ChunkDatastore {
    chunks_per_group: u32,
    grid_groups: Vec<TextureAndView>,
    atlas: TextureAndView,
    bind_group_layout_rw: BindGroupLayout,
    bind_group_layout_ro: BindGroupLayout,
    bind_group_rw: BindGroup,
    bind_group_ro: BindGroup,
    dummy_views: Vec<TextureView>,
}

impl ChunkDatastore {
    fn new_dummy_texture(ctx: &WgpuContext) -> TextureView {
        let texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("chunk_datastore dummy_texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        texture.create_view(&TextureViewDescriptor {
            label: Some("chunk_datastore dummy_view"),
            ..Default::default()
        })
    }
    fn new_bind_group_from_grid_groups(
        ctx: &WgpuContext,
        atlas: &TextureAndView,
        grid_groups: &[TextureAndView],
        bind_group_layout: &BindGroupLayout,
        dummy_views: &[TextureView],
    ) -> BindGroup {
        let mut grid_views = grid_groups.iter().map(|v| &v.view).collect::<Vec<_>>();
        for dummy in dummy_views[grid_views.len()..8].iter() {
            grid_views.push(dummy);
        }
        ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("chunk_datastore bind_group"),
            layout: bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&atlas.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureViewArray(&grid_views),
                },
            ],
        })
    }

    fn new_grid_group(ctx: &WgpuContext, chunks_per_group: u32) -> TextureAndView {
        let texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("chunk_datastore grid_group_texture"),
            size: Extent3d {
                width: 64 * chunks_per_group,
                height: 64,
                depth_or_array_layers: 64 * 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("chunk_datastore grid_group_view"),
            ..Default::default()
        });
        TextureAndView { texture, view }
    }

    pub fn new(ctx: &WgpuContext, chunks_per_group: u32) -> Self {
        // Initialize with 1 chunk buffer
        let grid_groups = vec![Self::new_grid_group(ctx, chunks_per_group)];

        let [bind_group_layout_rw, bind_group_layout_ro]: [BindGroupLayout; 2] = (0..2)
            .map(|i| {
                ctx.device
                    .create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: Some("chunk_datastore bind_group_layout"),
                        entries: &[
                            BindGroupLayoutEntry {
                                binding: 0,
                                visibility: ShaderStages::COMPUTE,
                                ty: BindingType::StorageTexture {
                                    access: StorageTextureAccess::ReadOnly,
                                    format: TextureFormat::R32Uint,
                                    view_dimension: TextureViewDimension::D3,
                                },
                                count: None,
                            },
                            BindGroupLayoutEntry {
                                binding: 1,
                                visibility: ShaderStages::COMPUTE,
                                ty: BindingType::StorageTexture {
                                    access: [
                                        StorageTextureAccess::ReadWrite,
                                        StorageTextureAccess::ReadOnly,
                                    ][i],
                                    format: TextureFormat::R32Uint,
                                    view_dimension: TextureViewDimension::D3,
                                },
                                count: NonZeroU32::new(8),
                            },
                        ],
                    })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let atlas_texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("chunk_datastore atlas_texture"),
            size: Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 64,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&TextureViewDescriptor {
            label: Some("chunk_datastore atlas_view"),
            ..Default::default()
        });
        let atlas = TextureAndView {
            texture: atlas_texture,
            view: atlas_view,
        };

        let dummy_views = (0..8)
            .map(|_| Self::new_dummy_texture(ctx))
            .collect::<Vec<_>>();

        let bind_group_rw = Self::new_bind_group_from_grid_groups(
            ctx,
            &atlas,
            &grid_groups,
            &bind_group_layout_rw,
            &dummy_views,
        );

        let bind_group_ro = Self::new_bind_group_from_grid_groups(
            ctx,
            &atlas,
            &grid_groups,
            &bind_group_layout_ro,
            &dummy_views,
        );

        Self {
            chunks_per_group,
            grid_groups,
            atlas,
            bind_group_layout_rw,
            bind_group_layout_ro,
            bind_group_rw,
            bind_group_ro,
            dummy_views,
        }
    }

    fn offset_and_which_to_group_and_origin(
        &self,
        offset_and_which: (u32, u32),
    ) -> (u32, glm::UVec3) {
        if offset_and_which.1 >= 2 {
            panic!("which must be 0 or 1");
        }
        let group = offset_and_which.0 / self.chunks_per_group;
        let origin = glm::UVec3::new(
            (offset_and_which.0 % self.chunks_per_group) * 64,
            0,
            offset_and_which.1 * 64,
        );
        (group, origin)
    }

    pub fn upload_chunk_data(&self, ctx: &WgpuContext, offset_and_which: (u32, u32), data: &[u32]) {
        let (group, origin) = self.offset_and_which_to_group_and_origin(offset_and_which);
        ctx.queue.write_texture(
            ImageCopyTexture {
                texture: &self.grid_groups[group as usize].texture,
                mip_level: 0,
                origin: Origin3d {
                    x: origin.x,
                    y: origin.y,
                    z: origin.z,
                },
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(64 * size_of::<u32>() as u32),
                rows_per_image: Some(64),
            },
            Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 64,
            },
        );
    }

    pub fn copy(&self, encoder: &mut CommandEncoder, from: (u32, u32), to: (u32, u32)) {
        let (from_group, from_origin) = self.offset_and_which_to_group_and_origin(from);
        let (to_group, to_origin) = self.offset_and_which_to_group_and_origin(to);
        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &self.grid_groups[from_group as usize].texture,
                mip_level: 0,
                origin: Origin3d {
                    x: from_origin.x,
                    y: from_origin.y,
                    z: from_origin.z,
                },
                aspect: TextureAspect::All,
            },
            ImageCopyTexture {
                texture: &self.grid_groups[to_group as usize].texture,
                mip_level: 0,
                origin: Origin3d {
                    x: to_origin.x,
                    y: to_origin.y,
                    z: to_origin.z,
                },
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 64,
            },
        );
    }

    pub fn update_atlas(&self, ctx: &WgpuContext, pos: glm::IVec3, data: u32) {
        let pos = pos + glm::vec3(32, 32, 32);
        ctx.queue.write_texture(
            ImageCopyTexture {
                texture: &self.atlas.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: pos.x as u32,
                    y: pos.y as u32,
                    z: pos.z as u32,
                },
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&[data]),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(64 * size_of::<u32>() as u32),
                rows_per_image: Some(64),
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    // pub fn download(&self, _ctx: &WgpuContext, _data: &mut [u32; 64 * 64 * 64]) {
    //     todo!("implement Chunk::download");
    // }

    pub fn ensure_size(&mut self, ctx: &WgpuContext, size: u32) {
        let required_groups = size.div_ceil(self.chunks_per_group);
        if required_groups > self.grid_groups.len() as u32 {
            self.grid_groups.resize_with(required_groups as usize, || {
                Self::new_grid_group(ctx, self.chunks_per_group)
            });
            self.bind_group_rw = Self::new_bind_group_from_grid_groups(
                ctx,
                &self.atlas,
                &self.grid_groups,
                &self.bind_group_layout_rw,
                &self.dummy_views,
            );
            self.bind_group_ro = Self::new_bind_group_from_grid_groups(
                ctx,
                &self.atlas,
                &self.grid_groups,
                &self.bind_group_layout_ro,
                &self.dummy_views,
            );
        }
    }

    pub fn chunks_per_group(&self) -> u32 {
        self.chunks_per_group
    }

    pub fn bind_group_layout(&self, read_write: bool) -> &BindGroupLayout {
        if read_write {
            &self.bind_group_layout_rw
        } else {
            &self.bind_group_layout_ro
        }
    }

    pub fn bind_group(&self, read_write: bool) -> &BindGroup {
        if read_write {
            &self.bind_group_rw
        } else {
            &self.bind_group_ro
        }
    }
}
