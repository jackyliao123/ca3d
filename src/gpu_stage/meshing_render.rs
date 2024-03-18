use std::collections::HashMap;
use std::mem::size_of;
use std::rc::Rc;

use bytemuck::{offset_of, Pod, Zeroable};
use nalgebra_glm as glm;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

use crate::chunk_manager::ChunkManager;
use crate::util::*;
use crate::wgpu_context::WgpuContext;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct MeshingPushConstants {
    max_faces: u32,
    group: u32,
    origin_x: u32,
    which: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct FaceInstance {
    color: u32,
    info: u32,
}

pub struct PerChunkResource {
    indirect_buffer: Buffer,
    instance_buffer: Buffer,
    bind_group: BindGroup,
}

impl PerChunkResource {
    fn new(ctx: &WgpuContext, bind_group_layout: &BindGroupLayout) -> Self {
        let indirect_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("meshing per_chunk indirect_buffer"),
            size: size_of::<DrawIndirectPod>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("meshing per_chunk instance_buffer"),
            size: 64 * 64 * 64 * size_of::<FaceInstance>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("meshing per_chunk bind_group"),
            layout: bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: indirect_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            indirect_buffer,
            instance_buffer,
            bind_group,
        }
    }
}

struct MeshingResources {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    indirect_buffer_init: Buffer,
    per_chunk_resources: HashMap<glm::IVec3, PerChunkResource>,
}

pub struct Meshing {
    res: MeshingResources,
}

impl MeshingResources {
    fn new(ctx: &WgpuContext, chunk_manager: &ChunkManager) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("meshing shader"),
            source: ShaderSource::Wgsl(include_str!("./meshing.wgsl").into()),
        });

        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("meshing bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("meshing pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout, chunk_manager.bind_group_layout(false)],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..size_of::<MeshingPushConstants>() as u32,
                }],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("meshing generate_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_generate",
            });

        let indirect_buffer_init = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("meshing indirect_buffer_init"),
            contents: bytemuck::cast_slice(&[DrawIndirectPod {
                vertex_count: 6,
                instance_count: 0,
                base_vertex: 0,
                base_instance: 0,
            }]),
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        });

        Self {
            bind_group_layout,
            pipeline,
            indirect_buffer_init,
            per_chunk_resources: HashMap::new(),
        }
    }
}

impl Meshing {
    pub fn new(ctx: &WgpuContext, chunk_manager: &ChunkManager) -> Self {
        let res = MeshingResources::new(ctx, chunk_manager);
        Self { res }
    }

    pub fn update(
        &mut self,
        ctx: &WgpuContext,
        command_encoder: &mut CommandEncoder,
        chunk_manager: &ChunkManager,
    ) -> &HashMap<glm::IVec3, PerChunkResource> {
        self.res
            .per_chunk_resources
            .retain(|chunk, _| chunk_manager.chunks().contains_key(chunk));

        for chunk in chunk_manager.chunks().values() {
            self.res
                .per_chunk_resources
                .entry(chunk.pos)
                .or_insert_with(|| PerChunkResource::new(ctx, &self.res.bind_group_layout));

            command_encoder.copy_buffer_to_buffer(
                &self.res.indirect_buffer_init,
                0,
                &self.res.per_chunk_resources[&chunk.pos].indirect_buffer,
                0,
                size_of::<DrawIndirectPod>() as u64,
            );
        }

        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("meshing compute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.res.pipeline);
            for chunk in chunk_manager.chunks().values() {
                let per_chunk_resource = &self.res.per_chunk_resources[&chunk.pos];

                let (group, origin_x) = chunk_manager.offset_to_group_and_origin_x(chunk.offset());

                compute_pass.set_push_constants(
                    0,
                    bytemuck::cast_slice(&[MeshingPushConstants {
                        max_faces: self.res.per_chunk_resources[&chunk.pos]
                            .instance_buffer
                            .size() as u32
                            / size_of::<FaceInstance>() as u32,
                        group,
                        origin_x,
                        which: chunk_manager.which(),
                    }]),
                );
                compute_pass.set_bind_group(0, &per_chunk_resource.bind_group, &[]);
                compute_pass.set_bind_group(1, chunk_manager.bind_group(false), &[]);
                compute_pass.dispatch_workgroups(
                    64u32.div_ceil(4),
                    64u32.div_ceil(4),
                    64u32.div_ceil(4),
                );
            }
        }

        &self.res.per_chunk_resources
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct RenderPushConstants {
    view_proj: glm::Mat4x4,
    translate: glm::Vec3,
}

struct RenderResources {
    shader: ShaderModule,
    pipeline_layout: PipelineLayout,
}

struct RenderDynamicResources {
    output_target: Rc<RenderTarget>,
    pipeline: RenderPipeline,
}

pub struct Render {
    res: RenderResources,
    dynamic: RenderDynamicResources,
}

impl RenderResources {
    fn new(ctx: &WgpuContext) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("render shader"),
            source: ShaderSource::Wgsl(include_str!("./render.wgsl").into()),
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("render pipeline_layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::VERTEX,
                    range: 0..size_of::<RenderPushConstants>() as u32,
                }],
            });

        Self {
            shader,
            pipeline_layout,
        }
    }
}

impl RenderDynamicResources {
    fn new(ctx: &WgpuContext, res: &mut RenderResources, output_target: Rc<RenderTarget>) -> Self {
        let pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("render pipeline"),
                layout: Some(&res.pipeline_layout),
                vertex: VertexState {
                    module: &res.shader,
                    entry_point: "vs_main",
                    buffers: &[VertexBufferLayout {
                        array_stride: size_of::<FaceInstance>() as u64,
                        step_mode: VertexStepMode::Instance,
                        attributes: &[
                            VertexAttribute {
                                format: VertexFormat::Uint32,
                                offset: offset_of!(FaceInstance, color) as u64,
                                shader_location: 0,
                            },
                            VertexAttribute {
                                format: VertexFormat::Uint32,
                                offset: offset_of!(FaceInstance, info) as u64,
                                shader_location: 1,
                            },
                        ],
                    }],
                },
                fragment: Some(FragmentState {
                    module: &res.shader,
                    entry_point: "fs_main",
                    targets: &[Some(output_target.info.format.into())],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(Face::Back),
                    unclipped_depth: true,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Greater,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: MultisampleState::default(),
                multiview: None,
            });

        Self {
            output_target,
            pipeline,
        }
    }
}

impl Render {
    pub fn new(ctx: &WgpuContext, output_target: Rc<RenderTarget>) -> Self {
        let mut res = RenderResources::new(ctx);
        let dynamic = RenderDynamicResources::new(ctx, &mut res, output_target);
        Self { res, dynamic }
    }
    pub fn resize(&mut self, ctx: &WgpuContext, output_target: Rc<RenderTarget>) {
        self.dynamic = RenderDynamicResources::new(ctx, &mut self.res, output_target);
    }

    pub fn update(
        &mut self,
        _ctx: &WgpuContext,
        command_encoder: &mut CommandEncoder,
        chunk_manager: &ChunkManager,
        per_chunk_resource: &HashMap<glm::IVec3, PerChunkResource>,
        view_proj: &glm::Mat4x4,
    ) {
        {
            let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render render_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.dynamic.output_target.render_target,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: self
                        .dynamic
                        .output_target
                        .depth_target
                        .as_ref()
                        .expect("no depth target"),
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.dynamic.pipeline);

            for (pos, chunk) in chunk_manager.chunks() {
                let per_chunk_resource = &per_chunk_resource[pos];

                render_pass.set_push_constants(
                    ShaderStages::VERTEX,
                    0,
                    bytemuck::cast_slice(&[RenderPushConstants {
                        view_proj: *view_proj,
                        translate: chunk.pos.cast::<f32>() * 64.0,
                    }]),
                );

                render_pass.set_vertex_buffer(0, per_chunk_resource.instance_buffer.slice(..));
                render_pass.draw_indirect(&per_chunk_resource.indirect_buffer, 0);
            }
        }
    }
}
