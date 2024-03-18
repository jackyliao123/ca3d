use crate::resource_size_helper::ResourceSizeHelper;
use crate::util::{RenderTarget, RenderTargetInfo};
use crate::wgpu_context::WgpuContext;
use bytemuck::{offset_of, Pod, Zeroable};
use nalgebra_glm as glm;
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

const CYLINDER_SEGMENTS: u32 = 60;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct PushConstants {
    proj: glm::Mat4x4,
    view: glm::Mat4x4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct WireframeInstanceInput {
    color: glm::Vec4,
    offset1: glm::Vec4,
    offset2: glm::Vec4,
}

struct Resources {
    shader: ShaderModule,
    depth_desc: TextureDescriptor<'static>,
    pipeline_layout: PipelineLayout,
    cylinder_vertex_buffer: Buffer,
    sphere_vertex_buffer: Buffer,
    cylinder_instance_buffer: ResourceSizeHelper<Buffer>,
    sphere_instance_buffer: ResourceSizeHelper<Buffer>,
}

struct DynamicResources {
    output_target: Rc<RenderTarget>,
    depth_view: Rc<TextureView>,
    pipeline: RenderPipeline,
}

pub struct Overlay {
    res: Resources,
    dynamic: DynamicResources,
    cylinder_instances: RefCell<Vec<WireframeInstanceInput>>,
    sphere_instances: RefCell<Vec<WireframeInstanceInput>>,
}

impl Resources {
    fn new(ctx: &WgpuContext) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("overlay shader"),
            source: ShaderSource::Wgsl(include_str!("./overlay.wgsl").into()),
        });

        let depth_desc = TextureDescriptor {
            label: Some("overlay depth_desc"),
            size: Extent3d {
                width: ctx.surface_config.width,
                height: ctx.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("overlay pipeline_layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::VERTEX,
                    range: 0..size_of::<PushConstants>() as u32,
                }],
            });

        let mut cylinder_vertices: Vec<glm::Vec4> = vec![];
        for i in 0..CYLINDER_SEGMENTS {
            let angle1 = (i as f32 / CYLINDER_SEGMENTS as f32) * 2.0 * std::f32::consts::PI;
            let angle2 = ((i + 1) as f32 / CYLINDER_SEGMENTS as f32) * 2.0 * std::f32::consts::PI;
            let loc1 = glm::vec2(angle1.cos(), angle1.sin());
            let loc2 = glm::vec2(angle2.cos(), angle2.sin());
            cylinder_vertices.push(glm::vec4(loc1.x, loc1.y, 0.0, 0.0));
            cylinder_vertices.push(glm::vec4(loc2.x, loc2.y, 0.0, 0.0));
            cylinder_vertices.push(glm::vec4(loc1.x, loc1.y, 0.0, 1.0));
            cylinder_vertices.push(glm::vec4(loc2.x, loc2.y, 0.0, 0.0));
            cylinder_vertices.push(glm::vec4(loc2.x, loc2.y, 0.0, 1.0));
            cylinder_vertices.push(glm::vec4(loc1.x, loc1.y, 0.0, 1.0));
        }

        let cylinder_vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("overlay cylinder_vertex_buffer"),
            contents: bytemuck::cast_slice(&cylinder_vertices),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        let sphere_vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("overlay sphere_vertex_buffer"),
            contents: &[],
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        Self {
            shader,
            depth_desc,
            pipeline_layout,
            cylinder_vertex_buffer,
            sphere_vertex_buffer,
            cylinder_instance_buffer: Default::default(),
            sphere_instance_buffer: Default::default(),
        }
    }
}

impl DynamicResources {
    fn new(ctx: &WgpuContext, res: &mut Resources, output_target: Rc<RenderTarget>) -> Self {
        res.depth_desc.size.width = output_target.info.width;
        res.depth_desc.size.height = output_target.info.height;
        let depth_texture = ctx.device.create_texture(&res.depth_desc);
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());
        let wireframe_pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("overlay pipeline"),
                layout: Some(&res.pipeline_layout),
                vertex: VertexState {
                    module: &res.shader,
                    entry_point: "vs_wireframe",
                    buffers: &[
                        VertexBufferLayout {
                            array_stride: size_of::<glm::Vec4>() as u64,
                            step_mode: VertexStepMode::Vertex,
                            attributes: &[VertexAttribute {
                                format: VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 0,
                            }],
                        },
                        VertexBufferLayout {
                            array_stride: size_of::<WireframeInstanceInput>() as u64,
                            step_mode: VertexStepMode::Instance,
                            attributes: &[
                                VertexAttribute {
                                    format: VertexFormat::Float32x4,
                                    offset: offset_of!(WireframeInstanceInput, color) as u64,
                                    shader_location: 1,
                                },
                                VertexAttribute {
                                    format: VertexFormat::Float32x4,
                                    offset: offset_of!(WireframeInstanceInput, offset1) as u64,
                                    shader_location: 2,
                                },
                                VertexAttribute {
                                    format: VertexFormat::Float32x4,
                                    offset: offset_of!(WireframeInstanceInput, offset2) as u64,
                                    shader_location: 3,
                                },
                            ],
                        },
                    ],
                },
                fragment: Some(FragmentState {
                    module: &res.shader,
                    entry_point: "fs_main",
                    targets: &[Some(ColorTargetState {
                        format: output_target.info.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
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
            depth_view: Rc::new(depth_view),
            pipeline: wireframe_pipeline,
        }
    }
}

impl Overlay {
    pub fn new(ctx: &WgpuContext, output_target: Rc<RenderTarget>) -> Self {
        let mut res = Resources::new(ctx);
        let dynamic = DynamicResources::new(ctx, &mut res, output_target);
        Self {
            res,
            dynamic,
            cylinder_instances: RefCell::new(vec![]),
            sphere_instances: RefCell::new(vec![]),
        }
    }

    pub fn line(&self, color: glm::Vec4, line: (glm::Vec3, glm::Vec3)) {
        let mut cylinder_instances = self.cylinder_instances.borrow_mut();
        cylinder_instances.push(WireframeInstanceInput {
            color,
            offset1: glm::vec4(line.0.x, line.0.y, line.0.z, 1.0),
            offset2: glm::vec4(line.1.x, line.1.y, line.1.z, 1.0),
        });
    }

    pub fn resize(&mut self, ctx: &WgpuContext, output_target: Rc<RenderTarget>) {
        self.dynamic = DynamicResources::new(ctx, &mut self.res, output_target);
    }

    pub fn input_target(&self) -> Rc<RenderTarget> {
        Rc::new(RenderTarget {
            render_target: self.dynamic.output_target.render_target.clone(),
            depth_target: Some(self.dynamic.depth_view.clone()),
            info: RenderTargetInfo {
                format: self.dynamic.output_target.info.format,
                width: self.dynamic.output_target.info.width,
                height: self.dynamic.output_target.info.height,
            },
        })
    }

    pub fn update(
        &mut self,
        ctx: &WgpuContext,
        command_encoder: &mut CommandEncoder,
        proj: &glm::Mat4x4,
        view: &glm::Mat4x4,
    ) {
        self.line(
            glm::vec4(1.0, 0.0, 1.0, 1.0),
            (glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.0)),
        );
        {
            let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("overlay render_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.dynamic.output_target.render_target,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.dynamic.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.dynamic.pipeline);
            render_pass.set_push_constants(
                ShaderStages::VERTEX,
                0,
                bytemuck::cast_slice(&[PushConstants {
                    proj: *proj,
                    view: *view,
                }]),
            );

            let mut cylinder_instances = self.cylinder_instances.borrow_mut();

            let cylinder_instance_buffer = self.res.cylinder_instance_buffer.get_or_recreate(
                cylinder_instances.len() as u32,
                |size| {
                    ctx.device.create_buffer(&BufferDescriptor {
                        label: Some("overlay cylinder_instance_buffer"),
                        size: size as u64 * size_of::<WireframeInstanceInput>() as u64,
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                },
            );

            ctx.queue.write_buffer(
                cylinder_instance_buffer,
                0,
                bytemuck::cast_slice(&cylinder_instances),
            );

            render_pass.set_vertex_buffer(0, self.res.cylinder_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, cylinder_instance_buffer.slice(..));
            render_pass.draw(
                0..self.res.cylinder_vertex_buffer.size() as u32 / size_of::<glm::Vec4>() as u32,
                0..1,
            );

            cylinder_instances.clear();
        }
    }
}
