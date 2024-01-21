use crate::user_event::UserEvent;
use crate::util::*;
use crate::wgpu_context::WgpuContext;
use crate::FinalDrawResources;
use bytemuck::{Pod, Zeroable};
use nalgebra_glm as glm;
use pod_enum::pod_enum;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::*;
use winit::event_loop::EventLoopProxy;

#[repr(u32)]
#[pod_enum]
enum TonemapType {
    None = 0,
    AcesLum = 1,
    AcesFull = 2,
}

impl Default for TonemapType {
    fn default() -> Self {
        TonemapType::None
    }
}

#[repr(u32)]
#[pod_enum]
enum TargetColorSpace {
    Linear = 0,
    Srgb = 1,
}

impl Default for TargetColorSpace {
    fn default() -> Self {
        TargetColorSpace::Linear
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct Uniforms {
    linear_transform: glm::Mat4x4,
    tonemapping: TonemapType,
    target_color_space: TargetColorSpace,
    _pad0: [f32; 2],
    output_scale: f32,
    _pad1: [f32; 3],
}

struct Resources {
    renderbuffer_desc: TextureDescriptor<'static>,
    pipeline_layout: PipelineLayout,
    shader: ShaderModule,
    uniform_buffer: Buffer,
    bind_group_layout: BindGroupLayout,
    linear_buffer_sampler: Sampler,
}

struct DynamicResources {
    output_target_info: Rc<RenderTargetInfo>,
    input_target: Rc<RenderTarget>,
    final_draw_resources: Arc<FinalDrawResources>,
}

pub struct Tonemap {
    res: Resources,
    dynamic: DynamicResources,
    exposure: f32,
    bleed: f32,
    tonemapping: TonemapType,
    output_scale: f32,
}

impl Resources {
    fn new(ctx: &WgpuContext) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tonemap shader"),
            source: ShaderSource::Wgsl(include_str!("./tonemap.wgsl").into()),
        });

        let renderbuffer_desc = TextureDescriptor {
            label: Some("tonemap renderbuffer_desc"),
            size: Extent3d {
                width: 0,
                height: 0,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("tonemap bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(size_of::<Uniforms>() as u64),
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("tonemap pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("tonemap uniform_buffer"),
            size: size_of::<Uniforms>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let linear_buffer_sampler = ctx.device.create_sampler(&SamplerDescriptor {
            label: Some("tonemap linear_buffer_sampler"),
            ..Default::default()
        });

        Self {
            renderbuffer_desc,
            pipeline_layout,
            shader,
            uniform_buffer,
            bind_group_layout,
            linear_buffer_sampler,
        }
    }
}

impl DynamicResources {
    fn new(
        ctx: &WgpuContext,
        res: &mut Resources,
        output_target_info: Rc<RenderTargetInfo>,
    ) -> Self {
        res.renderbuffer_desc.size.width = output_target_info.width;
        res.renderbuffer_desc.size.height = output_target_info.height;
        let renderbuffer = ctx.device.create_texture(&res.renderbuffer_desc);
        let renderbuffer_view = renderbuffer.create_view(&TextureViewDescriptor::default());
        let pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("tonemap pipeline"),
                layout: Some(&res.pipeline_layout),
                vertex: VertexState {
                    module: &res.shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &res.shader,
                    entry_point: "fs_main",
                    targets: &[Some(output_target_info.format.into())],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview: None,
            });

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tonemap bind_group"),
            layout: &res.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&res.linear_buffer_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&renderbuffer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let input_target = Rc::new(RenderTarget {
            render_target: renderbuffer_view,
            info: RenderTargetInfo {
                format: res.renderbuffer_desc.format,
                width: res.renderbuffer_desc.size.width,
                height: res.renderbuffer_desc.size.height,
            },
        });

        Self {
            output_target_info,
            input_target,
            final_draw_resources: Arc::new(FinalDrawResources {
                pipeline,
                bind_group,
            }),
        }
    }
}

impl Tonemap {
    pub fn new(ctx: &WgpuContext, output_target_info: Rc<RenderTargetInfo>) -> Self {
        let mut res = Resources::new(ctx);
        let dynamic = DynamicResources::new(ctx, &mut res, output_target_info);
        Self {
            res,
            dynamic,
            exposure: 1.0,
            bleed: 0.0,
            tonemapping: TonemapType::AcesFull,
            output_scale: 1.0,
        }
    }
    pub fn resize(&mut self, ctx: &WgpuContext, output_target_info: Rc<RenderTargetInfo>) {
        self.dynamic = DynamicResources::new(ctx, &mut self.res, output_target_info);
    }

    pub fn update(&mut self, ctx: &WgpuContext) {
        let output_linear = self.dynamic.output_target_info.format.is_srgb();
        let exposure = self.exposure;
        let bleed = exposure * self.bleed;
        let transform = glm::mat3(
            exposure, bleed, bleed, bleed, exposure, bleed, bleed, bleed, exposure,
        );

        let uniforms = Uniforms {
            linear_transform: glm::mat3_to_mat4(&transform),
            tonemapping: self.tonemapping,
            target_color_space: if output_linear {
                TargetColorSpace::Linear
            } else {
                TargetColorSpace::Srgb
            },
            output_scale: self.output_scale,
            ..Default::default()
        };
        ctx.queue
            .write_buffer(&self.res.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn final_draw_resources(&self) -> Arc<FinalDrawResources> {
        self.dynamic.final_draw_resources.clone()
    }

    pub fn input_target(&self) -> Rc<RenderTarget> {
        self.dynamic.input_target.clone()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, _elp: &EventLoopProxy<UserEvent>) {
        ui.collapsing("Tonemap", |ui| {
            ui.add(
                egui::Slider::new(&mut self.exposure, 0.01..=1000.0)
                    .logarithmic(true)
                    .text("Exposure"),
            );
            ui.add(egui::Slider::new(&mut self.bleed, 0.0..=0.1).text("Bleed"));
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.tonemapping, TonemapType::None, "None");
                ui.radio_value(&mut self.tonemapping, TonemapType::AcesLum, "AcesLum");
                ui.radio_value(&mut self.tonemapping, TonemapType::AcesFull, "AcesFull");
            });
            ui.add(egui::Slider::new(&mut self.output_scale, 0.0..=10.0).text("Output scale"));
        });
    }
}
