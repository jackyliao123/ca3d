use std::mem::size_of;
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use nalgebra_glm as glm;
use wgpu::*;
use winit::event_loop::EventLoopProxy;

use crate::user_event::UserEvent;
use crate::util::*;
use crate::wgpu_context::WgpuContext;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct Uniforms {
    scale_fact: glm::Vec2,
    _pad0: [f32; 2],
}

struct Resources {
    shader: ShaderModule,
    texture_desc: TextureDescriptor<'static>,
    downsample_group_layout: BindGroupLayout,
    upsample_group_layout: BindGroupLayout,
    sampler_bind_group: BindGroup,
    downsample_pipeline_layout: PipelineLayout,
    upsample_pipeline_layout: PipelineLayout,
}

struct DynamicResources {
    output_target: Rc<RenderTarget>,
    input_target: Rc<RenderTarget>,

    downsample_pipeline: ComputePipeline,
    upsample_pipeline: ComputePipeline,
    per_pass_bind_group_downsample: Vec<BindGroup>,
    per_pass_bind_group_upsample: Vec<BindGroup>,
    upsample_uniforms: Vec<Buffer>,
}

pub struct Bloom {
    res: Resources,
    dynamic: DynamicResources,
    mip_limit: u32,
    bloom_factor: f32,
}

impl Resources {
    fn new(ctx: &WgpuContext) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom shader"),
            source: ShaderSource::Wgsl(include_str!("./bloom.wgsl").into()),
        });

        let texture_desc = TextureDescriptor {
            label: Some("bloom texture_desc"),
            size: Extent3d {
                width: 0,
                height: 0,
                depth_or_array_layers: 1,
            },
            mip_level_count: 0,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        let downsample_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("bloom downsample_group_layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::WriteOnly,
                                format: texture_desc.format,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let upsample_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("bloom upsample_group_layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::WriteOnly,
                                format: texture_desc.format,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: BufferSize::new(size_of::<Uniforms>() as u64),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let sampler = ctx.device.create_sampler(&SamplerDescriptor {
            label: Some("bloom sampler"),
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            ..Default::default()
        });

        let sampler_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("bloom sampler_group_layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    }],
                });

        let sampler_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bloom sampler_bind_group"),
            layout: &sampler_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(&sampler),
            }],
        });

        let downsample_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("bloom downsample_pipeline_layout"),
                    bind_group_layouts: &[&sampler_group_layout, &downsample_group_layout],
                    push_constant_ranges: &[],
                });

        let upsample_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("bloom upsample_pipeline_layout"),
                    bind_group_layouts: &[&sampler_group_layout, &upsample_group_layout],
                    push_constant_ranges: &[],
                });

        Self {
            shader,
            texture_desc,
            downsample_group_layout,
            upsample_group_layout,
            sampler_bind_group,
            downsample_pipeline_layout,
            upsample_pipeline_layout,
        }
    }
}

impl DynamicResources {
    fn new(
        ctx: &WgpuContext,
        res: &mut Resources,
        mip_limit: u32,
        output_target: Rc<RenderTarget>,
    ) -> Self {
        res.texture_desc.size.width = output_target.info.width;
        res.texture_desc.size.height = output_target.info.height;
        let mips = res
            .texture_desc
            .size
            .max_mips(TextureDimension::D2)
            .min(16)
            .min(mip_limit);
        res.texture_desc.mip_level_count = mips;

        let create_resources_for_shader = |entry_point: &str, pipeline_layout: &PipelineLayout| {
            let texture = ctx.device.create_texture(&res.texture_desc);
            let view = (0..mips)
                .map(|i| {
                    texture.create_view(&TextureViewDescriptor {
                        base_mip_level: i,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect::<Vec<_>>();
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("bloom pipeline"),
                    layout: Some(pipeline_layout),
                    module: &res.shader,
                    entry_point,
                });
            (texture, view, pipeline)
        };

        let (downsample_buffer, downsample_view, downsample_pipeline) =
            create_resources_for_shader("cs_downsample", &res.downsample_pipeline_layout);
        let (_upsample_buffer, upsample_view, upsample_pipeline) =
            create_resources_for_shader("cs_upsample", &res.upsample_pipeline_layout);

        let renderbuffer_view = downsample_buffer.create_view(&TextureViewDescriptor {
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let per_pass_bind_group_downsample = (0..(mips - 1) as usize)
            .map(|i| {
                ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("bloom per_pass_bind_group_downsample"),
                    layout: &res.downsample_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&downsample_view[i]),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(&downsample_view[i + 1]),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>();

        let (per_pass_bind_group_upsample, upsample_uniforms): (Vec<BindGroup>, Vec<Buffer>) = (0
            ..(mips - 1) as usize)
            .map(|i| {
                let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
                    label: Some("bloom uniform_buffer"),
                    size: size_of::<Uniforms>() as u64,
                    usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                });

                ctx.queue.write_buffer(
                    &uniform_buffer,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        scale_fact: glm::vec2(
                            1.0 / mips as f32,
                            if i == (mips - 2) as usize {
                                1.0 / mips as f32
                            } else {
                                1.0
                            },
                        ),
                        ..Default::default()
                    }),
                );

                let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("bloom bind_group"),
                    layout: &res.upsample_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(if i == (mips - 2) as usize {
                                &downsample_view[i + 1]
                            } else {
                                &upsample_view[i + 1]
                            }),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(if i == 0 {
                                &output_target.render_target
                            } else {
                                &upsample_view[i]
                            }),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::TextureView(&downsample_view[i]),
                        },
                    ],
                });
                (bind_group, uniform_buffer)
            })
            .unzip();

        let input_target = Rc::new(RenderTarget {
            render_target: renderbuffer_view,
            info: RenderTargetInfo {
                format: res.texture_desc.format,
                width: res.texture_desc.size.width,
                height: res.texture_desc.size.height,
            },
        });

        Self {
            output_target,
            input_target,

            downsample_pipeline,
            upsample_pipeline,
            per_pass_bind_group_downsample,
            per_pass_bind_group_upsample,
            upsample_uniforms,
        }
    }
}

impl Bloom {
    pub fn new(ctx: &WgpuContext, output_target: Rc<RenderTarget>) -> Self {
        let mut res = Resources::new(ctx);
        let mip_limit = 12;
        let dynamic = DynamicResources::new(ctx, &mut res, mip_limit, output_target);
        Self {
            res,
            dynamic,
            bloom_factor: 0.05,
            mip_limit: 12,
        }
    }
    pub fn resize(&mut self, ctx: &WgpuContext, output_target: Rc<RenderTarget>) {
        self.dynamic = DynamicResources::new(ctx, &mut self.res, self.mip_limit, output_target);
    }

    pub fn update(&mut self, ctx: &WgpuContext, command_encoder: &mut CommandEncoder) {
        if self.res.texture_desc.mip_level_count == 1 {
            return;
        }

        let full_bloom = if self.res.texture_desc.mip_level_count <= 2 {
            glm::vec2(0.0, 1.0)
        } else {
            glm::vec2(1.0 / self.res.texture_desc.mip_level_count as f32, 1.0)
        };
        let no_bloom = glm::vec2(1.0f32, 0.0);
        let uniforms = Uniforms {
            scale_fact: glm::mix(&no_bloom, &full_bloom, self.bloom_factor),
            ..Default::default()
        };
        ctx.queue.write_buffer(
            self.dynamic.upsample_uniforms.first().unwrap(),
            0,
            bytemuck::bytes_of(&uniforms),
        );
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bloom compute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.dynamic.downsample_pipeline);
            compute_pass.set_bind_group(0, &self.res.sampler_bind_group, &[]);
            for i in 0..self.dynamic.per_pass_bind_group_downsample.len() {
                compute_pass.set_bind_group(
                    1,
                    &self.dynamic.per_pass_bind_group_downsample[i],
                    &[],
                );
                let div = 1 << (i + 1);
                compute_pass.dispatch_workgroups(
                    self.res.texture_desc.size.width.div_ceil(div * 8),
                    self.res.texture_desc.size.height.div_ceil(div * 8),
                    1,
                );
            }

            compute_pass.set_pipeline(&self.dynamic.upsample_pipeline);
            for i in (0..self.dynamic.per_pass_bind_group_upsample.len()).rev() {
                compute_pass.set_bind_group(1, &self.dynamic.per_pass_bind_group_upsample[i], &[]);
                let div = 1 << i;
                compute_pass.dispatch_workgroups(
                    self.res.texture_desc.size.width.div_ceil(div * 8),
                    self.res.texture_desc.size.height.div_ceil(div * 8),
                    1,
                );
            }
        }
    }

    pub fn input_target(&self) -> Rc<RenderTarget> {
        // If mip level is 1, bypass bloom altogether
        if self.res.texture_desc.mip_level_count == 1 {
            self.dynamic.output_target.clone()
        } else {
            self.dynamic.input_target.clone()
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, elp: &EventLoopProxy<UserEvent>) {
        ui.collapsing("Bloom", |ui| {
            ui.add(egui::Slider::new(&mut self.bloom_factor, 0.0..=1.0).text("Bloom Factor"));
            let prev_mip_limit = self.mip_limit;
            ui.add(egui::Slider::new(&mut self.mip_limit, 1..=16).text("Mip Limit"));
            if prev_mip_limit != self.mip_limit {
                let _ = elp.send_event(UserEvent::RequestResize);
            }
        });
    }
}
