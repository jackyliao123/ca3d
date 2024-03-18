use std::mem::size_of;
use std::rc::Rc;

use nalgebra_glm as glm;
use wgpu::*;

use crate::util::RenderTarget;
use crate::wgpu_context::WgpuContext;

struct Resources {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

struct DynamicResources {
    output_target: Rc<RenderTarget>,
    buffer: Buffer,
    cpu_buffer: Buffer,
    bind_group: BindGroup,
}

pub struct Picker {
    res: Resources,
    dynamic: DynamicResources,
}

impl Resources {
    fn new(ctx: &WgpuContext) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("picker shader"),
            source: ShaderSource::Wgsl(include_str!("picker.wgsl").into()),
        });
        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("picker bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
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
                label: Some("picker pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("picker pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_main",
            });
        Self {
            bind_group_layout,
            pipeline,
        }
    }
}

impl DynamicResources {
    fn new(ctx: &WgpuContext, res: &mut Resources, output_target: Rc<RenderTarget>) -> Self {
        let buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("picker buffer"),
            size: (output_target.info.width * output_target.info.height) as u64
                * size_of::<glm::Vec4>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cpu_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("picker cpu_buffer"),
            size: (output_target.info.width * output_target.info.height) as u64
                * size_of::<glm::Vec4>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("picker bind_group"),
            layout: &res.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&output_target.render_target),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        Self {
            output_target,
            buffer,
            cpu_buffer,
            bind_group,
        }
    }
}

impl Picker {
    pub fn new(ctx: &WgpuContext, output_target: Rc<RenderTarget>) -> Self {
        let mut res = Resources::new(ctx);
        let dynamic = DynamicResources::new(ctx, &mut res, output_target);
        Self { res, dynamic }
    }

    pub fn resize(&mut self, ctx: &WgpuContext, output_target: Rc<RenderTarget>) {
        self.dynamic = DynamicResources::new(ctx, &mut self.res, output_target);
    }

    pub fn input_target(&self) -> Rc<RenderTarget> {
        self.dynamic.output_target.clone()
    }

    pub fn update(&mut self, ctx: &WgpuContext, command_encoder: &mut CommandEncoder) {
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("picker compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.res.pipeline);
            compute_pass.set_bind_group(0, &self.dynamic.bind_group, &[]);
            compute_pass.dispatch_workgroups(
                self.dynamic.output_target.info.width.div_ceil(8),
                self.dynamic.output_target.info.height.div_ceil(8),
                1,
            );
        }
        command_encoder.copy_buffer_to_buffer(
            &self.dynamic.buffer,
            0,
            &self.dynamic.cpu_buffer,
            0,
            (self.dynamic.output_target.info.width * self.dynamic.output_target.info.height) as u64
                * size_of::<glm::Vec4>() as u64,
        );

        {
            let mapped_range = self.dynamic.cpu_buffer.slice(..).get_mapped_range();

            // let in_buf: &[glm::Vec4] = bytemuck::cast_slice(mapped_range.as_ref());
        }

        self.dynamic.cpu_buffer.unmap();
    }

    pub fn after_submit(&self) {
        self.dynamic
            .cpu_buffer
            .slice(..)
            .map_async(MapMode::Read, |result| {
                match result {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Failed to map buffer: {:?}", e);
                    }
                }
                // result.expect("Failed to map buffer");
            });
    }
}
