use bytemuck::{Pod, Zeroable};
use nalgebra_glm as glm;
use std::mem::size_of;
use wgpu::*;
use winit::event_loop::EventLoopProxy;

use crate::chunk_manager::ChunkManager;
use crate::user_event::UserEvent;
use crate::wgpu_context::WgpuContext;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct PushConstants {
    rng: u32,
    chunks_per_buffer_shift: u32,
    starting_which: u32,
    num_chunks: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
struct ChunkInfoEntry {
    pos: glm::IVec3,
    _pad0: u32,
}

struct Resources {
    chunk_info_buffer: Buffer,
    data_bind_group: BindGroup,
    pipeline: ComputePipeline,
}

pub struct Simulate {
    res: Resources,
    n_iter: u32,
    pub paused: bool,
    pub step: u32,
}

impl Resources {
    fn new(ctx: &WgpuContext, chunk_manager: &ChunkManager) -> Self {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("simulate shader"),
            source: ShaderSource::Wgsl(include_str!("simulate.wgsl").into()),
        });

        let data_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("simulate data_bind_group_layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                (4096 * size_of::<ChunkInfoEntry>()) as u64,
                            ),
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("simulate pipeline_layout"),
                bind_group_layouts: &[
                    &data_bind_group_layout,
                    chunk_manager.bind_group_layout(true),
                ],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..size_of::<PushConstants>() as u32,
                }],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("simulate pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_simulate",
            });

        let chunk_info_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("simulate chunk_info_buffer"),
            size: (4096 * size_of::<ChunkInfoEntry>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let data_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("simulate data_bind_group"),
            layout: &data_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: chunk_info_buffer.as_entire_binding(),
            }],
        });

        Self {
            chunk_info_buffer,
            data_bind_group,

            pipeline,
        }
    }
}

impl Simulate {
    pub fn new(ctx: &WgpuContext, chunk_manager: &ChunkManager) -> Self {
        let res = Resources::new(ctx, chunk_manager);
        Self {
            res,
            n_iter: 1,
            paused: true,
            step: 0,
        }
    }

    pub fn update(
        &mut self,
        ctx: &WgpuContext,
        command_encoder: &mut CommandEncoder,
        chunk_manager: &mut ChunkManager,
    ) {
        if self.paused && self.step == 0 {
            return;
        }
        if self.step > 0 {
            self.step -= 1;
        }
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("simulate compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.res.pipeline);

        let mut chunk_info = vec![ChunkInfoEntry::default(); chunk_manager.num_offsets() as usize];

        for chunk in chunk_manager.chunks().values() {
            chunk_info[chunk.offset() as usize] = ChunkInfoEntry {
                pos: chunk.pos,
                ..Default::default()
            };
        }

        ctx.queue.write_buffer(
            &self.res.chunk_info_buffer,
            0,
            bytemuck::cast_slice(&chunk_info),
        );

        compute_pass.set_bind_group(0, &self.res.data_bind_group, &[]);
        compute_pass.set_bind_group(1, chunk_manager.bind_group(true), &[]);

        for i in 0..self.n_iter {
            compute_pass.set_push_constants(
                0,
                bytemuck::bytes_of(&PushConstants {
                    rng: rand::random(),
                    chunks_per_buffer_shift: chunk_manager.chunks_per_group().ilog2(),
                    starting_which: chunk_manager.which() ^ (i & 1),
                    num_chunks: chunk_manager.num_offsets(),
                }),
            );
            compute_pass.dispatch_workgroups(chunk_manager.num_offsets(), 512, 1);
        }

        drop(compute_pass);
        chunk_manager.advance_which(self.n_iter);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, _elp: &EventLoopProxy<UserEvent>) {
        ui.collapsing("Simulate", |ui| {
            ui.add(egui::Slider::new(&mut self.n_iter, 1..=1024).text("Iterations"));
            ui.add(egui::Checkbox::new(&mut self.paused, "Pause"));
        });
    }
}
