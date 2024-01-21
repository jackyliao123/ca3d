use std::cell::RefCell;
use std::time::Duration;

use egui::Ui;
use egui_extras::{Column, TableBuilder};
use indexmap::IndexMap;
use wgpu::*;

struct CpuTimer {
    #[cfg(target_arch = "wasm32")]
    performance: web_sys::Performance,
}

#[derive(Debug, Clone, Copy)]
struct CpuTimestamp {
    #[cfg(not(target_arch = "wasm32"))]
    now: std::time::Instant,

    #[cfg(target_arch = "wasm32")]
    now: f64,
}

impl CpuTimer {
    pub fn new() -> Self {
        Self {
            #[cfg(target_arch = "wasm32")]
            performance: web_sys::window().unwrap().performance().unwrap(),
        }
    }

    pub fn now(&self) -> CpuTimestamp {
        #[cfg(not(target_arch = "wasm32"))]
        let now = std::time::Instant::now();

        #[cfg(target_arch = "wasm32")]
        let now = self.performance.now() / 1000.0;

        CpuTimestamp { now }
    }
}

impl CpuTimestamp {
    pub fn elapsed(&self, prev_timestamp: &Self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.now - prev_timestamp.now
        }

        #[cfg(target_arch = "wasm32")]
        {
            let elapsed = self.now - prev_timestamp.now;
            Duration::from_secs_f64(elapsed)
        }
    }
}

struct PendingQueryInfo {
    cpu_start: CpuTimestamp,
    cpu_end: Option<CpuTimestamp>,
    gpu_start_query_index: u32,
    gpu_end_query_index: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct QueryInfo {
    pub cpu: (Duration, Duration),
    pub gpu: Option<(Duration, Duration)>,
}

struct GpuResources {
    query_set: QuerySet,
    query_buffer: Buffer,
    query_buffer_staging: Buffer,
}

impl GpuResources {
    fn new(device: &Device, max_queries: u32) -> Self {
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("profiler query_set"),
            ty: QueryType::Timestamp,
            count: max_queries,
        });
        let query_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("profiler query_buffer"),
            size: max_queries as u64 * std::mem::size_of::<u64>() as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        let query_buffer_staging = device.create_buffer(&BufferDescriptor {
            label: Some("profiler query_buffer_staging"),
            size: max_queries as u64 * std::mem::size_of::<u64>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: true,
        });
        Self {
            query_set,
            query_buffer,
            query_buffer_staging,
        }
    }
}

struct Mutables {
    name_stack: Vec<String>,
    queries: IndexMap<Vec<String>, PendingQueryInfo>,
    query_index: u32,
}

impl Mutables {
    fn new() -> Self {
        Self {
            name_stack: Vec::new(),
            queries: IndexMap::new(),
            query_index: 0,
        }
    }
}

pub struct Profiler {
    cpu_timer: CpuTimer,
    gpu_resources: Option<GpuResources>,
    max_queries: u32,
    mutables: RefCell<Mutables>,
    timestamp_period: f32,
    prev_frame_info: IndexMap<String, QueryInfo>,
}

impl Profiler {
    pub fn new(device: &Device, queue: &Queue, cpu_only: bool) -> Self {
        let cpu_timer = CpuTimer::new();
        let max_queries = 2;

        let gpu_resources = if cpu_only {
            None
        } else {
            Some(GpuResources::new(device, max_queries))
        };

        let timestamp_period = queue.get_timestamp_period();

        Self {
            cpu_timer,
            mutables: RefCell::new(Mutables::new()),
            gpu_resources,
            max_queries,
            timestamp_period,
            prev_frame_info: IndexMap::new(),
        }
    }

    pub fn begin(&self, encoder: &mut CommandEncoder, name: &str) {
        let mutables = &mut *self.mutables.borrow_mut();

        if mutables.query_index < self.max_queries {
            if let Some(gpu_resources) = &self.gpu_resources {
                encoder.write_timestamp(&gpu_resources.query_set, mutables.query_index);
            }
        }
        mutables.name_stack.push(name.to_owned());
        let query_info = PendingQueryInfo {
            cpu_start: self.cpu_timer.now(),
            cpu_end: None,
            gpu_start_query_index: mutables.query_index,
            gpu_end_query_index: None,
        };
        mutables
            .queries
            .insert(mutables.name_stack.clone(), query_info);
        mutables.query_index += 1;
    }

    pub fn end(&self, encoder: &mut CommandEncoder) {
        let mutables = &mut *self.mutables.borrow_mut();

        if mutables.query_index < self.max_queries {
            if let Some(gpu_resources) = &self.gpu_resources {
                encoder.write_timestamp(&gpu_resources.query_set, mutables.query_index);
            }
        }
        let query_info = mutables
            .queries
            .get_mut(&mutables.name_stack)
            .expect("Profiler end called without begin");
        mutables
            .name_stack
            .pop()
            .expect("Profiler end called without begin");
        query_info.cpu_end = Some(self.cpu_timer.now());
        query_info.gpu_end_query_index = Some(mutables.query_index);
        mutables.query_index += 1;
    }

    pub fn profile<T>(
        &self,
        encoder: &mut CommandEncoder,
        name: &str,
        cb: impl FnOnce(&mut CommandEncoder) -> T,
    ) -> T {
        self.begin(encoder, name);
        let ret = cb(encoder);
        self.end(encoder);
        ret
    }

    pub fn gather_prev_frame_info(&mut self, device: &Device) {
        let mutables = &mut *self.mutables.borrow_mut();

        {
            let mapped_range = self.gpu_resources.as_ref().map(|gpu_resources| {
                gpu_resources
                    .query_buffer_staging
                    .slice(..)
                    .get_mapped_range()
            });
            let timestamps: Option<&[u64]> = mapped_range
                .as_ref()
                .map(|mapped_range| bytemuck::cast_slice(mapped_range));

            self.prev_frame_info = mutables
                .queries
                .first()
                .map(|(_, first_query)| {
                    let frame_cpu_start = &first_query.cpu_start;
                    let frame_gpu_start = timestamps
                        .map(|timestamps| timestamps[first_query.gpu_start_query_index as usize]);

                    mutables
                        .queries
                        .iter()
                        .map(|(name, query_info)| {
                            let cpu_start = query_info.cpu_start.elapsed(frame_cpu_start);
                            let cpu_duration = query_info
                                .cpu_end
                                .expect("No CPU end time, did you forget to call end()?")
                                .elapsed(&query_info.cpu_start);
                            let gpu_start_and_duration = query_info
                                .gpu_end_query_index
                                .zip(timestamps)
                                .zip(frame_gpu_start)
                                .and_then(
                                    |((gpu_end_query_index, timestamps), frame_gpu_start)| {
                                        timestamps
                                            .get(query_info.gpu_start_query_index as usize)
                                            .zip(timestamps.get(gpu_end_query_index as usize))
                                            .map(|(start, end)| {
                                                (
                                                    Duration::from_secs_f64(
                                                        (start.saturating_sub(frame_gpu_start))
                                                            as f64
                                                            * self.timestamp_period as f64
                                                            / 1e9,
                                                    ),
                                                    Duration::from_secs_f64(
                                                        (end.saturating_sub(*start)) as f64
                                                            * self.timestamp_period as f64
                                                            / 1e9,
                                                    ),
                                                )
                                            })
                                    },
                                );

                            (
                                name.join("."),
                                QueryInfo {
                                    cpu: (cpu_start, cpu_duration),
                                    gpu: gpu_start_and_duration,
                                },
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
        }

        if mutables.query_index > self.max_queries {
            while mutables.query_index > self.max_queries {
                self.max_queries *= 2;
            }
            if let Some(gpu_resources) = self.gpu_resources.as_mut() {
                log::info!("Resizing query buffer to {}", self.max_queries);
                *gpu_resources = GpuResources::new(device, self.max_queries);
            }
        }
    }

    pub fn ui(&self, ui: &mut Ui) {
        TableBuilder::new(ui)
            .column(Column::auto().resizable(true))
            .column(Column::auto().resizable(true))
            .column(Column::auto().resizable(true))
            .header(20.0, |mut header| {
                header.col(|ui| {
                    ui.heading("Stage");
                });
                header.col(|ui| {
                    ui.heading("CPU time");
                });
                header.col(|ui| {
                    ui.heading("GPU time");
                });
            })
            .body(|mut body| {
                for (name, query_info) in &self.prev_frame_info {
                    body.row(30.0, |mut row| {
                        row.col(|ui| {
                            ui.label(name);
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.6} ms", query_info.cpu.1.as_secs_f64() * 1000.0));
                        });
                        row.col(|ui| {
                            if let Some(gpu) = query_info.gpu {
                                ui.label(format!("{:.6} ms", gpu.1.as_secs_f64() * 1000.0));
                            }
                        });
                    });
                }
            })
    }

    pub fn begin_frame(&self, encoder: &mut CommandEncoder) {
        {
            let mutables = &mut *self.mutables.borrow_mut();

            mutables.queries.clear();
            assert!(
                mutables.name_stack.is_empty(),
                "Profiler stack not empty, did you forget to call end()?"
            );
            mutables.query_index = 0;
        }

        // Profile the frame
        self.begin(encoder, "main");
    }

    pub fn end_frame(&self, encoder: &mut CommandEncoder) {
        self.end(encoder);

        let queries = self.max_queries.min(self.mutables.borrow_mut().query_index);

        if let Some(gpu_resources) = &self.gpu_resources {
            encoder.resolve_query_set(
                &gpu_resources.query_set,
                0..queries,
                &gpu_resources.query_buffer,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &gpu_resources.query_buffer,
                0,
                &gpu_resources.query_buffer_staging,
                0,
                queries as u64 * std::mem::size_of::<u64>() as u64,
            );
            gpu_resources.query_buffer_staging.unmap();
        }
    }

    pub fn after_submit(&self) {
        if let Some(gpu_resources) = &self.gpu_resources {
            gpu_resources
                .query_buffer_staging
                .slice(..)
                .map_async(MapMode::Read, |result| {
                    result.expect("Failed to map buffer");
                });
        }
    }
}
