use crate::profiler::Profiler;

use wgpu::*;

pub struct WgpuContext<'window> {
    pub surface: Surface<'window>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_caps: SurfaceCapabilities,
    pub surface_format: TextureFormat,
    pub surface_config: SurfaceConfiguration,
    pub profiler: Profiler,
}
