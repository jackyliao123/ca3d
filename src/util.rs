use bytemuck::{Pod, Zeroable};
use std::rc::Rc;
use wgpu::{Texture, TextureFormat, TextureView};

use crate::wgpu_context::WgpuContext;

pub struct RenderTargetInfo {
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl From<&WgpuContext<'_>> for RenderTargetInfo {
    fn from(ctx: &WgpuContext) -> Self {
        Self {
            format: ctx.surface_format,
            width: ctx.surface_config.width,
            height: ctx.surface_config.height,
        }
    }
}

pub struct RenderTarget {
    pub render_target: Rc<TextureView>,
    pub depth_target: Option<Rc<TextureView>>,
    pub info: RenderTargetInfo,
}

pub struct TextureAndView {
    pub texture: Texture,
    pub view: TextureView,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DrawIndirectPod {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub base_vertex: u32,
    pub base_instance: u32,
}
