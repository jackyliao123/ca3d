mod chunk;
mod chunk_datastore;
mod chunk_manager;
mod game;
mod gpu_stage;
mod key_tracker;
mod profiler;
mod shared_resource_helper;
mod user_event;
mod util;
mod wgpu_context;

use crate::game::Game;
use crate::user_event::UserEvent;
use crate::wgpu_context::WgpuContext;
use egui::ViewportId;

use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoopBuilder;
use winit::window::CursorGrabMode;
use winit::{
    event::{Event, WindowEvent},
    window::WindowBuilder,
};

pub struct FinalDrawResources {
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
}

struct GamePaintCallback {}

impl egui_wgpu::CallbackTrait for GamePaintCallback {
    fn paint<'a>(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'a>,
        callback_resources: &'a egui_wgpu::CallbackResources,
    ) {
        let draw_resources = callback_resources.get::<Arc<FinalDrawResources>>().unwrap();
        render_pass.set_pipeline(&draw_resources.pipeline);
        render_pass.set_bind_group(0, &draw_resources.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

pub async fn start() {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event()
        .build()
        .unwrap();
    let event_loop_proxy = event_loop.create_proxy();

    let window = WindowBuilder::new()
        .with_title("CellularAutomata3d")
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    add_canvas_to_body(&window, event_loop_proxy.clone());

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::default()
    });

    let surface = instance
        .create_surface(&window)
        .expect("Could not create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Could not create adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: if cfg!(target_arch = "wasm32") {
                    wgpu::Features::default()
                } else {
                    wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                        | wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | wgpu::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::DEPTH_CLIP_CONTROL
                },
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits {
                        max_texture_dimension_1d: 4096,
                        max_texture_dimension_2d: 4096,
                        max_texture_dimension_3d: 1024,
                        max_uniform_buffer_binding_size: 16384,
                        max_vertex_buffer_array_stride: 0,
                        max_compute_invocations_per_workgroup: 512,
                        ..Default::default()
                    }
                } else {
                    wgpu::Limits {
                        max_compute_invocations_per_workgroup: 512,
                        max_storage_textures_per_shader_stage: 16,
                        max_push_constant_size: 128,
                        ..Default::default()
                    }
                },
            },
            None,
        )
        .await
        .expect("Could not create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|format| format.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    log::info!("Surface format: {:?}", surface_format);

    let preferred_present_modes = [
        wgpu::PresentMode::Fifo,
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::Immediate,
    ]
    .iter()
    .filter(|mode| surface_caps.present_modes.contains(mode))
    .copied()
    .collect::<Vec<_>>();

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: preferred_present_modes[0],
        desired_maximum_frame_latency: 2,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &surface_config);

    let mut requested_surface_size: Option<PhysicalSize<u32>> = None;

    let profiler = profiler::Profiler::new(&device, &queue, cfg!(target_arch = "wasm32"));
    let mut ctx = WgpuContext {
        surface,
        adapter,
        device,
        queue,
        surface_caps,
        surface_format,
        surface_config,
        profiler,
    };

    let mut egui_state = egui_winit::State::new(
        egui::Context::default(),
        ViewportId::default(),
        &window,
        None,
        Some(4096),
    );
    let mut egui_renderer =
        egui_wgpu::renderer::Renderer::new(&ctx.device, surface_format, None, 1);
    let mut cursor_locked = false;

    let mut game = Game::new(&ctx);

    event_loop
        .run(|event, elwt| {
            match event {
                Event::WindowEvent { window_id, event } if window_id == window.id() => {
                    if cursor_locked {
                        use WindowEvent::*;
                        match event {
                            KeyboardInput { .. } | MouseInput { .. } | MouseWheel { .. } => {
                                // Send these inputs to the game
                                game.input(&event, &event_loop_proxy);
                                return;
                            }
                            CursorMoved { .. } | AxisMotion { .. } => {
                                // Don't send these to the game nor egui
                                return;
                            }
                            CursorLeft { .. } | Focused(false) => {
                                let _ = event_loop_proxy
                                    .send_event(UserEvent::RequestCursorLock(false));
                            }
                            _ => (),
                        }
                    }
                    let response = egui_state.on_window_event(&window, &event);
                    if !response.consumed {
                        match event {
                            WindowEvent::Resized(size) => {
                                requested_surface_size = Some(size);
                            }
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            _ => (),
                        }
                    }
                    if let WindowEvent::RedrawRequested = event {
                        if let Some(size) = requested_surface_size.take() {
                            ctx.surface_config.width = size.width;
                            ctx.surface_config.height = size.height;
                            ctx.surface.configure(&ctx.device, &ctx.surface_config);
                            game.resize(&ctx);
                            requested_surface_size = None;
                        }
                        let output = ctx.surface.get_current_texture();
                        let mut encoder =
                            ctx.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("encoder main"),
                                });
                        match output {
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                requested_surface_size = Some(window.inner_size());
                                log::warn!("get_current_texture() Lost/Outdated");
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("get_current_texture() OutOfMemory");
                                elwt.exit();
                            }
                            Err(wgpu::SurfaceError::Timeout) => {
                                log::warn!("get_current_texture() Timeout")
                            }
                            Ok(surface_texture) => {
                                ctx.device.poll(wgpu::Maintain::Wait);

                                let surface_view = surface_texture
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());

                                ctx.profiler.gather_prev_frame_info(&ctx.device);

                                ctx.profiler.begin_frame(&mut encoder);

                                game.update(&ctx, &mut encoder);

                                egui_renderer
                                    .callback_resources
                                    .insert(game.final_draw_resources());

                                let raw_input = egui_state.take_egui_input(&window);
                                let full_output = egui_state.egui_ctx().run(raw_input, |ui_ctx| {
                                    game.ui(ui_ctx, &ctx, &event_loop_proxy);

                                    let response = egui::CentralPanel::default()
                                        .frame(egui::Frame::none())
                                        .show(ui_ctx, |ui| {
                                            let rect = ui.available_rect_before_wrap();

                                            ui.painter().add(
                                                egui_wgpu::Callback::new_paint_callback(
                                                    rect,
                                                    GamePaintCallback {},
                                                ),
                                            );

                                            if cfg!(debug_assertions) {
                                                egui::Frame::none()
                                                    .inner_margin(egui::Margin::same(10.0))
                                                    .show(ui, |ui| {
                                                        ui.with_layout(
                                                            egui::Layout::bottom_up(
                                                                egui::Align::LEFT,
                                                            ),
                                                            |ui| {
                                                                egui::warn_if_debug_build(ui);
                                                            },
                                                        );
                                                    });
                                            }
                                        })
                                        .response
                                        .interact(egui::Sense::click());

                                    if response.clicked() {
                                        let _ = event_loop_proxy
                                            .send_event(UserEvent::RequestCursorLock(true));
                                    }
                                });
                                egui_state
                                    .handle_platform_output(&window, full_output.platform_output);

                                let pixels_per_point = egui_state.egui_ctx().pixels_per_point();

                                let clipped_primitives = egui_state
                                    .egui_ctx()
                                    .tessellate(full_output.shapes, pixels_per_point);
                                for (id, image_delta) in &full_output.textures_delta.set {
                                    egui_renderer.update_texture(
                                        &ctx.device,
                                        &ctx.queue,
                                        *id,
                                        image_delta,
                                    );
                                }

                                let screen_descriptor = egui_wgpu::renderer::ScreenDescriptor {
                                    size_in_pixels: [
                                        ctx.surface_config.width,
                                        ctx.surface_config.height,
                                    ],
                                    pixels_per_point,
                                };

                                egui_renderer.update_buffers(
                                    &ctx.device,
                                    &ctx.queue,
                                    &mut encoder,
                                    &clipped_primitives,
                                    &screen_descriptor,
                                );
                                {
                                    let mut rpass =
                                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: Some("renderpass gui"),
                                            color_attachments: &[Some(
                                                wgpu::RenderPassColorAttachment {
                                                    view: &surface_view,
                                                    resolve_target: None,
                                                    ops: wgpu::Operations {
                                                        load: wgpu::LoadOp::Clear(
                                                            wgpu::Color::BLACK,
                                                        ),
                                                        store: wgpu::StoreOp::Store,
                                                    },
                                                },
                                            )],
                                            depth_stencil_attachment: None,
                                            timestamp_writes: None,
                                            occlusion_query_set: None,
                                        });

                                    egui_renderer.render(
                                        &mut rpass,
                                        &clipped_primitives,
                                        &screen_descriptor,
                                    );
                                }
                                for t in &full_output.textures_delta.free {
                                    egui_renderer.free_texture(t);
                                }

                                ctx.profiler.end_frame(&mut encoder);
                                ctx.queue.submit(Some(encoder.finish()));
                                ctx.profiler.after_submit();
                                surface_texture.present();
                            }
                        }
                    }
                }
                Event::DeviceEvent {
                    event: winit::event::DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    if cursor_locked {
                        game.mouse_motion(delta.0, delta.1);
                    }
                }
                Event::UserEvent(UserEvent::RequestCursorLock(locked)) => {
                    if locked {
                        if window.set_cursor_grab(CursorGrabMode::Locked).is_err()
                            && window.set_cursor_grab(CursorGrabMode::Confined).is_err()
                        {
                            log::error!("Could not grab cursor");
                        } else if !cfg!(target_arch = "wasm32") {
                            let _ = event_loop_proxy
                                .send_event(UserEvent::NotifyCursorLockStatus(true));
                        }
                    } else {
                        window.set_cursor_grab(CursorGrabMode::None).unwrap();
                        let _ =
                            event_loop_proxy.send_event(UserEvent::NotifyCursorLockStatus(false));
                    }
                }
                Event::UserEvent(UserEvent::NotifyCursorLockStatus(locked)) => {
                    if locked != cursor_locked {
                        window.set_cursor_visible(!locked);
                        cursor_locked = locked;
                        game.cursor_lock_update(locked);
                    }
                }
                Event::UserEvent(UserEvent::RequestResize) => {
                    game.resize(&ctx);
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => (),
            }
        })
        .unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn wasm_start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Trace).expect("Failed to initialize logger");
    start().await;
}

#[cfg(target_arch = "wasm32")]
fn add_canvas_to_body(
    window: &winit::window::Window,
    event_loop_proxy: winit::event_loop::EventLoopProxy<UserEvent>,
) {
    use winit::platform::web::WindowExtWebSys;

    let document = web_sys::window()
        .expect("No window")
        .document()
        .expect("No document");
    let body = document.body().expect("No body");

    body.append_child(&window.canvas().unwrap())
        .expect("Failed to append canvas to body");
    use wasm_bindgen::JsCast;
    let closure = wasm_bindgen::closure::Closure::<dyn FnMut(_)>::new(move |_: web_sys::Event| {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");
        let locked = document.pointer_lock_element().is_some();
        let _ = event_loop_proxy.send_event(UserEvent::NotifyCursorLockStatus(locked));
    });
    document
        .add_event_listener_with_callback("pointerlockchange", closure.as_ref().unchecked_ref())
        .expect("Failed to add pointerlockchange event listener");
    closure.forget();
}
