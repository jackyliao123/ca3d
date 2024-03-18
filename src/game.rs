use std::rc::Rc;
use std::sync::Arc;

use egui::Widget;
use nalgebra_glm as glm;
use rand::{thread_rng, Rng};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::EventLoopProxy;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::chunk::Chunk;
use crate::chunk_manager::ChunkManager;
use crate::gpu_stage::bloom::Bloom;
use crate::gpu_stage::meshing_render::{Meshing, Render};
use crate::gpu_stage::overlay::Overlay;
use crate::gpu_stage::picker::Picker;
use crate::gpu_stage::simulate::Simulate;
use crate::gpu_stage::tonemap::Tonemap;
use crate::key_tracker::KeyTracker;
use crate::user_event::UserEvent;
use crate::util::RenderTargetInfo;
use crate::wgpu_context::WgpuContext;
use crate::FinalDrawResources;

pub struct Game {
    position: glm::Vec3,
    projection: glm::Mat4,
    look: glm::Vec2,
    look_sensitivity: f32,
    speed: f32,
    fov: f32,

    key_tracker: KeyTracker,
    show_debug_window: bool,
    show_render_options: bool,
    show_profiler: bool,

    chunk_manager: ChunkManager,

    pub simulate: Simulate,
    pub meshing: Meshing,
    pub render: Render,
    pub picker: Picker,
    pub overlay: Overlay,
    pub bloom: Bloom,
    pub tonemap: Tonemap,
}

impl Game {
    pub fn new(ctx: &WgpuContext) -> Self {
        let chunk_manager = ChunkManager::new(ctx);

        let tonemap = Tonemap::new(ctx, Rc::new(RenderTargetInfo::from(ctx)));
        let bloom = Bloom::new(ctx, tonemap.input_target());
        let overlay = Overlay::new(ctx, bloom.input_target());
        let picker = Picker::new(ctx, overlay.input_target());
        let render = Render::new(ctx, picker.input_target());
        let meshing = Meshing::new(ctx, &chunk_manager);
        let simulate = Simulate::new(ctx, &chunk_manager);

        let mut game = Self {
            position: glm::vec3(80.0, 80.0, 80.0),
            projection: glm::identity(),
            look: glm::vec2(-45.0, 45.0),
            look_sensitivity: 0.1,
            speed: 0.1,
            fov: 90.0,

            key_tracker: KeyTracker::new(),
            show_debug_window: false,
            show_render_options: false,
            show_profiler: false,

            chunk_manager,

            simulate,
            meshing,
            render,
            picker,
            overlay,
            bloom,
            tonemap,
        };

        let mut rng = thread_rng();

        let mut blocks = vec![0u32; 64 * 64 * 64];

        let init_size = 2;

        for cx in 0..init_size {
            for cy in 0..init_size {
                for cz in 0..init_size {
                    let pos = glm::vec3(cx, cy, cz);

                    let chunk = Chunk::new(pos);
                    game.chunk_manager.add_chunk(chunk);
                }
            }
        }
        game.chunk_manager.finalize_changes_and_start_frame(ctx);
        for x in 0..64 {
            for z in 0..64 {
                for y in 0..64 {
                    if rng.gen_range(0..10000) == 0 {
                        blocks[x + y * 64 + z * 64 * 64] = rng.gen();
                    } else {
                        blocks[x + y * 64 + z * 64 * 64] = 0;
                    }
                }
            }
        }

        for cx in 0..init_size {
            for cy in 0..init_size {
                for cz in 0..init_size {
                    let pos = glm::vec3(cx, cy, cz);

                    game.chunk_manager.upload_chunk_data(ctx, pos, &blocks);
                }
            }
        }

        game
    }

    pub fn update(
        &mut self,
        ctx: &WgpuContext,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut rel_movement = glm::vec3(0.0, 0.0, 0.0);
        if self.key_tracker.is_key_pressed(KeyCode::KeyW) {
            rel_movement.z -= 1.0;
        }
        if self.key_tracker.is_key_pressed(KeyCode::KeyS) {
            rel_movement.z += 1.0;
        }
        if self.key_tracker.is_key_pressed(KeyCode::KeyA) {
            rel_movement.x -= 1.0;
        }
        if self.key_tracker.is_key_pressed(KeyCode::KeyD) {
            rel_movement.x += 1.0;
        }
        if self.key_tracker.is_key_pressed(KeyCode::Space) {
            rel_movement.y += 1.0;
        }
        if self.key_tracker.is_key_pressed(KeyCode::ShiftLeft) {
            rel_movement.y -= 1.0;
        }

        let abs_movement = glm::rotate_y_vec3(
            &glm::vec3(rel_movement.x, 0.0, rel_movement.z),
            self.look.y.to_radians(),
        ) + glm::vec3(0.0, rel_movement.y, 0.0);

        self.position += abs_movement * self.speed;

        self.projection = glm::reversed_infinite_perspective_rh_zo(
            ctx.surface_config.width as f32 / ctx.surface_config.height as f32,
            self.fov.to_radians(),
            0.1,
        );
        let view: glm::Mat4 = glm::identity();
        let view = glm::rotate_x(&view, -self.look.x.to_radians());
        let view = glm::rotate_y(&view, -self.look.y.to_radians());
        let view = glm::translate(&view, &-self.position);

        let mvp = self.projection * view;

        self.chunk_manager.finalize_changes_and_start_frame(ctx);
        ctx.profiler.profile(encoder, "simulate", |encoder| {
            self.simulate.update(ctx, encoder, &mut self.chunk_manager);
        });

        let meshing_result = ctx.profiler.profile(encoder, "meshing", |encoder| {
            self.meshing.update(ctx, encoder, &self.chunk_manager)
        });

        ctx.profiler.profile(encoder, "render", |encoder| {
            self.render
                .update(ctx, encoder, &self.chunk_manager, meshing_result, &mvp);
        });

        ctx.profiler.profile(encoder, "picker", |encoder| {
            self.picker.update(ctx, encoder);
        });

        ctx.profiler.profile(encoder, "overlay", |encoder| {
            self.overlay.update(ctx, encoder, &self.projection, &view);
        });

        ctx.profiler.profile(encoder, "bloom", |encoder| {
            self.bloom.update(ctx, encoder);
        });

        ctx.profiler.profile(encoder, "tonemap", |_encoder| {
            self.tonemap.update(ctx);
        });

        vec![]
    }

    pub fn final_draw_resources(&self) -> Arc<FinalDrawResources> {
        self.tonemap.final_draw_resources()
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        self.look.y -= dx as f32 * self.look_sensitivity;
        self.look.x -= dy as f32 * self.look_sensitivity;
        if self.look.x > 90.0 {
            self.look.x = 90.0;
        }
        if self.look.x < -90.0 {
            self.look.x = -90.0;
        }
    }

    pub fn resize(&mut self, ctx: &WgpuContext) {
        self.tonemap
            .resize(ctx, Rc::new(RenderTargetInfo::from(ctx)));
        self.bloom.resize(ctx, self.tonemap.input_target());
        self.overlay.resize(ctx, self.bloom.input_target());
        self.picker.resize(ctx, self.overlay.input_target());
        self.render.resize(ctx, self.picker.input_target());
    }

    pub fn input(&mut self, event: &WindowEvent, event_loop_proxy: &EventLoopProxy<UserEvent>) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state,
                        ..
                    },
                ..
            } => {
                if *state == ElementState::Pressed {
                    self.key_tracker.key_down(*key_code);
                    match *key_code {
                        KeyCode::Escape => {
                            let _ =
                                event_loop_proxy.send_event(UserEvent::RequestCursorLock(false));
                        }
                        KeyCode::KeyI => {
                            self.simulate.step = 1;
                        }
                        KeyCode::KeyP => {
                            self.simulate.paused = !self.simulate.paused;
                        }
                        _ => {}
                    }
                } else {
                    self.key_tracker.key_up(*key_code);
                }
            }
            WindowEvent::MouseWheel {
                delta: winit::event::MouseScrollDelta::LineDelta(_, y),
                ..
            } => {
                self.speed *= 1.0 + y / 100.0;
                self.speed = self.speed.clamp(0.0001, 10000.0);
            }
            _ => {}
        }
    }

    pub fn cursor_lock_update(&mut self, locked: bool) {
        if !locked {
            self.key_tracker.reset();
        }
    }

    pub fn ui(
        &mut self,
        ctx: &egui::Context,
        wgpu_ctx: &WgpuContext,
        event_loop_proxy: &EventLoopProxy<UserEvent>,
    ) {
        egui::TopBottomPanel::top("menubar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                let is_web = cfg!(target_arch = "wasm32");
                ui.menu_button("File", |ui| {
                    if !is_web {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                });
                ui.menu_button("View", |ui| {
                    egui::widgets::global_dark_light_mode_buttons(ui);
                    egui::widgets::Checkbox::new(&mut self.show_debug_window, "Debug window")
                        .ui(ui);
                    egui::widgets::Checkbox::new(&mut self.show_render_options, "Render options")
                        .ui(ui);
                    egui::widgets::Checkbox::new(&mut self.show_profiler, "Profiler").ui(ui);
                });
            });
        });

        egui::Window::new("Debug")
            .open(&mut self.show_debug_window)
            .show(ctx, |ui| {
                egui::collapsing_header::CollapsingHeader::new("Settings").show(ui, |ui| {
                    ctx.settings_ui(ui);
                });
                egui::collapsing_header::CollapsingHeader::new("Inspection").show(ui, |ui| {
                    ctx.inspection_ui(ui);
                });
                egui::collapsing_header::CollapsingHeader::new("Memory").show(ui, |ui| {
                    ctx.memory_ui(ui);
                });
            });

        egui::Window::new("Render options")
            .open(&mut self.show_render_options)
            .show(ctx, |ui| {
                self.simulate.ui(ui, event_loop_proxy);
                self.bloom.ui(ui, event_loop_proxy);
                self.tonemap.ui(ui, event_loop_proxy);
            });

        egui::Window::new("Profiler")
            .open(&mut self.show_profiler)
            .show(ctx, |ui| {
                wgpu_ctx.profiler.ui(ui);
            });
    }

    pub fn after_submit(&self) {
        self.picker.after_submit();
    }
}
