use ca3d::start;
use std::env;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info")
    }
    env_logger::init();
    pollster::block_on(start());
}
