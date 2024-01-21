use std::collections::HashSet;
use winit::keyboard::KeyCode;

pub struct KeyTracker {
    keys_pressed: HashSet<KeyCode>,
}

impl KeyTracker {
    pub fn new() -> Self {
        Self {
            keys_pressed: HashSet::new(),
        }
    }

    pub fn key_down(&mut self, key: KeyCode) {
        self.keys_pressed.insert(key);
    }

    pub fn key_up(&mut self, key: KeyCode) {
        self.keys_pressed.remove(&key);
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn reset(&mut self) {
        self.keys_pressed.clear();
    }
}
