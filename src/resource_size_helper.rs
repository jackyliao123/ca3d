pub struct ResourceSizeHelper<T> {
    data: Option<T>,
    created_size: u32,
}

impl<T> Default for ResourceSizeHelper<T> {
    fn default() -> Self {
        Self {
            data: None,
            created_size: 0,
        }
    }
}

impl<T> ResourceSizeHelper<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_or_recreate(&mut self, size_req: u32, create: impl FnOnce(u32) -> T) -> &T {
        let require_recreate = if self.created_size < size_req {
            self.created_size = size_req.next_power_of_two();
            true
        } else {
            false
        };
        match (&mut self.data, require_recreate) {
            (Some(data), false) => data,
            (data, _) => data.insert(create(self.created_size)),
        }
    }

    pub fn get_existing(&self) -> &T {
        self.data.as_ref().unwrap()
    }
}
