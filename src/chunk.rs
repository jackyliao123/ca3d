use nalgebra_glm as glm;

pub struct ResidencyOffset {
    pub index: u64,  // used by the offset tracker
    pub offset: u32, // offset into shared buffers
}

impl ResidencyOffset {
    pub fn new(index: u64, offset: u32) -> Self {
        Self { index, offset }
    }
}

pub struct Chunk {
    pub pos: glm::I32Vec3,
    pub neighbors: u32,
    pub residency: Option<ResidencyOffset>,
}

impl Chunk {
    pub fn new(pos: glm::I32Vec3) -> Self {
        Self {
            pos,
            residency: None,
            neighbors: 0,
        }
    }

    pub fn offset(&self) -> u32 {
        self.residency
            .as_ref()
            .unwrap_or_else(|| panic!("chunk {:?} offset not tracked", self.pos))
            .offset
    }
}
