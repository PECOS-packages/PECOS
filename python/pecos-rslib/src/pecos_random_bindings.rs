use crate::prelude::*;
use pyo3::prelude::*;

// use core::prelude::rng_pcg::PCGRandom;

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub struct RngPcg {
    global_state: PCGRandom,
}

impl Default for RngPcg {
    fn default() -> Self {
        Self {
            global_state: PCGRandom::init_global_state(),
        }
    }
}

#[pymethods]
impl RngPcg {
    #[new]
    #[must_use]
    pub fn new() -> RngPcg {
        Self::default()
    }

    pub fn random(&mut self) -> u32 {
        PCGRandom::pcg32_random_r(&mut self.global_state)
    }

    pub fn boundedrand(&mut self, bound: u32) -> u32 {
        PCGRandom::pcg32_boundedrand_r(&mut self.global_state, bound)
    }

    pub fn frandom(&mut self) -> f64 {
        PCGRandom::frandom(&mut self.global_state)
    }

    pub fn srandom(&mut self, seq: u64) {
        PCGRandom::pcg32_srandom_r(&mut self.global_state, 42_u64, seq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcg_functions() {
        let mut pcg = RngPcg::new();
        // Set seed
        pcg.srandom(15);

        // Test basic random
        let r1 = pcg.random();
        assert!(r1 > 0);

        // Test bounded random
        let bound = 100;
        let r2 = pcg.boundedrand(bound);
        assert!(r2 < bound);

        // Test float random
        let r3 = pcg.frandom();
        assert!((0.0..1.0).contains(&r3));
    }
}
