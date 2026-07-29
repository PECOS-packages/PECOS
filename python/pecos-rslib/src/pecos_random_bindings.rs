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

    /// Seed with any 64-bit integer from Python.
    ///
    /// Accepts:
    /// - signed range: `[-2^63, 2^63-1]`
    /// - unsigned range: `[0, 2^64-1]`
    ///
    /// Negative values are interpreted using two's complement when converted
    /// to the underlying `u64` sequence value.
    pub fn srandom(&mut self, seq: i128) -> PyResult<()> {
        let seq_u64 = normalize_seed(seq)?;
        PCGRandom::pcg32_srandom_r(&mut self.global_state, 42_u64, seq_u64);
        Ok(())
    }

    #[must_use]
    pub fn clone(&self) -> RngPcg {
        *self
    }
}

fn normalize_seed(seq: i128) -> PyResult<u64> {
    if seq >= 0 {
        u64::try_from(seq).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err("srandom seed out of 64-bit range")
        })
    } else {
        i64::try_from(seq).map(|signed| signed as u64).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err("srandom seed out of 64-bit range")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcg_functions() {
        let mut pcg = RngPcg::new();
        // Set seed
        pcg.srandom(15).expect("seed should be in range");

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

    #[test]
    fn test_srandom_accepts_negative_i64() {
        let mut pcg = RngPcg::new();
        pcg.srandom(-1).expect("negative i64 should be accepted");
        let _ = pcg.random();
    }

    #[test]
    fn test_srandom_accepts_u64_max() {
        let mut pcg = RngPcg::new();
        pcg.srandom(i128::from(u64::MAX))
            .expect("u64::MAX should be accepted");
        let _ = pcg.random();
    }

    #[test]
    fn test_srandom_rejects_out_of_range() {
        let mut pcg = RngPcg::new();
        let too_large = i128::from(u64::MAX) + 1;
        assert!(pcg.srandom(too_large).is_err());
    }
}
