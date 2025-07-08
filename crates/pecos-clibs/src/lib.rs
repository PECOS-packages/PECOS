//! FFI bindings for C libraries used by PECOS

pub mod rng {
    use std::os::raw::c_double;

    // External C functions from rng_pcg
    unsafe extern "C" {
        pub fn pcg32_random() -> u32;
        pub fn pcg32_boundedrand(bound: u32) -> u32;
        pub fn pcg32_frandom() -> c_double;
        pub fn pcg32_srandom(seq: u64);
    }

    /// Safe wrapper for PCG32 RNG
    pub struct Pcg32Rng {
        _private: (),  // Prevent construction outside this module
    }

    impl Pcg32Rng {
        /// Seed the RNG with a sequence number
        pub fn seed(seq: u64) {
            unsafe { pcg32_srandom(seq) }
        }

        /// Generate a random u32
        pub fn random() -> u32 {
            unsafe { pcg32_random() }
        }

        /// Generate a random u32 in the range [0, bound)
        pub fn bounded_random(bound: u32) -> u32 {
            unsafe { pcg32_boundedrand(bound) }
        }

        /// Generate a random f64 in the range [0.0, 1.0)
        pub fn random_float() -> f64 {
            unsafe { pcg32_frandom() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::rng::Pcg32Rng;

    #[test]
    fn test_pcg32_basic() {
        // Seed the RNG
        Pcg32Rng::seed(42);
        
        // Test basic random generation
        let r1 = Pcg32Rng::random();
        let r2 = Pcg32Rng::random();
        assert_ne!(r1, r2); // Should be different

        // Test bounded random
        let bounded = Pcg32Rng::bounded_random(100);
        assert!(bounded < 100);

        // Test float random
        let f = Pcg32Rng::random_float();
        assert!(f >= 0.0 && f < 1.0);
    }
}