#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
pub struct PCGRandom {
    pub state: u64,
    inc: u64,
}

impl Default for PCGRandom {
    fn default() -> Self {
        Self::init_global_state()
    }
}

impl PCGRandom {
    #[must_use]
    pub fn init_global_state() -> PCGRandom {
        PCGRandom {
            state: 0x853c_49e6_748f_ea9b,
            inc: 0xda3e_39cb_94b9_5bdb,
        }
    }

    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    fn pcg_rotr(value: u32, urot: u32) -> u32 {
        let rot = urot as i32;
        (value >> rot) | (value << ((-rot) & 31))
    }

    #[inline]
    fn pcg_setseq_64_step_r(rng: &mut PCGRandom) {
        const PCG_DEFAULT_MULTIPLIER_64: u64 = 6_364_136_223_846_793_005;
        rng.state = rng
            .state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_64)
            .wrapping_add(rng.inc);
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn pcg_output_xsh(state: u64) -> u32 {
        let value = ((state >> 18) ^ state) >> 27;
        let urot = state >> 59;
        PCGRandom::pcg_rotr(value as u32, urot as u32)
    }

    #[inline]
    pub fn pcg32_random_r(rng: &mut PCGRandom) -> u32 {
        let old_state: u64 = rng.state;
        PCGRandom::pcg_setseq_64_step_r(rng);
        PCGRandom::pcg_output_xsh(old_state)
    }

    #[inline]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn pcg32_boundedrand_r(rng: &mut PCGRandom, ubound: u32) -> u32 {
        let bound: i32 = ubound as i32;
        let threshold: u32 = (-bound % bound) as u32;
        loop {
            let random: u32 = PCGRandom::pcg32_random_r(rng);
            if random >= threshold {
                return random % bound as u32;
            }
        }
    }

    #[inline]
    pub fn frandom(rng: &mut PCGRandom) -> f64 {
        let random = f64::from(PCGRandom::pcg32_random_r(rng));
        let exp: i32 = -32;
        random * 2f64.powi(exp)
    }

    #[inline]
    pub fn pcg32_srandom_r(rng: &mut PCGRandom, initstate: u64, initseq: u64) {
        rng.state = 0_u64;
        rng.inc = (initseq << 1_u64) | 1_u64;
        PCGRandom::pcg_setseq_64_step_r(rng);
        rng.state += initstate;
        PCGRandom::pcg_setseq_64_step_r(rng);
    }

    /// Create a new `PCGRandom` seeded from a u64 value.
    ///
    /// This is a convenience method that creates a new instance and seeds it.
    #[must_use]
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut rng = Self::init_global_state();
        Self::pcg32_srandom_r(&mut rng, seed, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        rng
    }

    /// Generate a random u64 value by combining two u32 values.
    ///
    /// This is more efficient than calling `pcg32_random_r` twice externally
    /// because it avoids extra function call overhead.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = u64::from(Self::pcg32_random_r(self));
        let hi = u64::from(Self::pcg32_random_r(self));
        (hi << 32) | lo
    }

    /// Generate a random u32 value.
    ///
    /// This is a method-style wrapper around `pcg32_random_r`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        Self::pcg32_random_r(self)
    }

    /// Fill a slice with random u64 values efficiently.
    ///
    /// This is optimized for bulk generation by avoiding per-element function calls.
    #[inline]
    pub fn fill_u64(&mut self, dest: &mut [u64]) {
        for val in dest {
            *val = self.next_u64();
        }
    }

    /// Fill a slice with random bytes.
    #[inline]
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        // Process 8 bytes at a time using u64
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let val = self.next_u64();
            chunk.copy_from_slice(&val.to_le_bytes());
        }
        // Handle remaining bytes
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            remainder.copy_from_slice(&bytes[..remainder.len()]);
        }
    }
}

/// PCG64 Fast - A fast, high-quality 64-bit PCG generator.
///
/// This is equivalent to `pcg64_fast` (`Mcg128Xsl64`) from the PCG family.
/// It uses a Multiplicative Congruential Generator (MCG) with 128-bit state
/// and the XSL-RR output function to produce high-quality 64-bit random numbers.
///
/// **Quality**: Passes `BigCrush` and `PractRand`. This is a legitimate, well-tested
/// PCG variant - not a "fast but low quality" generator.
///
/// **Performance**: Faster than standard PCG64 (`Lcg128Xsl64`) because MCG
/// skips the addition step. The trade-off is no stream selection capability,
/// but for most applications a single stream is sufficient.
///
/// Use this as the default choice for fast, high-quality random number generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PCG64Fast {
    state: u128,
}

impl Default for PCG64Fast {
    fn default() -> Self {
        Self::new()
    }
}

impl PCG64Fast {
    /// PCG default multiplier for 128-bit MCG
    const MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    /// Create a new `PCG64Fast` with default seed.
    #[must_use]
    pub fn new() -> Self {
        // State must be odd for MCG
        Self {
            state: 0x979c_9a98_d849_0658_68dc_de48_1b87_85d7, // Note: odd
        }
    }

    /// Create a `PCG64Fast` from a u64 seed.
    #[must_use]
    pub fn seed_from_u64(seed: u64) -> Self {
        // Create a 128-bit state from seed, ensuring it's odd
        let seed128 =
            u128::from(seed) | (u128::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) << 64);
        Self {
            state: seed128 | 1, // Ensure odd
        }
    }

    /// Generate a random u64 value.
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // Intentional: extracting lower bits from u128
    pub fn next_u64(&mut self) -> u64 {
        // MCG step (just multiplication, no addition)
        let old_state = self.state;
        self.state = self.state.wrapping_mul(Self::MULTIPLIER);
        // XSL-RR output
        let rot = (old_state >> 122) as u32;
        let xsl = ((old_state >> 64) as u64) ^ (old_state as u64);
        xsl.rotate_right(rot)
    }

    /// Generate a random u32 value.
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // Intentional: extracting lower 32 bits
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Fill a slice with random u64 values efficiently.
    #[inline]
    pub fn fill_u64(&mut self, dest: &mut [u64]) {
        for val in dest {
            *val = self.next_u64();
        }
    }

    /// Fill a slice with random bytes.
    #[inline]
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let val = self.next_u64();
            chunk.copy_from_slice(&val.to_le_bytes());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            remainder.copy_from_slice(&bytes[..remainder.len()]);
        }
    }

    /// Generate a random f64 in [0, 1).
    #[inline]
    #[allow(clippy::cast_precision_loss)] // Intentional: standard technique for uniform f64 generation
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcg64_fast_generates_values() {
        let mut rng = PCG64Fast::seed_from_u64(42);
        let val1 = rng.next_u64();
        let val2 = rng.next_u64();
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_pcg64_fast_deterministic() {
        let mut rng1 = PCG64Fast::seed_from_u64(12345);
        let mut rng2 = PCG64Fast::seed_from_u64(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_pcg64_fast_different_seeds() {
        let mut rng1 = PCG64Fast::seed_from_u64(1);
        let mut rng2 = PCG64Fast::seed_from_u64(2);
        let vals1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let vals2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn test_pcg64_fast_fill_u64() {
        let mut rng = PCG64Fast::seed_from_u64(42);
        let mut dest = vec![0u64; 100];
        rng.fill_u64(&mut dest);
        let non_zero = dest.iter().filter(|&&x| x != 0).count();
        assert!(non_zero > 95);
    }

    #[test]
    fn test_pcg64_fast_f64_range() {
        let mut rng = PCG64Fast::seed_from_u64(42);
        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "f64 out of range: {f}");
        }
    }
}

#[derive(Default)]
pub struct RNGModel {
    pub rng_gen: PCGRandom,
    pub curr_bound: u32,
    pub count: u64,
}

impl std::fmt::Debug for RNGModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "RNGModel: {:?}; Current rng bound: {} with index count: {}",
            self.rng_gen, self.curr_bound, self.count
        )
    }
}

impl RNGModel {
    pub fn set_seed(&mut self, seed: u64) {
        PCGRandom::pcg32_srandom_r(&mut self.rng_gen, 42_u64, seed);
    }

    /// Advance the RNG to a specific index.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is less than the current count (i.e., the index has already been passed).
    pub fn set_index(&mut self, idx: u64) {
        assert!(
            self.count <= idx,
            "Invalid start index: index {} is before the current stream index: {}",
            idx,
            self.count
        );
        while self.count < idx {
            self.rng_num();
        }
    }

    pub fn rng_num(&mut self) -> u32 {
        self.count += 1;
        if self.curr_bound == 0 {
            PCGRandom::pcg32_random_r(&mut self.rng_gen)
        } else {
            PCGRandom::pcg32_boundedrand_r(&mut self.rng_gen, self.curr_bound)
        }
    }

    pub fn set_bound(&mut self, ubound: u32) {
        self.curr_bound = ubound;
    }
}
