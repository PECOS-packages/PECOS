#[derive(Clone, Copy)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
pub struct PCGRandom {
    pub state: u64,
    inc: u64,
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
}
