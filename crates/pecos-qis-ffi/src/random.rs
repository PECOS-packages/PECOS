//! Classical PCG32 state belongs to the program execution, independently of
//! quantum simulation seeds. Both QIS and Selene entry points use this state.

use pecos_random::{PCG32_INIT_STATE, PCGRandom};

use crate::{ExecutionContext, ProgramError, ffi::fatal_ffi_input, get_execution_context};

impl ExecutionContext {
    pub(crate) fn reset_program_rng(&self) {
        let result = self.program_rng.lock().map(|mut rng| *rng = None);
        if result.is_err() {
            self.record_program_error(ProgramError::InvalidInput {
                entry: "pecos_reset_program_rng".to_string(),
                detail: "poisoned program RNG".to_string(),
            });
        }
    }
}

/// Clear the program RNG at shot start without changing synchronization state.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_reset_program_rng() {
    if let Some(ctx) = get_execution_context() {
        unsafe { &*ctx }.reset_program_rng();
    }
}

fn with_state<T>(
    operation: impl FnOnce(&mut Option<PCGRandom>) -> Result<T, String>,
) -> Result<T, String> {
    let ctx = get_execution_context().ok_or("no execution context registered")?;
    let mut state = unsafe { &*ctx }
        .program_rng
        .lock()
        .map_err(|_| "poisoned program RNG")?;
    operation(&mut state)
}

fn with_rng<T>(operation: impl FnOnce(&mut PCGRandom) -> T) -> Result<T, String> {
    with_state(|state| {
        let rng = state
            .as_mut()
            .ok_or("random_seed must be called in this shot")?;
        Ok(operation(rng))
    })
}

fn seed(seed: u64) -> Result<(), String> {
    with_state(|state| {
        *state = Some(PCGRandom::from_seed_and_stream(PCG32_INIT_STATE, seed));
        Ok(())
    })
}

fn bounded(bound: u32) -> Result<u32, String> {
    if bound == 0 {
        return Err("bound must be positive".to_string());
    }
    with_rng(|rng| PCGRandom::pcg32_boundedrand_r(rng, bound))
}

fn advance(mut delta: u64) -> Result<(), String> {
    with_rng(|rng| {
        // Compose the LCG's affine transition by exponentiation by squaring.
        // Modulo 2^64 also gives signed deltas their backwards-jump semantics.
        let mut multiplier = 6_364_136_223_846_793_005_u64;
        let mut increment = rng.increment();
        let mut accumulated_multiplier = 1_u64;
        let mut accumulated_increment = 0_u64;
        while delta != 0 {
            if delta & 1 != 0 {
                accumulated_multiplier = accumulated_multiplier.wrapping_mul(multiplier);
                accumulated_increment = accumulated_increment
                    .wrapping_mul(multiplier)
                    .wrapping_add(increment);
            }
            increment = multiplier.wrapping_add(1).wrapping_mul(increment);
            multiplier = multiplier.wrapping_mul(multiplier);
            delta >>= 1;
        }
        rng.state = accumulated_multiplier
            .wrapping_mul(rng.state)
            .wrapping_add(accumulated_increment);
    })
}

// Each operation finishes and releases its mutex before the fatal-input path
// can transfer to C. Unguarded calls record an error and return an ABI placeholder.
macro_rules! export_rng {
    ($name:ident($($arg:ident: $ty:ty),*) -> $return:ty, $operation:expr, $invalid:expr) => {
        /// Access the registered execution context's classical RNG.
        ///
        /// # Safety
        /// With an execution guard installed, the caller's skipped frames must
        /// not own values requiring destruction if invalid input transfers to C.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($($arg: $ty),*) -> $return {
            match $operation {
                Ok(value) => value,
                Err(detail) => {
                    unsafe { fatal_ffi_input(stringify!($name), detail) };
                    $invalid
                }
            }
        }
    };
}

export_rng!(random_seed(seed_value: i64) -> (), seed(seed_value.cast_unsigned()), ());
export_rng!(random_int() -> i32, with_rng(|rng| rng.next_u32().cast_signed()), 0);
export_rng!(random_rng(bound: i32) -> i32,
    u32::try_from(bound).map_err(|_| "bound must be positive".to_string())
        .and_then(bounded).map(u32::cast_signed), 0);
export_rng!(random_float() -> f64, with_rng(PCGRandom::frandom), 0.0);
export_rng!(random_advance(delta: i64) -> (), advance(delta.cast_unsigned()), ());

export_rng!(random_seed_selene(seed_value: u64) -> (), seed(seed_value), ());
export_rng!(random_u32_selene() -> u32, with_rng(PCGRandom::next_u32), 0);
export_rng!(random_u32_bounded_selene(bound: u32) -> u32, bounded(bound), 0);
export_rng!(random_f64_selene() -> f64, with_rng(PCGRandom::frandom), 0.0);
export_rng!(random_advance_selene(delta: u64) -> (), advance(delta), ());
