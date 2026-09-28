//! Classical RNG exports use the registered shot context.
use crate::ffi::*;
use crate::*;

struct RegisteredContext(*mut ExecutionContext);

impl RegisteredContext {
    fn new() -> Self {
        let ptr = pecos_create_execution_context();
        unsafe { pecos_register_execution_context(ptr) };
        Self(ptr)
    }

    fn context(&self) -> &ExecutionContext {
        unsafe { &*self.0 }
    }
}

impl Drop for RegisteredContext {
    fn drop(&mut self) {
        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(self.0);
        }
    }
}

#[test]
fn selene_reference_stream_and_reseed() {
    let _registered = RegisteredContext::new();
    unsafe {
        for _ in 0..2 {
            random_seed(42);
            assert_eq!(random_int(), 1_085_446_021);
            assert_eq!(random_rng(10), 0);
            assert_eq!(
                random_float().to_bits(),
                0.183_732_153_614_982_96_f64.to_bits()
            );
            random_advance(3);
            assert_eq!(random_int(), 1_312_053_665);
        }
        for _ in 0..2 {
            random_seed_selene(42);
            assert_eq!(random_u32_selene(), 1_085_446_021);
            assert_eq!(random_u32_bounded_selene(10), 0);
            assert_eq!(
                random_f64_selene().to_bits(),
                0.183_732_153_614_982_96_f64.to_bits()
            );
            random_advance_selene(3);
            assert_eq!(random_u32_selene(), 1_312_053_665);
        }
        random_seed(42);
        assert_eq!(random_u32_selene(), 1_085_446_021);
        assert_eq!(random_rng(10), 0);
        assert_eq!(
            random_f64_selene().to_bits(),
            0.183_732_153_614_982_96_f64.to_bits()
        );
        random_advance_selene(3);
        assert_eq!(random_int(), 1_312_053_665);
    }
}

#[test]
fn unseeded_calls_and_invalid_bounds_record_function_names() {
    let registered = RegisteredContext::new();
    let ctx = registered.context();
    let calls: &[(&str, unsafe fn())] = &[
        ("random_int", || unsafe {
            random_int();
        }),
        ("random_rng", || unsafe {
            random_rng(10);
        }),
        ("random_float", || unsafe {
            random_float();
        }),
        ("random_advance", || unsafe {
            random_advance(3);
        }),
        ("random_u32_selene", || unsafe {
            random_u32_selene();
        }),
        ("random_u32_bounded_selene", || unsafe {
            random_u32_bounded_selene(10);
        }),
        ("random_f64_selene", || unsafe {
            random_f64_selene();
        }),
        ("random_advance_selene", || unsafe {
            random_advance_selene(3);
        }),
    ];
    for (entry, call) in calls {
        ctx.reset();
        unsafe { call() };
        assert!(matches!(ctx.program_error.lock().expect("error").as_ref(),
            Some(ProgramError::InvalidInput { entry: actual, detail })
            if actual == entry && detail.contains("random_seed")));
    }
    for bound in [0, -1, i32::MIN] {
        ctx.reset();
        unsafe {
            random_seed(42);
            random_rng(bound);
        }
        assert!(matches!(ctx.program_error.lock().expect("error").as_ref(),
            Some(ProgramError::InvalidInput { entry, .. }) if entry == "random_rng"));
    }
    ctx.reset();
    unsafe {
        random_seed_selene(42);
        random_u32_bounded_selene(0);
    }
    assert!(matches!(ctx.program_error.lock().expect("error").as_ref(),
        Some(ProgramError::InvalidInput { entry, .. }) if entry == "random_u32_bounded_selene"));
}

#[test]
fn every_shot_reset_requires_reseeding() {
    let registered = RegisteredContext::new();
    let ctx = registered.context();
    for reset in [ExecutionContext::reset, ExecutionContext::reset_outputs] {
        unsafe {
            random_seed(42);
            random_int();
        }
        reset(ctx);
        unsafe {
            random_int();
        }
        assert!(ctx.program_error.lock().expect("error").is_some());
        unsafe {
            random_seed(42);
            assert_eq!(random_int(), 1_085_446_021);
        }
    }
    ctx.clear_program_error();
    unsafe {
        pecos_reset_program_rng();
        random_int();
    }
    assert!(ctx.program_error.lock().expect("error").is_some());
}

#[test]
fn signed_draws_rejection_and_jump_ahead() {
    let _registered = RegisteredContext::new();
    unsafe {
        random_seed(42);
        random_advance(4);
        assert_eq!(random_int(), -65_901_028);
        random_advance(-1);
        assert_eq!(random_u32_selene(), 4_229_066_268);
        random_seed(42);
        // Threshold is 2^32 % (2^30 + 1) = 1073741821. After the first
        // draw, rejection skips 176895750 and 789123591.
        assert_eq!(random_rng(1_073_741_825), 11_704_196);
        assert_eq!(random_rng(1_073_741_825), 611_036_920);
        random_seed_selene(42);
        assert_eq!(random_u32_bounded_selene(1 << 31), 1_085_446_021);
        assert_eq!(random_u32_bounded_selene(u32::MAX), 176_895_750);
        random_seed(42);
        random_advance(i64::MIN);
        random_advance(i64::MIN);
        assert_eq!(random_int(), 1_085_446_021);
        random_advance(0);
        assert_eq!(random_int(), 176_895_750);
    }
}

#[test]
fn program_seeds_and_contexts_are_independent() {
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for (seed, expected) in [(42, 1_085_446_021), (43, 1_343_367_557)] {
            let barrier = &barrier;
            scope.spawn(move || {
                let _registered = RegisteredContext::new();
                unsafe { random_seed(seed) };
                barrier.wait();
                unsafe { assert_eq!(random_int(), expected) };
            });
        }
    });
    let _registered = RegisteredContext::new();
    unsafe {
        random_seed(-1);
        assert_eq!(random_u32_selene(), 2_319_459_346);
        random_seed_selene(u64::MAX);
        assert_eq!(random_int(), 2_319_459_346_u32.cast_signed());
    }
}
