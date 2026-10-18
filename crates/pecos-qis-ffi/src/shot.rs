//! Host-owned program identity, independent of native runtime and RNG state.
use crate::{ExecutionContext, ffi::fatal_ffi_input, get_execution_context};
use std::sync::atomic::Ordering;

/// Assign the program-visible shot before executing the program.
///
/// # Safety
/// A non-null pointer must refer to a live context owned by the caller. Call only
/// while no program is executing in that context. Null returns an error status.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_set_program_shot_with_context(
    ctx: *mut ExecutionContext,
    shot: u64,
) -> i32 {
    let Some(ctx) = (unsafe { ctx.as_ref() }) else {
        return 1;
    };
    ctx.program_shot.store(shot, Ordering::SeqCst);
    0
}

/// Return the current job-wide program shot index.
///
/// # Safety
/// Execution must follow the non-nested guard/recovery contract. With a guard,
/// a missing context may transfer control without running Rust destructors.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn get_current_shot() -> u64 {
    if let Some(ctx) = get_execution_context() {
        unsafe { &*ctx }.program_shot.load(Ordering::SeqCst)
    } else {
        unsafe { fatal_ffi_input("get_current_shot", "no execution context registered".into()) };
        0
    }
}
