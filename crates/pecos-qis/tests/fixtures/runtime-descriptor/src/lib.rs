//! Test-only native descriptor accessor, built by Cargo with the QIS tests.

use std::sync::atomic::{AtomicPtr, Ordering};

/// Library basename used to locate the Cargo-built shared library.
pub const LIBRARY_NAME: &str = "pecos_qis_test_runtime";

static DESCRIPTOR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Install the descriptor used by this test library.
///
/// # Safety
/// The pointer must refer to a valid native runtime descriptor. It and all
/// callback libraries must outlive every runtime using this fixture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_descriptor(value: *mut ()) {
    DESCRIPTOR.store(value, Ordering::SeqCst);
}

/// Return the descriptor installed by the test before plugin initialization.
#[unsafe(no_mangle)]
pub extern "C" fn selene_runtime_get_plugin_descriptor_v1() -> *mut () {
    DESCRIPTOR.load(Ordering::SeqCst)
}
