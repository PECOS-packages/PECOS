//! Minimal QIS Interface for Fast Linking
//!
//! This crate provides the minimal FFI interface needed to link QIS (Quantum Instruction Set)
//! programs with Rust functions. It's designed to be lightweight and compile quickly.
//!
//! The interface collects quantum operations during program execution without performing
//! any simulation or complex state management. These operations are later processed by
//! a `QisRuntime` implementation.

use std::cell::RefCell;

pub mod ffi;

// Re-export all types from pecos-qis-ffi-types
pub use pecos_qis_ffi_types::{Operation, OperationCollector, OperationList, QuantumOp};

thread_local! {
    /// Thread-local storage for the current operation collector
    static INTERFACE: RefCell<OperationCollector> = RefCell::new(OperationCollector::new());
}

/// Get the thread-local operation collector
pub fn with_interface<F, R>(f: F) -> R
where
    F: FnOnce(&mut OperationCollector) -> R,
{
    INTERFACE.with(|interface| f(&mut interface.borrow_mut()))
}

/// Reset the thread-local operation collector
pub fn reset_interface() {
    with_interface(OperationCollector::reset);
}

/// Get a clone of the thread-local operation collector
#[must_use]
pub fn get_interface_clone() -> OperationCollector {
    with_interface(|interface| interface.clone())
}

/// Set measurement results in the thread-local operation collector
pub fn set_measurements(measurements: impl IntoIterator<Item = (usize, bool)>) {
    with_interface(|interface| interface.set_measurement_results(measurements));
}
