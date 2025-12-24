//! Selene QIS Interface and Runtime
//!
//! This crate provides Selene-based implementations of `QisInterface` and `QisRuntime` traits.
//!
//! ## Helios Interface
//!
//! The Helios interface uses Selene's Helios compiler to execute quantum programs. It works by:
//!
//! 1. Linking user program bitcode with Selene's libhelios.a to create an executable
//! 2. Loading the executable in-process using dlopen
//! 3. Providing a shim .so that implements selene_* functions forwarding to PECOS FFI
//! 4. Calling `qmain()` directly to execute the program and collect operations
//!
//! # Architecture
//!
//! ```text
//! user_program.bc + libhelios.a → program.x
//!           ↓
//!       dlopen (in-process)
//!           ↓
//!   program.x calls ___qalloc(), ___rxy(), etc.
//!           ↓
//!   libhelios.a forwards to selene_qalloc(), selene_rxy(), etc.
//!           ↓
//!   libpecos_selene_shim.so implements selene_* functions
//!           ↓
//!   Shim forwards to pecos_qis_ffi::with_interface()
//!           ↓
//!   Operations collected in thread-local storage
//! ```

pub mod builder;
pub mod executor;
pub mod prelude;
pub mod selene_library_runtime;
pub mod selene_runtime;
pub mod selene_runtimes;
pub mod shim;

pub use builder::{HeliosInterfaceBuilder, helios_interface_builder};
pub use executor::QisHeliosInterface;
pub use selene_library_runtime::{
    QisSeleneLibraryRuntime, QisSeleneSimpleRuntime, SeleneRuntimeConfig, selene_library_runtime,
    selene_simple_runtime as selene_simple_runtime_v2, selene_simple_runtime_from_path,
};
pub use selene_runtime::SeleneRuntime;
pub use selene_runtimes::{
    RuntimeFetchError, find_selene_runtime, selene_runtime_auto, selene_simple_runtime,
    selene_soft_rz_runtime,
};

// Re-export pecos_qis_interface to ensure its FFI symbols are included
// when this crate is built as a cdylib
pub use pecos_qis_ffi_types;
