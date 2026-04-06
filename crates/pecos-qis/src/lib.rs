//! QIS (Quantum Instruction Set) Infrastructure for PECOS
//!
//! Complete QIS infrastructure for PECOS, including:
//! - `QisInterface` and `QisRuntime` traits for quantum program execution
//! - `QisEngine` - the classical control engine for QIS programs
//! - Selene-based implementations (`QisHeliosInterface`, `SeleneRuntime`)
//!
//! # Architecture
//!
//! The QIS system consists of:
//! - **Interface**: Links and executes quantum programs (e.g., `QisHeliosInterface`)
//! - **Runtime**: Interprets quantum operations (e.g., `SeleneRuntime`)
//! - **Engine**: Orchestrates interface and runtime, implements `ClassicalControlEngine`
//!
//! ## Helios Interface
//!
//! The Helios interface uses Selene's Helios compiler to execute quantum programs:
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
//!
//! # LLVM Setup
//!
//! This crate requires LLVM 14 for QIR (Quantum Intermediate Representation) support.
//!
//! If the build fails, run:
//!
//! ```bash
//! pecos setup
//! cargo build
//! ```
//!
//! This takes ~5 minutes, downloads ~400MB, and installs to `~/.pecos/deps/llvm`.
//!
//! **Don't need QIR?** Disable LLVM:
//! ```toml
//! [dependencies]
//! pecos-qis = { version = "0.1", default-features = false }
//! ```
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use pecos_qis::{qis_engine, selene_simple_runtime, helios_interface_builder};
//! use pecos_engines::ClassicalControlEngineBuilder;
//!
//! // Create a QIS engine with Selene runtime
//! let runtime = selene_simple_runtime().expect("Failed to find Selene runtime");
//! let engine = qis_engine()
//!     .runtime(runtime)
//!     .interface(helios_interface_builder())
//!     .build()
//!     .expect("Failed to build engine");
//! ```

// ============================================================================
// Prelude for common imports
// ============================================================================

pub mod prelude;

// ============================================================================
// Core interface and runtime traits
// ============================================================================

pub mod qis_interface;
pub mod runtime;

pub use qis_interface::{
    BoxedInterface, DynamicSyncHandle, InterfaceError, ProgramFormat, QisInterface,
};

pub use runtime::{
    CallFrame, ClassicalState, QisRuntime, Result as RuntimeResult, RuntimeError, Shot, Value,
};

// ============================================================================
// Engine implementation
// ============================================================================

pub mod ccengine;
#[path = "engine_builder.rs"]
pub mod engine_builder;
pub mod interface_impl;
pub mod program;

pub use ccengine::QisEngine;
pub use engine_builder::{QisEngineBuilder, qis_engine};

pub use program::{
    InterfaceChoice, IntoQisInterface, ProgramType, QisEngineProgram, QisInterfaceBuilder,
    QisInterfaceProvider,
};

// ============================================================================
// Selene implementation (feature-gated, enabled by default)
// ============================================================================

#[cfg(feature = "selene")]
pub mod executor;
#[cfg(feature = "selene")]
#[path = "selene_builder.rs"]
pub mod selene_builder;
#[cfg(feature = "selene")]
pub mod selene_runtime;
#[cfg(feature = "selene")]
pub mod selene_runtimes;
#[cfg(feature = "selene")]
pub mod shim;

#[cfg(feature = "selene")]
pub use executor::{HeliosSyncHandle, QisHeliosInterface};
#[cfg(feature = "selene")]
pub use selene_builder::{HeliosInterfaceBuilder, helios_interface_builder};
#[cfg(feature = "selene")]
pub use selene_runtime::SeleneRuntime;
#[cfg(feature = "selene")]
pub use selene_runtimes::{
    RuntimeFetchError, find_selene_runtime, selene_runtime_auto, selene_simple_runtime,
    selene_soft_rz_runtime,
};

// Re-export pecos_qis_ffi_types for downstream crates
pub use pecos_qis_ffi_types;

// ============================================================================
// Convenience functions
// ============================================================================

use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngine;
use pecos_programs::Qis;
use std::path::Path;

/// Setup a QIS control engine for a program file with an explicit runtime
///
/// This function loads a QIS program from a file and creates a control engine
/// using the provided runtime.
///
/// # Parameters
///
/// - `program_path`: Path to the QIS program file (.ll or .bc)
/// - `runtime`: The QIS runtime to use (e.g., `SeleneRuntime`)
///
/// # Returns
///
/// Returns a boxed `ClassicalControlEngine` on success.
///
/// # Errors
///
/// - `PecosError::IO`: If the program file cannot be read
/// - `PecosError::Processing`: If the engine creation fails
pub fn setup_qis_engine_with_runtime(
    program_path: &Path,
    runtime: impl QisRuntime + 'static,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    use pecos_engines::ClassicalControlEngineBuilder;

    log::debug!("Loading QIS program from: {}", program_path.display());
    // Load the QIS program from file
    let program = Qis::from_file(program_path)?;

    log::debug!("Creating QIS control engine with explicit runtime");
    let builder = qis_engine()
        .runtime(runtime)
        .try_program(program)
        .map_err(|e| PecosError::Processing(format!("Failed to load QIS program: {e}")))?;

    log::debug!("Building engine");
    let engine = builder
        .build()
        .map_err(|e| PecosError::Processing(format!("Failed to build engine: {e}")))?;

    log::debug!("Engine built successfully");
    Ok(Box::new(engine) as Box<dyn ClassicalControlEngine>)
}

/// Setup a QIS control engine for a program file (deprecated)
///
/// **Deprecated**: This function is deprecated because it relied on implicit runtime selection.
/// Use `setup_qis_engine_with_runtime` instead and provide an explicit runtime.
///
/// # Parameters
///
/// - `program_path`: Path to the QIS program file (.ll or .bc)
///
/// # Returns
///
/// Returns an error directing users to use the explicit runtime version.
///
/// # Errors
/// Always returns an error directing users to use `setup_qis_engine_with_runtime` instead.
#[deprecated(
    since = "0.1.1",
    note = "Use setup_qis_engine_with_runtime with an explicit runtime instead"
)]
pub fn setup_qis_engine(
    _program_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    Err(PecosError::Processing(
        "setup_qis_engine is deprecated.\n\
        \n\
        Please use setup_qis_engine_with_runtime and provide an explicit runtime:\n\
        \n\
        use pecos_qis::{setup_qis_engine_with_runtime, selene_simple_runtime};\n\
        \n\
        let engine = setup_qis_engine_with_runtime(path, selene_simple_runtime()?)?;"
            .to_string(),
    ))
}
