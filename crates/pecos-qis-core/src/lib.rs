//! QIS Classical Control Engine
//!
//! This crate provides the orchestration between `QisInterface` (linked programs)
//! and `QisRuntime` (interpreters), implementing `ClassicalControlEngine` for PECOS integration.
//!
//! The reference runtime implementation is:
//! - `SeleneRuntime`: Selene-based QIS runtime (in pecos-qis-selene crate)
//!
//! # LLVM Setup
//!
//! This crate requires LLVM 14 for QIR (Quantum Intermediate Representation) support.
//!
//! If the build fails, just run the commands shown in the error message. Typically:
//!
//! ```bash
//! cargo run -p pecos -- llvm install
//! export PECOS_LLVM=$(cargo run -p pecos -- llvm find)
//! export LLVM_SYS_140_PREFIX="$PECOS_LLVM"
//! cargo build
//! ```
//!
//! This takes ~5 minutes, downloads ~400MB, and installs to `~/.pecos/llvm`.
//!
//! **`PECOS_LLVM` vs `LLVM_SYS_140_PREFIX`**: Use `PECOS_LLVM` to specify which LLVM installation
//! PECOS should use. This takes priority over system-wide `LLVM_SYS_140_PREFIX`, allowing
//! you to use a different LLVM for PECOS than other projects.
//!
//! **Don't need QIR?** Disable LLVM:
//! ```toml
//! [dependencies]
//! pecos-qis-core = { version = "0.1", default-features = false }
//! ```
//!
//! See the [Getting Started guide](https://quantum-pecos.readthedocs.io/) for more details.
//!
//! # Example Usage
//!
//! This crate provides the core builder API for QIS engines. Specific runtime
//! implementations are provided by other crates (e.g., `pecos-qis-selene`).
//!
//! ```rust
//! use pecos_qis_core::qis_engine;
//! use pecos_qis_ffi_types::{OperationCollector, QuantumOp};
//!
//! // Create an interface with quantum operations
//! let mut interface = OperationCollector::new();
//! let q0 = interface.allocate_qubit();
//! interface.operations.push(QuantumOp::H(q0).into());
//!
//! // Create a builder (requires a runtime to build)
//! let builder = qis_engine().with_interface(interface.clone());
//!
//! // For complete examples with runtime, see the pecos-qis-selene crate
//! assert_eq!(interface.allocated_qubits.len(), 1);
//! ```
//!
//! # Builder API
//!
//! The QIS engine builder follows the standard PECOS builder pattern.
//! This example shows the API structure:
//!
//! ```rust
//! use pecos_qis_core::qis_engine;
//! use pecos_qis_ffi_types::{OperationCollector, QuantumOp};
//!
//! // Create a Bell state program
//! let mut interface = OperationCollector::new();
//! let q0 = interface.allocate_qubit();
//! let q1 = interface.allocate_qubit();
//! interface.operations.push(QuantumOp::H(q0).into());
//! interface.operations.push(QuantumOp::CX(q0, q1).into());
//!
//! // Create the builder (requires adding .runtime() and calling .build() to execute)
//! let builder = qis_engine().with_interface(interface.clone());
//!
//! // Verify the interface structure
//! assert_eq!(interface.allocated_qubits.len(), 2);
//! assert_eq!(interface.operations.len(), 2);
//! ```
//!
//! For more on Selene-based runtimes and interfaces (LLVM execution), see the
//! `pecos-qis-selene` crate.

pub mod builder;
pub mod ccengine;
pub mod interface_impl;
pub mod prelude;
pub mod program;
pub mod qis_interface;
pub mod runtime;

pub use builder::{QisEngineBuilder, qis_engine};
pub use ccengine::QisEngine;

// Re-export QisInterface trait and related types
pub use interface_impl::SimpleQisInterface;
pub use qis_interface::{BoxedInterface, InterfaceError, ProgramFormat, QisInterface};

pub use program::{
    InterfaceChoice, IntoQisInterface, ProgramType, QisEngineProgram, QisInterfaceBuilder,
    QisInterfaceProvider,
};

// Re-export the runtime trait and types for convenience
pub use runtime::{
    CallFrame, ClassicalState, QisRuntime, Result as RuntimeResult, RuntimeError, Shot, Value,
};

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
/// - `runtime`: The QIS runtime to use (e.g., `SeleneRuntime` from pecos-qis-selene)
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
/// This function attempts to load the program with the default Helios interface
/// and requires a runtime to be available. Since runtime selection is environment-dependent,
/// callers should use the explicit version.
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
        use pecos_qis_core::setup_qis_engine_with_runtime;\n\
        use pecos_qis_selene::selene_simple_runtime;\n\
        \n\
        let engine = setup_qis_engine_with_runtime(path, selene_simple_runtime()?)?;"
            .to_string(),
    ))
}
