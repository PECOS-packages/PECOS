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
//! user_program.bc + libpecos_qis_ffi → program.so (loaded locally)
//!                                         │
//!                                  QIS / selene_* calls
//!                                         ↓
//! in-process Selene QIS plugins → libpecos_qis_ffi (global singleton)
//!                                         ↓
//!                     Operations collected in thread-local storage
//! ```
//! The Helios archive is built for interface tooling; programs do not link it.
//!
//! # LLVM Setup
//!
//! This crate requires LLVM 21.1 for QIR (Quantum Intermediate Representation) support.
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
//! Requires the `selene` feature (enabled by default).
//!
//! ```rust,no_run
//! # #[cfg(feature = "selene")]
//! # {
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
//! # }
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
pub mod scheduled;
mod scheduled_transport;

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
#[cfg(any(feature = "selene", test))]
mod qir_detection;

pub use ccengine::{LoweredQuantumGateTrace, OperationTraceChunk, OperationTraceStore, QisEngine};
pub use engine_builder::{QisEngineBuilder, qis_engine};

pub use program::{InterfaceChoice, IntoQisInterface, QisEngineProgram, QisInterfaceBuilder};

// ============================================================================
// Selene implementation (feature-gated, enabled by default)
// ============================================================================

#[cfg(feature = "selene")]
pub mod executor;
#[cfg(feature = "selene")]
#[path = "selene_builder.rs"]
pub mod selene_builder;
#[cfg(feature = "selene")]
mod selene_native;
#[cfg(feature = "selene")]
pub mod selene_runtime;
#[cfg(feature = "selene")]
pub mod selene_runtimes;

#[cfg(feature = "selene")]
pub use executor::{HeliosSyncHandle, QisHeliosInterface};
#[cfg(feature = "selene")]
pub use selene_builder::{HeliosInterfaceBuilder, helios_interface_builder};
#[cfg(feature = "selene")]
pub use selene_runtime::{
    RuntimeCustomEvent, RuntimeCustomEventDisposition, RuntimeCustomEventPolicy, RuntimeNativeGate,
    RuntimeNativeGateSet, SeleneRuntime,
};
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

/// Create a QIS engine builder preconfigured with the default Selene simple runtime.
///
/// # Errors
///
/// Returns an error if the default Selene simple runtime cannot be located or loaded.
#[cfg(feature = "selene")]
pub fn selene_engine() -> Result<QisEngineBuilder, RuntimeFetchError> {
    Ok(qis_engine()
        .runtime(selene_simple_runtime()?)
        .interface(helios_interface_builder()))
}

/// Create a QIS engine builder preconfigured with a named Selene runtime plugin.
///
/// # Errors
///
/// Returns an error if the requested Selene runtime plugin cannot be located or loaded.
#[cfg(feature = "selene")]
pub fn selene_engine_auto(lib_name: &str) -> Result<QisEngineBuilder, RuntimeFetchError> {
    Ok(qis_engine()
        .runtime(selene_runtime_auto(lib_name)?)
        .interface(helios_interface_builder()))
}

/// Create a QIS engine builder preconfigured with the Selene soft-RZ runtime.
///
/// # Errors
///
/// Returns an error if the Selene soft-RZ runtime cannot be located or loaded.
#[cfg(feature = "selene")]
pub fn selene_soft_rz_engine() -> Result<QisEngineBuilder, RuntimeFetchError> {
    Ok(qis_engine()
        .runtime(selene_soft_rz_runtime()?)
        .interface(helios_interface_builder()))
}

#[cfg(all(test, feature = "selene"))]
pub(crate) mod test_env {
    use std::ffi::{OsStr, OsString};
    use std::sync::Mutex;

    // Environment variables are process-wide. Every test in this crate that
    // reads or mutates environment state must hold this lock.
    pub(crate) static ENV_MUTEX: Mutex<()> = Mutex::new(());

    pub(crate) struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        pub(crate) fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: Environment-mutating tests hold ENV_MUTEX for the guard's lifetime.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: Environment-mutating tests hold ENV_MUTEX for the guard's lifetime.
            unsafe {
                if let Some(previous) = &self.previous {
                    std::env::set_var(self.key, previous);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    pub(crate) struct TestChild {
        process: std::process::Child,
        stdout: Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>,
        stderr: Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    }

    impl TestChild {
        pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
            self.process.try_wait()
        }

        pub(crate) fn kill_and_wait(&mut self) -> std::process::ExitStatus {
            self.process.kill().expect("kill child");
            self.process.wait().expect("observe child exit")
        }
    }

    impl Drop for TestChild {
        fn drop(&mut self) {
            // Also reap a holder if its parent's assertion or barrier fails.
            let _ = self.process.kill();
            let _ = self.process.wait();
        }
    }

    fn drain_child_pipe(
        mut pipe: impl std::io::Read + Send + 'static,
    ) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            pipe.read_to_end(&mut bytes)?;
            Ok(bytes)
        })
    }

    /// Spawn with a stable inherited environment, releasing `ENV_MUTEX` before waiting.
    pub(crate) fn spawn_test_child(test_name: &str, envs: &[(&str, &OsStr)]) -> TestChild {
        let mut process = {
            let _env_lock = ENV_MUTEX
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::process::Command::new(std::env::current_exe().expect("test executable"))
                .args(["--exact", test_name, "--nocapture"])
                .envs(envs.iter().copied())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .expect("spawn child")
        };
        // Drain immediately: callers may wait for a child barrier before joining.
        let stdout = drain_child_pipe(process.stdout.take().expect("stdout pipe"));
        let stderr = drain_child_pipe(process.stderr.take().expect("stderr pipe"));
        TestChild {
            process,
            stdout: Some(stdout),
            stderr: Some(stderr),
        }
    }

    pub(crate) fn finish_test_child(
        mut child: TestChild,
        budget: std::time::Duration,
    ) -> Result<std::process::Output, String> {
        let watchdog = std::time::Instant::now();
        let (status, timed_out) = loop {
            if let Some(status) = child.process.try_wait().expect("child status") {
                break (status, false);
            }
            if watchdog.elapsed() >= budget {
                break (child.kill_and_wait(), true);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        let stdout = child
            .stdout
            .take()
            .expect("stdout reader")
            .join()
            .expect("stdout thread")
            .expect("read stdout");
        let stderr = child
            .stderr
            .take()
            .expect("stderr reader")
            .join()
            .expect("stderr thread")
            .expect("read stderr");
        let output = std::process::Output {
            status,
            stdout,
            stderr,
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if timed_out {
            return Err(format!(
                "child watchdog expired ({status}): {stdout}\n{stderr}"
            ));
        }
        if !status.success() {
            return Err(format!("child failed ({status}): {stdout}\n{stderr}"));
        }
        if !stdout.contains("test result: ok. 1 passed") {
            return Err(format!(
                "child did not run exactly one test ({status}): {stdout}\n{stderr}"
            ));
        }
        Ok(output)
    }

    pub(crate) fn join_test_child(child: TestChild) {
        finish_test_child(child, std::time::Duration::from_secs(60))
            .unwrap_or_else(|error| panic!("{error}"));
    }

    /// Run one test of this binary in a child process, for tests that must change
    /// process-wide state that concurrently running tests read.
    pub(crate) fn run_test_in_child(test_name: &str, envs: &[(&str, &OsStr)]) {
        join_test_child(spawn_test_child(test_name, envs));
    }
}
