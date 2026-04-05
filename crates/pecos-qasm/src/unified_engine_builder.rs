//! Unified engine builder for QASM that integrates with the common simulation API
//!
//! This module provides the engine builder that implements the `ClassicalControlEngineBuilder`
//! trait from pecos-engines, enabling the unified simulation API.

use crate::engine::QASMEngine;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngineBuilder;
use pecos_programs::Qasm;
#[cfg(feature = "wasm")]
use pecos_programs::{Wasm, Wat};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Builder for QASM engines that integrates with the unified simulation API
#[derive(Debug, Clone, Default)]
pub struct QasmEngineBuilder {
    /// The QASM source (either string or file path)
    source: Option<QasmSource>,
    /// Virtual includes to use (filename -> content)
    virtual_includes: Vec<(String, String)>,
    /// Additional search paths for include files
    include_paths: Vec<String>,
    /// When true, allows general expressions in if statements
    allow_complex_conditionals: bool,
    /// WebAssembly program for foreign function calls
    #[cfg(feature = "wasm")]
    wasm_program: Option<crate::QasmEngineWasm>,
}

#[derive(Debug, Clone)]
enum QasmSource {
    /// QASM string content
    String(String),
    /// Path to QASM file
    File(PathBuf),
}

/// Trait for types that can be converted to a WASM program
#[cfg(feature = "wasm")]
pub trait IntoWasm {
    /// Convert to a `QasmEngineWasm`
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails
    fn into_wasm_program(self) -> Result<crate::QasmEngineWasm, PecosError>;
}

#[cfg(feature = "wasm")]
impl IntoWasm for Wasm {
    fn into_wasm_program(self) -> Result<crate::QasmEngineWasm, PecosError> {
        Ok(self.into())
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for Wat {
    fn into_wasm_program(self) -> Result<crate::QasmEngineWasm, PecosError> {
        use std::convert::TryInto;
        self.try_into()
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for crate::QasmEngineWasm {
    fn into_wasm_program(self) -> Result<crate::QasmEngineWasm, PecosError> {
        Ok(self)
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for String {
    fn into_wasm_program(self) -> Result<crate::QasmEngineWasm, PecosError> {
        // Load from file path
        let bytes = std::fs::read(&self)
            .map_err(|e| PecosError::Input(format!("Failed to read WASM file '{self}': {e}")))?;
        Ok(crate::QasmEngineWasm::from_bytes(bytes).with_source_path(self))
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for &str {
    fn into_wasm_program(self) -> Result<crate::QasmEngineWasm, PecosError> {
        self.to_string().into_wasm_program()
    }
}

impl QasmEngineBuilder {
    /// Create a new QASM engine builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the QASM source from a string
    #[must_use]
    pub fn qasm(mut self, qasm: impl Into<String>) -> Self {
        self.source = Some(QasmSource::String(qasm.into()));
        self
    }

    /// Set the QASM source from a file path
    #[must_use]
    pub fn qasm_file(mut self, path: impl AsRef<Path>) -> Self {
        self.source = Some(QasmSource::File(path.as_ref().to_path_buf()));
        self
    }

    /// Set the QASM source from a `Qasm`
    #[must_use]
    pub fn program(mut self, program: impl Into<Qasm>) -> Self {
        let program = program.into();
        self.source = Some(QasmSource::String(program.source));
        self
    }

    /// Add a virtual include (filename -> content)
    #[must_use]
    pub fn with_virtual_include(mut self, filename: &str, content: &str) -> Self {
        self.virtual_includes
            .push((filename.to_string(), content.to_string()));
        self
    }

    /// Add multiple virtual includes
    #[must_use]
    pub fn with_virtual_includes(mut self, includes: &[(&str, &str)]) -> Self {
        for (filename, content) in includes {
            self.virtual_includes
                .push(((*filename).to_string(), (*content).to_string()));
        }
        self
    }

    /// Add an include search path
    #[must_use]
    pub fn with_include_path(mut self, path: &str) -> Self {
        self.include_paths.push(path.to_string());
        self
    }

    /// Add multiple include search paths
    #[must_use]
    pub fn with_include_paths(mut self, paths: &[&str]) -> Self {
        for path in paths {
            self.include_paths.push((*path).to_string());
        }
        self
    }

    /// Enable or disable complex conditionals
    #[must_use]
    pub fn allow_complex_conditionals(mut self, allow: bool) -> Self {
        self.allow_complex_conditionals = allow;
        self
    }

    /// Check if this builder has a QASM source configured
    #[must_use]
    pub fn has_source(&self) -> bool {
        self.source.is_some()
    }

    /// Get the `Qasm` from this builder (if any)
    #[must_use]
    pub fn get_program(&self) -> Option<pecos_programs::Qasm> {
        match &self.source {
            Some(QasmSource::String(content)) => {
                Some(pecos_programs::Qasm::from_string(content.clone()))
            }
            Some(QasmSource::File(path)) => pecos_programs::Qasm::from_file(path).ok(),
            None => None,
        }
    }

    /// Set the WebAssembly program for foreign function calls
    ///
    /// This method accepts:
    /// - `Wasm` - pre-loaded WASM binary
    /// - `Wat` - WebAssembly text format (parsed by wasmtime)
    /// - `QasmEngineWasm` - engine-specific WASM program
    /// - `&str` or `String` - path to a .wasm or .wat file
    #[cfg(feature = "wasm")]
    #[must_use]
    pub fn wasm(mut self, wasm: impl IntoWasm) -> Self {
        match wasm.into_wasm_program() {
            Ok(program) => {
                self.wasm_program = Some(program);
            }
            Err(e) => {
                // Store error for later reporting during build
                log::warn!("Failed to load WASM program: {e}");
            }
        }
        self
    }
}

impl ClassicalControlEngineBuilder for QasmEngineBuilder {
    type Engine = QASMEngine;

    /// Build the QASM engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No QASM source was specified
    /// - Failed to read QASM file from disk
    /// - Failed to parse QASM content
    /// - WASM module initialization failed
    /// - WASM module is missing required exports
    fn build(self) -> Result<Self::Engine, PecosError> {
        // Get the QASM content
        let qasm_content = match self.source {
            Some(QasmSource::String(s)) => s,
            Some(QasmSource::File(path)) => std::fs::read_to_string(&path)
                .map_err(|e| PecosError::Input(format!("Failed to read QASM file: {e}")))?,
            None => {
                return Err(PecosError::Input(
                    "No QASM source specified. Use .qasm() or .qasm_file()".to_string(),
                ));
            }
        };

        // Create the engine using FromStr
        let mut engine = QASMEngine::from_str(&qasm_content)?;

        // Apply configuration
        if self.allow_complex_conditionals {
            engine.allow_complex_conditionals(true);
        }

        // Handle WASM foreign object if specified
        #[cfg(feature = "wasm")]
        if let Some(wasm_program) = self.wasm_program {
            use crate::foreign_objects::ForeignObject;
            use crate::program::QASMProgram;
            use crate::wasm_foreign_object::WasmtimeForeignObject;

            // Create the WASM foreign object from bytes
            let wasm_obj = WasmtimeForeignObject::from_bytes(&wasm_program.wasm_bytes)?;

            // Get exported functions from WASM module
            let exported_functions = wasm_obj.get_funcs();

            // Check if init function exists
            if !exported_functions.contains(&"init".to_string()) {
                return Err(PecosError::Input(
                    "WebAssembly module must export an 'init' function".to_string(),
                ));
            }

            // Parse the QASM program to extract function calls
            let program = QASMProgram::from_str(&qasm_content)?;
            let non_builtin_calls = program.get_non_builtin_function_calls();

            // Validate that all non-builtin function calls exist in WASM module
            for func_name in non_builtin_calls {
                if !exported_functions.contains(&func_name) {
                    return Err(PecosError::Input(format!(
                        "Function '{func_name}' is called in QASM but not exported by WebAssembly module. Available functions: {exported_functions:?}"
                    )));
                }
            }

            // Initialize the WASM module
            let mut boxed_obj: Box<dyn ForeignObject> = Box::new(wasm_obj);
            boxed_obj.init()?;

            // Set the foreign object on the engine
            engine.set_foreign_object(boxed_obj);
        }

        // Note: virtual_includes and include_paths would need to be handled
        // during parsing, which happens in from_str. This is a limitation
        // of the current design that could be addressed in the future.

        Ok(engine)
    }
}

impl From<Qasm> for QasmEngineBuilder {
    fn from(program: Qasm) -> Self {
        Self::new().program(program)
    }
}

/// Create a new QASM engine builder
///
/// This is the entry point for the unified simulation API.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use pecos_qasm::unified_engine_builder::qasm_engine;
/// use pecos_engines::{ClassicalControlEngineBuilder, DepolarizingNoise};
///
/// // Basic usage
/// let results = qasm_engine()
///     .qasm("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;")
///     .to_sim()
///     .seed(42)
///     .noise(DepolarizingNoise { p: 0.01 })
///     .run(1000)?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn qasm_engine() -> QasmEngineBuilder {
    QasmEngineBuilder::new()
}
