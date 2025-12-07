// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! PHIR JSON engine builder following the unified simulation API pattern

use crate::common::{PhirJsonVersion, detect_version};
use crate::v0_1::engine::PhirJsonEngine;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngineBuilder;
use pecos_programs::PhirJson;
use std::path::{Path, PathBuf};

/// Engine-specific PHIR program that stores the validated JSON and version
#[derive(Debug, Clone)]
pub struct PhirJsonEngineProgram {
    json_content: String,
    version: PhirJsonVersion,
}

impl PhirJsonEngineProgram {
    /// Create from a JSON string, detecting and validating the version
    ///
    /// # Errors
    ///
    /// Returns an error if version detection fails
    pub fn from_json(json: &str) -> Result<Self, PecosError> {
        let version = detect_version(json)?;
        Ok(Self {
            json_content: json.to_string(),
            version,
        })
    }

    /// Get the JSON content
    #[must_use]
    pub fn json(&self) -> &str {
        &self.json_content
    }

    /// Get the detected version
    #[must_use]
    pub fn version(&self) -> PhirJsonVersion {
        self.version
    }
}

// Convert from the shared Phir type
impl From<PhirJson> for PhirJsonEngineProgram {
    fn from(program: PhirJson) -> Self {
        // We need to detect the version here, but if it fails, we'll handle it later in build()
        match detect_version(&program.source) {
            Ok(version) => Self {
                json_content: program.source,
                version,
            },
            // If version detection fails, we'll use a placeholder and let build() handle the error
            Err(_) => Self {
                json_content: program.source,
                version: PhirJsonVersion::V0_1, // Default to V0_1
            },
        }
    }
}

/// WebAssembly program for PHIR foreign function calls
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct PhirJsonEngineWasm {
    /// The WASM binary data
    pub wasm_bytes: Vec<u8>,
    /// Optional source path for debugging
    pub source_path: Option<String>,
}

#[cfg(feature = "wasm")]
impl PhirJsonEngineWasm {
    /// Create from WASM bytes
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            wasm_bytes: bytes,
            source_path: None,
        }
    }

    /// Create from a file path (WAT or WASM)
    ///
    /// # Errors
    ///
    /// Returns an error if file cannot be read or compiled
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PecosError> {
        let path_ref = path.as_ref();
        let bytes = std::fs::read(path_ref).map_err(PecosError::IO)?;

        // If it's a .wat file, compile it to WASM using wat crate
        let wasm_bytes = if path_ref.extension().and_then(|s| s.to_str()) == Some("wat") {
            wat::parse_bytes(&bytes)
                .map_err(|e| PecosError::Input(format!("Failed to parse WAT file: {e}")))?
                .to_vec()
        } else {
            bytes
        };

        Ok(Self {
            wasm_bytes,
            source_path: Some(path_ref.display().to_string()),
        })
    }
}

/// Trait for types that can be converted to a WASM program for PHIR
#[cfg(feature = "wasm")]
pub trait IntoWasm {
    /// Convert to a `PhirJsonEngineWasm`
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError>;
}

#[cfg(feature = "wasm")]
impl IntoWasm for PhirJsonEngineWasm {
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError> {
        Ok(self)
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for &str {
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError> {
        PhirJsonEngineWasm::from_file(self)
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for String {
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError> {
        PhirJsonEngineWasm::from_file(self)
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for &String {
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError> {
        PhirJsonEngineWasm::from_file(self)
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for PathBuf {
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError> {
        PhirJsonEngineWasm::from_file(self)
    }
}

#[cfg(feature = "wasm")]
impl IntoWasm for &Path {
    fn into_wasm_program(self) -> Result<PhirJsonEngineWasm, PecosError> {
        PhirJsonEngineWasm::from_file(self)
    }
}

/// Builder for PHIR JSON engines
#[derive(Clone)]
pub struct PhirJsonEngineBuilder {
    program: Option<PhirJsonEngineProgram>,
    /// WebAssembly program for foreign function calls
    #[cfg(feature = "wasm")]
    wasm_program: Option<PhirJsonEngineWasm>,
}

impl PhirJsonEngineBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            program: None,
            #[cfg(feature = "wasm")]
            wasm_program: None,
        }
    }

    /// Set the program for this engine (accepts either `Phir` or `PhirJsonEngineProgram`)
    #[must_use]
    pub fn program(mut self, program: impl Into<PhirJsonEngineProgram>) -> Self {
        self.program = Some(program.into());
        self
    }

    /// Set the program from a JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if JSON parsing or version detection fails
    pub fn json(mut self, json: &str) -> Result<Self, PecosError> {
        self.program = Some(PhirJsonEngineProgram::from_json(json)?);
        Ok(self)
    }

    /// Set the program from a file path
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or JSON parsing fails
    pub fn file(self, path: impl AsRef<Path>) -> Result<Self, PecosError> {
        let content = std::fs::read_to_string(path).map_err(PecosError::IO)?;
        self.json(&content)
    }

    /// Set the WebAssembly program for foreign function calls
    ///
    /// This method accepts:
    /// - `PhirJsonEngineWasm` - pre-loaded WASM binary
    /// - `&str` or `String` - path to a .wasm or .wat file
    /// - `PathBuf` or `&Path` - path to a .wasm or .wat file
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

impl Default for PhirJsonEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassicalControlEngineBuilder for PhirJsonEngineBuilder {
    type Engine = PhirJsonEngine;

    fn build(self) -> Result<Self::Engine, PecosError> {
        let program = self
            .program
            .ok_or_else(|| PecosError::Input("No program set for PHIR engine".to_string()))?;

        // Create engine from program
        let mut engine = match program.version {
            PhirJsonVersion::V0_1 => PhirJsonEngine::from_json(program.json())?,
        };

        // Set WASM foreign object if provided
        #[cfg(feature = "wasm")]
        if let Some(wasm_program) = self.wasm_program {
            use crate::v0_1::foreign_objects::ForeignObject;
            use crate::v0_1::wasm_foreign_object::WasmtimeForeignObject;

            let mut foreign_object = WasmtimeForeignObject::from_bytes(&wasm_program.wasm_bytes)?;
            foreign_object.init()?;
            engine.set_foreign_object(Box::new(foreign_object));
        }

        Ok(engine)
    }
}

/// Create a new PHIR JSON engine builder
///
/// This is the entry point for the unified API pattern:
/// ```rust
/// use pecos_phir_json::phir_json_engine;
/// use pecos_programs::PhirJson;
/// use pecos_engines::engine_builder::ClassicalControlEngineBuilder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let json = r#"{
///     "format": "PHIR/JSON",
///     "version": "0.1.0",
///     "metadata": {
///         "name": "simple_measurement",
///         "description": "Single qubit measurement example"
///     },
///     "ops": [
///         {
///             "data": "qvar_define",
///             "data_type": "qubits",
///             "variable": "q",
///             "size": 1
///         },
///         {
///             "data": "cvar_define",
///             "data_type": "i64",
///             "variable": "m",
///             "size": 1
///         },
///         {"qop": "H", "args": [["q", 0]]},
///         {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
///         {"cop": "Result", "args": ["m"], "returns": ["c"]}
///     ]
/// }"#;
///
/// let results = phir_json_engine()
///     .program(PhirJson::from_json(json))
///     .to_sim()
///     .run(100)?;
///
/// // Verify we got the expected number of shots
/// assert_eq!(results.len(), 100);
///
/// // Convert to columnar format and verify the result register exists
/// let shot_map = results.try_as_shot_map()?;
/// let register_names = shot_map.register_names();
/// assert!(register_names.iter().any(|n| *n == "c"),
///         "Expected 'c' register in results, found: {:?}", register_names);
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn phir_json_engine() -> PhirJsonEngineBuilder {
    PhirJsonEngineBuilder::new()
}

/// Convenience conversion from `Phir` to builder
impl From<PhirJson> for PhirJsonEngineBuilder {
    fn from(program: PhirJson) -> Self {
        Self::new().program(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phir_engine_program_from_json() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {},
            "ops": []
        }"#;

        let program = PhirJsonEngineProgram::from_json(json).unwrap();
        assert_eq!(program.version(), PhirJsonVersion::V0_1);
        assert_eq!(program.json(), json);
    }

    #[test]
    fn test_phir_program_conversion() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {},
            "ops": []
        }"#;

        let shared_program = PhirJson::from_json(json);
        let engine_program: PhirJsonEngineProgram = shared_program.into();
        assert_eq!(engine_program.version(), PhirJsonVersion::V0_1);
        assert_eq!(engine_program.json(), json);
    }

    #[test]
    fn test_phir_engine_builder() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {},
            "ops": [
                {"data": "cvar_define", "data_type": "u32", "variable": "result", "size": 1},
                {"cop": "Result", "args": [0], "returns": [["result", 0]]}
            ]
        }"#;

        let program = PhirJson::from_json(json);
        let builder = phir_json_engine().program(program);

        // Build should succeed
        let engine = builder.build();
        assert!(engine.is_ok(), "Failed to build engine: {:?}", engine.err());
    }

    #[test]
    fn test_phir_unified_api_pattern() {
        // Test that we can use the unified API pattern
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {},
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
                {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
                {"data": "cvar_define", "data_type": "u32", "variable": "result", "size": 1},
                {"cop": "Result", "args": [0], "returns": [["result", 0]]}
            ]
        }"#;

        let program = PhirJson::from_json(json);

        // This tests that the builder can be used with .to_sim()
        let _sim_builder = phir_json_engine().program(program).to_sim();

        // We can't actually run it without quantum backend setup,
        // but this verifies the API compiles correctly
    }
}
