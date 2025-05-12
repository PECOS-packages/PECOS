#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_phir::v0_1::ast::PHIRProgram;
    use pecos_phir::v0_1::engine::PHIREngine;
    use pecos_phir::v0_1::foreign_objects::ForeignObject;
    use pecos_phir::v0_1::wasm_foreign_object::WasmtimeForeignObject;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn test_wasm_add_function_in_phir() -> Result<(), PecosError> {
        // WASM path
        let wasm_path = Path::new("crates/pecos-phir/tests/assets/add.wat");

        // Skip the test if the WebAssembly file doesn't exist
        if !wasm_path.exists() {
            println!("Skipping test_wasm_add_function_in_phir: WebAssembly file not found");
            return Ok(());
        }

        // PHIR program inlined as JSON string
        let phir_json = r#"{
  "format": "PHIR/JSON",
  "version": "0.1.0",
  "metadata": {
    "num_qubits": 0,
    "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
  },
  "ops": [
    {"cop": "ffcall", "function": "add", "args": [7, 3], "returns": ["result"]},
    {"cop": "Result", "args": ["result"], "returns": ["output"]}
  ]
}"#;

        // Create a WebAssembly foreign object
        let mut foreign_object = WasmtimeForeignObject::new(wasm_path)?;

        // Initialize the foreign object
        foreign_object.init()?;

        // Wrap in Arc after initialization
        let foreign_object = Arc::new(foreign_object);

        // Create a PHIR engine from the JSON string
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;
        let mut engine = PHIREngine::from_program(program);

        // Set the foreign object for FFI calls
        engine.set_foreign_object(foreign_object);

        // Execute the program
        let result = engine.process(())?;

        // Verify the result - we expect "output" to be 10 (7 + 3)
        assert_eq!(result.registers.get("output"), Some(&10));

        Ok(())
    }
}
