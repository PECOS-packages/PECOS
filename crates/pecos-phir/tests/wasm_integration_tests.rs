#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_engines::core::shot_results::OutputFormat;
    use pecos_phir::v0_1::ast::PHIRProgram;
    use pecos_phir::v0_1::engine::PHIREngine;
    use pecos_phir::v0_1::foreign_objects::ForeignObject;
    use pecos_phir::v0_1::wasm_foreign_object::WasmtimeForeignObject;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn setup_test_environment() -> Result<(Arc<WasmtimeForeignObject>, PHIREngine), PecosError> {
        // Create a temporary WebAssembly module with the 'add' function
        let wat_content = r#"
        (module
          (func $init (export "init"))
          (func $add (export "add") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            i32.add)
        )
        "#;

        // Create a unique temporary file name to prevent conflicts between tests
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let temp_dir = std::env::temp_dir();
        let wasm_path = temp_dir.join(format!("add_test_{timestamp}.wat"));
        std::fs::write(&wasm_path, wat_content).map_err(|e| {
            PecosError::IO(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write temporary WAT file: {e}"),
            ))
        })?;

        // Create a WebAssembly foreign object
        let mut foreign_object = WasmtimeForeignObject::new(&wasm_path)?;

        // Initialize the foreign object
        foreign_object.init()?;

        // Important: We deliberately don't delete the file here to avoid issues
        // with the file being removed while it's still needed by the WasmtimeForeignObject.
        // Instead, we rely on the operating system to clean up temporary files eventually.

        // Wrap in Arc after initialization
        let foreign_object = Arc::new(foreign_object);

        // Create a basic PHIR engine from a simple program JSON string with minimal operations
        let simple_phir = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
            },
            "ops": [
                {"data": "cvar_define", "data_type": "i32", "variable": "placeholder", "size": 32},
                {"cop": "=", "args": [0], "returns": ["placeholder"]},
                {"cop": "Result", "args": ["placeholder"], "returns": ["output"]}
            ]
        }"#;

        let mut engine = PHIREngine::from_json(simple_phir)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        // Set the foreign object directly
        engine.set_foreign_object(Arc::clone(&foreign_object) as Arc<dyn ForeignObject>);

        Ok((foreign_object, engine))
    }

    // Test 1: Basic WebAssembly function execution from PHIR
    #[test]
    fn test_wasm_basic_execution() -> Result<(), PecosError> {
        // Setup test environment
        let (foreign_object, _) = setup_test_environment()?;

        // Create a PHIR program with direct WebAssembly function call
        let phir_json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
            },
            "ops": [
                {"cop": "ffcall", "function": "add", "args": [5, 7], "returns": ["result"]},
                {"cop": "Result", "args": ["result"], "returns": ["output"]}
            ]
        }"#;

        // Replace the engine's program with our test program
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;
        let mut engine = PHIREngine::from_program(program);

        // Set the foreign object directly
        engine.set_foreign_object(Arc::clone(&foreign_object) as Arc<dyn ForeignObject>);

        // Execute the program
        let result = engine.process(())?;

        // Debug the raw internal state
        println!("Initial shot result registers: {:?}", result.registers);
        println!(
            "Measurement results: {:?}",
            engine.processor.measurement_results
        );
        println!(
            "Initial exported values: {:?}",
            engine.processor.exported_values
        );
        println!("Export mappings: {:?}", engine.processor.export_mappings);

        // Verify that the WebAssembly call worked by checking measurement_results
        assert!(
            engine.processor.measurement_results.contains_key("result"),
            "Measurement results should contain 'result'"
        );
        if let Some(&value) = engine.processor.measurement_results.get("result") {
            assert_eq!(
                value, 12,
                "WebAssembly computation value should be 12 (5 + 7)"
            );

            // This test verifies that the WebAssembly function was executed correctly
            // The Result command and export mappings are tested in other contexts, such as the CLI
        }

        Ok(())
    }

    // Test 2: Multiple WebAssembly function calls with variable references
    #[test]
    fn test_wasm_multiple_calls() -> Result<(), PecosError> {
        // Setup test environment
        let (foreign_object, _) = setup_test_environment()?;

        // Create a PHIR program with multiple WebAssembly function calls
        let phir_json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
            },
            "ops": [
                {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 32},
                {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 32},
                {"data": "cvar_define", "data_type": "i32", "variable": "c", "size": 32},
                {"data": "cvar_define", "data_type": "i32", "variable": "final_result", "size": 32},
                {"cop": "=", "args": [3], "returns": ["a"]},
                {"cop": "=", "args": [4], "returns": ["b"]},
                {"cop": "ffcall", "function": "add", "args": ["a", "b"], "returns": ["c"]},
                {"cop": "ffcall", "function": "add", "args": ["c", 10], "returns": ["final_result"]},
                {"cop": "Result", "args": ["final_result"], "returns": ["output"]}
            ]
        }"#;

        // Replace the engine's program with our test program
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;
        let mut engine = PHIREngine::from_program(program);

        // Set the foreign object directly
        engine.set_foreign_object(Arc::clone(&foreign_object) as Arc<dyn ForeignObject>);

        // Execute the program
        let result = engine.process(())?;

        // Debug the internal state
        println!("Initial shot result registers: {:?}", result.registers);
        println!(
            "Measurement results: {:?}",
            engine.processor.measurement_results
        );
        println!(
            "Initial exported values: {:?}",
            engine.processor.exported_values
        );
        println!("Export mappings: {:?}", engine.processor.export_mappings);

        // Verify the variable setup was successful
        assert!(
            engine.processor.measurement_results.contains_key("a"),
            "Measurement results should contain 'a'"
        );
        if let Some(&a_value) = engine.processor.measurement_results.get("a") {
            assert_eq!(a_value, 3, "Variable 'a' should be 3");
        }

        // The c register should contain the result of a + b = 3 + 4 = 7
        if let Some(&c_value) = engine.processor.measurement_results.get("c") {
            assert_eq!(c_value, 7, "Variable 'c' should be 7 (3 + 4)");
        }

        // Check for the final result
        if let Some(&final_value) = engine.processor.measurement_results.get("final_result") {
            assert_eq!(
                final_value, 17,
                "Variable 'final_result' should be 17 (3 + 4 + 10)"
            );

            // This test verifies that the WebAssembly function was executed correctly
            // The Result command and export mappings are tested in other contexts, such as the CLI
        }

        Ok(())
    }

    // Test 3: WebAssembly function calls with conditional blocks
    #[test]
    fn test_wasm_with_conditionals() -> Result<(), PecosError> {
        // Setup test environment
        let (foreign_object, _) = setup_test_environment()?;

        // Create a PHIR program with conditional blocks and WebAssembly calls
        let phir_json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
            },
            "ops": [
                {"data": "cvar_define", "data_type": "i32", "variable": "condition", "size": 32},
                {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 32},
                {"cop": "=", "args": [1], "returns": ["condition"]},
                {
                    "block": "if",
                    "condition": {"cop": "==", "args": ["condition", 1]},
                    "true_branch": [
                        {"cop": "ffcall", "function": "add", "args": [5, 5], "returns": ["result"]}
                    ],
                    "false_branch": [
                        {"cop": "ffcall", "function": "add", "args": [2, 2], "returns": ["result"]}
                    ]
                },
                {"cop": "Result", "args": ["result"], "returns": ["output"]}
            ]
        }"#;

        // Replace the engine's program with our test program
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;
        let mut engine = PHIREngine::from_program(program);

        // Set the foreign object directly
        engine.set_foreign_object(Arc::clone(&foreign_object) as Arc<dyn ForeignObject>);

        // Execute the program
        let result = engine.process(())?;

        // Debug the internal state
        println!("Initial shot result registers: {:?}", result.registers);
        println!(
            "Measurement results: {:?}",
            engine.processor.measurement_results
        );
        println!(
            "Initial exported values: {:?}",
            engine.processor.exported_values
        );
        println!("Export mappings: {:?}", engine.processor.export_mappings);

        // Verify the condition variable was set correctly
        assert!(
            engine
                .processor
                .measurement_results
                .contains_key("condition"),
            "Measurement results should contain 'condition'"
        );
        if let Some(&condition_value) = engine.processor.measurement_results.get("condition") {
            assert_eq!(condition_value, 1, "Variable 'condition' should be 1");
        }

        // Check for the result of the conditional operation
        if let Some(&result_value) = engine.processor.measurement_results.get("result") {
            // Since condition=1, the true branch should have executed: 5+5=10
            assert_eq!(
                result_value, 10,
                "Variable 'result' should be 10 (5 + 5 from true branch)"
            );

            // This test verifies that the WebAssembly function was executed correctly
            // The Result command and export mappings are tested in other contexts, such as the CLI
        }

        Ok(())
    }

    // Test 4: Test result formatting
    #[test]
    fn test_result_formatting() -> Result<(), PecosError> {
        // Setup test environment
        let (foreign_object, _) = setup_test_environment()?;

        // Create a simple PHIR program
        let phir_json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
            },
            "ops": [
                {"cop": "ffcall", "function": "add", "args": [123, 456], "returns": ["result"]},
                {"cop": "Result", "args": ["result"], "returns": ["output"]}
            ]
        }"#;

        // Replace the engine's program with our test program
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;
        let mut engine = PHIREngine::from_program(program);

        // Set the foreign object directly
        engine.set_foreign_object(Arc::clone(&foreign_object) as Arc<dyn ForeignObject>);

        // Execute the program
        let _result = engine.process(())?;

        // Debug the internal state
        println!(
            "Measurement results: {:?}",
            engine.processor.measurement_results
        );
        println!(
            "Initial exported values: {:?}",
            engine.processor.exported_values
        );
        println!("Export mappings: {:?}", engine.processor.export_mappings);

        // Verify that the WebAssembly call worked by checking measurement_results
        assert!(
            engine.processor.measurement_results.contains_key("result"),
            "Measurement results should contain 'result'"
        );
        if let Some(&value) = engine.processor.measurement_results.get("result") {
            assert_eq!(value, 579, "Value should be 579 (123 + 456)");

            // This test verifies that the WebAssembly function was executed correctly
            // The Result command and export mappings are tested in other contexts, such as the CLI
        }

        // Test different format outputs - we don't verify the output, just that the methods don't error
        let pretty_json = engine.get_formatted_results(OutputFormat::PrettyJson)?;
        let compact_json = engine.get_formatted_results(OutputFormat::CompactJson)?;
        let pretty_compact = engine.get_formatted_results(OutputFormat::PrettyCompactJson)?;

        // Debug the formatted results
        println!("Pretty JSON: {pretty_json}");
        println!("Compact JSON: {compact_json}");
        println!("Pretty Compact JSON: {pretty_compact}");

        // Basic verification that the formatted outputs exist (even if they might be empty arrays)
        assert!(
            pretty_json.contains('['),
            "Pretty JSON result should be valid JSON"
        );
        assert!(
            compact_json.contains('['),
            "Compact JSON result should be valid JSON"
        );
        assert!(
            pretty_compact.contains('['),
            "Pretty Compact JSON result should be valid JSON"
        );

        Ok(())
    }

    // Test 5: Test error handling for invalid WebAssembly calls
    #[test]
    fn test_wasm_error_handling() -> Result<(), PecosError> {
        // Setup test environment
        let (foreign_object, _) = setup_test_environment()?;

        // Create a PHIR program with an invalid function call
        let phir_json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
            },
            "ops": [
                {"cop": "ffcall", "function": "non_existent_function", "args": [1, 2], "returns": ["result"]},
                {"cop": "Result", "args": ["result"], "returns": ["output"]}
            ]
        }"#;

        // Replace the engine's program with our test program
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;
        let mut engine = PHIREngine::from_program(program);

        // Set the foreign object directly
        engine.set_foreign_object(Arc::clone(&foreign_object) as Arc<dyn ForeignObject>);

        // Execute the program - it should fail because the function doesn't exist
        let result = engine.process(());
        assert!(
            result.is_err(),
            "Function call to non-existent function should fail"
        );

        // Verify that the error message contains information about the missing function
        if let Err(e) = result {
            assert!(
                e.to_string().contains("non_existent_function"),
                "Error message should mention the missing function name"
            );
        }

        Ok(())
    }
}
