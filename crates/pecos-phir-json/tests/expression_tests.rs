mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_phir_json::v0_1::ast::PHIRProgram;
    use pecos_phir_json::v0_1::engine::PhirJsonEngine;

    // Test 1: Basic arithmetic expressions
    #[test]
    fn test_arithmetic_expressions() -> Result<(), PecosError> {
        // Define test program inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 0
          },
          "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "c", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "d", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 31},
            {"cop": "=", "args": [10], "returns": ["a"]},
            {"cop": "=", "args": [5], "returns": ["b"]},
            {"cop": "=", "args": [{"cop": "+", "args": ["a", "b"]}], "returns": ["c"]},
            {"cop": "=", "args": [{"cop": "*", "args": ["a", "b"]}], "returns": ["d"]},
            {"cop": "=", "args": [{"cop": "-", "args": ["d", "c"]}], "returns": ["result"]},
            {"cop": "Result", "args": ["result"], "returns": ["output"]}
          ]
        }"#;

        // In a real scenario, this calculation would be:
        // a = 10
        // b = 5
        // c = a + b = 15
        // d = a * b = 50
        // result = d - c = 50 - 15 = 35

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        // Create engine directly
        let mut engine = PhirJsonEngine::from_program(program.clone())?;

        // Execute directly
        let shot = engine.process(())?;

        assert_eq!(shot.data["output"].as_u32(), Some(35));

        Ok(())
    }

    // Test 2: Comparison expressions and logical operators
    #[test]
    fn test_comparison_expressions() -> Result<(), PecosError> {
        // Define comparison expressions test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 0
          },
          "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "less_than", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "equal", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "greater_than", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "combined", "size": 31},
            {"cop": "=", "args": [10], "returns": ["a"]},
            {"cop": "=", "args": [5], "returns": ["b"]},
            {"cop": "=", "args": [{"cop": "<", "args": ["b", "a"]}], "returns": ["less_than"]},
            {"cop": "=", "args": [{"cop": "==", "args": ["a", 10]}], "returns": ["equal"]},
            {"cop": "=", "args": [{"cop": ">", "args": ["a", "b"]}], "returns": ["greater_than"]},
            {"cop": "=", "args": [{"cop": "&", "args": ["less_than", "equal"]}], "returns": ["combined"]},
            {"cop": "Result", "args": ["less_than"], "returns": ["less_than_result"]},
            {"cop": "Result", "args": ["equal"], "returns": ["equal_result"]},
            {"cop": "Result", "args": ["greater_than"], "returns": ["greater_than_result"]},
            {"cop": "Result", "args": ["combined"], "returns": ["combined_result"]}
          ]
        }"#;

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        // Create engine directly
        let mut engine = PhirJsonEngine::from_program(program.clone())?;

        // Execute directly
        let shot = engine.process(())?;

        assert_eq!(shot.data["less_than_result"].as_u32(), Some(1));
        assert_eq!(shot.data["equal_result"].as_u32(), Some(1));
        assert_eq!(shot.data["greater_than_result"].as_u32(), Some(1));
        assert_eq!(shot.data["combined_result"].as_u32(), Some(1));

        Ok(())
    }

    // Test 3: Bit manipulation operations
    #[test]
    fn test_bit_operations() -> Result<(), PecosError> {
        // Define bit operations test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 0
          },
          "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit_and", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit_or", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit_xor", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit_shift", "size": 31},
            {"cop": "=", "args": [3], "returns": ["a"]},
            {"cop": "=", "args": [5], "returns": ["b"]},
            {"cop": "=", "args": [{"cop": "&", "args": ["a", "b"]}], "returns": ["bit_and"]},
            {"cop": "=", "args": [{"cop": "|", "args": ["a", "b"]}], "returns": ["bit_or"]},
            {"cop": "=", "args": [{"cop": "^", "args": ["a", "b"]}], "returns": ["bit_xor"]},
            {"cop": "=", "args": [{"cop": "<<", "args": ["a", 2]}], "returns": ["bit_shift"]},
            {"cop": "Result", "args": ["bit_and"], "returns": ["bit_and_result"]},
            {"cop": "Result", "args": ["bit_or"], "returns": ["bit_or_result"]},
            {"cop": "Result", "args": ["bit_xor"], "returns": ["bit_xor_result"]},
            {"cop": "Result", "args": ["bit_shift"], "returns": ["bit_shift_result"]}
          ]
        }"#;

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        // Create engine directly
        let mut engine = PhirJsonEngine::from_program(program.clone())?;

        // Execute directly
        let shot = engine.process(())?;

        assert_eq!(shot.data["bit_and_result"].as_u32(), Some(1));
        assert_eq!(shot.data["bit_or_result"].as_u32(), Some(7));
        assert_eq!(shot.data["bit_xor_result"].as_u32(), Some(6));
        assert_eq!(shot.data["bit_shift_result"].as_u32(), Some(12));

        Ok(())
    }

    // Test 4: Nested expressions
    #[test]
    fn test_nested_expressions() -> Result<(), PecosError> {
        // Define nested expressions test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 0
          },
          "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "a", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "b", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "c", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 31},
            {"cop": "=", "args": [5], "returns": ["a"]},
            {"cop": "=", "args": [10], "returns": ["b"]},
            {"cop": "=", "args": [15], "returns": ["c"]},
            {"cop": "=", "args": [
              {"cop": "+", "args": [
                {"cop": "*", "args": ["a", "b"]},
                {"cop": "-", "args": ["c", 5]}
              ]}
            ], "returns": ["result"]},
            {"cop": "Result", "args": ["result"], "returns": ["output"]}
          ]
        }"#;

        // In a real scenario, this calculation would be:
        // a = 5
        // b = 10
        // c = 15
        // result = (a * b) + (c - 5) = (5 * 10) + (15 - 5) = 50 + 10 = 60

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        // Create engine directly
        let mut engine = PhirJsonEngine::from_program(program.clone())?;

        // Execute directly
        let shot = engine.process(())?;

        assert_eq!(shot.data["output"].as_u32(), Some(60));

        Ok(())
    }

    // Test 5: Variable bit access
    #[test]
    fn test_variable_bit_access() -> Result<(), PecosError> {
        // Define variable bit access test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 0
          },
          "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "value", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit0", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit1", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "bit2", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 31},
            {"cop": "=", "args": [5], "returns": ["value"]},
            {"cop": "=", "args": [{"cop": "&", "args": [{"cop": ">>", "args": ["value", 0]}, 1]}], "returns": ["bit0"]},
            {"cop": "=", "args": [{"cop": "&", "args": [{"cop": ">>", "args": ["value", 1]}, 1]}], "returns": ["bit1"]},
            {"cop": "=", "args": [{"cop": "&", "args": [{"cop": ">>", "args": ["value", 2]}, 1]}], "returns": ["bit2"]},
            {"cop": "=", "args": [1], "returns": [["value", 0]]},
            {"cop": "=", "args": [0], "returns": [["value", 1]]},
            {"cop": "=", "args": [1], "returns": [["value", 2]]},
            {"cop": "Result", "args": ["bit0"], "returns": ["bit0_result"]},
            {"cop": "Result", "args": ["bit1"], "returns": ["bit1_result"]},
            {"cop": "Result", "args": ["bit2"], "returns": ["bit2_result"]},
            {"cop": "Result", "args": ["value"], "returns": ["value_result"]}
          ]
        }"#;

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        // Create engine directly
        let mut engine = PhirJsonEngine::from_program(program.clone())?;

        // Execute directly
        let shot = engine.process(())?;

        assert_eq!(shot.data["bit0_result"].as_u32(), Some(1));
        assert_eq!(shot.data["bit1_result"].as_u32(), Some(0));
        assert_eq!(shot.data["bit2_result"].as_u32(), Some(1));
        assert_eq!(shot.data["value_result"].as_u32(), Some(5));

        Ok(())
    }
}
