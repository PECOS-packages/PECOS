mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_core::{Gate, QubitId};
    use pecos_engines::{
        ClassicalControlEngineBuilder, ControlEngine, EngineStage, StateVectorEngineBuilder,
    };
    use pecos_phir_json::phir_json_engine;
    use pecos_phir_json::v0_1::ast::PHIRProgram;
    use pecos_phir_json::v0_1::engine::PhirJsonEngine;
    use pecos_phir_json::v0_1::operations::{MachineOperationResult, OperationProcessor};
    use std::collections::BTreeMap;

    // Test direct machine operation processing
    #[test]
    fn test_machine_operations_processing() {
        let processor = OperationProcessor::new();

        // Test Idle operation
        let result =
            processor.process_machine_op("Idle", None, Some(&(5.0, "ms".to_string())), None);
        assert!(result.is_ok());
        if let Ok(MachineOperationResult::Idle { duration_ns, .. }) = result {
            assert_eq!(duration_ns, 5_000_000); // 5ms = 5,000,000ns
        } else {
            panic!("Expected Idle result but got: {result:?}");
        }

        // Test Delay operation
        let result =
            processor.process_machine_op("Delay", None, Some(&(10.0, "us".to_string())), None);
        assert!(result.is_ok());
        if let Ok(MachineOperationResult::Delay { duration_ns, .. }) = result {
            assert_eq!(duration_ns, 10_000); // 10us = 10,000ns
        } else {
            panic!("Expected Delay result but got: {result:?}");
        }

        // Test Timing operation
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "timing_type".to_string(),
            serde_json::Value::String("start".to_string()),
        );
        metadata.insert(
            "label".to_string(),
            serde_json::Value::String("test_label".to_string()),
        );

        let result = processor.process_machine_op("Timing", None, None, Some(&metadata));
        assert!(result.is_ok());
        if let Ok(MachineOperationResult::Timing {
            timing_type, label, ..
        }) = result
        {
            assert_eq!(timing_type, "start");
            assert_eq!(label, "test_label");
        } else {
            panic!("Expected Timing result but got: {result:?}");
        }

        // Note: Reset machine operation has been replaced with Init quantum operation
        // We'll test the Skip machine operation instead (which is part of the spec)
        let result = processor.process_machine_op("Skip", None, None, None);
        assert!(result.is_ok());
        if let Ok(MachineOperationResult::Skip) = result {
            // Skip operation has no parameters to check
        } else {
            panic!("Expected Skip result but got: {result:?}");
        }
    }

    // Test running a PHIR program with machine operations - Complex version
    #[test]
    fn test_phir_with_machine_operations() -> Result<(), PecosError> {
        // Define the PHIR program inline - simplified program for more reliable testing
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 2
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "var", "size": 31},
            {"qop": "H", "args": [["q", 0]]},
            {"mop": "Idle", "args": [["q", 0], ["q", 1]], "duration": [5.0, "ms"]},
            {"mop": "Delay", "args": [["q", 0]], "duration": [2.0, "us"]},
            {"mop": "Skip"},
            {"cop": "=", "args": [1], "returns": ["var"]},
            {"cop": "Result", "args": ["var"], "returns": ["x"]}
          ]
        }"#;

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        let mut engine = PhirJsonEngine::from_program(program)?;
        let EngineStage::NeedsProcessing(commands) = engine.start(())? else {
            panic!("expected machine commands");
        };
        assert_eq!(
            commands.quantum_ops()?,
            vec![
                Gate::h(&[0]),
                Gate::idle(0.005, vec![QubitId(0), QubitId(1)]),
                Gate::idle(0.000_002, vec![QubitId(0)]),
            ]
        );

        // The classical work after the machine operations must still run to completion.
        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(4)?;
        assert_eq!(results.shots.len(), 4);
        for shot in &results.shots {
            assert_eq!(shot.data["x"].as_u32(), Some(1));
        }

        Ok(())
    }

    // Test running a simplified PHIR program with machine operations
    #[test]
    fn test_simple_machine_operations() -> Result<(), PecosError> {
        // Define the PHIR program inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 2
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 31},
            {"qop": "H", "args": [["q", 0]]},
            {"mop": "Idle", "args": [["q", 0], ["q", 1]], "duration": [5.0, "ms"]},
            {"mop": "Delay", "args": [["q", 0]], "duration": [2.0, "us"]},
            {"mop": "Transport", "args": [["q", 1]], "duration": [1.0, "ms"], "metadata": {"from_position": [0, 0], "to_position": [1, 0]}},
            {"mop": "Timing", "args": [["q", 0], ["q", 1]], "metadata": {"timing_type": "sync", "label": "sync_point_1"}},
            {"qop": "CX", "args": [["q", 0], ["q", 1]]},
            {"cop": "=", "args": [42], "returns": ["result"]},
            {"cop": "Result", "args": ["result"], "returns": ["a"]}
          ]
        }"#;

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        let mut engine = PhirJsonEngine::from_program(program)?;
        let EngineStage::NeedsProcessing(commands) = engine.start(())? else {
            panic!("expected machine commands");
        };
        assert_eq!(
            commands.quantum_ops()?,
            vec![
                Gate::h(&[0]),
                Gate::idle(0.005, vec![QubitId(0), QubitId(1)]),
                Gate::idle(0.000_002, vec![QubitId(0)]),
                Gate::idle(0.001, vec![QubitId(1)]),
                Gate::cx(&[(0, 1)]),
            ]
        );

        // The classical work after the machine operations must still run to completion.
        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(4)?;
        assert_eq!(results.shots.len(), 4);
        for shot in &results.shots {
            assert_eq!(shot.data["a"].as_u32(), Some(42));
        }

        Ok(())
    }
}
