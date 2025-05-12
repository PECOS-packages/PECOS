mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_phir::v0_1::operations::{MachineOperationResult, OperationProcessor};
    use std::collections::HashMap;

    // We still need get_phir_results for the simple test
    use crate::common::phir_test_utils::get_phir_results;

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
        let mut metadata = HashMap::new();
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
        // We need direct access to the engine to check machine operation processing
        let phir_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/assets/advanced_machine_operations_test.json");

        // Create and run the engine directly
        let mut engine = pecos_phir::v0_1::engine::PHIREngine::new(phir_path)?;
        let result = engine.process(())?;

        // Print all information about the result for debugging
        println!("ShotResult: {result:?}");
        println!("Registers: {:?}", result.registers);

        // Verify the final result exists
        assert!(
            result.registers.contains_key("output"),
            "Expected 'output' register to be present"
        );
        assert_eq!(
            result.registers["output"], 1,
            "Expected output value to be 1 (m0 + m1 = 1 + 0 = 1)"
        );

        Ok(())
    }

    // Test running a simplified PHIR program with machine operations
    #[test]
    fn test_simple_machine_operations() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/simple_machine_operations_test.json")?;

        // Print all information about the result for debugging
        println!("ShotResult: {result:?}");
        println!("Registers: {:?}", result.registers);

        // Verify that the program executed successfully with machine operations
        assert!(
            result.registers.contains_key("output"),
            "Expected 'output' register to be present"
        );
        assert_eq!(
            result.registers["output"], 42,
            "Expected output value to be 42"
        );

        Ok(())
    }
}
