#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_phir::v0_1::engine::PHIREngine;
    use std::path::Path;

    // Test 1: Basic quantum gate operations and measurement
    #[test]
    fn test_basic_gates_and_measurement() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/basic_gates_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_basic_gates_and_measurement: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // We can't assert specific values since measurements are probabilistic,
        // but we can check that we got a result (0 or 1)
        assert!(result.registers.contains_key("output"));
        let value = result.registers.get("output").unwrap();
        assert!(*value == 0 || *value == 1);

        Ok(())
    }

    // Test 2: Bell state preparation
    #[test]
    fn test_bell_state() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/bell_state_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_bell_state: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Check that we have an output measurement
        assert!(result.registers.contains_key("output"));

        // Bell state should result in either 00 (0) or 11 (3) measurement outcomes
        let value = result.registers.get("output").unwrap();
        assert!(*value == 0 || *value == 3);

        Ok(())
    }

    // Test 3: Testing rotation gates
    #[test]
    fn test_rotation_gates() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/rotation_gates_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_rotation_gates: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify that we have an output
        assert!(result.registers.contains_key("output"));
        let value = result.registers.get("output").unwrap();
        assert!(*value == 0 || *value == 1);

        Ok(())
    }

    // Test 4: Testing qparallel blocks
    #[test]
    fn test_qparallel_blocks() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/qparallel_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_qparallel_blocks: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify that we have an output
        assert!(result.registers.contains_key("output"));

        // After qparallel with H on qubit 0 and X on qubit 1,
        // the possible measurement outcomes are 01 (1) or 11 (3)
        let value = result.registers.get("output").unwrap();
        assert!(*value == 1 || *value == 3);

        Ok(())
    }

    // Test 5: Complex example with control flow and quantum operations
    #[test]
    fn test_control_flow_with_quantum() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/control_flow_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_control_flow_with_quantum: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify that we have an output
        assert!(result.registers.contains_key("output"));

        // Since condition is 1, the X gate is applied, so we expect output to be 1
        let value = result.registers.get("output").unwrap();
        assert_eq!(*value, 1);

        Ok(())
    }
}
