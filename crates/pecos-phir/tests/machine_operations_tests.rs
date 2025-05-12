#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_phir::v0_1::engine::PHIREngine;
    use std::path::Path;

    // Test machine operations
    #[test]
    fn test_machine_operations() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/machine_operations_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_machine_operations: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // The actual result value will depend on the quantum simulation,
        // but we just need to verify that the engine successfully processes
        // machine operations without errors
        assert!(result.registers.contains_key("output"));

        Ok(())
    }
}
