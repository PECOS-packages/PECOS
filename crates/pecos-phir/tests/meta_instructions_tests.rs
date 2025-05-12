mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;

    // Test meta instructions
    #[test]
    fn test_meta_instructions() -> Result<(), PecosError> {
        // We need direct access to the engine to verify barrier handling
        let phir_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/assets/meta_instructions_test.json");

        // Create and run the engine directly
        let mut engine = pecos_phir::v0_1::engine::PHIREngine::new(phir_path)?;
        let result = engine.process(())?;

        // Print all information about the result for debugging
        println!("ShotResult: {result:?}");
        println!("Registers: {:?}", result.registers);

        // Verify that the program executed successfully
        assert!(
            result.registers.contains_key("output"),
            "Expected 'output' register to be present"
        );

        Ok(())
    }
}
