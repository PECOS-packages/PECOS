#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::Engine;
    use pecos_phir::v0_1::engine::PHIREngine;
    use std::path::Path;

    // Test 1: Basic arithmetic expressions
    #[test]
    fn test_arithmetic_expressions() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path =
            Path::new("crates/pecos-phir/tests/assets/arithmetic_expressions_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_arithmetic_expressions: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify the result - we expect output = (10 * 5) - (10 + 5) = 50 - 15 = 35
        assert_eq!(result.registers.get("output"), Some(&35));

        Ok(())
    }

    // Test 2: Comparison expressions and logical operators
    #[test]
    fn test_comparison_expressions() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path =
            Path::new("crates/pecos-phir/tests/assets/comparison_expressions_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_comparison_expressions: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify results
        assert_eq!(result.registers.get("less_than_result"), Some(&1)); // 5 < 10, so true (1)
        assert_eq!(result.registers.get("equal_result"), Some(&1)); // 10 == 10, so true (1)
        assert_eq!(result.registers.get("greater_than_result"), Some(&1)); // 10 > 5, so true (1)
        assert_eq!(result.registers.get("combined_result"), Some(&1)); // 1 & 1, so true (1)

        Ok(())
    }

    // Test 3: Bit manipulation operations
    #[test]
    fn test_bit_operations() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/bit_operations_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_bit_operations: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify results
        assert_eq!(result.registers.get("bit_and_result"), Some(&1)); // 3 & 5 = 1
        assert_eq!(result.registers.get("bit_or_result"), Some(&7)); // 3 | 5 = 7
        assert_eq!(result.registers.get("bit_xor_result"), Some(&6)); // 3 ^ 5 = 6
        assert_eq!(result.registers.get("bit_shift_result"), Some(&12)); // 3 << 2 = 12

        Ok(())
    }

    // Test 4: Nested expressions
    #[test]
    fn test_nested_expressions() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/nested_expressions_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_nested_expressions: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify result - we expect output = (5 * 10) + (15 - 5) = 50 + 10 = 60
        assert_eq!(result.registers.get("output"), Some(&60));

        Ok(())
    }

    // Test 5: Variable bit access
    #[test]
    fn test_variable_bit_access() -> Result<(), PecosError> {
        // Path to our test file
        let phir_path = Path::new("crates/pecos-phir/tests/assets/variable_bit_access_test.json");

        // Skip the test if the file doesn't exist
        if !phir_path.exists() {
            println!("Skipping test_variable_bit_access: test file not found");
            return Ok(());
        }

        // Create a PHIR engine from the program file
        let mut engine = PHIREngine::new(phir_path)?;

        // Execute the program
        let result = engine.process(())?;

        // Verify results
        // Initial value is 5 (binary 101), so bits 0 and 2 are 1, bit 1 is 0
        assert_eq!(result.registers.get("bit0_result"), Some(&1));
        assert_eq!(result.registers.get("bit1_result"), Some(&0));
        assert_eq!(result.registers.get("bit2_result"), Some(&1));

        // After bit modifications (setting bit 0 to 1, bit 1 to 0, bit 2 to 1),
        // value should be binary 101 = decimal 5 (unchanged in this case)
        assert_eq!(result.registers.get("value_result"), Some(&5));

        Ok(())
    }
}
