mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;

    // Import helpers from common module
    use crate::common::phir_test_utils::{assert_shotresult_value, get_phir_results};

    // Test machine operations
    #[test]
    fn test_machine_operations() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/machine_operations_test.json")?;

        // Print all information about the result for debugging
        println!("ShotResult: {result:?}");
        println!("Registers: {:?}", result.registers);
        println!("Registers_u64: {:?}", result.registers_u64);
        println!("Registers_i64: {:?}", result.registers_i64);

        // The actual result value will depend on the quantum simulation,
        // but we just need to verify that the engine successfully processes
        // machine operations without errors and exports the result value
        assert!(
            result.registers.contains_key("output"),
            "Expected 'output' register to be present"
        );

        // Since we've modified the test file to directly set result=2, check the value
        assert_shotresult_value(&result, "output", 2);

        Ok(())
    }

    // Test simple machine operations
    #[test]
    fn test_simple_machine_operations() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/simple_machine_operations_test.json")?;

        // Print all information about the result for debugging
        println!("ShotResult: {result:?}");
        println!("Registers: {:?}", result.registers);
        println!("Registers_u64: {:?}", result.registers_u64);
        println!("Registers_i64: {:?}", result.registers_i64);

        // The actual result value will depend on the quantum simulation,
        // but we just need to verify that the engine successfully processes
        // simple machine operations without errors
        assert!(
            result.registers.contains_key("output"),
            "Expected 'output' register to be present"
        );

        // Check that the value is 42 (from the assignment in the JSON file)
        assert_shotresult_value(&result, "output", 42);

        Ok(())
    }
}
