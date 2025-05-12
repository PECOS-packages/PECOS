mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    
    // Import helpers from common module
    use crate::common::phir_test_utils::{
        get_phir_results, assert_shotresult_value, assert_shotresult_values
    };

    // Test 1: Basic arithmetic expressions 
    #[test]
    fn test_arithmetic_expressions() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/arithmetic_expressions_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        println!("Registers_u64: {:?}", result.registers_u64);
        println!("Registers_i64: {:?}", result.registers_i64);
        
        // Verify the result - we expect output = (10 * 5) - (10 + 5) = 50 - 15 = 35
        assert_shotresult_value(&result, "output", 35);

        Ok(())
    }

    // Test 2: Comparison expressions and logical operators
    #[test]
    fn test_comparison_expressions() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/comparison_expressions_test.json")?;

        // Verify results
        assert_shotresult_values(&result, &[
            ("less_than_result", 1),      // 5 < 10, so true (1)
            ("equal_result", 1),          // 10 == 10, so true (1)
            ("greater_than_result", 1),   // 10 > 5, so true (1)
            ("combined_result", 1),       // 1 & 1, so true (1)
        ]);

        Ok(())
    }

    // Test 3: Bit manipulation operations
    #[test]
    fn test_bit_operations() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/bit_operations_test.json")?;

        // Verify results
        assert_shotresult_values(&result, &[
            ("bit_and_result", 1),       // 3 & 5 = 1
            ("bit_or_result", 7),        // 3 | 5 = 7
            ("bit_xor_result", 6),       // 3 ^ 5 = 6
            ("bit_shift_result", 12),    // 3 << 2 = 12
        ]);

        Ok(())
    }

    // Test 4: Nested expressions
    #[test]
    fn test_nested_expressions() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/nested_expressions_test.json")?;

        // Verify result - we expect output = (5 * 10) + (15 - 5) = 50 + 10 = 60
        assert_shotresult_value(&result, "output", 60);

        Ok(())
    }

    // Test 5: Variable bit access
    #[test]
    fn test_variable_bit_access() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/variable_bit_access_test.json")?;

        // Verify results
        // Initial value is 5 (binary 101), so bits 0 and 2 are 1, bit 1 is 0
        assert_shotresult_values(&result, &[
            ("bit0_result", 1),   // bit 0 of 5 (101) is 1
            ("bit1_result", 0),   // bit 1 of 5 (101) is 0
            ("bit2_result", 1),   // bit 2 of 5 (101) is 1
            ("value_result", 5),  // Final value after bit ops
        ]);

        Ok(())
    }
}