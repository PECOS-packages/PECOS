mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    
    // Import helpers from common module
    use crate::common::phir_test_utils::{
        get_phir_results, assert_shotresult_value
    };

    // Test meta instructions
    #[test]
    fn test_meta_instructions() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/meta_instructions_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        println!("Registers_u64: {:?}", result.registers_u64);
        println!("Registers_i64: {:?}", result.registers_i64);
        
        // The actual result value will depend on the quantum simulation,
        // but we just need to verify that the engine successfully processes
        // meta instructions without errors
        assert!(result.registers.contains_key("output"), "Expected 'output' register to be present");

        Ok(())
    }
}