mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    
    // Import helpers from common module
    use crate::common::phir_test_utils::{
        get_phir_results, assert_shotresult_value
    };

    // Test 1: Basic quantum gate operations and measurement
    #[test]
    fn test_basic_gates_and_measurement() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/basic_gates_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        
        // We can't assert specific values since measurements are probabilistic,
        // but we can check that we got a result (0 or 1)
        assert!(result.registers.contains_key("output"), "Expected 'output' register to be present");
        let value = result.registers.get("output").unwrap();
        assert!(*value == 0 || *value == 1, "Expected measurement value to be 0 or 1, got {}", value);

        Ok(())
    }

    // Test 2: Bell state preparation
    #[test]
    fn test_bell_state() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/bell_state_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        
        // Check that we have an output measurement
        assert!(result.registers.contains_key("output"), "Expected 'output' register to be present");

        // Bell state should result in either 00 (0) or 11 (3) measurement outcomes
        let value = result.registers.get("output").unwrap();
        assert!(*value == 0 || *value == 3, "Expected Bell state measurement value to be 0 or 3, got {}", value);

        Ok(())
    }

    // Test 3: Testing rotation gates
    #[test]
    fn test_rotation_gates() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/rotation_gates_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        
        // Verify that we have an output
        assert!(result.registers.contains_key("output"), "Expected 'output' register to be present");
        let value = result.registers.get("output").unwrap();
        assert!(*value == 0 || *value == 1, "Expected measurement value to be 0 or 1, got {}", value);

        Ok(())
    }

    // Test 4: Testing qparallel blocks
    #[test]
    fn test_qparallel_blocks() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/qparallel_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        
        // Verify that we have an output
        assert!(result.registers.contains_key("output"), "Expected 'output' register to be present");

        // After qparallel with H on qubit 0 and X on qubit 1,
        // the possible measurement outcomes are 01 (1) or 11 (3)
        let value = result.registers.get("output").unwrap();
        assert!(*value == 1 || *value == 3, 
            "Expected qparallel measurement value to be 1 or 3, got {}", value);

        Ok(())
    }

    // Test 5: Complex example with control flow and quantum operations
    #[test]
    fn test_control_flow_with_quantum() -> Result<(), PecosError> {
        let result = get_phir_results("tests/assets/control_flow_test.json")?;
        
        // Print all information about the result for debugging
        println!("ShotResult: {:?}", result);
        println!("Registers: {:?}", result.registers);
        
        // Verify that we have an output
        assert!(result.registers.contains_key("output"), "Expected 'output' register to be present");

        // Since condition is 1, the X gate is applied, so we expect output to be 1
        let value = result.registers.get("output").unwrap();
        assert_eq!(*value, 1, "Expected control flow output value to be 1, got {}", value);

        Ok(())
    }
}