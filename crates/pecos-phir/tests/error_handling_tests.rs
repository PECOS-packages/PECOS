#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_phir::v0_1::ast::{ArgItem, Expression};
    use pecos_phir::v0_1::operations::OperationProcessor;

    // Test the improved error handling for variable access
    #[test]
    fn test_variable_not_found_error() {
        let processor = OperationProcessor::new();

        // Create a variable reference for a non-existent variable
        let expr = Expression::Variable("nonexistent".to_string());

        // Evaluate the expression and check the error
        let result = processor.evaluate_expression(&expr);
        assert!(result.is_err());

        // Verify the error type and message
        match result {
            Err(PecosError::Computation(msg)) => {
                assert!(msg.contains("not found"));
                assert!(msg.contains("nonexistent"));
            }
            _ => panic!("Expected Computation error but got: {result:?}"),
        }
    }

    // Test the improved error handling for bit access
    #[test]
    fn test_bit_access_out_of_bounds() {
        let mut processor = OperationProcessor::new();

        // Define a variable with size 4 (bits 0-3)
        processor.handle_variable_definition("cvar_define", "i32", "test_var", 4);

        // Set a value for the variable
        processor
            .measurement_results
            .insert("test_var".to_string(), 15);

        // Direct test of validate_variable_access with an out-of-bounds index
        let result = processor.validate_variable_access("test_var", 5); // Size is 4, so index 5 is out of bounds

        assert!(result.is_err());

        // Check the error message
        match result {
            Err(e) => {
                let error_msg = e.to_string();
                assert!(error_msg.contains("out of bounds"));
                assert!(error_msg.contains("test_var"));
                assert!(error_msg.contains('5'));
                assert!(error_msg.contains('4')); // Size is 4
            }
            _ => panic!("Expected error but got success: {result:?}"),
        }
    }

    // Test the improved error handling for arithmetic operations
    #[test]
    fn test_division_by_zero() {
        let processor = OperationProcessor::new();

        // Create a division operation with zero divisor
        let expr = Expression::Operation {
            cop: "/".to_string(),
            args: vec![ArgItem::Integer(10), ArgItem::Integer(0)],
        };

        // Evaluate the expression and check the error
        let result = processor.evaluate_expression(&expr);
        assert!(result.is_err());

        // Verify the error type and message
        match result {
            Err(PecosError::Computation(msg)) => {
                assert!(msg.contains("Division by zero"));
                assert!(msg.contains("10 / 0"));
            }
            _ => panic!("Expected Computation error but got: {result:?}"),
        }
    }

    // Test the improved error handling for shift operations
    #[test]
    fn test_invalid_shift_amount() {
        let processor = OperationProcessor::new();

        // Create a left shift operation with invalid shift amount
        let expr = Expression::Operation {
            cop: "<<".to_string(),
            args: vec![
                ArgItem::Integer(10),
                ArgItem::Integer(100), // Too large for i64
            ],
        };

        // Evaluate the expression and check the error
        let result = processor.evaluate_expression(&expr);
        assert!(result.is_err());

        // Verify the error type and message
        match result {
            Err(PecosError::Computation(msg)) => {
                assert!(msg.contains("shift amount out of range"));
                assert!(msg.contains("10 << 100"));
            }
            _ => panic!("Expected Computation error but got: {result:?}"),
        }
    }

    // Test integer overflow detection
    #[test]
    fn test_integer_overflow() {
        let processor = OperationProcessor::new();

        // Create a multiplication that will overflow
        let expr = Expression::Operation {
            cop: "*".to_string(),
            args: vec![ArgItem::Integer(i64::MAX), ArgItem::Integer(2)],
        };

        // Evaluate the expression and check the error
        let result = processor.evaluate_expression(&expr);
        assert!(result.is_err());

        // Verify the error type and message
        match result {
            Err(PecosError::Computation(msg)) => {
                assert!(msg.contains("overflow"));
                assert!(msg.contains("multiplication"));
            }
            _ => panic!("Expected Computation error but got: {result:?}"),
        }
    }

    // Test our error handling directly on an expression with an invalid variable
    #[test]
    fn test_invalid_variable_in_expression() {
        let processor = OperationProcessor::new();

        // Create an expression with a reference to non-existent variable
        let expr = Expression::Operation {
            cop: "+".to_string(),
            args: vec![
                ArgItem::Integer(5),
                ArgItem::Simple("nonexistent".to_string()),
            ],
        };

        // Evaluate the expression and check the error
        let result = processor.evaluate_expression(&expr);
        assert!(result.is_err());

        // Verify the error type and message
        match result {
            Err(e) => {
                let error_str = e.to_string();
                assert!(
                    error_str.contains("nonexistent"),
                    "Error message should mention the missing variable: {error_str}"
                );
            }
            _ => panic!("Expected error but got success"),
        }
    }
}
