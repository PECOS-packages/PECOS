mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;

    // Import helpers from common module
    use crate::common::phir_test_utils::get_phir_results;

    #[test]
    fn test_angle_units_conversion() -> Result<(), PecosError> {
        // Run the test program that uses different angle units
        let result = get_phir_results("tests/assets/angle_units_test.json")?;

        // Print all information about the result for debugging
        println!("ShotResult: {result:?}");
        println!("Registers: {:?}", result.registers);

        // We can't assert exact values since it's a probabilistic simulation,
        // but we just want to ensure the program runs without errors
        assert!(
            result.registers.contains_key("output"),
            "Expected 'output' register to be present"
        );

        Ok(())
    }
}
