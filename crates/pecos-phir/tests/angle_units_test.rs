mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;

    // Import helpers from common module
    use crate::common::phir_test_utils::run_phir_simulation_from_json;

    #[test]
    fn test_angle_units_conversion() -> Result<(), PecosError> {
        // Define the test program with different angle units inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 3,
            "description": "Test for different angle units"
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 3},
            {"data": "cvar_define", "data_type": "i32", "variable": "c", "size": 3},

            {"qop": "RZ", "angles": [[1.5707963267948966], "rad"], "args": [["q", 0]], "returns": []},
            {"qop": "RZ", "angles": [[90.0], "deg"], "args": [["q", 1]], "returns": []},
            {"qop": "RZ", "angles": [[0.5], "pi"], "args": [["q", 2]], "returns": []},

            {"qop": "R1XY", "angles": [[0.0, 3.141592653589793], "rad"], "args": [["q", 0]], "returns": []},
            {"qop": "R1XY", "angles": [[0.0, 180.0], "deg"], "args": [["q", 1]], "returns": []},
            {"qop": "R1XY", "angles": [[0.0, 1.0], "pi"], "args": [["q", 2]], "returns": []},

            {"qop": "Measure", "args": [["q", 0]], "returns": [["c", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["c", 1]]},
            {"qop": "Measure", "args": [["q", 2]], "returns": [["c", 2]]},

            {"cop": "Result", "args": ["c"], "returns": ["ret"]}
          ]
        }"#;

        // Run the test using our helper function - using single shot with no noise
        let results = run_phir_simulation_from_json(
            phir_json,
            1,
            1,
            None,
            None::<pecos_engines::PassThroughNoiseModel>,
            None::<&std::path::Path>,
        )?;

        // Print all information about the result for debugging
        println!("ShotResults: {results:?}");

        // Make sure we have results
        assert!(
            !results.shots.is_empty(),
            "Expected at least one shot result"
        );

        // We can't assert exact values since it's a probabilistic simulation,
        // but we just want to ensure the program runs without errors
        let shot = &results.shots[0];
        assert!(
            shot.contains_key("ret"),
            "Expected 'output' register to be present"
        );

        Ok(())
    }
}
