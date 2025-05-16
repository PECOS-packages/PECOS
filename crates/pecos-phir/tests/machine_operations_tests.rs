mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_engines::PassThroughNoiseModel;

    // Import helpers from common module
    use crate::common::phir_test_utils::run_phir_simulation_from_json;

    // Test machine operations
    #[test]
    fn test_machine_operations() -> Result<(), PecosError> {
        // Define the PHIR program inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 2
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 32},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 32},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "CX", "args": [["q", 0], ["q", 1]]},
            {"mop": "Idle", "args": [["q", 0], ["q", 1]], "duration": [5.0, "ms"]},
            {"mop": "Transport", "args": [["q", 0]], "duration": [2.0, "us"], "metadata": {"from_position": [0, 0], "to_position": [1, 0]}},
            {"mop": "Skip"},
            {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]},
            {"cop": "=", "args": [2], "returns": ["result"]},
            {"cop": "Result", "args": ["result"], "returns": ["output"]}
          ]
        }"#;

        // Run the simulation with single shot
        let results = run_phir_simulation_from_json(
            phir_json,
            1,
            1,
            None,
            None::<PassThroughNoiseModel>,
            None::<&std::path::Path>,
        )?;

        // Print results information for debugging
        println!("ShotResults: {results:?}");

        // The actual result value will depend on the quantum simulation,
        // but we just need to verify that the engine successfully processes
        // machine operations without errors and exports the result value
        assert!(!results.shots.is_empty(), "Expected non-empty results");

        let shot = &results.shots[0];
        assert!(
            shot.contains_key("output"),
            "Expected 'output' register to be present"
        );

        // Check that the value is 2 (from the assignment in the JSON)
        assert_eq!(
            shot.get("output").unwrap(),
            "2",
            "Expected output to be 2, got {}",
            shot.get("output").unwrap()
        );

        Ok(())
    }

    // Test simple machine operations
    #[test]
    fn test_simple_machine_operations() -> Result<(), PecosError> {
        // Define the PHIR program inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 2
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 32},
            {"qop": "H", "args": [["q", 0]]},
            {"mop": "Idle", "args": [["q", 0], ["q", 1]], "duration": [5.0, "ms"]},
            {"mop": "Delay", "args": [["q", 0]], "duration": [2.0, "us"]},
            {"mop": "Transport", "args": [["q", 1]], "duration": [1.0, "ms"], "metadata": {"from_position": [0, 0], "to_position": [1, 0]}},
            {"mop": "Timing", "args": [["q", 0], ["q", 1]], "metadata": {"timing_type": "sync", "label": "sync_point_1"}},
            {"qop": "CX", "args": [["q", 0], ["q", 1]]},
            {"cop": "=", "args": [42], "returns": ["result"]},
            {"cop": "Result", "args": ["result"], "returns": ["output"]}
          ]
        }"#;

        // Run the simulation with single shot
        let results = run_phir_simulation_from_json(
            phir_json,
            1,
            1,
            None,
            None::<PassThroughNoiseModel>,
            None::<&std::path::Path>,
        )?;

        // Print results information for debugging
        println!("ShotResults: {results:?}");

        // The actual result value will depend on the quantum simulation,
        // but we just need to verify that the engine successfully processes
        // simple machine operations without errors
        assert!(!results.shots.is_empty(), "Expected non-empty results");

        let shot = &results.shots[0];
        assert!(
            shot.contains_key("output"),
            "Expected 'output' register to be present"
        );

        // Check that the value is 42 (from the assignment in the JSON file)
        assert_eq!(
            shot.get("output").unwrap(),
            "42",
            "Expected output to be 42, got {}",
            shot.get("output").unwrap()
        );

        Ok(())
    }
}
