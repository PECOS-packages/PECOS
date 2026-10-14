mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;

    use pecos_engines::{ClassicalControlEngineBuilder, StateVectorEngineBuilder};
    use pecos_phir_json::phir_json_engine;
    use std::collections::BTreeSet;

    // Test 1: Basic quantum gate operations and measurement
    #[test]
    fn test_basic_gates_and_measurement() -> Result<(), PecosError> {
        // Define the program inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 1
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 1},
            {"qop": "X", "args": [["q", 0]], "returns": []},
            {"cop": "=", "args": [0], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": ["m"], "returns": ["output"]}
          ]
        }"#;

        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(32)?;
        assert_eq!(results.shots.len(), 32);
        let outcomes: BTreeSet<_> = results
            .shots
            .iter()
            .map(|shot| shot.data["output"].as_u32().unwrap())
            .collect();
        assert_eq!(outcomes, BTreeSet::from([1]));

        Ok(())
    }

    // Test 1b: Hadamard pinned deterministically. H Z H is X, so this measures 1 every
    // shot -- and it fails if H is the identity or X, both of which measure 0. A
    // superposition test over shots cannot distinguish those; this can.
    #[test]
    fn test_hadamard_conjugates_z_to_x() -> Result<(), PecosError> {
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 1
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 1},
            {"qop": "H", "args": [["q", 0]], "returns": []},
            {"qop": "Z", "args": [["q", 0]], "returns": []},
            {"qop": "H", "args": [["q", 0]], "returns": []},
            {"cop": "=", "args": [0], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": ["m"], "returns": ["output"]}
          ]
        }"#;

        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(32)?;
        assert_eq!(results.shots.len(), 32);
        let outcomes: BTreeSet<_> = results
            .shots
            .iter()
            .map(|shot| shot.data["output"].as_u32().unwrap())
            .collect();
        assert_eq!(outcomes, BTreeSet::from([1]));

        Ok(())
    }

    // Test 2: Bell state preparation
    #[test]
    fn test_bell_state() -> Result<(), PecosError> {
        // Define the Bell state program inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 2
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]], "returns": []},
            {"qop": "CX", "args": [["q", 0], ["q", 1]], "returns": []},
            {"cop": "=", "args": [0], "returns": [["m", 0]]},
            {"cop": "=", "args": [0], "returns": [["m", 1]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["output"]}
          ]
        }"#;

        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(32)?;
        assert_eq!(results.shots.len(), 32);
        let outcomes: BTreeSet<_> = results
            .shots
            .iter()
            .map(|shot| shot.data["output"].as_u32().unwrap())
            .collect();
        assert_eq!(outcomes, BTreeSet::from([0, 3]));

        Ok(())
    }

    // Test 3: Testing rotation gates
    #[test]
    fn test_rotation_gates() -> Result<(), PecosError> {
        // Define rotation gates test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 1
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 1},
            {"qop": "X", "args": [["q", 0]], "returns": []},
            {"qop": "RZ", "angles": [[1.5707963267948966], "rad"], "args": [["q", 0]], "returns": []},
            {"qop": "R1XY", "angles": [[0.0, 3.141592653589793], "rad"], "args": [["q", 0]], "returns": []},
            {"cop": "=", "args": [0], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": ["m"], "returns": ["output"]}
          ]
        }"#;

        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(32)?;
        assert_eq!(results.shots.len(), 32);
        let outcomes: BTreeSet<_> = results
            .shots
            .iter()
            .map(|shot| shot.data["output"].as_u32().unwrap())
            .collect();
        assert_eq!(outcomes, BTreeSet::from([1]));

        Ok(())
    }

    // Test 4: Testing qparallel blocks
    #[test]
    fn test_qparallel_blocks() -> Result<(), PecosError> {
        // Define qparallel test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 2
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 2},
            {
              "block": "qparallel",
              "ops": [
                {"qop": "X", "args": [["q", 0]], "returns": []},
                {"qop": "X", "args": [["q", 1]], "returns": []}
              ]
            },
            {"cop": "=", "args": [0], "returns": [["m", 0]]},
            {"cop": "=", "args": [1], "returns": [["m", 1]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["output"]}
          ]
        }"#;

        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(32)?;
        assert_eq!(results.shots.len(), 32);
        let outcomes: BTreeSet<_> = results
            .shots
            .iter()
            .map(|shot| shot.data["output"].as_u32().unwrap())
            .collect();
        assert_eq!(outcomes, BTreeSet::from([3]));

        Ok(())
    }

    // Test 5: Complex example with control flow and quantum operations
    #[test]
    fn test_control_flow_with_quantum() -> Result<(), PecosError> {
        // Define control flow test inline
        let phir_json = r#"{
          "format": "PHIR/JSON",
          "version": "0.1.0",
          "metadata": {
            "num_qubits": 1
          },
          "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "i32", "variable": "condition", "size": 31},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 1},
            {"cop": "=", "args": [1], "returns": ["condition"]},
            {
              "block": "if",
              "condition": {"cop": "==", "args": ["condition", 1]},
              "true_branch": [
                {"qop": "X", "args": [["q", 0]], "returns": []}
              ],
              "false_branch": [
                {"qop": "H", "args": [["q", 0]], "returns": []}
              ]
            },
            {"cop": "=", "args": [0], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": ["m"], "returns": ["output"]}
          ]
        }"#;

        let results = phir_json_engine()
            .json(phir_json)?
            .to_sim()
            .quantum(StateVectorEngineBuilder::default())
            .seed(42)
            .run(32)?;
        assert_eq!(results.shots.len(), 32);
        let outcomes: BTreeSet<_> = results
            .shots
            .iter()
            .map(|shot| shot.data["output"].as_u32().unwrap())
            .collect();
        assert_eq!(outcomes, BTreeSet::from([1]));

        Ok(())
    }
}
