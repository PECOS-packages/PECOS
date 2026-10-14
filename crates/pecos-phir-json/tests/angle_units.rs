mod common;

#[cfg(test)]
mod tests {
    use pecos_core::errors::PecosError;
    use pecos_core::{Angle64, Gate};
    use pecos_engines::{ControlEngine, EngineStage};
    use pecos_phir_json::v0_1::ast::PHIRProgram;
    use pecos_phir_json::v0_1::engine::PhirJsonEngine;

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

        // Parse JSON into PHIRProgram
        let program: PHIRProgram = serde_json::from_str(phir_json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

        let mut engine = PhirJsonEngine::from_program(program)?;
        let EngineStage::NeedsProcessing(commands) = engine.start(())? else {
            panic!("expected rotation and measurement commands");
        };
        assert_eq!(
            commands.quantum_ops()?,
            vec![
                Gate::rz(Angle64::QUARTER_TURN, &[0]),
                Gate::rz(Angle64::QUARTER_TURN, &[1]),
                Gate::rz(Angle64::QUARTER_TURN, &[2]),
                Gate::rxy1q(Angle64::ZERO, Angle64::HALF_TURN, &[0]),
                Gate::rxy1q(Angle64::ZERO, Angle64::HALF_TURN, &[1]),
                Gate::rxy1q(Angle64::ZERO, Angle64::HALF_TURN, &[2]),
                Gate::mz(&[0]),
                Gate::mz(&[1]),
                Gate::mz(&[2]),
            ]
        );

        Ok(())
    }
}
