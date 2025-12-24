//! Tests for `GeneralNoise` JSON configuration

use pecos_qasm::config::{NoiseConfig, QuantumEngineConfig};
use pecos_qasm::simulation::NoiseModelType;
use serde_json::json;

#[test]
fn test_general_noise_json_simple() {
    let json_config = json!({
        "type": "GeneralNoise",
        "p1": 0.001,
        "p2": 0.01,
        "p_prep": 0.001,
        "p_meas_0": 0.001,
        "p_meas_1": 0.001
    });

    let noise_config: NoiseConfig = serde_json::from_value(json_config).unwrap();

    match &noise_config {
        NoiseConfig::GeneralNoise(fields) => {
            assert_eq!(fields.p1, Some(0.001));
            assert_eq!(fields.p2, Some(0.01));
            assert_eq!(fields.p_prep, Some(0.001));
            assert_eq!(fields.p_meas_0, Some(0.001));
            assert_eq!(fields.p_meas_1, Some(0.001));
        }
        _ => panic!("Expected GeneralNoise variant"),
    }

    // Test conversion to NoiseModelType
    let noise_model: NoiseModelType = noise_config.into();
    match noise_model {
        NoiseModelType::General(_) => {
            // Success - it should create a General variant
        }
        _ => panic!("Expected General variant"),
    }
}

#[test]
fn test_general_noise_json_complex() {
    let json_config = json!({
        "type": "GeneralNoise",
        "p1": 0.001,
        "p2": 0.01,
        "scale": 1.5,
        "noiseless_gates": ["H", "CX"],
        "p1_pauli_model": {
            "X": 0.5,
            "Y": 0.3,
            "Z": 0.2
        },
        "p2_pauli_model": {
            "IX": 0.1,
            "IY": 0.06,
            "IZ": 0.08,
            "XI": 0.1,
            "XX": 0.06,
            "XY": 0.06,
            "XZ": 0.06,
            "YI": 0.06,
            "YX": 0.06,
            "YY": 0.06,
            "YZ": 0.06,
            "ZI": 0.08,
            "ZX": 0.06,
            "ZY": 0.06,
            "ZZ": 0.04
        },
        "p_idle_coherent": false,
        "p_idle_linear_rate": 0.0001,
        "leakage_scale": 0.5,
        "emission_scale": 0.8,
        "p2_angle_params": [1.0, 0.5, 1.2, 0.3],
        "p2_angle_power": 2.0
    });

    let noise_config: NoiseConfig = serde_json::from_value(json_config).unwrap();

    match &noise_config {
        NoiseConfig::GeneralNoise(fields) => {
            assert_eq!(fields.p1, Some(0.001));
            assert_eq!(fields.p2, Some(0.01));
            assert_eq!(fields.scale, Some(1.5));
            assert_eq!(
                fields.noiseless_gates,
                Some(vec!["H".to_string(), "CX".to_string()])
            );
            assert!(fields.p1_pauli_model.is_some());
            assert!(fields.p2_pauli_model.is_some());
            assert_eq!(fields.p_idle_coherent, Some(false));
            assert_eq!(fields.p_idle_linear_rate, Some(0.0001));
            assert_eq!(fields.leakage_scale, Some(0.5));
            assert_eq!(fields.emission_scale, Some(0.8));
            assert_eq!(fields.p2_angle_params, Some((1.0, 0.5, 1.2, 0.3)));
            assert_eq!(fields.p2_angle_power, Some(2.0));
        }
        _ => panic!("Expected GeneralNoise variant"),
    }
}

#[test]
fn test_general_noise_json_minimal() {
    // Test that GeneralNoise works with no parameters (all defaults)
    let json_config = json!({
        "type": "GeneralNoise"
    });

    let noise_config: NoiseConfig = serde_json::from_value(json_config).unwrap();

    match &noise_config {
        NoiseConfig::GeneralNoise(_) => {
            // Success - should parse with all None values
        }
        _ => panic!("Expected GeneralNoise variant"),
    }
}

#[derive(serde::Deserialize)]
struct SimConfig {
    quantum_engine: QuantumEngineConfig,
    noise: NoiseConfig,
}

#[test]
fn test_full_simulation_config() {
    let json_config = json!({
        "quantum_engine": "StateVector",
        "noise": {
            "type": "GeneralNoise",
            "p1": 0.001,
            "p2": 0.01
        }
    });

    let config: SimConfig = serde_json::from_value(json_config).unwrap();

    match &config.quantum_engine {
        QuantumEngineConfig::StateVector => {}
        QuantumEngineConfig::SparseStabilizer => panic!("Expected StateVector engine"),
    }

    match &config.noise {
        NoiseConfig::GeneralNoise(fields) => {
            assert_eq!(fields.p1, Some(0.001));
            assert_eq!(fields.p2, Some(0.01));
        }
        _ => panic!("Expected GeneralNoise"),
    }
}

#[test]
fn test_general_noise_unknown_field_error() {
    // Test that unknown fields cause deserialization errors
    let json_config = json!({
        "type": "GeneralNoise",
        "p1": 0.001,
        "p2": 0.01,
        "unknown_parameter": 0.5  // This should cause an error
    });

    let result: Result<NoiseConfig, _> = serde_json::from_value(json_config);
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("unknown field"));
}

#[test]
fn test_general_noise_typo_in_field_name() {
    // Test that typos in field names are caught
    let json_config = json!({
        "type": "GeneralNoise",
        "p_1": 0.001,  // Should be "p1" not "p_1"
        "p2": 0.01
    });

    let result: Result<NoiseConfig, _> = serde_json::from_value(json_config);
    assert!(result.is_err());

    // The error should mention the unknown field
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("unknown field `p_1`"));
}
