//! Per-location probability cardinality is checked at both public entry points.

use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilderError, DemSampler, DetectorValidationError, NoiseConfig, SamplingEngine,
};
use pecos_quantum::DagCircuit;

fn check_probability_length(supplied: usize, engine: bool) {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.mz(&[0]);
    let map = InfluenceBuilder::new(&circuit).unwrap().build().unwrap();
    assert_eq!(map.locations.len(), 2);
    let probabilities = vec![0.1; supplied];
    let message = if engine {
        match SamplingEngine::from_influence_map(&map, &probabilities, &NoiseConfig::default()) {
            Err(DemBuilderError::ConfigurationError(message)) => message,
            result => panic!("expected ConfigurationError, got {result:?}"),
        }
    } else {
        match DemSampler::from_influence_map(&map, &probabilities) {
            Err(DetectorValidationError::InvalidConfiguration { message }) => message,
            result => panic!("expected InvalidConfiguration, got {result:?}"),
        }
    };
    assert!(
        message.contains(&format!("received {supplied}")),
        "{message}"
    );
    assert!(message.contains("expected 2"), "{message}");
    assert!(
        message.contains("pass one probability per influence-map location"),
        "{message}"
    );
}

#[test]
fn sampling_engine_rejects_empty_probability_slice() {
    check_probability_length(0, true);
}

#[test]
fn sampling_engine_rejects_short_probability_slice() {
    check_probability_length(1, true);
}

#[test]
fn sampling_engine_rejects_long_probability_slice() {
    check_probability_length(3, true);
}

#[test]
fn dem_sampler_rejects_empty_probability_slice() {
    check_probability_length(0, false);
}

#[test]
fn dem_sampler_rejects_short_probability_slice() {
    check_probability_length(1, false);
}

#[test]
fn dem_sampler_rejects_long_probability_slice() {
    check_probability_length(3, false);
}
