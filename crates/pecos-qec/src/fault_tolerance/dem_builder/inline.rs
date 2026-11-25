// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Exact DEM conversion for inline binary and uniform depolarizing Pauli channels.
//! Channel payloads are categorical. Never interpret their branches as independent
//! faults without an exact conversion, or add gate-triggered noise on top.

use super::{DemBuilder, DemBuilderError, DetectorErrorModel, FaultMechanism, NoiseConfig};
use crate::fault_tolerance::circuit_definitions::definitions_from_dag_circuit;
use crate::fault_tolerance::influence_builder::InfluenceBuilder;
use crate::fault_tolerance::propagator::{DagFaultAnalyzer, Pauli};
use pecos_core::gate_type::GateType;
use pecos_core::{ChannelExpr, Gate};
use pecos_quantum::{Attribute, DagCircuit, TickCircuit};
use std::collections::{BTreeMap, BTreeSet};

/// Circuit-level declaration for an inline-noise circuit, including zero noise.
/// Set to the string `inline` when no external gate-rate model should be applied.
pub const INLINE_NOISE_ATTRIBUTE: &str = "pecos_noise_source";

/// Whether a DAG carries inline channels or explicitly declares inline noise.
#[must_use]
pub fn uses_inline_noise(circuit: &DagCircuit) -> bool {
    circuit.iter_gates().any(|(_, g)| g.is_channel())
        || matches!(circuit.get_attr(INLINE_NOISE_ATTRIBUTE), Some(Attribute::String(s)) if s == "inline")
}

/// Whether a tick circuit carries inline channels or declares inline noise.
#[must_use]
pub fn tick_uses_inline_noise(circuit: &TickCircuit) -> bool {
    circuit.has_channel_operations()
        || matches!(circuit.get_meta(INLINE_NOISE_ATTRIBUTE), Some(Attribute::String(s)) if s == "inline")
}

fn invalid(message: impl Into<String>) -> DemBuilderError {
    DemBuilderError::ConfigurationError(message.into())
}

fn require_no_external_noise(noise: &NoiseConfig) -> Result<(), DemBuilderError> {
    let rates = [
        noise.p1,
        noise.p2,
        noise.p_meas,
        noise.p_prep,
        noise.p_idle,
        noise.idle_rz,
        noise.p_idle_linear.rate,
        noise.p_idle_quadratic.rate,
        noise.p_idle_quadratic_sine.rate,
        noise.p_meas_crosstalk_local,
        noise.p_meas_crosstalk_global,
    ];
    if rates
        .iter()
        .chain(noise.p1_gate_rates.values())
        .chain(noise.p2_gate_rates.values())
        .any(|&p| p != 0.0)
        || noise.t1.is_some()
        || noise.t2.is_some()
    {
        return Err(invalid(
            "inline channels cannot be combined with external gate/idle/readout noise; use zero rates",
        ));
    }
    Ok(())
}

/// Convert a payload to independent (Pauli word index, probability) terms.
/// Word digits are I,X,Y,Z, least-significant digit on gate.qubits[0].
fn independent_terms(gate: &Gate) -> Result<Vec<(usize, f64)>, DemBuilderError> {
    if !(1..=2).contains(&gate.qubits.len()) {
        return Err(invalid(
            "inline DEM supports one- and two-qubit Pauli channels",
        ));
    }
    let ops = match gate.channel_expr() {
        Some(ChannelExpr::Unitary(u)) => vec![(1.0, u.clone())],
        Some(ChannelExpr::MixedUnitary(ops)) => ops.clone(),
        _ => {
            return Err(invalid(
                "inline DEM requires a Pauli mixed-unitary channel payload",
            ));
        }
    };
    let size = 1 << (2 * gate.qubits.len());
    let mut probabilities = vec![0.0; size];
    for (p, unitary) in ops {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(invalid(
                "inline channel probabilities must be finite and in [0, 1]",
            ));
        }
        let pauli = unitary
            .try_to_pauli_string()
            .ok_or_else(|| invalid("inline DEM cannot represent a non-Pauli channel"))?;
        let mut word = 0;
        for (p, q) in pauli.paulis() {
            let position = gate
                .qubits
                .iter()
                .position(|target| target == q)
                .ok_or_else(|| {
                    invalid("inline channel Pauli target is outside its gate support")
                })?;
            let digit = match p {
                pecos_core::Pauli::I => 0,
                pecos_core::Pauli::X => 1,
                pecos_core::Pauli::Y => 2,
                pecos_core::Pauli::Z => 3,
            };
            word |= digit << (2 * position);
        }
        probabilities[word] += p;
    }
    match exact_pauli_terms(&probabilities)? {
        ExactPauliConversion::Exact(terms) => Ok(terms),
        ExactPauliConversion::Unavailable(message) => Err(invalid(message)),
    }
}

/// Outcome of the supported exact conversion, distinct from invalid input.
pub(crate) enum ExactPauliConversion {
    Exact(Vec<(usize, f64)>),
    Unavailable(&'static str),
}

/// Exact binary/uniform categorical conversion shared with located channels.
#[allow(clippy::float_cmp)] // Exact uniformity and the fully mixed endpoint must not be approximated.
pub(crate) fn exact_pauli_terms(
    probabilities: &[f64],
) -> Result<ExactPauliConversion, DemBuilderError> {
    let size = probabilities.len();
    let total: f64 = probabilities.iter().sum();
    if !total.is_finite() || (total - 1.0).abs() > 1e-12 {
        return Err(invalid("inline channel probabilities must sum to 1"));
    }
    let nonidentity: Vec<_> = probabilities
        .iter()
        .copied()
        .enumerate()
        .skip(1)
        .filter(|(_, p)| *p > 0.0)
        .collect();
    if nonidentity.len() <= 1 {
        return Ok(ExactPauliConversion::Exact(nonidentity));
    }
    let weight = probabilities[1];
    if probabilities[1..].iter().any(|&p| p != weight) {
        return Ok(ExactPauliConversion::Unavailable(
            "inline DEM supports binary Pauli and uniform depolarizing channels; this categorical distribution has no supported exact conversion",
        ));
    }
    // (1-2q)^(N/2) = 1-N*w, N=4^n and each nonidentity branch has weight w.
    // Above the fully mixed point the full-Pauli factorization is unavailable.
    // Inline callers reject it; located channels may still fit its projected signatures.
    let attenuation = weight * f64::from(u32::try_from(size).expect("size <= 16"));
    if attenuation > 1.0 {
        return Ok(ExactPauliConversion::Unavailable(
            "inline depolarization exceeds the supported exact independent-channel range",
        ));
    }
    let q = if attenuation == 1.0 {
        0.5
    } else {
        -((-attenuation).ln_1p() / (f64::from(u32::try_from(size).expect("size <= 16")) / 2.0))
            .exp_m1()
            / 2.0
    };
    Ok(ExactPauliConversion::Exact(
        (1..size).map(|word| (word, q)).collect(),
    ))
}

pub(super) fn build_inline_dem(
    circuit: &DagCircuit,
    noise: &NoiseConfig,
) -> Result<DetectorErrorModel, DemBuilderError> {
    require_no_external_noise(noise)?;
    let definitions = definitions_from_dag_circuit(circuit)?;
    let mut ideal = circuit.clone();
    let mut channels = Vec::new();
    for (node, gate) in circuit.iter_gates() {
        if gate.is_channel() {
            let terms = independent_terms(gate)
                .map_err(|e| invalid(format!("inline channel at node {node}: {e}")))?;
            channels.push((node, gate.qubits.clone(), terms));
            ideal
                .update_gate(node, |g| {
                    *g = Gate::simple(GateType::I, gate.qubits.clone());
                })
                .map_err(|e| invalid(e.to_string()))?;
        }
    }
    let mut influence = DagFaultAnalyzer::new(&ideal)
        .build_influence_map()
        .map_err(DemBuilderError::UnsupportedGate)?;
    let annotation_builder = InfluenceBuilder::new(&ideal)
        .map_err(DemBuilderError::UnsupportedGate)?
        .with_circuit_annotations()
        .map_err(|e| invalid(e.to_string()))?;
    let info = annotation_builder
        .run_symbolic_simulation()
        .map_err(|e| invalid(e.to_string()))?;
    // Symbolic outcomes carry independent random-bit supports. Their XOR must
    // be empty for each declared readout parity. Constant signs need not be zero:
    // a DEM describes deviations, so affine constants and ideal signs cancel.
    for (kind, id, measurements) in definitions
        .detectors
        .iter()
        .map(|d| ("detector", d.id, &d.measurements))
        .chain(
            definitions
                .observables
                .iter()
                .map(|o| ("observable", o.id, &o.measurements)),
        )
    {
        let mut dependencies = BTreeSet::new();
        for &m in measurements {
            let (node, _, _) = influence.measurements[m];
            let index = info
                .node_to_meas_idx
                .get(node)
                .copied()
                .flatten()
                .ok_or_else(|| invalid("inline DEM symbolic measurement order is incomplete"))?;
            let result = info
                .history
                .get(index)
                .ok_or_else(|| invalid("inline DEM symbolic measurement history is incomplete"))?;
            for bit in &result.outcome {
                if !dependencies.insert(bit) {
                    dependencies.remove(&bit);
                }
            }
        }
        if !dependencies.is_empty() {
            return Err(invalid(format!(
                "inline DEM requires deterministic ideal {kind} {id}"
            )));
        }
    }
    let annotations = annotation_builder
        .build()
        .map_err(|e| invalid(e.to_string()))?;
    influence.merge_dem_outputs_from(&annotations);
    let builder = DemBuilder::new(&influence)
        .with_noise(0.0, 0.0, 0.0, 0.0)
        .with_circuit_definitions(definitions)?;
    let mut dem = builder.try_build()?;
    let (detectors, observables) = builder.build_measurement_mappings();
    let locations: BTreeMap<_, _> = influence
        .locations
        .iter()
        .enumerate()
        .map(|(index, loc)| ((loc.node, loc.qubits[0]), index))
        .collect();
    for (node, qubits, terms) in channels {
        for (word, probability) in terms {
            if probability == 0.0 {
                continue;
            }
            let mut effect = FaultMechanism::from_unsorted([], []);
            for (position, q) in qubits.iter().enumerate() {
                let digit = u8::try_from((word >> (2 * position)) & 3).expect("Pauli digit <= 3");
                if digit != 0 {
                    let index = locations
                        .get(&(node, *q))
                        .ok_or_else(|| invalid("inline channel has no propagation location"))?;
                    effect = effect.xor(&builder.compute_mechanism(
                        *index,
                        Pauli::from_u8(digit),
                        &detectors,
                        &observables,
                    ));
                }
            }
            if !effect.is_empty() {
                dem.add_direct_contribution(effect, probability);
            }
        }
    }
    Ok(dem)
}

#[cfg(test)]
mod tests {
    use super::super::DemSampler;
    use super::*;
    use pecos_core::channel;

    fn memory(channel: ChannelExpr) -> TickCircuit {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().channel(channel);
        let measurements = circuit.tick().mz(&[1, 0]);
        circuit.detector(&measurements).unwrap();
        circuit.observable(&measurements[1..]).unwrap();
        circuit
    }

    fn build(circuit: &TickCircuit) -> Result<DetectorErrorModel, DemBuilderError> {
        DemBuilder::try_from_tick_circuit(circuit, 0.0, 0.0, 0.0, 0.0)
    }

    #[test]
    fn categorical_binary_fault_matches_records_and_annotations_once() {
        let circuit = memory(channel::BitFlip(0.125, 0));
        let dem = build(&circuit).unwrap();
        assert_eq!(dem.to_mechanisms().0, vec![(0.125, vec![0], vec![0])]);
        let dag = DagCircuit::try_from(&circuit).unwrap();
        assert_eq!(
            DemBuilder::from_circuit(&dag, 0.0, 0.0, 0.0, 0.0)
                .unwrap()
                .to_mechanisms(),
            dem.to_mechanisms()
        );
        let sampler =
            DemSampler::from_tick_circuit(&circuit, &NoiseConfig::new(0.0, 0.0, 0.0, 0.0)).unwrap();
        assert_eq!(sampler.num_detectors(), 1);
        assert_eq!(sampler.num_observables(), 1);
    }

    #[test]
    #[allow(clippy::float_cmp)] // The fixtures include exact channel-domain endpoints.
    fn depolarization_exact_projection_and_endpoints() {
        for p in [0.0, 1e-25, 0.01, 0.9375] {
            let dem = build(&memory(channel::Depolarizing2(p, 0, 1))).unwrap();
            let mechanisms = dem.to_mechanisms().0;
            if p == 0.0 {
                assert_eq!(mechanisms.len(), 0);
                continue;
            }
            let expected = if p == 0.9375 {
                0.5
            } else {
                -((-16.0 * p / 15.0).ln_1p() / 2.0).exp_m1() / 2.0
            };
            assert_eq!(mechanisms.len(), 3);
            for (actual, _, _) in mechanisms {
                assert!((actual - expected).abs() <= expected * 1e-12);
            }
        }
        assert!(
            build(&memory(channel::Depolarizing2(0.99, 0, 1)))
                .unwrap_err()
                .to_string()
                .contains("range")
        );
        assert!(
            build(&memory(channel::Depolarizing(1.0, 0)))
                .unwrap_err()
                .to_string()
                .contains("range")
        );
    }

    #[test]
    fn unsupported_categorical_and_non_pauli_channels_are_rejected() {
        assert!(
            build(&memory(channel::PauliChannel(0.1, 0.2, 0.03, 0)))
                .unwrap_err()
                .to_string()
                .contains("categorical")
        );
        assert!(
            build(&memory(channel::AmplitudeDamping(0.1, 0)))
                .unwrap_err()
                .to_string()
                .contains("Pauli")
        );
    }

    #[test]
    fn external_noise_and_nondeterministic_parities_are_rejected() {
        let circuit = memory(channel::BitFlip(0.1, 0));
        assert!(
            DemBuilder::try_from_tick_circuit(&circuit, 0.001, 0.0, 0.0, 0.0)
                .unwrap_err()
                .to_string()
                .contains("external")
        );
        let mut random = TickCircuit::new();
        random.tick().pz(&[0]);
        random.tick().h(&[0]);
        random.tick().channel(channel::BitFlip(0.1, 0));
        let m = random.tick().mz(&[0]);
        random.detector(&m).unwrap();
        assert!(
            build(&random)
                .unwrap_err()
                .to_string()
                .contains("deterministic ideal detector")
        );
    }

    #[test]
    fn duplicate_branches_keep_sum_error() {
        let expression = channel::BitFlip(0.6, 0);
        let ChannelExpr::MixedUnitary(mut ops) = expression else {
            panic!("binary mixture")
        };
        let x = ops.pop().unwrap().1;
        let gate = Gate::channel(ChannelExpr::MixedUnitary(vec![(0.6, x.clone()), (0.6, x)]));
        let error = independent_terms(&gate).unwrap_err().to_string();
        assert!(
            error.contains("inline channel probabilities must sum to 1"),
            "{error}"
        );
    }

    #[test]
    fn declaration_without_payload_is_explicit_zero_noise() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        let m = circuit.tick().mz(&[0]);
        circuit.detector(&m).unwrap();
        circuit.set_meta(INLINE_NOISE_ATTRIBUTE, Attribute::String("inline".into()));
        assert!(tick_uses_inline_noise(&circuit));
        assert_eq!(build(&circuit).unwrap().to_mechanisms().0.len(), 0);
    }
}
