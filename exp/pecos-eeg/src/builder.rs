// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! EEG DEM builder: TickCircuit + noise → DEM events.

use crate::Bm;
use crate::circuit::{self, NoiseModel};
use crate::dem_mapping::{self, DemEntry, Detector, Observable};
use crate::expand;
use crate::stabilizer::StabilizerGroup;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_quantum::{AnnotationKind, TickCircuit};

pub struct EegDemBuilder<'a> {
    tc: &'a TickCircuit,
    noise: NoiseModel,
    config: dem_mapping::EegConfig,
}

impl<'a> EegDemBuilder<'a> {
    #[must_use]
    pub fn from_tick_circuit(tc: &'a TickCircuit) -> Self {
        Self {
            tc,
            noise: NoiseModel::coherent_only(0.0),
            config: dem_mapping::EegConfig::default(),
        }
    }

    #[must_use]
    pub fn noise(mut self, noise: NoiseModel) -> Self {
        self.noise = noise;
        self
    }

    /// Set the full EEG configuration.
    #[must_use]
    pub fn config(mut self, config: dem_mapping::EegConfig) -> Self {
        self.config = config;
        self
    }

    /// Use the exact sin^2(h) formula instead of leading-order h^2.
    #[must_use]
    pub fn exact_h_formula(mut self) -> Self {
        self.config.h_formula = dem_mapping::HFormula::SinSquared;
        self
    }

    /// Use second-order BCH (includes [H,H] commutator corrections).
    #[must_use]
    pub fn bch_order_2(mut self) -> Self {
        self.config.bch_order = dem_mapping::BchOrder::Second;
        self
    }

    #[must_use]
    pub fn build(&self) -> Vec<DemEntry> {
        let gates: Vec<pecos_core::Gate> = self
            .tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates);
        let result = circuit::analyze_expanded(&expanded.gates, &self.noise);
        let (detectors, observables) = build_detectors(self.tc, &expanded);

        // Compute stabilizer group from the EXPANDED circuit (pre-readout).
        // This includes auxiliary qubits, so beta function checks happen
        // directly in the expanded frame without lossy frame mapping.
        // Exclude the final deferred MZ(aux) gates at the end.
        let expanded_pre_readout = exclude_final_mz(&expanded.gates);
        let stab_group = StabilizerGroup::from_circuit(&expanded_pre_readout, expanded.num_qubits);

        dem_mapping::build_dem_configured(
            &result.generators,
            &detectors,
            &observables,
            Some(&stab_group),
            &self.config,
        )
    }

    #[must_use]
    pub fn build_dem_string(&self) -> String {
        dem_mapping::format_dem(&self.build())
    }

    #[must_use]
    pub fn summary(&self) -> EegSummary {
        let gates: Vec<pecos_core::Gate> = self
            .tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates);
        let result = circuit::analyze_expanded(&expanded.gates, &self.noise);
        let (detectors, observables) = build_detectors(self.tc, &expanded);

        let expanded_pre = exclude_final_mz(&expanded.gates);
        let stab_group = StabilizerGroup::from_circuit(&expanded_pre, expanded.num_qubits);
        let entries = dem_mapping::build_dem_configured(
            &result.generators,
            &detectors,
            &observables,
            Some(&stab_group),
            &self.config,
        );

        let h_count = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == crate::eeg::EegType::H)
            .count();
        let s_count = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == crate::eeg::EegType::S)
            .count();

        EegSummary {
            num_original_gates: gates.len(),
            num_expanded_gates: expanded.gates.len(),
            num_expanded_qubits: expanded.num_qubits,
            num_h_generators: h_count,
            num_s_generators: s_count,
            num_detectors: detectors.len(),
            num_observables: observables.len(),
            num_dem_events: entries.len(),
            generator_fidelity: result.generator_fidelity(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EegSummary {
    pub num_original_gates: usize,
    pub num_expanded_gates: usize,
    pub num_expanded_qubits: usize,
    pub num_h_generators: usize,
    pub num_s_generators: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    pub num_dem_events: usize,
    /// Generator fidelity: ε_gen = Σ h_P² + Σ |s_P|. DEM error scales as ε_gen^{1.5}.
    pub generator_fidelity: f64,
}

/// Strip all trailing MZ gates from the expanded circuit.
///
/// The expanded circuit ends with deferred MZ(aux) gates. Stripping them
/// gives the pre-readout expanded state for stabilizer group computation.
fn exclude_final_mz(gates: &[pecos_core::Gate]) -> Vec<pecos_core::Gate> {
    let last_non_mz = gates
        .iter()
        .rposition(|g| g.gate_type != pecos_core::gate_type::GateType::MZ);
    match last_non_mz {
        Some(idx) => gates[..=idx].to_vec(),
        None => Vec::new(),
    }
}

/// Build detectors for the expanded circuit from TickCircuit annotations.
///
/// Each detector is defined by measurement records (negative indices from
/// the end of the measurement sequence). In the expanded circuit, each
/// measurement record k maps to a Z-measurement on auxiliary qubit
/// `expanded.measurement_qubit[k]`.
///
/// The detector stabilizer in the expanded circuit is:
///   Z_{aux_r1} * Z_{aux_r2} * ...
/// where aux_ri = expanded.measurement_qubit[abs_index(ri)]
fn build_detectors(
    tc: &TickCircuit,
    expanded: &expand::ExpandedCircuit,
) -> (Vec<Detector>, Vec<Observable>) {
    let mut detectors = Vec::new();
    let mut observables = Vec::new();
    let num_meas = expanded.measurement_qubit.len();

    for annotation in tc.annotations() {
        match &annotation.kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => {
                // measurement_nodes are gate indices in the ORIGINAL circuit.
                // We need to map these to measurement record indices, then
                // to auxiliary qubits in the expanded circuit.
                //
                // The gate indices correspond to MZ gates. Each MZ gate can
                // measure multiple qubits. We need to find which measurement
                // record each gate index maps to.
                //
                // Strategy: the k-th measurement record corresponds to the
                // k-th qubit measured across all MZ gates in order. Each
                // measurement_node is a gate index — we find which measurement
                // records that gate produced.
                let bitmask =
                    measurement_nodes_to_aux_bitmask(measurement_nodes, tc, expanded, num_meas);
                detectors.push(Detector {
                    id: detectors.len(),
                    stabilizer: bitmask,
                });
            }
            AnnotationKind::Observable {
                measurement_nodes, ..
            } => {
                let bitmask =
                    measurement_nodes_to_aux_bitmask(measurement_nodes, tc, expanded, num_meas);
                observables.push(Observable {
                    id: observables.len(),
                    pauli: bitmask,
                });
            }
            AnnotationKind::TrackedPauli => {}
        }
    }

    (detectors, observables)
}

/// Map measurement record indices to a Z bitmask on auxiliary qubits.
///
/// Each measurement_node is a measurement record index (counting all
/// MZ qubits in circuit order). Maps directly to an auxiliary qubit
/// in the expanded circuit via `expanded.measurement_qubit[record]`.
fn measurement_nodes_to_aux_bitmask(
    measurement_nodes: &[usize],
    _tc: &TickCircuit,
    expanded: &expand::ExpandedCircuit,
    num_meas: usize,
) -> Bm {
    let mut bitmask = Bm::default();

    for &record_idx in measurement_nodes {
        if record_idx < num_meas {
            let aux_qubit = expanded.measurement_qubit[record_idx];
            bitmask.z_bits.xor_bit(aux_qubit);
        }
    }

    bitmask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_no_noise() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().mz(&[0]);
        let entries = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::coherent_only(0.0))
            .build();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_summary_coherent() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);
        let summary = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::coherent_only(0.1))
            .summary();
        assert!(summary.num_h_generators > 0);
        assert_eq!(summary.num_s_generators, 0);
    }

    #[test]
    fn test_builder_matches_manual_pipeline() {
        // Same circuit through builder and manual pipeline should give same DEM
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);

        let noise = NoiseModel::coherent_only(0.05);

        // Builder path
        let builder_entries = EegDemBuilder::from_tick_circuit(&tc)
            .noise(noise.clone())
            .build();

        // Manual path
        let gates: Vec<pecos_core::Gate> = tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates);
        let result = circuit::analyze_expanded(&expanded.gates, &noise);

        let expanded_pre = exclude_final_mz(&expanded.gates);
        let stab_group = StabilizerGroup::from_circuit(&expanded_pre, expanded.num_qubits);

        let (detectors, observables) = build_detectors(&tc, &expanded);
        let manual_entries = dem_mapping::build_dem_with_stabilizers(
            &result.generators,
            &detectors,
            &observables,
            Some(&stab_group),
        );

        // Same number of entries
        assert_eq!(
            builder_entries.len(),
            manual_entries.len(),
            "Builder and manual should produce same number of DEM entries"
        );

        // Same probabilities (order may differ, so sort)
        let mut bp: Vec<f64> = builder_entries.iter().map(|e| e.probability).collect();
        let mut mp: Vec<f64> = manual_entries.iter().map(|e| e.probability).collect();
        bp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (b, m) in bp.iter().zip(mp.iter()) {
            assert!(
                (b - m).abs() < 1e-15,
                "Probability mismatch: builder={b}, manual={m}"
            );
        }
    }

    #[test]
    fn test_no_annotations_empty_dem() {
        // Without detector/observable annotations, builder should produce empty DEM
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);

        let entries = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::depolarizing(0.01))
            .build();

        assert!(
            entries.is_empty(),
            "No annotations → no detectors → no DEM entries"
        );
    }

    #[test]
    fn test_with_detector_annotations() {
        // Build a circuit with detector annotations
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1, 2]);

        // Round 1: syndrome extraction
        tc.tick().cx(&[(0, 2)]);
        tc.tick().cx(&[(1, 2)]);
        let m1 = tc.tick().mz(&[2]);

        // Round 2: syndrome extraction
        tc.tick().pz(&[2]);
        tc.tick().cx(&[(0, 2)]);
        tc.tick().cx(&[(1, 2)]);
        let m2 = tc.tick().mz(&[2]);

        // Detector: compare m1 and m2
        tc.detector(&[m1[0], m2[0]]);

        // Final readout
        tc.tick().mz(&[0, 1]);

        let entries = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::depolarizing(0.01))
            .build();

        assert!(
            !entries.is_empty(),
            "Circuit with detector annotation should produce DEM entries"
        );
        for e in &entries {
            assert!(e.probability > 0.0);
            assert!(e.probability < 0.5);
        }
    }

    #[test]
    fn test_summary_counts() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1, 2]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().cx(&[(1, 2)]);
        tc.tick().mz(&[0, 1, 2]);

        let summary = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::depolarizing(0.01).with_idle_rz(0.05))
            .summary();

        assert!(
            summary.num_h_generators > 0,
            "Should have H generators from idle RZ"
        );
        assert!(
            summary.num_s_generators > 0,
            "Should have S generators from depolarizing"
        );
        assert_eq!(summary.num_expanded_qubits, 6, "3 original + 3 aux");
    }
}
