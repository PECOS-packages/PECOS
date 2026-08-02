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

pub use crate::expand::EegBuildError;

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

    /// # Errors
    ///
    /// Returns [`EegBuildError`] when the circuit contains a measurement type
    /// the MZ-only expansion cannot represent, or when an annotation references
    /// a measurement record the expansion did not produce. Both used to be
    /// silent -- the measurement vanished, or the reference was skipped.
    pub fn build(&self) -> Result<Vec<DemEntry>, EegBuildError> {
        let gates: Vec<pecos_core::Gate> = self
            .tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates)?;
        let result = circuit::analyze_expanded(&expanded.gates, &self.noise);
        let (detectors, observables) = build_detectors(self.tc, &expanded)?;

        // Compute stabilizer group from the EXPANDED circuit (pre-readout).
        // This includes auxiliary qubits, so beta function checks happen
        // directly in the expanded frame without lossy frame mapping.
        // Exclude the final deferred MZ(aux) gates at the end.
        let expanded_pre_readout = exclude_final_mz(&expanded.gates);
        let stab_group = StabilizerGroup::from_circuit(&expanded_pre_readout, expanded.num_qubits);

        Ok(dem_mapping::build_dem_configured(
            &result.generators,
            &detectors,
            &observables,
            Some(&stab_group),
            &self.config,
        ))
    }

    /// # Errors
    ///
    /// Same conditions as [`build`](Self::build).
    pub fn build_dem_string(&self) -> Result<String, EegBuildError> {
        Ok(dem_mapping::format_dem(&self.build()?))
    }

    /// # Errors
    ///
    /// Same conditions as [`build`](Self::build).
    pub fn summary(&self) -> Result<EegSummary, EegBuildError> {
        let gates: Vec<pecos_core::Gate> = self
            .tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates)?;
        let result = circuit::analyze_expanded(&expanded.gates, &self.noise);
        let (detectors, observables) = build_detectors(self.tc, &expanded)?;

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

        Ok(EegSummary {
            num_original_gates: gates.len(),
            num_expanded_gates: expanded.gates.len(),
            num_expanded_qubits: expanded.num_qubits,
            num_h_generators: h_count,
            num_s_generators: s_count,
            num_detectors: detectors.len(),
            num_observables: observables.len(),
            num_dem_events: entries.len(),
            generator_fidelity: result.generator_fidelity(),
        })
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
) -> Result<(Vec<Detector>, Vec<Observable>), EegBuildError> {
    let mut detectors = Vec::new();
    let mut observables = Vec::new();

    for annotation in tc.annotations() {
        match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            } => {
                let bitmask = measurement_ids_to_aux_bitmask(measurement_ids, expanded)?;
                detectors.push(Detector {
                    id: detectors.len(),
                    stabilizer: bitmask,
                });
            }
            AnnotationKind::Observable {
                measurement_ids, ..
            } => {
                let bitmask = measurement_ids_to_aux_bitmask(measurement_ids, expanded)?;
                observables.push(Observable {
                    id: observables.len(),
                    pauli: bitmask,
                });
            }
            AnnotationKind::TrackedPauli => {}
        }
    }

    Ok((detectors, observables))
}

/// Map measurement ids to a Z bitmask on auxiliary qubits.
///
/// Each id resolves through the expansion's own id-rank map, so external
/// (non-positional) ids land on the right auxiliary qubit. An unknown id is
/// an error, never a skip.
fn measurement_ids_to_aux_bitmask(
    measurement_ids: &[pecos_core::MeasId],
    expanded: &expand::ExpandedCircuit,
) -> Result<Bm, EegBuildError> {
    let mut bitmask = Bm::default();
    for &meas_id in measurement_ids {
        let aux_qubit = expanded.aux_qubit_for_id(meas_id)?;
        bitmask.z_bits.xor_bit(aux_qubit);
    }
    Ok(bitmask)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two measurements carrying the same supplied id would make id
    /// resolution last-wins -- a silently wrong DEM. `TickCircuit` permits the
    /// duplicate; the expansion must refuse it.
    #[test]
    fn duplicate_supplied_ids_are_refused() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        for qubit in [0usize, 1] {
            let mut gate = pecos_core::Gate::mz(&[qubit]);
            gate.meas_ids = smallvec::smallvec![pecos_core::MeasId::from_raw(5)];
            tc.tick().try_add_gate(gate).expect("gate is valid");
        }

        let err = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::depolarizing(0.01))
            .build()
            .expect_err("two measurements hold id 5");
        assert_eq!(
            err,
            EegBuildError::DuplicateMeasId {
                meas_id: pecos_core::MeasId::from_raw(5),
            }
        );
    }

    /// A supplied id colliding with a later minted one is the same ambiguity
    /// arriving through two different doors; the expansion refuses it too.
    #[test]
    fn a_supplied_id_colliding_with_a_minted_id_is_refused() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        let mut gate = pecos_core::Gate::mz(&[0usize]);
        gate.meas_ids = smallvec::smallvec![pecos_core::MeasId::from_raw(0)];
        tc.tick().try_add_gate(gate).expect("gate is valid");
        // `try_add_gate` does not advance the mint counter, so this mints 0.
        let minted = tc.tick().mz(&[1]);
        assert_eq!(minted[0].meas_id, pecos_core::MeasId::from_raw(0));

        let err = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::depolarizing(0.01))
            .build()
            .expect_err("supplied and minted ids collide on 0");
        assert!(matches!(err, EegBuildError::DuplicateMeasId { .. }));
    }

    /// Scrambled (non-positional) annotation ids must produce a byte-identical
    /// DEM to positional ids on the same physical circuit: eeg orders its
    /// auxiliaries by expansion rank, which is invariant under id relabeling.
    #[test]
    fn scrambled_annotation_ids_build_a_byte_identical_dem() {
        let build = |ids: [usize; 3]| {
            let mut dag = pecos_quantum::DagCircuit::new();
            dag.pz(&[0, 1, 2]);
            dag.cx(&[(0, 2)]);
            dag.cx(&[(1, 2)]);
            for (qubit, id) in [(2usize, ids[0]), (0, ids[1]), (1, ids[2])] {
                let mut gate = pecos_core::Gate::mz(&[qubit]);
                gate.meas_ids = smallvec::smallvec![pecos_core::MeasId::from_raw(id)];
                dag.try_add_gate_auto_wire(gate).expect("gate is valid");
            }
            let refs: Vec<_> = ids
                .iter()
                .map(|&id| {
                    dag.find_measurement(pecos_core::MeasId::from_raw(id))
                        .expect("the id was just supplied")
                })
                .collect();
            dag.detector(&refs).expect("refs are from this circuit");
            dag.observable(&[refs[1]])
                .expect("refs are from this circuit");
            let tc = TickCircuit::from(&dag);
            EegDemBuilder::from_tick_circuit(&tc)
                .noise(NoiseModel::depolarizing(0.01))
                .build_dem_string()
                .expect("the circuit is MZ-only and every id resolves")
        };
        let positional = build([0, 1, 2]);
        let scrambled = build([9, 4, 7]);
        assert!(
            positional.contains("error("),
            "the comparison is vacuous without error mechanisms:\n{positional}"
        );
        assert_eq!(positional, scrambled);
    }

    /// A `MeasureFree` used to pass through the expansion unchanged, silently
    /// deleting its measurement record. It now lowers to `MZ`.
    #[test]
    fn a_measure_free_circuit_expands_with_its_record_intact() {
        // Surface-code circuits read every ancilla with `mz_free`; refusing it
        // was a regression, not a guard. It lowers to `MZ` -- the record is
        // real, the free has no stabilizer effect.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().mz_free(&[0]);
        tc.tick().mz(&[1]);

        EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::coherent_only(0.01))
            .build()
            .expect("MeasureFree lowers to MZ in the expansion");

        let gates: Vec<pecos_core::Gate> = tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = crate::expand::expand_circuit(&gates).expect("supported");
        assert_eq!(
            expanded.measurement_qubit.len(),
            2,
            "both the MeasureFree and the MZ keep their measurement records"
        );
    }

    /// `MeasureLeaked` stays refused: it consumes no record, so the
    /// record-aligned expansion genuinely cannot represent it.
    #[test]
    fn a_measure_leaked_circuit_is_refused() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick()
            .try_add_gate(pecos_core::Gate::measure_leaked(&[0usize]))
            .expect("gate is valid");
        tc.tick().mz(&[1]);

        let err = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::coherent_only(0.01))
            .build()
            .expect_err("MeasureLeaked consumes no record");
        assert!(matches!(err, EegBuildError::UnsupportedMeasurement { .. }));
    }

    /// An annotation referencing an id the expansion never recorded used to
    /// be skipped, thinning the observable. It is now an error naming the id.
    ///
    /// Construction validation cannot catch this reference: it arrives
    /// pre-built on a converted circuit, which is exactly how a stale
    /// reference reaches eeg in practice.
    #[test]
    fn an_unknown_id_annotation_is_an_error() {
        let mut dag = pecos_quantum::DagCircuit::new();
        dag.pz(&[0]);
        dag.mz(&[0]);
        dag.add_annotation(pecos_quantum::PauliAnnotation {
            pauli: pecos_core::PauliString::zs(&[0usize]),
            kind: pecos_quantum::AnnotationKind::Observable {
                measurement_ids: vec![pecos_core::MeasId::from_raw(7)],
            },
            label: None,
        });
        let tc = TickCircuit::from(&dag);

        let err = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::coherent_only(0.01))
            .build()
            .expect_err("id 7 was never recorded by the expansion");
        assert_eq!(
            err,
            EegBuildError::UnresolvableAnnotationId {
                meas_id: pecos_core::MeasId::from_raw(7),
                num_measurements: 1,
            }
        );
    }

    #[test]
    fn test_empty_no_noise() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().mz(&[0]);
        let entries = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::coherent_only(0.0))
            .build()
            .expect("circuit is MZ-only with in-range records");
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
            .summary()
            .expect("circuit is MZ-only with in-range records");
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
            .build()
            .expect("circuit is MZ-only with in-range records");

        // Manual path
        let gates: Vec<pecos_core::Gate> = tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates).expect("MZ-only circuit");
        let result = circuit::analyze_expanded(&expanded.gates, &noise);

        let expanded_pre = exclude_final_mz(&expanded.gates);
        let stab_group = StabilizerGroup::from_circuit(&expanded_pre, expanded.num_qubits);

        let (detectors, observables) = build_detectors(&tc, &expanded).expect("records in range");
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
            .build()
            .expect("circuit is MZ-only with in-range records");

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
        tc.detector(&[m1[0], m2[0]])
            .expect("refs are from this circuit");

        // Final readout
        tc.tick().mz(&[0, 1]);

        let entries = EegDemBuilder::from_tick_circuit(&tc)
            .noise(NoiseModel::depolarizing(0.01))
            .build()
            .expect("circuit is MZ-only with in-range records");

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
            .summary()
            .expect("circuit is MZ-only with in-range records");

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
