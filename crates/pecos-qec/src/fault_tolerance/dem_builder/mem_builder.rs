// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Measurement Noise Model (MNM) builder implementation.
//!
//! This module builds a MNM from a fault influence map. Unlike the DEM builder
//! which maps to detector effects, the MNM maps directly to measurement effects
//! for fast approximate sampling.
//!
//! # Usage
//!
//! ```
//! use pecos_qec::fault_tolerance::DagFaultAnalyzer;
//! use pecos_qec::fault_tolerance::dem_builder::MemBuilder;
//! use pecos_quantum::DagCircuit;
//! use rand::SeedableRng;
//! use rand::rngs::SmallRng;
//!
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);
//! dag.cx(&[(0, 2)]);
//! dag.cx(&[(1, 2)]);
//! dag.mz(&[2]);
//!
//! let analyzer = DagFaultAnalyzer::new(&dag);
//! let influence_map = analyzer.build_influence_map();
//!
//! let mnm = MemBuilder::new(&influence_map)
//!     .with_noise(0.01, 0.01, 0.01, 0.01)
//!     .build();
//!
//! // Sample measurement outcomes
//! let mut rng = SmallRng::seed_from_u64(42);
//! let outcomes = mnm.sample(&mut rng);
//! ```

use super::types::{MeasurementMechanism, MeasurementNoiseModel, NoiseConfig};
use crate::fault_tolerance::propagator::{DagFaultInfluenceMap, Pauli};
use pecos_core::gate_type::GateType;
use smallvec::SmallVec;

/// Builder for Measurement Noise Models (MNMs).
///
/// Constructs a MNM from a fault influence map. The MNM aggregates fault locations
/// by their measurement effects (which measurements flip), enabling fast approximate
/// sampling.
///
/// # Comparison with DEM
///
/// | Aspect | DEM | MNM |
/// |--------|-----|-----|
/// | Maps to | Detectors | Measurements |
/// | Use case | Decoding | Sampling |
/// | Aggregates by | Detector signature | Measurement signature |
/// | Output | Stim-compatible DEM | Raw measurement outcomes |
pub struct MemBuilder<'a> {
    /// Reference to the fault influence map.
    influence_map: &'a DagFaultInfluenceMap,
    /// Noise configuration.
    noise: NoiseConfig,
    /// Measurement order from the original circuit (e.g., `TickCircuit`).
    /// This is a list of qubits in the order they were measured.
    /// `measurement_order[tc_idx] = qubit` means the tc_idx-th measurement
    /// in the `TickCircuit` was on this qubit.
    measurement_order: Option<Vec<usize>>,
}

impl<'a> MemBuilder<'a> {
    /// Creates a new MNM builder from a fault influence map.
    #[must_use]
    pub fn new(influence_map: &'a DagFaultInfluenceMap) -> Self {
        Self {
            influence_map,
            noise: NoiseConfig::default(),
            measurement_order: None,
        }
    }

    /// Sets the noise configuration.
    #[must_use]
    pub fn with_noise(mut self, p1: f64, p2: f64, p_meas: f64, p_init: f64) -> Self {
        self.noise = NoiseConfig::new(p1, p2, p_meas, p_init);
        self
    }

    /// Sets the measurement order from the original circuit (e.g., `TickCircuit`).
    ///
    /// This is needed when detector definitions use `TickCircuit` measurement indices
    /// but the influence map uses a different ordering based on DAG topology.
    ///
    /// # Arguments
    ///
    /// * `order` - List of qubit indices in measurement execution order.
    ///   `order[tc_idx] = qubit` means the tc_idx-th measurement in the `TickCircuit`
    ///   was on this qubit.
    #[must_use]
    pub fn with_measurement_order(mut self, order: Vec<usize>) -> Self {
        self.measurement_order = Some(order);
        self
    }

    /// Computes the mapping from influence map measurement indices to `TickCircuit` indices.
    ///
    /// Returns a vector where `result[im_idx] = tc_idx`, mapping each influence map
    /// measurement to its corresponding `TickCircuit` measurement.
    fn compute_im_to_tc_mapping(&self, tc_order: &[usize]) -> Vec<usize> {
        let im_measurements = &self.influence_map.measurements;
        let num_measurements = im_measurements.len();

        // Build map: qubit -> list of TC indices where that qubit is measured
        let mut tc_indices_by_qubit: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (tc_idx, &qubit) in tc_order.iter().enumerate() {
            tc_indices_by_qubit.entry(qubit).or_default().push(tc_idx);
        }

        // Build map: qubit -> list of IM indices where that qubit is measured
        let mut im_indices_by_qubit: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (im_idx, &(_node, qubit, _basis)) in im_measurements.iter().enumerate() {
            im_indices_by_qubit.entry(qubit).or_default().push(im_idx);
        }

        // Build the mapping: for each qubit, match IM indices to TC indices in order
        let mut im_to_tc = vec![0; num_measurements];
        for (qubit, im_indices) in &im_indices_by_qubit {
            if let Some(tc_indices) = tc_indices_by_qubit.get(qubit) {
                // Match in order - the i-th IM measurement of this qubit maps to
                // the i-th TC measurement of this qubit
                for (i, &im_idx) in im_indices.iter().enumerate() {
                    if i < tc_indices.len() {
                        im_to_tc[im_idx] = tc_indices[i];
                    }
                }
            }
        }

        im_to_tc
    }

    /// Builds the Measurement Noise Model.
    ///
    /// This aggregates all fault locations by their measurement effects.
    /// Locations that produce the same measurement signature have their
    /// probabilities combined using the independent error formula.
    #[must_use]
    pub fn build(&self) -> MeasurementNoiseModel {
        let num_measurements = self.influence_map.measurements.len();
        let mut mem = MeasurementNoiseModel::new(num_measurements);

        // Compute im_to_tc mapping if measurement order is provided
        if let Some(ref tc_order) = self.measurement_order {
            let im_to_tc = self.compute_im_to_tc_mapping(tc_order);
            mem.set_measurement_order(im_to_tc);
        }

        let locations = &self.influence_map.locations;

        // Group CX locations by node for two-qubit gate processing
        let mut cx_groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();

        for (loc_idx, loc) in locations.iter().enumerate() {
            match loc.gate_type {
                GateType::PZ | GateType::QAlloc => {
                    // Prep errors: only "after" locations
                    if self.noise.p_init > 0.0 && !loc.before {
                        self.process_prep_fault(loc_idx, &mut mem);
                    }
                }
                GateType::MZ | GateType::MeasureFree => {
                    // Measurement errors: only "before" locations
                    if self.noise.p_meas > 0.0 && loc.before {
                        self.process_meas_fault(loc_idx, &mut mem);
                    }
                }
                GateType::CX
                | GateType::CZ
                | GateType::CY
                | GateType::SWAP
                | GateType::RXX
                | GateType::RYY
                | GateType::RZZ => {
                    // Two-qubit gate errors: only "after" locations
                    if !loc.before {
                        cx_groups.entry(loc.node).or_default().push(loc_idx);
                    }
                }
                GateType::H
                | GateType::SZ
                | GateType::SZdg
                | GateType::SX
                | GateType::SXdg
                | GateType::SY
                | GateType::SYdg
                | GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::T
                | GateType::Tdg
                | GateType::RX
                | GateType::RY
                | GateType::RZ
                | GateType::U
                | GateType::R1XY => {
                    // Single-qubit gate errors: only "after" locations
                    if self.noise.p1 > 0.0 && !loc.before {
                        self.process_single_qubit_fault(loc_idx, &mut mem);
                    }
                }
                _ => {}
            }
        }

        // Process two-qubit gates
        if self.noise.p2 > 0.0 {
            for (_, loc_indices) in cx_groups {
                if loc_indices.len() == 2 {
                    self.process_two_qubit_fault(loc_indices[0], loc_indices[1], &mut mem);
                }
            }
        }

        mem
    }

    /// Processes a prep/initialization fault location.
    fn process_prep_fault(&self, loc_idx: usize, mem: &mut MeasurementNoiseModel) {
        // For Z-basis prep, X error matters
        let mechanism = self.compute_mechanism(loc_idx, Pauli::X);
        if !mechanism.is_empty() {
            mem.add_mechanism(mechanism, self.noise.p_init);
        }
    }

    /// Processes a measurement fault location.
    fn process_meas_fault(&self, loc_idx: usize, mem: &mut MeasurementNoiseModel) {
        // Measurement error is a bit flip (X error)
        let mechanism = self.compute_mechanism(loc_idx, Pauli::X);
        if !mechanism.is_empty() {
            mem.add_mechanism(mechanism, self.noise.p_meas);
        }
    }

    /// Processes a single-qubit gate fault location.
    fn process_single_qubit_fault(&self, loc_idx: usize, mem: &mut MeasurementNoiseModel) {
        // Depolarizing: each of X, Y, Z with probability p1/3
        let prob = self.noise.p1 / 3.0;

        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let mechanism = self.compute_mechanism(loc_idx, pauli);
            if !mechanism.is_empty() {
                mem.add_mechanism(mechanism, prob);
            }
        }
    }

    /// Processes a two-qubit gate fault (CX or CZ).
    fn process_two_qubit_fault(&self, loc1: usize, loc2: usize, mem: &mut MeasurementNoiseModel) {
        // Two-qubit depolarizing: 15 non-identity Pauli combinations with p2/15 each
        let prob = self.noise.p2 / 15.0;

        let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

        // Cache single-qubit effects for each Pauli on each qubit
        let mut effects1: [Option<MeasurementMechanism>; 4] = [None, None, None, None];
        let mut effects2: [Option<MeasurementMechanism>; 4] = [None, None, None, None];

        for &p in &[Pauli::X, Pauli::Y, Pauli::Z] {
            effects1[p.as_u8() as usize] = Some(self.compute_mechanism(loc1, p));
            effects2[p.as_u8() as usize] = Some(self.compute_mechanism(loc2, p));
        }

        // Process all 15 non-trivial Pauli combinations
        for &p1 in &paulis {
            for &p2 in &paulis {
                if p1 == Pauli::I && p2 == Pauli::I {
                    continue; // Skip II
                }

                let mechanism = if p1 == Pauli::I {
                    // IX, IY, IZ
                    effects2[p2.as_u8() as usize].clone().unwrap_or_default()
                } else if p2 == Pauli::I {
                    // XI, YI, ZI
                    effects1[p1.as_u8() as usize].clone().unwrap_or_default()
                } else {
                    // Correlated: XOR the measurement effects
                    let e1 = effects1[p1.as_u8() as usize].as_ref();
                    let e2 = effects2[p2.as_u8() as usize].as_ref();
                    xor_measurement_mechanisms(e1, e2)
                };

                if !mechanism.is_empty() {
                    mem.add_mechanism(mechanism, prob);
                }
            }
        }
    }

    /// Computes the measurement mechanism for a fault at the given location and Pauli type.
    fn compute_mechanism(&self, loc_idx: usize, pauli: Pauli) -> MeasurementMechanism {
        // Get the measurement indices that this fault flips
        let measurements = self
            .influence_map
            .get_detector_indices(loc_idx, pauli.as_u8());

        let mut meas_vec: SmallVec<[u32; 4]> = measurements.iter().copied().collect();
        meas_vec.sort_unstable();

        MeasurementMechanism::from_sorted(meas_vec)
    }
}

/// XORs two measurement mechanisms (symmetric difference).
fn xor_measurement_mechanisms(
    a: Option<&MeasurementMechanism>,
    b: Option<&MeasurementMechanism>,
) -> MeasurementMechanism {
    match (a, b) {
        (Some(m1), Some(m2)) => {
            let mut result: SmallVec<[u32; 4]> = SmallVec::new();
            let mut i = 0;
            let mut j = 0;

            while i < m1.measurements.len() && j < m2.measurements.len() {
                match m1.measurements[i].cmp(&m2.measurements[j]) {
                    std::cmp::Ordering::Less => {
                        result.push(m1.measurements[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        result.push(m2.measurements[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        // Same element in both - XOR cancels
                        i += 1;
                        j += 1;
                    }
                }
            }

            result.extend_from_slice(&m1.measurements[i..]);
            result.extend_from_slice(&m2.measurements[j..]);

            MeasurementMechanism::from_sorted(result)
        }
        (Some(m), None) | (None, Some(m)) => m.clone(),
        (None, None) => MeasurementMechanism::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::propagator::DagFaultAnalyzer;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_mem_builder_simple() {
        // Simple circuit: prep, cx, measure
        let mut dag = DagCircuit::new();
        dag.pz(&[2]); // Prep ancilla
        dag.cx(&[(0, 2)]); // CNOT data -> ancilla
        dag.cx(&[(1, 2)]); // CNOT data -> ancilla
        dag.mz(&[2]); // Measure ancilla

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let mem = MemBuilder::new(&influence_map)
            .with_noise(0.01, 0.01, 0.01, 0.01)
            .build();

        // Should have some mechanisms
        assert!(mem.num_mechanisms() > 0);
        assert_eq!(mem.num_measurements, 1);
    }

    #[test]
    fn test_mem_builder_aggregation() {
        // Circuit where multiple locations produce the same measurement effect
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let mem = MemBuilder::new(&influence_map)
            .with_noise(0.01, 0.01, 0.01, 0.01)
            .build();

        // Count how many mechanisms flip measurement 0
        let single_meas_mechanisms: Vec<_> = mem
            .iter()
            .filter(|(m, _)| m.measurements.as_slice() == [0])
            .collect();

        // Should have aggregated multiple sources into one mechanism
        // (prep X error + measurement X error both flip measurement 0)
        assert!(
            single_meas_mechanisms.len() == 1,
            "Expected aggregation of mechanisms with same effect"
        );

        // Combined probability should be > individual probability
        let combined_prob = single_meas_mechanisms[0].1;
        assert!(
            *combined_prob > 0.01,
            "Combined probability should be greater than single source"
        );
    }

    #[test]
    fn test_mem_sampling() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let mem = MemBuilder::new(&influence_map)
            .with_noise(0.1, 0.1, 0.1, 0.1) // High error rate for testing
            .build();

        let mut rng = SmallRng::seed_from_u64(42);

        // Sample many shots and count flips
        let num_shots = 10000;
        let mut flip_count = 0;

        for _ in 0..num_shots {
            let outcomes = mem.sample(&mut rng);
            if outcomes.first().copied().unwrap_or(false) {
                flip_count += 1;
            }
        }

        // Should have some flips (not 0) and some non-flips (not all)
        assert!(flip_count > 0, "Should have some measurement flips");
        assert!(
            flip_count < num_shots,
            "Should not have all measurements flipped"
        );
    }

    #[test]
    fn test_xor_measurement_mechanisms() {
        let m1 = MeasurementMechanism::from_unsorted([0, 1, 2]);
        let m2 = MeasurementMechanism::from_unsorted([1, 2, 3]);

        let result = xor_measurement_mechanisms(Some(&m1), Some(&m2));

        // {0, 1, 2} XOR {1, 2, 3} = {0, 3}
        assert_eq!(result.measurements.as_slice(), &[0, 3]);
    }
}
