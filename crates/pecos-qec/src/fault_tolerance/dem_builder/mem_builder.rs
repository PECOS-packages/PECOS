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
//! This module builds a measurement-level noise model from a fault influence
//! map. Unlike a DEM, which maps faults to detector and observable effects,
//! the MNM maps faults directly to raw measurement flips for fast approximate
//! sampling.

use super::types::{MeasurementMechanism, MeasurementNoiseModel, NoiseConfig};
use crate::fault_tolerance::propagator::{DagFaultInfluenceMap, Pauli};
use pecos_core::gate_type::GateType;
use smallvec::SmallVec;

/// Builder for Measurement Noise Models (MNMs).
pub struct MemBuilder<'a> {
    /// Reference to the fault influence map.
    influence_map: &'a DagFaultInfluenceMap,
    /// Noise configuration.
    noise: NoiseConfig,
    /// Optional measurement order from the original circuit.
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

    /// Sets the scalar noise configuration.
    #[must_use]
    pub fn with_noise(mut self, p1: f64, p2: f64, p_meas: f64, p_prep: f64) -> Self {
        self.noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        self
    }

    /// Sets the full noise configuration.
    #[must_use]
    pub fn with_noise_config(mut self, noise: NoiseConfig) -> Self {
        self.noise = noise;
        self
    }

    /// Sets the measurement order from the original circuit.
    #[must_use]
    pub fn with_measurement_order(mut self, order: Vec<usize>) -> Self {
        self.measurement_order = Some(order);
        self
    }

    /// Builds the Measurement Noise Model.
    #[must_use]
    pub fn build(&self) -> MeasurementNoiseModel {
        let num_measurements = self.influence_map.measurements.len();
        let mut mem = MeasurementNoiseModel::new(num_measurements);

        if let Some(ref tc_order) = self.measurement_order {
            mem.set_measurement_order(self.compute_im_to_tc_mapping(tc_order));
        }

        let locations = &self.influence_map.locations;
        let mut two_qubit_groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();

        for (loc_idx, loc) in locations.iter().enumerate() {
            match loc.gate_type {
                GateType::PZ | GateType::QAlloc if self.noise.p_prep > 0.0 && !loc.before => {
                    self.process_single_pauli_fault(loc_idx, Pauli::X, self.noise.p_prep, &mut mem);
                }
                GateType::MZ | GateType::MeasureFree if self.noise.p_meas > 0.0 && loc.before => {
                    self.process_single_pauli_fault(loc_idx, Pauli::X, self.noise.p_meas, &mut mem);
                }
                GateType::CX
                | GateType::CZ
                | GateType::CY
                | GateType::SZZ
                | GateType::SZZdg
                | GateType::SXX
                | GateType::SXXdg
                | GateType::SYY
                | GateType::SYYdg
                | GateType::SWAP
                | GateType::RXX
                | GateType::RYY
                | GateType::RZZ
                    if !loc.before =>
                {
                    two_qubit_groups.entry(loc.node).or_default().push(loc_idx);
                }
                GateType::H
                | GateType::F
                | GateType::Fdg
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
                | GateType::R1XY
                    if self.noise.p1 > 0.0 && !loc.before =>
                {
                    self.process_single_qubit_fault(loc_idx, &mut mem);
                }
                GateType::Idle if !loc.before => {
                    if self.noise.uses_dedicated_idle_noise() {
                        #[allow(clippy::cast_precision_loss)]
                        let duration = loc.idle_duration.max(1) as f64;
                        let probs = self.noise.idle_pauli_probs(duration);
                        if probs.px > 0.0 {
                            self.process_single_pauli_fault(loc_idx, Pauli::X, probs.px, &mut mem);
                        }
                        if probs.py > 0.0 {
                            self.process_single_pauli_fault(loc_idx, Pauli::Y, probs.py, &mut mem);
                        }
                        if probs.pz > 0.0 {
                            self.process_single_pauli_fault(loc_idx, Pauli::Z, probs.pz, &mut mem);
                        }
                    } else if self.noise.p1 > 0.0 {
                        self.process_single_qubit_fault(loc_idx, &mut mem);
                    }
                }
                _ => {}
            }
        }

        if self.noise.p2 > 0.0 {
            for loc_indices in two_qubit_groups.values() {
                for pair in loc_indices.chunks(2) {
                    if pair.len() == 2 {
                        self.process_two_qubit_fault(pair[0], pair[1], &mut mem);
                    }
                }
            }
        }

        mem
    }

    fn compute_im_to_tc_mapping(&self, tc_order: &[usize]) -> Vec<usize> {
        let im_measurements = &self.influence_map.measurements;
        let mut tc_indices_by_qubit: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (tc_idx, &qubit) in tc_order.iter().enumerate() {
            tc_indices_by_qubit.entry(qubit).or_default().push(tc_idx);
        }

        let mut im_indices_by_qubit: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (im_idx, &(_node, qubit, _basis)) in im_measurements.iter().enumerate() {
            im_indices_by_qubit.entry(qubit).or_default().push(im_idx);
        }

        let mut im_to_tc = vec![0; im_measurements.len()];
        for (qubit, im_indices) in &im_indices_by_qubit {
            if let Some(tc_indices) = tc_indices_by_qubit.get(qubit) {
                for (i, &im_idx) in im_indices.iter().enumerate() {
                    if let Some(&tc_idx) = tc_indices.get(i) {
                        im_to_tc[im_idx] = tc_idx;
                    }
                }
            }
        }
        im_to_tc
    }

    fn process_single_pauli_fault(
        &self,
        loc_idx: usize,
        pauli: Pauli,
        prob: f64,
        mem: &mut MeasurementNoiseModel,
    ) {
        let mechanism = self.compute_mechanism(loc_idx, pauli);
        if !mechanism.is_empty() {
            mem.add_mechanism(mechanism, prob);
        }
    }

    fn process_single_qubit_fault(&self, loc_idx: usize, mem: &mut MeasurementNoiseModel) {
        let prob = self.noise.p1 / 3.0;
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            self.process_single_pauli_fault(loc_idx, pauli, prob, mem);
        }
    }

    fn process_two_qubit_fault(&self, loc1: usize, loc2: usize, mem: &mut MeasurementNoiseModel) {
        let prob = self.noise.p2 / 15.0;
        let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

        let mut effects1: [Option<MeasurementMechanism>; 4] = [None, None, None, None];
        let mut effects2: [Option<MeasurementMechanism>; 4] = [None, None, None, None];

        for &p in &[Pauli::X, Pauli::Y, Pauli::Z] {
            effects1[p.as_u8() as usize] = Some(self.compute_mechanism(loc1, p));
            effects2[p.as_u8() as usize] = Some(self.compute_mechanism(loc2, p));
        }

        for &p1 in &paulis {
            for &p2 in &paulis {
                if p1 == Pauli::I && p2 == Pauli::I {
                    continue;
                }

                let mechanism = if p1 == Pauli::I {
                    effects2[p2.as_u8() as usize].clone().unwrap_or_default()
                } else if p2 == Pauli::I {
                    effects1[p1.as_u8() as usize].clone().unwrap_or_default()
                } else {
                    xor_measurement_mechanisms(
                        effects1[p1.as_u8() as usize].as_ref(),
                        effects2[p2.as_u8() as usize].as_ref(),
                    )
                };

                if !mechanism.is_empty() {
                    mem.add_mechanism(mechanism, prob);
                }
            }
        }
    }

    fn compute_mechanism(&self, loc_idx: usize, pauli: Pauli) -> MeasurementMechanism {
        let measurements = self
            .influence_map
            .get_detector_indices(loc_idx, pauli.as_u8());

        let mut meas_vec: SmallVec<[u32; 4]> = measurements.iter().copied().collect();
        meas_vec.sort_unstable();

        MeasurementMechanism::from_sorted(meas_vec)
    }
}

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
