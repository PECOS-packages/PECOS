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

//! Influence-based fault checking.
//!
//! This module provides [`InfluenceBasedChecker`] for O(1) fault classification
//! using pre-computed influence maps.

use super::SpacetimeLocation;
use super::types::{DetectorId, FaultInfluenceMap};

/// Efficient fault checker using pre-computed influence maps.
///
/// This provides O(1) fault classification instead of `O(circuit_depth)`
/// forward propagation.
pub struct InfluenceBasedChecker<'a> {
    influence_map: &'a FaultInfluenceMap,
}

impl<'a> InfluenceBasedChecker<'a> {
    /// Creates a new checker from a pre-computed influence map.
    #[must_use]
    pub fn new(influence_map: &'a FaultInfluenceMap) -> Self {
        Self { influence_map }
    }

    /// Classifies a fault at the given location with the given Pauli type.
    ///
    /// For single-qubit locations, returns whether any qubit causes syndrome or
    /// flips a tracked Pauli.
    /// For multi-qubit locations where the same Pauli is applied to all qubits,
    /// use `classify_uniform` which properly handles cancellation effects.
    ///
    /// Returns (`has_syndrome`, `flips_tracked_pauli`).
    #[must_use]
    pub fn classify(&self, location: &SpacetimeLocation, pauli: u8) -> (bool, bool) {
        self.influence_map.classify_fault(location, pauli)
    }

    /// Classifies a multi-qubit fault where the same Pauli is applied to all qubits.
    ///
    /// This properly handles cancellation: if the same Pauli on two different qubits
    /// both flip the same detector, they cancel out (XOR semantics).
    ///
    /// For Y faults (single or multi-qubit), we decompose Y = XZ and combine the
    /// X and Z contributions with XOR semantics.
    ///
    /// Returns (`has_syndrome`, `flips_tracked_pauli`).
    #[must_use]
    pub fn classify_uniform(&self, location: &SpacetimeLocation, pauli: u8) -> (bool, bool) {
        // Always use multi-qubit logic for Y faults (even single-qubit)
        // because Y = XZ needs to combine X and Z contributions
        if pauli == 2 || location.qubits.len() > 1 {
            self.influence_map
                .classify_multi_qubit_fault(location, pauli)
        } else {
            // Single-qubit X or Z: simple lookup
            self.influence_map.classify_fault(location, pauli)
        }
    }

    /// Returns all detectors flipped by the given fault.
    #[must_use]
    pub fn detectors_flipped(&self, location: &SpacetimeLocation, pauli: u8) -> Vec<&DetectorId> {
        self.influence_map
            .get_influence(location)
            .map_or(Vec::new(), |inf| {
                inf.detectors_for_pauli(pauli).iter().collect()
            })
    }

    /// Checks if a fault silently flips a tracked Pauli.
    #[must_use]
    pub fn is_silent_tracked_pauli_flip(&self, location: &SpacetimeLocation, pauli: u8) -> bool {
        let (has_syndrome, flips_tracked_pauli) = self.classify(location, pauli);
        !has_syndrome && flips_tracked_pauli
    }
}
