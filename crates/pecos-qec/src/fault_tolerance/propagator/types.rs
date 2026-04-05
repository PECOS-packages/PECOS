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

//! Core types for fault tolerance propagation and analysis.

use super::SpacetimeLocation;
use smallvec::SmallVec;
use std::collections::BTreeMap;

// ============================================================================
// Entity IDs (Type-Safe Indices)
// ============================================================================

/// A node (gate) in the DAG circuit.
///
/// This is a type-safe wrapper around a raw index, following ECS principles
/// where entities are just IDs and components hold the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Creates a new `NodeId` from a raw index.
    #[inline]
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the raw index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Creates from usize (for compatibility).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // node index fits in u32
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for NodeId {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<NodeId> for usize {
    #[inline]
    fn from(id: NodeId) -> Self {
        id.index()
    }
}

/// A fault location in the circuit.
///
/// Identifies a specific spacetime point where a fault can occur
/// (before or after a gate on specific qubits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LocationId(pub u32);

impl LocationId {
    #[inline]
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // location index fits in u32
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for LocationId {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<LocationId> for usize {
    #[inline]
    fn from(id: LocationId) -> Self {
        id.index()
    }
}

/// A detector (measurement-based syndrome bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct DetectorIdx(pub u32);

impl DetectorIdx {
    #[inline]
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // detector index fits in u32
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for DetectorIdx {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<DetectorIdx> for usize {
    #[inline]
    fn from(id: DetectorIdx) -> Self {
        id.index()
    }
}

/// A logical observable index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LogicalIdx(pub u32);

impl LogicalIdx {
    #[inline]
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // logical index fits in u32
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for LogicalIdx {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<LogicalIdx> for usize {
    #[inline]
    fn from(id: LogicalIdx) -> Self {
        id.index()
    }
}

/// Pauli type for faults (I=0, X=1, Y=2, Z=3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum Pauli {
    #[default]
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

impl Pauli {
    /// Creates from raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::X,
            2 => Self::Y,
            3 => Self::Z,
            _ => Self::I,
        }
    }

    /// Returns the raw u8 value.
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns true if this is a non-identity Pauli.
    #[inline]
    #[must_use]
    pub const fn is_nontrivial(self) -> bool {
        self.as_u8() != 0
    }
}

// ============================================================================
// Influence Map Types
// ============================================================================

/// Unique identifier for a measurement in the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeasurementId {
    /// Which tick the measurement occurs in.
    pub tick: usize,
    /// Which qubit is measured.
    pub qubit: usize,
    /// Measurement basis: 0 = Z, 1 = X.
    pub basis: u8,
}

/// Unique identifier for a detector (syndrome bit).
///
/// A detector is typically defined as the XOR of two measurements,
/// detecting changes in syndrome between rounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DetectorId {
    /// The measurements that make up this detector.
    /// For a simple detector: [`m_i`]
    /// For a comparison detector: [`m_i`, m_{i-1}]
    /// Using `SmallVec` to avoid heap allocation for common 1-2 measurement cases.
    pub measurements: SmallVec<[MeasurementId; 2]>,
    /// Optional name/label for the detector.
    pub name: Option<String>,
}

impl DetectorId {
    /// Creates a single-measurement detector.
    #[inline]
    #[must_use]
    pub fn single(measurement: MeasurementId) -> Self {
        Self {
            measurements: smallvec::smallvec![measurement],
            name: None,
        }
    }

    /// Creates a comparison detector (XOR of two measurements).
    #[inline]
    #[must_use]
    pub fn comparison(m1: MeasurementId, m2: MeasurementId) -> Self {
        Self {
            measurements: smallvec::smallvec![m1, m2],
            name: None,
        }
    }

    /// Adds a name to the detector.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Unique identifier for a logical observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalId {
    /// Index of the logical qubit.
    pub logical_qubit: usize,
    /// Which observable: 0 = Z, 1 = X.
    pub observable: u8,
}

/// What a single fault location influences.
///
/// Uses fixed-size arrays indexed by Pauli type (0=I, 1=X, 2=Y, 3=Z) for fast access.
#[derive(Debug, Clone, Default)]
pub struct FaultInfluence {
    /// Which detectors this fault flips, indexed by Pauli type (1=X, 2=Y, 3=Z).
    /// Index 0 is unused (identity fault has no effect).
    pub detector_flips: [Vec<DetectorId>; 4],

    /// Which logical observables this fault flips, indexed by Pauli type.
    pub logical_flips: [Vec<LogicalId>; 4],

    /// Which raw measurements this fault flips, indexed by Pauli type.
    pub measurement_flips: [Vec<MeasurementId>; 4],

    /// Per-qubit detector flips for multi-qubit locations.
    /// Key: (`qubit_index_in_location`, `pauli_type`), Value: detector IDs flipped by that qubit
    pub per_qubit_detector_flips: BTreeMap<(usize, u8), Vec<DetectorId>>,
}

impl FaultInfluence {
    /// Returns true if this fault has no effect.
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        self.detector_flips.iter().all(std::vec::Vec::is_empty)
            && self.logical_flips.iter().all(std::vec::Vec::is_empty)
            && self.measurement_flips.iter().all(std::vec::Vec::is_empty)
    }

    /// Returns all detectors flipped by a specific Pauli type.
    #[inline]
    #[must_use]
    pub fn detectors_for_pauli(&self, pauli: u8) -> &[DetectorId] {
        self.detector_flips
            .get(pauli as usize)
            .map_or(&[], |v| v.as_slice())
    }

    /// Returns all logicals flipped by a specific Pauli type.
    #[inline]
    #[must_use]
    pub fn logicals_for_pauli(&self, pauli: u8) -> &[LogicalId] {
        self.logical_flips
            .get(pauli as usize)
            .map_or(&[], |v| v.as_slice())
    }
}

/// Pre-computed map from fault locations to their influences.
///
/// This is the main output of backward propagation - a lookup table
/// that tells you what each fault location affects.
#[derive(Debug, Clone)]
pub struct FaultInfluenceMap {
    /// For each spacetime location, what it influences.
    pub influences: BTreeMap<SpacetimeLocation, FaultInfluence>,

    /// All detectors in the circuit.
    pub detectors: Vec<DetectorId>,

    /// All logical observables being tracked.
    pub logicals: Vec<LogicalId>,

    /// All measurements in the circuit.
    pub measurements: Vec<MeasurementId>,

    /// Reverse map: for each detector, which fault locations flip it.
    pub detector_to_faults: BTreeMap<DetectorId, Vec<(SpacetimeLocation, u8)>>,

    /// Reverse map: for each logical, which fault locations flip it.
    pub logical_to_faults: BTreeMap<LogicalId, Vec<(SpacetimeLocation, u8)>>,
}

impl FaultInfluenceMap {
    /// Creates an empty influence map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            influences: BTreeMap::new(),
            detectors: Vec::new(),
            logicals: Vec::new(),
            measurements: Vec::new(),
            detector_to_faults: BTreeMap::new(),
            logical_to_faults: BTreeMap::new(),
        }
    }

    /// Returns the influence of a fault at the given location.
    #[must_use]
    pub fn get_influence(&self, location: &SpacetimeLocation) -> Option<&FaultInfluence> {
        self.influences.get(location)
    }

    /// Quickly classifies a single-qubit fault based on pre-computed influences.
    ///
    /// Returns (`has_syndrome`, `has_logical_error`) for the given Pauli type.
    /// For multi-qubit locations, use `classify_multi_qubit_fault` instead.
    #[must_use]
    pub fn classify_fault(&self, location: &SpacetimeLocation, pauli: u8) -> (bool, bool) {
        if let Some(influence) = self.influences.get(location) {
            let has_syndrome = !influence.detectors_for_pauli(pauli).is_empty();
            let has_logical = !influence.logicals_for_pauli(pauli).is_empty();
            (has_syndrome, has_logical)
        } else {
            (false, false)
        }
    }

    /// Classifies a multi-qubit fault where the same Pauli is applied to all qubits.
    ///
    /// For multi-qubit locations (e.g., CX gate), applying the same Pauli to both
    /// qubits can have cancellation effects. This method properly computes the
    /// combined effect by `XORing` the per-qubit influences.
    ///
    /// For Y faults, we decompose Y = XZ and combine the X and Z contributions,
    /// since Y anticommutes with both X and Z components of the observable.
    ///
    /// Returns (`has_syndrome`, `has_logical_error`).
    #[must_use]
    pub fn classify_multi_qubit_fault(
        &self,
        location: &SpacetimeLocation,
        pauli: u8,
    ) -> (bool, bool) {
        if let Some(influence) = self.influences.get(location) {
            // Count detector flips per detector
            let mut detector_flip_counts: BTreeMap<&DetectorId, usize> = BTreeMap::new();

            // Collect flips from each qubit in the location
            for qubit_idx in 0..location.qubits.len() {
                if pauli == 2 {
                    // Y = XZ: a Y fault flips a detector if EITHER the X component
                    // OR the Z component would flip it. Count contributions from both.
                    // X component flips detectors sensitive to Z
                    if let Some(detectors) = influence.per_qubit_detector_flips.get(&(qubit_idx, 1))
                    {
                        for detector in detectors {
                            *detector_flip_counts.entry(detector).or_insert(0) += 1;
                        }
                    }
                    // Z component flips detectors sensitive to X
                    if let Some(detectors) = influence.per_qubit_detector_flips.get(&(qubit_idx, 3))
                    {
                        for detector in detectors {
                            *detector_flip_counts.entry(detector).or_insert(0) += 1;
                        }
                    }
                } else {
                    // X or Z fault: straightforward
                    if let Some(detectors) =
                        influence.per_qubit_detector_flips.get(&(qubit_idx, pauli))
                    {
                        for detector in detectors {
                            *detector_flip_counts.entry(detector).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Syndrome = odd number of flips for any detector
            let has_syndrome = detector_flip_counts.values().any(|&count| count % 2 == 1);

            // For logicals, use the same approach
            // (simplified: just check if any qubit flips logical, proper handling TBD)
            let has_logical = !influence.logicals_for_pauli(pauli).is_empty();

            (has_syndrome, has_logical)
        } else {
            (false, false)
        }
    }

    /// Returns all fault locations that flip a specific detector.
    #[must_use]
    pub fn faults_for_detector(&self, detector: &DetectorId) -> &[(SpacetimeLocation, u8)] {
        self.detector_to_faults
            .get(detector)
            .map_or(&[], |v| v.as_slice())
    }

    /// Returns the number of fault locations tracked.
    #[must_use]
    pub fn num_fault_locations(&self) -> usize {
        self.influences.len()
    }
}

impl Default for FaultInfluenceMap {
    fn default() -> Self {
        Self::new()
    }
}
