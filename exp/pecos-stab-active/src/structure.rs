// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Signed coordinate structure shared by simulation and width profiling.

use pecos_core::{Angle64, BitmaskStorage, PauliBitmaskVec, QubitId};
use pecos_simulators::{CliffordGateable, SparseStabY};
use pecos_stab_tn::stab_mps::coordinate_tableau::{
    self, CoordinateDecomposition, CoordinateGate, MeasurementCase,
};

use crate::PauliKindForDecomp;

#[derive(Clone, Debug)]
pub(crate) struct ActiveStructure {
    tableau: SparseStabY,
    active: Vec<usize>,
    peak_width: usize,
}

#[must_use]
pub(crate) struct Rotation<'a> {
    structure: &'a mut ActiveStructure,
    pauli: &'a [(usize, PauliKindForDecomp)],
    theta: Angle64,
    kind: RotationKind,
}

enum RotationKind {
    Identity,
    Clifford(usize),
    NonClifford {
        parts: CoordinateDecomposition,
        promote: bool,
    },
}

impl Rotation<'_> {
    pub(crate) fn apply_rotation(self) -> Option<RotationWork> {
        self.apply_rotation_observed(|_, _| {})
    }

    pub(crate) fn plan_rotation(self) -> (Option<RotationWork>, Option<PauliBitmaskVec>) {
        let mut correction = None;
        let work = self.apply_rotation_observed(|structure, pivot| {
            correction = Some(structure.row_bits(true, pivot));
        });
        (work, correction)
    }

    fn apply_rotation_observed(
        self,
        mut promoted: impl FnMut(&ActiveStructure, usize),
    ) -> Option<RotationWork> {
        let Rotation {
            structure,
            pauli,
            theta,
            kind,
        } = self;
        match kind {
            RotationKind::Identity => None,
            RotationKind::Clifford(turns) => {
                rotate_tableau(&mut structure.tableau, pauli, turns);
                None
            }
            RotationKind::NonClifford { mut parts, promote } => {
                if promote {
                    let pivot = coordinate_tableau::promote(
                        &mut structure.tableau,
                        &mut structure.active,
                        pauli,
                        false,
                    );
                    // localize_dormant leaves this stabilizer unchanged; read it
                    // after promote without splitting the shared primitive.
                    promoted(structure, pivot);
                    structure.peak_width = structure.peak_width.max(structure.width());
                    parts = coordinate_tableau::decompose(
                        &structure.tableau,
                        &structure.active,
                        pauli,
                        false,
                    );
                }
                Some(RotationWork {
                    theta,
                    parts,
                    double: promote,
                })
            }
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.structure.width()
    }

    pub(crate) fn needs_promotion(&self) -> bool {
        matches!(self.kind, RotationKind::NonClifford { promote: true, .. })
    }

    #[cfg(test)]
    pub(crate) fn is_identity(&self) -> bool {
        matches!(
            self.kind,
            RotationKind::Identity | RotationKind::Clifford(0)
        )
    }
}

#[must_use]
pub(crate) struct RotationWork {
    pub(crate) theta: Angle64,
    pub(crate) parts: CoordinateDecomposition,
    pub(crate) double: bool,
}

#[must_use]
pub(crate) struct Measurement<'a> {
    structure: &'a mut ActiveStructure,
    pauli: &'a [(usize, PauliKindForDecomp)],
    negative: bool,
    data: MeasurementData,
}

pub(crate) struct MeasurementData {
    parts: CoordinateDecomposition,
    case: MeasurementCase,
}

impl MeasurementData {
    pub(crate) fn new(parts: CoordinateDecomposition) -> Self {
        let case = parts.measurement_case();
        Self { parts, case }
    }

    pub(crate) fn parts(&self) -> &CoordinateDecomposition {
        &self.parts
    }

    pub(crate) fn case(&self) -> MeasurementCase {
        self.case
    }
}

impl Measurement<'_> {
    pub(crate) fn apply_measurement(self, outcome: bool) -> Option<ActiveMeasurement> {
        self.apply_measurement_observed(outcome, false, |_, _, _| {})
    }

    pub(crate) fn plan_measurement(self) -> MeasurementPlanWork {
        let mut row = None;
        let mut eta = false;
        let active = self.apply_measurement_observed(false, true, |structure, pivot, negative| {
            row = Some(structure.row_bits(false, pivot));
            eta = negative;
        });
        MeasurementPlanWork { active, row, eta }
    }

    fn apply_measurement_observed(
        self,
        outcome: bool,
        reference: bool,
        mut basis_ready: impl FnMut(&ActiveStructure, usize, bool),
    ) -> Option<ActiveMeasurement> {
        let Measurement {
            structure,
            pauli,
            negative,
            data,
        } = self;
        match data.case {
            MeasurementCase::Random => {
                coordinate_tableau::measure_random(
                    &mut structure.tableau,
                    &structure.active,
                    pauli,
                    negative,
                    outcome,
                );
                // measure_random uses the first dormant flip as its pivot.
                basis_ready(structure, data.parts.dormant_flips[0], false);
                None
            }
            MeasurementCase::Deterministic => None,
            MeasurementCase::Active => {
                let basis = coordinate_tableau::measurement_basis(
                    &mut structure.tableau,
                    &structure.active,
                    pauli,
                    negative,
                );
                basis_ready(structure, structure.active[basis.pivot_bit], basis.negative);
                let value = if reference {
                    false
                } else {
                    outcome ^ basis.negative
                };
                coordinate_tableau::demote(
                    &mut structure.tableau,
                    &mut structure.active,
                    basis.pivot_bit,
                    value,
                );
                Some(ActiveMeasurement {
                    gates: basis.gates,
                    projection: Projection {
                        pivot_bit: basis.pivot_bit,
                        value,
                    },
                })
            }
        }
    }

    pub(crate) fn data(&self) -> &MeasurementData {
        &self.data
    }
}

pub(crate) struct MeasurementPlanWork {
    pub(crate) active: Option<ActiveMeasurement>,
    pub(crate) row: Option<PauliBitmaskVec>,
    pub(crate) eta: bool,
}

#[must_use]
pub(crate) struct ActiveMeasurement {
    gates: Vec<CoordinateGate>,
    projection: Projection,
}

#[derive(Clone, Copy)]
#[must_use]
pub(crate) struct Projection {
    pivot_bit: usize,
    value: bool,
}

impl ActiveMeasurement {
    pub(crate) fn gates(&self) -> &[CoordinateGate] {
        &self.gates
    }

    pub(crate) fn projection(&self) -> &Projection {
        &self.projection
    }
}

impl Projection {
    pub(crate) fn pivot_bit(&self) -> usize {
        self.pivot_bit
    }

    pub(crate) fn value(&self) -> bool {
        self.value
    }
}

impl ActiveStructure {
    // Reuse SparseStabY's row accessors and PauliBitmaskVec's word-level
    // symplectic commutation. Row scalar phases do not affect conjugation.
    pub(crate) fn row_bits(&self, stabilizer: bool, index: usize) -> PauliBitmaskVec {
        let rows = if stabilizer {
            self.tableau.stabs()
        } else {
            self.tableau.destabs()
        };
        let mut bits = PauliBitmaskVec::identity();
        for q in &rows.row_x[index] {
            bits.x_bits.set_bit(q);
        }
        for q in &rows.row_z[index] {
            bits.z_bits.set_bit(q);
        }
        bits
    }

    #[cfg(test)]
    pub(crate) fn tableau(&self) -> &SparseStabY {
        &self.tableau
    }

    #[cfg(test)]
    pub(crate) fn active(&self) -> &[usize] {
        &self.active
    }

    pub(crate) fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            tableau: SparseStabY::with_seed(num_qubits, seed).with_destab_sign_tracking(),
            active: Vec::new(),
            peak_width: 0,
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.active.len()
    }

    pub(crate) fn peak_width(&self) -> usize {
        self.peak_width
    }

    pub(crate) fn num_qubits(&self) -> usize {
        self.tableau.num_qubits()
    }

    pub(crate) fn reset(&mut self) {
        self.tableau.reset();
        self.active.clear();
        self.peak_width = 0;
    }

    pub(crate) fn sz(&mut self, qubits: &[QubitId]) {
        self.tableau.sz(qubits);
    }

    pub(crate) fn h(&mut self, qubits: &[QubitId]) {
        self.tableau.h(qubits);
    }

    pub(crate) fn cx(&mut self, pairs: &[(QubitId, QubitId)]) {
        self.tableau.cx(pairs);
    }

    // Plan without mutation so the amplitude owner can reject a promotion.
    pub(crate) fn rotation<'a>(
        &'a mut self,
        theta: Angle64,
        pauli: &'a [(usize, PauliKindForDecomp)],
        negative: bool,
    ) -> Rotation<'a> {
        for (i, &(q, _)) in pauli.iter().enumerate() {
            assert!(
                q < self.num_qubits() && !pauli[..i].iter().any(|&(r, _)| r == q),
                "Pauli factors must name distinct valid qubits"
            );
        }
        let theta = if negative { -theta } else { theta };
        let kind = if pauli.is_empty() {
            RotationKind::Identity
        } else if let Some(turns) = clifford_turns(theta) {
            RotationKind::Clifford(turns)
        } else {
            let parts = coordinate_tableau::decompose(&self.tableau, &self.active, pauli, false);
            let promote = !parts.dormant_flips.is_empty();
            RotationKind::NonClifford { parts, promote }
        };
        Rotation {
            structure: self,
            pauli,
            theta,
            kind,
        }
    }

    pub(crate) fn measurement<'a>(
        &'a mut self,
        pauli: &'a [(usize, PauliKindForDecomp)],
        negative: bool,
    ) -> Measurement<'a> {
        let parts = coordinate_tableau::decompose(&self.tableau, &self.active, pauli, negative);
        Measurement {
            structure: self,
            pauli,
            negative,
            data: MeasurementData::new(parts),
        }
    }
}

pub(crate) fn clifford_turns(theta: Angle64) -> Option<usize> {
    [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ]
    .iter()
    .position(|&angle| theta == angle)
}

pub(crate) fn rotate_tableau(
    tableau: &mut SparseStabY,
    pauli: &[(usize, PauliKindForDecomp)],
    turns: usize,
) {
    let target = QubitId(pauli[pauli.len() - 1].0);
    for &(q, kind) in pauli {
        if kind == PauliKindForDecomp::Y {
            tableau.szdg(&[QubitId(q)]);
        }
        if kind != PauliKindForDecomp::Z {
            tableau.h(&[QubitId(q)]);
        }
    }
    for &(q, _) in &pauli[..pauli.len() - 1] {
        tableau.cx(&[(QubitId(q), target)]);
    }
    for _ in 0..turns {
        tableau.sz(&[target]);
    }
    for &(q, _) in pauli[..pauli.len() - 1].iter().rev() {
        tableau.cx(&[(QubitId(q), target)]);
    }
    for &(q, kind) in pauli.iter().rev() {
        if kind != PauliKindForDecomp::Z {
            tableau.h(&[QubitId(q)]);
        }
        if kind == PauliKindForDecomp::Y {
            tableau.sz(&[QubitId(q)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_measurement_finishes_before_amplitude_work() {
        let pauli = [(0, PauliKindForDecomp::X)];
        for negative in [false, true] {
            for outcome in [false, true] {
                let mut structure = ActiveStructure::with_seed(1, 0);
                let _ = structure
                    .rotation(Angle64::from_radians(0.37), &pauli, false)
                    .apply_rotation();
                let work = structure
                    .measurement(&pauli, negative)
                    .apply_measurement(outcome)
                    .unwrap();
                // The result describes amplitude work; the structure is already complete.
                assert_eq!(structure.width(), 0);
                assert_eq!(structure.peak_width(), 1);
                assert!(!work.gates().is_empty());
                let measurement = structure.measurement(&pauli, negative);
                assert_eq!(measurement.data().case(), MeasurementCase::Deterministic);
                assert_eq!(measurement.data().parts().phase.re < 0.0, outcome);
            }
        }
    }
}
