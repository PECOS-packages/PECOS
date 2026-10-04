// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Signed coordinate structure shared by simulation and width profiling.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{CliffordGateable, SparseStabY};
use pecos_stab_tn::stab_mps::coordinate_tableau::{
    self, CoordinateDecomposition, CoordinateGate, MeasurementCase,
};

use crate::PauliKindForDecomp;

#[derive(Clone, Debug)]
pub(crate) struct ActiveStructure {
    pub(crate) tableau: SparseStabY,
    pub(crate) active: Vec<usize>,
    peak_width: usize,
}

pub(crate) struct Rotation<'a> {
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

pub(crate) struct RotationWork {
    pub(crate) theta: Angle64,
    pub(crate) parts: CoordinateDecomposition,
    pub(crate) double: bool,
}

pub(crate) struct Measurement<'a> {
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
    pub(crate) fn data(&self) -> &MeasurementData {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn into_parts(self) -> CoordinateDecomposition {
        self.data.parts
    }
}

pub(crate) struct ActiveMeasurement {
    pub(crate) gates: Vec<CoordinateGate>,
    pub(crate) projection: Projection,
}

#[derive(Clone, Copy)]
pub(crate) struct Projection {
    pub(crate) pivot_bit: usize,
    pub(crate) value: bool,
}

impl ActiveStructure {
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
        &self,
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
        Rotation { pauli, theta, kind }
    }

    pub(crate) fn apply_rotation(&mut self, rotation: Rotation<'_>) -> Option<RotationWork> {
        let Rotation { pauli, theta, kind } = rotation;
        match kind {
            RotationKind::Identity => None,
            RotationKind::Clifford(turns) => {
                rotate_tableau(&mut self.tableau, pauli, turns);
                None
            }
            RotationKind::NonClifford { mut parts, promote } => {
                if promote {
                    coordinate_tableau::promote(&mut self.tableau, &mut self.active, pauli, false);
                    self.peak_width = self.peak_width.max(self.width());
                    parts =
                        coordinate_tableau::decompose(&self.tableau, &self.active, pauli, false);
                }
                Some(RotationWork {
                    theta,
                    parts,
                    double: promote,
                })
            }
        }
    }

    pub(crate) fn measurement<'a>(
        &self,
        pauli: &'a [(usize, PauliKindForDecomp)],
        negative: bool,
    ) -> Measurement<'a> {
        let parts = coordinate_tableau::decompose(&self.tableau, &self.active, pauli, negative);
        Measurement {
            pauli,
            negative,
            data: MeasurementData::new(parts),
        }
    }

    pub(crate) fn apply_measurement(
        &mut self,
        measurement: Measurement<'_>,
        outcome: bool,
    ) -> Option<ActiveMeasurement> {
        let Measurement {
            pauli,
            negative,
            data,
        } = measurement;
        match data.case {
            MeasurementCase::Random => {
                coordinate_tableau::measure_random(
                    &mut self.tableau,
                    &self.active,
                    pauli,
                    negative,
                    outcome,
                );
                None
            }
            MeasurementCase::Deterministic => None,
            MeasurementCase::Active => {
                let basis = coordinate_tableau::measurement_basis(
                    &mut self.tableau,
                    &self.active,
                    pauli,
                    negative,
                );
                let value = outcome ^ basis.negative;
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

    pub(crate) fn demote(&mut self, projection: Projection) {
        coordinate_tableau::demote(
            &mut self.tableau,
            &mut self.active,
            projection.pivot_bit,
            projection.value,
        );
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
