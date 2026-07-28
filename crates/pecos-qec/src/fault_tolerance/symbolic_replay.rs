// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! Shared unitary-Clifford dispatch for symbolic stabilizer replay.
//!
//! Two places replay a circuit against a [`SymbolicSparseStab`]: the influence
//! builder's forward symbolic simulation, and the DEM builder's measurement
//! crosstalk replay. They previously carried separate hand-written copies of
//! the same gate table. Issue #325 was exactly a disagreement between two
//! layers about what a gate does, so the table lives here once and both callers
//! dispatch through it.
//!
//! Only unitary Cliffords belong here. Measurement, preparation, and meta gates
//! carry caller-specific bookkeeping (measurement indices, payload nodes), so
//! each caller keeps its own arms for those and handles [`Dispatch::Unhandled`].

use pecos_quantum::GateType;
use pecos_simulators::SymbolicSparseStab;

/// Whether the shared table recognised and applied a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dispatch {
    /// The gate is a unitary Clifford and has been applied to the simulator.
    Applied,
    /// The gate is not in the unitary-Clifford table. The caller must handle it.
    Unhandled,
}

/// A gate's qubit list does not fit its arity.
///
/// Callers report this differently -- the crosstalk replay turns it into a
/// configuration error naming the payload node, the influence builder treats it
/// as a malformed-circuit panic -- so the table reports what is wrong and lets
/// them phrase it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArityError {
    /// The gate needed at least `required` qubits but carried `actual`.
    TooFew { required: usize, actual: usize },
    /// A two-qubit gate needed an even qubit count but carried `actual`.
    OddPairing { actual: usize },
}

/// Apply `gate_type` to `sim` if it is a unitary Clifford.
///
/// `qubits` is the gate's full qubit list: single-qubit gates apply to every
/// entry, two-qubit gates to every consecutive pair. A `DagCircuit` node may
/// genuinely carry several gate instances -- `DagCircuit::gate_count` counts
/// them individually -- so applying only the first would silently drop gates.
/// Only `DagCircuit::from(&TickCircuit)` and the `DagCircuit` builder helpers
/// split batches; `add_gate_auto_wire` stores a batched `Gate` unchanged.
///
/// # Errors
///
/// Returns [`ArityError`] when a gate carries too few qubits, or an odd number
/// for a two-qubit gate.
pub(crate) fn apply_unitary_clifford(
    sim: &mut SymbolicSparseStab,
    gate_type: GateType,
    qubits: &[usize],
) -> Result<Dispatch, ArityError> {
    let require = |n: usize| -> Result<(), ArityError> {
        if qubits.len() < n {
            return Err(ArityError::TooFew {
                required: n,
                actual: qubits.len(),
            });
        }
        Ok(())
    };
    let pairs = || -> Result<Vec<(usize, usize)>, ArityError> {
        require(2)?;
        if !qubits.len().is_multiple_of(2) {
            return Err(ArityError::OddPairing {
                actual: qubits.len(),
            });
        }
        Ok(qubits
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect())
    };

    match gate_type {
        GateType::H => {
            require(1)?;
            sim.h(qubits);
        }
        // F is the SX-then-SZ face rotation; keeping it as the decomposition
        // rather than a native call is what makes this table agree with the
        // Pauli propagator by construction (issue #325).
        GateType::F => {
            require(1)?;
            sim.sx(qubits);
            sim.sz(qubits);
        }
        GateType::Fdg => {
            require(1)?;
            sim.szdg(qubits);
            sim.sxdg(qubits);
        }
        GateType::SX => {
            require(1)?;
            sim.sx(qubits);
        }
        GateType::SXdg => {
            require(1)?;
            sim.sxdg(qubits);
        }
        GateType::SY => {
            require(1)?;
            sim.sy(qubits);
        }
        GateType::SYdg => {
            require(1)?;
            sim.sydg(qubits);
        }
        GateType::SZ => {
            require(1)?;
            sim.sz(qubits);
        }
        GateType::SZdg => {
            require(1)?;
            sim.szdg(qubits);
        }
        GateType::X => {
            require(1)?;
            sim.x(qubits);
        }
        GateType::Y => {
            require(1)?;
            sim.y(qubits);
        }
        GateType::Z => {
            require(1)?;
            sim.z(qubits);
        }
        GateType::CX => {
            sim.cx(&pairs()?);
        }
        GateType::CY => {
            sim.cy(&pairs()?);
        }
        GateType::CZ => {
            sim.cz(&pairs()?);
        }
        GateType::SXX => {
            sim.sxx(&pairs()?);
        }
        GateType::SXXdg => {
            sim.sxxdg(&pairs()?);
        }
        GateType::SYY => {
            sim.syy(&pairs()?);
        }
        GateType::SYYdg => {
            sim.syydg(&pairs()?);
        }
        GateType::SZZ => {
            sim.szz(&pairs()?);
        }
        GateType::SZZdg => {
            sim.szzdg(&pairs()?);
        }
        GateType::SWAP => {
            sim.swap(&pairs()?);
        }
        _ => return Ok(Dispatch::Unhandled),
    }

    Ok(Dispatch::Applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Apply a gate through the shared table, asserting it was recognised.
    fn apply(sim: &mut SymbolicSparseStab, gate_type: GateType, qubits: &[usize]) {
        assert_eq!(
            apply_unitary_clifford(sim, gate_type, qubits),
            Ok(Dispatch::Applied),
            "{gate_type:?} should be handled by the unitary-Clifford table"
        );
    }

    /// Rotations bringing Z, X, and Y onto the Z axis for readout.
    const READOUTS: [&[GateType]; 3] = [&[], &[GateType::H], &[GateType::SZdg, GateType::H]];

    /// |0>, |+>, and |+i> -- enough distinct stabilizer states that agreeing on
    /// all three pins a single-qubit Clifford up to global phase.
    const PREPS: [&[GateType]; 3] = [&[], &[GateType::H], &[GateType::H, GateType::SZ]];

    /// Signed fingerprint of the state `gates` produces from `prep`: per readout
    /// axis, whether the measurement is deterministic and, when it is, its
    /// value. `SymbolicMeasurementResult::flip` accumulates unitary gate phases,
    /// so this separates gates differing only by sign -- which is exactly what
    /// the unsigned `PauliProp` layer cannot do.
    fn fingerprint(prep: &[GateType], gates: &[GateType]) -> Vec<(bool, bool)> {
        READOUTS
            .iter()
            .map(|readout| {
                let mut sim = SymbolicSparseStab::new(1);
                sim.pz(0);
                for &gate_type in prep.iter().chain(gates).chain(*readout) {
                    apply(&mut sim, gate_type, &[0]);
                }
                let result = sim.mz(&[0]).pop().expect("one measurement per readout");
                // `flip` is only meaningful for a deterministic outcome.
                (
                    result.is_deterministic,
                    result.is_deterministic && result.flip,
                )
            })
            .collect()
    }

    /// Issue #325 in its general form: a native gate must act exactly like the
    /// decomposition the rest of the stack assumes for it. Checked with signs,
    /// which the DEM layer cannot see.
    #[test]
    fn native_gates_match_their_decompositions_including_sign() {
        let cases: [(GateType, &[GateType]); 4] = [
            (GateType::F, &[GateType::SX, GateType::SZ]),
            (GateType::Fdg, &[GateType::SZdg, GateType::SXdg]),
            (GateType::SY, &[GateType::SX, GateType::SZ, GateType::SXdg]),
            (
                GateType::SYdg,
                &[GateType::SX, GateType::SZdg, GateType::SXdg],
            ),
        ];

        for prep in PREPS {
            for (native, decomposed) in cases {
                assert_eq!(
                    fingerprint(prep, &[native]),
                    fingerprint(prep, decomposed),
                    "{native:?} must act as {decomposed:?} (prep {prep:?})"
                );
            }
        }
    }

    /// The teeth the DEM-level test in `dem_builder` provably cannot have.
    /// `SY` and `SYdg` share an unsigned action, so `PauliProp` and any
    /// phase-free DEM are blind to swapping them; this layer is not.
    #[test]
    fn adjoint_pairs_are_distinguishable_here_unlike_in_the_dem() {
        for (gate, adjoint) in [
            (GateType::SY, GateType::SYdg),
            (GateType::SX, GateType::SXdg),
            (GateType::SZ, GateType::SZdg),
            (GateType::F, GateType::Fdg),
        ] {
            assert!(
                PREPS
                    .iter()
                    .any(|prep| fingerprint(prep, &[gate]) != fingerprint(prep, &[adjoint])),
                "{gate:?} and {adjoint:?} must be separable by some stabilizer state"
            );
        }
    }

    /// Two-qubit gates act on every consecutive pair, not just the first. The
    /// two tables this module replaced disagreed on exactly this point.
    #[test]
    fn two_qubit_gates_apply_to_every_pair() {
        fn bell_pairs_then_measure(cx_batches: &[&[usize]]) -> String {
            let mut sim = SymbolicSparseStab::new(4);
            for q in 0..4 {
                sim.pz(q);
            }
            apply(&mut sim, GateType::H, &[0]);
            apply(&mut sim, GateType::H, &[2]);
            for batch in cx_batches {
                apply(&mut sim, GateType::CX, batch);
            }
            sim.mz(&[0, 1, 2, 3]);
            sim.measurement_history().format_all()
        }

        assert_eq!(
            bell_pairs_then_measure(&[&[0, 1, 2, 3]]),
            bell_pairs_then_measure(&[&[0, 1], &[2, 3]]),
            "a batched CX must entangle both pairs, not just the first"
        );
    }

    /// Non-unitary gates fall through so each caller keeps its own bookkeeping.
    #[test]
    fn non_unitary_gates_are_left_to_the_caller() {
        let mut sim = SymbolicSparseStab::new(1);
        for gate_type in [GateType::MZ, GateType::PZ, GateType::QAlloc, GateType::I] {
            assert_eq!(
                apply_unitary_clifford(&mut sim, gate_type, &[0]),
                Ok(Dispatch::Unhandled),
                "{gate_type:?} must be left to the caller"
            );
        }
    }

    #[test]
    fn malformed_qubit_lists_are_reported_not_panicked_on() {
        let mut sim = SymbolicSparseStab::new(2);
        assert_eq!(
            apply_unitary_clifford(&mut sim, GateType::H, &[]),
            Err(ArityError::TooFew {
                required: 1,
                actual: 0
            })
        );
        assert_eq!(
            apply_unitary_clifford(&mut sim, GateType::CX, &[0]),
            Err(ArityError::TooFew {
                required: 2,
                actual: 1
            })
        );
        assert_eq!(
            apply_unitary_clifford(&mut sim, GateType::CX, &[0, 1, 0]),
            Err(ArityError::OddPairing { actual: 3 })
        );
    }
}
