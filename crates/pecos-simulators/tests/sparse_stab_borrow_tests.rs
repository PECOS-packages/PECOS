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

use pecos_core::QubitId;
use pecos_simulators::{
    CliffordGateable, DenseStab, DenseStabColOnly, DenseStabRowOnly, SparseColOnly, SparseRowOnly,
    SparseStab, SparseStabHybrid, SparseStabUnsortedVecSet, SparseStabVecSet, SparseStabY,
    SparseStabYUnsortedVecSet, SparseStabYVecSet, StabilizerTableauSimulator,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

type Gate<S> = for<'a> fn(&'a mut S, &[(QubitId, QubitId)]) -> &'a mut S;
type BatchGate<S, T> = for<'a> fn(&'a mut S, &[T]) -> &'a mut S;
type NamedBatchGate<S, T> = (&'static str, BatchGate<S, T>);
type NamedGate<S> = (&'static str, Gate<S>);

fn gates<S: CliffordGateable>() -> [NamedGate<S>; 14] {
    [
        ("CX", S::cx),
        ("CY", S::cy),
        ("CZ", S::cz),
        ("SWAP", S::swap),
        ("SXX", S::sxx),
        ("SXXdg", S::sxxdg),
        ("SYY", S::syy),
        ("SYYdg", S::syydg),
        ("SZZ", S::szz),
        ("SZZdg", S::szzdg),
        ("iSWAP", S::iswap),
        ("iSWAPdg", S::iswapdg),
        ("G", S::g),
        ("Gdg", S::gdg),
    ]
}

fn check_repeated_qubits<S: CliffordGateable + StabilizerTableauSimulator + Clone>(mut sim: S) {
    sim.h(&[QubitId(0), QubitId(1), QubitId(2)])
        .sz(&[QubitId(1)]);
    let before = sim.clone();
    for (name, gate) in gates::<S>() {
        let error = catch_unwind(AssertUnwindSafe(|| {
            gate(
                &mut sim,
                &[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(2))],
            );
        }))
        .expect_err("repeated qubits must panic in every build profile");
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .expect("panic should explain the invalid qubit");
        assert!(message.contains(&name.to_uppercase()), "{name}: {message}");
        assert!(message.contains("distinct qubits"), "{name}: {message}");
        assert!(message.contains("repeated qubit 2"), "{name}: {message}");
        assert_eq!(
            sim.full_tableau(),
            before.full_tableau(),
            "{name} mutated before panicking"
        );
    }
}

// DenseStab prints the coefficient of X^x Z^z even when displaying XZ as Y.
// SparseStabY prints the coefficient of actual Pauli Y = iXZ. Convert only the
// reference representation, using XZ = -iY independently of either gate code.
fn reference_tableau(tableau: String, y_convention: bool) -> String {
    if !y_convention {
        return tableau;
    }
    let mut result = String::new();
    for line in tableau.lines() {
        let (phase, paulis) = if let Some(rest) = line.strip_prefix("+i") {
            (1, rest)
        } else if let Some(rest) = line.strip_prefix("-i") {
            (3, rest)
        } else if let Some(rest) = line.strip_prefix('+') {
            (0, rest)
        } else {
            (2, line.strip_prefix('-').expect("signed generator"))
        };
        let ys = paulis.chars().filter(|&ch| ch == 'Y').count();
        result.push_str(["+", "+i", "-", "-i"][(phase + 4 - ys % 4) % 4]);
        result.push_str(paulis);
        result.push('\n');
    }
    result
}

fn check_disjoint_gates<S: CliffordGateable + StabilizerTableauSimulator + Clone>(
    mut sim: S,
    tracked: bool,
    y_convention: bool,
) {
    // Prepare independently using only single-qubit gates, so a CX mutation cannot
    // contaminate both the tested gate and its starting state.
    sim.h(&[QubitId(0), QubitId(2)])
        .sz(&[QubitId(2)])
        .x(&[QubitId(1)]);
    let mut reference = DenseStab::with_seed(3, 42);
    reference
        .h(&[QubitId(0), QubitId(2)])
        .sz(&[QubitId(2)])
        .x(&[QubitId(1)]);
    for pair in [(QubitId(0), QubitId(2)), (QubitId(2), QubitId(0))] {
        for ((name, gate), (_, reference_gate)) in
            gates::<S>().into_iter().zip(gates::<DenseStab>())
        {
            let mut actual = sim.clone();
            let mut expected = reference.clone();
            gate(&mut actual, &[pair]);
            reference_gate(&mut expected, &[pair]);
            assert_eq!(
                actual.stab_tableau(),
                reference_tableau(expected.stab_tableau(), y_convention),
                "{name}, pair {pair:?}"
            );
            if tracked {
                assert_eq!(
                    actual.destab_tableau(),
                    reference_tableau(expected.destab_tableau(), y_convention),
                    "{name}, pair {pair:?}, tracked destabilizers"
                );
            }
        }
    }
}

#[test]
fn sparse_stab_repeated_qubits() {
    check_repeated_qubits(SparseStab::with_seed(3, 42));
    check_repeated_qubits(SparseStabVecSet::with_seed(3, 42));
    check_repeated_qubits(SparseStabUnsortedVecSet::with_seed(3, 42));
    check_repeated_qubits(SparseStabHybrid::with_seed(3, 42));
    check_repeated_qubits(pecos_simulators::Stabilizer::with_seed(3, 42));
}

#[test]
fn sparse_stab_batched_repeated_qubits() {
    use pecos_simulators::{BatchedOps, RawOps};

    let mut sim = SparseStab::with_seed(3, 42);
    sim.h(&[QubitId(0), QubitId(1), QubitId(2)])
        .sz(&[QubitId(1)]);
    let before = sim.clone();
    for gate in [
        SparseStab::cx_batched,
        SparseStab::cz_batched,
        SparseStab::cx_raw,
    ] {
        let error = catch_unwind(AssertUnwindSafe(|| {
            gate(&mut sim, &[0, 1, 2, 2]);
        }))
        .expect_err("batched and raw gates must reject repeated qubits");
        let message = error.downcast_ref::<String>().expect("descriptive panic");
        assert!(message.contains("repeated qubit 2"), "{message}");
        assert_eq!(sim.full_tableau(), before.full_tableau());
    }
}

#[test]
fn sparse_stab_y_repeated_qubits() {
    check_repeated_qubits(SparseStabY::with_seed(3, 42));
    check_repeated_qubits(SparseStabYVecSet::with_seed(3, 42));
    check_repeated_qubits(SparseStabYUnsortedVecSet::with_seed(3, 42));
}

#[test]
fn sparse_stab_disjoint_gates() {
    check_disjoint_gates(SparseStab::with_seed(3, 42), false, false);
    check_disjoint_gates(
        SparseStab::with_seed(3, 42).with_destab_sign_tracking(),
        true,
        false,
    );
    check_disjoint_gates(SparseStabVecSet::with_seed(3, 42), false, false);
    check_disjoint_gates(
        SparseStabVecSet::with_seed(3, 42).with_destab_sign_tracking(),
        true,
        false,
    );
    check_disjoint_gates(SparseStabUnsortedVecSet::with_seed(3, 42), false, false);
    check_disjoint_gates(
        SparseStabUnsortedVecSet::with_seed(3, 42).with_destab_sign_tracking(),
        true,
        false,
    );
}

#[test]
fn sparse_stab_y_disjoint_gates() {
    check_disjoint_gates(SparseStabY::with_seed(3, 42), false, true);
    check_disjoint_gates(
        SparseStabY::with_seed(3, 42).with_destab_sign_tracking(),
        true,
        true,
    );
    check_disjoint_gates(SparseStabYVecSet::with_seed(3, 42), false, true);
    check_disjoint_gates(
        SparseStabYVecSet::with_seed(3, 42).with_destab_sign_tracking(),
        true,
        true,
    );
    check_disjoint_gates(SparseStabYUnsortedVecSet::with_seed(3, 42), false, true);
    check_disjoint_gates(
        SparseStabYUnsortedVecSet::with_seed(3, 42).with_destab_sign_tracking(),
        true,
        true,
    );
}

#[test]
fn dense_batches_reject_before_mutation() {
    check_repeated_qubits(DenseStab::with_seed(3, 42));
    check_repeated_qubits(DenseStabColOnly::with_seed(3, 42));
    check_repeated_qubits(DenseStabRowOnly::with_seed(3, 42));
    check_repeated_qubits(SparseColOnly::new(3));
    check_repeated_qubits(SparseRowOnly::new(3));
}

fn check_bounds_hook<S: CliffordGateable>(mut sim: S) {
    let repeated = [QubitId(0); 64];
    sim.validate_qubits(&repeated);
    sim.identity(&repeated);
    sim.apply_global_phase(pecos_core::Angle64::QUARTER_TURN, &repeated);
    sim.mx(&[QubitId(0), QubitId(0)]);
    sim.my(&[QubitId(1), QubitId(1)]);
}

#[test]
fn bounds_hook_allows_repeated_single_qubit_operands() {
    check_bounds_hook(SparseStab::with_seed(3, 42));
    check_bounds_hook(SparseStabHybrid::with_seed(3, 42));
    check_bounds_hook(SparseStabY::with_seed(3, 42));
    check_bounds_hook(pecos_simulators::Stabilizer::with_seed(3, 42));
}

fn check_pair_bounds<S: CliffordGateable + StabilizerTableauSimulator + Clone>(mut sim: S) {
    sim.h(&[QubitId(0), QubitId(1)]).sz(&[QubitId(1)]);
    let before = sim.clone();
    for (name, gate) in gates::<S>() {
        let error = catch_unwind(AssertUnwindSafe(|| {
            gate(
                &mut sim,
                &[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))],
            );
        }))
        .expect_err("invalid later pair must reject the batch");
        let message = error.downcast_ref::<String>().expect("descriptive panic");
        assert!(message.contains("qubit 3"), "{name}: {message}");
        assert!(message.contains("3-qubit"), "{name}: {message}");
        assert_eq!(sim.full_tableau(), before.full_tableau(), "{name}");
    }
}

#[test]
fn sparse_pair_bounds_reject_entire_batch_with_actionable_message() {
    check_pair_bounds(SparseStab::with_seed(3, 42));
}

#[test]
fn sparse_stab_y_pair_bounds_reject_entire_batch() {
    check_pair_bounds(SparseStabY::with_seed(3, 42));
}

#[test]
fn sparse_stab_hybrid_pair_bounds_reject_entire_batch() {
    check_pair_bounds(SparseStabHybrid::with_seed(3, 42));
}

fn check_invalid_bounds_hook<S: CliffordGateable + StabilizerTableauSimulator + Clone>(mut sim: S) {
    sim.h(&[QubitId(0), QubitId(1)]).sz(&[QubitId(1)]);
    let before = sim.clone();
    let error = catch_unwind(AssertUnwindSafe(|| {
        // Identity uses the bounds hook without a later indexing operation that
        // could accidentally hide removal of that hook's check.
        sim.identity(&[QubitId(0), QubitId(3)]);
    }))
    .expect_err("bounds hook must reject an invalid later operand");
    let message = error.downcast_ref::<String>().expect("descriptive panic");
    assert!(message.contains("qubit 3"), "{message}");
    assert!(message.contains("3-qubit"), "{message}");
    assert_eq!(sim.full_tableau(), before.full_tableau());
}

#[test]
fn sparse_stab_y_bounds_hook_rejects_invalid_operand() {
    check_invalid_bounds_hook(SparseStabY::with_seed(3, 42));
}

#[test]
fn sparse_stab_hybrid_bounds_hook_rejects_invalid_operand() {
    check_invalid_bounds_hook(SparseStabHybrid::with_seed(3, 42));
}

#[test]
fn sparse_batched_bounds_reject_entire_batch() {
    use pecos_simulators::BatchedOps;
    let mut sim = SparseStab::with_seed(3, 42);
    sim.h(&[QubitId(0), QubitId(1)]).sz(&[QubitId(1)]);
    let before = sim.clone();
    let gates: [NamedBatchGate<SparseStab, usize>; 2] = [
        ("CX", SparseStab::cx_batched),
        ("CZ", SparseStab::cz_batched),
    ];
    for (name, gate) in gates {
        let error = catch_unwind(AssertUnwindSafe(|| {
            gate(&mut sim, &[0, 1, 2, 3]);
        }))
        .expect_err("invalid later pair must reject the batch");
        let message = error.downcast_ref::<String>().expect("descriptive panic");
        assert!(message.contains(name), "{message}");
        assert!(message.contains("qubit 3"), "{message}");
        assert!(message.contains("3-qubit"), "{message}");
        assert_eq!(sim.full_tableau(), before.full_tableau(), "{name}");
    }
}

#[test]
fn command_buffer_flush_rejects_bounds_before_mutation() {
    use pecos_simulators::CommandBuffer;
    for gate in ["X", "Z", "SZ", "H", "CX", "CZ"] {
        let mut sim = SparseStab::with_seed(3, 42);
        let before = sim.clone();
        let mut buffer = CommandBuffer::new();
        buffer.h(&[0]);
        match gate {
            "X" => buffer.x(&[5]),
            "Z" => buffer.z(&[5]),
            "SZ" => buffer.sz(&[5]),
            "H" => buffer.h(&[5]),
            "CX" => buffer.cx(&[(0, 5)]),
            "CZ" => buffer.cz(&[(0, 5)]),
            _ => panic!("unlisted buffer {gate}"),
        };
        let buffered_before = format!("{buffer:?}");
        let error = catch_unwind(AssertUnwindSafe(|| buffer.flush(&mut sim)))
            .expect_err("flush must reject all invalid buffers before the first phase");
        let message = error.downcast_ref::<String>().expect("descriptive panic");
        assert!(message.contains("qubit 5"), "{gate}: {message}");
        assert!(message.contains("3-qubit"), "{gate}: {message}");
        assert_eq!(sim.full_tableau(), before.full_tableau(), "{gate}");
        assert_eq!(
            format!("{buffer:?}"),
            buffered_before,
            "{gate} cleared a buffer"
        );
    }
}

#[test]
fn command_buffer_rejects_repeated_pairs_before_appending() {
    use pecos_simulators::CommandBuffer;
    let gates: [NamedBatchGate<CommandBuffer, (usize, usize)>; 2] =
        [("CX", CommandBuffer::cx), ("CZ", CommandBuffer::cz)];
    for (name, gate) in gates {
        let mut buffer = CommandBuffer::new();
        buffer.h(&[0]);
        let before = format!("{buffer:?}");
        let error = catch_unwind(AssertUnwindSafe(|| {
            gate(&mut buffer, &[(0, 1), (2, 2)]);
        }))
        .expect_err("buffer must reject repeated pair");
        let message = error.downcast_ref::<String>().expect("descriptive panic");
        assert!(message.contains(name), "{message}");
        assert!(message.contains("repeated qubit 2"), "{message}");
        assert_eq!(format!("{buffer:?}"), before, "{name} partially appended");
    }
}
