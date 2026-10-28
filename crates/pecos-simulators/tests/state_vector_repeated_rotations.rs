// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, SparseStateVecAoS, SparseStateVecSoA, StabVec,
    StateVecAoS, StateVecSoA, StateVecSoA32, StateVectorAccess,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

type Rotation<S> = for<'a> fn(&'a mut S, Angle64, &[(QubitId, QubitId)]) -> &'a mut S;

fn check_repeated_rotations<S: ArbitraryRotationGateable + StateVectorAccess>(mut sim: S) {
    sim.h(&[QubitId(0), QubitId(1), QubitId(2)])
        .sz(&[QubitId(1)]);
    let before = sim.state_vector().expect("snapshot the entire state");
    let gates: [(&str, Rotation<S>); 3] = [("RXX", S::rxx), ("RYY", S::ryy), ("RZZ", S::rzz)];
    for (name, gate) in gates {
        for qubit in [0, 2] {
            for angle in [Angle64::ZERO, Angle64::QUARTER_TURN] {
                let error = catch_unwind(AssertUnwindSafe(|| {
                    gate(
                        &mut sim,
                        angle,
                        &[(QubitId(0), QubitId(1)), (QubitId(qubit), QubitId(qubit))],
                    );
                }))
                .expect_err("repeated operands must fail before changing the state");
                let message = error.downcast_ref::<String>().expect("descriptive panic");
                assert!(message.contains(name), "{message}");
                assert!(
                    message.contains(&format!("requires distinct qubits; repeated qubit {qubit}")),
                    "{message}"
                );
                assert_eq!(sim.state_vector().expect("state remains readable"), before);
            }
        }
    }
}

#[test]
fn state_vector_repeated_rotations() {
    check_repeated_rotations(StateVecSoA::with_seed(3, 42));
    check_repeated_rotations(StateVecAoS::with_seed(3, 42));
    check_repeated_rotations(StateVecSoA32::with_seed(3, 42));
    check_repeated_rotations(SparseStateVecSoA::with_seed(3, 42));
    // Sparse AoS inherits the trait's RXX/RYY decompositions.
    check_repeated_rotations(SparseStateVecAoS::with_seed(3, 42));
}

#[test]
fn stab_vec_repeated_rotations_name_the_public_gate() {
    let mut sim: StabVec = StabVec::with_seed(3, 42);
    sim.h(&[QubitId(0), QubitId(1)]).sz(&[QubitId(1)]);
    assert!(!sim.is_dense(), "test the stabilizer decomposition path");
    let mut before = sim.clone();
    let gates: [(&str, Rotation<StabVec>); 3] = [
        ("RXX", StabVec::rxx),
        ("RYY", StabVec::ryy),
        ("RZZ", StabVec::rzz),
    ];
    for (name, gate) in gates {
        let error = catch_unwind(AssertUnwindSafe(|| {
            gate(
                &mut sim,
                Angle64::QUARTER_TURN,
                &[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(2))],
            );
        }))
        .expect_err("reject invalid batch before decomposition");
        let message = error.downcast_ref::<String>().expect("descriptive panic");
        assert!(message.contains(name), "{message}");
        assert!(message.contains("repeated qubit 2"), "{message}");
        assert_eq!(sim.state_vector(), before.state_vector());
    }
}
