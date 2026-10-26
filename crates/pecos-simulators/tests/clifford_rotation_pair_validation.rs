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

use pecos_core::{Angle64, QubitId};
use pecos_simulators::clifford_rotation::CliffordRotation;
use pecos_simulators::{CliffordGateable, SparseStab, StabilizerTableauSimulator};
use std::panic::{AssertUnwindSafe, catch_unwind};

const GATES: [&str; 6] = ["RZZ", "RXX", "RYY", "RXYXY2Q", "RXXRYYRZZ", "U2Q"];

fn apply(
    sim: &mut SparseStab,
    gate: &str,
    angle: Angle64,
    pairs: &[(QubitId, QubitId)],
) -> Result<(), String> {
    let quarter = Angle64::QUARTER_TURN;
    match gate {
        "RZZ" => sim.try_rzz(angle, pairs),
        "RXX" => sim.try_rxx(angle, pairs),
        "RYY" => sim.try_ryy(angle, pairs),
        "RXYXY2Q" => sim.try_rxyxy2q(angle, quarter, pairs),
        // Put the variable angle last to catch partial application of components.
        "RXXRYYRZZ" => sim.try_rxxryyrzz(quarter, quarter, angle, pairs),
        "U2Q" => sim.try_u2q(
            [[Angle64::ZERO; 3]; 2],
            [quarter, quarter, angle],
            [[quarter; 3]; 2],
            pairs,
        ),
        _ => panic!("unlisted gate {gate}"),
    }
    .map(|_| ())
}

fn prepared() -> SparseStab {
    let mut sim = SparseStab::with_seed(3, 42).with_destab_sign_tracking();
    sim.h(&[QubitId(0), QubitId(1), QubitId(2)])
        .sz(&[QubitId(1)]);
    sim
}

// Zero checks the identity shortcut and an eighth turn checks that the pair check
// runs before angle admission. The half turn decomposes into single-qubit Paulis.
fn angles() -> [Angle64; 4] {
    [
        Angle64::HALF_TURN,
        Angle64::QUARTER_TURN,
        Angle64::ZERO,
        Angle64::from_turn_ratio(1, 8),
    ]
}

#[test]
fn clifford_rotations_return_err_for_repeated_pairs_before_mutation() {
    for gate in GATES {
        for angle in angles() {
            let mut sim = prepared();
            let before = sim.clone();
            let error = apply(
                &mut sim,
                gate,
                angle,
                &[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(2))],
            )
            .expect_err("a repeated qubit in a later pair must reject the whole batch");
            assert_eq!(
                error,
                format!(
                    "{gate} requires distinct qubits; repeated qubit 2 appears twice in a pair"
                ),
                "{gate}, {angle}"
            );
            assert_eq!(sim.full_tableau(), before.full_tableau(), "{gate}, {angle}");
        }
    }
}

#[test]
fn clifford_rotations_check_bounds_before_mutation() {
    for gate in GATES {
        for angle in angles() {
            let mut sim = prepared();
            let before = sim.clone();
            let panic = catch_unwind(AssertUnwindSafe(|| {
                let _ = apply(
                    &mut sim,
                    gate,
                    angle,
                    &[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))],
                );
            }))
            .expect_err("an out-of-range qubit in a later pair must panic");
            let message = panic.downcast_ref::<String>().expect("actionable panic");
            assert!(
                message.contains("qubit 3 is outside the 3-qubit"),
                "{gate}: {message}"
            );
            assert_eq!(sim.full_tableau(), before.full_tableau(), "{gate}, {angle}");
        }
    }
}
