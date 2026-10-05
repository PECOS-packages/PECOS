// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Projective gate conformance on entangled inputs.

use num_complex::Complex64;
use pecos_core::{Angle64, Clifford, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, DenseStateVec};
use pecos_stab_tn::stab_mps::StabMps;

fn assert_equal_up_to_phase(a: &[Complex64], b: &[Complex64], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: state dimensions");
    let overlap: Complex64 = a.iter().zip(b).map(|(x, y)| x * y.conj()).sum();
    assert!(overlap.norm() > 1e-12, "{label}: orthogonal states");
    let phase = overlap / overlap.norm();
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (*x - phase * y).norm() < 1e-10,
            "{label}: amplitude {index}: {x} vs {y}, phase {phase}"
        );
    }
}

fn apply_single_qubit_clifford<S: CliffordGateable>(
    sim: &mut S,
    gate: Clifford,
    qubits: &[QubitId],
) {
    match gate {
        Clifford::I => sim.identity(qubits),
        Clifford::X => sim.x(qubits),
        Clifford::Y => sim.y(qubits),
        Clifford::Z => sim.z(qubits),
        Clifford::H => sim.h(qubits),
        Clifford::H2 => sim.h2(qubits),
        Clifford::H3 => sim.h3(qubits),
        Clifford::H4 => sim.h4(qubits),
        Clifford::H5 => sim.h5(qubits),
        Clifford::H6 => sim.h6(qubits),
        Clifford::SX => sim.sx(qubits),
        Clifford::SXdg => sim.sxdg(qubits),
        Clifford::SY => sim.sy(qubits),
        Clifford::SYdg => sim.sydg(qubits),
        Clifford::SZ => sim.sz(qubits),
        Clifford::SZdg => sim.szdg(qubits),
        Clifford::F => sim.f(qubits),
        Clifford::Fdg => sim.fdg(qubits),
        Clifford::F2 => sim.f2(qubits),
        Clifford::F2dg => sim.f2dg(qubits),
        Clifford::F3 => sim.f3(qubits),
        Clifford::F3dg => sim.f3dg(qubits),
        Clifford::F4 => sim.f4(qubits),
        Clifford::F4dg => sim.f4dg(qubits),
        _ => panic!("expected a single-qubit Clifford, got {gate}"),
    };
}

fn single_qubit_clifford_order(gate: Clifford) -> usize {
    let mut power = Clifford::I;
    for order in 1..=4 {
        power = gate.compose(power);
        if power == Clifford::I {
            return order;
        }
    }
    panic!("single-qubit Clifford {gate} has order greater than four");
}

fn apply_two_qubit_clifford(
    sim: &mut impl CliffordGateable,
    gate: Clifford,
    pairs: &[(QubitId, QubitId)],
) {
    match gate {
        Clifford::CX => sim.cx(pairs),
        Clifford::CY => sim.cy(pairs),
        Clifford::CZ => sim.cz(pairs),
        Clifford::SXX => sim.sxx(pairs),
        Clifford::SXXdg => sim.sxxdg(pairs),
        Clifford::SYY => sim.syy(pairs),
        Clifford::SYYdg => sim.syydg(pairs),
        Clifford::SZZ => sim.szz(pairs),
        Clifford::SZZdg => sim.szzdg(pairs),
        Clifford::SWAP => sim.swap(pairs),
        Clifford::ISWAP => sim.iswap(pairs),
        Clifford::ISWAPdg => sim.iswapdg(pairs),
        Clifford::G => sim.g(pairs),
        Clifford::Gdg => sim.gdg(pairs),
        _ => panic!("expected a two-qubit Clifford, got {gate}"),
    };
}

fn prepare_entangled_pair(sim: &mut impl CliffordGateable) {
    sim.h(&[QubitId(0)])
        .sz(&[QubitId(0)])
        .cx(&[(QubitId(0), QubitId(1))])
        .h(&[QubitId(1)]);
}

fn prepare_entangled_partners(sim: &mut impl CliffordGateable) {
    sim.h(&[QubitId(0), QubitId(1)])
        .cx(&[(QubitId(0), QubitId(2)), (QubitId(1), QubitId(3))])
        .sz(&[QubitId(0)])
        .h(&[QubitId(1)])
        .sz(&[QubitId(1)]);
}

fn apply_rotation(sim: &mut impl ArbitraryRotationGateable, gate: &str, angle: Angle64, q: usize) {
    let qubits = &[QubitId(q)];
    match gate {
        "rz" => sim.rz(angle, qubits),
        "rx" => sim.rx(angle, qubits),
        "ry" => sim.ry(angle, qubits),
        "rzz" => sim.rzz(angle, &[(QubitId(q), QubitId(1 - q))]),
        "u" => sim.u(angle, angle, angle, qubits),
        _ => panic!("unknown rotation {gate}"),
    };
}

fn dense_state(sim: &mut DenseStateVec, n: usize) -> Vec<Complex64> {
    (0..1 << n).map(|index| sim.get_amplitude(index)).collect()
}

#[test]
fn single_qubit_cliffords_match_dense_and_have_group_order() {
    for &gate in Clifford::all_1q() {
        for q in 0..2 {
            let mut sim = StabMps::new(2);
            let mut dense = DenseStateVec::new(2);
            prepare_entangled_pair(&mut sim);
            prepare_entangled_pair(&mut dense);
            let start = sim.state_vector_up_to_phase();
            apply_single_qubit_clifford(&mut sim, gate, &[QubitId(q)]);
            apply_single_qubit_clifford(&mut dense, gate, &[QubitId(q)]);
            assert_equal_up_to_phase(
                &sim.state_vector_up_to_phase(),
                &dense_state(&mut dense, 2),
                &format!("{gate:?} q={q}"),
            );
            for _ in 1..single_qubit_clifford_order(gate) {
                apply_single_qubit_clifford(&mut sim, gate, &[QubitId(q)]);
            }
            assert_equal_up_to_phase(
                &sim.state_vector_up_to_phase(),
                &start,
                &format!("{gate:?} power q={q}"),
            );
        }
    }
}

#[test]
fn two_qubit_cliffords_match_dense_in_both_orders() {
    for &gate in Clifford::all_2q() {
        for (a, b) in [(0, 1), (1, 0)] {
            let mut sim = StabMps::new(4);
            let mut dense = DenseStateVec::new(4);
            prepare_entangled_partners(&mut sim);
            prepare_entangled_partners(&mut dense);
            let pairs = &[(QubitId(a), QubitId(b))];
            apply_two_qubit_clifford(&mut sim, gate, pairs);
            apply_two_qubit_clifford(&mut dense, gate, pairs);
            assert_equal_up_to_phase(
                &sim.state_vector_up_to_phase(),
                &dense_state(&mut dense, 4),
                &format!("{gate:?} ({a},{b})"),
            );
        }
    }
}

#[test]
fn rotations_match_dense_with_and_without_merging() {
    for merge in [false, true] {
        for gate in ["rz", "rx", "ry", "rzz", "u"] {
            for angle in [
                Angle64::ZERO,
                Angle64::QUARTER_TURN,
                Angle64::HALF_TURN,
                Angle64::THREE_QUARTERS_TURN,
                Angle64::from_radians(0.37),
            ] {
                for q in 0..2 {
                    for repetitions in [1, 2] {
                        let mut sim = StabMps::builder(4)
                            .merge_rz(merge)
                            .svd_cutoff(0.0)
                            .max_truncation_error(0.0)
                            .build();
                        let mut dense = DenseStateVec::new(4);
                        prepare_entangled_partners(&mut sim);
                        prepare_entangled_partners(&mut dense);
                        // Repetition exercises the merge buffer, including angle wraparound.
                        for _ in 0..repetitions {
                            apply_rotation(&mut sim, gate, angle, q);
                            apply_rotation(&mut dense, gate, angle, q);
                        }
                        sim.flush();
                        assert_equal_up_to_phase(
                            &sim.state_vector_up_to_phase(),
                            &dense_state(&mut dense, 4),
                            &format!(
                                "{gate} {angle:?}, q={q}, merge={merge}, repetitions={repetitions}"
                            ),
                        );
                    }
                }
            }
        }
    }
}
