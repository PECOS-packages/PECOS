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

//! `F` and `Fdg` must match their decompositions across the FFI boundary.
//!
//! `CppSparseStab` forwards `f()` to a C++ implementation instead of inheriting
//! the `sx()`-then-`sz()` default, so the Rust and C++ sides could disagree with
//! nothing to notice. Issue #379 was exactly that kind of drift between two
//! independent implementations of the same gate.
//!
//! `F` cycles `X -> Y -> Z -> X` and `Fdg` reverses it, `X -> Z -> Y -> X`.

use pecos_core::QubitId;
use pecos_cppsparsestab::CppSparseStab;
use pecos_simulators::CliffordGateable;

/// Signed stabilizer and destabilizer tableaux after preparing a state and
/// applying `gates`.
fn tableaux(prep: &[&str], gates: &[&str]) -> (String, String) {
    let mut sim = CppSparseStab::new(1);
    let q = [QubitId(0)];
    for gate in prep.iter().chain(gates) {
        match *gate {
            "h" => sim.h(&q),
            "sx" => sim.sx(&q),
            "sxdg" => sim.sxdg(&q),
            "sz" => sim.sz(&q),
            "szdg" => sim.szdg(&q),
            "f" => sim.f(&q),
            "fdg" => sim.fdg(&q),
            other => panic!("unexpected gate {other}"),
        };
    }
    (sim.stab_tableau(), sim.destab_tableau())
}

#[test]
fn face_gates_match_their_decompositions() {
    const PREPS: [&[&str]; 4] = [&[], &["h"], &["h", "sz"], &["sx"]];
    for prep in PREPS {
        assert_eq!(
            tableaux(prep, &["f"]),
            tableaux(prep, &["sx", "sz"]),
            "CppSparseStab f() must equal sx then sz (prep {prep:?})"
        );
        assert_eq!(
            tableaux(prep, &["fdg"]),
            tableaux(prep, &["szdg", "sxdg"]),
            "CppSparseStab fdg() must equal szdg then sxdg (prep {prep:?})"
        );
    }
}

#[test]
fn face_gate_and_its_adjoint_are_inverse() {
    const PREPS: [&[&str]; 4] = [&[], &["h"], &["h", "sz"], &["sx"]];
    for prep in PREPS {
        assert_eq!(
            tableaux(prep, &["f", "fdg"]),
            tableaux(prep, &[]),
            "f() then fdg() must be the identity (prep {prep:?})"
        );
    }
}
