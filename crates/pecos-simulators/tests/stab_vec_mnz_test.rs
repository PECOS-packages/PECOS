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

//! Differential pinning of the -Z measurement family against `SparseStab`.

use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, SparseStab, StabVec};

/// The old Z-frame shortcut composed a frame that commutes with a Z readout
/// and could never flip the outcome. `mnz` now follows the trait's reference
/// decomposition and must agree with `SparseStab` on the deterministic cases.
#[test]
fn mnz_and_pnz_agree_with_sparse_stab() {
    // mnz on |0>: outcome inverted, state unchanged.
    let mut s = StabVec::new(1);
    let mut r = SparseStab::new(1);
    let (sm, rm) = (s.mnz(&[QubitId(0)]), r.mnz(&[QubitId(0)]));
    assert_eq!(sm[0].outcome, rm[0].outcome);
    assert_eq!(
        s.mz(&[QubitId(0)])[0].outcome,
        r.mz(&[QubitId(0)])[0].outcome
    );

    // X; mnz; mz: the X flips both readings.
    let mut s = StabVec::new(1);
    let mut r = SparseStab::new(1);
    s.x(&[QubitId(0)]);
    r.x(&[QubitId(0)]);
    assert_eq!(
        s.mnz(&[QubitId(0)])[0].outcome,
        r.mnz(&[QubitId(0)])[0].outcome
    );
    assert_eq!(
        s.mz(&[QubitId(0)])[0].outcome,
        r.mz(&[QubitId(0)])[0].outcome
    );

    // pnz then mz reads 1 on both.
    let mut s = StabVec::new(1);
    let mut r = SparseStab::new(1);
    s.pnz(&[QubitId(0)]);
    r.pnz(&[QubitId(0)]);
    assert_eq!(
        s.mz(&[QubitId(0)])[0].outcome,
        r.mz(&[QubitId(0)])[0].outcome
    );
    assert!(s.mz(&[QubitId(0)])[0].outcome, "pnz prepares |1>");
}
