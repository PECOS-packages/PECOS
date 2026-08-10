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

//! Collapse alignment for the lazily-framed stabilizer simulator.

use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, StabVec};

/// Collapse projects; it does not reset. The diagonal-frame shortcut used to
/// flip the reported outcome while collapsing the stored (unflipped)
/// eigenstate, so a repeated measurement read the pre-error value -- unlike
/// every eager simulator.
#[test]
fn a_framed_flip_survives_its_own_measurement() {
    let mut s = StabVec::new(1);
    s.x(&[QubitId(0)]);
    let m1 = s.mz(&[QubitId(0)]);
    let m2 = s.mz(&[QubitId(0)]);
    assert!(m1[0].outcome, "the X flips the first readout");
    assert!(
        m2[0].outcome,
        "and the post-measurement state must match the report"
    );
}

/// The default measure-and-prepare correction relies on the post-measurement
/// state matching the reported outcome.
#[test]
fn mpz_resets_after_a_framed_flip() {
    let mut s = StabVec::new(1);
    s.x(&[QubitId(0)]);
    let m1 = s.mpz(&[QubitId(0)]);
    let m2 = s.mz(&[QubitId(0)]);
    assert!(m1[0].outcome);
    assert!(!m2[0].outcome, "the built-in preparation left |0>");
}
