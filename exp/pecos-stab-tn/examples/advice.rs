// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Replay a circuit, recommend a simulator, and compare three ancilla budgets.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::compile::{ExecutionAdvice, StabMpsCompile};

fn analyzed_circuit() -> StabMpsCompile {
    let mut analysis = StabMpsCompile::new(20);
    analysis.h(&[QubitId(0), QubitId(1)]);
    analysis.cx(&[(QubitId(0), QubitId(2)), (QubitId(1), QubitId(3))]);
    analysis.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0), QubitId(1)]);
    analysis
}

fn print_advice(label: &str, advice: &ExecutionAdvice) {
    println!("{label}:");
    println!("  simulator: {:?}", advice.simulator);
    println!("  injection: {:?}", advice.injection);
    println!("  injectable_count: {}", advice.injectable_count);
    println!(
        "  deferred_ancillas_required: {}",
        advice.deferred_ancillas_required
    );
    println!("  deferred_feasible: {:?}", advice.deferred_feasible);
    println!("  warnings: {:?}", advice.warnings);
    println!("  reason: {}", advice.reason);
}

fn main() {
    let analysis = analyzed_circuit();
    let recommendation = analysis.recommend();
    println!(
        "recommendation: {:?}: {}",
        recommendation.kind, recommendation.reason
    );

    let required = usize::try_from(analysis.nonclifford_rz_total())
        .expect("the example's gate count fits usize");
    print_advice("sufficient budget", &analysis.advise(Some(required)));
    print_advice(
        "insufficient budget",
        &analysis.advise(Some(required.saturating_sub(1))),
    );
    print_advice("unspecified budget", &analysis.advise(None));
}
