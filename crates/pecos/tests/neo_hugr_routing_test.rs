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

//! HUGR program routing to the neo stack.
//!
//! The neo stack runs HUGR through the PHIR engine (HUGR -> PHIR), so its
//! results use the same NAMED classical register contract (`c`) as the
//! engines/QASM path -- not the per-qubit `q0`/`q1` + `measurements` shape that
//! `pecos_hugr::hugr_engine` would emit -- and needs no Selene/LLVM. These
//! tests lock that, using the Guppy-generated fixtures shared with
//! `hugr_execution_tests.rs`, and cross-check against the engines PHIR engine.

#![cfg(feature = "neo")]

use std::collections::BTreeSet;

use pecos::prelude::Data;
use pecos::{SimStack, sim};
use pecos_programs::Hugr;

/// Run a HUGR fixture on the neo stack and return the `c` register of each shot.
/// Panics if a shot has no `c` register (the contract this test guards).
fn neo_hugr_c(bytes: &[u8], seed: u64, shots: usize) -> Vec<u32> {
    let results = sim(Hugr::from_bytes(bytes.to_vec()))
        .stack(SimStack::Neo)
        .seed(seed)
        .shots(shots)
        .run()
        .expect("neo HUGR run");
    results
        .shots
        .iter()
        .map(|shot| {
            shot.data
                .get("c")
                .and_then(Data::as_u32)
                .expect("neo HUGR results must expose the named `c` register")
        })
        .collect()
}

/// The engines-side reference: the same HUGR through `pecos_phir`'s engine on
/// the engines `sim_builder` (the `hugr_execution_tests.rs` path).
fn engines_phir_c(bytes: &[u8], seed: u64, shots: usize) -> Vec<u32> {
    let builder = pecos_phir::phir_engine()
        .from_hugr_bytes(bytes)
        .expect("HUGR -> PhirEngineBuilder");
    let results = pecos_engines::sim_builder()
        .classical(builder)
        .seed(seed)
        .run(shots)
        .expect("engines PHIR run");
    results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(Data::as_u32))
        .collect()
}

#[test]
fn neo_hugr_results_use_the_named_c_register() {
    // The contract: the only register key is `c`, NOT `q0`/`q1`/`measurements`.
    let bytes = include_bytes!("test_data/hugr/bell_state.hugr");
    let results = sim(Hugr::from_bytes(bytes.to_vec()))
        .stack(SimStack::Neo)
        .seed(42)
        .shots(5)
        .run()
        .expect("neo HUGR run");
    let keys: Vec<&String> = results.shots[0].data.keys().collect();
    assert!(
        results.shots[0].data.contains_key("c"),
        "neo HUGR must expose `c`; got keys {keys:?}"
    );
    assert!(
        !results.shots[0].data.contains_key("measurements")
            && !results.shots[0].data.contains_key("q0"),
        "neo HUGR must NOT expose the raw q0/measurements shape; got keys {keys:?}"
    );
}

#[test]
fn neo_hugr_bell_state_correlations() {
    let results = neo_hugr_c(include_bytes!("test_data/hugr/bell_state.hugr"), 42, 200);
    assert_eq!(results.len(), 200);
    for &v in &results {
        assert!(v == 0 || v == 3, "Bell on neo must be 00 or 11, got {v}");
    }
    assert!(results.contains(&0), "expected some 00");
    assert!(results.contains(&3), "expected some 11");
}

#[test]
fn neo_hugr_ghz_state_correlations() {
    let results = neo_hugr_c(include_bytes!("test_data/hugr/ghz_state.hugr"), 42, 200);
    assert_eq!(results.len(), 200);
    for &v in &results {
        assert!(v == 0 || v == 7, "GHZ on neo must be 000 or 111, got {v}");
    }
}

#[test]
fn neo_hugr_rz_x_is_deterministic() {
    // Rz(pi)|0> stays |0>, X|0> -> |1>: result 0b10 = 2 every shot.
    let results = neo_hugr_c(include_bytes!("test_data/hugr/rz_x.hugr"), 42, 20);
    assert_eq!(results.len(), 20);
    for &v in &results {
        assert_eq!(
            v, 2,
            "Rz(pi)|0> + X|0> on neo should give 0b10 = 2, got {v}"
        );
    }
}

#[test]
fn neo_hugr_support_matches_engines_phir() {
    // Cross-check the neo facade route against the engines PHIR engine (the
    // hugr_execution_tests path): both run the HUGR via PHIR, so the outcome
    // SUPPORT must agree. Independent seeds -- agreement is from the shared
    // contract, not a shared RNG stream.
    let bytes = include_bytes!("test_data/hugr/bell_state.hugr");
    let neo: BTreeSet<u32> = neo_hugr_c(bytes, 1, 400).into_iter().collect();
    let engines: BTreeSet<u32> = engines_phir_c(bytes, 2, 400).into_iter().collect();
    assert_eq!(
        neo, engines,
        "neo-facade HUGR and engines-PHIR HUGR must explore the same Bell support"
    );
    assert_eq!(neo, BTreeSet::from([0, 3]));
}

#[test]
fn neo_hugr_control_flow_is_rejected() {
    // The PHIR HUGR converter is straight-line only. HUGR with classical
    // control flow (loops, conditionals) must be REJECTED up front, not
    // silently converted into empty/partial results that look like a
    // successful run. Each of these fixtures compiles to a CFG with multiple
    // basic blocks.
    for fixture in [
        &include_bytes!("test_data/hugr/simple_while_loop.hugr")[..],
        &include_bytes!("test_data/hugr/forloop_h_test.hugr")[..],
        &include_bytes!("test_data/hugr/simple_conditional.hugr")[..],
        &include_bytes!("test_data/hugr/conditional_x.hugr")[..],
    ] {
        let err = sim(Hugr::from_bytes(fixture.to_vec()))
            .stack(SimStack::Neo)
            .seed(42)
            .shots(4)
            .run()
            .expect_err("control-flow HUGR must be rejected on the neo stack");
        assert!(
            err.to_string().contains("classical control flow"),
            "expected a control-flow rejection, got: {err}"
        );
    }
}
