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

//! Integration tests for HUGR program support in pecos-neo.

#![cfg(feature = "hugr")]

use pecos_core::QubitId;
use pecos_neo::tool::sim_neo;
use pecos_programs::Hugr;
use std::path::PathBuf;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("crates/pecos/tests/test_data/hugr")
}

fn load_hugr(filename: &str) -> Hugr {
    Hugr::from_file(test_data_dir().join(filename)).unwrap()
}

#[test]
fn test_hugr_bell_state_auto() {
    let results = sim_neo(load_hugr("bell_state.hugr"))
        .auto()
        .seed(42)
        .shots(100)
        .build()
        .run();

    assert_eq!(results.len(), 100);

    for outcome in &results.outcomes {
        let q0 = outcome.get_bit(QubitId(0)).unwrap_or(false);
        let q1 = outcome.get_bit(QubitId(1)).unwrap_or(false);
        assert_eq!(q0, q1, "Bell state qubits should be correlated");
    }
}

#[test]
fn test_hugr_single_hadamard_auto() {
    let results = sim_neo(load_hugr("single_hadamard.hugr"))
        .auto()
        .seed(42)
        .shots(200)
        .build()
        .run();

    assert_eq!(results.len(), 200);

    let ones: usize = results
        .outcomes
        .iter()
        .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
        .count();
    let zeros = 200 - ones;

    assert!(zeros > 50, "Expected roughly half zeros, got {zeros}");
    assert!(ones > 50, "Expected roughly half ones, got {ones}");
}

#[test]
fn test_hugr_explicit_engine() {
    let hugr = load_hugr("bell_state.hugr");
    let source = String::from_utf8_lossy(&hugr.hugr).into_owned();
    let results = sim_neo(source)
        .classical(pecos_hugr::hugr_engine())
        .seed(42)
        .shots(10)
        .build()
        .run();

    assert_eq!(results.len(), 10);
}

#[test]
fn test_hugr_via_program_enum_auto() {
    let program = pecos_programs::Program::Hugr(load_hugr("bell_state.hugr"));
    let results = sim_neo(program).auto().seed(42).shots(10).build().run();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_hugr_seeded_reproducibility() {
    let results1 = sim_neo(load_hugr("single_hadamard.hugr"))
        .auto()
        .seed(123)
        .shots(50)
        .build()
        .run();

    let results2 = sim_neo(load_hugr("single_hadamard.hugr"))
        .auto()
        .seed(123)
        .shots(50)
        .build()
        .run();

    assert_eq!(results1.len(), results2.len());
    for (o1, o2) in results1.outcomes.iter().zip(results2.outcomes.iter()) {
        assert_eq!(
            o1.get_bit(QubitId(0)),
            o2.get_bit(QubitId(0)),
            "Same seed should produce identical outcomes"
        );
    }
}

#[test]
fn test_hugr_different_seeds_differ() {
    let results1 = sim_neo(load_hugr("single_hadamard.hugr"))
        .auto()
        .seed(42)
        .shots(50)
        .build()
        .run();

    let results2 = sim_neo(load_hugr("single_hadamard.hugr"))
        .auto()
        .seed(99)
        .shots(50)
        .build()
        .run();

    let same_count: usize = results1
        .outcomes
        .iter()
        .zip(results2.outcomes.iter())
        .filter(|(o1, o2)| o1.get_bit(QubitId(0)) == o2.get_bit(QubitId(0)))
        .count();

    assert!(
        same_count < 50,
        "Different seeds should produce different results"
    );
}
