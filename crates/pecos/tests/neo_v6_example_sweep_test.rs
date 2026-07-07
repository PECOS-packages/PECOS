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

//! Validation-gate item V6: the neo-routable EXAMPLE sweep.
//!
//! The repository's `sim()`-facade examples (`crates/pecos/examples/*.rs`)
//! exercise a handful of distinct circuits. Most pin an explicit quantum
//! backend (`.quantum(...)`), which the neo stack deliberately rejects, but the
//! CIRCUITS themselves are neo-routable. This sweep runs each example-derived
//! circuit through `sim().stack(...)` on BOTH stacks (auto-backend) and checks
//! that they agree, covering circuit shapes the curated V1 matrix does not:
//! a 3-qubit entangling circuit, whole-register `measure q -> c` syntax, and a
//! purely deterministic program.
//!
//! Circuits that only run via `sim_neo` directly, or that need QIS/PHIR/custom
//! backends, are NOT part of this sweep (they are not facade-routable); the
//! curated cross-stack physics lives in `neo_equivalence_matrix_test.rs`.

#![cfg(feature = "neo")]

use std::collections::BTreeSet;

use pecos::{SimStack, sim};
use pecos_engines::shot_results::ShotVec;
use pecos_num::jeffreys_interval;
use pecos_programs::Qasm;

const SHOTS: usize = 10_000;
/// ~4.4 sigma per side: a real stack disagreement, not sampling noise, fails.
const CONFIDENCE: f64 = 0.99999;
const SEED: u64 = 42;

// --- Example-derived circuits (verbatim from crates/pecos/examples/*.rs) ---

/// Bell pair with WHOLE-REGISTER measurement (`measure q -> c`), the form used
/// by every facade example. Noiseless support is exactly {00, 11}.
const BELL_WHOLE_REG: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
"#;

/// The 3-qubit circuit from `unified_sim_demo.rs`: q2 = q0 XOR q1, so every
/// noiseless shot has even parity (q0^q1^q2 = 0) over outcomes {000,011,101,110}.
const GHZ3: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    h q[1];
    cx q[0], q[2];
    cx q[1], q[2];
    measure q -> c;
"#;

/// Deterministic program (`unified_sim_demo.rs`): always reads "1".
const X_DETERMINISTIC: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q[0];
    measure q -> c;
"#;

/// Single Hadamard with per-bit measurement (`sim_api_examples.rs`): {0, 1}.
const SINGLE_H: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    measure q[0] -> c[0];
"#;

fn run(program: &str, stack: SimStack, seed: u64, noise: Option<f64>) -> ShotVec {
    let builder = sim(Qasm::from_string(program)).stack(stack).seed(seed);
    let results = match noise {
        Some(p) => builder
            .noise(pecos_engines::DepolarizingNoise { p })
            .shots(SHOTS)
            .run(),
        None => builder.shots(SHOTS).run(),
    };
    results.expect("simulation run")
}

fn bitstrings(v: &ShotVec) -> Vec<String> {
    v.shots
        .iter()
        .map(|s| s.data["c"].to_bitstring().expect("c register bits"))
        .collect()
}

fn support(v: &ShotVec) -> BTreeSet<String> {
    bitstrings(v).into_iter().collect()
}

/// Fraction of shots whose `c` register satisfies `pred`, with its count.
#[allow(clippy::cast_precision_loss)]
fn rate_where(v: &ShotVec, pred: impl Fn(&str) -> bool) -> (u64, f64) {
    let count = bitstrings(v).iter().filter(|b| pred(b)).count() as u64;
    (count, count as f64 / SHOTS as f64)
}

/// Assert the two stacks' rates for `pred` are statistically compatible
/// (independent seeds, Jeffreys overlap) and print the sweep row.
fn assert_cross_stack_rate(
    name: &str,
    program: &str,
    noise: Option<f64>,
    pred: impl Fn(&str) -> bool,
) {
    // Independent seeds: agreement must come from matching conventions, not a
    // shared RNG stream.
    let engines = run(program, SimStack::Engines, SEED, noise);
    let neo = run(program, SimStack::Neo, SEED ^ 0xA5A5, noise);
    let (e_count, _) = rate_where(&engines, &pred);
    let (n_count, _) = rate_where(&neo, &pred);
    let e_ci = jeffreys_interval(e_count, SHOTS as u64, CONFIDENCE);
    let n_ci = jeffreys_interval(n_count, SHOTS as u64, CONFIDENCE);
    println!(
        "V6 {name}: engines {e_count}/{SHOTS} CI [{:.4}, {:.4}], neo {n_count}/{SHOTS} CI [{:.4}, {:.4}]",
        e_ci.0, e_ci.1, n_ci.0, n_ci.1
    );
    assert!(
        e_ci.0 <= n_ci.1 && n_ci.0 <= e_ci.1,
        "V6 {name}: stack rates are statistically incompatible: \
         engines {e_count}/{SHOTS} vs neo {n_count}/{SHOTS}"
    );
}

#[test]
fn v6_deterministic_example_is_bit_identical_cross_stack() {
    // A noiseless deterministic program must produce IDENTICAL ShotVecs on both
    // stacks (same seed, no randomness) and read "1" every shot.
    let engines = run(X_DETERMINISTIC, SimStack::Engines, SEED, None);
    let neo = run(X_DETERMINISTIC, SimStack::Neo, SEED, None);
    assert_eq!(
        engines, neo,
        "deterministic example must be bit-identical across stacks"
    );
    assert_eq!(support(&neo), BTreeSet::from(["1".to_string()]));
}

#[test]
fn v6_bell_whole_register_measure_matches() {
    // Noiseless Bell with whole-register `measure q -> c`: both stacks must
    // produce exactly the correlated support {00, 11} and a compatible P(00).
    for stack in [SimStack::Engines, SimStack::Neo] {
        let v = run(BELL_WHOLE_REG, stack, SEED, None);
        assert_eq!(
            support(&v),
            BTreeSet::from(["00".to_string(), "11".to_string()]),
            "Bell ({stack:?}) must only produce the correlated outcomes 00/11"
        );
    }
    assert_cross_stack_rate("bell_p00", BELL_WHOLE_REG, None, |b| b == "00");
}

#[test]
fn v6_three_qubit_example_preserves_parity() {
    // The 3-qubit example sets q2 = q0 XOR q1, so every noiseless shot has even
    // parity. Both stacks must honor that (covers a circuit shape and a
    // whole-register measurement wider than the V1 matrix).
    let even_parity = |b: &str| b.bytes().filter(|&c| c == b'1').count() % 2 == 0;
    for stack in [SimStack::Engines, SimStack::Neo] {
        let v = run(GHZ3, stack, SEED, None);
        assert!(
            bitstrings(&v).iter().all(|b| even_parity(b)),
            "3-qubit example ({stack:?}) must yield only even-parity outcomes"
        );
        assert!(
            support(&v).len() >= 3,
            "3-qubit example ({stack:?}) should explore its 4-outcome support, got {:?}",
            support(&v)
        );
    }
    // The even-parity fraction is 1.0 noiselessly on both stacks; under
    // depolarizing noise it drops below 1 on BOTH stacks at a compatible rate.
    assert_cross_stack_rate("ghz3_even_parity_noisy", GHZ3, Some(0.02), even_parity);
}

#[test]
fn v6_single_hadamard_example_matches() {
    // Single Hadamard, per-bit measurement: support {0, 1} on both stacks and a
    // compatible P(0) ~ 0.5.
    for stack in [SimStack::Engines, SimStack::Neo] {
        let v = run(SINGLE_H, stack, SEED, None);
        assert_eq!(
            support(&v),
            BTreeSet::from(["0".to_string(), "1".to_string()]),
            "single-H ({stack:?}) must produce both 0 and 1"
        );
    }
    assert_cross_stack_rate("single_h_p0", SINGLE_H, None, |b| b == "0");
}
