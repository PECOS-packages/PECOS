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

//! Differential test for the `GeneralNoiseModel` EMISSION channel across stacks.
//!
//! Engines models spontaneous emission as REPLACING the gate (the gate is
//! dropped on an emission fault). neo now matches this by undoing the gate
//! (applying its dagger) before the emission error. Each test pins three
//! configurations against the gate-removing analytic and against one another:
//! engines, neo configured DIRECTLY via `sim_neo`, and neo reached through the
//! `sim().stack(Neo)` FACADE (which now maps the engines emission ratios onto
//! neo's builder).
//!
//! Analytic for `x; measure` with uniform Pauli and emission weights, gate
//! error `p1` and emission ratio `e`: an emission fault (probability `p1*e`)
//! drops the X so the qubit is `|0>` and a uniform Pauli reads `0` only on Z
//! (`P(0) = 1/3`); a Pauli fault (probability `p1*(1-e)`) keeps the X so the
//! qubit is `|1>` and a uniform Pauli reads `0` on X or Y (`P(0) = 2/3`). Hence
//! `P(0) = p1 * (e/3 + (1-e)*2/3)`. At `p1 = 0.3`, `e = 0.5` this is `0.15` (the
//! gate-PRESERVING model would give `0.2`).

#![cfg(feature = "neo")]

use pecos::{SimStack, sim};
use pecos_num::jeffreys_interval;
use pecos_programs::Qasm;

const SHOTS: usize = 20_000;
const CONFIDENCE: f64 = 0.99999;
const P1: f64 = 0.3;
const EMISSION: f64 = 0.5;

const X_MEASURE: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q[0];
    measure q[0] -> c[0];
"#;

#[allow(clippy::cast_precision_loss)]
fn rate_zero(shots: &pecos_engines::shot_results::ShotVec) -> (u64, f64) {
    let zeros = shots
        .shots
        .iter()
        .filter(|s| s.data["c"].to_bitstring().as_deref() == Some("0"))
        .count() as u64;
    (zeros, zeros as f64 / SHOTS as f64)
}

/// The single-qubit emission noise: gate error `p1`, emission ratio
/// `EMISSION`, everything else (prep, meas, p2, leakage, idle) zeroed so the
/// only physics is the single-qubit emission/Pauli channel on the X gate. A
/// fresh builder each call since `.noise()` consumes it.
fn emission_noise_1q() -> pecos_engines::noise::GeneralNoiseModelBuilder {
    pecos_engines::noise::GeneralNoiseModel::builder()
        .with_p1_probability(P1)
        .with_p1_emission_ratio(EMISSION)
        .with_p2_probability(0.0)
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_prep_leak_ratio(0.0)
        .with_p_idle_linear_rate(0.0)
}

fn engines_zero_count() -> u64 {
    let results = sim(Qasm::from_string(X_MEASURE))
        .stack(SimStack::Engines)
        .noise(emission_noise_1q())
        .seed(42)
        .shots(SHOTS)
        .run()
        .expect("engines run");
    rate_zero(&results).0
}

/// neo reached through the `sim().stack(Neo)` FACADE with the SAME engines
/// `GeneralNoiseModel`: the facade maps the emission ratio onto neo's builder.
fn neo_facade_zero_count() -> u64 {
    let results = sim(Qasm::from_string(X_MEASURE))
        .stack(SimStack::Neo)
        .noise(emission_noise_1q())
        .seed(7) // independent seed; agreement must be physical
        .shots(SHOTS)
        .run()
        .expect("neo facade run");
    rate_zero(&results).0
}

/// neo configured directly (the facade does not map emission yet): the
/// `GeneralNoiseModelBuilder` mirrors the same single-qubit emission channel.
fn neo_zero_count() -> u64 {
    use pecos_neo::noise::GeneralNoiseModelBuilder;
    use pecos_neo::tool::{monte_carlo, sim_neo};

    let noise = GeneralNoiseModelBuilder::new()
        .with_p1(P1)
        .with_p1_emission_ratio(EMISSION)
        .with_p2(0.0)
        .with_p_prep(0.0)
        .with_p_meas_symmetric(0.0);
    let results = sim_neo(Qasm::from_string(X_MEASURE))
        .auto()
        .sampling(monte_carlo(SHOTS))
        .noise(noise)
        .seed(99) // independent of the engines seed; agreement must be physical
        .run();
    let shots = results.shots.expect("neo produced shots");
    rate_zero(&shots).0
}

#[test]
fn emission_is_gate_removing_and_matches_engines() {
    let analytic = P1 * (EMISSION / 3.0 + (1.0 - EMISSION) * 2.0 / 3.0); // 0.15

    let engines = engines_zero_count();
    let neo = neo_zero_count();
    let facade = neo_facade_zero_count();
    let engines_ci = jeffreys_interval(engines, SHOTS as u64, CONFIDENCE);
    let neo_ci = jeffreys_interval(neo, SHOTS as u64, CONFIDENCE);
    let facade_ci = jeffreys_interval(facade, SHOTS as u64, CONFIDENCE);
    println!(
        "emission: engines {engines}/{SHOTS} CI [{:.4}, {:.4}], neo-direct {neo}/{SHOTS} CI \
         [{:.4}, {:.4}], neo-facade {facade}/{SHOTS} CI [{:.4}, {:.4}], gate-removing analytic \
         {analytic:.4} (gate-preserving would be {:.4})",
        engines_ci.0,
        engines_ci.1,
        neo_ci.0,
        neo_ci.1,
        facade_ci.0,
        facade_ci.1,
        P1 * 2.0 / 3.0
    );

    // All three configurations contain the gate-REMOVING analytic (0.15),
    // proving the gate is dropped on emission; the gate-PRESERVING value (0.2)
    // is excluded. The facade route additionally proves the engines->neo
    // emission-ratio mapping is wired through `sim().stack(Neo)`.
    for (name, ci) in [
        ("engines", engines_ci),
        ("neo-direct", neo_ci),
        ("neo-facade", facade_ci),
    ] {
        assert!(
            ci.0 <= analytic && analytic <= ci.1,
            "{name} P(0) excludes the gate-removing analytic {analytic}"
        );
    }
    // And every pair of stacks agrees.
    assert!(
        engines_ci.0 <= neo_ci.1 && neo_ci.0 <= engines_ci.1,
        "engines and neo-direct emission rates disagree: {engines}/{SHOTS} vs {neo}/{SHOTS}"
    );
    assert!(
        engines_ci.0 <= facade_ci.1 && facade_ci.0 <= engines_ci.1,
        "engines and neo-facade emission rates disagree: {engines}/{SHOTS} vs {facade}/{SHOTS}"
    );
}

// --- Two-qubit emission ---------------------------------------------------

const P2: f64 = 0.6;

/// `x q0; cx q0,q1; measure q1`. Pure two-qubit emission (`emission=1.0`,
/// `p1=0`) on the CX. If the CX is DROPPED, q1 stays 0 and a uniform two-qubit
/// Pauli flips it on 8/15 -> `P(q1=0) = p2 * 7/15`. If the CX is KEPT, q1 is 1
/// and the Pauli flips it on 8/15 -> `P(q1=0) = p2 * 8/15`. At `p2=0.6` that is
/// `0.28` (gate-removing) vs `0.32` (gate-preserving).
const CX_MEASURE: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[1];
    x q[0];
    cx q[0], q[1];
    measure q[1] -> c[0];
"#;

fn emission_noise_2q() -> pecos_engines::noise::GeneralNoiseModelBuilder {
    pecos_engines::noise::GeneralNoiseModel::builder()
        .with_p1_probability(0.0)
        .with_p2_probability(P2)
        .with_p2_emission_ratio(1.0)
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_prep_leak_ratio(0.0)
        .with_p_idle_linear_rate(0.0)
}

fn engines_2q_zero_count() -> u64 {
    let results = sim(Qasm::from_string(CX_MEASURE))
        .stack(SimStack::Engines)
        .noise(emission_noise_2q())
        .seed(42)
        .shots(SHOTS)
        .run()
        .expect("engines run");
    rate_zero(&results).0
}

fn neo_facade_2q_zero_count() -> u64 {
    let results = sim(Qasm::from_string(CX_MEASURE))
        .stack(SimStack::Neo)
        .noise(emission_noise_2q())
        .seed(7)
        .shots(SHOTS)
        .run()
        .expect("neo facade run");
    rate_zero(&results).0
}

fn neo_2q_zero_count() -> u64 {
    use pecos_neo::noise::GeneralNoiseModelBuilder;
    use pecos_neo::tool::{monte_carlo, sim_neo};

    let noise = GeneralNoiseModelBuilder::new()
        .with_p1(0.0)
        .with_p2(P2)
        .with_p2_emission_ratio(1.0)
        .with_p_prep(0.0)
        .with_p_meas_symmetric(0.0);
    let results = sim_neo(Qasm::from_string(CX_MEASURE))
        .auto()
        .sampling(monte_carlo(SHOTS))
        .noise(noise)
        .seed(99)
        .run();
    let shots = results.shots.expect("neo produced shots");
    rate_zero(&shots).0
}

#[test]
fn two_qubit_emission_is_gate_removing_and_matches_engines() {
    let analytic = P2 * 7.0 / 15.0; // 0.28 (gate-preserving would be P2*8/15 = 0.32)

    let engines = engines_2q_zero_count();
    let neo = neo_2q_zero_count();
    let facade = neo_facade_2q_zero_count();
    let engines_ci = jeffreys_interval(engines, SHOTS as u64, CONFIDENCE);
    let neo_ci = jeffreys_interval(neo, SHOTS as u64, CONFIDENCE);
    let facade_ci = jeffreys_interval(facade, SHOTS as u64, CONFIDENCE);
    println!(
        "2q emission: engines {engines}/{SHOTS} CI [{:.4}, {:.4}], neo-direct {neo}/{SHOTS} CI \
         [{:.4}, {:.4}], neo-facade {facade}/{SHOTS} CI [{:.4}, {:.4}], gate-removing analytic \
         {analytic:.4} (gate-preserving would be {:.4})",
        engines_ci.0,
        engines_ci.1,
        neo_ci.0,
        neo_ci.1,
        facade_ci.0,
        facade_ci.1,
        P2 * 8.0 / 15.0
    );

    for (name, ci) in [
        ("engines", engines_ci),
        ("neo-direct", neo_ci),
        ("neo-facade", facade_ci),
    ] {
        assert!(
            ci.0 <= analytic && analytic <= ci.1,
            "{name} P(q1=0) excludes the gate-removing analytic {analytic}"
        );
    }
    assert!(
        engines_ci.0 <= neo_ci.1 && neo_ci.0 <= engines_ci.1,
        "engines and neo-direct 2q emission rates disagree: {engines}/{SHOTS} vs {neo}/{SHOTS}"
    );
    assert!(
        engines_ci.0 <= facade_ci.1 && facade_ci.0 <= engines_ci.1,
        "engines and neo-facade 2q emission rates disagree: {engines}/{SHOTS} vs {facade}/{SHOTS}"
    );
}
