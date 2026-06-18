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

//! Contract tests for routing `sim()` to the pecos-neo stack.
//!
//! The neo stack must return the same `ShotVec` contract as the engines
//! stack: for deterministic programs, results are compared for exact
//! equality across stacks.

#![cfg(feature = "neo")]

use pecos::{SimStack, monte_carlo, sim};
use pecos_programs::Qasm;

/// Deterministic program exercising measurement feedback: c ends as "11".
fn deterministic_conditional_qasm() -> Qasm {
    Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) x q[1];
        measure q[1] -> c[1];
        "#,
    )
}

#[test]
fn neo_stack_matches_engines_for_deterministic_qasm() {
    let engines = sim(deterministic_conditional_qasm())
        .stack(SimStack::Engines)
        .seed(42)
        .shots(5)
        .run()
        .expect("engines run");

    let neo = sim(deterministic_conditional_qasm())
        .stack(SimStack::Neo)
        .seed(42)
        .shots(5)
        .run()
        .expect("neo run");

    assert_eq!(engines.shots.len(), 5);
    assert_eq!(
        engines, neo,
        "Deterministic program must produce identical ShotVecs on both stacks"
    );
    for shot in &neo.shots {
        assert_eq!(shot.data["c"].to_bitstring().unwrap(), "11");
    }
}

#[test]
fn neo_stack_parallel_matches_engines() {
    let engines = sim(deterministic_conditional_qasm())
        .stack(SimStack::Engines)
        .seed(7)
        .workers(2)
        .shots(6)
        .run()
        .expect("engines run");

    let neo = sim(deterministic_conditional_qasm())
        .stack(SimStack::Neo)
        .seed(7)
        .workers(2)
        .shots(6)
        .run()
        .expect("neo run");

    assert_eq!(engines, neo);
}

#[test]
fn neo_stack_worker_count_invariant_for_noisy_program() {
    // The neo determinism guarantee (V4): each shot's RNG is derived from its
    // GLOBAL shot index, so a STOCHASTIC program's results must be bit-identical
    // regardless of how the shots are split across workers. The depolarizing
    // noise is what makes this a real RNG test -- a deterministic program would
    // pass trivially. (Same seed throughout; only the worker count varies.)
    let noise = pecos_engines::DepolarizingNoise { p: 0.3 };
    let run = |workers: usize| {
        sim(x_measure_qasm())
            .stack(SimStack::Neo)
            .noise(noise)
            .seed(42)
            .workers(workers)
            .shots(128)
            .run()
            .expect("neo run")
    };
    let w1 = run(1);
    // Self-check: the noise must actually produce a MIX of outcomes, or
    // worker-invariance would hold trivially (all shots identical).
    let rate0 = rate_of(&w1, "0");
    assert!(
        rate0 > 0.0 && rate0 < 1.0,
        "noisy program should produce varied outcomes (got rate(0)={rate0}), \
         otherwise worker-invariance is vacuous"
    );
    assert_eq!(
        w1,
        run(2),
        "neo noisy results must be invariant to worker count (1 vs 2)"
    );
    assert_eq!(
        w1,
        run(4),
        "neo noisy results must be invariant to worker count (1 vs 4)"
    );
}

#[test]
fn neo_stack_same_seed_is_reproducible() {
    // V3 reproducibility: identical config + identical seed -> bit-identical
    // ShotVec on neo across independent runs (a noisy program, so the RNG is
    // genuinely exercised).
    let noise = pecos_engines::DepolarizingNoise { p: 0.1 };
    let run = || {
        sim(x_measure_qasm())
            .stack(SimStack::Neo)
            .noise(noise)
            .seed(123)
            .shots(64)
            .run()
            .expect("neo run")
    };
    assert_eq!(
        run(),
        run(),
        "neo must reproduce identical results for a fixed seed"
    );
}

/// One-qubit program whose only error source is what the noise model adds.
fn x_measure_qasm() -> Qasm {
    Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        "#,
    )
}

/// Fraction of shots where register `c` reads the given bitstring.
#[allow(clippy::cast_precision_loss)] // shot counts are far below 2^52
fn rate_of(results: &pecos_engines::shot_results::ShotVec, bits: &str) -> f64 {
    let matching = results
        .shots
        .iter()
        .filter(|shot| shot.data["c"].to_bitstring().as_deref() == Some(bits))
        .count();
    matching as f64 / results.shots.len() as f64
}

#[test]
fn neo_stack_measurement_noise_rate_matches_engines() {
    // Measurement-only noise: P(c = 0) = p_meas exactly on both stacks.
    let p_meas = 0.2;
    let shots = 4000;
    let noise = pecos_engines::noise::DepolarizingNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_probability(p_meas)
        .with_p1_probability(0.0)
        .with_p2_probability(0.0);

    let engines = sim(x_measure_qasm())
        .stack(SimStack::Engines)
        .noise(noise.clone())
        .seed(42)
        .shots(shots)
        .run()
        .expect("engines run");
    let neo = sim(x_measure_qasm())
        .stack(SimStack::Neo)
        .noise(noise)
        .seed(42)
        .shots(shots)
        .run()
        .expect("neo run");

    let engines_rate = rate_of(&engines, "0");
    let neo_rate = rate_of(&neo, "0");

    // Bands: ~5 sigma for p=0.2 at 4000 shots is ~0.032.
    assert!(
        (engines_rate - p_meas).abs() < 0.035,
        "engines rate {engines_rate} should be near {p_meas}"
    );
    assert!(
        (neo_rate - p_meas).abs() < 0.035,
        "neo rate {neo_rate} should be near {p_meas}"
    );
}

#[test]
fn neo_stack_uniform_depolarizing_rate_matches_engines() {
    // Uniform depolarizing through the convenience struct: the compound
    // error rate must agree across stacks. This is a direct stack-vs-stack
    // comparison, so the two stacks use INDEPENDENT seeds — agreement must
    // come from matching conventions, not from a shared RNG stream (which
    // would make the check tautological if the streams ever converged).
    let shots = 4000;
    let run = |stack: SimStack| {
        let seed = if matches!(stack, SimStack::Neo) {
            7 ^ 0xA5A5
        } else {
            7
        };
        sim(x_measure_qasm())
            .stack(stack)
            .noise(pecos_engines::DepolarizingNoise { p: 0.1 })
            .seed(seed)
            .shots(shots)
            .run()
            .expect("run")
    };

    let engines_rate = rate_of(&run(SimStack::Engines), "0");
    let neo_rate = rate_of(&run(SimStack::Neo), "0");

    assert!(
        (engines_rate - neo_rate).abs() < 0.035,
        "compound error rates should agree: engines={engines_rate}, neo={neo_rate}"
    );
}

#[test]
fn neo_stack_biased_depolarizing_struct_rate_matches_engines() {
    // The BiasedDepolarizingNoise convenience struct (uniform p, with the
    // biased family's record-flip measurement) must agree cross-stack through
    // the facade mapping. Independent seeds, as above.
    let shots = 4000;
    let run = |stack: SimStack| {
        let seed = if matches!(stack, SimStack::Neo) {
            7 ^ 0xA5A5
        } else {
            7
        };
        sim(x_measure_qasm())
            .stack(stack)
            .noise(pecos_engines::BiasedDepolarizingNoise { p: 0.1 })
            .seed(seed)
            .shots(shots)
            .run()
            .expect("run")
    };

    let engines_rate = rate_of(&run(SimStack::Engines), "0");
    let neo_rate = rate_of(&run(SimStack::Neo), "0");

    assert!(
        (engines_rate - neo_rate).abs() < 0.035,
        "biased struct compound rates should agree: engines={engines_rate}, neo={neo_rate}"
    );
}

#[test]
fn neo_stack_general_noise_average_convention_matches() {
    // The critical convention test: engines' with_average_p1_probability
    // stores p1 = 1.5 x average internally (standard depolarizing
    // convention), which the mapping carries one-to-one to neo. With
    // average_p1 = 0.2 the effective depolarizing p1 is 0.3, so the
    // outcome flip rate on a single 1q gate is 2/3 x 0.3 = 0.2 on BOTH
    // stacks. A convention mismatch (double- or un-scaled) would shift
    // one stack's rate to ~0.13 or ~0.3 and fail loudly.
    let shots = 4000;
    let expected_flip = 0.2;
    let run = |stack: SimStack| {
        // GeneralNoiseModel defaults are realistic (nonzero emission, prep
        // leak, idle, and base probabilities); zero everything except the
        // 1q Pauli channel so the physics is plain depolarizing.
        let noise = pecos_engines::noise::GeneralNoiseModel::builder()
            .with_average_p1_probability(0.2)
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0)
            .with_prep_probability(0.0)
            .with_meas_0_probability(0.0)
            .with_meas_1_probability(0.0)
            .with_average_p2_probability(0.0);
        sim(x_measure_qasm())
            .stack(stack)
            .noise(noise)
            .seed(11)
            .shots(shots)
            .run()
            .expect("run")
    };

    let engines_rate = rate_of(&run(SimStack::Engines), "0");
    let neo_rate = rate_of(&run(SimStack::Neo), "0");

    assert!(
        (engines_rate - expected_flip).abs() < 0.035,
        "engines flip rate {engines_rate} should be near {expected_flip}"
    );
    assert!(
        (neo_rate - expected_flip).abs() < 0.035,
        "neo flip rate {neo_rate} should be near {expected_flip}"
    );
}

#[test]
fn neo_stack_rejects_unmapped_noise() {
    // A bare GeneralNoiseModel keeps its realistic defaults for prep leak
    // (0.5) and linear idling (0.001) — physics beyond the simple Pauli
    // subset, so the mapping must refuse rather than silently change the
    // model. (Spontaneous emission IS now mapped, so it is the prep-leak
    // and idle defaults that force the rejection here.)
    let general =
        pecos_engines::noise::GeneralNoiseModel::builder().with_average_p1_probability(0.01);
    let err = sim(deterministic_conditional_qasm())
        .stack(SimStack::Neo)
        .noise(general)
        .shots(5)
        .run()
        .expect_err("beyond-subset GeneralNoiseModel configs are not mapped");
    assert!(
        err.to_string()
            .contains("beyond the simple probability subset"),
        "unexpected error: {err}"
    );
}

#[test]
fn neo_stack_rejects_nonunit_emission_scale() {
    // `with_emission_scale` multiplies the emission ratios at build() but the
    // facade subset surfaces the RAW ratios, so a non-unit scale would map a
    // DIFFERENT emission rate to neo than engines runs. The facade must reject
    // it rather than silently diverge. (Codex batch-4 finding 1.)
    let general = pecos_engines::noise::GeneralNoiseModel::builder()
        .with_p1_probability(0.3)
        .with_p1_emission_ratio(0.25)
        .with_emission_scale(2.0)
        .with_p2_probability(0.0)
        .with_prep_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0)
        .with_prep_leak_ratio(0.0)
        .with_p_idle_linear_rate(0.0);
    let err = sim(deterministic_conditional_qasm())
        .stack(SimStack::Neo)
        .noise(general)
        .shots(5)
        .run()
        .expect_err("non-unit emission_scale must not be silently mapped to neo");
    assert!(
        err.to_string()
            .contains("beyond the simple probability subset"),
        "unexpected error: {err}"
    );
}

#[test]
fn neo_stack_rejects_unrouted_quantum_backend() {
    let err = sim(deterministic_conditional_qasm())
        .stack(SimStack::Neo)
        .quantum(pecos_engines::state_vector())
        .shots(5)
        .run()
        .expect_err("explicit quantum backends are not yet routed");
    assert!(err.to_string().contains("not yet routed to the neo stack"));
}

#[test]
fn neo_stack_rejects_build() {
    let Err(err) = sim(deterministic_conditional_qasm())
        .stack(SimStack::Neo)
        .build()
    else {
        panic!("neo stack has no MonteCarloEngine; build() must error");
    };
    assert!(err.to_string().contains("MonteCarloEngine"));
}

// --- Shared sampling vocabulary (.sampling(monte_carlo(n))) ----------------

/// The shared `monte_carlo()` run-spec drives BOTH stacks through the facade,
/// and `.shots(n)` is exactly its shorthand. A deterministic program lets us
/// assert exact `ShotVec` equality across the two spellings and the two stacks.
#[test]
fn facade_sampling_monte_carlo_drives_both_stacks() {
    for stack in [SimStack::Engines, SimStack::Neo] {
        let via_sampling = sim(deterministic_conditional_qasm())
            .stack(stack)
            .seed(42)
            .sampling(monte_carlo(5))
            .run()
            .expect("sampling run");
        let via_shots = sim(deterministic_conditional_qasm())
            .stack(stack)
            .seed(42)
            .shots(5)
            .run()
            .expect("shots run");

        assert_eq!(via_sampling.shots.len(), 5);
        assert_eq!(
            via_sampling, via_shots,
            "{stack:?}: .sampling(monte_carlo(5)) must equal .shots(5)"
        );
        for shot in &via_sampling.shots {
            assert_eq!(shot.data["c"].to_bitstring().unwrap(), "11");
        }
    }
}

/// `monte_carlo(n).workers(w)` carries worker parallelism through the facade on
/// both stacks; the deterministic program's results are worker-count invariant.
#[test]
fn facade_sampling_workers_runs_parallel_on_both_stacks() {
    for stack in [SimStack::Engines, SimStack::Neo] {
        let serial = sim(deterministic_conditional_qasm())
            .stack(stack)
            .seed(7)
            .sampling(monte_carlo(6))
            .run()
            .expect("serial run");
        let parallel = sim(deterministic_conditional_qasm())
            .stack(stack)
            .seed(7)
            .sampling(monte_carlo(6).workers(2))
            .run()
            .expect("parallel run");

        assert_eq!(parallel.shots.len(), 6);
        // A deterministic program yields identical outcomes regardless of how
        // shots are split across workers, on either stack.
        assert_eq!(
            serial, parallel,
            "{stack:?}: worker count must not change deterministic results"
        );
    }
}
