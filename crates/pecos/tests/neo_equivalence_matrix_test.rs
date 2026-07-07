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

//! Statistical equivalence matrix between the engines and neo stacks
//! (validation-gate item V1).
//!
//! Each cell runs the same QASM program with the same mapped noise
//! configuration through `sim()` on both stacks and compares the target
//! outcome rate with Jeffreys credible intervals: the two stacks'
//! intervals must overlap, and where the rate has an exact analytic
//! value, each stack's interval must contain it.
//!
//! Program-type coverage beyond QASM (HUGR) and exact worker-count
//! invariance are covered by `neo_routing_test.rs`; surface-code-scale
//! decoded equivalence is covered by `neo_surface_ler_test.rs`.

#![cfg(feature = "neo")]

use pecos::{SimStack, sim};
use pecos_engines::shot_results::ShotVec;
use pecos_num::jeffreys_interval;
use pecos_programs::Qasm;

const SHOTS: usize = 20_000;
/// ~4.4 sigma per side: stack disagreement, not sampling noise, is what
/// fails a cell.
const CONFIDENCE: f64 = 0.99999;
const SEED: u64 = 42;

const X_MEASURE: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q[0];
    measure q[0] -> c[0];
"#;

const RESET_MEASURE: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    reset q[0];
    measure q[0] -> c[0];
"#;

const MEASURE_TWICE: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    reset q[0];
    measure q[0] -> c[0];
    measure q[0] -> c[0];
"#;

const BELL: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
"#;

const FEEDBACK: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    x q[0];
    measure q[0] -> c[0];
    if (c == 1) x q[1];
    measure q[1] -> c[1];
"#;

/// RZZ at +pi/2 on |00> (a ZZ=+1 eigenstate): the gate leaves the state in
/// |00> (up to phase), so the only outcome change is from the angle-scaled
/// two-qubit depolarizing noise on the RZZ. The 8 of 15 non-identity Paulis
/// that anticommute with Z(x)Z flip the parity, giving `P(01 or 10) = 8*p_eff/15`.
const RZZ_POS: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    rzz(0.5*pi) q[0],q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
"#;

/// Same as `RZZ_POS` but at -pi/2, exercising the NEGATIVE-angle branch of the
/// asymmetric scaling (engines `(a, b)` / neo `neg_*`).
const RZZ_NEG: &str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    rzz(-0.5*pi) q[0],q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
"#;

/// One noise configuration of the matrix, applied identically to both
/// stacks (the facade maps it to neo's noise channels).
enum NoiseCell {
    Meas(f64),
    Prep(f64),
    P1(f64),
    P2(f64),
    Uniform(f64),
    GnmSimple {
        average_p1: f64,
        p_meas: f64,
    },
    /// `GeneralNoiseModel` two-qubit noise with angle-dependent scaling: stored
    /// `p2`, asymmetric coefficients `(a, b, c, d)` and `power`. Everything
    /// outside the two-qubit channel is zeroed so the physics is exactly the
    /// angle-scaled depolarizing channel on the RZZ gate.
    GnmAngle {
        p2: f64,
        angle_params: (f64, f64, f64, f64),
        angle_power: f64,
    },
    /// `BiasedDepolarizingNoiseModel` with zero gate/prep noise and asymmetric
    /// record-flip measurement: `p_meas_0` flips a 0 outcome to 1, `p_meas_1`
    /// flips a 1 outcome to 0. The bias is applied to the recorded outcome
    /// after readout (never the state), so it must map to neo's record-flip
    /// channel, not the state-flip one.
    BiasedMeas {
        p_meas_0: f64,
        p_meas_1: f64,
    },
}

impl NoiseCell {
    fn run(&self, qasm: &str, stack: SimStack) -> ShotVec {
        // Independent seed per stack. Each cell compares the two stacks'
        // empirical rates (Jeffreys overlap), so the comparison must be
        // between INDEPENDENT samples — a shared seed would make it
        // tautological if the two stacks' per-shot RNG streams ever
        // converged. Each stack is also checked against its analytic value,
        // which holds for any seed.
        let seed = if matches!(stack, SimStack::Neo) {
            SEED ^ 0xA5A5
        } else {
            SEED
        };
        let builder = sim(Qasm::from_string(qasm)).stack(stack).seed(seed);
        let depol = |p_prep: f64, p_meas: f64, p1: f64, p2: f64| {
            pecos_engines::noise::DepolarizingNoiseModel::builder()
                .with_prep_probability(p_prep)
                .with_meas_probability(p_meas)
                .with_p1_probability(p1)
                .with_p2_probability(p2)
        };
        let results = match *self {
            Self::Meas(p) => builder.noise(depol(0.0, p, 0.0, 0.0)).shots(SHOTS).run(),
            Self::Prep(p) => builder.noise(depol(p, 0.0, 0.0, 0.0)).shots(SHOTS).run(),
            Self::P1(p) => builder.noise(depol(0.0, 0.0, p, 0.0)).shots(SHOTS).run(),
            Self::P2(p) => builder.noise(depol(0.0, 0.0, 0.0, p)).shots(SHOTS).run(),
            Self::Uniform(p) => builder
                .noise(pecos_engines::DepolarizingNoise { p })
                .shots(SHOTS)
                .run(),
            Self::GnmSimple { average_p1, p_meas } => builder
                .noise(
                    // GeneralNoiseModel has realistic non-zero defaults;
                    // zero everything outside the simple Pauli subset so
                    // the cell physics is exactly known.
                    pecos_engines::noise::GeneralNoiseModel::builder()
                        .with_average_p1_probability(average_p1)
                        .with_average_p2_probability(0.0)
                        .with_p1_emission_ratio(0.0)
                        .with_p2_emission_ratio(0.0)
                        .with_prep_leak_ratio(0.0)
                        .with_p_idle_linear_rate(0.0)
                        .with_prep_probability(0.0)
                        .with_meas_0_probability(p_meas)
                        .with_meas_1_probability(p_meas),
                )
                .shots(SHOTS)
                .run(),
            Self::GnmAngle {
                p2,
                angle_params: (a, b, c, d),
                angle_power,
            } => builder
                .noise(
                    // Plain Pauli two-qubit noise with angle scaling; zero
                    // every other channel and the non-neutral GNM defaults so
                    // only the angle-scaled RZZ depolarizing noise remains.
                    pecos_engines::noise::GeneralNoiseModel::builder()
                        .with_p2_probability(p2)
                        .with_p2_angle_params(a, b, c, d)
                        .with_p2_angle_power(angle_power)
                        .with_average_p1_probability(0.0)
                        .with_p1_emission_ratio(0.0)
                        .with_p2_emission_ratio(0.0)
                        .with_prep_leak_ratio(0.0)
                        .with_p_idle_linear_rate(0.0)
                        .with_prep_probability(0.0)
                        .with_meas_0_probability(0.0)
                        .with_meas_1_probability(0.0),
                )
                .shots(SHOTS)
                .run(),
            Self::BiasedMeas { p_meas_0, p_meas_1 } => builder
                .noise(
                    // Asymmetric record-flip measurement, no gate/prep noise.
                    pecos_engines::noise::BiasedDepolarizingNoiseModel::builder()
                        .with_prep_probability(0.0)
                        .with_meas_0_probability(p_meas_0)
                        .with_meas_1_probability(p_meas_1)
                        .with_single_qubit_probability(0.0)
                        .with_two_qubit_probability(0.0),
                )
                .shots(SHOTS)
                .run(),
        };
        results.expect("simulation run")
    }
}

/// Count shots whose register `c` reads any of the target bitstrings.
///
/// Every target set used here is symmetric under bit reversal, so the
/// count is independent of register bit ordering.
fn count_targets(results: &ShotVec, targets: &[&str]) -> u64 {
    results
        .shots
        .iter()
        .filter(|shot| {
            let bits = shot.data["c"].to_bitstring().expect("c register bits");
            targets.contains(&bits.as_str())
        })
        .count() as u64
}

/// Run one matrix cell on both stacks and apply the equivalence (and
/// optional analytic-truth) assertions.
fn check_cell(name: &str, qasm: &str, cell: &NoiseCell, targets: &[&str], analytic: Option<f64>) {
    let engines = count_targets(&cell.run(qasm, SimStack::Engines), targets);
    let neo = count_targets(&cell.run(qasm, SimStack::Neo), targets);

    let engines_ci = jeffreys_interval(engines, SHOTS as u64, CONFIDENCE);
    let neo_ci = jeffreys_interval(neo, SHOTS as u64, CONFIDENCE);
    println!(
        "{name}: engines {engines}/{SHOTS} CI [{:.5}, {:.5}], \
         neo {neo}/{SHOTS} CI [{:.5}, {:.5}], analytic {analytic:?}",
        engines_ci.0, engines_ci.1, neo_ci.0, neo_ci.1
    );

    assert!(
        engines_ci.0 <= neo_ci.1 && neo_ci.0 <= engines_ci.1,
        "{name}: stack rates are statistically incompatible: \
         engines {engines}/{SHOTS} vs neo {neo}/{SHOTS}"
    );

    if let Some(truth) = analytic {
        assert!(
            engines_ci.0 <= truth && truth <= engines_ci.1,
            "{name}: engines rate {engines}/{SHOTS} excludes the analytic value {truth}"
        );
        assert!(
            neo_ci.0 <= truth && truth <= neo_ci.1,
            "{name}: neo rate {neo}/{SHOTS} excludes the analytic value {truth}"
        );
    }
}

#[test]
fn meas_only_rates_match() {
    // Measurement flip only: P(c = 0 after X) = p_meas exactly.
    check_cell(
        "meas_only",
        X_MEASURE,
        &NoiseCell::Meas(0.2),
        &["0"],
        Some(0.2),
    );
}

#[test]
fn prep_only_rates_match() {
    // Preparation error only: P(c = 1 after reset) = p_prep exactly.
    check_cell(
        "prep_only",
        RESET_MEASURE,
        &NoiseCell::Prep(0.15),
        &["1"],
        Some(0.15),
    );
}

#[test]
fn p1_only_rates_match() {
    // Uniform 1q depolarizing after the X gate: X and Y flip the Z-basis
    // outcome, Z does not, so P(c = 0) = 2p/3 exactly.
    check_cell("p1_only", X_MEASURE, &NoiseCell::P1(0.3), &["0"], Some(0.2));
}

#[test]
fn p2_only_anticorrelation_matches() {
    // Uniform 2q depolarizing after the Bell CX: of the 15 two-qubit
    // Paulis, the 8 with exactly one X/Y factor anticommute with Z(x)Z
    // and break the outcome correlation, so P(01 or 10) = 8p/15 exactly.
    check_cell(
        "p2_only",
        BELL,
        &NoiseCell::P2(0.3),
        &["01", "10"],
        Some(8.0 * 0.3 / 15.0),
    );
}

#[test]
fn uniform_depolarizing_compound_matches() {
    // All channels at p = 0.1 on the x-measure program: the compound
    // error rate has no simple closed form; cross-stack agreement only.
    check_cell(
        "uniform_depol",
        X_MEASURE,
        &NoiseCell::Uniform(0.1),
        &["0"],
        None,
    );
}

#[test]
fn gnm_simple_subset_matches() {
    // GeneralNoiseModel's "average" convention stores p1 = 1.5 x average
    // (0.3 here, flip 0.2) on both stacks, composed with a 5% measurement
    // flip: P(c = 0) = 0.2 * 0.95 + 0.8 * 0.05 = 0.23 exactly.
    check_cell(
        "gnm_simple",
        X_MEASURE,
        &NoiseCell::GnmSimple {
            average_p1: 0.2,
            p_meas: 0.05,
        },
        &["0"],
        Some(0.23),
    );
}

#[test]
fn feedback_under_measurement_noise_matches() {
    // Conditional feedback with noisy measurement: the recorded c[0]
    // drives the correction, so both stacks must apply the measurement
    // flip with the same record-vs-state semantics to agree here.
    check_cell(
        "feedback_meas_noise",
        FEEDBACK,
        &NoiseCell::Meas(0.1),
        &["11"],
        None,
    );
}

#[test]
fn meas_twice_without_reset_matches() {
    // Measurement noise in the depolarizing family is a physical X
    // injected before readout, so the error persists in the state: the
    // SECOND measurement of an un-reset qubit flips at 2p(1-p), not p.
    // The creg bit is overwritten, so c reads the second outcome. A
    // record-flip mapping (the bug this cell guards against) would
    // produce p here.
    let p = 0.25;
    check_cell(
        "meas_twice",
        MEASURE_TWICE,
        &NoiseCell::Meas(p),
        &["1"],
        Some(2.0 * p * (1.0 - p)),
    );
}

#[test]
fn meas_twice_gnm_is_record_flip_not_state_flip() {
    // The COMPLEMENT of `meas_twice_without_reset_matches`, locking the
    // OTHER engines measurement convention. GeneralNoiseModel readout error
    // flips only the recorded outcome, never the post-measurement state, so
    // the qubit stays |0> across both measurements and the SECOND outcome
    // flips at exactly p (record flip) — NOT 2p(1-p) (state flip). Both
    // stacks must agree (engines GNM record-flip maps to neo's record-
    // flipping MeasurementChannel). Together with the depolarizing cell
    // above, this pins the engines depolarizing-vs-GNM measurement-physics
    // distinction (the B1 root cause) on BOTH sides, cross-stack.
    let p = 0.25;
    check_cell(
        "meas_twice_gnm",
        MEASURE_TWICE,
        &NoiseCell::GnmSimple {
            average_p1: 0.0,
            p_meas: p,
        },
        &["1"],
        Some(p),
    );
}

/// One `GnmAngle` configuration shared by both angle cells: stored p2 = 0.3,
/// asymmetric coefficients with the NEGATIVE branch (a = 1.5) steeper than the
/// POSITIVE branch (c = 1.0), linear power. The facade translates engines'
/// `(a, b, c, d, power)` into neo's asymmetric `AngleScaling`; both cells pin
/// the cross-stack rate AND the analytic value, so a dropped or neg/pos-swapped
/// mapping fails loudly.
const ANGLE_CELL: NoiseCell = NoiseCell::GnmAngle {
    p2: 0.3,
    angle_params: (1.5, 0.0, 1.0, 0.0),
    angle_power: 1.0,
};

#[test]
fn gnm_angle_scaling_positive_matches() {
    // RZZ(+pi/2): the POSITIVE branch scales p2 by c*|theta/pi|^power + d =
    // 1.0*0.5 + 0 = 0.5, so the effective p2 is 0.3*0.5 = 0.15 and the 8/15
    // parity-flipping Paulis give P(01 or 10) = 8*0.15/15 = 0.08. Dropping the
    // angle scaling (the pre-mapping behavior) would give the unscaled
    // 8*0.3/15 = 0.16, and the neg/pos-swapped mapping would give 0.12 — both
    // far outside the band, so this discriminates.
    check_cell(
        "gnm_angle_pos",
        RZZ_POS,
        &ANGLE_CELL,
        &["01", "10"],
        Some(8.0 * (0.3 * 0.5) / 15.0),
    );
}

#[test]
fn gnm_angle_scaling_negative_matches() {
    // RZZ(-pi/2): the NEGATIVE branch scales p2 by a*|theta/pi|^power + b =
    // 1.5*0.5 + 0 = 0.75, so the effective p2 is 0.3*0.75 = 0.225 and the rate
    // is 8*0.225/15 = 0.12. Both stacks read the gate angle as the SIGNED
    // principal value (-pi, pi] -- neo always did; engines' noise call site was
    // aligned with its own gate unitaries (which all use `to_radians_signed`),
    // fixing a bug where the unsigned [0, 2pi) angle made RZZ(-pi/2) take the
    // POSITIVE branch as 3pi/2 and never reached the negative coefficients. So
    // the stacks AGREE here, and the value differs from the positive cell
    // (0.08), locking the asymmetry direction. A regression to the unsigned
    // angle would push engines to 8*0.45/15 = 0.24 and fail the cross-stack
    // overlap and the analytic containment.
    check_cell(
        "gnm_angle_neg",
        RZZ_NEG,
        &ANGLE_CELL,
        &["01", "10"],
        Some(8.0 * (0.3 * 0.75) / 15.0),
    );
}

#[test]
fn biased_meas_flip_1_to_0_matches() {
    // BiasedDepolarizing flips the RECORDED outcome after readout. A |1> (from
    // X, with gate noise zeroed) reads 0 with probability p_meas_1 (the 1->0
    // rate). The bias is asymmetric (p_meas_0 = 0.1 != p_meas_1 = 0.3), so a
    // swapped 0<->1 mapping would read 0.1 here instead of 0.3.
    check_cell(
        "biased_meas_1to0",
        X_MEASURE,
        &NoiseCell::BiasedMeas {
            p_meas_0: 0.1,
            p_meas_1: 0.3,
        },
        &["0"],
        Some(0.3),
    );
}

#[test]
fn biased_meas_flip_0_to_1_matches() {
    // The complementary direction: a |0> (from reset) reads 1 with probability
    // p_meas_0 (the 0->1 rate, 0.1). Together with the cell above this pins
    // both the magnitude AND the direction of the asymmetric bias.
    check_cell(
        "biased_meas_0to1",
        RESET_MEASURE,
        &NoiseCell::BiasedMeas {
            p_meas_0: 0.1,
            p_meas_1: 0.3,
        },
        &["1"],
        Some(0.1),
    );
}

#[test]
fn biased_meas_twice_is_record_flip_not_state_flip() {
    // Like meas_twice_gnm: BiasedDepolarizing flips the record, never the
    // state, so a qubit reset to |0> and measured twice reads 1 on the SECOND
    // measurement at exactly p_meas_0 = 0.25 -- NOT 2p(1-p) = 0.375 (state
    // flip). This pins that BiasedDepolarizing maps to neo's record-flip
    // channel (with_p_meas), not the state-flip channel the plain
    // depolarizing family uses.
    let p = 0.25;
    check_cell(
        "biased_meas_twice",
        MEASURE_TWICE,
        &NoiseCell::BiasedMeas {
            p_meas_0: p,
            p_meas_1: p,
        },
        &["1"],
        Some(p),
    );
}
