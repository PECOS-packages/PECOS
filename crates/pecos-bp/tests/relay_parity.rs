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

//! Parity of the native Relay-BP against the external `relay-bp` crate
//! (through the `pecos-relay-bp` wrapper, a dev-dependency here) on the
//! [[144,12,12]] gross-code circuit-level DEM shipped in `tests/data`.
//!
//! The two implementations share the update rule, the flooding schedule, the
//! memory term, and the relay structure, but not floating-point summation
//! order, so per-shot agreement is high but not bitwise. Three contracts:
//!
//! 1. Leg 0 only (no randomness on either side), at iteration budgets small
//!    enough that some shots fail: per-shot convergence flags and corrections
//!    must agree on almost every shot, and at least one budget must sit in a
//!    mixed regime so the comparison discriminates.
//! 2. Relayed legs with one explicit gamma vector shared by both decoders:
//!    the same per-shot agreement, now exercising the memory term with
//!    signed gammas and the warm start.
//! 3. Relayed legs with each side's own random gammas: convergence counts
//!    agree within a margin, and every native leg demonstrably ran.
//!
//! The full operating point is kept as an end-to-end smoke test.

use pecos_bp::BpGraph;
use pecos_bp::relay::{RelayBp, RelayConfig, Schedule};
use pecos_decoder_core::dem::DemCheckMatrix;
use pecos_relay_bp::{RelayBpBuilder, RelayBpDecoder, StoppingCriterion};

/// One fixture shot: `(detector bits, true observable flips)`.
type Shot = (Vec<u8>, Vec<u8>);

const SHOTS: usize = 100;
/// Per-shot agreement floor for the deterministic comparisons.
const MIN_AGREEMENT: usize = 98;

fn load_fixture() -> (DemCheckMatrix, Vec<Shot>) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/");
    let dem = std::fs::read_to_string(format!("{dir}gross_144_12_12.dem")).unwrap();
    let dcm = DemCheckMatrix::from_dem_str(&dem).unwrap();
    let shots = std::fs::read_to_string(format!("{dir}gross_144_12_12_shots.txt")).unwrap();
    let parse = |s: &str| s.bytes().map(|b| b - b'0').collect::<Vec<u8>>();
    let shots = shots
        .lines()
        .map(|line| {
            let (d, o) = line.split_once(' ').unwrap();
            (parse(d), parse(o))
        })
        .collect::<Vec<_>>();
    assert_eq!(dcm.num_detectors, 1008);
    assert_eq!(dcm.num_mechanisms, 8785);
    assert_eq!(dcm.num_observables, 12);
    assert_eq!(shots.len(), SHOTS);
    (dcm, shots)
}

fn observables(dcm: &DemCheckMatrix, correction: &[u8]) -> Vec<u8> {
    (0..dcm.num_observables)
        .map(|o| {
            (0..dcm.num_mechanisms).fold(0u8, |acc, v| {
                acc ^ (dcm.observable_matrix[[o, v]] & correction[v])
            })
        })
        .collect()
}

/// Deterministic gamma vector in `[-0.24, 0.66)` (no RNG dependency).
fn shared_gamma_vector(n: usize, mut state: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = f64::from(u32::try_from(state >> 32).unwrap()) / 4_294_967_296.0;
            -0.24 + 0.90 * u
        })
        .collect()
}

struct Tally {
    native_converged: usize,
    external_converged: usize,
    native_errors: usize,
    external_errors: usize,
    flag_agreement: usize,
    both_converged: usize,
    same_correction: usize,
    native_legs_ran: Vec<usize>,
    external_max_iterations: usize,
}

fn compare(
    dcm: &DemCheckMatrix,
    shots: &[Shot],
    native: &mut RelayBp,
    external: &mut RelayBpDecoder,
) -> Tally {
    let mut t = Tally {
        native_converged: 0,
        external_converged: 0,
        native_errors: 0,
        external_errors: 0,
        flag_agreement: 0,
        both_converged: 0,
        same_correction: 0,
        native_legs_ran: Vec::new(),
        external_max_iterations: 0,
    };
    for (shot, (syndrome, truth)) in shots.iter().enumerate() {
        let n = native
            .decode(syndrome, u64::try_from(shot).unwrap())
            .unwrap();
        let e = external
            .decode(&ndarray::ArrayView1::from(syndrome.as_slice()))
            .unwrap();
        let e_corr: Vec<u8> = e.decoding.iter().copied().collect();

        t.native_converged += usize::from(n.converged);
        t.external_converged += usize::from(e.converged);
        t.native_errors += usize::from(observables(dcm, &n.correction) != *truth);
        t.external_errors += usize::from(observables(dcm, &e_corr) != *truth);
        t.flag_agreement += usize::from(n.converged == e.converged);
        if n.converged && e.converged {
            t.both_converged += 1;
            t.same_correction += usize::from(n.correction == e_corr);
        }
        t.native_legs_ran.push(n.legs.len());
        t.external_max_iterations = t.external_max_iterations.max(e.iterations);
    }
    println!(
        "native: converged {}/{SHOTS}, logical errors {}/{SHOTS}; external: converged {}/{SHOTS}, \
         logical errors {}/{SHOTS}; flag agreement {}/{SHOTS}; same correction {}/{} jointly \
         converged",
        t.native_converged,
        t.native_errors,
        t.external_converged,
        t.external_errors,
        t.flag_agreement,
        t.same_correction,
        t.both_converged
    );
    t
}

fn assert_per_shot_agreement(t: &Tally, label: &str) {
    assert!(
        t.flag_agreement >= MIN_AGREEMENT,
        "{label}: convergence flags agree on only {} of {SHOTS} shots",
        t.flag_agreement
    );
    // Almost every shot one side converged on, the other did too.
    assert!(
        t.both_converged + 2 >= t.native_converged.min(t.external_converged),
        "{label}: only {} jointly converged shots",
        t.both_converged
    );
    // Corrections identical on almost every jointly converged shot.
    assert!(
        t.same_correction * 20 >= t.both_converged * 19,
        "{label}: identical corrections on only {} of {} jointly converged shots",
        t.same_correction,
        t.both_converged
    );
}

#[test]
fn leg_zero_agrees_per_shot_across_iteration_budgets() {
    let (dcm, shots) = load_fixture();
    let graph = BpGraph::from_dcm(&dcm);
    let gamma0 = 0.1;
    let mut mixed_regime_seen = false;
    for pre_iterations in [2, 4, 8, 16] {
        let cfg = RelayConfig {
            schedule: Schedule::Flooding,
            alpha: 1.0,
            gamma0,
            pre_iterations,
            num_legs: 0,
            leg_iterations: 1,
            gamma_range: (-0.24, 0.66),
            stop_after_converged: 1,
            explicit_gammas: None,
        };
        let mut native = RelayBp::new(graph.clone(), cfg).unwrap();
        let mut external = RelayBpBuilder::new(&dcm.check_matrix.view())
            .error_priors(&dcm.error_priors)
            .alpha(Some(1.0))
            .gamma0(Some(gamma0))
            .pre_iter(pre_iterations)
            .num_sets(1)
            .set_max_iter(1)
            .stopping_criterion(StoppingCriterion::PreIter)
            .seed(7)
            .build()
            .unwrap();
        println!("pre_iterations = {pre_iterations}");
        let t = compare(&dcm, &shots, &mut native, &mut external);
        assert_per_shot_agreement(&t, &format!("pre_iterations={pre_iterations}"));
        assert!(t.native_legs_ran.iter().all(|&legs| legs == 1));
        if (5..=95).contains(&t.native_converged) {
            mixed_regime_seen = true;
        }
    }
    assert!(
        mixed_regime_seen,
        "no iteration budget produced a mixed success regime"
    );
}

fn shared_explicit_gamma_case(rows: Vec<Vec<f64>>, label: &str) {
    let (dcm, shots) = load_fixture();
    let graph = BpGraph::from_dcm(&dcm);
    for row in &rows {
        assert!(row.iter().any(|&g| g < 0.0) && row.iter().any(|&g| g > 0.0));
    }
    let (pre_iterations, num_legs, leg_iterations, gamma0) = (2, 4, 10, 0.1);

    let cfg = RelayConfig {
        schedule: Schedule::Flooding,
        alpha: 1.0,
        gamma0,
        pre_iterations,
        num_legs,
        leg_iterations,
        gamma_range: (-0.24, 0.66),
        stop_after_converged: usize::MAX,
        explicit_gammas: Some(rows.clone()),
    };
    let mut native = RelayBp::new(graph, cfg).unwrap();
    let mut external = RelayBpBuilder::new(&dcm.check_matrix.view())
        .error_priors(&dcm.error_priors)
        .alpha(Some(1.0))
        .gamma0(Some(gamma0))
        .pre_iter(pre_iterations)
        .num_sets(num_legs)
        .set_max_iter(leg_iterations)
        .explicit_gammas(rows)
        .stopping_criterion(StoppingCriterion::All)
        .seed(7)
        .build()
        .unwrap();
    let t = compare(&dcm, &shots, &mut native, &mut external);
    assert_per_shot_agreement(&t, label);
    assert!(t.native_legs_ran.iter().all(|&legs| legs == num_legs + 1));
    assert!(
        t.external_max_iterations > pre_iterations,
        "the external decoder never ran a relay leg"
    );
    assert!(
        t.native_converged >= 5,
        "relayed legs converged on only {} shots",
        t.native_converged
    );
}

#[test]
fn relayed_legs_with_one_shared_explicit_gamma_row_agree_per_shot() {
    let rows = vec![shared_gamma_vector(8785, 0x00C0_FFEE)];
    shared_explicit_gamma_case(rows, "one shared gamma row");
}

#[test]
fn relayed_legs_with_two_shared_explicit_gamma_rows_agree_per_shot() {
    // Two distinct rows: leg order (row 0 first, then cycling) must match on
    // both sides, which the one-row case cannot detect.
    let rows = vec![
        shared_gamma_vector(8785, 0x00C0_FFEE),
        shared_gamma_vector(8785, 0x0BAD_F00D),
    ];
    shared_explicit_gamma_case(rows, "two shared gamma rows");
}

#[test]
fn relayed_legs_with_independent_random_gammas_agree_in_aggregate() {
    let (dcm, shots) = load_fixture();
    let graph = BpGraph::from_dcm(&dcm);
    let (pre_iterations, num_legs, leg_iterations, gamma0) = (2, 8, 10, 0.1);
    let cfg = RelayConfig {
        schedule: Schedule::Flooding,
        alpha: 1.0,
        gamma0,
        pre_iterations,
        num_legs,
        leg_iterations,
        gamma_range: (-0.24, 0.66),
        stop_after_converged: usize::MAX,
        explicit_gammas: None,
    };
    let mut native = RelayBp::new(graph, cfg).unwrap();
    let mut external = RelayBpBuilder::new(&dcm.check_matrix.view())
        .error_priors(&dcm.error_priors)
        .alpha(Some(1.0))
        .gamma0(Some(gamma0))
        .pre_iter(pre_iterations)
        .num_sets(num_legs)
        .set_max_iter(leg_iterations)
        .gamma_dist_interval((-0.24, 0.66))
        .stopping_criterion(StoppingCriterion::All)
        .seed(7)
        .build()
        .unwrap();
    let t = compare(&dcm, &shots, &mut native, &mut external);
    assert!(t.native_legs_ran.iter().all(|&legs| legs == num_legs + 1));
    assert!(t.external_max_iterations > pre_iterations);
    let gap = t.native_converged.abs_diff(t.external_converged);
    assert!(gap <= 5, "convergence counts differ by {gap} shots");
}

#[test]
fn full_operating_point_smoke() {
    let (dcm, shots) = load_fixture();
    let graph = BpGraph::from_dcm(&dcm);
    let cfg = RelayConfig::default();
    let (pre_iterations, num_legs, leg_iterations, gamma0, gamma_range) = (
        cfg.pre_iterations,
        cfg.num_legs,
        cfg.leg_iterations,
        cfg.gamma0,
        cfg.gamma_range,
    );
    let mut native = RelayBp::new(graph, cfg).unwrap();
    let mut external = RelayBpBuilder::new(&dcm.check_matrix.view())
        .error_priors(&dcm.error_priors)
        .alpha(Some(1.0))
        .gamma0(Some(gamma0))
        .pre_iter(pre_iterations)
        .num_sets(num_legs)
        .set_max_iter(leg_iterations)
        .gamma_dist_interval(gamma_range)
        .stopping_criterion(StoppingCriterion::NConv { stop_after: 1 })
        .seed(7)
        .build()
        .unwrap();
    let t = compare(&dcm, &shots, &mut native, &mut external);
    assert!(t.native_converged + 5 >= t.external_converged);
    assert!(t.native_errors <= t.external_errors + 5);
    assert!(t.native_converged >= 80);
}
