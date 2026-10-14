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

//! Dense reference oracle for the native Relay-BP kernel.
//!
//! The reference below is a deliberately naive, dense, allocation-heavy
//! implementation of memory-BP with relay legs. It is the specification of
//! the algorithm: the sparse kernel in `pecos_bp::relay` must reproduce its
//! posteriors bit for bit on toy graphs, for both schedules, with and without
//! memory, across several relayed legs with explicit per-variable gammas.
//!
//! Algorithm (log-likelihood ratios, positive means "no error"):
//!
//! - `prior[v] = ln((1 - p_v) / p_v)` clamped to `[-30, 30]`, exactly
//!   `BpGraph::prior_llrs`.
//! - Leg 0 starts with posterior `P = prior`, `gamma[v] = gamma0` for every
//!   variable. Leg `r >= 1` keeps `P` from the end of leg `r - 1` and draws a
//!   fresh `gamma[v]` per variable (explicit here, cycling through the
//!   supplied vectors).
//! - Every leg starts message passing from scratch: `c2v = 0` and
//!   `v2c[v -> c] = prior[v]`. The memory bias
//!   `bias[v] = prior[v] + gamma[v] * (P[v] - prior[v])` enters at the
//!   first variable pass of the leg, computed from the warm-started `P`.
//! - One flooding iteration: (1) every check computes its outgoing min-sum
//!   messages from the current `v2c`, scaled by `alpha`; (2) every variable
//!   computes `bias[v]` from the posterior of the previous iteration, then
//!   `v2c[v -> c] = bias[v] + sum_{c' != c} c2v[c' -> v]` and
//!   `P[v] = bias[v] + sum_c c2v[c -> v]`.
//! - One check-serial iteration: `bias[v]` is computed once for every
//!   variable from the previous iteration's posterior; then for each check in
//!   index order, (1) that check computes its outgoing messages from the
//!   current `v2c`, and (2) each variable it touches immediately recomputes
//!   `P[v] = bias[v] + sum_c c2v[c -> v]` and `v2c[v -> c'] = P[v] - c2v[c' -> v]`
//!   for all of its checks.
//! - A check with a single neighbour has an empty exclusive minimum. Its
//!   message magnitude is `LLR_SATURATION` (30, the certainty bound priors
//!   are clamped to): certainty-strength evidence that competes like any
//!   other and can tie against a zero-probability prior. No message is ever
//!   infinite, and the syndrome check below guarantees an unsatisfied
//!   correction is never reported as converged.
//! - After every iteration the hard decision is `e[v] = (P[v] < 0)`. A leg
//!   stops early when `H e == s`; the leg is then "converged" and its weight
//!   is the signed prior-LLR cost `sum_v e[v] * prior[v]` (lower is more
//!   likely; a mechanism with probability above 0.5 lowers the cost).
//! - The relay stops after the configured number of converged legs, or when
//!   all legs ran. The reported correction is the minimum-weight converged
//!   correction (first one on ties), or the last hard decision if none.

use pecos_bp::relay::{RelayBp, RelayConfig, Schedule};
use pecos_bp::{BpGraph, LLR_SATURATION};
use pecos_decoder_core::dem::DemCheckMatrix;

const TOY_DEMS: [&str; 5] = [
    // Small graph with a degree-3 check and a boundary-like weight-1 column.
    "error(0.05) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.02) D0 D2 L1\nerror(0.07) D2 D3\nerror(0.2) D3\nerror(0.03) D0 D1 D3\n",
    // Repetition-code-like chain with a hyperedge.
    "error(0.01) D0\nerror(0.02) D0 D1\nerror(0.03) D1 D2\nerror(0.04) D2 D3\nerror(0.05) D3 D4\nerror(0.06) D4\nerror(0.015) D0 D2 D4 L0\nerror(0.025) D1 D3 L0\n",
    // Denser graph with two observables.
    "error(0.1) D0 D1 D2\nerror(0.1) D1 D2 D3\nerror(0.1) D2 D3 D4\nerror(0.1) D0 D3 D4 L0\nerror(0.05) D0 D4 L1\nerror(0.05) D1 D4\nerror(0.05) D0 D2 L0 L1\nerror(0.2) D3\nerror(0.2) D1\n",
    // A degree-1 check: D3 is touched by mechanism 2 only.
    "error(0.05) D0 D1\nerror(0.1) D1 D2 L0\nerror(0.02) D2 D3\nerror(0.2) D0\nerror(0.03) D0 D2\n",
    // Priors above 0.5 (negative LLRs): the signed weight must prefer
    // selecting them, where an absolute weight would penalise them.
    "error(0.9) D0 D1\nerror(0.7) D1 D2 L0\nerror(0.05) D0 D2\nerror(0.6) D2\nerror(0.1) D0\n",
];

/// Index of the toy with the degree-1 check, and the mechanism that pins it.
const DEGREE_ONE_TOY: usize = 3;
const DEGREE_ONE_MECHANISM: usize = 2;

fn sign(x: f64) -> f64 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// Dense reference state for one shot.
struct Reference {
    checks: Vec<Vec<usize>>, // check -> variables
    vars: Vec<Vec<usize>>,   // variable -> checks
    prior: Vec<f64>,
    posterior: Vec<f64>,
    gamma: Vec<f64>,
    bias: Vec<f64>,
    c2v: Vec<Vec<f64>>, // [check][variable] (dense, zero off-graph)
    v2c: Vec<Vec<f64>>, // [variable][check]
}

impl Reference {
    fn new(dcm: &DemCheckMatrix, prior: &[f64]) -> Self {
        let m = dcm.num_detectors;
        let n = dcm.num_mechanisms;
        let mut checks = vec![Vec::new(); m];
        let mut vars = vec![Vec::new(); n];
        for ((c, v), &entry) in dcm.check_matrix.indexed_iter() {
            if entry != 0 {
                checks[c].push(v);
                vars[v].push(c);
            }
        }
        Self {
            checks,
            vars,
            prior: prior.to_vec(),
            posterior: prior.to_vec(),
            gamma: vec![0.0; n],
            bias: vec![0.0; n],
            c2v: vec![vec![0.0; n]; m],
            v2c: vec![vec![0.0; m]; n],
        }
    }

    fn start_leg(&mut self, gamma: &[f64]) {
        self.gamma.copy_from_slice(gamma);
        for row in &mut self.c2v {
            row.fill(0.0);
        }
        for v in 0..self.vars.len() {
            for &c in &self.vars[v] {
                self.v2c[v][c] = self.prior[v];
            }
        }
    }

    fn compute_bias(&mut self) {
        for v in 0..self.vars.len() {
            self.bias[v] = self.prior[v] + self.gamma[v] * (self.posterior[v] - self.prior[v]);
        }
    }

    fn check_update(&mut self, c: usize, syndrome: &[u8], alpha: f64) {
        let mut total_sign = if syndrome[c] != 0 { -1.0 } else { 1.0 };
        let mut min1 = f64::INFINITY;
        let mut min2 = f64::INFINITY;
        let mut min1_var = usize::MAX;
        for &v in &self.checks[c] {
            let msg = self.v2c[v][c];
            total_sign *= sign(msg);
            let mag = msg.abs();
            if mag < min1 {
                min2 = min1;
                min1 = mag;
                min1_var = v;
            } else if mag < min2 {
                min2 = mag;
            }
        }
        for &v in &self.checks[c] {
            let msg = self.v2c[v][c];
            let excl_sign = total_sign * sign(msg);
            let excl_min = if v == min1_var { min2 } else { min1 };
            // Empty exclusive minimum (single-neighbour check): saturate.
            let excl_min = if excl_min.is_finite() {
                excl_min
            } else {
                LLR_SATURATION
            };
            self.c2v[c][v] = alpha * excl_sign * excl_min;
        }
    }

    fn variable_update(&mut self, v: usize) {
        let total: f64 = self.vars[v].iter().map(|&c| self.c2v[c][v]).sum();
        for &c in &self.vars[v] {
            self.v2c[v][c] = self.bias[v] + total - self.c2v[c][v];
        }
        self.posterior[v] = self.bias[v] + total;
    }

    fn hard_decision(&self) -> Vec<u8> {
        self.posterior.iter().map(|&p| u8::from(p < 0.0)).collect()
    }

    fn satisfied(&self, e: &[u8], syndrome: &[u8]) -> bool {
        self.checks.iter().enumerate().all(|(c, vs)| {
            let parity = vs.iter().fold(0u8, |acc, &v| acc ^ e[v]);
            parity == syndrome[c]
        })
    }

    /// Run one leg. Returns `(converged, iterations_used)`.
    fn run_leg(
        &mut self,
        syndrome: &[u8],
        iterations: usize,
        alpha: f64,
        schedule: Schedule,
    ) -> (bool, usize) {
        for it in 1..=iterations {
            match schedule {
                Schedule::Flooding => {
                    for c in 0..self.checks.len() {
                        self.check_update(c, syndrome, alpha);
                    }
                    self.compute_bias();
                    for v in 0..self.vars.len() {
                        self.variable_update(v);
                    }
                }
                Schedule::CheckSerial => {
                    self.compute_bias();
                    for c in 0..self.checks.len() {
                        self.check_update(c, syndrome, alpha);
                        let touched = self.checks[c].clone();
                        for v in touched {
                            self.variable_update(v);
                        }
                    }
                }
            }
            let e = self.hard_decision();
            if self.satisfied(&e, syndrome) {
                return (true, it);
            }
        }
        (false, iterations)
    }

    fn weight(&self, e: &[u8]) -> f64 {
        e.iter()
            .zip(&self.prior)
            .map(|(&b, &p)| f64::from(b) * p)
            .sum()
    }
}

/// Reference relay run. `gammas[r % len]` is the per-variable gamma vector
/// of leg `r + 1`; leg 0 uses `gamma0` everywhere.
struct ReferenceOutcome {
    converged: bool,
    correction: Vec<u8>,
    weight: Option<f64>,
    posterior: Vec<f64>,
    legs_converged: Vec<bool>,
    legs_iterations: Vec<usize>,
}

fn reference_relay(
    dcm: &DemCheckMatrix,
    prior: &[f64],
    syndrome: &[u8],
    cfg: &RelayConfig,
    gammas: &[Vec<f64>],
) -> ReferenceOutcome {
    let n = dcm.num_mechanisms;
    let mut r = Reference::new(dcm, prior);
    let mut legs_converged = Vec::new();
    let mut legs_iterations = Vec::new();
    let mut best: Option<(f64, Vec<u8>)> = None;
    let mut num_converged = 0;

    let leg0 = vec![cfg.gamma0; n];
    r.start_leg(&leg0);
    let (conv, its) = r.run_leg(syndrome, cfg.pre_iterations, cfg.alpha, cfg.schedule);
    legs_converged.push(conv);
    legs_iterations.push(its);
    if conv {
        num_converged += 1;
        let e = r.hard_decision();
        best = Some((r.weight(&e), e));
    }
    let mut leg = 0;
    while num_converged < cfg.stop_after_converged && leg < cfg.num_legs {
        r.start_leg(&gammas[leg % gammas.len()]);
        let (conv, its) = r.run_leg(syndrome, cfg.leg_iterations, cfg.alpha, cfg.schedule);
        legs_converged.push(conv);
        legs_iterations.push(its);
        if conv {
            num_converged += 1;
            let e = r.hard_decision();
            let w = r.weight(&e);
            if best.as_ref().is_none_or(|(bw, _)| w < *bw) {
                best = Some((w, e));
            }
        }
        leg += 1;
    }
    let (converged, correction, weight) = if let Some((w, e)) = best {
        (true, e, Some(w))
    } else {
        let e = r.hard_decision();
        (false, e, None)
    };
    ReferenceOutcome {
        converged,
        correction,
        weight,
        posterior: r.posterior.clone(),
        legs_converged,
        legs_iterations,
    }
}

/// Deterministic pseudo-random helpers for the fixtures (no RNG dependency).
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 11
}

/// Uniform in `[0, 1)` from the top 32 of the 53 bits `lcg` returns.
fn unit(state: &mut u64) -> f64 {
    f64::from(u32::try_from(lcg(state) >> 21).unwrap()) / 4_294_967_296.0
}

fn syndrome_of(dcm: &DemCheckMatrix, e: &[u8]) -> Vec<u8> {
    (0..dcm.num_detectors)
        .map(|c| {
            (0..dcm.num_mechanisms).fold(0u8, |acc, v| acc ^ (dcm.check_matrix[[c, v]] & e[v]))
        })
        .collect()
}

fn random_syndrome(dcm: &DemCheckMatrix, state: &mut u64) -> Vec<u8> {
    let e: Vec<u8> = (0..dcm.num_mechanisms)
        .map(|_| u8::from(unit(state) < 0.3))
        .collect();
    syndrome_of(dcm, &e)
}

fn explicit_gammas(n: usize, sets: usize, range: (f64, f64), state: &mut u64) -> Vec<Vec<f64>> {
    (0..sets)
        .map(|_| {
            (0..n)
                .map(|_| range.0 + (range.1 - range.0) * unit(state))
                .collect()
        })
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn fixture_sampler_covers_its_ranges() {
    let mut state = 12345u64;
    let draws: Vec<f64> = (0..2000).map(|_| unit(&mut state)).collect();
    assert!(draws.iter().all(|&u| (0.0..1.0).contains(&u)));
    assert!(draws.iter().any(|&u| u < 0.05));
    assert!(draws.iter().any(|&u| u > 0.95));
    let below = draws.iter().filter(|&&u| u < 0.3).count();
    assert!((400..800).contains(&below), "p=0.3 draws: {below} of 2000");
}

struct Coverage {
    strict_early_stop: bool,
    negative_gamma: bool,
    positive_gamma: bool,
    distinct_syndromes: usize,
}

fn check_against_reference(
    schedule: Schedule,
    gamma0: f64,
    num_legs: usize,
    gamma_sets: usize,
    stop_after: usize,
    alpha: f64,
) -> Coverage {
    let mut state =
        0x5eed_u64 ^ (u64::try_from(num_legs).unwrap() << 8) ^ u64::try_from(stop_after).unwrap();
    let mut coverage = Coverage {
        strict_early_stop: false,
        negative_gamma: false,
        positive_gamma: false,
        distinct_syndromes: 0,
    };
    for dem in TOY_DEMS {
        let dcm = DemCheckMatrix::from_dem_str(dem).unwrap();
        let graph = BpGraph::from_dcm(&dcm);
        let prior = graph.prior_llrs().to_vec();
        let n = dcm.num_mechanisms;
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..6 {
            let syndrome = random_syndrome(&dcm, &mut state);
            seen.insert(syndrome.clone());
            let gammas = explicit_gammas(n, gamma_sets, (-0.24, 0.66), &mut state);
            for g in gammas.iter().flatten() {
                coverage.negative_gamma |= *g < 0.0;
                coverage.positive_gamma |= *g > 0.0;
            }
            let cfg = RelayConfig {
                schedule,
                alpha,
                gamma0,
                pre_iterations: 7,
                num_legs,
                leg_iterations: 5,
                gamma_range: (-0.24, 0.66),
                stop_after_converged: stop_after,
                explicit_gammas: Some(gammas.clone()),
            };
            let expected = reference_relay(&dcm, &prior, &syndrome, &cfg, &gammas);

            let mut relay = RelayBp::new(graph.clone(), cfg).unwrap();
            let got = relay.decode(&syndrome, 0).unwrap();

            assert_eq!(
                got.converged, expected.converged,
                "converged flag ({dem:?}, {syndrome:?})"
            );
            assert_eq!(
                got.correction, expected.correction,
                "correction ({dem:?}, {syndrome:?})"
            );
            assert_eq!(
                got.weight.map(f64::to_bits),
                expected.weight.map(f64::to_bits),
                "weight ({dem:?}, {syndrome:?})"
            );
            assert_eq!(
                bits(&got.posterior),
                bits(&expected.posterior),
                "final posterior ({dem:?}, {syndrome:?})"
            );
            assert!(
                got.posterior.iter().all(|p| p.is_finite()),
                "posterior must stay finite ({dem:?}, {syndrome:?})"
            );
            let got_conv: Vec<bool> = got.legs.iter().map(|l| l.converged).collect();
            let got_its: Vec<usize> = got.legs.iter().map(|l| l.iterations).collect();
            assert_eq!(
                got_conv, expected.legs_converged,
                "per-leg convergence ({dem:?}, {syndrome:?})"
            );
            assert_eq!(
                got_its, expected.legs_iterations,
                "per-leg iterations ({dem:?}, {syndrome:?})"
            );
            assert_eq!(
                got.total_iterations,
                expected.legs_iterations.iter().sum::<usize>()
            );
            for (leg_index, leg) in got.legs.iter().enumerate() {
                let limit = if leg_index == 0 { 7 } else { 5 };
                assert!(leg.iterations >= 1 && leg.iterations <= limit);
                assert_eq!(leg.converged, leg.correction.is_some());
                assert_eq!(leg.converged, leg.weight.is_some());
                if leg.converged && leg.iterations > 1 && leg.iterations < limit {
                    coverage.strict_early_stop = true;
                }
            }
            if got.converged {
                assert_eq!(
                    syndrome_of(&dcm, &got.correction),
                    syndrome,
                    "converged correction must satisfy the syndrome"
                );
            }
        }
        coverage.distinct_syndromes += seen.len();
    }
    assert!(
        coverage.distinct_syndromes >= 12,
        "the random syndromes collapsed: {} distinct over 24 draws",
        coverage.distinct_syndromes
    );
    coverage
}

#[test]
fn flooding_without_memory_matches_reference() {
    // gamma0 = 0 and no relay legs: plain normalized min-sum with early stop.
    let cov = check_against_reference(Schedule::Flooding, 0.0, 0, 1, 1, 0.8);
    assert!(
        cov.strict_early_stop,
        "no leg stopped strictly between 1 and its limit"
    );
}

#[test]
fn check_serial_without_memory_matches_reference() {
    check_against_reference(Schedule::CheckSerial, 0.0, 0, 1, 1, 0.8);
}

#[test]
fn flooding_memory_leg_matches_reference() {
    check_against_reference(Schedule::Flooding, 0.1, 0, 1, 1, 1.0);
}

#[test]
fn flooding_relay_legs_match_reference() {
    // Enough legs and a high stop count so every leg runs and warm-starts.
    let cov = check_against_reference(Schedule::Flooding, 0.1, 4, 4, usize::MAX, 1.0);
    assert!(
        cov.negative_gamma && cov.positive_gamma,
        "gammas must span both signs"
    );
}

#[test]
fn check_serial_relay_legs_match_reference() {
    check_against_reference(Schedule::CheckSerial, 0.1, 4, 4, usize::MAX, 0.9);
}

#[test]
fn explicit_gammas_cycle_when_fewer_sets_than_legs() {
    // Four legs, two gamma vectors: legs 3 and 4 reuse vectors 1 and 2.
    check_against_reference(Schedule::Flooding, 0.1, 4, 2, usize::MAX, 1.0);
}

#[test]
fn relay_stops_after_requested_converged_legs() {
    check_against_reference(Schedule::Flooding, 0.1, 6, 6, 2, 1.0);
}

#[test]
fn fired_degree_one_check_pins_its_mechanism_and_converges() {
    // gamma0 = 0 is the case where an infinite message would have poisoned
    // the bias with 0 * inf; the saturated message must instead pin the
    // mechanism so the syndrome is explained by exactly that mechanism.
    let dcm = DemCheckMatrix::from_dem_str(TOY_DEMS[DEGREE_ONE_TOY]).unwrap();
    let graph = BpGraph::from_dcm(&dcm);
    let prior = graph.prior_llrs().to_vec();
    let mut truth = vec![0u8; dcm.num_mechanisms];
    truth[DEGREE_ONE_MECHANISM] = 1;
    let syndrome = syndrome_of(&dcm, &truth);
    assert_eq!(syndrome[3], 1, "the degree-1 check must fire in this case");
    for schedule in [Schedule::Flooding, Schedule::CheckSerial] {
        for gamma0 in [0.0, 0.1, -0.24] {
            let cfg = RelayConfig {
                schedule,
                alpha: 1.0,
                gamma0,
                pre_iterations: 20,
                num_legs: 0,
                leg_iterations: 1,
                gamma_range: (-0.24, 0.66),
                stop_after_converged: 1,
                explicit_gammas: None,
            };
            let expected = reference_relay(&dcm, &prior, &syndrome, &cfg, &[]);
            let mut relay = RelayBp::new(graph.clone(), cfg).unwrap();
            let got = relay.decode(&syndrome, 3).unwrap();
            assert!(
                got.converged,
                "{schedule:?} gamma0={gamma0}: did not converge"
            );
            assert_eq!(got.correction, truth, "{schedule:?} gamma0={gamma0}");
            assert_eq!(bits(&got.posterior), bits(&expected.posterior));
            assert!(got.posterior.iter().all(|p| p.is_finite()));
        }
    }
}

#[test]
fn gamma_range_is_honoured_when_sampling() {
    // A degenerate range makes sampling deterministic: it must equal the
    // same run with that constant supplied as explicit gammas.
    let dcm = DemCheckMatrix::from_dem_str(TOY_DEMS[2]).unwrap();
    let graph = BpGraph::from_dcm(&dcm);
    let mut state = 99u64;
    let syndrome = random_syndrome(&dcm, &mut state);
    let base = RelayConfig {
        schedule: Schedule::Flooding,
        alpha: 1.0,
        gamma0: 0.1,
        pre_iterations: 3,
        num_legs: 4,
        leg_iterations: 4,
        gamma_range: (0.3, 0.3),
        stop_after_converged: usize::MAX,
        explicit_gammas: None,
    };
    let mut sampled = RelayBp::new(graph.clone(), base.clone()).unwrap();
    let explicit_cfg = RelayConfig {
        explicit_gammas: Some(vec![vec![0.3; dcm.num_mechanisms]]),
        ..base
    };
    let mut explicit = RelayBp::new(graph, explicit_cfg).unwrap();
    let a = sampled.decode(&syndrome, 5).unwrap();
    let b = explicit.decode(&syndrome, 5).unwrap();
    assert_eq!(bits(&a.posterior), bits(&b.posterior));
    assert_eq!(a.correction, b.correction);
    assert_eq!(a.legs.len(), 5);
}

#[test]
fn decode_is_a_pure_function_of_syndrome_and_shot_seed() {
    let dcm = DemCheckMatrix::from_dem_str(TOY_DEMS[2]).unwrap();
    let graph = BpGraph::from_dcm(&dcm);
    let cfg = RelayConfig {
        schedule: Schedule::Flooding,
        alpha: 1.0,
        gamma0: 0.1,
        pre_iterations: 3,
        num_legs: 5,
        leg_iterations: 4,
        gamma_range: (-0.24, 0.66),
        stop_after_converged: usize::MAX,
        explicit_gammas: None,
    };
    let mut state = 77u64;
    let syndromes: Vec<Vec<u8>> = (0..5).map(|_| random_syndrome(&dcm, &mut state)).collect();

    let mut relay = RelayBp::new(graph.clone(), cfg.clone()).unwrap();
    let in_order: Vec<_> = syndromes
        .iter()
        .enumerate()
        .map(|(i, s)| relay.decode(s, u64::try_from(i).unwrap()).unwrap())
        .collect();

    // Same shots, reversed order, fresh decoder: bitwise identical outcomes.
    let mut relay = RelayBp::new(graph.clone(), cfg.clone()).unwrap();
    for (i, s) in syndromes.iter().enumerate().rev() {
        let again = relay.decode(s, u64::try_from(i).unwrap()).unwrap();
        assert_eq!(bits(&again.posterior), bits(&in_order[i].posterior));
        assert_eq!(again.correction, in_order[i].correction);
    }

    // A different shot seed draws different gammas, so some posterior differs.
    let mut relay = RelayBp::new(graph, cfg).unwrap();
    let other = relay.decode(&syndromes[0], 1_000_003).unwrap();
    assert_ne!(bits(&other.posterior), bits(&in_order[0].posterior));
}

#[test]
fn config_validation_rejects_nonsense() {
    let dcm = DemCheckMatrix::from_dem_str(TOY_DEMS[0]).unwrap();
    let graph = BpGraph::from_dcm(&dcm);
    let good = RelayConfig {
        schedule: Schedule::Flooding,
        alpha: 1.0,
        gamma0: 0.1,
        pre_iterations: 3,
        num_legs: 2,
        leg_iterations: 4,
        gamma_range: (-0.24, 0.66),
        stop_after_converged: 1,
        explicit_gammas: None,
    };
    let rejects = |cfg: RelayConfig| RelayBp::new(graph.clone(), cfg).is_err();
    assert!(RelayBp::new(graph.clone(), good.clone()).is_ok());
    assert!(rejects(RelayConfig {
        alpha: 0.0,
        ..good.clone()
    }));
    assert!(rejects(RelayConfig {
        alpha: f64::NAN,
        ..good.clone()
    }));
    assert!(rejects(RelayConfig {
        pre_iterations: 0,
        ..good.clone()
    }));
    assert!(rejects(RelayConfig {
        num_legs: 1,
        leg_iterations: 0,
        ..good.clone()
    }));
    assert!(rejects(RelayConfig {
        gamma_range: (0.7, 0.2),
        ..good.clone()
    }));
    // Finite endpoints whose span overflows.
    assert!(rejects(RelayConfig {
        gamma_range: (-f64::MAX, f64::MAX),
        ..good.clone()
    }));
    assert!(rejects(RelayConfig {
        stop_after_converged: 0,
        ..good.clone()
    }));
    // Explicit gammas must have one entry per variable.
    let short = vec![vec![0.1; dcm.num_mechanisms - 1]];
    assert!(rejects(RelayConfig {
        explicit_gammas: Some(short),
        ..good.clone()
    }));

    // Syndromes must be binary and the right length.
    let mut relay = RelayBp::new(graph, good).unwrap();
    assert!(relay.decode(&[0, 1, 0], 0).is_err());
    assert!(relay.decode(&[0, 2, 0, 0], 0).is_err());
    assert!(relay.decode(&[0, 1, 0, 0], 0).is_ok());
}
