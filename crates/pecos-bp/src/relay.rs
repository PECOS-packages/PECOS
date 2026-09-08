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

//! Native Relay-BP decoding for detector error models.
//!
//! Relay-BP runs memory belief propagation in a sequence of warm-started
//! legs. Message state is reset between legs, while each leg's initial memory
//! bias uses the posterior left by the preceding leg. A degree-one check sends
//! a message of certainty magnitude [`LLR_SATURATION`]. It competes like any
//! other certain evidence and can tie against a zero-probability prior; the
//! syndrome check after each iteration guarantees no unsatisfied correction
//! is reported as converged.

use crate::{BpGraph, LLR_SATURATION};
use pecos_decoder_core::errors::DecoderError;
use pecos_random::PecosRng;

/// Message-passing update schedule used within each Relay-BP leg.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// Update every check before updating every variable.
    Flooding,
    /// Update checks in index order and immediately refresh touched variables.
    CheckSerial,
}

/// Configuration for [`RelayBp`].
#[derive(Clone, Debug)]
pub struct RelayConfig {
    /// Message-passing schedule used within every leg.
    pub schedule: Schedule,
    /// Positive, finite scale applied to min-sum check messages.
    pub alpha: f64,
    /// Memory strength used for every variable in leg zero.
    pub gamma0: f64,
    /// Maximum number of iterations in leg zero.
    pub pre_iterations: usize,
    /// Number of relayed legs after leg zero; must be less than [`usize::MAX`].
    pub num_legs: usize,
    /// Maximum number of iterations in each relayed leg.
    pub leg_iterations: usize,
    /// Inclusive parameter interval used to form uniform gamma samples.
    pub gamma_range: (f64, f64),
    /// Stop after this many legs have converged.
    pub stop_after_converged: usize,
    /// Optional reproducible per-variable gamma vectors for relayed legs.
    ///
    /// Vectors are reused cyclically when fewer vectors than relay legs are
    /// supplied. Each vector must contain one entry per graph mechanism.
    pub explicit_gammas: Option<Vec<Vec<f64>>>,
}

impl Default for RelayConfig {
    /// Return the Relay-BP operating point of Müller et al.,
    /// arXiv:2506.01779 (gamma interval, leg lengths), as also used by the
    /// staged decoder of arXiv:2607.28795.
    ///
    /// These values reproduce that operating point; they are not a tuned
    /// general-purpose PECOS default.
    fn default() -> Self {
        Self {
            schedule: Schedule::Flooding,
            alpha: 1.0,
            gamma0: 0.1,
            pre_iterations: 80,
            num_legs: 40,
            leg_iterations: 60,
            gamma_range: (-0.24, 0.66),
            stop_after_converged: 1,
            explicit_gammas: None,
        }
    }
}

/// Result of one executed Relay-BP leg.
///
/// A coset-quorum consumer derives this leg's observable coset from
/// [`correction`](Self::correction) with its own observable matrix.
#[derive(Clone, Debug)]
pub struct LegOutcome {
    /// Whether this leg produced a correction satisfying every check.
    pub converged: bool,
    /// Number of iterations executed by this leg.
    pub iterations: usize,
    /// Signed prior-LLR cost, present only on convergence; lower is more likely.
    pub weight: Option<f64>,
    /// Correction produced by this leg, present only on convergence.
    pub correction: Option<Vec<u8>>,
}

/// Result of a Relay-BP decode.
///
/// Setting [`RelayConfig::stop_after_converged`] to `q` collects the first
/// `q` converged legs in [`legs`](Self::legs). A coset-quorum consumer can
/// derive each such leg's observable coset from [`LegOutcome::correction`]
/// with its own observable matrix.
#[derive(Clone, Debug)]
pub struct RelayOutcome {
    /// Whether at least one executed leg converged.
    pub converged: bool,
    /// Minimum-cost converged correction, or the final hard decision.
    pub correction: Vec<u8>,
    /// Signed prior-LLR cost, present only when `correction` converged.
    pub weight: Option<f64>,
    /// Posterior at the end of the last executed leg.
    pub posterior: Vec<f64>,
    /// Outcomes of the executed legs, beginning with leg zero.
    pub legs: Vec<LegOutcome>,
    /// Sum of the iteration counts in `legs`.
    pub total_iterations: usize,
}

/// Native memory-BP decoder with relayed legs.
///
/// Graph-sized work buffers are allocated during construction and reset for
/// every call to [`decode`](Self::decode).
pub struct RelayBp {
    graph: BpGraph,
    config: RelayConfig,
    posterior: Vec<f64>,
    gamma: Vec<f64>,
    bias: Vec<f64>,
    c_to_v: Vec<f64>,
    v_to_c: Vec<f64>,
    hard_decision: Vec<u8>,
    best_correction: Vec<u8>,
}

impl RelayBp {
    /// Construct a Relay-BP decoder and allocate its reusable work buffers.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for non-finite or
    /// inconsistent parameters, zero required iteration counts, a zero
    /// convergence target, or malformed explicit gamma vectors.
    pub fn new(graph: BpGraph, config: RelayConfig) -> Result<Self, DecoderError> {
        validate_config(&graph, &config)?;

        let num_vars = graph.mechanism_count();
        let num_edges = graph.edge_count();
        Ok(Self {
            graph,
            config,
            posterior: vec![0.0; num_vars],
            gamma: vec![0.0; num_vars],
            bias: vec![0.0; num_vars],
            c_to_v: vec![0.0; num_edges],
            v_to_c: vec![0.0; num_edges],
            hard_decision: vec![0; num_vars],
            best_correction: vec![0; num_vars],
        })
    }

    /// Decode one syndrome with a deterministic, per-shot gamma stream.
    ///
    /// Relayed leg `r` samples its gammas from a fresh [`PecosRng`] seeded by
    /// the combined `(shot_seed, r)` pair. `PecosRng` expands that seed with
    /// `SplitMix64`, and no RNG or message state is shared between shots.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidDimensions`] unless `syndrome` contains
    /// exactly one entry per graph check, [`DecoderError::InvalidSyndrome`] if
    /// an entry is not binary, or [`DecoderError::InternalError`] if message
    /// accumulation produces a non-finite posterior.
    pub fn decode(
        &mut self,
        syndrome: &[u8],
        shot_seed: u64,
    ) -> Result<RelayOutcome, DecoderError> {
        if syndrome.len() != self.graph.check_count() {
            return Err(DecoderError::InvalidDimensions {
                expected: self.graph.check_count(),
                actual: syndrome.len(),
            });
        }
        if let Some((check, &value)) = syndrome.iter().enumerate().find(|&(_, &value)| value > 1) {
            return Err(DecoderError::InvalidSyndrome(format!(
                "Relay-BP syndrome entry {check} must be zero or one, got {value}"
            )));
        }

        self.posterior.copy_from_slice(self.graph.prior_llrs());

        let mut legs = Vec::new();
        let mut total_iterations = 0;
        let mut num_converged = 0;
        let mut best_weight = None;

        for leg in 0..=self.config.num_legs {
            if leg > 0 && num_converged >= self.config.stop_after_converged {
                break;
            }

            if leg == 0 {
                self.gamma.fill(self.config.gamma0);
            } else {
                self.prepare_relay_gamma(shot_seed, leg);
            }
            self.start_leg();

            let iteration_limit = if leg == 0 {
                self.config.pre_iterations
            } else {
                self.config.leg_iterations
            };
            let (converged, iterations) = self.run_leg(syndrome, iteration_limit);
            total_iterations += iterations;
            if let Some(variable) = self
                .posterior
                .iter()
                .position(|posterior| !posterior.is_finite())
            {
                return Err(DecoderError::InternalError(format!(
                    "Relay belief propagation produced a non-finite posterior for mechanism \
                     {variable}; the model's degree, prior, memory, and scaling regime exceeds \
                     what min-sum message accumulation can represent"
                )));
            }

            let (weight, correction) = if converged {
                num_converged += 1;
                let weight = self.correction_weight();
                if best_weight.is_none_or(|current| weight < current) {
                    best_weight = Some(weight);
                    self.best_correction.copy_from_slice(&self.hard_decision);
                }
                (Some(weight), Some(self.hard_decision.clone()))
            } else {
                (None, None)
            };
            legs.push(LegOutcome {
                converged,
                iterations,
                weight,
                correction,
            });
        }

        let converged = best_weight.is_some();
        let (correction, weight) = if let Some(weight) = best_weight {
            (self.best_correction.clone(), Some(weight))
        } else {
            (self.hard_decision.clone(), None)
        };

        Ok(RelayOutcome {
            converged,
            correction,
            weight,
            posterior: self.posterior.clone(),
            legs,
            total_iterations,
        })
    }

    /// Return the immutable Tanner graph used by this decoder.
    #[must_use]
    pub const fn graph(&self) -> &BpGraph {
        &self.graph
    }

    /// Return this decoder's validated configuration.
    #[must_use]
    pub const fn config(&self) -> &RelayConfig {
        &self.config
    }

    fn prepare_relay_gamma(&mut self, shot_seed: u64, leg: usize) {
        if let Some(explicit) = &self.config.explicit_gammas {
            self.gamma
                .copy_from_slice(&explicit[(leg - 1) % explicit.len()]);
            return;
        }

        let mut rng = PecosRng::seed_from_u64(leg_seed(shot_seed, leg));
        let (lo, hi) = self.config.gamma_range;
        for gamma in &mut self.gamma {
            *gamma = lo + (hi - lo) * rng.next_f64();
        }
    }

    fn start_leg(&mut self) {
        self.c_to_v.fill(0.0);
        for variable in 0..self.graph.mechanism_count() {
            for &(_, message) in self.graph.var_entries(variable) {
                self.v_to_c[message as usize] = self.graph.prior_llrs()[variable];
            }
        }
    }

    fn run_leg(&mut self, syndrome: &[u8], iteration_limit: usize) -> (bool, usize) {
        for iteration in 1..=iteration_limit {
            match self.config.schedule {
                Schedule::Flooding => self.flooding_iteration(syndrome),
                Schedule::CheckSerial => self.check_serial_iteration(syndrome),
            }
            if self.update_hard_decision_and_check(syndrome) {
                return (true, iteration);
            }
        }
        (false, iteration_limit)
    }

    fn compute_bias(&mut self) {
        for variable in 0..self.graph.mechanism_count() {
            let prior = self.graph.prior_llrs()[variable];
            self.bias[variable] = prior + self.gamma[variable] * (self.posterior[variable] - prior);
        }
    }

    fn flooding_iteration(&mut self, syndrome: &[u8]) {
        for check in 0..self.graph.check_count() {
            update_check(
                &self.graph,
                syndrome,
                self.config.alpha,
                check,
                &self.v_to_c,
                &mut self.c_to_v,
            );
        }
        self.compute_bias();
        for variable in 0..self.graph.mechanism_count() {
            update_variable_flooding(
                &self.graph,
                variable,
                &self.bias,
                &self.c_to_v,
                &mut self.v_to_c,
                &mut self.posterior,
            );
        }
    }

    fn check_serial_iteration(&mut self, syndrome: &[u8]) {
        self.compute_bias();
        for check in 0..self.graph.check_count() {
            update_check(
                &self.graph,
                syndrome,
                self.config.alpha,
                check,
                &self.v_to_c,
                &mut self.c_to_v,
            );
            for &(variable, _) in self.graph.check_entries(check) {
                update_variable_serial(
                    &self.graph,
                    variable as usize,
                    &self.bias,
                    &self.c_to_v,
                    &mut self.v_to_c,
                    &mut self.posterior,
                );
            }
        }
    }

    fn update_hard_decision_and_check(&mut self, syndrome: &[u8]) -> bool {
        for (decision, &posterior) in self.hard_decision.iter_mut().zip(&self.posterior) {
            *decision = u8::from(posterior < 0.0);
        }

        for (check, &expected) in syndrome.iter().enumerate() {
            let parity = self
                .graph
                .check_entries(check)
                .iter()
                .fold(0, |parity, &(variable, _)| {
                    parity ^ self.hard_decision[variable as usize]
                });
            if parity != expected {
                return false;
            }
        }
        true
    }

    fn correction_weight(&self) -> f64 {
        self.hard_decision
            .iter()
            .zip(self.graph.prior_llrs())
            .map(|(&bit, &prior)| f64::from(bit) * prior)
            .sum()
    }
}

fn validate_config(graph: &BpGraph, config: &RelayConfig) -> Result<(), DecoderError> {
    if !config.alpha.is_finite() || config.alpha <= 0.0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "Relay-BP alpha must be finite and greater than zero, got {}",
            config.alpha
        )));
    }
    if !config.gamma0.is_finite() {
        return Err(DecoderError::InvalidConfiguration(format!(
            "Relay-BP gamma0 must be finite, got {}",
            config.gamma0
        )));
    }
    if config.pre_iterations == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "Relay-BP pre_iterations must be at least one".into(),
        ));
    }
    if config.num_legs == usize::MAX {
        return Err(DecoderError::InvalidConfiguration(
            "Relay-BP num_legs must be less than usize::MAX".into(),
        ));
    }
    if config.num_legs > 0 && config.leg_iterations == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "Relay-BP leg_iterations must be at least one when num_legs is nonzero".into(),
        ));
    }

    let (lo, hi) = config.gamma_range;
    if !lo.is_finite() || !hi.is_finite() || lo > hi || !(hi - lo).is_finite() {
        return Err(DecoderError::InvalidConfiguration(format!(
            "Relay-BP gamma_range endpoints and span must be finite and ordered, got ({lo}, {hi})"
        )));
    }
    if config.stop_after_converged == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "Relay-BP stop_after_converged must be at least one".into(),
        ));
    }

    if let Some(explicit) = &config.explicit_gammas {
        if config.num_legs > 0 && explicit.is_empty() {
            return Err(DecoderError::InvalidConfiguration(
                "Relay-BP explicit_gammas must not be empty when relay legs are configured".into(),
            ));
        }
        for (set, gammas) in explicit.iter().enumerate() {
            if gammas.len() != graph.mechanism_count() {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "Relay-BP explicit gamma set {set} has {} entries; expected {}",
                    gammas.len(),
                    graph.mechanism_count()
                )));
            }
            if let Some(variable) = gammas.iter().position(|gamma| !gamma.is_finite()) {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "Relay-BP explicit gamma set {set}, mechanism {variable} must be finite"
                )));
            }
        }
    }
    Ok(())
}

#[inline]
fn message_sign(message: f64) -> f64 {
    if message < 0.0 { -1.0 } else { 1.0 }
}

/// Update one check. A degree-one check sends a message of certainty magnitude
/// [`LLR_SATURATION`]; it competes like any other certain evidence and can tie
/// against a zero-probability prior. The per-iteration syndrome check prevents
/// an unsatisfied correction from being reported as converged.
fn update_check(
    graph: &BpGraph,
    syndrome: &[u8],
    alpha: f64,
    check: usize,
    v_to_c: &[f64],
    c_to_v: &mut [f64],
) {
    let entries = graph.check_entries(check);
    let mut total_sign = if syndrome[check] != 0 { -1.0 } else { 1.0 };
    let mut min1 = f64::INFINITY;
    let mut min2 = f64::INFINITY;
    let mut min1_variable = u32::MAX;

    for &(variable, message) in entries {
        let value = v_to_c[message as usize];
        total_sign *= message_sign(value);
        let magnitude = value.abs();
        if magnitude < min1 {
            min2 = min1;
            min1 = magnitude;
            min1_variable = variable;
        } else if magnitude < min2 {
            min2 = magnitude;
        }
    }

    for &(variable, message) in entries {
        let value = v_to_c[message as usize];
        let exclusive_sign = total_sign * message_sign(value);
        let exclusive_min = if variable == min1_variable {
            min2
        } else {
            min1
        };
        let exclusive_min = if exclusive_min.is_finite() {
            exclusive_min
        } else {
            LLR_SATURATION
        };
        c_to_v[message as usize] = alpha * exclusive_sign * exclusive_min;
    }
}

/// Combine the shot and one-based leg indices before `PecosRng` applies its
/// own `SplitMix64` seed expansion.
fn leg_seed(shot_seed: u64, leg: usize) -> u64 {
    let leg = u64::try_from(leg).expect("relay leg index does not fit in u64");
    shot_seed ^ leg.wrapping_add(1)
}

fn update_variable_flooding(
    graph: &BpGraph,
    variable: usize,
    bias: &[f64],
    c_to_v: &[f64],
    v_to_c: &mut [f64],
    posterior: &mut [f64],
) {
    let entries = graph.var_entries(variable);
    let total: f64 = entries
        .iter()
        .map(|&(_, message)| c_to_v[message as usize])
        .sum();
    for &(_, message) in entries {
        v_to_c[message as usize] = bias[variable] + total - c_to_v[message as usize];
    }
    posterior[variable] = bias[variable] + total;
}

fn update_variable_serial(
    graph: &BpGraph,
    variable: usize,
    bias: &[f64],
    c_to_v: &[f64],
    v_to_c: &mut [f64],
    posterior: &mut [f64],
) {
    let entries = graph.var_entries(variable);
    let total: f64 = entries
        .iter()
        .map(|&(_, message)| c_to_v[message as usize])
        .sum();
    posterior[variable] = bias[variable] + total;
    for &(_, message) in entries {
        v_to_c[message as usize] = posterior[variable] - c_to_v[message as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::{RelayBp, RelayConfig, Schedule, leg_seed, message_sign};
    use crate::{BpGraph, LLR_SATURATION};
    use pecos_decoder_core::dem::DemCheckMatrix;
    use pecos_decoder_core::errors::DecoderError;
    use pecos_random::PecosRng;

    fn degree_one_graph() -> BpGraph {
        let dcm = DemCheckMatrix::from_dem_str("error(0.1) D0\n").unwrap();
        BpGraph::from_dcm(&dcm)
    }

    fn one_iteration_config(schedule: Schedule) -> RelayConfig {
        RelayConfig {
            schedule,
            alpha: 1.0,
            gamma0: 0.0,
            pre_iterations: 1,
            num_legs: 0,
            leg_iterations: 0,
            gamma_range: (-0.24, 0.66),
            stop_after_converged: 1,
            explicit_gammas: None,
        }
    }

    #[test]
    fn degree_one_checks_send_saturated_not_maximum_messages() {
        for schedule in [Schedule::Flooding, Schedule::CheckSerial] {
            let mut decoder =
                RelayBp::new(degree_one_graph(), one_iteration_config(schedule)).unwrap();
            let outcome = decoder.decode(&[1], 0).unwrap();

            assert!(outcome.converged);
            assert_eq!(outcome.correction, vec![1]);
            assert!(outcome.posterior.iter().all(|value| value.is_finite()));
            assert!(decoder.c_to_v[0].is_finite());
            assert_eq!(decoder.c_to_v[0].to_bits(), (-LLR_SATURATION).to_bits());
        }
    }

    #[test]
    fn zero_posterior_has_zero_hard_decision() {
        let dcm = DemCheckMatrix::from_dem_str("error(0) D0\n").unwrap();
        let graph = BpGraph::from_dcm(&dcm);

        for schedule in [Schedule::Flooding, Schedule::CheckSerial] {
            let mut decoder = RelayBp::new(graph.clone(), one_iteration_config(schedule)).unwrap();
            let outcome = decoder.decode(&[1], 0).unwrap();

            assert_eq!(outcome.posterior[0].to_bits(), 0.0_f64.to_bits());
            assert!(!outcome.converged);
            assert_eq!(outcome.correction, vec![0]);
            assert_eq!(outcome.weight, None);
        }
    }

    #[test]
    fn negative_zero_message_has_positive_sign() {
        let alpha = 1.0;
        let negative_sign = -1.0;
        let zero_magnitude = 0.0;
        let reachable_negative_zero: f64 = alpha * negative_sign * zero_magnitude;
        assert_eq!(reachable_negative_zero.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(
            message_sign(reachable_negative_zero).to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn signed_cost_prefers_a_high_probability_correction() {
        let dcm = DemCheckMatrix::from_dem_str("error(0.4) D0\nerror(0.9) D0\n").unwrap();
        let graph = BpGraph::from_dcm(&dcm);
        let high_probability_cost = graph.prior_llrs()[1];
        let config = RelayConfig {
            schedule: Schedule::Flooding,
            alpha: 1.0,
            gamma0: 0.0,
            pre_iterations: 1,
            num_legs: 1,
            leg_iterations: 1,
            gamma_range: (-10.0, -10.0),
            stop_after_converged: 2,
            explicit_gammas: Some(vec![vec![-10.0, -10.0]]),
        };
        let mut decoder = RelayBp::new(graph, config).unwrap();

        let outcome = decoder.decode(&[1], 0).unwrap();

        assert_eq!(outcome.legs.len(), 2);
        assert_eq!(outcome.legs[0].correction, Some(vec![0, 1]));
        assert_eq!(outcome.legs[1].correction, Some(vec![1, 0]));
        for leg in &outcome.legs {
            let correction = leg.correction.as_ref().unwrap();
            assert_eq!(correction[0] ^ correction[1], 1);
        }
        assert_eq!(outcome.correction, vec![0, 1]);
        assert_eq!(
            outcome.weight.unwrap().to_bits(),
            high_probability_cost.to_bits()
        );
        assert!(outcome.legs[0].weight.unwrap() < outcome.legs[1].weight.unwrap());
    }

    #[test]
    fn failed_relay_returns_the_last_legs_hard_decision_without_a_weight() {
        let dcm = DemCheckMatrix::from_dem_str("error(0.1) D0 D1\nerror(0.4) D1\n").unwrap();
        let config = RelayConfig {
            schedule: Schedule::Flooding,
            alpha: 1.0,
            gamma0: 0.0,
            pre_iterations: 1,
            num_legs: 1,
            leg_iterations: 1,
            gamma_range: (-2.0, 1.0),
            stop_after_converged: usize::MAX,
            explicit_gammas: Some(vec![vec![-2.0, 1.0]]),
        };
        let mut decoder = RelayBp::new(BpGraph::from_dcm(&dcm), config).unwrap();

        let outcome = decoder.decode(&[1, 0], 0).unwrap();

        assert_eq!(outcome.legs.len(), 2);
        assert!(outcome.legs.iter().all(|leg| !leg.converged));
        assert_eq!(outcome.correction, vec![0, 0]);
        assert_eq!(
            outcome
                .posterior
                .iter()
                .map(|p| u8::from(*p < 0.0))
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
        assert_eq!(outcome.weight, None);
    }

    #[test]
    fn relay_ranking_uses_saturated_prior_costs() {
        let dcm = DemCheckMatrix::from_dem_str("error(1e-300) D0\nerror(9.3e-14) D0\n").unwrap();
        let graph = BpGraph::from_dcm(&dcm);
        assert_eq!(graph.prior_llrs(), &[LLR_SATURATION, LLR_SATURATION]);
        let config = RelayConfig {
            schedule: Schedule::Flooding,
            alpha: 1.0,
            gamma0: 0.0,
            pre_iterations: 1,
            num_legs: 2,
            leg_iterations: 1,
            gamma_range: (-1.0, 2.0),
            stop_after_converged: usize::MAX,
            explicit_gammas: Some(vec![vec![0.5, -0.5], vec![-1.0, 2.0]]),
        };
        let mut decoder = RelayBp::new(graph, config).unwrap();

        let outcome = decoder.decode(&[1], 0).unwrap();

        assert_eq!(outcome.legs[0].correction, None);
        assert_eq!(outcome.legs[1].correction, Some(vec![1, 0]));
        assert_eq!(outcome.legs[2].correction, Some(vec![0, 1]));
        assert_eq!(outcome.legs[1].weight, Some(LLR_SATURATION));
        assert_eq!(outcome.legs[2].weight, Some(LLR_SATURATION));
        assert_eq!(outcome.correction, vec![1, 0]);
        assert_eq!(outcome.weight, Some(LLR_SATURATION));
    }

    #[test]
    fn decode_rejects_non_finite_posteriors() {
        let mut config = one_iteration_config(Schedule::Flooding);
        config.alpha = f64::MAX;
        let mut decoder = RelayBp::new(degree_one_graph(), config).unwrap();

        let error = decoder.decode(&[0], 0).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::InternalError(message) if message.contains("mechanism 0")
        ));
    }

    #[test]
    fn validation_rejects_non_finite_gamma_span() {
        let mut config = one_iteration_config(Schedule::Flooding);
        config.gamma_range = (-f64::MAX, f64::MAX);
        assert!(matches!(
            RelayBp::new(degree_one_graph(), config),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn validation_rejects_unbounded_leg_count() {
        let mut config = one_iteration_config(Schedule::Flooding);
        config.num_legs = usize::MAX;
        config.leg_iterations = 1;
        assert!(matches!(
            RelayBp::new(degree_one_graph(), config),
            Err(DecoderError::InvalidConfiguration(message))
                if message.contains("num_legs")
        ));
    }

    #[test]
    fn non_binary_syndrome_is_rejected_before_state_mutation() {
        let mut decoder =
            RelayBp::new(degree_one_graph(), one_iteration_config(Schedule::Flooding)).unwrap();
        let posterior_before = decoder.posterior.clone();

        let error = decoder.decode(&[2], 0).unwrap_err();
        assert!(matches!(error, DecoderError::InvalidSyndrome(_)));
        assert_eq!(decoder.posterior, posterior_before);
    }

    #[test]
    fn leg_gamma_streams_depend_on_the_shot_and_leg() {
        fn gamma_bits(shot_seed: u64, leg: usize) -> [u64; 8] {
            let mut rng = PecosRng::seed_from_u64(leg_seed(shot_seed, leg));
            std::array::from_fn(|_| (-0.24 + (0.66 - -0.24) * rng.next_f64()).to_bits())
        }

        assert_eq!(leg_seed(0x1234_5678_9abc_def0, 7), 0x1234_5678_9abc_def8);
        let mut shot_zero_leg_one = None;
        let mut shot_one_leg_one = None;
        for shot_seed in [0, 1, u64::MAX] {
            let streams: Vec<_> = (1..=4).map(|leg| gamma_bits(shot_seed, leg)).collect();
            for left in 0..streams.len() {
                for right in left + 1..streams.len() {
                    for shift in 0..streams[left].len() {
                        let overlap = streams[left].len() - shift;
                        assert_ne!(
                            &streams[left][shift..],
                            &streams[right][..overlap],
                            "shot {shot_seed}, legs {} and {} share a shifted window",
                            left + 1,
                            right + 1
                        );
                        assert_ne!(
                            &streams[right][shift..],
                            &streams[left][..overlap],
                            "shot {shot_seed}, legs {} and {} share a shifted window",
                            right + 1,
                            left + 1
                        );
                    }
                }
            }
            if shot_seed == 0 {
                shot_zero_leg_one = Some(streams[0]);
            } else if shot_seed == 1 {
                shot_one_leg_one = Some(streams[0]);
            }
        }
        assert_ne!(shot_zero_leg_one, shot_one_leg_one);
    }
}
