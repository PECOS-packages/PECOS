// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use crate::Gate;
use crate::byte_message::{ByteMessage, ByteMessageBuilder, GateType};
use crate::engine_system::{ControlEngine, EngineStage};
use crate::noise::{NoiseModel, NoiseRng, NoiseUtils, ProbabilityValidator, RngManageable};
use log::trace;
use pecos_core::errors::PecosError;
use pecos_random::PecosRng;
use pecos_random::rng_ext::RngProbabilityExt;
use std::any::Any;
use std::collections::BTreeMap;
use std::collections::HashSet;

/////////////////////////////////////////////////////////
/// Tools for cataloging error opportunities and outcomes
/////////////////////////////////////////////////////////

/// The kinds of faults supported in cataloging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepolarizingFaultSiteKind {
    Prep,
    Meas,
    SingleQubit,
    TwoQubit,
}

/// This class stores one possible outcome and its probability]
#[derive(Debug, Clone, PartialEq)]
pub struct DepolarizingFaultOutcome {
    /// Human-readable outcome label.
    pub label: &'static str,
    /// Outcome probability at this site.
    pub probability: f64,
}

/// Sites at which faults can occur
#[derive(Debug, Clone, PartialEq)]
pub struct DepolarizingFaultSite {
    // A globally unique identifier for this fault site
    pub uid: usize,
    pub gate_index: usize,
    pub kind: DepolarizingFaultSiteKind,
    pub gate_type: GateType,
    // Specify the error location
    pub qubits: Vec<usize>,
    // Specify possible states and their probabilities
    pub outcomes: Vec<DepolarizingFaultOutcome>,
}

impl DepolarizingFaultSite {
    #[must_use]
    pub fn outcome_label_probability(&self, outcome_label: &str) -> Option<f64> {
        self.outcomes
            .iter()
            .find(|outcome| outcome.label == outcome_label)
            .map(|outcome| outcome.probability)
    }

    #[must_use]
    pub fn outcome_probability(&self, outcome: DepolarizingFaultOutcome) -> Option<f64> {
        self.outcome_label_probability(outcome.label)
    }

    pub fn random_outcome(&self) -> DepolarizingFaultOutcome {
        // Get a random number between zero and 1
        let rand_val = rand::random::<f64>();
        // Scale the random number to the cumulative probabilities of the outcomes
        let scaled_val = rand_val * self.outcomes.iter().map(|o| o.probability).sum::<f64>();
        // Loop through the outcomes and find the first one whose cumulative probability exceeds the random number
        let mut cumulative_probability = 0.0;
        for outcome in &self.outcomes {
            cumulative_probability += outcome.probability;
            if scaled_val < cumulative_probability {
                return outcome.clone();
            }
        }
        self.outcomes.last().unwrap().clone()
    }

    // Find a random outcome that is not the current outcome
    pub fn random_outcome_except(&self, current_outcome_label: &str) -> DepolarizingFaultOutcome {
        // Grab a list of all the outcomes except the specified one
        let filtered_outcomes: Vec<&DepolarizingFaultOutcome> = self
            .outcomes
            .iter()
            .filter(|outcome| outcome.label != current_outcome_label)
            .collect();

        assert!(
            !filtered_outcomes.is_empty(),
            "No outcomes available except the current one"
        );

        // Get a random number between zero and 1
        let rand_val = rand::random::<f64>();

        // Scale the random number to the cumulative probabilities of the filtered outcomes
        let scaled_val = rand_val * filtered_outcomes.iter().map(|o| o.probability).sum::<f64>();
        assert!(
            scaled_val > f64::EPSILON,
            "Random value is too small, check probabilities of outcomes"
        );

        // Loop through the outcomes and find the first one whose cumulative probability exceeds
        // the random number
        let mut cumulative_probability = 0.0;
        for outcome in &filtered_outcomes {
            cumulative_probability += outcome.probability;
            if scaled_val < cumulative_probability {
                return (*outcome).clone();
            }
        }
        (*filtered_outcomes.last().unwrap()).clone()
    }

    #[must_use]
    pub fn no_fault_probability(&self) -> Option<f64> {
        self.outcome_label_probability("NoFault")
    }
}

/// A single fault-site sampled fault (not a history of faults)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepolarizingSampledFault {
    /// Deterministic site identifier from traversal order.
    pub site_uid: usize,
    /// Outcome index for this fault, excludes 0 (no-fault) outcome.
    pub outcome_index: u8,
    /// Human-readable outcome label (for example `X`, `YZ`, `IX`).
    pub outcome_label: &'static str,
}

// A fault catalog
#[derive(Debug, Clone, PartialEq)]
pub struct DepolarizingFaultCatalog {
    /// Ordered fault sites.
    pub sites: Vec<DepolarizingFaultSite>,
    rng: Option<PecosRng>,
}

impl Default for DepolarizingFaultCatalog {
    fn default() -> Self {
        Self {
            sites: Vec::new(),
            rng: None,
        }
    }
}

impl DepolarizingFaultCatalog {
    // Function that computes the probability of a sampled fault history
    pub fn fault_history_probability(
        &self,
        sampled_fault_history: &[DepolarizingSampledFault],
    ) -> f64 {
        self.check_valid_fault_history(sampled_fault_history);
        let mut probability = 1.0;
        let mut next_fault_history_ind = 0;

        for site in &self.sites {
            let no_fault_probability = site.no_fault_probability().unwrap_or_else(|| {
                panic!("No-fault outcome not found for fault site {}", site.uid);
            });

            match sampled_fault_history.get(next_fault_history_ind) {
                Some(fault) if fault.site_uid == site.uid => {
                    let label = fault.outcome_label;
                    probability *= site.outcome_label_probability(label).unwrap_or_else(|| {
                        panic!(
                            "Outcome label {} not found for fault site {}",
                            label, site.uid
                        );
                    });
                    next_fault_history_ind += 1;
                }
                Some(_) => {
                    probability *= no_fault_probability;
                }
                None => {
                    probability *= no_fault_probability;
                }
            }
        }
        probability
    }

    // TODO Implement a "next" function that takes a sampled
    // fault history and returns the next sampled fault history

    // TODO Implement a iterator function that returns an iterator
    // over all fault histories

    pub fn get_site(&self, site_uid: usize) -> &DepolarizingFaultSite {
        self.sites
            .iter()
            .find(|s| s.uid == site_uid)
            .unwrap_or_else(|| {
                panic!("Site uid {} not found in fault catalog", site_uid);
            })
    }

    // Function to set a random number generator seed for fault sampling
    pub fn set_seed(&mut self, seed: u64) {
        // Set the seed for the random number generator
        self.rng = Some(PecosRng::seed_from_u64(seed));
    }

    // Function to grab a random site
    fn random_site(&mut self) -> usize {
        // Error if the rng is not set
        if self.rng.is_none() {
            panic!("Random number generator not set for fault catalog, set using catalog.set_seed()");
        }

        let nsite: u64 = self.sites.len() as u64;
        let invalid_threshold = nsite.wrapping_neg() % nsite;

        // Find a random site between these two
        let mut bit = self.rng.as_mut().unwrap().next_u64();
        while bit < invalid_threshold {
            bit = self.rng.as_mut().unwrap().next_u64();
        }

        // Convert it to a site with modulo
        (bit % nsite) as usize
    }

    // Given a site uid, this performs a random flip to one of the other outcomes
    pub fn random_flip_at_site(
        &mut self,
        site_uid: usize,
        sampled_fault_history: &[DepolarizingSampledFault],
    ) -> Vec<DepolarizingSampledFault> {

        self.check_valid_fault_history(sampled_fault_history);
        // Generate a random value between 0 and 1
        let rand_val = self.rng.as_mut().unwrap().next_f64();

        let mut flipped_fault_history = sampled_fault_history.to_vec();

        // Grab the site and capture the current outcome before removing the site.
        let site = self.get_site(site_uid);
        let current_outcome_label = sampled_fault_history
            .iter()
            .find(|fault| fault.site_uid == site_uid)
            .map_or("NoFault", |fault| fault.outcome_label);

        // Remove any existing sample at this site before inserting the flipped outcome.
        flipped_fault_history.retain(|fault| fault.site_uid != site_uid);

        // Grab a list of all of the outcomes except the current one
        let outcomes = site
            .outcomes
            .iter()
            .filter(|outcome| outcome.label != current_outcome_label)
            .collect::<Vec<_>>();

        // Scale down the probability to the sum of the probabilities of flips
        let scaled_val = rand_val * outcomes.iter().map(|outcome| outcome.probability).sum::<f64>();

        // Loop over outcomes until we find the first one whose cumulative probability exceeds the random number
        let mut cumulative_probability = 0.0;
        for outcome in &outcomes {
            cumulative_probability += outcome.probability;
            if scaled_val < cumulative_probability {
                flipped_fault_history.push(DepolarizingSampledFault {
                    site_uid,
                    outcome_index: site
                        .outcomes
                        .iter()
                        .position(|o| o.label == outcome.label)
                        .unwrap() as u8,
                    outcome_label: outcome.label,
                });
            }
        }

        // Resort by site_uid to maintain order
        flipped_fault_history.sort_by_key(|fault| fault.site_uid);

        flipped_fault_history
    }

    // Takes a fault history and randomly selects a site and flips it to a
    // different outcome
    pub fn random_flip(
        &mut self,
        sampled_fault_history: &[DepolarizingSampledFault],
    ) -> Vec<DepolarizingSampledFault> {
        // Pick a random site
        let flip_site_uid = self.random_site();

        // Flip the site
        self.random_flip_at_site(flip_site_uid, sampled_fault_history)
    }

    // Checks that two catalogs are compatible with each other
    fn is_catalog_compatible(&self, other: &DepolarizingFaultCatalog) -> bool {
        // Check that they have the same number of sites
        if self.sites.len() != other.sites.len() {
            return false;
        }
        // check that each site has the same uid and gate type
        for (site, other_site) in self.sites.iter().zip(other.sites.iter()) {
            if site.uid != other_site.uid || site.gate_type != other_site.gate_type {
                return false;
            }
        }
        true
    }

    // Checks that a history is compatible with this catalog
    fn check_valid_fault_history(&self, history: &[DepolarizingSampledFault]) -> bool {
        let mut fault_site_uids = history.iter().map(|fault| fault.site_uid).collect::<Vec<_>>();
        let mut catalog_site_uids = self.sites.iter().map(|site| site.uid).collect::<Vec<_>>();

        // Check that all of the site_uids in the history are present in the catalog
        for site in fault_site_uids.iter() {
            assert!(
                catalog_site_uids.contains(site),
                "Fault history contains site uid {} not present in catalog",
                site
            );
        }

        // Check that all of the site_uids are in ascending order
        for (site1, site2) in fault_site_uids.iter().zip(fault_site_uids.iter().skip(1)) {
            assert!(
                site1 < site2,
                "Fault catalog sites are not in ascending order: {} >= {}",
                site1,
                site2
            );
        }

        // Check that there are no duplicate site_uids in the history
        let fault_site_uids_set: HashSet<_> = fault_site_uids.into_iter().collect();
        assert!(
            fault_site_uids_set.len() == history.len(),
            "Fault history contains duplicate site uids"
        );

        true
    }

    // Takes another fault catalog and returns the ratio of the
    // probabilities of a single fault history.
    pub fn fault_catalog_probability_ratio(
        &self,
        other: &DepolarizingFaultCatalog,
        sampled_fault_history: &[DepolarizingSampledFault],
    ) -> f64 {

        self.check_valid_fault_history(sampled_fault_history);
        assert!(
            self.is_catalog_compatible(other),
            "Fault catalogs are not compatible"
        );
        // Easiest to compute probabilities separately since
        // we no fault probabilities will be different for each catalog
        let prob_self = self.fault_history_probability(sampled_fault_history);
        let prob_other = other.fault_history_probability(sampled_fault_history);
        prob_self / prob_other
    }

    // Takes two fault histories and returns the ratio of their probabilitieis
    pub fn fault_histories_probability_ratio(
        &self,
        sampled_fault_history_a: &[DepolarizingSampledFault],
        sampled_fault_history_b: &[DepolarizingSampledFault],
    ) -> f64 {

        self.check_valid_fault_history(sampled_fault_history_a);
        self.check_valid_fault_history(sampled_fault_history_b);

        // Track the ratio
        let mut ratio = 1.0;

        // Counters to track the next fault sites
        let mut next_fault_history_ind_a = 0;
        let mut next_fault_history_ind_b = 0;
        let mut next_fault_a = sampled_fault_history_a.get(next_fault_history_ind_a);
        let mut next_fault_b = sampled_fault_history_b.get(next_fault_history_ind_b);
        let mut next_fault_site_a = next_fault_a.map(|f| f.site_uid);
        let mut next_fault_site_b = next_fault_b.map(|f| f.site_uid);

        // Iterate through sites, only updating if there is a fault site
        for site in &self.sites {
            // Check if history_a or history_b both have a fault here
            let has_fault_a = next_fault_site_a == Some(site.uid);
            let has_fault_b = next_fault_site_b == Some(site.uid);
            if has_fault_a && has_fault_b {
                // Both histories have a fault here, compute ratio of probabilities
                let label_a = next_fault_a.unwrap().outcome_label;
                let label_b = next_fault_b.unwrap().outcome_label;
                let prob_a = site.outcome_label_probability(label_a).unwrap_or_else(|| {
                    panic!(
                        "Outcome label {} not found for fault site {}",
                        label_a, site.uid
                    );
                });
                let prob_b = site.outcome_label_probability(label_b).unwrap_or_else(|| {
                    panic!(
                        "Outcome label {} not found for fault site {}",
                        label_b, site.uid
                    );
                });
                // Update the ratio
                ratio *= prob_a / prob_b;
            } else if has_fault_a {
                // Only history_a has a fault here, multiply by its probability and divide by no-fault probability
                let label_a = next_fault_a.unwrap().outcome_label;
                let prob_a = site.outcome_label_probability(label_a).unwrap_or_else(|| {
                    panic!(
                        "Outcome label {} not found for fault site {}",
                        label_a, site.uid
                    );
                });
                let prob_b = site.no_fault_probability().unwrap_or_else(|| {
                    panic!("No-fault outcome not found for fault site {}", site.uid);
                });
                // Update the ratio
                ratio *= prob_a / prob_b;
            } else if has_fault_b {
                // Only history_b has a fault here, multiply by no-fault probability and divide by its probability
                let label_b = next_fault_b.unwrap().outcome_label;
                let prob_b = site.outcome_label_probability(label_b).unwrap_or_else(|| {
                    panic!(
                        "Outcome label {} not found for fault site {}",
                        label_b, site.uid
                    );
                });
                let prob_a = site.no_fault_probability().unwrap_or_else(|| {
                    panic!("No-fault outcome not found for fault site {}", site.uid);
                });
                // Update the ratio
                ratio *= prob_a / prob_b;
            }

            // Update to the next sites
            if has_fault_a {
                next_fault_history_ind_a += 1;
                next_fault_a = sampled_fault_history_a.get(next_fault_history_ind_a);
                next_fault_site_a = next_fault_a.map(|f| f.site_uid);
            }
            if has_fault_b {
                next_fault_history_ind_b += 1;
                next_fault_b = sampled_fault_history_b.get(next_fault_history_ind_b);
                next_fault_site_b = next_fault_b.map(|f| f.site_uid);
            }
        }
        ratio
    }
}

/// Implements depolarizing channel noise for quantum simulations
///
/// This model applies different error probabilities to various quantum operations:
/// - `p_prep`: Preparation error probability
/// - `p_meas`: Measurement error probability
/// - `p1`: Single-qubit gate error probability
/// - `p2`: Two-qubit gate error probability
///
/// # Usage
///
/// ```rust
/// use pecos_engines::noise::DepolarizingNoiseModel;
/// use pecos_engines::noise::{NoiseModel, RngManageable};
///
/// // Create with direct constructor
/// let mut noise_model = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);
/// noise_model.set_seed(42); // For reproducibility
///
/// // Or use the builder pattern
/// let noise_model = DepolarizingNoiseModel::builder()
///     .with_p_prep(0.01)
///     .with_p_meas(0.02)
///     .with_p1(0.03)
///     .with_p2(0.04)
///     .with_seed(42)
///     .build();
///
/// // Or use uniform probability
/// let noise_model = DepolarizingNoiseModel::builder()
///     .with_uniform_probability(0.01)
///     .build();
/// ```
#[derive(Clone)]
pub struct DepolarizingNoiseModel {
    /// Probability of applying an error during preparation
    p_prep: f64,
    /// Probability of applying an error during measurement
    p_meas: f64,
    /// Probability of applying an error after single-qubit gates
    p1: f64,
    /// Probability of applying an error after two-qubit gates
    p2: f64,
    /// Precomputed threshold for preparation error probability
    p_prep_threshold: u64,
    /// Precomputed threshold for measurement error probability
    p_meas_threshold: u64,
    /// Precomputed threshold for single-qubit gate error probability
    p1_threshold: u64,
    /// Precomputed threshold for two-qubit gate error probability
    p2_threshold: u64,
    /// Random number generator
    rng: NoiseRng<PecosRng>,
    /// Scratch builder reused across batches to avoid repeated allocations.
    scratch_builder: ByteMessageBuilder,
    /// Scratch gate storage reused across batches to avoid repeated allocations.
    scratch_gates: Vec<Gate>,
    /// If True, then all faults will be cataloged
    catalog_faults_enabled: bool,
    /// Stores the catalog of faults
    catalog: Option<DepolarizingFaultCatalog>,
    /// If True, cache non-identity faults for most recent run
    sampled_fault_history_enabled: bool,
    /// Realized non-identity faults from the last `start()` call.
    sampled_fault_history: Option<Vec<DepolarizingSampledFault>>,
    /// Optional replay history to deterministically inject specific outcomes.
    replay_fault_history: Option<Vec<DepolarizingSampledFault>>,
}

impl ProbabilityValidator for DepolarizingNoiseModel {}

impl DepolarizingNoiseModel {
    fn channel_gate_error() -> PecosError {
        PecosError::Input(
            "ByteMessage noise models cannot process GateType::Channel; channel operations carry typed payloads and must use a channel-aware circuit path"
                .to_string(),
        )
    }

    /// Compute a probability threshold from a f64 probability
    #[inline]
    fn compute_threshold(p: f64) -> u64 {
        // Convert probability to fixed-point threshold for fast comparison
        // This matches the formula used in RngProbabilityExt::probability_threshold
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        {
            (p * (u64::MAX as f64)) as u64
        }
    }

    /// Create a new depolarizing noise model with the given probabilities
    #[must_use]
    pub fn new(p_prep: f64, p_meas: f64, p1: f64, p2: f64) -> Self {
        // Validate all probabilities
        Self::validate_probability(p_prep);
        Self::validate_probability(p_meas);
        Self::validate_probability(p1);
        Self::validate_probability(p2);

        Self {
            p_prep,
            p_meas,
            p1,
            p2,
            p_prep_threshold: Self::compute_threshold(p_prep),
            p_meas_threshold: Self::compute_threshold(p_meas),
            p1_threshold: Self::compute_threshold(p1),
            p2_threshold: Self::compute_threshold(p2),
            rng: NoiseRng::default(),
            scratch_builder: NoiseUtils::create_quantum_builder(),
            scratch_gates: Vec::new(),
            catalog_faults_enabled: false,
            catalog: None,
            sampled_fault_history_enabled: false,
            sampled_fault_history: None,
            replay_fault_history: None,
        }
    }

    /// Create a new noise model with uniform probability for all error types
    #[must_use]
    pub fn new_uniform(probability: f64) -> Self {
        Self::new(probability, probability, probability, probability)
    }

    /// Create a new builder for the depolarizing noise model
    #[must_use]
    pub fn builder() -> DepolarizingNoiseModelBuilder {
        DepolarizingNoiseModelBuilder::new()
    }

    /// Set all probabilities of error
    pub fn set_probabilities(&mut self, p_prep: f64, p_meas: f64, p1: f64, p2: f64) {
        Self::validate_probability(p_prep);
        Self::validate_probability(p_meas);
        Self::validate_probability(p1);
        Self::validate_probability(p2);

        self.p_prep = p_prep;
        self.p_meas = p_meas;
        self.p1 = p1;
        self.p2 = p2;

        // Recompute thresholds for optimized probability checks
        self.p_prep_threshold = Self::compute_threshold(p_prep);
        self.p_meas_threshold = Self::compute_threshold(p_meas);
        self.p1_threshold = Self::compute_threshold(p1);
        self.p2_threshold = Self::compute_threshold(p2);
    }

    /// Set a uniform probability for all error types
    pub fn set_uniform_probability(&mut self, probability: f64) {
        self.set_probabilities(probability, probability, probability, probability);
    }

    /// Get the current error probabilities
    #[must_use]
    pub fn probabilities(&self) -> (f64, f64, f64, f64) {
        (self.p_prep, self.p_meas, self.p1, self.p2)
    }

    /// Enable or disable depolarizing fault-catalog capture during `start()`.
    pub fn set_catalog_faults_enabled(&mut self, enabled: bool) {
        self.catalog_faults_enabled = enabled;
    }

    /// Returns whether fault-catalog capture is enabled.
    #[must_use]
    pub fn catalog_faults_enabled(&self) -> bool {
        self.catalog_faults_enabled
    }

    /// Returns the catalog captured during `start()`
    #[must_use]
    pub fn fault_catalog(&self) -> Option<&DepolarizingFaultCatalog> {
        self.catalog.as_ref()
    }

    /// Takes ownership of the last captured catalog.
    pub fn take_fault_catalog(&mut self) -> Option<DepolarizingFaultCatalog> {
        self.catalog.take()
    }

    /// Enable or disable sampled-fault history capture during `start()`.
    pub fn set_sampled_fault_history_enabled(&mut self, enabled: bool) {
        self.sampled_fault_history_enabled = enabled;
    }

    /// Returns whether sampled-fault history capture is enabled.
    #[must_use]
    pub fn sampled_fault_history_enabled(&self) -> bool {
        self.sampled_fault_history_enabled
    }

    /// Returns sampled non-identity faults from the last `start()`.
    #[must_use]
    pub fn sampled_fault_history(&self) -> Option<&[DepolarizingSampledFault]> {
        self.sampled_fault_history.as_deref()
    }

    /// Takes sampled non-identity faults from the last `start()`.
    pub fn take_sampled_fault_history(&mut self) -> Option<Vec<DepolarizingSampledFault>> {
        self.sampled_fault_history.take()
    }

    /// Set a deterministic fault history (to be replayed on the next `start()` call)
    pub fn set_replay_fault_history(&mut self, history: Option<Vec<DepolarizingSampledFault>>) {
        self.replay_fault_history = history;
    }

    /// Clear previously set fault history for replay
    pub fn clear_replay_fault_history(&mut self) {
        self.replay_fault_history = None;
    }

    /// Build a fault catalog from a quantum-operation message.
    ///
    /// This does not modify the simulator state and is intended for
    /// pre-sampling catalog construction.
    ///
    /// # Errors
    /// Returns [`PecosError::Input`] if the message is not quantum-operations
    /// formatted.
    pub fn build_fault_catalog_from_message(
        &self,
        input: &ByteMessage,
    ) -> Result<DepolarizingFaultCatalog, PecosError> {
        // Convert message into vector of gate object
        let mut gates = Vec::new();
        input
            .quantum_ops_into(&mut gates)
            .map_err(|e| PecosError::Input(format!("Failed to parse quantum operations: {e}")))?;

        // Build up the fault catalog from the gates
        Ok(Self::build_fault_catalog_from_gates(&self, &gates))
    }

    /// Build a fault catalog from a vector of gates
    ///
    /// This does not modify the simulator state and is intended for
    /// pre-sampling catalog construction.
    pub fn build_fault_catalog_from_gates(&self, gates: &[Gate]) -> DepolarizingFaultCatalog {
        // Create a vector to store the fault sites
        let mut fault_sites = Vec::new();

        // Track unique identifier for fault sites
        let mut next_fault_site_id = 0_usize;

        // Loop through provided gates
        for (gate_index, gate) in gates.iter().enumerate() {
            // Collect the qubits that the gate acts on
            let qubits = gate.qubits.iter().map(|q| q.0).collect::<Vec<_>>();

            // Collect gate kind and outcomes using matches of gate types
            let kind_outcomes = match gate.gate_type {
                // Single qubit gates
                GateType::X
                | GateType::Z
                | GateType::Y
                | GateType::SX
                | GateType::SXdg
                | GateType::SY
                | GateType::SYdg
                | GateType::SZ
                | GateType::SZdg
                | GateType::H
                | GateType::F
                | GateType::Fdg
                | GateType::RX
                | GateType::RY
                | GateType::RZ
                | GateType::T
                | GateType::Tdg
                | GateType::U
                | GateType::R1XY => Some((
                    DepolarizingFaultSiteKind::SingleQubit,
                    Self::single_qubit_outcomes(self.p1),
                )),
                // Two qubit gates
                GateType::CX
                | GateType::CY
                | GateType::CZ
                | GateType::CH
                | GateType::SXX
                | GateType::SXXdg
                | GateType::SYY
                | GateType::SYYdg
                | GateType::SZZ
                | GateType::SZZdg
                | GateType::SWAP
                | GateType::CRZ
                | GateType::RXX
                | GateType::RYY
                | GateType::RZZ
                | GateType::RXXRYYRZZ
                | GateType::U2q
                | GateType::CCX => Some((
                    DepolarizingFaultSiteKind::TwoQubit,
                    Self::two_qubit_outcomes(self.p2),
                )),
                // Measure
                GateType::MZ
                | GateType::MX
                | GateType::MPZ
                | GateType::MeasureLeaked
                | GateType::MeasureFree => Some((
                    DepolarizingFaultSiteKind::Meas,
                    Self::binary_x_outcomes(self.p_meas),
                )),
                // Prepare
                GateType::PZ | GateType::PX | GateType::QAlloc => Some((
                    DepolarizingFaultSiteKind::Prep,
                    Self::binary_x_outcomes(self.p_prep),
                )),
                // Gates that do not get a fault event
                GateType::Channel
                | GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree
                | GateType::Custom
                | GateType::TrackedPauliMeta => None,
            };

            // Add the new fault site
            if let Some((kind, outcomes)) = kind_outcomes {
                fault_sites.push(DepolarizingFaultSite {
                    uid: next_fault_site_id,
                    gate_index,
                    kind,
                    gate_type: gate.gate_type,
                    qubits,
                    outcomes,
                });

                // Increment the unique fault id
                next_fault_site_id += 1;
            }
        }

        DepolarizingFaultCatalog {
            sites: fault_sites,
            rng: None, // By default, the catalog does not have a random number generator; it can be set later
        }
    }

    fn binary_x_outcomes(p: f64) -> Vec<DepolarizingFaultOutcome> {
        vec![
            DepolarizingFaultOutcome {
                label: "NoFault",
                probability: 1.0 - p,
            },
            DepolarizingFaultOutcome {
                label: "X",
                probability: p,
            },
        ]
    }

    fn single_qubit_outcomes(p: f64) -> Vec<DepolarizingFaultOutcome> {
        let branch = p / 3.0;
        vec![
            DepolarizingFaultOutcome {
                label: "NoFault",
                probability: 1.0 - p,
            },
            DepolarizingFaultOutcome {
                label: "X",
                probability: branch,
            },
            DepolarizingFaultOutcome {
                label: "Y",
                probability: branch,
            },
            DepolarizingFaultOutcome {
                label: "Z",
                probability: branch,
            },
        ]
    }

    fn two_qubit_outcomes(p: f64) -> Vec<DepolarizingFaultOutcome> {
        const LABELS: [&str; 15] = [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY",
            "ZZ",
        ];

        let mut outcomes = Vec::with_capacity(16);
        outcomes.push(DepolarizingFaultOutcome {
            label: "NoFault",
            probability: 1.0 - p,
        });

        let branch = p / 15.0;
        for label in LABELS {
            outcomes.push(DepolarizingFaultOutcome {
                label,
                probability: branch,
            });
        }

        outcomes
    }

    fn apply_noise_to_gate(
        rng: &mut NoiseRng<PecosRng>,
        p_prep_threshold: u64,
        p_meas_threshold: u64,
        p1_threshold: u64,
        p2_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        // Tracks the fault site unique identifier for the catalog
        next_fault_site_uid: &mut usize,
        // TODO Why do these need to be passed in instead of taken from self?
        sampled_fault_history_enabled: bool,
        sampled_fault_history: &mut Vec<DepolarizingSampledFault>,
        replay_outcomes_by_site: Option<&BTreeMap<usize, u8>>,
    ) {
        let mut sampled_outcome: Option<(usize, u8, &'static str)> = None;

        match gate.gate_type {
            GateType::X
            | GateType::Z
            | GateType::Y
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::H
            | GateType::F
            | GateType::Fdg
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::T
            | GateType::Tdg
            | GateType::U
            | GateType::R1XY => {
                // Apply ideal gate
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying single-qubit gate with possible fault");
                // Apply noise after the gate and cache the noise event
                let site_uid = *next_fault_site_uid;
                *next_fault_site_uid += 1;
                // While replaying, every site is forced (absent sites force no-fault) so no RNG is consumed.
                let forced_outcome = replay_outcomes_by_site
                    .map(|replay| replay.get(&site_uid).copied().unwrap_or(0));
                if let Some((outcome_index, outcome_label)) =
                    Self::apply_sq_faults(rng, p1_threshold, builder, gate, forced_outcome)
                {
                    sampled_outcome = Some((site_uid, outcome_index, outcome_label));
                }
            }
            GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::CH
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
            | GateType::CRZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::RXXRYYRZZ
            | GateType::U2q => {
                // Apply ideal gate
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying two-qubit gate with possible fault");
                // Applying noise after the gate and cache the noise event
                let site_uid = *next_fault_site_uid;
                *next_fault_site_uid += 1;
                // While replaying, every site is forced (absent sites force no-fault) so no RNG is consumed.
                let forced_outcome = replay_outcomes_by_site
                    .map(|replay| replay.get(&site_uid).copied().unwrap_or(0));
                if let Some((outcome_index, outcome_label)) =
                    Self::apply_tq_faults(rng, p2_threshold, builder, gate, forced_outcome)
                {
                    sampled_outcome = Some((site_uid, outcome_index, outcome_label));
                }
            }
            GateType::CCX => {
                // Apply ideal gate
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying three-qubit gate with possible fault");
                // Apply noise after the gate and cache the noise event
                let site_uid = *next_fault_site_uid;
                *next_fault_site_uid += 1;
                // While replaying, every site is forced (absent sites force no-fault) so no RNG is consumed.
                let forced_outcome = replay_outcomes_by_site
                    .map(|replay| replay.get(&site_uid).copied().unwrap_or(0));
                if let Some((outcome_index, outcome_label)) =
                    Self::apply_tq_faults(rng, p2_threshold, builder, gate, forced_outcome)
                {
                    sampled_outcome = Some((site_uid, outcome_index, outcome_label));
                }
            }
            // Measure-and-prep draws the measurement-half fault only, so
            // every noise path (engines, DEM builders, eeg) models MPZ
            // identically; the prepare-half lands with the dedicated
            // measure-prepare channel across all of them at onces
            GateType::MPZ
            | GateType::MX
            | GateType::MZ
            | GateType::MeasureLeaked
            | GateType::MeasureFree => {
                if gate.gate_type != GateType::MPZ {
                    trace!("Applying measurement with possible fault");
                }
                // Apply noise before the gate and cache the noise event
                let site_uid = *next_fault_site_uid;
                *next_fault_site_uid += 1;
                // While replaying, every site is forced (absent sites force no-fault) so no RNG is consumed.
                let forced_outcome = replay_outcomes_by_site
                    .map(|replay| replay.get(&site_uid).copied().unwrap_or(0));
                if let Some((outcome_index, outcome_label)) =
                    Self::apply_meas_faults(rng, p_meas_threshold, builder, gate, forced_outcome)
                {
                    sampled_outcome = Some((site_uid, outcome_index, outcome_label));
                }
                // Apply the ideal measurement
                NoiseUtils::add_gate_to_builder(builder, gate);
            }
            GateType::PX | GateType::PZ | GateType::QAlloc => {
                // Apply the ideal preparation gate
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying preparation with possible fault");
                // Apply noise after the prep and cache the noise event
                let site_uid = *next_fault_site_uid;
                *next_fault_site_uid += 1;
                // While replaying, every site is forced (absent sites force no-fault) so no RNG is consumed.
                let forced_outcome = replay_outcomes_by_site
                    .map(|replay| replay.get(&site_uid).copied().unwrap_or(0));
                if let Some((outcome_index, outcome_label)) =
                    Self::apply_prep_faults(rng, p_prep_threshold, builder, gate, forced_outcome)
                {
                    sampled_outcome = Some((site_uid, outcome_index, outcome_label));
                }
            }
            GateType::Channel => unreachable!("channel gates are rejected before noise is applied"),
            GateType::I
            | GateType::Idle
            | GateType::MeasCrosstalkLocalPayload
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::QFree
            | GateType::Custom
            | GateType::TrackedPauliMeta => {
                // Just pass through with no added noise.
            }
        }

        if sampled_fault_history_enabled {
            if let Some((site_uid, outcome_index, outcome_label)) = sampled_outcome {
                sampled_fault_history.push(DepolarizingSampledFault {
                    site_uid,
                    outcome_index,
                    outcome_label,
                });
            }
        }
    }

    fn apply_prep_faults(
        rng: &mut NoiseRng<PecosRng>,
        p_prep_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        forced_outcome: Option<u8>,
    ) -> Option<(u8, &'static str)> {
        let apply_fault = match forced_outcome {
            Some(1) => true,
            Some(_) => false,
            None => rng.inner_mut().check_probability(p_prep_threshold),
        };

        if apply_fault {
            trace!("Applying prep fault on qubits {:?}", gate.qubits);
            match gate.gate_type {
                GateType::PX => {
                    NoiseUtils::apply_z(builder, *gate.qubits[0]);
                    return Some((1, "Z"));
                }
                _ => {
                    NoiseUtils::apply_x(builder, *gate.qubits[0]);
                    return Some((1, "X"));
                }
            }
        }
        None
    }

    fn apply_meas_faults(
        rng: &mut NoiseRng<PecosRng>,
        p_meas_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        forced_outcome: Option<u8>,
    ) -> Option<(u8, &'static str)> {
        let apply_fault = match forced_outcome {
            Some(1) => true,
            Some(_) => false,
            None => rng.inner_mut().check_probability(p_meas_threshold),
        };

        if apply_fault {
            trace!("Applying meas fault on qubits {:?}", gate.qubits);
            match gate.gate_type {
                GateType::MX => {
                    NoiseUtils::apply_z(builder, *gate.qubits[0]);
                    return Some((1, "Z"));
                }
                _ => {
                    NoiseUtils::apply_x(builder, *gate.qubits[0]);
                    return Some((1, "X"));
                }
            }
        }

        None
    }

    fn apply_sq_outcome(
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        outcome_index: u8,
    ) -> Option<(u8, &'static str)> {
        let qubit = gate.qubits[0];

        match outcome_index {
            1 => {
                NoiseUtils::apply_x(builder, *qubit);
                Some((1, "X"))
            }
            2 => {
                NoiseUtils::apply_y(builder, *qubit);
                Some((2, "Y"))
            }
            3 => {
                NoiseUtils::apply_z(builder, *qubit);
                Some((3, "Z"))
            }
            _ => None,
        }
    }

    fn apply_tq_outcome(
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        outcome_index: u8,
    ) -> Option<(u8, &'static str)> {
        let qubit0 = gate.qubits[0];
        let qubit1 = gate.qubits[1];

        match outcome_index {
            1 => {
                NoiseUtils::apply_x(builder, *qubit1);
                Some((1, "IX"))
            }
            2 => {
                NoiseUtils::apply_y(builder, *qubit1);
                Some((2, "IY"))
            }
            3 => {
                NoiseUtils::apply_z(builder, *qubit1);
                Some((3, "IZ"))
            }
            4 => {
                NoiseUtils::apply_x(builder, *qubit0);
                Some((4, "XI"))
            }
            5 => {
                NoiseUtils::apply_x(builder, *qubit0);
                NoiseUtils::apply_x(builder, *qubit1);
                Some((5, "XX"))
            }
            6 => {
                NoiseUtils::apply_x(builder, *qubit0);
                NoiseUtils::apply_y(builder, *qubit1);
                Some((6, "XY"))
            }
            7 => {
                NoiseUtils::apply_x(builder, *qubit0);
                NoiseUtils::apply_z(builder, *qubit1);
                Some((7, "XZ"))
            }
            8 => {
                NoiseUtils::apply_y(builder, *qubit0);
                Some((8, "YI"))
            }
            9 => {
                NoiseUtils::apply_y(builder, *qubit0);
                NoiseUtils::apply_x(builder, *qubit1);
                Some((9, "YX"))
            }
            10 => {
                NoiseUtils::apply_y(builder, *qubit0);
                NoiseUtils::apply_y(builder, *qubit1);
                Some((10, "YY"))
            }
            11 => {
                NoiseUtils::apply_y(builder, *qubit0);
                NoiseUtils::apply_z(builder, *qubit1);
                Some((11, "YZ"))
            }
            12 => {
                NoiseUtils::apply_z(builder, *qubit0);
                Some((12, "ZI"))
            }
            13 => {
                NoiseUtils::apply_z(builder, *qubit0);
                NoiseUtils::apply_x(builder, *qubit1);
                Some((13, "ZX"))
            }
            14 => {
                NoiseUtils::apply_z(builder, *qubit0);
                NoiseUtils::apply_y(builder, *qubit1);
                Some((14, "ZY"))
            }
            15 => {
                NoiseUtils::apply_z(builder, *qubit0);
                NoiseUtils::apply_z(builder, *qubit1);
                Some((15, "ZZ"))
            }
            _ => None,
        }
    }

    fn apply_sq_faults(
        rng: &mut NoiseRng<PecosRng>,
        p1_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        forced_outcome: Option<u8>,
    ) -> Option<(u8, &'static str)> {
        // A forced outcome (including a forced no-fault of 0) must skip the RNG
        // draw entirely, or replay would desync from the recorded history.
        let outcome_index = match forced_outcome {
            Some(outcome) => Some(outcome),
            None => rng.inner_mut().noise_sample_1q(p1_threshold).map(|v| v + 1),
        };
        outcome_index.and_then(|outcome| Self::apply_sq_outcome(builder, gate, outcome))
    }

    fn apply_tq_faults(
        rng: &mut NoiseRng<PecosRng>,
        p2_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
        forced_outcome: Option<u8>,
    ) -> Option<(u8, &'static str)> {
        // A forced outcome (including a forced no-fault of 0) must skip the RNG
        // draw entirely, or replay would desync from the recorded history.
        let outcome_index = match forced_outcome {
            Some(outcome) => Some(outcome),
            None => rng.inner_mut().noise_sample_2q(p2_threshold).map(|v| v + 1),
        };
        outcome_index.and_then(|outcome| Self::apply_tq_outcome(builder, gate, outcome))
    }
}

impl NoiseModel for DepolarizingNoiseModel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl RngManageable for DepolarizingNoiseModel {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: PecosRng) {
        self.rng = NoiseRng::new(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.rng.inner()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.rng.inner_mut()
    }
}

/// Builder for creating depolarizing noise models.
///
/// The retired descriptive probability setters are intentionally unavailable:
///
/// ```compile_fail
/// use pecos_engines::noise::DepolarizingNoiseModel;
/// let _ = DepolarizingNoiseModel::builder().with_single_qubit_probability(0.01);
/// ```
///
/// ```compile_fail
/// use pecos_engines::noise::DepolarizingNoiseModel;
/// let _ = DepolarizingNoiseModel::builder().with_two_qubit_probability(0.01);
/// ```
#[derive(Debug, Clone)]
pub struct DepolarizingNoiseModelBuilder {
    p_prep: Option<f64>,
    p_meas: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
    seed: Option<u64>,
}

impl Default for DepolarizingNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DepolarizingNoiseModelBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            p_prep: None,
            p_meas: None,
            p1: None,
            p2: None,
            seed: None,
        }
    }

    /// Set the same probability for all error types
    ///
    /// This is a convenience method to set all probabilities to the same value.
    ///
    /// # Arguments
    /// * `probability` - The probability value to set for all error types
    #[must_use]
    pub fn with_uniform_probability(mut self, probability: f64) -> Self {
        self.p_prep = Some(probability);
        self.p_meas = Some(probability);
        self.p1 = Some(probability);
        self.p2 = Some(probability);
        self
    }

    /// Set the probability of error during preparation
    #[must_use]
    pub fn with_p_prep(mut self, probability: f64) -> Self {
        self.p_prep = Some(probability);
        self
    }

    /// Set the probability of error during measurement
    #[must_use]
    pub fn with_p_meas(mut self, probability: f64) -> Self {
        self.p_meas = Some(probability);
        self
    }

    /// Set the probability of error after single-qubit gates
    #[must_use]
    pub fn with_p1(mut self, probability: f64) -> Self {
        self.p1 = Some(probability);
        self
    }

    /// Set the probability of error after two-qubit gates
    #[must_use]
    pub fn with_p2(mut self, probability: f64) -> Self {
        self.p2 = Some(probability);
        self
    }

    /// Set the seed for the random number generator
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the depolarizing noise model
    ///
    /// # Returns
    /// A `DepolarizingNoiseModel` instance
    ///
    /// # Panics
    /// Panics if any probabilities are not between 0 and 1.
    #[must_use]
    pub fn build(self) -> DepolarizingNoiseModel {
        let p_prep = self.p_prep.expect("Preparation probability must be set");
        let p_meas = self.p_meas.expect("Measurement probability must be set");
        let p1 = self.p1.expect("Single-qubit probability must be set");
        let p2 = self.p2.expect("Two-qubit probability must be set");

        // Create the noise model
        let mut noise = DepolarizingNoiseModel::new(p_prep, p_meas, p1, p2);

        // Set the seed if provided
        if let Some(seed) = self.seed {
            // Use RngManageable::set_seed directly
            noise.set_seed(seed);
        }

        noise
    }
}

impl crate::noise::IntoNoiseModel for DepolarizingNoiseModelBuilder {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(self.build())
    }
}

impl ControlEngine for DepolarizingNoiseModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // For quantum operations, apply gate noise
        trace!("DepolarizingNoise::start - applying noise to quantum operations");

        self.scratch_gates.clear();
        input
            .quantum_ops_into(&mut self.scratch_gates)
            .map_err(|e| PecosError::Input(format!("Failed to parse quantum operations: {e}")))?;

        if self.scratch_gates.iter().any(Gate::is_channel) {
            return Err(Self::channel_gate_error());
        }

        self.sampled_fault_history = None;

        // Initialize an empty fault catalog
        if self.catalog_faults_enabled {
            self.catalog = Some(Self::build_fault_catalog_from_gates(
                &self,
                &self.scratch_gates,
            ));
        } else {
            self.catalog = None;
        }

        if self.p_prep_threshold == 0
            && self.p_meas_threshold == 0
            && self.p1_threshold == 0
            && self.p2_threshold == 0
            // Also check if there is no replay fault history, in which case we can skip processing
            && self.replay_fault_history.is_none()
        {
            return Ok(EngineStage::NeedsProcessing(input));
        }

        self.scratch_builder.reset();
        let _ = self.scratch_builder.for_quantum_operations();

        let p_prep_threshold = self.p_prep_threshold;
        let p_meas_threshold = self.p_meas_threshold;
        let p1_threshold = self.p1_threshold;
        let p2_threshold = self.p2_threshold;
        let rng = &mut self.rng;
        let builder = &mut self.scratch_builder;
        let mut sampled_fault_history = Vec::new();
        let mut next_fault_site_uid = 0_usize;

        // Get the replay history
        let replay_outcomes_by_site = self.replay_fault_history.as_ref().map(|history| {
            history
                .iter()
                .map(|fault| (fault.site_uid, fault.outcome_index))
                .collect::<BTreeMap<usize, u8>>()
        });

        for gate in &self.scratch_gates {
            Self::apply_noise_to_gate(
                rng,
                p_prep_threshold,
                p_meas_threshold,
                p1_threshold,
                p2_threshold,
                builder,
                gate,
                &mut next_fault_site_uid,
                self.sampled_fault_history_enabled,
                &mut sampled_fault_history,
                replay_outcomes_by_site.as_ref(),
            );
        }

        self.sampled_fault_history = if self.sampled_fault_history_enabled {
            Some(sampled_fault_history)
        } else {
            None
        };

        let noisy_gates = self.scratch_builder.build();

        // Return the noisy operations
        Ok(EngineStage::NeedsProcessing(noisy_gates))
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // This noise model doesn't directly modify measurement results, just pass through
        trace!("DepolarizingNoise::continue_processing - passing through measurement results");
        Ok(EngineStage::Complete(result))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // No state to reset
        self.catalog = None;
        self.sampled_fault_history = None;
        self.replay_fault_history = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_system::{ControlEngine, EngineStage};

    #[test]
    fn test_probabilities_getter_and_setter() {
        // Create a noise model with initial probabilities
        let mut noise = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);

        // Check initial probabilities
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.01).abs() < f64::EPSILON);
        assert!((p_meas - 0.02).abs() < f64::EPSILON);
        assert!((p1 - 0.03).abs() < f64::EPSILON);
        assert!((p2 - 0.04).abs() < f64::EPSILON);

        // Update probabilities and check they were updated
        noise.set_probabilities(0.05, 0.06, 0.07, 0.08);
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.05).abs() < f64::EPSILON);
        assert!((p_meas - 0.06).abs() < f64::EPSILON);
        assert!((p1 - 0.07).abs() < f64::EPSILON);
        assert!((p2 - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn test_uniform_probability() {
        // Test the uniform probability constructor
        let noise = DepolarizingNoiseModel::new_uniform(0.05);
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.05).abs() < f64::EPSILON);
        assert!((p_meas - 0.05).abs() < f64::EPSILON);
        assert!((p1 - 0.05).abs() < f64::EPSILON);
        assert!((p2 - 0.05).abs() < f64::EPSILON);

        // Test the uniform probability setter
        let mut noise = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);
        noise.set_uniform_probability(0.07);
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.07).abs() < f64::EPSILON);
        assert!((p_meas - 0.07).abs() < f64::EPSILON);
        assert!((p1 - 0.07).abs() < f64::EPSILON);
        assert!((p2 - 0.07).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_invalid_probability_panics() {
        let mut noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);
        noise.set_probabilities(0.1, 0.2, 1.1, 0.4); // Should panic
    }

    #[test]
    fn test_builder() {
        // Create a noise model with the builder
        let mut noise = DepolarizingNoiseModel::builder()
            .with_p_prep(0.1)
            .with_p_meas(0.2)
            .with_p1(0.3)
            .with_p2(0.4)
            .build();

        // Create a direct instance with the same probabilities
        let mut direct_noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        // Create a simple message for testing
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        let input = builder.build();

        // Process using the ControlEngine API instead of the old apply_noise method
        let result1 = noise
            .start(input.clone())
            .expect("Builder-created noise model failed");
        let result2 = direct_noise
            .start(input)
            .expect("Directly created noise model failed");

        // Verify we got a valid result that needs processing
        match result1 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }

        match result2 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }
    }

    #[test]
    fn test_builder_with_uniform_probability() {
        // Create a noise model with the builder using uniform probability
        let noise = DepolarizingNoiseModel::builder()
            .with_uniform_probability(0.05)
            .build();

        // Create a direct instance with the same uniform probability
        let direct_noise = DepolarizingNoiseModel::new_uniform(0.05);

        // Check that probabilities match
        let (p_prep1, p_meas1, p1_1, p2_1) = direct_noise.probabilities();

        // Get the boxed noise model's probabilities using any_ref downcast
        let noise_ref = noise
            .as_any()
            .downcast_ref::<DepolarizingNoiseModel>()
            .unwrap();
        let (p_prep2, p_meas2, p1_2, p2_2) = noise_ref.probabilities();

        assert!((p_prep1 - p_prep2).abs() < f64::EPSILON);
        assert!((p_meas1 - p_meas2).abs() < f64::EPSILON);
        assert!((p1_1 - p1_2).abs() < f64::EPSILON);
        assert!((p2_1 - p2_2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_as_any_methods() {
        // Create a noise model
        let mut noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        // Test as_any for type checking
        assert!(noise.as_any().is::<DepolarizingNoiseModel>());

        // Test as_any_mut for downcasting and modifying
        let downcast_noise = noise
            .as_any_mut()
            .downcast_mut::<DepolarizingNoiseModel>()
            .unwrap();
        downcast_noise.set_probabilities(0.5, 0.5, 0.5, 0.5);

        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.5).abs() < f64::EPSILON);
        assert!((p_meas - 0.5).abs() < f64::EPSILON);
        assert!((p1 - 0.5).abs() < f64::EPSILON);
        assert!((p2 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn field_name_setters_match_pre_removal_alias_bytes() {
        let mut noise = DepolarizingNoiseModel::builder()
            .with_p_prep(0.0)
            .with_p_meas(0.0)
            .with_p1(1.0)
            .with_p2(1.0)
            .with_seed(0x5eed)
            .build();

        let mut builder = ByteMessage::quantum_operations_builder();
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);

        let EngineStage::NeedsProcessing(output) = noise.start(builder.build()).unwrap() else {
            panic!("noise model unexpectedly completed");
        };
        assert_eq!(
            output.as_bytes(),
            [
                83, 67, 69, 80, 1, 0, 0, 0, 4, 0, 0, 0, 84, 0, 0, 0, 10, 0, 0, 0, 8, 0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 8, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 12,
                0, 0, 0, 50, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0, 0, 8, 0, 0, 0, 2, 1, 0, 0,
                1, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn test_builder_with_probability() {
        // Create a noise model with the builder
        let mut noise = DepolarizingNoiseModel::builder()
            .with_p_prep(0.01)
            .with_p_meas(0.02)
            .with_p1(0.03)
            .with_p2(0.04)
            .build();

        // Create a direct instance with the same probabilities
        let mut direct_noise = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);

        // Create a simple quantum operations message for testing
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        let input = builder.build();

        // Process using the ControlEngine API instead of the old apply_noise method
        let result1 = noise
            .start(input.clone())
            .expect("Builder-created noise model failed");
        let result2 = direct_noise
            .start(input)
            .expect("Directly created noise model failed");

        // Verify we got a valid result that needs processing
        match result1 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }

        match result2 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }
    }

    #[test]
    fn test_fault_catalog_from_message_basic_attributes() {
        // Check that the catalog has the correct information stored
        // in it from a very simple circuit
        let noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        // Build a simple circuit
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);

        let msg = builder.build();

        // Try to build the catalog
        let catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        // Check that the catalog has the correct information stored
        assert_eq!(catalog.sites.len(), 4);

        // Check that the sites have the correct kind
        assert_eq!(catalog.sites[0].kind, DepolarizingFaultSiteKind::Prep);
        assert_eq!(
            catalog.sites[1].kind,
            DepolarizingFaultSiteKind::SingleQubit
        );
        assert_eq!(catalog.sites[2].kind, DepolarizingFaultSiteKind::TwoQubit);
        assert_eq!(catalog.sites[3].kind, DepolarizingFaultSiteKind::Meas);

        // Check that the sites have the correct unique ids
        assert_eq!(catalog.sites[0].uid, 0);
        assert_eq!(catalog.sites[1].uid, 1);
        assert_eq!(catalog.sites[2].uid, 2);
        assert_eq!(catalog.sites[3].uid, 3);

        // Check that the gates have been indexed correct
        assert_eq!(catalog.sites[0].gate_index, 0);
        assert_eq!(catalog.sites[1].gate_index, 1);
        assert_eq!(catalog.sites[2].gate_index, 2);
        assert_eq!(catalog.sites[3].gate_index, 3);

        // Check that the gate types have been indexed correctly
        assert_eq!(catalog.sites[0].gate_type, GateType::PZ);
        assert_eq!(catalog.sites[1].gate_type, GateType::X);
        assert_eq!(catalog.sites[2].gate_type, GateType::CX);
        assert_eq!(catalog.sites[3].gate_type, GateType::MZ);

        // Check that the qubits are stored correctly
        assert_eq!(catalog.sites[0].qubits, vec![0]);
        assert_eq!(catalog.sites[1].qubits, vec![0]);
        assert_eq!(catalog.sites[2].qubits, vec![0, 1]);
        assert_eq!(catalog.sites[3].qubits, vec![1]);

        // Check that the outcomes are correct
        assert_eq!(catalog.sites[0].outcomes.len(), 2);
        assert_eq!(catalog.sites[1].outcomes.len(), 4);
        assert_eq!(catalog.sites[2].outcomes.len(), 16);
        assert_eq!(catalog.sites[3].outcomes.len(), 2);

        assert_eq!(catalog.sites[0].outcomes[0].label, "NoFault");
        assert_eq!(catalog.sites[0].outcomes[1].label, "X");

        assert_eq!(catalog.sites[1].outcomes[0].label, "NoFault");
        assert_eq!(catalog.sites[1].outcomes[1].label, "X");
        assert_eq!(catalog.sites[1].outcomes[2].label, "Y");
        assert_eq!(catalog.sites[1].outcomes[3].label, "Z");

        assert_eq!(catalog.sites[2].outcomes[0].label, "NoFault");
        assert_eq!(catalog.sites[2].outcomes[1].label, "IX");
        assert_eq!(catalog.sites[2].outcomes[2].label, "IY");
        assert_eq!(catalog.sites[2].outcomes[3].label, "IZ");
        assert_eq!(catalog.sites[2].outcomes[4].label, "XI");
        assert_eq!(catalog.sites[2].outcomes[5].label, "XX");
        assert_eq!(catalog.sites[2].outcomes[6].label, "XY");
        assert_eq!(catalog.sites[2].outcomes[7].label, "XZ");
        assert_eq!(catalog.sites[2].outcomes[8].label, "YI");
        assert_eq!(catalog.sites[2].outcomes[9].label, "YX");
        assert_eq!(catalog.sites[2].outcomes[10].label, "YY");
        assert_eq!(catalog.sites[2].outcomes[11].label, "YZ");
        assert_eq!(catalog.sites[2].outcomes[12].label, "ZI");
        assert_eq!(catalog.sites[2].outcomes[13].label, "ZX");
        assert_eq!(catalog.sites[2].outcomes[14].label, "ZY");
        assert_eq!(catalog.sites[2].outcomes[15].label, "ZZ");
    }

    #[test]
    fn test_fault_catalog_probabilities_sum_to_one() {
        // Checks that all the fault catalog outcomes
        // sum to 1
        let noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.45);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        builder.pz(&[2]);

        let msg = builder.build();
        let catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        for site in &catalog.sites {
            let sum: f64 = site.outcomes.iter().map(|o| o.probability).sum();
            assert!((sum - 1.0).abs() < 1e-12, "outcomes must sum to one");
            // Check that the first outcome is always NoFault
            assert_eq!(site.outcomes[0].label, "NoFault");
        }
    }

    #[test]
    fn test_fault_catalog_capture_in_start() {
        // Check that the fault catalog is captured when enabled
        let mut noise = DepolarizingNoiseModel::new_uniform(0.0);
        noise.set_catalog_faults_enabled(true);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        let msg = builder.build();

        let _ = noise.start(msg).expect("noise start should succeed");

        let catalog = noise
            .fault_catalog()
            .expect("catalog should be captured when enabled");
        assert_eq!(catalog.sites.len(), 2);
    }

    #[test]
    fn test_sampled_fault_history_tracks_non_identity_faults() {
        // Checks that when you force errors, they are correctly cached in the sampled fault history
        // TODO Check what this test is doing
        let mut noise = DepolarizingNoiseModel::new_uniform(1.0);
        noise.set_seed(7);
        noise.set_sampled_fault_history_enabled(true);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let _ = noise.start(msg).expect("noise start should succeed");

        let history = noise
            .sampled_fault_history()
            .expect("sampled fault history should exist when enabled");
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].site_uid, 0);
        assert_eq!(history[0].outcome_label, "X");
        assert_eq!(history[1].site_uid, 1);
        assert!(matches!(history[1].outcome_label, "X" | "Y" | "Z"));
        assert_eq!(history[2].site_uid, 2);
        assert_eq!(history[3].site_uid, 3);
        assert_eq!(history[3].outcome_label, "X");
    }

    #[test]
    fn test_replay_fault_history_forces_specified_outcome() {
        // Test that if we specify a fault history, then that is the one that
        // is executed on replay
        let mut noise = DepolarizingNoiseModel::new_uniform(0.0);
        noise.set_sampled_fault_history_enabled(true);
        noise.set_replay_fault_history(Some(vec![DepolarizingSampledFault {
            site_uid: 0,
            outcome_index: 5,
            outcome_label: "XX",
        }]));

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.cx(&[(0, 1)]);
        let msg = builder.build();

        let _ = noise.start(msg).expect("noise start should succeed");

        let history = noise
            .sampled_fault_history()
            .expect("history should exist when enabled");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].site_uid, 0);
        assert_eq!(history[0].outcome_index, 5);
        assert_eq!(history[0].outcome_label, "XX");
    }

    #[test]
    fn test_replay_fault_history_uses_expected_site_indices() {
        // Check a more complex replay history with multiple sites
        // and with skipped fault sites
        let mut noise = DepolarizingNoiseModel::new_uniform(0.0);
        noise.set_sampled_fault_history_enabled(true);

        noise.set_replay_fault_history(Some(vec![
            DepolarizingSampledFault {
                site_uid: 0,
                outcome_index: 1,
                outcome_label: "X",
            },
            DepolarizingSampledFault {
                site_uid: 2,
                outcome_index: 15,
                outcome_label: "ZZ",
            },
            DepolarizingSampledFault {
                site_uid: 3,
                outcome_index: 1,
                outcome_label: "X",
            },
        ]));

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let _ = noise.start(msg).expect("noise start should succeed");

        let history = noise
            .sampled_fault_history()
            .expect("history should exist when enabled");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].site_uid, 0);
        assert_eq!(history[1].site_uid, 2);
        assert_eq!(history[2].site_uid, 3);
        assert_eq!(history[0].outcome_label, "X");
        assert_eq!(history[1].outcome_label, "ZZ");
        assert_eq!(history[2].outcome_label, "X");
    }

    #[test]
    fn test_sampled_and_replayed_histories_match() {
        // Checks that if we sample a fault history and then replay it, we get the same history back
        let mut source = DepolarizingNoiseModel::new_uniform(0.5);
        source.set_seed(11);
        source.set_sampled_fault_history_enabled(true);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let _ = source
            .start(msg.clone())
            .expect("source start should succeed");
        let sampled_history = source
            .sampled_fault_history()
            .expect("sampled history should exist")
            .to_vec();

        let mut replay = DepolarizingNoiseModel::new_uniform(0.0);
        replay.set_sampled_fault_history_enabled(true);
        replay.set_replay_fault_history(Some(sampled_history.clone()));
        let _ = replay.start(msg).expect("replay start should succeed");

        let replayed_history = replay
            .sampled_fault_history()
            .expect("replayed history should exist")
            .to_vec();

        assert_eq!(replayed_history, sampled_history);
    }

    #[test]
    fn test_catalog_derived_history_replay_is_repeatable() {
        let mut noise = DepolarizingNoiseModel::new_uniform(0.4);
        noise.set_sampled_fault_history_enabled(true);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        let derived_history: Vec<DepolarizingSampledFault> = catalog
            .sites
            .iter()
            .filter_map(|site| {
                site.outcomes
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find(|(_, outcome)| outcome.probability > 0.0)
                    .and_then(|(idx, outcome)| {
                        u8::try_from(idx)
                            .ok()
                            .map(|outcome_index| DepolarizingSampledFault {
                                site_uid: site.uid,
                                outcome_index,
                                outcome_label: outcome.label,
                            })
                    })
            })
            .collect();

        assert_eq!(derived_history.len(), catalog.sites.len());

        noise.set_replay_fault_history(Some(derived_history.clone()));
        let first_stage = noise
            .start(msg.clone())
            .expect("first replay start should succeed");
        let first_noisy = match first_stage {
            EngineStage::NeedsProcessing(noisy) => noisy,
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        };
        let first_history = noise
            .sampled_fault_history()
            .expect("first replay history should exist")
            .to_vec();

        noise.set_replay_fault_history(Some(derived_history));
        let second_stage = noise
            .start(msg)
            .expect("second replay start should succeed");
        let second_noisy = match second_stage {
            EngineStage::NeedsProcessing(noisy) => noisy,
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        };
        let second_history = noise
            .sampled_fault_history()
            .expect("second replay history should exist")
            .to_vec();

        assert_eq!(first_noisy.as_bytes(), second_noisy.as_bytes());
        assert_eq!(first_history, second_history);
    }

    #[test]
    fn test_empty_replay_matches_no_fault_run_message() {
        // Ensures that an empty replay history produces the same output as a run with no faults
        let mut no_replay = DepolarizingNoiseModel::new_uniform(0.0);
        let mut with_empty_replay = DepolarizingNoiseModel::new_uniform(0.0);
        with_empty_replay.set_replay_fault_history(Some(Vec::new()));

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let stage_no_replay = no_replay
            .start(msg.clone())
            .expect("no-replay start should succeed");
        let out_no_replay = match stage_no_replay {
            EngineStage::NeedsProcessing(noisy) => noisy,
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        };

        let stage_empty_replay = with_empty_replay
            .start(msg)
            .expect("empty-replay start should succeed");
        let out_empty_replay = match stage_empty_replay {
            EngineStage::NeedsProcessing(noisy) => noisy,
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        };

        assert_eq!(out_no_replay.as_bytes(), out_empty_replay.as_bytes());
    }

    // Tests that sums of all possible fault histories add to 1
    #[test]
    fn test_fault_history_probability_sums_to_one() {
        // Specify a circuit
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        // Build the noise model and fault catalog
        let source = DepolarizingNoiseModel::new_uniform(0.5);
        let catalog = source
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        let mut prob = 0.0;
        for outcome1 in &catalog.sites[0].outcomes {
            for outcome2 in &catalog.sites[1].outcomes {
                for outcome3 in &catalog.sites[2].outcomes {
                    for outcome4 in &catalog.sites[3].outcomes {
                        let mut history = Vec::new();
                        if outcome1.label != "NoFault" {
                            let index = catalog.sites[0]
                                .outcomes
                                .iter()
                                .position(|o| o.label == outcome1.label)
                                .unwrap() as u8;
                            history.push(DepolarizingSampledFault {
                                site_uid: 0,
                                outcome_index: index,
                                outcome_label: &outcome1.label,
                            });
                        }
                        if outcome2.label != "NoFault" {
                            let index = catalog.sites[1]
                                .outcomes
                                .iter()
                                .position(|o| o.label == outcome2.label)
                                .unwrap() as u8;
                            history.push(DepolarizingSampledFault {
                                site_uid: 1,
                                outcome_index: index,
                                outcome_label: &outcome2.label,
                            });
                        }
                        if outcome3.label != "NoFault" {
                            let index = catalog.sites[2]
                                .outcomes
                                .iter()
                                .position(|o| o.label == outcome3.label)
                                .unwrap() as u8;
                            history.push(DepolarizingSampledFault {
                                site_uid: 2,
                                outcome_index: index,
                                outcome_label: &outcome3.label,
                            });
                        }
                        if outcome4.label != "NoFault" {
                            let index = catalog.sites[3]
                                .outcomes
                                .iter()
                                .position(|o| o.label == outcome4.label)
                                .unwrap() as u8;
                            history.push(DepolarizingSampledFault {
                                site_uid: 3,
                                outcome_index: index,
                                outcome_label: &outcome4.label,
                            });
                        }
                        let history_prob = catalog.fault_history_probability(&history);
                        prob += history_prob;
                    }
                }
            }
        }
        let tolerance = 1e-12;
        assert!(
            (prob - 1.0).abs() < tolerance,
            "total probability should sum to 1"
        );
    }

    // Tests that probabilities are correctly computed when compared to known examples
    #[test]
    fn test_fault_history_probability_computation() {
        // Specify a circuit
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        // Create a noise model where no faults are allowed and
        // check that the probability of the empty history is 1
        // and any other histories have probability 0
        let source = DepolarizingNoiseModel::new_uniform(0.0);
        let catalog = source
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");
        let empty_history: Vec<DepolarizingSampledFault> = Vec::new();

        let prob_empty = catalog.fault_history_probability(&empty_history);
        assert!(
            (prob_empty - 1.0).abs() < f64::EPSILON,
            "empty history should have probability 1"
        );

        let non_empty_history = vec![DepolarizingSampledFault {
            site_uid: 0,
            outcome_index: 1,
            outcome_label: "X",
        }];
        let prob_non_empty = catalog.fault_history_probability(&non_empty_history);
        assert!(
            (prob_non_empty).abs() < f64::EPSILON,
            "non-empty history should have probability 0"
        );

        // Create a noise model where faults are forced
        // and check that the probability of the empty history is 0
        let source = DepolarizingNoiseModel::new_uniform(1.0);
        let catalog = source
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");
        let empty_history: Vec<DepolarizingSampledFault> = Vec::new();

        let prob_empty = catalog.fault_history_probability(&empty_history);
        assert!(
            (prob_empty).abs() < f64::EPSILON,
            "empty history should have probability 0"
        );

        // Create a non-trivial noise model and check that the probability of a specified
        // history is computed correctly
        let source = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);
        let catalog = source
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        let empty_history: Vec<DepolarizingSampledFault> = Vec::new();
        let prob_empty = catalog.fault_history_probability(&empty_history);
        assert!(
            (prob_empty - 0.9 * 0.8 * 0.7 * 0.6).abs() < f64::EPSILON,
            "empty history should have correct probability"
        );

        let history = vec![
            DepolarizingSampledFault {
                site_uid: 0,
                outcome_index: 1,
                outcome_label: "X",
            },
            DepolarizingSampledFault {
                site_uid: 1,
                outcome_index: 2,
                outcome_label: "Y",
            },
        ];
        let prob_history = catalog.fault_history_probability(&history);
        let expected_prob = 0.1 * 0.1 * 0.6 * 0.8;
        let tolerance = 1e-12;
        assert!(
            (prob_history - expected_prob).abs() < tolerance,
            "history should have correct probability"
        );
    }

    #[test]
    fn test_fault_history_probability_ratios() {
        let mut source = DepolarizingNoiseModel::new_uniform(0.5);
        source.set_seed(0);
        source.set_sampled_fault_history_enabled(true);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        // Get the fault catalog
        let catalog = source
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        // Get two history samples
        source
            .start(msg.clone())
            .expect("source start should succeed");
        let history1 = source
            .sampled_fault_history()
            .expect("sampled history should exist")
            .to_vec();
        source
            .start(msg.clone())
            .expect("source start should succeed");
        let history2 = source
            .sampled_fault_history()
            .expect("sampled history should exist")
            .to_vec();

        // Compute the probabilities & compare
        let prob1 = catalog.fault_history_probability(&history1);
        let prob2 = catalog.fault_history_probability(&history2);

        let ratio = prob1 / prob2;

        let ratio_from_function = catalog.fault_histories_probability_ratio(&history1, &history2);

        assert!(
            (ratio - ratio_from_function).abs() < f64::EPSILON,
            "ratios should match"
        );
    }

    #[test]
    fn test_random_flips_are_deterministic_given_seed() {
        let noise = DepolarizingNoiseModel::new_uniform(0.5);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let mut catalog_a = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");
        let mut catalog_b = catalog_a.clone();
        catalog_a.set_seed(0);
        catalog_b.set_seed(0);

        let mut history_a = Vec::new();
        let mut history_b = Vec::new();
        for _ in 0..10 {
            history_a = catalog_a.random_flip(&history_a);
            history_b = catalog_b.random_flip(&history_b);
            assert_eq!(history_a, history_b);
        }
    }

    #[test]
    fn test_random_flip_at_site_returns_a_different_fault() {
        let noise = DepolarizingNoiseModel::new_uniform(0.5);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        let msg = builder.build();

        let mut catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");
        catalog.set_seed(0);

        let outcomes = catalog.sites[0].outcomes.clone();
        for (outcome_index, outcome) in outcomes.iter().enumerate() {
            let history = vec![DepolarizingSampledFault {
                site_uid: 0,
                outcome_index: outcome_index as u8,
                outcome_label: outcome.label,
            }];

            let flipped_history = catalog.random_flip_at_site(0, &history);

            assert_ne!(flipped_history[0].outcome_label, outcome.label);
        }
    }

    #[test]
    fn test_random_flip_at_site_produces_valid_history() {
        let noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        let msg = builder.build();

        let mut catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");
        catalog.set_seed(0);

        let original_history = Vec::new();
        let site_uid = 2;
        let sampled_history = catalog.random_flip_at_site(site_uid, &original_history);

        assert_ne!(sampled_history, original_history);
        assert_eq!(sampled_history.len(), 1);

        let sampled_fault = &sampled_history[0];
        assert_eq!(sampled_fault.site_uid, site_uid);

        let site = catalog.get_site(site_uid);
        let outcome = &site.outcomes[usize::from(sampled_fault.outcome_index)];
        assert_eq!(sampled_fault.outcome_label, outcome.label);
        assert!(outcome.probability > 0.0);

        let expected_probability = catalog
            .sites
            .iter()
            .map(|catalog_site| {
                if catalog_site.uid == site_uid {
                    outcome.probability
                } else {
                    catalog_site
                        .no_fault_probability()
                        .expect("catalog site should have a no-fault outcome")
                }
            })
            .product::<f64>();
        let actual_probability = catalog.fault_history_probability(&sampled_history);
        assert!((actual_probability - expected_probability).abs() < f64::EPSILON);
    }

    #[test]
    fn test_random_flip_at_site_does_not_modify_provided_history() {
        let noise = DepolarizingNoiseModel::new_uniform(0.5);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        let msg = builder.build();

        let mut catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");
        catalog.set_seed(0);

        let history = vec![DepolarizingSampledFault {
            site_uid: 0,
            outcome_index: 1,
            outcome_label: "X",
        }];
        let original_history = history.clone();

        let _ = catalog.random_flip_at_site(0, &history);

        assert_eq!(history, original_history);
    }
}
