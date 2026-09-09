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

use pecos_engines::faults::{FaultCatalog, FaultHistory, FaultOutcome};
use pecos_random::PecosRng;

/// Returns a random integer between 0 and the provided upper bound
pub fn random_int(rng: &mut PecosRng, upper_bound: u64) -> u64 {
    let invalid_threshold = upper_bound.wrapping_neg() % upper_bound;
    let mut bit = rng.next_u64();
    while bit < invalid_threshold {
        bit = rng.next_u64();
    }
    bit % upper_bound
}

/// Enum representing different types of fault flippers.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultFlipper {
    BasicSingle(BasicSingleFaultFlipper),
    WeightedSingle(WeightedSingleFaultFlipper),
}

impl FaultFlipper {
    #[must_use]
    pub fn new_basic(fault_catalog: FaultCatalog) -> Self {
        Self::BasicSingle(BasicSingleFaultFlipper::new(fault_catalog))
    }

    #[must_use]
    pub fn new_weighted(fault_catalog: FaultCatalog) -> Self {
        Self::WeightedSingle(WeightedSingleFaultFlipper::new(fault_catalog))
    }

    pub fn set_seed(&mut self, seed: u64) {
        match self {
            Self::BasicSingle(flipper) => flipper.set_seed(seed),
            Self::WeightedSingle(flipper) => flipper.set_seed(seed),
        }
    }

    pub fn random_flip_and_ratio(&mut self, history: &FaultHistory) -> (FaultHistory, f64) {
        match self {
            Self::BasicSingle(flipper) => flipper.random_flip_and_ratio(history),
            Self::WeightedSingle(flipper) => flipper.random_flip_and_ratio(history),
        }
    }

    pub fn random_flip(&mut self, history: &FaultHistory) -> FaultHistory {
        match self {
            Self::BasicSingle(flipper) => flipper.random_flip(history),
            Self::WeightedSingle(flipper) => flipper.random_flip(history),
        }
    }
}

/// Simplest single site fault flipper
/// where a site is chosen at random and then
/// a new event is chosen entirely at random
/// with no respect given to the relative probabilities
/// of events
#[derive(Debug, Clone, PartialEq)]
pub struct BasicSingleFaultFlipper {
    pub rng: Option<PecosRng>,
    pub fault_catalog: FaultCatalog,
}

impl BasicSingleFaultFlipper {
    #[must_use]
    pub fn new(fault_catalog: FaultCatalog) -> Self {
        Self {
            rng: None,
            fault_catalog,
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Some(PecosRng::seed_from_u64(seed));
    }

    fn random_site_uid(&mut self) -> usize {
        let rng = self
            .rng
            .as_mut()
            .expect("Set the fault flipper seed before requesting a flip");
        let nsite = self.fault_catalog.len() as u64;
        assert!(nsite > 0, "Cannot flip a fault in an empty catalog");
        let random_idx = random_int(rng, nsite) as usize;
        self.fault_catalog
            .sites()
            .nth(random_idx)
            .expect("Valid site index")
            .uid()
    }

    pub fn random_flip_and_ratio_at_site(
        &mut self,
        site_uid: usize,
        history: &FaultHistory,
    ) -> (FaultHistory, f64) {

        // Get the site and a label for the current outcome
        let site = self.fault_catalog.get_site(site_uid);
        let current_label = history
            .iter()
            .find(|fault| fault.site_uid() == site_uid)
            .map_or("NoFault", |fault| fault.outcome_label());
        
        // Grab all the outcomes (except the current one)
        let outcomes = site
            .outcomes()
            .into_iter()
            .filter(|outcome| outcome.label() != current_label)
            .collect::<Vec<_>>();
        assert!(!outcomes.is_empty(), "Fault site has no alternative outcome");

        // Select a random outcome
        let rng = self
            .rng
            .as_mut()
            .expect("Set the fault flipper seed before requesting a flip");
        let chosen_idx = random_int(rng, outcomes.len() as u64) as usize;
        let chosen_outcome = &outcomes[chosen_idx];

        // Get the new history with the randomly selected outcome
        let new_history = history.with_outcome(&site, chosen_outcome.label());

        // Compute the ratio of probabilities
        // No further correction is needed because probabilities are equal
        // in both directions.
        let prev_log_prob = site
            .outcome_label_log_probability(current_label)
            .expect("Current outcome must belong to its fault site");
        let new_log_prob = site
            .outcome_label_log_probability(chosen_outcome.label())
            .expect("Chosen outcome must belong to its fault site");
        let ratio = (new_log_prob - prev_log_prob).exp();

        (new_history, ratio)
    }

    pub fn random_flip_at_site(&mut self, site_uid: usize, history: &FaultHistory) -> FaultHistory {
        let (result, _) = self.random_flip_and_ratio_at_site(site_uid, history);
        result
    }

    pub fn random_flip_and_ratio(&mut self, history: &FaultHistory) -> (FaultHistory, f64) {
        let site_uid = self.random_site_uid();
        self.random_flip_and_ratio_at_site(site_uid, history)
    }

    pub fn random_flip(&mut self, history: &FaultHistory) -> FaultHistory {
        let site_uid = self.random_site_uid();
        self.random_flip_at_site(site_uid, history)
    }
}

/// Fault flipper for depolarizing fault sampling, weighted by model probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedSingleFaultFlipper {
    pub rng: Option<PecosRng>,
    pub fault_catalog: FaultCatalog,
}

impl WeightedSingleFaultFlipper {
    #[must_use]
    pub fn new(fault_catalog: FaultCatalog) -> Self {
        Self {
            rng: None,
            fault_catalog,
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Some(PecosRng::seed_from_u64(seed));
    }

    fn random_site_uid(&mut self) -> usize {
        let rng = self
            .rng
            .as_mut()
            .expect("Set the fault flipper seed before requesting a flip");
        let nsite = self.fault_catalog.len() as u64;
        assert!(nsite > 0, "Cannot flip a fault in an empty catalog");
        let random_idx = random_int(rng, nsite) as usize;
        self.fault_catalog
            .sites()
            .nth(random_idx)
            .expect("Valid site index")
            .uid()
    }

    pub fn random_flip_and_ratio_at_site(
        &mut self,
        site_uid: usize,
        history: &FaultHistory,
    ) -> (FaultHistory, f64) {

        // Get the site and a label for the current outcome
        let site = self.fault_catalog.get_site(site_uid);
        let current_label = history
            .iter()
            .find(|fault| fault.site_uid() == site_uid)
            .map_or("NoFault", |fault| fault.outcome_label());
        
        // Grab all the outcomes (except the current one)
        let outcomes = site
            .outcomes()
            .into_iter()
            .filter(|outcome| outcome.label() != current_label)
            .collect::<Vec<_>>();
        assert!(!outcomes.is_empty(), "Fault site has no alternative outcome");

        // Select a random value and scale it down according to the probability
        // of all possible new outcomes
        let rand_val = self.rng.as_mut()
            .expect("Set the fault flipper seed before requesting a flip")
            .next_f64();
        let total_prob: f64 = outcomes.iter().map(FaultOutcome::probability).sum();
        let scaled_val = rand_val * total_prob;

        // Get the random outcome based on the scaled random value
        let mut cumulative = 0.0;
        let chosen_outcome = outcomes
            .iter()
            .find(|outcome| {
                cumulative += outcome.probability();
                scaled_val < cumulative
            })
            .or_else(|| outcomes.last())
            .expect("There are no alternative outcomes for this fault site");
        
        // Get the new history with the randomly selected outcome
        let new_history = history.with_outcome(&site, chosen_outcome.label());

        // Compute the acceptance probability ratio
        // (with the hastings correction term)
        // p(x')/p(x) * p(x|x')/p(x'|x) = 
        // (1 - p(x)) / (1 - p(x'))
        let prev_prob = site
            .outcome_label_probability(current_label)
            .expect("Current outcome must belong to its fault site");
        let new_prob = site
            .outcome_label_probability(chosen_outcome.label())
            .expect("Selected outcome must belong to its fault site");

        let ratio = (1.0 - prev_prob) / (1.0 - new_prob);

        (new_history, ratio)
    }

    pub fn random_flip_at_site(&mut self, site_uid: usize, history: &FaultHistory) -> FaultHistory {
        let (result, _) = self.random_flip_and_ratio_at_site(site_uid, history);
        result
    }

    pub fn random_flip_and_ratio(&mut self, history: &FaultHistory) -> (FaultHistory, f64) {
        let site_uid = self.random_site_uid();
        self.random_flip_and_ratio_at_site(site_uid, history)
    }

    pub fn random_flip(&mut self, history: &FaultHistory) -> FaultHistory {
        let site_uid = self.random_site_uid();
        self.random_flip_at_site(site_uid, history)
    }
}

// TODO Implement multi-site fault flipper.
