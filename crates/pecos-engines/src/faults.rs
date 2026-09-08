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

use crate::GateType;
use crate::noise::depolarizing::{
    DepolarizingFaultCatalog, DepolarizingFaultOutcome, DepolarizingFaultSite,
    DepolarizingSampledFault,
};
use pecos_random::PecosRng;
use std::collections::HashSet;

/// FaultOutcome
///
/// At every FaultSite, there are a number of possible
/// outcomes that can occur. This is one of the possible
/// outcomes. For example, this could be an "X" flip
/// for a single qubit or a correlated "XZ" flip for two
/// qubits.
///
/// In the current design, each fault is associated with a:
///  - probability: the probability of the fault
///  - label: a human-readable label for the fault, e.g. "X"
/// The `label()` and `probability()` functions provide
/// access to these.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultOutcome {
    Depolarizing(DepolarizingFaultOutcome),
}

impl FaultOutcome {
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Depolarizing(outcome) => outcome.label,
        }
    }

    #[must_use]
    pub fn probability(&self) -> f64 {
        match self {
            Self::Depolarizing(outcome) => outcome.probability,
        }
    }
}

/// FaultSite
///
/// A single location where a fault can occur in a given circuit.
/// An example of this might include information about where
/// the fault occurs (i.e. the associated gate, an index for the qubit, etc.)
/// and information about what faults are available, such as a vector
/// of FaultOutcomes.
///
/// In the current implementation, each FaultSite has
///  - uid: a unique identifier
///  - gate_index: a unique index identifying that gate with which it occurs
///  - gate_type: the type of the associated gate
///  - qubits: the qubits involved in the fault
///  - outcomes: a list of possible FaultOutcomes
///
#[derive(Debug, Clone, PartialEq)]
pub enum FaultSite {
    Depolarizing(DepolarizingFaultSite),
}

impl FaultSite {
    #[must_use]
    pub fn uid(&self) -> usize {
        match self {
            Self::Depolarizing(site) => site.uid,
        }
    }

    #[must_use]
    pub fn gate_index(&self) -> usize {
        match self {
            Self::Depolarizing(site) => site.gate_index,
        }
    }

    #[must_use]
    pub fn gate_type(&self) -> GateType {
        match self {
            Self::Depolarizing(site) => site.gate_type,
        }
    }

    #[must_use]
    pub fn qubits(&self) -> &[usize] {
        match self {
            Self::Depolarizing(site) => &site.qubits,
        }
    }

    #[must_use]
    pub fn outcomes(&self) -> Vec<FaultOutcome> {
        match self {
            Self::Depolarizing(site) => site.outcomes.iter().cloned().map(Into::into).collect(),
        }
    }

    // Returns the probability of a given outcome, as specified
    // by its human-readable string
    #[must_use]
    pub fn outcome_label_probability(&self, label: &str) -> Option<f64> {
        self.outcomes()
            .into_iter()
            .find(|outcome| outcome.label() == label)
            .map(|outcome| outcome.probability())
    }

    // Returns the probability of no fault occurring
    #[must_use]
    pub fn no_fault_probability(&self) -> Option<f64> {
        self.outcome_label_probability("NoFault")
    }
}

/// SampledFault
///
/// This is a (non-identity) fault that *has* occurred
/// in a circuit.
///
/// Each sampled fault has:
///  - site_uid: the unique identifier of the associated FaultSite
///  - outcome_index: the index of the sampled outcome
///  - outcome_label: the human-readable label of the sampled outcome
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampledFault {
    Depolarizing(DepolarizingSampledFault),
}

impl SampledFault {
    #[must_use]
    pub fn site_uid(&self) -> usize {
        match self {
            Self::Depolarizing(fault) => fault.site_uid,
        }
    }

    #[must_use]
    pub fn outcome_index(&self) -> u8 {
        match self {
            Self::Depolarizing(fault) => fault.outcome_index,
        }
    }

    #[must_use]
    pub fn outcome_label(&self) -> &'static str {
        match self {
            Self::Depolarizing(fault) => fault.outcome_label,
        }
    }
}

/// FaultCatalog
///
/// A full catalog of all faults that *may* occur during
/// execution of a circuit.
///
/// Each FaultCatalog contains the following methods:
///  - len: returns the number of sites in the fault catalog
///  - set_seed: specifies the seed for the PecosRng to be used
///  - fault_history_probability: returns the probability of a given fault history
///  - fault_histories_probability_ratio: returns the ratio of probabilities of two fault histories

#[derive(Debug, Clone, PartialEq)]
pub enum FaultCatalog {
    Depolarizing(DepolarizingFaultCatalog),
}

impl FaultCatalog {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Depolarizing(catalog) => catalog.sites.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate without exposing the concrete catalog.
    pub fn sites(&self) -> impl Iterator<Item = FaultSite> + '_ {
        match self {
            Self::Depolarizing(catalog) => catalog.sites.iter().cloned().map(Into::into),
        }
    }

    #[must_use]
    pub fn get_site(&self, site_uid: usize) -> FaultSite {
        self.sites()
            .find(|site| site.uid() == site_uid)
            .unwrap_or_else(|| panic!("Site uid {site_uid} not found in fault catalog"))
    }

    /// Seed the proposal RNG used when perturbing histories.
    pub fn set_seed(&mut self, seed: u64) {
        match self {
            Self::Depolarizing(catalog) => catalog.rng = Some(PecosRng::seed_from_u64(seed)),
        }
    }

    #[must_use]
    pub fn fault_history_probability(&self, history: &FaultHistory) -> f64 {
        self.check_valid_fault_history(history);
        let mut probability = 1.0;
        let mut faults = history.iter().peekable();
        for site in self.sites() {
            let label = faults
                .peek()
                .filter(|fault| fault.site_uid() == site.uid())
                .map_or("NoFault", SampledFault::outcome_label);
            if label != "NoFault" {
                faults.next();
            }
            probability *= site.outcome_label_probability(label).unwrap_or_else(|| {
                panic!(
                    "Outcome label {label} not found for fault site {}",
                    site.uid()
                )
            });
        }
        probability
    }

    // Computes the ratio of probabilities between two fault histories
    #[must_use]
    pub fn fault_histories_probability_ratio(&self, a: &FaultHistory, b: &FaultHistory) -> f64 {
        // TODO there used to be a better way to do this where you go site-by-site
        self.fault_history_probability(a) / self.fault_history_probability(b)
    }

    // Computes the ratio of probabilities when a single fault history is specified
    // but you want to compare two fault catalogs (each generated by a different error model)
    #[must_use]
    pub fn fault_catalog_probability_ratio(&self, other: &Self, history: &FaultHistory) -> f64 {
        assert!(
            self.is_catalog_compatible(other),
            "Fault catalogs are not compatible"
        );
        self.fault_history_probability(history) / other.fault_history_probability(history)
    }

    /// Randomly change one site's outcome and return the proposed history.
    pub fn random_flip(&mut self, history: &FaultHistory) -> FaultHistory {
        let site_uid = self.random_site_uid();
        self.random_flip_at_site(site_uid, history)
    }

    /// Propose one random flip and return its Metropolis-Hastings correction.
    pub fn random_flip_hastings_correction(
        &mut self,
        history: &FaultHistory,
    ) -> (FaultHistory, f64) {
        let site_uid = self.random_site_uid();
        let site = self.get_site(site_uid);
        let label_at_site = |candidate: &FaultHistory| {
            candidate
                .iter()
                .find(|fault| fault.site_uid() == site_uid)
                .map_or("NoFault", |fault| fault.outcome_label())
        };
        let old_label = label_at_site(history);
        let proposed = self.random_flip_at_site(site_uid, history);
        let new_label = label_at_site(&proposed);
        let old_probability = site
            .outcome_label_probability(old_label)
            .expect("Current outcome must belong to its fault site");
        let new_probability = site
            .outcome_label_probability(new_label)
            .expect("Proposed outcome must belong to its fault site");

        // The proposal selects alternatives in proportion to their model probabilities.
        let correction = (old_probability / (1.0 - new_probability))
            / (new_probability / (1.0 - old_probability));
        (proposed, correction)
    }

    fn random_site_uid(&mut self) -> usize {
        let (rng, sites) = match self {
            Self::Depolarizing(catalog) => (&mut catalog.rng, &catalog.sites),
        };
        assert!(!sites.is_empty(), "Cannot flip a fault in an empty catalog");
        let rng = rng
            .as_mut()
            .expect("Set the fault catalog seed before requesting a flip");
        sites[(rng.next_u64() % sites.len() as u64) as usize].uid
    }

    pub fn random_flip_at_site(&mut self, site_uid: usize, history: &FaultHistory) -> FaultHistory {
        self.check_valid_fault_history(history);
        let random_value = match self {
            Self::Depolarizing(catalog) => {
                catalog
                    .rng
                    .as_mut()
                    .expect("Fault catalog RNG is not set")
                    .next_u64() as f64
                    / u64::MAX as f64
            }
        };
        let site = self.get_site(site_uid);
        let current_label = history
            .iter()
            .find(|fault| fault.site_uid() == site_uid)
            .map_or("NoFault", |fault| fault.outcome_label());
        let outcomes = site
            .outcomes()
            .into_iter()
            .filter(|outcome| outcome.label() != current_label)
            .collect::<Vec<_>>();
        let total = outcomes.iter().map(FaultOutcome::probability).sum::<f64>();
        let mut cumulative = 0.0;
        let selected = outcomes
            .iter()
            .find(|outcome| {
                cumulative += outcome.probability();
                random_value * total < cumulative
            })
            .or_else(|| outcomes.last())
            .expect("Fault site has no alternative outcome");

        // Conversion back to the child type remains local to this enum implementation.
        let mut proposed = history.as_depolarizing().to_vec();
        proposed.retain(|fault| fault.site_uid != site_uid);
        if selected.label() != "NoFault" {
            let outcome_index =
                site.outcomes()
                    .iter()
                    .position(|outcome| outcome.label() == selected.label())
                    .expect("Selected outcome came from this site") as u8;
            proposed.push(DepolarizingSampledFault {
                site_uid,
                outcome_index,
                outcome_label: selected.label(),
            });
            proposed.sort_by_key(|fault| fault.site_uid);
        }
        proposed.into()
    }

    fn is_catalog_compatible(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.sites().zip(other.sites()).all(|(left, right)| {
                left.uid() == right.uid() && left.gate_type() == right.gate_type()
            })
    }

    fn check_valid_fault_history(&self, history: &FaultHistory) {
        // A history cannot be used with a catalog from another noise model.
        assert!(
            matches!(
                (self, history),
                (Self::Depolarizing(_), FaultHistory::Depolarizing(_))
            ),
            "Fault history and catalog were produced by different noise models"
        );
        let catalog_uids = self.sites().map(|site| site.uid()).collect::<HashSet<_>>();
        let history_uids = history
            .iter()
            .map(|fault| fault.site_uid())
            .collect::<Vec<_>>();
        assert!(
            history_uids.iter().all(|uid| catalog_uids.contains(uid)),
            "Fault history contains a site uid not present in the catalog"
        );
        assert!(
            history_uids.windows(2).all(|uids| uids[0] < uids[1]),
            "Fault history site uids must be unique and ascending"
        );
    }
}

/// FaultHistory
///
/// A list of all faults that *have* occured during
/// execution of a circuit
///
/// Each fault history contains:
///  - len: the number of faults in the history
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultHistory {
    Depolarizing(Vec<DepolarizingSampledFault>),
}

impl FaultHistory {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Depolarizing(history) => history.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate without exposing the concrete history vector.
    pub fn iter(&self) -> impl Iterator<Item = SampledFault> + '_ {
        match self {
            Self::Depolarizing(history) => history.iter().cloned().map(Into::into),
        }
    }

    /// Access the concrete representation only at the noise-model boundary.
    pub(crate) fn as_depolarizing(&self) -> &[DepolarizingSampledFault] {
        match self {
            Self::Depolarizing(history) => history,
        }
    }
}

// Promotion of depolarizing instances of fault classes
// to generic versions. This allows the depolarizing versions
// to be used without exposing details to simulation engines.
impl From<DepolarizingFaultSite> for FaultSite {
    fn from(value: DepolarizingFaultSite) -> Self {
        Self::Depolarizing(value)
    }
}

impl From<DepolarizingFaultOutcome> for FaultOutcome {
    fn from(value: DepolarizingFaultOutcome) -> Self {
        Self::Depolarizing(value)
    }
}

impl From<DepolarizingSampledFault> for SampledFault {
    fn from(value: DepolarizingSampledFault) -> Self {
        Self::Depolarizing(value)
    }
}

impl From<DepolarizingFaultCatalog> for FaultCatalog {
    fn from(value: DepolarizingFaultCatalog) -> Self {
        Self::Depolarizing(value)
    }
}

impl From<Vec<DepolarizingSampledFault>> for FaultHistory {
    fn from(value: Vec<DepolarizingSampledFault>) -> Self {
        Self::Depolarizing(value)
    }
}
