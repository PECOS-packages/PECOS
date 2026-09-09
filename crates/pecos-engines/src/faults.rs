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
///  - log_probability: the log probability of the fault
///  - label: a human-readable label for the fault, e.g. "X"
/// The `label()`, `log_probability()`, and `probability()` functions provide
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

    // We store the log of the probability for numerical
    // precision reasons
    #[must_use]
    pub fn log_probability(&self) -> f64 {
        match self {
            Self::Depolarizing(outcome) => outcome.log_probability,
        }
    }

    // Compute the probability from the log of the probability
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.log_probability().exp()
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

    // Returns the log of the probability of a given outcome, as specified
    // by its human-readable string
    #[must_use]
    pub fn outcome_label_log_probability(&self, label: &str) -> Option<f64> {
        self.outcomes()
            .into_iter()
            .find(|outcome| outcome.label() == label)
            .map(|outcome| outcome.log_probability())
    }

    // Returns the probability of a given outcome, as specified
    // by its human-readable string
    #[must_use]
    pub fn outcome_label_probability(&self, label: &str) -> Option<f64> {
        self.outcome_label_log_probability(label).map(f64::exp)
    }

    // Returns the log of the probability of no fault occurring
    #[must_use]
    pub fn no_fault_log_probability(&self) -> Option<f64> {
        self.outcome_label_log_probability("NoFault")
    }

    // Returns the probability of no fault occurring
    #[must_use]
    pub fn no_fault_probability(&self) -> Option<f64> {
        self.no_fault_log_probability().map(f64::exp)
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
///  - fault_history_log_probability: returns the log of the probability of a given fault history
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

    #[must_use]
    pub fn fault_history_log_probability(&self, history: &FaultHistory) -> f64 {
        self.check_valid_fault_history(history);
        let mut log_probability = 0.0;
        let mut faults = history.iter().peekable();
        for site in self.sites() {
            let label = faults
                .peek()
                .filter(|fault| fault.site_uid() == site.uid())
                .map_or("NoFault", SampledFault::outcome_label);
            if label != "NoFault" {
                faults.next();
            }
            log_probability += site.outcome_label_log_probability(label).unwrap_or_else(|| {
                panic!(
                    "Outcome label {label} not found for fault site {}",
                    site.uid()
                )
            });
        }
        log_probability
    }

    #[must_use]
    pub fn fault_history_probability(&self, history: &FaultHistory) -> f64 {
        self.fault_history_log_probability(history).exp()
    }

    // Computes the log of the ratio of probabilities between two fault histories
    #[must_use]
    pub fn fault_histories_log_probability_ratio(&self, a: &FaultHistory, b: &FaultHistory) -> f64 {
        // Check that they are both valid histories
        self.check_valid_fault_history(a);
        self.check_valid_fault_history(b);

        let mut ratio: f64 = 0.0;
        let mut a_faults = a.iter().peekable();
        let mut b_faults = b.iter().peekable();

        // Iterate through sites, only updating if there is a fault site
        for site in self.sites() {
            // Check if the next faults are both at this site
            let a_label = a_faults
                .peek()
                .filter(|fault| fault.site_uid() == site.uid())
                .map_or("NoFault", SampledFault::outcome_label);

            let b_label = b_faults
                .peek()
                .filter(|fault| fault.site_uid() == site.uid())
                .map_or("NoFault", SampledFault::outcome_label);

            // Move both to the next fault
            if a_label != "NoFault" {
                a_faults.next();
            }
            if b_label != "NoFault" {
                b_faults.next();
            }

            // Update the ratio if the labels are different
            if a_label != b_label {
                ratio += site.outcome_label_log_probability(a_label).unwrap_or_else(|| {
                    panic!(
                        "Outcome label {a_label} not found for fault site {}",
                        site.uid()
                    )
                });
                ratio -= site.outcome_label_log_probability(b_label).unwrap_or_else(|| {
                    panic!(
                        "Outcome label {b_label} not found for fault site {}",
                        site.uid()
                    )
                });
            }
        }
        ratio
    }

    // Computes the ratio of probabilities between two fault histories
    #[must_use]
    pub fn fault_histories_probability_ratio(&self, a: &FaultHistory, b: &FaultHistory) -> f64 {
        self.fault_histories_log_probability_ratio(a, b).exp()
    }

    // Computes the ratio of probabilities when a single fault history is specified
    // but you want to compare two fault catalogs (each generated by a different error model)
    #[must_use]
    pub fn fault_catalog_probability_ratio(&self, other: &Self, history: &FaultHistory) -> f64 {
        assert!(
            self.is_catalog_compatible(other),
            "Fault catalogs are not compatible"
        );
        self.fault_catalog_log_probability_ratio(other, history).exp()
    }

    #[must_use]
    pub fn fault_catalog_log_probability_ratio(&self, other: &Self, history: &FaultHistory) -> f64 {
        assert!(
            self.is_catalog_compatible(other),
            "Fault catalogs are not compatible"
        );
        self.fault_history_log_probability(history) - other.fault_history_log_probability(history)
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

    fn is_catalog_compatible(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.sites().zip(other.sites()).all(|(left, right)| {
                left.uid() == right.uid() && left.gate_type() == right.gate_type()
            })
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
    pub fn with_outcome(
        &self,
        site: &FaultSite,
        outcome_label: &'static str,
    ) -> Self {

        let outcome_index = site
            .outcomes()
            .iter()
            .position(|outcome| outcome.label() == outcome_label)
            .expect("Selected outcome came from this site") as u8;
        let site_uid = site.uid();

        match (self, site) {
            (Self::Depolarizing(history), FaultSite::Depolarizing(_)) => {
                let mut proposed = history.clone();
                proposed.retain(|fault| fault.site_uid != site_uid);
                if outcome_label != "NoFault" {
                    proposed.push(DepolarizingSampledFault {
                        site_uid,
                        outcome_index,
                        outcome_label,
                    });
                    proposed.sort_by_key(|fault| fault.site_uid);
                }
                Self::Depolarizing(proposed)
            }
        }
    }

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
    pub fn as_depolarizing(&self) -> &[DepolarizingSampledFault] {
        match self {
            Self::Depolarizing(history) => history,
        }
    }

    /// Returns a sampled fault at the specified index
    pub fn get(&self, index: usize) -> Option<SampledFault> {
        match self {
            Self::Depolarizing(history) => history.get(index).cloned().map(Into::into),
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

// TODO: Add some tests for all of this.
