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

use std::collections::BTreeMap;
use std::ops::Mul;

use crate::Gate;
use crate::noise::noise_rng::NoiseRng;
use crate::noise::utils::{SingleQubitNoiseResult, TwoQubitNoiseResult};
use rand::distr::weighted::WeightedIndex;

/// Tolerance for weight normalization - total weights should be within this amount of 1.0
const NORMALIZATION_TOLERANCE: f64 = 1e-5;
/// Small margin for floating-point equality comparisons
const FLOAT_EPSILON: f64 = 1e-10;

/// A sampler that selects keys with probability proportional to their weights
///
/// Uses `BTreeMap` for deterministic key ordering, ensuring consistent behavior
/// when using the same seed across multiple runs or threads.
#[derive(Debug, Clone)]
pub struct WeightedSampler<K: Clone + Ord> {
    keys: Vec<K>,
    distribution: WeightedIndex<f64>,
    weighted_map: BTreeMap<K, f64>,
}

impl<K: Clone + std::fmt::Debug + std::hash::Hash + Eq + Ord> WeightedSampler<K> {
    /// Create a new weighted sampler from a map of keys to weights
    ///
    /// The weights are normalized to sum to 1.0 with a default tolerance of 1e-10
    ///
    /// # Panics
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    /// - If the weighted index distribution cannot be created
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<K, f64>) -> Self {
        Self::new_with_tolerance(weighted_map, NORMALIZATION_TOLERANCE)
    }

    /// Create a sampler from non-negative relative weights.
    ///
    /// Unlike [`Self::new`], the weights need not sum to one. They are normalized internally.
    ///
    /// # Panics
    ///
    /// Panics if the map is empty, a weight is negative or non-finite, the total is not positive,
    /// or the weighted distribution cannot be created.
    #[must_use]
    pub fn new_relative(weighted_map: &BTreeMap<K, f64>) -> Self {
        Self::new_impl(weighted_map, NORMALIZATION_TOLERANCE, false)
    }

    /// Create a new weighted sampler with a specific tolerance for weight normalization
    ///
    /// # Panics
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new_with_tolerance(weighted_map: &BTreeMap<K, f64>, tolerance: f64) -> Self {
        Self::new_impl(weighted_map, tolerance, true)
    }

    fn new_impl(weighted_map: &BTreeMap<K, f64>, tolerance: f64, require_normalized: bool) -> Self {
        let (normalized_weighted_map, normalized_weights) =
            Self::validate_and_normalize(weighted_map, tolerance, require_normalized);

        // BTreeMap already provides deterministic ordering of keys
        let keys: Vec<K> = weighted_map.keys().cloned().collect();

        // Create the distribution using deterministically ordered weights
        let distribution = WeightedIndex::new(&normalized_weights)
            .expect("WeightedSampler: failed to create weighted distribution");

        WeightedSampler {
            keys,
            distribution,
            weighted_map: normalized_weighted_map,
        }
    }

    /// Validates that the weights are positive and approximately sum to 1.0
    /// Returns a normalized `BTreeMap` and a Vec of normalized weights for creating the distribution
    fn validate_and_normalize(
        weighted_map: &BTreeMap<K, f64>,
        tolerance: f64,
        require_normalized: bool,
    ) -> (BTreeMap<K, f64>, Vec<f64>) {
        assert!(
            !weighted_map.is_empty(),
            "WeightedSampler: weighted_map cannot be empty"
        );

        for weight in weighted_map.values() {
            assert!(
                weight.is_finite() && *weight >= 0.0,
                "WeightedSampler: weights must be finite and non-negative, got {weight}"
            );
        }

        let total_weight: f64 = weighted_map.values().sum();

        assert!(
            total_weight > 0.0,
            "WeightedSampler: total weight must be positive, got {total_weight}"
        );

        // Check if weights are within tolerance of 1.0
        if require_normalized {
            assert!(
                (total_weight - 1.0).abs() <= tolerance,
                "WeightedSampler: total weight {total_weight} deviates from 1.0 by more than tolerance {tolerance}"
            );
        }

        // Determine if we need to normalize (only normalize if not already very close to 1.0)
        let needs_normalization = (total_weight - 1.0).abs() > FLOAT_EPSILON;

        // Collect normalized weights for the distribution
        let normalized_weights: Vec<f64> = if needs_normalization {
            weighted_map.values().map(|&w| w / total_weight).collect()
        } else {
            weighted_map.values().copied().collect()
        };

        // Create normalized BTreeMap
        let mut normalized_map = BTreeMap::new();
        for (key, &value) in weighted_map {
            let normalized_value = if needs_normalization {
                value / total_weight
            } else {
                value
            };
            normalized_map.insert(key.clone(), normalized_value);
        }

        (normalized_map, normalized_weights)
    }

    /// Sample a key from the distribution
    ///
    /// # Panics
    /// - If the keys vector is empty (should never happen if constructed properly)
    /// - If the distribution sampling fails
    #[must_use]
    pub fn sample(&self, rng: &mut NoiseRng) -> K {
        let index = rng.sample(&self.distribution);
        self.keys[index].clone()
    }

    /// Get a reference to the normalized weighted map
    #[must_use]
    pub fn get_weighted_map(&self) -> &BTreeMap<K, f64> {
        &self.weighted_map
    }
}

/// Create a Pauli gate based on the Pauli operator character
/// Returns None for identity ('I') operations
fn create_pauli_gate(op: char, qubit: usize) -> Option<Gate> {
    match op {
        'X' => Some(Gate::x(&[qubit])),
        'Y' => Some(Gate::y(&[qubit])),
        'Z' => Some(Gate::z(&[qubit])),
        'I' => None, // Identity - no operation
        _ => panic!("Invalid Pauli operator '{op}'"),
    }
}

fn validate_pauli_leakage_label(label: &str, arity: usize) {
    assert_eq!(
        label.chars().count(),
        arity,
        "PauliLeakageDict: event {label:?} must contain exactly {arity} symbols"
    );
    for symbol in label.chars() {
        assert!(
            matches!(symbol, 'I' | 'X' | 'Y' | 'Z' | 'L'),
            "PauliLeakageDict: invalid symbol {symbol:?} in event {label:?}; expected 'I', 'X', 'Y', 'Z', or 'L'"
        );
    }
    assert!(
        label.chars().any(|symbol| symbol != 'I'),
        "PauliLeakageDict: the all-identity event {label:?} is implicit and cannot be supplied"
    );
}

/// A validated relative distribution of stochastic Pauli-plus-leakage events.
///
/// Event labels contain `I`, `X`, `Y`, `Z`, or `L`, where `L` means `any -> L` on that leg.
/// The all-identity event is excluded because channel identity is represented by the outer
/// application probability. Event weights form a normalized conditional distribution.
///
/// Use [`Self::tensor`] (or `*`) to form a higher-arity Kronecker product distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliLeakageDict {
    arity: usize,
    events: BTreeMap<String, f64>,
}

impl PauliLeakageDict {
    /// Validate and construct an event dictionary, inferring arity from its labels.
    ///
    /// # Panics
    ///
    /// Panics if the dictionary is empty, labels are invalid or inconsistent, the all-identity
    /// event is supplied, or the weights do not form a normalized distribution.
    #[must_use]
    pub fn new(events: &BTreeMap<String, f64>) -> Self {
        assert!(
            !events.is_empty(),
            "PauliLeakageDict: events cannot be empty"
        );
        let arity = events
            .keys()
            .next()
            .expect("nonempty event dictionary")
            .chars()
            .count();
        assert!(arity > 0, "PauliLeakageDict: event labels cannot be empty");
        for event in events.keys() {
            validate_pauli_leakage_label(event, arity);
        }
        let sampler = WeightedSampler::new(events);
        Self {
            arity,
            events: sampler.get_weighted_map().clone(),
        }
    }

    /// Number of qubit legs represented by each event label.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Return the normalized event distribution.
    #[must_use]
    pub fn events(&self) -> &BTreeMap<String, f64> {
        &self.events
    }

    /// Form the tensor/Kronecker product of two event distributions.
    #[must_use]
    pub fn tensor(&self, other: &Self) -> Self {
        let mut events = BTreeMap::new();
        for (left, left_probability) in &self.events {
            for (right, right_probability) in &other.events {
                *events.entry(format!("{left}{right}")).or_insert(0.0) +=
                    left_probability * right_probability;
            }
        }
        Self::new(&events)
    }
}

impl Mul for PauliLeakageDict {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.tensor(&rhs)
    }
}

impl Mul<&PauliLeakageDict> for &PauliLeakageDict {
    type Output = PauliLeakageDict;

    fn mul(self, rhs: &PauliLeakageDict) -> Self::Output {
        self.tensor(rhs)
    }
}

impl Mul<&PauliLeakageDict> for PauliLeakageDict {
    type Output = Self;

    fn mul(self, rhs: &PauliLeakageDict) -> Self::Output {
        self.tensor(rhs)
    }
}

impl Mul<PauliLeakageDict> for &PauliLeakageDict {
    type Output = PauliLeakageDict;

    fn mul(self, rhs: PauliLeakageDict) -> Self::Output {
        self.tensor(&rhs)
    }
}

/// A stochastic single-qubit Pauli-plus-leakage channel.
///
/// With the outer `probability`, one event is sampled from the normalized relative distribution.
/// If the outer coin fails, identity is implicit. `L` means `any -> L`; Pauli events have no
/// effect while the qubit is already leaked.
///
/// # Example
///
/// ```
/// use std::collections::BTreeMap;
/// use pecos_engines::noise::{GeneralNoiseModel, PauliLeakageChannel};
///
/// let events = BTreeMap::from([
///     ("X".to_string(), 0.4),
///     ("Y".to_string(), 0.2),
///     ("Z".to_string(), 0.3),
///     ("L".to_string(), 0.1),
/// ]);
/// let channel = PauliLeakageChannel::new(0.001, &events);
/// let noise = GeneralNoiseModel::builder()
///     .add_p1_pauli_leakage_channel_after_gate(&channel)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct PauliLeakageChannel {
    probability: f64,
    events: PauliLeakageDict,
}

impl PauliLeakageChannel {
    /// Configure a single-qubit channel from a plain event dictionary.
    ///
    /// # Panics
    ///
    /// Panics if `probability` or the event dictionary is invalid.
    #[must_use]
    pub fn new(probability: f64, events: &BTreeMap<String, f64>) -> Self {
        Self::from_event_dict(probability, &PauliLeakageDict::new(events))
    }

    /// Configure a channel from an already validated, single-qubit event dictionary.
    ///
    /// # Panics
    ///
    /// Panics if `probability` is invalid or `events` does not have arity one.
    #[must_use]
    pub fn from_event_dict(probability: f64, events: &PauliLeakageDict) -> Self {
        assert!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "PauliLeakageChannel probability must be finite and between 0 and 1, got {probability}"
        );
        assert_eq!(
            events.arity(),
            1,
            "PauliLeakageChannel requires an event dictionary of arity one"
        );
        Self {
            probability,
            events: events.clone(),
        }
    }

    /// Return the outer application probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Return the normalized event weights.
    #[must_use]
    pub fn events(&self) -> &BTreeMap<String, f64> {
        self.events.events()
    }

    /// Return the validated event dictionary.
    #[must_use]
    pub fn event_dict(&self) -> &PauliLeakageDict {
        &self.events
    }
}

/// A stochastic joint two-qubit Pauli-plus-leakage channel.
#[derive(Clone, Debug)]
pub struct TwoQubitPauliLeakageChannel {
    probability: f64,
    events: PauliLeakageDict,
}

impl TwoQubitPauliLeakageChannel {
    /// Configure a joint two-qubit channel from a plain event dictionary.
    ///
    /// # Panics
    ///
    /// Panics if `probability` or the event dictionary is invalid.
    #[must_use]
    pub fn new(probability: f64, events: &BTreeMap<String, f64>) -> Self {
        Self::from_event_dict(probability, &PauliLeakageDict::new(events))
    }

    /// Configure a channel from an already validated, two-qubit event dictionary.
    ///
    /// # Panics
    ///
    /// Panics if `probability` is invalid or `events` does not have arity two.
    #[must_use]
    pub fn from_event_dict(probability: f64, events: &PauliLeakageDict) -> Self {
        assert!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "TwoQubitPauliLeakageChannel probability must be finite and between 0 and 1, got {probability}"
        );
        assert_eq!(
            events.arity(),
            2,
            "TwoQubitPauliLeakageChannel requires an event dictionary of arity two"
        );
        Self {
            probability,
            events: events.clone(),
        }
    }

    /// Return the outer application probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Return the normalized event weights.
    #[must_use]
    pub fn events(&self) -> &BTreeMap<String, f64> {
        self.events.events()
    }

    /// Return the validated event dictionary.
    #[must_use]
    pub fn event_dict(&self) -> &PauliLeakageDict {
        &self.events
    }
}

/// Samples single qubit noise operations (Pauli gates or leakage)
#[derive(Clone, Debug)]
pub struct SingleQubitWeightedSampler {
    sampler: WeightedSampler<String>,
}

impl SingleQubitWeightedSampler {
    /// Create a new single qubit sampler from a weighted map
    ///
    /// Valid keys are: "X", "Y", "Z", "L" (for leakage)
    ///
    /// # Panics
    /// - If the weighted map contains invalid keys
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<String, f64>) -> Self {
        Self::validate_pauli_leakage_keys(weighted_map);

        Self {
            sampler: WeightedSampler::new(weighted_map),
        }
    }

    fn validate_pauli_leakage_keys(weighted_map: &BTreeMap<String, f64>) {
        const VALID_KEYS: [&str; 4] = ["X", "Y", "Z", "L"];

        for key in weighted_map.keys() {
            assert!(
                VALID_KEYS.contains(&key.as_str()),
                "SingleQubitWeightedSampler: invalid key '{key}' - must be one of X, Y, Z, or L"
            );
        }
    }

    /// Get a reference to the normalized weighted map
    #[must_use]
    pub fn get_weighted_map(&self) -> &BTreeMap<String, f64> {
        self.sampler.get_weighted_map()
    }

    /// Sample a raw key from the distribution
    #[must_use]
    pub fn sample_keys(&self, rng: &mut NoiseRng) -> String {
        self.sampler.sample(rng)
    }

    /// Sample a gate operation for the given qubit
    ///
    /// # Panics
    /// - If the sampled key is invalid (this should never happen if the sampler was created properly)
    #[must_use]
    pub fn sample_gates(&self, rng: &mut NoiseRng, qubit: usize) -> SingleQubitNoiseResult {
        let key = self.sample_keys(rng);

        // Check for leakage first
        if key == "L" {
            return SingleQubitNoiseResult {
                gate: None,
                qubit_leaked: true,
            };
        }

        // For Pauli gates, create appropriate gate
        let gate = match key.as_str() {
            "X" => Gate::x(&[qubit]),
            "Y" => Gate::y(&[qubit]),
            "Z" => Gate::z(&[qubit]),
            _ => panic!(
                "SingleQubitWeightedSampler: invalid key '{key}' - must be one of \"X\", \"Y\", \"Z\", or \"L\""
            ),
        };

        SingleQubitNoiseResult {
            gate: Some(gate),
            qubit_leaked: false,
        }
    }
}

/// Samples two-qubit noise operations (pairs of Pauli gates or leakage)
#[derive(Clone, Debug)]
pub struct TwoQubitWeightedSampler {
    sampler: WeightedSampler<String>,
}

/// A basis-population state in PECOS's effective `{|0>, |1>, |L>}` space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QubitTransitionState {
    /// Computational basis state |0>.
    Zero,
    /// Computational basis state |1>.
    One,
    /// Abstract leakage state |L>.
    Leakage,
}

impl QubitTransitionState {
    fn parse(label: &str) -> Self {
        match label {
            "0" => Self::Zero,
            "1" => Self::One,
            "L" => Self::Leakage,
            _ => {
                panic!("QubitTransitionChannel: invalid state {label:?}; expected '0', '1', or 'L'")
            }
        }
    }
}

fn validate_transition_label(label: &str, arity: usize) {
    assert_eq!(
        label.chars().count(),
        arity,
        "TransitionDict: state {label:?} must contain exactly {arity} symbols"
    );
    for symbol in label.chars() {
        assert!(
            matches!(symbol, '0' | '1' | 'L'),
            "TransitionDict: invalid symbol {symbol:?} in state {label:?}; expected '0', '1', or 'L'"
        );
    }
}

fn transition_wildcard_position(label: &str) -> Option<usize> {
    let positions = label
        .chars()
        .enumerate()
        .filter_map(|(position, symbol)| (symbol == '*').then_some(position))
        .collect::<Vec<_>>();
    assert_eq!(
        positions.len(),
        1,
        "TransitionDict: wildcard state {label:?} must contain exactly one '*'"
    );
    positions.first().copied()
}

fn validate_wildcard_transition_label(label: &str, wildcard_position: usize) {
    assert_eq!(
        label.chars().count(),
        2,
        "TransitionDict: wildcard state {label:?} must contain exactly two symbols"
    );
    assert_eq!(
        transition_wildcard_position(label),
        Some(wildcard_position),
        "TransitionDict: destination {label:?} must preserve '*' in position {wildcard_position}"
    );
    for symbol in label.chars().filter(|symbol| *symbol != '*') {
        assert!(
            matches!(symbol, '0' | '1' | 'L'),
            "TransitionDict: invalid symbol {symbol:?} in state {label:?}; expected '0', '1', 'L', or '*'"
        );
    }
}

fn transition_basis_labels(arity: usize) -> Vec<String> {
    let mut labels = vec![String::new()];
    for _ in 0..arity {
        labels = labels
            .into_iter()
            .flat_map(|prefix| {
                ['0', '1', 'L'].map(move |symbol| {
                    let mut label = prefix.clone();
                    label.push(symbol);
                    label
                })
            })
            .collect();
    }
    labels
}

/// A validated conditional population-transition dictionary.
///
/// `transitions[source][destination]` is `P(destination | source)` over strings in
/// `{0, 1, L}^arity`. Supplied rows must sum to one. An omitted source row is exact identity and
/// retains that distinction through tensor products and sequential composition. Two-qubit maps
/// may instead use one matching `*` per source and destination as an unresolved identity wire.
///
/// Use [`Self::tensor`] (or `*`) to form a higher-arity Kronecker product. Use [`Self::compose`]
/// for matrix-style composition: `after.compose(&before)` applies `before` first and `after`
/// second.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionDict {
    arity: usize,
    transitions: BTreeMap<String, BTreeMap<String, f64>>,
    expanded_transitions: BTreeMap<String, BTreeMap<String, f64>>,
    computational_dependencies: Vec<bool>,
}

impl TransitionDict {
    /// Validate and construct a transition dictionary, inferring arity from its source labels.
    ///
    /// # Panics
    ///
    /// Panics if the dictionary is empty, labels are invalid or have inconsistent arity, or a
    /// supplied row is empty, invalid, or does not sum to one.
    #[must_use]
    pub fn new(transitions: &BTreeMap<String, BTreeMap<String, f64>>) -> Self {
        assert!(
            !transitions.is_empty(),
            "TransitionDict: transitions cannot be empty"
        );
        let arity = transitions
            .keys()
            .next()
            .expect("nonempty transition dictionary")
            .chars()
            .count();
        assert!(arity > 0, "TransitionDict: state labels cannot be empty");

        let contains_wildcard = transitions.iter().any(|(source, row)| {
            source.contains('*') || row.keys().any(|destination| destination.contains('*'))
        });
        if contains_wildcard {
            return Self::new_two_qubit_wildcard(transitions, arity);
        }

        let mut computational_dependencies = vec![false; arity];
        for (source, row) in transitions {
            validate_transition_label(source, arity);
            for (leg, symbol) in source.chars().enumerate() {
                computational_dependencies[leg] |= matches!(symbol, '0' | '1');
            }
            for destination in row.keys() {
                validate_transition_label(destination, arity);
            }
            let _ = WeightedSampler::new(row);
        }

        Self {
            arity,
            transitions: transitions.clone(),
            expanded_transitions: transitions.clone(),
            computational_dependencies,
        }
    }

    fn identity(arity: usize) -> Self {
        Self {
            arity,
            transitions: BTreeMap::new(),
            expanded_transitions: BTreeMap::new(),
            computational_dependencies: vec![false; arity],
        }
    }

    fn new_two_qubit_wildcard(
        transitions: &BTreeMap<String, BTreeMap<String, f64>>,
        arity: usize,
    ) -> Self {
        assert_eq!(
            arity, 2,
            "TransitionDict: '*' identity labels are supported only for two-qubit channels"
        );
        let mut factor_rows: [BTreeMap<String, BTreeMap<String, f64>>; 2] =
            std::array::from_fn(|_| BTreeMap::new());

        for (source, row) in transitions {
            let wildcard_position = transition_wildcard_position(source)
                .expect("wildcard transition source must contain '*'");
            validate_wildcard_transition_label(source, wildcard_position);
            assert!(
                !row.is_empty(),
                "TransitionDict: transition row for {source:?} cannot be empty"
            );
            let acted_position = 1 - wildcard_position;
            let source_state = source
                .chars()
                .nth(acted_position)
                .expect("two-symbol wildcard source");
            let mut factor_row = BTreeMap::new();
            for (destination, probability) in row {
                validate_wildcard_transition_label(destination, wildcard_position);
                let destination_state = destination
                    .chars()
                    .nth(acted_position)
                    .expect("two-symbol wildcard destination");
                factor_row.insert(destination_state.to_string(), *probability);
            }
            factor_rows[acted_position].insert(source_state.to_string(), factor_row);
        }

        let factors = factor_rows.map(|rows| {
            if rows.is_empty() {
                Self::identity(1)
            } else {
                Self::new(&rows)
            }
        });
        let expanded = factors[0].tensor(&factors[1]);
        Self {
            arity,
            transitions: transitions.clone(),
            expanded_transitions: expanded.expanded_transitions,
            computational_dependencies: expanded.computational_dependencies,
        }
    }

    /// Number of effective qutrits represented by each source and destination label.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Return the validated nested transition mapping.
    #[must_use]
    pub fn transitions(&self) -> &BTreeMap<String, BTreeMap<String, f64>> {
        &self.transitions
    }

    pub(crate) fn expanded_transitions(&self) -> &BTreeMap<String, BTreeMap<String, f64>> {
        &self.expanded_transitions
    }

    /// Whether applying the map can require resolving a computational source on each leg.
    #[must_use]
    pub(crate) fn computational_dependencies(&self) -> &[bool] {
        &self.computational_dependencies
    }

    fn effective_row(&self, source: &str) -> BTreeMap<String, f64> {
        self.expanded_transitions
            .get(source)
            .cloned()
            .unwrap_or_else(|| BTreeMap::from([(source.to_string(), 1.0)]))
    }

    /// Form the tensor/Kronecker product of two conditional transition dictionaries.
    ///
    /// Omitted rows are expanded as identity only where the other factor acts. The returned
    /// object retains per-leg dependency metadata so an identity factor does not cause an
    /// unnecessary computational-basis measurement when used as a joint channel.
    #[must_use]
    pub fn tensor(&self, other: &Self) -> Self {
        let mut transitions = BTreeMap::new();
        for left_source in transition_basis_labels(self.arity) {
            for right_source in transition_basis_labels(other.arity) {
                let left_explicit = self.expanded_transitions.contains_key(&left_source);
                let right_explicit = other.expanded_transitions.contains_key(&right_source);
                if !left_explicit && !right_explicit {
                    continue;
                }

                let source = format!("{left_source}{right_source}");
                let mut row = BTreeMap::new();
                for (left_destination, left_probability) in self.effective_row(&left_source) {
                    for (right_destination, right_probability) in other.effective_row(&right_source)
                    {
                        *row.entry(format!("{left_destination}{right_destination}"))
                            .or_insert(0.0) += left_probability * right_probability;
                    }
                }
                transitions.insert(source, row);
            }
        }

        let mut result = Self::new(&transitions);
        result.computational_dependencies = self
            .computational_dependencies
            .iter()
            .chain(&other.computational_dependencies)
            .copied()
            .collect();
        result
    }

    /// Compose equal-arity maps, applying `before` first and `self` second.
    ///
    /// # Panics
    ///
    /// Panics if the maps have different arities.
    #[must_use]
    pub fn compose(&self, before: &Self) -> Self {
        assert_eq!(
            self.arity, before.arity,
            "TransitionDict: sequential composition requires equal arity"
        );
        let mut transitions = BTreeMap::new();
        for source in transition_basis_labels(self.arity) {
            let before_row = before.effective_row(&source);
            let is_explicit = before.expanded_transitions.contains_key(&source)
                || before_row
                    .keys()
                    .any(|middle| self.expanded_transitions.contains_key(middle));
            if !is_explicit {
                continue;
            }

            let mut row = BTreeMap::new();
            for (middle, before_probability) in before_row {
                for (destination, after_probability) in self.effective_row(&middle) {
                    *row.entry(destination).or_insert(0.0) +=
                        before_probability * after_probability;
                }
            }
            transitions.insert(source, row);
        }

        let mut result = Self::new(&transitions);
        result.computational_dependencies = self
            .computational_dependencies
            .iter()
            .zip(&before.computational_dependencies)
            .map(|(after, before)| *after || *before)
            .collect();
        result
    }

    /// Apply `self` first and `next` second.
    #[must_use]
    pub fn then(&self, next: &Self) -> Self {
        next.compose(self)
    }
}

impl Mul for TransitionDict {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.tensor(&rhs)
    }
}

impl Mul<&TransitionDict> for &TransitionDict {
    type Output = TransitionDict;

    fn mul(self, rhs: &TransitionDict) -> Self::Output {
        self.tensor(rhs)
    }
}

impl Mul<&TransitionDict> for TransitionDict {
    type Output = Self;

    fn mul(self, rhs: &TransitionDict) -> Self::Output {
        self.tensor(rhs)
    }
}

impl Mul<TransitionDict> for &TransitionDict {
    type Output = TransitionDict;

    fn mul(self, rhs: TransitionDict) -> Self::Output {
        self.tensor(&rhs)
    }
}

/// A conditional population-transition channel on `{|0>, |1>, |L>}`.
///
/// `transitions[y][x]` is `P(x | y)`: the outer key is the source state and the inner key is the
/// destination. Each supplied source row must sum to one. An omitted source row is an exact
/// identity for that source, while an omitted destination inside a supplied row has probability
/// zero.
///
/// The channel is selected with the outer `probability`, giving the effective transition matrix
/// `T_eff = (1 - p) I + p T`. This outer coin provides a fast path for the low error rates typical
/// of QEC simulations.
///
/// A supplied computational row uses Z-basis measurement-and-transition semantics. It therefore
/// removes computational-basis coherence when the channel is selected. This is a classical
/// population channel, not a Pauli channel: use the general noise model's existing p1/p2 models
/// for coherent `rho -> P rho P` faults. PECOS tracks leakage as a classical state and does not
/// preserve coherence between the computational and leakage sectors.
///
/// # Example
///
/// Recover a leaked qubit with 90% probability, equally often to |0> and |1>:
///
/// ```
/// use std::collections::BTreeMap;
/// use pecos_engines::noise::{GeneralNoiseModel, QubitTransitionChannel};
///
/// let transitions = BTreeMap::from([(
///     "L".to_string(),
///     BTreeMap::from([("0".to_string(), 0.5), ("1".to_string(), 0.5)]),
/// )]);
/// let recovery = QubitTransitionChannel::new(0.9, &transitions);
/// let noise = GeneralNoiseModel::builder()
///     .add_p2_transition_channel_after_gate(&recovery)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct QubitTransitionChannel {
    probability: f64,
    transitions: TransitionDict,
}

impl QubitTransitionChannel {
    /// Configure an overall application probability and conditional transition rows.
    ///
    /// # Panics
    ///
    /// Panics if the outer probability is invalid, the transition map is empty, a state label is
    /// invalid, a supplied row is empty, or a row contains invalid probabilities or does not sum
    /// to one.
    #[must_use]
    pub fn new(probability: f64, transitions: &BTreeMap<String, BTreeMap<String, f64>>) -> Self {
        Self::from_transition_dict(probability, &TransitionDict::new(transitions))
    }

    /// Configure a channel from an already validated, single-qubit transition dictionary.
    ///
    /// # Panics
    ///
    /// Panics if `transitions` does not have arity one or `probability` is invalid.
    #[must_use]
    pub fn from_transition_dict(probability: f64, transitions: &TransitionDict) -> Self {
        assert!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "transition-channel probability must be finite and between 0 and 1, got {probability}"
        );
        assert_eq!(
            transitions.arity(),
            1,
            "QubitTransitionChannel requires a transition dictionary of arity one"
        );
        Self {
            probability,
            transitions: transitions.clone(),
        }
    }

    /// Return the unscaled outer application probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Return the configured conditional transition rows.
    #[must_use]
    pub fn transitions(&self) -> &BTreeMap<String, BTreeMap<String, f64>> {
        self.transitions.transitions()
    }

    /// Return the validated transition dictionary, including composition metadata.
    #[must_use]
    pub fn transition_dict(&self) -> &TransitionDict {
        &self.transitions
    }

    /// Configure leakage recovery with a biased computational-basis destination.
    ///
    /// # Panics
    ///
    /// Panics if either probability is not finite or lies outside `[0, 1]`.
    #[must_use]
    pub fn leak_recovery(recovery_probability: f64, p_zero: f64) -> Self {
        assert!(
            p_zero.is_finite() && (0.0..=1.0).contains(&p_zero),
            "p_zero must be finite and between 0 and 1, got {p_zero}"
        );
        let transitions = BTreeMap::from([(
            "L".to_string(),
            BTreeMap::from([("0".to_string(), p_zero), ("1".to_string(), 1.0 - p_zero)]),
        )]);
        Self::new(recovery_probability, &transitions)
    }
}

/// Compiled sampler for a [`QubitTransitionChannel`].
#[derive(Clone, Debug)]
pub struct QubitTransitionWeightedSampler {
    probability: f64,
    rows: BTreeMap<QubitTransitionState, WeightedSampler<QubitTransitionState>>,
}

impl QubitTransitionWeightedSampler {
    /// Compile an already validated transition channel.
    #[must_use]
    pub fn new(channel: &QubitTransitionChannel) -> Self {
        let rows = channel
            .transitions
            .transitions()
            .iter()
            .map(|(source, row)| {
                let typed_row = row
                    .iter()
                    .map(|(destination, probability)| {
                        (QubitTransitionState::parse(destination), *probability)
                    })
                    .collect();
                (
                    QubitTransitionState::parse(source),
                    WeightedSampler::new(&typed_row),
                )
            })
            .collect();
        Self {
            probability: channel.probability,
            rows,
        }
    }

    /// Draw the channel's outer application coin.
    #[must_use]
    pub fn is_selected(&self, rng: &mut NoiseRng) -> bool {
        rng.occurs(self.probability)
    }

    /// Whether a source state has an explicitly supplied transition row.
    #[must_use]
    pub fn has_row(&self, source: QubitTransitionState) -> bool {
        self.rows.contains_key(&source)
    }

    /// Whether the channel can depend on an unknown computational-basis source.
    #[must_use]
    pub(crate) fn has_computational_row(&self) -> bool {
        self.has_row(QubitTransitionState::Zero) || self.has_row(QubitTransitionState::One)
    }

    /// Sample `P(destination | source)`, using identity for an omitted source row.
    #[must_use]
    pub fn sample_destination(
        &self,
        rng: &mut NoiseRng,
        source: QubitTransitionState,
    ) -> QubitTransitionState {
        self.rows.get(&source).map_or(source, |row| row.sample(rng))
    }

    /// Return the scaled outer probability used by the compiled channel.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    pub(crate) fn scale_probability(&mut self, scale: f64) {
        self.probability = (self.probability * scale).clamp(0.0, 1.0);
    }
}

fn parse_two_qubit_transition_state(label: &str) -> [QubitTransitionState; 2] {
    let mut states = label
        .chars()
        .map(|state| QubitTransitionState::parse(&state.to_string()));
    let first = states.next().unwrap_or_else(|| {
        panic!("TwoQubitTransitionChannel: state {label:?} must contain exactly two symbols")
    });
    let second = states.next().unwrap_or_else(|| {
        panic!("TwoQubitTransitionChannel: state {label:?} must contain exactly two symbols")
    });
    assert!(
        states.next().is_none(),
        "TwoQubitTransitionChannel: state {label:?} must contain exactly two symbols"
    );
    [first, second]
}

/// A joint conditional population-transition channel on two effective qutrits.
///
/// `transitions[xy][wz]` is `P(wz | xy)`, where every concrete symbol is one of `0`, `1`, or `L`.
/// Thus there are nine possible source and destination labels, from `00` through `LL`. Supplied
/// rows must sum to one; omitted source rows are identity. A `*` in the same position in a source
/// and all its destinations is an identity wire that is neither measured nor otherwise resolved.
/// Factorized rules for both legs, such as `*L` and `L*`, may coexist in one channel.
#[derive(Clone, Debug)]
pub struct TwoQubitTransitionChannel {
    probability: f64,
    transitions: TransitionDict,
}

impl TwoQubitTransitionChannel {
    /// Configure a joint two-qubit conditional transition matrix.
    ///
    /// # Panics
    ///
    /// Panics under the same probability and row-validation conditions as
    /// [`QubitTransitionChannel::new`], or if a source or destination is not a two-symbol state.
    #[must_use]
    pub fn new(probability: f64, transitions: &BTreeMap<String, BTreeMap<String, f64>>) -> Self {
        Self::from_transition_dict(probability, &TransitionDict::new(transitions))
    }

    /// Configure a joint channel from an already validated, two-qubit transition dictionary.
    ///
    /// Passing a [`TransitionDict`] produced by [`TransitionDict::tensor`] preserves enough
    /// information to avoid resolving a computational-basis state on an identity factor.
    ///
    /// # Panics
    ///
    /// Panics if `transitions` does not have arity two or `probability` is invalid.
    #[must_use]
    pub fn from_transition_dict(probability: f64, transitions: &TransitionDict) -> Self {
        assert!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "two-qubit transition-channel probability must be finite and between 0 and 1, got {probability}"
        );
        assert_eq!(
            transitions.arity(),
            2,
            "TwoQubitTransitionChannel requires a transition dictionary of arity two"
        );
        Self {
            probability,
            transitions: transitions.clone(),
        }
    }

    /// Return the unscaled outer application probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Return the configured joint conditional transition rows.
    #[must_use]
    pub fn transitions(&self) -> &BTreeMap<String, BTreeMap<String, f64>> {
        self.transitions.transitions()
    }

    /// Return the validated transition dictionary, including composition metadata.
    #[must_use]
    pub fn transition_dict(&self) -> &TransitionDict {
        &self.transitions
    }
}

/// One ordered transition step attached to a two-qubit gate.
#[derive(Clone, Debug)]
pub enum P2TransitionStep {
    /// Apply distinct single-qubit channels independently to the first and second gate legs.
    Independent {
        /// Channel for the first gate leg.
        first: QubitTransitionChannel,
        /// Channel for the second gate leg.
        second: QubitTransitionChannel,
    },
    /// Apply one joint conditional matrix to the two-leg state.
    Joint(TwoQubitTransitionChannel),
}

impl P2TransitionStep {
    /// Construct an independent-leg step from two potentially different channels.
    #[must_use]
    pub fn independent(first: QubitTransitionChannel, second: QubitTransitionChannel) -> Self {
        Self::Independent { first, second }
    }

    /// Alias for [`Self::independent`] emphasizing the product of two one-qubit channels.
    #[must_use]
    pub fn tensor_product(first: QubitTransitionChannel, second: QubitTransitionChannel) -> Self {
        Self::independent(first, second)
    }

    /// Apply the same single-qubit channel independently to both gate legs.
    #[must_use]
    pub fn same_on_each(channel: QubitTransitionChannel) -> Self {
        Self::Independent {
            first: channel.clone(),
            second: channel,
        }
    }

    /// Construct a joint two-qubit transition step.
    #[must_use]
    pub fn joint(channel: TwoQubitTransitionChannel) -> Self {
        Self::Joint(channel)
    }
}

impl From<QubitTransitionChannel> for P2TransitionStep {
    fn from(channel: QubitTransitionChannel) -> Self {
        Self::same_on_each(channel)
    }
}

impl From<&QubitTransitionChannel> for P2TransitionStep {
    fn from(channel: &QubitTransitionChannel) -> Self {
        Self::same_on_each(channel.clone())
    }
}

impl From<TwoQubitTransitionChannel> for P2TransitionStep {
    fn from(channel: TwoQubitTransitionChannel) -> Self {
        Self::joint(channel)
    }
}

impl From<&TwoQubitTransitionChannel> for P2TransitionStep {
    fn from(channel: &TwoQubitTransitionChannel) -> Self {
        Self::joint(channel.clone())
    }
}

impl Mul for QubitTransitionChannel {
    type Output = P2TransitionStep;

    fn mul(self, rhs: Self) -> Self::Output {
        P2TransitionStep::tensor_product(self, rhs)
    }
}

impl Mul<&QubitTransitionChannel> for &QubitTransitionChannel {
    type Output = P2TransitionStep;

    fn mul(self, rhs: &QubitTransitionChannel) -> Self::Output {
        P2TransitionStep::tensor_product(self.clone(), rhs.clone())
    }
}

impl Mul<&QubitTransitionChannel> for QubitTransitionChannel {
    type Output = P2TransitionStep;

    fn mul(self, rhs: &QubitTransitionChannel) -> Self::Output {
        P2TransitionStep::tensor_product(self, rhs.clone())
    }
}

impl Mul<QubitTransitionChannel> for &QubitTransitionChannel {
    type Output = P2TransitionStep;

    fn mul(self, rhs: QubitTransitionChannel) -> Self::Output {
        P2TransitionStep::tensor_product(self.clone(), rhs)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum P2TransitionWeightedStep {
    Independent {
        first: QubitTransitionWeightedSampler,
        second: QubitTransitionWeightedSampler,
    },
    Joint {
        probability: f64,
        rows: BTreeMap<[QubitTransitionState; 2], WeightedSampler<[QubitTransitionState; 2]>>,
        computational_dependencies: [bool; 2],
    },
}

#[derive(Clone, Debug)]
pub(crate) enum SelectedP2TransitionStep {
    Independent {
        first: Option<QubitTransitionWeightedSampler>,
        second: Option<QubitTransitionWeightedSampler>,
    },
    Joint {
        rows: BTreeMap<[QubitTransitionState; 2], WeightedSampler<[QubitTransitionState; 2]>>,
        computational_dependencies: [bool; 2],
    },
}

impl P2TransitionWeightedStep {
    pub(crate) fn new(step: &P2TransitionStep) -> Self {
        match step {
            P2TransitionStep::Independent { first, second } => Self::Independent {
                first: QubitTransitionWeightedSampler::new(first),
                second: QubitTransitionWeightedSampler::new(second),
            },
            P2TransitionStep::Joint(channel) => Self::Joint {
                probability: channel.probability,
                computational_dependencies: channel
                    .transitions
                    .computational_dependencies()
                    .try_into()
                    .expect("two-qubit transition dictionary must have two dependency entries"),
                rows: channel
                    .transitions
                    .expanded_transitions()
                    .iter()
                    .map(|(source, row)| {
                        let typed_row = row
                            .iter()
                            .map(|(destination, probability)| {
                                (parse_two_qubit_transition_state(destination), *probability)
                            })
                            .collect();
                        (
                            parse_two_qubit_transition_state(source),
                            WeightedSampler::new(&typed_row),
                        )
                    })
                    .collect(),
            },
        }
    }

    pub(crate) fn scale_probability(&mut self, scale: f64) {
        match self {
            Self::Independent { first, second } => {
                first.scale_probability(scale);
                second.scale_probability(scale);
            }
            Self::Joint { probability, .. } => {
                *probability = (*probability * scale).clamp(0.0, 1.0);
            }
        }
    }

    pub(crate) fn select(&self, rng: &mut NoiseRng) -> Option<SelectedP2TransitionStep> {
        match self {
            Self::Independent { first, second } => {
                let first = first.is_selected(rng).then(|| first.clone());
                let second = second.is_selected(rng).then(|| second.clone());
                (first.is_some() || second.is_some())
                    .then_some(SelectedP2TransitionStep::Independent { first, second })
            }
            Self::Joint {
                probability,
                rows,
                computational_dependencies,
            } => rng
                .occurs(*probability)
                .then(|| SelectedP2TransitionStep::Joint {
                    rows: rows.clone(),
                    computational_dependencies: *computational_dependencies,
                }),
        }
    }
}

impl SelectedP2TransitionStep {
    pub(crate) fn sample_destinations(
        &self,
        rng: &mut NoiseRng,
        sources: [QubitTransitionState; 2],
    ) -> [QubitTransitionState; 2] {
        match self {
            Self::Independent { first, second } => [
                first.as_ref().map_or(sources[0], |channel| {
                    channel.sample_destination(rng, sources[0])
                }),
                second.as_ref().map_or(sources[1], |channel| {
                    channel.sample_destination(rng, sources[1])
                }),
            ],
            Self::Joint { rows, .. } => rows.get(&sources).map_or(sources, |row| row.sample(rng)),
        }
    }

    pub(crate) fn requires_computational_source(&self) -> [bool; 2] {
        match self {
            Self::Independent { first, second } => [
                first
                    .as_ref()
                    .is_some_and(QubitTransitionWeightedSampler::has_computational_row),
                second
                    .as_ref()
                    .is_some_and(QubitTransitionWeightedSampler::has_computational_row),
            ],
            Self::Joint {
                computational_dependencies,
                ..
            } => *computational_dependencies,
        }
    }

    pub(crate) fn sample_known_destinations(
        &self,
        rng: &mut NoiseRng,
        sources: [Option<QubitTransitionState>; 2],
    ) -> [Option<QubitTransitionState>; 2] {
        if let [Some(first), Some(second)] = sources {
            return self.sample_destinations(rng, [first, second]).map(Some);
        }

        match self {
            Self::Independent { first, second } => [
                sources[0].map(|source| {
                    first
                        .as_ref()
                        .map_or(source, |channel| channel.sample_destination(rng, source))
                }),
                sources[1].map(|source| {
                    second
                        .as_ref()
                        .map_or(source, |channel| channel.sample_destination(rng, source))
                }),
            ],
            Self::Joint { rows, .. } => {
                let concrete_sources = [
                    sources[0].unwrap_or(QubitTransitionState::Zero),
                    sources[1].unwrap_or(QubitTransitionState::Zero),
                ];
                let destinations = rows
                    .get(&concrete_sources)
                    .map_or(concrete_sources, |row| row.sample(rng));
                [
                    sources[0].map(|_| destinations[0]),
                    sources[1].map(|_| destinations[1]),
                ]
            }
        }
    }
}

impl TwoQubitWeightedSampler {
    /// Create a new two-qubit sampler from a weighted map
    ///
    /// Valid keys are two-character strings where each character is one of:
    /// "X", "Y", "Z", "I" (identity), or "L" (leakage)
    /// Note: "II" is not allowed as it represents no operation
    ///
    /// # Panics
    /// - If the weighted map contains invalid keys
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<String, f64>) -> Self {
        Self::validate_two_qubit_keys(weighted_map);

        Self {
            sampler: WeightedSampler::new(weighted_map),
        }
    }

    fn validate_two_qubit_keys(weighted_map: &BTreeMap<String, f64>) {
        const VALID_CHARS: [char; 5] = ['X', 'Y', 'Z', 'I', 'L'];

        for key in weighted_map.keys() {
            // Key must be exactly 2 characters long
            assert_eq!(
                key.len(),
                2,
                "TwoQubitWeightedSampler: invalid key '{key}' - must be exactly 2 characters"
            );

            // Check each character is valid
            for c in key.chars() {
                assert!(
                    VALID_CHARS.contains(&c),
                    "TwoQubitWeightedSampler: invalid character '{c}' in key '{key}' - must be one of X, Y, Z, I, or L"
                );
            }

            // Special case: "II" is not allowed
            assert_ne!(
                key.as_str(),
                "II",
                "TwoQubitWeightedSampler: key 'II' is not allowed as it represents no operation"
            );
        }
    }

    /// Get a reference to the normalized weighted map
    #[must_use]
    pub fn get_weighted_map(&self) -> &BTreeMap<String, f64> {
        self.sampler.get_weighted_map()
    }

    /// Sample a raw key from the distribution
    #[must_use]
    pub fn sample_keys(&self, rng: &mut NoiseRng) -> String {
        self.sampler.sample(rng)
    }

    /// Sample gate operations for the given qubit pair
    ///
    /// # Panics
    /// - If the sampled key is invalid (this should never happen if the sampler was created properly)
    #[must_use]
    pub fn sample_gates(
        &self,
        rng: &mut NoiseRng,
        qubit0: usize,
        qubit1: usize,
    ) -> TwoQubitNoiseResult {
        // Sample a key and extract the characters
        let key_str = self.sample_keys(rng);
        let chars: Vec<char> = key_str.chars().collect();

        // Determine leakage status
        let qubit0_leaked = chars[0] == 'L';
        let qubit1_leaked = chars[1] == 'L';

        // If both qubits leaked, no gates needed
        if qubit0_leaked && qubit1_leaked {
            return TwoQubitNoiseResult::with_leakage(true, true, None);
        }

        // Build gates for non-leaked qubits only
        let mut gates = Vec::new();

        // Convert the first operation if not leaked
        if !qubit0_leaked && let Some(gate) = create_pauli_gate(chars[0], qubit0) {
            gates.push(gate);
        }

        // Convert the second operation if not leaked
        if !qubit1_leaked && let Some(gate) = create_pauli_gate(chars[1], qubit1) {
            gates.push(gate);
        }

        // Only return gates if we have some
        let gates_option = if gates.is_empty() { None } else { Some(gates) };

        TwoQubitNoiseResult::with_leakage(qubit0_leaked, qubit1_leaked, gates_option)
    }
}

/// One Pauli-plus-leakage step attached to a two-qubit gate hook.
#[derive(Clone, Debug)]
pub enum P2PauliLeakageStep {
    /// Draw distinct single-qubit channels independently on the two gate legs.
    Independent {
        /// Channel for the first gate leg.
        first: PauliLeakageChannel,
        /// Channel for the second gate leg.
        second: PauliLeakageChannel,
    },
    /// Draw one correlated joint event for the pair.
    Joint(TwoQubitPauliLeakageChannel),
}

impl P2PauliLeakageStep {
    /// Construct an independent-leg step from two potentially different channels.
    #[must_use]
    pub fn independent(first: PauliLeakageChannel, second: PauliLeakageChannel) -> Self {
        Self::Independent { first, second }
    }

    /// Alias for [`Self::independent`] emphasizing the product of two one-qubit channels.
    #[must_use]
    pub fn tensor_product(first: PauliLeakageChannel, second: PauliLeakageChannel) -> Self {
        Self::independent(first, second)
    }

    /// Apply the same independently drawn channel to both legs.
    #[must_use]
    pub fn same_on_each(channel: PauliLeakageChannel) -> Self {
        Self::Independent {
            first: channel.clone(),
            second: channel,
        }
    }

    /// Apply one correlated two-qubit event channel.
    #[must_use]
    pub fn joint(channel: TwoQubitPauliLeakageChannel) -> Self {
        Self::Joint(channel)
    }
}

impl Mul for PauliLeakageChannel {
    type Output = P2PauliLeakageStep;

    fn mul(self, rhs: Self) -> Self::Output {
        P2PauliLeakageStep::tensor_product(self, rhs)
    }
}

impl Mul<&PauliLeakageChannel> for &PauliLeakageChannel {
    type Output = P2PauliLeakageStep;

    fn mul(self, rhs: &PauliLeakageChannel) -> Self::Output {
        P2PauliLeakageStep::tensor_product(self.clone(), rhs.clone())
    }
}

impl Mul<&PauliLeakageChannel> for PauliLeakageChannel {
    type Output = P2PauliLeakageStep;

    fn mul(self, rhs: &PauliLeakageChannel) -> Self::Output {
        P2PauliLeakageStep::tensor_product(self, rhs.clone())
    }
}

impl Mul<PauliLeakageChannel> for &PauliLeakageChannel {
    type Output = P2PauliLeakageStep;

    fn mul(self, rhs: PauliLeakageChannel) -> Self::Output {
        P2PauliLeakageStep::tensor_product(self.clone(), rhs)
    }
}

/// Compiled sampler for a [`PauliLeakageChannel`].
#[derive(Clone, Debug)]
pub struct PauliLeakageWeightedSampler {
    probability: f64,
    sampler: SingleQubitWeightedSampler,
}

impl PauliLeakageWeightedSampler {
    /// Compile an already validated channel.
    #[must_use]
    pub fn new(channel: &PauliLeakageChannel) -> Self {
        Self {
            probability: channel.probability,
            sampler: SingleQubitWeightedSampler::new(channel.events()),
        }
    }

    /// Draw the outer coin and, when selected, one event symbol.
    pub(crate) fn sample_event(&self, rng: &mut NoiseRng) -> Option<char> {
        self.is_selected(rng).then(|| {
            self.sampler
                .sample_keys(rng)
                .chars()
                .next()
                .expect("arity-one event")
        })
    }

    /// Draw the channel's outer application coin.
    #[must_use]
    pub fn is_selected(&self, rng: &mut NoiseRng) -> bool {
        rng.occurs(self.probability)
    }

    /// Return the scaled outer probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    pub(crate) fn scale_probability(&mut self, scale: f64) {
        self.probability = (self.probability * scale).clamp(0.0, 1.0);
    }
}

/// Compiled sampler for a [`TwoQubitPauliLeakageChannel`].
#[derive(Clone, Debug)]
pub struct TwoQubitPauliLeakageWeightedSampler {
    probability: f64,
    sampler: TwoQubitWeightedSampler,
}

impl TwoQubitPauliLeakageWeightedSampler {
    /// Compile an already validated channel.
    #[must_use]
    pub fn new(channel: &TwoQubitPauliLeakageChannel) -> Self {
        Self {
            probability: channel.probability,
            sampler: TwoQubitWeightedSampler::new(channel.events()),
        }
    }

    fn sample_events(&self, rng: &mut NoiseRng) -> Option<[char; 2]> {
        self.is_selected(rng).then(|| {
            self.sampler
                .sample_keys(rng)
                .chars()
                .collect::<Vec<_>>()
                .try_into()
                .expect("arity-two event")
        })
    }

    /// Draw the channel's outer application coin.
    #[must_use]
    pub fn is_selected(&self, rng: &mut NoiseRng) -> bool {
        rng.occurs(self.probability)
    }

    /// Return the scaled outer probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    pub(crate) fn scale_probability(&mut self, scale: f64) {
        self.probability = (self.probability * scale).clamp(0.0, 1.0);
    }
}

/// Compiled sampler for a [`P2PauliLeakageStep`].
#[derive(Clone, Debug)]
pub(crate) enum P2PauliLeakageWeightedStep {
    Independent {
        first: PauliLeakageWeightedSampler,
        second: PauliLeakageWeightedSampler,
    },
    Joint(TwoQubitPauliLeakageWeightedSampler),
}

impl P2PauliLeakageWeightedStep {
    pub(crate) fn new(step: &P2PauliLeakageStep) -> Self {
        match step {
            P2PauliLeakageStep::Independent { first, second } => Self::Independent {
                first: PauliLeakageWeightedSampler::new(first),
                second: PauliLeakageWeightedSampler::new(second),
            },
            P2PauliLeakageStep::Joint(channel) => {
                Self::Joint(TwoQubitPauliLeakageWeightedSampler::new(channel))
            }
        }
    }

    pub(crate) fn sample_events(&self, rng: &mut NoiseRng) -> [Option<char>; 2] {
        match self {
            Self::Independent { first, second } => {
                [first.sample_event(rng), second.sample_event(rng)]
            }
            Self::Joint(channel) => channel
                .sample_events(rng)
                .map_or([None, None], |events| events.map(Some)),
        }
    }

    pub(crate) fn scale_probability(&mut self, scale: f64) {
        match self {
            Self::Independent { first, second } => {
                first.scale_probability(scale);
                second.scale_probability(scale);
            }
            Self::Joint(channel) => channel.scale_probability(scale),
        }
    }
}

/// Samples crosstalk noise transitions
#[derive(Clone, Debug)]
pub struct CrosstalkWeightedSampler {
    sampler_from_0: WeightedSampler<String>,
    sampler_from_1: WeightedSampler<String>,
}

impl CrosstalkWeightedSampler {
    /// Create a new crosstalk sampler from a weighted map
    ///
    /// Valid keys are: "0->0", "0->1", "0->L", "1->1", "1->0", "1->L"
    ///
    /// # Panics
    /// - If the weighted map contains invalid keys
    /// - If the weighted map is empty
    /// - If the total weight of each sampler is not positive
    /// - If the total weight of each sampler deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<String, f64>) -> Self {
        const KEYS_FROM_0: [&str; 3] = ["0->0", "0->1", "0->L"];
        const KEYS_FROM_1: [&str; 3] = ["1->1", "1->0", "1->L"];
        Self::validate_crosstalk_keys(weighted_map);

        // Separate the 0->* components from the 1->* components
        let weighted_map_from_0 = KEYS_FROM_0
            .into_iter()
            .filter_map(|key| weighted_map.get(key).map(|&val| (key.to_string(), val)))
            .collect();
        let weighted_map_from_1 = KEYS_FROM_1
            .into_iter()
            .filter_map(|key| weighted_map.get(key).map(|&val| (key.to_string(), val)))
            .collect();

        Self {
            sampler_from_0: WeightedSampler::new(&weighted_map_from_0),
            sampler_from_1: WeightedSampler::new(&weighted_map_from_1),
        }
    }

    fn validate_crosstalk_keys(weighted_map: &BTreeMap<String, f64>) {
        const VALID_KEYS: [&str; 6] = ["0->0", "0->1", "0->L", "1->1", "1->0", "1->L"];

        for key in weighted_map.keys() {
            assert!(
                VALID_KEYS.contains(&key.as_str()),
                "CrosstalkWeightedSampler: invalid key '{key}' - must be one of {VALID_KEYS:?}",
            );
        }
    }

    /// Get a reference to the normalized weighted map, for keys 0->* or 1->*
    /// # Panics
    /// - If `from_state` is not either 0 or 1.
    #[must_use]
    pub fn get_weighted_map(&self, from_state: u32) -> &BTreeMap<String, f64> {
        assert!(from_state == 0 || from_state == 1);
        if from_state == 0 {
            self.sampler_from_0.get_weighted_map()
        } else {
            self.sampler_from_1.get_weighted_map()
        }
    }

    /// Sample a raw key from the distribution, for keys 0->* or 1->*.
    /// # Panics
    /// - If `from_state` is not either 0 or 1.
    #[must_use]
    pub fn sample_keys(&self, rng: &mut NoiseRng, from_state: u32) -> String {
        assert!(from_state == 0 || from_state == 1);
        if from_state == 0 {
            self.sampler_from_0.sample(rng)
        } else {
            self.sampler_from_1.sample(rng)
        }
    }

    /// Sample a gate operation for the given qubit
    ///
    /// # Panics
    /// - If the sampled key is invalid (this should never happen if the sampler was created properly)
    #[must_use]
    pub fn sample_gates(
        &self,
        rng: &mut NoiseRng,
        qubit: usize,
        from_state: u32,
    ) -> SingleQubitNoiseResult {
        let key = self.sample_keys(rng, from_state);

        match key.as_str() {
            "0->0" | "1->1" => SingleQubitNoiseResult {
                gate: None,
                qubit_leaked: false,
            },
            "0->1" | "1->0" => SingleQubitNoiseResult {
                gate: Some(Gate::x(&[qubit])),
                qubit_leaked: false,
            },
            "0->L" | "1->L" => SingleQubitNoiseResult {
                gate: None,
                qubit_leaked: true,
            },
            _ => panic!("CrosstalkWeightedSampler: invalid key '{key}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::noise_rng::NoiseRng;
    use pecos_random::PecosRng;

    const SAMPLE_SIZE: usize = 100;

    #[test]
    fn test_different_sampler_instances_same_results() {
        // Create two weighted samplers with the same weights
        let mut weights1 = BTreeMap::new();
        weights1.insert("A".to_string(), 0.3);
        weights1.insert("B".to_string(), 0.7);

        // Make a separate instance with the same data
        let mut weights2 = BTreeMap::new();
        weights2.insert("A".to_string(), 0.3);
        weights2.insert("B".to_string(), 0.7);

        let sampler1 = WeightedSampler::new(&weights1);
        let sampler2 = WeightedSampler::new(&weights2);

        // Use the same seed for both RNGs
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Sample from both samplers
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler1.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler2.sample(&mut rng2))
            .collect();

        // Results should be identical with same seed
        assert_eq!(
            results1, results2,
            "Different sampler instances with same weights should produce identical results with same seed"
        );
    }

    #[test]
    fn test_deterministic_ordering_with_shuffled_keys() {
        // Create two weighted samplers with the same weights but different insertion order
        let mut weights1 = BTreeMap::new();
        weights1.insert("A".to_string(), 0.3);
        weights1.insert("B".to_string(), 0.2);
        weights1.insert("C".to_string(), 0.5);

        // Insert in different order
        let mut weights2 = BTreeMap::new();
        weights2.insert("C".to_string(), 0.5);
        weights2.insert("A".to_string(), 0.3);
        weights2.insert("B".to_string(), 0.2);

        let sampler1 = WeightedSampler::new(&weights1);
        let sampler2 = WeightedSampler::new(&weights2);

        // Use the same seed for both RNGs
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Sample from both samplers
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler1.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler2.sample(&mut rng2))
            .collect();

        // Results should be identical despite different insertion order
        assert_eq!(
            results1, results2,
            "Samplers with differently ordered but equivalent maps should produce identical results"
        );
    }

    #[test]
    fn test_deterministic_sampling_basic() {
        // Test basic deterministic sampling with same seed
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng2))
            .collect();

        // Verify exact sequence match
        assert_eq!(
            results1, results2,
            "Sampling results should be identical with same seed"
        );
    }

    #[test]
    fn test_deterministic_sampling_multiple_seeds() {
        // Test deterministic sampling with multiple different seeds
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);

        // Test multiple seed pairs
        let seed_pairs = [(42, 42), (123, 123), (999, 999), (0, 0)];

        for (seed1, seed2) in seed_pairs {
            let mut rng1 = NoiseRng::<PecosRng>::with_seed(seed1);
            let mut rng2 = NoiseRng::<PecosRng>::with_seed(seed2);

            let results1: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng1))
                .collect();
            let results2: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng2))
                .collect();

            assert_eq!(
                results1, results2,
                "Sampling results should be identical with same seed pair ({seed1}, {seed2})"
            );
        }
    }

    #[test]
    fn test_deterministic_sampling_different_seeds() {
        // Test that different seeds produce different sequences
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);

        // Test multiple different seed pairs
        let seed_pairs = [(42, 43), (123, 124), (999, 1000), (0, 1)];

        for (seed1, seed2) in seed_pairs {
            let mut rng1 = NoiseRng::<PecosRng>::with_seed(seed1);
            let mut rng2 = NoiseRng::<PecosRng>::with_seed(seed2);

            let results1: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng1))
                .collect();
            let results2: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng2))
                .collect();

            assert_ne!(
                results1, results2,
                "Sampling results should differ with different seed pair ({seed1}, {seed2})"
            );
        }
    }

    #[test]
    fn test_deterministic_sampling_single_qubit() {
        // Test deterministic sampling with single qubit sampler
        let mut weights = BTreeMap::new();
        weights.insert("X".to_string(), 0.25);
        weights.insert("Y".to_string(), 0.25);
        weights.insert("Z".to_string(), 0.25);
        weights.insert("L".to_string(), 0.25);

        let sampler = SingleQubitWeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            match (&r1.gate, &r2.gate) {
                (Some(g1), Some(g2)) => assert_eq!(
                    g1.gate_type, g2.gate_type,
                    "Gate type mismatch at index {i}"
                ),
                (None, None) => (),
                _ => panic!("Gate presence mismatch at index {i}"),
            }
        }
    }

    #[test]
    fn test_deterministic_sampling_two_qubit() {
        // Test deterministic sampling with two qubit sampler
        let mut weights = BTreeMap::new();
        weights.insert("XX".to_string(), 0.2);
        weights.insert("YY".to_string(), 0.2);
        weights.insert("ZZ".to_string(), 0.2);
        weights.insert("XL".to_string(), 0.2);
        weights.insert("LX".to_string(), 0.2);

        let sampler = TwoQubitWeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 1))
            .collect();
        let results2: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 1))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit0_leaked, r2.qubit0_leaked,
                "Qubit 0 leakage mismatch at index {i}"
            );
            assert_eq!(
                r1.qubit1_leaked, r2.qubit1_leaked,
                "Qubit 1 leakage mismatch at index {i}"
            );
            match (&r1.gates, &r2.gates) {
                (Some(g1), Some(g2)) => {
                    assert_eq!(g1.len(), g2.len(), "Gate count mismatch at index {i}");
                    for (j, (gate1, gate2)) in g1.iter().zip(g2.iter()).enumerate() {
                        assert_eq!(
                            gate1.gate_type, gate2.gate_type,
                            "Gate type mismatch at index {i} for gate {j}"
                        );
                    }
                }
                (None, None) => (),
                _ => panic!("Gate presence mismatch at index {i}"),
            }
        }
    }

    #[test]
    fn test_deterministic_sampling_crosstalk() {
        // Test deterministic sampling with single qubit sampler
        let mut weights = BTreeMap::new();
        weights.insert("0->1".to_string(), 0.5);
        weights.insert("0->L".to_string(), 0.5);
        weights.insert("1->0".to_string(), 0.5);
        weights.insert("1->L".to_string(), 0.5);

        let sampler = CrosstalkWeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 0))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 0))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            match (&r1.gate, &r2.gate) {
                (Some(g1), Some(g2)) => assert_eq!(
                    g1.gate_type, g2.gate_type,
                    "Gate type mismatch at index {i}"
                ),
                (None, None) => (),
                _ => panic!("Gate presence mismatch at index {i}"),
            }
        }
    }

    #[test]
    fn test_deterministic_sampling_reset() {
        // Test that resetting the RNG and using the same seed produces the same sequence
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);
        let seed = 42;

        // First sequence
        let mut rng = NoiseRng::<PecosRng>::with_seed(seed);
        let results1: Vec<String> = (0..SAMPLE_SIZE).map(|_| sampler.sample(&mut rng)).collect();

        // Reset RNG with same seed
        rng = NoiseRng::<PecosRng>::with_seed(seed);
        let results2: Vec<String> = (0..SAMPLE_SIZE).map(|_| sampler.sample(&mut rng)).collect();

        // Verify exact sequence match
        assert_eq!(
            results1, results2,
            "Sampling results should be identical after RNG reset with same seed"
        );
    }

    #[test]
    fn test_deterministic_sampling_consecutive() {
        // Test that consecutive samples from the same RNG are deterministic
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        // Take two consecutive samples
        let result1 = sampler.sample(&mut rng);
        let result2 = sampler.sample(&mut rng);

        // Reset RNG and take the same two samples
        rng = NoiseRng::<PecosRng>::with_seed(42);
        let result3 = sampler.sample(&mut rng);
        let result4 = sampler.sample(&mut rng);

        // Verify the sequences match
        assert_eq!(result1, result3, "First sample should be deterministic");
        assert_eq!(result2, result4, "Second sample should be deterministic");
    }

    #[test]
    fn test_deterministic_sampling_interleaved() {
        // Test that interleaved sampling from different samplers is deterministic
        let mut weights1 = BTreeMap::new();
        weights1.insert("A".to_string(), 0.3);
        weights1.insert("B".to_string(), 0.7);

        let mut weights2 = BTreeMap::new();
        weights2.insert("X".to_string(), 0.4);
        weights2.insert("Y".to_string(), 0.6);

        let sampler1 = WeightedSampler::new(&weights1);
        let sampler2 = WeightedSampler::new(&weights2);

        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Interleaved sampling
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| {
                if rng1.random_float() < 0.5 {
                    sampler1.sample(&mut rng1)
                } else {
                    sampler2.sample(&mut rng2)
                }
            })
            .collect();

        // Reset RNGs and repeat
        rng1 = NoiseRng::<PecosRng>::with_seed(42);
        rng2 = NoiseRng::<PecosRng>::with_seed(42);

        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| {
                if rng1.random_float() < 0.5 {
                    sampler1.sample(&mut rng1)
                } else {
                    sampler2.sample(&mut rng2)
                }
            })
            .collect();

        assert_eq!(
            results1, results2,
            "Interleaved sampling should be deterministic"
        );
    }

    #[test]
    fn test_deterministic_sampling_edge_cases() {
        // Test edge cases for sampling
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 1.0); // Single outcome with probability 1.0

        let sampler = WeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        // Should always get "A" regardless of RNG state
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng2))
            .collect();

        assert_eq!(
            results1, results2,
            "Sampling should be deterministic even with single outcome"
        );
        assert!(
            results1.iter().all(|x| x == "A"),
            "All results should be 'A'"
        );
    }

    #[test]
    fn test_deterministic_sampling_single_qubit_edge_cases() {
        // Test edge cases for single qubit sampling
        let mut weights = BTreeMap::new();
        weights.insert("L".to_string(), 1.0); // Always leak

        let sampler = SingleQubitWeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            assert!(r1.qubit_leaked, "All results should indicate leakage");
            assert!(r1.gate.is_none(), "No gates should be present");
        }
    }

    #[test]
    fn test_deterministic_sampling_two_qubit_edge_cases() {
        // Test edge cases for two qubit sampling
        let mut weights = BTreeMap::new();
        weights.insert("LL".to_string(), 1.0); // Always leak both qubits

        let sampler = TwoQubitWeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        let results1: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 1))
            .collect();
        let results2: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 1))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit0_leaked, r2.qubit0_leaked,
                "Qubit 0 leakage mismatch at index {i}"
            );
            assert_eq!(
                r1.qubit1_leaked, r2.qubit1_leaked,
                "Qubit 1 leakage mismatch at index {i}"
            );
            assert!(
                r1.qubit0_leaked && r1.qubit1_leaked,
                "Both qubits should leak"
            );
            assert!(r1.gates.is_none(), "No gates should be present");
        }
    }

    #[test]
    fn test_deterministic_sampling_crosstalk_edge_cases() {
        // Test edge cases for single qubit sampling
        let mut weights = BTreeMap::new();
        weights.insert("0->L".to_string(), 1.0); // Always leak
        weights.insert("1->L".to_string(), 1.0); // Always leak

        let sampler = CrosstalkWeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 1))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 1))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            assert!(r1.qubit_leaked, "All results should indicate leakage");
            assert!(r1.gate.is_none(), "No gates should be present");
        }
    }

    #[test]
    fn pauli_leakage_dict_tensor_product_multiplies_relative_weights() {
        let first = PauliLeakageDict::new(&BTreeMap::from([
            ("X".to_string(), 0.25),
            ("L".to_string(), 0.75),
        ]));
        let second = PauliLeakageDict::new(&BTreeMap::from([
            ("Y".to_string(), 0.4),
            ("Z".to_string(), 0.6),
        ]));

        let product = &first * &second;

        assert_eq!(product.arity(), 2);
        assert!((product.events()["XY"] - 0.10).abs() < f64::EPSILON);
        assert!((product.events()["XZ"] - 0.15).abs() < f64::EPSILON);
        assert!((product.events()["LY"] - 0.30).abs() < f64::EPSILON);
        assert!((product.events()["LZ"] - 0.45).abs() < f64::EPSILON);
    }

    #[test]
    fn pauli_leakage_channel_product_preserves_independent_outer_coins() {
        let first = PauliLeakageChannel::new(0.9, &BTreeMap::from([("L".to_string(), 1.0)]));
        let second = PauliLeakageChannel::new(0.8, &BTreeMap::from([("X".to_string(), 1.0)]));

        let P2PauliLeakageStep::Independent { first, second } = &first * &second else {
            panic!("channel product must produce an independent step");
        };

        assert!((first.probability() - 0.9).abs() < f64::EPSILON);
        assert!((second.probability() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "all-identity event")]
    fn pauli_leakage_dict_rejects_all_identity_event() {
        let _ = PauliLeakageDict::new(&BTreeMap::from([("II".to_string(), 1.0)]));
    }

    #[test]
    fn transition_channel_uses_conditional_rows_and_omitted_rows_are_identity() {
        let transitions = BTreeMap::from([
            (
                "0".to_string(),
                BTreeMap::from([
                    ("0".to_string(), 0.05),
                    ("1".to_string(), 0.05),
                    ("L".to_string(), 0.90),
                ]),
            ),
            (
                "L".to_string(),
                BTreeMap::from([
                    ("0".to_string(), 0.45),
                    ("1".to_string(), 0.45),
                    ("L".to_string(), 0.10),
                ]),
            ),
        ]);
        let channel = QubitTransitionChannel::new(0.25, &transitions);
        let sampler = QubitTransitionWeightedSampler::new(&channel);
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        assert!((channel.probability() - 0.25).abs() < f64::EPSILON);
        assert!(sampler.has_computational_row());
        assert_eq!(
            sampler.sample_destination(&mut rng, QubitTransitionState::One),
            QubitTransitionState::One
        );
    }

    #[test]
    fn transition_dict_tensor_product_expands_only_affected_rows() {
        let first = TransitionDict::new(&BTreeMap::from([(
            "L".to_string(),
            BTreeMap::from([("0".to_string(), 1.0)]),
        )]));
        let second = TransitionDict::new(&BTreeMap::from([(
            "L".to_string(),
            BTreeMap::from([("1".to_string(), 1.0)]),
        )]));

        let product = &first * &second;

        assert_eq!(product.arity(), 2);
        assert_eq!(product.transitions().len(), 5);
        assert_eq!(
            product.transitions()["0L"],
            BTreeMap::from([("01".to_string(), 1.0)])
        );
        assert_eq!(
            product.transitions()["L0"],
            BTreeMap::from([("00".to_string(), 1.0)])
        );
        assert_eq!(
            product.transitions()["LL"],
            BTreeMap::from([("01".to_string(), 1.0)])
        );
        assert_eq!(product.computational_dependencies(), &[false, false]);
    }

    #[test]
    fn transition_dict_composes_in_matrix_order() {
        let leak_zero = TransitionDict::new(&BTreeMap::from([(
            "0".to_string(),
            BTreeMap::from([("L".to_string(), 1.0)]),
        )]));
        let recover_to_one = TransitionDict::new(&BTreeMap::from([(
            "L".to_string(),
            BTreeMap::from([("1".to_string(), 1.0)]),
        )]));

        let composed = recover_to_one.compose(&leak_zero);

        assert_eq!(
            composed.transitions()["0"],
            BTreeMap::from([("1".to_string(), 1.0)])
        );
        assert_eq!(leak_zero.then(&recover_to_one), composed);
    }

    #[test]
    fn multiplying_single_qubit_channels_preserves_independent_outer_coins() {
        let first = QubitTransitionChannel::leak_recovery(0.9, 1.0);
        let second = QubitTransitionChannel::leak_recovery(0.8, 0.0);

        let P2TransitionStep::Independent { first, second } = &first * &second else {
            panic!("channel product must produce an independent transition step");
        };

        assert!((first.probability() - 0.9).abs() < f64::EPSILON);
        assert!((second.probability() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn transition_application_probability_is_close_to_requested_probability() {
        let channel = QubitTransitionChannel::leak_recovery(0.9, 0.5);
        let sampler = QubitTransitionWeightedSampler::new(&channel);
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let samples = 10_000_u32;
        let selected = (0..samples)
            .filter(|_| sampler.is_selected(&mut rng))
            .count();
        let observed = f64::from(u32::try_from(selected).unwrap()) / f64::from(samples);

        assert!(
            (observed - 0.9).abs() < 0.02,
            "observed probability {observed}"
        );
    }

    #[test]
    fn leak_recovery_helper_uses_conditional_destination_probabilities() {
        let channel = QubitTransitionChannel::leak_recovery(1.0, 0.25);
        let sampler = QubitTransitionWeightedSampler::new(&channel);
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let samples = 10_000_u32;
        let zero = (0..samples)
            .filter(|_| {
                sampler.sample_destination(&mut rng, QubitTransitionState::Leakage)
                    == QubitTransitionState::Zero
            })
            .count();
        let observed = f64::from(u32::try_from(zero).unwrap()) / f64::from(samples);

        assert!((observed - 0.25).abs() < 0.02, "observed P(0|L) {observed}");
        assert_eq!(
            channel.transitions()["L"],
            BTreeMap::from([("0".to_string(), 0.25), ("1".to_string(), 0.75)])
        );
    }

    #[test]
    fn joint_two_qubit_transition_supports_all_qutrit_pair_labels() {
        let transitions =
            BTreeMap::from([("0L".to_string(), BTreeMap::from([("L1".to_string(), 1.0)]))]);
        let channel = TwoQubitTransitionChannel::new(1.0, &transitions);
        let step = P2TransitionWeightedStep::new(&P2TransitionStep::joint(channel));
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let selected = step.select(&mut rng).unwrap();

        assert_eq!(
            selected.sample_known_destinations(
                &mut rng,
                [
                    Some(QubitTransitionState::Zero),
                    Some(QubitTransitionState::Leakage),
                ],
            ),
            [
                Some(QubitTransitionState::Leakage),
                Some(QubitTransitionState::One),
            ]
        );
    }

    #[test]
    fn wildcard_transition_rules_preserve_the_other_leg_without_resolving_it() {
        let transitions =
            BTreeMap::from([("*L".to_string(), BTreeMap::from([("*0".to_string(), 1.0)]))]);
        let channel = TwoQubitTransitionChannel::new(1.0, &transitions);
        let step = P2TransitionWeightedStep::new(&P2TransitionStep::joint(channel.clone()));
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let selected = step.select(&mut rng).unwrap();

        assert_eq!(channel.transitions(), &transitions);
        assert_eq!(selected.requires_computational_source(), [false, false]);
        assert_eq!(
            selected
                .sample_known_destinations(&mut rng, [None, Some(QubitTransitionState::Leakage)],),
            [None, Some(QubitTransitionState::Zero)]
        );
    }

    #[test]
    fn wildcard_rules_for_both_legs_compose_for_double_leakage() {
        let transitions = BTreeMap::from([
            ("*L".to_string(), BTreeMap::from([("*0".to_string(), 1.0)])),
            ("L*".to_string(), BTreeMap::from([("1*".to_string(), 1.0)])),
        ]);
        let channel = TwoQubitTransitionChannel::new(1.0, &transitions);
        let step = P2TransitionWeightedStep::new(&P2TransitionStep::joint(channel));
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let selected = step.select(&mut rng).unwrap();

        assert_eq!(
            selected.sample_known_destinations(
                &mut rng,
                [
                    Some(QubitTransitionState::Leakage),
                    Some(QubitTransitionState::Leakage),
                ],
            ),
            [
                Some(QubitTransitionState::One),
                Some(QubitTransitionState::Zero),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "must contain exactly one '*'")]
    fn wildcard_transition_rejects_destination_that_touches_identity_leg() {
        let transitions =
            BTreeMap::from([("*L".to_string(), BTreeMap::from([("00".to_string(), 1.0)]))]);
        let _ = TwoQubitTransitionChannel::new(1.0, &transitions);
    }

    #[test]
    fn independent_p2_step_can_use_distinct_leg_channels() {
        let first = QubitTransitionChannel::leak_recovery(1.0, 1.0);
        let second = QubitTransitionChannel::leak_recovery(1.0, 0.0);
        let step = P2TransitionWeightedStep::new(&P2TransitionStep::independent(first, second));
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let selected = step.select(&mut rng).unwrap();

        assert_eq!(
            selected.sample_known_destinations(
                &mut rng,
                [
                    Some(QubitTransitionState::Leakage),
                    Some(QubitTransitionState::Leakage),
                ],
            ),
            [
                Some(QubitTransitionState::Zero),
                Some(QubitTransitionState::One),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "deviates from 1.0")]
    fn transition_channel_rejects_non_normalized_rows() {
        let transitions = BTreeMap::from([(
            "L".to_string(),
            BTreeMap::from([("0".to_string(), 0.8), ("1".to_string(), 0.1)]),
        )]);
        let _ = QubitTransitionChannel::new(1.0, &transitions);
    }

    #[test]
    #[should_panic(expected = "invalid symbol")]
    fn transition_channel_rejects_unknown_states() {
        let transitions =
            BTreeMap::from([("C".to_string(), BTreeMap::from([("0".to_string(), 1.0)]))]);
        let _ = QubitTransitionChannel::new(1.0, &transitions);
    }

    #[test]
    #[should_panic(expected = "transition-channel probability")]
    fn transition_channel_rejects_invalid_application_probability() {
        let transitions =
            BTreeMap::from([("L".to_string(), BTreeMap::from([("0".to_string(), 1.0)]))]);
        let _ = QubitTransitionChannel::new(1.1, &transitions);
    }
}
