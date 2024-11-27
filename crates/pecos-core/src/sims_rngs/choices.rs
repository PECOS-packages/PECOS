// Copyright 2024 The PECOS Developers
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

use rand::distributions::{Distribution, WeightedIndex};
use rand::RngCore;

const EPSILON: f64 = 1e-9;

/// Struct to hold choices and pre-validated `WeightedIndex`
/// The weights need to sum up close to 1.0 and will be re-normalized if they do so.
#[derive(Debug)]
pub struct Choices<T> {
    items: Vec<T>,
    weighted_index: WeightedIndex<f64>,
}

impl<T> Choices<T> {
    /// Validate and normalize weights, then create Choices struct
    /// # Panics
    /// This will panic if the number of weights and number of items are not the same.
    #[inline]
    #[allow(clippy::float_arithmetic)]
    #[must_use]
    pub fn new(items: Vec<T>, weights: &[f64]) -> Self {
        assert_eq!(
            items.len(),
            weights.len(),
            "Number of items needs to equal number of weights"
        );
        assert!(
            weights.iter().all(|&w| w >= 0.0f64),
            "All weights must be positive numbers since they represent probabilities."
        );

        let sum_weights: f64 = weights.iter().sum();
        assert!(
            isclose(sum_weights, 1.0, EPSILON),
            "Weights do not sum to 1 \u{b1} \u{3b5}" // 1 ± ε
        );

        let normalized_weights: Vec<f64> = weights.iter().map(|&w| w / sum_weights).collect();
        let weighted_index = WeightedIndex::new(normalized_weights)
            .expect("Failed to create WeightedIndex due to invalid weights");

        Choices {
            items,
            weighted_index,
        }
    }

    /// Sample a choice based on the weights
    #[inline]
    pub fn sample<R: RngCore>(&self, rng: &mut R) -> &T {
        let index = self.weighted_index.sample(rng);
        &self.items[index]
    }
}

/// Determine if two floats are close to each other.
#[inline]
#[allow(clippy::single_call_fn, clippy::float_arithmetic)]
fn isclose(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() <= epsilon
}
