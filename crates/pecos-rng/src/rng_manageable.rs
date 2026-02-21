// Copyright 2025 The PECOS Developers
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

//! Traits and utilities for managing RNG state.
//!
//! This module provides:
//! - [`RngManageable`]: A trait for components that can have their RNG replaced or reseeded
//! - [`derive_seed`]: A function for deriving related seeds from a base seed

use crate::{PecosRng, Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Derive a new seed from a base seed and a purpose string.
///
/// This function provides a standardized way to derive related seeds from a base seed,
/// which is useful for creating deterministic but uncorrelated random sequences
/// across different components.
///
/// # Arguments
/// * `base_seed` - The original seed value
/// * `purpose` - A string describing the purpose of the derived seed
///
/// # Returns
/// A new seed value derived from the base seed and purpose
///
/// # Example
///
/// ```
/// use pecos_rng::rng_manageable::derive_seed;
///
/// let base = 42;
/// let noise_seed = derive_seed(base, "noise");
/// let measurement_seed = derive_seed(base, "measurement");
///
/// // Different purposes give different seeds
/// assert_ne!(noise_seed, measurement_seed);
/// ```
#[must_use]
pub fn derive_seed(base_seed: u64, purpose: &str) -> u64 {
    // Create a purpose-specific seed by hashing the purpose string
    let mut purpose_hasher = DefaultHasher::new();
    purpose.hash(&mut purpose_hasher);
    let purpose_hash = purpose_hasher.finish();

    // Combine the base seed with the purpose hash
    let combined_seed = base_seed.wrapping_add(purpose_hash);

    // Use the combined seed to initialize an RNG and get a random number
    let mut rng = PecosRng::seed_from_u64(combined_seed);
    rng.next_u64()
}

/// Trait for components that can have their random number generator replaced or reseeded.
///
/// This trait defines methods for managing the random number generator (RNG) used by
/// a component. It allows for replacing the RNG with a new one or reseeding it,
/// which is useful for:
///
/// - Ensuring deterministic behavior in tests
/// - Reproducing specific random sequences
/// - Coordinating randomness across different components
///
/// # Usage
///
/// ```
/// use pecos_rng::{PecosRng, SeedableRng};
/// use pecos_rng::rng_manageable::RngManageable;
///
/// struct MySimulator {
///     rng: PecosRng,
/// }
///
/// impl RngManageable for MySimulator {
///     type Rng = PecosRng;
///
///     fn set_rng(&mut self, rng: Self::Rng) {
///         self.rng = rng;
///     }
///
///     fn rng(&self) -> &Self::Rng {
///         &self.rng
///     }
///
///     fn rng_mut(&mut self) -> &mut Self::Rng {
///         &mut self.rng
///     }
/// }
///
/// let mut sim = MySimulator { rng: PecosRng::seed_from_u64(42) };
/// sim.set_seed(123); // Reseed with a new value
/// ```
pub trait RngManageable {
    /// The type of random number generator used by this component.
    type Rng: Rng + SeedableRng;

    /// Replace the random number generator with a new one.
    ///
    /// This method allows replacing the RNG without recreating the entire component,
    /// preserving its current state.
    fn set_rng(&mut self, rng: Self::Rng);

    /// Replace the random number generator with a new one created from a seed.
    ///
    /// This is the preferred method for most users who need deterministic behavior.
    ///
    /// # Arguments
    /// * `seed` - Seed value for the new random number generator
    fn set_seed(&mut self, seed: u64) {
        self.set_rng(Self::Rng::seed_from_u64(seed));
    }

    /// Get a read-only reference to the internal random number generator.
    fn rng(&self) -> &Self::Rng;

    /// Get a mutable reference to the internal random number generator.
    fn rng_mut(&mut self) -> &mut Self::Rng;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestComponent {
        rng: PecosRng,
    }

    impl RngManageable for TestComponent {
        type Rng = PecosRng;

        fn set_rng(&mut self, rng: Self::Rng) {
            self.rng = rng;
        }

        fn rng(&self) -> &Self::Rng {
            &self.rng
        }

        fn rng_mut(&mut self) -> &mut Self::Rng {
            &mut self.rng
        }
    }

    #[test]
    fn test_derive_seed_deterministic() {
        let seed1 = derive_seed(42, "test");
        let seed2 = derive_seed(42, "test");
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn test_derive_seed_different_purposes() {
        let seed1 = derive_seed(42, "noise");
        let seed2 = derive_seed(42, "measurement");
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn test_derive_seed_different_bases() {
        let seed1 = derive_seed(42, "test");
        let seed2 = derive_seed(43, "test");
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn test_rng_manageable_set_seed() {
        let mut comp = TestComponent {
            rng: PecosRng::seed_from_u64(0),
        };

        comp.set_seed(42);
        let val1 = comp.rng_mut().next_u64();

        comp.set_seed(42);
        let val2 = comp.rng_mut().next_u64();

        assert_eq!(val1, val2, "Same seed should produce same values");
    }
}
