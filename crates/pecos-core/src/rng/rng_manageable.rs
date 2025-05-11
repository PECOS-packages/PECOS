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

use crate::errors::PecosError;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Derive a new seed from a base seed and a purpose string
///
/// This function provides a standardized way to derive related seeds from a base seed,
/// which is useful for creating deterministic but uncorrelated random sequences
/// across different components.
///
/// The implementation uses a temporary RNG seeded with a combination of the base seed
/// and a hash of the purpose string, ensuring that different purposes produce different
/// derived seeds even with the same base seed.
///
/// # Arguments
/// * `base_seed` - The original seed value
/// * `purpose` - A string describing the purpose of the derived seed
///
/// # Returns
/// A new seed value derived from the base seed and purpose
#[must_use]
pub fn derive_seed(base_seed: u64, purpose: &str) -> u64 {
    // Create a purpose-specific seed by hashing the purpose string
    let mut purpose_hasher = DefaultHasher::new();
    purpose.hash(&mut purpose_hasher);
    let purpose_hash = purpose_hasher.finish();

    // Combine the base seed with the purpose hash
    let combined_seed = base_seed.wrapping_add(purpose_hash);

    // Use the combined seed to initialize an RNG and get a random number
    let mut rng = ChaCha8Rng::seed_from_u64(combined_seed);
    rng.next_u64()
}

/// Trait for components that can have their random number generator replaced or reseeded
///
/// This trait defines methods for managing the random number generator (RNG) used by
/// a component. It allows for replacing the RNG with a new one or reseeding it,
/// which is useful for:
///
/// - Ensuring deterministic behavior in tests
/// - Reproducing specific random sequences
/// - Coordinating randomness across different components
///
/// # Usage Guidelines
/// - For most users, `set_seed()` is the preferred method as it provides a simpler interface
/// - `set_rng()` is primarily intended for implementers and advanced use cases
///
/// # Implementation Notes
/// - Implementers only need to implement `set_rng()` to get a default implementation of `set_seed()`
/// - Implementers should ensure that replacing the RNG does not affect the current state
///   of the component beyond the randomness source
pub trait RngManageable {
    /// The type of random number generator used by this component
    type Rng: RngCore + SeedableRng;

    /// Replace the random number generator with a new one
    ///
    /// This method is primarily intended for implementers and advanced use cases.
    /// Most users should prefer `set_seed()` for a simpler interface.
    ///
    /// This method allows replacing the RNG without recreating the entire component,
    /// preserving its current state.
    ///
    /// # Arguments
    /// * `rng` - A new random number generator
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// Returns an error if setting the RNG fails
    fn set_rng(&mut self, rng: Self::Rng) -> Result<(), PecosError>;

    /// Replace the random number generator with a new one created from a seed
    ///
    /// This is the preferred method for most users who need deterministic behavior.
    /// It creates a new RNG from the provided seed and sets it using `set_rng()`.
    ///
    /// This method allows replacing the RNG with a seeded one without recreating
    /// the entire component, preserving its current state.
    ///
    /// # Arguments
    /// * `seed` - Seed value for the new random number generator
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// Returns an error if setting the RNG fails
    ///
    /// # Implementation Note
    /// The default implementation creates a new RNG using `SeedableRng::seed_from_u64`
    /// and sets it using `set_rng()`. Implementers typically only need to implement
    /// `set_rng()` unless they need custom seed handling.
    fn set_seed(&mut self, seed: u64) -> Result<(), PecosError>
    where
        Self::Rng: SeedableRng,
    {
        self.set_rng(Self::Rng::seed_from_u64(seed))
    }

    /// Get a read-only reference to the internal random number generator
    ///
    /// This method provides access to the RNG for inspection or to retrieve
    /// information from it (such as recorded values from a `RecordingRng`).
    ///
    /// # Returns
    /// A reference to the internal RNG
    fn rng(&self) -> &Self::Rng;

    /// Get a mutable reference to the internal random number generator
    ///
    /// This method provides mutable access to the RNG for direct manipulation.
    /// This is an advanced feature that should be used with care.
    ///
    /// # Returns
    /// A mutable reference to the internal RNG
    fn rng_mut(&mut self) -> &mut Self::Rng;
}
