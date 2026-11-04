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
//! Provides:
//! - [`RngManageable`]: A trait for components that can have their RNG replaced or reseeded
//! - [`derive_seed`]: A function for deriving related seeds from a base seed
//! - [`time_seed`]: A function for generating a seed from system time

use crate::{PecosRng, Rng, SeedableRng};
use std::time::{SystemTime, UNIX_EPOCH};

/// Derive a new seed from a base seed and a purpose string.
///
/// Standardized way to derive related seeds from a base seed,
/// useful for creating deterministic but uncorrelated random sequences
/// across different components.
///
/// # Arguments
/// * `base_seed` - The original seed value
/// * `purpose` - A string describing the purpose of the derived seed
///
/// # Returns
/// A new seed value derived from the base seed and purpose
///
/// The purpose is hashed with a fixed SipHash-1-3, so derived seeds do not
/// depend on the Rust release.
///
/// # Example
///
/// ```
/// use pecos_random::rng_manageable::derive_seed;
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
    // Combine the base seed with the purpose hash
    let combined_seed = base_seed.wrapping_add(purpose_hash(purpose));

    // Use the combined seed to initialize an RNG and get a random number
    let mut rng = PecosRng::seed_from_u64(combined_seed);
    rng.next_u64()
}

/// Hash a purpose string for [`derive_seed`].
///
/// SipHash-1-3 with zero keys over the string's bytes followed by `0xFF`
/// (Aumasson and Bernstein, "`SipHash`: a fast short-input PRF", INDOCRYPT 2012).
/// This is what `std::hash::DefaultHasher::new()` computed for a `str` when it
/// replaced that hasher here, so existing derived seeds keep their values. Unlike
/// `DefaultHasher`, whose algorithm the standard library leaves unspecified across
/// releases, it never changes.
fn purpose_hash(purpose: &str) -> u64 {
    fn sip_round(v: &mut [u64; 4]) {
        v[0] = v[0].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(13) ^ v[0];
        v[0] = v[0].rotate_left(32);
        v[2] = v[2].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(16) ^ v[2];
        v[0] = v[0].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(21) ^ v[0];
        v[2] = v[2].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(17) ^ v[2];
        v[2] = v[2].rotate_left(32);
    }

    // Initial state for the all-zero key: the ASCII of "somepseudorandomlygeneratedbytes".
    let mut v: [u64; 4] = [
        0x736f_6d65_7073_6575,
        0x646f_7261_6e64_6f6d,
        0x6c79_6765_6e65_7261,
        0x7465_6462_7974_6573,
    ];
    let mut compress = |m: u64| {
        v[3] ^= m;
        sip_round(&mut v);
        v[0] ^= m;
    };

    let bytes = purpose.as_bytes();
    let (blocks, remainder) = bytes.as_chunks::<8>();
    for block in blocks {
        compress(u64::from_le_bytes(*block));
    }

    // The leftover bytes (at most 7) as a little-endian word, built with shifts.
    let mut tail = 0u64;
    for (i, &byte) in remainder.iter().enumerate() {
        tail |= u64::from(byte) << (8 * i);
    }

    // The 0xFF terminator follows the leftover bytes, and the final block carries
    // the total length (string plus terminator) mod 256 in its top byte. Seven
    // leftover bytes plus the terminator fill a whole block, leaving the final
    // block with only the length.
    let total_len = (bytes.len() as u64).wrapping_add(1);
    if remainder.len() == 7 {
        compress(tail | (0xFF << 56));
        compress(total_len << 56);
    } else {
        compress(tail | (0xFF << (8 * remainder.len())) | (total_len << 56));
    }

    v[2] ^= 0xFF;
    for _ in 0..3 {
        sip_round(&mut v);
    }
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

/// Generate a seed from the current system time.
///
/// Standardized way to generate a seed when no explicit
/// seed is provided. Uses nanosecond precision for better uniqueness.
///
/// # Returns
/// A seed value derived from the current system time, or a fallback value (12345)
/// if the system time is unavailable.
///
/// # Example
///
/// ```
/// use pecos_random::rng_manageable::time_seed;
/// use pecos_random::{PecosRng, SeedableRng};
///
/// // Use time_seed when no explicit seed is provided
/// let seed = time_seed();
/// let rng = PecosRng::seed_from_u64(seed);
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)] // timestamp truncation is fine for seeding
pub fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(12345, |d| d.as_nanos() as u64)
}

/// Resolve an optional seed, using system time if none provided.
///
/// This is a convenience function that combines `time_seed()` with `Option` handling.
///
/// # Arguments
/// * `seed` - An optional seed value
///
/// # Returns
/// The provided seed if `Some`, otherwise a seed from system time.
///
/// # Example
///
/// ```
/// use pecos_random::rng_manageable::resolve_seed;
///
/// let explicit = resolve_seed(Some(42));
/// assert_eq!(explicit, 42);
///
/// let auto = resolve_seed(None);
/// // auto is derived from system time
/// ```
#[must_use]
pub fn resolve_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(time_seed)
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
/// use pecos_random::{PecosRng, SeedableRng};
/// use pecos_random::rng_manageable::RngManageable;
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
