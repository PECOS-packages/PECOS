// Copyright 2026 The PECOS Developers
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

//! Thin wrapper over [`rapidrand::RapidRng`] preserving the seeded output
//! streams of the rapidhash 4.4 `rng::RapidRng` this crate previously used.
//!
//! rapidhash 4.5 deprecated its `rng` module in favor of the `rapidrand`
//! crate. The mixing function and secrets are identical, but the seeding
//! paths differ: the old `RapidRng::new(seed)` stored the seed RAW, which in
//! rapidrand corresponds to `SeedableRng::from_seed(seed.to_le_bytes())`.
//! rapidrand's `seed_from_u64` pre-mixes the seed once and produces a
//! DIFFERENT stream -- do not switch to it without a deliberate,
//! release-noted reproducibility break. Stream identity is locked by
//! `tests/seed_stream_stability.rs`.

use rand_core::{SeedableRng, TryRng};

/// Single-stream rapidhash-mixing RNG with the legacy seeding semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RapidRng(rapidrand::RapidRng);

impl RapidRng {
    /// Create a generator whose stream matches the old `RapidRng::new(seed)`.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(rapidrand::RapidRng::from_seed(seed.to_le_bytes()))
    }

    /// Next value in the stream.
    #[inline]
    pub fn next(&mut self) -> u64 {
        match self.0.try_next_u64() {
            Ok(v) => v,
        }
    }
}
