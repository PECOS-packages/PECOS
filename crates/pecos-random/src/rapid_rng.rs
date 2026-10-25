// Copyright 2026 The PECOS Developers
// Portions copyright 2026 Liam Gray <gmail@liamg.me> (rapidrand, used under Apache-2.0)
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

//! Single-stream RNG built on the rapidhash folded-multiply mix.
//!
//! Each step adds an odd constant to a 64-bit state, then XORs the high and
//! low halves of the 128-bit product of the state and the state XOR a second
//! constant.
//!
//! Ported from the `rapidrand` crate (<https://github.com/hoxxep/rapidrand>,
//! dual-licensed MIT OR Apache-2.0, used here under Apache-2.0), which
//! replaced the rapidhash 4.4 `rng::RapidRng` this crate depended on before.
//! Changes: kept only the raw-seeded single-stream generator, without the
//! `rand_core` trait implementations.
//!
//! Lineage: the step is wyrand, from wyhash by Wang Yi
//! (<https://github.com/wangyi-fudan/wyhash>, Unlicense). The two constants
//! are the first two default secrets of rapidhash by Nicolas De Carli
//! (<https://github.com/Nicoshev/rapidhash>, MIT; BSD 2-Clause before 2025).
//!
//! The seed is stored RAW as the initial state, matching the old rapidhash
//! `RapidRng::new(seed)`. rapidrand's `seed_from_u64` pre-mixes the seed once
//! and produces a DIFFERENT stream -- do not switch to that seeding without a
//! deliberate, release-noted reproducibility break. Stream identity is locked
//! by `tests/seed_stream_stability.rs`.

/// Rapidhash V1 secret[0]. Odd, so the state cycles through every `u64`.
const RAPID_SECRET_ADD: u64 = 0x2d35_8dcc_aa6c_78a5;

/// Rapidhash V1 secret[1].
const RAPID_SECRET_XOR: u64 = 0x8bb8_4b93_962e_acc9;

/// Single-stream rapidhash-mixing RNG with raw seeding.
///
/// Not a cryptographic generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RapidRng {
    state: u64,
}

impl RapidRng {
    /// Create a generator whose initial state is `seed`, unmixed.
    ///
    /// Because the state only ever has a constant added, related seeds give
    /// strongly correlated streams: for `s` and `s ^ (1 << 63)` some output bit
    /// agrees on every draw, and even for `s` and `s + 1` some output bit
    /// agrees on about three draws in four. [`PecosRng`](crate::PecosRng) and
    /// [`PecosScalarRng`](crate::PecosScalarRng) scramble the seed with
    /// `SplitMix64` before it reaches this constructor; use them unless you need
    /// the raw stream.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next value in the stream.
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // the fold keeps each 64-bit half on purpose
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(RAPID_SECRET_ADD);
        let product =
            u128::from(self.state).wrapping_mul(u128::from(self.state ^ RAPID_SECRET_XOR));
        (product as u64) ^ ((product >> 64) as u64)
    }
}
