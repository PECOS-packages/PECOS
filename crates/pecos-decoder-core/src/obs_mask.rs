// Copyright 2026 The PECOS Developers
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

//! Observable flip mask supporting arbitrarily many logical observables.
//!
//! Decoders report "which logical observables flipped" as a bitmask (bit `i` =
//! observable `i`). Historically this was a `u64`, capping observables at 64.
//! [`ObsMask`] lifts that cap while keeping the common case fast:
//!
//! - **<= 64 observables -> one inline word, zero heap allocation** (a
//!   `SmallVec<[u64; 1]>` keeps the single word on the stack). The per-shot
//!   decode hot path is unchanged from the old `u64`.
//! - **> 64 observables -> spills to N words on the heap**, no truncation.
//!
//! Word `w` holds observable bits `64*w ..= 64*w + 63` (little-endian). Trailing
//! zero words are permitted; equality compares the represented bit set, not the
//! storage, so `ObsMask::from_u64(0)`, `ObsMask::new()`, and a two-word `[5, 0]`
//! vs one-word `[5]` all compare as expected.

use smallvec::{SmallVec, smallvec};
use std::ops::BitXorAssign;

const WORD_BITS: usize = u64::BITS as usize;

/// A logical-observable flip mask of unbounded width.
///
/// See the module docs. Cheap (`Copy`-like, one inline word) for the common
/// `<= 64` case; spills to the heap only beyond 64 observables.
#[derive(Clone, Debug, Default)]
pub struct ObsMask {
    words: SmallVec<[u64; 1]>,
}

impl ObsMask {
    /// An empty (all-zero) mask. No heap allocation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            words: SmallVec::new(),
        }
    }

    /// A mask from a single 64-bit word (observables 0..=63). No heap allocation.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        Self {
            words: smallvec![value],
        }
    }

    /// Sets observable bit `bit` to 1, growing the storage if needed.
    pub fn set(&mut self, bit: usize) {
        let word = bit / WORD_BITS;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << (bit % WORD_BITS);
    }

    /// Returns whether observable bit `bit` is set.
    #[must_use]
    pub fn get(&self, bit: usize) -> bool {
        let word = bit / WORD_BITS;
        self.words
            .get(word)
            .is_some_and(|w| (w >> (bit % WORD_BITS)) & 1 != 0)
    }

    /// Returns whether no observable bit is set.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Number of set observable bits.
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// The mask as a single `u64` if it fits in 64 bits, else `None`.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        if self.words.iter().skip(1).all(|&w| w == 0) {
            Some(self.words.first().copied().unwrap_or(0))
        } else {
            None
        }
    }

    /// The backing little-endian words (lowest observables first). Trailing zero
    /// words may be present. Used to bridge to/from external representations
    /// (e.g. a Python arbitrary-precision integer).
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Builds a mask from little-endian words (lowest observables first).
    #[must_use]
    pub fn from_words(words: &[u64]) -> Self {
        Self {
            words: SmallVec::from_slice(words),
        }
    }

    /// Iterates the indices of the set observable bits, ascending.
    pub fn iter_set_bits(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            (0..WORD_BITS)
                .filter(move |b| (word >> b) & 1 != 0)
                .map(move |b| w * WORD_BITS + b)
        })
    }
}

impl From<u64> for ObsMask {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl BitXorAssign<&ObsMask> for ObsMask {
    fn bitxor_assign(&mut self, rhs: &ObsMask) {
        if rhs.words.len() > self.words.len() {
            self.words.resize(rhs.words.len(), 0);
        }
        for (w, &r) in self.words.iter_mut().zip(rhs.words.iter()) {
            *w ^= r;
        }
    }
}

impl PartialEq for ObsMask {
    fn eq(&self, other: &Self) -> bool {
        // Value equality: compare every word, treating missing words as zero, so
        // representations that differ only by trailing zero words are equal.
        let n = self.words.len().max(other.words.len());
        (0..n).all(|i| {
            self.words.get(i).copied().unwrap_or(0) == other.words.get(i).copied().unwrap_or(0)
        })
    }
}

impl Eq for ObsMask {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_zero() {
        let m = ObsMask::new();
        assert!(m.is_zero());
        assert_eq!(m.count_ones(), 0);
        assert_eq!(m.to_u64(), Some(0));
    }

    #[test]
    fn from_u64_roundtrips() {
        let m = ObsMask::from_u64(0b1011);
        assert!(m.get(0));
        assert!(m.get(1));
        assert!(!m.get(2));
        assert!(m.get(3));
        assert_eq!(m.to_u64(), Some(0b1011));
        assert_eq!(m.count_ones(), 3);
    }

    #[test]
    fn set_below_64_stays_inline() {
        let mut m = ObsMask::new();
        m.set(0);
        m.set(63);
        assert!(m.get(0));
        assert!(m.get(63));
        assert_eq!(m.words().len(), 1, "<=64 observables must use one word");
        assert_eq!(m.to_u64(), Some((1u64 << 63) | 1));
    }

    #[test]
    fn set_at_and_above_64_spills_wide() {
        let mut m = ObsMask::new();
        m.set(5);
        m.set(64);
        m.set(130);
        assert!(m.get(5));
        assert!(m.get(64));
        assert!(m.get(130));
        assert!(!m.get(63));
        assert_eq!(m.words().len(), 3, "bit 130 needs 3 words");
        assert_eq!(m.to_u64(), None, "does not fit in 64 bits");
        assert_eq!(m.count_ones(), 3);
        assert_eq!(m.iter_set_bits().collect::<Vec<_>>(), vec![5, 64, 130]);
    }

    #[test]
    fn xor_same_and_mixed_width() {
        let mut a = ObsMask::from_u64(0b1100);
        a ^= &ObsMask::from_u64(0b1010);
        assert_eq!(a.to_u64(), Some(0b0110));

        // Wide ^ narrow grows the narrow side.
        let mut wide = ObsMask::new();
        wide.set(70);
        let mut narrow = ObsMask::from_u64(0b1);
        narrow ^= &wide;
        assert!(narrow.get(0));
        assert!(narrow.get(70));
        assert_eq!(narrow.count_ones(), 2);

        // x ^ x == 0 at any width.
        let mut w = ObsMask::new();
        w.set(200);
        let snapshot = w.clone();
        w ^= &snapshot;
        assert!(w.is_zero());
    }

    #[test]
    fn equality_ignores_trailing_zero_words() {
        assert_eq!(ObsMask::new(), ObsMask::from_u64(0));
        assert_eq!(ObsMask::from_u64(5), ObsMask::from_words(&[5, 0, 0]));
        assert_ne!(ObsMask::from_u64(5), ObsMask::from_words(&[5, 1]));
        let mut wide_zero = ObsMask::new();
        wide_zero.set(100);
        wide_zero.set(100); // toggle on, still set
        assert_ne!(wide_zero, ObsMask::new());
    }

    #[test]
    fn from_words_and_words_roundtrip() {
        let m = ObsMask::from_words(&[0xff, 0x1]);
        assert_eq!(m.words(), &[0xff, 0x1]);
        assert!(m.get(0));
        assert!(m.get(64));
        assert_eq!(m.count_ones(), 9);
    }
}
