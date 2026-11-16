// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Adaptive word and element storage for stabilizer indices.

use crate::IndexSet;
use rustc_hash::FxBuildHasher;
use std::collections::HashSet;

/// Always-dense word span: trades reset allocation reuse against small-set memory.
const DENSE_MIN_WORDS: usize = 32;
/// Words per element: trades dense XOR speed against sparse storage efficiency.
const DENSE_WORDS_PER_ELEMENT: usize = 2;

#[derive(Clone)]
enum Storage {
    Dense {
        words: Vec<u64>,
        /// Valid only when `words.len() > DENSE_MIN_WORDS`; ignored and
        /// unspecified for narrow sets, whose cardinality is computed on demand.
        len: usize,
    },
    Sparse(HashSet<u32, FxBuildHasher>),
}

/// A set of indices using dense words for compact spans and Fx-hashed `u32` keys
/// otherwise. Narrow dense cardinality scans at most `DENSE_MIN_WORDS` words;
/// wide dense and sparse cardinality are cached. Narrow emptiness scans at most
/// `DENSE_MIN_WORDS` words with early exit; wide and sparse emptiness are O(1).
/// Narrow storage retains its high-water length, including trailing zero words;
/// wide storage is exactly trimmed.
///
/// Dense removal uses twice the growth density threshold to avoid conversion
/// thrashing. Iteration order is unspecified; debug output is sorted.
/// Index-domain capacity hints do not allocate. A sparse set returns to dense
/// storage only through bulk XOR results, `clear`, or `take_clearing`, never
/// through single-element operations. Memory remains O(nnz) in either mode.
///
/// # Panics
///
/// Index operations panic when an index exceeds `u32::MAX`, in either mode.
pub struct AdaptiveIndexSet {
    storage: Storage,
}

impl Default for AdaptiveIndexSet {
    fn default() -> Self {
        Self {
            storage: Storage::Dense {
                words: Vec::new(),
                len: 0,
            },
        }
    }
}

impl AdaptiveIndexSet {
    /// Create an empty set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        <Self as IndexSet>::new()
    }

    /// Create an empty set; the index-domain capacity hint does not allocate.
    #[inline]
    #[must_use]
    pub fn with_capacity(max_index: usize) -> Self {
        <Self as IndexSet>::with_capacity(max_index)
    }

    /// Insert an index, returning whether it was newly inserted.
    #[inline]
    pub fn insert(&mut self, index: usize) -> bool {
        <Self as IndexSet>::insert(self, index)
    }

    /// Remove an index, returning whether it was present.
    #[inline]
    pub fn remove(&mut self, index: usize) -> bool {
        <Self as IndexSet>::remove(self, index)
    }

    /// Check whether an index is present.
    #[inline]
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        <Self as IndexSet>::contains(self, index)
    }

    /// Iterate over indices in unspecified order.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> AdaptiveIndexSetIter<'_> {
        <Self as IndexSet>::iter(self)
    }

    /// Return the cardinality, scanning at most the narrow word bound.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        <Self as IndexSet>::len(self)
    }

    /// Check emptiness, scanning only in narrow mode.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        <Self as IndexSet>::is_empty(self)
    }

    /// Remove all elements, retaining bounded narrow storage.
    #[inline]
    pub fn clear(&mut self) {
        <Self as IndexSet>::clear(self);
    }

    #[inline]
    fn key(index: usize) -> u32 {
        u32::try_from(index).unwrap_or_else(|_| panic!("index {index} exceeds limit {}", u32::MAX))
    }

    #[inline]
    fn dense_limit(len: usize) -> usize {
        DENSE_MIN_WORDS.max(DENSE_WORDS_PER_ELEMENT.saturating_mul(len))
    }

    #[inline]
    fn dense_len(words: &[u64], len: usize) -> usize {
        if words.len() <= DENSE_MIN_WORDS {
            words.iter().map(|word| word.count_ones() as usize).sum()
        } else {
            len
        }
    }

    #[inline]
    fn trim(words: &mut Vec<u64>) {
        while words.last() == Some(&0) {
            words.pop();
        }
    }

    #[inline]
    fn shrink_narrow(words: &mut Vec<u64>) {
        if words.capacity() > 2 * DENSE_MIN_WORDS {
            words.shrink_to(words.len());
        }
    }

    #[inline]
    fn finish_narrow(words: &mut Vec<u64>) {
        Self::shrink_narrow(words);
    }

    #[inline]
    fn shrink_sparse(elements: &mut HashSet<u32, FxBuildHasher>) {
        let len = elements.len();
        if elements.capacity() > 4 * len.max(8) {
            elements.shrink_to(len + len / 4);
        }
    }

    #[inline]
    fn shrink_dense(words: &mut Vec<u64>, len: usize) {
        if words.capacity()
            > (2 * DENSE_MIN_WORDS).max((4 * DENSE_WORDS_PER_ELEMENT).saturating_mul(len))
        {
            words.shrink_to(words.len());
        }
    }

    fn demote(&mut self) {
        let elements = self.iter().map(Self::key).collect();
        self.storage = Storage::Sparse(elements);
        if let Storage::Sparse(elements) = &mut self.storage {
            Self::shrink_sparse(elements);
        }
    }

    /// Finish a wide operation, including a possible crossing back to narrow.
    #[inline]
    fn finish_dense(&mut self) {
        if let Storage::Dense { words, len } = &mut self.storage {
            Self::trim(words);
            if words.len() <= DENSE_MIN_WORDS {
                Self::shrink_narrow(words);
            } else if words.len()
                > DENSE_MIN_WORDS.max((2 * DENSE_WORDS_PER_ELEMENT).saturating_mul(*len))
            {
                self.demote();
            } else {
                Self::shrink_dense(words, *len);
            }
        }
    }

    /// Scan sparse storage only when the dense operand pays for that scan.
    #[inline]
    fn promote_for_xor(&mut self, span: usize, other_len: usize) {
        if let Storage::Sparse(elements) = &self.storage {
            if elements.len() > other_len {
                return;
            }
            let own_span = elements
                .iter()
                .map(|&i| i as usize / 64 + 1)
                .max()
                .unwrap_or(0);
            let span = span.max(own_span);
            if span <= Self::dense_limit(other_len - elements.len()) {
                let mut words = vec![0; span];
                for &index in elements {
                    words[index as usize / 64] |= 1 << (index % 64);
                }
                self.storage = Storage::Dense {
                    words,
                    len: elements.len(),
                };
            }
        }
    }

    /// XOR a dense operand (or a fused expression of two dense operands).
    /// `span` covers the operand (possibly with narrow trailing zeros), and
    /// `other_len` is its cardinality.
    fn xor_words(&mut self, span: usize, other_len: usize, word: impl Fn(usize) -> u64) {
        self.promote_for_xor(span, other_len);
        if let Storage::Dense { words, len } = &mut self.storage {
            if words.len() <= DENSE_MIN_WORDS && span <= DENSE_MIN_WORDS {
                if span > words.len() {
                    words.resize(span, 0);
                }
                for (i, value) in words.iter_mut().take(span).enumerate() {
                    *value ^= word(i);
                }
                Self::finish_narrow(words);
                return;
            }
            // Initialize once before a narrow-to-wide crossing; wide caches
            // remain valid until trimming crosses back below the threshold.
            *len = Self::dense_len(words, *len);
            // The sum bounds growth; cancellations are handled by finish_dense.
            if span > words.len() && span > Self::dense_limit(len.saturating_add(other_len)) {
                self.demote();
            }
        }
        if let Storage::Dense { words, len } = &mut self.storage {
            if span > words.len() {
                words.resize(span, 0);
            }
            for (i, value) in words.iter_mut().take(span).enumerate() {
                let before = value.count_ones() as usize;
                *value ^= word(i);
                // Keep the post-XOR popcount: pivot selection needs cached length.
                *len = *len - before + value.count_ones() as usize;
            }
            self.finish_dense();
        } else {
            for i in 0..span {
                let mut bits = word(i);
                while bits != 0 {
                    self.toggle(i * 64 + bits.trailing_zeros() as usize);
                    bits &= bits - 1;
                }
            }
        }
    }

    fn xor_fused_words(&mut self, span: usize, word: impl Fn(usize) -> u64) {
        let mut exact_span = 0;
        let mut len = 0;
        for i in 0..span {
            let value = word(i);
            if value != 0 {
                exact_span = i + 1;
                len += value.count_ones() as usize;
            }
        }
        self.xor_words(exact_span, len, word);
    }

    /// Toggle when the set may grow, convert, or is wide or sparse.
    #[inline(never)]
    fn toggle_slow(&mut self, index: usize) {
        match &mut self.storage {
            Storage::Dense { words, len } => {
                Self::key(index);
                let needed = index / 64 + 1;
                if words.len() <= DENSE_MIN_WORDS && needed <= DENSE_MIN_WORDS {
                    if needed > words.len() {
                        words.resize(needed, 0);
                    }
                    words[index / 64] ^= 1 << (index % 64);
                    Self::shrink_narrow(words);
                    return;
                }
                *len = Self::dense_len(words, *len);
                if needed > words.len() {
                    if needed > Self::dense_limit(*len + 1) {
                        self.demote();
                        self.toggle(index);
                        return;
                    }
                    words.resize(needed, 0);
                }
                let mask = 1 << (index % 64);
                words[index / 64] ^= mask;
                if words[index / 64] & mask != 0 {
                    *len += 1;
                    Self::shrink_dense(words, *len);
                } else {
                    *len -= 1;
                    self.finish_dense();
                }
            }
            Storage::Sparse(elements) => {
                let key = Self::key(index);
                if elements.remove(&key) {
                    Self::shrink_sparse(elements);
                } else {
                    elements.insert(key);
                }
            }
        }
    }

    /// Remove from a wide or sparse set.
    #[inline(never)]
    fn remove_slow(&mut self, index: usize) -> bool {
        match &mut self.storage {
            Storage::Dense { words, len } => {
                Self::key(index);
                let Some(word) = words.get_mut(index / 64) else {
                    return false;
                };
                let mask = 1 << (index % 64);
                if *word & mask == 0 {
                    return false;
                }
                *word &= !mask;
                if words.len() <= DENSE_MIN_WORDS {
                    Self::finish_narrow(words);
                    return true;
                }
                *len -= 1;
                self.finish_dense();
                true
            }
            Storage::Sparse(elements) => {
                let removed = elements.remove(&Self::key(index));
                if removed {
                    Self::shrink_sparse(elements);
                }
                removed
            }
        }
    }

    /// XOR when either operand is wide or sparse.
    #[inline(never)]
    fn xor_assign_slow(&mut self, other: &Self) {
        match &other.storage {
            Storage::Dense { words, len } => {
                self.xor_words(words.len(), Self::dense_len(words, *len), |i| words[i]);
            }
            Storage::Sparse(elements) => {
                for &index in elements {
                    self.toggle(index as usize);
                }
            }
        }
    }

    #[cfg(test)]
    fn assert_invariants(&self) {
        match &self.storage {
            Storage::Dense { words, len } => {
                if words.len() <= DENSE_MIN_WORDS {
                    assert!(words.capacity() <= 2 * DENSE_MIN_WORDS);
                } else {
                    assert_ne!(words.last(), Some(&0));
                    assert!(words.len() <= DENSE_MIN_WORDS.max(2 * DENSE_WORDS_PER_ELEMENT * len));
                    assert_eq!(
                        *len,
                        words.iter().map(|w| w.count_ones() as usize).sum::<usize>()
                    );
                    assert!(
                        words.capacity()
                            <= (2 * DENSE_MIN_WORDS).max(4 * DENSE_WORDS_PER_ELEMENT * len)
                    );
                }
            }
            Storage::Sparse(elements) => assert!(elements.capacity() <= 4 * elements.len().max(8)),
        }
    }
}

impl FromIterator<usize> for AdaptiveIndexSet {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut set = Self::new();
        for index in iter {
            set.insert(index);
        }
        set
    }
}

impl<'a> IntoIterator for &'a AdaptiveIndexSet {
    type Item = usize;
    type IntoIter = AdaptiveIndexSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Clone for AdaptiveIndexSet {
    fn clone(&self) -> Self {
        let mut copy = Self {
            storage: self.storage.clone(),
        };
        match &mut copy.storage {
            Storage::Dense { words, .. } if words.len() <= DENSE_MIN_WORDS => {
                Self::shrink_narrow(words);
            }
            Storage::Dense { words, len } => Self::shrink_dense(words, *len),
            Storage::Sparse(elements) => Self::shrink_sparse(elements),
        }
        copy
    }
}

impl IndexSet for AdaptiveIndexSet {
    type Iter<'a> = AdaptiveIndexSetIter<'a>;

    #[inline]
    fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn insert(&mut self, index: usize) -> bool {
        match &mut self.storage {
            Storage::Dense { words, len } => {
                Self::key(index);
                let needed = index / 64 + 1;
                if words.len() <= DENSE_MIN_WORDS && needed <= DENSE_MIN_WORDS {
                    if needed > words.len() {
                        words.resize(needed, 0);
                    }
                    let mask = 1 << (index % 64);
                    let word = &mut words[index / 64];
                    let inserted = *word & mask == 0;
                    *word |= mask;
                    Self::shrink_narrow(words);
                    return inserted;
                }
                *len = Self::dense_len(words, *len);
                if needed > words.len() {
                    if needed > Self::dense_limit(*len + 1) {
                        self.demote();
                        return self.insert(index);
                    }
                    words.resize(needed, 0);
                }
                let mask = 1 << (index % 64);
                let word = &mut words[index / 64];
                let inserted = *word & mask == 0;
                *word |= mask;
                *len += usize::from(inserted);
                Self::shrink_dense(words, *len);
                inserted
            }
            Storage::Sparse(elements) => elements.insert(Self::key(index)),
        }
    }

    #[inline]
    fn remove(&mut self, index: usize) -> bool {
        // Hot path: a narrow set; `word < words.len() <= DENSE_MIN_WORDS` bounds the
        // index, and an index beyond the span is absent.
        if let Storage::Dense { words, .. } = &mut self.storage
            && words.len() <= DENSE_MIN_WORDS
            && index < 64 * DENSE_MIN_WORDS
        {
            let word = index / 64;
            let mask = 1 << (index % 64);
            let Some(value) = words.get_mut(word) else {
                return false;
            };
            if *value & mask == 0 {
                return false;
            }
            *value &= !mask;
            return true;
        }
        self.remove_slow(index)
    }

    #[inline]
    fn contains(&self, index: usize) -> bool {
        match &self.storage {
            Storage::Dense { words, .. } => {
                Self::key(index);
                words
                    .get(index / 64)
                    .is_some_and(|word| word & (1 << (index % 64)) != 0)
            }
            Storage::Sparse(elements) => elements.contains(&Self::key(index)),
        }
    }

    #[inline]
    fn toggle(&mut self, index: usize) {
        // Hot path (CX toggles one row bit per column element): a narrow set whose
        // span already covers the index cannot grow, convert or need a capacity
        // check, and `word < words.len() <= DENSE_MIN_WORDS` bounds the index.
        if let Storage::Dense { words, .. } = &mut self.storage {
            let word = index / 64;
            if words.len() <= DENSE_MIN_WORDS && word < words.len() {
                words[word] ^= 1 << (index % 64);
                return;
            }
        }
        self.toggle_slow(index);
    }

    #[inline]
    fn xor_assign(&mut self, other: &Self) {
        // Hot path (column XORs in CX): narrow ^ narrow is BitSet's word loop.
        if let (Storage::Dense { words: a, .. }, Storage::Dense { words: b, .. }) =
            (&mut self.storage, &other.storage)
            && a.len() <= DENSE_MIN_WORDS
            && b.len() <= DENSE_MIN_WORDS
        {
            if b.len() > a.len() {
                a.resize(b.len(), 0);
            }
            for (a, b) in a.iter_mut().zip(b) {
                *a ^= b;
            }
            Self::finish_narrow(a);
            return;
        }
        self.xor_assign_slow(other);
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        match &self.storage {
            Storage::Dense { words, .. } => AdaptiveIndexSetIter::Dense {
                words: words.iter(),
                bits: 0,
                base: 0,
            },
            Storage::Sparse(elements) => AdaptiveIndexSetIter::Sparse(elements.iter()),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match &self.storage {
            Storage::Dense { words, len } => Self::dense_len(words, *len),
            Storage::Sparse(elements) => elements.len(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        match &self.storage {
            Storage::Dense { words, len } => {
                if words.len() <= DENSE_MIN_WORDS {
                    words.iter().all(|&word| word == 0)
                } else {
                    *len == 0
                }
            }
            Storage::Sparse(elements) => elements.is_empty(),
        }
    }

    #[inline]
    fn clear(&mut self) {
        if let Storage::Dense { words, len } = &mut self.storage
            && words.len() <= DENSE_MIN_WORDS
            && words.capacity() <= 2 * DENSE_MIN_WORDS
        {
            words.fill(0);
            *len = 0;
            return;
        }
        *self = Self::default();
    }

    fn intersection_count(&self, other: &Self) -> usize {
        if let (Storage::Dense { words: a, .. }, Storage::Dense { words: b, .. }) =
            (&self.storage, &other.storage)
        {
            return a
                .iter()
                .zip(b)
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum();
        }
        let (small, large) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        small.iter().filter(|&index| large.contains(index)).count()
    }

    #[inline]
    fn xor_intersection_into(&self, other: &Self, target: &mut Self) {
        if let (Storage::Dense { words: a, .. }, Storage::Dense { words: b, .. }) =
            (&self.storage, &other.storage)
        {
            if let Storage::Dense { words, .. } = &mut target.storage
                && words.len() <= DENSE_MIN_WORDS
                && a.len() <= DENSE_MIN_WORDS
                && b.len() <= DENSE_MIN_WORDS
            {
                let span = a.len().min(b.len());
                if span > words.len() {
                    words.resize(span, 0);
                }
                for ((target, a), b) in words.iter_mut().zip(a).zip(b) {
                    *target ^= a & b;
                }
                Self::finish_narrow(words);
                return;
            }
            target.xor_fused_words(a.len().min(b.len()), |i| a[i] & b[i]);
            return;
        }
        let (small, large) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        for index in small {
            if large.contains(index) {
                target.toggle(index);
            }
        }
    }

    #[inline]
    fn xor_symmetric_difference_into(&self, other: &Self, target: &mut Self) {
        if let (Storage::Dense { words: a, .. }, Storage::Dense { words: b, .. }) =
            (&self.storage, &other.storage)
        {
            if let Storage::Dense { words, .. } = &mut target.storage
                && words.len() <= DENSE_MIN_WORDS
                && a.len() <= DENSE_MIN_WORDS
                && b.len() <= DENSE_MIN_WORDS
            {
                let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
                if long.len() > words.len() {
                    words.resize(long.len(), 0);
                }
                for ((target, a), b) in words.iter_mut().zip(long).zip(short) {
                    *target ^= a ^ b;
                }
                for (target, value) in words[short.len()..].iter_mut().zip(&long[short.len()..]) {
                    *target ^= value;
                }
                Self::finish_narrow(words);
                return;
            }
            target.xor_fused_words(a.len().max(b.len()), |i| {
                a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)
            });
            return;
        }
        for index in self {
            if !other.contains(index) {
                target.toggle(index);
            }
        }
        for index in other {
            if !self.contains(index) {
                target.toggle(index);
            }
        }
    }
}

impl std::fmt::Debug for AdaptiveIndexSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut elements: Vec<_> = self.iter().collect();
        elements.sort_unstable();
        f.debug_set().entries(elements).finish()
    }
}

/// Iterator over either representation of an [`AdaptiveIndexSet`].
pub enum AdaptiveIndexSetIter<'a> {
    /// Dense words, with the current word's unvisited bits and next word's base.
    Dense {
        words: std::slice::Iter<'a, u64>,
        bits: u64,
        base: usize,
    },
    /// Sparse keys in arbitrary order.
    Sparse(std::collections::hash_set::Iter<'a, u32>),
}

impl Iterator for AdaptiveIndexSetIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Dense { words, bits, base } => {
                while *bits == 0 {
                    *bits = *words.next()?;
                    *base += 64;
                }
                let index = *base - 64 + bits.trailing_zeros() as usize;
                *bits &= *bits - 1;
                Some(index)
            }
            Self::Sparse(elements) => elements.next().map(|&index| index as usize),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_random::PecosRng;
    use rand::RngExt;
    use std::collections::BTreeSet;

    fn invariant(set: &AdaptiveIndexSet) {
        set.assert_invariants();
    }

    fn check(set: &AdaptiveIndexSet, expected: &BTreeSet<usize>) {
        assert_eq!(&set.iter().collect::<BTreeSet<_>>(), expected);
        assert_eq!(set.iter().count(), expected.len());
        assert_eq!(set.len(), expected.len());
        assert_eq!(set.is_empty(), expected.is_empty());
        invariant(set);
    }

    fn toggle_reference(set: &mut BTreeSet<usize>, index: usize) {
        if !set.remove(&index) {
            set.insert(index);
        }
    }

    fn sample_index(rng: &mut PecosRng, distribution: u64) -> usize {
        match distribution {
            0 => rng.random_range(0..1000),
            1 => [0, 63, 64, 10_000_000, u32::MAX as usize][rng.random_range(0..5)],
            2 => rng.random_range(0..64 * (DENSE_MIN_WORDS + 16)),
            _ => {
                if rng.random_bool(0.5) {
                    rng.random_range(0..1000)
                } else {
                    rng.random_range(1_000_000..1_000_100)
                }
            }
        }
    }

    #[test]
    fn differential_random_operations() {
        for seed in 0..24 {
            let mut rng = PecosRng::seed_from_u64(seed);
            let mut set = AdaptiveIndexSet::with_capacity(usize::MAX);
            let mut expected = BTreeSet::new();
            check(&set, &expected);
            for _ in 0..1000 {
                let index = sample_index(&mut rng, seed % 4);
                let mut other = AdaptiveIndexSet::new();
                let mut rhs = BTreeSet::new();
                for _ in 0..rng.random_range(0..80) {
                    let value = sample_index(&mut rng, seed % 4);
                    other.insert(value);
                    rhs.insert(value);
                }
                match rng.random_range(0..12) {
                    0 => {
                        assert_eq!(set.insert(index), expected.insert(index));
                    }
                    1 => {
                        assert_eq!(set.remove(index), expected.remove(&index));
                    }
                    2 => {
                        set.toggle(index);
                        toggle_reference(&mut expected, index);
                    }
                    3 => {
                        set.xor_assign(&other);
                        expected = expected.symmetric_difference(&rhs).copied().collect();
                    }
                    4 => {
                        assert_eq!(
                            set.intersection_count(&other),
                            expected.intersection(&rhs).count()
                        );
                        assert_eq!(
                            other.intersection_count(&set),
                            rhs.intersection(&expected).count()
                        );
                    }
                    5 | 6 => {
                        let mut target = AdaptiveIndexSet::new();
                        let mut target_ref = BTreeSet::new();
                        for value in (0..2048)
                            .step_by(17)
                            .chain([sample_index(&mut rng, seed % 4)])
                        {
                            target.insert(value);
                            target_ref.insert(value);
                        }
                        if rng.random() {
                            set.xor_intersection_into(&other, &mut target);
                            for &value in expected.intersection(&rhs) {
                                toggle_reference(&mut target_ref, value);
                            }
                        } else {
                            other.xor_symmetric_difference_into(&set, &mut target);
                            for &value in expected.symmetric_difference(&rhs) {
                                toggle_reference(&mut target_ref, value);
                            }
                        }
                        check(&target, &target_ref);
                        set = target;
                        expected = target_ref;
                    }
                    7 => {
                        set.clear();
                        expected.clear();
                    }
                    8 => {
                        let taken = set.take_clearing();
                        check(&taken, &expected);
                        expected.clear();
                    }
                    9 => {
                        set.set_single(index);
                        expected.clear();
                        expected.insert(index);
                    }
                    10 => {
                        let cloned = set.clone();
                        check(&cloned, &expected);
                        other.clone_from(&set);
                        check(&other, &expected);
                        // Restore other to keep the input-preservation check below meaningful.
                        rhs = expected.clone();
                    }
                    _ => {
                        set.toggle_unchecked(index);
                        toggle_reference(&mut expected, index);
                    }
                }
                check(&set, &expected);
                check(&other, &rhs);
                assert_eq!(set.contains(index), expected.contains(&index));
            }
        }
    }

    fn dense(set: &AdaptiveIndexSet) -> bool {
        matches!(set.storage, Storage::Dense { .. })
    }

    fn block(end: usize) -> AdaptiveIndexSet {
        let mut set = AdaptiveIndexSet::new();
        for i in 0..end {
            set.insert(i);
            invariant(&set);
        }
        assert!(dense(&set));
        set
    }

    fn assert_width(set: &AdaptiveIndexSet, wide: bool) {
        let Storage::Dense { words, .. } = &set.storage else {
            panic!("expected dense storage");
        };
        assert_eq!(words.len() > DENSE_MIN_WORDS, wide);
    }

    #[test]
    fn repeated_narrow_wide_crossings() {
        for seed in 0..8 {
            let mut rng = PecosRng::seed_from_u64(seed);
            let mut set = block(4 * DENSE_MIN_WORDS);
            let mut expected: BTreeSet<_> = set.iter().collect();
            for cycle in 0..24 {
                // Vary narrow contents while retaining enough low bits to keep
                // the following growth dense for every supported W0/F pair.
                for _ in 0..4 {
                    let index = rng.random_range(4 * DENSE_MIN_WORDS..64 * DENSE_MIN_WORDS);
                    match rng.random_range(0..3) {
                        0 => assert_eq!(set.insert(index), expected.insert(index)),
                        1 => assert_eq!(set.remove(index), expected.remove(&index)),
                        _ => {
                            set.toggle(index);
                            toggle_reference(&mut expected, index);
                        }
                    }
                    check(&set, &expected);
                }
                // A narrow cache is deliberately invalid: no consumer may trust
                // it, including clone and the first growth back to wide.
                let Storage::Dense { len, .. } = &mut set.storage else {
                    unreachable!()
                };
                *len = usize::MAX;
                check(&set.clone(), &expected);
                let high = rng.random_range(64 * DENSE_MIN_WORDS..192 * DENSE_MIN_WORDS);
                if cycle % 2 == 0 {
                    assert!(set.insert(high));
                } else {
                    set.toggle(high);
                }
                expected.insert(high);
                check(&set, &expected);
                assert_width(&set, true);
                check(&set.clone(), &expected);
                if cycle % 2 == 0 {
                    assert!(set.remove(high));
                } else {
                    set.toggle(high);
                }
                expected.remove(&high);
                check(&set, &expected);
                assert_width(&set, false);

                let mut operand = block(16 * DENSE_MIN_WORDS);
                operand.insert(high);
                let rhs: BTreeSet<_> = operand.iter().collect();
                check(&operand, &rhs);
                // Repeat the crossing through XOR and both fused target paths.
                for operation in 0..3 {
                    let Storage::Dense { len, .. } = &mut set.storage else {
                        unreachable!()
                    };
                    *len = usize::MAX;
                    match operation {
                        0 => set.xor_assign(&operand),
                        1 => operand.xor_intersection_into(&operand, &mut set),
                        _ => operand
                            .xor_symmetric_difference_into(&AdaptiveIndexSet::new(), &mut set),
                    }
                    expected = expected.symmetric_difference(&rhs).copied().collect();
                    check(&set, &expected);
                    assert_width(&set, true);
                    set.xor_assign(&operand);
                    expected = expected.symmetric_difference(&rhs).copied().collect();
                    check(&set, &expected);
                    assert_width(&set, false);
                }

                let mut singleton = AdaptiveIndexSet::new();
                singleton.insert(high);
                assert!(!dense(&singleton));
                for wide in [true, false] {
                    set.xor_assign(&singleton);
                    toggle_reference(&mut expected, high);
                    check(&set, &expected);
                    assert_width(&set, wide);
                }
            }
        }
    }

    #[test]
    fn narrow_clearing_toggle_is_empty() {
        let mut set = AdaptiveIndexSet::new();
        // Gaps and the last surviving bit must not hide an empty high-water span.
        for index in [0, 63, 64, 64 * DENSE_MIN_WORDS - 1] {
            set.toggle(index);
            check(&set, &BTreeSet::from([index]));
            set.toggle(index);
            assert!(set.is_empty());
            check(&set, &BTreeSet::new());
        }
    }

    #[test]
    fn narrow_mutations_preserve_high_water() {
        let high = 64 * DENSE_MIN_WORDS - 1;
        let mut set = AdaptiveIndexSet::from_iter([0, high]);
        let operand = AdaptiveIndexSet::from_iter([high]);
        for operation in 0..5 {
            set.insert(high);
            match operation {
                0 => {
                    assert!(set.remove(high));
                }
                1 => set.toggle(high),
                2 => set.xor_assign(&operand),
                3 => operand.xor_intersection_into(&operand, &mut set),
                _ => operand.xor_symmetric_difference_into(&AdaptiveIndexSet::new(), &mut set),
            }
            check(&set, &BTreeSet::from([0]));
            let Storage::Dense { words, .. } = &set.storage else {
                unreachable!()
            };
            assert_eq!(words.len(), DENSE_MIN_WORDS);
            assert_eq!(words.last(), Some(&0));
        }
        let Storage::Dense { len, .. } = &mut set.storage else {
            unreachable!()
        };
        *len = usize::MAX;
        assert!(!set.remove(64 * DENSE_MIN_WORDS));
        check(&set, &BTreeSet::from([0]));
        let taken = set.take_clearing();
        check(&taken, &BTreeSet::from([0]));
        let Storage::Dense { words, len } = &set.storage else {
            unreachable!()
        };
        assert_eq!(words.as_slice(), &[] as &[u64]);
        assert_eq!(*len, 0);
    }

    #[test]
    fn wide_is_empty_uses_cached_len() {
        // Internal dispatch fixture: normal finishing would demote this singleton.
        // Its sole bit is in the top word, beyond the narrow scan bound.
        let mut words = vec![0; DENSE_MIN_WORDS + 1];
        words[DENSE_MIN_WORDS] = 1;
        let mut set = AdaptiveIndexSet {
            storage: Storage::Dense { words, len: 1 },
        };
        assert!(!set.is_empty());
        // Deliberately disagree with the words to prove wide emptiness reads the
        // cache, rather than scanning storage. This is not a public valid state.
        let Storage::Dense { len, .. } = &mut set.storage else {
            unreachable!()
        };
        *len = 0;
        assert!(set.is_empty());
    }

    #[test]
    fn modes_clear_and_layout() {
        fn send_sync<T: Send + Sync>() {}

        let mut set = block(1000);
        let Storage::Dense { words, .. } = &set.storage else {
            unreachable!()
        };
        let capacity = words.capacity();
        let width = words.len();
        set.clear();
        check(&set, &BTreeSet::new());
        let Storage::Dense { words, .. } = &set.storage else {
            unreachable!()
        };
        assert_eq!(words.capacity(), capacity);
        assert_eq!(words.len(), width);
        set.set_single(999);
        invariant(&set);
        let Storage::Dense { words, .. } = &set.storage else {
            unreachable!()
        };
        assert_eq!(words.capacity(), capacity);
        set.clear();
        set.insert(0);
        set.insert(10_000_000);
        assert!(!dense(&set));
        check(&set, &BTreeSet::from([0, 10_000_000]));
        set.clear();
        check(&set, &BTreeSet::new());
        let Storage::Dense { words, .. } = &set.storage else {
            unreachable!()
        };
        assert_eq!(words.capacity(), 0);
        let mut large = block(64 * (2 * DENSE_MIN_WORDS + 1));
        large.clear();
        let Storage::Dense { words, .. } = &large.storage else {
            unreachable!()
        };
        assert_eq!(words.capacity(), 0);
        invariant(&large);
        let size = std::mem::size_of::<AdaptiveIndexSet>();
        println!("size_of::<AdaptiveIndexSet>()={size}");
        assert!(size <= 48);
        send_sync::<AdaptiveIndexSet>();
    }

    #[test]
    fn clear_drops_even_a_small_wide_allocation() {
        let mut set = block(64 * (DENSE_MIN_WORDS + 1));
        let Storage::Dense { words, .. } = &set.storage else {
            unreachable!()
        };
        assert!(words.capacity() <= 2 * DENSE_MIN_WORDS);
        set.clear();
        check(&set, &BTreeSet::new());
        let Storage::Dense { words, len } = &set.storage else {
            unreachable!()
        };
        assert_eq!(words.capacity(), 0);
        assert_eq!(*len, 0);
    }

    #[test]
    fn dense_block_thins_and_releases_capacity() {
        let end = 64 * (2 * DENSE_MIN_WORDS + 10);
        let mut set = block(end);
        let mut expected: BTreeSet<_> = (0..end).collect();
        // Keep the highest bit: cardinality, rather than span, forces demotion.
        for i in 0..end - 1 {
            assert!(set.remove(i));
            expected.remove(&i);
            invariant(&set);
            assert_eq!(set.len(), expected.len());
        }
        check(&set, &expected);
        assert!(!dense(&set));
        // Removing the high tail instead must shrink the retained dense allocation.
        let mut set = block(end);
        for i in (2..end).rev() {
            set.toggle(i);
            invariant(&set);
        }
        assert!(dense(&set));
        check(&set, &BTreeSet::from([0, 1]));
        let Storage::Dense { words, .. } = &set.storage else {
            unreachable!()
        };
        assert!(words.capacity() <= 2 * DENSE_MIN_WORDS);
    }

    #[test]
    fn sparse_column_promotes_and_cancels_repeatedly() {
        let end = 64 * (DENSE_MIN_WORDS + 8);
        let hub = block(end);
        let rhs: BTreeSet<_> = hub.iter().collect();
        let mut column = AdaptiveIndexSet::new();
        column.set_single(end - 1);
        assert!(!dense(&column));
        let mut expected = BTreeSet::from([end - 1]);
        for step in 0..9 {
            column.xor_assign(&hub);
            expected = expected.symmetric_difference(&rhs).copied().collect();
            check(&column, &expected);
            assert_eq!(dense(&column), step % 2 == 0);
        }
        assert!(dense(&column));
        for intersection in [true, false] {
            let mut target = AdaptiveIndexSet::new();
            target.set_single(end - 1);
            if intersection {
                hub.xor_intersection_into(&hub, &mut target);
            } else {
                hub.xor_symmetric_difference_into(&AdaptiveIndexSet::new(), &mut target);
            }
            assert!(dense(&target));
            check(&target, &expected);
        }
    }

    #[test]
    fn bulk_operations_across_every_mode_combination() {
        let end = 64 * (DENSE_MIN_WORDS + 8);
        let compact = block(end);
        let mut sparse = AdaptiveIndexSet::new();
        for index in [0, 63, end - 1, 10_000_000] {
            sparse.insert(index);
            invariant(&sparse);
        }
        let mut hysteresis = compact.clone();
        let survivors = (end / 64).div_ceil(2 * DENSE_WORDS_PER_ELEMENT);
        for i in 0..end - survivors {
            hysteresis.remove(i);
            invariant(&hysteresis);
        }
        assert!(dense(&hysteresis));
        let mut same_sparse = AdaptiveIndexSet::new();
        same_sparse.insert(10_000_000);
        for i in &compact {
            same_sparse.insert(i);
        }
        same_sparse.remove(10_000_000);
        assert!(!dense(&same_sparse));
        check(&same_sparse, &compact.iter().collect());
        let sets = [
            AdaptiveIndexSet::new(),
            compact,
            sparse,
            hysteresis,
            same_sparse,
        ];
        for a in &sets {
            for b in &sets {
                let ar: BTreeSet<_> = a.iter().collect();
                let br: BTreeSet<_> = b.iter().collect();
                assert_eq!(a.intersection_count(b), ar.intersection(&br).count());
                let mut result = a.clone();
                result.xor_assign(b);
                check(&result, &ar.symmetric_difference(&br).copied().collect());
                for target in &sets {
                    let tr: BTreeSet<_> = target.iter().collect();
                    for intersection in [true, false] {
                        let mut result = target.clone();
                        let operand: BTreeSet<_> = if intersection {
                            a.xor_intersection_into(b, &mut result);
                            ar.intersection(&br).copied().collect()
                        } else {
                            a.xor_symmetric_difference_into(b, &mut result);
                            ar.symmetric_difference(&br).copied().collect()
                        };
                        check(
                            &result,
                            &tr.symmetric_difference(&operand).copied().collect(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn index_limit_and_sorted_debug() {
        let limit = u64::from(u32::MAX);
        let Ok(index) = usize::try_from(limit + 1) else {
            return;
        };
        for sparse in [false, true] {
            for operation in 0..6 {
                let panic = std::panic::catch_unwind(|| {
                    let mut set = AdaptiveIndexSet::new();
                    if sparse {
                        set.insert(u32::MAX as usize);
                    }
                    match operation {
                        0 => {
                            set.insert(index);
                        }
                        1 => {
                            set.remove(index);
                        }
                        2 => {
                            let _ = set.contains(index);
                        }
                        3 => set.toggle(index),
                        4 => set.toggle_unchecked(index),
                        _ => set.set_single(index),
                    }
                })
                .unwrap_err();
                let message = panic.downcast_ref::<String>().unwrap();
                assert!(message.contains(&index.to_string()), "{message}");
                assert!(message.contains(&limit.to_string()), "{message}");
            }
        }
        let mut set = AdaptiveIndexSet::new();
        let mut expected = BTreeSet::new();
        for i in [u32::MAX as usize, 128, 64, 0, 63, 127, 256] {
            set.insert(i);
            expected.insert(i);
            check(&set, &expected);
        }
        assert_eq!(format!("{set:?}"), format!("{expected:?}"));
    }
}
