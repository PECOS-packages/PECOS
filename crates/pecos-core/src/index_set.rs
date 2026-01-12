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

//! A trait for set operations on `usize` indices.
//!
//! This module provides [`IndexSet`], a trait that abstracts over different set
//! implementations for storing `usize` indices. It enables generic implementations
//! of stabilizer simulators that can use either [`BitSet`](crate::BitSet) or
//! [`VecSet<usize>`](crate::VecSet) as the underlying storage.
//!
//! # Implementations
//!
//! - [`BitSet`](crate::BitSet): O(1) toggle operations, efficient for large circuits
//! - [`VecSet<usize>`](crate::VecSet): Lower overhead for small sets, better iteration
//!
//! # Example
//!
//! ```rust
//! use pecos_core::{BitSet, IndexSet};
//!
//! fn process_set<S: IndexSet>(set: &mut S) {
//!     set.insert(0);
//!     set.toggle(1);
//!     set.toggle(1);  // 1 is now removed
//!     assert!(set.contains(0));
//!     assert!(!set.contains(1));
//! }
//!
//! let mut bitset = BitSet::new();
//! process_set(&mut bitset);
//! ```

use core::fmt::Debug;

/// A trait for set operations on `usize` indices.
///
/// This trait provides a common interface for set types used in stabilizer
/// simulation, abstracting over the differences between [`BitSet`](crate::BitSet)
/// and [`VecSet<usize>`](crate::VecSet).
///
/// # Key Operations
///
/// - `toggle`: Insert if absent, remove if present (XOR with single element)
/// - `xor_assign`: Symmetric difference with another set
/// - Standard set operations: `insert`, `remove`, `contains`, `iter`
pub trait IndexSet: Clone + Default + Debug {
    /// Iterator type for iterating over set elements.
    type Iter<'a>: Iterator<Item = usize>
    where
        Self: 'a;

    /// Create a new empty set.
    fn new() -> Self;

    /// Insert an index into the set.
    ///
    /// Returns `true` if the index was newly inserted, `false` if already present.
    fn insert(&mut self, index: usize) -> bool;

    /// Remove an index from the set.
    ///
    /// Returns `true` if the index was present, `false` otherwise.
    fn remove(&mut self, index: usize) -> bool;

    /// Check if the set contains an index.
    fn contains(&self, index: usize) -> bool;

    /// Toggle an index (insert if absent, remove if present).
    ///
    /// This is the key operation for CX gate implementation in stabilizer
    /// simulation, where we need to XOR single elements into sets.
    fn toggle(&mut self, index: usize);

    /// XOR (symmetric difference) with another set in place.
    ///
    /// Elements present in exactly one of the two sets will be in the result.
    fn xor_assign(&mut self, other: &Self);

    /// Iterate over indices in the set.
    fn iter(&self) -> Self::Iter<'_>;

    /// Check if the set is empty.
    fn is_empty(&self) -> bool;

    /// Get the number of elements in the set.
    fn len(&self) -> usize;

    /// Remove all elements from the set.
    fn clear(&mut self);

    /// Clear the set and insert a single element.
    ///
    /// This is an optimization for initializing identity matrices where
    /// set[i] = {i}. Avoids the `contains()` check since we know the set is empty.
    fn set_single(&mut self, index: usize) {
        self.clear();
        self.insert(index);
    }

    /// Count elements present in both this set and another.
    ///
    /// Used for computing commutation relations in stabilizer simulation.
    fn intersection_count(&self, other: &Self) -> usize;

    /// XOR elements in the intersection of self and other into target.
    ///
    /// For each element in (self AND other), toggle it in target.
    /// This is used for sign propagation in stabilizer simulation.
    fn xor_intersection_into(&self, other: &Self, target: &mut Self);

    /// XOR elements in the symmetric difference of self and other into target.
    ///
    /// For each element in (self XOR other), toggle it in target.
    /// This is used for the Y gate sign propagation in stabilizer simulation.
    fn xor_symmetric_difference_into(&self, other: &Self, target: &mut Self);
}
