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

use crate::VecSet;
use std::fmt;
use std::ops::Deref;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct QubitId(pub usize);

/// Type alias for a set of qubit IDs, useful for collections of qubits.
pub type QubitIdSet = VecSet<QubitId>;

// Automatic conversion from usize to QubitId
impl From<usize> for QubitId {
    #[inline]
    fn from(value: usize) -> Self {
        QubitId(value)
    }
}

// Automatic conversion from QubitId to usize
impl From<QubitId> for usize {
    #[inline]
    fn from(qubit: QubitId) -> usize {
        qubit.0
    }
}

// Add Deref implementation to match QubitIndex
impl Deref for QubitId {
    type Target = usize;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Add Display implementation to match QubitIndex
impl fmt::Display for QubitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Add convenience methods to match QubitIndex
impl QubitId {
    /// Create a new `QubitId`
    #[inline]
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the underlying index value
    #[inline]
    #[must_use]
    pub const fn index(&self) -> usize {
        self.0
    }
}

/// Helper function to create a single-qubit array for gate operations.
///
/// # Example
/// ```
/// use pecos_core::{QubitId, qid};
/// let qubits = qid(0);
/// assert_eq!(qubits, [QubitId(0)]);
/// ```
#[inline]
#[must_use]
pub const fn qid(n: usize) -> [QubitId; 1] {
    [QubitId(n)]
}

/// Helper function to create a two-qubit array for gate operations.
///
/// # Example
/// ```
/// use pecos_core::{QubitId, qid2};
/// let qubits = qid2(0, 1);
/// assert_eq!(qubits, [QubitId(0), QubitId(1)]);
/// ```
#[inline]
#[must_use]
pub const fn qid2(a: usize, b: usize) -> [QubitId; 2] {
    [QubitId(a), QubitId(b)]
}

/// Helper function to create a `Vec<QubitId>` from a collection of qubit indices.
///
/// Useful for batch single-qubit gate operations.
///
/// # Example
/// ```
/// use pecos_core::{QubitId, qids};
/// let qubits = qids([0, 1, 2]);
/// assert_eq!(qubits, vec![QubitId(0), QubitId(1), QubitId(2)]);
/// ```
#[inline]
#[must_use]
pub fn qids<I>(indices: I) -> Vec<QubitId>
where
    I: IntoIterator<Item = usize>,
{
    indices.into_iter().map(QubitId).collect()
}

/// Helper function to create a flattened `Vec<QubitId>` from pairs of qubit indices.
///
/// Useful for batch two-qubit gate operations where pairs are flattened to
/// `[control0, target0, control1, target1, ...]`.
///
/// # Example
/// ```
/// use pecos_core::{QubitId, qids2};
/// let qubits = qids2([(0, 1), (2, 3)]);
/// assert_eq!(qubits, vec![QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
/// ```
#[inline]
#[must_use]
pub fn qids2<I>(pairs: I) -> Vec<QubitId>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    pairs
        .into_iter()
        .flat_map(|(a, b)| [QubitId(a), QubitId(b)])
        .collect()
}
