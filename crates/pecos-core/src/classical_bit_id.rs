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

use std::fmt;

/// A unique identifier for a classical bit in a quantum circuit.
///
/// Classical bits receive measurement outcomes and can be used to condition
/// gates (classical control). This type mirrors [`QubitId`](crate::QubitId)
/// for the classical register space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct ClassicalBitId(pub usize);

impl ClassicalBitId {
    /// Create a new `ClassicalBitId`.
    #[inline]
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the underlying index value.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for ClassicalBitId {
    #[inline]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<ClassicalBitId> for usize {
    #[inline]
    fn from(cbit: ClassicalBitId) -> usize {
        cbit.0
    }
}

impl fmt::Display for ClassicalBitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_and_index() {
        let cbit = ClassicalBitId::new(42);
        assert_eq!(cbit.index(), 42);
        assert_eq!(cbit.0, 42);
    }

    #[test]
    fn test_from_usize() {
        let cbit = ClassicalBitId::from(5);
        assert_eq!(cbit.index(), 5);
    }

    #[test]
    fn test_into_usize() {
        let cbit = ClassicalBitId::new(7);
        let idx: usize = cbit.into();
        assert_eq!(idx, 7);
    }

    #[test]
    fn test_display() {
        let cbit = ClassicalBitId::new(3);
        assert_eq!(format!("{cbit}"), "3");
    }

    #[test]
    fn test_ordering() {
        let a = ClassicalBitId::new(1);
        let b = ClassicalBitId::new(2);
        assert!(a < b);
        assert_eq!(a, ClassicalBitId::new(1));
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ClassicalBitId::new(0));
        set.insert(ClassicalBitId::new(1));
        set.insert(ClassicalBitId::new(0)); // duplicate
        assert_eq!(set.len(), 2);
    }
}
