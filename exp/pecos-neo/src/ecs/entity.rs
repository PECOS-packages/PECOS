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

//! Entity identifiers.
//!
//! Entities are lightweight handles to simulation instances. They are:
//! - Cheap to copy (just a `u64`)
//! - Deterministically ordered (for reproducible iteration)
//! - Stable (IDs are never reused within a World's lifetime)

/// A unique identifier for a simulation instance.
///
/// Entity IDs are assigned sequentially and never reused, ensuring
/// deterministic behavior when iterating over entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub u64);

impl EntityId {
    /// Create a new entity ID with the given value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[inline]
    #[must_use]
    pub const fn id(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_id_ordering() {
        let e1 = EntityId(1);
        let e2 = EntityId(2);
        let e3 = EntityId(3);

        assert!(e1 < e2);
        assert!(e2 < e3);

        // BTreeMap ordering should be deterministic
        let mut entities = std::collections::BTreeSet::new();
        entities.insert(e3);
        entities.insert(e1);
        entities.insert(e2);

        let ordered: Vec<_> = entities.iter().copied().collect();
        assert_eq!(ordered, vec![e1, e2, e3]);
    }

    #[test]
    fn test_entity_id_display() {
        let e = EntityId(42);
        assert_eq!(format!("{e}"), "Entity(42)");
    }
}
