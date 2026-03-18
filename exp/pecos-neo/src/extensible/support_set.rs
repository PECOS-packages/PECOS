//! Compact support set for gates using `BitVec`.

use super::GateId;
use crate::noise::BitVec;

/// Compact set of gate IDs using a bit vector.
///
/// Provides O(1) insert, contains, and set operations.
/// Much faster than `HashSet` for gate ID operations.
#[derive(Clone, Debug, Default)]
pub struct GateSupportSet {
    /// Bits indexed by `GateId` - bit is set if gate is in the set
    bits: BitVec,
}

impl GateSupportSet {
    /// Create a new empty support set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bits: BitVec::default(),
        }
    }

    /// Create a support set with capacity for the given max ID.
    #[must_use]
    pub fn with_capacity(max_id: usize) -> Self {
        Self {
            bits: BitVec::with_capacity(max_id + 1),
        }
    }

    /// Insert a gate ID into the set.
    #[inline]
    pub fn insert(&mut self, id: GateId) {
        let idx = id.0 as usize;
        self.bits.set(idx); // Auto-resizes if needed
    }

    /// Check if the set contains a gate ID.
    #[inline]
    #[must_use]
    pub fn contains(&self, id: GateId) -> bool {
        let idx = id.0 as usize;
        self.bits.get(idx)
    }

    /// Remove a gate ID from the set.
    #[inline]
    pub fn remove(&mut self, id: GateId) {
        let idx = id.0 as usize;
        self.bits.clear(idx);
    }

    /// Check if the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.count_ones() == 0
    }

    /// Get the number of gates in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.count_ones()
    }

    /// Iterate over all gate IDs in the set.
    pub fn iter(&self) -> impl Iterator<Item = GateId> + '_ {
        self.bits.iter_ones().map(|i| GateId(i as u16))
    }

    /// Union with another set (modifies self).
    pub fn union_with(&mut self, other: &GateSupportSet) {
        for id in other.iter() {
            self.insert(id);
        }
    }

    /// Intersection with another set (modifies self).
    pub fn intersect_with(&mut self, other: &GateSupportSet) {
        // Collect IDs to remove first to avoid iterator invalidation
        let to_remove: Vec<_> = self.iter().filter(|id| !other.contains(*id)).collect();

        for id in to_remove {
            self.remove(id);
        }
    }

    /// Compute set difference (self - other), returns new set.
    #[must_use]
    pub fn difference(&self, other: &GateSupportSet) -> GateSupportSet {
        let mut result = GateSupportSet::new();
        for id in self.iter() {
            if !other.contains(id) {
                result.insert(id);
            }
        }
        result
    }

    /// Check if this set is a subset of another.
    #[must_use]
    pub fn is_subset_of(&self, other: &GateSupportSet) -> bool {
        for id in self.iter() {
            if !other.contains(id) {
                return false;
            }
        }
        true
    }

    /// Get the underlying bits for low-level operations.
    #[must_use]
    pub fn bits(&self) -> &BitVec {
        &self.bits
    }

    /// Get mutable access to the underlying bits.
    pub fn bits_mut(&mut self) -> &mut BitVec {
        &mut self.bits
    }
}

impl FromIterator<GateId> for GateSupportSet {
    fn from_iter<I: IntoIterator<Item = GateId>>(iter: I) -> Self {
        let mut set = GateSupportSet::new();
        for id in iter {
            set.insert(id);
        }
        set
    }
}
