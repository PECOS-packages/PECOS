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

//! Component storage and component types.
//!
//! Components are plain data associated with entities. Each component type
//! is stored separately (Structure of Arrays pattern) for:
//!
//! - Cache-friendly iteration over one component type
//! - Easy addition of new component types
//! - Sparse storage (not all entities need all components)

use super::EntityId;
use crate::noise::NoiseContext;
use crate::outcome::MeasurementOutcomes;
use crate::sampling::weight::SampleWeight;
use pecos_random::PecosRng;
use std::collections::BTreeMap;

/// Generic storage for a component type.
///
/// Uses `BTreeMap` for deterministic iteration order.
#[derive(Debug, Clone)]
pub struct ComponentStorage<T> {
    data: BTreeMap<EntityId, T>,
}

impl<T> Default for ComponentStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ComponentStorage<T> {
    /// Create empty storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }

    /// Insert a component for an entity.
    pub fn insert(&mut self, entity: EntityId, component: T) -> Option<T> {
        self.data.insert(entity, component)
    }

    /// Get a component reference.
    #[must_use]
    pub fn get(&self, entity: EntityId) -> Option<&T> {
        self.data.get(&entity)
    }

    /// Get a mutable component reference.
    #[must_use]
    pub fn get_mut(&mut self, entity: EntityId) -> Option<&mut T> {
        self.data.get_mut(&entity)
    }

    /// Remove a component.
    pub fn remove(&mut self, entity: EntityId) -> Option<T> {
        self.data.remove(&entity)
    }

    /// Check if entity has this component.
    #[must_use]
    pub fn contains(&self, entity: EntityId) -> bool {
        self.data.contains_key(&entity)
    }

    /// Iterate over all (entity, component) pairs in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &T)> {
        self.data.iter().map(|(&e, c)| (e, c))
    }

    /// Iterate mutably over all (entity, component) pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        self.data.iter_mut().map(|(&e, c)| (e, c))
    }

    /// Get the number of components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get all entity IDs that have this component.
    pub fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.data.keys().copied()
    }

    /// Clear all components.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl<T: Clone> ComponentStorage<T> {
    /// Clone a component from one entity to another.
    ///
    /// Returns `Some(old)` if destination already had a component.
    pub fn clone_from(&mut self, src: EntityId, dst: EntityId) -> Option<T> {
        if let Some(component) = self.data.get(&src) {
            let cloned = component.clone();
            self.data.insert(dst, cloned)
        } else {
            None
        }
    }
}

// ============================================================================
// Specific Component Types
// ============================================================================

/// Component wrapping a quantum simulator state.
///
/// Generic over the simulator type to preserve type safety.
#[derive(Debug, Clone)]
pub struct SimulatorComponent<S> {
    pub simulator: S,
}

impl<S> SimulatorComponent<S> {
    #[must_use]
    pub fn new(simulator: S) -> Self {
        Self { simulator }
    }
}

/// Component for per-entity random number generator.
///
/// Each entity gets its own RNG, seeded deterministically from
/// the world's base seed and the entity's ID.
#[derive(Debug, Clone)]
pub struct RngComponent {
    pub rng: PecosRng,
}

impl RngComponent {
    #[must_use]
    pub fn new(rng: PecosRng) -> Self {
        Self { rng }
    }
}

/// Component for importance sampling weight.
#[derive(Debug, Clone, Default)]
pub struct WeightComponent {
    pub weight: SampleWeight,
}

impl WeightComponent {
    #[must_use]
    pub fn new(weight: SampleWeight) -> Self {
        Self { weight }
    }

    /// Create with unit weight.
    #[must_use]
    pub fn one() -> Self {
        Self::default()
    }
}

/// Component for noise context (leakage, prepared qubits, etc.).
#[derive(Debug, Clone, Default)]
pub struct NoiseContextComponent {
    pub context: NoiseContext,
}

impl NoiseContextComponent {
    #[must_use]
    pub fn new(context: NoiseContext) -> Self {
        Self { context }
    }
}

/// Component for measurement outcomes accumulated during a shot.
#[derive(Debug, Clone, Default)]
pub struct OutcomeComponent {
    pub outcomes: MeasurementOutcomes,
}

impl OutcomeComponent {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.outcomes.clear();
    }
}

/// Component tracking simulation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatusComponent {
    /// Simulation is active and running.
    #[default]
    Active,
    /// Simulation completed normally.
    Complete,
    /// Simulation failed (uncorrectable error).
    Failed,
    /// Simulation was pruned (weight too low).
    Pruned,
}

impl StatusComponent {
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    #[must_use]
    pub fn is_terminal(self) -> bool {
        !self.is_active()
    }
}

/// Component for tracking position in a branching program.
#[derive(Debug, Clone, Default)]
pub struct PathComponent {
    /// Current block in the program graph.
    pub current_block: usize,
    /// History of branch choices made.
    pub branch_history: Vec<usize>,
}

impl PathComponent {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn at_block(block: usize) -> Self {
        Self {
            current_block: block,
            branch_history: Vec::new(),
        }
    }

    /// Record a branch choice and move to a new block.
    pub fn branch_to(&mut self, choice: usize, new_block: usize) {
        self.branch_history.push(choice);
        self.current_block = new_block;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_storage_basic() {
        let mut storage: ComponentStorage<i32> = ComponentStorage::new();

        let e1 = EntityId(1);
        let e2 = EntityId(2);

        storage.insert(e1, 100);
        storage.insert(e2, 200);

        assert_eq!(storage.get(e1), Some(&100));
        assert_eq!(storage.get(e2), Some(&200));
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn test_component_storage_deterministic_order() {
        let mut storage: ComponentStorage<i32> = ComponentStorage::new();

        // Insert in non-sequential order
        storage.insert(EntityId(5), 50);
        storage.insert(EntityId(1), 10);
        storage.insert(EntityId(3), 30);

        // Iteration should be in EntityId order
        let pairs: Vec<_> = storage.iter().map(|(e, &v)| (e.0, v)).collect();
        assert_eq!(pairs, vec![(1, 10), (3, 30), (5, 50)]);
    }

    #[test]
    fn test_component_storage_clone_from() {
        let mut storage: ComponentStorage<String> = ComponentStorage::new();

        let e1 = EntityId(1);
        let e2 = EntityId(2);

        storage.insert(e1, "hello".to_string());
        storage.clone_from(e1, e2);

        assert_eq!(storage.get(e1), Some(&"hello".to_string()));
        assert_eq!(storage.get(e2), Some(&"hello".to_string()));
    }

    #[test]
    fn test_weight_component_default() {
        let weight = WeightComponent::default();
        assert!((weight.weight.weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_status_component() {
        assert!(StatusComponent::Active.is_active());
        assert!(!StatusComponent::Complete.is_active());
        assert!(StatusComponent::Complete.is_terminal());
        assert!(StatusComponent::Pruned.is_terminal());
    }

    #[test]
    fn test_path_component() {
        let mut path = PathComponent::at_block(0);
        assert_eq!(path.current_block, 0);

        path.branch_to(1, 5);
        assert_eq!(path.current_block, 5);
        assert_eq!(path.branch_history, vec![1]);

        path.branch_to(0, 10);
        assert_eq!(path.current_block, 10);
        assert_eq!(path.branch_history, vec![1, 0]);
    }
}
