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

//! The World - central container for entities, components, and resources.
//!
//! The World is the main entry point for the ECS-inspired simulation infrastructure.
//! It manages:
//!
//! - Entity allocation (sequential, deterministic IDs)
//! - Component storage (per-entity data)
//! - Resources (shared state like seed management)
//! - Entity lifecycle (spawn, clone, despawn)

use super::component::{
    ComponentStorage, NoiseContextComponent, OutcomeComponent, PathComponent, RngComponent,
    SimulatorComponent, StatusComponent, WeightComponent,
};
use super::entity::EntityId;
use super::resource::Resources;
use crate::noise::NoiseContext;
use crate::sampling::SampleWeight;
use pecos_core::rng::rng_manageable::{RngManageable, derive_seed};
use pecos_random::PecosRng;
use pecos_simulators::CliffordGateable;
use std::collections::BTreeSet;

/// Transferable entity state for moving entities between worlds.
///
/// This struct holds all the state needed to recreate an entity in a different
/// world, used for entity redistribution in parallel rare event simulation.
#[derive(Debug, Clone)]
pub struct EntityTransfer<S: CliffordGateable> {
    /// The simulator state.
    pub simulator: S,
    /// The entity's weight for importance sampling.
    pub weight: SampleWeight,
    /// The noise context (leakage, prepared qubits, etc.).
    pub noise_context: NoiseContext,
    /// The entity's status.
    pub status: StatusComponent,
}

/// The simulation world - container for all entities and their components.
///
/// Generic over the simulator type `S` to preserve type safety.
///
/// # Determinism
///
/// The World ensures deterministic behavior through:
/// - Sequential entity ID allocation (no reuse)
/// - `BTreeMap`-based component storage (ordered iteration)
/// - Centralized seed derivation (each entity gets a deterministic RNG)
///
/// # Example
///
/// ```
/// use pecos_neo::ecs::World;
/// use pecos_simulators::SparseStab;
///
/// // Create world with base seed
/// let mut world: World<SparseStab> = World::new(42);
///
/// // Spawn entities with simulators
/// let e1 = world.spawn_with_simulator(SparseStab::new(2));
/// let e2 = world.spawn_with_simulator(SparseStab::new(2));
///
/// // Each entity has its own RNG
/// assert!(world.rngs.get(e1).is_some());
/// assert!(world.rngs.get(e2).is_some());
/// ```
#[derive(Debug)]
pub struct World<S: CliffordGateable> {
    // Entity management
    next_entity_id: u64,
    alive_entities: BTreeSet<EntityId>,

    // Component storage (public for direct access in systems)
    /// Simulator state for each entity.
    pub simulators: ComponentStorage<SimulatorComponent<S>>,
    /// Per-entity RNG for noise and measurement.
    pub rngs: ComponentStorage<RngComponent>,
    /// Importance sampling weights.
    pub weights: ComponentStorage<WeightComponent>,
    /// Noise context (leakage, prepared qubits).
    pub noise_contexts: ComponentStorage<NoiseContextComponent>,
    /// Measurement outcomes.
    pub outcomes: ComponentStorage<OutcomeComponent>,
    /// Simulation status.
    pub statuses: ComponentStorage<StatusComponent>,
    /// Path through program graph.
    pub paths: ComponentStorage<PathComponent>,

    // Resources (shared state)
    /// Shared resources.
    pub resources: Resources,
}

impl<S: CliffordGateable> World<S> {
    /// Create a new world with the given base seed.
    #[must_use]
    pub fn new(base_seed: u64) -> Self {
        Self {
            next_entity_id: 0,
            alive_entities: BTreeSet::new(),
            simulators: ComponentStorage::new(),
            rngs: ComponentStorage::new(),
            weights: ComponentStorage::new(),
            noise_contexts: ComponentStorage::new(),
            outcomes: ComponentStorage::new(),
            statuses: ComponentStorage::new(),
            paths: ComponentStorage::new(),
            resources: Resources::new(base_seed),
        }
    }

    /// Get the base seed for this world.
    #[must_use]
    pub fn base_seed(&self) -> u64 {
        self.resources.seed.base_seed()
    }

    // ========================================================================
    // Entity Management
    // ========================================================================

    /// Spawn a new entity with no components.
    ///
    /// Returns the new entity's ID. The ID is deterministic based on
    /// spawn order.
    pub fn spawn(&mut self) -> EntityId {
        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;
        self.alive_entities.insert(id);
        id
    }

    /// Spawn a new entity with a simulator and default components.
    ///
    /// This is the common case for Monte Carlo simulation. The entity gets:
    /// - The provided simulator
    /// - A deterministically-seeded RNG
    /// - Unit weight (for importance sampling)
    /// - Empty noise context
    /// - Empty outcomes
    /// - Active status
    pub fn spawn_with_simulator(&mut self, simulator: S) -> EntityId {
        let entity = self.spawn();

        // Create RNG from deterministic seed
        let entity_seed = self.resources.seed.seed_for_entity(entity.0);
        let rng = PecosRng::seed_from_u64(entity_seed);

        // Add components
        self.simulators
            .insert(entity, SimulatorComponent::new(simulator));
        self.rngs.insert(entity, RngComponent::new(rng));
        self.weights.insert(entity, WeightComponent::one());
        self.noise_contexts
            .insert(entity, NoiseContextComponent::default());
        self.outcomes.insert(entity, OutcomeComponent::new());
        self.statuses.insert(entity, StatusComponent::Active);

        entity
    }

    /// Despawn an entity and remove all its components.
    pub fn despawn(&mut self, entity: EntityId) {
        if self.alive_entities.remove(&entity) {
            self.simulators.remove(entity);
            self.rngs.remove(entity);
            self.weights.remove(entity);
            self.noise_contexts.remove(entity);
            self.outcomes.remove(entity);
            self.statuses.remove(entity);
            self.paths.remove(entity);
        }
    }

    /// Check if an entity is alive.
    #[must_use]
    pub fn is_alive(&self, entity: EntityId) -> bool {
        self.alive_entities.contains(&entity)
    }

    /// Get the number of alive entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.alive_entities.len()
    }

    /// Iterate over all alive entities in deterministic order.
    pub fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.alive_entities.iter().copied()
    }

    /// Get entities with a specific status.
    #[must_use]
    pub fn entities_with_status(&self, status: StatusComponent) -> Vec<EntityId> {
        self.statuses
            .iter()
            .filter(|&(_, s)| *s == status)
            .map(|(e, _)| e)
            .collect()
    }

    /// Get all active entities.
    #[must_use]
    pub fn active_entities(&self) -> Vec<EntityId> {
        self.entities_with_status(StatusComponent::Active)
    }

    // ========================================================================
    // Entity Cloning (for splitting/branching)
    // ========================================================================

    /// Clone an entity, creating a new entity with copied components.
    ///
    /// The new entity gets:
    /// - A new, unique entity ID
    /// - Cloned simulator state
    /// - A fresh RNG (derived from base seed + new entity ID)
    /// - Cloned weight, noise context, outcomes, status, path
    ///
    /// This is the core operation for trajectory splitting in rare event
    /// simulation.
    pub fn clone_entity(&mut self, source: EntityId) -> Option<EntityId>
    where
        S: Clone,
    {
        if !self.is_alive(source) {
            return None;
        }

        let new_entity = self.spawn();

        // Clone simulator
        self.simulators.clone_from(source, new_entity);

        // New entity gets its own RNG (don't clone the RNG state!)
        let entity_seed = self.resources.seed.seed_for_entity(new_entity.0);
        let rng = PecosRng::seed_from_u64(entity_seed);
        self.rngs.insert(new_entity, RngComponent::new(rng));

        // Clone other components
        self.weights.clone_from(source, new_entity);
        self.noise_contexts.clone_from(source, new_entity);
        self.outcomes.clone_from(source, new_entity);
        self.statuses.clone_from(source, new_entity);
        self.paths.clone_from(source, new_entity);

        Some(new_entity)
    }

    /// Split an entity into multiple clones, dividing the weight.
    ///
    /// Creates `count - 1` new entities (the original counts as one).
    /// The weight is divided equally among all `count` entities.
    ///
    /// Returns the IDs of the new entities (not including the original).
    pub fn split_entity(&mut self, source: EntityId, count: usize) -> Vec<EntityId>
    where
        S: Clone,
    {
        if count <= 1 || !self.is_alive(source) {
            return vec![];
        }

        // Split the weight
        if let Some(weight_comp) = self.weights.get_mut(source) {
            weight_comp.weight = weight_comp.weight.split(count);
        }

        // Clone the entity (count - 1) times
        let mut clones = Vec::with_capacity(count - 1);
        for _ in 1..count {
            if let Some(clone) = self.clone_entity(source) {
                // The cloned weight is already split (from clone_from)
                clones.push(clone);
            }
        }

        clones
    }

    // ========================================================================
    // Entity Transfer (for redistribution across workers)
    // ========================================================================

    /// Extract an entity from this world, removing it and returning its transferable state.
    ///
    /// This is used for entity redistribution in parallel rare event simulation.
    /// The entity is removed from this world and can be imported into another.
    ///
    /// Returns `None` if the entity doesn't exist or is not alive.
    pub fn extract_entity(&mut self, entity: EntityId) -> Option<EntityTransfer<S>>
    where
        S: Clone,
    {
        if !self.is_alive(entity) {
            return None;
        }

        // Extract components
        let simulator = self.simulators.remove(entity)?.simulator;
        let weight = self
            .weights
            .remove(entity)
            .map_or_else(SampleWeight::one, |w| w.weight);
        let noise_context = self
            .noise_contexts
            .remove(entity)
            .map(|c| c.context)
            .unwrap_or_default();
        let status = self
            .statuses
            .remove(entity)
            .unwrap_or(StatusComponent::Active);

        // Remove remaining components
        self.rngs.remove(entity);
        self.outcomes.remove(entity);
        self.paths.remove(entity);

        // Remove from alive set
        self.alive_entities.remove(&entity);

        Some(EntityTransfer {
            simulator,
            weight,
            noise_context,
            status,
        })
    }

    /// Import an entity from a transfer, creating it in this world.
    ///
    /// This is used for entity redistribution in parallel rare event simulation.
    /// A new entity ID is allocated in this world.
    ///
    /// Returns the new entity ID.
    pub fn import_entity(&mut self, transfer: EntityTransfer<S>) -> EntityId {
        let entity = self.spawn();

        // Set up the simulator
        self.simulators
            .insert(entity, SimulatorComponent::new(transfer.simulator));

        // Create a new RNG for this entity (based on this world's seed)
        let entity_seed = self.resources.seed.seed_for_entity(entity.0);
        let rng = PecosRng::seed_from_u64(entity_seed);
        self.rngs.insert(entity, RngComponent::new(rng));

        // Import other components
        self.weights
            .insert(entity, WeightComponent::new(transfer.weight));
        self.noise_contexts.insert(
            entity,
            NoiseContextComponent {
                context: transfer.noise_context,
            },
        );
        self.outcomes.insert(entity, OutcomeComponent::new());
        self.statuses.insert(entity, transfer.status);

        entity
    }

    // ========================================================================
    // Batch Operations
    // ========================================================================

    /// Reset all entities for a new shot.
    ///
    /// Clears outcomes and resets noise contexts, but preserves entity IDs
    /// and RNG state (which advances naturally).
    pub fn reset_for_new_shot(&mut self) {
        for (_, outcome) in self.outcomes.iter_mut() {
            outcome.clear();
        }
        for (_, ctx) in self.noise_contexts.iter_mut() {
            ctx.context = NoiseContext::new();
        }
        for (_, status) in self.statuses.iter_mut() {
            *status = StatusComponent::Active;
        }
    }

    /// Prune entities below a weight threshold.
    ///
    /// Marks entities with weight below `threshold` as `Pruned`.
    /// Returns the number of entities pruned.
    pub fn prune_by_weight(&mut self, threshold: f64) -> usize {
        let to_prune: Vec<EntityId> = self
            .weights
            .iter()
            .filter(|(_, w)| w.weight.weight() < threshold)
            .map(|(e, _)| e)
            .collect();

        let count = to_prune.len();
        for entity in to_prune {
            if let Some(status) = self.statuses.get_mut(entity) {
                *status = StatusComponent::Pruned;
            }
        }
        count
    }

    /// Get total weight of all active entities.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.active_entities()
            .iter()
            .filter_map(|&e| self.weights.get(e))
            .map(|w| w.weight.weight())
            .sum()
    }

    /// Apply splitting decisions to entities.
    ///
    /// Takes a list of (entity, copies) pairs and:
    /// - Prunes entities with copies=0
    /// - Keeps entities with copies=1 unchanged
    /// - Splits entities with copies>1
    ///
    /// Returns the number of new entities created.
    pub fn apply_split_decisions(&mut self, decisions: &[(EntityId, usize)]) -> usize
    where
        S: Clone,
    {
        let mut created = 0;

        for &(entity, copies) in decisions {
            match copies {
                0 => {
                    // Prune
                    if let Some(status) = self.statuses.get_mut(entity) {
                        *status = StatusComponent::Pruned;
                    }
                }
                1 => {
                    // Keep unchanged
                }
                n => {
                    // Split into n copies
                    let clones = self.split_entity(entity, n);
                    created += clones.len();
                }
            }
        }

        created
    }

    /// Resample entities using multinomial resampling based on weights.
    ///
    /// This is used in subset simulation and splitting methods. It:
    /// 1. Computes selection probabilities from weights
    /// 2. Resamples to `target_count` entities
    /// 3. Adjusts weights to maintain unbiased estimation
    ///
    /// Returns the number of entities after resampling.
    #[allow(clippy::cast_precision_loss)] // weight calculation
    pub fn resample_by_weight(&mut self, target_count: usize, rng: &mut PecosRng) -> usize
    where
        S: Clone,
    {
        use rand::RngExt;

        let active: Vec<EntityId> = self.active_entities();
        if active.is_empty() || target_count == 0 {
            return 0;
        }

        // Collect weights and compute cumulative distribution
        let weights: Vec<f64> = active
            .iter()
            .map(|&e| self.weights.get(e).map_or(1.0, |w| w.weight.weight()))
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return 0;
        }

        // Build cumulative distribution function
        let mut cdf = Vec::with_capacity(weights.len());
        let mut cumsum = 0.0;
        for w in &weights {
            cumsum += w / total_weight;
            cdf.push(cumsum);
        }

        // Sample target_count entities with replacement using inverse CDF
        let mut selection_counts = vec![0usize; active.len()];
        for _ in 0..target_count {
            let u: f64 = rng.random();
            // Binary search for the first index where cdf[idx] >= u
            let idx = cdf.partition_point(|&x| x < u).min(cdf.len() - 1);
            selection_counts[idx] += 1;
        }

        // Apply decisions: prune unselected, clone selected multiple times
        let mut decisions = Vec::with_capacity(active.len());
        for (idx, &entity) in active.iter().enumerate() {
            decisions.push((entity, selection_counts[idx]));
        }

        self.apply_split_decisions(&decisions);

        // Reweight: each surviving entity gets weight = total_weight / target_count
        // Note: We must get the NEW active list since apply_split_decisions creates clones
        let new_weight =
            crate::sampling::SampleWeight::from_linear(total_weight / target_count as f64);
        let current_active = self.active_entities();
        for entity in &current_active {
            if let Some(weight_comp) = self.weights.get_mut(*entity) {
                weight_comp.weight = new_weight;
            }
        }

        current_active.len()
    }
}

// Additional impl block for simulators that support RNG management
impl<S> World<S>
where
    S: CliffordGateable + RngManageable<Rng = PecosRng>,
{
    /// Spawn an entity with full seed derivation for both simulator and noise RNG.
    ///
    /// This mirrors the hierarchical seeding pattern from `MonteCarloEngine`:
    /// ```text
    /// entity_seed (derived from base_seed + entity_id)
    /// ├── noise (for noise channel RNG)
    /// └── simulator (for simulator's internal RNG)
    /// ```
    pub fn spawn_with_full_seeding(&mut self, mut simulator: S) -> EntityId {
        let entity = self.spawn();

        // Derive separate seeds for noise and simulator
        let entity_seed = self.resources.seed.seed_for_entity(entity.0);
        let noise_seed = derive_seed(entity_seed, "noise");
        let sim_seed = derive_seed(entity_seed, "simulator");

        // Seed the simulator
        simulator.set_seed(sim_seed);

        // Create noise RNG
        let rng = PecosRng::seed_from_u64(noise_seed);

        // Add components
        self.simulators
            .insert(entity, SimulatorComponent::new(simulator));
        self.rngs.insert(entity, RngComponent::new(rng));
        self.weights.insert(entity, WeightComponent::one());
        self.noise_contexts
            .insert(entity, NoiseContextComponent::default());
        self.outcomes.insert(entity, OutcomeComponent::new());
        self.statuses.insert(entity, StatusComponent::Active);

        entity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::weight::SampleWeight;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_world_spawn() {
        let mut world: World<SparseStab> = World::new(42);

        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();

        assert_eq!(e1, EntityId(0));
        assert_eq!(e2, EntityId(1));
        assert_eq!(e3, EntityId(2));
        assert_eq!(world.entity_count(), 3);
    }

    #[test]
    fn test_world_spawn_with_simulator() {
        let mut world: World<SparseStab> = World::new(42);

        let e = world.spawn_with_simulator(SparseStab::new(2));

        assert!(world.simulators.contains(e));
        assert!(world.rngs.contains(e));
        assert!(world.weights.contains(e));
        assert!(world.noise_contexts.contains(e));
        assert!(world.outcomes.contains(e));
        assert!(world.statuses.contains(e));
    }

    #[test]
    fn test_world_despawn() {
        let mut world: World<SparseStab> = World::new(42);

        let e = world.spawn_with_simulator(SparseStab::new(2));
        assert!(world.is_alive(e));

        world.despawn(e);
        assert!(!world.is_alive(e));
        assert!(!world.simulators.contains(e));
        assert!(!world.rngs.contains(e));
    }

    #[test]
    fn test_world_clone_entity() {
        let mut world: World<SparseStab> = World::new(42);

        let original = world.spawn_with_simulator(SparseStab::new(2));
        let clone = world.clone_entity(original).unwrap();

        assert_ne!(original, clone);
        assert!(world.simulators.contains(clone));
        assert!(world.rngs.contains(clone));

        // Both entities should have RNGs (cloned entity gets its own derived seed)
        assert!(world.rngs.get(original).is_some());
        assert!(world.rngs.get(clone).is_some());
        assert!(world.is_alive(clone));
    }

    #[test]
    fn test_world_split_entity() {
        let mut world: World<SparseStab> = World::new(42);

        let original = world.spawn_with_simulator(SparseStab::new(2));

        // Get original weight
        let orig_weight = world.weights.get(original).unwrap().weight.weight();
        assert!((orig_weight - 1.0).abs() < 1e-10);

        // Split into 4
        let clones = world.split_entity(original, 4);
        assert_eq!(clones.len(), 3);

        // All 4 entities should have 1/4 weight
        let expected_weight = 0.25;
        let orig_weight = world.weights.get(original).unwrap().weight.weight();
        assert!(
            (orig_weight - expected_weight).abs() < 1e-10,
            "Original weight: {orig_weight}"
        );

        for &clone in &clones {
            let clone_weight = world.weights.get(clone).unwrap().weight.weight();
            assert!(
                (clone_weight - expected_weight).abs() < 1e-10,
                "Clone weight: {clone_weight}"
            );
        }
    }

    #[test]
    fn test_world_deterministic_entity_order() {
        let mut world: World<SparseStab> = World::new(42);

        for _ in 0..10 {
            world.spawn_with_simulator(SparseStab::new(2));
        }

        let entities: Vec<EntityId> = world.entities().collect();
        let expected: Vec<EntityId> = (0..10).map(EntityId).collect();
        assert_eq!(entities, expected);
    }

    #[test]
    fn test_world_prune_by_weight() {
        let mut world: World<SparseStab> = World::new(42);

        let e1 = world.spawn_with_simulator(SparseStab::new(2));
        let e2 = world.spawn_with_simulator(SparseStab::new(2));

        // Set e2 to very low weight
        world.weights.get_mut(e2).unwrap().weight = SampleWeight::from_linear(0.001);

        let pruned = world.prune_by_weight(0.01);
        assert_eq!(pruned, 1);

        assert_eq!(*world.statuses.get(e1).unwrap(), StatusComponent::Active);
        assert_eq!(*world.statuses.get(e2).unwrap(), StatusComponent::Pruned);
    }

    #[test]
    fn test_world_spawn_with_full_seeding() {
        let mut world: World<SparseStab> = World::new(42);

        let e1 = world.spawn_with_full_seeding(SparseStab::new(2));
        let e2 = world.spawn_with_full_seeding(SparseStab::new(2));

        assert!(world.is_alive(e1));
        assert!(world.is_alive(e2));

        // Both should have simulators and RNGs
        assert!(world.simulators.contains(e1));
        assert!(world.rngs.contains(e1));
    }

    #[test]
    fn test_world_full_seeding_determinism() {
        // Two worlds with same seed should produce same entity seeds
        let mut world1: World<SparseStab> = World::new(42);
        let mut world2: World<SparseStab> = World::new(42);

        let e1a = world1.spawn_with_full_seeding(SparseStab::new(2));
        let e1b = world2.spawn_with_full_seeding(SparseStab::new(2));

        // Entity IDs should match
        assert_eq!(e1a, e1b);

        // The derived seeds should be the same (we can verify via resource)
        assert_eq!(
            world1.resources.seed.seed_for_entity(e1a.0),
            world2.resources.seed.seed_for_entity(e1b.0)
        );
    }
}
