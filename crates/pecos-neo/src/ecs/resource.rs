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

//! Resources - shared state across all entities.
//!
//! Unlike components (per-entity data), resources are singletons that
//! provide shared configuration or state for the entire simulation.

use pecos_core::rng::rng_manageable::derive_seed;

/// Resource managing deterministic seed derivation.
///
/// All entity RNGs are derived from this base seed, ensuring
/// reproducible simulations.
#[derive(Debug, Clone)]
pub struct SeedResource {
    base_seed: u64,
}

impl SeedResource {
    /// Create a new seed resource with the given base seed.
    #[must_use]
    pub fn new(base_seed: u64) -> Self {
        Self { base_seed }
    }

    /// Get the base seed.
    #[must_use]
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Derive a seed for an entity.
    ///
    /// Seeds are derived using: `derive_seed(base_seed, "entity_{id}")`
    /// This ensures each entity gets a unique, deterministic seed.
    #[must_use]
    pub fn seed_for_entity(&self, entity_id: u64) -> u64 {
        derive_seed(self.base_seed, &format!("entity_{entity_id}"))
    }

    /// Derive a seed for a specific purpose within an entity.
    ///
    /// Useful when an entity needs multiple independent RNGs.
    /// Seeds are derived using: `derive_seed(entity_seed, purpose)`
    #[must_use]
    pub fn seed_for_purpose(&self, entity_id: u64, purpose: &str) -> u64 {
        let entity_seed = self.seed_for_entity(entity_id);
        derive_seed(entity_seed, purpose)
    }
}

/// Container for all resources in the simulation.
///
/// Resources are shared, read-mostly state. Unlike components,
/// there's exactly one of each resource type.
#[derive(Debug, Clone)]
pub struct Resources {
    /// Seed management for deterministic RNGs.
    pub seed: SeedResource,

    /// Global time step counter (optional, for time-based simulations).
    pub time_step: u64,
}

impl Resources {
    /// Create resources with the given base seed.
    #[must_use]
    pub fn new(base_seed: u64) -> Self {
        Self {
            seed: SeedResource::new(base_seed),
            time_step: 0,
        }
    }

    /// Advance the time step.
    pub fn advance_time(&mut self) {
        self.time_step += 1;
    }

    /// Reset time to zero.
    pub fn reset_time(&mut self) {
        self.time_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_resource_determinism() {
        let seed1 = SeedResource::new(42);
        let seed2 = SeedResource::new(42);

        // Same base seed should produce same entity seeds
        assert_eq!(seed1.seed_for_entity(0), seed2.seed_for_entity(0));
        assert_eq!(seed1.seed_for_entity(100), seed2.seed_for_entity(100));
    }

    #[test]
    fn test_seed_resource_uniqueness() {
        let seed = SeedResource::new(42);

        // Different entities should get different seeds
        let s0 = seed.seed_for_entity(0);
        let s1 = seed.seed_for_entity(1);
        let s2 = seed.seed_for_entity(2);

        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s0, s2);
    }

    #[test]
    fn test_seed_for_purpose() {
        let seed = SeedResource::new(42);

        let noise_seed = seed.seed_for_purpose(0, "noise");
        let sim_seed = seed.seed_for_purpose(0, "simulator");

        // Different purposes should get different seeds
        assert_ne!(noise_seed, sim_seed);

        // Same purpose should be deterministic
        assert_eq!(noise_seed, seed.seed_for_purpose(0, "noise"));
    }

    #[test]
    fn test_resources_time() {
        let mut resources = Resources::new(42);
        assert_eq!(resources.time_step, 0);

        resources.advance_time();
        assert_eq!(resources.time_step, 1);

        resources.advance_time();
        assert_eq!(resources.time_step, 2);

        resources.reset_time();
        assert_eq!(resources.time_step, 0);
    }
}
