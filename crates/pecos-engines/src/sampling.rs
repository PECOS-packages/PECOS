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

//! Cross-stack sampling vocabulary.
//!
//! [`monte_carlo()`] builds a [`MonteCarloBuilder`] -- the stack-agnostic
//! run-spec for Monte Carlo sampling (shot count plus optional worker
//! parallelism). It is the single source of truth shared by both simulation
//! stacks: the engines [`MonteCarloEngine`](crate::MonteCarloEngine) consumes
//! it directly, and the `pecos-neo` stack converts it into its own `Sampling`
//! strategy. The unified facade (`pecos::sim().stack(...).sampling(...)`) and
//! the neo builder (`sim_neo().sampling(...)`) therefore accept the SAME
//! `monte_carlo(n).workers(m)` spelling.
//!
//! Monte Carlo is the only strategy both stacks share; richer rare-event
//! strategies (importance sampling, subset simulation) are `pecos-neo`-only and
//! live there.

use std::num::NonZero;

/// Builder for the Monte Carlo sampling strategy.
///
/// Created by [`monte_carlo()`]. The shot count is the defining argument;
/// worker parallelism is optional and unset by default (sequential). Worker
/// resolution is deferred to [`resolved_workers()`](Self::resolved_workers) so
/// an unset count does not silently override a worker count configured
/// elsewhere on a builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonteCarloBuilder {
    shots: usize,
    /// Explicit worker count, or `None` when unset. `Some` from `.workers(n)`.
    workers: Option<usize>,
    /// `true` from `.auto_workers()`: resolve to available parallelism at use.
    auto_workers: bool,
}

impl MonteCarloBuilder {
    /// Set the number of parallel workers.
    ///
    /// Parallel execution distributes shots across workers, each with its own
    /// simulator, command source, and noise model built from the shared
    /// configuration. Per-shot seeding uses global shot indices on the neo
    /// stack, so neo results are identical for any worker count (the engines
    /// stack does not make that guarantee).
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self.auto_workers = false;
        self
    }

    /// Request a worker count derived from the machine's available parallelism,
    /// resolved when the spec is consumed (see [`workers()`](Self::workers)).
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.auto_workers = true;
        self
    }

    /// The configured shot count.
    #[must_use]
    pub fn shots(&self) -> usize {
        self.shots
    }

    /// The explicitly-set worker count, or `None` when unset.
    ///
    /// `None` means neither `.workers(n)` nor `.auto_workers()` was called;
    /// callers that need a concrete count should use
    /// [`resolved_workers()`](Self::resolved_workers).
    #[must_use]
    pub fn worker_count(&self) -> Option<usize> {
        self.workers
    }

    /// Whether [`auto_workers()`](Self::auto_workers) was requested.
    #[must_use]
    pub fn auto_workers_requested(&self) -> bool {
        self.auto_workers
    }

    /// The concrete worker count: available parallelism when `.auto_workers()`
    /// was requested, the explicit `.workers(n)` otherwise, and `1` when unset.
    #[must_use]
    pub fn resolved_workers(&self) -> usize {
        if self.auto_workers {
            std::thread::available_parallelism().map_or(1, NonZero::get)
        } else {
            self.workers.unwrap_or(1)
        }
    }
}

/// Create a Monte Carlo sampling spec running `shots` shots.
///
/// This is the standard execution strategy: each shot runs the program once
/// and records its outcomes. Sequential by default; add
/// [`workers(n)`](MonteCarloBuilder::workers) or
/// [`auto_workers()`](MonteCarloBuilder::auto_workers) for parallel execution.
#[must_use]
pub fn monte_carlo(shots: usize) -> MonteCarloBuilder {
    MonteCarloBuilder {
        shots,
        workers: None,
        auto_workers: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monte_carlo_defaults_to_sequential_unset_workers() {
        let mc = monte_carlo(100);
        assert_eq!(mc.shots(), 100);
        assert_eq!(mc.worker_count(), None);
        assert!(!mc.auto_workers_requested());
        // Unset resolves to a single worker.
        assert_eq!(mc.resolved_workers(), 1);
    }

    #[test]
    fn workers_sets_explicit_count() {
        let mc = monte_carlo(100).workers(8);
        assert_eq!(mc.worker_count(), Some(8));
        assert_eq!(mc.resolved_workers(), 8);
        assert!(!mc.auto_workers_requested());
    }

    #[test]
    fn auto_workers_resolves_to_available_parallelism() {
        let mc = monte_carlo(100).auto_workers();
        assert!(mc.auto_workers_requested());
        assert_eq!(mc.worker_count(), None);
        let expected = std::thread::available_parallelism().map_or(1, NonZero::get);
        assert_eq!(mc.resolved_workers(), expected);
    }

    #[test]
    fn last_worker_setter_wins() {
        // auto then explicit -> explicit
        assert_eq!(
            monte_carlo(1).auto_workers().workers(3).resolved_workers(),
            3
        );
        // explicit then auto -> auto
        let auto = std::thread::available_parallelism().map_or(1, NonZero::get);
        assert_eq!(
            monte_carlo(1).workers(3).auto_workers().resolved_workers(),
            auto
        );
    }
}
