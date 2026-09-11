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

use pecos_engines::{
    ClassicalControlEngine, HybridEngine, NoiseModel, QuantumEngine, QuantumSystem,
    monte_carlo::MonteCarloEngineBuilder,
};
use pecos_core::rng::PecosRng;
use pecos_engines::depolarizing::{
    DepolarizingFaultCatalog, DepolarizingSampledFault, DepolarizingFaultHistory,
};

// MetropolisStep
// Represents a single step in the Metropolis algorithm,
// including the state, whether it was accepted, and the acceptance probability.
struct MetropolisStep<State> {
    state: State,
    accepted: bool,
    acceptance_probability: f64,
}

// Metropolis Stepper
// 
// Implements a `step` method, which takes two states and performs a metropolis step
// returning a metropolis step that has been accepted with probability `min(1,acceptance_ratio)`,
// and updating the state of the metropolis step based on if the step was accepted or not.
struct MetropolisStepper {rng: PecosRng }

impl MetropolisStepper {
    #[must_use]
    pub fn step<State>(
        &mut self,
        current: State,
        proposal: State, // Probably don't need two of these
        acceptance_ratio: f64)
        -> MetropolisStep<State> {
        assert!(
            acceptance_ratio >= 0.0,
            "Acceptance ratio must be a non-negative value! got {}", acceptance_ratio
        );

        // Metropolis-hastings acceptance probability
        // assuming that the hastings correction comes in `acceptance_ratio` if needed.
        let acceptance_probability: f64 = acceptance_ratio.min(1.0);
        // accept the new state with probability `acceptance_probability`
        let accepted: bool = self.rng.next_f64() < acceptance_probability;
        // return the updated state
        MetropolisStep {
            state: if accepted {proposal} else {current},
            accepted,
            acceptance_probability,
        }
    }
}

#[derive(Default)]
pub struct MetropolisEngineBuilder {
    /// MonteCarloEngineBuilder
    monte_carlo_engine_builder: Option<MonteCarloEngineBuilder>,
    /// Optional MonteCarloEngine (for building fault catalogs and histories)
    monte_carlo_engine: Option<MonteCarloEngine>,
    /// Optional seed for the `MetropolisEngine`'s RNG
    seed: Option<u64>,
    /// Optional FaultCatalog input
    fault_catalog: Option<DepolarizingFaultCatalog>,
    /// Optional initial FaultHistory input
    fault_history: Option<DepolarizingFaultHistory>,
}

impl MetropolisEngineBuilder {
    /// Create a new `MetropolisEngineBuilder` with default settings
    ///
    /// # Returns
    /// A new `MetropolisEngineBuilder` with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            monte_carlo_engine_builder: Some(MonteCarloEngineBuilder::new()),
            seed: None,
        }
    }

    fn update_monte_carlo_builder(
        mut self,
        update: impl FnOnce(MonteCarloEngineBuilder) -> MonteCarloEngineBuilder,
    ) -> Self {
        let builder = self
            .monte_carlo_engine_builder
            .take()
            .unwrap_or_else(MonteCarloEngineBuilder::new);
        self.monte_carlo_engine_builder = Some(update(builder));
        self
    }

    #[must_use]
    pub fn with_classical_engine(self, engine: Box<dyn ClassicalControlEngine>) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_classical_engine(engine))
    }

    #[must_use]
    pub fn with_quantum_engine(self, engine: Box<dyn QuantumEngine>) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_quantum_engine(engine))
    }

    #[must_use]
    pub fn with_noise_model(self, model: Box<dyn NoiseModel>) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_noise_model(model))
    }

    #[must_use]
    pub fn with_depolarizing_noise(self, probability: f64) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_depolarizing_noise(probability))
    }

    #[must_use]
    pub fn with_quantum_system(self, system: QuantumSystem) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_quantum_system(system))
    }

    #[must_use]
    pub fn with_hybrid_engine(self, engine: HybridEngine) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_hybrid_engine(engine))
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        // TODO: Need to sort out how the different seeds work
        self.seed = Some(seed);
        self.update_monte_carlo_builder(|builder| builder.with_seed(seed))
    }

    #[must_use]
    pub fn with_default_workers(self, workers: usize) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_default_workers(workers))
    }

    #[must_use]
    pub fn with_num_qubits(self, num_qubits: usize) -> Self {
        self.update_monte_carlo_builder(|builder| builder.with_num_qubits(num_qubits))
    }

    pub fn with_fault_history(mut self, fault_history: FaultHistory) -> Self {
        self.fault_history = Some(fault_history);
        self
    }

    pub fn with_fault_catalog(mut self, fault_catalog: FaultCatalog) -> Self {
        self.fault_catalog = Some(fault_catalog);
        self
    }

    // Uses the monte carlo engine to obtain a sampled fault catalog
    fn obtain_fault_catalog(&mut self) -> FaultCatalog {
        // Generate a new fault catalog if it doesn't exist
        self.monte_carlo_engine.as_mut()
            .unwrap_or_else(|| panic!("MonteCarloEngine not initialized"))
            .return_fault_catalog()
    }

    // Uses the monte carlo engine to obtain a sampled fault history
    fn obtain_fault_history(&mut self) -> FaultHistory {
        // Generate a new fault history if it doesn't exist
        self.monte_carlo_engine.as_mut()
            .unwrap_or_else(|| panic!("MonteCarloEngine not initialized"))
            .run_with_fault_tracking().fault_histories[0]
    }

    /// Builds the Monte Carlo engine with the configured settings.
    pub fn build(self) -> MetropolisEngine {

        // Create a new Monte Carlo engine with the hybrid engine
        let seed = resolve_seed(self.seed);
        let rng = PecosRng::seed_from_u64(seed);

        // See if we need to build the MonteCarloEngine
        if !self.fault_catalog.is_some() || !self.fault_history.is_some() {
            let monte_carlo_engine = self.monte_carlo_engine_builder.build();
            // Ensure the MonteCarloEngine is available for subsequent use
            self.monte_carlo_engine = Some(monte_carlo_engine);
        }

        // If needed, generate the fault catalog
        if !self.fault_catalog.is_some() {
            self.fault_catalog = Some(self.obtain_fault_catalog());
        }
        let fault_catalog = self.fault_catalog.as_mut().unwrap();

        // If needed, generate the fault history
        if !self.fault_history.is_some() {
            self.fault_history = Some(self.obtain_fault_history());
        } 
        let fault_history = self.fault_history.as_mut().unwrap();

        MetropolisEngine {
            rng: rng,
            seed: seed,
            catalog: fault_catalog,
            history: fault_history,
        }
    }
}

/// MetropolisEngine runs a metropolis algorithm
struct MetropolisEngine {
    /// Random number generator for seed generation
    pub rng: PecosRng,
    /// The seed used to initialize the RNG
    pub seed: u64,
    /// The fault catalog used in the Metropolis algorithm
    pub catalog: FaultCatalog,
    /// The fault history used in the Metropolis algorithm
    pub history: FaultHistory,
    /// The flipper used to propose new fault histories
    pub flipper: FaultFlipper,
    /// TODO Implement this checker
    /// Checks whether a fault is an "allowed" move.
    /// For our purposes, this should only return true when
    /// the proposed fault history is a logical failure.
    // pub checker: FaultChecker,
}

impl MetropolisEngine {
    /// TODO: Implement a builder() function that returns a 
    /// (not-yet implemented) MetropolisEngineBuilder.
    /// # Example Usage
    ///
    /// ```
    /// // Create a Monte Carlo engine with default settings
    /// let classical_engine = Box::new(ExternalClassicalEngine::new());
    /// let mut engine = MetropolisEngine::builder()
    ///     .build();
    /// ```
    // #[must_use]
    // pub fn builder() -> MetropolisEngineBuilder {
    //     MetropolisEngineBuilder::new()
    // }


    pub fn new_from_catalog_and_history(
        seed: u64,
        catalog: DepolarizingFaultCatalog,
        history: DepolarizingFaultHistory
    ) -> Self {
        Self::new(seed, catalog, history)
    }

    /// Set a specific seed for the random number generator.
    ///
    /// Setting a seed ensures deterministic behavior across runs with the same seed.
    /// This method sets the seed for:
    /// - The internal `PecosRng` used for shot distribution
    /// - The template `HybridEngine` (which sets seeds for the noise model and quantum engine)
    ///
    /// # Arguments
    /// * `seed` - The seed value for the random number generators
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = PecosRng::seed_from_u64(seed);
    }

    pub fn step(&mut self) {
        let (proposal, ratio) = self.flipper.random_flip_and_ratio(&self.history.current());
        // TODO: Implement checker logic here to determine if the proposed move is allowed.
        


        let result = self.history.step(self.history.current().clone(), proposal.clone(), ratio);
        if result.accepted {
            let probability = self.catalog.fault_history_probability(&self.history.current());
            println!("Proposal ratio: {}", ratio);
            println!("Hastings correction: {}", correction);
            println!("new state probability: {:e}", probability);
        }
    }

    // Run a number of Metropolis steps, given a fault catalog
    pub fn run(&mut self, nstep: usize) -> ? {
        for i in 0..nstep {
            self.step();
        }
    }


for _i in 1..1_000 {
        let (proposal, correction) = fault_catalog.random_flip_hastings_correction(&current);
        let ratio: f64 = correction * fault_catalog.fault_histories_probability_ratio(&proposal, &current);
        let result = stepper.step(current.clone(), proposal.clone(), ratio);
        current = result.state;
        if result.accepted {
            let probability = fault_catalog.fault_history_probability(&current);
            println!("Proposal ratio: {}", ratio);
            println!("Hastings correction: {}", correction);
            println!("new state probability: {:e}", probability);
        }
    }
    }


}

// TODO: Implement a parallel metropolis engine for running multiple Metropolis chains concurrently.