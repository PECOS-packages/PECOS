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

//! Program execution interfaces for classical-quantum hybrid programs.
//!
//! This module provides traits and types for executing programs with classical
//! control flow, enabling mid-circuit measurement and feedback.
//!
//! ## Design Philosophy
//!
//! Instead of the `ControlEngine` pattern (`start`/`continue_processing`) from pecos-engines,
//! we use a simpler functional approach:
//!
//! 1. **`CommandSource`**: Generates batches of quantum commands
//! 2. **Outcome-driven**: Each batch is executed, outcomes fed back for next batch
//! 3. **Pure functions where possible**: Branch decisions as functions of outcomes
//!
//! This maps better to the DOD philosophy where data flows through transformations.
//!
//! ## Example
//!
//! ```no_run
//! use pecos_neo::program::{CommandSource, ProgramRunner};
//! use pecos_neo::prelude::*;
//! use pecos_simulators::SparseStab;
//!
//! // A simple repeat-until-success program
//! struct RepeatUntilSuccess {
//!     max_attempts: usize,
//!     current_attempt: usize,
//!     succeeded: bool,
//! }
//!
//! impl CommandSource for RepeatUntilSuccess {
//!     fn next_commands(&mut self, outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
//!         // Check if previous attempt succeeded
//!         if let Some(outcomes) = outcomes {
//!             if outcomes.get_bit(QubitId(0)) == Some(true) {
//!                 self.succeeded = true;
//!                 return None; // Done!
//!             }
//!         }
//!
//!         if self.current_attempt >= self.max_attempts {
//!             return None; // Give up
//!         }
//!
//!         self.current_attempt += 1;
//!
//!         // Try again: prep, rotate, measure
//!         Some(CommandBuilder::new()
//!             .pz(&[0])
//!             .h(&[0])
//!             .mz(&[0])
//!             .build())
//!     }
//!
//!     fn is_complete(&self) -> bool {
//!         self.succeeded || self.current_attempt >= self.max_attempts
//!     }
//!
//!     fn reset(&mut self) {
//!         self.current_attempt = 0;
//!         self.succeeded = false;
//!     }
//!
//!     fn num_qubits(&self) -> usize { 1 }
//! }
//! ```

use crate::command::CommandQueue;
use crate::extensible::GateDefinitions;
use crate::noise::ComposableNoiseModel;
use crate::outcome::MeasurementOutcomes;
use crate::runner::{CircuitRunner, EventHandlers, GateOverrides};
use pecos_core::rng::RngManageable;
use pecos_random::PecosRng;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};

/// A source of quantum commands for program execution.
///
/// This trait represents the classical control side of a hybrid program.
/// Implementations generate batches of quantum commands based on
/// measurement outcomes from previous batches.
pub trait CommandSource {
    /// Generate the next batch of commands.
    ///
    /// # Arguments
    /// * `outcomes` - Measurement outcomes from the previous batch, or `None` for the first batch
    ///
    /// # Returns
    /// * `Some(commands)` - The next batch of commands to execute
    /// * `None` - The program is complete
    fn next_commands(&mut self, outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue>;

    /// Check if the program is complete.
    fn is_complete(&self) -> bool;

    /// Reset the program state for a new shot.
    fn reset(&mut self);

    /// Get the number of qubits required.
    fn num_qubits(&self) -> usize;
}

/// Result of a single program execution (shot).
#[derive(Debug, Clone)]
pub struct ProgramResult {
    /// All measurement outcomes collected during execution.
    pub outcomes: MeasurementOutcomes,
    /// Number of command batches executed.
    pub num_batches: usize,
}

/// Runs hybrid programs with classical-quantum feedback.
///
/// The `ProgramRunner` owns both a [`CircuitRunner`] and a simulator, providing
/// a stateful wrapper for program execution. This is the higher-level
/// interface that manages simulator lifecycle.
pub struct ProgramRunner<S: CliffordGateable> {
    runner: CircuitRunner<S>,
    simulator: S,
}

impl<S: CliffordGateable> ProgramRunner<S> {
    /// Create a new program runner with the given simulator.
    pub fn new(simulator: S) -> Self {
        Self {
            runner: CircuitRunner::new(),
            simulator,
        }
    }

    /// Create a new program runner with explicit gate definitions.
    pub fn with_definitions(simulator: S, definitions: GateDefinitions) -> Self {
        Self {
            runner: CircuitRunner::with_definitions(definitions),
            simulator,
        }
    }

    /// Set the noise model.
    #[must_use]
    pub fn with_noise(mut self, noise: ComposableNoiseModel) -> Self {
        self.runner = self.runner.with_noise(noise);
        self
    }

    /// Set the RNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.runner = self.runner.with_seed(seed);
        self
    }

    /// Set custom gate overrides.
    #[must_use]
    pub fn with_overrides(mut self, overrides: GateOverrides<S>) -> Self {
        self.runner = self.runner.with_overrides(overrides);
        self
    }

    /// Execute a single shot of the program.
    ///
    /// Runs the program until `CommandSource::is_complete()` returns true
    /// or `next_commands()` returns `None`.
    ///
    /// The simulator is reset to |0>^n at the start of each shot, ensuring
    /// clean state for programs that don't explicitly prepare qubits.
    pub fn run_shot<P: CommandSource + ?Sized>(&mut self, program: &mut P) -> ProgramResult {
        program.reset();

        // Reset simulator to |0>^n state at the start of each shot
        self.simulator.reset();

        let mut all_outcomes = MeasurementOutcomes::new();
        let mut num_batches = 0;
        let mut last_outcomes: Option<MeasurementOutcomes> = None;

        loop {
            let commands = program.next_commands(last_outcomes.as_ref());

            match commands {
                Some(cmds) if !cmds.is_empty() => {
                    let outcomes = self
                        .runner
                        .apply_circuit(&mut self.simulator, &cmds)
                        .expect("core gates should not fail");
                    num_batches += 1;

                    for outcome in outcomes.iter() {
                        all_outcomes.record(*outcome);
                    }

                    last_outcomes = Some(outcomes);
                }
                _ => {
                    break;
                }
            }

            if program.is_complete() {
                break;
            }
        }

        ProgramResult {
            outcomes: all_outcomes,
            num_batches,
        }
    }

    /// Get a reference to the underlying circuit runner.
    #[must_use]
    pub fn circuit_runner(&self) -> &CircuitRunner<S> {
        &self.runner
    }

    /// Get a mutable reference to the underlying circuit runner.
    pub fn circuit_runner_mut(&mut self) -> &mut CircuitRunner<S> {
        &mut self.runner
    }

    /// Get a reference to the simulator.
    #[must_use]
    pub fn simulator(&self) -> &S {
        &self.simulator
    }

    /// Get a mutable reference to the simulator.
    pub fn simulator_mut(&mut self) -> &mut S {
        &mut self.simulator
    }

    /// Set maximum decomposition depth for gate resolution.
    #[must_use]
    pub fn with_max_decomp_depth(mut self, depth: usize) -> Self {
        self.runner = self.runner.with_max_decomp_depth(depth);
        self
    }

    /// Merge an [`EventHandlers`] collection into the underlying runner.
    #[must_use]
    pub fn with_event_handlers(mut self, handlers: EventHandlers) -> Self {
        self.runner = self.runner.with_event_handlers(handlers);
        self
    }
}

impl<S> ProgramRunner<S>
where
    S: CliffordGateable + RngManageable<Rng = PecosRng>,
{
    /// Set the full seed for deterministic execution.
    ///
    /// Seeds both the noise RNG and the simulator's internal RNG using
    /// derived seeds from a single base seed.
    pub fn set_full_seed(&mut self, seed: u64) {
        self.runner.set_full_seed(&mut self.simulator, seed);
    }
}

impl<S: CliffordGateable + ArbitraryRotationGateable> ProgramRunner<S> {
    /// Create a program runner with rotation gate support and default definitions.
    ///
    /// For simulators implementing `ArbitraryRotationGateable`, this constructor
    /// enables native execution of rotation gates (T, Tdg, RX, RY, RZ, etc.).
    pub fn rotations(simulator: S) -> Self {
        Self {
            runner: CircuitRunner::rotations(),
            simulator,
        }
    }

    /// Create a program runner with rotation gate support and explicit definitions.
    pub fn rotations_with_definitions(simulator: S, definitions: GateDefinitions) -> Self {
        Self {
            runner: CircuitRunner::rotations_with_definitions(definitions),
            simulator,
        }
    }
}

// ============================================================================
// DynProgramRunner (type-erased program execution)
// ============================================================================

/// Type-erased program runner for dynamic dispatch.
///
/// This trait abstracts `ProgramRunner<S>` operations, allowing custom simulator
/// backends to be used through the high-level `sim_neo()` API without knowing
/// the concrete simulator type at compile time.
pub trait DynProgramRunner: Send + Sync {
    /// Execute a single shot of the program.
    fn run_shot(&mut self, source: &mut dyn CommandSource) -> ProgramResult;

    /// Set the full seed for deterministic execution.
    fn set_full_seed(&mut self, seed: u64);
}

impl<S> DynProgramRunner for ProgramRunner<S>
where
    S: CliffordGateable + RngManageable<Rng = PecosRng> + Send + Sync,
{
    fn run_shot(&mut self, source: &mut dyn CommandSource) -> ProgramResult {
        ProgramRunner::run_shot(self, source)
    }

    fn set_full_seed(&mut self, seed: u64) {
        self.runner.set_full_seed(&mut self.simulator, seed);
    }
}

// ============================================================================
// Simple Program Implementations
// ============================================================================

/// A static program that executes a single batch of commands.
///
/// This is the simplest case - no classical feedback, just run the circuit once.
#[derive(Debug, Clone)]
pub struct StaticProgram {
    commands: CommandQueue,
    executed: bool,
    num_qubits: usize,
}

impl StaticProgram {
    /// Create a new static program from a command queue.
    #[must_use]
    pub fn new(commands: CommandQueue, num_qubits: usize) -> Self {
        Self {
            commands,
            executed: false,
            num_qubits,
        }
    }
}

impl CommandSource for StaticProgram {
    fn next_commands(&mut self, _outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        if self.executed {
            None
        } else {
            self.executed = true;
            Some(self.commands.clone())
        }
    }

    fn is_complete(&self) -> bool {
        self.executed
    }

    fn reset(&mut self) {
        self.executed = false;
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// A program that repeats a circuit for multiple rounds (e.g., QEC syndrome extraction).
#[derive(Debug, Clone)]
pub struct RepeatedProgram {
    /// Commands for each round.
    round_commands: CommandQueue,
    /// Total number of rounds.
    num_rounds: usize,
    /// Current round (0-indexed).
    current_round: usize,
    /// Number of qubits.
    num_qubits: usize,
}

impl RepeatedProgram {
    /// Create a new repeated program.
    #[must_use]
    pub fn new(round_commands: CommandQueue, num_rounds: usize, num_qubits: usize) -> Self {
        Self {
            round_commands,
            num_rounds,
            current_round: 0,
            num_qubits,
        }
    }
}

impl CommandSource for RepeatedProgram {
    fn next_commands(&mut self, _outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        if self.current_round >= self.num_rounds {
            return None;
        }

        self.current_round += 1;
        Some(self.round_commands.clone())
    }

    fn is_complete(&self) -> bool {
        self.current_round >= self.num_rounds
    }

    fn reset(&mut self) {
        self.current_round = 0;
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// A program with conditional branching based on measurement outcomes.
///
/// Executes an initial circuit, then based on measurement outcomes,
/// chooses which branch to execute next.
#[derive(Debug, Clone)]
pub struct ConditionalProgram<F>
where
    F: Fn(&MeasurementOutcomes) -> Option<CommandQueue>,
{
    /// Initial commands to execute.
    initial_commands: CommandQueue,
    /// Function that decides what to execute next based on outcomes.
    branch_fn: F,
    /// Current state.
    state: ConditionalState,
    /// Number of qubits.
    num_qubits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditionalState {
    Initial,
    Branching,
    Complete,
}

impl<F> ConditionalProgram<F>
where
    F: Fn(&MeasurementOutcomes) -> Option<CommandQueue>,
{
    /// Create a new conditional program.
    pub fn new(initial_commands: CommandQueue, branch_fn: F, num_qubits: usize) -> Self {
        Self {
            initial_commands,
            branch_fn,
            state: ConditionalState::Initial,
            num_qubits,
        }
    }
}

impl<F> CommandSource for ConditionalProgram<F>
where
    F: Fn(&MeasurementOutcomes) -> Option<CommandQueue>,
{
    fn next_commands(&mut self, outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        match self.state {
            ConditionalState::Initial => {
                self.state = ConditionalState::Branching;
                Some(self.initial_commands.clone())
            }
            ConditionalState::Branching => {
                self.state = ConditionalState::Complete;
                outcomes.and_then(|o| (self.branch_fn)(o))
            }
            ConditionalState::Complete => None,
        }
    }

    fn is_complete(&self) -> bool {
        self.state == ConditionalState::Complete
    }

    fn reset(&mut self) {
        self.state = ConditionalState::Initial;
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use pecos_core::QubitId;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_static_program() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut program = StaticProgram::new(commands, 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

        let result = runner.run_shot(&mut program);

        assert_eq!(result.num_batches, 1);
        assert_eq!(result.outcomes.len(), 1);
    }

    #[test]
    fn test_repeated_program() {
        // Simulate QEC: prep, measure syndrome, repeat
        let round_commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut program = RepeatedProgram::new(round_commands, 3, 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

        let result = runner.run_shot(&mut program);

        assert_eq!(result.num_batches, 3);
        // 3 measurements (one per round)
        assert_eq!(result.outcomes.len(), 3);
    }

    #[test]
    fn test_conditional_program() {
        // Initial: prep and measure
        let initial = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        // Branch: if measured 1, apply X correction
        let branch = |outcomes: &MeasurementOutcomes| {
            if outcomes.get_bit(QubitId(0)) == Some(true) {
                Some(CommandBuilder::new().x(&[0]).mz(&[0]).build())
            } else {
                None
            }
        };

        let mut program = ConditionalProgram::new(initial, branch, 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

        let result = runner.run_shot(&mut program);

        // Either 1 or 2 batches depending on measurement outcome
        assert!(result.num_batches >= 1 && result.num_batches <= 2);
    }

    #[test]
    fn test_program_reset() {
        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();
        let mut program = StaticProgram::new(commands, 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

        // Run first shot
        let result1 = runner.run_shot(&mut program);
        assert_eq!(result1.num_batches, 1);

        // Run second shot (program should reset)
        let result2 = runner.run_shot(&mut program);
        assert_eq!(result2.num_batches, 1);
    }

    #[test]
    fn test_bell_state_program() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut program = StaticProgram::new(commands, 2);
        let mut runner = ProgramRunner::new(SparseStab::new(2)).with_seed(42);

        let result = runner.run_shot(&mut program);

        // Bell state: both measurements should be correlated
        let o0 = result.outcomes.get_bit(QubitId(0));
        let o1 = result.outcomes.get_bit(QubitId(1));

        assert_eq!(o0, o1, "Bell state measurements should be equal");
    }
}
