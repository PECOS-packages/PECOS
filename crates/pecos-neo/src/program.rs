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
//! ```ignore
//! use pecos_neo::program::{CommandSource, ProgramRunner};
//! use pecos_neo::prelude::*;
//! use pecos_qsim::SparseStab;
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
//!             .prep(0)
//!             .h(0)
//!             .measure(0)
//!             .build())
//!     }
//!
//!     fn is_complete(&self) -> bool {
//!         self.succeeded || self.current_attempt >= self.max_attempts
//!     }
//! }
//! ```

use crate::command::CommandQueue;
use crate::noise::ComposableNoiseModel;
use crate::outcome::MeasurementOutcomes;
use crate::runner::ShotRunner;
use pecos_qsim::CliffordGateable;

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
/// The `ProgramRunner` executes programs that implement `CommandSource`,
/// handling the back-and-forth between classical control and quantum execution.
pub struct ProgramRunner<S: CliffordGateable> {
    runner: ShotRunner<S>,
}

impl<S: CliffordGateable> ProgramRunner<S> {
    /// Create a new program runner with the given simulator.
    pub fn new(simulator: S) -> Self {
        Self {
            runner: ShotRunner::new(simulator),
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

    /// Execute a single shot of the program.
    ///
    /// Runs the program until `CommandSource::is_complete()` returns true
    /// or `next_commands()` returns `None`.
    ///
    /// The simulator is reset to |0⟩^n at the start of each shot, ensuring
    /// clean state for programs that don't explicitly prepare qubits.
    pub fn run_shot<P: CommandSource + ?Sized>(&mut self, program: &mut P) -> ProgramResult {
        program.reset();

        // Reset simulator to |0⟩^n state at the start of each shot
        // This is important for programs (like QASM) that assume qubits start in |0>
        self.runner.simulator_mut().reset();

        let mut all_outcomes = MeasurementOutcomes::new();
        let mut num_batches = 0;
        let mut last_outcomes: Option<MeasurementOutcomes> = None;

        loop {
            // Get next batch of commands
            let commands = program.next_commands(last_outcomes.as_ref());

            match commands {
                Some(cmds) if !cmds.is_empty() => {
                    // Execute this batch (without resetting - state carries over between batches)
                    let outcomes = self.runner.run_shot(&cmds);
                    num_batches += 1;

                    // Merge outcomes into total
                    for outcome in outcomes.iter() {
                        all_outcomes.record(*outcome);
                    }

                    last_outcomes = Some(outcomes);
                }
                _ => {
                    // Program complete
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

    /// Get a reference to the underlying shot runner.
    #[must_use]
    pub fn shot_runner(&self) -> &ShotRunner<S> {
        &self.runner
    }

    /// Get a mutable reference to the underlying shot runner.
    pub fn shot_runner_mut(&mut self) -> &mut ShotRunner<S> {
        &mut self.runner
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
    use pecos_qsim::SparseStab;

    #[test]
    fn test_static_program() {
        let commands = CommandBuilder::new().prep(0).h(0).measure(0).build();

        let mut program = StaticProgram::new(commands, 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

        let result = runner.run_shot(&mut program);

        assert_eq!(result.num_batches, 1);
        assert_eq!(result.outcomes.len(), 1);
    }

    #[test]
    fn test_repeated_program() {
        // Simulate QEC: prep, measure syndrome, repeat
        let round_commands = CommandBuilder::new().prep(0).h(0).measure(0).build();

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
        let initial = CommandBuilder::new().prep(0).h(0).measure(0).build();

        // Branch: if measured 1, apply X correction
        let branch = |outcomes: &MeasurementOutcomes| {
            if outcomes.get_bit(QubitId(0)) == Some(true) {
                Some(CommandBuilder::new().x(0).measure(0).build())
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
        let commands = CommandBuilder::new().prep(0).measure(0).build();
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
            .prep(0)
            .prep(1)
            .h(0)
            .cx(0, 1)
            .measure(0)
            .measure(1)
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
