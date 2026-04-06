use crate::Engine;
use crate::byte_message::ByteMessage;
use crate::engine_system::{ControlEngine, EngineStage};
use crate::shot_results::Shot;
use dyn_clone::DynClone;
use pecos_core::errors::PecosError;
use std::any::Any;

/// Classical engine that processes programs and handles measurements
pub trait ClassicalEngine: Engine<Input = (), Output = Shot> + DynClone + Send + Sync {
    fn num_qubits(&self) -> usize;

    /// Generate a `ByteMessage` containing the next batch of quantum commands to execute.
    /// An empty message indicates no more commands are available.
    ///
    /// # Errors
    /// Returns `PecosError` if program processing fails.
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError>;

    /// Handles a `ByteMessage` containing measurements from the quantum engine.
    ///
    /// # Errors
    /// Returns `PecosError` if measurement processing fails.
    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError>;

    /// Retrieves the results of the execution process after all measurements are handled.
    ///
    /// # Errors
    /// Returns `PecosError` if result retrieval fails.
    fn get_results(&self) -> Result<Shot, PecosError>;

    /// Sets a specific seed for the classical engine.
    fn set_seed(&mut self, _seed: u64) {
        // Default implementation does nothing
    }

    /// Compiles the classical program into an intermediate representation or directly
    /// into commands that can be executed by the engine.
    ///
    /// # Errors
    /// Returns `PecosError` if compilation fails.
    fn compile(&self) -> Result<(), PecosError>;

    /// Resets the state of the classical engine to its initial configuration.
    ///
    /// # Errors
    /// Returns `PecosError` if the reset operation fails.
    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }

    /// Returns a reference to self as Any
    ///
    /// This allows for type-checking and downcasting without requiring
    /// experimental trait upcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable reference to self as Any
    ///
    /// This allows for type-checking and downcasting without requiring
    /// experimental trait upcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// Register the ClassicalEngine trait with dyn_clone
dyn_clone::clone_trait_object!(ClassicalEngine);

/// Combines `ClassicalEngine` with `ControlEngine` for use in `HybridEngine`.
///
/// Both traits must be explicitly implemented -- there is no default implementation
/// because control flow is highly specific to each engine type (e.g., batching,
/// finalization, single-shot processing).
///
/// See `PhirEngine`, `QasmEngine`, and `QisEngine` for concrete examples.
pub trait ClassicalControlEngine: ClassicalEngine
    + ControlEngine<Input = (), Output = Shot, EngineInput = ByteMessage, EngineOutput = ByteMessage>
{
}

// Blanket implementation for all types that implement both traits
impl<T> ClassicalControlEngine for T where
    T: ClassicalEngine
        + ControlEngine<
            Input = (),
            Output = Shot,
            EngineInput = ByteMessage,
            EngineOutput = ByteMessage,
        >
{
}

// Register the combined trait with dyn_clone
dyn_clone::clone_trait_object!(ClassicalControlEngine);

// Implement ClassicalEngine for Box<dyn ClassicalControlEngine> to enable trait object usage
impl ClassicalEngine for Box<dyn ClassicalControlEngine> {
    fn num_qubits(&self) -> usize {
        (**self).num_qubits()
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        (**self).generate_commands()
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        (**self).handle_measurements(message)
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        (**self).get_results()
    }

    fn set_seed(&mut self, seed: u64) {
        (**self).set_seed(seed);
    }

    fn compile(&self) -> Result<(), PecosError> {
        (**self).compile()
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        ClassicalEngine::reset(&mut **self)
    }

    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        (**self).as_any_mut()
    }
}

// Implement ControlEngine for Box<dyn ClassicalControlEngine> to enable trait object usage
impl ControlEngine for Box<dyn ClassicalControlEngine> {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        (**self).start(input)
    }

    fn continue_processing(
        &mut self,
        result: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        (**self).continue_processing(result)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        <dyn ControlEngine<
                Input = (),
                Output = Shot,
                EngineInput = ByteMessage,
                EngineOutput = ByteMessage,
            >>::reset(&mut **self)
    }
}

// Implement Engine for Box<dyn ClassicalControlEngine>
impl Engine for Box<dyn ClassicalControlEngine> {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        let mut stage = self.start(input)?;

        loop {
            match stage {
                EngineStage::NeedsProcessing(_engine_input) => {
                    // In a real system, this would process through a quantum engine
                    // For now, we'll just return an empty message
                    let engine_output = ByteMessage::builder().build();
                    stage = self.continue_processing(engine_output)?;
                }
                EngineStage::Complete(output) => return Ok(output),
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Use fully qualified path to disambiguate
        ClassicalEngine::reset(&mut **self)
    }
}
