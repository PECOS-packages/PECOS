use crate::Engine;
pub use crate::classical::{ClassicalControlEngine, ClassicalEngine};
pub use crate::hybrid::HybridEngine;
pub use crate::hybrid::HybridEngineBuilder;
pub use crate::monte_carlo::MonteCarloEngine;
pub use crate::monte_carlo::MonteCarloEngineBuilder;
pub use crate::quantum::QuantumEngine;
pub use crate::quantum_system::QuantumSystem;
use dyn_clone::DynClone;
use pecos_core::errors::PecosError;

/// A control engine that orchestrates execution flow with another engine
///
/// Control engines manage complex workflows by:
/// - Breaking down input into smaller pieces for processing
/// - Sending those pieces to another engine
/// - Handling results from that engine
/// - Determining when processing is complete
///
/// # Type Parameters
/// * `Input` - Type received as input
/// * `Output` - Type returned as final output
/// * `EngineInput` - Type sent to the controlled engine
/// * `EngineOutput` - Type received from the controlled engine
pub trait ControlEngine: DynClone + Send + Sync {
    type Input;
    type Output;
    type EngineInput;
    type EngineOutput;

    /// Start processing new input. Returns `NeedsProcessing` or `Complete`.
    ///
    /// # Errors
    /// Returns `PecosError` if processing cannot be started.
    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError>;

    /// Continue processing with result from controlled engine.
    ///
    /// # Errors
    /// Returns `PecosError` if processing cannot continue.
    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError>;

    /// Reset engine state for reuse between simulation runs.
    ///
    /// # Errors
    /// Returns `PecosError` if the reset fails.
    fn reset(&mut self) -> Result<(), PecosError>;
}

/// Represents the stage of processing in a control engine
///
/// Control engines orchestrate the execution flow by:
/// 1. Taking input and determining what needs to be sent to a controlled engine
/// 2. Processing results from the controlled engine
/// 3. Deciding whether to continue processing or complete
#[derive(Debug)]
pub enum EngineStage<I, O> {
    /// Indicates more processing is needed by sending input to controlled engine
    NeedsProcessing(I),
    /// Processing is complete with final output
    Complete(O),
}

/// A system that combines a controller and a controlled engine
///
/// This trait extends Engine with additional capabilities, representing
/// a complete engine system that consists of:
/// 1. A controller component that manages the execution flow
/// 2. A controlled engine component that performs the actual processing
///
/// Any type implementing `EngineSystem` must also implement Engine.
pub trait EngineSystem: Engine {
    /// The type of the controller component
    type Controller: ControlEngine<
            Input = Self::Input,
            Output = Self::Output,
            EngineInput = Self::EngineInput,
            EngineOutput = Self::EngineOutput,
        >;

    /// The type of the controlled engine component
    type ControlledEngine: Engine<Input = Self::EngineInput, Output = Self::EngineOutput>;

    /// The input type for the controlled engine
    type EngineInput;

    /// The output type from the controlled engine
    type EngineOutput;

    /// Get a reference to the controller component
    fn controller(&self) -> &Self::Controller;

    /// Get a mutable reference to the controller component
    fn controller_mut(&mut self) -> &mut Self::Controller;

    /// Get a reference to the controlled engine component
    fn engine(&self) -> &Self::ControlledEngine;

    /// Get a mutable reference to the controlled engine component
    fn engine_mut(&mut self) -> &mut Self::ControlledEngine;

    /// Process an input using the system's controller and engine components
    ///
    /// This method implements the complete execution flow:
    /// 1. Start processing with the controller
    /// 2. In a loop:
    ///    a. If more processing is needed, send input to the controlled engine
    ///    b. Pass the engine's output back to the controller
    ///    c. Continue until the controller indicates processing is complete
    ///
    /// # Parameters
    /// * `input` - The input to process
    ///
    /// # Returns
    /// * The final output of processing
    ///
    /// # Errors
    /// This function may return an error if:
    /// - Resetting the quantum or classical engine fails.
    /// - Generating commands through the classical engine fails.
    /// - Processing commands through the quantum engine fails.
    /// - Handling measurements through the classical engine fails.
    fn process_as_system(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        let mut stage = self.controller_mut().start(input)?;

        loop {
            match stage {
                EngineStage::NeedsProcessing(engine_input) => {
                    let engine_output = self.engine_mut().process(engine_input)?;
                    stage = self.controller_mut().continue_processing(engine_output)?;
                }
                EngineStage::Complete(output) => {
                    return Ok(output);
                }
            }
        }
    }
}

// Register the Engine trait with dyn_clone
dyn_clone::clone_trait_object!(<I, O> Engine<Input=I, Output=O>);

// Register the ControlEngine trait with dyn_clone
dyn_clone::clone_trait_object!(<I, O, EI, EO> ControlEngine<Input=I, Output=O, EngineInput=EI, EngineOutput=EO>);

// Implement Engine for Box<dyn Engine> to allow using it directly
// in EngineSystem implementations
impl<I, O> Engine for Box<dyn Engine<Input = I, Output = O>> {
    type Input = I;
    type Output = O;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        (**self).process(input)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        (**self).reset()
    }
}

// Implement ControlEngine for Box<dyn ControlEngine> to allow using it directly
// in EngineSystem implementations
impl<I, O, EI, EO> ControlEngine
    for Box<dyn ControlEngine<Input = I, Output = O, EngineInput = EI, EngineOutput = EO>>
{
    type Input = I;
    type Output = O;
    type EngineInput = EI;
    type EngineOutput = EO;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        (**self).start(input)
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        (**self).continue_processing(result)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        (**self).reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // A simple test engine that just returns its input
    #[derive(Clone)]
    struct EchoEngine {
        calls: Arc<AtomicUsize>,
    }

    impl EchoEngine {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Engine for EchoEngine {
        type Input = u32;
        type Output = u32;

        fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(input)
        }

        fn reset(&mut self) -> Result<(), PecosError> {
            self.calls.store(0, Ordering::SeqCst);
            Ok(())
        }
    }

    // A controller that will require a configurable number of iterations
    #[derive(Clone)]
    struct IterativeController {
        target_iterations: usize,
        current_iteration: usize,
        calls: Arc<AtomicUsize>,
    }

    impl IterativeController {
        fn new(target_iterations: usize) -> Self {
            Self {
                target_iterations,
                current_iteration: 0,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl ControlEngine for IterativeController {
        type Input = u32;
        type Output = u32;
        type EngineInput = u32;
        type EngineOutput = u32;

        fn start(
            &mut self,
            input: Self::Input,
        ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
            // Reset counters on start
            self.current_iteration = 0;
            self.calls.fetch_add(1, Ordering::SeqCst);

            // Return NeedsProcessing to start the loop
            Ok(EngineStage::NeedsProcessing(input))
        }

        fn continue_processing(
            &mut self,
            result: Self::EngineOutput,
        ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
            self.current_iteration += 1;
            self.calls.fetch_add(1, Ordering::SeqCst);

            // If we've reached our target iterations, return the result
            if self.current_iteration >= self.target_iterations {
                Ok(EngineStage::Complete(result))
            } else {
                // Otherwise, request another round of processing
                Ok(EngineStage::NeedsProcessing(result))
            }
        }

        fn reset(&mut self) -> Result<(), PecosError> {
            self.current_iteration = 0;
            self.calls.store(0, Ordering::SeqCst);
            Ok(())
        }
    }

    // A system combining our iterative controller and echo engine
    #[derive(Clone)]
    struct TestSystem {
        controller: IterativeController,
        engine: EchoEngine,
    }

    impl TestSystem {
        fn new(target_iterations: usize) -> Self {
            Self {
                controller: IterativeController::new(target_iterations),
                engine: EchoEngine::new(),
            }
        }
    }

    impl Engine for TestSystem {
        type Input = u32;
        type Output = u32;

        fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
            self.process_as_system(input)
        }

        fn reset(&mut self) -> Result<(), PecosError> {
            self.controller.reset()?;
            self.engine.reset()
        }
    }

    impl EngineSystem for TestSystem {
        type Controller = IterativeController;
        type ControlledEngine = EchoEngine;
        type EngineInput = u32;
        type EngineOutput = u32;

        fn controller(&self) -> &Self::Controller {
            &self.controller
        }

        fn controller_mut(&mut self) -> &mut Self::Controller {
            &mut self.controller
        }

        fn engine(&self) -> &Self::ControlledEngine {
            &self.engine
        }

        fn engine_mut(&mut self) -> &mut Self::ControlledEngine {
            &mut self.engine
        }
    }

    #[test]
    fn test_engine_system_looping() {
        // Create a system that should loop 3 times
        let mut system = TestSystem::new(3);

        // Process an input
        let result = system.process(42).unwrap();

        // Verify the result is still 42
        assert_eq!(result, 42);

        // Verify controller was called 4 times (start + 3 continue_processing)
        assert_eq!(system.controller.calls.load(Ordering::SeqCst), 4);

        // Verify engine was called 3 times
        assert_eq!(system.engine.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_no_loops() {
        // Create a system that should process immediately (0 iterations)
        let mut system = TestSystem::new(0);

        // Process an input
        let result = system.process(42).unwrap();

        // Verify the result is still 42
        assert_eq!(result, 42);

        // Verify controller was called 2 times (start + 1 continue_processing)
        assert_eq!(system.controller.calls.load(Ordering::SeqCst), 2);

        // Verify engine was called 1 time
        assert_eq!(system.engine.calls.load(Ordering::SeqCst), 1);
    }
}
