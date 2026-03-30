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

    /// Generate a `ByteMessage` containing the next batch of quantum commands to execute
    ///
    /// # Returns
    ///
    /// Returns a `ByteMessage` containing the quantum commands to execute if successful.
    /// An empty message indicates no more commands are available.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - Operation error: If the program processing fails or encounters unsupported operations.
    /// - Lock error: If a lock cannot be acquired during the execution process.
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError>;

    /// Handles a `ByteMessage` containing measurements from the quantum engine
    ///
    /// # Parameters
    ///
    /// - `message`: A `ByteMessage` containing the measurement data to process.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - Operation error: If the measurement processing fails.
    /// - Lock error: If a lock cannot be acquired during the measurement handling process.
    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError>;

    /// Retrieves the results of the execution process after all measurements are handled.
    ///
    /// # Returns
    ///
    /// Returns a `Shot` containing the measurements and results generated
    /// during the execution process.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - Operation error: If result retrieval fails or is unsupported.
    /// - Lock error: If a lock cannot be acquired to access required resources.
    fn get_results(&self) -> Result<Shot, PecosError>;

    /// Sets a specific seed for the classical engine
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    fn set_seed(&mut self, _seed: u64) {
        // Default implementation does nothing
    }

    /// Compiles the classical program into an intermediate representation or directly
    /// into commands that can be executed by the engine.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the compilation is successful, or an `Err` containing
    /// a boxed error if the compilation fails.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - `Box<dyn std::error::Error>`: If there is a compilation error due to syntax issues,
    ///   unsupported features, or internal errors in the engine's implementation.
    fn compile(&self) -> Result<(), PecosError>;

    /// Resets the state of the classical engine to its initial configuration.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the reset operation completes successfully.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - Operation error: If the reset operation encounters unsupported actions or fails.
    /// - Lock error: If a lock cannot be acquired during the reset process.
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

/// A trait that combines `ClassicalEngine` with `ControlEngine` for use in `HybridEngine`
///
/// This trait ensures that engines used by `HybridEngine` implement both the
/// `ClassicalEngine` interface (for quantum command generation and measurement handling)
/// and the `ControlEngine` interface (for orchestrating the execution flow).
///
/// # Important
///
/// **Both traits must be explicitly implemented** by any engine that wants to be used
/// with `HybridEngine`. There is no default implementation because control flow is
/// highly specific to each engine type:
///
/// - Some engines may need to batch operations (like `PhirEngine`)
/// - Some engines may need to finalize state after measurements (like `PhirEngine`'s exports)
/// - Some engines may process everything in one shot (like `QasmEngine`)
///
/// # Example Implementation Pattern
///
/// ```rust
/// use pecos_engines::{
///     ClassicalEngine, ControlEngine, Engine, EngineStage,
///     ByteMessage, ByteMessageBuilder, Shot
/// };
/// use pecos_core::errors::PecosError;
/// use std::any::Any;
///
/// // Example engine implementation
/// #[derive(Clone)]
/// struct MyEngine {
///     num_qubits: usize,
///     commands_generated: bool,
///     shot_result: Shot,
/// }
///
/// impl MyEngine {
///     fn new(num_qubits: usize) -> Self {
///         Self {
///             num_qubits,
///             commands_generated: false,
///             shot_result: Shot::default(),
///         }
///     }
/// }
///
/// // First implement the base Engine trait
/// impl Engine for MyEngine {
///     type Input = ();
///     type Output = Shot;
///
///     fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
///         // Process a single shot
///         Ok(self.shot_result.clone())
///     }
///
///     fn reset(&mut self) -> Result<(), PecosError> {
///         // Reset engine state
///         self.commands_generated = false;
///         self.shot_result = Shot::default();
///         Ok(())
///     }
/// }
///
/// // Then implement ClassicalEngine for quantum-specific functionality
/// impl ClassicalEngine for MyEngine {
///     fn num_qubits(&self) -> usize {
///         self.num_qubits
///     }
///
///     fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
///         let mut builder = ByteMessageBuilder::new();
///         builder.for_quantum_operations();
///
///         // Generate commands only once in this example
///         if !self.commands_generated {
///             // Add quantum operations (e.g., H gate on qubit 0)
///             builder.h(&[0]);
///             self.commands_generated = true;
///         }
///
///         Ok(builder.build())
///     }
///
///     fn handle_measurements(&mut self, msg: ByteMessage) -> Result<(), PecosError> {
///         // Process measurement results from quantum engine
///         // In a real implementation, you would parse the message
///         // and update internal state accordingly
///         Ok(())
///     }
///
///     fn get_results(&self) -> Result<Shot, PecosError> {
///         Ok(self.shot_result.clone())
///     }
///
///     fn compile(&self) -> Result<(), PecosError> {
///         // Perform any necessary compilation/validation
///         Ok(())
///     }
///
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
///
///     fn as_any_mut(&mut self) -> &mut dyn Any {
///         self
///     }
/// }
///
/// // Finally implement ControlEngine for execution flow control
/// impl ControlEngine for MyEngine {
///     type Input = ();
///     type Output = Shot;
///     type EngineInput = ByteMessage;
///     type EngineOutput = ByteMessage;
///
///     fn start(&mut self, _: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
///         // Generate initial quantum commands
///         let commands = self.generate_commands()?;
///
///         if commands.is_empty()? {
///             // No commands to execute, return results
///             Ok(EngineStage::Complete(self.get_results()?))
///         } else {
///             // Send commands to quantum engine
///             Ok(EngineStage::NeedsProcessing(commands))
///         }
///     }
///
///     fn continue_processing(&mut self, measurements: ByteMessage)
///         -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
///         // Handle measurements from quantum engine
///         self.handle_measurements(measurements)?;
///
///         // Check if there are more commands to execute
///         let commands = self.generate_commands()?;
///
///         if commands.is_empty()? {
///             // All done, return final results
///             Ok(EngineStage::Complete(self.get_results()?))
///         } else {
///             // More commands to execute
///             Ok(EngineStage::NeedsProcessing(commands))
///         }
///     }
///
///     fn reset(&mut self) -> Result<(), PecosError> {
///         // Reset control engine state
///         self.commands_generated = false;
///         self.shot_result = Shot::default();
///         Ok(())
///     }
/// }
///
/// // Verify the implementation
/// let mut engine = MyEngine::new(2);
/// assert_eq!(engine.num_qubits(), 2);
///
/// // Test compilation
/// engine.compile().unwrap();
///
/// // Test command generation
/// let commands = engine.generate_commands().unwrap();
/// assert!(!commands.is_empty().unwrap());
///
/// // Second call returns empty (no more commands)
/// let commands = engine.generate_commands().unwrap();
/// assert!(commands.is_empty().unwrap());
/// ```
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
