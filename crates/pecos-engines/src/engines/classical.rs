use crate::byte_message::ByteMessage;
use crate::core::shot_results::ShotResult;
use crate::engines::{ControlEngine, Engine, EngineStage, phir, qir};
use crate::errors::QueueError;
use dyn_clone::DynClone;
use log::debug;
use std::any::Any;
use std::error::Error;
use std::path::Path;

/// Classical engine that processes programs and handles measurements
pub trait ClassicalEngine:
    Engine<Input = (), Output = ShotResult> + DynClone + Send + Sync
{
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
    /// - `QueueError::OperationError`: If the program processing fails or encounters unsupported operations.
    /// - `QueueError::LockError`: If a lock cannot be acquired during the execution process.
    fn generate_commands(&mut self) -> Result<ByteMessage, QueueError>;

    /// Handles a `ByteMessage` containing measurements from the quantum engine
    ///
    /// # Parameters
    ///
    /// - `message`: A `ByteMessage` containing the measurement data to process.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - `QueueError::OperationError`: If the measurement processing fails.
    /// - `QueueError::LockError`: If a lock cannot be acquired during the measurement handling process.
    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), QueueError>;

    /// Retrieves the results of the execution process after all measurements are handled.
    ///
    /// # Returns
    ///
    /// Returns a `ShotResult` containing the measurements and results generated
    /// during the execution process.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - `QueueError::OperationError`: If result retrieval fails or is unsupported.
    /// - `QueueError::LockError`: If a lock cannot be acquired to access required resources.
    fn get_results(&self) -> Result<ShotResult, QueueError>;

    /// Sets a specific seed for the classical engine
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// Returns a `QueueError` if setting the seed fails
    fn set_seed(&mut self, _seed: u64) -> Result<(), QueueError> {
        // Default implementation just succeeds without doing anything
        Ok(())
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
    fn compile(&self) -> Result<(), Box<dyn std::error::Error>>;

    /// Resets the state of the classical engine to its initial configuration.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the reset operation completes successfully.
    ///
    /// # Errors
    ///
    /// This function may return the following errors:
    /// - `QueueError::OperationError`: If the reset operation encounters unsupported actions or fails.
    /// - `QueueError::LockError`: If a lock cannot be acquired during the reset process.
    fn reset(&mut self) -> Result<(), QueueError> {
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

impl ControlEngine for Box<dyn ClassicalEngine> {
    type Input = ();
    type Output = ShotResult;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, ShotResult>, QueueError> {
        // Build up first batch of commands until measurement needed
        let commands = self.generate_commands()?;

        // Check if we have an empty message (no more commands)
        if let Ok(is_empty) = commands.is_empty() {
            if is_empty {
                // No more commands, return results
                let results = self.get_results()?;
                return Ok(EngineStage::Complete(results));
            }
        }

        // Need to process these commands
        Ok(EngineStage::NeedsProcessing(commands))
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ShotResult>, QueueError> {
        // Handle measurements from quantum engine
        self.handle_measurements(measurements)?;

        // Generate next batch of commands
        let commands = self.generate_commands()?;

        // Check if we have an empty message (no more commands)
        if let Ok(is_empty) = commands.is_empty() {
            if is_empty {
                // No more commands, return results
                let results = self.get_results()?;
                return Ok(EngineStage::Complete(results));
            }
        }

        Ok(EngineStage::NeedsProcessing(commands))
    }

    fn reset(&mut self) -> Result<(), QueueError> {
        // Use fully qualified path to disambiguate
        ClassicalEngine::reset(&mut **self)
    }
}

impl Engine for Box<dyn ClassicalEngine> {
    type Input = ();
    type Output = ShotResult;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, QueueError> {
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

    fn reset(&mut self) -> Result<(), QueueError> {
        // Use fully qualified path to disambiguate
        ClassicalEngine::reset(&mut **self)
    }
}

/// Sets up a basic QIR engine.
///
/// This function creates a QIR engine from the provided path.
///
/// # Parameters
///
/// - `program_path`: A reference to the path of the QIR program file
/// - `shots`: Optional number of shots to set for the engine
///
/// # Returns
///
/// Returns a `Box<dyn ClassicalEngine>` containing the QIR engine
pub fn setup_qir_engine(
    program_path: &Path,
    shots: Option<usize>,
) -> Result<Box<dyn ClassicalEngine>, Box<dyn Error>> {
    debug!("Setting up QIR engine for: {}", program_path.display());
    let mut engine = qir::QirEngine::new(program_path.to_path_buf());

    // Set the number of shots assigned to this engine if specified
    if let Some(num_shots) = shots {
        engine.set_assigned_shots(num_shots)?;
    }

    // Pre-compile the QIR library to prepare for efficient cloning
    engine.pre_compile()?;

    Ok(Box::new(engine))
}

/// Sets up a basic PHIR engine.
///
/// This function creates a PHIR engine from the provided path.
///
/// # Parameters
///
/// - `program_path`: A reference to the path of the PHIR program file
///
/// # Returns
///
/// Returns a `Box<dyn ClassicalEngine>` containing the PHIR engine
pub fn setup_phir_engine(program_path: &Path) -> Result<Box<dyn ClassicalEngine>, Box<dyn Error>> {
    debug!("Setting up PHIR engine for: {}", program_path.display());
    let engine = phir::PHIREngine::new(program_path)?;
    Ok(Box::new(engine))
}
