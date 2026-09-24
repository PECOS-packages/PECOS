use crate::Engine;
use crate::byte_message::ByteMessage;
use crate::engine_system::EngineSystem;
use crate::noise::{NoiseModel, PassThroughNoiseModel};
use crate::quantum::QuantumEngine;
use crate::runtime_frame::{self, FrameExecutor, RuntimeGeneralNoise, ShotContext};
use pecos_core::errors::PecosError;
use std::fmt::Debug;

/// A system that coordinates quantum simulation with noise application
///
/// The `QuantumSystem` combines:
/// 1. A `NoiseModel` that transforms quantum operations
/// 2. A `QuantumEngine` that processes those operations
///
/// This is a controlled execution environment where noise transforms the idealized
/// quantum operations before they are passed to the quantum engine.
///
/// # Examples
///
/// ```
/// use pecos_engines::quantum_system::QuantumSystem;
/// use pecos_engines::noise::depolarizing::DepolarizingNoiseModel;
/// use pecos_engines::quantum::StateVecEngine;
///
/// // Create a quantum system with 2 qubits
/// let noise_model = DepolarizingNoiseModel::new_uniform(0.01);
/// let engine = StateVecEngine::new(2);
/// let system = QuantumSystem::new(Box::new(noise_model), Box::new(engine));
/// ```
pub struct QuantumSystem {
    // Core components
    noise_model: Box<dyn NoiseModel>,
    quantum_engine: Box<dyn QuantumEngine>,
    shot_context: Option<ShotContext>,
    frame_poisoned: bool,
    host_blocked: bool,
}

impl QuantumSystem {
    /// Create a new `QuantumSystem` with the given noise model and quantum engine
    ///
    /// # Parameters
    /// - `noise_model`: A boxed noise model implementing the `NoiseModel` trait
    /// - `quantum_engine`: A boxed quantum engine implementing the `QuantumEngine` trait
    ///
    /// # Returns
    /// A new `QuantumSystem` with the specified components
    #[must_use]
    pub fn new(noise_model: Box<dyn NoiseModel>, quantum_engine: Box<dyn QuantumEngine>) -> Self {
        Self {
            noise_model,
            quantum_engine,
            shot_context: None,
            frame_poisoned: false,
            host_blocked: false,
        }
    }

    fn drive_legacy(&mut self, input: ByteMessage) -> Result<ByteMessage, PecosError> {
        let mut stage = self.noise_model.start(input)?;
        loop {
            match stage {
                crate::EngineStage::Complete(output) => return Ok(output),
                crate::EngineStage::NeedsProcessing(commands) => {
                    let reply = self.quantum_engine.process(commands)?;
                    stage = self.noise_model.continue_processing(reply)?;
                }
            }
        }
    }

    /// Establish host identity after reset. Identity never changes RNG streams.
    ///
    /// # Errors
    /// Rejects a failed owner or an already active shot. Multiple process inputs
    /// retain this context until reset; no per-input shot identity is inferred.
    pub fn begin_shot(&mut self, context: ShotContext) -> Result<(), PecosError> {
        if self.frame_poisoned || self.host_blocked || self.shot_context.is_some() {
            return Err(runtime_frame::error("shot requires successful reset"));
        }
        self.shot_context = Some(context);
        Ok(())
    }

    /// Current explicit host identity, if established.
    #[must_use]
    pub fn shot_context(&self) -> Option<ShotContext> {
        self.shot_context
    }

    pub(crate) fn uses_runtime_frames(&self) -> bool {
        self.noise_model.as_any().is::<RuntimeGeneralNoise>()
    }
    pub(crate) fn block_host(&mut self) {
        self.host_blocked = true;
    }
    pub(crate) fn finish_host_reset(&mut self) {
        self.host_blocked = false;
    }

    /// Create a new `QuantumSystem` with the given quantum engine and no noise
    ///
    /// This is a convenience method that creates a new `QuantumSystem` with a
    /// `PassThroughNoise` model, which does not apply any noise transformations.
    ///
    /// # Parameters
    /// - `quantum_engine`: A boxed quantum engine implementing the `QuantumEngine` trait
    ///
    /// # Returns
    /// A new `QuantumSystem` with the specified engine and a pass-through noise model
    #[must_use]
    pub fn new_without_noise(quantum_engine: Box<dyn QuantumEngine>) -> Self {
        Self::new(
            Box::new(PassThroughNoiseModel::builder().build()),
            quantum_engine,
        )
    }

    /// Set a specific seed for all components of the quantum system
    ///
    /// This method sets different but deterministic seeds for each component:
    /// - The noise model
    /// - The quantum engine
    ///
    /// The seeds are derived from the base seed using a standard seed derivation protocol
    /// to ensure they don't produce correlated random sequences.
    ///
    /// # Arguments
    /// * `seed` - Base seed value for the random number generators
    pub fn set_seed(&mut self, seed: u64) {
        // Derive a different seed for the noise model using the standard protocol
        let noise_seed = pecos_core::rng::rng_manageable::derive_seed(seed, "noise_model");

        // Derive a different seed for the quantum engine using the standard protocol
        let engine_seed = pecos_core::rng::rng_manageable::derive_seed(seed, "quantum_engine");

        // Set the seed for the noise model using RngManageable::set_seed
        self.noise_model.set_seed(noise_seed);

        // Directly set the seed for the quantum engine using the trait method
        self.quantum_engine.set_seed(engine_seed);
    }

    /// Returns a reference to the noise model
    #[must_use]
    pub fn noise_model(&self) -> &dyn NoiseModel {
        &*self.noise_model
    }

    /// Returns a mutable reference to the noise model
    #[must_use]
    pub fn noise_model_mut(&mut self) -> &mut dyn NoiseModel {
        if self.uses_runtime_frames() {
            self.frame_poisoned = true;
        }
        self.shot_context = None;
        &mut *self.noise_model
    }

    /// Returns a reference to the quantum engine
    #[must_use]
    pub fn quantum_engine(&self) -> &dyn QuantumEngine {
        &*self.quantum_engine
    }

    /// Returns a mutable reference to the quantum engine
    #[must_use]
    pub fn quantum_engine_mut(&mut self) -> &mut dyn QuantumEngine {
        if self.uses_runtime_frames() {
            self.frame_poisoned = true;
        }
        self.shot_context = None;
        &mut *self.quantum_engine
    }

    /// Helper method for tests to check if the engine is a specific type
    #[cfg(test)]
    fn is_engine_type(&self) -> bool {
        // Since QuantumEngine doesn't have as_any, we need to check the debug representation
        // StateVecEngine is now a type alias for StateVectorEngine<StateVec>
        let debug_str = format!("{:?}", self.quantum_engine);
        debug_str.contains("StateVectorEngine") || debug_str.contains("StateVecEngine")
    }
}

// Explicitly implement Engine for QuantumSystem
impl Engine for QuantumSystem {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        let framed = input.as_bytes().get(4) == Some(&2);
        if !self.uses_runtime_frames() {
            // Ordinary consumers reject v2 through their existing parser.
            if framed {
                return Err(runtime_frame::error("runtime frame capability required"));
            }
            return self.drive_legacy(input);
        }
        if self.frame_poisoned || self.host_blocked {
            return Err(runtime_frame::error(
                "execution owner poisoned; reset required",
            ));
        }
        if self.shot_context.is_none() {
            return Err(runtime_frame::error("explicit shot context required"));
        }
        let model = self
            .noise_model
            .as_any_mut()
            .downcast_mut::<RuntimeGeneralNoise>()
            .expect("checked private compiled model");
        if framed {
            let records = runtime_frame::decode(&input, model)?;
            // Only the tested built-in state-vector consumer is admitted initially.
            let sim = self
                .quantum_engine
                .as_any()
                .downcast_ref::<crate::StateVecEngine>()
                .ok_or_else(|| runtime_frame::error("unsupported frame simulator"))?;
            if sim.simulator().num_qubits() < model.qubits {
                return Err(runtime_frame::error(
                    "simulator capacity below frame profile",
                ));
            }
            // Latch before mutation; errors and unwinding leave this set.
            self.frame_poisoned = true;
            let result = FrameExecutor {
                model,
                simulator: &mut *self.quantum_engine,
            }
            .execute(records)?;
            self.frame_poisoned = false;
            Ok(result)
        } else {
            model.preflight_legacy(&input)?;
            self.frame_poisoned = true;
            let result = self.drive_legacy(input)?;
            self.frame_poisoned = false;
            Ok(result)
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.frame_poisoned = true;
        self.shot_context = None;
        // Clear poison only after both resets succeed.
        self.noise_model.reset()?;

        // Reset the quantum engine
        self.quantum_engine.reset()?;
        self.frame_poisoned = false;
        Ok(())
    }
}

// Implement EngineSystem for QuantumSystem using core components directly
impl EngineSystem for QuantumSystem {
    // Use the core components directly for the controller and controlled engine
    type Controller = Box<dyn NoiseModel>;
    type ControlledEngine = Box<dyn QuantumEngine>;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn process_as_system(&mut self, input: ByteMessage) -> Result<ByteMessage, PecosError> {
        self.process(input)
    }

    fn controller(&self) -> &Self::Controller {
        &self.noise_model
    }

    fn controller_mut(&mut self) -> &mut Self::Controller {
        if self.uses_runtime_frames() {
            self.frame_poisoned = true;
        }
        self.shot_context = None;
        &mut self.noise_model
    }

    fn engine(&self) -> &Self::ControlledEngine {
        &self.quantum_engine
    }

    fn engine_mut(&mut self) -> &mut Self::ControlledEngine {
        if self.uses_runtime_frames() {
            self.frame_poisoned = true;
        }
        self.shot_context = None;
        &mut self.quantum_engine
    }
}

impl Clone for QuantumSystem {
    fn clone(&self) -> Self {
        Self {
            noise_model: dyn_clone::clone_box(&*self.noise_model),
            quantum_engine: dyn_clone::clone_box(&*self.quantum_engine),
            shot_context: None,
            frame_poisoned: self.frame_poisoned,
            host_blocked: self.host_blocked,
        }
    }
}

// Manual implementation of Debug for QuantumSystem
impl Debug for QuantumSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumSystem")
            .field("noise_model", &format!("{:p}", &self.noise_model))
            .field("quantum_engine", &format!("{:p}", &self.quantum_engine))
            .field("shot_context", &self.shot_context)
            .field("frame_poisoned", &self.frame_poisoned)
            .field("host_blocked", &self.host_blocked)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::ByteMessageBuilder;
    use crate::engine_system::ControlEngine;
    use crate::noise::{DepolarizingNoiseModel, PassThroughNoiseModel};
    use crate::quantum::StateVecEngine;

    // Note: QuantumSystem implements EngineSystem and uses the blanket implementation
    // of Engine for EngineSystem. This allows it to be used as a controlled engine
    // in higher-level engine systems like HybridEngine.

    /// Creates a new `QuantumSystem` with a state vector quantum engine and depolarizing noise
    ///
    /// # Parameters
    /// - `num_qubits`: Number of qubits for the quantum engine
    /// - `probability`: Probability parameter for the depolarizing noise model (between 0.0 and 1.0)
    ///
    /// # Returns
    /// A new `QuantumSystem` configured with the specified parameters
    #[must_use]
    pub fn create_quantume_system_with_state_vec_and_depolarizing_noise(
        num_qubits: usize,
        probability: f64,
    ) -> QuantumSystem {
        // Create a quantum engine using a state vector simulator
        let quantum_engine = Box::new(StateVecEngine::new(num_qubits));

        // Create a QuantumSystem with depolarizing noise
        QuantumSystem::new(
            Box::new(DepolarizingNoiseModel::new_uniform(probability)),
            quantum_engine,
        )
    }

    /// Creates a new `QuantumSystem` with a state vector quantum engine and depolarizing noise with a specific seed
    ///
    /// This function first creates a quantum system with the specified number of qubits and
    /// depolarizing noise probability, then sets the seed using the `set_seed` method, which
    /// handles the derivation of component-specific seeds.
    ///
    /// # Parameters
    /// - `num_qubits`: Number of qubits for the quantum engine
    /// - `probability`: Probability parameter for the depolarizing noise model (between 0.0 and 1.0)
    /// - `seed`: Seed value for the random number generators
    ///
    /// # Returns
    /// A new `QuantumSystem` configured with the specified parameters and seeded randomness
    #[must_use]
    pub fn create_quantume_system_with_state_vec_and_depolarizing_noise_with_seed(
        num_qubits: usize,
        probability: f64,
        seed: u64,
    ) -> QuantumSystem {
        // Create a quantum engine
        let quantum_engine = Box::new(StateVecEngine::new(num_qubits));

        let mut system = // Create a QuantumSystem with depolarizing noise
        QuantumSystem::new(
            Box::new(DepolarizingNoiseModel::new_uniform(probability)),
            quantum_engine,
        );

        system.set_seed(seed);

        system
    }

    /// Test that verifies the ability to update the probability of a depolarizing noise model
    #[test]
    fn test_access_and_update_noise_model() {
        // Create a quantum system with 2 qubits and 1% depolarizing noise
        let mut system = create_quantume_system_with_state_vec_and_depolarizing_noise(2, 0.01);

        // Get a reference to the noise model and verify it's a DepolarizingNoise
        let noise_model = system.noise_model();
        assert!(noise_model.as_any().is::<DepolarizingNoiseModel>());

        // Create a simple quantum circuit with an X gate on qubit 0
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        let input = builder.build();

        // Process the input with 1% noise
        let _result1 = system
            .process(input.clone())
            .expect("Failed to process input with initial noise");

        // Get a mutable reference to the noise model and update the probability
        if let Some(depolarizing_noise) = system
            .noise_model_mut()
            .as_any_mut()
            .downcast_mut::<DepolarizingNoiseModel>()
        {
            depolarizing_noise.set_uniform_probability(0.05);
        } else {
            panic!("Failed to downcast noise model to DepolarizingNoise");
        }

        // With the simplified design, we no longer need to update components as the
        // noise_model is used directly as the controller

        // Process the same input with 5% noise
        let _result2 = system
            .process(input)
            .expect("Failed to process input with updated noise");

        // Verify that a system with PassThroughNoise cannot be downcast to DepolarizingNoise
        let mut system_without_noise =
            QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(2)));

        // Verify the noise model is not a DepolarizingNoise
        assert!(
            system_without_noise
                .noise_model()
                .as_any()
                .is::<PassThroughNoiseModel>()
        );
        assert!(
            !system_without_noise
                .noise_model()
                .as_any()
                .is::<DepolarizingNoiseModel>()
        );

        // Attempt to downcast to DepolarizingNoise should fail
        assert!(
            system_without_noise
                .noise_model_mut()
                .as_any_mut()
                .downcast_mut::<DepolarizingNoiseModel>()
                .is_none()
        );
    }

    /// Test that verifies the seed management functionality
    #[test]
    fn test_seed_management() {
        // Create two quantum systems with the same seed
        let seed = 42u64;
        let mut system1 =
            create_quantume_system_with_state_vec_and_depolarizing_noise_with_seed(2, 0.5, seed);
        let mut system2 =
            create_quantume_system_with_state_vec_and_depolarizing_noise_with_seed(2, 0.5, seed);

        // Create a simple quantum circuit with a Hadamard gate and measurement
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.h(&[0]);
        builder.mz(&[0]);
        let input = builder.build();

        // Process the input with both systems - they should produce the same results
        let result1 = system1
            .process(input.clone())
            .expect("Failed to process input with system1");
        let result2 = system2
            .process(input.clone())
            .expect("Failed to process input with system2");

        // Extract and compare measurement results
        let meas1 = result1
            .outcomes()
            .expect("Failed to parse measurement results from system1");
        let meas2 = result2
            .outcomes()
            .expect("Failed to parse measurement results from system2");

        assert_eq!(
            meas1, meas2,
            "Systems with the same seed should produce the same results"
        );

        // Now create a system with a different seed
        let different_seed = 43u64;
        let mut system3 = create_quantume_system_with_state_vec_and_depolarizing_noise_with_seed(
            2,
            0.5,
            different_seed,
        );

        // Reset system1 and set it to use the different seed
        system1.reset().expect("Failed to reset system1");
        system1.set_seed(different_seed);

        // Process the input again with system1 and system3
        let result1 = system1
            .process(input.clone())
            .expect("Failed to process input with system1 after seed change");
        let result3 = system3
            .process(input)
            .expect("Failed to process input with system3");

        // Extract and compare measurement results
        let meas1 = result1
            .outcomes()
            .expect("Failed to parse measurement results from system1");
        let meas3 = result3
            .outcomes()
            .expect("Failed to parse measurement results from system3");

        assert_eq!(
            meas1, meas3,
            "System1 with updated seed should match system3"
        );
    }

    /// Test that verifies our engine type checking functionality
    #[test]
    fn test_engine_type_checking() {
        // Create a quantum system with 2 qubits and 5% depolarizing noise
        let system = create_quantume_system_with_state_vec_and_depolarizing_noise(2, 0.05);

        // Verify the engine is a StateVecEngine
        assert!(system.is_engine_type());
    }

    /// Test that verifies the blanket implementation of process works correctly
    #[test]
    fn test_blanket_process_implementation() {
        // Create a quantum system with 2 qubits and 5% depolarizing noise
        let mut system = create_quantume_system_with_state_vec_and_depolarizing_noise(2, 0.05);

        // Create a simple quantum circuit with an X gate on qubit 0 and measurement
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.mz(&[0]);
        let input = builder.build();

        // Process the input using the blanket implementation of Engine for EngineSystem
        let result = system
            .process(input.clone())
            .expect("Failed to process input");

        // Verify the result contains measurements
        assert!(result.outcomes().is_ok());
    }

    /// Test that the `EngineSystem` pattern works correctly with direct access to
    /// controller and engine components
    #[test]
    fn test_engine_system_pattern() {
        // Create a quantum system with 2 qubits and 5% depolarizing noise
        let mut system = create_quantume_system_with_state_vec_and_depolarizing_noise(2, 0.05);

        // Create a simple quantum circuit with an X gate on qubit 0 and measurement
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.mz(&[0]);
        let input = builder.build();

        // Process the input through the system
        let result = system
            .process(input.clone())
            .expect("Failed to process input");
        assert!(result.outcomes().is_ok());

        // Test that we can use controller and engine components directly
        {
            // Test controller_mut which gives a mutable reference to the controller
            let stage_result = system.controller_mut().start(input.clone());
            assert!(stage_result.is_ok());
        }

        {
            // Test engine_mut which gives a mutable reference to the engine
            let reset_result = system.engine_mut().reset();
            assert!(reset_result.is_ok());
        }
    }
}
