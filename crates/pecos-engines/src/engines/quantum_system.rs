use crate::byte_message::ByteMessage;
use crate::engines::noise::{NoiseModel, PassThroughNoiseModel};
use crate::engines::quantum::QuantumEngine;
use crate::engines::{Engine, EngineSystem};
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
/// use pecos_engines::engines::quantum_system::QuantumSystem;
/// use pecos_engines::engines::noise::depolarizing::DepolarizingNoiseModel;
/// use pecos_engines::engines::quantum::StateVecEngine;
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
        }
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
        Self::new(Box::new(PassThroughNoiseModel), quantum_engine)
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
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// Returns a `PecosError` if setting the seed fails for either component
    ///
    /// # Panics
    /// This function will panic if the engine type changes between the check for engine type
    /// and the attempt to get a mutable reference to it, which should never happen in practice.
    pub fn set_seed(&mut self, seed: u64) -> Result<(), PecosError> {
        // Derive a different seed for the noise model using the standard protocol
        let noise_seed = pecos_core::rng::rng_manageable::derive_seed(seed, "noise_model");

        // Derive a different seed for the quantum engine using the standard protocol
        let engine_seed = pecos_core::rng::rng_manageable::derive_seed(seed, "quantum_engine");

        // Set the seed for the noise model using RngManageable::set_seed
        // Convert the error type to PecosError
        self.noise_model
            .set_seed(noise_seed)
            .map_err(|e| PecosError::Processing(format!("Failed to set noise model seed: {e}")))?;

        // Directly set the seed for the quantum engine using the trait method
        self.quantum_engine.set_seed(engine_seed)?;

        Ok(())
    }

    /// Returns a reference to the noise model
    #[must_use]
    pub fn noise_model(&self) -> &dyn NoiseModel {
        &*self.noise_model
    }

    /// Returns a mutable reference to the noise model
    #[must_use]
    pub fn noise_model_mut(&mut self) -> &mut dyn NoiseModel {
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
        &mut *self.quantum_engine
    }

    /// Helper method for tests to check if the engine is a specific type
    #[cfg(test)]
    fn is_engine_type(&self) -> bool {
        // Since QuantumEngine doesn't have as_any, we need to check the debug representation
        format!("{:?}", self.quantum_engine).contains("StateVecEngine")
    }
}

// Explicitly implement Engine for QuantumSystem
impl Engine for QuantumSystem {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        // Delegate to process_as_system for the standard implementation
        self.process_as_system(input)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Reset the noise model using the ControlEngine trait
        self.noise_model.reset()?;

        // Reset the quantum engine
        self.quantum_engine.reset()?;

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

    fn controller(&self) -> &Self::Controller {
        &self.noise_model
    }

    fn controller_mut(&mut self) -> &mut Self::Controller {
        &mut self.noise_model
    }

    fn engine(&self) -> &Self::ControlledEngine {
        &self.quantum_engine
    }

    fn engine_mut(&mut self) -> &mut Self::ControlledEngine {
        &mut self.quantum_engine
    }
}

impl Clone for QuantumSystem {
    fn clone(&self) -> Self {
        Self {
            noise_model: dyn_clone::clone_box(&*self.noise_model),
            quantum_engine: dyn_clone::clone_box(&*self.quantum_engine),
        }
    }
}

// Manual implementation of Debug for QuantumSystem
impl Debug for QuantumSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumSystem")
            .field("noise_model", &format!("{:p}", &self.noise_model))
            .field("quantum_engine", &format!("{:p}", &self.quantum_engine))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::ByteMessageBuilder;
    use crate::engines::ControlEngine;
    use crate::engines::noise::{DepolarizingNoiseModel, PassThroughNoiseModel};
    use crate::engines::quantum::StateVecEngine;

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

        system
            .set_seed(seed)
            .expect("Failed to set seed for system");

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
        builder.add_x(&[0]);
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
        builder.add_h(&[0]);
        builder.add_measurements(&[0], &[0]);
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
            .parse_measurements()
            .expect("Failed to parse measurement results from system1");
        let meas2 = result2
            .parse_measurements()
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
        system1
            .set_seed(different_seed)
            .expect("Failed to set seed for system1");

        // Process the input again with system1 and system3
        let result1 = system1
            .process(input.clone())
            .expect("Failed to process input with system1 after seed change");
        let result3 = system3
            .process(input)
            .expect("Failed to process input with system3");

        // Extract and compare measurement results
        let meas1 = result1
            .parse_measurements()
            .expect("Failed to parse measurement results from system1");
        let meas3 = result3
            .parse_measurements()
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
        builder.add_x(&[0]);
        builder.add_measurements(&[0], &[0]);
        let input = builder.build();

        // Process the input using the blanket implementation of Engine for EngineSystem
        let result = system
            .process(input.clone())
            .expect("Failed to process input");

        // Verify the result contains measurements
        assert!(result.parse_measurements().is_ok());
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
        builder.add_x(&[0]);
        builder.add_measurements(&[0], &[0]);
        let input = builder.build();

        // Process the input through the system
        let result = system
            .process(input.clone())
            .expect("Failed to process input");
        assert!(result.parse_measurements().is_ok());

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
