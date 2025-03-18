use crate::byte_message::ByteMessage;
use crate::byte_message::GateType;
use crate::engines::Engine;
use crate::errors::QueueError;
use dyn_clone::DynClone;
use log::debug;
use pecos_core::RngManageable;
use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec, StdSparseStab,
};
use std::fmt::Debug;

/// Trait for quantum engines that can process quantum operations
pub trait QuantumEngine:
    Engine<Input = ByteMessage, Output = ByteMessage> + DynClone + Debug
{
    /// Set a specific seed for the random number generator
    ///
    /// This method allows for deterministic behavior by setting a specific seed
    /// for the random number generator used by the quantum simulator.
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    ///
    /// # Errors
    /// Returns a `QueueError` if setting the seed fails
    fn set_seed(&mut self, seed: u64) -> Result<(), QueueError>;

    /// Reset the seed to the original value
    ///
    /// This method resets the random number generator to use the original seed,
    /// which is useful for repeating simulations with the same randomness.
    ///
    /// # Errors
    /// Returns a `QueueError` if resetting the seed fails
    fn reset_seed(&mut self, seed: u64) -> Result<(), QueueError> {
        self.set_seed(seed)
    }
}

dyn_clone::clone_trait_object!(QuantumEngine);

/// A quantum engine that uses a state vector simulator
#[derive(Debug, Clone)]
pub struct StateVecEngine {
    simulator: StateVec,
}

impl StateVecEngine {
    /// Create a new state vector engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: StateVec::new(num_qubits),
        }
    }

    /// Create a new state vector engine with a specific seed
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Seed value for the random number generator
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: StateVec::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for StateVecEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, QueueError> {
        // Parse commands from the message
        let batch = message.parse_quantum_operations()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    debug!("Processing X gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.x(cmd.qubits[0]);
                }
                GateType::Y => {
                    debug!("Processing Y gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.y(cmd.qubits[0]);
                }
                GateType::Z => {
                    debug!("Processing Z gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.z(cmd.qubits[0]);
                }
                GateType::H => {
                    debug!("Processing H gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.h(cmd.qubits[0]);
                }
                GateType::CX => {
                    debug!(
                        "Processing CX gate with control {:?} and target {:?}",
                        cmd.qubits[0], cmd.qubits[1]
                    );
                    self.simulator.cx(cmd.qubits[0], cmd.qubits[1]);
                }
                GateType::RZZ => {
                    debug!(
                        "Processing RZZ gate on qubits {:?} and {:?}",
                        cmd.qubits[0], cmd.qubits[1]
                    );
                    self.simulator
                        .rzz(cmd.params[0], cmd.qubits[0], cmd.qubits[1]);
                }
                GateType::SZZ => {
                    debug!(
                        "Processing SZZ gate on qubits {:?} and {:?}",
                        cmd.qubits[0], cmd.qubits[1]
                    );
                    self.simulator.szz(cmd.qubits[0], cmd.qubits[1]);
                }
                GateType::RZ => {
                    if !cmd.params.is_empty() {
                        debug!(
                            "Processing RZ gate with angle {:?} on qubit {:?}",
                            cmd.params[0], cmd.qubits[0]
                        );
                        self.simulator.rz(cmd.params[0], cmd.qubits[0]);
                    }
                }
                GateType::R1XY => {
                    if cmd.params.len() >= 2 {
                        debug!(
                            "Processing R1XY gate with angles theta={:?}, phi={:?} on qubit {:?}",
                            cmd.params[0], cmd.params[1], cmd.qubits[0]
                        );
                        self.simulator
                            .r1xy(cmd.params[0], cmd.params[1], cmd.qubits[0]);
                    }
                }
                GateType::Measure => {
                    if let Some(result_id) = cmd.result_id {
                        debug!(
                            "Processing measurement on qubit {:?} with result_id {:?}",
                            cmd.qubits[0], result_id
                        );
                        let meas_result = self.simulator.mz(cmd.qubits[0]);
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push((result_id, outcome));
                    }
                }
                GateType::Prep => {
                    debug!("Processing Y gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.pz(cmd.qubits[0]);
                }
            }
        }

        // Create a message with the measurement results
        let result_message = ByteMessage::record_measurement_results(&measurements);
        Ok(result_message)
    }

    fn reset(&mut self) -> Result<(), QueueError> {
        self.simulator.reset();
        Ok(())
    }
}

impl QuantumEngine for StateVecEngine {
    fn set_seed(&mut self, seed: u64) -> Result<(), QueueError> {
        self.simulator.set_seed(seed);
        Ok(())
    }
}

/// A quantum engine that uses a stabilizer simulator
#[derive(Debug, Clone)]
pub struct SparseStabEngine {
    simulator: StdSparseStab,
}

impl SparseStabEngine {
    /// Create a new stabilizer engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: StdSparseStab::new(num_qubits),
        }
    }

    /// Create a new stabilizer engine with a specific seed
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Seed value for the random number generator
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: StdSparseStab::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for SparseStabEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, QueueError> {
        // Parse commands from the message
        let batch = message.parse_quantum_operations()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    debug!("Processing X gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.x(cmd.qubits[0]);
                }
                GateType::Y => {
                    debug!("Processing Y gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.y(cmd.qubits[0]);
                }
                GateType::Z => {
                    debug!("Processing Z gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.z(cmd.qubits[0]);
                }
                GateType::H => {
                    debug!("Processing H gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.h(cmd.qubits[0]);
                }
                GateType::CX => {
                    debug!(
                        "Processing CX gate with control {:?} and target {:?}",
                        cmd.qubits[0], cmd.qubits[1]
                    );
                    self.simulator.cx(cmd.qubits[0], cmd.qubits[1]);
                }
                GateType::SZZ => {
                    debug!(
                        "Processing SZZ gate on qubits {:?} and {:?}",
                        cmd.qubits[0], cmd.qubits[1]
                    );
                    self.simulator.szz(cmd.qubits[0], cmd.qubits[1]);
                }
                GateType::Measure => {
                    if let Some(result_id) = cmd.result_id {
                        debug!(
                            "Processing measurement on qubit {:?} with result_id {:?}",
                            cmd.qubits[0], result_id
                        );
                        let meas_result = self.simulator.mz(cmd.qubits[0]);
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push((result_id, outcome));
                    }
                }
                GateType::Prep => {
                    debug!("Processing Y gate on qubit {:?}", cmd.qubits[0]);
                    self.simulator.pz(cmd.qubits[0]);
                }
                // Skip gates not supported by the stabilizer simulator
                _ => {
                    debug!("Skipping unsupported gate {:?}", cmd.gate_type);
                }
            }
        }

        // Create a message with the measurement results
        let result_message = ByteMessage::record_measurement_results(&measurements);
        Ok(result_message)
    }

    fn reset(&mut self) -> Result<(), QueueError> {
        self.simulator.reset();
        Ok(())
    }
}

impl QuantumEngine for SparseStabEngine {
    fn set_seed(&mut self, seed: u64) -> Result<(), QueueError> {
        self.simulator.set_seed(seed);
        Ok(())
    }
}

/// Creates a new quantum engine that supports both Clifford gates and arbitrary rotation gates
///
/// This factory function creates a new `StateVecEngine` and returns it as a boxed `QuantumEngine`
/// trait object, allowing for polymorphic usage.
///
/// # Parameters
/// * `simulator` - A state vector simulator
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_quantum_engine_arbitrary_qgate(simulator: StateVec) -> Box<dyn QuantumEngine> {
    Box::new(StateVecEngine { simulator })
}

/// Creates a new quantum engine with a specific seed
///
/// This factory function creates a new `StateVecEngine` with a specific seed
/// and returns it as a boxed `QuantumEngine` trait object.
///
/// # Parameters
/// * `num_qubits` - Number of qubits in the system
/// * `seed` - Seed value for the random number generator
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_quantum_engine_with_seed(num_qubits: usize, seed: u64) -> Box<dyn QuantumEngine> {
    Box::new(StateVecEngine::with_seed(num_qubits, seed))
}

/// Creates a new stabilizer quantum engine
///
/// This factory function creates a new `SparseStabEngine` and returns it as a boxed `QuantumEngine`
/// trait object, allowing for polymorphic usage.
///
/// # Parameters
/// * `num_qubits` - Number of qubits in the system
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_stabilizer_engine(num_qubits: usize) -> Box<dyn QuantumEngine> {
    Box::new(SparseStabEngine::new(num_qubits))
}

/// Creates a new stabilizer quantum engine with a specific seed
///
/// This factory function creates a new `SparseStabEngine` with a specific seed
/// and returns it as a boxed `QuantumEngine` trait object.
///
/// # Parameters
/// * `num_qubits` - Number of qubits in the system
/// * `seed` - Seed value for the random number generator
///
/// # Returns
/// A boxed `QuantumEngine` trait object
#[must_use]
pub fn new_stabilizer_engine_with_seed(num_qubits: usize, seed: u64) -> Box<dyn QuantumEngine> {
    Box::new(SparseStabEngine::with_seed(num_qubits, seed))
}
