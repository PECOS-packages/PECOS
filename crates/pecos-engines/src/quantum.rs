use crate::Engine;
use crate::byte_message::ByteMessage;
use crate::byte_message::GateType;
use dyn_clone::DynClone;
use log::debug;
use pecos_core::QubitId;
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec, StdSparseStab,
};
use std::any::Any;
use std::fmt::Debug;

/// Helper function to create quantum engine errors
fn quantum_error<S: Into<String>>(msg: S) -> PecosError {
    PecosError::Processing(msg.into())
}

/// Trait for quantum engines that can process quantum operations
pub trait QuantumEngine:
    Engine<Input = ByteMessage, Output = ByteMessage> + DynClone + Debug
{
    /// Sets a specific seed for the quantum engine
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    fn set_seed(&mut self, seed: u64);

    /// Returns a reference to this object as Any, for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Returns a mutable reference to this object as Any, for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

dyn_clone::clone_trait_object!(QuantumEngine);

// Implement Engine for Box<dyn QuantumEngine> to allow using it directly
// as a controlled engine in EngineSystem
impl Engine for Box<dyn QuantumEngine> {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        // Delegate to the underlying QuantumEngine
        (**self).process(input)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Delegate to the underlying QuantumEngine
        (**self).reset()
    }
}

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

    /// Ensure the simulator has the correct number of qubits, recreating if necessary
    ///
    /// This method checks if the current simulator has enough qubits.
    /// If not, it recreates the simulator with more qubits to prevent
    /// memory corruption during quantum operations.
    ///
    /// Note: The simulator can only grow, never shrink, to preserve quantum state.
    ///
    /// # Arguments
    /// * `required_qubits` - The minimum number of qubits required for the simulation
    pub fn ensure_qubit_count(&mut self, required_qubits: usize) {
        if self.simulator.num_qubits() < required_qubits {
            debug!(
                "StateVecEngine: Expanding simulator (was {} qubits, now {} qubits)",
                self.simulator.num_qubits(),
                required_qubits
            );
            // Preserve the RNG state if possible
            let rng = self.simulator.rng().clone();
            self.simulator = StateVec::with_rng(required_qubits, rng);
        }
    }
}

impl Engine for StateVecEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        // Parse commands from the message
        let batch = message.quantum_ops()?;

        // Calculate required number of qubits from operations and ensure simulator has correct size
        if !batch.is_empty() {
            let max_qubit_index = batch
                .iter()
                .flat_map(|cmd| cmd.qubits.iter())
                .map(|q| usize::from(*q))
                .max()
                .unwrap_or(0);
            let required_qubits = max_qubit_index + 1;
            self.ensure_qubit_count(required_qubits);
        }

        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    for q in &cmd.qubits {
                        debug!("Processing X gate on qubit {q:?}");
                        self.simulator.x(usize::from(*q));
                    }
                }
                GateType::Y => {
                    for q in &cmd.qubits {
                        debug!("Processing Y gate on qubit {q:?}");
                        self.simulator.y(usize::from(*q));
                    }
                }
                GateType::Z => {
                    for q in &cmd.qubits {
                        debug!("Processing Z gate on qubit {q:?}");
                        self.simulator.z(usize::from(*q));
                    }
                }
                GateType::H => {
                    for q in &cmd.qubits {
                        debug!("Processing H gate on qubit {q:?}");
                        self.simulator.h(usize::from(*q));
                    }
                }
                GateType::SZ => {
                    for q in &cmd.qubits {
                        debug!("Processing SZ gate on qubit {q:?}");
                        self.simulator.sz(usize::from(*q));
                    }
                }
                GateType::SZdg => {
                    for q in &cmd.qubits {
                        debug!("Processing SZdg gate on qubit {q:?}");
                        self.simulator.szdg(usize::from(*q));
                    }
                }
                GateType::T => {
                    for q in &cmd.qubits {
                        debug!("Processing T gate on qubit {q:?}");
                        self.simulator.t(usize::from(*q));
                    }
                }
                GateType::Tdg => {
                    for q in &cmd.qubits {
                        debug!("Processing Tdg gate on qubit {q:?}");
                        self.simulator.tdg(usize::from(*q));
                    }
                }
                GateType::CX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing CX gate with control {:?} and target {:?}",
                            qubits[0], qubits[1]
                        );
                        self.simulator
                            .cx(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::RZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "RZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.params.is_empty() {
                        return Err(quantum_error("RZZ gate requires at least one parameter"));
                    }
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing RZZ gate on qubits {:?} and {:?}",
                            qubits[0], qubits[1]
                        );
                        self.simulator.rzz(cmd.params[0], *qubits[0], *qubits[1]);
                    }
                }
                GateType::SZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing SZZ gate on qubits {:?} and {:?}",
                            qubits[0], qubits[1]
                        );
                        self.simulator
                            .szz(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::SZZdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SZZdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing SZZdg gate on qubits {:?} and {:?}",
                            qubits[0], qubits[1]
                        );
                        self.simulator
                            .szzdg(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                // TODO: Consider setting exact numbers of parameters
                GateType::RX => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            debug!(
                                "Processing RX gate with angle {:?} on qubit {:?}",
                                cmd.params[0], q
                            );
                            self.simulator.rx(cmd.params[0], **q);
                        }
                    }
                }
                GateType::RY => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            debug!(
                                "Processing RY gate with angle {:?} on qubit {:?}",
                                cmd.params[0], q
                            );
                            self.simulator.ry(cmd.params[0], **q);
                        }
                    }
                }
                GateType::RZ => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            debug!(
                                "Processing RZ gate with angle {:?} on qubit {:?}",
                                cmd.params[0], q
                            );
                            self.simulator.rz(cmd.params[0], **q);
                        }
                    }
                }
                // TODO: Consider setting exact number of parameters
                GateType::R1XY => {
                    if cmd.params.len() >= 2 {
                        for q in &cmd.qubits {
                            debug!(
                                "Processing R1XY gate with angles theta={:?}, phi={:?} on qubit {:?}",
                                cmd.params[0], cmd.params[1], q
                            );
                            self.simulator.r1xy(cmd.params[0], cmd.params[1], **q);
                        }
                    }
                }

                // TODO: Fix it so we have multiple result_ids or get rid of result ids...
                GateType::Measure | GateType::MeasureLeaked => {
                    for q in &cmd.qubits {
                        debug!("Processing measurement on qubit {q:?}");
                        let meas_result = self.simulator.mz(**q);
                        // According to the documentation:
                        // mz() outcome: true if projected to |1⟩, false if projected to |0⟩
                        // So we can directly convert the boolean to u32
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep => {
                    for q in &cmd.qubits {
                        debug!("Processing Prep gate on qubit {q:?}");
                        self.simulator.pz(**q);
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload => {
                    // Just let the system naturally evolve for the specified duration
                    // No active operation needed in the simulator
                }
                GateType::U => {
                    // TODO: Consider checking for the exact number of parameters
                    if cmd.params.len() >= 3 {
                        for q in &cmd.qubits {
                            debug!(
                                "Processing U gate with angles theta={:?}, phi={:?}, lambda={:?} on qubit {:?}",
                                cmd.params[0], cmd.params[1], cmd.params[2], q
                            );
                            self.simulator
                                .u(cmd.params[0], cmd.params[1], cmd.params[2], **q);
                        }
                    }
                }
            }
        }

        // Create a message with the measurement results
        let mut builder = ByteMessage::outcomes_builder();

        // Convert measurements from u32 to usize and add to builder
        let outcomes: Vec<usize> = measurements.iter().map(|&m| m as usize).collect();
        builder.add_outcomes(&outcomes);

        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("StateVecEngine: reset() called");
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for StateVecEngine {
    type Rng = <StateVec as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    /// Get a read-only reference to the internal random number generator
    ///
    /// This method delegates to the underlying simulator's RNG
    ///
    /// # Returns
    /// A reference to the internal RNG
    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    /// Get a mutable reference to the internal random number generator
    ///
    /// This method delegates to the underlying simulator's RNG
    ///
    /// # Returns
    /// A mutable reference to the internal RNG
    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for StateVecEngine {
    fn set_seed(&mut self, seed: u64) {
        // Create a new RNG with the given seed
        let rng = <StateVec as RngManageable>::Rng::seed_from_u64(seed);

        // Set the simulator's RNG
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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

impl SparseStabEngine {
    fn process_single_qubit_gate(
        &mut self,
        gate_type: GateType,
        qubits: &[QubitId],
    ) -> Result<(), PecosError> {
        match gate_type {
            GateType::X => {
                for q in qubits {
                    debug!("Processing X gate on qubit {q:?}");
                    self.simulator.x(usize::from(*q));
                }
            }
            GateType::Y => {
                for q in qubits {
                    debug!("Processing Y gate on qubit {q:?}");
                    self.simulator.y(usize::from(*q));
                }
            }
            GateType::Z => {
                for q in qubits {
                    debug!("Processing Z gate on qubit {q:?}");
                    self.simulator.z(usize::from(*q));
                }
            }
            GateType::H => {
                for q in qubits {
                    debug!("Processing H gate on qubit {q:?}");
                    self.simulator.h(usize::from(*q));
                }
            }
            GateType::SZ => {
                for q in qubits {
                    debug!("Processing SZ gate on qubit {q:?}");
                    self.simulator.sz(usize::from(*q));
                }
            }
            GateType::SZdg => {
                for q in qubits {
                    debug!("Processing SZdg gate on qubit {q:?}");
                    self.simulator.szdg(usize::from(*q));
                }
            }
            GateType::T => {
                return Err(quantum_error(
                    "T gate is not supported by stabilizer simulator",
                ));
            }
            GateType::Tdg => {
                return Err(quantum_error(
                    "Tdg gate is not supported by stabilizer simulator",
                ));
            }
            GateType::RX | GateType::RY => {
                return Err(quantum_error(
                    "RX/RY gates are not supported by stabilizer simulator",
                ));
            }
            _ => {} // Handled elsewhere
        }
        Ok(())
    }

    fn process_two_qubit_gate(&mut self, gate_type: GateType, qubits: &[QubitId]) {
        // Verify even number of qubits for all two-qubit gates
        if !qubits.len().is_multiple_of(2) {
            log::warn!(
                "{:?} gate requires even number of qubits, got {} - skipping",
                gate_type,
                qubits.len()
            );
            return;
        }

        match gate_type {
            GateType::CX => {
                for qubits in qubits.chunks_exact(2) {
                    debug!(
                        "Processing CX gate with control {:?} and target {:?}",
                        qubits[0], qubits[1]
                    );
                    self.simulator
                        .cx(usize::from(qubits[0]), usize::from(qubits[1]));
                }
            }
            GateType::SZZ => {
                for qubits in qubits.chunks_exact(2) {
                    debug!(
                        "Processing SZZ gate on qubits {:?} and {:?}",
                        qubits[0], qubits[1]
                    );
                    self.simulator
                        .szz(usize::from(qubits[0]), usize::from(qubits[1]));
                }
            }
            GateType::SZZdg => {
                for qubits in qubits.chunks_exact(2) {
                    debug!(
                        "Processing SZZdg gate on qubits {:?} and {:?}",
                        qubits[0], qubits[1]
                    );
                    self.simulator
                        .szzdg(usize::from(qubits[0]), usize::from(qubits[1]));
                }
            }
            _ => {} // Not a two-qubit gate
        }
    }
}

impl Engine for SparseStabEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        let batch = message.quantum_ops()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                // Single-qubit gates
                GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::H
                | GateType::SZ
                | GateType::SZdg
                | GateType::T
                | GateType::Tdg
                | GateType::RX
                | GateType::RY => {
                    self.process_single_qubit_gate(cmd.gate_type, &cmd.qubits)?;
                }
                // Two-qubit gates
                GateType::CX | GateType::SZZ | GateType::SZZdg => {
                    self.process_two_qubit_gate(cmd.gate_type, &cmd.qubits);
                }
                // Special operations
                GateType::Measure | GateType::MeasureLeaked => {
                    for q in &cmd.qubits {
                        debug!("Processing measurement on qubit {q:?}");
                        let meas_result = self.simulator.mz(**q);
                        // According to the documentation:
                        // mz() outcome: true if projected to |1⟩, false if projected to |0⟩
                        // So we can directly convert the boolean to u32
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep => {
                    for q in &cmd.qubits {
                        debug!("Processing Prep gate on qubit {q:?}");
                        self.simulator.pz(**q);
                    }
                }
                GateType::Idle => {
                    // For idle gates, just let the system naturally evolve
                    // No active operation needed in the simulator
                }
                _ => {
                    return Err(PecosError::Processing(format!(
                        "Gate {:?} is not supported by the stabilizer simulator. Only Clifford gates are supported.",
                        cmd.gate_type
                    )));
                }
            }
        }

        // Create a message with the measurement results
        let mut builder = ByteMessage::outcomes_builder();
        let outcomes: Vec<usize> = measurements.iter().map(|&m| m as usize).collect();
        builder.add_outcomes(&outcomes);

        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for SparseStabEngine {
    type Rng = <StdSparseStab as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    /// Get a read-only reference to the internal random number generator
    ///
    /// This method delegates to the underlying simulator's RNG
    ///
    /// # Returns
    /// A reference to the internal RNG
    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    /// Get a mutable reference to the internal random number generator
    ///
    /// This method delegates to the underlying simulator's RNG
    ///
    /// # Returns
    /// A mutable reference to the internal RNG
    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for SparseStabEngine {
    fn set_seed(&mut self, seed: u64) {
        // Create a new RNG with the given seed
        let rng = <StdSparseStab as RngManageable>::Rng::seed_from_u64(seed);

        // Set the simulator's RNG
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
