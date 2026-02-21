use crate::Engine;
use crate::byte_message::ByteMessage;
use crate::byte_message::GateType;
use dyn_clone::DynClone;
use log::debug;
use pecos_core::Angle64;
use pecos_core::QubitId;
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, CoinToss, QuantumSimulator, SparseStab, StateVec,
    StateVecAoS, StateVecSoA,
};
use pecos_rng::{PecosRng, SeedableRng};
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

/// Trait for simulators that can be used with `StateVectorEngine`.
///
/// This trait combines all the requirements for a state vector simulator.
pub trait StateVectorSimulator:
    QuantumSimulator
    + CliffordGateable
    + ArbitraryRotationGateable
    + RngManageable
    + Clone
    + Debug
    + Send
    + Sync
where
    <Self as RngManageable>::Rng: Clone,
{
    /// Returns the number of qubits in the simulator.
    fn num_qubits(&self) -> usize;

    /// Create a new simulator with the specified number of qubits.
    fn create(num_qubits: usize) -> Self;

    /// Create a new simulator with a specific seed.
    fn create_with_seed(num_qubits: usize, seed: u64) -> Self;

    /// Create a new simulator with a custom RNG.
    fn create_with_rng(num_qubits: usize, rng: <Self as RngManageable>::Rng) -> Self;
}

impl StateVectorSimulator for StateVec {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }

    fn create(num_qubits: usize) -> Self {
        StateVec::new(num_qubits)
    }

    fn create_with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVec::with_seed(num_qubits, seed)
    }

    fn create_with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        StateVec::with_rng(num_qubits, rng)
    }
}

impl StateVectorSimulator for StateVecAoS {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }

    fn create(num_qubits: usize) -> Self {
        StateVecAoS::new(num_qubits)
    }

    fn create_with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVecAoS::with_seed(num_qubits, seed)
    }

    fn create_with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        StateVecAoS::with_rng(num_qubits, rng)
    }
}

impl StateVectorSimulator for StateVecSoA {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }

    fn create(num_qubits: usize) -> Self {
        StateVecSoA::new(num_qubits)
    }

    fn create_with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVecSoA::with_seed(num_qubits, seed)
    }

    fn create_with_rng(num_qubits: usize, rng: PecosRng) -> Self {
        StateVecSoA::with_rng(num_qubits, rng)
    }
}

/// A generic quantum engine that uses any state vector simulator.
///
/// This engine works with any simulator implementing `StateVectorSimulator`.
#[derive(Debug, Clone)]
pub struct StateVectorEngine<S: StateVectorSimulator>
where
    <S as RngManageable>::Rng: Clone,
{
    simulator: S,
}

impl<S: StateVectorSimulator> StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    /// Create a new state vector engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: S::create(num_qubits),
        }
    }

    /// Create a new state vector engine with a specific seed
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: S::create_with_seed(num_qubits, seed),
        }
    }

    /// Ensure the simulator has the correct number of qubits, recreating if necessary
    pub fn ensure_qubit_count(&mut self, required_qubits: usize) {
        if self.simulator.num_qubits() < required_qubits {
            debug!(
                "StateVectorEngine: Expanding simulator (was {} qubits, now {} qubits)",
                self.simulator.num_qubits(),
                required_qubits
            );
            let rng = self.simulator.rng().clone();
            self.simulator = S::create_with_rng(required_qubits, rng);
        }
    }
}

/// Type alias for state vector engine using the default `StateVec` simulator
/// (sparse `SoA`, optimized for QEC workloads).
pub type StateVecEngine = StateVectorEngine<StateVec>;

/// Type alias for state vector engine using the dense `StateVecSoA` simulator.
///
/// `DenseStateVecEngine` uses:
/// - `SoA` (Structure of Arrays) layout for better SIMD performance
/// - Strided iteration for cache-efficient access patterns
/// - Fused gate primitives for reduced memory bandwidth
/// - Optional parallel execution for large state vectors
pub type DenseStateVecEngine = StateVectorEngine<StateVecSoA>;

impl DenseStateVecEngine {
    /// Create a new dense state vector engine with parallel execution enabled/disabled
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `parallel` - Whether to enable parallel execution for large states
    /// * `num_threads` - Number of threads for parallel execution (None = use Rayon's default)
    #[must_use]
    pub fn with_parallel(num_qubits: usize, parallel: bool, num_threads: Option<usize>) -> Self {
        let mut simulator = StateVecSoA::new(num_qubits);
        simulator.set_parallel(parallel);
        simulator.set_num_threads(num_threads);
        Self { simulator }
    }
}

impl<S: StateVectorSimulator + 'static> Engine for StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
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

        let mut measurements: Vec<usize> = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    debug!("Processing X gate on qubits {:?}", cmd.qubits);
                    self.simulator.x(&cmd.qubits);
                }
                GateType::Y => {
                    debug!("Processing Y gate on qubits {:?}", cmd.qubits);
                    self.simulator.y(&cmd.qubits);
                }
                GateType::Z => {
                    debug!("Processing Z gate on qubits {:?}", cmd.qubits);
                    self.simulator.z(&cmd.qubits);
                }
                GateType::H => {
                    debug!("Processing H gate on qubits {:?}", cmd.qubits);
                    self.simulator.h(&cmd.qubits);
                }
                GateType::SZ => {
                    debug!("Processing SZ gate on qubits {:?}", cmd.qubits);
                    self.simulator.sz(&cmd.qubits);
                }
                GateType::SZdg => {
                    debug!("Processing SZdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.szdg(&cmd.qubits);
                }
                GateType::SX => {
                    debug!("Processing SX gate on qubits {:?}", cmd.qubits);
                    self.simulator.sx(&cmd.qubits);
                }
                GateType::SXdg => {
                    debug!("Processing SXdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.sxdg(&cmd.qubits);
                }
                GateType::T => {
                    debug!("Processing T gate on qubits {:?}", cmd.qubits);
                    self.simulator.t(&cmd.qubits);
                }
                GateType::Tdg => {
                    debug!("Processing Tdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.tdg(&cmd.qubits);
                }
                GateType::CX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing CX gate on qubits {:?}", cmd.qubits);
                    self.simulator.cx(&cmd.qubits);
                }
                GateType::CY => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CY gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing CY gate on qubits {:?}", cmd.qubits);
                    self.simulator.cy(&cmd.qubits);
                }
                GateType::CZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing CZ gate on qubits {:?}", cmd.qubits);
                    self.simulator.cz(&cmd.qubits);
                }
                // CH = Ry(π/4)_target, CX(control, target), Ry(-π/4)_target
                GateType::CH => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CH gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing CH gate with control {:?} and target {:?}",
                            qubits[0], qubits[1]
                        );
                        let target_slice = &[qubits[1]];
                        self.simulator.ry(
                            Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                            target_slice,
                        );
                        self.simulator.cx(qubits);
                        self.simulator.ry(
                            Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                            target_slice,
                        );
                    }
                }
                GateType::RZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "RZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.is_empty() {
                        return Err(quantum_error("RZZ gate requires at least one angle"));
                    }
                    let angle = cmd.angles[0];
                    debug!("Processing RZZ gate on qubits {:?}", cmd.qubits);
                    self.simulator.rzz(angle, &cmd.qubits);
                }
                GateType::SZZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SZZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SZZ gate on qubits {:?}", cmd.qubits);
                    self.simulator.szz(&cmd.qubits);
                }
                GateType::SZZdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SZZdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SZZdg gate on qubits {:?}", cmd.qubits);
                    self.simulator.szzdg(&cmd.qubits);
                }
                GateType::SWAP => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "SWAP gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    debug!("Processing SWAP gate on qubits {:?}", cmd.qubits);
                    self.simulator.swap(&cmd.qubits);
                }
                GateType::CRZ => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(quantum_error(format!(
                            "CRZ gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    if cmd.angles.is_empty() {
                        return Err(quantum_error("CRZ gate requires at least one angle"));
                    }
                    let angle = cmd.angles[0];
                    let half_angle = angle / 2u64;
                    // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                    for qubits in cmd.qubits.chunks_exact(2) {
                        debug!(
                            "Processing CRZ gate on qubits {:?} and {:?} with angle {:?}",
                            qubits[0], qubits[1], angle
                        );
                        self.simulator.rz(half_angle, &[qubits[1]]);
                        self.simulator.cx(&[qubits[0], qubits[1]]);
                        self.simulator.rz(-half_angle, &[qubits[1]]);
                        self.simulator.cx(&[qubits[0], qubits[1]]);
                    }
                }
                GateType::CCX => {
                    if cmd.qubits.len() % 3 != 0 {
                        return Err(quantum_error(format!(
                            "CCX gate requires a multiple of 3 qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    for qubits in cmd.qubits.chunks_exact(3) {
                        debug!(
                            "Processing CCX gate with controls {:?}, {:?} and target {:?}",
                            qubits[0], qubits[1], qubits[2]
                        );
                        // Toffoli decomposition into Clifford+T gates
                        let c0 = qubits[0];
                        let c1 = qubits[1];
                        let target = qubits[2];
                        // Standard decomposition (15 gates)
                        self.simulator.h(&[target]);
                        self.simulator.cx(&[c1, target]);
                        self.simulator.tdg(&[target]);
                        self.simulator.cx(&[c0, target]);
                        self.simulator.t(&[target]);
                        self.simulator.cx(&[c1, target]);
                        self.simulator.tdg(&[target]);
                        self.simulator.cx(&[c0, target]);
                        self.simulator.t(&[c1]);
                        self.simulator.t(&[target]);
                        self.simulator.cx(&[c0, c1]);
                        self.simulator.h(&[target]);
                        self.simulator.t(&[c0]);
                        self.simulator.tdg(&[c1]);
                        self.simulator.cx(&[c0, c1]);
                    }
                }
                GateType::RX => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        debug!(
                            "Processing RX gate with angle {angle:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.rx(angle, &cmd.qubits);
                    }
                }
                GateType::RY => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        debug!(
                            "Processing RY gate with angle {angle:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.ry(angle, &cmd.qubits);
                    }
                }
                GateType::RZ => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        debug!(
                            "Processing RZ gate with angle {angle:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.rz(angle, &cmd.qubits);
                    }
                }
                GateType::R1XY => {
                    if cmd.angles.len() >= 2 {
                        let theta = cmd.angles[0];
                        let phi = cmd.angles[1];
                        debug!(
                            "Processing R1XY gate with angles theta={theta:?}, phi={phi:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.r1xy(theta, phi, &cmd.qubits);
                    }
                }

                // TODO: Fix it so we have multiple result_ids or get rid of result ids...
                GateType::Measure | GateType::MeasureLeaked => {
                    debug!("Processing measurement on qubits {:?}", cmd.qubits);
                    let meas_results = self.simulator.mz(&cmd.qubits);
                    for meas_result in meas_results {
                        // mz() outcome: true if projected to |1⟩, false if projected to |0⟩
                        measurements.push(usize::from(meas_result.outcome));
                    }
                }
                GateType::Prep => {
                    debug!("Processing Prep gate on qubits {:?}", cmd.qubits);
                    self.simulator.pz(&cmd.qubits);
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree => {
                    // Just let the system naturally evolve for the specified duration
                    // No active operation needed in the simulator
                    // QFree is a no-op for state vector simulation (qubit tracking is handled elsewhere)
                }
                GateType::SY | GateType::SYdg | GateType::RXX | GateType::RYY => {
                    return Err(quantum_error(format!(
                        "Gate type {:?} is not yet supported by StateVecEngine",
                        cmd.gate_type
                    )));
                }
                GateType::QAlloc => {
                    // Allocate qubits in |0⟩ state - for state vector sim, same as Prep
                    debug!("Processing QAlloc gate on qubits {:?}", cmd.qubits);
                    self.simulator.pz(&cmd.qubits);
                }
                GateType::MeasureFree => {
                    // Measure and deallocate - measure first, then the qubit is implicitly freed
                    debug!("Processing MeasureFree gate on qubits {:?}", cmd.qubits);
                    let meas_results = self.simulator.mz(&cmd.qubits);
                    for meas_result in meas_results {
                        measurements.push(usize::from(meas_result.outcome));
                    }
                }
                GateType::U => {
                    if cmd.angles.len() >= 3 {
                        let theta = cmd.angles[0];
                        let phi = cmd.angles[1];
                        let lambda = cmd.angles[2];
                        debug!(
                            "Processing U gate with angles theta={theta:?}, phi={phi:?}, lambda={lambda:?} on qubits {:?}",
                            cmd.qubits
                        );
                        self.simulator.u(theta, phi, lambda, &cmd.qubits);
                    }
                }
            }
        }

        // Create a message with the measurement results
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&measurements);

        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("StateVecEngine: reset() called");
        self.simulator.reset();
        Ok(())
    }
}

impl<S: StateVectorSimulator> RngManageable for StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    type Rng = <S as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl<S: StateVectorSimulator + 'static> QuantumEngine for StateVectorEngine<S>
where
    <S as RngManageable>::Rng: Clone,
{
    fn set_seed(&mut self, seed: u64) {
        let rng = <S as RngManageable>::Rng::seed_from_u64(seed);
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
    simulator: SparseStab,
}

impl SparseStabEngine {
    /// Create a new stabilizer engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: SparseStab::new(num_qubits),
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
            simulator: SparseStab::with_seed(num_qubits, seed),
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
                debug!("Processing X gate on qubits {qubits:?}");
                self.simulator.x(qubits);
            }
            GateType::Y => {
                debug!("Processing Y gate on qubits {qubits:?}");
                self.simulator.y(qubits);
            }
            GateType::Z => {
                debug!("Processing Z gate on qubits {qubits:?}");
                self.simulator.z(qubits);
            }
            GateType::H => {
                debug!("Processing H gate on qubits {qubits:?}");
                self.simulator.h(qubits);
            }
            GateType::SZ => {
                debug!("Processing SZ gate on qubits {qubits:?}");
                self.simulator.sz(qubits);
            }
            GateType::SZdg => {
                debug!("Processing SZdg gate on qubits {qubits:?}");
                self.simulator.szdg(qubits);
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
                debug!("Processing CX gate on qubits {qubits:?}");
                self.simulator.cx(qubits);
            }
            GateType::SZZ => {
                debug!("Processing SZZ gate on qubits {qubits:?}");
                self.simulator.szz(qubits);
            }
            GateType::SZZdg => {
                debug!("Processing SZZdg gate on qubits {qubits:?}");
                self.simulator.szzdg(qubits);
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
        let mut measurements: Vec<usize> = Vec::new();

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
                    debug!("Processing measurement on qubits {:?}", cmd.qubits);
                    let meas_results = self.simulator.mz(&cmd.qubits);
                    for meas_result in meas_results {
                        // mz() outcome: true if projected to |1⟩, false if projected to |0⟩
                        measurements.push(usize::from(meas_result.outcome));
                    }
                }
                GateType::Prep => {
                    debug!("Processing Prep gate on qubits {:?}", cmd.qubits);
                    self.simulator.pz(&cmd.qubits);
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
        builder.add_outcomes(&measurements);

        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.simulator.reset();
        Ok(())
    }
}

impl RngManageable for SparseStabEngine {
    type Rng = <SparseStab as RngManageable>::Rng;

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
        let rng = <SparseStab as RngManageable>::Rng::seed_from_u64(seed);

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

/// A quantum engine that uses a coin toss simulator
///
/// This engine ignores all quantum gates and returns random measurement results.
/// Useful for testing classical control logic without quantum overhead.
#[derive(Debug, Clone)]
pub struct CoinTossEngine {
    simulator: CoinToss,
}

impl CoinTossEngine {
    /// Create a new coin toss engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: CoinToss::new(num_qubits),
        }
    }

    /// Create a new coin toss engine with a specific seed
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: CoinToss::with_seed(num_qubits, Some(seed)),
        }
    }

    /// Create a new coin toss engine with custom probability
    #[must_use]
    pub fn with_prob(num_qubits: usize, prob: f64) -> Self {
        Self {
            simulator: CoinToss::with_prob(num_qubits, prob),
        }
    }

    /// Create a new coin toss engine with custom probability and seed
    #[must_use]
    pub fn with_prob_and_seed(num_qubits: usize, prob: f64, seed: u64) -> Self {
        Self {
            simulator: CoinToss::with_prob_and_seed(num_qubits, prob, Some(seed)),
        }
    }
}

impl Engine for CoinTossEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        let batch = message.quantum_ops()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                // All gates are no-ops for CoinToss - only measurements matter
                GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                    for q in &cmd.qubits {
                        debug!("CoinToss: Processing measurement on qubit {q:?}");
                        let meas_results = self.simulator.mz(&[*q]);
                        for meas_result in &meas_results {
                            let outcome = u32::from(meas_result.outcome);
                            measurements.push(outcome);
                        }
                    }
                }
                // All other gates are ignored
                _ => {}
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

impl RngManageable for CoinTossEngine {
    type Rng = <CoinToss as RngManageable>::Rng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.simulator.set_rng(rng);
    }

    fn rng(&self) -> &Self::Rng {
        self.simulator.rng()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.simulator.rng_mut()
    }
}

impl QuantumEngine for CoinTossEngine {
    fn set_seed(&mut self, seed: u64) {
        self.simulator.set_seed(seed);
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
