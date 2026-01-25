//! Quest quantum engine integration with PECOS engine system
//!
//! This module provides wrappers and builders to integrate `QuEST` simulators
//! with the PECOS engine system, allowing them to be used with the `sim()` API.

use crate::{QuestDensityMatrix, QuestStateVec};
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_engines::{
    Engine, IntoQuantumEngineBuilder, QuantumEngine, QuantumEngineBuilder,
    byte_message::{ByteMessage, GateType},
};
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use std::any::Any;
use std::fmt::Debug;

/// Quest state vector quantum engine wrapper
#[derive(Debug, Clone)]
pub struct QuestStateVecEngine {
    simulator: QuestStateVec,
}

impl QuestStateVecEngine {
    /// Create a new Quest state vector engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: QuestStateVec::new(num_qubits),
        }
    }

    /// Create a new Quest state vector engine with a specific seed
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: QuestStateVec::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for QuestStateVecEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        // Parse commands from the message
        let batch = message.quantum_ops()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    for q in &cmd.qubits {
                        self.simulator.x(usize::from(*q));
                    }
                }
                GateType::Y => {
                    for q in &cmd.qubits {
                        self.simulator.y(usize::from(*q));
                    }
                }
                GateType::Z => {
                    for q in &cmd.qubits {
                        self.simulator.z(usize::from(*q));
                    }
                }
                GateType::H => {
                    for q in &cmd.qubits {
                        self.simulator.h(usize::from(*q));
                    }
                }
                GateType::SZ => {
                    for q in &cmd.qubits {
                        self.simulator.sz(usize::from(*q));
                    }
                }
                GateType::SZdg => {
                    for q in &cmd.qubits {
                        self.simulator.szdg(usize::from(*q));
                    }
                }
                GateType::T => {
                    for q in &cmd.qubits {
                        self.simulator.t(usize::from(*q));
                    }
                }
                GateType::Tdg => {
                    for q in &cmd.qubits {
                        self.simulator.tdg(usize::from(*q));
                    }
                }
                GateType::CX => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .cx(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::CY => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .cy(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::CZ => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .cz(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                // CH = Ry(π/4)_target, CX(control, target), Ry(-π/4)_target
                GateType::CH => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let control = usize::from(qubits[0]);
                        let target = usize::from(qubits[1]);
                        self.simulator.ry(std::f64::consts::FRAC_PI_4, target);
                        self.simulator.cx(control, target);
                        self.simulator.ry(-std::f64::consts::FRAC_PI_4, target);
                    }
                }
                GateType::RZZ => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator.rzz(cmd.params[0], *qubits[0], *qubits[1]);
                    }
                }
                GateType::SZZ => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .szz(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::SZZdg => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .szzdg(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::SWAP => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        // SWAP = CX(0,1) CX(1,0) CX(0,1)
                        let q0 = usize::from(qubits[0]);
                        let q1 = usize::from(qubits[1]);
                        self.simulator.cx(q0, q1);
                        self.simulator.cx(q1, q0);
                        self.simulator.cx(q0, q1);
                    }
                }
                GateType::CRZ => {
                    if !cmd.params.is_empty() {
                        let angle = cmd.params[0];
                        let half_angle = angle / 2.0;
                        for qubits in cmd.qubits.chunks_exact(2) {
                            // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                            let control = usize::from(qubits[0]);
                            let target = usize::from(qubits[1]);
                            self.simulator.rz(half_angle, target);
                            self.simulator.cx(control, target);
                            self.simulator.rz(-half_angle, target);
                            self.simulator.cx(control, target);
                        }
                    }
                }
                GateType::CCX => {
                    for qubits in cmd.qubits.chunks_exact(3) {
                        // Toffoli decomposition into Clifford+T gates
                        let c0 = usize::from(qubits[0]);
                        let c1 = usize::from(qubits[1]);
                        let target = usize::from(qubits[2]);
                        self.simulator.h(target);
                        self.simulator.cx(c1, target);
                        self.simulator.tdg(target);
                        self.simulator.cx(c0, target);
                        self.simulator.t(target);
                        self.simulator.cx(c1, target);
                        self.simulator.tdg(target);
                        self.simulator.cx(c0, target);
                        self.simulator.t(c1);
                        self.simulator.t(target);
                        self.simulator.cx(c0, c1);
                        self.simulator.h(target);
                        self.simulator.t(c0);
                        self.simulator.tdg(c1);
                        self.simulator.cx(c0, c1);
                    }
                }
                GateType::SX => {
                    for q in &cmd.qubits {
                        self.simulator.sx(usize::from(*q));
                    }
                }
                GateType::SXdg => {
                    for q in &cmd.qubits {
                        self.simulator.sxdg(usize::from(*q));
                    }
                }
                GateType::RX => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            self.simulator.rx(cmd.params[0], **q);
                        }
                    }
                }
                GateType::RY => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            self.simulator.ry(cmd.params[0], **q);
                        }
                    }
                }
                GateType::RZ => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            self.simulator.rz(cmd.params[0], **q);
                        }
                    }
                }
                GateType::R1XY => {
                    if cmd.params.len() >= 2 {
                        for q in &cmd.qubits {
                            self.simulator.r1xy(cmd.params[0], cmd.params[1], **q);
                        }
                    }
                }
                GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                    for q in &cmd.qubits {
                        let meas_result = self.simulator.mz(**q);
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep | GateType::QAlloc => {
                    for q in &cmd.qubits {
                        self.simulator.pz(**q);
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree => {
                    // No operation needed (QFree is just a marker for qubit lifecycle)
                }
                GateType::U => {
                    if cmd.params.len() >= 3 {
                        for q in &cmd.qubits {
                            self.simulator
                                .u(cmd.params[0], cmd.params[1], cmd.params[2], **q);
                        }
                    }
                }
                GateType::SY | GateType::SYdg | GateType::RXX | GateType::RYY => {
                    return Err(PecosError::Processing(format!(
                        "Gate type {:?} is not yet supported by QuestStateVecEngine",
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

impl QuantumEngine for QuestStateVecEngine {
    fn set_seed(&mut self, seed: u64) {
        let rng = <QuestStateVec as RngManageable>::Rng::seed_from_u64(seed);
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Quest density matrix quantum engine wrapper
#[derive(Debug, Clone)]
pub struct QuestDensityMatrixEngine {
    simulator: QuestDensityMatrix,
}

impl QuestDensityMatrixEngine {
    /// Create a new Quest density matrix engine with the specified number of qubits
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: QuestDensityMatrix::new(num_qubits),
        }
    }

    /// Create a new Quest density matrix engine with a specific seed
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            simulator: QuestDensityMatrix::with_seed(num_qubits, seed),
        }
    }
}

impl Engine for QuestDensityMatrixEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        // Parse commands from the message
        let batch = message.quantum_ops()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    for q in &cmd.qubits {
                        self.simulator.x(usize::from(*q));
                    }
                }
                GateType::Y => {
                    for q in &cmd.qubits {
                        self.simulator.y(usize::from(*q));
                    }
                }
                GateType::Z => {
                    for q in &cmd.qubits {
                        self.simulator.z(usize::from(*q));
                    }
                }
                GateType::H => {
                    for q in &cmd.qubits {
                        self.simulator.h(usize::from(*q));
                    }
                }
                GateType::SZ => {
                    for q in &cmd.qubits {
                        self.simulator.sz(usize::from(*q));
                    }
                }
                GateType::SZdg => {
                    for q in &cmd.qubits {
                        self.simulator.szdg(usize::from(*q));
                    }
                }
                GateType::T => {
                    for q in &cmd.qubits {
                        self.simulator.t(usize::from(*q));
                    }
                }
                GateType::Tdg => {
                    for q in &cmd.qubits {
                        self.simulator.tdg(usize::from(*q));
                    }
                }
                GateType::CX => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .cx(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::CY => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .cy(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::CZ => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .cz(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                // CH = Ry(π/4)_target, CX(control, target), Ry(-π/4)_target
                GateType::CH => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let control = usize::from(qubits[0]);
                        let target = usize::from(qubits[1]);
                        self.simulator.ry(std::f64::consts::FRAC_PI_4, target);
                        self.simulator.cx(control, target);
                        self.simulator.ry(-std::f64::consts::FRAC_PI_4, target);
                    }
                }
                GateType::RZZ => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator.rzz(cmd.params[0], *qubits[0], *qubits[1]);
                    }
                }
                GateType::SZZ => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .szz(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::SZZdg => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        self.simulator
                            .szzdg(usize::from(qubits[0]), usize::from(qubits[1]));
                    }
                }
                GateType::SWAP => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        // SWAP = CX(0,1) CX(1,0) CX(0,1)
                        let q0 = usize::from(qubits[0]);
                        let q1 = usize::from(qubits[1]);
                        self.simulator.cx(q0, q1);
                        self.simulator.cx(q1, q0);
                        self.simulator.cx(q0, q1);
                    }
                }
                GateType::CRZ => {
                    if !cmd.params.is_empty() {
                        let angle = cmd.params[0];
                        let half_angle = angle / 2.0;
                        for qubits in cmd.qubits.chunks_exact(2) {
                            // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                            let control = usize::from(qubits[0]);
                            let target = usize::from(qubits[1]);
                            self.simulator.rz(half_angle, target);
                            self.simulator.cx(control, target);
                            self.simulator.rz(-half_angle, target);
                            self.simulator.cx(control, target);
                        }
                    }
                }
                GateType::CCX => {
                    for qubits in cmd.qubits.chunks_exact(3) {
                        // Toffoli decomposition into Clifford+T gates
                        let c0 = usize::from(qubits[0]);
                        let c1 = usize::from(qubits[1]);
                        let target = usize::from(qubits[2]);
                        self.simulator.h(target);
                        self.simulator.cx(c1, target);
                        self.simulator.tdg(target);
                        self.simulator.cx(c0, target);
                        self.simulator.t(target);
                        self.simulator.cx(c1, target);
                        self.simulator.tdg(target);
                        self.simulator.cx(c0, target);
                        self.simulator.t(c1);
                        self.simulator.t(target);
                        self.simulator.cx(c0, c1);
                        self.simulator.h(target);
                        self.simulator.t(c0);
                        self.simulator.tdg(c1);
                        self.simulator.cx(c0, c1);
                    }
                }
                GateType::SX => {
                    for q in &cmd.qubits {
                        self.simulator.sx(usize::from(*q));
                    }
                }
                GateType::SXdg => {
                    for q in &cmd.qubits {
                        self.simulator.sxdg(usize::from(*q));
                    }
                }
                GateType::RX => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            self.simulator.rx(cmd.params[0], **q);
                        }
                    }
                }
                GateType::RY => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            self.simulator.ry(cmd.params[0], **q);
                        }
                    }
                }
                GateType::RZ => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            self.simulator.rz(cmd.params[0], **q);
                        }
                    }
                }
                GateType::R1XY => {
                    if cmd.params.len() >= 2 {
                        for q in &cmd.qubits {
                            self.simulator.r1xy(cmd.params[0], cmd.params[1], **q);
                        }
                    }
                }
                GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                    for q in &cmd.qubits {
                        let meas_result = self.simulator.mz(**q);
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep | GateType::QAlloc => {
                    for q in &cmd.qubits {
                        self.simulator.pz(**q);
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree => {
                    // No operation needed (QFree is just a marker for qubit lifecycle)
                }
                GateType::U => {
                    if cmd.params.len() >= 3 {
                        for q in &cmd.qubits {
                            self.simulator
                                .u(cmd.params[0], cmd.params[1], cmd.params[2], **q);
                        }
                    }
                }
                GateType::SY | GateType::SYdg | GateType::RXX | GateType::RYY => {
                    return Err(PecosError::Processing(format!(
                        "Gate type {:?} is not yet supported by QuestDensityMatrixEngine",
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

impl QuantumEngine for QuestDensityMatrixEngine {
    fn set_seed(&mut self, seed: u64) {
        let rng = <QuestDensityMatrix as RngManageable>::Rng::seed_from_u64(seed);
        self.simulator.set_rng(rng);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Builder for Quest state vector quantum engine
#[derive(Debug, Clone, Default)]
pub struct QuestStateVectorEngineBuilder {
    /// Number of qubits (if explicitly set)
    num_qubits: Option<usize>,
    /// CUDA acceleration mode flag
    #[allow(dead_code)]
    use_cuda: bool,
}

impl QuestStateVectorEngineBuilder {
    /// Create a new Quest state vector engine builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of qubits
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = Some(num_qubits);
        self
    }

    /// Use CPU-only mode (default)
    #[must_use]
    pub fn with_cpu(mut self) -> Self {
        self.use_cuda = false;
        self
    }

    /// Use GPU acceleration mode
    ///
    /// This enables GPU acceleration using the best available backend.
    /// Currently supports NVIDIA CUDA via the `QuEST` CUDA backend.
    /// The backend is loaded at runtime, so systems without GPU support
    /// can still use the CPU mode.
    ///
    /// # Panics
    /// Panics if the `cuda` feature is not enabled at compile time
    #[must_use]
    pub fn with_gpu(self) -> Self {
        #[cfg(not(feature = "cuda"))]
        {
            panic!(
                "GPU feature is not enabled. Rebuild with --features cuda to use GPU acceleration"
            );
        }
        #[cfg(feature = "cuda")]
        {
            Self {
                use_cuda: true,
                ..self
            }
        }
    }
}

impl QuantumEngineBuilder for QuestStateVectorEngineBuilder {
    fn build(&mut self) -> Result<Box<dyn QuantumEngine>, PecosError> {
        let num_qubits = self.num_qubits.ok_or_else(|| {
            PecosError::Input("Number of qubits not specified for Quest engine".to_string())
        })?;

        // Check if CUDA was requested
        #[cfg(feature = "cuda")]
        if self.use_cuda {
            // Create and return CUDA-backed engine
            let engine = QuestCudaStateVecEngine::new(num_qubits)?;
            return Ok(Box::new(engine));
        }

        #[cfg(not(feature = "cuda"))]
        if self.use_cuda {
            return Err(PecosError::Processing(
                "CUDA acceleration requested but 'cuda' feature is not enabled. \
                 Rebuild with --features cuda to use GPU acceleration."
                    .to_string(),
            ));
        }

        // CPU mode - use the standard implementation
        Ok(Box::new(QuestStateVecEngine::new(num_qubits)))
    }

    fn set_qubits_if_needed(&mut self, num_qubits: usize) {
        if self.num_qubits.is_none() {
            self.num_qubits = Some(num_qubits);
        }
    }
}

impl IntoQuantumEngineBuilder for QuestStateVectorEngineBuilder {
    type Builder = Self;

    fn into_quantum_engine_builder(self) -> Self::Builder {
        self
    }
}

/// Builder for Quest density matrix quantum engine
#[derive(Debug, Clone, Default)]
pub struct QuestDensityMatrixEngineBuilder {
    /// Number of qubits (if explicitly set)
    num_qubits: Option<usize>,
    /// CUDA acceleration mode flag
    #[allow(dead_code)]
    use_cuda: bool,
}

impl QuestDensityMatrixEngineBuilder {
    /// Create a new Quest density matrix engine builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of qubits
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = Some(num_qubits);
        self
    }

    /// Use CPU-only mode (default)
    #[must_use]
    pub fn with_cpu(mut self) -> Self {
        self.use_cuda = false;
        self
    }

    /// Use GPU acceleration mode
    ///
    /// This enables GPU acceleration using the best available backend.
    /// Currently supports NVIDIA CUDA via the `QuEST` CUDA backend.
    /// The backend is loaded at runtime, so systems without GPU support
    /// can still use the CPU mode.
    ///
    /// # Panics
    /// Panics if the `cuda` feature is not enabled at compile time
    #[must_use]
    pub fn with_gpu(self) -> Self {
        #[cfg(not(feature = "cuda"))]
        {
            panic!(
                "GPU feature is not enabled. Rebuild with --features cuda to use GPU acceleration"
            );
        }
        #[cfg(feature = "cuda")]
        {
            Self {
                use_cuda: true,
                ..self
            }
        }
    }
}

impl QuantumEngineBuilder for QuestDensityMatrixEngineBuilder {
    fn build(&mut self) -> Result<Box<dyn QuantumEngine>, PecosError> {
        let num_qubits = self.num_qubits.ok_or_else(|| {
            PecosError::Input("Number of qubits not specified for Quest engine".to_string())
        })?;

        // Check if CUDA was requested
        if self.use_cuda {
            // CUDA density matrix engine not yet implemented
            return Err(PecosError::Processing(
                "CUDA acceleration for density matrix simulation is not yet implemented. \
                 Use QuestStateVectorEngineBuilder for GPU-accelerated state vector simulation, \
                 or use CPU mode for density matrix simulation."
                    .to_string(),
            ));
        }

        // CPU mode - use the standard implementation
        Ok(Box::new(QuestDensityMatrixEngine::new(num_qubits)))
    }

    fn set_qubits_if_needed(&mut self, num_qubits: usize) {
        if self.num_qubits.is_none() {
            self.num_qubits = Some(num_qubits);
        }
    }
}

impl IntoQuantumEngineBuilder for QuestDensityMatrixEngineBuilder {
    type Builder = Self;

    fn into_quantum_engine_builder(self) -> Self::Builder {
        self
    }
}

/// Create a Quest state vector quantum engine builder
#[must_use]
pub fn quest_state_vec() -> QuestStateVectorEngineBuilder {
    QuestStateVectorEngineBuilder::new()
}

/// Create a Quest density matrix quantum engine builder
#[must_use]
pub fn quest_density_matrix() -> QuestDensityMatrixEngineBuilder {
    QuestDensityMatrixEngineBuilder::new()
}

// ============================================================================
// CUDA-backed quantum engine
// ============================================================================

/// CUDA-backed `QuEST` state vector quantum engine
///
/// This engine uses the dynamically-loaded `QuEST` CUDA backend for GPU-accelerated
/// quantum simulation. The CUDA backend is loaded at runtime via dlopen, allowing
/// the same binary to work on systems with and without CUDA installed.
#[cfg(feature = "cuda")]
pub struct QuestCudaStateVecEngine {
    /// Opaque handle to the `QuEST` environment (owned by CUDA backend)
    env_handle: *mut u8,
    /// Opaque handle to the quantum register (owned by CUDA backend)
    qureg_handle: *mut u8,
    /// Reference to the CUDA backend (static lifetime, lazily loaded)
    backend: &'static crate::cuda_loader::CudaBackend,
    /// Number of qubits
    num_qubits: usize,
}

#[cfg(feature = "cuda")]
impl QuestCudaStateVecEngine {
    /// Create a new CUDA-backed state vector engine
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The CUDA backend library cannot be loaded
    /// - The CUDA environment cannot be created
    /// - The quantum register cannot be allocated
    ///
    /// # Panics
    /// Panics if `num_qubits` exceeds `i32::MAX` (extremely unlikely in practice).
    pub fn new(num_qubits: usize) -> Result<Self, PecosError> {
        let backend = crate::cuda_loader::try_load_cuda().map_err(|e| {
            PecosError::Processing(format!(
                "Failed to load CUDA backend: {e}\n\n{}",
                crate::cuda_loader::cuda_unavailable_error_message()
            ))
        })?;

        // Create environment
        let env_handle = unsafe { (backend.create_env)() };
        if env_handle.is_null() {
            return Err(PecosError::Processing(
                "Failed to create CUDA QuEST environment".to_string(),
            ));
        }

        // Create quantum register
        let qureg_handle =
            unsafe { (backend.create_qureg)(env_handle, i32::try_from(num_qubits).unwrap()) };
        if qureg_handle.is_null() {
            unsafe {
                (backend.destroy_env)(env_handle);
            }
            return Err(PecosError::Processing(format!(
                "Failed to create CUDA quantum register with {num_qubits} qubits"
            )));
        }

        // Initialize to zero state
        unsafe {
            (backend.init_zero_state)(qureg_handle);
        }

        log::info!("Created CUDA-backed QuEST state vector engine with {num_qubits} qubits");

        Ok(Self {
            env_handle,
            qureg_handle,
            backend,
            num_qubits,
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for QuestCudaStateVecEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.qureg_handle.is_null() {
                (self.backend.destroy_qureg)(self.qureg_handle);
            }
            if !self.env_handle.is_null() {
                (self.backend.destroy_env)(self.env_handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl Debug for QuestCudaStateVecEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuestCudaStateVecEngine")
            .field("num_qubits", &self.num_qubits)
            .finish_non_exhaustive()
    }
}

// Safety: The CUDA backend handles are thread-safe through QuEST's internal synchronization
#[cfg(feature = "cuda")]
unsafe impl Send for QuestCudaStateVecEngine {}
#[cfg(feature = "cuda")]
unsafe impl Sync for QuestCudaStateVecEngine {}

#[cfg(feature = "cuda")]
impl Clone for QuestCudaStateVecEngine {
    /// Clone creates a new CUDA engine with the same configuration but reset to zero state.
    ///
    /// Note: This does NOT preserve the quantum state of the original engine.
    /// Cloning GPU resources is expensive, so this creates a fresh engine.
    fn clone(&self) -> Self {
        Self::new(self.num_qubits).expect("Failed to clone CUDA engine")
    }
}

#[cfg(feature = "cuda")]
impl Engine for QuestCudaStateVecEngine {
    type Input = ByteMessage;
    type Output = ByteMessage;

    // Allow cast warnings: qubit indices are always small (quantum computers don't have billions of qubits)
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        let batch = message.quantum_ops()?;
        let mut measurements = Vec::new();

        for cmd in &batch {
            match cmd.gate_type {
                GateType::X => {
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_pauli_x)(self.qureg_handle, qubit);
                        }
                    }
                }
                GateType::Y => {
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_pauli_y)(self.qureg_handle, qubit);
                        }
                    }
                }
                GateType::Z => {
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_pauli_z)(self.qureg_handle, qubit);
                        }
                    }
                }
                GateType::H => {
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_hadamard)(self.qureg_handle, qubit);
                        }
                    }
                }
                GateType::SZ => {
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_s_gate)(self.qureg_handle, qubit);
                        }
                    }
                }
                GateType::SZdg => {
                    // S-dagger = S^3 = phase(-pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                qubit,
                                -std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::T => {
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_t_gate)(self.qureg_handle, qubit);
                        }
                    }
                }
                GateType::Tdg => {
                    // T-dagger = T^7 = phase(-pi/4)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                qubit,
                                -std::f64::consts::FRAC_PI_4,
                            );
                        }
                    }
                }
                GateType::CX => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (ctrl, tgt) =
                            (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, ctrl, tgt);
                        }
                    }
                }
                GateType::CY => {
                    // CY = (I ⊗ S†) · CX · (I ⊗ S) = Controlled-Y
                    // Decompose as: S†(target) · CX · S(target)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (ctrl, tgt) =
                            (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            // S†(tgt) = phase(-pi/2)
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                tgt,
                                -std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, ctrl, tgt);
                            // S(tgt) = phase(pi/2)
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                tgt,
                                std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::CZ => {
                    // CZ = H(target) · CX · H(target)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (ctrl, tgt) =
                            (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_hadamard)(self.qureg_handle, tgt);
                            (self.backend.apply_cnot)(self.qureg_handle, ctrl, tgt);
                            (self.backend.apply_hadamard)(self.qureg_handle, tgt);
                        }
                    }
                }
                // CH = Ry(π/4)_target · CX · Ry(-π/4)_target
                GateType::CH => {
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (ctrl, tgt) =
                            (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_rotation_y)(
                                self.qureg_handle,
                                tgt,
                                std::f64::consts::FRAC_PI_4,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, ctrl, tgt);
                            (self.backend.apply_rotation_y)(
                                self.qureg_handle,
                                tgt,
                                -std::f64::consts::FRAC_PI_4,
                            );
                        }
                    }
                }
                GateType::RX => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            unsafe {
                                (self.backend.apply_rotation_x)(
                                    self.qureg_handle,
                                    qubit,
                                    cmd.params[0],
                                );
                            }
                        }
                    }
                }
                GateType::RY => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            unsafe {
                                (self.backend.apply_rotation_y)(
                                    self.qureg_handle,
                                    qubit,
                                    cmd.params[0],
                                );
                            }
                        }
                    }
                }
                GateType::RZ => {
                    if !cmd.params.is_empty() {
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            unsafe {
                                (self.backend.apply_rotation_z)(
                                    self.qureg_handle,
                                    qubit,
                                    cmd.params[0],
                                );
                            }
                        }
                    }
                }
                GateType::RZZ => {
                    // RZZ(theta) = exp(-i * theta/2 * Z_a Z_b)
                    // Decompose as: CNOT(a,b) - RZ(theta, b) - CNOT(a,b)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (*qubits[0] as i32, *qubits[1] as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_z)(self.qureg_handle, b, cmd.params[0]);
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::SZZ => {
                    // SZZ = RZZ(pi/2) = exp(-i * pi/4 * Z_a Z_b)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_z)(
                                self.qureg_handle,
                                b,
                                std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::SZZdg => {
                    // SZZdg = RZZ(-pi/2) = exp(i * pi/4 * Z_a Z_b)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_z)(
                                self.qureg_handle,
                                b,
                                -std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::R1XY => {
                    // R1XY(theta, phi) gate
                    // Decompose as: RZ(-phi) - RX(theta) - RZ(phi)
                    if cmd.params.len() >= 2 {
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            let (theta, phi) = (cmd.params[0], cmd.params[1]);
                            unsafe {
                                (self.backend.apply_rotation_z)(self.qureg_handle, qubit, -phi);
                                (self.backend.apply_rotation_x)(self.qureg_handle, qubit, theta);
                                (self.backend.apply_rotation_z)(self.qureg_handle, qubit, phi);
                            }
                        }
                    }
                }
                GateType::U => {
                    // U(theta, phi, lambda) = RZ(phi) - RY(theta) - RZ(lambda)
                    if cmd.params.len() >= 3 {
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            let (theta, phi, lambda) =
                                (cmd.params[0], cmd.params[1], cmd.params[2]);
                            unsafe {
                                (self.backend.apply_rotation_z)(self.qureg_handle, qubit, lambda);
                                (self.backend.apply_rotation_y)(self.qureg_handle, qubit, theta);
                                (self.backend.apply_rotation_z)(self.qureg_handle, qubit, phi);
                            }
                        }
                    }
                }
                GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                    for q in &cmd.qubits {
                        let qubit = **q as i32;
                        let outcome = unsafe { (self.backend.measure)(self.qureg_handle, qubit) };
                        measurements.push(u32::try_from(outcome).unwrap());
                    }
                }
                GateType::Prep | GateType::QAlloc => {
                    // Prepare in |0> state: measure and flip if result is 1
                    for q in &cmd.qubits {
                        let qubit = **q as i32;
                        let outcome = unsafe { (self.backend.measure)(self.qureg_handle, qubit) };
                        if outcome == 1 {
                            unsafe {
                                (self.backend.apply_pauli_x)(self.qureg_handle, qubit);
                            }
                        }
                    }
                }
                GateType::SWAP => {
                    // SWAP = CX(0,1) CX(1,0) CX(0,1)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (q0, q1) =
                            (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, q0, q1);
                            (self.backend.apply_cnot)(self.qureg_handle, q1, q0);
                            (self.backend.apply_cnot)(self.qureg_handle, q0, q1);
                        }
                    }
                }
                GateType::CRZ => {
                    // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                    if !cmd.params.is_empty() {
                        let angle = cmd.params[0];
                        let half_angle = angle / 2.0;
                        for qubits in cmd.qubits.chunks_exact(2) {
                            let (control, target) =
                                (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                            unsafe {
                                (self.backend.apply_rotation_z)(
                                    self.qureg_handle,
                                    target,
                                    half_angle,
                                );
                                (self.backend.apply_cnot)(self.qureg_handle, control, target);
                                (self.backend.apply_rotation_z)(
                                    self.qureg_handle,
                                    target,
                                    -half_angle,
                                );
                                (self.backend.apply_cnot)(self.qureg_handle, control, target);
                            }
                        }
                    }
                }
                GateType::CCX => {
                    // Toffoli decomposition into Clifford+T gates
                    for qubits in cmd.qubits.chunks_exact(3) {
                        let c0 = usize::from(qubits[0]) as i32;
                        let c1 = usize::from(qubits[1]) as i32;
                        let target = usize::from(qubits[2]) as i32;
                        unsafe {
                            (self.backend.apply_hadamard)(self.qureg_handle, target);
                            (self.backend.apply_cnot)(self.qureg_handle, c1, target);
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                target,
                                -std::f64::consts::FRAC_PI_4,
                            ); // Tdg
                            (self.backend.apply_cnot)(self.qureg_handle, c0, target);
                            (self.backend.apply_t_gate)(self.qureg_handle, target);
                            (self.backend.apply_cnot)(self.qureg_handle, c1, target);
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                target,
                                -std::f64::consts::FRAC_PI_4,
                            ); // Tdg
                            (self.backend.apply_cnot)(self.qureg_handle, c0, target);
                            (self.backend.apply_t_gate)(self.qureg_handle, c1);
                            (self.backend.apply_t_gate)(self.qureg_handle, target);
                            (self.backend.apply_cnot)(self.qureg_handle, c0, c1);
                            (self.backend.apply_hadamard)(self.qureg_handle, target);
                            (self.backend.apply_t_gate)(self.qureg_handle, c0);
                            (self.backend.apply_phase_shift)(
                                self.qureg_handle,
                                c1,
                                -std::f64::consts::FRAC_PI_4,
                            ); // Tdg
                            (self.backend.apply_cnot)(self.qureg_handle, c0, c1);
                        }
                    }
                }
                GateType::SX => {
                    // SX = RX(pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_rotation_x)(
                                self.qureg_handle,
                                qubit,
                                std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::SXdg => {
                    // SXdg = RX(-pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_rotation_x)(
                                self.qureg_handle,
                                qubit,
                                -std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree => {
                    // No operation needed (QFree is just a marker for qubit lifecycle)
                }
                GateType::SY | GateType::SYdg | GateType::RXX | GateType::RYY => {
                    return Err(PecosError::Processing(format!(
                        "Gate type {:?} is not yet supported by QuestCudaStateVecEngine",
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
        unsafe {
            (self.backend.init_zero_state)(self.qureg_handle);
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl QuantumEngine for QuestCudaStateVecEngine {
    fn set_seed(&mut self, _seed: u64) {
        // CUDA backend doesn't currently support seeding via the loaded library
        // The seed would need to be passed to QuEST's internal RNG
        log::warn!("set_seed not yet implemented for CUDA backend");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
