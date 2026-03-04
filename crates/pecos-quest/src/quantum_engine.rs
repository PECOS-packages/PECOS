//! Quest quantum engine integration with PECOS engine system
//!
//! This module provides wrappers and builders to integrate `QuEST` simulators
//! with the PECOS engine system, allowing them to be used with the `sim()` API.

use crate::{QuestDensityMatrix, QuestStateVec};
use pecos_core::Angle64;
#[cfg(feature = "cuda")]
use pecos_core::QubitId;
use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_engines::{
    Engine, IntoQuantumEngineBuilder, QuantumEngine, QuantumEngineBuilder,
    byte_message::{ByteMessage, GateType},
};
#[cfg(feature = "cuda")]
use pecos_qsim::MeasurementResult;
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
                    self.simulator.x(&cmd.qubits);
                }
                GateType::Y => {
                    self.simulator.y(&cmd.qubits);
                }
                GateType::Z => {
                    self.simulator.z(&cmd.qubits);
                }
                GateType::H => {
                    self.simulator.h(&cmd.qubits);
                }
                GateType::SZ => {
                    self.simulator.sz(&cmd.qubits);
                }
                GateType::SZdg => {
                    self.simulator.szdg(&cmd.qubits);
                }
                GateType::T => {
                    self.simulator.t(&cmd.qubits);
                }
                GateType::Tdg => {
                    self.simulator.tdg(&cmd.qubits);
                }
                GateType::CX => {
                    self.simulator.cx(&cmd.qubits);
                }
                GateType::CY => {
                    self.simulator.cy(&cmd.qubits);
                }
                GateType::CZ => {
                    self.simulator.cz(&cmd.qubits);
                }
                // CH = Ry(π/4)_target, CX(control, target), Ry(-π/4)_target
                GateType::CH => {
                    for qubits in cmd.qubits.chunks_exact(2) {
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
                    self.simulator.rzz(cmd.angles[0], &cmd.qubits);
                }
                GateType::SZZ => {
                    self.simulator.szz(&cmd.qubits);
                }
                GateType::SZZdg => {
                    self.simulator.szzdg(&cmd.qubits);
                }
                GateType::F => {
                    self.simulator.f(&cmd.qubits);
                }
                GateType::Fdg => {
                    self.simulator.fdg(&cmd.qubits);
                }
                GateType::SY => {
                    self.simulator.sy(&cmd.qubits);
                }
                GateType::SYdg => {
                    self.simulator.sydg(&cmd.qubits);
                }
                GateType::SXX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SXX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.sxx(&cmd.qubits);
                }
                GateType::SXXdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SXXdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.sxxdg(&cmd.qubits);
                }
                GateType::SYY => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SYY gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.syy(&cmd.qubits);
                }
                GateType::SYYdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SYYdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.syydg(&cmd.qubits);
                }
                GateType::SWAP => {
                    self.simulator.swap(&cmd.qubits);
                }
                GateType::CRZ => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        let half_angle = angle / 2u64;
                        for pair in cmd.qubits.chunks_exact(2) {
                            // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                            self.simulator.rz(half_angle, &[pair[1]]);
                            self.simulator.cx(pair);
                            self.simulator.rz(-half_angle, &[pair[1]]);
                            self.simulator.cx(pair);
                        }
                    }
                }
                GateType::CCX => {
                    for qubits in cmd.qubits.chunks_exact(3) {
                        // Toffoli decomposition into Clifford+T gates
                        let c0 = qubits[0];
                        let c1 = qubits[1];
                        let target = qubits[2];
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
                GateType::SX => {
                    self.simulator.sx(&cmd.qubits);
                }
                GateType::SXdg => {
                    self.simulator.sxdg(&cmd.qubits);
                }
                GateType::RX => {
                    if !cmd.angles.is_empty() {
                        self.simulator.rx(cmd.angles[0], &cmd.qubits);
                    }
                }
                GateType::RY => {
                    if !cmd.angles.is_empty() {
                        self.simulator.ry(cmd.angles[0], &cmd.qubits);
                    }
                }
                GateType::RZ => {
                    if !cmd.angles.is_empty() {
                        self.simulator.rz(cmd.angles[0], &cmd.qubits);
                    }
                }
                GateType::R1XY => {
                    if cmd.angles.len() >= 2 {
                        self.simulator
                            .r1xy(cmd.angles[0], cmd.angles[1], &cmd.qubits);
                    }
                }
                GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                    let meas_results = self.simulator.mz(&cmd.qubits);
                    for meas_result in meas_results {
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep | GateType::QAlloc => {
                    self.simulator.pz(&cmd.qubits);
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree
                | GateType::Custom => {
                    // No operation needed (QFree is just a marker for qubit lifecycle)
                }
                GateType::U => {
                    if cmd.angles.len() >= 3 {
                        self.simulator
                            .u(cmd.angles[0], cmd.angles[1], cmd.angles[2], &cmd.qubits);
                    }
                }
                GateType::RXX => {
                    if cmd.angles.is_empty() {
                        return Err(PecosError::Processing(
                            "RXX gate requires at least one angle".to_string(),
                        ));
                    }
                    self.simulator.rxx(cmd.angles[0], &cmd.qubits);
                }
                GateType::RYY => {
                    if cmd.angles.is_empty() {
                        return Err(PecosError::Processing(
                            "RYY gate requires at least one angle".to_string(),
                        ));
                    }
                    self.simulator.ryy(cmd.angles[0], &cmd.qubits);
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
                    self.simulator.x(&cmd.qubits);
                }
                GateType::Y => {
                    self.simulator.y(&cmd.qubits);
                }
                GateType::Z => {
                    self.simulator.z(&cmd.qubits);
                }
                GateType::H => {
                    self.simulator.h(&cmd.qubits);
                }
                GateType::SZ => {
                    self.simulator.sz(&cmd.qubits);
                }
                GateType::SZdg => {
                    self.simulator.szdg(&cmd.qubits);
                }
                GateType::T => {
                    self.simulator.t(&cmd.qubits);
                }
                GateType::Tdg => {
                    self.simulator.tdg(&cmd.qubits);
                }
                GateType::CX => {
                    self.simulator.cx(&cmd.qubits);
                }
                GateType::CY => {
                    self.simulator.cy(&cmd.qubits);
                }
                GateType::CZ => {
                    self.simulator.cz(&cmd.qubits);
                }
                // CH = Ry(π/4)_target, CX(control, target), Ry(-π/4)_target
                GateType::CH => {
                    for qubits in cmd.qubits.chunks_exact(2) {
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
                    self.simulator.rzz(cmd.angles[0], &cmd.qubits);
                }
                GateType::SZZ => {
                    self.simulator.szz(&cmd.qubits);
                }
                GateType::SZZdg => {
                    self.simulator.szzdg(&cmd.qubits);
                }
                GateType::F => {
                    self.simulator.f(&cmd.qubits);
                }
                GateType::Fdg => {
                    self.simulator.fdg(&cmd.qubits);
                }
                GateType::SY => {
                    self.simulator.sy(&cmd.qubits);
                }
                GateType::SYdg => {
                    self.simulator.sydg(&cmd.qubits);
                }
                GateType::SXX => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SXX gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.sxx(&cmd.qubits);
                }
                GateType::SXXdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SXXdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.sxxdg(&cmd.qubits);
                }
                GateType::SYY => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SYY gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.syy(&cmd.qubits);
                }
                GateType::SYYdg => {
                    if cmd.qubits.len() % 2 != 0 {
                        return Err(PecosError::Processing(format!(
                            "SYYdg gate requires even number of qubits, got {}",
                            cmd.qubits.len()
                        )));
                    }
                    self.simulator.syydg(&cmd.qubits);
                }
                GateType::SWAP => {
                    self.simulator.swap(&cmd.qubits);
                }
                GateType::CRZ => {
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0];
                        let half_angle = angle / 2u64;
                        for pair in cmd.qubits.chunks_exact(2) {
                            // CRZ(θ) = Rz(θ/2) on target, CX, Rz(-θ/2) on target, CX
                            self.simulator.rz(half_angle, &[pair[1]]);
                            self.simulator.cx(pair);
                            self.simulator.rz(-half_angle, &[pair[1]]);
                            self.simulator.cx(pair);
                        }
                    }
                }
                GateType::CCX => {
                    for qubits in cmd.qubits.chunks_exact(3) {
                        // Toffoli decomposition into Clifford+T gates
                        let c0 = qubits[0];
                        let c1 = qubits[1];
                        let target = qubits[2];
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
                GateType::SX => {
                    self.simulator.sx(&cmd.qubits);
                }
                GateType::SXdg => {
                    self.simulator.sxdg(&cmd.qubits);
                }
                GateType::RX => {
                    if !cmd.angles.is_empty() {
                        self.simulator.rx(cmd.angles[0], &cmd.qubits);
                    }
                }
                GateType::RY => {
                    if !cmd.angles.is_empty() {
                        self.simulator.ry(cmd.angles[0], &cmd.qubits);
                    }
                }
                GateType::RZ => {
                    if !cmd.angles.is_empty() {
                        self.simulator.rz(cmd.angles[0], &cmd.qubits);
                    }
                }
                GateType::R1XY => {
                    if cmd.angles.len() >= 2 {
                        self.simulator
                            .r1xy(cmd.angles[0], cmd.angles[1], &cmd.qubits);
                    }
                }
                GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                    let meas_results = self.simulator.mz(&cmd.qubits);
                    for meas_result in meas_results {
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep | GateType::QAlloc => {
                    self.simulator.pz(&cmd.qubits);
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree
                | GateType::Custom => {
                    // No operation needed (QFree is just a marker for qubit lifecycle)
                }
                GateType::U => {
                    if cmd.angles.len() >= 3 {
                        self.simulator
                            .u(cmd.angles[0], cmd.angles[1], cmd.angles[2], &cmd.qubits);
                    }
                }
                GateType::RXX => {
                    if cmd.angles.is_empty() {
                        return Err(PecosError::Processing(
                            "RXX gate requires at least one angle".to_string(),
                        ));
                    }
                    self.simulator.rxx(cmd.angles[0], &cmd.qubits);
                }
                GateType::RYY => {
                    if cmd.angles.is_empty() {
                        return Err(PecosError::Processing(
                            "RYY gate requires at least one angle".to_string(),
                        ));
                    }
                    self.simulator.ryy(cmd.angles[0], &cmd.qubits);
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
///
/// The engine uses a shared CUDA environment that persists for the lifetime of the
/// process, avoiding `QuEST` CUDA recreation issues. Only the quantum register (qureg)
/// is created/destroyed per engine instance.
#[cfg(feature = "cuda")]
pub struct QuestCudaStateVecEngine {
    /// Opaque handle to the quantum register (owned by this instance)
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
    /// - The shared CUDA environment cannot be created
    /// - The quantum register cannot be allocated
    ///
    /// # Panics
    /// Panics if `num_qubits` exceeds `i32::MAX` (extremely unlikely in practice).
    pub fn new(num_qubits: usize) -> Result<Self, PecosError> {
        // Get the shared CUDA environment (created once, reused across all engines)
        let (env_handle, backend) = crate::cuda_loader::get_shared_cuda_env().map_err(|e| {
            PecosError::Processing(format!(
                "Failed to get shared CUDA environment: {e}\n\n{}",
                crate::cuda_loader::cuda_unavailable_error_message()
            ))
        })?;

        // Create quantum register using the shared environment
        let qureg_handle =
            unsafe { (backend.create_qureg)(env_handle, i32::try_from(num_qubits).unwrap()) };
        if qureg_handle.is_null() {
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
            qureg_handle,
            backend,
            num_qubits,
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for QuestCudaStateVecEngine {
    fn drop(&mut self) {
        // Destroy the qureg to free GPU memory.
        // NOTE: QuEST's CUDA backend only supports one qureg at a time,
        // so this must be called before creating a new engine.
        unsafe {
            if !self.qureg_handle.is_null() {
                (self.backend.destroy_qureg)(self.qureg_handle);
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
                    if !cmd.angles.is_empty() {
                        let theta = cmd.angles[0].to_radians();
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            unsafe {
                                (self.backend.apply_rotation_x)(self.qureg_handle, qubit, theta);
                            }
                        }
                    }
                }
                GateType::RY => {
                    if !cmd.angles.is_empty() {
                        let theta = cmd.angles[0].to_radians();
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            unsafe {
                                (self.backend.apply_rotation_y)(self.qureg_handle, qubit, theta);
                            }
                        }
                    }
                }
                GateType::RZ => {
                    if !cmd.angles.is_empty() {
                        let theta = cmd.angles[0].to_radians();
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
                            unsafe {
                                (self.backend.apply_rotation_z)(self.qureg_handle, qubit, theta);
                            }
                        }
                    }
                }
                GateType::RZZ => {
                    // RZZ(theta) = exp(-i * theta/2 * Z_a Z_b)
                    // Decompose as: CNOT(a,b) - RZ(theta, b) - CNOT(a,b)
                    let theta = cmd.angles[0].to_radians();
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (*qubits[0] as i32, *qubits[1] as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_z)(self.qureg_handle, b, theta);
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
                    if cmd.angles.len() >= 2 {
                        let theta = cmd.angles[0].to_radians();
                        let phi = cmd.angles[1].to_radians();
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
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
                    if cmd.angles.len() >= 3 {
                        let theta = cmd.angles[0].to_radians();
                        let phi = cmd.angles[1].to_radians();
                        let lambda = cmd.angles[2].to_radians();
                        for q in &cmd.qubits {
                            let qubit = **q as i32;
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
                    if !cmd.angles.is_empty() {
                        let angle = cmd.angles[0].to_radians();
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
                GateType::F => {
                    // F = SX · SZ = RX(pi/2) · RZ(pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_rotation_z)(
                                self.qureg_handle,
                                qubit,
                                std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_rotation_x)(
                                self.qureg_handle,
                                qubit,
                                std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::Fdg => {
                    // Fdg = F† = SZ† · SX† = RZ(-pi/2) · RX(-pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_rotation_x)(
                                self.qureg_handle,
                                qubit,
                                -std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_rotation_z)(
                                self.qureg_handle,
                                qubit,
                                -std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::SY => {
                    // SY = RY(pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_rotation_y)(
                                self.qureg_handle,
                                qubit,
                                std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::SYdg => {
                    // SYdg = RY(-pi/2)
                    for q in &cmd.qubits {
                        let qubit = usize::from(*q) as i32;
                        unsafe {
                            (self.backend.apply_rotation_y)(
                                self.qureg_handle,
                                qubit,
                                -std::f64::consts::FRAC_PI_2,
                            );
                        }
                    }
                }
                GateType::SXX => {
                    // SXX = RXX(pi/2): decompose as H⊗H · SZZ · H⊗H
                    // Or equivalently: CNOT(a,b) · RX(pi/2, b) · CNOT(a,b)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_x)(
                                self.qureg_handle,
                                b,
                                std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::SXXdg => {
                    // SXXdg = RXX(-pi/2)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_x)(
                                self.qureg_handle,
                                b,
                                -std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::SYY => {
                    // SYY = RYY(pi/2): decompose as CNOT(a,b) · RY(pi/2, b) · CNOT(a,b)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_y)(
                                self.qureg_handle,
                                b,
                                std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::SYYdg => {
                    // SYYdg = RYY(-pi/2)
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_y)(
                                self.qureg_handle,
                                b,
                                -std::f64::consts::FRAC_PI_2,
                            );
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::Custom
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree => {
                    // No operation needed (Custom is a placeholder whose actual gate name is in metadata)
                }
                GateType::RXX => {
                    // RXX(theta) = CNOT(a,b) · RX(theta, b) · CNOT(a,b)
                    if cmd.angles.is_empty() {
                        return Err(PecosError::Processing(
                            "RXX gate requires at least one angle".to_string(),
                        ));
                    }
                    let theta = cmd.angles[0].to_radians();
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_x)(self.qureg_handle, b, theta);
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
                }
                GateType::RYY => {
                    // RYY(theta) = CNOT(a,b) · RY(theta, b) · CNOT(a,b)
                    if cmd.angles.is_empty() {
                        return Err(PecosError::Processing(
                            "RYY gate requires at least one angle".to_string(),
                        ));
                    }
                    let theta = cmd.angles[0].to_radians();
                    for qubits in cmd.qubits.chunks_exact(2) {
                        let (a, b) = (usize::from(qubits[0]) as i32, usize::from(qubits[1]) as i32);
                        unsafe {
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                            (self.backend.apply_rotation_y)(self.qureg_handle, b, theta);
                            (self.backend.apply_cnot)(self.qureg_handle, a, b);
                        }
                    }
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

// ============================================================================
// CliffordGateable and ArbitraryRotationGateable implementations for CUDA engine
// ============================================================================

#[cfg(feature = "cuda")]
impl QuantumSimulator for QuestCudaStateVecEngine {
    fn reset(&mut self) -> &mut Self {
        unsafe {
            (self.backend.init_zero_state)(self.qureg_handle);
        }
        self
    }
}

#[cfg(feature = "cuda")]
impl CliffordGateable for QuestCudaStateVecEngine {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            unsafe {
                (self.backend.apply_s_gate)(self.qureg_handle, q.index() as i32);
            }
        }
        self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            unsafe {
                (self.backend.apply_hadamard)(self.qureg_handle, q.index() as i32);
            }
        }
        self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            unsafe {
                (self.backend.apply_cnot)(
                    self.qureg_handle,
                    pair[0].index() as i32,
                    pair[1].index() as i32,
                );
            }
        }
        self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| {
                let outcome =
                    unsafe { (self.backend.measure)(self.qureg_handle, q.index() as i32) };
                MeasurementResult {
                    outcome: outcome != 0,
                    is_deterministic: false, // CUDA backend doesn't report determinism
                }
            })
            .collect()
    }
}

#[cfg(feature = "cuda")]
impl ArbitraryRotationGateable for QuestCudaStateVecEngine {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            unsafe {
                (self.backend.apply_rotation_x)(self.qureg_handle, q.index() as i32, theta);
            }
        }
        self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            unsafe {
                (self.backend.apply_rotation_z)(self.qureg_handle, q.index() as i32, theta);
            }
        }
        self
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn rzz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RZZ(theta) = exp(-i * theta/2 * Z⊗Z)
        // Decomposition: CNOT(q1,q2) . RZ(theta, q2) . CNOT(q1,q2)
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RZZ requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index() as i32;
            let q2 = pair[1].index() as i32;
            unsafe {
                (self.backend.apply_cnot)(self.qureg_handle, q1, q2);
                (self.backend.apply_rotation_z)(self.qureg_handle, q2, theta);
                (self.backend.apply_cnot)(self.qureg_handle, q1, q2);
            }
        }
        self
    }
}
