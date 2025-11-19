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
use rand::SeedableRng;
use std::any::Any;
use std::fmt::Debug;

/// Helper function to create quantum engine errors
fn quantum_error<S: Into<String>>(msg: S) -> PecosError {
    PecosError::Processing(msg.into())
}

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
                GateType::Measure | GateType::MeasureLeaked => {
                    for q in &cmd.qubits {
                        let meas_result = self.simulator.mz(**q);
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep => {
                    for q in &cmd.qubits {
                        self.simulator.pz(**q);
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload => {
                    // No operation needed
                }
                GateType::U => {
                    if cmd.params.len() >= 3 {
                        for q in &cmd.qubits {
                            self.simulator
                                .u(cmd.params[0], cmd.params[1], cmd.params[2], **q);
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
        self.simulator.reset();
        Ok(())
    }
}

impl QuantumEngine for QuestStateVecEngine {
    fn set_seed(&mut self, seed: u64) -> Result<(), PecosError> {
        let rng = <QuestStateVec as RngManageable>::Rng::seed_from_u64(seed);
        self.simulator
            .set_rng(rng)
            .map_err(|e| quantum_error(format!("Failed to set seed: {e}")))
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
                GateType::Measure | GateType::MeasureLeaked => {
                    for q in &cmd.qubits {
                        let meas_result = self.simulator.mz(**q);
                        let outcome = u32::from(meas_result.outcome);
                        measurements.push(outcome);
                    }
                }
                GateType::Prep => {
                    for q in &cmd.qubits {
                        self.simulator.pz(**q);
                    }
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload => {
                    // No operation needed
                }
                GateType::U => {
                    if cmd.params.len() >= 3 {
                        for q in &cmd.qubits {
                            self.simulator
                                .u(cmd.params[0], cmd.params[1], cmd.params[2], **q);
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
        self.simulator.reset();
        Ok(())
    }
}

impl QuantumEngine for QuestDensityMatrixEngine {
    fn set_seed(&mut self, seed: u64) -> Result<(), PecosError> {
        let rng = <QuestDensityMatrix as RngManageable>::Rng::seed_from_u64(seed);
        self.simulator
            .set_rng(rng)
            .map_err(|e| quantum_error(format!("Failed to set seed: {e}")))
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
    /// GPU mode flag (only used if gpu feature is enabled)
    #[allow(dead_code)]
    use_gpu: bool,
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
        self.use_gpu = false;
        self
    }

    /// Use GPU acceleration mode
    ///
    /// # Panics
    /// Panics if the `gpu` feature is not enabled at compile time
    #[must_use]
    pub fn with_gpu(self) -> Self {
        #[cfg(not(feature = "gpu"))]
        {
            panic!(
                "GPU feature is not enabled. Rebuild with --features gpu to use GPU acceleration"
            );
        }
        #[cfg(feature = "gpu")]
        {
            Self {
                use_gpu: true,
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
    /// GPU mode flag (only used if gpu feature is enabled)
    #[allow(dead_code)]
    use_gpu: bool,
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
        self.use_gpu = false;
        self
    }

    /// Use GPU acceleration mode
    ///
    /// # Panics
    /// Panics if the `gpu` feature is not enabled at compile time
    #[must_use]
    pub fn with_gpu(self) -> Self {
        #[cfg(not(feature = "gpu"))]
        {
            panic!(
                "GPU feature is not enabled. Rebuild with --features gpu to use GPU acceleration"
            );
        }
        #[cfg(feature = "gpu")]
        {
            Self {
                use_gpu: true,
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
