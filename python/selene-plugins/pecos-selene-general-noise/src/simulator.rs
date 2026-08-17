use std::any::Any;
use std::f64::consts::PI;

use pecos_core::errors::PecosError;
use pecos_engines::byte_message::GateType;
use pecos_engines::prelude::*;
use selene_core::simulator::{Simulator, SimulatorInterface};
use selene_core::utils::MetricValue;

pub struct SeleneSimulator {
    inner: Simulator,
}

impl SeleneSimulator {
    pub fn new(inner: Simulator) -> Self {
        Self { inner }
    }

    fn error(context: &str, error: impl std::fmt::Display) -> PecosError {
        PecosError::Generic(format!("Selene simulator {context} failed: {error}"))
    }

    pub fn shot_start(&mut self, shot_id: u64, seed: u64) -> Result<(), PecosError> {
        self.inner
            .shot_start(shot_id, seed)
            .map_err(|error| Self::error("shot_start", error))
    }

    pub fn shot_end(&mut self) -> Result<(), PecosError> {
        self.inner
            .shot_end()
            .map_err(|error| Self::error("shot_end", error))
    }

    pub fn metric(&mut self, index: u8) -> anyhow::Result<Option<(String, MetricValue)>> {
        self.inner.get_metric(index)
    }

    pub fn dump_state(&mut self, file: &std::path::Path, qubits: &[u64]) -> anyhow::Result<()> {
        self.inner.dump_state(file, qubits)
    }

    fn rz(&mut self, qubit: usize, angle: f64) -> Result<(), PecosError> {
        self.inner
            .rz(qubit as u64, angle)
            .map_err(|error| Self::error("RZ", error))
    }

    fn rxy(&mut self, qubit: usize, theta: f64, phi: f64) -> Result<(), PecosError> {
        self.inner
            .rxy(qubit as u64, theta, phi)
            .map_err(|error| Self::error("RXY", error))
    }

    fn rzz(&mut self, first: usize, second: usize, theta: f64) -> Result<(), PecosError> {
        self.inner
            .rzz(first as u64, second as u64, theta)
            .map_err(|error| Self::error("RZZ", error))
    }
}

impl Clone for SeleneSimulator {
    fn clone(&self) -> Self {
        panic!("a loaded Selene simulator cannot be cloned")
    }
}

unsafe impl Send for SeleneSimulator {}
unsafe impl Sync for SeleneSimulator {}

impl std::fmt::Debug for SeleneSimulator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SeleneSimulator")
            .finish_non_exhaustive()
    }
}

impl Engine for SeleneSimulator {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, message: Self::Input) -> Result<Self::Output, PecosError> {
        let mut outcomes = Vec::new();
        for gate in message.quantum_ops()? {
            let qubits = gate
                .qubits
                .iter()
                .map(|qubit| usize::from(*qubit))
                .collect::<Vec<_>>();
            let angle = |index: usize| gate.angles[index].to_radians_signed();
            match gate.gate_type {
                GateType::X => self.rxy(qubits[0], PI, 0.0)?,
                GateType::Y => self.rxy(qubits[0], PI, PI / 2.0)?,
                GateType::Z => self.rz(qubits[0], PI)?,
                GateType::H => self.rxy(qubits[0], PI / 2.0, -PI / 2.0)?,
                GateType::RX => self.rxy(qubits[0], angle(0), 0.0)?,
                GateType::RY => self.rxy(qubits[0], angle(0), PI / 2.0)?,
                GateType::RZ => self.rz(qubits[0], angle(0))?,
                GateType::R1XY => self.rxy(qubits[0], angle(0), angle(1))?,
                GateType::RZZ => self.rzz(qubits[0], qubits[1], angle(0))?,
                GateType::SZZ => self.rzz(qubits[0], qubits[1], PI / 2.0)?,
                GateType::SZZdg => self.rzz(qubits[0], qubits[1], -PI / 2.0)?,
                GateType::SZ => self.rz(qubits[0], PI / 2.0)?,
                GateType::SZdg => self.rz(qubits[0], -PI / 2.0)?,
                GateType::T => self.rz(qubits[0], PI / 4.0)?,
                GateType::Tdg => self.rz(qubits[0], -PI / 4.0)?,
                GateType::MZ | GateType::MeasureLeaked => {
                    outcomes.push(usize::from(
                        self.inner
                            .measure(qubits[0] as u64)
                            .map_err(|error| Self::error("measurement", error))?,
                    ));
                }
                GateType::PZ => self
                    .inner
                    .reset(qubits[0] as u64)
                    .map_err(|error| Self::error("reset", error))?,
                GateType::Idle
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::MeasCrosstalkLocalPayload => {}
                unsupported => {
                    return Err(PecosError::Generic(format!(
                        "PECOS general-noise bridge produced unsupported gate {unsupported}"
                    )));
                }
            }
        }
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&outcomes);
        Ok(builder.build())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}

impl QuantumEngine for SeleneSimulator {
    fn set_seed(&mut self, _seed: u64) {}

    fn as_any(&self) -> &(dyn Any + 'static) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + 'static) {
        self
    }
}
