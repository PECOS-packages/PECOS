// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! PECOS `StabMps` simulator plugin for the Selene quantum emulator.
//!
//! Stabilizer tableau + MPS hybrid simulator. Clifford gates are O(n) on the
//! tableau; non-Clifford rotations decompose in the stabilizer basis and
//! apply to the MPS. Cost is polynomial when non-Clifford count is bounded.

use anyhow::{Result, anyhow, bail};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;
use selene_core::error_model::BatchResult;
use selene_core::export_simulator_plugin;
use selene_core::operation::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::sync::Arc;

pub struct StabMpsSimulator {
    simulator: StabMps,
    n_qubits: u64,
}

impl StabMpsSimulator {
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }
}

#[allow(clippy::unnecessary_wraps, clippy::unused_self)]
impl StabMpsSimulator {
    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        self.simulator = StabMps::builder(Self::to_usize(self.n_qubits))
            .seed(seed)
            .for_qec()
            .build();
        Ok(())
    }

    fn shot_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn rxy(&mut self, qubit: u64, theta: f64, phi: f64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RXY(qubit={qubit}) out of bounds (n_qubits={})",
                self.n_qubits
            ));
        }
        let q = QubitId(Self::to_usize(qubit));
        self.simulator
            .rz(Angle64::from_radians(-phi), &[q])
            .rx(Angle64::from_radians(theta), &[q])
            .rz(Angle64::from_radians(phi), &[q]);
        Ok(())
    }

    fn rz(&mut self, qubit: u64, theta: f64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RZ(qubit={qubit}) out of bounds (n_qubits={})",
                self.n_qubits
            ));
        }
        self.simulator.rz(
            Angle64::from_radians(theta),
            &[QubitId(Self::to_usize(qubit))],
        );
        Ok(())
    }

    fn rzz(&mut self, qubit1: u64, qubit2: u64, theta: f64) -> Result<()> {
        if qubit1 >= self.n_qubits || qubit2 >= self.n_qubits {
            return Err(anyhow!(
                "RZZ(qubit1={qubit1}, qubit2={qubit2}) out of bounds (n_qubits={})",
                self.n_qubits
            ));
        }
        self.simulator.rzz(
            Angle64::from_radians(theta),
            &[(
                QubitId(Self::to_usize(qubit1)),
                QubitId(Self::to_usize(qubit2)),
            )],
        );
        Ok(())
    }

    fn measure(&mut self, qubit: u64) -> Result<bool> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Measure(qubit={qubit}) out of bounds (n_qubits={})",
                self.n_qubits
            ));
        }
        let results = self.simulator.mz(&[QubitId(Self::to_usize(qubit))]);
        Ok(results[0].outcome)
    }

    fn postselect(&mut self, qubit: u64, target_value: bool) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Postselect(qubit={qubit}) out of bounds (n_qubits={})",
                self.n_qubits
            ));
        }
        let results = self.simulator.mz(&[QubitId(Self::to_usize(qubit))]);
        if results[0].outcome != target_value {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target={target_value}) failed: got {}",
                results[0].outcome
            ));
        }
        Ok(())
    }

    fn reset(&mut self, qubit: u64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Reset(qubit={qubit}) out of bounds (n_qubits={})",
                self.n_qubits
            ));
        }
        self.simulator.reset_qubit(QubitId(Self::to_usize(qubit)));
        Ok(())
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        match nth_metric {
            0 => Ok(Some((
                "max_bond_dim".to_string(),
                MetricValue::U64(self.simulator.max_bond_dim() as u64),
            ))),
            1 => Ok(Some((
                "uncompensated_pre_reduction_count".to_string(),
                MetricValue::U64(self.simulator.uncompensated_pre_reduction_count()),
            ))),
            _ => Ok(None),
        }
    }

    fn dump_state(&mut self, _file: &std::path::Path, _qubits: &[u64]) -> Result<()> {
        Err(anyhow!("State dumping not supported for StabMps"))
    }
}

impl SimulatorInterface for StabMpsSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, shot_id: u64, seed: u64) -> Result<()> {
        Self::shot_start(self, shot_id, seed)
    }

    fn shot_end(&mut self) -> Result<()> {
        Self::shot_end(self)
    }

    fn handle_operations(&mut self, operations: BatchOperation) -> Result<BatchResult> {
        let mut results = BatchResult::default();
        for operation in operations {
            match operation {
                Operation::RXYGate {
                    qubit_id,
                    theta,
                    phi,
                } => Self::rxy(self, qubit_id, theta, phi)?,
                Operation::RZGate { qubit_id, theta } => Self::rz(self, qubit_id, theta)?,
                Operation::RZZGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => {
                    Self::rzz(self, qubit_id_1, qubit_id_2, theta)?;
                }
                Operation::Measure {
                    qubit_id,
                    result_id,
                } => {
                    results.set_bool_result(result_id, Self::measure(self, qubit_id)?);
                }
                Operation::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    results.set_u64_result(result_id, u64::from(Self::measure(self, qubit_id)?));
                }
                Operation::Reset { qubit_id } => Self::reset(self, qubit_id)?,
                Operation::RPPGate { .. } => {
                    anyhow::bail!("RPP gates are not supported by StabMps")
                }
                Operation::Custom { .. } => {}
                _ => anyhow::bail!("Unsupported Selene operation"),
            }
        }
        Ok(results)
    }

    fn postselect(&mut self, qubit: u64, target_value: bool) -> Result<()> {
        Self::postselect(self, qubit, target_value)
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        Self::get_metric(self, nth_metric)
    }

    fn dump_state(&mut self, file: &std::path::Path, qubits: &[u64]) -> Result<()> {
        Self::dump_state(self, file, qubits)
    }
}

#[derive(Default)]
pub struct StabMpsSimulatorFactory;

impl SimulatorInterfaceFactory for StabMpsSimulatorFactory {
    type Interface = StabMpsSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();
        if args.len() > 1 {
            bail!(
                "Expected no arguments for StabMps plugin, got {}: {:?}",
                args.len() - 1,
                args.iter().skip(1).collect::<Vec<_>>()
            );
        }
        if n_qubits == 0 {
            bail!("Number of qubits must be greater than 0");
        }
        Ok(Box::new(StabMpsSimulator {
            simulator: StabMps::builder(StabMpsSimulator::to_usize(n_qubits))
                .seed(0)
                .for_qec()
                .build(),
            n_qubits,
        }))
    }
}

export_simulator_plugin!(crate::StabMpsSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::{StabMpsSimulator, StabMpsSimulatorFactory};
    use pecos_stab_tn::stab_mps::{MeasurementMode, StabMps};
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(StabMpsSimulatorFactory);
        let args: Vec<String> = vec![String::new()];
        run_basic_tests(interface, args);
    }

    #[test]
    fn selene_keeps_for_qec_measurement_policy() {
        let mut interface = StabMpsSimulator {
            simulator: StabMps::builder(2).for_qec().build(),
            n_qubits: 2,
        };
        assert_eq!(
            interface.simulator.measurement_mode(),
            MeasurementMode::Exact
        );
        interface.shot_start(0, 7).unwrap();
        assert_eq!(
            interface.simulator.measurement_mode(),
            MeasurementMode::Exact
        );
    }
}
