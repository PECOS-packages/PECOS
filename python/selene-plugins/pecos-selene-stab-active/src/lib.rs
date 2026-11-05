// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! PECOS `StabActive` simulator plugin for the Selene quantum emulator.
//!
//! This crate provides a Selene-compatible plugin wrapping the PECOS Clifford+RZ simulator.
//! Clifford gates update a tableau; non-Clifford rotations use dense active coordinates.
//! Amplitude storage is exponential in active width, bounded only by platform memory.

use anyhow::{Result, anyhow, bail};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_active::StabActive;
use selene_core::error_model::BatchResult;
use selene_core::export_simulator_plugin;
use selene_core::operation::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::sync::Arc;

/// The PECOS `StabActive` simulator wrapped for Selene compatibility.
pub struct StabActiveSimulator {
    /// The underlying PECOS Clifford+RZ simulator
    simulator: StabActive,
    /// Number of qubits in the system
    n_qubits: u64,
}

impl StabActiveSimulator {
    fn rxy(&mut self, qubit: u64, theta: f64, phi: f64) -> Result<()> {
        let theta_angle = Angle64::try_from_radians(theta)?;
        let phi_angle = Angle64::try_from_radians(phi)?;
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RXY(qubit={qubit}, theta={theta}, phi={phi}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = QubitId(usize::try_from(qubit)?);

        // RXY(theta, phi) = Rz(phi) * Rx(theta) * Rz(-phi)
        // Gates are applied left-to-right in code but the matrix multiplication
        // is right-to-left, so we apply Rz(-phi) first
        self.simulator
            .rz(Angle64::try_from_radians(-phi)?, &[q])
            .rx(theta_angle, &[q])
            .rz(phi_angle, &[q]);

        Ok(())
    }

    fn rz(&mut self, qubit: u64, theta: f64) -> Result<()> {
        let theta_angle = Angle64::try_from_radians(theta)?;
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RZ(qubit={qubit}, theta={theta}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        self.simulator
            .rz(theta_angle, &[QubitId(usize::try_from(qubit)?)]);
        Ok(())
    }

    fn rzz(&mut self, qubit1: u64, qubit2: u64, theta: f64) -> Result<()> {
        if qubit1 == qubit2 {
            return Err(anyhow!(
                "RZZ requires distinct qubits; repeated qubit {qubit1}"
            ));
        }
        let theta_angle = Angle64::try_from_radians(theta)?;
        if qubit1 >= self.n_qubits || qubit2 >= self.n_qubits {
            return Err(anyhow!(
                "RZZ(qubit1={qubit1}, qubit2={qubit2}, theta={theta}) is out of bounds. \
                 qubits must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        self.simulator.rzz(
            theta_angle,
            &[(
                QubitId(usize::try_from(qubit1)?),
                QubitId(usize::try_from(qubit2)?),
            )],
        );
        Ok(())
    }

    fn rxyxy2q(&mut self, qubit1: u64, qubit2: u64, theta: f64, phi: f64) -> Result<()> {
        let theta_angle = Angle64::try_from_radians(theta)?;
        let phi_angle = Angle64::try_from_radians(phi)?;
        if qubit1 >= self.n_qubits || qubit2 >= self.n_qubits || qubit1 == qubit2 {
            return Err(anyhow!(
                "RXYXY2Q requires two distinct qubits below {}, got ({qubit1}, {qubit2})",
                self.n_qubits
            ));
        }
        self.simulator.rxyxy2q(
            theta_angle,
            phi_angle,
            &[(
                QubitId(usize::try_from(qubit1)?),
                QubitId(usize::try_from(qubit2)?),
            )],
        );
        Ok(())
    }

    fn measure(&mut self, qubit: u64) -> Result<bool> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Measure(qubit={qubit}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let results = self.simulator.mz(&[QubitId(usize::try_from(qubit)?)]);
        Ok(results[0].outcome)
    }

    fn postselect(&mut self, qubit: u64, target_value: bool) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target_value={target_value}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let results = self.simulator.mz(&[QubitId(usize::try_from(qubit)?)]);
        let outcome = results[0].outcome;

        if outcome != target_value {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target_value={target_value}) failed. \
                 The measurement outcome was {outcome} but postselection to {target_value} was requested.",
            ));
        }

        Ok(())
    }

    fn reset(&mut self, qubit: u64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Reset(qubit={qubit}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = QubitId(usize::try_from(qubit)?);

        // Measure the qubit and flip if needed to get |0>
        let results = self.simulator.mz(&[q]);
        if results[0].outcome {
            self.simulator.x(&[q]);
        }

        Ok(())
    }
}

impl SimulatorInterface for StabActiveSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        // Reconstructing also resets peak_active_width to zero for this shot.
        self.simulator = StabActive::with_seed(usize::try_from(self.n_qubits)?, seed)
            .with_max_active_width(StabActive::max_supported_active_width());
        Ok(())
    }

    fn shot_end(&mut self) -> Result<()> {
        Ok(())
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
                Operation::RPPGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                    phi,
                } => {
                    self.rxyxy2q(qubit_id_1, qubit_id_2, theta, phi)?;
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
        match nth_metric {
            0 => Ok(Some((
                "active_width".to_string(),
                MetricValue::U64(u64::try_from(self.simulator.active_width())?),
            ))),
            1 => Ok(Some((
                "peak_active_width".to_string(),
                MetricValue::U64(u64::try_from(self.simulator.peak_active_width())?),
            ))),
            _ => Ok(None),
        }
    }

    fn dump_state(&mut self, _file: &std::path::Path, _qubits: &[u64]) -> Result<()> {
        Err(anyhow!(
            "State dumping is not supported for the StabActive simulator"
        ))
    }
}

/// Factory for creating `StabActiveSimulator` instances.
#[derive(Default)]
pub struct StabActiveSimulatorFactory;

impl SimulatorInterfaceFactory for StabActiveSimulatorFactory {
    type Interface = StabActiveSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();

        if args.len() > 1 {
            bail!(
                "Expected no arguments for the PECOS StabActive plugin, got {} arguments: {:?}",
                args.len() - 1,
                args.iter().skip(1).collect::<Vec<_>>()
            );
        }

        if n_qubits == 0 {
            bail!("Number of qubits must be greater than 0");
        }

        Ok(Box::new(StabActiveSimulator {
            simulator: StabActive::with_seed(usize::try_from(n_qubits)?, 0)
                .with_max_active_width(StabActive::max_supported_active_width()),
            n_qubits,
        }))
    }
}

// Export the plugin using Selene's macro
export_simulator_plugin!(crate::StabActiveSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::StabActiveSimulatorFactory;
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    fn assert_metric(sim: &mut super::StabActiveSimulator, index: u8, name: &str, width: u64) {
        use selene_core::simulator::SimulatorInterface;
        let (actual_name, value) = sim
            .get_metric(index)
            .expect("metric")
            .expect("known metric");
        assert_eq!(actual_name, name);
        assert!(matches!(value, selene_core::utils::MetricValue::U64(value) if value == width));
    }

    #[test]
    fn platform_limit_and_shot_metrics() {
        use pecos_core::{Angle64, QubitId};
        use pecos_simulators::CliffordGateable;
        use pecos_stab_active::StabActive;
        use selene_core::simulator::{SimulatorInterface, interface::SimulatorInterfaceFactory};

        let mut sim = Arc::new(StabActiveSimulatorFactory)
            .init(2, &[String::new()])
            .expect("create plugin");
        // A public read-only accessor is needed across the backend crate boundary.
        assert_eq!(
            sim.simulator.max_active_width(),
            StabActive::max_supported_active_width()
        );
        assert_metric(&mut sim, 0, "active_width", 0);
        assert_metric(&mut sim, 1, "peak_active_width", 0);
        for shot in 0..2 {
            sim.shot_start(shot, 17).expect("start shot");
            assert_eq!(
                sim.simulator.max_active_width(),
                StabActive::max_supported_active_width()
            );
            assert_metric(&mut sim, 0, "active_width", 0);
            assert_metric(&mut sim, 1, "peak_active_width", 0);
            sim.simulator.h(&[QubitId(0)]);
            sim.rz(0, (Angle64::QUARTER_TURN / 2).to_radians())
                .expect("rotate");
            assert_metric(&mut sim, 0, "active_width", 1);
            assert_metric(&mut sim, 1, "peak_active_width", 1);
            sim.measure(0).expect("measure");
            assert_metric(&mut sim, 0, "active_width", 0);
            assert_metric(&mut sim, 1, "peak_active_width", 1);
        }
        assert!(sim.get_metric(2).expect("unknown metric").is_none());
        assert!(sim.get_metric(u8::MAX).expect("unknown metric").is_none());
    }

    #[test]
    #[ignore = "27 active coordinates require a 2 GiB amplitude vector"]
    fn width_above_default_limit() {
        use pecos_core::QubitId;
        use pecos_simulators::CliffordGateable;
        use selene_core::simulator::{SimulatorInterface, interface::SimulatorInterfaceFactory};

        let mut sim = Arc::new(StabActiveSimulatorFactory)
            .init(27, &[String::new()])
            .expect("create plugin");
        sim.shot_start(0, 17).expect("start shot");
        for qubit in 0..27 {
            sim.simulator.h(&[QubitId(qubit)]);
            sim.rz(u64::try_from(qubit).expect("small qubit index"), 0.3)
                .expect("rotate");
        }
        assert_metric(&mut sim, 0, "active_width", 27);
        assert_metric(&mut sim, 1, "peak_active_width", 27);
    }

    #[test]
    fn two_qubit_operations_reject_repeated_operands() {
        use selene_core::operation::{BatchOperation, Operation};
        use selene_core::simulator::{SimulatorInterface, interface::SimulatorInterfaceFactory};
        for theta in [
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            3.0 * std::f64::consts::FRAC_PI_2,
        ] {
            for (gate, operation) in [
                (
                    "RZZ",
                    Operation::RZZGate {
                        qubit_id_1: 1,
                        qubit_id_2: 1,
                        theta,
                    },
                ),
                (
                    "RXYXY2Q",
                    Operation::RPPGate {
                        qubit_id_1: 1,
                        qubit_id_2: 1,
                        theta,
                        phi: 0.0,
                    },
                ),
            ] {
                let mut sim = Arc::new(StabActiveSimulatorFactory)
                    .init(2, &[String::new()])
                    .expect("create plugin");
                sim.shot_start(0, 17).expect("start shot");
                let error = sim
                    .handle_operations(BatchOperation::error_model(vec![operation]))
                    .err()
                    .expect("repeated operands must return an error, never unwind through FFI");
                let message = error.to_string();
                assert!(message.contains(gate), "{message}");
                assert!(message.contains("distinct"), "{message}");
                assert!(message.contains('1'), "{message}");
            }
        }
    }

    #[test]
    fn nonfinite_runtime_angles_are_errors() {
        use selene_core::operation::{BatchOperation, Operation};
        use selene_core::simulator::{SimulatorInterface, interface::SimulatorInterfaceFactory};
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut sim = std::sync::Arc::new(super::StabActiveSimulatorFactory)
                .init(2, &[String::new()])
                .expect("plugin operation succeeds");
            sim.shot_start(0, 17).expect("plugin operation succeeds");
            let error = sim
                .handle_operations(BatchOperation::error_model(vec![Operation::RPPGate {
                    qubit_id_1: 0,
                    qubit_id_2: 1,
                    theta: 0.0,
                    phi: value,
                }]))
                .err()
                .expect("nonfinite axis must fail");
            assert!(error.to_string().contains(&value.to_string()));
        }
    }

    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(StabActiveSimulatorFactory);
        let args: Vec<String> = vec![String::new()];
        run_basic_tests(interface, args);
    }

    #[test]
    fn rpp_batch_rotates_both_qubits_and_checks_targets() {
        use selene_core::operation::{BatchOperation, Operation};
        use selene_core::simulator::{SimulatorInterface, interface::SimulatorInterfaceFactory};

        let args = vec![String::new()];
        let mut sim = Arc::new(StabActiveSimulatorFactory)
            .init(3, &args)
            .expect("plugin operation succeeds");
        sim.shot_start(0, 17).expect("plugin operation succeeds");
        // Two quarter turns give a half turn, so |00> becomes |11> for
        // any shared axis. Qubit 1 isn't targeted and should stay at zero.
        for _ in 0..2 {
            sim.handle_operations(BatchOperation::error_model(vec![Operation::RPPGate {
                qubit_id_1: 2,
                qubit_id_2: 0,
                theta: std::f64::consts::FRAC_PI_2,
                phi: 0.37,
            }]))
            .expect("plugin operation succeeds");
        }
        let results = sim
            .handle_operations(BatchOperation::error_model(vec![
                Operation::Measure {
                    qubit_id: 0,
                    result_id: 0,
                },
                Operation::Measure {
                    qubit_id: 1,
                    result_id: 1,
                },
                Operation::Measure {
                    qubit_id: 2,
                    result_id: 2,
                },
            ]))
            .expect("plugin operation succeeds");
        assert_eq!(
            results
                .bool_results
                .iter()
                .map(|r| r.value)
                .collect::<Vec<_>>(),
            vec![true, false, true]
        );
        for (first, second) in [(0, 3), (3, 0), (1, 1)] {
            assert!(
                sim.handle_operations(BatchOperation::error_model(vec![Operation::RPPGate {
                    qubit_id_1: first,
                    qubit_id_2: second,
                    theta: 0.0,
                    phi: 0.0,
                }]))
                .is_err()
            );
        }
    }
}
