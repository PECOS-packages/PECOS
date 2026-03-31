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

//! PECOS `CliffordRz` simulator plugin for the Selene quantum emulator.
//!
//! This crate provides a Selene-compatible plugin wrapping the PECOS Clifford+RZ simulator.
//! It supports Clifford gates efficiently plus arbitrary RZ rotations via a sum-over-Cliffords
//! decomposition. Cost is polynomial in qubits and Clifford gates, exponential in RZ gate count.

use anyhow::{Result, anyhow, bail};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, CliffordRz};
use selene_core::export_simulator_plugin;
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::sync::Arc;

/// The PECOS `CliffordRz` simulator wrapped for Selene compatibility.
pub struct CliffordRzSimulator {
    /// The underlying PECOS Clifford+RZ simulator
    simulator: CliffordRz,
    /// Number of qubits in the system
    n_qubits: u64,
}

impl CliffordRzSimulator {
    /// Convert a `u64` to `usize` for use with the simulator.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }
}

impl SimulatorInterface for CliffordRzSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        self.simulator = CliffordRz::new_with_seed(Self::to_usize(self.n_qubits), seed);
        Ok(())
    }

    fn shot_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn rxy(&mut self, qubit: u64, theta: f64, phi: f64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RXY(qubit={qubit}, theta={theta}, phi={phi}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = QubitId(Self::to_usize(qubit));

        // RXY(theta, phi) = Rz(phi) * Rx(theta) * Rz(-phi)
        // Gates are applied left-to-right in code but the matrix multiplication
        // is right-to-left, so we apply Rz(-phi) first
        self.simulator
            .rz(Angle64::from_radians(-phi), &[q])
            .rx(Angle64::from_radians(theta), &[q])
            .rz(Angle64::from_radians(phi), &[q]);

        Ok(())
    }

    fn rz(&mut self, qubit: u64, theta: f64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RZ(qubit={qubit}, theta={theta}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
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
                "RZZ(qubit1={qubit1}, qubit2={qubit2}, theta={theta}) is out of bounds. \
                 qubits must be less than the number of qubits ({}).",
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
                "Measure(qubit={qubit}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let results = self.simulator.mz(&[QubitId(Self::to_usize(qubit))]);
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

        let results = self.simulator.mz(&[QubitId(Self::to_usize(qubit))]);
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

        let q = QubitId(Self::to_usize(qubit));

        // Measure the qubit and flip if needed to get |0>
        let results = self.simulator.mz(&[q]);
        if results[0].outcome {
            self.simulator.x(&[q]);
        }

        Ok(())
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        match nth_metric {
            0 => Ok(Some((
                "num_terms".to_string(),
                MetricValue::U64(self.simulator.num_terms() as u64),
            ))),
            _ => Ok(None),
        }
    }

    fn dump_state(&mut self, _file: &std::path::Path, _qubits: &[u64]) -> Result<()> {
        Err(anyhow!(
            "State dumping is not supported for the CliffordRz simulator"
        ))
    }
}

/// Factory for creating `CliffordRzSimulator` instances.
#[derive(Default)]
pub struct CliffordRzSimulatorFactory;

impl SimulatorInterfaceFactory for CliffordRzSimulatorFactory {
    type Interface = CliffordRzSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();

        if args.len() > 1 {
            bail!(
                "Expected no arguments for the PECOS CliffordRz plugin, got {} arguments: {:?}",
                args.len() - 1,
                args.iter().skip(1).collect::<Vec<_>>()
            );
        }

        if n_qubits == 0 {
            bail!("Number of qubits must be greater than 0");
        }

        Ok(Box::new(CliffordRzSimulator {
            simulator: CliffordRz::new_with_seed(CliffordRzSimulator::to_usize(n_qubits), 0),
            n_qubits,
        }))
    }
}

// Export the plugin using Selene's macro
export_simulator_plugin!(crate::CliffordRzSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::CliffordRzSimulatorFactory;
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(CliffordRzSimulatorFactory);
        let args: Vec<String> = vec![String::new()];
        run_basic_tests(interface, args);
    }
}
