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

//! PECOS `Mast` (Magic State injection) simulator plugin for the Selene quantum emulator.
//!
//! Wraps the MAST simulator which handles non-Clifford gates via deferred ancilla
//! projection. Bond dimension stays bounded for Clifford+T circuits.

use anyhow::{Result, anyhow, bail};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::mast::Mast;
use selene_core::export_simulator_plugin;
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::sync::Arc;

pub struct MastSimulator {
    simulator: Mast,
    n_qubits: u64,
    max_non_clifford: usize,
}

impl MastSimulator {
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }
}

impl SimulatorInterface for MastSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        self.simulator =
            Mast::with_seed(Self::to_usize(self.n_qubits), self.max_non_clifford, seed);
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
        let q = QubitId(Self::to_usize(qubit));
        let results = self.simulator.mz(&[q]);
        if results[0].outcome {
            self.simulator.x(&[q]);
        }
        Ok(())
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        match nth_metric {
            0 => Ok(Some((
                "max_bond_dim".to_string(),
                MetricValue::U64(self.simulator.max_bond_dim() as u64),
            ))),
            1 => Ok(Some((
                "num_ancillas_used".to_string(),
                MetricValue::U64(self.simulator.num_ancillas_used() as u64),
            ))),
            _ => Ok(None),
        }
    }

    fn dump_state(&mut self, _file: &std::path::Path, _qubits: &[u64]) -> Result<()> {
        Err(anyhow!("State dumping not supported for Mast"))
    }
}

#[derive(Default)]
pub struct MastSimulatorFactory;

impl SimulatorInterfaceFactory for MastSimulatorFactory {
    type Interface = MastSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();
        // Optional arg: max_non_clifford (default 100)
        let max_nc = if args.len() > 1 {
            args[1].parse::<usize>().map_err(|_| {
                anyhow!(
                    "Mast plugin expects optional integer argument max_non_clifford, got '{}'",
                    args[1]
                )
            })?
        } else {
            100
        };
        if n_qubits == 0 {
            bail!("Number of qubits must be greater than 0");
        }
        Ok(Box::new(MastSimulator {
            simulator: Mast::with_seed(MastSimulator::to_usize(n_qubits), max_nc, 0),
            n_qubits,
            max_non_clifford: max_nc,
        }))
    }
}

export_simulator_plugin!(crate::MastSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::MastSimulatorFactory;
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(MastSimulatorFactory);
        let args: Vec<String> = vec![String::new()];
        run_basic_tests(interface, args);
    }
}
