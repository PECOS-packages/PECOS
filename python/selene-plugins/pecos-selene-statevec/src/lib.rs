// Copyright 2025 The PECOS Developers
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

//! PECOS `StateVec` simulator plugin for the Selene quantum emulator.
//!
//! This crate provides a Selene-compatible plugin wrapping the PECOS state vector simulator.
//! Unlike stabilizer simulators, this supports arbitrary rotation angles, making it suitable
//! for simulating any quantum circuit.

use anyhow::{Result, anyhow, bail};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};
use selene_core::export_simulator_plugin;
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::io::Write;
use std::sync::Arc;

/// The PECOS `StateVec` simulator wrapped for Selene compatibility.
pub struct StateVecSimulator {
    /// The underlying PECOS state vector simulator
    simulator: StateVec,
    /// Number of qubits in the system
    n_qubits: u64,
    /// Cumulative probability of postselection outcomes
    cumulative_postselect_probability: f64,
}

impl StateVecSimulator {
    /// Convert a `u64` to `usize` for use with the simulator.
    ///
    /// # Safety
    ///
    /// This is safe because `check_memory()` validates that `n_qubits <= 60` before
    /// any simulator is created, and all qubit indices are bounds-checked against
    /// `n_qubits` before this function is called. Thus, the value will always fit
    /// in a `usize` on any platform (even 32-bit).
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }
}

impl SimulatorInterface for StateVecSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        // Create a fresh simulator with the given seed for deterministic behavior
        self.simulator = StateVec::with_seed(Self::to_usize(self.n_qubits), seed);
        self.cumulative_postselect_probability = 1.0;
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

        let q = Self::to_usize(qubit);

        // Calculate the probability of measuring the target value
        let state = self.simulator.state();
        let mut prob_target = 0.0;

        for (i, amp) in state.iter().enumerate() {
            let bit = (i >> q) & 1;
            if (bit == 1) == target_value {
                prob_target += amp.norm_sqr();
            }
        }

        self.cumulative_postselect_probability *= prob_target;

        if prob_target < 1e-10 {
            return Err(anyhow!(
                "Postselection of {target_value} on qubit {qubit} is too unlikely to postselect. \
                 The probability of this outcome is {prob_target:.2e}."
            ));
        }

        // Project onto the target value by measuring
        // Note: StateVec doesn't expose direct state manipulation for postselection,
        // so we measure and verify we got the expected outcome
        let results = self.simulator.mz(&[QubitId(q)]);
        let outcome = results[0].outcome;

        if outcome != target_value {
            // The measurement collapsed to the wrong value
            // Since this is a state vector simulator, we need to recreate the state
            // This is an approximation - in practice, postselection should be done differently
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
            // If we measured 1, apply X to flip to 0
            self.simulator.x(&[q]);
        }

        Ok(())
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        match nth_metric {
            0 => Ok(Some((
                "cumulative_postselect_probability".to_string(),
                MetricValue::F64(self.cumulative_postselect_probability),
            ))),
            _ => Ok(None),
        }
    }

    fn dump_state(&mut self, file: &std::path::Path, qubits: &[u64]) -> Result<()> {
        let handle = std::fs::File::create(file)?;
        let mut writer = std::io::BufWriter::new(handle);

        // Write header identifier
        writer.write_all(b"selene-statevec")?;

        // Write number of qubits and qubit list
        writer.write_all(self.n_qubits.to_le_bytes().as_slice())?;
        writer.write_all((qubits.len() as u64).to_le_bytes().as_slice())?;
        for &q in qubits {
            writer.write_all(q.to_le_bytes().as_slice())?;
        }

        // Write state vector amplitudes
        let state = self.simulator.state();
        for amp in state {
            writer.write_all(amp.re.to_le_bytes().as_slice())?;
            writer.write_all(amp.im.to_le_bytes().as_slice())?;
        }

        Ok(())
    }
}

/// Factory for creating `StateVecSimulator` instances.
#[derive(Default)]
pub struct StateVecSimulatorFactory;

/// Check if there is enough memory to allocate a state vector of the given size.
fn check_memory(n_qubits: u64) -> Result<()> {
    if n_qubits == 0 {
        bail!("Number of qubits must be greater than 0");
    } else if n_qubits > 60 {
        bail!(
            "It is impossible to describe more than 60 qubits in a statevector \
             on a computer with a 64-bit address space."
        );
    }

    // Each amplitude is a Complex64 = 16 bytes (2 * f64)
    let bytes_required = 16_u64.checked_mul(1_u64 << n_qubits);

    match bytes_required {
        Some(bytes) => {
            // Just log a warning for large allocations, but let the OS handle
            // actual memory allocation
            if bytes > 1024 * 1024 * 1024 {
                // > 1GB
                eprintln!(
                    "Warning: Allocating state vector for {n_qubits} qubits requires \
                     approximately {bytes} bytes"
                );
            }
            Ok(())
        }
        None => {
            bail!("Memory requirement overflow for {n_qubits} qubits");
        }
    }
}

impl SimulatorInterfaceFactory for StateVecSimulatorFactory {
    type Interface = StateVecSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();

        // StateVec plugin doesn't require any arguments
        if args.len() > 1 {
            bail!(
                "Expected no arguments for the PECOS StateVec plugin, got {} arguments: {:?}",
                args.len() - 1,
                args.iter().skip(1).collect::<Vec<_>>()
            );
        }

        check_memory(n_qubits)?;

        Ok(Box::new(StateVecSimulator {
            simulator: StateVec::with_seed(StateVecSimulator::to_usize(n_qubits), 0),
            n_qubits,
            cumulative_postselect_probability: 1.0,
        }))
    }
}

// Export the plugin using Selene's macro
export_simulator_plugin!(crate::StateVecSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::StateVecSimulatorFactory;
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(StateVecSimulatorFactory);
        let args: Vec<String> = vec![String::new()];
        run_basic_tests(interface, args);
    }
}
