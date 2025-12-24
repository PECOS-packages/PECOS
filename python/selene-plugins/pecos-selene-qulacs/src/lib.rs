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

//! PECOS Qulacs simulator plugin for the Selene quantum emulator.
//!
//! This crate provides a Selene-compatible plugin wrapping the PECOS Qulacs state vector simulator.
//! Qulacs is a high-performance quantum simulator that supports arbitrary rotation angles.
//!
//! # Attribution
//!
//! This plugin wraps Qulacs, a high-performance quantum circuit simulator developed by the Qulacs team.
//!
//! - **Repository:** <https://github.com/qulacs/qulacs>
//! - **License:** MIT License

use anyhow::{Result, anyhow, bail};
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable};
use pecos_qulacs::QulacsStateVec;
use pecos_rng::PecosRng;
use selene_core::export_simulator_plugin;
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::io::Write;
use std::sync::Arc;

/// The PECOS Qulacs simulator wrapped for Selene compatibility.
pub struct QulacsSimulator {
    /// The underlying PECOS Qulacs state vector simulator
    simulator: QulacsStateVec<PecosRng>,
    /// Number of qubits in the system
    n_qubits: u64,
    /// Cumulative probability of postselection outcomes
    cumulative_postselect_probability: f64,
}

impl QulacsSimulator {
    /// Convert a `u64` to `usize` for use with the simulator.
    ///
    /// # Safety
    ///
    /// This is safe because `check_memory()` validates that `n_qubits <= 60` before
    /// any simulator is created, and all qubit indices are bounds-checked against
    /// `n_qubits` before this function is called. Thus, the value will always fit
    /// in a `usize` on any platform.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }

    /// Convert Selene qubit index to PECOS qubit index.
    ///
    /// PECOS Qulacs internally converts qubit indices from PECOS convention (MSB-first,
    /// qubit 0 = most significant) to Qulacs convention (LSB-first, qubit 0 = least
    /// significant).
    ///
    /// Selene uses LSB-first convention (like Qulacs), so Selene qubit 0 should
    /// ultimately map to Qulacs qubit 0. Since PECOS Qulacs converts PECOS index i
    /// to Qulacs index (n-1-i), we need:
    ///   Selene qubit i -> PECOS qubit (n-1-i) -> Qulacs qubit (n-1-(n-1-i)) = i
    ///
    /// This double conversion ensures Selene qubit indices are preserved in Qulacs.
    #[inline]
    fn convert_qubit(&self, selene_qubit: u64) -> usize {
        Self::to_usize(self.n_qubits - 1 - selene_qubit)
    }
}

impl SimulatorInterface for QulacsSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        // Create a fresh simulator with the given seed for deterministic behavior
        self.simulator = QulacsStateVec::with_seed(Self::to_usize(self.n_qubits), seed);
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

        let q = self.convert_qubit(qubit);

        // RXY(theta, phi) = Rz(phi) * Rx(theta) * Rz(-phi)
        // Gates are applied left-to-right in code but the matrix multiplication
        // is right-to-left, so we apply Rz(-phi) first
        self.simulator.rz(-phi, q);
        self.simulator.rx(theta, q);
        self.simulator.rz(phi, q);

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

        self.simulator.rz(theta, self.convert_qubit(qubit));
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

        let q1 = self.convert_qubit(qubit1);
        let q2 = self.convert_qubit(qubit2);

        // PECOS Qulacs's rzz is implemented correctly using CX decomposition
        // RZZ(theta) = CX(q1, q2) * Rz(theta, q2) * CX(q1, q2)
        self.simulator.rzz(theta, q1, q2);

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

        let converted = self.convert_qubit(qubit);
        let result = self.simulator.mz(converted);
        Ok(result.outcome)
    }

    fn postselect(&mut self, qubit: u64, target_value: bool) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target_value={target_value}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = self.convert_qubit(qubit);

        // Calculate the probability of measuring the target value
        let mut prob_target = 0.0;
        let n_states = 1usize << self.n_qubits;
        for i in 0..n_states {
            let bit = (i >> q) & 1;
            if (bit == 1) == target_value {
                prob_target += self.simulator.probability(i);
            }
        }

        self.cumulative_postselect_probability *= prob_target;

        if prob_target < 1e-10 {
            return Err(anyhow!(
                "Postselection of {target_value} on qubit {qubit} is too unlikely to postselect. \
                 The probability of this outcome is {prob_target:.2e}."
            ));
        }

        // Measure and check if we got the expected outcome
        let result = self.simulator.mz(q);

        if result.outcome != target_value {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target_value={target_value}) failed. \
                 The measurement outcome was {} but postselection to {target_value} was requested.",
                result.outcome
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

        let q = self.convert_qubit(qubit);

        // Measure the qubit and flip if needed to get |0>
        let result = self.simulator.mz(q);
        if result.outcome {
            // If we measured 1, apply X to flip to 0
            self.simulator.x(q);
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
        writer.write_all(b"selene-qulacs")?;

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

/// Factory for creating `QulacsSimulator` instances.
#[derive(Debug, Clone, Copy, Default)]
pub struct QulacsSimulatorFactory;

/// Parse command-line style arguments.
fn parse_args(args: &[impl AsRef<str>]) -> Result<()> {
    for arg in args {
        let arg = arg.as_ref();
        if arg.is_empty() {
            continue;
        }

        if let Some(value) = arg.strip_prefix("--mode=") {
            if value != "state_vector" {
                bail!(
                    "Qulacs plugin only supports state_vector mode, got '{value}'. \
                     Density matrix simulation is not yet implemented."
                );
            }
        } else if arg.starts_with("--") {
            bail!("Unknown argument: {arg}");
        }
        // Ignore positional args (like the empty string from Selene)
    }

    Ok(())
}

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

impl SimulatorInterfaceFactory for QulacsSimulatorFactory {
    type Interface = QulacsSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        parse_args(args)?;
        check_memory(n_qubits)?;

        Ok(Box::new(QulacsSimulator {
            simulator: QulacsStateVec::with_seed(QulacsSimulator::to_usize(n_qubits), 0),
            n_qubits,
            cumulative_postselect_probability: 1.0,
        }))
    }
}

// Export the plugin using Selene's macro
export_simulator_plugin!(crate::QulacsSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::{QulacsSimulatorFactory, parse_args};
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn test_parse_args_default() {
        let args: Vec<&str> = vec![];
        assert!(parse_args(&args).is_ok());
    }

    #[test]
    fn test_parse_args_state_vector() {
        let args = vec!["--mode=state_vector"];
        assert!(parse_args(&args).is_ok());
    }

    #[test]
    fn test_parse_args_density_matrix_fails() {
        let args = vec!["--mode=density_matrix"];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_parse_args_empty_strings() {
        let args = vec!["", "--mode=state_vector", ""];
        assert!(parse_args(&args).is_ok());
    }

    /// Test that a Bell state through the Selene wrapper produces correlated measurements.
    #[test]
    fn test_bell_state_correlation() {
        use selene_core::simulator::SimulatorInterface;
        use selene_core::simulator::interface::SimulatorInterfaceFactory;

        const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;
        const PI: f64 = std::f64::consts::PI;

        let factory = Arc::new(QulacsSimulatorFactory);
        let mut outcomes = [0u32; 4];

        for seed in 0..100u64 {
            let mut sim = factory.clone().init(2, &["--mode=state_vector"]).unwrap();
            sim.shot_start(0, seed).unwrap();

            // Selene's H decomposition on qubit 0
            sim.rxy(0, HALF_PI, -HALF_PI).unwrap();
            sim.rz(0, PI).unwrap();

            // Selene's CNOT decomposition (control=0, target=1)
            sim.rxy(1, HALF_PI, HALF_PI).unwrap();
            sim.rzz(0, 1, HALF_PI).unwrap();
            sim.rz(0, HALF_PI).unwrap();
            sim.rxy(1, HALF_PI, 0.0).unwrap();
            sim.rz(1, -HALF_PI).unwrap();

            // Measure both qubits
            let m0 = sim.measure(0).unwrap();
            let m1 = sim.measure(1).unwrap();

            let idx = usize::from(m0) | (if m1 { 2 } else { 0 });
            outcomes[idx] += 1;
        }

        // Bell state should only produce |00> and |11>, never |01> or |10>
        assert!(
            outcomes[0b01] == 0 && outcomes[0b10] == 0,
            "Bell state should only have |00> and |11>, got {outcomes:?}"
        );
    }

    /// Run Selene's basic conformance tests for the Qulacs plugin.
    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(QulacsSimulatorFactory);
        let args: Vec<String> = vec!["--mode=state_vector".to_string()];
        run_basic_tests(interface, args);
    }
}
