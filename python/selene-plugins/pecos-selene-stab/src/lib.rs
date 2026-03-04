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

//! PECOS `Stab` simulator plugin for the Selene quantum emulator.
//!
//! This crate provides a Selene-compatible plugin wrapping the PECOS stabilizer simulator.
//! As a stabilizer simulator, it can only simulate Clifford operations (rotations that are
//! multiples of pi/2).

use anyhow::{Result, anyhow};
use clap::Parser;
use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, Stab};
use selene_core::export_simulator_plugin;
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::sync::Arc;

/// Represents angles that can be approximated to Clifford rotations (multiples of pi/2).
enum ApproxAngle {
    /// Angle is approximately 0
    Zero,
    /// Angle is approximately pi/2
    FracPi2,
    /// Angle is approximately pi
    Pi,
    /// Angle is approximately 3*pi/2 (or -pi/2)
    Frac3Pi2,
    /// Angle cannot be approximated to a Clifford rotation
    NoSuitableApproximation,
}

/// Command-line parameters for the `Stab` plugin.
#[derive(Parser, Debug)]
struct Params {
    /// Threshold for angle approximation. Angles within this threshold of a
    /// multiple of pi/2 will be rounded to that multiple.
    #[arg(long)]
    angle_threshold: f64,
}

/// The PECOS `Stab` simulator wrapped for Selene compatibility.
pub struct StabSimulator {
    /// The underlying PECOS stabilizer simulator
    simulator: Stab,
    /// Number of qubits in the system
    n_qubits: u64,
    /// Threshold for angle approximation to Clifford rotations
    angle_threshold: f64,
}

impl StabSimulator {
    /// Convert a `u64` to `usize` for use with the simulator.
    ///
    /// # Safety
    ///
    /// This is safe because stabilizer simulators are limited to a reasonable number of qubits
    /// (typically < 10000), and all qubit indices are bounds-checked against `n_qubits` before
    /// this function is called. Thus, the value will always fit in a `usize` on any platform.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }

    /// Attempts to approximate an angle to a multiple of pi/2.
    ///
    /// Returns the closest Clifford angle if within the threshold, otherwise
    /// returns `NoSuitableApproximation`.
    fn get_approximate_angle(&self, theta: f64) -> ApproxAngle {
        // Convert angle to units of pi/2
        let quadrant_float = theta * 2.0 / std::f64::consts::PI;
        // Round to nearest integer
        let quadrant_rounded = quadrant_float.round();
        // Check if we're within the threshold of a multiple of pi/2
        let within_threshold = (quadrant_float - quadrant_rounded).abs() < self.angle_threshold;

        // Map to the appropriate Clifford angle (0, 1, 2, or 3)
        // Using rem_euclid on the f64 to handle negative values correctly,
        // then converting to integer. The result is always 0, 1, 2, or 3.
        let quadrant_mod4 = quadrant_rounded.rem_euclid(4.0);
        // The result of rem_euclid(4.0) on a rounded f64 is always exactly
        // 0.0, 1.0, 2.0, or 3.0, so we can safely compare with epsilon tolerance
        let quadrant = if (quadrant_mod4 - 0.0).abs() < 0.5 {
            0
        } else if (quadrant_mod4 - 1.0).abs() < 0.5 {
            1
        } else if (quadrant_mod4 - 2.0).abs() < 0.5 {
            2
        } else {
            3
        };

        match (within_threshold, quadrant) {
            (true, 0) => ApproxAngle::Zero,
            (true, 1) => ApproxAngle::FracPi2,
            (true, 2) => ApproxAngle::Pi,
            (true, 3) => ApproxAngle::Frac3Pi2,
            _ => ApproxAngle::NoSuitableApproximation,
        }
    }
}

impl SimulatorInterface for StabSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        // Create a fresh simulator with the given seed for deterministic behavior
        self.simulator = Stab::with_seed(Self::to_usize(self.n_qubits), seed);
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

        let approx_theta = self.get_approximate_angle(theta);
        let approx_phi = self.get_approximate_angle(phi);
        let q = QubitId(Self::to_usize(qubit));

        // RXY(theta, phi) = Rz(phi) * Rx(theta) * Rz(-phi)
        // Gates are applied left-to-right in code but the matrix multiplication
        // is right-to-left, so we apply Rz(-phi) first
        match approx_phi {
            ApproxAngle::Zero => (),
            ApproxAngle::FracPi2 => {
                // Rz(-pi/2) = S^dagger = szdg
                self.simulator.szdg(&[q]);
            }
            ApproxAngle::Pi => {
                // Rz(-pi) = Z (same as Rz(pi))
                self.simulator.z(&[q]);
            }
            ApproxAngle::Frac3Pi2 => {
                // Rz(-3pi/2) = Rz(pi/2) = S = sz
                self.simulator.sz(&[q]);
            }
            ApproxAngle::NoSuitableApproximation => {
                return Err(anyhow!(
                    "RXY(qubit={qubit}, theta={theta}, phi={phi}) is not representable in \
                     stabilizer form. Angles must be (approximate) multiples of pi/2 to use \
                     the PECOS Stab simulator."
                ));
            }
        }

        // Apply Rx(theta)
        match approx_theta {
            ApproxAngle::Zero => (),
            ApproxAngle::FracPi2 => {
                self.simulator.sx(&[q]);
            }
            ApproxAngle::Pi => {
                self.simulator.x(&[q]);
            }
            ApproxAngle::Frac3Pi2 => {
                self.simulator.sxdg(&[q]);
            }
            ApproxAngle::NoSuitableApproximation => {
                return Err(anyhow!(
                    "RXY(qubit={qubit}, theta={theta}, phi={phi}) is not representable in \
                     stabilizer form. Angles must be (approximate) multiples of pi/2 to use \
                     the PECOS Stab simulator."
                ));
            }
        }

        // Apply Rz(phi)
        match approx_phi {
            ApproxAngle::Zero => (),
            ApproxAngle::FracPi2 => {
                // Rz(pi/2) = S = sz
                self.simulator.sz(&[q]);
            }
            ApproxAngle::Pi => {
                // Rz(pi) = Z
                self.simulator.z(&[q]);
            }
            ApproxAngle::Frac3Pi2 => {
                // Rz(3pi/2) = Rz(-pi/2) = S^dagger = szdg
                self.simulator.szdg(&[q]);
            }
            ApproxAngle::NoSuitableApproximation => {
                // Already handled above, but included for completeness
                return Err(anyhow!(
                    "RXY(qubit={qubit}, theta={theta}, phi={phi}) is not representable in \
                     stabilizer form. Angles must be (approximate) multiples of pi/2 to use \
                     the PECOS Stab simulator."
                ));
            }
        }

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

        let approx = self.get_approximate_angle(theta);
        let q = QubitId(Self::to_usize(qubit));

        match approx {
            ApproxAngle::Zero => (),
            ApproxAngle::FracPi2 => {
                self.simulator.sz(&[q]);
            }
            ApproxAngle::Pi => {
                self.simulator.z(&[q]);
            }
            ApproxAngle::Frac3Pi2 => {
                self.simulator.szdg(&[q]);
            }
            ApproxAngle::NoSuitableApproximation => {
                return Err(anyhow!(
                    "RZ(qubit={qubit}, theta={theta}) is not representable in stabilizer form. \
                     Angles must be (approximate) multiples of pi/2 to use the PECOS Stab \
                     simulator."
                ));
            }
        }
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

        let q1 = QubitId(Self::to_usize(qubit1));
        let q2 = QubitId(Self::to_usize(qubit2));
        let approx = self.get_approximate_angle(theta);

        match approx {
            ApproxAngle::Zero => (),
            ApproxAngle::FracPi2 => {
                // sqrt(ZZ) = szz in PECOS
                self.simulator.szz(&[q1, q2]);
            }
            ApproxAngle::Pi => {
                // ZZ = Z tensor Z (up to global phase)
                self.simulator.z(&[q1]).z(&[q2]);
            }
            ApproxAngle::Frac3Pi2 => {
                // sqrt(ZZ)^dagger = szzdg in PECOS
                self.simulator.szzdg(&[q1, q2]);
            }
            ApproxAngle::NoSuitableApproximation => {
                return Err(anyhow!(
                    "RZZ(qubit1={qubit1}, qubit2={qubit2}, theta={theta}) is not representable \
                     in stabilizer form. Angles must be (approximate) multiples of pi/2 to use \
                     the PECOS Stab simulator."
                ));
            }
        }
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

        let q = QubitId(Self::to_usize(qubit));

        // Measure the qubit
        let results = self.simulator.mz(&[q]);
        let result = &results[0];

        // If the outcome doesn't match the target, we need to flip it
        // But for stabilizer states, if the measurement was deterministic and
        // didn't match, postselection is impossible
        if result.outcome != target_value {
            if result.is_deterministic {
                return Err(anyhow!(
                    "Postselect(qubit={qubit}, target_value={target_value}) failed. \
                     The measurement outcome was deterministically {} and cannot be changed.",
                    result.outcome
                ));
            }
            // For non-deterministic measurements, we already collapsed to the wrong state
            // In a proper implementation, we'd need to handle this differently
            // For now, return an error since we can't change a collapsed state
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

        let q = QubitId(Self::to_usize(qubit));

        // Use the measure-and-prepare operation to reset to |0>
        // mpz measures in Z basis and prepares |0> (the +1 eigenstate of Z)
        self.simulator.mpz(&[q]);

        Ok(())
    }

    fn get_metric(&mut self, _nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        // Currently no metrics are exposed
        Ok(None)
    }

    fn dump_state(&mut self, _file: &std::path::Path, _qubits: &[u64]) -> Result<()> {
        // State dumping is not yet implemented for the stabilizer simulator
        Err(anyhow!(
            "State dumping is not yet supported for the PECOS Stab simulator."
        ))
    }
}

/// Factory for creating `StabSimulator` instances.
#[derive(Default)]
pub struct StabSimulatorFactory;

impl SimulatorInterfaceFactory for StabSimulatorFactory {
    type Interface = StabSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();

        match Params::try_parse_from(args) {
            Err(e) => Err(anyhow!("Error parsing arguments to PECOS Stab plugin: {e}")),
            Ok(params) => Ok(Box::new(StabSimulator {
                simulator: Stab::with_seed(StabSimulator::to_usize(n_qubits), 0),
                n_qubits,
                angle_threshold: params.angle_threshold,
            })),
        }
    }
}

// Export the plugin using Selene's macro
export_simulator_plugin!(crate::StabSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::StabSimulatorFactory;
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn basic_conformance_test() {
        let interface = Arc::new(StabSimulatorFactory);
        let args = vec![String::new(), "--angle-threshold=0.001".to_string()];
        run_basic_tests(interface, args);
    }
}
