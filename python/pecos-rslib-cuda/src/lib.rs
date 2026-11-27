#![allow(clippy::needless_pass_by_value)] // PyO3 requires owned values from Python
//! CUDA/cuQuantum Python bindings for PECOS quantum simulators.
//!
//! This crate provides PyO3-based Python bindings for the Rust cuQuantum wrappers,
//! enabling GPU-accelerated quantum simulation from Python.

// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, QubitId};
use pecos_cuquantum::{
    ArbitraryRotationGateable, CliffordGateable, CuDensityMat, CuStateVec, CuTensorNet,
    QuantumSimulator, is_cudensitymat_usable as cudensitymat_usable,
    is_cuquantum_available as cuquantum_available, is_custatevec_usable as custatevec_usable,
    is_cutensornet_usable as cutensornet_usable,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn validate_distinct_qubits(gate: &str, qubits: &[usize]) -> PyResult<()> {
    let dups = pecos_core::duplicate_qubits(qubits.iter().copied());
    if !dups.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{gate} requires distinct qubits; duplicated qubits: {dups:?}"
        )));
    }
    Ok(())
}

/// Check if cuQuantum is available on this system.
///
/// Returns True if cuQuantum SDK is installed and accessible.
#[pyfunction]
fn is_cuquantum_available() -> bool {
    cuquantum_available()
}

/// Check if the cuStateVec backend can create a simulator on this system.
#[pyfunction]
fn is_custatevec_usable() -> bool {
    custatevec_usable()
}

/// Check if the cuTensorNet backend can create a handle on this system.
#[pyfunction]
fn is_cutensornet_usable() -> bool {
    cutensornet_usable()
}

/// Check if the cuDensityMat backend can create a simulator on this system.
#[pyfunction]
fn is_cudensitymat_usable() -> bool {
    cudensitymat_usable()
}

/// GPU-accelerated state vector quantum simulator using cuQuantum.
///
/// This simulator can handle up to approximately 30 qubits (limited by GPU memory).
/// It supports all quantum gates including arbitrary rotations.
///
/// Args:
///     `num_qubits`: Number of qubits to simulate.
///
/// Example:
///     >>> sim = CuStateVecSim(4)
///     >>> sim.h([0])
///     >>> sim.cx([0, 1])
///     >>> results = sim.mz([0, 1])
#[pyclass(name = "CuStateVec", unsendable)]
struct PyCuStateVec {
    inner: CuStateVec,
}

impl PyCuStateVec {
    fn apply_lowered_controlled_rotation(&mut self, gates: impl IntoIterator<Item = Gate>) {
        for gate in gates {
            match gate.gate_type {
                GateType::Z => {
                    self.inner.z(&gate.qubits);
                }
                GateType::H => {
                    self.inner.h(&gate.qubits);
                }
                GateType::SX => {
                    self.inner.sx(&gate.qubits);
                }
                GateType::SXdg => {
                    self.inner.sxdg(&gate.qubits);
                }
                GateType::RZ => {
                    self.inner.rz(gate.angles[0], &gate.qubits);
                }
                GateType::RZZ => {
                    self.inner
                        .rzz(gate.angles[0], &[(gate.qubits[0], gate.qubits[1])]);
                }
                _ => unreachable!("controlled-rotation lowering emitted an unexpected gate"),
            }
        }
    }
}

#[pymethods]
impl PyCuStateVec {
    /// Create a new state vector simulator with the given number of qubits.
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        let inner =
            CuStateVec::new(num_qubits).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a new state vector simulator with a specific random seed.
    #[staticmethod]
    fn with_seed(num_qubits: usize, seed: u64) -> PyResult<Self> {
        let inner = CuStateVec::with_seed(num_qubits, seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the number of qubits in this simulator.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Reset the simulator to the |0...0> state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    // =========================================================================
    // Clifford gates
    // =========================================================================

    /// Apply Hadamard gate to the specified qubits.
    fn h(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("H", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h(&qubits);
        Ok(())
    }

    /// Apply Pauli X gate to the specified qubits.
    fn x(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("X", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.x(&qubits);
        Ok(())
    }

    /// Apply Pauli Y gate to the specified qubits.
    fn y(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("Y", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.y(&qubits);
        Ok(())
    }

    /// Apply Pauli Z gate to the specified qubits.
    fn z(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("Z", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.z(&qubits);
        Ok(())
    }

    /// Apply S gate (sqrt(Z)) to the specified qubits.
    fn s(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("S", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sz(&qubits);
        Ok(())
    }

    /// Apply S-dagger gate to the specified qubits.
    fn sdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SDG", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.szdg(&qubits);
        Ok(())
    }

    /// Apply CNOT (CX) gate. First qubit is control, second is target.
    fn cx(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("CX", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.cx(&pairs);
        Ok(())
    }

    /// Apply CY gate. First qubit is control, second is target.
    fn cy(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("CY", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.cy(&pairs);
        Ok(())
    }

    /// Apply CZ gate.
    fn cz(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("CZ", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.cz(&pairs);
        Ok(())
    }

    /// Apply SWAP gate.
    fn swap(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SWAP", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.swap(&pairs);
        Ok(())
    }

    /// Apply iSWAP gate.
    fn iswap(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("ISWAP", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.iswap(&pairs);
        Ok(())
    }

    /// Apply sqrt(X) gate.
    fn sx(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SX", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sx(&qubits);
        Ok(())
    }

    /// Apply sqrt(X)-dagger gate.
    fn sxdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SXDG", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sxdg(&qubits);
        Ok(())
    }

    /// Apply sqrt(Y) gate.
    fn sy(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SY", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sy(&qubits);
        Ok(())
    }

    /// Apply sqrt(Y)-dagger gate.
    fn sydg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SYDG", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sydg(&qubits);
        Ok(())
    }

    /// Apply sqrt(Z) gate (same as S).
    fn sz(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SZ", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sz(&qubits);
        Ok(())
    }

    /// Apply sqrt(Z)-dagger gate (same as Sdg).
    fn szdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SZDG", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.szdg(&qubits);
        Ok(())
    }

    /// Apply sqrt(XX) gate.
    fn sxx(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SXX", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.sxx(&pairs);
        Ok(())
    }

    /// Apply sqrt(XX)-dagger gate.
    fn sxxdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SXXDG", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.sxxdg(&pairs);
        Ok(())
    }

    /// Apply sqrt(YY) gate.
    fn syy(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SYY", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.syy(&pairs);
        Ok(())
    }

    /// Apply sqrt(YY)-dagger gate.
    fn syydg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SYYDG", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.syydg(&pairs);
        Ok(())
    }

    /// Apply sqrt(ZZ) gate.
    fn szz(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SZZ", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.szz(&pairs);
        Ok(())
    }

    /// Apply sqrt(ZZ)-dagger gate.
    fn szzdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("SZZDG", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.szzdg(&pairs);
        Ok(())
    }

    /// Apply F gate (face rotation X->Y->Z->X).
    fn f(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("F", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.f(&qubits);
        Ok(())
    }

    /// Apply F-dagger gate.
    fn fdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("FDG", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.fdg(&qubits);
        Ok(())
    }

    /// Apply H2 gate (Z*H*Z).
    fn h2(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("H2", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h2(&qubits);
        Ok(())
    }

    /// Apply H3 gate.
    fn h3(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("H3", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h3(&qubits);
        Ok(())
    }

    /// Apply H4 gate.
    fn h4(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("H4", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h4(&qubits);
        Ok(())
    }

    /// Apply H5 gate.
    fn h5(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("H5", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h5(&qubits);
        Ok(())
    }

    /// Apply H6 gate.
    fn h6(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("H6", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h6(&qubits);
        Ok(())
    }

    /// Apply G gate (Quantinuum native two-qubit gate).
    fn g(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("G", &qubits)?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.g(&pairs);
        Ok(())
    }

    // =========================================================================
    // Rotation gates (non-Clifford)
    // =========================================================================

    /// Apply T gate (pi/8 gate) to the specified qubits.
    fn t(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("T", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.t(&qubits);
        Ok(())
    }

    /// Apply T-dagger gate to the specified qubits.
    fn tdg(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("TDG", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.tdg(&qubits);
        Ok(())
    }

    /// Apply RX rotation gate.
    fn rx(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RX", &qubits)?;
        let angle = Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.rx(angle, &qubits);
        Ok(())
    }

    /// Apply RY rotation gate.
    fn ry(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RY", &qubits)?;
        let angle = Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.ry(angle, &qubits);
        Ok(())
    }

    /// Apply RZ rotation gate.
    fn rz(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RZ", &qubits)?;
        let angle = Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.rz(angle, &qubits);
        Ok(())
    }

    /// Apply RXX rotation gate.
    fn rxx(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RXX", &qubits)?;
        let angle = Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.rxx(angle, &pairs);
        Ok(())
    }

    /// Apply RYY rotation gate.
    fn ryy(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RYY", &qubits)?;
        let angle = Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.ryy(angle, &pairs);
        Ok(())
    }

    /// Apply RZZ rotation gate.
    fn rzz(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RZZ", &qubits)?;
        let angle = Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.rzz(angle, &pairs);
        Ok(())
    }

    /// Apply RXYXY2Q with angles (theta, phi) in radians.
    fn rxyxy2q(&mut self, theta: f64, phi: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RXYXY2Q", &qubits)?;
        let theta = Angle64::try_from_radians(theta)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let phi = Angle64::try_from_radians(phi)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let pairs: Vec<(QubitId, QubitId)> = qubits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (QubitId(c[0]), QubitId(c[1])))
            .collect();
        self.inner.rxyxy2q(theta, phi, &pairs);
        Ok(())
    }

    /// Apply controlled-RX gate. Pairs of qubits = (control, target).
    fn crx(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("CRX", &qubits)?;
        Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        for pair in qubits.as_chunks::<2>().0 {
            self.apply_lowered_controlled_rotation(
                pecos_core::controlled_rotations::lower_crx(
                    angle,
                    QubitId(pair[0]),
                    QubitId(pair[1]),
                )
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            );
        }
        Ok(())
    }

    /// Apply controlled-RY gate. Pairs of qubits = (control, target).
    fn cry(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("CRY", &qubits)?;
        Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        for pair in qubits.as_chunks::<2>().0 {
            self.apply_lowered_controlled_rotation(
                pecos_core::controlled_rotations::lower_cry(
                    angle,
                    QubitId(pair[0]),
                    QubitId(pair[1]),
                )
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            );
        }
        Ok(())
    }

    /// Apply controlled-RZ gate. Pairs of qubits = (control, target).
    fn crz(&mut self, angle: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("CRZ", &qubits)?;
        Angle64::try_from_radians(angle)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        for pair in qubits.as_chunks::<2>().0 {
            self.apply_lowered_controlled_rotation(
                pecos_core::controlled_rotations::lower_crz(
                    angle,
                    QubitId(pair[0]),
                    QubitId(pair[1]),
                )
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            );
        }
        Ok(())
    }

    /// Apply U gate (general single-qubit rotation).
    fn u(&mut self, theta: f64, phi: f64, lambda: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("U", &qubits)?;
        let theta = Angle64::try_from_radians(theta)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let phi = Angle64::try_from_radians(phi)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let lambda = Angle64::try_from_radians(lambda)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.u(theta, phi, lambda, &qubits);
        Ok(())
    }

    /// Apply RXY1Q gate (rotation in XY plane).
    fn rxy1q(&mut self, theta: f64, phi: f64, qubits: Vec<usize>) -> PyResult<()> {
        validate_distinct_qubits("RXY1Q", &qubits)?;
        let theta = Angle64::try_from_radians(theta)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let phi = Angle64::try_from_radians(phi)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.rxy1q(theta, phi, &qubits);
        Ok(())
    }

    // =========================================================================
    // Measurement
    // =========================================================================

    /// Measure qubits in the X basis.
    fn mx(&mut self, qubits: Vec<usize>) -> PyResult<Vec<u8>> {
        validate_distinct_qubits("MX", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        let results = self.inner.mx(&qubits);
        Ok(results.iter().map(|r| u8::from(r.outcome)).collect())
    }

    /// Measure qubits in the -X basis.
    fn mnx(&mut self, qubits: Vec<usize>) -> PyResult<Vec<u8>> {
        validate_distinct_qubits("MNX", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        let results = self.inner.mnx(&qubits);
        Ok(results.iter().map(|r| u8::from(r.outcome)).collect())
    }

    /// Measure qubits in the Y basis.
    fn my(&mut self, qubits: Vec<usize>) -> PyResult<Vec<u8>> {
        validate_distinct_qubits("MY", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        let results = self.inner.my(&qubits);
        Ok(results.iter().map(|r| u8::from(r.outcome)).collect())
    }

    /// Measure qubits in the -Y basis.
    fn mny(&mut self, qubits: Vec<usize>) -> PyResult<Vec<u8>> {
        validate_distinct_qubits("MNY", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        let results = self.inner.mny(&qubits);
        Ok(results.iter().map(|r| u8::from(r.outcome)).collect())
    }

    /// Measure qubits in the Z basis.
    ///
    /// Returns a list of measurement results (0 or 1) for each qubit.
    fn mz(&mut self, qubits: Vec<usize>) -> PyResult<Vec<u8>> {
        validate_distinct_qubits("MZ", &qubits)?;
        let qubits: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        let results = self.inner.mz(&qubits);
        Ok(results.iter().map(|r| u8::from(r.outcome)).collect())
    }

    /// Sample measurement outcomes from the current state without collapsing it.
    ///
    /// Args:
    ///     `num_samples`: Number of samples to draw.
    ///
    /// Returns:
    ///     List of bitstrings as integers. Each integer represents a measurement outcome
    ///     where bit i corresponds to qubit i.
    fn sample(&mut self, num_samples: usize) -> Vec<u64> {
        self.inner.sample(num_samples)
    }
}

/// Tensor network simulator using NVIDIA cuTensorNet.
///
/// This class manages a cuTensorNet handle for tensor network contractions.
/// Tensor network methods can be used for simulating quantum circuits by
/// contracting tensor networks representing the circuit.
///
/// Use Cases:
///     - Simulating quantum circuits with many qubits but shallow depth
///     - Calculating expectation values
///     - Approximate simulation of larger circuits
///
/// Example:
///     >>> net = `CuTensorNet()`
///     >>> print(f"cuTensorNet version: {`CuTensorNet.version()`}")
#[pyclass(name = "CuTensorNet", unsendable)]
struct PyCuTensorNet {
    #[allow(dead_code)]
    inner: CuTensorNet,
}

#[pymethods]
impl PyCuTensorNet {
    /// Create a new tensor network handle.
    #[new]
    fn new() -> PyResult<Self> {
        let inner = CuTensorNet::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the cuTensorNet version.
    ///
    /// Returns the version as a single integer (e.g., 20000 for version 2.0.0).
    #[staticmethod]
    fn version() -> usize {
        CuTensorNet::version()
    }
}

/// Density matrix simulator using NVIDIA cuDensityMat.
///
/// This simulator manages a cuDensityMat handle and state, providing methods for
/// density matrix operations including noisy quantum simulation.
///
/// Advantages over State Vector:
///     - Can represent mixed states (statistical mixtures)
///     - Natural representation for noise and decoherence
///     - Essential for open quantum system simulation
///
/// Memory Requirements:
///     Density matrices require O(4^n) memory vs O(2^n) for state vectors,
///     limiting practical simulation to fewer qubits.
///
/// Example:
///     >>> sim = CuDensityMat(4)  # 4-qubit density matrix
///     >>> print(f"cuDensityMat version: {`CuDensityMat.version()`}")
#[pyclass(name = "CuDensityMat", unsendable)]
struct PyCuDensityMat {
    inner: CuDensityMat,
}

#[pymethods]
impl PyCuDensityMat {
    /// Create a new density matrix simulator.
    ///
    /// Initializes the state to the pure state |0...0><0...0|.
    ///
    /// Args:
    ///     `num_qubits`: Number of qubits to simulate.
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        let inner =
            CuDensityMat::new(num_qubits).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the number of qubits in this simulator.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Get the cuDensityMat version.
    ///
    /// Returns the version as a single integer.
    #[staticmethod]
    fn version() -> usize {
        CuDensityMat::version()
    }
}

/// A Python module for CUDA/cuQuantum quantum simulation.
#[pymodule]
fn pecos_rslib_cuda(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    log::debug!("pecos_rslib_cuda module initializing...");

    // Add availability check functions
    m.add_function(wrap_pyfunction!(is_cuquantum_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_custatevec_usable, m)?)?;
    m.add_function(wrap_pyfunction!(is_cutensornet_usable, m)?)?;
    m.add_function(wrap_pyfunction!(is_cudensitymat_usable, m)?)?;

    // Add simulator classes
    m.add_class::<PyCuStateVec>()?;
    m.add_class::<PyCuTensorNet>()?;
    m.add_class::<PyCuDensityMat>()?;

    // The Python distribution version from pyproject.toml, injected by build.rs. The crate
    // version (CARGO_PKG_VERSION) is a different number -- it rides the Rust workspace train.
    m.add("__version__", env!("PECOS_PYTHON_VERSION"))?;

    Ok(())
}
