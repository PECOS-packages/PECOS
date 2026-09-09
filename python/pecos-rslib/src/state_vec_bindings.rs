// Copyright 2026 The PECOS Developers
use crate::prelude::*;
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

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PySet, PyTuple};

use crate::pecos_array::Array;
use crate::simulator_utils::{
    SymbolEntry, extract_angle, extract_angles, supports_exact, validate_supported_symbol,
};

/// The struct represents the state-vector simulator exposed to Python
#[pyclass(name = "StateVec", module = "pecos_rslib")]
pub struct PyStateVec {
    inner: StateVec,
}

#[cfg(test)]
crate::simulator_utils::direct_surface_test!(direct_surface_matches_predicate, {
    PyStateVec::new(2, None)
});

fn supports(entry: &SymbolEntry) -> bool {
    supports_exact! { entry;
        I => ["I"];
        X => ["X"];
        Y => ["Y"];
        Z => ["Z"];
        H => ["H", "H1", "H+z+x"];
        H2 => ["H2", "H-z-x"];
        H3 => ["H3", "H+y-z"];
        H4 => ["H4", "H-y-z"];
        H5 => ["H5", "H-x+y"];
        H6 => ["H6", "H-x-y"];
        F => ["F", "F1"];
        Fdg => ["Fdg", "F1d", "F1dg"];
        F2 => ["F2"];
        F2dg => ["F2dg", "F2d"];
        F3 => ["F3"];
        F3dg => ["F3dg", "F3d"];
        F4 => ["F4"];
        F4dg => ["F4dg", "F4d"];
        Sx => ["Q", "SX", "SqrtX"];
        Sxdg => ["Qd", "SXdg", "SqrtXd"];
        Sy => ["R", "SY", "SqrtY"];
        Sydg => ["Rd", "SYdg", "SqrtYd"];
        Sz => ["S", "SZ", "SqrtZ"];
        Szdg => ["Sd", "SZdg", "SqrtZd"];
        T => ["T"];
        Tdg => ["Tdg"];
        Pz => ["PZ"];
        Pnz => ["PNZ"];
        Px => ["PX"];
        Pnx => ["PNX"];
        Py => ["PY"];
        Pny => ["PNY"];
        InitZ => ["Init", "Init +Z", "init |0>", "leak", "leak |0>", "unleak |0>"];
        InitNz => ["Init -Z", "init |1>", "leak |1>", "unleak |1>"];
        InitX => ["Init +X", "init |+>"];
        InitNx => ["Init -X", "init |->"];
        InitY => ["Init +Y", "init |+i>"];
        InitNy => ["Init -Y", "init |-i>"];
        Mz => ["MZ"];
        MeasureZ => ["Measure", "measure Z", "Measure +Z"];
        Mx => ["MX", "Measure +X"];
        My => ["MY", "Measure +Y"];
        Rx => ["RX"];
        Ry => ["RY"];
        Rz => ["RZ"];
        Rxy1q => ["RXY1Q", "R1XY"];
        U => ["U"];
        Cx => ["CX", "CNOT"];
        Cy => ["CY"];
        Cz => ["CZ"];
        Sxx => ["SXX", "SqrtXX"];
        Sxxdg => ["SXXdg", "SqrtXXd", "SqrtXXdg"];
        Syy => ["SYY", "SqrtYY"];
        Syydg => ["SYYdg", "SqrtYYd", "SqrtYYdg"];
        Szz => ["SZZ", "SqrtZZ"];
        Szzdg => ["SZZdg", "SqrtZZd", "SqrtZZdg"];
        Swap => ["SWAP"];
        G => ["G", "G2"];
        Rxx => ["RXX"];
        Ryy => ["RYY"];
        Rzz => ["RZZ"];
        RxxRyyRzz => ["RXXRYYRZZ", "RZZRYYRXX", "R2XXYYZZ", "RXXYYZZ"];
        Ii => ["II"];
        Crx => ["CRX"];
        Cry => ["CRY"];
        Crz => ["CRZ"];
    }
}

#[pymethods]
impl PyStateVec {
    /// Creates a new state-vector simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    pub fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        PyStateVec {
            inner: match seed {
                Some(s) => StateVec::with_seed(num_qubits, s),
                None => StateVec::new(num_qubits),
            },
        }
    }

    /// Resets the quantum state to the all-zero state
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    /// Executes a single-qubit gate based on the provided symbol and location
    ///
    /// `symbol`: The gate symbol (e.g., "X", "H", "Z")
    /// `location`: The qubit index to apply the gate to
    /// `params`: Optional parameters for parameterized gates (currently unused here)
    ///
    /// Returns an optional result, usually `None` unless a measurement is performed
    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        let q = &[QubitId(location)];
        match symbol {
            "X" => {
                self.inner.x(q);
                Ok(None)
            }
            "Y" => {
                self.inner.y(q);
                Ok(None)
            }
            "Z" => {
                self.inner.z(q);
                Ok(None)
            }
            "RX" => {
                let angle = extract_angle(params, "RX")?;
                self.inner.rx(angle, q);
                Ok(None)
            }
            "RY" => {
                let angle = extract_angle(params, "RY")?;
                self.inner.ry(angle, q);
                Ok(None)
            }
            "RZ" => {
                let angle = extract_angle(params, "RZ")?;
                self.inner.rz(angle, q);
                Ok(None)
            }
            "RXY1Q" | "R1XY" => {
                let angles = extract_angles(params, "RXY1Q", GateType::RXY1Q.angle_arity())?;
                self.inner.rxy1q(angles[0], angles[1], q);
                Ok(None)
            }
            "U" => {
                let angles = extract_angles(params, "U", GateType::U.angle_arity())?;
                self.inner.u(angles[0], angles[1], angles[2], q);
                Ok(None)
            }

            "T" => {
                self.inner.t(q);
                Ok(None)
            }

            "Tdg" => {
                self.inner.tdg(q);
                Ok(None)
            }

            "H" | "H1" | "H+z+x" => {
                self.inner.h(q);
                Ok(None)
            }
            "H2" | "H-z-x" => {
                self.inner.h2(q);
                Ok(None)
            }
            "H3" | "H+y-z" => {
                self.inner.h3(q);
                Ok(None)
            }
            "H4" | "H-y-z" => {
                self.inner.h4(q);
                Ok(None)
            }
            "H5" | "H-x+y" => {
                self.inner.h5(q);
                Ok(None)
            }
            "H6" | "H-x-y" => {
                self.inner.h6(q);
                Ok(None)
            }
            "F" | "F1" => {
                self.inner.f(q);
                Ok(None)
            }
            "Fdg" | "F1d" | "F1dg" => {
                self.inner.fdg(q);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(q);
                Ok(None)
            }
            "F2dg" | "F2d" => {
                self.inner.f2dg(q);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(q);
                Ok(None)
            }
            "F3dg" | "F3d" => {
                self.inner.f3dg(q);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(q);
                Ok(None)
            }
            "F4dg" | "F4d" => {
                self.inner.f4dg(q);
                Ok(None)
            }
            "MZ" | "Measure" | "Measure +Z" | "measure Z" => {
                let result = self
                    .inner
                    .mz(q)
                    .into_iter()
                    .next()
                    .expect("single-qubit measurement returned no result");
                Ok(Some(u8::from(result.outcome)))
            }
            "MX" | "Measure +X" => {
                let result = self
                    .inner
                    .mx(q)
                    .into_iter()
                    .next()
                    .expect("single-qubit measurement returned no result");
                Ok(Some(u8::from(result.outcome)))
            }
            "MY" | "Measure +Y" => {
                let result = self
                    .inner
                    .my(q)
                    .into_iter()
                    .next()
                    .expect("single-qubit measurement returned no result");
                Ok(Some(u8::from(result.outcome)))
            }
            // Gate aliases - alternative names for common gates
            "I" => Ok(None), // Identity gate - no operation
            "Q" | "SX" | "SqrtX" => {
                self.inner.sx(q);
                Ok(None)
            }
            "Qd" | "SXdg" | "SqrtXd" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "R" | "SY" | "SqrtY" => {
                self.inner.sy(q);
                Ok(None)
            }
            "Rd" | "SYdg" | "SqrtYd" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "S" | "SZ" | "SqrtZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "Sd" | "SZdg" | "SqrtZd" => {
                self.inner.szdg(q);
                Ok(None)
            }
            "Init" | "Init +Z" | "init |0>" | "leak" | "leak |0>" | "unleak |0>" | "PZ" => {
                self.inner.pz(q);
                Ok(None)
            }
            "Init -Z" | "init |1>" | "leak |1>" | "unleak |1>" | "PNZ" => {
                self.inner.pnz(q);
                Ok(None)
            }
            "Init +X" | "init |+>" | "PX" => {
                self.inner.px(q);
                Ok(None)
            }
            "Init -X" | "init |->" | "PNX" => {
                self.inner.pnx(q);
                Ok(None)
            }
            "Init +Y" | "init |+i>" | "PY" => {
                self.inner.py(q);
                Ok(None)
            }
            "Init -Y" | "init |-i>" | "PNY" => {
                self.inner.pny(q);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported single-qubit gate",
            )),
        }
    }

    /// Executes a two-qubit gate based on the provided symbol and locations
    ///
    /// `symbol`: The gate symbol (e.g., "CX", "CZ")
    /// `location`: A tuple specifying the two qubits to apply the gate to
    /// `params`: Optional parameters for parameterized gates (currently unused here)
    ///
    /// Returns an optional result, usually `None` unless a measurement is performed
    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params))]
    fn run_2q_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        if location.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires exactly 2 qubit locations",
            ));
        }

        let q1: usize = location.get_item(0)?.extract()?;
        let q2: usize = location.get_item(1)?.extract()?;
        let pair = &[(QubitId(q1), QubitId(q2))];

        match symbol {
            "CX" | "CNOT" => {
                self.inner.cx(pair);
                Ok(None)
            }
            "CY" => {
                self.inner.cy(pair);
                Ok(None)
            }
            "CZ" => {
                self.inner.cz(pair);
                Ok(None)
            }
            "SXX" | "SqrtXX" => {
                self.inner.sxx(pair);
                Ok(None)
            }
            "SXXdg" | "SqrtXXd" | "SqrtXXdg" => {
                self.inner.sxxdg(pair);
                Ok(None)
            }
            "SYY" | "SqrtYY" => {
                self.inner.syy(pair);
                Ok(None)
            }
            "SYYdg" | "SqrtYYd" | "SqrtYYdg" => {
                self.inner.syydg(pair);
                Ok(None)
            }
            "SZZ" | "SqrtZZ" => {
                self.inner.szz(pair);
                Ok(None)
            }
            "SZZdg" | "SqrtZZd" | "SqrtZZdg" => {
                self.inner.szzdg(pair);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(pair);
                Ok(None)
            }
            "G2" | "G" => {
                self.inner.g(pair);
                Ok(None)
            }
            "RXX" => {
                let angle = extract_angle(params, "RXX")?;
                self.inner.rxx(angle, pair);
                Ok(None)
            }
            "RYY" => {
                let angle = extract_angle(params, "RYY")?;
                self.inner.ryy(angle, pair);
                Ok(None)
            }
            "RZZ" => {
                let angle = extract_angle(params, "RZZ")?;
                self.inner.rzz(angle, pair);
                Ok(None)
            }

            "CRX" | "CRY" | "CRZ" => {
                let Some(params) = params else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter missing for controlled rotation gate",
                    ));
                };
                let angle = match params.get_item("angle") {
                    Ok(Some(py_any)) => py_any.extract::<f64>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Expected a float radians parameter for controlled rotation gate",
                        )
                    })?,
                    Ok(None) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for controlled rotation gate",
                        ));
                    }
                    Err(err) => return Err(err),
                };
                let gates: Vec<Gate> = match symbol {
                    "CRX" => {
                        pecos_core::controlled_rotations::lower_crx(angle, pair[0].0, pair[0].1)
                            .into()
                    }
                    "CRY" => {
                        pecos_core::controlled_rotations::lower_cry(angle, pair[0].0, pair[0].1)
                            .into()
                    }
                    "CRZ" => {
                        pecos_core::controlled_rotations::lower_crz(angle, pair[0].0, pair[0].1)
                            .into()
                    }
                    _ => unreachable!(),
                };
                for gate in gates {
                    match gate.gate_type {
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
                        _ => {
                            unreachable!("controlled-rotation lowering emitted an unexpected gate")
                        }
                    }
                }
                Ok(None)
            }

            "RXXRYYRZZ" | "RZZRYYRXX" | "R2XXYYZZ" | "RXXYYZZ" => {
                let angles =
                    extract_angles(params, "RXXRYYRZZ", GateType::RXXRYYRZZ.angle_arity())?;
                self.inner.rxxryyrzz(angles[0], angles[1], angles[2], pair);
                Ok(None)
            }
            // Gate aliases - alternative names for two-qubit gates
            "II" => Ok(None), // Two-qubit identity - no operation

            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported two-qubit gate",
            )),
        }
    }

    /// Internal gate dispatcher (tuple-based) - for internal use
    ///
    /// `symbol`: The gate symbol
    /// `location`: A tuple specifying the qubits to apply the gate to
    /// `params`: Optional parameters for parameterized gates
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_gate_internal(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match location.len() {
            1 => {
                let qubit: usize = location.get_item(0)?.extract()?;
                self.run_1q_gate(symbol, qubit, params)
            }
            2 => self.run_2q_gate(symbol, location, params),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Gate location must be specified for either 1 or 2 qubits",
            )),
        }
    }

    /// High-level `run_gate` that accepts a set of locations (Python wrapper compatible)
    ///
    /// This is the main API that matches the Python wrapper behavior
    /// Unsupported symbols raise for empty locations; supported symbols return an empty result.
    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        self.run_gate_highlevel(symbol, locations, params, py)
    }

    /// Provides direct access to the current state vector as a Python property
    #[getter]
    #[allow(clippy::items_after_statements)] // Use statements for type imports are clearer when near usage
    fn vector(&mut self, py: Python<'_>) -> PyResult<Py<Array>> {
        // Convert the state vector to a 1D complex ndarray
        use ndarray::Array1;
        let state = self.inner.state();
        let complex_array: Vec<num_complex::Complex64> = state.clone();
        let nd_array = Array1::from(complex_array);

        // Create ArrayData from the ndarray
        use crate::pecos_array::ArrayData;
        let array_data = ArrayData::Complex128(nd_array.into_dyn());

        // Create Array and wrap it as a Python object
        let pecos_array = Array::new(array_data);
        Py::new(py, pecos_array)
    }

    /// Returns the probability of each computational basis state as a real-valued array.
    ///
    /// Each entry is |amplitude|^2 for the corresponding basis state.
    #[getter]
    #[allow(clippy::items_after_statements)]
    fn probabilities(&mut self, py: Python<'_>) -> PyResult<Py<Array>> {
        use ndarray::Array1;

        let state = self.inner.state();
        let probs: Vec<f64> = state.iter().map(num_complex::Complex::norm_sqr).collect();
        let nd_array = Array1::from(probs);

        use crate::pecos_array::ArrayData;
        let array_data = ArrayData::F64(nd_array.into_dyn());
        let pecos_array = Array::new(array_data);
        Py::new(py, pecos_array)
    }

    /// Get the probability of measuring a specific basis state.
    fn probability(&mut self, basis_state: usize) -> PyResult<f64> {
        let state = self.inner.state();
        if basis_state >= state.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "basis_state {basis_state} out of range for {} qubits",
                self.inner.num_qubits()
            )));
        }
        Ok(state[basis_state].norm_sqr())
    }

    /// Get state vector with big-endian qubit ordering (PECOS convention)
    ///
    /// Converts the state vector from little-endian (Rust/hardware convention) to
    /// big-endian (PECOS convention) by reversing the bit order of indices.
    ///
    /// This is significantly faster than doing the conversion in Python as it:
    /// 1. Uses Rust's built-in `reverse_bits()` (often a single CPU instruction)
    /// 2. Avoids Python string formatting and parsing
    /// 3. Performs all indexing operations in contiguous Rust memory
    fn vector_big_endian(&mut self, py: Python<'_>) -> PyResult<Py<Array>> {
        use crate::pecos_array::ArrayData;
        use ndarray::Array1;

        let state = self.inner.state();
        let num_qubits = self.inner.num_qubits();
        let length = state.len();

        // Pre-allocate result vector
        let mut reordered = Vec::with_capacity(length);
        reordered.resize(length, num_complex::Complex64::new(0.0, 0.0));

        // Compute bit-reversed indices and reorder
        // This is much faster than Python's string-based approach
        for (idx, &value) in state.iter().enumerate() {
            // Reverse the bits and shift to keep only num_qubits bits
            // The cast is intentional - num_qubits is always small (< 64)
            #[allow(clippy::cast_possible_truncation)]
            let reversed_idx = idx.reverse_bits() >> (usize::BITS - num_qubits as u32);
            reordered[reversed_idx] = value;
        }

        // Convert to ndarray
        let nd_array = Array1::from(reordered);
        let array_data = ArrayData::Complex128(nd_array.into_dyn());

        // Create Array and wrap it as a Python object
        let pecos_array = Array::new(array_data);
        Py::new(py, pecos_array)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// High-level `run_gate` method that accepts a set of locations
    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate_highlevel(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        let locations_set: Bound<PySet> = locations.clone().cast_into()?;
        validate_supported_symbol(symbol, &locations_set, supports)?;

        // Check if simulate_gate is False
        if let Some(p) = params
            && let Ok(Some(sg)) = p.get_item("simulate_gate")
            && let Ok(false) = sg.extract::<bool>()
        {
            return Ok(output.into());
        }

        for location in locations_set.iter() {
            // Convert location to tuple
            let loc_tuple: Bound<'_, PyTuple> = if location.is_instance_of::<PyTuple>() {
                location.clone().cast_into()?
            } else {
                // Single qubit - wrap in tuple
                PyTuple::new(py, std::slice::from_ref(&location))?
            };

            // Call the underlying run_gate_internal
            let result = self.run_gate_internal(symbol, &loc_tuple, params)?;

            // Only add to output if result is Some (non-zero measurement)
            if let Some(value) = result {
                output.set_item(location, value)?;
            }
        }

        Ok(output.into())
    }

    /// Execute a quantum circuit
    #[pyo3(signature = (circuit, removed_locations=None))]
    fn run_circuit(
        &mut self,
        circuit: &Bound<'_, PyAny>,
        removed_locations: Option<&Bound<'_, PySet>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let results = PyDict::new(py);

        // Iterate over circuit items
        for item in circuit.call_method0("items")?.try_iter()? {
            let item = item?;
            let tuple: Bound<PyTuple> = item.clone().cast_into()?;

            let symbol: String = tuple.get_item(0)?.extract()?;
            let locations_item = tuple.get_item(1)?;
            let locations: Bound<PySet> = locations_item.clone().cast_into()?;
            let params_item = tuple.get_item(2)?;
            let params: Bound<PyDict> = params_item.clone().cast_into()?;

            // Subtract removed_locations if provided
            let final_locations = if let Some(removed) = removed_locations {
                locations.call_method1("__sub__", (removed,))?
            } else {
                locations.clone().into_any()
            };

            // Run the gate
            let gate_results =
                self.run_gate_highlevel(&symbol, &final_locations, Some(&params), py)?;

            // Update results
            results.call_method1("update", (gate_results,))?;
        }

        Ok(results.into())
    }

    fn __reduce__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let state = self.inner.state();
        // Serialize state vector as raw little-endian bytes (16 bytes per Complex64: 2 x f64)
        let mut bytes = Vec::with_capacity(state.len() * 16);
        for c in state {
            bytes.extend_from_slice(&c.re.to_le_bytes());
            bytes.extend_from_slice(&c.im.to_le_bytes());
        }
        let state_bytes = PyBytes::new(py, &bytes);
        let num_qubits = self.inner.num_qubits();

        // Return (StateVec._from_pickle, (num_qubits, state_bytes))
        let cls = py.get_type::<PyStateVec>();
        let from_pickle = cls.getattr("_from_pickle")?;
        PyTuple::new(
            py,
            &[
                from_pickle.into_any(),
                PyTuple::new(
                    py,
                    &[
                        num_qubits.into_pyobject(py)?.into_any(),
                        state_bytes.into_any(),
                    ],
                )?
                .into_any(),
            ],
        )
    }

    #[staticmethod]
    fn _from_pickle(num_qubits: usize, state_bytes: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = state_bytes.as_bytes();
        let expected_len = (1usize << num_qubits) * 16;
        if bytes.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid state bytes length: expected {expected_len}, got {}",
                bytes.len()
            )));
        }

        let mut state = Vec::with_capacity(1 << num_qubits);
        for chunk in bytes.as_chunks::<16>().0 {
            let re = f64::from_le_bytes(
                chunk[..8]
                    .try_into()
                    .expect("16-byte chunks contain an 8-byte prefix"),
            );
            let im = f64::from_le_bytes(
                chunk[8..16]
                    .try_into()
                    .expect("16-byte chunks contain an 8-byte suffix"),
            );
            state.push(num_complex::Complex64::new(re, im));
        }

        let rng: PecosRng = rand::make_rng();
        Ok(PyStateVec {
            inner: StateVec::from_state(&state, rng),
        })
    }

    #[getter]
    fn bindings(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::GateBindingsDict> {
        // Create a Rust GateBindingsDict directly
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::GateBindingsDict::new(
            sim_obj, supports,
        ))
    }
}
