// Copyright 2025 The PECOS Developers
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

use pecos::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

// Monte Carlo engines create independent simulator copies for each thread.
// CppSparseStab implements Send, so each thread gets exclusive access to its own instance.
#[pyclass(name = "SparseSimCpp")]
pub struct PySparseSimCpp {
    inner: CppSparseStab,
}

#[pymethods]
impl PySparseSimCpp {
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        let inner = match seed {
            Some(s) => CppSparseStab::with_seed(num_qubits, s),
            None => CppSparseStab::new(num_qubits),
        };
        PySparseSimCpp { inner }
    }

    fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    fn __repr__(&self) -> String {
        format!("SparseSimCpp(num_qubits={})", self.inner.num_qubits())
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match symbol {
            "X" => {
                self.inner.x(location);
                Ok(None)
            }
            "Y" => {
                self.inner.y(location);
                Ok(None)
            }
            "Z" => {
                self.inner.z(location);
                Ok(None)
            }
            "H" => {
                self.inner.h(location);
                Ok(None)
            }
            "H2" => {
                self.inner.h2(location);
                Ok(None)
            }
            "H3" => {
                self.inner.h3(location);
                Ok(None)
            }
            "H4" => {
                self.inner.h4(location);
                Ok(None)
            }
            "H5" => {
                self.inner.h5(location);
                Ok(None)
            }
            "H6" => {
                self.inner.h6(location);
                Ok(None)
            }
            "F" | "F1" => {
                self.inner.f(location);
                Ok(None)
            }
            "Fdg" | "F1d" => {
                self.inner.fdg(location);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(location);
                Ok(None)
            }
            "F2dg" | "F2d" => {
                self.inner.f2dg(location);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(location);
                Ok(None)
            }
            "F3dg" | "F3d" => {
                self.inner.f3dg(location);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(location);
                Ok(None)
            }
            "F4dg" | "F4d" => {
                self.inner.f4dg(location);
                Ok(None)
            }
            "MZ" => {
                let result = self.inner.mz(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "MX" | "Measure +X" => {
                let result = self.inner.mx(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "MY" | "Measure +Y" => {
                let result = self.inner.my(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "MZForced" => {
                if let Some(params) = params {
                    // Extract forced_outcome as integer first, then convert to bool
                    let forced_int = params
                        .get_item("forced_outcome")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "MZForced requires a 'forced_outcome' parameter",
                            )
                        })?
                        .extract::<i32>()?;
                    let forced_value = forced_int != 0;
                    let result = self.inner.force_measure(location, forced_value);
                    Ok(Some(u8::from(result.outcome)))
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "MZForced requires a 'forced_outcome' parameter",
                    ))
                }
            }
            // Gate aliases - alternative names for common gates
            "I" => Ok(None), // Identity gate - no operation
            "Q" | "SX" | "SqrtX" => {
                self.inner.sx(location);
                Ok(None)
            }
            "Qd" | "SXdg" | "SqrtXdg" => {
                self.inner.sxdg(location);
                Ok(None)
            }
            "R" | "SY" | "SqrtY" => {
                self.inner.sy(location);
                Ok(None)
            }
            "Rd" | "SYdg" | "SqrtYdg" => {
                self.inner.sydg(location);
                Ok(None)
            }
            "S" | "SZ" | "SqrtZ" => {
                self.inner.sz(location);
                Ok(None)
            }
            "Sd" | "SZdg" | "SqrtZdg" => {
                self.inner.szdg(location);
                Ok(None)
            }
            "Measure" | "Measure +Z" | "measure Z" => {
                // Check if forced_outcome parameter is provided
                if let Some(params) = params
                    && let Ok(Some(forced_item)) = params.get_item("forced_outcome")
                {
                    // Has forced_outcome, use forced measurement
                    let forced_int: i32 = forced_item.extract()?;
                    let forced_value = forced_int != 0;
                    let result = self.inner.force_measure(location, forced_value);
                    return Ok(Some(u8::from(result.outcome)));
                }
                // No forced_outcome, use regular measurement
                let result = self.inner.mz(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "Init" | "init |0>" => {
                // Check if forced_outcome parameter is provided
                // If so, do forced measurement + correction (matches old Python behavior)
                if let Some(params) = params
                    && let Ok(Some(forced_item)) = params.get_item("forced_outcome")
                {
                    let forced_int: i32 = forced_item.extract()?;
                    if forced_int != -1 {
                        // Use forced measurement approach
                        let forced_value = forced_int != 0;
                        let result = self.inner.force_measure(location, forced_value);
                        // If measured |1>, flip to |0>
                        if result.outcome {
                            self.inner.x(location);
                        }
                        return Ok(None);
                    }
                }
                // No forced_outcome or forced_outcome==-1, use native preparation
                self.inner.pz(location);
                Ok(None)
            }
            "init |1>" => {
                // Use native preparation gate
                self.inner.pnz(location);
                Ok(None)
            }
            "init |+>" => {
                // Use native preparation gate
                self.inner.px(location);
                Ok(None)
            }
            "init |->" => {
                // Use native preparation gate
                self.inner.pnx(location);
                Ok(None)
            }
            "init |+i>" => {
                // Use native preparation gate
                self.inner.py(location);
                Ok(None)
            }
            "init |-i>" => {
                // Use native preparation gate
                self.inner.pny(location);
                Ok(None)
            }
            "PZForced" => {
                // Alias for "init |0>" with forced_outcome - used in random circuit tests
                // Just handle it the same way as "init |0>"
                if let Some(params) = params
                    && let Ok(Some(forced_item)) = params.get_item("forced_outcome")
                {
                    let forced_int: i32 = forced_item.extract()?;
                    if forced_int != -1 {
                        // Use forced measurement approach
                        let forced_value = forced_int != 0;
                        let result = self.inner.force_measure(location, forced_value);
                        // If measured |1>, flip to |0>
                        if result.outcome {
                            self.inner.x(location);
                        }
                        return Ok(None);
                    }
                }
                // No forced_outcome or forced_outcome==-1, use native preparation
                self.inner.pz(location);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported single-qubit gate: {symbol}"
            ))),
        }
    }

    fn run_2q_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        _params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        if location.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires exactly 2 qubit locations",
            ));
        }

        let q1: usize = location.get_item(0)?.extract()?;
        let q2: usize = location.get_item(1)?.extract()?;
        match symbol {
            "CX" | "CNOT" => {
                self.inner.cx(q1, q2);
                Ok(None)
            }
            "CY" => {
                self.inner.cy(q1, q2);
                Ok(None)
            }
            "CZ" => {
                self.inner.cz(q1, q2);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(q1, q2);
                Ok(None)
            }
            "G2" | "G" => {
                self.inner.g2(q1, q2);
                Ok(None)
            }
            "SXX" | "SqrtXX" => {
                self.inner.sxx(q1, q2);
                Ok(None)
            }
            "SXXdg" | "SqrtXXdg" => {
                self.inner.sxxdg(q1, q2);
                Ok(None)
            }
            // Gate aliases - alternative names for two-qubit gates
            "II" => Ok(None), // Two-qubit identity - no operation
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported two-qubit gate: {symbol}"
            ))),
        }
    }

    /// Internal gate dispatcher (tuple-based) - for internal use
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
                "Gates must have either 1 or 2 qubit locations",
            )),
        }
    }

    /// High-level `run_gate` that accepts a set of locations (Python wrapper compatible)
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

    // Additional methods that mirror SparseSim's API
    fn h(&mut self, qubit: usize) {
        self.inner.h(qubit);
    }

    fn x(&mut self, qubit: usize) {
        self.inner.x(qubit);
    }

    fn y(&mut self, qubit: usize) {
        self.inner.y(qubit);
    }

    fn z(&mut self, qubit: usize) {
        self.inner.z(qubit);
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.inner.cx(control, target);
    }

    fn mz(&mut self, qubit: usize) -> bool {
        self.inner.mz(qubit).outcome
    }

    fn mx(&mut self, qubit: usize) -> bool {
        self.inner.mx(qubit).outcome
    }

    fn my(&mut self, qubit: usize) -> bool {
        self.inner.my(qubit).outcome
    }

    fn stab_tableau(&self) -> String {
        self.inner.stab_tableau()
    }

    fn destab_tableau(&self) -> String {
        self.inner.destab_tableau()
    }

    // Expose preparation gates for testing
    fn py(&mut self, qubit: usize) {
        self.inner.py(qubit);
    }

    fn pny(&mut self, qubit: usize) {
        self.inner.pny(qubit);
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

        // Check if simulate_gate is False
        if let Some(p) = params
            && let Ok(Some(sg)) = p.get_item("simulate_gate")
            && let Ok(false) = sg.extract::<bool>()
        {
            return Ok(output.into());
        }

        // Convert locations to a vector
        let locations_set: Bound<PySet> = locations.clone().cast_into()?;

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

    /// Add faults by running a circuit
    #[pyo3(signature = (circuit, removed_locations=None))]
    fn add_faults(
        &mut self,
        circuit: &Bound<'_, PyAny>,
        removed_locations: Option<&Bound<'_, PySet>>,
        py: Python<'_>,
    ) -> PyResult<()> {
        self.run_circuit(circuit, removed_locations, py)?;
        Ok(())
    }

    #[getter]
    fn bindings(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::GateBindingsDict> {
        // Create a Rust GateBindingsDict directly
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::GateBindingsDict::new(sim_obj))
    }

    #[getter]
    fn stabs(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::TableauWrapper> {
        // Create a Rust TableauWrapper directly with is_stab=true
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::TableauWrapper::new(sim_obj, true))
    }

    #[getter]
    fn destabs(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::TableauWrapper> {
        // Create a Rust TableauWrapper directly with is_stab=false
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::TableauWrapper::new(sim_obj, false))
    }

    #[pyo3(signature = (verbose=None, print_y=None, print_destabs=None))]
    fn print_stabs(
        &self,
        verbose: Option<bool>,
        print_y: Option<bool>,
        print_destabs: Option<bool>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let verbose = verbose.unwrap_or(true);
        let print_y = print_y.unwrap_or(true);
        let print_destabs = print_destabs.unwrap_or(false);

        // Get raw tableaus
        let stabs_raw = self.inner.stab_tableau();
        let adjust_fn = py.import("pecos_rslib")?.getattr("adjust_tableau_string")?;

        // Process stabilizers
        let stabs_lines: Vec<&str> = stabs_raw.lines().collect();
        let mut stabs_formatted = Vec::new();
        for line in stabs_lines {
            let adjusted = adjust_fn.call1((line, true, print_y))?;
            stabs_formatted.push(adjusted.extract::<String>()?);
        }

        if print_destabs {
            // Process destabilizers
            let destabs_raw = self.inner.destab_tableau();
            let destabs_lines: Vec<&str> = destabs_raw.lines().collect();
            let mut destabs_formatted = Vec::new();
            for line in destabs_lines {
                let adjusted = adjust_fn.call1((line, false, print_y))?;
                destabs_formatted.push(adjusted.extract::<String>()?);
            }

            if verbose {
                println!("Stabilizers:");
                for line in &stabs_formatted {
                    println!("{line}");
                }
                println!("Destabilizers:");
                for line in &destabs_formatted {
                    println!("{line}");
                }
            }

            // Return tuple of (stabs, destabs) - convert to Python lists first, then tuple
            let stabs_list = PyList::new(py, stabs_formatted)?;
            let destabs_list = PyList::new(py, destabs_formatted)?;
            let tuple = PyTuple::new(py, [stabs_list.as_any(), destabs_list.as_any()])?;
            Ok(tuple.into())
        } else {
            if verbose {
                println!("Stabilizers:");
                for line in &stabs_formatted {
                    println!("{line}");
                }
            }
            // Return just stabs as a list
            let stabs_list = PyList::new(py, stabs_formatted)?;
            Ok(stabs_list.into())
        }
    }
}
