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

use pecos::core::BitSet;
use pecos::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PySet, PyTuple};

#[pyclass(module = "pecos_rslib")]
pub struct SparseSim {
    inner: SparseStab,
}

#[pymethods]
impl SparseSim {
    #[new]
    fn new(num_qubits: usize) -> Self {
        SparseSim {
            inner: SparseStab::new(num_qubits),
        }
    }

    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    fn __repr__(&self) -> String {
        format!("SparseSim(num_qubits={})", self.inner.num_qubits())
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
            "H" => {
                self.inner.h(q);
                Ok(None)
            }
            "H2" => {
                self.inner.h2(q);
                Ok(None)
            }
            "H3" => {
                self.inner.h3(q);
                Ok(None)
            }
            "H4" => {
                self.inner.h4(q);
                Ok(None)
            }
            "H5" => {
                self.inner.h5(q);
                Ok(None)
            }
            "H6" => {
                self.inner.h6(q);
                Ok(None)
            }
            "F" => {
                self.inner.f(q);
                Ok(None)
            }
            "Fdg" => {
                self.inner.fdg(q);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(q);
                Ok(None)
            }
            "F2dg" => {
                self.inner.f2dg(q);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(q);
                Ok(None)
            }
            "F3dg" => {
                self.inner.f3dg(q);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(q);
                Ok(None)
            }
            "F4dg" => {
                self.inner.f4dg(q);
                Ok(None)
            }
            "SX" => {
                self.inner.sx(q);
                Ok(None)
            }
            "SXdg" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "SY" => {
                self.inner.sy(q);
                Ok(None)
            }
            "SYdg" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "SZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "SZdg" => {
                self.inner.szdg(q);
                Ok(None)
            }
            "PZ" => {
                self.inner.pz(q);
                Ok(None)
            }
            "PX" => {
                self.inner.px(q);
                Ok(None)
            }
            "PY" => {
                self.inner.py(q);
                Ok(None)
            }
            "PnZ" => {
                self.inner.pnz(q);
                Ok(None)
            }
            "PnX" => {
                self.inner.pnx(q);
                Ok(None)
            }
            "PnY" => {
                self.inner.pny(q);
                Ok(None)
            }
            "PZForced" => {
                let forced_value = params
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("PZForced requires params")
                    })?
                    .get_item("forced_outcome")?
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "PZForced requires a 'forced_outcome' parameter",
                        )
                    })?
                    .call_method0("__bool__")?
                    .extract::<bool>()?;
                // pz_forced is an inherent method still using old API
                self.inner.pz_forced(location, forced_value);
                Ok(None)
            }
            "MZ" | "MX" | "MY" | "MZForced" => {
                let result = match symbol {
                    "MZ" => self.inner.mz(q).into_iter().next().unwrap(),
                    "MX" => self.inner.mx(q).into_iter().next().unwrap(),
                    "MY" => self.inner.my(q).into_iter().next().unwrap(),
                    "MZForced" => {
                        let forced_value = params
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "MZForced requires params",
                                )
                            })?
                            .get_item("forced_outcome")?
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "MZForced requires a 'forced_outcome' parameter",
                                )
                            })?
                            .call_method0("__bool__")?
                            .extract::<bool>()?;
                        // mz_forced is an inherent method still using old API
                        self.inner.mz_forced(location, forced_value)
                    }
                    _ => unreachable!(),
                };
                Ok(Some(u8::from(result.outcome)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported single-qubit gate",
            )),
        }
    }

    #[pyo3(signature = (symbol, location, _params))]
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
        let pair = &[(QubitId(q1), QubitId(q2))];

        match symbol {
            "CX" => {
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
            "SXX" => {
                self.inner.sxx(pair);
                Ok(None)
            }
            "SXXdg" => {
                self.inner.sxxdg(pair);
                Ok(None)
            }
            "SYY" => {
                self.inner.syy(pair);
                Ok(None)
            }
            "SYYdg" => {
                self.inner.syydg(pair);
                Ok(None)
            }
            "SZZ" => {
                self.inner.szz(pair);
                Ok(None)
            }
            "SZZdg" => {
                self.inner.szzdg(pair);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(pair);
                Ok(None)
            }
            "G2" => {
                self.inner.g(pair);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported two-qubit gate",
            )),
        }
    }

    /// Internal gate dispatcher (tuple-based) - for internal use
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

    fn stab_tableau(&self) -> String {
        self.inner.stab_tableau()
    }

    fn destab_tableau(&self) -> String {
        self.inner.destab_tableau()
    }

    #[pyo3(signature = (verbose=None, _print_y=None, print_destabs=None))]
    fn print_stabs(
        &self,
        verbose: Option<bool>,
        _print_y: Option<bool>,
        print_destabs: Option<bool>,
    ) -> Vec<String> {
        let verbose = verbose.unwrap_or(true);
        // let print_y = print_y.unwrap_or(true);
        let print_destabs = print_destabs.unwrap_or(false);

        let stabs = self.inner.stab_tableau();
        let stab_lines: Vec<String> = stabs.lines().map(String::from).collect();

        if print_destabs {
            let destabs = self.inner.destab_tableau();
            let destab_lines: Vec<String> = destabs.lines().map(String::from).collect();

            if verbose {
                log::debug!("Stabilizers:");
                for line in &stab_lines {
                    log::debug!("{line}");
                }
                log::debug!("Destabilizers:");
                for line in &destab_lines {
                    log::debug!("{line}");
                }
            }

            [stab_lines, destab_lines].concat()
        } else {
            if verbose {
                log::debug!("Stabilizers:");
                for line in &stab_lines {
                    log::debug!("{line}");
                }
            }

            stab_lines
        }
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

    /// Returns the raw gens data (`col_x`, `col_z`, `row_x`, `row_z`) for stabs or destabs.
    fn _gens_data(&self, is_stab: bool) -> crate::simulator_utils::GensData {
        let gens = if is_stab {
            self.inner.stabs()
        } else {
            self.inner.destabs()
        };
        let to_vecs = |sets: &[BitSet]| -> Vec<Vec<usize>> {
            sets.iter().map(|s| s.iter().collect()).collect()
        };
        (
            to_vecs(&gens.col_x),
            to_vecs(&gens.col_z),
            to_vecs(&gens.row_x),
            to_vecs(&gens.row_z),
        )
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
}
