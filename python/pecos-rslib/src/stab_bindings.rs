// Copyright 2026 The PECOS Developers
use pecos::prelude::*;
use pecos::simulators::{ForcedMeasurement, Stab, StabilizerTableauSimulator};
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
use pyo3::types::{PyAny, PyDict, PySet, PyTuple};

#[pyclass(name = "Stab", module = "pecos_rslib")]
pub struct PyStab {
    inner: Stab,
}

#[pymethods]
impl PyStab {
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        PyStab {
            inner: match seed {
                Some(s) => Stab::with_seed(num_qubits, s),
                None => Stab::new(num_qubits),
            },
        }
    }

    fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
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
            // No-op gates
            "I" => Ok(None),
            // Pauli gates
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
            "PZ" => {
                self.inner.pz(q);
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
                // Stab lacks pz_forced, so use mz_forced + conditional X
                let result = self.inner.mz_forced(location, forced_value);
                if result.outcome {
                    self.inner.x(q);
                }
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
                        self.inner.mz_forced(location, forced_value)
                    }
                    _ => unreachable!(),
                };
                Ok(Some(u8::from(result.outcome)))
            }
            // Gate aliases - alternative names for common gates
            "Q" | "SX" | "SqrtX" => {
                self.inner.sx(q);
                Ok(None)
            }
            "Qd" | "SXdg" | "SqrtXd" | "SqrtXdg" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "R" | "SY" | "SqrtY" => {
                self.inner.sy(q);
                Ok(None)
            }
            "Rd" | "SYdg" | "SqrtYd" | "SqrtYdg" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "S" | "SZ" | "SqrtZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "Sd" | "SZdg" | "SqrtZd" | "SqrtZdg" => {
                self.inner.szdg(q);
                Ok(None)
            }
            // Initialization aliases
            "Init" | "Init +Z" | "init |0>" | "leak" | "leak |0>" | "unleak |0>" => {
                // Check if forced_outcome parameter is provided
                // If so, do forced measurement + correction (matches old Python behavior)
                if let Some(params) = params
                    && let Ok(Some(forced_item)) = params.get_item("forced_outcome")
                {
                    let forced_int: i32 = forced_item.extract()?;
                    if forced_int != -1 {
                        // Use forced measurement approach
                        let forced_value = forced_int != 0;
                        let result = self.inner.mz_forced(location, forced_value);
                        // If measured |1>, flip to |0>
                        if result.outcome {
                            self.inner.x(q);
                        }
                        return Ok(None);
                    }
                }
                // No forced_outcome or forced_outcome==-1, use native preparation
                self.inner.pz(q);
                Ok(None)
            }
            "Init -Z" | "init |1>" | "leak |1>" | "unleak |1>" | "PnZ" => {
                self.inner.pnz(q);
                Ok(None)
            }
            "Init +X" | "init |+>" | "PX" => {
                self.inner.px(q);
                Ok(None)
            }
            "Init -X" | "init |->" | "PnX" => {
                self.inner.pnx(q);
                Ok(None)
            }
            "Init +Y" | "init |+i>" | "PY" => {
                self.inner.py(q);
                Ok(None)
            }
            "Init -Y" | "init |-i>" | "PnY" => {
                self.inner.pny(q);
                Ok(None)
            }
            // Measurement aliases
            "Measure" | "measure Z" | "Measure +Z" => {
                // Check if forced_outcome parameter is provided
                if let Some(params) = params
                    && let Ok(Some(forced_item)) = params.get_item("forced_outcome")
                {
                    // Has forced_outcome, use forced measurement
                    let forced_int: i32 = forced_item.extract()?;
                    let forced_value = forced_int != 0;
                    let result = self.inner.mz_forced(location, forced_value);
                    return Ok(Some(u8::from(result.outcome)));
                }
                // No forced_outcome, use regular measurement
                let result = self.inner.mz(q).into_iter().next().unwrap();
                Ok(Some(u8::from(result.outcome)))
            }
            "Measure +X" => {
                let result = self.inner.mx(q).into_iter().next().unwrap();
                Ok(Some(u8::from(result.outcome)))
            }
            "Measure +Y" => {
                let result = self.inner.my(q).into_iter().next().unwrap();
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
        let pair = &[QubitId(q1), QubitId(q2)];

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
            // Two-qubit gate aliases
            "II" => Ok(None), // Two-qubit identity - no operation
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

    fn stab_tableau(&self) -> String {
        self.inner.stab_tableau()
    }

    fn destab_tableau(&self) -> String {
        self.inner.destab_tableau()
    }

    fn _gens_data(&self, is_stab: bool) -> crate::simulator_utils::GensData {
        self.inner.gens_data(is_stab)
    }

    #[getter]
    fn stabs(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::TableauWrapper> {
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::TableauWrapper::new(sim_obj, true))
    }

    #[getter]
    fn destabs(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::TableauWrapper> {
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::TableauWrapper::new(sim_obj, false))
    }

    #[getter]
    fn gens(
        slf: PyRef<'_, Self>,
    ) -> PyResult<(
        crate::simulator_utils::TableauWrapper,
        crate::simulator_utils::TableauWrapper,
    )> {
        let py = slf.py();
        let sim_obj_stab: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        let sim_obj_destab = sim_obj_stab.clone_ref(py);
        Ok((
            crate::simulator_utils::TableauWrapper::new(sim_obj_stab, true),
            crate::simulator_utils::TableauWrapper::new(sim_obj_destab, false),
        ))
    }

    #[getter]
    fn bindings(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::GateBindingsDict> {
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::GateBindingsDict::new(sim_obj))
    }
}
