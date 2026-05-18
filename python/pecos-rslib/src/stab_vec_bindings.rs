// Copyright 2026 The PECOS Developers
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

use crate::dtypes::AngleParam;
use crate::prelude::*;
use pecos_simulators::StabVec;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PySet, PyTuple};

#[pyclass(name = "StabVec", module = "pecos_rslib")]
pub struct PyStabVec {
    inner: StabVec,
}

#[pymethods]
impl PyStabVec {
    /// Create a new Clifford+RZ simulator.
    ///
    /// Args:
    ///     `num_qubits`: Number of qubits
    ///     seed: Optional RNG seed for reproducibility
    ///     `pruning_threshold`: Relative pruning threshold (default 1e-8)
    ///     `mc_threshold`: Monte Carlo measurement threshold. Positive integer = MC for
    ///         T > threshold. None = exact measurement only. Default: 2048.
    #[new]
    #[pyo3(signature = (num_qubits, seed=None, pruning_threshold=None, mc_threshold=Some(2048)))]
    fn new(
        num_qubits: usize,
        seed: Option<u64>,
        pruning_threshold: Option<f64>,
        mc_threshold: Option<usize>,
    ) -> Self {
        let mut builder = StabVec::builder(num_qubits);
        if let Some(s) = seed {
            builder = builder.seed(s);
        }
        if let Some(pt) = pruning_threshold {
            builder = builder.pruning_threshold(pt);
        }
        builder = builder.mc_threshold(mc_threshold);
        PyStabVec {
            inner: builder.build(),
        }
    }

    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    fn state_vector(&mut self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let sv = self.inner.state_vector();
        let list: Vec<(f64, f64)> = sv.iter().map(|c| (c.re, c.im)).collect();
        Ok(PyList::new(py, &list)?.unbind())
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
            "I" => Ok(None),
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

            "T" => {
                self.inner.t(q);
                Ok(None)
            }
            "Tdg" => {
                self.inner.tdg(q);
                Ok(None)
            }

            // Rotation gates
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

            // Preparations
            "PZ" | "Init" | "Init +Z" | "init |0>" | "leak" | "leak |0>" | "unleak |0>" => {
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

            // Measurements
            "MZ" | "Measure" | "measure Z" | "Measure +Z" => {
                let result = self
                    .inner
                    .mz(q)
                    .into_iter()
                    .next()
                    .expect("measurement returned no results");
                Ok(Some(u8::from(result.outcome)))
            }

            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported single-qubit gate: {symbol}"
            ))),
        }
    }

    #[pyo3(signature = (symbol, location, params=None))]
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
            "II" => Ok(None),

            // Two-qubit rotation gates
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

            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported two-qubit gate: {symbol}"
            ))),
        }
    }

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

    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate_highlevel(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);

        if let Some(p) = params
            && let Ok(Some(sg)) = p.get_item("simulate_gate")
            && let Ok(false) = sg.extract::<bool>()
        {
            return Ok(output.into());
        }

        let locations_set: Bound<PySet> = locations.clone().cast_into()?;
        if locations_set.is_empty() {
            return Ok(output.into());
        }

        // Fast path: batch dispatch for common Clifford gates without special params
        let has_special_params = params.is_some_and(|p| !p.is_empty());
        if !has_special_params
            && let Some(result) = crate::simulator_utils::try_clifford_batch_dispatch(
                &mut self.inner,
                symbol,
                &locations_set,
                py,
            )?
        {
            return Ok(result);
        }

        // Fallback: per-location dispatch
        for location in locations_set.iter() {
            let loc_tuple: Bound<'_, PyTuple> = if location.is_instance_of::<PyTuple>() {
                location.clone().cast_into()?
            } else {
                PyTuple::new(py, std::slice::from_ref(&location))?
            };

            let result = self.run_gate_internal(symbol, &loc_tuple, params)?;

            if let Some(value) = result {
                output.set_item(location, value)?;
            }
        }

        Ok(output.into())
    }

    #[pyo3(signature = (circuit, removed_locations=None))]
    fn run_circuit(
        &mut self,
        circuit: &Bound<'_, PyAny>,
        removed_locations: Option<&Bound<'_, PySet>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let results = PyDict::new(py);

        for item in circuit.call_method0("items")?.try_iter()? {
            let item = item?;
            let tuple: Bound<PyTuple> = item.clone().cast_into()?;

            let symbol: String = tuple.get_item(0)?.extract()?;
            let locations_item = tuple.get_item(1)?;
            let locations: Bound<PySet> = locations_item.clone().cast_into()?;
            let params_item = tuple.get_item(2)?;
            let params: Bound<PyDict> = params_item.clone().cast_into()?;

            let final_locations = if let Some(removed) = removed_locations {
                locations.call_method1("__sub__", (removed,))?
            } else {
                locations.clone().into_any()
            };

            let gate_results =
                self.run_gate_highlevel(&symbol, &final_locations, Some(&params), py)?;
            results.call_method1("update", (gate_results,))?;
        }

        Ok(results.into())
    }

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
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::GateBindingsDict::new(sim_obj))
    }
}

/// Extract an angle from params dict under the "angle" key.
fn extract_angle(params: Option<&Bound<'_, PyDict>>, gate_name: &str) -> PyResult<Angle64> {
    let params = params.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{gate_name} requires params with 'angle'"
        ))
    })?;
    let py_any = params.get_item("angle")?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{gate_name} requires an 'angle' parameter"
        ))
    })?;
    let angle: AngleParam = py_any.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Expected a valid angle parameter for {gate_name}"
        ))
    })?;
    Ok(angle.0)
}
