// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Simulator utilities implemented in Rust.
//!
//! This module provides `GateBindingsDict` and `TableauWrapper` classes
//! that were previously implemented in Python.

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::collections::HashMap;

use crate::sparse_stab_bindings::adjust_tableau_string;

/// Raw generators data: `(col_x, col_z, row_x, row_z)`.
pub type GensData = (
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
);

/// Special dict that delegates all gate lookups to Rust's `run_gate()`.
///
/// This provides backwards compatibility for code that accesses sim.bindings[`gate_name`].
/// Instead of storing lambdas for every gate, we create them on-demand.
#[pyclass(mapping)]
pub struct GateBindingsDict {
    sim: Py<PyAny>,
    cache: HashMap<String, Py<PyAny>>,
}

impl GateBindingsDict {
    /// Create a new `GateBindingsDict` from Rust code.
    pub fn new(sim: Py<PyAny>) -> Self {
        Self {
            sim,
            cache: HashMap::new(),
        }
    }
}

#[pymethods]
impl GateBindingsDict {
    #[new]
    fn py_new(sim: Py<PyAny>) -> Self {
        Self::new(sim)
    }

    fn __getitem__(&mut self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.cache.get(key) {
            return Ok(cached.clone_ref(py));
        }

        // Create a closure that calls run_gate
        let sim = self.sim.clone_ref(py);
        let gate_name = key.to_string();

        // Create a Python function that wraps the gate call
        let locals = PyDict::new(py);
        locals.set_item("sim", sim)?;
        locals.set_item("gate_name", &gate_name)?;

        // Define a wrapper function in Python using PyModule::from_code
        let code = c_str!(
            r#"
def gate_lambda(simulator, location, **params):
    # Convert location to tuple
    if isinstance(location, int):
        loc_tuple = (location,)
    elif isinstance(location, list):
        loc_tuple = tuple(location)
    else:
        loc_tuple = location

    # Wrap in a set (run_gate expects a set of locations)
    loc_set = {loc_tuple}

    # Call run_gate
    result_dict = sim.run_gate(gate_name, loc_set, **params)

    # Extract the result for this specific location
    if result_dict:
        return result_dict.get(location) or result_dict.get(loc_tuple)
    return None
"#
        );

        // Create a module with the code and inject the sim and gate_name
        let module =
            PyModule::from_code(py, code, c_str!("gate_bindings"), c_str!("gate_bindings"))?;
        module.setattr("sim", self.sim.clone_ref(py))?;
        module.setattr("gate_name", &gate_name)?;
        let gate_lambda = module.getattr("gate_lambda")?.unbind();

        // Cache the lambda
        self.cache
            .insert(key.to_string(), gate_lambda.clone_ref(py));

        Ok(gate_lambda)
    }

    fn __setitem__(&mut self, _py: Python<'_>, key: &str, value: Py<PyAny>) {
        // Store the value in the cache (allows overriding gate lambdas)
        self.cache.insert(key.to_string(), value);
    }

    fn __contains__(&mut self, py: Python<'_>, key: &str) -> bool {
        // Try to get the item - always return true since gates are dynamically created
        self.__getitem__(py, key).is_ok()
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&mut self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> Py<PyAny> {
        match self.__getitem__(py, key) {
            Ok(val) => val,
            Err(_) => default.unwrap_or_else(|| py.None()),
        }
    }

    fn __len__(&self) -> usize {
        self.cache.len()
    }

    fn keys(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
}

/// Wrapper for accessing stabilizer/destabilizer tableaus from simulators.
#[pyclass]
pub struct TableauWrapper {
    sim: Py<PyAny>,
    is_stab: bool,
}

impl TableauWrapper {
    /// Create a new `TableauWrapper` from Rust code.
    pub fn new(sim: Py<PyAny>, is_stab: bool) -> Self {
        Self { sim, is_stab }
    }
}

#[pymethods]
impl TableauWrapper {
    #[new]
    #[pyo3(signature = (sim, *, is_stab))]
    fn py_new(sim: Py<PyAny>, is_stab: bool) -> Self {
        Self::new(sim, is_stab)
    }

    #[pyo3(signature = (*, verbose = false))]
    fn print_tableau(&self, py: Python<'_>, verbose: bool) -> PyResult<Vec<String>> {
        // Get the tableau from the simulator
        let tableau: String = if self.is_stab {
            self.sim.call_method0(py, "stab_tableau")?.extract(py)?
        } else {
            self.sim.call_method0(py, "destab_tableau")?.extract(py)?
        };

        // Split into lines and adjust each
        let lines: Vec<String> = tableau
            .lines()
            .map(|line| adjust_tableau_string(line, self.is_stab, false))
            .collect();

        // Print if verbose
        if verbose {
            for line in &lines {
                println!("{line}");
            }
        }

        Ok(lines)
    }

    /// Helper to get raw gens data from the simulator.
    fn get_gens_data(&self, py: Python<'_>) -> PyResult<GensData> {
        self.sim
            .call_method1(py, "_gens_data", (self.is_stab,))?
            .extract(py)
    }

    #[getter]
    fn col_x(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.0)
    }

    #[getter]
    fn col_z(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.1)
    }

    #[getter]
    fn row_x(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.2)
    }

    #[getter]
    fn row_z(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.3)
    }
}

/// Register the simulator utils module
pub fn register_simulator_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GateBindingsDict>()?;
    m.add_class::<TableauWrapper>()?;
    Ok(())
}

// --- Shared batch dispatch for simulator bindings ---

use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, MeasurementResult};
use pyo3::types::{PySet, PyTuple};

/// Extract a single qubit index from a Python location.
/// Handles both bare ints and 1-tuples like `(0,)` (the `GateBindingsDict` wraps ints in tuples).
pub fn extract_single_qubit(location: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(q) = location.extract::<usize>() {
        return Ok(q);
    }
    if let Ok(tuple) = location.cast::<PyTuple>()
        && tuple.len() == 1
    {
        return tuple.get_item(0)?.extract::<usize>();
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected int or 1-tuple for single-qubit location, got {:?}",
        location.get_type().name()?
    )))
}

/// Collect single-qubit locations from a Python set into a Vec of `QubitIds`.
fn collect_single_qubits(locations: &Bound<'_, PySet>) -> PyResult<Vec<QubitId>> {
    locations
        .iter()
        .map(|l| Ok(QubitId(extract_single_qubit(&l)?)))
        .collect()
}

/// Collect single-qubit locations as raw usize values.
fn collect_single_qubit_indices(locations: &Bound<'_, PySet>) -> PyResult<Vec<usize>> {
    locations.iter().map(|l| extract_single_qubit(&l)).collect()
}

/// Collect two-qubit pair locations from a Python set.
fn collect_pairs(locations: &Bound<'_, PySet>) -> PyResult<Vec<(QubitId, QubitId)>> {
    locations
        .iter()
        .map(|l| {
            let t: (usize, usize) = l.extract()?;
            Ok((QubitId(t.0), QubitId(t.1)))
        })
        .collect()
}

/// Build a measurement output dict from qubit indices and results.
fn build_meas_output(
    py: Python<'_>,
    qubits: &[usize],
    results: Vec<MeasurementResult>,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    for (&q, r) in qubits.iter().zip(results) {
        if r.outcome {
            output.set_item(q, 1u8)?;
        }
    }
    Ok(output.into())
}

/// Try to dispatch a gate in batch mode for any `CliffordGateable` simulator.
///
/// Returns `Some(output_dict)` if the gate was handled, `None` to fall back to
/// per-location dispatch (for parameterized gates, unknown symbols, etc.).
pub fn try_clifford_batch_dispatch<S: CliffordGateable>(
    sim: &mut S,
    symbol: &str,
    locations: &Bound<'_, PySet>,
    py: Python<'_>,
) -> PyResult<Option<Py<PyDict>>> {
    match symbol {
        // Identity
        "I" => return Ok(Some(PyDict::new(py).into())),

        // Single-qubit Clifford gates (no return value)
        "X" | "Y" | "Z" | "H" | "H1" | "H+z+x" | "H2" | "H-z-x" | "H3" | "H+y-z" | "H4"
        | "H-y-z" | "H5" | "H-x+y" | "H6" | "H-x-y" | "F" | "F1" | "Fdg" | "F1d" | "F1dg"
        | "F2" | "F2dg" | "F2d" | "F3" | "F3dg" | "F3d" | "F4" | "F4dg" | "F4d" | "Q" | "SX"
        | "SqrtX" | "Qd" | "SXdg" | "SqrtXd" | "SqrtXdg" | "R" | "SY" | "SqrtY" | "Rd" | "SYdg"
        | "SqrtYd" | "SqrtYdg" | "S" | "SZ" | "SqrtZ" | "Sd" | "SZdg" | "SqrtZd" | "SqrtZdg" => {
            let qubits = collect_single_qubits(locations)?;
            match symbol {
                "X" => {
                    sim.x(&qubits);
                }
                "Y" => {
                    sim.y(&qubits);
                }
                "Z" => {
                    sim.z(&qubits);
                }
                "H" | "H1" | "H+z+x" => {
                    sim.h(&qubits);
                }
                "H2" | "H-z-x" => {
                    sim.h2(&qubits);
                }
                "H3" | "H+y-z" => {
                    sim.h3(&qubits);
                }
                "H4" | "H-y-z" => {
                    sim.h4(&qubits);
                }
                "H5" | "H-x+y" => {
                    sim.h5(&qubits);
                }
                "H6" | "H-x-y" => {
                    sim.h6(&qubits);
                }
                "F" | "F1" => {
                    sim.f(&qubits);
                }
                "Fdg" | "F1d" | "F1dg" => {
                    sim.fdg(&qubits);
                }
                "F2" => {
                    sim.f2(&qubits);
                }
                "F2dg" | "F2d" => {
                    sim.f2dg(&qubits);
                }
                "F3" => {
                    sim.f3(&qubits);
                }
                "F3dg" | "F3d" => {
                    sim.f3dg(&qubits);
                }
                "F4" => {
                    sim.f4(&qubits);
                }
                "F4dg" | "F4d" => {
                    sim.f4dg(&qubits);
                }
                "Q" | "SX" | "SqrtX" => {
                    sim.sx(&qubits);
                }
                "Qd" | "SXdg" | "SqrtXd" | "SqrtXdg" => {
                    sim.sxdg(&qubits);
                }
                "R" | "SY" | "SqrtY" => {
                    sim.sy(&qubits);
                }
                "Rd" | "SYdg" | "SqrtYd" | "SqrtYdg" => {
                    sim.sydg(&qubits);
                }
                "S" | "SZ" | "SqrtZ" => {
                    sim.sz(&qubits);
                }
                "Sd" | "SZdg" | "SqrtZd" | "SqrtZdg" => {
                    sim.szdg(&qubits);
                }
                _ => unreachable!(),
            }
            return Ok(Some(PyDict::new(py).into()));
        }

        // Preparations (no return value)
        "PZ" | "Init" | "Init +Z" | "init |0>" | "leak" | "leak |0>" | "unleak |0>" => {
            sim.pz(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PNZ" | "Init -Z" | "init |1>" | "leak |1>" | "unleak |1>" => {
            sim.pnz(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PX" | "Init +X" | "init |+>" => {
            sim.px(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PNX" | "Init -X" | "init |->" => {
            sim.pnx(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PY" | "Init +Y" | "init |+i>" => {
            sim.py(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PNY" | "Init -Y" | "init |-i>" => {
            sim.pny(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }

        // Measurements (return outcomes)
        "MZ" | "Measure" | "measure Z" | "Measure +Z" => {
            let qubits = collect_single_qubit_indices(locations)?;
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
            let results = sim.mz(&qubit_ids);
            return Ok(Some(build_meas_output(py, &qubits, results)?));
        }
        "MX" | "Measure +X" => {
            let qubits = collect_single_qubit_indices(locations)?;
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
            let results = sim.mx(&qubit_ids);
            return Ok(Some(build_meas_output(py, &qubits, results)?));
        }
        "MY" | "Measure +Y" => {
            let qubits = collect_single_qubit_indices(locations)?;
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
            let results = sim.my(&qubit_ids);
            return Ok(Some(build_meas_output(py, &qubits, results)?));
        }

        // Two-qubit Clifford gates (no return value)
        "CX" | "CNOT" | "CY" | "CZ" | "SZZ" | "SZZdg" | "SXX" | "SXXdg" | "SYY" | "SYYdg"
        | "SqrtZZ" | "SqrtZZd" | "SqrtXX" | "SqrtXXd" | "SqrtYY" | "SqrtYYd" | "SWAP" | "G"
        | "G2" => {
            let pairs = collect_pairs(locations)?;
            match symbol {
                "CX" | "CNOT" => {
                    sim.cx(&pairs);
                }
                "CY" => {
                    sim.cy(&pairs);
                }
                "CZ" => {
                    sim.cz(&pairs);
                }
                "SZZ" | "SqrtZZ" => {
                    sim.szz(&pairs);
                }
                "SZZdg" | "SqrtZZd" => {
                    sim.szzdg(&pairs);
                }
                "SXX" | "SqrtXX" => {
                    sim.sxx(&pairs);
                }
                "SXXdg" | "SqrtXXd" => {
                    sim.sxxdg(&pairs);
                }
                "SYY" | "SqrtYY" => {
                    sim.syy(&pairs);
                }
                "SYYdg" | "SqrtYYd" => {
                    sim.syydg(&pairs);
                }
                "SWAP" => {
                    sim.swap(&pairs);
                }
                "G" | "G2" => {
                    sim.g(&pairs);
                }
                _ => unreachable!(),
            }
            return Ok(Some(PyDict::new(py).into()));
        }

        _ => {}
    }

    Ok(None)
}
