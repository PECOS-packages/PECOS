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
}

/// Register the simulator utils module
pub fn register_simulator_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GateBindingsDict>()?;
    m.add_class::<TableauWrapper>()?;
    Ok(())
}
