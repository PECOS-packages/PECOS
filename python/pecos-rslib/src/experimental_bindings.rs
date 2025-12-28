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

//! Experimental bindings for HUGR symbolic execution.
//!
//! This module provides Python bindings for the symbolic HUGR execution pipeline:
//! 1. Execute a `SimpleHugr` through `SymbolicSparseStab`
//! 2. Get symbolic measurement dependencies (`MeasurementHistory`)
//! 3. Sample efficiently using `MeasurementSampler`

use pecos::qsim::{
    HugrExecutionError, MeasurementHistory, MeasurementSampler, StdSymbolicSparseStab, execute_hugr,
};
use pecos::quantum::read_hugr_envelope;
use pecos_quantum::Circuit;
use pecos_quantum::hugr_convert::SimpleHugr;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::dag_circuit_bindings::PyDagCircuit;

/// Python wrapper for `MeasurementHistory` with sampling capabilities
#[pyclass(name = "SymbolicExecutionResult")]
pub struct PySymbolicExecutionResult {
    history: MeasurementHistory,
}

#[pymethods]
impl PySymbolicExecutionResult {
    /// Number of measurements in the history
    #[getter]
    fn num_measurements(&self) -> usize {
        self.history.len()
    }

    /// Number of deterministic measurements
    #[getter]
    fn num_deterministic(&self) -> usize {
        self.history.deterministic().len()
    }

    /// Number of non-deterministic (random) measurements
    #[getter]
    fn num_nondeterministic(&self) -> usize {
        self.history.nondeterministic().len()
    }

    /// Sample measurement outcomes efficiently.
    ///
    /// This is extremely fast because sampling is reduced to XOR operations
    /// on random bits - no quantum simulation is performed.
    ///
    /// Args:
    ///     `num_shots`: Number of samples to generate
    ///
    /// Returns:
    ///     List of measurement outcome tuples, where each tuple contains
    ///     the outcomes for all measurements in order.
    fn sample(&self, num_shots: usize) -> Vec<Vec<bool>> {
        eprintln!(
            "[DEBUG] sample: num_shots={}, num_measurements={}, deterministic={}, nondeterministic={}",
            num_shots,
            self.history.len(),
            self.history.deterministic().len(),
            self.history.nondeterministic().len()
        );

        eprintln!("[DEBUG] sample: Creating MeasurementSampler...");
        let sampler = MeasurementSampler::new(&self.history);
        eprintln!("[DEBUG] sample: Calling sampler.sample({num_shots})...");

        let result = sampler.sample(num_shots);
        eprintln!(
            "[DEBUG] sample: sampler.sample() returned, shots={}, num_measurements={}",
            result.shots(),
            result.num_measurements()
        );

        // Convert from column-major to row-major format for Python
        let n_shots = result.shots();
        let n_meas = result.num_measurements();
        eprintln!("[DEBUG] sample: Converting to Python format...");

        let output: Vec<Vec<bool>> = (0..n_shots)
            .map(|shot| {
                (0..n_meas)
                    .map(|meas| result.get(shot, meas).into())
                    .collect()
            })
            .collect();

        eprintln!(
            "[DEBUG] sample: Conversion complete, returning {} rows",
            output.len()
        );
        output
    }

    /// Sample and return counts of unique outcomes.
    ///
    /// Args:
    ///     `num_shots`: Number of samples to generate
    ///
    /// Returns:
    ///     Dictionary mapping outcome tuples to their counts
    fn sample_counts(&self, py: Python<'_>, num_shots: usize) -> PyResult<Py<PyDict>> {
        eprintln!(
            "[DEBUG] sample_counts: num_shots={}, num_measurements={}, deterministic={}, nondeterministic={}",
            num_shots,
            self.history.len(),
            self.history.deterministic().len(),
            self.history.nondeterministic().len()
        );

        eprintln!("[DEBUG] Creating MeasurementSampler...");
        let sampler = MeasurementSampler::new(&self.history);
        eprintln!(
            "[DEBUG] MeasurementSampler created, calling sample({num_shots})..."
        );

        let result = sampler.sample(num_shots);
        eprintln!(
            "[DEBUG] sample() returned: shots={}, num_measurements={}",
            result.shots(),
            result.num_measurements()
        );

        // Count occurrences
        let mut counts: std::collections::HashMap<Vec<bool>, usize> =
            std::collections::HashMap::new();
        let n_shots = result.shots();
        let n_meas = result.num_measurements();
        eprintln!(
            "[DEBUG] Counting occurrences: n_shots={n_shots}, n_meas={n_meas}"
        );

        for shot in 0..n_shots {
            let outcome: Vec<bool> = (0..n_meas)
                .map(|meas| result.get(shot, meas).into())
                .collect();
            *counts.entry(outcome).or_insert(0) += 1;
        }
        eprintln!(
            "[DEBUG] Counting complete, unique outcomes: {}",
            counts.len()
        );

        // Convert to Python dict with tuple keys
        let dict = PyDict::new(py);
        for (outcome, count) in counts {
            // Convert bool vec to tuple of ints for use as dict key
            let key: Vec<u8> = outcome.iter().map(|&b| u8::from(b)).collect();
            dict.set_item(key, count)?;
        }
        eprintln!("[DEBUG] sample_counts complete, returning dict");

        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "SymbolicExecutionResult(measurements={}, deterministic={}, random={})",
            self.history.len(),
            self.history.deterministic().len(),
            self.history.nondeterministic().len()
        )
    }

    fn __str__(&self) -> String {
        self.history.to_string()
    }
}

/// Execute a HUGR symbolically and return a result that can be sampled efficiently.
///
/// This function performs symbolic stabilizer simulation on a HUGR circuit.
/// Instead of collapsing measurements to concrete outcomes, it tracks the
/// symbolic dependencies between measurements. This allows generating
/// millions of samples extremely quickly.
///
/// Args:
///     `hugr_bytes`: The HUGR program as bytes (envelope format)
///     `num_qubits`: Number of qubits in the circuit (optional, auto-detected if None)
///
/// Returns:
///     `SymbolicExecutionResult` that can be sampled efficiently
///
/// Raises:
///     `RuntimeError`: If the HUGR contains unsupported gates (non-Clifford)
///     `RuntimeError`: If the HUGR contains control flow (use `SimpleHugr` validation)
///
/// Example:
///     >>> from pecos.experimental import `execute_hugr_symbolic`
///     >>> result = `execute_hugr_symbolic(hugr_bytes`, `num_qubits=5`)
///     >>> samples = `result.sample(1_000_000)`  # Very fast!
///     >>> counts = `result.sample_counts(1_000_000)`
#[pyfunction]
#[pyo3(signature = (hugr_bytes, num_qubits=None))]
pub fn execute_hugr_symbolic(
    hugr_bytes: &Bound<'_, PyBytes>,
    num_qubits: Option<usize>,
) -> PyResult<PySymbolicExecutionResult> {
    let bytes = hugr_bytes.as_bytes();

    // Parse HUGR bytes into a Hugr
    let hugr = read_hugr_envelope(bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse HUGR bytes: {e}")))?;

    // Convert to SimpleHugr using relaxed mode to allow guppy-generated HUGRs
    // which may have CFG wrapper structures but no actual control flow
    let simple_hugr = SimpleHugr::new_relaxed(hugr);

    // Determine number of qubits
    let n_qubits = num_qubits.unwrap_or_else(|| simple_hugr.qubits().len());

    // Create symbolic simulator and execute
    let mut sim = StdSymbolicSparseStab::new(n_qubits);

    execute_hugr(&mut sim, &simple_hugr).map_err(|e| match e {
        HugrExecutionError::UnsupportedGate { gate_type, .. } => PyRuntimeError::new_err(format!(
            "Unsupported gate for stabilizer simulation: {gate_type}. \
                 Only Clifford gates (H, S, CX, CY, CZ, X, Y, Z) are supported."
        )),
        HugrExecutionError::InvalidQubitCount {
            gate_type,
            expected,
            actual,
            ..
        } => PyRuntimeError::new_err(format!(
            "Gate {gate_type} expected {expected} qubits but got {actual}"
        )),
        HugrExecutionError::QubitOutOfBounds {
            qubit, num_qubits, ..
        } => PyRuntimeError::new_err(format!(
            "Qubit {qubit} out of bounds (circuit has {num_qubits} qubits)"
        )),
    })?;

    // Return the measurement history wrapped for Python
    Ok(PySymbolicExecutionResult {
        history: sim.measurement_history().clone(),
    })
}

/// Execute a `DagCircuit` symbolically and return a result that can be sampled efficiently.
///
/// This function performs symbolic stabilizer simulation on a `DagCircuit`.
/// It's a convenience function that avoids HUGR serialization/deserialization
/// when you have a `DagCircuit` directly.
///
/// Args:
///     circuit: The `DagCircuit` to execute
///     `num_qubits`: Number of qubits in the circuit (optional, auto-detected if None)
///
/// Returns:
///     `SymbolicExecutionResult` that can be sampled efficiently
///
/// Raises:
///     `RuntimeError`: If the circuit contains unsupported gates (non-Clifford)
///
/// Example:
///     >>> from pecos.experimental import `execute_dag_circuit_symbolic`
///     >>> from `pecos_rslib` import `DagCircuit`, Gate
///     >>> circuit = `DagCircuit()`
///     >>> `circuit.add_gate(Gate.h`([0]))
///     >>> `circuit.add_gate(Gate.cx`([(0, 1)]))
///     >>> circuit.add_gate(Gate.mz([0]))
///     >>> circuit.add_gate(Gate.mz([1]))
///     >>> result = `execute_dag_circuit_symbolic(circuit`, `num_qubits=2`)
///     >>> samples = `result.sample(1_000_000)`  # Very fast!
#[pyfunction]
#[pyo3(signature = (circuit, num_qubits=None))]
pub fn execute_dag_circuit_symbolic(
    circuit: &PyDagCircuit,
    num_qubits: Option<usize>,
) -> PyResult<PySymbolicExecutionResult> {
    // Determine number of qubits
    let n_qubits = num_qubits.unwrap_or_else(|| circuit.inner.qubits().len());

    // Create symbolic simulator and execute
    let mut sim = StdSymbolicSparseStab::new(n_qubits);

    execute_hugr(&mut sim, &circuit.inner).map_err(|e| match e {
        HugrExecutionError::UnsupportedGate { gate_type, .. } => PyRuntimeError::new_err(format!(
            "Unsupported gate for stabilizer simulation: {gate_type}. \
                 Only Clifford gates (H, S, CX, CY, CZ, X, Y, Z) are supported."
        )),
        HugrExecutionError::InvalidQubitCount {
            gate_type,
            expected,
            actual,
            ..
        } => PyRuntimeError::new_err(format!(
            "Gate {gate_type} expected {expected} qubits but got {actual}"
        )),
        HugrExecutionError::QubitOutOfBounds {
            qubit, num_qubits, ..
        } => PyRuntimeError::new_err(format!(
            "Qubit {qubit} out of bounds (circuit has {num_qubits} qubits)"
        )),
    })?;

    // Return the measurement history wrapped for Python
    Ok(PySymbolicExecutionResult {
        history: sim.measurement_history().clone(),
    })
}

/// Register the experimental module
pub fn register_experimental_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let experimental = pyo3::types::PyModule::new(py, "experimental")?;

    // Add the main functions
    experimental.add_function(wrap_pyfunction!(execute_hugr_symbolic, &experimental)?)?;
    experimental.add_function(wrap_pyfunction!(
        execute_dag_circuit_symbolic,
        &experimental
    )?)?;

    // Add the result class
    experimental.add_class::<PySymbolicExecutionResult>()?;

    // Register in sys.modules for import support
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.experimental", &experimental)?;

    parent.add_submodule(&experimental)?;
    Ok(())
}
