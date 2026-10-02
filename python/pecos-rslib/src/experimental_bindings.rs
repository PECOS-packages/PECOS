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

//! Experimental bindings for circuit symbolic execution.
//!
//! This module provides Python bindings for the symbolic circuit execution pipeline:
//! 1. Execute a `DagCircuit` through `SymbolicSparseStab`
//! 2. Get symbolic measurement dependencies (`MeasurementHistory`)
//! 3. Sample efficiently using `MeasurementSampler`

use pecos_experimental::{
    DepolarizingNoiseModel, NoisyMeasurementHistory, NoisyMeasurementHistoryBuilder,
    NoisyMeasurementSampler, SymbolicExecutionError, execute_circuit_symbolic,
};
use pecos_simulators::{MeasurementHistory, MeasurementSampler, SymbolicSparseStab};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dag_circuit_bindings::PyDagCircuit;

fn symbolic_execution_error(error: SymbolicExecutionError) -> PyErr {
    match error {
        SymbolicExecutionError::UnsupportedGate { gate_type, .. } => {
            PyRuntimeError::new_err(format!(
                "Unsupported gate for stabilizer simulation: {gate_type}. \
                 Only supported Clifford gates and rotations at Clifford angles are supported."
            ))
        }
        SymbolicExecutionError::InvalidQubitCount {
            gate_type,
            expected,
            actual,
            ..
        } => PyRuntimeError::new_err(format!(
            "Gate {gate_type} expected {expected} qubits but got {actual}"
        )),
        SymbolicExecutionError::QubitOutOfBounds {
            qubit, num_qubits, ..
        } => PyRuntimeError::new_err(format!(
            "Qubit {qubit} out of bounds (circuit has {num_qubits} qubits)"
        )),
    }
}

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
        let sampler = MeasurementSampler::new(&self.history);
        let result = sampler.sample(num_shots);

        // Convert from column-major to row-major format for Python
        let n_shots = result.shots();
        let n_meas = result.num_measurements();
        (0..n_shots)
            .map(|shot| {
                (0..n_meas)
                    .map(|meas| result.get(shot, meas).into())
                    .collect()
            })
            .collect()
    }

    /// Sample and return counts of unique outcomes.
    ///
    /// Args:
    ///     `num_shots`: Number of samples to generate
    ///
    /// Returns:
    ///     Dictionary mapping outcome tuples to their counts
    fn sample_counts(&self, py: Python<'_>, num_shots: usize) -> PyResult<Py<PyDict>> {
        let sampler = MeasurementSampler::new(&self.history);
        let result = sampler.sample(num_shots);

        // Count occurrences
        let mut counts: std::collections::HashMap<Vec<bool>, usize> =
            std::collections::HashMap::new();
        let n_shots = result.shots();
        let n_meas = result.num_measurements();
        for shot in 0..n_shots {
            let outcome: Vec<bool> = (0..n_meas)
                .map(|meas| result.get(shot, meas).into())
                .collect();
            *counts.entry(outcome).or_insert(0) += 1;
        }

        // Convert to Python dict with tuple keys
        let dict = PyDict::new(py);
        for (outcome, count) in counts {
            // Convert bool vec to tuple of ints for use as dict key
            let key: Vec<u8> = outcome.iter().map(|&b| u8::from(b)).collect();
            dict.set_item(key, count)?;
        }

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

/// Execute a `DagCircuit` symbolically and return a result that can be sampled efficiently.
///
/// This function performs symbolic stabilizer simulation on a `DagCircuit`.
/// Measurement records retain the circuit measurement order.
/// Guppy callers can obtain a `DagCircuit` with
/// `trace_program_to_tick_circuit(...).to_dag_circuit()`.
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
    let mut sim = SymbolicSparseStab::new(n_qubits);

    execute_circuit_symbolic(&mut sim, &circuit.inner).map_err(symbolic_execution_error)?;

    // Return the measurement history wrapped for Python
    Ok(PySymbolicExecutionResult {
        history: sim.measurement_history().clone(),
    })
}

// ============================================================================
// Noisy symbolic execution
// ============================================================================

/// Python wrapper for `NoisyMeasurementHistory` with noisy sampling capabilities
#[pyclass(name = "NoisySymbolicExecutionResult")]
pub struct PyNoisySymbolicExecutionResult {
    history: NoisyMeasurementHistory,
}

#[pymethods]
impl PyNoisySymbolicExecutionResult {
    /// Number of measurements in the history
    #[getter]
    fn num_measurements(&self) -> usize {
        self.history.num_measurements()
    }

    /// Number of fault events in the noise model
    #[getter]
    fn num_faults(&self) -> usize {
        self.history.num_faults()
    }

    /// Sample measurement outcomes with noise.
    ///
    /// This samples fault bits (with their probabilities) and random bits,
    /// then computes measurement outcomes via XOR chains.
    ///
    /// Args:
    ///     `num_shots`: Number of samples to generate
    ///
    /// Returns:
    ///     List of measurement outcome lists
    fn sample(&self, num_shots: usize) -> Vec<Vec<bool>> {
        let sampler = NoisyMeasurementSampler::new(&self.history);
        let result = sampler.sample(num_shots);

        // Convert from column-major to row-major format for Python
        let n_shots = result.shots();
        let n_meas = result.num_measurements();
        (0..n_shots)
            .map(|shot| {
                (0..n_meas)
                    .map(|meas| result.get(shot, meas).into())
                    .collect()
            })
            .collect()
    }

    /// Sample and return counts of unique outcomes.
    ///
    /// Args:
    ///     `num_shots`: Number of samples to generate
    ///
    /// Returns:
    ///     Dictionary mapping outcome tuples to their counts
    fn sample_counts(&self, py: Python<'_>, num_shots: usize) -> PyResult<Py<PyDict>> {
        let sampler = NoisyMeasurementSampler::new(&self.history);
        let result = sampler.sample(num_shots);

        // Count occurrences
        let mut counts: std::collections::HashMap<Vec<bool>, usize> =
            std::collections::HashMap::new();
        let n_shots = result.shots();
        let n_meas = result.num_measurements();
        for shot in 0..n_shots {
            let outcome: Vec<bool> = (0..n_meas)
                .map(|meas| result.get(shot, meas).into())
                .collect();
            *counts.entry(outcome).or_insert(0) += 1;
        }

        // Convert to Python dict with bytes keys
        let dict = PyDict::new(py);
        for (outcome, count) in counts {
            let key: Vec<u8> = outcome.iter().map(|&b| u8::from(b)).collect();
            dict.set_item(key, count)?;
        }

        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "NoisySymbolicExecutionResult(measurements={}, faults={})",
            self.history.num_measurements(),
            self.history.num_faults()
        )
    }

    fn __str__(&self) -> String {
        self.history.to_string()
    }
}

/// Execute a `DagCircuit` symbolically with depolarizing noise.
///
/// Args:
///     circuit: The `DagCircuit` to execute
///     `p1`: Single-qubit gate error probability
///     `p2`: Two-qubit gate error probability
///     `p_meas`: Measurement error probability
///     `p_prep`: State preparation error probability
///     `num_qubits`: Number of qubits (optional, auto-detected if None)
///
/// Returns:
///     `NoisySymbolicExecutionResult` that can be sampled with noise
#[pyfunction]
#[pyo3(signature = (circuit, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0, num_qubits=None))]
pub fn execute_dag_circuit_symbolic_noisy(
    circuit: &PyDagCircuit,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
    num_qubits: Option<usize>,
) -> PyResult<PyNoisySymbolicExecutionResult> {
    // Determine number of qubits
    let n_qubits = num_qubits.unwrap_or_else(|| circuit.inner.qubits().len());

    // Create symbolic simulator and execute
    let mut sim = SymbolicSparseStab::new(n_qubits);

    execute_circuit_symbolic(&mut sim, &circuit.inner).map_err(symbolic_execution_error)?;

    // Build noisy measurement history
    let noise_model = DepolarizingNoiseModel::new(p1, p2, p_meas, p_prep);
    let builder = NoisyMeasurementHistoryBuilder::new().with_noise_model(noise_model);
    let noisy_history = builder
        .build_from_circuit(&circuit.inner, sim.measurement_history())
        .map_err(symbolic_execution_error)?;

    Ok(PyNoisySymbolicExecutionResult {
        history: noisy_history,
    })
}

/// Register the experimental module
pub fn register_experimental_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let experimental = pyo3::types::PyModule::new(py, "experimental")?;

    // Add the main functions (noiseless)
    experimental.add_function(wrap_pyfunction!(
        execute_dag_circuit_symbolic,
        &experimental
    )?)?;

    // Add noisy execution functions
    experimental.add_function(wrap_pyfunction!(
        execute_dag_circuit_symbolic_noisy,
        &experimental
    )?)?;

    // Add the result classes
    experimental.add_class::<PySymbolicExecutionResult>()?;
    experimental.add_class::<PyNoisySymbolicExecutionResult>()?;

    // Register in sys.modules for import support
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.experimental", &experimental)?;

    parent.add_submodule(&experimental)?;
    Ok(())
}
