// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

// PyO3 requires specific signatures for Python bindings that conflict with these lints:
// - Dunder methods like __eq__, __hash__ must take &self (not self)
// - Python arguments must be owned types (Vec<T>, not &[T])
// - Py<Self> is the standard pattern for methods that need to return the same object
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::needless_pass_by_value)]

//! Python bindings for quantum circuit representation.
//!
//! This module provides Python bindings for `DagCircuit`, `Gate`, `GateType`, and `QubitId`
//! from the pecos-quantum crate, as well as HUGR conversion utilities.

use crate::dtypes::AngleParam;
use crate::gate_registry_bindings::PyGateRegistry;
use pecos::core::{Angle64, GateQubits, GateSignature, TimeUnits};
use pecos::quantum::{Attribute, DagCircuit, Gate, GateType, QubitId, Tick, TickCircuit};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;

/// Convert a Rust Attribute to a Python object.
fn attribute_to_py(py: Python<'_>, attr: &Attribute) -> Py<PyAny> {
    match attr {
        Attribute::Float(f) => f.into_pyobject(py).unwrap().into_any().unbind(),
        Attribute::Int(i) => i.into_pyobject(py).unwrap().into_any().unbind(),
        Attribute::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        Attribute::Bool(b) => b.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        Attribute::IntList(list) => list.into_pyobject(py).unwrap().into_any().unbind(),
        Attribute::StringList(list) => list.into_pyobject(py).unwrap().into_any().unbind(),
        Attribute::Json(value) => {
            // Convert serde_json::Value to Python via JSON string
            let json_str = serde_json::to_string(value).unwrap_or_default();
            // Use Python's json.loads to parse it
            py.import("json")
                .and_then(|json_mod| json_mod.call_method1("loads", (json_str,)))
                .map_or_else(|_| py.None(), pyo3::Bound::unbind)
        }
    }
}

/// Convert a Python object to a Rust Attribute.
fn py_to_attribute(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Attribute> {
    // Try each type in order of specificity
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Attribute::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Attribute::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Attribute::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Attribute::String(s));
    }
    // Check for list types
    if obj.is_instance_of::<PyList>() {
        // Try to extract as list of ints
        if let Ok(int_list) = obj.extract::<Vec<i64>>() {
            return Ok(Attribute::IntList(int_list));
        }
        // Try to extract as list of strings
        if let Ok(str_list) = obj.extract::<Vec<String>>() {
            return Ok(Attribute::StringList(str_list));
        }
    }
    // Fall back to JSON for dicts and other complex types
    if obj.is_instance_of::<PyDict>() || obj.is_instance_of::<PyList>() {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
        let value: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {e}"))
        })?;
        return Ok(Attribute::Json(value));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Cannot convert {} to Attribute",
        obj.get_type().name()?
    )))
}

/// Convert a `BTreeMap` of attributes to a Python dict.
fn attrs_to_py_dict(
    py: Python<'_>,
    attrs: &std::collections::BTreeMap<String, Attribute>,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in attrs {
        dict.set_item(key, attribute_to_py(py, value))?;
    }
    Ok(dict.into())
}

/// Convert a Python dict to a `BTreeMap` of attributes.
fn py_dict_to_attrs(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
) -> PyResult<std::collections::BTreeMap<String, Attribute>> {
    let mut attrs = std::collections::BTreeMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let attr = py_to_attribute(py, &value)?;
        attrs.insert(key_str, attr);
    }
    Ok(attrs)
}

/// Python wrapper for `QubitId`.
#[pyclass(name = "QubitId", module = "pecos_rslib.quantum", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyQubitId {
    inner: QubitId,
}

#[pymethods]
impl PyQubitId {
    /// Create a new `QubitId` from an integer.
    #[new]
    fn new(id: usize) -> Self {
        Self {
            inner: QubitId::from(id),
        }
    }

    /// Get the integer value of this qubit ID.
    fn __int__(&self) -> usize {
        usize::from(self.inner)
    }

    fn __repr__(&self) -> String {
        format!("QubitId({})", usize::from(self.inner))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

impl From<QubitId> for PyQubitId {
    fn from(inner: QubitId) -> Self {
        Self { inner }
    }
}

impl From<PyQubitId> for QubitId {
    fn from(py_qubit: PyQubitId) -> Self {
        py_qubit.inner
    }
}

/// Python wrapper for `GateType`.
#[pyclass(name = "GateType", module = "pecos_rslib.quantum", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyGateType {
    inner: GateType,
}

#[pymethods]
impl PyGateType {
    /// Get the name of this gate type.
    #[getter]
    fn name(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("GateType.{}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    // Common gate types as class attributes
    #[classattr]
    #[pyo3(name = "H")]
    fn h() -> Self {
        Self { inner: GateType::H }
    }

    #[classattr]
    #[pyo3(name = "X")]
    fn x() -> Self {
        Self { inner: GateType::X }
    }

    #[classattr]
    #[pyo3(name = "Y")]
    fn y() -> Self {
        Self { inner: GateType::Y }
    }

    #[classattr]
    #[pyo3(name = "Z")]
    fn z() -> Self {
        Self { inner: GateType::Z }
    }

    #[classattr]
    #[pyo3(name = "S")]
    fn s() -> Self {
        Self {
            inner: GateType::SZ,
        }
    }

    #[classattr]
    #[pyo3(name = "Sdg")]
    fn sdg() -> Self {
        Self {
            inner: GateType::SZdg,
        }
    }

    #[classattr]
    #[pyo3(name = "SX")]
    fn sx() -> Self {
        Self {
            inner: GateType::SX,
        }
    }

    #[classattr]
    #[pyo3(name = "SXdg")]
    fn sxdg() -> Self {
        Self {
            inner: GateType::SXdg,
        }
    }

    #[classattr]
    #[pyo3(name = "SY")]
    fn sy() -> Self {
        Self {
            inner: GateType::SY,
        }
    }

    #[classattr]
    #[pyo3(name = "SYdg")]
    fn sydg() -> Self {
        Self {
            inner: GateType::SYdg,
        }
    }

    #[classattr]
    #[pyo3(name = "T")]
    fn t() -> Self {
        Self { inner: GateType::T }
    }

    #[classattr]
    #[pyo3(name = "Tdg")]
    fn tdg() -> Self {
        Self {
            inner: GateType::Tdg,
        }
    }

    #[classattr]
    #[pyo3(name = "I")]
    fn i() -> Self {
        Self { inner: GateType::I }
    }

    #[classattr]
    #[pyo3(name = "CX")]
    fn cx() -> Self {
        Self {
            inner: GateType::CX,
        }
    }

    #[classattr]
    #[pyo3(name = "CY")]
    fn cy() -> Self {
        Self {
            inner: GateType::CY,
        }
    }

    #[classattr]
    #[pyo3(name = "CZ")]
    fn cz() -> Self {
        Self {
            inner: GateType::CZ,
        }
    }

    #[classattr]
    #[pyo3(name = "RX")]
    fn rx() -> Self {
        Self {
            inner: GateType::RX,
        }
    }

    #[classattr]
    #[pyo3(name = "RY")]
    fn ry() -> Self {
        Self {
            inner: GateType::RY,
        }
    }

    #[classattr]
    #[pyo3(name = "RZ")]
    fn rz() -> Self {
        Self {
            inner: GateType::RZ,
        }
    }

    #[classattr]
    #[pyo3(name = "RXX")]
    fn rxx() -> Self {
        Self {
            inner: GateType::RXX,
        }
    }

    #[classattr]
    #[pyo3(name = "RYY")]
    fn ryy() -> Self {
        Self {
            inner: GateType::RYY,
        }
    }

    #[classattr]
    #[pyo3(name = "RZZ")]
    fn rzz_attr() -> Self {
        Self {
            inner: GateType::RZZ,
        }
    }

    #[classattr]
    #[pyo3(name = "R1XY")]
    fn r1xy() -> Self {
        Self {
            inner: GateType::R1XY,
        }
    }

    #[classattr]
    #[pyo3(name = "U")]
    fn u() -> Self {
        Self { inner: GateType::U }
    }

    #[classattr]
    #[pyo3(name = "F")]
    fn f() -> Self {
        Self { inner: GateType::F }
    }

    #[classattr]
    #[pyo3(name = "Fdg")]
    fn fdg() -> Self {
        Self {
            inner: GateType::Fdg,
        }
    }

    #[classattr]
    #[pyo3(name = "SXX")]
    fn sxx() -> Self {
        Self {
            inner: GateType::SXX,
        }
    }

    #[classattr]
    #[pyo3(name = "SXXdg")]
    fn sxxdg() -> Self {
        Self {
            inner: GateType::SXXdg,
        }
    }

    #[classattr]
    #[pyo3(name = "SYY")]
    fn syy() -> Self {
        Self {
            inner: GateType::SYY,
        }
    }

    #[classattr]
    #[pyo3(name = "SYYdg")]
    fn syydg() -> Self {
        Self {
            inner: GateType::SYYdg,
        }
    }

    #[classattr]
    #[pyo3(name = "SZZ")]
    fn szz() -> Self {
        Self {
            inner: GateType::SZZ,
        }
    }

    #[classattr]
    #[pyo3(name = "SZZdg")]
    fn szzdg() -> Self {
        Self {
            inner: GateType::SZZdg,
        }
    }

    #[classattr]
    #[pyo3(name = "SWAP")]
    fn swap() -> Self {
        Self {
            inner: GateType::SWAP,
        }
    }

    #[classattr]
    #[pyo3(name = "CH")]
    fn ch() -> Self {
        Self {
            inner: GateType::CH,
        }
    }

    #[classattr]
    #[pyo3(name = "CRZ")]
    fn crz() -> Self {
        Self {
            inner: GateType::CRZ,
        }
    }

    #[classattr]
    #[pyo3(name = "CCX")]
    fn ccx() -> Self {
        Self {
            inner: GateType::CCX,
        }
    }

    #[classattr]
    #[pyo3(name = "Measure")]
    fn mz() -> Self {
        Self {
            inner: GateType::MZ,
        }
    }

    #[classattr]
    #[pyo3(name = "MeasureFree")]
    fn mz_free() -> Self {
        Self {
            inner: GateType::MeasureFree,
        }
    }

    #[classattr]
    #[pyo3(name = "Prep")]
    fn pz() -> Self {
        Self {
            inner: GateType::PZ,
        }
    }

    #[classattr]
    #[pyo3(name = "QAlloc")]
    fn qalloc() -> Self {
        Self {
            inner: GateType::QAlloc,
        }
    }

    #[classattr]
    #[pyo3(name = "QFree")]
    fn qfree() -> Self {
        Self {
            inner: GateType::QFree,
        }
    }

    #[classattr]
    #[pyo3(name = "Custom")]
    fn custom() -> Self {
        Self {
            inner: GateType::Custom,
        }
    }
}

impl From<GateType> for PyGateType {
    fn from(inner: GateType) -> Self {
        Self { inner }
    }
}

impl From<PyGateType> for GateType {
    fn from(py_gate: PyGateType) -> Self {
        py_gate.inner
    }
}

/// Python wrapper for `Gate`.
#[pyclass(name = "Gate", module = "pecos_rslib.quantum", from_py_object)]
#[derive(Clone)]
pub struct PyGate {
    inner: Gate,
}

#[pymethods]
impl PyGate {
    /// Create a new gate.
    ///
    /// # Arguments
    ///
    /// * `gate_type` - The type of gate
    /// * `params` - Gate parameters (angles, etc.)
    /// * `qubits` - Qubit IDs the gate acts on
    #[new]
    #[pyo3(signature = (gate_type, params=None, qubits=None))]
    fn new(gate_type: PyGateType, params: Option<Vec<f64>>, qubits: Option<Vec<usize>>) -> Self {
        let params = params.unwrap_or_default();
        let qubits: Vec<QubitId> = qubits
            .unwrap_or_default()
            .into_iter()
            .map(QubitId::from)
            .collect();
        // For Python API, params are passed as radians for rotation gates
        // Split them into angles and other params based on gate type
        let angle_count = gate_type.inner.angle_arity();
        let angles: Vec<Angle64> = params
            .iter()
            .take(angle_count)
            .map(|&r| Angle64::from_radians(r))
            .collect();
        let other_params: Vec<f64> = params.into_iter().skip(angle_count).collect();
        Self {
            inner: Gate::new(gate_type.inner, angles, other_params, qubits),
        }
    }

    /// Get the gate type.
    #[getter]
    fn gate_type(&self) -> PyGateType {
        PyGateType {
            inner: self.inner.gate_type,
        }
    }

    /// Get the non-angle parameters (e.g., duration for Idle gate).
    #[getter]
    fn params(&self) -> Vec<f64> {
        self.inner.params.to_vec()
    }

    /// Get the rotation angles in radians.
    #[getter]
    fn angles(&self) -> Vec<f64> {
        self.inner
            .angles
            .iter()
            .map(pecos::core::Angle::to_radians)
            .collect()
    }

    /// Get the qubits this gate acts on.
    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.inner.qubits.iter().map(|q| usize::from(*q)).collect()
    }

    /// Check if this is a single-qubit gate.
    fn is_single_qubit(&self) -> bool {
        self.inner.is_single_qubit()
    }

    /// Check if this is a two-qubit gate.
    fn is_two_qubit(&self) -> bool {
        self.inner.is_two_qubit()
    }

    // Factory methods for common gates

    /// Create a Hadamard gate.
    #[staticmethod]
    fn h(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::h(&qubits),
        }
    }

    /// Create an X gate.
    #[staticmethod]
    fn x(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::x(&qubits),
        }
    }

    /// Create a Y gate.
    #[staticmethod]
    fn y(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::y(&qubits),
        }
    }

    /// Create a Z gate.
    #[staticmethod]
    fn z(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::z(&qubits),
        }
    }

    /// Create an Identity gate.
    #[staticmethod]
    fn i(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::i(&qubits),
        }
    }

    /// Create an SX gate (sqrt-X).
    #[staticmethod]
    fn sx(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::sx(&qubits),
        }
    }

    /// Create an `SXdg` gate (sqrt-X dagger).
    #[staticmethod]
    fn sxdg(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::sxdg(&qubits),
        }
    }

    /// Create an SY gate (sqrt-Y).
    #[staticmethod]
    fn sy(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::sy(&qubits),
        }
    }

    /// Create an `SYdg` gate (sqrt-Y dagger).
    #[staticmethod]
    fn sydg(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::sydg(&qubits),
        }
    }

    /// Create a CX (CNOT) gate.
    #[staticmethod]
    fn cx(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Gate::cx(&pairs),
        }
    }

    /// Create a CY gate.
    #[staticmethod]
    fn cy(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Gate::cy(&pairs),
        }
    }

    /// Create a CZ gate.
    #[staticmethod]
    fn cz(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Gate::cz(&pairs),
        }
    }

    /// Create an RX gate.
    #[staticmethod]
    fn rx(angle: AngleParam, qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::rx(angle.0, &qubits),
        }
    }

    /// Create an RY gate.
    #[staticmethod]
    fn ry(angle: AngleParam, qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::ry(angle.0, &qubits),
        }
    }

    /// Create an RZ gate.
    #[staticmethod]
    fn rz(angle: AngleParam, qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::rz(angle.0, &qubits),
        }
    }

    /// Create an RXX gate.
    #[staticmethod]
    fn rxx(angle: AngleParam, pairs: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Gate::rxx(angle.0, &pairs),
        }
    }

    /// Create an RYY gate.
    #[staticmethod]
    fn ryy(angle: AngleParam, pairs: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Gate::ryy(angle.0, &pairs),
        }
    }

    /// Create an RZZ gate.
    #[staticmethod]
    #[pyo3(name = "rzz")]
    fn rzz_gate(angle: AngleParam, pairs: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Gate::rzz(angle.0, &pairs),
        }
    }

    /// Create an R1XY gate.
    #[staticmethod]
    fn r1xy(theta: AngleParam, phi: AngleParam, qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::r1xy(theta.0, phi.0, &qubits),
        }
    }

    /// Create a U gate.
    #[staticmethod]
    #[pyo3(name = "u")]
    fn u_gate(theta: AngleParam, phi: AngleParam, lam: AngleParam, qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::u(theta.0, phi.0, lam.0, &qubits),
        }
    }

    /// Create a Measure gate.
    #[staticmethod]
    fn mz(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::mz(&qubits),
        }
    }

    /// Create a `QAlloc` gate.
    #[staticmethod]
    fn qalloc(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::qalloc(&qubits),
        }
    }

    /// Create a `QFree` gate.
    #[staticmethod]
    fn qfree(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::qfree(&qubits),
        }
    }

    /// Create a `MeasureFree` gate.
    #[staticmethod]
    fn mz_free(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::mz_free(&qubits),
        }
    }

    /// Create a PZ (preparation/reset) gate.
    #[staticmethod]
    fn pz(qubits: Vec<usize>) -> Self {
        Self {
            inner: Gate::pz(&qubits),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Gate({}, params={:?}, qubits={:?})",
            self.inner.gate_type,
            self.inner.params,
            self.inner
                .qubits
                .iter()
                .map(|q| usize::from(*q))
                .collect::<Vec<_>>()
        )
    }
}

impl From<Gate> for PyGate {
    fn from(inner: Gate) -> Self {
        Self { inner }
    }
}

impl From<PyGate> for Gate {
    fn from(py_gate: PyGate) -> Self {
        py_gate.inner
    }
}

// Exception for cycle errors
pyo3::create_exception!(
    pecos_rslib,
    DagCircuitWouldCycleError,
    pyo3::exceptions::PyException
);

/// Python wrapper for `DagCircuit`.
///
/// A directed acyclic graph representation of a quantum circuit where nodes are gates
/// and edges are qubit wires flowing between gates.
#[pyclass(name = "DagCircuit", module = "pecos_rslib.quantum", from_py_object)]
#[derive(Clone)]
pub struct PyDagCircuit {
    pub(crate) inner: DagCircuit,
}

#[pymethods]
impl PyDagCircuit {
    /// Create a new empty circuit.
    #[new]
    fn new() -> Self {
        Self {
            inner: DagCircuit::new(),
        }
    }

    /// Create a new circuit with pre-allocated capacity.
    #[staticmethod]
    fn with_capacity(gates: usize, wires: usize) -> Self {
        Self {
            inner: DagCircuit::with_capacity(gates, wires),
        }
    }

    /// Add a gate to the circuit.
    ///
    /// Returns the node index of the newly added gate.
    fn add_gate(&mut self, gate: PyGate) -> usize {
        self.inner.add_gate(gate.inner)
    }

    /// Remove a gate from the circuit.
    ///
    /// Returns the removed gate if it existed.
    fn remove_gate(&mut self, node: usize) -> Option<PyGate> {
        self.inner.remove_gate(node).map(PyGate::from)
    }

    /// Get the gate at a node.
    fn gate(&self, node: usize) -> Option<PyGate> {
        self.inner.gate(node).cloned().map(PyGate::from)
    }

    /// Returns the number of gates in the circuit.
    fn gate_count(&self) -> usize {
        self.inner.gate_count()
    }

    /// Returns all node indices in the circuit.
    fn nodes(&self) -> Vec<usize> {
        self.inner.nodes()
    }

    /// Connect two gates with a qubit wire.
    ///
    /// Creates an edge from `from_node` to `to_node` representing the given qubit
    /// flowing between the gates.
    ///
    /// Returns the edge ID of the new wire.
    ///
    /// Raises `DagCircuitWouldCycleError` if adding this wire would create a cycle.
    fn connect(&mut self, from_node: usize, to_node: usize, qubit: usize) -> PyResult<usize> {
        self.inner
            .connect(from_node, to_node, QubitId::from(qubit))
            .map_err(|_| {
                PyErr::new::<DagCircuitWouldCycleError, _>("Adding this wire would create a cycle")
            })
    }

    /// Connect two gates on all shared qubits.
    ///
    /// Returns a list of (qubit, `edge_id`) pairs for each connection made.
    fn connect_all(&mut self, from_node: usize, to_node: usize) -> PyResult<Vec<(usize, usize)>> {
        self.inner
            .connect_all(from_node, to_node)
            .map(|connections| {
                connections
                    .into_iter()
                    .map(|(q, e)| (usize::from(q), e))
                    .collect()
            })
            .map_err(|_| {
                PyErr::new::<DagCircuitWouldCycleError, _>("Adding this wire would create a cycle")
            })
    }

    /// Remove a wire by its edge ID.
    ///
    /// Returns the qubit that was carried by this wire.
    fn remove_wire(&mut self, edge_id: usize) -> Option<usize> {
        self.inner.remove_wire(edge_id).map(usize::from)
    }

    /// Returns the number of wires in the circuit.
    fn wire_count(&self) -> usize {
        self.inner.wire_count()
    }

    /// Returns the qubit carried by a wire.
    fn wire_qubit(&self, edge_id: usize) -> Option<usize> {
        self.inner.wire_qubit(edge_id).map(usize::from)
    }

    /// Returns all wires as (from, to, qubit) tuples.
    fn wires(&self) -> Vec<(usize, usize, usize)> {
        self.inner
            .wires()
            .into_iter()
            .map(|(f, t, q)| (f, t, usize::from(q)))
            .collect()
    }

    /// Returns incoming wires to a gate as (`edge_id`, qubit) pairs.
    fn incoming_wires(&self, node: usize) -> Vec<(usize, usize)> {
        self.inner
            .incoming_wires(node)
            .into_iter()
            .map(|(e, q)| (e, usize::from(q)))
            .collect()
    }

    /// Returns outgoing wires from a gate as (`edge_id`, qubit) pairs.
    fn outgoing_wires(&self, node: usize) -> Vec<(usize, usize)> {
        self.inner
            .outgoing_wires(node)
            .into_iter()
            .map(|(e, q)| (e, usize::from(q)))
            .collect()
    }

    /// Returns the predecessor gate for a specific qubit input.
    fn predecessor_on_qubit(&self, node: usize, qubit: usize) -> Option<usize> {
        self.inner.predecessor_on_qubit(node, QubitId::from(qubit))
    }

    /// Returns the successor gate for a specific qubit output.
    fn successor_on_qubit(&self, node: usize, qubit: usize) -> Option<usize> {
        self.inner.successor_on_qubit(node, QubitId::from(qubit))
    }

    /// Returns all predecessor gates.
    fn predecessors(&self, node: usize) -> Vec<usize> {
        self.inner.predecessors(node)
    }

    /// Returns all successor gates.
    fn successors(&self, node: usize) -> Vec<usize> {
        self.inner.successors(node)
    }

    /// Returns the circuit depth (longest path from any root to any leaf).
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Returns the circuit width (number of unique qubits used).
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Returns all unique qubits used in the circuit.
    fn qubits(&self) -> Vec<usize> {
        self.inner.qubits().into_iter().map(usize::from).collect()
    }

    /// Returns the count of single-qubit gates.
    fn single_qubit_gate_count(&self) -> usize {
        self.inner.single_qubit_gate_count()
    }

    /// Returns the count of two-qubit gates.
    fn two_qubit_gate_count(&self) -> usize {
        self.inner.two_qubit_gate_count()
    }

    /// Returns the count of gates of a specific type.
    fn gate_type_count(&self, gate_type: PyGateType) -> usize {
        self.inner.gate_type_count(gate_type.inner)
    }

    /// Returns gates in topological order.
    fn topological_order(&self) -> Vec<usize> {
        self.inner.topological_order()
    }

    /// Returns circuit layers (gates that can execute in parallel).
    fn layers(&self) -> Vec<Vec<usize>> {
        self.inner.layers().collect()
    }

    /// Returns the root gates (gates with no incoming wires).
    fn roots(&self) -> Vec<usize> {
        self.inner.roots()
    }

    /// Returns the leaf gates (gates with no outgoing wires).
    fn leaves(&self) -> Vec<usize> {
        self.inner.leaves()
    }

    /// Returns all gates acting on a specific qubit.
    fn gates_on_qubit(&self, qubit: usize) -> Vec<usize> {
        self.inner.gates_on_qubit(QubitId::from(qubit))
    }

    /// Returns gates acting on a specific qubit in topological order.
    fn qubit_timeline(&self, qubit: usize) -> Vec<usize> {
        self.inner.qubit_timeline(QubitId::from(qubit))
    }

    /// Returns all wires carrying a specific qubit.
    fn wires_for_qubit(&self, qubit: usize) -> Vec<usize> {
        self.inner.wires_for_qubit(QubitId::from(qubit))
    }

    // ==================== Builder Methods ====================
    //
    // These methods provide a fluent API for building circuits, matching
    // the simulator APIs. Each method returns self for method chaining.

    /// Apply a Hadamard gate.
    fn h(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.h(&qubits);
        slf
    }

    /// Apply a Pauli-X gate.
    fn x(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.x(&qubits);
        slf
    }

    /// Apply a Pauli-Y gate.
    fn y(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.y(&qubits);
        slf
    }

    /// Apply a Pauli-Z gate.
    fn z(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.z(&qubits);
        slf
    }

    /// Apply a sqrt(Z) gate (S gate).
    fn sz(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.sz(&qubits);
        slf
    }

    /// Apply a sqrt(Z)-dagger gate (S-dagger gate).
    fn szdg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.szdg(&qubits);
        slf
    }

    /// Apply a T gate (fourth root of Z).
    fn t(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.t(&qubits);
        slf
    }

    /// Apply a T-dagger gate.
    fn tdg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.tdg(&qubits);
        slf
    }

    /// Apply a CNOT (CX) gate.
    ///
    /// Args:
    ///     pairs: List of (control, target) qubit pairs.
    fn cx(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> Py<Self> {
        slf.borrow_mut(py).inner.cx(&pairs);
        slf
    }

    /// Apply a CZ (controlled-Z) gate.
    ///
    /// Args:
    ///     pairs: List of (q1, q2) qubit pairs.
    fn cz(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> Py<Self> {
        slf.borrow_mut(py).inner.cz(&pairs);
        slf
    }

    /// Apply a sqrt(ZZ) gate.
    fn szz(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> Py<Self> {
        slf.borrow_mut(py).inner.szz(&pairs);
        slf
    }

    /// Apply a sqrt(ZZ)-dagger gate.
    fn szzdg(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> Py<Self> {
        slf.borrow_mut(py).inner.szzdg(&pairs);
        slf
    }

    /// Apply an RX rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle (angle64 or float radians).
    ///     qubits: List of qubits to rotate.
    fn rx(slf: Py<Self>, py: Python<'_>, theta: AngleParam, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.rx(theta.0, &qubits);
        slf
    }

    /// Apply an RY rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle (angle64 or float radians).
    ///     qubits: List of qubits to rotate.
    fn ry(slf: Py<Self>, py: Python<'_>, theta: AngleParam, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.ry(theta.0, &qubits);
        slf
    }

    /// Apply an RZ rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle (angle64 or float radians).
    ///     qubits: List of qubits to rotate.
    fn rz(slf: Py<Self>, py: Python<'_>, theta: AngleParam, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.rz(theta.0, &qubits);
        slf
    }

    /// Apply an RZZ rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle (angle64 or float radians).
    ///     pairs: List of (q1, q2) qubit pairs.
    fn rzz(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        pairs: Vec<(usize, usize)>,
    ) -> Py<Self> {
        slf.borrow_mut(py).inner.rzz(theta.0, &pairs);
        slf
    }

    /// Apply an idle gate with a specified duration.
    ///
    /// Idle gates represent waiting time on qubits, useful for noise modeling.
    /// Duration is in abstract time units - interpretation depends on your noise model.
    ///
    /// Args:
    ///     duration: Duration as `TimeUnits`, `Nanoseconds` (deprecated), or integer.
    ///     qubits: List of qubits to idle.
    fn idle(
        slf: Py<Self>,
        py: Python<'_>,
        duration: &Bound<'_, PyAny>,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        // Try to extract as PyTimeUnits, PyNanoseconds (deprecated), or u64
        let units = if let Ok(py_tu) = duration.extract::<PyTimeUnits>() {
            py_tu.inner
        } else if let Ok(py_ns) = duration.extract::<PyNanoseconds>() {
            // Deprecated: treat nanoseconds value as time units directly
            TimeUnits::new(py_ns.ns)
        } else if let Ok(raw) = duration.extract::<u64>() {
            TimeUnits::new(raw)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "duration must be TimeUnits, Nanoseconds, or an integer",
            ));
        };
        slf.borrow_mut(py).inner.idle(units, &qubits);
        Ok(slf)
    }

    /// Measure qubits in the Z basis.
    ///
    /// Note: Unlike gates, measurements break the chain in simulators.
    /// This method still returns self for convenience in Python.
    fn mz(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.mz(&qubits);
        slf
    }

    /// Measure and free qubits (destructive measurement).
    fn mz_free(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.mz_free(&qubits);
        slf
    }

    /// Prepare qubits in the |0> state (Z-basis preparation).
    fn pz(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.pz(&qubits);
        slf
    }

    /// Allocate qubits in the |0> state.
    fn qalloc(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.qalloc(&qubits);
        slf
    }

    /// Free/deallocate qubits.
    fn qfree(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> Py<Self> {
        slf.borrow_mut(py).inner.qfree(&qubits);
        slf
    }

    /// Add metadata to the last added gate.
    ///
    /// Args:
    ///     key: The attribute name.
    ///     value: The attribute value.
    fn meta(
        slf: Py<Self>,
        py: Python<'_>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let attr = py_to_attribute(py, value)?;
        slf.borrow_mut(py).inner.meta(key, attr);
        Ok(slf)
    }

    /// Add multiple metadata attributes to the last added gate.
    ///
    /// Args:
    ///     attrs: A dictionary of attribute names to values.
    fn metas(slf: Py<Self>, py: Python<'_>, attrs: &Bound<'_, PyDict>) -> PyResult<Py<Self>> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        slf.borrow_mut(py).inner.metas(attrs_map);
        Ok(slf)
    }

    // ==================== Attributes ====================

    /// Get all circuit-level attributes as a dictionary.
    ///
    /// Returns:
    ///     A dictionary of attribute name to value.
    fn attrs(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        attrs_to_py_dict(py, self.inner.attrs())
    }

    /// Get a circuit-level attribute by key.
    ///
    /// Args:
    ///     key: The attribute name.
    ///
    /// Returns:
    ///     The attribute value, or None if not found.
    fn get_attr(&self, py: Python<'_>, key: &str) -> Option<Py<PyAny>> {
        self.inner
            .get_attr(key)
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Set a circuit-level attribute.
    ///
    /// Args:
    ///     key: The attribute name.
    ///     value: The attribute value (int, float, str, bool, list of ints, list of strings, or dict).
    fn set_attr(&mut self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        self.inner.set_attr(key, attr);
        Ok(())
    }

    /// Get all attributes on a gate.
    ///
    /// Args:
    ///     node: The gate node index.
    ///
    /// Returns:
    ///     A dictionary of attribute name to value, or None if the node doesn't exist.
    fn gate_attrs(&self, py: Python<'_>, node: usize) -> PyResult<Option<Py<PyDict>>> {
        match self.inner.gate_attrs(node) {
            Some(attrs) => Ok(Some(attrs_to_py_dict(py, attrs)?)),
            None => Ok(None),
        }
    }

    /// Get an attribute from a gate.
    ///
    /// Args:
    ///     node: The gate node index.
    ///     key: The attribute name.
    ///
    /// Returns:
    ///     The attribute value, or None if not found.
    fn get_gate_attr(&self, py: Python<'_>, node: usize, key: &str) -> Option<Py<PyAny>> {
        self.inner
            .get_gate_attr(node, key)
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Set an attribute on a gate.
    ///
    /// Args:
    ///     node: The gate node index.
    ///     key: The attribute name.
    ///     value: The attribute value.
    ///
    /// Returns:
    ///     True if the gate exists, False otherwise.
    fn set_gate_attr(
        &mut self,
        py: Python<'_>,
        node: usize,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let attr = py_to_attribute(py, value)?;
        Ok(self.inner.set_gate_attr(node, key, attr))
    }

    /// Set multiple attributes on a gate at once.
    ///
    /// Args:
    ///     node: The gate node index.
    ///     attrs: A dictionary of attribute names to values.
    ///
    /// Returns:
    ///     True if the gate exists, False otherwise.
    fn set_gate_attrs(
        &mut self,
        py: Python<'_>,
        node: usize,
        attrs: &Bound<'_, PyDict>,
    ) -> PyResult<bool> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        Ok(self.inner.set_gate_attrs(node, attrs_map))
    }

    /// Get all attributes on a wire.
    ///
    /// Args:
    ///     `edge_id`: The wire edge ID.
    ///
    /// Returns:
    ///     A dictionary of attribute name to value, or None if the wire doesn't exist.
    fn wire_attrs(&self, py: Python<'_>, edge_id: usize) -> PyResult<Option<Py<PyDict>>> {
        match self.inner.wire_attrs(edge_id) {
            Some(attrs) => Ok(Some(attrs_to_py_dict(py, attrs)?)),
            None => Ok(None),
        }
    }

    /// Get an attribute from a wire.
    ///
    /// Args:
    ///     `edge_id`: The wire edge ID.
    ///     key: The attribute name.
    ///
    /// Returns:
    ///     The attribute value, or None if not found.
    fn get_wire_attr(&self, py: Python<'_>, edge_id: usize, key: &str) -> Option<Py<PyAny>> {
        self.inner
            .get_wire_attr(edge_id, key)
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Set an attribute on a wire.
    ///
    /// Args:
    ///     `edge_id`: The wire edge ID.
    ///     key: The attribute name.
    ///     value: The attribute value.
    ///
    /// Returns:
    ///     True if the wire exists, False otherwise.
    fn set_wire_attr(
        &mut self,
        py: Python<'_>,
        edge_id: usize,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let attr = py_to_attribute(py, value)?;
        Ok(self.inner.set_wire_attr(edge_id, key, attr))
    }

    /// Convert this `DagCircuit` to a `TickCircuit`.
    ///
    /// Each layer of parallel gates in the DAG becomes a tick.
    /// Circuit-level and gate-level attributes are preserved.
    ///
    /// Returns:
    ///     A new `TickCircuit` with the same gates organized by layers.
    fn to_tick_circuit(&self) -> PyTickCircuit {
        PyTickCircuit {
            inner: TickCircuit::from(&self.inner),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DagCircuit(gates={}, wires={}, depth={}, width={})",
            self.inner.gate_count(),
            self.inner.wire_count(),
            self.inner.depth(),
            self.inner.width()
        )
    }
}

// HUGR conversion exception
pyo3::create_exception!(
    pecos_rslib,
    HugrConversionError,
    pyo3::exceptions::PyException
);

// Qubit conflict exception
pyo3::create_exception!(
    pecos_rslib,
    GateSignatureMismatchError,
    pyo3::exceptions::PyValueError
);

pyo3::create_exception!(
    pecos_rslib,
    QubitConflictError,
    pyo3::exceptions::PyValueError
);

/// Convert HUGR bytes to a `DagCircuit`.
///
/// This function takes serialized HUGR data (JSON or binary envelope format)
/// and converts it to a `DagCircuit` for circuit analysis and manipulation.
///
/// Args:
///     `hugr_bytes`: Serialized HUGR data as bytes. Can be:
///         - JSON format (starts with '{')
///         - Binary envelope format (HUGR package)
///
/// Returns:
///     A `DagCircuit` representing the quantum circuit.
///
/// Raises:
///     `HugrConversionError`: If the HUGR cannot be parsed or contains unsupported structures.
///
/// Example:
///     >>> from `pecos_rslib.quantum` import `hugr_to_dag_circuit`
///     >>> # Get HUGR bytes from a compiled Guppy program
///     >>> `hugr_bytes` = `guppy_func.compile().package.to_bytes()`
///     >>> circuit = `hugr_to_dag_circuit(hugr_bytes)`
///     >>> `print(circuit.gate_count())`
#[pyfunction]
#[pyo3(name = "hugr_to_dag_circuit")]
fn py_hugr_to_dag_circuit(hugr_bytes: &Bound<'_, PyBytes>) -> PyResult<PyDagCircuit> {
    use pecos::quantum::{hugr_to_dag_circuit, read_hugr_envelope};

    let bytes = hugr_bytes.as_bytes();

    // Parse the HUGR bytes
    let hugr = read_hugr_envelope(bytes)
        .map_err(|e| PyErr::new::<HugrConversionError, _>(format!("Failed to parse HUGR: {e}")))?;

    // Convert to DagCircuit
    let dag = hugr_to_dag_circuit(&hugr).map_err(|e| {
        PyErr::new::<HugrConversionError, _>(format!("Failed to convert HUGR to DagCircuit: {e}"))
    })?;

    Ok(PyDagCircuit { inner: dag })
}

/// Map a HUGR operation name to a `GateType`.
///
/// Args:
///     `op_name`: The HUGR operation name (e.g., "H", "CX", "`QAlloc`").
///
/// Returns:
///     The corresponding `GateType`, or None if the operation is not recognized.
#[pyfunction]
#[pyo3(name = "hugr_op_to_gate_type")]
fn py_hugr_op_to_gate_type(op_name: &str) -> Option<PyGateType> {
    use pecos::quantum::hugr_op_to_gate_type;
    hugr_op_to_gate_type(op_name).map(|gt| PyGateType { inner: gt })
}

/// Map a `GateType` to a HUGR operation name.
///
/// Args:
///     `gate_type`: The `GateType` to convert.
///
/// Returns:
///     The corresponding HUGR operation name, or None if the gate type is not supported.
#[pyfunction]
#[pyo3(name = "gate_type_to_hugr_op")]
fn py_gate_type_to_hugr_op(gate_type: PyGateType) -> Option<String> {
    use pecos::quantum::gate_type_to_hugr_op;
    gate_type_to_hugr_op(gate_type.inner).map(String::from)
}

/// Check if an operation name is a recognized quantum operation.
///
/// Args:
///     `op_name`: The operation name to check.
///
/// Returns:
///     True if the operation is a recognized quantum operation.
#[pyfunction]
#[pyo3(name = "is_quantum_operation")]
fn py_is_quantum_operation(op_name: &str) -> bool {
    use pecos::quantum::is_quantum_operation;
    is_quantum_operation(op_name)
}

// ==================== Time Unit Types ====================

/// Python wrapper for nanosecond durations.
///
/// Deprecated: Prefer using `TimeUnits` with `TimeScale` for new code.
/// This type is kept for backwards compatibility.
#[pyclass(name = "Nanoseconds", module = "pecos_rslib", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyNanoseconds {
    /// Duration in nanoseconds.
    ns: u64,
}

#[pymethods]
impl PyNanoseconds {
    /// Create from nanoseconds.
    #[new]
    fn new(ns: u64) -> Self {
        Self { ns }
    }

    /// Create from nanoseconds.
    #[staticmethod]
    fn from_ns(ns: u64) -> Self {
        Self { ns }
    }

    /// Create from microseconds.
    #[staticmethod]
    fn from_us(us: u64) -> Self {
        Self { ns: us * 1_000 }
    }

    /// Create from milliseconds.
    #[staticmethod]
    fn from_ms(ms: u64) -> Self {
        Self { ns: ms * 1_000_000 }
    }

    /// Create from seconds.
    #[staticmethod]
    fn from_secs(secs: u64) -> Self {
        Self {
            ns: secs * 1_000_000_000,
        }
    }

    /// Get the duration in nanoseconds.
    fn as_ns(&self) -> u64 {
        self.ns
    }

    /// Get the duration in microseconds (truncated).
    fn as_us(&self) -> u64 {
        self.ns / 1_000
    }

    /// Get the duration in milliseconds (truncated).
    fn as_ms(&self) -> u64 {
        self.ns / 1_000_000
    }

    /// Get the duration in seconds (truncated).
    fn as_secs(&self) -> u64 {
        self.ns / 1_000_000_000
    }

    fn __repr__(&self) -> String {
        format!("Nanoseconds({})", self.ns)
    }

    fn __str__(&self) -> String {
        format!("{}ns", self.ns)
    }

    fn __int__(&self) -> u64 {
        self.ns
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.ns == other.ns
    }

    fn __lt__(&self, other: &Self) -> bool {
        self.ns < other.ns
    }

    fn __le__(&self, other: &Self) -> bool {
        self.ns <= other.ns
    }

    fn __gt__(&self, other: &Self) -> bool {
        self.ns > other.ns
    }

    fn __ge__(&self, other: &Self) -> bool {
        self.ns >= other.ns
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            ns: self.ns + other.ns,
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            ns: self.ns - other.ns,
        }
    }

    fn __mul__(&self, rhs: u64) -> Self {
        Self { ns: self.ns * rhs }
    }

    fn __hash__(&self) -> u64 {
        self.ns
    }
}

/// Python wrapper for `TimeUnits`.
///
/// Represents an abstract time duration in arbitrary units.
#[pyclass(name = "TimeUnits", module = "pecos_rslib", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyTimeUnits {
    inner: TimeUnits,
}

#[pymethods]
impl PyTimeUnits {
    /// Create a new time duration.
    #[new]
    fn new(units: u64) -> Self {
        Self {
            inner: TimeUnits::new(units),
        }
    }

    /// Get the duration as a u64.
    fn as_u64(&self) -> u64 {
        self.inner.as_u64()
    }

    fn __repr__(&self) -> String {
        format!("TimeUnits({})", self.inner.as_u64())
    }

    fn __str__(&self) -> String {
        format!("{} units", self.inner.as_u64())
    }

    fn __int__(&self) -> u64 {
        self.inner.as_u64()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __lt__(&self, other: &Self) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &Self) -> bool {
        self.inner <= other.inner
    }

    fn __gt__(&self, other: &Self) -> bool {
        self.inner > other.inner
    }

    fn __ge__(&self, other: &Self) -> bool {
        self.inner >= other.inner
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner - other.inner,
        }
    }

    fn __mul__(&self, rhs: u64) -> Self {
        Self {
            inner: self.inner * rhs,
        }
    }

    fn __hash__(&self) -> u64 {
        self.inner.as_u64()
    }
}

impl From<TimeUnits> for PyTimeUnits {
    fn from(inner: TimeUnits) -> Self {
        Self { inner }
    }
}

impl From<PyTimeUnits> for TimeUnits {
    fn from(py_tu: PyTimeUnits) -> Self {
        py_tu.inner
    }
}

// ============================================================================
// TickCircuit bindings
// ============================================================================

/// Python wrapper for a single tick (parallel time slice).
#[pyclass(name = "Tick", module = "pecos_rslib.quantum")]
pub struct PyTick {
    inner: Tick,
}

#[pymethods]
impl PyTick {
    /// Get the number of gates in this tick.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the tick is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the gates in this tick as a list.
    fn gates(&self) -> Vec<PyGate> {
        self.inner
            .gates()
            .iter()
            .map(|g: &Gate| PyGate { inner: g.clone() })
            .collect()
    }

    /// Get metadata from a gate by index.
    fn get_gate_attr(&self, gate_idx: usize, key: &str, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner
            .get_gate_attr(gate_idx, key)
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Set metadata on a gate.
    fn set_gate_attr(
        &mut self,
        py: Python<'_>,
        gate_idx: usize,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        self.inner.set_gate_attr(gate_idx, key, attr);
        Ok(())
    }

    /// Set multiple metadata attributes on a gate at once.
    fn set_gate_attrs(
        &mut self,
        py: Python<'_>,
        gate_idx: usize,
        attrs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        self.inner.set_gate_attrs(gate_idx, attrs_map);
        Ok(())
    }

    /// Get tick-level metadata.
    fn get_attr(&self, key: &str, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner
            .get_attr(key)
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Set tick-level metadata.
    fn meta(&mut self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        self.inner.set_attr(key, attr);
        Ok(())
    }

    /// Set multiple tick-level metadata attributes at once.
    fn metas(&mut self, py: Python<'_>, attrs: &Bound<'_, PyDict>) -> PyResult<()> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        self.inner.set_attrs(attrs_map);
        Ok(())
    }

    /// Get the set of qubits used in this tick.
    ///
    /// Returns a sorted list of qubit IDs that are acted upon by gates in this tick.
    fn active_qubits(&self) -> Vec<usize> {
        self.inner
            .active_qubits()
            .into_iter()
            .map(|q| q.0)
            .collect()
    }

    /// Check if a specific qubit is used in this tick.
    fn uses_qubit(&self, qubit: usize) -> bool {
        self.inner.uses_qubit(QubitId::from(qubit))
    }

    /// Check if any of the given qubits are already in use in this tick.
    ///
    /// Returns a list of conflicting qubit IDs.
    fn find_conflicts(&self, qubits: Vec<usize>) -> Vec<PyQubitId> {
        let qubit_ids: Vec<QubitId> = qubits.into_iter().map(QubitId::from).collect();
        self.inner
            .find_conflicts(&qubit_ids)
            .into_iter()
            .map(|q| PyQubitId { inner: q })
            .collect()
    }

    /// Add a gate to this tick.
    ///
    /// Returns the index of the added gate within this tick.
    fn add_gate(&mut self, gate: &PyGate) -> usize {
        self.inner.add_gate(gate.inner.clone())
    }

    /// Try to add a gate to this tick, returning an error if any qubit is already in use.
    ///
    /// Returns the gate index if successful.
    ///
    /// Raises:
    ///     `ValueError`: If any qubit in the gate is already used by another gate in this tick.
    fn try_add_gate(&mut self, gate: &PyGate) -> PyResult<usize> {
        self.inner
            .try_add_gate(gate.inner.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Remove all gates that use any of the specified qubits.
    ///
    /// Returns the number of gates removed.
    fn discard(&mut self, qubits: Vec<usize>) -> usize {
        let qubit_ids: Vec<QubitId> = qubits.into_iter().map(QubitId::from).collect();
        self.inner.discard(&qubit_ids)
    }

    /// Remove a specific gate by index.
    ///
    /// Returns the removed gate, or None if the index is out of bounds.
    fn remove_gate(&mut self, idx: usize) -> Option<PyGate> {
        self.inner.remove_gate(idx).map(|g| PyGate { inner: g })
    }

    fn __repr__(&self) -> String {
        format!("Tick(gates={})", self.inner.len())
    }
}

/// Python wrapper for `TickCircuit`.
///
/// A tick-based quantum circuit where each tick contains gates that
/// execute in parallel on non-overlapping qubits.
///
/// Use `tick()` to create a new tick and get a handle for adding gates.
#[pyclass(name = "TickCircuit", module = "pecos_rslib.quantum")]
pub struct PyTickCircuit {
    inner: TickCircuit,
}

#[pymethods]
impl PyTickCircuit {
    /// Create a new empty tick circuit.
    #[new]
    fn new() -> Self {
        Self {
            inner: TickCircuit::new(),
        }
    }

    /// Get the number of ticks (excluding trailing empty ticks).
    fn num_ticks(&self) -> usize {
        self.inner.num_ticks()
    }

    /// Get the total number of gates across all ticks.
    fn gate_count(&self) -> usize {
        self.inner.gate_count()
    }

    /// Get the next tick index that will be allocated.
    fn next_tick_index(&self) -> usize {
        self.inner.next_tick_index()
    }

    /// Get a tick by index.
    fn get_tick(&self, idx: usize) -> Option<PyTick> {
        self.inner
            .get_tick(idx)
            .map(|t: &Tick| PyTick { inner: t.clone() })
    }

    /// Create a new tick and return a handle for adding gates.
    ///
    /// The tick acts as a mini-circuit where gates can be chained.
    fn tick(slf: Py<Self>, py: Python<'_>) -> PyTickHandle {
        PyTickHandle::new(slf, py)
    }

    /// Set circuit-level metadata.
    fn set_meta(
        slf: Py<Self>,
        py: Python<'_>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        slf.borrow_mut(py).inner.set_meta(key, attr);
        Ok(())
    }

    /// Set multiple circuit-level metadata attributes at once.
    fn metas(slf: Py<Self>, py: Python<'_>, attrs: &Bound<'_, PyDict>) -> PyResult<()> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        slf.borrow_mut(py).inner.set_metas(attrs_map);
        Ok(())
    }

    /// Get circuit-level metadata.
    fn get_meta(&self, key: &str, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner
            .get_meta(key)
            .map(|attr| attribute_to_py(py, attr))
    }

    // =========================================================================
    // Circuit manipulation
    // =========================================================================

    /// Clear the circuit and start fresh.
    ///
    /// This completely replaces the circuit with a new empty instance.
    /// For performance-critical code, consider using `reset()` instead.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Reset the circuit state while preserving allocated memory.
    ///
    /// This is faster than `clear()` when reusing the same circuit multiple times.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Reserve empty ticks in advance.
    ///
    /// Args:
    ///     n: The number of empty ticks to reserve.
    fn reserve_ticks(&mut self, n: usize) {
        self.inner.reserve_ticks(n);
    }

    /// Insert an empty tick at a specific position.
    ///
    /// All ticks at or after `idx` are shifted to the right.
    /// Returns a `TickHandle` to the newly inserted tick.
    ///
    /// Args:
    ///     idx: The position at which to insert the new tick.
    ///
    /// Raises:
    ///     `IndexError`: If `idx > num_ticks()`.
    fn insert_tick(slf: Py<Self>, py: Python<'_>, idx: usize) -> PyResult<PyTickHandle> {
        {
            let mut circuit = slf.borrow_mut(py);
            let num_ticks = circuit.inner.ticks().len();
            if idx > num_ticks {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "insert_tick index {idx} out of bounds for circuit with {num_ticks} ticks"
                )));
            }
            // Insert the tick
            let _ = circuit.inner.insert_tick(idx);
        }
        // Return a handle to the inserted tick
        Ok(PyTickHandle {
            circuit: slf,
            tick_idx: idx,
            last_gate_idx: None,
        })
    }

    /// Get a handle to an existing tick for adding more gates.
    ///
    /// This allows adding gates to a tick that was previously created.
    ///
    /// Args:
    ///     idx: The index of the tick to get a handle for.
    ///
    /// Raises:
    ///     `IndexError`: If `idx >= num_ticks()`.
    fn tick_at(slf: Py<Self>, py: Python<'_>, idx: usize) -> PyResult<PyTickHandle> {
        {
            let circuit = slf.borrow(py);
            let num_ticks = circuit.inner.ticks().len();
            if idx >= num_ticks {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "tick_at index {idx} out of bounds for circuit with {num_ticks} ticks"
                )));
            }
        }
        Ok(PyTickHandle {
            circuit: slf,
            tick_idx: idx,
            last_gate_idx: None,
        })
    }

    // =========================================================================
    // Iteration helpers
    // =========================================================================

    /// Get all qubits used in the circuit.
    ///
    /// Returns:
    ///     A list of qubit IDs used in the circuit.
    fn all_qubits(&self) -> Vec<usize> {
        self.inner
            .all_qubits()
            .into_iter()
            .map(usize::from)
            .collect()
    }

    /// Count gates by type across the entire circuit.
    ///
    /// Returns:
    ///     A dictionary mapping gate type names to counts.
    fn gate_counts_by_type(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let counts = self.inner.gate_counts_by_type();
        let dict = PyDict::new(py);
        for (gate_type, count) in counts {
            dict.set_item(format!("{gate_type:?}"), count)?;
        }
        Ok(dict.into())
    }

    /// Get all gates in the circuit as a list.
    ///
    /// Returns:
    ///     A list of (`tick_index`, gate) tuples.
    fn gates(&self) -> Vec<(usize, PyGate)> {
        self.inner
            .iter_gates_with_tick()
            .map(|(tick_idx, gate)| {
                (
                    tick_idx,
                    PyGate {
                        inner: gate.clone(),
                    },
                )
            })
            .collect()
    }

    /// Remove all gates that use any of the specified qubits from a tick.
    ///
    /// Args:
    ///     qubits: List of qubit IDs. Gates using any of these qubits will be removed.
    ///     `tick_idx`: The index of the tick to modify.
    ///
    /// Returns:
    ///     The number of gates removed, or None if the tick index is out of bounds.
    ///
    /// Example:
    ///     >>> circuit = `TickCircuit()`
    ///     >>> circuit.tick().h(&[0]).x(&[1]).cx(2, 3)
    ///     >>> circuit.discard([0, 2], 0)  # Remove H on q0 and CX on q2,q3
    ///     2
    fn discard(&mut self, qubits: Vec<usize>, tick_idx: usize) -> Option<usize> {
        self.inner.discard(&qubits, tick_idx)
    }

    // =========================================================================
    // Tick-level and gate-level metadata setters (by index)
    // =========================================================================

    /// Set tick-level metadata on a specific tick by index.
    ///
    /// Unlike `get_tick().meta()` which operates on a copy, this method
    /// modifies the tick in place.
    ///
    /// Args:
    ///     `tick_idx`: The index of the tick.
    ///     key: The metadata key.
    ///     value: The metadata value.
    ///
    /// Raises:
    ///     `IndexError`: If `tick_idx` is out of bounds.
    fn set_tick_meta(
        &mut self,
        py: Python<'_>,
        tick_idx: usize,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        if let Some(tick) = self.inner.get_tick_mut(tick_idx) {
            tick.set_attr(key, attr);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tick index {tick_idx} out of bounds"
            )))
        }
    }

    /// Get tick-level metadata from a specific tick by index.
    ///
    /// Args:
    ///     `tick_idx`: The index of the tick.
    ///     key: The metadata key.
    ///
    /// Returns:
    ///     The metadata value, or None if not found or `tick_idx` is out of bounds.
    fn get_tick_meta(&self, py: Python<'_>, tick_idx: usize, key: &str) -> Option<Py<PyAny>> {
        self.inner
            .get_tick(tick_idx)
            .and_then(|tick| tick.get_attr(key))
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Set gate-level metadata on a specific gate within a tick.
    ///
    /// Unlike `get_tick().set_gate_attr()` which operates on a copy, this method
    /// modifies the tick in place.
    ///
    /// Args:
    ///     `tick_idx`: The index of the tick.
    ///     `gate_idx`: The index of the gate within the tick.
    ///     key: The metadata key.
    ///     value: The metadata value.
    ///
    /// Raises:
    ///     `IndexError`: If `tick_idx` is out of bounds.
    fn set_gate_meta(
        &mut self,
        py: Python<'_>,
        tick_idx: usize,
        gate_idx: usize,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        if let Some(tick) = self.inner.get_tick_mut(tick_idx) {
            tick.set_gate_attr(gate_idx, key, attr);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tick index {tick_idx} out of bounds"
            )))
        }
    }

    /// Get gate-level metadata from a specific gate within a tick.
    ///
    /// Args:
    ///     `tick_idx`: The index of the tick.
    ///     `gate_idx`: The index of the gate within the tick.
    ///     key: The metadata key.
    ///
    /// Returns:
    ///     The metadata value, or None if not found or indices are out of bounds.
    fn get_gate_meta(
        &self,
        py: Python<'_>,
        tick_idx: usize,
        gate_idx: usize,
        key: &str,
    ) -> Option<Py<PyAny>> {
        self.inner
            .get_tick(tick_idx)
            .and_then(|tick| tick.get_gate_attr(gate_idx, key))
            .map(|attr| attribute_to_py(py, attr))
    }

    /// Convert this `TickCircuit` to a `DagCircuit`.
    ///
    /// Gates are added in tick order, with qubit wires connecting
    /// consecutive gates on the same qubit.
    /// Circuit-level and gate-level attributes are preserved.
    ///
    /// Returns:
    ///     A new `DagCircuit` with the same gates and qubit wire connections.
    fn to_dag_circuit(&self) -> PyDagCircuit {
        PyDagCircuit {
            inner: DagCircuit::from(&self.inner),
        }
    }

    // =========================================================================
    // Gate signature validation
    // =========================================================================

    /// Import gate signatures for validation.
    ///
    /// Args:
    ///     sigs: A dictionary mapping gate names to (`quantum_arity`, `angle_arity`) tuples.
    fn import_gate_signatures(&mut self, sigs: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut sig_map = HashMap::new();
        for (key, value) in sigs.iter() {
            let name: String = key.extract()?;
            let (quantum_arity, angle_arity): (usize, usize) = value.extract()?;
            sig_map.insert(
                name,
                GateSignature {
                    quantum_arity,
                    angle_arity,
                },
            );
        }
        self.inner.import_signatures(&sig_map);
        Ok(())
    }

    /// Get gate signatures as a dictionary.
    ///
    /// Returns:
    ///     A dictionary mapping gate names to (`quantum_arity`, `angle_arity`) tuples.
    fn gate_signatures(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (name, sig) in self.inner.gate_signatures() {
            dict.set_item(name, (sig.quantum_arity, sig.angle_arity))?;
        }
        Ok(dict.into())
    }

    /// Import signatures from a `GateRegistry`.
    ///
    /// Extracts signatures from all registered gates and imports them
    /// for validation when adding custom gates.
    fn import_registry(&mut self, registry: &PyGateRegistry) {
        let sigs = registry.inner.signatures();
        self.inner.import_signatures(&sigs);
    }

    fn __repr__(&self) -> String {
        format!(
            "TickCircuit(ticks={}, gates={})",
            self.inner.num_ticks(),
            self.inner.gate_count()
        )
    }
}

/// Handle to a specific tick for adding gates.
///
/// Gates added through the handle are placed in the associated tick.
/// The handle chains for fluent API usage.
#[pyclass(name = "TickHandle", module = "pecos_rslib.quantum")]
pub struct PyTickHandle {
    circuit: Py<PyTickCircuit>,
    tick_idx: usize,
    last_gate_idx: Option<usize>,
}

impl PyTickHandle {
    fn new(circuit: Py<PyTickCircuit>, py: Python<'_>) -> Self {
        let tick_idx = {
            let mut circuit_ref = circuit.borrow_mut(py);
            // Call tick() on the inner circuit to allocate a new tick
            let handle = circuit_ref.inner.tick();
            handle.index()
        };
        Self {
            circuit,
            tick_idx,
            last_gate_idx: None,
        }
    }

    fn add_gate_internal(&mut self, py: Python<'_>, gate: Gate) -> PyResult<()> {
        let mut circuit = self.circuit.borrow_mut(py);
        if let Some(tick) = circuit.inner.get_tick_mut(self.tick_idx) {
            match tick.try_add_gate(gate) {
                Ok(idx) => {
                    self.last_gate_idx = Some(idx);
                    Ok(())
                }
                Err(err) => {
                    let msg = format!(
                        "Qubit(s) {:?} already in use in tick {}",
                        err.conflicting_qubits
                            .iter()
                            .map(std::string::ToString::to_string)
                            .collect::<Vec<_>>(),
                        self.tick_idx
                    );
                    Err(PyErr::new::<QubitConflictError, _>(msg))
                }
            }
        } else {
            Ok(())
        }
    }

    fn add_gate_get_idx(&mut self, py: Python<'_>, gate: Gate) -> PyResult<usize> {
        let mut circuit = self.circuit.borrow_mut(py);
        if let Some(tick) = circuit.inner.get_tick_mut(self.tick_idx) {
            match tick.try_add_gate(gate) {
                Ok(idx) => {
                    self.last_gate_idx = Some(idx);
                    Ok(idx)
                }
                Err(err) => {
                    let msg = format!(
                        "Qubit(s) {:?} already in use in tick {}",
                        err.conflicting_qubits
                            .iter()
                            .map(std::string::ToString::to_string)
                            .collect::<Vec<_>>(),
                        self.tick_idx
                    );
                    Err(PyErr::new::<QubitConflictError, _>(msg))
                }
            }
        } else {
            Ok(0)
        }
    }
}

/// Handle returned by preparation operations on a tick.
///
/// This handle breaks the method chain (unlike regular gates),
/// but still allows attaching metadata via `.meta()`.
#[pyclass(name = "TickPrepHandle", module = "pecos_rslib.quantum")]
pub struct PyTickPrepHandle {
    circuit: Py<PyTickCircuit>,
    tick_idx: usize,
    gate_idx: usize,
}

#[pymethods]
impl PyTickPrepHandle {
    /// Add metadata to this preparation.
    fn meta(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        let mut circuit = self.circuit.borrow_mut(py);
        if let Some(tick) = circuit.inner.get_tick_mut(self.tick_idx) {
            tick.set_gate_attr(self.gate_idx, key, attr);
        }
        Ok(())
    }

    /// Add multiple metadata attributes to this preparation.
    fn metas(&self, py: Python<'_>, attrs: &Bound<'_, PyDict>) -> PyResult<()> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        let mut circuit = self.circuit.borrow_mut(py);
        if let Some(tick) = circuit.inner.get_tick_mut(self.tick_idx) {
            tick.set_gate_attrs(self.gate_idx, attrs_map);
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "TickPrepHandle(tick={}, gate={})",
            self.tick_idx, self.gate_idx
        )
    }
}

/// Handle returned by measurement operations on a tick.
///
/// This handle breaks the method chain (unlike regular gates),
/// but still allows attaching metadata via `.meta()`.
#[pyclass(name = "TickMeasureHandle", module = "pecos_rslib.quantum")]
pub struct PyTickMeasureHandle {
    circuit: Py<PyTickCircuit>,
    tick_idx: usize,
    gate_idx: usize,
}

#[pymethods]
impl PyTickMeasureHandle {
    /// Add metadata to this measurement.
    fn meta(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let attr = py_to_attribute(py, value)?;
        let mut circuit = self.circuit.borrow_mut(py);
        if let Some(tick) = circuit.inner.get_tick_mut(self.tick_idx) {
            tick.set_gate_attr(self.gate_idx, key, attr);
        }
        Ok(())
    }

    /// Add multiple metadata attributes to this measurement.
    fn metas(&self, py: Python<'_>, attrs: &Bound<'_, PyDict>) -> PyResult<()> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        let mut circuit = self.circuit.borrow_mut(py);
        if let Some(tick) = circuit.inner.get_tick_mut(self.tick_idx) {
            tick.set_gate_attrs(self.gate_idx, attrs_map);
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "TickMeasureHandle(tick={}, gate={})",
            self.tick_idx, self.gate_idx
        )
    }
}

#[pymethods]
impl PyTickHandle {
    /// Get the tick index this handle refers to.
    fn index(&self) -> usize {
        self.tick_idx
    }

    /// Set metadata on the last added gate, or tick-level metadata if no gate added yet.
    fn meta(
        slf: Py<Self>,
        py: Python<'_>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let attr = py_to_attribute(py, value)?;
        {
            let handle = slf.borrow_mut(py);
            let tick_idx = handle.tick_idx;
            let last_gate_idx = handle.last_gate_idx;

            let mut circuit = handle.circuit.borrow_mut(py);
            if let Some(tick) = circuit.inner.get_tick_mut(tick_idx) {
                if let Some(gate_idx) = last_gate_idx {
                    tick.set_gate_attr(gate_idx, key, attr);
                } else {
                    // No gate yet - set tick-level metadata
                    tick.set_attr(key, attr);
                }
            }
        }
        Ok(slf)
    }

    /// Set multiple metadata attributes on the last added gate, or tick-level if no gate added yet.
    fn metas(slf: Py<Self>, py: Python<'_>, attrs: &Bound<'_, PyDict>) -> PyResult<Py<Self>> {
        let attrs_map = py_dict_to_attrs(py, attrs)?;
        {
            let handle = slf.borrow_mut(py);
            let tick_idx = handle.tick_idx;
            let last_gate_idx = handle.last_gate_idx;

            let mut circuit = handle.circuit.borrow_mut(py);
            if let Some(tick) = circuit.inner.get_tick_mut(tick_idx) {
                if let Some(gate_idx) = last_gate_idx {
                    tick.set_gate_attrs(gate_idx, attrs_map);
                } else {
                    // No gate yet - set tick-level metadata
                    tick.set_attrs(attrs_map);
                }
            }
        }
        Ok(slf)
    }

    // =========================================================================
    // Single-qubit gates
    // =========================================================================

    /// Apply a Hadamard gate.
    fn h(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::h(&qubits))?;
        Ok(slf)
    }

    /// Apply a Pauli-X gate.
    fn x(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::x(&qubits))?;
        Ok(slf)
    }

    /// Apply a Pauli-Y gate.
    fn y(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::y(&qubits))?;
        Ok(slf)
    }

    /// Apply a Pauli-Z gate.
    fn z(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::z(&qubits))?;
        Ok(slf)
    }

    /// Apply an Identity gate.
    fn i(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::i(&qubits))?;
        Ok(slf)
    }

    /// Apply an SX gate (sqrt-X).
    fn sx(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sx(&qubits))?;
        Ok(slf)
    }

    /// Apply an SX-dagger gate.
    fn sxdg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sxdg(&qubits))?;
        Ok(slf)
    }

    /// Apply an SY gate (sqrt-Y).
    fn sy(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sy(&qubits))?;
        Ok(slf)
    }

    /// Apply an SY-dagger gate.
    fn sydg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sydg(&qubits))?;
        Ok(slf)
    }

    /// Apply an SZ gate (sqrt-Z).
    fn sz(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sz(&qubits))?;
        Ok(slf)
    }

    /// Apply an SZ-dagger gate.
    fn szdg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::szdg(&qubits))?;
        Ok(slf)
    }

    /// Apply a T gate.
    fn t(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::t(&qubits))?;
        Ok(slf)
    }

    /// Apply a T-dagger gate.
    fn tdg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::tdg(&qubits))?;
        Ok(slf)
    }

    /// Apply an RX rotation.
    fn rx(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::rx(theta.0, &qubits))?;
        Ok(slf)
    }

    /// Apply an RY rotation.
    fn ry(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::ry(theta.0, &qubits))?;
        Ok(slf)
    }

    /// Apply an RZ rotation.
    fn rz(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::rz(theta.0, &qubits))?;
        Ok(slf)
    }

    /// Apply an R1XY rotation (single-qubit gate with two angle parameters).
    ///
    /// Args:
    ///     theta: First rotation angle (angle64 or float radians).
    ///     phi: Second rotation angle (angle64 or float radians).
    ///     qubits: List of qubits to rotate.
    fn r1xy(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        phi: AngleParam,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::r1xy(theta.0, phi.0, &qubits))?;
        Ok(slf)
    }

    /// Apply a U gate (general single-qubit unitary with three angle parameters).
    ///
    /// Args:
    ///     theta: First rotation angle (angle64 or float radians).
    ///     phi: Second rotation angle (angle64 or float radians).
    ///     lam: Third rotation angle (angle64 or float radians).
    ///     qubits: List of qubits to rotate.
    #[pyo3(name = "u")]
    fn u_gate(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        phi: AngleParam,
        lam: AngleParam,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::u(theta.0, phi.0, lam.0, &qubits))?;
        Ok(slf)
    }

    // =========================================================================
    // Two-qubit gates
    // =========================================================================

    /// Apply a CNOT (CX) gate.
    fn cx(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::cx(&pairs))?;
        Ok(slf)
    }

    /// Apply a CY gate.
    fn cy(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::cy(&pairs))?;
        Ok(slf)
    }

    /// Apply a CZ gate.
    fn cz(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::cz(&pairs))?;
        Ok(slf)
    }

    /// Apply an SZZ gate (sqrt-ZZ).
    fn szz(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::szz(&pairs))?;
        Ok(slf)
    }

    /// Apply an SZZ-dagger gate.
    fn szzdg(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::szzdg(&pairs))?;
        Ok(slf)
    }

    /// Apply an F gate.
    fn f(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::f(&qubits))?;
        Ok(slf)
    }

    /// Apply an F-dagger gate.
    fn fdg(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::fdg(&qubits))?;
        Ok(slf)
    }

    /// Apply an SXX gate (sqrt-XX).
    fn sxx(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sxx(&pairs))?;
        Ok(slf)
    }

    /// Apply an SXX-dagger gate.
    fn sxxdg(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::sxxdg(&pairs))?;
        Ok(slf)
    }

    /// Apply an SYY gate (sqrt-YY).
    fn syy(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::syy(&pairs))?;
        Ok(slf)
    }

    /// Apply an SYY-dagger gate.
    fn syydg(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::syydg(&pairs))?;
        Ok(slf)
    }

    /// Apply a SWAP gate.
    fn swap(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::swap(&pairs))?;
        Ok(slf)
    }

    /// Apply a CH gate (controlled-Hadamard).
    fn ch(slf: Py<Self>, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py).add_gate_internal(py, Gate::ch(&pairs))?;
        Ok(slf)
    }

    /// Apply a CRZ gate (controlled-RZ).
    fn crz(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::crz(theta.0, &pairs))?;
        Ok(slf)
    }

    /// Apply a CCX gate (Toffoli).
    fn ccx(
        slf: Py<Self>,
        py: Python<'_>,
        triples: Vec<(usize, usize, usize)>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::ccx(&triples))?;
        Ok(slf)
    }

    /// Apply an RXX rotation.
    fn rxx(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::rxx(theta.0, &pairs))?;
        Ok(slf)
    }

    /// Apply an RYY rotation.
    fn ryy(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::ryy(theta.0, &pairs))?;
        Ok(slf)
    }

    /// Apply an RZZ rotation.
    fn rzz(
        slf: Py<Self>,
        py: Python<'_>,
        theta: AngleParam,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::rzz(theta.0, &pairs))?;
        Ok(slf)
    }

    // =========================================================================
    // Generic gate dispatch (name-based)
    // =========================================================================

    /// Add a gate by name, resolving to a native `GateType` if possible.
    ///
    /// If the name matches a known gate type (e.g., "H", "CX", "SZZ"), it is
    /// added as that native type. Otherwise, it falls through to `custom_gate`.
    ///
    /// Args:
    ///     name: The gate name (case-insensitive for standard gates).
    ///     qubits: List of qubit IDs.
    ///     angles: Optional list of angle values (radians).
    #[pyo3(signature = (name, qubits, angles=None))]
    fn add_gate(
        slf: Py<Self>,
        py: Python<'_>,
        name: &str,
        qubits: Vec<usize>,
        angles: Option<Vec<f64>>,
    ) -> PyResult<Py<Self>> {
        use std::str::FromStr;

        match GateType::from_str(name) {
            Ok(gate_type) => {
                let arity = gate_type.quantum_arity();
                let angle_arity = gate_type.angle_arity();

                // Validate angle count for parameterized gates
                let angle_vals: Vec<Angle64> = angles
                    .unwrap_or_default()
                    .into_iter()
                    .map(Angle64::from_radians)
                    .collect();
                if angle_arity > 0 && angle_vals.len() != angle_arity {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Gate '{name}' requires {angle_arity} angle(s), got {}",
                        angle_vals.len()
                    )));
                }

                // Determine if we need to broadcast (e.g. single-qubit gate on multiple qubits)
                let needs_broadcast =
                    arity > 0 && qubits.len() > arity && qubits.len().is_multiple_of(arity);

                if arity > 0 && qubits.len() != arity && !needs_broadcast {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Gate '{name}' requires {} qubit(s), got {} (not a valid multiple)",
                        arity,
                        qubits.len()
                    )));
                }

                let handle = slf.borrow_mut(py);
                let tick_idx = handle.tick_idx;
                let circuit_py = handle.circuit.clone_ref(py);

                let mut circuit = circuit_py.borrow_mut(py);
                let tick = circuit.inner.get_tick_mut(tick_idx).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Tick {tick_idx} does not exist"
                    ))
                })?;

                if needs_broadcast {
                    // Broadcast: create one gate per arity-chunk of qubits
                    let mut last_idx = None;
                    for chunk in qubits.chunks(arity) {
                        let qubit_ids: GateQubits =
                            chunk.iter().copied().map(QubitId::from).collect();
                        let gate = Gate::new(gate_type, angle_vals.clone(), vec![], qubit_ids);
                        match tick.try_add_gate(gate) {
                            Ok(idx) => {
                                tick.set_gate_attr(
                                    idx,
                                    "_symbol",
                                    Attribute::String(name.to_string()),
                                );
                                last_idx = Some(idx);
                            }
                            Err(err) => {
                                let msg = format!(
                                    "Qubit(s) {:?} already in use in tick {}",
                                    err.conflicting_qubits
                                        .iter()
                                        .map(std::string::ToString::to_string)
                                        .collect::<Vec<_>>(),
                                    tick_idx
                                );
                                return Err(PyErr::new::<QubitConflictError, _>(msg));
                            }
                        }
                    }
                    drop(circuit);
                    drop(handle);
                    slf.borrow_mut(py).last_gate_idx = last_idx;
                    Ok(slf)
                } else {
                    // Normal: create single gate
                    let qubit_ids: GateQubits = qubits.into_iter().map(QubitId::from).collect();
                    let gate = Gate::new(gate_type, angle_vals, vec![], qubit_ids);
                    match tick.try_add_gate(gate) {
                        Ok(idx) => {
                            tick.set_gate_attr(idx, "_symbol", Attribute::String(name.to_string()));
                            drop(circuit);
                            drop(handle);
                            slf.borrow_mut(py).last_gate_idx = Some(idx);
                            Ok(slf)
                        }
                        Err(err) => {
                            let msg = format!(
                                "Qubit(s) {:?} already in use in tick {}",
                                err.conflicting_qubits
                                    .iter()
                                    .map(std::string::ToString::to_string)
                                    .collect::<Vec<_>>(),
                                tick_idx
                            );
                            Err(PyErr::new::<QubitConflictError, _>(msg))
                        }
                    }
                }
            }
            Err(_) => {
                // Unknown gate name - fall through to custom_gate
                PyTickHandle::custom_gate(slf, py, name, qubits, angles)
            }
        }
    }

    // =========================================================================
    // Custom (unrecognized) gates
    // =========================================================================

    /// Add a custom (unrecognized) gate on the given qubits.
    fn custom(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        let qubit_ids: GateQubits = qubits.into_iter().map(QubitId::from).collect();
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::custom(qubit_ids))?;
        Ok(slf)
    }

    /// Add a custom gate with signature validation.
    ///
    /// On first use, the gate name's signature (quantum arity, angle arity)
    /// is recorded. Subsequent uses are validated against this signature.
    ///
    /// Args:
    ///     name: The gate name.
    ///     qubits: List of qubit IDs.
    ///     angles: Optional list of angle values (radians).
    ///
    /// Raises:
    ///     `GateSignatureMismatchError`: If the arity does not match a previous use.
    ///     `QubitConflictError`: If a qubit is already in use in this tick.
    #[pyo3(signature = (name, qubits, angles=None))]
    fn custom_gate(
        slf: Py<Self>,
        py: Python<'_>,
        name: &str,
        qubits: Vec<usize>,
        angles: Option<Vec<f64>>,
    ) -> PyResult<Py<Self>> {
        let angle_vals: Vec<Angle64> = angles
            .unwrap_or_default()
            .into_iter()
            .map(Angle64::from_radians)
            .collect();

        let handle = slf.borrow_mut(py);
        let tick_idx = handle.tick_idx;
        let circuit_py = handle.circuit.clone_ref(py);

        // Validate/register and add gate
        let mut circuit = circuit_py.borrow_mut(py);
        match circuit
            .inner
            .validate_or_register_gate(name, qubits.len(), angle_vals.len())
        {
            Ok(()) => {}
            Err(e) => {
                return Err(PyErr::new::<GateSignatureMismatchError, _>(e.to_string()));
            }
        }

        let qubit_ids: GateQubits = qubits.into_iter().map(QubitId::from).collect();
        let gate = Gate::new(GateType::Custom, angle_vals, vec![], qubit_ids);

        if let Some(tick) = circuit.inner.get_tick_mut(tick_idx) {
            match tick.try_add_gate(gate) {
                Ok(idx) => {
                    tick.set_gate_attr(idx, "_symbol", Attribute::String(name.to_string()));
                    drop(circuit);
                    drop(handle);
                    // Update last_gate_idx through a fresh borrow
                    slf.borrow_mut(py).last_gate_idx = Some(idx);
                    Ok(slf)
                }
                Err(err) => {
                    let msg = format!(
                        "Qubit(s) {:?} already in use in tick {}",
                        err.conflicting_qubits
                            .iter()
                            .map(std::string::ToString::to_string)
                            .collect::<Vec<_>>(),
                        tick_idx
                    );
                    Err(PyErr::new::<QubitConflictError, _>(msg))
                }
            }
        } else {
            drop(circuit);
            drop(handle);
            Ok(slf)
        }
    }

    // =========================================================================
    // State preparation and measurement
    // =========================================================================

    /// Prepare qubits in the |0> state.
    ///
    /// Returns a `TickPrepHandle` that allows attaching metadata via `.meta()`.
    /// This breaks the chain - only `.meta()` can be called on the result.
    fn pz(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<PyTickPrepHandle> {
        let (circuit, tick_idx, gate_idx) = {
            let mut handle = slf.borrow_mut(py);
            let gate_idx = handle.add_gate_get_idx(py, Gate::pz(&qubits))?;
            (handle.circuit.clone_ref(py), handle.tick_idx, gate_idx)
        };
        Ok(PyTickPrepHandle {
            circuit,
            tick_idx,
            gate_idx,
        })
    }

    /// Measure qubits in the Z basis.
    ///
    /// Returns a `TickMeasureHandle` that allows attaching metadata via `.meta()`.
    /// This breaks the chain - only `.meta()` can be called on the result.
    fn mz(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<PyTickMeasureHandle> {
        let (circuit, tick_idx, gate_idx) = {
            let mut handle = slf.borrow_mut(py);
            let gate_idx = handle.add_gate_get_idx(py, Gate::mz(&qubits))?;
            (handle.circuit.clone_ref(py), handle.tick_idx, gate_idx)
        };
        Ok(PyTickMeasureHandle {
            circuit,
            tick_idx,
            gate_idx,
        })
    }

    /// Measure and free qubits (destructive measurement).
    ///
    /// Returns a `TickMeasureHandle` that allows attaching metadata via `.meta()`.
    fn mz_free(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<PyTickMeasureHandle> {
        let (circuit, tick_idx, gate_idx) = {
            let mut handle = slf.borrow_mut(py);
            let gate_idx = handle.add_gate_get_idx(py, Gate::mz_free(&qubits))?;
            (handle.circuit.clone_ref(py), handle.tick_idx, gate_idx)
        };
        Ok(PyTickMeasureHandle {
            circuit,
            tick_idx,
            gate_idx,
        })
    }

    // =========================================================================
    // Resource management
    // =========================================================================

    /// Allocate qubits.
    fn qalloc(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::qalloc(&qubits))?;
        Ok(slf)
    }

    /// Free qubits.
    fn qfree(slf: Py<Self>, py: Python<'_>, qubits: Vec<usize>) -> PyResult<Py<Self>> {
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::qfree(&qubits))?;
        Ok(slf)
    }

    // =========================================================================
    // Timing
    // =========================================================================

    /// Apply an idle gate with a specified duration.
    ///
    /// Duration is in abstract time units - interpretation depends on your noise model.
    ///
    /// Args:
    ///     duration: Duration as `TimeUnits`, `Nanoseconds` (deprecated), or integer.
    ///     qubits: List of qubits to idle.
    fn idle(
        slf: Py<Self>,
        py: Python<'_>,
        duration: &Bound<'_, PyAny>,
        qubits: Vec<usize>,
    ) -> PyResult<Py<Self>> {
        let units = if let Ok(py_tu) = duration.extract::<PyTimeUnits>() {
            py_tu.inner
        } else if let Ok(py_ns) = duration.extract::<PyNanoseconds>() {
            // Deprecated: treat nanoseconds value as time units directly
            TimeUnits::new(py_ns.ns)
        } else if let Ok(raw) = duration.extract::<u64>() {
            TimeUnits::new(raw)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "duration must be TimeUnits, Nanoseconds, or an integer",
            ));
        };
        let qubit_ids: Vec<QubitId> = qubits.into_iter().map(QubitId::from).collect();
        slf.borrow_mut(py)
            .add_gate_internal(py, Gate::idle(units.as_f64(), qubit_ids))?;
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        format!("TickHandle(tick={})", self.tick_idx)
    }
}

/// Register the quantum module with Python.
pub fn register_quantum_circuit_types(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent_module.py();

    // Add classes to parent module
    parent_module.add_class::<PyQubitId>()?;
    parent_module.add_class::<PyGateType>()?;
    parent_module.add_class::<PyGate>()?;
    parent_module.add_class::<PyDagCircuit>()?;
    parent_module.add_class::<PyTick>()?;
    parent_module.add_class::<PyTickCircuit>()?;
    parent_module.add_class::<PyTickHandle>()?;
    parent_module.add_class::<PyTickPrepHandle>()?;
    parent_module.add_class::<PyTickMeasureHandle>()?;

    // Add exceptions
    parent_module.add(
        "DagCircuitWouldCycleError",
        py.get_type::<DagCircuitWouldCycleError>(),
    )?;
    parent_module.add("HugrConversionError", py.get_type::<HugrConversionError>())?;
    parent_module.add("QubitConflictError", py.get_type::<QubitConflictError>())?;
    parent_module.add(
        "GateSignatureMismatchError",
        py.get_type::<GateSignatureMismatchError>(),
    )?;

    // Add HUGR conversion functions
    parent_module.add_function(wrap_pyfunction!(py_hugr_to_dag_circuit, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(py_hugr_op_to_gate_type, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(py_gate_type_to_hugr_op, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(py_is_quantum_operation, parent_module)?)?;

    Ok(())
}

/// Register time unit types with Python.
///
/// These are registered separately so they can be added at the pecos namespace level.
pub fn register_time_unit_types(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyNanoseconds>()?;
    parent_module.add_class::<PyTimeUnits>()?;
    Ok(())
}
