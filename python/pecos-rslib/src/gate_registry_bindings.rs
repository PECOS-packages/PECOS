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

//! Python bindings for the gate registration system.

use pecos::core::Value;
use pecos::core::gate_type::GateType;
use pecos::core::{Angle64, AngleSource, GateDefinitionBuilder, GateRegistry, QubitId};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::HashMap;

/// Parse a gate name string into a `GateType`.
fn parse_gate_type(name: &str) -> PyResult<GateType> {
    match name {
        "I" => Ok(GateType::I),
        "X" => Ok(GateType::X),
        "Y" => Ok(GateType::Y),
        "Z" => Ok(GateType::Z),
        "SX" => Ok(GateType::SX),
        "SXdg" => Ok(GateType::SXdg),
        "SY" => Ok(GateType::SY),
        "SYdg" => Ok(GateType::SYdg),
        "SZ" | "S" => Ok(GateType::SZ),
        "SZdg" | "Sdg" => Ok(GateType::SZdg),
        "H" => Ok(GateType::H),
        "RX" => Ok(GateType::RX),
        "RY" => Ok(GateType::RY),
        "RZ" => Ok(GateType::RZ),
        "T" => Ok(GateType::T),
        "Tdg" => Ok(GateType::Tdg),
        "U" => Ok(GateType::U),
        "R1XY" => Ok(GateType::R1XY),
        "CX" | "CNOT" => Ok(GateType::CX),
        "CY" => Ok(GateType::CY),
        "CZ" => Ok(GateType::CZ),
        "CH" => Ok(GateType::CH),
        "SZZ" => Ok(GateType::SZZ),
        "SZZdg" => Ok(GateType::SZZdg),
        "SWAP" => Ok(GateType::SWAP),
        "CRZ" => Ok(GateType::CRZ),
        "RXX" => Ok(GateType::RXX),
        "RYY" => Ok(GateType::RYY),
        "RZZ" => Ok(GateType::RZZ),
        "CCX" | "Toffoli" => Ok(GateType::CCX),
        "Measure" => Ok(GateType::Measure),
        "MeasureLeaked" => Ok(GateType::MeasureLeaked),
        "MeasureFree" => Ok(GateType::MeasureFree),
        "Prep" => Ok(GateType::Prep),
        "QAlloc" => Ok(GateType::QAlloc),
        "QFree" => Ok(GateType::QFree),
        "Idle" => Ok(GateType::Idle),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown gate type: '{name}'"
        ))),
    }
}

/// Convert a Python object to a `Value`.
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    // Try bool before int since Python bools are ints
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return Ok(Value::Dict(py_dict_to_value_map(dict)?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let items: PyResult<Vec<Value>> = list.iter().map(|item| py_to_value(&item)).collect();
        return Ok(Value::List(items?));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Metadata values must be str, int, float, bool, list, or dict, got {}",
        obj.get_type().name()?
    )))
}

/// Convert a `Value` to a Python object.
fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<Py<PyAny>> {
    match val {
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Int(i) => Ok(i.into_pyobject(py)?.into_any().unbind()),
        Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::List(items) => {
            let py_list = PyList::empty(py);
            for item in items {
                py_list.append(value_to_py(py, item)?)?;
            }
            Ok(py_list.unbind().into_any())
        }
        Value::Dict(map) => {
            let py_dict = PyDict::new(py);
            for (k, v) in map {
                py_dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(py_dict.unbind().into_any())
        }
    }
}

/// Convert a Python dict to a `HashMap<String, Value>`.
fn py_dict_to_value_map(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Value>> {
    let mut metadata = HashMap::new();
    for (key, val) in dict.iter() {
        let k: String = key.extract()?;
        let v = py_to_value(&val)?;
        metadata.insert(k, v);
    }
    Ok(metadata)
}

/// Python-friendly angle source specification for decomposition steps.
#[pyclass(name = "AngleSource", from_py_object)]
#[derive(Clone)]
pub struct PyAngleSource {
    inner: AngleSource,
}

#[pymethods]
impl PyAngleSource {
    /// Forward the i-th input angle from the parent gate.
    #[staticmethod]
    fn input(index: u8) -> Self {
        Self {
            inner: AngleSource::Input(index),
        }
    }

    /// Use a fixed angle value (in turns, where 1.0 = full turn).
    #[staticmethod]
    fn fixed(value: f64) -> Self {
        Self {
            inner: AngleSource::Fixed(Angle64::from_turns(value)),
        }
    }

    /// Negate the i-th input angle from the parent gate.
    #[staticmethod]
    fn neg_input(index: u8) -> Self {
        Self {
            inner: AngleSource::NegInput(index),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            AngleSource::Input(i) => format!("AngleSource.input({i})"),
            AngleSource::Fixed(a) => format!("AngleSource.fixed({a})"),
            AngleSource::NegInput(i) => format!("AngleSource.neg_input({i})"),
        }
    }
}

/// Builder for constructing gate definitions with a fluent API.
#[pyclass(name = "GateDefBuilder")]
pub struct PyGateDefBuilder {
    inner: Option<GateDefinitionBuilder>,
}

#[pymethods]
impl PyGateDefBuilder {
    /// Set the number of angle parameters this gate accepts.
    fn angle_arity(slf: Py<Self>, py: Python<'_>, arity: usize) -> Py<Self> {
        let mut this = slf.borrow_mut(py);
        let builder = this.inner.take().expect("Builder already consumed");
        this.inner = Some(builder.angle_arity(arity));
        drop(this);
        slf
    }

    /// Add a non-parameterized gate step to the decomposition.
    fn step(
        slf: Py<Self>,
        py: Python<'_>,
        gate_name: &str,
        qubit_indices: Vec<u8>,
    ) -> PyResult<Py<Self>> {
        let gate_type = parse_gate_type(gate_name)?;
        let mut this = slf.borrow_mut(py);
        let builder = this.inner.take().expect("Builder already consumed");
        this.inner = Some(builder.step(gate_type, &qubit_indices));
        drop(this);
        Ok(slf)
    }

    /// Add a parameterized gate step to the decomposition.
    fn step_with_angles(
        slf: Py<Self>,
        py: Python<'_>,
        gate_name: &str,
        qubit_indices: Vec<u8>,
        angle_sources: Vec<PyAngleSource>,
    ) -> PyResult<Py<Self>> {
        let gate_type = parse_gate_type(gate_name)?;
        let sources: Vec<AngleSource> = angle_sources.into_iter().map(|s| s.inner).collect();
        let mut this = slf.borrow_mut(py);
        let builder = this.inner.take().expect("Builder already consumed");
        this.inner = Some(builder.step_with_angles(gate_type, &qubit_indices, &sources));
        drop(this);
        Ok(slf)
    }

    /// Add a gate step with angles and per-step metadata.
    ///
    /// Metadata values can be str, int, float, or bool.
    fn step_with_metadata(
        slf: Py<Self>,
        py: Python<'_>,
        gate_name: &str,
        qubit_indices: Vec<u8>,
        angle_sources: Vec<PyAngleSource>,
        metadata: &Bound<'_, PyDict>,
    ) -> PyResult<Py<Self>> {
        let gate_type = parse_gate_type(gate_name)?;
        let sources: Vec<AngleSource> = angle_sources.into_iter().map(|s| s.inner).collect();
        let meta = py_dict_to_value_map(metadata)?;
        let mut this = slf.borrow_mut(py);
        let builder = this.inner.take().expect("Builder already consumed");
        this.inner = Some(builder.step_with_metadata(gate_type, &qubit_indices, &sources, meta));
        drop(this);
        Ok(slf)
    }

    /// Finalize and register this gate definition into a registry.
    fn register_into(&mut self, registry: &mut PyGateRegistry) -> PyResult<()> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed"))?;
        registry.inner.register(builder.build());
        Ok(())
    }
}

/// Registry mapping gate names to definitions with decompositions.
#[pyclass(name = "GateRegistry")]
pub struct PyGateRegistry {
    pub(crate) inner: GateRegistry,
}

#[pymethods]
impl PyGateRegistry {
    #[new]
    fn new() -> Self {
        Self {
            inner: GateRegistry::new(),
        }
    }

    /// Start building a gate definition.
    fn define(&self, name: String, quantum_arity: usize) -> PyGateDefBuilder {
        PyGateDefBuilder {
            inner: Some(GateDefinitionBuilder::new(name, quantum_arity)),
        }
    }

    /// Check if a gate is registered.
    fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Decompose a registered gate into concrete steps.
    ///
    /// Returns a list of (`gate_name`, qubits, angles, metadata) tuples, or None if
    /// the gate is not registered or has no decomposition.
    fn decompose(
        &self,
        py: Python<'_>,
        name: &str,
        qubits: Vec<usize>,
        angles: Vec<f64>,
    ) -> PyResult<Py<PyAny>> {
        let qubit_ids: Vec<QubitId> = qubits.into_iter().map(QubitId::from).collect();
        let angle_vals: Vec<Angle64> = angles.into_iter().map(Angle64::from_turns).collect();

        match self.inner.decompose(name, &qubit_ids, &angle_vals) {
            None => Ok(py.None()),
            Some(steps) => {
                let result = PyList::empty(py);
                for (gate_type, step_qubits, step_angles, step_meta) in steps {
                    let gate_name = format!("{gate_type}");
                    let py_qubits: Vec<usize> =
                        step_qubits.iter().map(|q| usize::from(*q)).collect();
                    let py_angles: Vec<f64> = step_angles
                        .iter()
                        .map(|a| a.to_radians() / std::f64::consts::TAU)
                        .collect();
                    let py_meta = PyDict::new(py);
                    for (k, v) in &step_meta {
                        py_meta.set_item(k, value_to_py(py, v)?)?;
                    }
                    result.append((gate_name, py_qubits, py_angles, py_meta))?;
                }
                Ok(result.unbind().into_any())
            }
        }
    }
}

/// Register gate registry types into a Python module.
pub fn register_gate_registry_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGateRegistry>()?;
    m.add_class::<PyGateDefBuilder>()?;
    m.add_class::<PyAngleSource>()?;
    Ok(())
}
