//! Python batch normalization for the existing checked Rust v7 controller.
use crate::dag_circuit_bindings::PyGate;
use pecos_core::errors::PecosError;
use pecos_engines::scheduled_events::{
    MAX_BATCH_OPERATIONS, ScheduledBatchAdapter, ScheduledCrosstalk, ScheduledEventBatch,
    ScheduledEventNoise, ScheduledEventOp, ScheduledGateBuffer,
};
use pecos_engines::scheduled_frame::{ScheduledIdleZ, ScheduledNoise};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::cell::Cell;

/// An ordered spectator exposure returned alongside gates by a batch adapter.
/// It creates no physical busy time and has no program-visible measurement.
#[pyclass(name = "ScheduledCrosstalk", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyScheduledCrosstalk {
    inner: ScheduledCrosstalk,
}
fn check_spectator_ids(ids: &[usize]) -> PyResult<()> {
    let mut seen = std::collections::BTreeSet::new();
    if ids.iter().any(|q| !seen.insert(*q)) {
        return Err(PyValueError::new_err(
            "spectator IDs must be distinct; the profile capacity is checked at admission",
        ));
    }
    Ok(())
}
#[pymethods]
impl PyScheduledCrosstalk {
    /// Explicit potential victims in sampling order; unprepared/leaked are skipped.
    #[staticmethod]
    fn local(victims: Vec<usize>) -> PyResult<Self> {
        check_spectator_ids(&victims)?;
        Ok(Self {
            inner: ScheduledCrosstalk::Local(victims),
        })
    }
    /// Prepared logical qubits except these IDs, in the general model's order.
    #[staticmethod]
    fn global_excluding(exclusions: Vec<usize>) -> PyResult<Self> {
        check_spectator_ids(&exclusions)?;
        Ok(Self {
            inner: ScheduledCrosstalk::GlobalExcluding(exclusions),
        })
    }
}
/// Opt into explicit measurement exposures while retaining all local/idle settings.
/// No automatic exposure insertion. Wrap the returned profile with
/// scheduled_event_idle_noise(profile, adapter_factory) for live v7 execution.
/// Accrued idle noise runs at the next physical operation, after the exposure.
/// Non-Z idle channels can change output distributions under this ordering even
/// with symmetric transitions. Z/RZ channels need not preserve seeded paths:
/// exposure leakage can suppress later idle-noise RNG draws.
#[pyfunction]
#[pyo3(signature = (profile, model, *, local_probability = 0.0, global_probability = 0.0))]
pub fn scheduled_measurement_crosstalk(
    profile: Py<PyAny>,
    model: std::collections::BTreeMap<String, f64>,
    local_probability: f64,
    global_probability: f64,
    py: Python<'_>,
) -> PyResult<crate::engine_builders::PyScheduledIdleNoise> {
    let profile =
        crate::engine_builders::extract_scheduled_idle(&profile, py).ok_or_else(|| {
            PyTypeError::new_err("expected a checked scheduled idle or local noise profile")
        })?;
    Ok(crate::engine_builders::PyScheduledIdleNoise {
        inner: profile
            .with_measurement_crosstalk(local_probability, global_probability, model)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

thread_local! {
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}
struct CallbackScope(bool);
impl CallbackScope {
    fn enter() -> Self {
        Self(IN_CALLBACK.replace(true))
    }
}
impl Drop for CallbackScope {
    fn drop(&mut self) {
        IN_CALLBACK.set(self.0);
    }
}
/// Waiting for an engine from one of its callbacks would create a worker cycle.
/// Reject built simulation operations from callbacks; external threads may wait.
pub(crate) fn reject_callback_reentry() -> PyResult<()> {
    if IN_CALLBACK.get() {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Built simulation run/reset cannot be called from a scheduled adapter callback",
        ))
    } else {
        Ok(())
    }
}

/// Owned read-only snapshot of one original batch. No execution state is exposed.
#[pyclass(name = "ScheduledEventBatch", frozen)]
struct PyScheduledEventBatch {
    inner: ScheduledEventBatch,
}
#[pymethods]
impl PyScheduledEventBatch {
    #[getter]
    fn runtime_shot_id(&self) -> u64 {
        self.inner.runtime_shot_id
    }
    #[getter]
    fn batch_index(&self) -> u64 {
        self.inner.batch_index
    }
    #[getter]
    fn start_nanos(&self) -> u64 {
        self.inner.start_nanos
    }
    #[getter]
    fn duration_nanos(&self) -> u64 {
        self.inner.duration_nanos
    }
    /// Copies in original order: Gate objects or (tag, bytes) custom-event tuples.
    #[getter]
    fn operations(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .operations
            .iter()
            .map(|op| match op {
                ScheduledEventOp::Gate(g) => {
                    Ok(Py::new(py, PyGate::from(g.as_ref().clone()))?.into_any())
                }
                ScheduledEventOp::Custom { tag, payload } => Ok((*tag, PyBytes::new(py, payload))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()),
            })
            .collect()
    }
    /// (original operation position, native result ID, program result ID).
    /// Native IDs are unique; program IDs may repeat when a slot is re-measured.
    #[getter]
    fn measurements(&self) -> Vec<(usize, u64, u64)> {
        self.inner
            .measurements
            .iter()
            .map(|m| (m.operation_index, m.runtime_result, m.program_result))
            .collect()
    }
}

fn callback_error(stage: &str, error: PyErr) -> PecosError {
    PecosError::Processing(format!("scheduled adapter {stage}: {error}"))
}
struct PythonAdapter {
    object: Py<PyAny>,
}
impl ScheduledBatchAdapter for PythonAdapter {
    fn validate(&self, batch: &ScheduledEventBatch) -> Result<(), PecosError> {
        let _scope = CallbackScope::enter();
        Python::attach(|py| -> PyResult<()> {
            let input = Py::new(
                py,
                PyScheduledEventBatch {
                    inner: batch.clone(),
                },
            )?;
            let result = self.object.call_method1(py, "validate", (input,))?;
            if !result.is_none(py) {
                return Err(PyTypeError::new_err("validate must return None or raise"));
            }
            Ok(())
        })
        .map_err(|e| callback_error("validate", e))
    }
    fn translate(
        &mut self,
        batch: &ScheduledEventBatch,
        output: &mut ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError> {
        let _scope = CallbackScope::enter();
        Python::attach(|py| {
            let input = Py::new(
                py,
                PyScheduledEventBatch {
                    inner: batch.clone(),
                },
            )
            .map_err(|e| callback_error("input", e))?;
            let result = self
                .object
                .call_method1(py, "translate", (input,))
                .map_err(|e| callback_error("translate", e))?;
            let gates = result
                .bind(py)
                .cast::<PyList>()
                .map_err(|e| callback_error("output", e.into()))?;
            if gates.len() > MAX_BATCH_OPERATIONS {
                return Err(PecosError::Input(
                    "scheduled adapter expansion limit".into(),
                ));
            }
            for gate in gates.iter() {
                if let Ok(effect) = gate.extract::<PyScheduledCrosstalk>() {
                    output.push_crosstalk(effect.inner).map_err(|e| {
                        PecosError::Input(format!("scheduled adapter spectator: {e}"))
                    })?;
                    continue;
                }
                let gate = gate
                    .extract::<PyGate>()
                    .map_err(|e| callback_error("output gate", e.into()))?;
                output.push(gate.into()).map_err(|e| {
                    PecosError::Input(format!("scheduled adapter output gate: {e}"))
                })?;
            }
            Ok(())
        })
    }
}

/// Checked v7 idle-Z profile with a trusted Python factory.
#[pyclass(name = "ScheduledEventIdleZ", from_py_object)]
#[derive(Clone)]
pub struct PyScheduledEventIdleZ {
    pub(crate) inner: ScheduledEventNoise,
}
/// Factory receives (run, worker, shot), returns a fresh validate/translate object.
#[pyfunction]
#[pyo3(signature = (qubits, adapter_factory, *, linear = 0.0, sine = 0.0, coherent = 0.0))]
pub fn scheduled_event_idle_z(
    qubits: usize,
    adapter_factory: Py<PyAny>,
    linear: f64,
    sine: f64,
    coherent: f64,
    py: Python<'_>,
) -> PyResult<PyScheduledEventIdleZ> {
    let profile = ScheduledIdleZ::new(qubits, linear, sine, coherent)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyScheduledEventIdleZ {
        inner: event_profile(profile.into(), adapter_factory, py)?,
    })
}
/// Checked v7 local idle-noise profile with a trusted Python factory.
#[pyclass(name = "ScheduledEventIdleNoise", from_py_object)]
#[derive(Clone)]
pub struct PyScheduledEventIdleNoise {
    pub(crate) inner: ScheduledEventNoise,
}
/// Attach a per-shot batch adapter to a checked scheduled idle profile.
#[pyfunction]
pub fn scheduled_event_idle_noise(
    profile: crate::engine_builders::PyScheduledIdleNoise,
    adapter_factory: Py<PyAny>,
    py: Python<'_>,
) -> PyResult<PyScheduledEventIdleNoise> {
    Ok(PyScheduledEventIdleNoise {
        inner: event_profile(profile.inner, adapter_factory, py)?,
    })
}
/// A per-shot adapter with the checked local fault profile.
#[pyclass(name = "ScheduledEventLocalNoise", from_py_object)]
#[derive(Clone)]
pub struct PyScheduledEventLocalNoise {
    pub(crate) inner: ScheduledEventNoise,
}
/// Attach a per-shot batch adapter to a checked scheduled local fault profile.
/// Factory receives (run, worker, shot), returning a fresh validate/translate object.
#[pyfunction]
pub fn scheduled_event_local_noise(
    profile: crate::engine_builders::PyScheduledLocalNoise,
    adapter_factory: Py<PyAny>,
    py: Python<'_>,
) -> PyResult<PyScheduledEventLocalNoise> {
    Ok(PyScheduledEventLocalNoise {
        inner: event_profile(profile.inner.into(), adapter_factory, py)?,
    })
}
fn event_profile(
    profile: ScheduledNoise,
    adapter_factory: Py<PyAny>,
    py: Python<'_>,
) -> PyResult<ScheduledEventNoise> {
    if !adapter_factory.bind(py).is_callable() {
        return Err(PyTypeError::new_err("adapter_factory must be callable"));
    }
    Ok(ScheduledEventNoise::new(profile, move |context| {
        let _scope = CallbackScope::enter();
        Python::attach(|py| -> PyResult<Box<dyn ScheduledBatchAdapter>> {
            let object =
                adapter_factory.call1(py, ((context.run, context.worker, context.shot),))?;
            for method in ["validate", "translate"] {
                if !object.getattr(py, method)?.bind(py).is_callable() {
                    return Err(PyTypeError::new_err(format!(
                        "adapter.{method} must be callable"
                    )));
                }
            }
            Ok(Box::new(PythonAdapter { object }))
        })
        .map_err(|e| callback_error("factory", e))
    }))
}
pub(crate) fn extract_event_noise(
    noise: &Py<PyAny>,
    py: Python<'_>,
) -> Option<ScheduledEventNoise> {
    if let Ok(profile) = noise.extract::<PyScheduledEventLocalNoise>(py) {
        Some(profile.inner)
    } else if let Ok(profile) = noise.extract::<PyScheduledEventIdleNoise>(py) {
        Some(profile.inner)
    } else {
        noise
            .extract::<PyScheduledEventIdleZ>(py)
            .ok()
            .map(|p| p.inner)
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScheduledCrosstalk>()?;
    m.add_function(wrap_pyfunction!(scheduled_measurement_crosstalk, m)?)?;
    m.add_class::<PyScheduledEventBatch>()?;
    m.add_class::<PyScheduledEventIdleZ>()?;
    m.add_class::<PyScheduledEventIdleNoise>()?;
    m.add_class::<PyScheduledEventLocalNoise>()?;
    m.add_function(wrap_pyfunction!(scheduled_event_local_noise, m)?)?;
    m.add_function(wrap_pyfunction!(scheduled_event_idle_noise, m)?)?;
    m.add_function(wrap_pyfunction!(scheduled_event_idle_z, m)?)?;
    Ok(())
}

pub(crate) fn is_event_noise(noise: &Py<PyAny>) -> bool {
    Python::attach(|py| {
        noise
            .bind(py)
            .is_instance_of::<PyScheduledEventLocalNoise>()
            || noise.bind(py).is_instance_of::<PyScheduledEventIdleZ>()
            || noise.bind(py).is_instance_of::<PyScheduledEventIdleNoise>()
    })
}
pub(crate) fn unsupported_route() -> PyErr {
    PyTypeError::new_err(
        "scheduled event noise requires QIS/HUGR on the engines stack without operation tracing",
    )
}
