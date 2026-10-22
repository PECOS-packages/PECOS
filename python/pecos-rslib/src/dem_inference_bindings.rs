// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Shared Rust DEM inference settings and algorithms.
use crate::dag_circuit_bindings::PyTickCircuit;
use crate::engine_builders::{PyQis, configured_selene_runtime, configured_selene_runtime_plugin};
use pecos_qec::dem_inference::{DemInferenceData, DemInferenceSettings};
use pecos_qec::{DemOutputSelector, DemProgramRequest, SeleneDemProgram};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Detector supports followed by observable supports, using stable measurement IDs.
type InferredSupports = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Fluent configuration for empirical detector/observable annotation inference.
/// Setters return a new settings value; defaults use automatic probe sizing and
/// require measurement provenance. Seed/runtime belong to the DEM builder.
#[pyclass(
    name = "DemInferenceSettings",
    module = "pecos_rslib.qec",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug, Default)]
pub struct PyDemInferenceSettings {
    inner: DemInferenceSettings,
}

#[pymethods]
impl PyDemInferenceSettings {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn with_raw_tag(&self, tag: String) -> Self {
        Self {
            inner: self.inner.clone().with_raw_tag(tag),
        }
    }
    fn with_detector_tag(&self, tag: String) -> Self {
        Self {
            inner: self.inner.clone().with_detector_tag(tag),
        }
    }
    fn with_observable_tags(&self, tags: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().with_observable_tags(tags),
        }
    }
    fn with_probe_shots(&self, shots: usize) -> Self {
        Self {
            inner: self.inner.clone().with_probe_shots(shots),
        }
    }
    fn with_provenance_shots(&self, shots: usize) -> Self {
        Self {
            inner: self.inner.clone().with_provenance_shots(shots),
        }
    }
    fn with_validation_rows(&self, rows: usize) -> Self {
        Self {
            inner: self.inner.clone().with_validation_rows(rows),
        }
    }
    fn with_require_raw_provenance(&self, required: bool) -> Self {
        Self {
            inner: self.inner.clone().with_require_raw_provenance(required),
        }
    }
    #[getter]
    fn raw_tag(&self) -> &str {
        &self.inner.raw_tag
    }
    #[getter]
    fn detector_tag(&self) -> &str {
        &self.inner.detector_tag
    }
    #[getter]
    fn observable_tags(&self) -> Vec<String> {
        self.inner.observable_tags.clone()
    }
    #[getter]
    fn probe_shots(&self) -> Option<usize> {
        self.inner.probe_shots
    }
    #[getter]
    fn provenance_shots(&self) -> usize {
        self.inner.provenance_shots
    }
    #[getter]
    fn validation_rows(&self) -> usize {
        self.inner.validation_rows
    }
    #[getter]
    fn require_raw_provenance(&self) -> bool {
        self.inner.require_raw_provenance
    }
    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
    fn resolved_probe_shots(&self, raw_width: usize) -> PyResult<usize> {
        self.inner
            .resolved_probe_shots(raw_width)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
    fn infer(
        &self,
        raw_measurement_ids: Vec<usize>,
        raw_rows: Vec<Vec<u8>>,
        detector_rows: Vec<Vec<u8>>,
        observable_rows: Vec<Vec<u8>>,
    ) -> PyResult<InferredSupports> {
        let report = self
            .inner
            .infer(&DemInferenceData {
                raw_measurement_ids,
                raw_rows,
                detector_rows,
                observable_rows,
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((report.detector_supports, report.observable_supports))
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Runtime leaf configuration, resolved by Python's shared runtime dispatch.
#[pyclass(
    name = "_SeleneRuntimeSpec",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub struct PySeleneRuntimeSpec {
    runtime: Option<pecos_qis::SeleneRuntime>,
}

#[pymethods]
impl PySeleneRuntimeSpec {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (runtime_name = None, *, custom_event_policy = "reject_unhandled", native_gates = None))]
    fn selene_runtime(
        &mut self,
        runtime_name: Option<&str>,
        custom_event_policy: &str,
        native_gates: Option<Vec<String>>,
    ) -> PyResult<Self> {
        self.runtime = Some(configured_selene_runtime(
            runtime_name,
            custom_event_policy,
            native_gates,
        )?);
        Ok(self.clone())
    }

    #[pyo3(signature = (library_file, init_args = None, library_search_dirs = None, *, custom_event_policy = "reject_unhandled", native_gates = None))]
    fn selene_runtime_plugin(
        &mut self,
        library_file: &str,
        init_args: Option<Vec<String>>,
        library_search_dirs: Option<Vec<String>>,
        custom_event_policy: &str,
        native_gates: Option<Vec<String>>,
    ) -> PyResult<Self> {
        self.runtime = Some(configured_selene_runtime_plugin(
            library_file,
            init_args,
            library_search_dirs,
            custom_event_policy,
            native_gates,
        )?);
        Ok(self.clone())
    }
}

#[pyfunction(name = "_capture_dem_evidence")]
#[pyo3(signature = (program, num_qubits, seed, runtime, settings, detector_outputs = None, observable_outputs = None))]
fn capture_dem_evidence<'py>(
    py: Python<'py>,
    program: &PyQis,
    num_qubits: usize,
    seed: u64,
    runtime: &PySeleneRuntimeSpec,
    settings: &PyDemInferenceSettings,
    detector_outputs: Option<Vec<(String, usize)>>,
    observable_outputs: Option<Vec<(String, usize)>>,
) -> PyResult<Bound<'py, PyDict>> {
    let runtime = runtime
        .runtime
        .clone()
        .ok_or_else(|| PyValueError::new_err("Selene runtime spec is not configured"))?;
    let program = SeleneDemProgram::new(program.inner.clone()).with_runtime(runtime);
    let convert = |outputs: Vec<(String, usize)>| {
        outputs
            .into_iter()
            .map(|(tag, element)| DemOutputSelector { tag, element })
            .collect::<Vec<_>>()
    };
    let detector_outputs = detector_outputs.map(convert);
    let observable_outputs = observable_outputs.map(convert);
    let settings = settings.inner.clone();
    let evidence = py
        .detach(|| {
            program.capture_and_analyze(&DemProgramRequest {
                num_qubits,
                seed,
                inference: Some(&settings),
                detector_outputs: detector_outputs.as_deref(),
                observable_outputs: observable_outputs.as_deref(),
            })
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let data = evidence
        .probes
        .expect("inference request produces probe data");
    let result = PyDict::new(py);
    result.set_item(
        "circuit",
        PyTickCircuit {
            inner: evidence.circuit,
        },
    )?;
    result.set_item("raw_measurement_ids", data.raw_measurement_ids)?;
    result.set_item("raw_rows", data.raw_rows)?;
    result.set_item("detector_rows", data.detector_rows)?;
    result.set_item("observable_rows", data.observable_rows)?;
    result.set_item("binding", evidence.binding)?;
    let labels = |outputs: Vec<DemOutputSelector>| {
        outputs
            .into_iter()
            .map(|output| (output.tag, output.element))
            .collect::<Vec<_>>()
    };
    result.set_item("detector_outputs", labels(evidence.detector_outputs))?;
    result.set_item("observable_outputs", labels(evidence.observable_outputs))?;
    result.set_item("probe_shots", evidence.probe_shots)?;
    result.set_item("correlation_shots", evidence.correlation_shots)?;
    result.set_item("schedule_shots", evidence.schedule_shots)?;
    let traces = serde_json::to_string(&evidence.named_result_traces)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    result.set_item(
        "named_result_traces",
        py.import("json")?.call_method1("loads", (traces,))?,
    )?;
    Ok(result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDemInferenceSettings>()?;
    m.add_class::<PySeleneRuntimeSpec>()?;
    m.add_function(wrap_pyfunction!(capture_dem_evidence, m)?)?;
    Ok(())
}
