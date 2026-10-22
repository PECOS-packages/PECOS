// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Value-only boundary for reusable Guppy DEM preparation.

use crate::dag_circuit_bindings::{PyTickCircuit, circuit_definitions_to_py};
use crate::dem_inference_bindings::{PyDemInferenceSettings, PySeleneRuntimeSpec};
use crate::engine_builders::PyQis;
use crate::fault_tolerance_bindings::PyDetectorErrorModel;
use crate::observable_flips_bindings::obsmask_to_py;
use pecos_qec::dem_build::{MeasurementRef, ResultRef};
use pecos_qec::{
    DemBuild, DemDefinition, DemDefinitionSource, DemOutputSelector, PreparedDem, SeleneDemProgram,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(FromPyObject)]
enum ReferenceValue {
    Record(usize),
    Result((String, usize, Option<usize>)),
}

/// ID, direct IDs, typed references, inferred selector, coordinates, metadata JSON.
/// Exactly one of the three source slots must be present.
type DefinitionValue = (
    Option<usize>,
    Option<Vec<usize>>,
    Option<Vec<ReferenceValue>>,
    Option<(String, usize)>,
    Option<Vec<f64>>,
    String,
);
type GeneratorCertificate = (Vec<u8>, Vec<(String, usize)>, String);
type Evaluation = (Py<PyList>, Py<PyAny>);

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn definitions(values: Vec<DefinitionValue>) -> PyResult<Vec<DemDefinition>> {
    values
        .into_iter()
        .map(|(id, ids, refs, inferred, coords, metadata)| {
            let source = match (ids, refs, inferred) {
                (Some(ids), None, None) => DemDefinitionSource::Measurements(ids),
                (None, Some(refs), None) => {
                    let refs = refs
                        .into_iter()
                        .map(|reference| match reference {
                            ReferenceValue::Record(offset) => MeasurementRef::record(offset),
                            ReferenceValue::Result((tag, occurrence, element)) => {
                                ResultRef::new(tag, occurrence, element).map(MeasurementRef::Result)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(value_error)?;
                    DemDefinitionSource::references(refs).map_err(value_error)?
                }
                (None, None, Some((tag, element))) => {
                    DemDefinitionSource::inferred(tag, element).map_err(value_error)?
                }
                _ => return Err(value_error("definition requires exactly one source")),
            };
            Ok(DemDefinition {
                id,
                source,
                coords,
                metadata: serde_json::from_str(&metadata).map_err(value_error)?,
            })
        })
        .collect()
}

fn json_value(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("json")?
        .call_method1("loads", (value.to_string(),))?
        .unbind())
}

fn traces_json(prepared: &PreparedDem, bound: bool) -> Value {
    let raw_tag = prepared
        .inference_settings()
        .map(|settings| settings.raw_tag.as_str());
    let mut bound_traces = prepared
        .result_traces()
        .iter()
        .filter(|trace| Some(trace.name.as_str()) == raw_tag);
    let traces = prepared
        .source_result_traces()
        .iter()
        .enumerate()
        .filter_map(|(index, trace)| {
            if bound && Some(trace.name.as_str()) != raw_tag {
                return None;
            }
            let provenance = if bound {
                bound_traces
                    .next()
                    .expect("certified raw traces preserve source call order")
            } else {
                trace
            };
            let ids = provenance.measurement_ids.as_deref().unwrap_or_default();
            let mut value = json!({"name": trace.name, "width": trace.width, "result_ids": ids,
            "values": prepared.result_trace_values().get(index)});
            if let Some(occurrence) = trace.occurrence {
                value["occurrence"] = json!(occurrence);
            }
            Some(value)
        })
        .collect::<Vec<_>>();
    json!(traces)
}

#[pyfunction(name = "_prepare_guppy_dem")]
#[pyo3(signature = (program, num_qubits, seed, runtime, *, settings=None, detectors=None, observables=None, detectors_json=None, observables_json=None, num_measurements=None, frontend_static_schedule=None, trusted_generator_layout=None, require_hosted_operation_order=false, max_hosted_tick_separation=None, hosted_context="GuppyDemBuilder.build", idle_after_2q=None, strip_traced_idles=None, simplify_single_qubit_clifford_chains=false))]
fn prepare_guppy_dem(
    py: Python<'_>,
    program: &PyQis,
    num_qubits: usize,
    seed: u64,
    runtime: &PySeleneRuntimeSpec,
    settings: Option<&PyDemInferenceSettings>,
    detectors: Option<Vec<DefinitionValue>>,
    observables: Option<Vec<DefinitionValue>>,
    detectors_json: Option<String>,
    observables_json: Option<String>,
    num_measurements: Option<usize>,
    frontend_static_schedule: Option<String>,
    trusted_generator_layout: Option<GeneratorCertificate>,
    require_hosted_operation_order: bool,
    max_hosted_tick_separation: Option<u64>,
    hosted_context: &str,
    idle_after_2q: Option<f64>,
    strip_traced_idles: Option<bool>,
    simplify_single_qubit_clifford_chains: bool,
) -> PyResult<PyPreparedDem> {
    let runtime = runtime
        .runtime
        .clone()
        .ok_or_else(|| value_error("Selene runtime spec is not configured"))?;
    let mut program = if let Some((bytes, layout, digest)) = trusted_generator_layout {
        SeleneDemProgram::from_frontend_compiled_hugr(program.inner.clone(), bytes)
            .with_trusted_generator_layout(layout, digest)
    } else {
        SeleneDemProgram::new(program.inner.clone())
    }
    .with_runtime(runtime)
    .with_schedule_validation_shots(0)
    .with_require_hosted_operation_order(require_hosted_operation_order)
    .with_max_hosted_tick_separation(max_hosted_tick_separation)
    .with_hosted_validation_context(hosted_context)
    .with_simplify_single_qubit_clifford_chains(simplify_single_qubit_clifford_chains);
    if let Some(source) = frontend_static_schedule {
        program = program.with_frontend_static_schedule(source);
    }
    let settings = settings.map(|settings| settings.inner.clone());
    let detectors = detectors.map(definitions).transpose()?;
    let observables = observables.map(definitions).transpose()?;
    py.detach(move || {
        let mut builder = pecos_qec::DetectorErrorModel::builder()
            .with_program(program)
            .with_qubits(num_qubits)
            .with_seed(seed);
        if let Some(settings) = settings {
            builder = builder.with_inference(settings);
        }
        if let Some(values) = detectors {
            builder = builder.with_detector_definitions(values);
        }
        if let Some(values) = observables {
            builder = builder.with_observable_definitions(values);
        }
        if let Some(json) = detectors_json {
            builder = builder.with_detectors_json(json);
        }
        if let Some(json) = observables_json {
            builder = builder.with_observables_json(json);
        }
        if let Some(count) = num_measurements {
            builder = builder.with_num_measurements(count);
        }
        if let Some(duration) = idle_after_2q {
            builder = builder.with_idle_after_2q(duration);
        }
        if let Some(strip) = strip_traced_idles {
            builder = builder.with_strip_traced_idles(strip);
        }
        builder.prepare()
    })
    .map(|inner| PyPreparedDem { inner })
    .map_err(value_error)
}

#[pyclass(name = "_PreparedDem", module = "pecos_rslib.qec", frozen)]
pub(crate) struct PyPreparedDem {
    inner: PreparedDem,
}

crate::dem_noise_bindings::dem_noise_methods! {
    PyPreparedDem,
    /// Build with noise while retaining the prepared schema.
    build_dem, [&self], [], noise, PyDemBuild, {
        self.inner.build_dem(noise).map(|inner| PyDemBuild { inner }).map_err(value_error)
    }, {
        #[getter]
        fn circuit(&self) -> PyTickCircuit {
            PyTickCircuit {
                inner: self.inner.circuit().clone(),
            }
        }

        #[getter]
        fn detectors_json(&self) -> Option<&str> {
            match self.inner.circuit().get_meta("detectors") {
                Some(pecos_quantum::Attribute::String(value)) => Some(value.as_str()),
                _ => None,
            }
        }

        #[getter]
        fn observables_json(&self) -> Option<&str> {
            match self.inner.circuit().get_meta("observables") {
                Some(pecos_quantum::Attribute::String(value)) => Some(value.as_str()),
                _ => None,
            }
        }

        #[getter]
        fn detectors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            definition_values(py, self.inner.definitions(), "detectors")
        }

        #[getter]
        fn observables(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            definition_values(py, self.inner.definitions(), "observables")
        }

        #[getter]
        fn measurement_ids(&self) -> Vec<usize> {
            self.inner.measurement_ids().to_vec()
        }

        #[getter]
        fn measurement_ledger(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            json_value(
                py,
                &json!(
                    self.inner
                        .measurement_ledger()
                        .iter()
                        .map(pecos_qec::dem_build::bindings::MeasurementLedgerEntry::to_json)
                        .collect::<Vec<_>>()
                ),
            )
        }

        #[getter]
        fn named_result_measurements(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
            let result = PyDict::new(py);
            for (tag, ids) in self.inner.named_result_measurements() {
                result.set_item(tag, PyTuple::new(py, ids)?)?;
            }
            Ok(result.unbind())
        }

        #[getter]
        fn schema_fingerprint(&self) -> &str {
            self.inner.schema_fingerprint()
        }

        #[getter]
        fn definition_origins(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            json_value(py, &self.inner.audit().to_json()["definition_origins"])
        }

        #[getter]
        fn audit(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let mut audit = self.inner.audit().to_json();
            audit["idle_noise_residuals"] = json!([]);
            json_value(py, &audit)
        }

        #[getter]
        fn inference(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let Some(report) = self.inner.inference() else {
                return Ok(py.None());
            };
            let labels = |outputs: Option<&[DemOutputSelector]>| {
                outputs.map(|outputs| {
                    outputs
                        .iter()
                        .map(|o| (o.tag.clone(), o.element))
                        .collect::<Vec<_>>()
                })
            };
            let mut data = self
                .inner
                .audit()
                .inference
                .expect("inference report has audit")
                .to_json();
            data["detector_supports"] = json!(report.detector_supports);
            data["observable_supports"] = json!(report.observable_supports);
            data["raw_tag"] = json!(self.inner.inference_settings().map(|s| &s.raw_tag));
            data["result_traces"] = traces_json(&self.inner, true);
            data["source_result_traces"] = traces_json(&self.inner, false);
            let result = json_value(py, &data)?;
            let dict = result.bind(py).cast::<pyo3::types::PyDict>()?;
            dict.set_item("detector_labels", labels(self.inner.detector_outputs()))?;
            dict.set_item("observable_labels", labels(self.inner.observable_outputs()))?;
            Ok(result)
        }

        fn evaluate_runtime_record(&self, py: Python<'_>, values: Vec<u8>) -> PyResult<Evaluation> {
            evaluation(py, self.inner.evaluate_runtime_record(&values))
        }
        fn runtime_record_parities(&self, py: Python<'_>, values: Vec<u8>) -> PyResult<Evaluation> {
            evaluation(py, self.inner.runtime_record_parities(&values))
        }
        fn evaluate_measurements(
            &self,
            py: Python<'_>,
            values: BTreeMap<usize, u8>,
        ) -> PyResult<Evaluation> {
            evaluation(py, self.inner.evaluate_measurements(&values))
        }
        fn measurement_parities(
            &self,
            py: Python<'_>,
            values: BTreeMap<usize, u8>,
        ) -> PyResult<Evaluation> {
            evaluation(py, self.inner.measurement_parities(&values))
        }
        fn evaluate_results(
            &self,
            py: Python<'_>,
            values: BTreeMap<String, Vec<u8>>,
        ) -> PyResult<Evaluation> {
            evaluation(py, self.inner.evaluate_results(&values))
        }
        fn result_parities(
            &self,
            py: Python<'_>,
            values: BTreeMap<String, Vec<u8>>,
        ) -> PyResult<Evaluation> {
            evaluation(py, self.inner.result_parities(&values))
        }
        fn evaluate_result_columns(
            &self,
            py: Python<'_>,
            columns: BTreeMap<String, Vec<Vec<u8>>>,
        ) -> PyResult<Vec<Evaluation>> {
            self.inner
                .evaluate_result_columns(&columns)
                .map_err(value_error)?
                .into_iter()
                .map(|row| evaluation(py, Ok(row)))
                .collect()
        }
        fn result_column_parities(
            &self,
            py: Python<'_>,
            columns: BTreeMap<String, Vec<Vec<u8>>>,
        ) -> PyResult<Vec<Evaluation>> {
            self.inner
                .result_column_parities(&columns)
                .map_err(value_error)?
                .into_iter()
                .map(|row| evaluation(py, Ok(row)))
                .collect()
        }
    }
}

fn definition_values(
    py: Python<'_>,
    definitions: &pecos_qec::fault_tolerance::circuit_definitions::CircuitDefinitions,
    role: &str,
) -> PyResult<Py<PyAny>> {
    Ok(circuit_definitions_to_py(py, definitions)?
        .bind(py)
        .get_item(role)?
        .expect("circuit definitions contain detectors and observables")
        .unbind())
}

fn evaluation(
    py: Python<'_>,
    result: Result<(Vec<u8>, pecos_decoder_core::obs_mask::ObsMask), pecos_qec::DemBuildError>,
) -> PyResult<Evaluation> {
    let (detectors, observables) = result.map_err(value_error)?;
    Ok((
        PyList::new(py, detectors)?.unbind(),
        obsmask_to_py(py, &observables)?,
    ))
}

#[pyclass(name = "_DemBuild", module = "pecos_rslib.qec", frozen)]
struct PyDemBuild {
    inner: DemBuild,
}

#[pymethods]
impl PyDemBuild {
    #[getter]
    fn dem(&self) -> PyDetectorErrorModel {
        PyDetectorErrorModel {
            inner: self.inner.dem().clone(),
        }
    }
    #[getter]
    fn prepared(&self) -> PyPreparedDem {
        PyPreparedDem {
            inner: self.inner.prepared().clone(),
        }
    }
    #[getter]
    fn idle_noise_residuals(&self, py: Python<'_>) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.dem().idle_noise_residuals(py)
    }
    fn residual_summary(&self, py: Python<'_>, relative_threshold: f64) -> PyResult<Py<PyAny>> {
        let summary = self
            .inner
            .residual_summary(relative_threshold)
            .map_err(value_error)?;
        json_value(
            py,
            &json!(
                summary
                    .iter()
                    .map(|group| json!({
                        "channel_kind": group.channel_kind,
                        "count": group.count,
                        "max_relative_magnitude": group.max_relative_magnitude,
                        "max_tv_magnitude": group.max_tv_magnitude,
                    }))
                    .collect::<Vec<_>>()
            ),
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedDem>()?;
    m.add_class::<PyDemBuild>()?;
    m.add_function(wrap_pyfunction!(prepare_guppy_dem, m)?)?;
    Ok(())
}
