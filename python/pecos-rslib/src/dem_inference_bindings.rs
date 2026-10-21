// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Shared Rust DEM inference settings and algorithms.
use pecos_qec::dem_inference::{
    DemInferenceData, DemInferenceSettings, correlate_measurement_ids, infer_affine_columns,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pyfunction(name = "_infer_affine_columns")]
fn infer_affine_columns_py(
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
    validation_rows: usize,
) -> PyResult<Vec<(u8, Vec<usize>)>> {
    infer_affine_columns(&inputs, &outputs, validation_rows)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction(name = "_correlate_measurement_ids")]
fn correlate_measurement_ids_py(
    raw_rows: Vec<Vec<u8>>,
    physical_rows: Vec<Vec<u8>>,
    source_ids: Vec<usize>,
) -> PyResult<Vec<usize>> {
    correlate_measurement_ids(&raw_rows, &physical_rows, &source_ids)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDemInferenceSettings>()?;
    m.add_function(wrap_pyfunction!(infer_affine_columns_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlate_measurement_ids_py, m)?)?;
    Ok(())
}
