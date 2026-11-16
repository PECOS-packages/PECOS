//! Immutable Python projection diagnostics and DEM input conversion.

use super::{PyDetectorErrorModel, decoder_build_error_to_py};
use pecos_decoder_core::graphlike_projection::ProjectionReport;
use pecos_decoder_core::window::StructuredDem;
use pecos_decoders::batch::{DecoderFactory, PreparedModel};
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Diagnostics from graphlike projection. Pickling is not supported.
#[pyclass(
    name = "ProjectionReport",
    module = "pecos_rslib.qec",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(super) struct PyProjectionReport {
    pub(super) inner: ProjectionReport,
}

impl From<ProjectionReport> for PyProjectionReport {
    fn from(inner: ProjectionReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyProjectionReport {
    #[getter]
    fn input_hyperedges(&self) -> usize {
        self.inner.input_hyperedges
    }

    #[getter]
    fn projected(&self) -> usize {
        self.inner.projected
    }

    #[getter]
    fn zero_probability_omitted(&self) -> Vec<usize> {
        self.inner.zero_probability_omitted.clone()
    }

    #[getter]
    fn dropped_indices(&self) -> Vec<usize> {
        self.inner.dropped_indices.clone()
    }

    #[getter]
    fn cross_sector_components(&self) -> usize {
        self.inner.cross_sector_components
    }

    #[getter]
    fn label_conflicts(&self) -> usize {
        self.inner.label_conflicts
    }

    #[getter]
    fn unlabelled_detectors(&self) -> usize {
        self.inner.unlabelled_detectors
    }

    #[getter]
    fn distinct_sectors(&self) -> usize {
        self.inner.distinct_sectors
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.extract::<PyRef<'_, Self>>() else {
            return Ok(py.NotImplemented());
        };
        Ok((self.inner == other.inner)
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

pub(super) enum DemInput {
    Text(String),
    Object {
        text: String,
        structured: StructuredDem,
    },
}

impl DemInput {
    pub(super) fn extract(dem: &Bound<'_, PyAny>) -> PyResult<Self> {
        if dem.is_instance_of::<PyString>() {
            return Ok(Self::Text(dem.extract()?));
        }
        if let Ok(dem) = dem.extract::<PyRef<'_, PyDetectorErrorModel>>() {
            // Both immutable calls render the same deterministic standard text.
            let text = dem.inner.to_string();
            let structured = dem
                .inner
                .to_structured_decoder_dem()
                .map_err(decoder_build_error_to_py)?;
            return Ok(Self::Object { text, structured });
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "dem must be a str or DetectorErrorModel",
        ))
    }

    pub(super) fn text(&self) -> &str {
        match self {
            Self::Text(text) | Self::Object { text, .. } => text,
        }
    }

    pub(super) fn prepare<F: DecoderFactory>(
        &self,
        spec: &F,
    ) -> Result<PreparedModel, F::BuildError> {
        match self {
            Self::Text(text) => spec.decode_model(text),
            Self::Object { structured, .. } => spec.decode_structured_model(structured),
        }
    }
}
