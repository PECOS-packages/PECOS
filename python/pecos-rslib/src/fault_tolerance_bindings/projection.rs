//! Immutable Python projection diagnostics and DEM input conversion.

use super::{PyDetectorErrorModel, RustDetectorErrorModel, decoder_build_error_to_py};
use pecos_decoder_core::DecoderError;
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

/// Parse exactly the text used for corpus identity, then overlay object-only labels.
pub(super) fn structured_object_model(
    text: &str,
    dem: &RustDetectorErrorModel,
) -> Result<StructuredDem, DecoderError> {
    let mut structured = StructuredDem::from_dem_str(text)?;
    structured.detector_fault_sectors = vec![None; structured.num_detectors];
    for detector in &dem.detectors {
        if let Some(sector) = &detector.fault_sector {
            let slot = structured
                .detector_fault_sectors
                .get_mut(detector.id as usize)
                .ok_or_else(|| {
                    DecoderError::InvalidConfiguration(format!(
                        "labelled detector D{} is outside the standard DEM text's detector count {}",
                        detector.id, structured.num_detectors
                    ))
                })?;
            *slot = Some(sector.clone());
        }
    }
    Ok(structured)
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
            let text = dem.inner.to_string();
            let structured =
                structured_object_model(&text, &dem.inner).map_err(decoder_build_error_to_py)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_qec::fault_tolerance::dem_builder::DetectorDef;

    // Contract: text owns dimensions, probabilities and mechanism order; objects add labels only.
    #[test]
    fn object_overlay_preserves_parsed_model() {
        let mut dem = RustDetectorErrorModel::new();
        let mut detector = DetectorDef::new(2);
        detector.fault_sector = Some("X".into());
        dem.detectors.push(detector);
        let text = "error(0.012346) D2 L0\nerror(0.023457) D0 D1\ndetector D3\n";
        let mut expected = StructuredDem::from_dem_str(text).unwrap();
        expected.detector_fault_sectors = vec![None, None, Some("X".into()), None];
        assert_eq!(structured_object_model(text, &dem).unwrap(), expected);
    }

    // Contract: labels cannot silently grow the dimensions certified by corpus identity.
    #[test]
    fn object_overlay_rejects_label_outside_text_dimensions() {
        let mut dem = RustDetectorErrorModel::new();
        let mut detector = DetectorDef::new(2);
        detector.fault_sector = Some("X".into());
        dem.detectors.push(detector);
        let error = structured_object_model("error(0.1) D0", &dem).unwrap_err();
        assert!(
            error.to_string().contains(
                "labelled detector D2 is outside the standard DEM text's detector count 1"
            )
        );
    }
}
