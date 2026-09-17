//! Versioned Python decoder-provider bridge. The standard extension never imports
//! optional providers. Providers own their configuration and native workers;
//! only Python objects and integer word arrays cross extension boundaries.

use crate::decoder_spec_bindings::PyDecoderSpec;
use pecos_decoder_core::{DecoderError, ObservableDecoder, obs_mask::ObsMask};
use pecos_decoders::{DecodeModel, DecoderSpec, spec::ExecutionTraits};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

pub(crate) enum BatchDecoderSpec {
    Builtin(DecoderSpec),
    Provider {
        spec: Py<PyAny>,
        traits: ExecutionTraits,
    },
}

impl BatchDecoderSpec {
    pub(crate) fn extract(decoder: &Bound<'_, PyAny>) -> PyResult<Self> {
        if decoder.is_instance_of::<PyString>() {
            return DecoderSpec::parse(decoder.extract::<&str>()?)
                .map(Self::Builtin)
                .map_err(crate::fault_tolerance_bindings::decoder_parse_error_to_py);
        }
        if let Ok(spec) = decoder.extract::<PyRef<'_, PyDecoderSpec>>() {
            return Ok(Self::Builtin(spec.inner.clone()));
        }
        let version = decoder
            .getattr("_pecos_decoder_api_version")
            .and_then(|v| v.extract::<u32>());
        if !matches!(version, Ok(1)) {
            return Err(PyTypeError::new_err(
                "decoder must be a DecoderSpec, legacy decoder string, or a version-1 decoder provider",
            ));
        }
        if !decoder.getattr("_pecos_build_decoder")?.is_callable() {
            return Err(PyTypeError::new_err(
                "decoder provider _pecos_build_decoder must be callable",
            ));
        }
        Ok(Self::Provider {
            spec: decoder.clone().unbind(),
            traits: ExecutionTraits {
                history_dependent: decoder.getattr("history_dependent")?.extract()?,
                wall_clock_dependent: decoder.getattr("wall_clock_dependent")?.extract()?,
            },
        })
    }

    pub(crate) fn execution_traits(&self) -> ExecutionTraits {
        match self {
            Self::Builtin(spec) => spec.execution_traits(),
            Self::Provider { traits, .. } => *traits,
        }
    }
    pub(crate) fn native_batch_capable(&self) -> bool {
        match self {
            Self::Builtin(spec) => spec.native_batch_capable(),
            Self::Provider { .. } => false,
        }
    }
    pub(crate) fn embedded_hybrid_full_dem(&self) -> Option<&str> {
        match self {
            Self::Builtin(spec) => spec.embedded_hybrid_full_dem(),
            Self::Provider { .. } => None,
        }
    }
    pub(crate) fn build(
        &self,
        model: &DecodeModel,
    ) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
        match self {
            Self::Builtin(spec) => spec.build(model),
            Self::Provider { spec, .. } => {
                let dem = match model {
                    DecodeModel::SingleDem(text) => text.clone(),
                    DecodeModel::StructuredDem(model) => model.to_dem_string(),
                    DecodeModel::HybridDem { .. } => {
                        return Err(DecoderError::InvalidConfiguration(
                            "decoder providers require a single DEM".into(),
                        ));
                    }
                };
                Python::attach(|py| -> PyResult<Box<dyn ObservableDecoder>> {
                    let worker = spec.bind(py).call_method1("_pecos_build_decoder", (dem,))?;
                    let num_detectors = worker.getattr("num_detectors")?.extract::<usize>()?;
                    if !worker.getattr("_pecos_decode_obs")?.is_callable() {
                        return Err(PyTypeError::new_err(
                            "decoder provider _pecos_decode_obs must be callable",
                        ));
                    }
                    Ok(Box::new(ProviderDecoder {
                        worker: worker.unbind(),
                        num_detectors,
                    }))
                })
                .map_err(|e| DecoderError::InvalidConfiguration(e.to_string()))
            }
        }
    }
}

struct ProviderDecoder {
    worker: Py<PyAny>,
    num_detectors: usize,
}
impl ObservableDecoder for ProviderDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.num_detectors)
    }
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Python::attach(|py| {
            let words = self
                .worker
                .bind(py)
                .call_method1("_pecos_decode_obs", (PyBytes::new(py, syndrome),))?
                .extract::<Vec<u64>>()?;
            Ok::<_, PyErr>(ObsMask::from_words(&words))
        })
        .map_err(|e| DecoderError::DecodingFailed(e.to_string()))
    }
}
