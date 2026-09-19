//! Versioned Python decoder-provider bridge. The standard extension never imports
//! optional providers. Providers own their configuration and native workers;
//! only Python objects and integer word arrays cross extension boundaries.
//!
//! This is an internal protocol between PECOS packages, not a public API. A
//! version-1 provider is any Python object with:
//!
//! - `_pecos_decoder_api_version == 1`. Any other value is rejected, so a change
//!   to this contract takes a new version number.
//! - `history_dependent: bool` and `wall_clock_dependent: bool`, read once when the
//!   provider is passed to `decode`. Either one keeps automatic planning
//!   sequential. An explicit parallel worker count is rejected for a
//!   history-dependent provider and runs with a reproducibility warning for a
//!   wall-clock-dependent one, exactly as for built-in specifications.
//! - `_pecos_build_decoder(dem: str)`, called once per decoder the batch planner
//!   needs, possibly from several threads at once, and returning a worker object.
//!   An exception the provider raises reaches the caller of `decode` unchanged.
//!
//! A worker object has:
//!
//! - `num_detectors: int` and `num_observables: int`, the dimensions of the model
//!   the worker was built from. Detectors are checked against the batch before any
//!   shot is decoded.
//! - `_pecos_decode_obs(syndrome: bytes) -> Sequence[int]`. `syndrome` holds one
//!   byte per detector, each 0 or 1. The result is the predicted observable mask as
//!   little-endian 64-bit words, lowest observables first, no more words than
//!   `num_observables` needs and no observable at or above it. A worker is only
//!   ever called from the one thread that built it, one shot at a time. An
//!   exception it raises is reported as a decode failure naming the shot.
//!
//! The bridge holds the GIL only around those calls. A provider that does native
//! work should release it inside them, or its workers run one at a time.

use crate::decoder_spec_bindings::PyDecoderSpec;
use pecos_decoder_core::{DecoderError, ObservableDecoder, obs_mask::ObsMask};
use pecos_decoders::{DecodeModel, DecoderSpec, spec::ExecutionTraits};
use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

pub(crate) enum BatchDecoderSpec {
    Builtin(DecoderSpec),
    Provider {
        spec: Py<PyAny>,
        traits: ExecutionTraits,
    },
}

pub(crate) enum DecoderBuildError {
    Decoder(DecoderError),
    Python(PyErr),
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
        let py = decoder.py();
        let not_a_provider = || {
            PyTypeError::new_err(
                "decoder must be a DecoderSpec, legacy decoder string, or a version-1 decoder provider",
            )
        };
        // An object without the version member is simply not a provider; any
        // other failure while reading it belongs to the caller.
        let version = match decoder.getattr("_pecos_decoder_api_version") {
            Ok(version) => version.extract::<u32>().ok(),
            Err(error)
                if error.is_instance_of::<PyAttributeError>(py)
                    || error.is_instance_of::<PyTypeError>(py) =>
            {
                None
            }
            Err(error) => return Err(error),
        };
        if version != Some(1) {
            return Err(not_a_provider());
        }
        let members = || -> PyResult<ExecutionTraits> {
            if !decoder.getattr("_pecos_build_decoder")?.is_callable() {
                return Err(PyTypeError::new_err(
                    "_pecos_build_decoder must be callable",
                ));
            }
            Ok(ExecutionTraits {
                history_dependent: decoder.getattr("history_dependent")?.extract()?,
                wall_clock_dependent: decoder.getattr("wall_clock_dependent")?.extract()?,
            })
        };
        let traits = members().map_err(|cause| {
            // Only a missing or wrong-typed member is a protocol error; anything
            // else a provider's own attribute access raises belongs to the caller.
            if !(cause.is_instance_of::<PyAttributeError>(py)
                || cause.is_instance_of::<PyTypeError>(py))
            {
                return cause;
            }
            let error = PyTypeError::new_err(
                "version-1 decoder providers require callable _pecos_build_decoder and boolean history_dependent and wall_clock_dependent members",
            );
            error.set_cause(py, Some(cause));
            error
        })?;
        Ok(Self::Provider {
            spec: decoder.clone().unbind(),
            traits,
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
    ) -> Result<Box<dyn ObservableDecoder>, DecoderBuildError> {
        match self {
            Self::Builtin(spec) => spec.build(model).map_err(DecoderBuildError::Decoder),
            Self::Provider { spec, .. } => {
                let DecodeModel::SingleDem(dem) = model else {
                    return Err(DecoderBuildError::Decoder(
                        DecoderError::InvalidConfiguration(
                            "decoder providers require a single DEM".into(),
                        ),
                    ));
                };
                Python::attach(|py| -> PyResult<Box<dyn ObservableDecoder>> {
                    let worker = spec.bind(py).call_method1("_pecos_build_decoder", (dem,))?;
                    let num_detectors = worker.getattr("num_detectors")?.extract::<usize>()?;
                    let num_observables = worker.getattr("num_observables")?.extract::<usize>()?;
                    if !worker.getattr("_pecos_decode_obs")?.is_callable() {
                        return Err(PyTypeError::new_err(
                            "decoder provider _pecos_decode_obs must be callable",
                        ));
                    }
                    Ok(Box::new(ProviderDecoder {
                        worker: worker.unbind(),
                        num_detectors,
                        num_observables,
                    }))
                })
                .map_err(DecoderBuildError::Python)
            }
        }
    }
}

struct ProviderDecoder {
    worker: Py<PyAny>,
    num_detectors: usize,
    num_observables: usize,
}
impl ObservableDecoder for ProviderDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.num_detectors)
    }
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        let mask = Python::attach(|py| {
            let words = self
                .worker
                .bind(py)
                .call_method1("_pecos_decode_obs", (PyBytes::new(py, syndrome),))?
                .extract::<Vec<u64>>()?;
            Ok::<_, PyErr>(ObsMask::from_words(&words))
        })
        .map_err(|e| DecoderError::DecodingFailed(e.to_string()))?;
        // The provider is outside this extension, so its answer is checked
        // against the model it reports rather than trusted.
        if mask.words().len() > self.num_observables.div_ceil(64) {
            return Err(DecoderError::DecodingFailed(format!(
                "decoder provider returned {} observable words, but {} observables need at most {}",
                mask.words().len(),
                self.num_observables,
                self.num_observables.div_ceil(64)
            )));
        }
        if let Some(observable) = mask
            .iter_set_bits()
            .find(|&bit| bit >= self.num_observables)
        {
            return Err(DecoderError::DecodingFailed(format!(
                "decoder provider predicted observable {observable}, but its model has {} observables",
                self.num_observables
            )));
        }
        Ok(mask)
    }
}
