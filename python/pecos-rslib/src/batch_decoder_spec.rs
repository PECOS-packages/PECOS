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
//! Version 2 replaces the spec method with `_pecos_prepare_decoder(dem: str)`,
//! returning a prepared object with `_pecos_build_decoder() -> worker`. The call's
//! one DEM is prepared once per decode call and shared by every worker build,
//! including concurrent builds. A failed preparation is cached the same way, so
//! every build reports that one exception unchanged. The prepared object is
//! released after the decode call. Preparation may run on any worker thread and
//! blocks the other builds until it finishes, so it must not re-enter PECOS batch
//! decoding.
//!
//! Version 2 also replaces the worker method with `_pecos_decode_outcome(bytes)`,
//! returning `(words_or_none, fallback_used, syndrome_consistent_or_none, deltas)`.
//! None words mean an undecoded shot, never a zero correction. A fallback must
//! supply words; consistency can only be specified for a fallback. Deltas are
//! nonnegative, strictly increasing actual attempts. Version 1 stays strict.
//!
//! The bridge holds the GIL only around those calls. A provider that does native
//! work should release it inside them, or its workers run one at a time.

use crate::decoder_spec_bindings::PyDecoderSpec;
use pecos_decoder_core::{DecoderError, ObservableDecoder, obs_mask::ObsMask};
use pecos_decoders::batch::{DecoderFactory, PreparedModel};
use pecos_decoders::{DecodeModel, DecoderSpec, spec::ExecutionTraits};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyString};

pub(crate) enum BatchDecoderSpec {
    Builtin(DecoderSpec),
    Provider {
        spec: Py<PyAny>,
        traits: ExecutionTraits,
        version: u32,
        /// The one DEM of this decode call and the outcome of preparing it.
        prepared: PyOnceLock<(String, PyResult<Py<PyAny>>)>,
    },
}

pub(crate) enum DecoderBuildError {
    Decoder(DecoderError),
    Python(PyErr),
}

impl std::fmt::Display for DecoderBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decoder(error) => error.fmt(formatter),
            Self::Python(error) => error.fmt(formatter),
        }
    }
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
                "decoder must be a DecoderSpec, legacy decoder string, or a version-1 or version-2 decoder provider",
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
        if !matches!(version, Some(1 | 2)) {
            return Err(not_a_provider());
        }
        let version = version.expect("validated protocol version");
        let method = if version == 2 {
            "_pecos_prepare_decoder"
        } else {
            "_pecos_build_decoder"
        };
        let members = || -> PyResult<ExecutionTraits> {
            if !decoder.getattr(method)?.is_callable() {
                return Err(PyTypeError::new_err(format!("{method} must be callable")));
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
            let error = PyTypeError::new_err(format!(
                "version-{version} decoder providers require callable {method} and boolean history_dependent and wall_clock_dependent members",
            ));
            error.set_cause(py, Some(cause));
            error
        })?;
        Ok(Self::Provider {
            spec: decoder.clone().unbind(),
            traits,
            version,
            prepared: PyOnceLock::new(),
        })
    }
}

impl DecoderFactory for BatchDecoderSpec {
    type BuildError = DecoderBuildError;

    fn execution_traits(&self) -> ExecutionTraits {
        match self {
            Self::Builtin(spec) => spec.execution_traits(),
            Self::Provider { traits, .. } => *traits,
        }
    }
    fn native_batch_capable(&self) -> bool {
        match self {
            Self::Builtin(spec) => spec.native_batch_capable(),
            Self::Provider { .. } => false,
        }
    }
    fn decode_model(&self, dem: &str) -> Result<PreparedModel, Self::BuildError> {
        match self {
            Self::Builtin(spec) => spec.decode_model(dem).map_err(DecoderBuildError::Decoder),
            Self::Provider { .. } => Ok(PreparedModel {
                model: DecodeModel::SingleDem(dem.to_string()),
                projection: None,
            }),
        }
    }
    fn decode_structured_model(
        &self,
        dem: &pecos_decoder_core::window::StructuredDem,
    ) -> Result<PreparedModel, Self::BuildError> {
        match self {
            Self::Builtin(spec) => spec.decode_structured_model(dem).map_err(DecoderBuildError::Decoder),
            Self::Provider { .. } => Err(DecoderBuildError::Decoder(
                DecoderError::InvalidConfiguration(
                    "decoder providers require DEM text by protocol; DetectorErrorModel objects are not supported".into(),
                ),
            )),
        }
    }
    fn build(&self, model: &DecodeModel) -> Result<Box<dyn ObservableDecoder>, DecoderBuildError> {
        match self {
            Self::Builtin(spec) => {
                DecoderFactory::build(spec, model).map_err(DecoderBuildError::Decoder)
            }
            Self::Provider {
                spec,
                version,
                prepared,
                ..
            } => {
                let DecodeModel::SingleDem(dem) = model else {
                    return Err(DecoderBuildError::Decoder(
                        DecoderError::InvalidConfiguration(
                            "decoder providers require a single DEM".into(),
                        ),
                    ));
                };
                Python::attach(|py| -> PyResult<Box<dyn ObservableDecoder>> {
                    let worker = if *version == 2 {
                        // A failure is cached too, so concurrent builders report the
                        // one exception instead of each repeating the preparation.
                        let (prepared_dem, prepared) = prepared.get_or_init(py, || {
                            let prepared = spec
                                .bind(py)
                                .call_method1("_pecos_prepare_decoder", (dem,))
                                .map(Bound::unbind);
                            (dem.clone(), prepared)
                        });
                        if prepared_dem != dem {
                            return Err(PyRuntimeError::new_err(
                                "internal error: a decoder provider preparation was reused for a different DEM",
                            ));
                        }
                        let prepared = prepared.as_ref().map_err(|error| error.clone_ref(py))?;
                        prepared.bind(py).call_method0("_pecos_build_decoder")?
                    } else {
                        spec.bind(py).call_method1("_pecos_build_decoder", (dem,))?
                    };
                    let num_detectors = worker.getattr("num_detectors")?.extract::<usize>()?;
                    let num_observables = worker.getattr("num_observables")?.extract::<usize>()?;
                    let method = if *version == 2 {
                        "_pecos_decode_outcome"
                    } else {
                        "_pecos_decode_obs"
                    };
                    if !worker.getattr(method)?.is_callable() {
                        return Err(PyTypeError::new_err(format!(
                            "decoder provider {method} must be callable"
                        )));
                    }
                    Ok(Box::new(ProviderDecoder {
                        worker: worker.unbind(),
                        num_detectors,
                        num_observables,
                        version: *version,
                    }))
                })
                .map_err(DecoderBuildError::Python)
            }
        }
    }
}

struct ProviderDecoder {
    worker: Py<PyAny>,
    version: u32,
    num_detectors: usize,
    num_observables: usize,
}
impl ObservableDecoder for ProviderDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.num_detectors)
    }
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        if self.version == 2 {
            return self.decode_outcome(syndrome)?.prediction.ok_or_else(|| {
                DecoderError::DecodingFailed(
                    "no prediction; use decode_outcome() for reporting outcomes".into(),
                )
            });
        }
        let mask = Python::attach(|py| {
            let words = self
                .worker
                .bind(py)
                .call_method1("_pecos_decode_obs", (PyBytes::new(py, syndrome),))?
                .extract::<Vec<u64>>()?;
            Ok::<_, PyErr>(ObsMask::from_words(&words))
        })
        .map_err(|e| DecoderError::DecodingFailed(e.to_string()))?;
        self.validate_mask(&mask)?;
        Ok(mask)
    }
    fn decode_outcome(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::outcome::ObservableDecodeOutcome, DecoderError> {
        use pecos_decoder_core::outcome::ObservableDecodeOutcome;
        if self.version == 1 {
            return self
                .decode_obs(syndrome)
                .map(ObservableDecodeOutcome::decoded);
        }
        let (words, fallback_used, syndrome_consistent, deltas_tried) = Python::attach(|py| {
            self.worker
                .bind(py)
                .call_method1("_pecos_decode_outcome", (PyBytes::new(py, syndrome),))?
                .extract::<(Option<Vec<u64>>, bool, Option<bool>, Vec<f64>)>()
        })
        .map_err(|e| DecoderError::DecodingFailed(e.to_string()))?;
        if fallback_used && words.is_none() {
            return Err(DecoderError::DecodingFailed(
                "decoder provider fallback_used=True requires prediction words, got None".into(),
            ));
        }
        if let Some(consistent) = syndrome_consistent
            && !fallback_used
        {
            let consistent = if consistent { "True" } else { "False" };
            return Err(DecoderError::DecodingFailed(format!(
                "decoder provider syndrome_consistent={consistent} requires fallback_used=True, got False"
            )));
        }
        if let Some(delta) = deltas_tried
            .iter()
            .find(|delta| delta.is_nan() || **delta < 0.0)
        {
            return Err(DecoderError::DecodingFailed(format!(
                "decoder provider delta {delta:?} must be nonnegative and not NaN"
            )));
        }
        if let Some([previous, next]) = deltas_tried.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(DecoderError::DecodingFailed(format!(
                "decoder provider deltas must be strictly increasing, got {previous:?} then {next:?}"
            )));
        }
        let prediction = words.map(|words| ObsMask::from_words(&words));
        if let Some(mask) = &prediction {
            self.validate_mask(mask)?;
        }
        Ok(ObservableDecodeOutcome {
            prediction,
            fallback_used,
            syndrome_consistent,
            deltas_tried,
        })
    }
}
impl ProviderDecoder {
    fn validate_mask(&self, mask: &ObsMask) -> Result<(), DecoderError> {
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // contract: structured input is rejected before any Python provider build call.
    #[test]
    fn provider_refuses_structured_preparation() {
        Python::initialize();
        Python::attach(|py| {
            let spec = BatchDecoderSpec::Provider {
                spec: py.None(),
                traits: ExecutionTraits::default(),
                version: 1,
                prepared: PyOnceLock::new(),
            };
            let dem =
                pecos_decoder_core::window::StructuredDem::from_dem_str("error(0.1) D0").unwrap();
            let Err(error) = spec.decode_structured_model(&dem) else {
                panic!("provider accepted structured input");
            };
            assert!(
                error
                    .to_string()
                    .contains("providers require DEM text by protocol")
            );
        });
    }
}
