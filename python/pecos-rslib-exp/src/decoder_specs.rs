//! Optional experimental decoder factories and native workers for batch decoding.
use crate::bp_trellis_bindings::{TrellisOrderArgument, parse_ordering};
use crate::frontier_bindings::{ColumnOrderArgument, parse_column_order, parse_metric_mode};
use pecos_bp_trellis::BpTrellisConfig;
use pecos_decoder_core::{DecoderError, ObservableDecoder};
use pecos_frontier::{FrontierConfig, SparseDem, TrellisOrdering};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::Mutex;

pub(crate) fn decoder_error_to_py(error: &DecoderError) -> PyErr {
    decoder_error_with_message(error, error.to_string())
}

pub(crate) fn indexed_decoder_error_to_py(shot_index: usize, error: &DecoderError) -> PyErr {
    decoder_error_with_message(error, format!("shot {shot_index}: {error}"))
}

fn decoder_error_with_message(error: &DecoderError, message: String) -> PyErr {
    match error {
        DecoderError::InvalidDemSyntax(_) => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ExperimentalSpec {
    Frontier(FrontierConfig, TrellisOrdering),
    BpTrellis(BpTrellisConfig),
}

#[pyclass(
    name = "ExperimentalDecoderSpec",
    module = "pecos_rslib_exp",
    frozen,
    from_py_object
)]
#[derive(Clone)]
struct PyExperimentalDecoderSpec {
    inner: ExperimentalSpec,
}
impl PyExperimentalDecoderSpec {
    fn new(inner: ExperimentalSpec) -> Self {
        Self { inner }
    }
}
#[pymethods]
impl PyExperimentalDecoderSpec {
    #[getter]
    fn family(&self) -> &'static str {
        match self.inner {
            ExperimentalSpec::Frontier(..) => "frontier",
            ExperimentalSpec::BpTrellis(_) => "bp_trellis",
        }
    }
    #[getter]
    fn history_dependent(&self) -> bool {
        false
    }
    #[getter]
    fn wall_clock_dependent(&self) -> bool {
        false
    }
    #[getter]
    fn _pecos_decoder_api_version(&self) -> u32 {
        1
    }
    fn __repr__(&self) -> String {
        match &self.inner {
            ExperimentalSpec::Frontier(c, ordering) => frontier_repr(c, ordering),
            ExperimentalSpec::BpTrellis(c) => bp_trellis_repr(c),
        }
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
    // Hashing only the family ensures equal specs have equal hashes, including
    // float options such as -0.0 and 0.0. Collisions within a family use __eq__,
    // which is sufficient for the small collections of decoder specs.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        self.family().hash(&mut hasher);
        hasher.finish()
    }
    /// Internal batch protocol: construct an independent native worker.
    fn _pecos_build_decoder(&self, py: Python<'_>, dem: &str) -> PyResult<PyExperimentalWorker> {
        py.detach(|| {
            // The engine and the reported dimensions come from the same parse.
            let dem = SparseDem::from_dem_str(dem)?;
            let inner: Box<dyn ObservableDecoder + Send> = match &self.inner {
                ExperimentalSpec::Frontier(c, ordering) => {
                    let mut config = c.clone();
                    config.column_order = ordering.resolve(&dem)?;
                    Box::new(pecos_frontier::FrontierDecoder::from_sparse_dem(
                        &dem, config,
                    )?)
                }
                ExperimentalSpec::BpTrellis(c) => Box::new(
                    pecos_bp_trellis::BpTrellisDecoder::from_sparse_dem(&dem, c.clone())?,
                ),
            };
            Ok::<_, DecoderError>(PyExperimentalWorker {
                inner: Mutex::new(inner),
                num_detectors: dem.num_detectors,
                num_observables: dem.num_observables,
            })
        })
        .map_err(|error| decoder_error_to_py(&error))
    }
}

#[pyclass(name = "ExperimentalDecoderWorker", module = "pecos_rslib_exp")]
struct PyExperimentalWorker {
    inner: Mutex<Box<dyn ObservableDecoder + Send>>,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}
#[pymethods]
impl PyExperimentalWorker {
    /// Decode without the GIL and return little-endian observable words.
    fn _pecos_decode_obs(&self, py: Python<'_>, syndrome: Vec<u8>) -> PyResult<Vec<u64>> {
        py.detach(|| {
            let mut decoder = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder lock poisoned"))?;
            decoder
                .decode_obs(&syndrome)
                .map(|mask| mask.words().to_vec())
                .map_err(|error| decoder_error_to_py(&error))
        })
    }
}

fn finish_repr(family: &str, args: Vec<String>) -> String {
    format!("{family}({})", args.join(", "))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExperimentalDecoderSpec>()?;
    module.add_class::<PyExperimentalWorker>()?;
    module.add_function(wrap_pyfunction!(frontier, module)?)?;
    module.add_function(wrap_pyfunction!(bp_trellis, module)?)?;
    Ok(())
}

/// Native Rust Frontier decoder for raw DEMs, including hyperedges.
/// Batch decoding supports independent Rust workers. Pruning can make predictions
/// approximate; use pecos_rslib_exp.FrontierDecoder for per-shot confidence data.
#[pyfunction]
#[pyo3(signature = (*, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order=ColumnOrderArgument::default(), merge_indistinguishable=false, metric_mode="logsumexp_float", int_metric_scale=1024),
    text_signature = "(*, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order='deadline_reorder', merge_indistinguishable=False, metric_mode='logsumexp_float', int_metric_scale=1024)")]
fn frontier(
    k: usize,
    delta: f64,
    score_alpha: f64,
    bp_score_iterations: usize,
    column_order: ColumnOrderArgument,
    merge_indistinguishable: bool,
    metric_mode: &str,
    int_metric_scale: i32,
) -> PyResult<PyExperimentalDecoderSpec> {
    let config = FrontierConfig {
        k,
        delta,
        score_alpha,
        column_order: None,
        merge_indistinguishable,
        bp_score_iterations,
        metric_mode: parse_metric_mode(metric_mode)?,
        int_metric_scale,
    };
    config
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyExperimentalDecoderSpec::new(ExperimentalSpec::Frontier(
        config,
        parse_column_order(column_order)?,
    )))
}

fn frontier_repr(config: &FrontierConfig, ordering: &TrellisOrdering) -> String {
    let default = FrontierConfig::default();
    let mut args = Vec::new();
    if config.k != default.k {
        args.push(format!("k={}", config.k));
    }
    if config.delta.to_bits() != default.delta.to_bits() {
        args.push(if config.delta.is_infinite() {
            "delta=float('inf')".to_owned()
        } else {
            format!("delta={:?}", config.delta)
        });
    }
    if config.score_alpha.to_bits() != default.score_alpha.to_bits() {
        args.push(format!("score_alpha={:?}", config.score_alpha));
    }
    if config.bp_score_iterations != default.bp_score_iterations {
        args.push(format!(
            "bp_score_iterations={}",
            config.bp_score_iterations
        ));
    }
    match ordering {
        TrellisOrdering::Deadline => {}
        TrellisOrdering::TimeOrder => args.push("column_order='time_order'".to_owned()),
        TrellisOrdering::BackwardDeadline => {
            args.push("column_order='backward_deadline_reorder'".to_owned());
        }
        TrellisOrdering::Explicit(order) => args.push(format!("column_order={order:?}")),
    }
    if config.merge_indistinguishable != default.merge_indistinguishable {
        args.push("merge_indistinguishable=True".to_owned());
    }
    if config.metric_mode != default.metric_mode {
        args.push("metric_mode='maxlog_int'".to_owned());
    }
    if config.int_metric_scale != default.int_metric_scale {
        args.push(format!("int_metric_scale={}", config.int_metric_scale));
    }
    finish_repr("frontier", args)
}

/// Native Rust BP-guided trellis decoder for raw DEMs, including hyperedges.
/// Batch decoding supports independent Rust workers. Each worker prebuilds the
/// optional escalation ladder, retried only after a no-path result. Use
/// pecos_rslib_exp.BpTrellisDecoder for per-shot confidence and retry telemetry.
#[pyfunction]
#[pyo3(signature = (*, k=8, delta=100.0, score_alpha=0.8, bp_score_iterations=5, merge_indistinguishable=true, ordering=TrellisOrderArgument::default(), escalation_ks=None),
    text_signature = "(*, k=8, delta=100.0, score_alpha=0.8, bp_score_iterations=5, merge_indistinguishable=True, ordering='deadline', escalation_ks=None)")]
fn bp_trellis(
    k: usize,
    delta: f64,
    score_alpha: f64,
    bp_score_iterations: usize,
    merge_indistinguishable: bool,
    ordering: TrellisOrderArgument,
    escalation_ks: Option<Vec<usize>>,
) -> PyResult<PyExperimentalDecoderSpec> {
    let config = BpTrellisConfig {
        k,
        delta,
        score_alpha,
        bp_score_iterations,
        merge_indistinguishable,
        ordering: parse_ordering(ordering)?,
        escalation_ks: escalation_ks.unwrap_or_default(),
    };
    config
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyExperimentalDecoderSpec::new(ExperimentalSpec::BpTrellis(
        config,
    )))
}

fn bp_trellis_repr(config: &BpTrellisConfig) -> String {
    let default = BpTrellisConfig::default();
    let mut args = Vec::new();
    if config.k != default.k {
        args.push(format!("k={}", config.k));
    }
    if config.delta.to_bits() != default.delta.to_bits() {
        args.push(if config.delta.is_infinite() {
            "delta=float('inf')".to_owned()
        } else {
            format!("delta={:?}", config.delta)
        });
    }
    if config.score_alpha.to_bits() != default.score_alpha.to_bits() {
        args.push(format!("score_alpha={:?}", config.score_alpha));
    }
    if config.bp_score_iterations != default.bp_score_iterations {
        args.push(format!(
            "bp_score_iterations={}",
            config.bp_score_iterations
        ));
    }
    if config.merge_indistinguishable != default.merge_indistinguishable {
        args.push("merge_indistinguishable=False".to_owned());
    }
    match &config.ordering {
        TrellisOrdering::Deadline => {}
        TrellisOrdering::TimeOrder => args.push("ordering='time_order'".to_owned()),
        TrellisOrdering::BackwardDeadline => args.push("ordering='backward_deadline'".to_owned()),
        TrellisOrdering::Explicit(order) => args.push(format!("ordering={order:?}")),
    }
    if config.escalation_ks != default.escalation_ks {
        args.push(format!("escalation_ks={:?}", config.escalation_ks));
    }
    finish_repr("bp_trellis", args)
}
