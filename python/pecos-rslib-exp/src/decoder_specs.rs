//! Optional experimental decoder factories and native workers for batch decoding.
use pecos_bp_trellis::{BpTrellisConfig, TrellisOrdering as BpTrellisOrdering};
use pecos_decoder_core::{DecoderError, ObservableDecoder};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
enum ExperimentalSpec {
    Frontier(FrontierConfig),
    BpTrellis(BpTrellisConfig),
}

#[pyclass(
    name = "ExperimentalDecoderSpec",
    module = "pecos_rslib_exp",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyExperimentalDecoderSpec {
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
            ExperimentalSpec::Frontier(_) => "frontier",
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
            ExperimentalSpec::Frontier(c) => frontier_repr(c),
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
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        self.family().hash(&mut hasher);
        hasher.finish()
    }
    /// Internal batch protocol: construct an independent native worker.
    fn _pecos_build_decoder(&self, py: Python<'_>, dem: &str) -> PyResult<PyExperimentalWorker> {
        let (inner, num_detectors) = py
            .detach(|| {
                let inner = match &self.inner {
                    ExperimentalSpec::Frontier(c) => build_frontier(dem, c),
                    ExperimentalSpec::BpTrellis(c) => build_bp_trellis(dem, c),
                }?;
                let num_detectors = pecos_decoder_core::dem::utils::parse_dem_metadata(dem)?.0;
                Ok::<_, DecoderError>((inner, num_detectors))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyExperimentalWorker {
            inner: Mutex::new(inner),
            num_detectors,
        })
    }
}

#[pyclass(module = "pecos_rslib_exp")]
pub struct PyExperimentalWorker {
    inner: Mutex<Box<dyn ObservableDecoder + Send>>,
    #[pyo3(get)]
    num_detectors: usize,
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
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

fn finish_repr(family: &str, args: Vec<String>) -> String {
    format!("{family}({})", args.join(", "))
}
fn invalid_choice(parameter: &str, value: &str, accepted: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{parameter} has invalid value {value:?}; accepted values: {accepted}"
    ))
}
fn non_negative(parameter: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{parameter} must be finite and non-negative"
        )))
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExperimentalDecoderSpec>()?;
    module.add_function(wrap_pyfunction!(frontier, module)?)?;
    module.add_function(wrap_pyfunction!(bp_trellis, module)?)?;
    Ok(())
}

/// Mechanism ordering for the Frontier decoder.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum FrontierColumnOrder {
    #[default]
    Deadline,
    Time,
    BackwardDeadline,
    Explicit(Vec<usize>),
}

/// Route metric for the Frontier decoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrontierMetricMode {
    #[default]
    LogSumExpFloat,
    MaxLogInt,
}

/// Frontier options, preserving the existing Python ordering defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierConfig {
    pub k: usize,
    pub delta: f64,
    pub score_alpha: f64,
    pub column_order: FrontierColumnOrder,
    pub merge_indistinguishable: bool,
    pub bp_score_iterations: usize,
    pub metric_mode: FrontierMetricMode,
    pub int_metric_scale: i32,
}

impl Default for FrontierConfig {
    fn default() -> Self {
        Self {
            k: 64,
            delta: 50.0,
            score_alpha: 0.8,
            column_order: FrontierColumnOrder::Deadline,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
            metric_mode: FrontierMetricMode::LogSumExpFloat,
            int_metric_scale: 1024,
        }
    }
}

fn build_frontier(
    dem: &str,
    config: &self::FrontierConfig,
) -> Result<Box<dyn ObservableDecoder + Send>, DecoderError> {
    use self::{FrontierColumnOrder, FrontierMetricMode};
    use pecos_frontier::{FrontierConfig, FrontierDecoder, MetricMode, SparseDem};
    let dem = SparseDem::from_dem_str(dem)?;
    let column_order = match &config.column_order {
        FrontierColumnOrder::Deadline => Some(pecos_frontier::deadline_column_order(&dem)?),
        FrontierColumnOrder::Time => None,
        FrontierColumnOrder::BackwardDeadline => {
            Some(pecos_frontier::backward_deadline_column_order(&dem)?)
        }
        FrontierColumnOrder::Explicit(order) => Some(order.clone()),
    };
    let decoder = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: config.k,
            delta: config.delta,
            score_alpha: config.score_alpha,
            column_order,
            merge_indistinguishable: config.merge_indistinguishable,
            bp_score_iterations: config.bp_score_iterations,
            metric_mode: match config.metric_mode {
                FrontierMetricMode::LogSumExpFloat => MetricMode::LogSumExpFloat,
                FrontierMetricMode::MaxLogInt => MetricMode::MaxLogInt,
            },
            int_metric_scale: config.int_metric_scale,
        },
    )?;
    Ok(Box::new(decoder))
}

fn build_bp_trellis(
    dem: &str,
    config: &self::BpTrellisConfig,
) -> Result<Box<dyn ObservableDecoder + Send>, DecoderError> {
    Ok(Box::new(pecos_bp_trellis::BpTrellisDecoder::from_dem_str(
        dem,
        config.clone(),
    )?))
}

#[derive(FromPyObject)]
enum FrontierOrderArgument {
    Name(String),
    Explicit(Vec<usize>),
}

impl Default for FrontierOrderArgument {
    fn default() -> Self {
        Self::Name("deadline_reorder".to_owned())
    }
}

/// Native Rust Frontier decoder for raw DEMs, including hyperedges.
/// Batch decoding supports independent Rust workers. Pruning makes predictions
/// approximate; use pecos_rslib_exp.FrontierDecoder for per-shot confidence data.
#[pyfunction]
#[pyo3(signature = (*, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order=FrontierOrderArgument::default(), merge_indistinguishable=false, metric_mode="logsumexp_float", int_metric_scale=1024),
    text_signature = "(*, k=64, delta=50.0, score_alpha=0.8, bp_score_iterations=0, column_order='deadline_reorder', merge_indistinguishable=False, metric_mode='logsumexp_float', int_metric_scale=1024)")]
fn frontier(
    k: usize,
    delta: f64,
    score_alpha: f64,
    bp_score_iterations: usize,
    column_order: FrontierOrderArgument,
    merge_indistinguishable: bool,
    metric_mode: &str,
    int_metric_scale: i32,
) -> PyResult<PyExperimentalDecoderSpec> {
    use self::{FrontierColumnOrder, FrontierConfig, FrontierMetricMode};
    if k == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }
    if delta.is_nan() || delta < 0.0 {
        return Err(PyValueError::new_err(
            "delta must be non-negative and not NaN",
        ));
    }
    let score_alpha = non_negative("score_alpha", score_alpha)?;
    if int_metric_scale <= 0 {
        return Err(PyValueError::new_err("int_metric_scale must be positive"));
    }
    let metric_mode = match metric_mode.trim() {
        "logsumexp_float" | "float" | "exact" => FrontierMetricMode::LogSumExpFloat,
        "maxlog_int" | "max_log_int" | "viterbi_int" | "frontierLite" | "frontier_lite"
        | "frontier-lite" | "frontierlite" => FrontierMetricMode::MaxLogInt,
        value => {
            return Err(invalid_choice(
                "metric_mode",
                value,
                "'logsumexp_float', 'maxlog_int'",
            ));
        }
    };
    if metric_mode == FrontierMetricMode::MaxLogInt {
        if !delta.is_finite() {
            return Err(PyValueError::new_err(
                "delta must be finite under maxlog_int",
            ));
        }
        if merge_indistinguishable {
            return Err(PyValueError::new_err(
                "merge_indistinguishable is incompatible with maxlog_int",
            ));
        }
    }
    let column_order = match column_order {
        FrontierOrderArgument::Explicit(order) => FrontierColumnOrder::Explicit(order),
        FrontierOrderArgument::Name(name) => match name.as_str() {
            "deadline_reorder" => FrontierColumnOrder::Deadline,
            "time_order" => FrontierColumnOrder::Time,
            "backward_deadline_reorder" => FrontierColumnOrder::BackwardDeadline,
            value => {
                return Err(invalid_choice(
                    "column_order",
                    value,
                    "'deadline_reorder', 'time_order', 'backward_deadline_reorder', or a list of column indices",
                ));
            }
        },
    };
    Ok(PyExperimentalDecoderSpec::new(ExperimentalSpec::Frontier(
        FrontierConfig {
            k,
            delta,
            score_alpha,
            column_order,
            merge_indistinguishable,
            bp_score_iterations,
            metric_mode,
            int_metric_scale,
        },
    )))
}

fn frontier_repr(config: &self::FrontierConfig) -> String {
    use self::{FrontierColumnOrder, FrontierConfig, FrontierMetricMode};
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
    if config.bp_score_iterations != 0 {
        args.push(format!(
            "bp_score_iterations={}",
            config.bp_score_iterations
        ));
    }
    match &config.column_order {
        FrontierColumnOrder::Deadline => {}
        FrontierColumnOrder::Time => args.push("column_order='time_order'".to_owned()),
        FrontierColumnOrder::BackwardDeadline => {
            args.push("column_order='backward_deadline_reorder'".to_owned());
        }
        FrontierColumnOrder::Explicit(order) => args.push(format!("column_order={order:?}")),
    }
    if config.merge_indistinguishable {
        args.push("merge_indistinguishable=True".to_owned());
    }
    if config.metric_mode == FrontierMetricMode::MaxLogInt {
        args.push("metric_mode='maxlog_int'".to_owned());
    }
    if config.int_metric_scale != default.int_metric_scale {
        args.push(format!("int_metric_scale={}", config.int_metric_scale));
    }
    finish_repr("frontier", args)
}

#[derive(FromPyObject)]
enum BpTrellisOrderArgument {
    Name(String),
    Explicit(Vec<usize>),
}

impl Default for BpTrellisOrderArgument {
    fn default() -> Self {
        Self::Name("deadline".to_owned())
    }
}

/// Native Rust BP-guided trellis decoder for raw DEMs, including hyperedges.
/// Batch decoding supports independent Rust workers. Each worker prebuilds the
/// optional escalation ladder, retried only after a no-path result. Use
/// pecos_rslib_exp.BpTrellisDecoder for per-shot confidence and retry telemetry.
#[pyfunction]
#[pyo3(signature = (*, k=8, delta=100.0, score_alpha=0.8, bp_score_iterations=5, merge_indistinguishable=true, ordering=BpTrellisOrderArgument::default(), escalation_ks=None),
    text_signature = "(*, k=8, delta=100.0, score_alpha=0.8, bp_score_iterations=5, merge_indistinguishable=True, ordering='deadline', escalation_ks=None)")]
fn bp_trellis(
    k: usize,
    delta: f64,
    score_alpha: f64,
    bp_score_iterations: usize,
    merge_indistinguishable: bool,
    ordering: BpTrellisOrderArgument,
    escalation_ks: Option<Vec<usize>>,
) -> PyResult<PyExperimentalDecoderSpec> {
    use self::{BpTrellisConfig, BpTrellisOrdering};
    if k == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }
    if delta.is_nan() || delta < 0.0 {
        return Err(PyValueError::new_err(
            "delta must be non-negative and not NaN",
        ));
    }
    let score_alpha = non_negative("score_alpha", score_alpha)?;
    let escalation_ks = escalation_ks.unwrap_or_default();
    if escalation_ks.contains(&0) {
        return Err(PyValueError::new_err(
            "escalation_ks widths must be at least 1",
        ));
    }
    let ordering = match ordering {
        BpTrellisOrderArgument::Explicit(order) => BpTrellisOrdering::Explicit(order),
        BpTrellisOrderArgument::Name(name) => match name.as_str() {
            "deadline" => BpTrellisOrdering::Deadline,
            "backward_deadline" => BpTrellisOrdering::BackwardDeadline,
            "time_order" => BpTrellisOrdering::TimeOrder,
            value => {
                return Err(invalid_choice(
                    "ordering",
                    value,
                    "'deadline', 'backward_deadline', 'time_order', or a list of mechanism indices",
                ));
            }
        },
    };
    Ok(PyExperimentalDecoderSpec::new(ExperimentalSpec::BpTrellis(
        BpTrellisConfig {
            k,
            delta,
            score_alpha,
            bp_score_iterations,
            merge_indistinguishable,
            ordering,
            escalation_ks,
        },
    )))
}

fn bp_trellis_repr(config: &self::BpTrellisConfig) -> String {
    use self::{BpTrellisConfig, BpTrellisOrdering};
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
    if !config.merge_indistinguishable {
        args.push("merge_indistinguishable=False".to_owned());
    }
    match &config.ordering {
        BpTrellisOrdering::Deadline => {}
        BpTrellisOrdering::TimeOrder => args.push("ordering='time_order'".to_owned()),
        BpTrellisOrdering::BackwardDeadline => args.push("ordering='backward_deadline'".to_owned()),
        BpTrellisOrdering::Explicit(order) => args.push(format!("ordering={order:?}")),
    }
    if !config.escalation_ks.is_empty() {
        args.push(format!("escalation_ks={:?}", config.escalation_ks));
    }
    finish_repr("bp_trellis", args)
}
