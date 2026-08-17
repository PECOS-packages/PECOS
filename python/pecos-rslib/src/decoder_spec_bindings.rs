//! Python value-object bindings for typed decoder specifications.

use pecos_decoders::spec::{
    BeamSearchConfig, BeliefMatchingConfig, BeliefMatchingMode, BpLsdConfig, BpOsdConfig,
    BpSchedule, EnsembleConfig, FusionBlossomConfig, FusionBlossomSolverType, KMwpmConfig,
    MinSumBpConfig, MwpfConfig, MwpfSolverType, PecosUfPreset, PerturbedConfig,
    PerturbedFusionBlossomConfig, PyMatchingConfig, RelayBpConfig, RelayStoppingCriterion,
    TesseractConfig, TesseractPreset, WindowedConfig, WindowedMode,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyTuple};

/// Immutable Python wrapper around a feature-independent decoder specification.
#[pyclass(
    name = "DecoderSpec",
    module = "pecos_rslib.decoders",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDecoderSpec {
    pub(crate) inner: pecos_decoders::DecoderSpec,
}

impl PyDecoderSpec {
    fn new(inner: pecos_decoders::DecoderSpec) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDecoderSpec {
    /// Parse a legacy decoder type string into a typed specification.
    #[staticmethod]
    fn parse(type_string: &str) -> PyResult<Self> {
        pecos_decoders::DecoderSpec::parse(type_string)
            .map(Self::new)
            .map_err(crate::fault_tolerance_bindings::decoder_parse_error_to_py)
    }

    #[getter]
    fn history_dependent(&self) -> bool {
        self.inner.execution_traits().history_dependent
    }

    #[getter]
    fn wall_clock_dependent(&self) -> bool {
        self.inner.execution_traits().wall_clock_dependent
    }

    fn __repr__(&self) -> String {
        spec_repr(&self.inner)
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
        // Deliberately coarse: hashing only the family keeps the
        // equal-implies-equal-hash invariant trivially true (float knobs make a
        // finer structural hash subtle, e.g. -0.0 == 0.0 with distinct bits).
        // Same-family specs collide and fall back to __eq__, which is fine for
        // the small spec collections this type is keyed in.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        spec_family_name(&self.inner).hash(&mut hasher);
        hasher.finish()
    }
}

fn invalid_choice(parameter: &str, value: &str, accepted: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{parameter} has invalid value {value:?}; accepted values: {accepted}"
    ))
}

fn finite(parameter: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{parameter} must be finite; accepted values are finite numbers"
        )))
    }
}

fn non_negative(parameter: &str, value: f64) -> PyResult<f64> {
    let value = finite(parameter, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{parameter} must be non-negative; accepted values are finite numbers >= 0"
        )))
    }
}

fn optional_non_negative(parameter: &str, value: Option<f64>) -> PyResult<Option<f64>> {
    value
        .map(|value| non_negative(parameter, value))
        .transpose()
}

fn probability(parameter: &str, value: f64) -> PyResult<f64> {
    let value = finite(parameter, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{parameter} must be a probability; accepted values are finite numbers in [0, 1]"
        )))
    }
}

fn optional_probability(parameter: &str, value: Option<f64>) -> PyResult<Option<f64>> {
    value.map(|value| probability(parameter, value)).transpose()
}

fn usize_value(parameter: &str, value: i64, allow_zero: bool) -> PyResult<usize> {
    let valid = if allow_zero { value >= 0 } else { value > 0 };
    if !valid {
        let accepted = if allow_zero {
            "non-negative integers"
        } else {
            "positive integers"
        };
        return Err(PyValueError::new_err(format!(
            "{parameter} has invalid value {value}; accepted values: {accepted}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "{parameter} has invalid value {value}; accepted values: integers representable as usize"
        ))
    })
}

fn optional_i32(parameter: &str, value: Option<i64>) -> PyResult<Option<i32>> {
    value
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "{parameter} has invalid value {value}; accepted values: 32-bit signed integers"
                ))
            })
        })
        .transpose()
}

fn tesseract_preset(value: &str) -> PyResult<TesseractPreset> {
    match value {
        "default" => Ok(TesseractPreset::Default),
        "fast" => Ok(TesseractPreset::Fast),
        "accurate" => Ok(TesseractPreset::Accurate),
        value => Err(invalid_choice(
            "preset",
            value,
            "'default', 'fast', 'accurate'",
        )),
    }
}

fn bp_schedule(value: &str) -> PyResult<BpSchedule> {
    match value {
        "serial" => Ok(BpSchedule::Serial),
        "parallel" => Ok(BpSchedule::Parallel),
        "serial_relative" => Ok(BpSchedule::SerialRelative),
        value => Err(invalid_choice(
            "bp_schedule",
            value,
            "'serial', 'parallel', 'serial_relative'",
        )),
    }
}

fn fusion_solver(value: &str) -> PyResult<FusionBlossomSolverType> {
    match value {
        "auto" => Ok(FusionBlossomSolverType::Auto),
        "legacy" => Ok(FusionBlossomSolverType::Legacy),
        "serial" => Ok(FusionBlossomSolverType::Serial),
        "parallel" => Ok(FusionBlossomSolverType::Parallel),
        value => Err(invalid_choice(
            "solver",
            value,
            "'auto', 'legacy', 'serial', 'parallel'",
        )),
    }
}

fn pecos_uf_preset(value: &str) -> PyResult<PecosUfPreset> {
    match value {
        "fast" => Ok(PecosUfPreset::Fast),
        "balanced" => Ok(PecosUfPreset::Balanced),
        "accurate" => Ok(PecosUfPreset::Accurate),
        "bp" => Ok(PecosUfPreset::Bp),
        "bp_serial" => Ok(PecosUfPreset::BpSerial),
        value => Err(invalid_choice(
            "preset",
            value,
            "'fast', 'balanced', 'accurate', 'bp', 'bp_serial'",
        )),
    }
}

fn belief_matching_mode(value: &str) -> PyResult<BeliefMatchingMode> {
    match value {
        "standard" => Ok(BeliefMatchingMode::Standard),
        "correlated" => Ok(BeliefMatchingMode::Correlated),
        "matching_graph_bp" => Ok(BeliefMatchingMode::MatchingGraphBp),
        value => Err(invalid_choice(
            "mode",
            value,
            "'standard', 'correlated', 'matching_graph_bp'",
        )),
    }
}

fn windowed_mode(value: &str) -> PyResult<WindowedMode> {
    match value {
        "auto" => Ok(WindowedMode::Auto),
        "sandwich" => Ok(WindowedMode::Sandwich),
        "overlap" => Ok(WindowedMode::Overlap),
        "non_overlapping" => Ok(WindowedMode::NonOverlapping),
        value => Err(invalid_choice(
            "mode",
            value,
            "'auto', 'sandwich', 'overlap', 'non_overlapping'",
        )),
    }
}

fn mwpf_solver(value: &str) -> PyResult<MwpfSolverType> {
    match value {
        "union_find" => Ok(MwpfSolverType::UnionFind),
        "single_hair" => Ok(MwpfSolverType::SingleHair),
        "bp_hybrid" => Ok(MwpfSolverType::BpHybrid),
        "joint_single_hair" => Ok(MwpfSolverType::JointSingleHair),
        value => Err(invalid_choice(
            "solver",
            value,
            "'union_find', 'single_hair', 'bp_hybrid', 'joint_single_hair'",
        )),
    }
}

#[derive(FromPyObject)]
enum PyRelayStoppingCriterion {
    // Listed first so a Python bool is caught here and rejected instead of
    // being read as the int subclass it is (True would mean NConvergences(1)).
    Bool(bool),
    Name(String),
    Count(i64),
}

impl Default for PyRelayStoppingCriterion {
    fn default() -> Self {
        Self::Name("first_convergence".to_string())
    }
}

fn parse_stopping_criterion(value: PyRelayStoppingCriterion) -> PyResult<RelayStoppingCriterion> {
    match value {
        PyRelayStoppingCriterion::Bool(value) => Err(invalid_choice(
            "stopping_criterion",
            if value { "True" } else { "False" },
            "'pre_iter', 'all', 'first_convergence', or a positive integer",
        )),
        PyRelayStoppingCriterion::Name(value) => match value.as_str() {
            "pre_iter" => Ok(RelayStoppingCriterion::PreIter),
            "all" => Ok(RelayStoppingCriterion::All),
            "first_convergence" => Ok(RelayStoppingCriterion::FirstConvergence),
            _ => Err(invalid_choice(
                "stopping_criterion",
                &value,
                "'pre_iter', 'all', 'first_convergence', or a positive integer",
            )),
        },
        PyRelayStoppingCriterion::Count(value) => Ok(RelayStoppingCriterion::NConvergences(
            usize_value("stopping_criterion", value, false)?,
        )),
    }
}

fn cloned_or_default(
    value: Option<PyRef<'_, PyDecoderSpec>>,
    default: &pecos_decoders::DecoderSpec,
) -> pecos_decoders::DecoderSpec {
    value.map_or_else(|| default.clone(), |value| value.inner.clone())
}

#[pyfunction]
#[pyo3(signature = (*, correlated, error_probability=None))]
fn pymatching(correlated: bool, error_probability: Option<f64>) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::PyMatching(
        PyMatchingConfig {
            correlated,
            error_probability: optional_probability("error_probability", error_probability)?,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, preset="default", det_beam=None, beam_climbing=None, verbose=None, no_revisit_dets=None, pqlimit=None, det_penalty=None))]
fn tesseract(
    preset: &str,
    det_beam: Option<i64>,
    beam_climbing: Option<bool>,
    verbose: Option<bool>,
    no_revisit_dets: Option<bool>,
    pqlimit: Option<i64>,
    det_penalty: Option<f64>,
) -> PyResult<PyDecoderSpec> {
    let det_beam = det_beam
        .map(|value| usize_value("det_beam", value, false))
        .transpose()?
        .map(|value| {
            u16::try_from(value).map_err(|_| {
                PyValueError::new_err(
                    "det_beam is too large; accepted values are positive 16-bit integers",
                )
            })
        })
        .transpose()?;
    let pqlimit = pqlimit
        .map(|value| usize_value("pqlimit", value, false))
        .transpose()?;
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::Tesseract(
        TesseractConfig {
            preset: tesseract_preset(preset)?,
            det_beam,
            beam_climbing,
            verbose,
            no_revisit_dets,
            pqlimit,
            det_penalty: optional_non_negative("det_penalty", det_penalty)?,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, error_rate=None, max_iter=100, bp_schedule="parallel", ms_scaling_factor=None, osd_order=0, random_schedule_seed=None))]
fn bp_osd(
    error_rate: Option<f64>,
    max_iter: i64,
    bp_schedule: &str,
    ms_scaling_factor: Option<f64>,
    osd_order: i64,
    random_schedule_seed: Option<i64>,
) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::BpOsd(
        BpOsdConfig {
            error_rate: optional_probability("error_rate", error_rate)?,
            max_iter: usize_value("max_iter", max_iter, false)?,
            bp_schedule: self::bp_schedule(bp_schedule)?,
            ms_scaling_factor: optional_non_negative("ms_scaling_factor", ms_scaling_factor)?,
            osd_order: usize_value("osd_order", osd_order, true)?,
            random_schedule_seed: optional_i32("random_schedule_seed", random_schedule_seed)?,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, error_rate=None, max_iter=100, bp_schedule="parallel", ms_scaling_factor=None, lsd_order=0, bits_per_step=0, random_schedule_seed=None))]
fn bp_lsd(
    error_rate: Option<f64>,
    max_iter: i64,
    bp_schedule: &str,
    ms_scaling_factor: Option<f64>,
    lsd_order: i64,
    bits_per_step: i64,
    random_schedule_seed: Option<i64>,
) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::BpLsd(
        BpLsdConfig {
            error_rate: optional_probability("error_rate", error_rate)?,
            max_iter: usize_value("max_iter", max_iter, false)?,
            bp_schedule: self::bp_schedule(bp_schedule)?,
            ms_scaling_factor: optional_non_negative("ms_scaling_factor", ms_scaling_factor)?,
            lsd_order: usize_value("lsd_order", lsd_order, true)?,
            bits_per_step: usize_value("bits_per_step", bits_per_step, true)?,
            random_schedule_seed: optional_i32("random_schedule_seed", random_schedule_seed)?,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, correlated=false, solver="auto"))]
fn fusion_blossom(correlated: bool, solver: &str) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(
        pecos_decoders::DecoderSpec::FusionBlossom(FusionBlossomConfig {
            correlated,
            solver_type: fusion_solver(solver)?,
        }),
    ))
}

#[pyfunction]
#[pyo3(
    signature = (*, error_rate=None, max_iter=200, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=0.65, pre_iter=80, num_sets=300, set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stopping_criterion=PyRelayStoppingCriterion::default(), seed=0),
    text_signature = "(*, error_rate=None, max_iter=200, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=0.65, pre_iter=80, num_sets=300, set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stopping_criterion='first_convergence', seed=0)"
)]
fn relay_bp(
    error_rate: Option<f64>,
    max_iter: i64,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    gamma0: Option<f64>,
    pre_iter: i64,
    num_sets: i64,
    set_max_iter: i64,
    gamma_dist_interval: (f64, f64),
    stopping_criterion: PyRelayStoppingCriterion,
    seed: u64,
) -> PyResult<PyDecoderSpec> {
    let gamma_min = finite("gamma_dist_interval", gamma_dist_interval.0)?;
    let gamma_max = finite("gamma_dist_interval", gamma_dist_interval.1)?;
    if gamma_min > gamma_max {
        return Err(PyValueError::new_err(format!(
            "gamma_dist_interval has invalid value {gamma_dist_interval:?}; accepted values: (minimum, maximum) with minimum <= maximum"
        )));
    }
    let stopping_criterion = parse_stopping_criterion(stopping_criterion)?;
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::RelayBp(
        RelayBpConfig {
            error_rate: optional_probability("error_rate", error_rate)?,
            max_iter: usize_value("max_iter", max_iter, false)?,
            alpha: optional_non_negative("alpha", alpha)?,
            alpha_iteration_scaling_factor: non_negative(
                "alpha_iteration_scaling_factor",
                alpha_iteration_scaling_factor,
            )?,
            gamma0: gamma0.map(|value| finite("gamma0", value)).transpose()?,
            pre_iter: usize_value("pre_iter", pre_iter, true)?,
            num_sets: usize_value("num_sets", num_sets, false)?,
            set_max_iter: usize_value("set_max_iter", set_max_iter, false)?,
            gamma_dist_interval: (gamma_min, gamma_max),
            stopping_criterion,
            seed,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, error_rate=None, max_iter=200, alpha=None))]
fn min_sum_bp(
    error_rate: Option<f64>,
    max_iter: i64,
    alpha: Option<f64>,
) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::MinSumBp(
        MinSumBpConfig {
            error_rate: optional_probability("error_rate", error_rate)?,
            max_iter: usize_value("max_iter", max_iter, false)?,
            alpha: optional_non_negative("alpha", alpha)?,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, preset="fast"))]
fn pecos_uf(preset: &str) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::PecosUf(
        pecos_uf_preset(preset)?,
    )))
}

#[pyfunction]
#[pyo3(signature = (*, mode="standard"))]
fn belief_matching(mode: &str) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(
        pecos_decoders::DecoderSpec::BeliefMatching(BeliefMatchingConfig {
            mode: belief_matching_mode(mode)?,
            embedded_full_dem: None,
        }),
    ))
}

#[pyfunction]
#[pyo3(signature = (*, step=0, buffer=0, mode="auto", seam=0, core_extend=0, commit_weight_max=0.0, inner=None, sandwich_phase2=None))]
fn windowed(
    step: i64,
    buffer: i64,
    mode: &str,
    seam: i64,
    core_extend: i64,
    commit_weight_max: f64,
    inner: Option<PyRef<'_, PyDecoderSpec>>,
    sandwich_phase2: Option<PyRef<'_, PyDecoderSpec>>,
) -> PyResult<PyDecoderSpec> {
    let defaults = WindowedConfig::default();
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::Windowed(
        WindowedConfig {
            step_size: usize_value("step", step, true)?,
            buffer_size: usize_value("buffer", buffer, true)?,
            mode: windowed_mode(mode)?,
            seam_half_width: usize_value("seam", seam, true)?,
            core_extend: usize_value("core_extend", core_extend, true)?,
            commit_weight_max: non_negative("commit_weight_max", commit_weight_max)?,
            inner: Box::new(cloned_or_default(inner, &defaults.inner)),
            sandwich_phase2: Box::new(cloned_or_default(
                sandwich_phase2,
                &defaults.sandwich_phase2,
            )),
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, solver="joint_single_hair", cluster_node_limit=50, timeout=None, only_solve_primal_once=false))]
fn mwpf(
    solver: &str,
    cluster_node_limit: i64,
    timeout: Option<f64>,
    only_solve_primal_once: bool,
) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::Mwpf(
        MwpfConfig {
            solver_type: mwpf_solver(solver)?,
            cluster_node_limit: usize_value("cluster_node_limit", cluster_node_limit, false)?,
            timeout: optional_non_negative("timeout", timeout)?,
            only_solve_primal_once,
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, inner=None, k=15, sigma=0.7, seed=42))]
fn perturbed(
    inner: Option<PyRef<'_, PyDecoderSpec>>,
    k: i64,
    sigma: f64,
    seed: u64,
) -> PyResult<PyDecoderSpec> {
    let defaults = PerturbedConfig::default();
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::Perturbed(
        PerturbedConfig {
            k: usize_value("k", k, false)?,
            sigma: non_negative("sigma", sigma)?,
            seed,
            inner: Box::new(cloned_or_default(inner, &defaults.inner)),
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, beam_width=5, sigma=0.5, seed=42, step=0, buffer=0, commit_weight_max=0.0, phase2=None))]
fn beamsearch(
    beam_width: i64,
    sigma: f64,
    seed: u64,
    step: i64,
    buffer: i64,
    commit_weight_max: f64,
    phase2: Option<PyRef<'_, PyDecoderSpec>>,
) -> PyResult<PyDecoderSpec> {
    let defaults = BeamSearchConfig::default();
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::BeamSearch(
        BeamSearchConfig {
            beam_width: usize_value("beam_width", beam_width, false)?,
            perturbation_sigma: non_negative("sigma", sigma)?,
            seed,
            step_size: usize_value("step", step, true)?,
            buffer_size: usize_value("buffer", buffer, true)?,
            commit_weight_max: non_negative("commit_weight_max", commit_weight_max)?,
            phase2: Box::new(cloned_or_default(phase2, &defaults.phase2)),
        },
    )))
}

#[pyfunction]
#[pyo3(signature = (*members))]
fn ensemble(members: &Bound<'_, PyTuple>) -> PyResult<PyDecoderSpec> {
    if members.is_empty() {
        return Err(PyValueError::new_err(
            "members requires at least one DecoderSpec; accepted values: one or more DecoderSpec objects",
        ));
    }
    let members = members
        .iter()
        .map(|member| {
            member
                .extract::<PyRef<'_, PyDecoderSpec>>()
                .map_err(|error| PyTypeError::new_err(error.to_string()))
                .map(|member| member.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::Ensemble(
        EnsembleConfig { members },
    )))
}

#[pyfunction]
#[pyo3(signature = (*, k=10))]
fn k_mwpm(k: i64) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(pecos_decoders::DecoderSpec::KMwpm(
        KMwpmConfig {
            k: usize_value("k", k, false)?,
        },
    )))
}

#[pyfunction]
fn astar() -> PyDecoderSpec {
    PyDecoderSpec::new(pecos_decoders::DecoderSpec::AStar)
}

#[pyfunction]
fn astar_full() -> PyDecoderSpec {
    PyDecoderSpec::new(pecos_decoders::DecoderSpec::AStarFull)
}

#[pyfunction]
fn union_find() -> PyDecoderSpec {
    PyDecoderSpec::new(pecos_decoders::DecoderSpec::UnionFind)
}

#[pyfunction]
fn belief_find() -> PyDecoderSpec {
    PyDecoderSpec::new(pecos_decoders::DecoderSpec::BeliefFind)
}

#[pyfunction]
#[pyo3(signature = (*, k=5, sigma=0.5, seed=42))]
fn perturbed_fb_corr(k: i64, sigma: f64, seed: u64) -> PyResult<PyDecoderSpec> {
    Ok(PyDecoderSpec::new(
        pecos_decoders::DecoderSpec::PerturbedFusionBlossomCorrelated(
            PerturbedFusionBlossomConfig {
                k: usize_value("k", k, false)?,
                sigma: non_negative("sigma", sigma)?,
                seed,
            },
        ),
    ))
}

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn push_option<T: std::fmt::Debug>(args: &mut Vec<String>, name: &str, value: Option<T>) {
    if let Some(value) = value {
        args.push(format!("{name}={value:?}"));
    }
}

fn finish_repr(family: &str, args: Vec<String>) -> String {
    // The repr names the real factory callable in pecos.decoders; there is no
    // DecoderSpec.<family> constructor.
    format!("{family}({})", args.join(", "))
}

/// Family name of a spec, used for the deliberately coarse `__hash__`.
fn spec_family_name(spec: &pecos_decoders::DecoderSpec) -> &'static str {
    match spec {
        pecos_decoders::DecoderSpec::PyMatching(_) => "pymatching",
        pecos_decoders::DecoderSpec::Tesseract(_) => "tesseract",
        pecos_decoders::DecoderSpec::KMwpm(_) => "k_mwpm",
        pecos_decoders::DecoderSpec::AStar => "astar",
        pecos_decoders::DecoderSpec::AStarFull => "astar_full",
        pecos_decoders::DecoderSpec::FusionBlossom(_) => "fusion_blossom",
        pecos_decoders::DecoderSpec::PerturbedFusionBlossomCorrelated(_) => "perturbed_fb_corr",
        pecos_decoders::DecoderSpec::BpOsd(_) => "bp_osd",
        pecos_decoders::DecoderSpec::BpLsd(_) => "bp_lsd",
        pecos_decoders::DecoderSpec::BeliefFind => "belief_find",
        pecos_decoders::DecoderSpec::UnionFind => "union_find",
        pecos_decoders::DecoderSpec::RelayBp(_) => "relay_bp",
        pecos_decoders::DecoderSpec::MinSumBp(_) => "min_sum_bp",
        pecos_decoders::DecoderSpec::PecosUf(_) => "pecos_uf",
        pecos_decoders::DecoderSpec::BeliefMatching(_) => "belief_matching",
        pecos_decoders::DecoderSpec::Windowed(_) => "windowed",
        pecos_decoders::DecoderSpec::Mwpf(_) => "mwpf",
        pecos_decoders::DecoderSpec::Perturbed(_) => "perturbed",
        pecos_decoders::DecoderSpec::BeamSearch(_) => "beamsearch",
        pecos_decoders::DecoderSpec::Ensemble(_) => "ensemble",
    }
}

fn spec_repr(spec: &pecos_decoders::DecoderSpec) -> String {
    match spec {
        pecos_decoders::DecoderSpec::PyMatching(config) => {
            // `correlated` is a required argument, so the repr always shows it.
            let mut args = vec![format!("correlated={}", py_bool(config.correlated))];
            push_option(&mut args, "error_probability", config.error_probability);
            finish_repr("pymatching", args)
        }
        pecos_decoders::DecoderSpec::Tesseract(config) => {
            let mut args = Vec::new();
            if config.preset != TesseractPreset::Default {
                args.push(format!("preset={:?}", tesseract_preset_name(config.preset)));
            }
            push_option(&mut args, "det_beam", config.det_beam);
            if let Some(value) = config.beam_climbing {
                args.push(format!("beam_climbing={}", py_bool(value)));
            }
            if let Some(value) = config.verbose {
                args.push(format!("verbose={}", py_bool(value)));
            }
            if let Some(value) = config.no_revisit_dets {
                args.push(format!("no_revisit_dets={}", py_bool(value)));
            }
            push_option(&mut args, "pqlimit", config.pqlimit);
            push_option(&mut args, "det_penalty", config.det_penalty);
            finish_repr("tesseract", args)
        }
        pecos_decoders::DecoderSpec::KMwpm(config) => {
            let args = (config.k != KMwpmConfig::default().k)
                .then(|| format!("k={}", config.k))
                .into_iter()
                .collect();
            finish_repr("k_mwpm", args)
        }
        pecos_decoders::DecoderSpec::AStar => finish_repr("astar", Vec::new()),
        pecos_decoders::DecoderSpec::AStarFull => finish_repr("astar_full", Vec::new()),
        pecos_decoders::DecoderSpec::FusionBlossom(config) => {
            let mut args = Vec::new();
            if config.correlated {
                args.push("correlated=True".to_string());
            }
            if config.solver_type != FusionBlossomSolverType::Auto {
                args.push(format!(
                    "solver={:?}",
                    fusion_solver_name(config.solver_type)
                ));
            }
            finish_repr("fusion_blossom", args)
        }
        pecos_decoders::DecoderSpec::PerturbedFusionBlossomCorrelated(config) => {
            let default = PerturbedFusionBlossomConfig::default();
            let mut args = Vec::new();
            if config.k != default.k {
                args.push(format!("k={}", config.k));
            }
            if config.sigma.to_bits() != default.sigma.to_bits() {
                args.push(format!("sigma={:?}", config.sigma));
            }
            if config.seed != default.seed {
                args.push(format!("seed={}", config.seed));
            }
            finish_repr("perturbed_fb_corr", args)
        }
        pecos_decoders::DecoderSpec::BpOsd(config) => bp_osd_repr(config),
        pecos_decoders::DecoderSpec::BpLsd(config) => bp_lsd_repr(config),
        pecos_decoders::DecoderSpec::BeliefFind => finish_repr("belief_find", Vec::new()),
        pecos_decoders::DecoderSpec::UnionFind => finish_repr("union_find", Vec::new()),
        pecos_decoders::DecoderSpec::RelayBp(config) => relay_bp_repr(config),
        pecos_decoders::DecoderSpec::MinSumBp(config) => {
            let default = MinSumBpConfig::default();
            let mut args = Vec::new();
            push_option(&mut args, "error_rate", config.error_rate);
            if config.max_iter != default.max_iter {
                args.push(format!("max_iter={}", config.max_iter));
            }
            push_option(&mut args, "alpha", config.alpha);
            finish_repr("min_sum_bp", args)
        }
        pecos_decoders::DecoderSpec::PecosUf(preset) => {
            let args = (*preset != PecosUfPreset::Fast)
                .then(|| format!("preset={:?}", pecos_uf_preset_name(*preset)))
                .into_iter()
                .collect();
            finish_repr("pecos_uf", args)
        }
        pecos_decoders::DecoderSpec::BeliefMatching(config) => {
            let mut args = Vec::new();
            if config.mode != BeliefMatchingMode::Standard {
                args.push(format!("mode={:?}", belief_matching_mode_name(config.mode)));
            }
            if let Some(full_dem) = &config.embedded_full_dem {
                // The embedded DEM can be megabytes; never inline it in a repr.
                args.push(format!("embedded_full_dem=<{} bytes>", full_dem.len()));
            }
            finish_repr("belief_matching", args)
        }
        pecos_decoders::DecoderSpec::Windowed(config) => windowed_repr(config),
        pecos_decoders::DecoderSpec::Mwpf(config) => mwpf_repr(config),
        pecos_decoders::DecoderSpec::Perturbed(config) => perturbed_repr(config),
        pecos_decoders::DecoderSpec::BeamSearch(config) => beamsearch_repr(config),
        pecos_decoders::DecoderSpec::Ensemble(config) => {
            finish_repr("ensemble", config.members.iter().map(spec_repr).collect())
        }
    }
}

fn bp_osd_repr(config: &BpOsdConfig) -> String {
    let default = BpOsdConfig::default();
    let mut args = Vec::new();
    push_option(&mut args, "error_rate", config.error_rate);
    if config.max_iter != default.max_iter {
        args.push(format!("max_iter={}", config.max_iter));
    }
    if config.bp_schedule != default.bp_schedule {
        args.push(format!(
            "bp_schedule={:?}",
            bp_schedule_name(config.bp_schedule)
        ));
    }
    push_option(&mut args, "ms_scaling_factor", config.ms_scaling_factor);
    if config.osd_order != default.osd_order {
        args.push(format!("osd_order={}", config.osd_order));
    }
    push_option(
        &mut args,
        "random_schedule_seed",
        config.random_schedule_seed,
    );
    finish_repr("bp_osd", args)
}

fn bp_lsd_repr(config: &BpLsdConfig) -> String {
    let default = BpLsdConfig::default();
    let mut args = Vec::new();
    push_option(&mut args, "error_rate", config.error_rate);
    if config.max_iter != default.max_iter {
        args.push(format!("max_iter={}", config.max_iter));
    }
    if config.bp_schedule != default.bp_schedule {
        args.push(format!(
            "bp_schedule={:?}",
            bp_schedule_name(config.bp_schedule)
        ));
    }
    push_option(&mut args, "ms_scaling_factor", config.ms_scaling_factor);
    if config.lsd_order != default.lsd_order {
        args.push(format!("lsd_order={}", config.lsd_order));
    }
    if config.bits_per_step != default.bits_per_step {
        args.push(format!("bits_per_step={}", config.bits_per_step));
    }
    push_option(
        &mut args,
        "random_schedule_seed",
        config.random_schedule_seed,
    );
    finish_repr("bp_lsd", args)
}

fn relay_bp_repr(config: &RelayBpConfig) -> String {
    let default = RelayBpConfig::default();
    let mut args = Vec::new();
    push_option(&mut args, "error_rate", config.error_rate);
    if config.max_iter != default.max_iter {
        args.push(format!("max_iter={}", config.max_iter));
    }
    push_option(&mut args, "alpha", config.alpha);
    if config.alpha_iteration_scaling_factor.to_bits()
        != default.alpha_iteration_scaling_factor.to_bits()
    {
        args.push(format!(
            "alpha_iteration_scaling_factor={:?}",
            config.alpha_iteration_scaling_factor
        ));
    }
    if config.gamma0 != default.gamma0 {
        args.push(format!("gamma0={:?}", config.gamma0));
    }
    if config.pre_iter != default.pre_iter {
        args.push(format!("pre_iter={}", config.pre_iter));
    }
    if config.num_sets != default.num_sets {
        args.push(format!("num_sets={}", config.num_sets));
    }
    if config.set_max_iter != default.set_max_iter {
        args.push(format!("set_max_iter={}", config.set_max_iter));
    }
    if config.gamma_dist_interval != default.gamma_dist_interval {
        args.push(format!(
            "gamma_dist_interval={:?}",
            config.gamma_dist_interval
        ));
    }
    if config.stopping_criterion != default.stopping_criterion {
        args.push(format!(
            "stopping_criterion={}",
            stopping_repr(config.stopping_criterion)
        ));
    }
    if config.seed != default.seed {
        args.push(format!("seed={}", config.seed));
    }
    finish_repr("relay_bp", args)
}

fn windowed_repr(config: &WindowedConfig) -> String {
    let default = WindowedConfig::default();
    let mut args = Vec::new();
    if config.step_size != default.step_size {
        args.push(format!("step={}", config.step_size));
    }
    if config.buffer_size != default.buffer_size {
        args.push(format!("buffer={}", config.buffer_size));
    }
    if config.mode != default.mode {
        args.push(format!("mode={:?}", windowed_mode_name(config.mode)));
    }
    if config.seam_half_width != default.seam_half_width {
        args.push(format!("seam={}", config.seam_half_width));
    }
    if config.core_extend != default.core_extend {
        args.push(format!("core_extend={}", config.core_extend));
    }
    if config.commit_weight_max.to_bits() != default.commit_weight_max.to_bits() {
        args.push(format!("commit_weight_max={:?}", config.commit_weight_max));
    }
    if config.inner != default.inner {
        args.push(format!("inner={}", spec_repr(&config.inner)));
    }
    if config.sandwich_phase2 != default.sandwich_phase2 {
        args.push(format!(
            "sandwich_phase2={}",
            spec_repr(&config.sandwich_phase2)
        ));
    }
    finish_repr("windowed", args)
}

fn mwpf_repr(config: &MwpfConfig) -> String {
    let default = MwpfConfig::default();
    let mut args = Vec::new();
    if config.solver_type != default.solver_type {
        args.push(format!("solver={:?}", mwpf_solver_name(config.solver_type)));
    }
    if config.cluster_node_limit != default.cluster_node_limit {
        args.push(format!("cluster_node_limit={}", config.cluster_node_limit));
    }
    push_option(&mut args, "timeout", config.timeout);
    if config.only_solve_primal_once {
        args.push("only_solve_primal_once=True".to_string());
    }
    finish_repr("mwpf", args)
}

fn perturbed_repr(config: &PerturbedConfig) -> String {
    let default = PerturbedConfig::default();
    let mut args = Vec::new();
    if config.inner != default.inner {
        args.push(format!("inner={}", spec_repr(&config.inner)));
    }
    if config.k != default.k {
        args.push(format!("k={}", config.k));
    }
    if config.sigma.to_bits() != default.sigma.to_bits() {
        args.push(format!("sigma={:?}", config.sigma));
    }
    if config.seed != default.seed {
        args.push(format!("seed={}", config.seed));
    }
    finish_repr("perturbed", args)
}

fn beamsearch_repr(config: &BeamSearchConfig) -> String {
    let default = BeamSearchConfig::default();
    let mut args = Vec::new();
    if config.beam_width != default.beam_width {
        args.push(format!("beam_width={}", config.beam_width));
    }
    if config.perturbation_sigma.to_bits() != default.perturbation_sigma.to_bits() {
        args.push(format!("sigma={:?}", config.perturbation_sigma));
    }
    if config.seed != default.seed {
        args.push(format!("seed={}", config.seed));
    }
    if config.step_size != default.step_size {
        args.push(format!("step={}", config.step_size));
    }
    if config.buffer_size != default.buffer_size {
        args.push(format!("buffer={}", config.buffer_size));
    }
    if config.commit_weight_max.to_bits() != default.commit_weight_max.to_bits() {
        args.push(format!("commit_weight_max={:?}", config.commit_weight_max));
    }
    if config.phase2 != default.phase2 {
        args.push(format!("phase2={}", spec_repr(&config.phase2)));
    }
    finish_repr("beamsearch", args)
}

fn tesseract_preset_name(value: TesseractPreset) -> &'static str {
    match value {
        TesseractPreset::Default => "default",
        TesseractPreset::Fast => "fast",
        TesseractPreset::Accurate => "accurate",
    }
}
fn bp_schedule_name(value: BpSchedule) -> &'static str {
    match value {
        BpSchedule::Serial => "serial",
        BpSchedule::Parallel => "parallel",
        BpSchedule::SerialRelative => "serial_relative",
    }
}
fn fusion_solver_name(value: FusionBlossomSolverType) -> &'static str {
    match value {
        FusionBlossomSolverType::Auto => "auto",
        FusionBlossomSolverType::Legacy => "legacy",
        FusionBlossomSolverType::Serial => "serial",
        FusionBlossomSolverType::Parallel => "parallel",
    }
}
fn pecos_uf_preset_name(value: PecosUfPreset) -> &'static str {
    match value {
        PecosUfPreset::Fast => "fast",
        PecosUfPreset::Balanced => "balanced",
        PecosUfPreset::Accurate => "accurate",
        PecosUfPreset::Bp => "bp",
        PecosUfPreset::BpSerial => "bp_serial",
    }
}
fn belief_matching_mode_name(value: BeliefMatchingMode) -> &'static str {
    match value {
        BeliefMatchingMode::Standard => "standard",
        BeliefMatchingMode::Correlated => "correlated",
        BeliefMatchingMode::MatchingGraphBp => "matching_graph_bp",
        BeliefMatchingMode::Hybrid => "hybrid",
    }
}
fn windowed_mode_name(value: WindowedMode) -> &'static str {
    match value {
        WindowedMode::Auto => "auto",
        WindowedMode::Sandwich => "sandwich",
        WindowedMode::Overlap => "overlap",
        WindowedMode::NonOverlapping => "non_overlapping",
    }
}
fn mwpf_solver_name(value: MwpfSolverType) -> &'static str {
    match value {
        MwpfSolverType::UnionFind => "union_find",
        MwpfSolverType::SingleHair => "single_hair",
        MwpfSolverType::BpHybrid => "bp_hybrid",
        MwpfSolverType::JointSingleHair => "joint_single_hair",
    }
}
fn stopping_repr(value: RelayStoppingCriterion) -> String {
    match value {
        RelayStoppingCriterion::PreIter => "'pre_iter'".to_string(),
        RelayStoppingCriterion::All => "'all'".to_string(),
        RelayStoppingCriterion::FirstConvergence => "'first_convergence'".to_string(),
        RelayStoppingCriterion::NConvergences(value) => value.to_string(),
    }
}

/// Register typed decoder-spec factories on `pecos_rslib.decoders`.
pub fn register_decoder_specs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDecoderSpec>()?;
    module.add_function(wrap_pyfunction!(pymatching, module)?)?;
    module.add_function(wrap_pyfunction!(tesseract, module)?)?;
    module.add_function(wrap_pyfunction!(bp_osd, module)?)?;
    module.add_function(wrap_pyfunction!(bp_lsd, module)?)?;
    module.add_function(wrap_pyfunction!(fusion_blossom, module)?)?;
    module.add_function(wrap_pyfunction!(relay_bp, module)?)?;
    module.add_function(wrap_pyfunction!(min_sum_bp, module)?)?;
    module.add_function(wrap_pyfunction!(pecos_uf, module)?)?;
    module.add_function(wrap_pyfunction!(belief_matching, module)?)?;
    module.add_function(wrap_pyfunction!(windowed, module)?)?;
    module.add_function(wrap_pyfunction!(mwpf, module)?)?;
    module.add_function(wrap_pyfunction!(perturbed, module)?)?;
    module.add_function(wrap_pyfunction!(beamsearch, module)?)?;
    module.add_function(wrap_pyfunction!(ensemble, module)?)?;
    module.add_function(wrap_pyfunction!(k_mwpm, module)?)?;
    module.add_function(wrap_pyfunction!(astar, module)?)?;
    module.add_function(wrap_pyfunction!(astar_full, module)?)?;
    module.add_function(wrap_pyfunction!(union_find, module)?)?;
    module.add_function(wrap_pyfunction!(belief_find, module)?)?;
    module.add_function(wrap_pyfunction!(perturbed_fb_corr, module)?)?;
    Ok(())
}
