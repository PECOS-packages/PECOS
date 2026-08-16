//! Feature-independent decoder configuration data.

use super::DecoderSpec;

/// `PyMatching` construction options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyMatchingConfig {
    /// Enable correlated matching.
    pub correlated: bool,
    /// Replace every edge probability when set.
    pub error_probability: Option<f64>,
}

/// Tesseract configuration preset.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TesseractPreset {
    /// Upstream defaults.
    #[default]
    Default,
    /// Runtime-oriented defaults used by the legacy string factory.
    Fast,
    /// Accuracy-oriented defaults.
    Accurate,
}

/// Tesseract construction options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TesseractConfig {
    pub preset: TesseractPreset,
    pub det_beam: Option<u16>,
    pub beam_climbing: Option<bool>,
    pub verbose: Option<bool>,
    pub no_revisit_dets: Option<bool>,
    pub pqlimit: Option<usize>,
    pub det_penalty: Option<f64>,
}

/// K-MWPM construction options.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KMwpmConfig {
    pub k: usize,
}

impl Default for KMwpmConfig {
    fn default() -> Self {
        Self { k: 10 }
    }
}

/// Fusion Blossom solver selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FusionBlossomSolverType {
    /// Select serial or parallel from the DEM shape.
    #[default]
    Auto,
    Legacy,
    Serial,
    Parallel,
}

/// Fusion Blossom construction options.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FusionBlossomConfig {
    pub correlated: bool,
    pub solver_type: FusionBlossomSolverType,
}

/// BP update schedule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BpSchedule {
    Serial,
    #[default]
    Parallel,
    SerialRelative,
}

impl BpSchedule {
    pub(crate) fn uses_random_serial_order(self) -> bool {
        matches!(self, Self::Serial | Self::SerialRelative)
    }
}

/// BP+OSD construction options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpOsdConfig {
    pub error_rate: Option<f64>,
    pub max_iter: usize,
    pub bp_schedule: BpSchedule,
    pub ms_scaling_factor: Option<f64>,
    pub osd_order: usize,
    pub random_schedule_seed: Option<i32>,
}

impl Default for BpOsdConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: 100,
            bp_schedule: BpSchedule::Parallel,
            ms_scaling_factor: None,
            osd_order: 0,
            random_schedule_seed: None,
        }
    }
}

/// BP+LSD construction options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpLsdConfig {
    pub error_rate: Option<f64>,
    pub max_iter: usize,
    pub bp_schedule: BpSchedule,
    pub ms_scaling_factor: Option<f64>,
    pub lsd_order: usize,
    pub bits_per_step: usize,
    pub random_schedule_seed: Option<i32>,
}

impl Default for BpLsdConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: 100,
            bp_schedule: BpSchedule::Parallel,
            ms_scaling_factor: None,
            lsd_order: 0,
            bits_per_step: 0,
            random_schedule_seed: None,
        }
    }
}

/// Relay BP stopping rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RelayStoppingCriterion {
    PreIter,
    All,
    #[default]
    FirstConvergence,
    NConvergences(usize),
}

/// Relay BP construction options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelayBpConfig {
    pub error_rate: Option<f64>,
    pub max_iter: usize,
    pub alpha: Option<f64>,
    pub alpha_iteration_scaling_factor: f64,
    pub gamma0: Option<f64>,
    pub pre_iter: usize,
    pub num_sets: usize,
    pub set_max_iter: usize,
    pub gamma_dist_interval: (f64, f64),
    pub stopping_criterion: RelayStoppingCriterion,
    pub seed: u64,
}

impl Default for RelayBpConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(0.65),
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            stopping_criterion: RelayStoppingCriterion::FirstConvergence,
            seed: 0,
        }
    }
}

/// Standalone min-sum BP construction options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinSumBpConfig {
    pub error_rate: Option<f64>,
    pub max_iter: usize,
    pub alpha: Option<f64>,
}

impl Default for MinSumBpConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: 200,
            alpha: None,
        }
    }
}

/// PECOS Union-Find preset.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PecosUfPreset {
    #[default]
    Fast,
    Balanced,
    Accurate,
    Bp,
    BpSerial,
}

/// Belief-matching construction mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BeliefMatchingMode {
    #[default]
    Standard,
    Correlated,
    MatchingGraphBp,
    Hybrid,
}

/// Belief-matching construction options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeliefMatchingConfig {
    pub mode: BeliefMatchingMode,
    /// Full DEM carried only by the legacy `belief_matching_hybrid:<DEM>` grammar.
    /// `build` still requires a `HybridDem`; the Python adapter combines this
    /// fragment with its separately supplied decomposed DEM.
    pub embedded_full_dem: Option<String>,
}

/// Sliding-window construction mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WindowedMode {
    #[default]
    NonOverlapping,
    Sandwich,
    Overlap,
    Auto,
}

/// Sliding-window construction options.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowedConfig {
    pub step_size: usize,
    pub buffer_size: usize,
    pub mode: WindowedMode,
    pub seam_half_width: usize,
    pub core_extend: usize,
    pub commit_weight_max: f64,
    pub inner: Box<DecoderSpec>,
    pub sandwich_phase2: Box<DecoderSpec>,
}

impl Default for WindowedConfig {
    fn default() -> Self {
        Self {
            step_size: 0,
            buffer_size: 0,
            mode: WindowedMode::Auto,
            seam_half_width: 0,
            core_extend: 0,
            commit_weight_max: 0.0,
            inner: Box::new(DecoderSpec::PecosUf(PecosUfPreset::Fast)),
            sandwich_phase2: Box::new(DecoderSpec::PyMatching(PyMatchingConfig {
                correlated: true,
                error_probability: None,
            })),
        }
    }
}

/// MWPF solver selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MwpfSolverType {
    UnionFind,
    SingleHair,
    BpHybrid,
    #[default]
    JointSingleHair,
}

/// MWPF construction options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MwpfConfig {
    pub solver_type: MwpfSolverType,
    pub cluster_node_limit: usize,
    pub timeout: Option<f64>,
    pub only_solve_primal_once: bool,
}

impl Default for MwpfConfig {
    fn default() -> Self {
        Self {
            solver_type: MwpfSolverType::JointSingleHair,
            cluster_node_limit: 50,
            timeout: None,
            only_solve_primal_once: false,
        }
    }
}

/// Perturbed-weight ensemble construction options.
#[derive(Clone, Debug, PartialEq)]
pub struct PerturbedConfig {
    pub k: usize,
    pub sigma: f64,
    pub seed: u64,
    pub inner: Box<DecoderSpec>,
}

impl Default for PerturbedConfig {
    fn default() -> Self {
        Self {
            k: 15,
            sigma: 0.7,
            seed: 42,
            inner: Box::new(DecoderSpec::PyMatching(PyMatchingConfig {
                correlated: true,
                error_probability: None,
            })),
        }
    }
}

/// Fast correlated Fusion Blossom perturbation options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerturbedFusionBlossomConfig {
    pub k: usize,
    pub sigma: f64,
    pub seed: u64,
}

impl Default for PerturbedFusionBlossomConfig {
    fn default() -> Self {
        Self {
            k: 5,
            sigma: 0.5,
            seed: 42,
        }
    }
}

/// Beam-search windowed decoder construction options.
#[derive(Clone, Debug, PartialEq)]
pub struct BeamSearchConfig {
    pub beam_width: usize,
    pub perturbation_sigma: f64,
    pub seed: u64,
    pub step_size: usize,
    pub buffer_size: usize,
    pub commit_weight_max: f64,
    pub phase2: Box<DecoderSpec>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            perturbation_sigma: 0.5,
            seed: 42,
            step_size: 0,
            buffer_size: 0,
            commit_weight_max: 0.0,
            phase2: Box::new(DecoderSpec::PyMatching(PyMatchingConfig {
                correlated: true,
                error_probability: None,
            })),
        }
    }
}

/// Voting ensemble construction options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnsembleConfig {
    pub members: Vec<DecoderSpec>,
}
