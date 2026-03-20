//! Configuration types for Relay BP decoders
//!
//! These types map to relay-bp's internal configuration structs, providing
//! a PECOS-friendly API.

pub use relay_bp::bp::relay::StoppingCriterion;

/// Configuration for the Relay ensemble decoder
///
/// Controls the relay algorithm that runs multiple BP legs with disordered
/// memory strengths for improved convergence on qLDPC codes.
#[derive(Debug, Clone)]
pub struct RelayConfig {
    /// Number of initial BP iterations before relay starts (default: 80)
    pub pre_iter: usize,
    /// Number of relay legs in the ensemble (default: 300)
    pub num_sets: usize,
    /// Maximum iterations per relay leg (default: 60)
    pub set_max_iter: usize,
    /// Range for sampling disordered memory strengths (default: (-0.24, 0.66))
    pub gamma_dist_interval: (f64, f64),
    /// When to stop relay iterations (default: `NConv { stop_after: 1 }`)
    pub stopping_criterion: StoppingCriterion,
    /// Random seed for reproducibility (default: 0)
    pub seed: u64,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            stopping_criterion: StoppingCriterion::NConv { stop_after: 1 },
            seed: 0,
        }
    }
}

impl RelayConfig {
    /// Convert to relay-bp's internal config type
    pub(crate) fn to_relay_config(&self) -> relay_bp::bp::relay::RelayDecoderConfig {
        relay_bp::bp::relay::RelayDecoderConfig {
            pre_iter: self.pre_iter,
            num_sets: self.num_sets,
            set_max_iter: self.set_max_iter,
            gamma_dist_interval: self.gamma_dist_interval,
            explicit_gammas: None,
            stopping_criterion: self.stopping_criterion.clone(),
            logging: false,
            seed: self.seed,
        }
    }
}

/// Configuration for the min-sum BP decoder
///
/// Controls a single instance of min-sum belief propagation, used either
/// standalone or as the inner decoder for the relay ensemble.
#[derive(Debug, Clone)]
pub struct MinSumConfig {
    /// Per-error prior probabilities (required)
    pub error_priors: Vec<f64>,
    /// Maximum number of BP iterations (default: 200)
    pub max_iter: usize,
    /// Min-sum scaling factor (None = no scaling)
    pub alpha: Option<f64>,
    /// Per-iteration scaling factor for alpha (default: 1.0)
    pub alpha_iteration_scaling_factor: f64,
    /// Memory BP strength (None = disabled)
    pub gamma0: Option<f64>,
}

impl MinSumConfig {
    /// Create a new config with the given error priors
    #[must_use]
    pub fn new(error_priors: Vec<f64>) -> Self {
        Self {
            error_priors,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: None,
        }
    }

    /// Convert to relay-bp's internal config type.
    ///
    /// This creates an `ndarray_016::Array1<f64>` (relay-bp's pinned ndarray 0.16),
    /// not the workspace ndarray 0.17. The conversion goes through raw slices
    /// to cross the version boundary.
    pub(crate) fn to_min_sum_config(&self) -> relay_bp::bp::min_sum::MinSumDecoderConfig {
        relay_bp::bp::min_sum::MinSumDecoderConfig {
            error_priors: crate::convert::vec_to_relay_array1_f64(&self.error_priors),
            max_iter: self.max_iter,
            alpha: self.alpha,
            alpha_iteration_scaling_factor: self.alpha_iteration_scaling_factor,
            gamma0: self.gamma0,
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
        }
    }
}
