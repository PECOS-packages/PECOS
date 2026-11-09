//! Configuration for PECOS native Relay-BP and min-sum decoders.

use crate::errors::{RelayBpError, Result};
use pecos_bp::{Alpha, Schedule, StopRule};

/// When to stop the relay ensemble.
#[derive(Clone, PartialEq, Debug)]
pub enum StoppingCriterion {
    /// Stop if leg zero converges; otherwise run all relay legs.
    PreIter,
    /// Stop after this many converged legs, including leg zero.
    NConv { stop_after: usize },
    /// Run all legs and retain the lowest-cost converged correction.
    All,
}

impl Default for StoppingCriterion {
    fn default() -> Self {
        Self::NConv { stop_after: 1 }
    }
}

/// Configuration for the native Relay-BP ensemble.
#[derive(Debug, Clone)]
pub struct RelayConfig {
    /// None means unit scaling; zero selects dynamic scaling.
    pub alpha: Option<f64>,
    /// Finite, strictly positive dynamic alpha iteration scaling factor.
    pub alpha_iteration_scaling_factor: f64,
    /// Leg-zero memory strength; None disables memory in leg zero only.
    /// Relay legs retain their sampled or explicit memory strengths.
    pub gamma0: Option<f64>,
    /// Number of initial BP iterations before relay starts (default: 80)
    pub pre_iter: usize,
    /// Number of relay legs in the ensemble (default: 300)
    pub num_sets: usize,
    /// Maximum iterations per relay leg (default: 60)
    pub set_max_iter: usize,
    /// Range for sampling disordered memory strengths (default: (-0.24, 0.66))
    pub gamma_dist_interval: (f64, f64),
    /// Optional explicit gamma sets, with one row per set and one value per variable.
    ///
    /// Public row zero is used by the first relay leg. Sets are reused
    /// cyclically when fewer sets than relay legs are supplied. `None` samples
    /// strengths from `gamma_dist_interval`.
    pub explicit_gammas: Option<Vec<Vec<f64>>>,
    /// When to stop relay iterations (default: `NConv { stop_after: 1 }`)
    pub stopping_criterion: StoppingCriterion,
    /// Random seed for reproducibility (default: 0)
    pub seed: u64,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(crate::DEFAULT_GAMMA0),
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            explicit_gammas: None,
            stopping_criterion: StoppingCriterion::default(),
            seed: 0,
        }
    }
}

impl RelayConfig {
    pub(crate) fn to_native(&self, num_columns: usize) -> Result<pecos_bp::RelayConfig> {
        let alpha = native_alpha(self.alpha, self.alpha_iteration_scaling_factor)?;
        if self.pre_iter == 0 && self.num_sets == 0 {
            return Err(RelayBpError::Configuration(
                "pre_iter must be positive when num_sets is 0, got pre_iter=0, num_sets=0".into(),
            ));
        }
        if self.num_sets > 0 && self.set_max_iter == 0 {
            return Err(RelayBpError::Configuration(format!(
                "set_max_iter must be positive when num_sets > 0, got set_max_iter=0, num_sets={}",
                self.num_sets
            )));
        }
        if self.num_sets == usize::MAX {
            return Err(RelayBpError::Configuration(format!(
                "num_sets must be less than usize::MAX, got {}",
                self.num_sets
            )));
        }
        let (lo, hi) = self.gamma_dist_interval;
        if !lo.is_finite() || !hi.is_finite() || lo > hi || !(hi - lo).is_finite() {
            return Err(RelayBpError::Configuration(format!(
                "gamma_dist_interval endpoints and span must be finite and ordered, got ({lo}, {hi})"
            )));
        }
        validate_gamma0(self.gamma0)?;
        if let Some(rows) = &self.explicit_gammas {
            if self.num_sets > 0 && rows.is_empty() {
                return Err(RelayBpError::Configuration(format!(
                    "explicit_gammas must contain a row when num_sets > 0, got row count 0, num_sets={}",
                    self.num_sets
                )));
            }
            for (row, gammas) in rows.iter().enumerate() {
                if gammas.len() != num_columns {
                    return Err(RelayBpError::Configuration(format!(
                        "explicit_gammas row {row} must have {num_columns} columns, got column count {}",
                        gammas.len()
                    )));
                }
                for (column, &gamma) in gammas.iter().enumerate() {
                    if !gamma.is_finite() {
                        return Err(RelayBpError::Configuration(format!(
                            "explicit_gammas[{row}][{column}] must be finite, got {gamma}"
                        )));
                    }
                }
            }
        }
        let stop_rule = match self.stopping_criterion {
            StoppingCriterion::PreIter => StopRule::PreIter,
            StoppingCriterion::All => StopRule::Converged(usize::MAX),
            StoppingCriterion::NConv { stop_after: 0 } => {
                return Err(RelayBpError::Configuration(
                    "stopping_criterion.stop_after must be positive, got 0".into(),
                ));
            }
            StoppingCriterion::NConv { stop_after } => StopRule::Converged(stop_after),
        };
        Ok(pecos_bp::RelayConfig {
            schedule: Schedule::Flooding,
            alpha,
            gamma0: self.gamma0.unwrap_or(0.0),
            pre_iterations: self.pre_iter,
            num_legs: self.num_sets,
            leg_iterations: self.set_max_iter,
            gamma_range: self.gamma_dist_interval,
            explicit_gammas: self.explicit_gammas.clone(),
            stop_rule,
        })
    }
}

/// Configuration for one native memory-BP leg, without a relay ensemble.
#[derive(Debug, Clone)]
pub struct MinSumConfig {
    /// One probability per check-matrix column.
    pub error_priors: Vec<f64>,
    /// Maximum iterations, strictly positive (default: 200).
    pub max_iter: usize,
    /// None means unit scaling; zero selects dynamic scaling.
    pub alpha: Option<f64>,
    /// Finite, strictly positive dynamic alpha iteration scaling factor.
    pub alpha_iteration_scaling_factor: f64,
    /// Memory strength. None means exactly plain min-sum BP.
    pub gamma0: Option<f64>,
}

impl MinSumConfig {
    /// Create a config with the given error priors.
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

    /// Validate the tuning parameters.
    ///
    /// # Errors
    /// Returns [`RelayBpError::Configuration`] for invalid alpha, scaling factor,
    /// gamma0 or `max_iter`.
    pub fn validate(&self) -> Result<()> {
        self.validated_alpha().map(|_| ())
    }

    fn validated_alpha(&self) -> Result<Alpha> {
        let alpha = native_alpha(self.alpha, self.alpha_iteration_scaling_factor)?;
        if self.max_iter == 0 {
            return Err(RelayBpError::Configuration(
                "max_iter must be positive, got 0".into(),
            ));
        }
        validate_gamma0(self.gamma0)?;
        Ok(alpha)
    }

    pub(crate) fn to_native(&self) -> Result<pecos_bp::RelayConfig> {
        let alpha = self.validated_alpha()?;
        Ok(pecos_bp::RelayConfig {
            schedule: Schedule::Flooding,
            alpha,
            gamma0: self.gamma0.unwrap_or(0.0),
            num_legs: 0,
            pre_iterations: self.max_iter,
            stop_rule: StopRule::Converged(1),
            ..pecos_bp::RelayConfig::default()
        })
    }
}

fn validate_gamma0(gamma0: Option<f64>) -> Result<()> {
    if let Some(gamma) = gamma0
        && !gamma.is_finite()
    {
        return Err(RelayBpError::Configuration(format!(
            "gamma0 must be finite, got {gamma}"
        )));
    }
    Ok(())
}

fn native_alpha(alpha: Option<f64>, scaling_factor: f64) -> Result<Alpha> {
    if !scaling_factor.is_finite() || scaling_factor <= 0.0 {
        return Err(RelayBpError::Configuration(format!(
            "alpha_iteration_scaling_factor must be finite and positive, got {scaling_factor}"
        )));
    }
    match alpha {
        None => Ok(Alpha::Constant(1.0)),
        Some(0.0) => Ok(Alpha::Dynamic { scaling_factor }),
        Some(value) if value.is_finite() && value > 0.0 => Ok(Alpha::Constant(value)),
        Some(value) => Err(RelayBpError::Configuration(format!(
            "alpha must be finite and non-negative, got {value}"
        ))),
    }
}
