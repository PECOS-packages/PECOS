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
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            explicit_gammas: None,
            stopping_criterion: StoppingCriterion::NConv { stop_after: 1 },
            seed: 0,
        }
    }
}

impl RelayConfig {
    /// Convert to relay-bp's internal config type.
    ///
    /// # Errors
    ///
    /// Returns [`crate::errors::RelayBpError::Configuration`] if the gamma
    /// interval is invalid. Returns [`crate::errors::RelayBpError::InvalidMatrix`]
    /// if an explicit gamma row has the wrong width or contains a non-finite
    /// value, or if relay legs are configured with no explicit gamma rows.
    pub(crate) fn to_relay_config(
        &self,
        num_variables: usize,
    ) -> crate::errors::Result<relay_bp::bp::relay::RelayDecoderConfig> {
        let (lo, hi) = self.gamma_dist_interval;
        let span = hi - lo;
        if !lo.is_finite() || !hi.is_finite() || lo > hi || !span.is_finite() {
            return Err(crate::errors::RelayBpError::Configuration(format!(
                "gamma_dist_interval endpoints and span must be finite and ordered, got ({lo}, {hi})"
            )));
        }
        let constant_interval = span <= 0.0;

        let explicit_gammas = if let Some(gamma_sets) = &self.explicit_gammas {
            if self.num_sets > 0 && gamma_sets.is_empty() {
                return Err(crate::errors::RelayBpError::InvalidMatrix(
                    "Explicit gammas must contain at least one row when relay legs are configured"
                        .to_string(),
                ));
            }
            for (set, gammas) in gamma_sets.iter().enumerate() {
                if gammas.len() != num_variables {
                    return Err(crate::errors::RelayBpError::InvalidMatrix(format!(
                        "Explicit gamma row {set} has {} entries; expected {num_variables}",
                        gammas.len()
                    )));
                }
                if let Some(variable) = gammas.iter().position(|gamma| !gamma.is_finite()) {
                    return Err(crate::errors::RelayBpError::InvalidMatrix(format!(
                        "Explicit gamma row {set}, variable {variable} must be finite"
                    )));
                }
            }
            let row_count = gamma_sets.len();
            let mut data = Vec::new();
            for row in 0..row_count {
                let public_row = (row + row_count - 1) % row_count;
                data.extend(gamma_sets[public_row].iter().copied());
            }
            Some(
                ndarray_016::Array2::from_shape_vec((row_count, num_variables), data).map_err(
                    |error| {
                        crate::errors::RelayBpError::InvalidMatrix(format!(
                            "Failed to create explicit gamma matrix: {error}"
                        ))
                    },
                )?,
            )
        } else if constant_interval && self.num_sets > 0 {
            // Upstream constructs an exclusive Uniform even when it will use
            // explicit gammas, and rand 0.8 rejects equal endpoints. Preserve
            // this wrapper's `lo <= hi` contract by expressing a zero-width
            // distribution as one constant gamma row.
            Some(ndarray_016::Array2::from_elem((1, num_variables), lo))
        } else {
            None
        };
        let upstream_gamma_interval = if constant_interval {
            (0.0, 1.0)
        } else {
            self.gamma_dist_interval
        };

        Ok(relay_bp::bp::relay::RelayDecoderConfig {
            pre_iter: self.pre_iter,
            num_sets: self.num_sets,
            set_max_iter: self.set_max_iter,
            gamma_dist_interval: upstream_gamma_interval,
            explicit_gammas,
            stopping_criterion: self.stopping_criterion.clone(),
            logging: false,
            seed: self.seed,
        })
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
    /// Memory BP strength. `None` disables memory-BP for the entire relay
    /// ensemble, which also renders `gamma_dist_interval`, `num_sets` and the
    /// seed inert; prefer [`crate::DEFAULT_GAMMA0`] unless you specifically
    /// want plain min-sum.
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

    /// Validate min-sum tuning parameters.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if a scaling value is not finite or is negative.
    pub fn validate(&self) -> crate::errors::Result<()> {
        if self
            .alpha
            .is_some_and(|alpha| !alpha.is_finite() || alpha < 0.0)
        {
            return Err(crate::errors::RelayBpError::Configuration(
                "alpha must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::RelayConfig;
    use crate::errors::RelayBpError;

    #[test]
    fn explicit_gammas_cross_the_ndarray_version_boundary() {
        let config = RelayConfig {
            num_sets: 3,
            explicit_gammas: Some(vec![vec![0.1, 0.1], vec![0.9, 0.9]]),
            ..Default::default()
        };
        let converted = config.to_relay_config(2).unwrap();
        let gammas = converted.explicit_gammas.unwrap();

        assert_eq!(gammas.shape(), &[2, 2]);
        // The external crate requests its first relay set with index one, so
        // internal row one must be public row zero.
        assert_eq!(gammas[[0, 0]].to_bits(), 0.9_f64.to_bits());
        assert_eq!(gammas[[0, 1]].to_bits(), 0.9_f64.to_bits());
        assert_eq!(gammas[[1, 0]].to_bits(), 0.1_f64.to_bits());
        assert_eq!(gammas[[1, 1]].to_bits(), 0.1_f64.to_bits());
    }

    #[test]
    fn explicit_gamma_rows_must_match_variable_count() {
        let config = RelayConfig {
            explicit_gammas: Some(vec![vec![0.1]]),
            ..Default::default()
        };

        assert!(matches!(
            config.to_relay_config(2),
            Err(RelayBpError::InvalidMatrix(_))
        ));
    }

    #[test]
    fn configured_relay_legs_require_an_explicit_gamma_row() {
        let config = RelayConfig {
            num_sets: 1,
            explicit_gammas: Some(Vec::new()),
            ..Default::default()
        };

        assert!(matches!(
            config.to_relay_config(2),
            Err(RelayBpError::InvalidMatrix(_))
        ));
    }

    #[test]
    fn explicit_gamma_values_must_be_finite() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let config = RelayConfig {
                explicit_gammas: Some(vec![vec![0.1, value]]),
                ..Default::default()
            };

            assert!(matches!(
                config.to_relay_config(2),
                Err(RelayBpError::InvalidMatrix(_))
            ));
        }
    }

    #[test]
    fn gamma_interval_must_be_ordered() {
        let config = RelayConfig {
            gamma_dist_interval: (0.7, -0.3),
            ..Default::default()
        };

        assert!(matches!(
            config.to_relay_config(2),
            Err(RelayBpError::Configuration(_))
        ));
    }

    #[test]
    fn constant_gamma_interval_avoids_upstream_uniform_panic() {
        let config = RelayConfig {
            num_sets: 2,
            gamma_dist_interval: (0.25, 0.25),
            ..Default::default()
        };
        let converted = config.to_relay_config(2).unwrap();
        let gammas = converted.explicit_gammas.unwrap();

        assert_eq!(converted.gamma_dist_interval.0.to_bits(), 0.0_f64.to_bits());
        assert_eq!(converted.gamma_dist_interval.1.to_bits(), 1.0_f64.to_bits());
        assert_eq!(gammas.shape(), &[1, 2]);
        assert!(
            gammas
                .iter()
                .all(|gamma| gamma.to_bits() == 0.25_f64.to_bits())
        );
    }
}
