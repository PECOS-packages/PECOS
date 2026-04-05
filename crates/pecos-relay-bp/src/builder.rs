//! Builder patterns for Relay BP and min-sum BP decoders

use crate::config::{MinSumConfig, RelayConfig, StoppingCriterion};
use crate::decoder::{MinSumBpDecoder, RelayBpDecoder};
use crate::errors::Result;
use ndarray::ArrayView2;

/// Builder for `RelayBpDecoder`
///
/// # Example
///
/// ```rust,ignore
/// let decoder = RelayBpDecoder::builder(&check_matrix.view())
///     .error_priors(&[0.1, 0.1, 0.1])
///     .max_iter(200)
///     .gamma0(Some(0.9))
///     .pre_iter(80)
///     .num_sets(300)
///     .seed(42)
///     .build()?;
/// ```
pub struct RelayBpBuilder<'a> {
    check_matrix: ArrayView2<'a, u8>,
    error_priors: Option<Vec<f64>>,
    max_iter: usize,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    gamma0: Option<f64>,
    pre_iter: usize,
    num_sets: usize,
    set_max_iter: usize,
    gamma_dist_interval: (f64, f64),
    stopping_criterion: StoppingCriterion,
    seed: u64,
}

impl<'a> RelayBpBuilder<'a> {
    /// Create a new builder with the given check matrix
    #[must_use]
    pub fn new(check_matrix: &ArrayView2<'a, u8>) -> Self {
        let relay_defaults = RelayConfig::default();
        Self {
            check_matrix: *check_matrix,
            error_priors: None,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: None,
            pre_iter: relay_defaults.pre_iter,
            num_sets: relay_defaults.num_sets,
            set_max_iter: relay_defaults.set_max_iter,
            gamma_dist_interval: relay_defaults.gamma_dist_interval,
            stopping_criterion: relay_defaults.stopping_criterion,
            seed: relay_defaults.seed,
        }
    }

    /// Set per-error prior probabilities (required)
    #[must_use]
    pub fn error_priors(mut self, priors: &[f64]) -> Self {
        self.error_priors = Some(priors.to_vec());
        self
    }

    /// Set maximum BP iterations (default: 200)
    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set min-sum scaling factor
    #[must_use]
    pub fn alpha(mut self, alpha: Option<f64>) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set per-iteration scaling factor for alpha (default: 1.0)
    #[must_use]
    pub fn alpha_iteration_scaling_factor(mut self, factor: f64) -> Self {
        self.alpha_iteration_scaling_factor = factor;
        self
    }

    /// Set memory BP strength (None = disabled)
    #[must_use]
    pub fn gamma0(mut self, gamma0: Option<f64>) -> Self {
        self.gamma0 = gamma0;
        self
    }

    /// Set initial BP iterations before relay (default: 80)
    #[must_use]
    pub fn pre_iter(mut self, pre_iter: usize) -> Self {
        self.pre_iter = pre_iter;
        self
    }

    /// Set number of relay legs (default: 300)
    #[must_use]
    pub fn num_sets(mut self, num_sets: usize) -> Self {
        self.num_sets = num_sets;
        self
    }

    /// Set max iterations per relay leg (default: 60)
    #[must_use]
    pub fn set_max_iter(mut self, set_max_iter: usize) -> Self {
        self.set_max_iter = set_max_iter;
        self
    }

    /// Set disordered memory strength sampling range (default: (-0.24, 0.66))
    #[must_use]
    pub fn gamma_dist_interval(mut self, interval: (f64, f64)) -> Self {
        self.gamma_dist_interval = interval;
        self
    }

    /// Set stopping criterion (default: `NConv { stop_after: 1 }`)
    #[must_use]
    pub fn stopping_criterion(mut self, criterion: StoppingCriterion) -> Self {
        self.stopping_criterion = criterion;
        self
    }

    /// Set random seed (default: 0)
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the Relay BP decoder
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::Configuration`] if error priors are not set.
    /// Returns [`RelayBpError::InvalidMatrix`] if the check matrix is invalid.
    pub fn build(self) -> Result<RelayBpDecoder> {
        let error_priors = self.error_priors.ok_or_else(|| {
            crate::errors::RelayBpError::Configuration("error_priors must be set".to_string())
        })?;

        let ms_config = MinSumConfig {
            error_priors,
            max_iter: self.max_iter,
            alpha: self.alpha,
            alpha_iteration_scaling_factor: self.alpha_iteration_scaling_factor,
            gamma0: self.gamma0,
        };

        let relay_config = RelayConfig {
            pre_iter: self.pre_iter,
            num_sets: self.num_sets,
            set_max_iter: self.set_max_iter,
            gamma_dist_interval: self.gamma_dist_interval,
            stopping_criterion: self.stopping_criterion,
            seed: self.seed,
        };

        RelayBpDecoder::new(&self.check_matrix, &ms_config, &relay_config)
    }
}

/// Builder for `MinSumBpDecoder`
///
/// # Example
///
/// ```rust,ignore
/// let decoder = MinSumBpDecoder::builder(&check_matrix.view())
///     .error_priors(&[0.1, 0.1, 0.1])
///     .max_iter(200)
///     .alpha(Some(0.0))
///     .build()?;
/// ```
pub struct MinSumBpBuilder<'a> {
    check_matrix: ArrayView2<'a, u8>,
    error_priors: Option<Vec<f64>>,
    max_iter: usize,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    gamma0: Option<f64>,
}

impl<'a> MinSumBpBuilder<'a> {
    /// Create a new builder with the given check matrix
    #[must_use]
    pub fn new(check_matrix: &ArrayView2<'a, u8>) -> Self {
        Self {
            check_matrix: *check_matrix,
            error_priors: None,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: None,
        }
    }

    /// Set per-error prior probabilities (required)
    #[must_use]
    pub fn error_priors(mut self, priors: &[f64]) -> Self {
        self.error_priors = Some(priors.to_vec());
        self
    }

    /// Set maximum BP iterations (default: 200)
    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set min-sum scaling factor
    #[must_use]
    pub fn alpha(mut self, alpha: Option<f64>) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set per-iteration scaling factor for alpha (default: 1.0)
    #[must_use]
    pub fn alpha_iteration_scaling_factor(mut self, factor: f64) -> Self {
        self.alpha_iteration_scaling_factor = factor;
        self
    }

    /// Set memory BP strength (None = disabled)
    #[must_use]
    pub fn gamma0(mut self, gamma0: Option<f64>) -> Self {
        self.gamma0 = gamma0;
        self
    }

    /// Build the min-sum BP decoder
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::Configuration`] if error priors are not set.
    /// Returns [`RelayBpError::InvalidMatrix`] if the check matrix is invalid.
    pub fn build(self) -> Result<MinSumBpDecoder> {
        let error_priors = self.error_priors.ok_or_else(|| {
            crate::errors::RelayBpError::Configuration("error_priors must be set".to_string())
        })?;

        let config = MinSumConfig {
            error_priors,
            max_iter: self.max_iter,
            alpha: self.alpha,
            alpha_iteration_scaling_factor: self.alpha_iteration_scaling_factor,
            gamma0: self.gamma0,
        };

        MinSumBpDecoder::new(&self.check_matrix, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_relay_builder() {
        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let mut decoder = RelayBpBuilder::new(&h.view())
            .error_priors(&[0.1, 0.1, 0.1])
            .pre_iter(40)
            .num_sets(100)
            .seed(42)
            .build()
            .unwrap();

        let syndrome = Array1::from_vec(vec![1u8, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert_eq!(result.decoding.len(), 3);
    }

    #[test]
    fn test_min_sum_builder() {
        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let mut decoder = MinSumBpBuilder::new(&h.view())
            .error_priors(&[0.1, 0.1, 0.1])
            .max_iter(100)
            .build()
            .unwrap();

        let syndrome = Array1::from_vec(vec![0u8, 1]);
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert_eq!(result.decoding.len(), 3);
    }

    #[test]
    fn test_builder_missing_priors() {
        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let result = MinSumBpBuilder::new(&h.view()).build();
        assert!(result.is_err());
    }
}
