//! Integration between the extensible gate system and noise model.
//!
//! This module provides:
//! - `GateIdNoiseConfig`: O(1) indexed noise config by `GateId`
//! - `DecompositionNoiseStrategy`: How to apply noise when gates are decomposed
//! - Integration helpers for the existing noise system

use super::{GateCategory, GateId, GateSpec, gates};
use crate::noise::validation::validate_distribution;
use crate::noise::validation::{ValidationResult, validate_probability};

/// Noise configuration for a specific gate.
#[derive(Clone, Debug)]
pub struct GateNoiseParams {
    /// Error probability per qubit.
    pub(crate) error_probability: f64,
    /// Optional gate duration (for idle noise calculations).
    pub(crate) duration_ns: Option<f64>,
    /// Pauli weights for depolarizing noise [px, py, pz].
    /// If None, uniform depolarizing is used.
    pub(crate) pauli_weights: Option<[f64; 3]>,
}

impl Default for GateNoiseParams {
    fn default() -> Self {
        Self {
            error_probability: 0.0,
            duration_ns: None,
            pauli_weights: None,
        }
    }
}

impl GateNoiseParams {
    /// Return the configured `error_probability`.
    #[must_use]
    pub fn error_probability(&self) -> f64 {
        self.error_probability
    }

    /// Return the configured `duration_ns`.
    #[must_use]
    pub fn duration_ns(&self) -> Option<f64> {
        self.duration_ns
    }

    /// Return the configured `pauli_weights`.
    #[must_use]
    pub fn pauli_weights(&self) -> Option<[f64; 3]> {
        self.pauli_weights
    }

    /// Create noise parameters with a given error probability.
    ///
    /// # Errors
    /// Returns an error naming an invalid probability or sampling weight and its remedy.
    pub fn with_error(error_probability: f64) -> ValidationResult<Self> {
        validate_probability(error_probability, "error_probability")?;
        Ok(Self {
            error_probability,
            ..Default::default()
        })
    }

    /// Set the gate duration.
    ///
    /// # Errors
    /// Returns an error if the duration is outside the finite `TimeUnits` domain.
    pub fn with_duration(
        mut self,
        duration_ns: f64,
    ) -> crate::noise::validation::ValidationResult<Self> {
        self.duration_ns = Some(crate::noise::validation::validate_duration(
            duration_ns,
            "duration_ns",
        )?);
        Ok(self)
    }

    /// Set custom Pauli weights.
    ///
    /// # Errors
    /// Rejects invalid Pauli weights or a sum other than one.
    pub fn with_pauli_weights(
        mut self,
        px: f64,
        py: f64,
        pz: f64,
    ) -> crate::noise::validation::ValidationResult<Self> {
        validate_distribution(&[px, py, pz], "pauli_weights")?;
        self.pauli_weights = Some([px, py, pz]);
        Ok(self)
    }

    /// Check if this is noiseless.
    #[must_use]
    pub fn is_noiseless(&self) -> bool {
        self.error_probability == 0.0
    }
}

/// Convert `GateCategory` to index for array lookup.
fn category_to_index(category: GateCategory) -> usize {
    match category {
        GateCategory::SingleQubitUnitary => 0,
        GateCategory::TwoQubitUnitary => 1,
        GateCategory::MultiQubitUnitary => 2,
        GateCategory::Preparation => 3,
        GateCategory::Measurement => 4,
        GateCategory::Idle => 5,
        GateCategory::QubitManagement => 6,
        GateCategory::Custom(_) => 7,
    }
}

/// O(1) indexed noise configuration by `GateId`.
///
/// Uses array indexing for core gates (0-255) and sparse storage for user gates.
#[derive(Clone, Debug)]
pub struct GateIdNoiseConfig {
    /// Per-gate noise parameters (indexed by `GateId`).
    core_params: Vec<Option<GateNoiseParams>>,
    /// Default by category when per-gate config is missing.
    category_defaults: [Option<GateNoiseParams>; 8],
    /// Global fallback when no category default exists.
    global_default: Option<GateNoiseParams>,
}

impl Default for GateIdNoiseConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl GateIdNoiseConfig {
    /// Create a new empty noise configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            core_params: vec![None; 256],
            category_defaults: [None, None, None, None, None, None, None, None],
            global_default: None,
        }
    }

    /// Set the global default error probability.
    ///
    /// # Errors
    /// Returns an error naming an invalid probability or sampling weight and its remedy.
    pub fn with_global_default(mut self, error_probability: f64) -> ValidationResult<Self> {
        self.global_default = Some(GateNoiseParams::with_error(error_probability)?);
        Ok(self)
    }

    /// Set the default for a gate category.
    ///
    /// # Errors
    /// Returns an error naming an invalid probability or sampling weight and its remedy.
    pub fn with_category_default(
        mut self,
        category: GateCategory,
        error_probability: f64,
    ) -> ValidationResult<Self> {
        self.category_defaults[category_to_index(category)] =
            Some(GateNoiseParams::with_error(error_probability)?);
        Ok(self)
    }

    /// Set noise parameters for a specific gate.
    pub fn set_gate(&mut self, gate: GateId, params: GateNoiseParams) {
        let idx = gate.0 as usize;
        if idx >= self.core_params.len() {
            self.core_params.resize(idx + 1, None);
        }
        self.core_params[idx] = Some(params);
    }

    /// Set error probability for a specific gate.
    ///
    /// # Errors
    /// Returns an error naming an invalid probability or sampling weight and its remedy.
    pub fn set_gate_error(&mut self, gate: GateId, error_probability: f64) -> ValidationResult<()> {
        self.set_gate(gate, GateNoiseParams::with_error(error_probability)?);
        Ok(())
    }

    /// Get noise parameters for a gate.
    ///
    /// Lookup priority:
    /// 1. Per-gate config
    /// 2. Category default (requires `GateSpec` for category lookup)
    /// 3. Global default
    /// 4. None (noiseless)
    #[must_use]
    pub fn get(&self, gate: GateId, spec: Option<&GateSpec>) -> Option<&GateNoiseParams> {
        // Try per-gate first
        if let Some(Some(params)) = self.core_params.get(gate.0 as usize) {
            return Some(params);
        }

        // Try category default
        if let Some(spec) = spec {
            let idx = category_to_index(spec.category);
            if let Some(params) = &self.category_defaults[idx] {
                return Some(params);
            }
        }

        // Try global default
        self.global_default.as_ref()
    }

    /// Get error probability for a gate.
    #[must_use]
    pub fn get_error_probability(&self, gate: GateId, spec: Option<&GateSpec>) -> f64 {
        self.get(gate, spec).map_or(0.0, |p| p.error_probability)
    }

    /// Mark a gate as noiseless.
    pub fn set_noiseless(&mut self, gate: GateId) {
        self.set_gate(gate, GateNoiseParams::default());
    }

    /// Set typical error rates for single and two-qubit gates.
    ///
    /// # Errors
    /// Returns an error naming an invalid probability or sampling weight and its remedy.
    pub fn with_typical_rates(mut self, p1: f64, p2: f64) -> ValidationResult<Self> {
        validate_probability(p1, "p1")?;
        validate_probability(p2, "p2")?;
        // Single-qubit gates
        for &gate in &[
            gates::I,
            gates::X,
            gates::Y,
            gates::Z,
            gates::H,
            gates::SX,
            gates::SXdg,
            gates::SY,
            gates::SYdg,
            gates::SZ,
            gates::SZdg,
            gates::T,
            gates::Tdg,
            gates::RX,
            gates::RY,
            gates::RZ,
        ] {
            self.set_gate_error(gate, p1)?;
        }

        // Two-qubit gates
        for &gate in &[gates::CX, gates::CY, gates::CZ, gates::SWAP, gates::ISWAP] {
            self.set_gate_error(gate, p2)?;
        }

        Ok(self)
    }
}

/// Validated fraction of high-level noise in a blended decomposition strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HighLevelWeight(f64);

impl HighLevelWeight {
    /// Validate a high-level noise fraction.
    ///
    /// # Errors
    /// Rejects non-finite values and values outside [0, 1].
    pub fn new(weight: f64) -> ValidationResult<Self> {
        Ok(Self(validate_probability(weight, "high_level_weight")?))
    }

    /// Read the validated fraction.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Strategy for applying noise when gates are decomposed.
///
/// When a high-level gate (like SWAP) is decomposed into native gates (CX, CX, CX),
/// this controls how noise is applied.
///
/// A raw weight cannot bypass validation through the public variant:
/// ```compile_fail
/// use pecos_neo::extensible::DecompositionNoiseStrategy;
/// let invalid = DecompositionNoiseStrategy::Blended { high_level_weight: f64::NAN };
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub enum DecompositionNoiseStrategy {
    /// Apply noise to each gate in the decomposition.
    ///
    /// This is the most physically accurate model: each native gate has its
    /// associated error rate applied independently.
    #[default]
    PerGate,

    /// Apply noise only to the high-level gate, skip decomposition noise.
    ///
    /// Use this when the high-level gate has a calibrated error rate that
    /// already accounts for its implementation.
    HighLevel,

    /// Blend high-level and per-gate noise.
    ///
    /// Applies `(1 - weight) * sum(per_gate_errors) + weight * high_level_error`.
    Blended {
        /// Weight for high-level error (0.0 = per-gate only, 1.0 = high-level only).
        high_level_weight: HighLevelWeight,
    },
}

impl PartialEq for DecompositionNoiseStrategy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::PerGate, Self::PerGate) | (Self::HighLevel, Self::HighLevel) => true,
            (
                Self::Blended {
                    high_level_weight: w1,
                },
                Self::Blended {
                    high_level_weight: w2,
                },
            ) => (w1.get() - w2.get()).abs() < f64::EPSILON,
            _ => false,
        }
    }
}

impl DecompositionNoiseStrategy {
    /// Create a blended strategy.
    ///
    /// # Errors
    /// Returns an error if a probability or fraction is outside [0, 1] or NaN.
    pub fn blended(high_level_weight: f64) -> ValidationResult<Self> {
        Ok(Self::Blended {
            high_level_weight: HighLevelWeight::new(high_level_weight)?,
        })
    }

    /// Calculate effective error probability for a decomposed gate.
    ///
    /// # Arguments
    /// * `high_level_error` - Error probability of the high-level gate
    /// * `decomposed_errors` - Error probabilities of each gate in the decomposition
    #[must_use]
    pub fn effective_error(&self, high_level_error: f64, decomposed_errors: &[f64]) -> f64 {
        match self {
            Self::PerGate => {
                // 1 - (1-p1)(1-p2)... ≈ p1 + p2 + ... for small p
                // For accuracy, compute the product of success probabilities
                let success_prob: f64 = decomposed_errors.iter().map(|&p| 1.0 - p).product();
                1.0 - success_prob
            }
            Self::HighLevel => high_level_error,
            Self::Blended { high_level_weight } => {
                let high_level_weight = high_level_weight.get();
                let per_gate_error: f64 = {
                    let success_prob: f64 = decomposed_errors.iter().map(|&p| 1.0 - p).product();
                    1.0 - success_prob
                };
                (1.0 - high_level_weight) * per_gate_error + high_level_weight * high_level_error
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_noise_params_default() {
        let params = GateNoiseParams::default();
        assert!(params.is_noiseless());
        assert!(params.duration_ns.is_none());
        assert!(params.pauli_weights.is_none());
    }

    #[test]
    fn test_gate_noise_params_with_error() {
        let params = GateNoiseParams::with_error(0.01).unwrap();
        assert!(!params.is_noiseless());
        assert_eq!(params.error_probability, 0.01);
    }

    #[test]
    fn test_gate_noise_params_builder() {
        let params = GateNoiseParams::with_error(0.01)
            .unwrap()
            .with_duration(100.0)
            .unwrap()
            .with_pauli_weights(0.5, 0.3, 0.2)
            .unwrap();

        assert_eq!(params.error_probability, 0.01);
        assert_eq!(params.duration_ns, Some(100.0));
        assert_eq!(params.pauli_weights, Some([0.5, 0.3, 0.2]));
    }

    #[test]
    fn test_gate_id_noise_config_new() {
        let config = GateIdNoiseConfig::new();
        assert!(config.get(gates::H, None).is_none());
    }

    #[test]
    fn test_gate_id_noise_config_per_gate() {
        let mut config = GateIdNoiseConfig::new();
        config.set_gate_error(gates::H, 0.001).unwrap();

        let params = config.get(gates::H, None).unwrap();
        assert_eq!(params.error_probability, 0.001);

        // Other gates should not have config
        assert!(config.get(gates::X, None).is_none());
    }

    #[test]
    fn test_gate_id_noise_config_global_default() {
        let config = GateIdNoiseConfig::new().with_global_default(0.01).unwrap();

        // All gates should use global default
        let params = config.get(gates::H, None).unwrap();
        assert_eq!(params.error_probability, 0.01);

        let params = config.get(gates::CX, None).unwrap();
        assert_eq!(params.error_probability, 0.01);
    }

    #[test]
    fn test_gate_id_noise_config_category_default() {
        let config = GateIdNoiseConfig::new()
            .with_category_default(GateCategory::SingleQubitUnitary, 0.001)
            .unwrap()
            .with_category_default(GateCategory::TwoQubitUnitary, 0.01)
            .unwrap();

        // Need GateSpec to use category lookup
        let single_qubit_spec = GateSpec::new("H")
            .with_quantum_arity(1)
            .with_category(GateCategory::SingleQubitUnitary);

        let params = config.get(gates::H, Some(&single_qubit_spec)).unwrap();
        assert_eq!(params.error_probability, 0.001);

        let two_qubit_spec = GateSpec::new("CX")
            .with_quantum_arity(2)
            .with_category(GateCategory::TwoQubitUnitary);

        let params = config.get(gates::CX, Some(&two_qubit_spec)).unwrap();
        assert_eq!(params.error_probability, 0.01);
    }

    #[test]
    fn test_gate_id_noise_config_priority() {
        let mut config = GateIdNoiseConfig::new()
            .with_global_default(0.1)
            .unwrap()
            .with_category_default(GateCategory::SingleQubitUnitary, 0.01)
            .unwrap();

        // Per-gate should override category and global
        config.set_gate_error(gates::H, 0.001).unwrap();

        let single_qubit_spec = GateSpec::new("H")
            .with_quantum_arity(1)
            .with_category(GateCategory::SingleQubitUnitary);

        // H has per-gate config
        let params = config.get(gates::H, Some(&single_qubit_spec)).unwrap();
        assert_eq!(params.error_probability, 0.001);

        // X should use category default
        let params = config.get(gates::X, Some(&single_qubit_spec)).unwrap();
        assert_eq!(params.error_probability, 0.01);

        // Unknown gate with no spec should use global
        let params = config.get(gates::RZ, None).unwrap();
        assert_eq!(params.error_probability, 0.1);
    }

    #[test]
    fn test_gate_id_noise_config_typical_rates() {
        let config = GateIdNoiseConfig::new()
            .with_typical_rates(0.001, 0.01)
            .unwrap();

        assert_eq!(config.get_error_probability(gates::H, None), 0.001);
        assert_eq!(config.get_error_probability(gates::X, None), 0.001);
        assert_eq!(config.get_error_probability(gates::CX, None), 0.01);
        assert_eq!(config.get_error_probability(gates::SWAP, None), 0.01);
    }

    #[test]
    fn test_decomposition_noise_strategy_per_gate() {
        let strategy = DecompositionNoiseStrategy::PerGate;

        // 3 gates with 1% error each
        // Success = 0.99^3 ≈ 0.970299, Error ≈ 0.029701
        let effective = strategy.effective_error(0.03, &[0.01, 0.01, 0.01]);
        assert!((effective - 0.029_701).abs() < 0.0001);
    }

    #[test]
    fn test_decomposition_noise_strategy_high_level() {
        let strategy = DecompositionNoiseStrategy::HighLevel;

        // High-level error should be used directly
        let effective = strategy.effective_error(0.03, &[0.01, 0.01, 0.01]);
        assert_eq!(effective, 0.03);
    }

    #[test]
    fn test_decomposition_noise_strategy_blended() {
        let strategy = DecompositionNoiseStrategy::blended(0.5).unwrap();

        // 50% blend of per-gate (~0.0297) and high-level (0.03)
        let per_gate = 1.0 - 0.99_f64.powi(3);
        let expected = 0.5 * per_gate + 0.5 * 0.03;
        let effective = strategy.effective_error(0.03, &[0.01, 0.01, 0.01]);
        assert!((effective - expected).abs() < 0.0001);
    }

    #[test]
    fn test_decomposition_noise_strategy_blended_weights() {
        // weight 0 = per-gate only
        let strategy0 = DecompositionNoiseStrategy::blended(0.0).unwrap();
        let per_gate = strategy0.effective_error(0.03, &[0.01, 0.01, 0.01]);

        // weight 1 = high-level only
        let strategy1 = DecompositionNoiseStrategy::blended(1.0).unwrap();
        let high_level = strategy1.effective_error(0.03, &[0.01, 0.01, 0.01]);

        assert!((per_gate - 0.029_701).abs() < 0.0001);
        assert_eq!(high_level, 0.03);
    }

    #[test]
    fn test_noiseless_gate() {
        let mut config = GateIdNoiseConfig::new().with_global_default(0.01).unwrap();
        config.set_noiseless(gates::I);

        // I should be noiseless
        assert_eq!(config.get_error_probability(gates::I, None), 0.0);

        // Other gates use default
        assert_eq!(config.get_error_probability(gates::H, None), 0.01);
    }

    #[test]
    fn test_strategy_equality() {
        assert_eq!(
            DecompositionNoiseStrategy::PerGate,
            DecompositionNoiseStrategy::PerGate
        );
        assert_eq!(
            DecompositionNoiseStrategy::HighLevel,
            DecompositionNoiseStrategy::HighLevel
        );
        assert_eq!(
            DecompositionNoiseStrategy::blended(0.5).unwrap(),
            DecompositionNoiseStrategy::blended(0.5).unwrap()
        );
        assert_ne!(
            DecompositionNoiseStrategy::PerGate,
            DecompositionNoiseStrategy::HighLevel
        );
    }
}
