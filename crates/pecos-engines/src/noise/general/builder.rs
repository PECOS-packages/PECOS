use crate::GateType;
use crate::noise::{
    GeneralNoiseModel, NoiseRng, SingleQubitWeightedSampler, TwoQubitWeightedSampler,
};
use std::collections::{BTreeMap, HashSet};

/// Builder for creating general noise models
#[derive(Debug, Clone)]
pub struct GeneralNoiseModelBuilder {
    // global params
    noiseless_gates: Option<HashSet<GateType>>,
    seed: Option<u64>,
    scale: Option<f64>,
    leakage_scale: Option<f64>,
    emission_scale: Option<f64>,
    // idle noise
    p_idle_coherent: Option<bool>,
    p_idle_linear_rate: Option<f64>,
    p_idle_linear_model: Option<SingleQubitWeightedSampler>,
    p_idle_quadratic_rate: Option<f64>,
    p_idle_coherent_to_incoherent_factor: Option<f64>,
    idle_scale: Option<f64>,
    // prep noise
    p_prep: Option<f64>,
    p_prep_leak_ratio: Option<f64>,
    p_prep_crosstalk: Option<f64>,
    prep_scale: Option<f64>,
    p_prep_crosstalk_scale: Option<f64>,
    // single-qubit gate noise
    p1: Option<f64>,
    p1_emission_ratio: Option<f64>,
    p1_emission_model: Option<SingleQubitWeightedSampler>,
    p1_seepage_prob: Option<f64>,
    p1_pauli_model: Option<SingleQubitWeightedSampler>,
    p1_scale: Option<f64>,
    // two-qubit gate noise
    p2: Option<f64>,
    p2_angle_params: Option<(f64, f64, f64, f64)>,
    p2_angle_power: Option<f64>,
    p2_emission_ratio: Option<f64>,
    p2_emission_model: Option<TwoQubitWeightedSampler>,
    p2_seepage_prob: Option<f64>,
    p2_pauli_model: Option<TwoQubitWeightedSampler>,
    p2_idle: Option<f64>,
    p2_scale: Option<f64>,
    // measurement noise
    p_meas_0: Option<f64>,
    p_meas_1: Option<f64>,
    p_meas_crosstalk: Option<f64>,
    meas_scale: Option<f64>,
    p_meas_crosstalk_scale: Option<f64>,
}

impl Default for GeneralNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneralNoiseModelBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            // global params
            noiseless_gates: None,
            seed: None,
            scale: None,
            leakage_scale: None,
            emission_scale: None,
            // idle noise
            p_idle_linear_rate: None,
            p_idle_linear_model: None,
            p_idle_quadratic_rate: None,
            p_idle_coherent: None,
            p_idle_coherent_to_incoherent_factor: None,
            idle_scale: None,
            // prep noise
            p_prep: None,
            p_prep_leak_ratio: None,
            p_prep_crosstalk: None,
            prep_scale: None,
            p_prep_crosstalk_scale: None,
            // single-qubit gate noise
            p1: None,
            p1_emission_ratio: None,
            p1_emission_model: None,
            p1_seepage_prob: None,
            p1_pauli_model: None,
            p1_scale: None,
            // two-qubit gate noise
            p2: None,
            p2_angle_params: None,
            p2_angle_power: None,
            p2_emission_ratio: None,
            p2_emission_model: None,
            p2_seepage_prob: None,
            p2_pauli_model: None,
            p2_idle: None,
            p2_scale: None,
            // measurement noise
            p_meas_0: None,
            p_meas_1: None,
            p_meas_crosstalk: None,
            meas_scale: None,
            p_meas_crosstalk_scale: None,
        }
    }

    /// Build the general noise model
    ///
    /// TODO: Consider another build with noiseless default
    ///
    /// # Returns
    /// A `GeneralNoiseModel`
    ///
    /// # Panics
    /// Panics if any probabilities are not set or are not between 0 and 1.
    #[must_use]
    pub fn build(mut self) -> GeneralNoiseModel {
        // Start with the default noise model as a base
        let mut model = GeneralNoiseModel::default();

        // global params
        // -----------------------------------------------------------------------------------------
        if let Some(gates) = self.noiseless_gates.clone() {
            for gate in gates {
                model.add_noiseless_gate(gate);
            }
        }

        if let Some(seed) = self.seed {
            // Use the with_seed constructor for NoiseRng
            model.rng = NoiseRng::with_seed(seed);
        }

        if let Some(leakage_scale) = self.leakage_scale {
            model.leakage_scale = leakage_scale;
        }

        // idle noise
        // -----------------------------------------------------------------------------------------
        if let Some(coherent) = self.p_idle_coherent {
            model.p_idle_coherent = coherent;
        }

        if let Some(p_idle_linear_rate) = self.p_idle_linear_rate {
            model.p_idle_linear_rate = p_idle_linear_rate;
        }

        if let Some(model_map) = self.p_idle_linear_model.clone() {
            model.p_idle_linear_model = model_map;
        }

        if let Some(p_idle_quadratic_rate) = self.p_idle_quadratic_rate {
            model.p_idle_quadratic_rate = p_idle_quadratic_rate;
        }

        if let Some(factor) = self.p_idle_coherent_to_incoherent_factor {
            model.p_idle_coherent_to_incoherent_factor = factor;
        }

        // prep noise
        // -----------------------------------------------------------------------------------------
        if let Some(p_prep) = self.p_prep {
            model.p_prep = p_prep;
        }
        if let Some(ratio) = self.p_prep_leak_ratio {
            model.p_prep_leak_ratio = ratio;
        }
        if let Some(prob) = self.p_prep_crosstalk {
            model.p_prep_crosstalk = prob;
        }

        //single-qubit gate noise
        // -----------------------------------------------------------------------------------------
        if let Some(p1) = self.p1 {
            model.p1 = p1;
        }

        if let Some(ratio) = self.p1_emission_ratio {
            model.p1_emission_ratio = ratio;
        }

        if let Some(model_map) = self.p1_emission_model.clone() {
            model.p1_emission_model = model_map;
        }

        if let Some(prob) = self.p1_seepage_prob {
            model.p1_seepage_prob = prob;
        }

        if let Some(model_map) = self.p1_pauli_model.clone() {
            model.p1_pauli_model = model_map;
        }

        // two-qubit gate noise
        // -----------------------------------------------------------------------------------------
        if let Some(p2) = self.p2 {
            model.p2 = p2;
        }

        if let Some(p2_angle_params) = self.p2_angle_params {
            model.p2_angle_a = p2_angle_params.0;
            model.p2_angle_b = p2_angle_params.1;
            model.p2_angle_c = p2_angle_params.2;
            model.p2_angle_d = p2_angle_params.3;
        }

        if let Some(power) = self.p2_angle_power {
            model.p2_angle_power = power;
        }
        if let Some(ratio) = self.p2_emission_ratio {
            model.p2_emission_ratio = ratio;
        }

        if let Some(model_map) = self.p2_emission_model.clone() {
            model.p2_emission_model = model_map;
        }

        if let Some(prob) = self.p2_seepage_prob {
            model.p2_seepage_prob = prob;
        }

        if let Some(model_map) = self.p2_pauli_model.clone() {
            model.p2_pauli_model = model_map;
        }

        if let Some(p2_idle) = self.p2_idle {
            model.p2_idle = p2_idle;
        }

        // measurement noise
        // -----------------------------------------------------------------------------------------
        if let Some(p_meas_0) = self.p_meas_0 {
            model.p_meas_0 = p_meas_0;
        }

        if let Some(p_meas_1) = self.p_meas_1 {
            model.p_meas_1 = p_meas_1;
        }

        model.p_meas_max = model.p_meas_0.max(model.p_meas_1);

        if let Some(prob) = self.p_meas_crosstalk {
            model.p_meas_crosstalk = prob;
        }

        // scale
        // -----------------------------------------------------------------------------------------
        self.scale_parameters(&mut model);
        model
    }

    // ========================================================================================== //
    // with global params
    // ========================================================================================== //

    /// Add a gate type to the set of noiseless gates
    #[must_use]
    pub fn with_noiseless_gate(mut self, gate_type: GateType) -> Self {
        if self.noiseless_gates.is_none() {
            self.noiseless_gates = Some(HashSet::new());
        }

        if let Some(ref mut gates) = self.noiseless_gates {
            gates.insert(gate_type);
        }

        self
    }

    /// Set the seed for the random number generator
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the overall scaling factor for error probabilities
    ///
    /// A global multiplier applied to all error rates. This allows easy adjustment of the
    /// overall noise level without changing individual parameters. Typically used to
    /// simulate different device qualities or to study the effect of noise strength.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Set the scaling factor for leakage errors
    ///
    /// Scales how much leakage is applied and instead is replaced by completely depolarizing noise.
    /// 1.0 means all leakage events are applied as leakage. 0.0 means all leakage events are
    /// replaced by completely depolarizing noise.
    #[must_use]
    pub fn with_leakage_scale(mut self, scale: f64) -> Self {
        self.leakage_scale = Some(Self::validate_probability(scale));
        self
    }

    /// Set the scaling factor for spontaneous emission errors
    ///
    /// Multiplier for spontaneous-emission-related error probabilities. Controls the relative
    /// strength of errors that involve transitions outside the standard computational basis.
    #[must_use]
    pub fn with_emission_scale(mut self, scale: f64) -> Self {
        self.emission_scale = Some(scale);
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_seepage_prob(mut self, prob: f64) -> Self {
        self.p1_seepage_prob = Some(Self::validate_probability(prob));
        self.p2_seepage_prob = Some(Self::validate_probability(prob));
        self
    }

    // --- idle noise --- //

    /// Set whether to use coherent dephasing
    #[must_use]
    pub fn with_p_idle_coherent(mut self, use_coherent: bool) -> Self {
        self.p_idle_coherent = Some(use_coherent);
        self
    }

    /// Set the idling noise error rate for the linear term
    #[must_use]
    pub fn with_p_idle_linear_rate(mut self, rate: f64) -> Self {
        self.p_idle_linear_rate = Some(Self::validate_non_negative(rate, "linear idling rate"));
        self
    }

    // TODO: See if we should put a average scaling...
    /// Set the average idling noise error rate per channel for the linear term
    #[must_use]
    pub fn with_average_p_idle_linear_rate(mut self, rate: f64) -> Self {
        let rate: f64 = rate * 3.0 / 2.0;
        self.p_idle_linear_rate = Some(rate);
        self
    }

    /// Set the stochastic model for idling that is linearly dependent on time
    #[must_use]
    pub fn with_p_idle_linear_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p_idle_linear_model = Some(SingleQubitWeightedSampler::new(model));
        self
    }

    /// Set the idling noise error rate for the quadratic term
    #[must_use]
    pub fn with_p_idle_quadratic_rate(mut self, rate: f64) -> Self {
        self.p_idle_quadratic_rate = Some(rate);
        self
    }

    /// Set the average idling noise error rate per channel for the quadratic term
    #[must_use]
    pub fn with_average_p_idle_quadratic_rate(mut self, rate: f64) -> Self {
        let rate: f64 = rate * (3.0 / 2.0 as f64).sqrt();
        self.p_idle_quadratic_rate = Some(rate);
        self
    }

    /// Set the coherent-to-incoherent conversion factor
    ///
    /// # Parameters
    /// * `factor` - The conversion factor between coherent and incoherent noise
    #[must_use]
    pub fn with_p_idle_coherent_to_incoherent_factor(mut self, factor: f64) -> Self {
        self.p_idle_coherent_to_incoherent_factor = Some(Self::validate_positive(
            factor,
            "Coherent-to-incoherent factor",
        ));
        self
    }

    /// Set the scaling factor for idle noise
    ///
    /// Controls the strength of errors that occur during idle periods or memory operations.
    /// In ion trap systems, this could represent heating or dephasing during storage times.
    #[must_use]
    pub fn with_idle_scale(mut self, scale: f64) -> Self {
        self.idle_scale = Some(scale);
        self
    }

    // ========================================================================================== //
    // prep noise
    // ========================================================================================== //

    /// Set the probability of error during preparation
    #[must_use]
    pub fn with_prep_probability(mut self, probability: f64) -> Self {
        self.p_prep = Some(Self::validate_probability(probability));
        self
    }

    /// Set the preparation leakage ratio
    #[must_use]
    pub fn with_prep_leak_ratio(mut self, ratio: f64) -> Self {
        self.p_prep_leak_ratio = Some(Self::validate_probability(ratio));
        self
    }

    /// Set the probability of crosstalk during initialization operations
    #[must_use]
    pub fn with_p_prep_crosstalk(mut self, prob: f64) -> Self {
        self.p_prep_crosstalk = Some(Self::validate_probability(prob));
        self
    }

    /// Set the scaling factor for initialization errors
    ///
    /// Multiplier for preparation error probabilities. Allows adjustment of the relative
    /// strength of initialization errors compared to other error types.
    #[must_use]
    pub fn with_prep_scale(mut self, scale: f64) -> Self {
        self.prep_scale = Some(scale);
        self
    }

    /// Set the scaling factor for initialization crosstalk probability
    ///
    /// Additional scaling factor specifically for initialization crosstalk probability.
    #[must_use]
    pub fn with_p_prep_crosstalk_scale(mut self, scale: f64) -> Self {
        self.p_prep_crosstalk_scale = Some(Self::validate_non_negative(
            scale,
            "Preparation crosstalk rescale factor",
        ));
        self
    }

    // ========================================================================================== //
    // single-qubit gate noise
    // ========================================================================================== //

    /// Set the probability of error after single-qubit gates
    #[must_use]
    pub fn with_p1_probability(mut self, probability: f64) -> Self {
        self.p1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the average probability of error after single-qubit gates
    ///
    /// Rescaling from average error to total error
    ///
    /// This conversion is necessary because experiments report average error rates,
    /// but our noise models use total error rates.
    ///
    /// For a single-qubit gate with uniform error distribution across 3 Pauli errors,
    /// the ratio of total error rate to average error rate is 3/2.
    #[must_use]
    pub fn with_average_p1_probability(mut self, probability: f64) -> Self {
        self.p1 = Some(Self::validate_probability(probability * 3.0 / 2.0));
        self
    }

    /// Set the emission ratio for single-qubit gate errors
    #[must_use]
    pub fn with_p1_emission_ratio(mut self, ratio: f64) -> Self {
        self.p1_emission_ratio = Some(Self::validate_probability(ratio));
        self
    }

    /// Set the emission error model for single-qubit gates
    #[must_use]
    pub fn with_p1_emission_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p1_emission_model = Some(SingleQubitWeightedSampler::new(model));
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_p1_seepage_prob(mut self, prob: f64) -> Self {
        self.p1_seepage_prob = Some(Self::validate_probability(prob));
        self
    }

    /// Set the Pauli error model for single-qubit gates
    #[must_use]
    pub fn with_p1_pauli_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p1_pauli_model = Some(SingleQubitWeightedSampler::new(model));
        self
    }

    /// Set the scaling factor for single-qubit gate errors
    ///
    /// Multiplier for single-qubit gate error probabilities. Allows adjustment of the
    /// relative strength of single-qubit gate errors compared to other error types.
    #[must_use]
    pub fn with_p1_scale(mut self, scale: f64) -> Self {
        self.p1_scale = Some(scale);
        self
    }

    // ========================================================================================== //
    // two-qubit gate noise
    // ========================================================================================== //

    /// Set the probability of error after two-qubit gates
    #[must_use]
    pub fn with_p2_probability(mut self, probability: f64) -> Self {
        self.p2 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of error after two-qubit gates
    ///
    /// Rescaling from average error to total error
    ///
    /// This conversion is necessary because experiments report average error rates,
    /// but our noise models use total error rates.
    ///
    /// For a two-qubit gate with uniform error distribution across 15 Pauli errors,
    /// the ratio of total error rate to average error rate is 5/4.
    #[must_use]
    pub fn with_average_p2_probability(mut self, probability: f64) -> Self {
        self.p2 = Some(Self::validate_probability(probability * 5.0 / 4.0));
        self
    }

    /// Set RZZ parameter scaling for angle dependent error.
    ///
    /// The PECOS gate set has a parameterized-angle ZZ gate, RZZ(θ). For implementation
    /// Certain parameters relate to the strength of the asymmetric
    /// depolarizing noise. These parameters depend on the angle θ and are normalized so that
    /// θ = π/2 gives the 2-qubit fault probability (p2).
    ///
    /// The parameters for asymmetric depolarizing noise are fit parameters that model how the
    /// noise changes as the angle θ changes according to these equations:
    ///
    /// For θ < 0:
    ///     (`p2_angle_a` × (|`θ|/π)^p2_angle_power` + `p2_angle_b`) × p2
    ///
    /// For θ > 0:
    ///     (`p2_angle_c` × (|`θ|/π)^p2_angle_power` + `p2_angle_d`) × p2
    ///
    /// For θ = 0:
    ///     (`p2_angle_b` + `p2_angle_d`) × 0.5 × p2
    ///
    /// # Parameters
    /// * `a` - Coefficient for scaling negative angles (`p2_angle_a`)
    /// * `b` - Offset for negative angles (`p2_angle_b`)
    /// * `c` - Coefficient for scaling positive angles (`p2_angle_c`)
    /// * `d` - Offset for positive angles (`p2_angle_d`)
    #[must_use]
    pub fn with_p2_angle_params(mut self, a: f64, b: f64, c: f64, d: f64) -> Self {
        self.p2_angle_params = Some((a, b, c, d));
        self
    }

    /// Set power parameter for RZZ error scaling
    ///
    /// # Parameters
    /// * `power` - The power to which theta is raised in the RZZ error rate formula
    #[must_use]
    pub fn with_p2_angle_power(mut self, power: f64) -> Self {
        self.p2_angle_power = Some(Self::validate_positive(power, "RZZ power parameter"));
        self
    }

    /// Set the two-qubit emission ratio
    #[must_use]
    pub fn with_p2_emission_ratio(mut self, ratio: f64) -> Self {
        self.p2_emission_ratio = Some(Self::validate_probability(ratio));
        self
    }

    /// Set the probability model for two-qubit emission errors
    #[must_use]
    pub fn with_p2_emission_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p2_emission_model = Some(TwoQubitWeightedSampler::new(model));
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_p2_seepage_prob(mut self, prob: f64) -> Self {
        self.p2_seepage_prob = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability model for two-qubit Pauli errors
    #[must_use]
    pub fn with_p2_pauli_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p2_pauli_model = Some(TwoQubitWeightedSampler::new(model));
        self
    }

    #[must_use]
    pub fn with_p2_idle(mut self, probability: f64) -> Self {
        self.p2_idle = Some(Self::validate_probability(probability));
        self
    }

    /// Set the scaling factor for two-qubit gate errors
    ///
    /// Multiplier for two-qubit gate error probabilities. Allows adjustment of the relative
    /// strength of two-qubit gate errors compared to other error types. In most quantum
    /// technologies, two-qubit gates are typically more error-prone than single-qubit gates.
    #[must_use]
    pub fn with_p2_scale(mut self, scale: f64) -> Self {
        self.p2_scale = Some(scale);
        self
    }

    // ========================================================================================== //
    // measurement noise
    // ========================================================================================== //

    /// Set the probability of flipping 0 to 1 during measurement
    #[must_use]
    pub fn with_meas_0_probability(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of flipping 1 to 0 during measurement
    #[must_use]
    pub fn with_meas_1_probability(mut self, probability: f64) -> Self {
        self.p_meas_1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of bit flipping the measurement result
    #[must_use]
    pub fn with_meas_probability(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(Self::validate_probability(probability));
        self.p_meas_1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of crosstalk during measurement operations
    #[must_use]
    pub fn with_p_meas_crosstalk(mut self, prob: f64) -> Self {
        self.p_meas_crosstalk = Some(Self::validate_probability(prob));
        self
    }

    /// Set the scaling factor for measurement faults
    ///
    /// Multiplier for measurement error probabilities. Allows adjustment of the relative
    /// strength of readout errors compared to other error types.
    #[must_use]
    pub fn with_meas_scale(mut self, scale: f64) -> Self {
        self.meas_scale = Some(scale);
        self
    }

    /// Set the scaling factor for measurement crosstalk probability
    ///
    /// Additional scaling factor specifically for measurement crosstalk probability.
    #[must_use]
    pub fn with_p_meas_crosstalk_scale(mut self, scale: f64) -> Self {
        self.p_meas_crosstalk_scale = Some(Self::validate_non_negative(
            scale,
            "Measurement crosstalk rescale factor",
        ));
        self
    }

    // ========================================================================================== //
    // validation
    // ========================================================================================== //

    /// Validate that a value is a valid probability (between 0 and 1)
    fn validate_probability(prob: f64) -> f64 {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0 and 1, got {prob}"
        );
        prob
    }

    /// Validate that a value is positive
    fn validate_positive(value: f64, name: &str) -> f64 {
        assert!(value > 0.0, "{name} must be positive, got {value}");
        value
    }

    /// Validate that a value is non-negative
    fn validate_non_negative(value: f64, name: &str) -> f64 {
        assert!(value >= 0.0, "{name} must be non-negative, got {value}");
        value
    }

    // ========================================================================================== //
    // scaling
    // ========================================================================================== //

    /// Scale error probabilities based on scaling factors
    ///
    /// This method applies all scaling factors to the error probabilities:
    /// - Global scale factor
    /// - Type-specific scale factors (measurement, preparation, memory, etc.)
    /// - Conversion factors from average to total error rates (3/2 for p1, 5/4 for p2)
    ///
    /// This method should be called exactly once after setting all parameters
    /// and before using the noise model for simulation. Calling it multiple times will
    /// compound the scaling factors incorrectly.
    pub fn scale_parameters(&mut self, model: &mut GeneralNoiseModel) {
        // Note, leakage_scale is not included here as it is used as an active parameter in the
        // noise model
        let scale = self.scale.unwrap_or(1.0);
        let idle_scale = self.idle_scale.unwrap_or(1.0);
        let prep_scale = self.prep_scale.unwrap_or(1.0);
        let meas_scale = self.meas_scale.unwrap_or(1.0);
        let p1_scale = self.p1_scale.unwrap_or(1.0);
        let p2_scale = self.p2_scale.unwrap_or(1.0);
        let emission_scale = self.emission_scale.unwrap_or(1.0);
        let p_meas_crosstalk_scale = self.p_meas_crosstalk_scale.unwrap_or(1.0);
        let p_prep_crosstalk_scale = self.p_prep_crosstalk_scale.unwrap_or(1.0);

        // Scale single-qubit gate error probability
        model.p1 *= p1_scale * scale;

        // Scale two-qubit gate error probability
        model.p2 *= p2_scale * scale;

        model.p_meas_0 *= meas_scale * scale;
        model.p_meas_1 *= meas_scale * scale;

        // Scale preparation error probability
        model.p_prep *= prep_scale * scale;

        // Scale preparation leakage ratio - include the global scale factor
        model.p_prep_leak_ratio *= scale;
        model.p_prep_leak_ratio = model.p_prep_leak_ratio.min(1.0);

        // Apply crosstalk rescaling factors
        model.p_meas_crosstalk *= p_meas_crosstalk_scale;
        model.p_prep_crosstalk *= p_prep_crosstalk_scale;

        // Then apply the regular scaling to crosstalks
        model.p_meas_crosstalk *= meas_scale * scale;
        model.p_prep_crosstalk *= prep_scale * scale;

        // Scale emission ratios
        model.p1_emission_ratio *= emission_scale * scale;
        model.p1_emission_ratio = model.p1_emission_ratio.min(1.0);

        model.p2_emission_ratio *= emission_scale * scale;
        model.p2_emission_ratio = model.p2_emission_ratio.min(1.0);

        model.p_idle_quadratic_rate *= (idle_scale * scale).sqrt();

        // If we need to do incoherent noise instead of coherent
        if !model.p_idle_coherent {
            // 0.5 to deal with the 0.5 in sin(rate x duration x 0.5)^2
            let factor = model.p_idle_coherent_to_incoherent_factor * 0.5;
            model.p_idle_quadratic_rate *= factor;
        }
        // frequency is in units of 2pi so convert to radians
        model.p_idle_quadratic_rate *= 2.0 * std::f64::consts::PI;

        model.p_idle_linear_rate = model.p_idle_linear_rate * scale * idle_scale;
        model.p2_idle = Self::validate_probability(model.p2_idle * scale * idle_scale);
    }
}
