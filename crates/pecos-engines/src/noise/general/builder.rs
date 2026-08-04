use crate::GateType;
use crate::noise::{
    CrosstalkWeightedSampler, GeneralNoiseModel, NoiseRng, SingleQubitWeightedSampler,
    TwoQubitWeightedSampler,
};
use std::collections::{BTreeMap, BTreeSet};

/// The plain-Pauli probabilities plus optional angle-dependent two-qubit
/// scaling and the spontaneous-emission ratios, as returned by
/// [`GeneralNoiseModelBuilder::pauli_with_angle_scaling`].
///
/// Layout:
/// `(p_prep, p_meas_0, p_meas_1, p1, p2, angle, p1_emission_ratio, p2_emission_ratio)`
/// where `angle` is `Some((a, b, c, d, power))` when angle scaling is
/// configured. The emission ratios use the model default (zero) when unset, and
/// the emission DISTRIBUTION is required to be the default (uniform Pauli) --
/// custom emission models keep the config out of this subset.
pub type PauliWithAngleScaling = (
    f64,
    f64,
    f64,
    f64,
    f64,
    Option<(f64, f64, f64, f64, f64)>,
    f64,
    f64,
);

/// Builder for creating general noise models
#[derive(Debug, Clone)]
pub struct GeneralNoiseModelBuilder {
    // global params
    noiseless_gates: Option<BTreeSet<GateType>>,
    seed: Option<u64>,
    scale: Option<f64>,
    leakage_scale: Option<f64>,
    emission_scale: Option<f64>,
    // idle noise
    p_idle_coherent: Option<bool>,
    p_idle_linear_rate: Option<f64>,
    p_idle_linear_model: Option<SingleQubitWeightedSampler>,
    p_idle_quadratic_rate: Option<f64>,
    p_idle_sin_squared: Option<(f64, BTreeMap<String, f64>)>,
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
    /// Duration of the idle-noise sites applied after two-qubit gates.
    ///
    /// These sites use the configured linear and quadratic idle mechanisms; this value is not a
    /// standalone error probability.
    idle_after_2q: Option<f64>,
    p2_scale: Option<f64>,
    // measurement noise
    p_meas_0: Option<f64>,
    p_meas_1: Option<f64>,
    meas_scale: Option<f64>,
    p_meas_crosstalk_global: Option<f64>,
    p_meas_crosstalk_local: Option<f64>,
    p_meas_crosstalk_model: Option<CrosstalkWeightedSampler>,
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
            p_idle_sin_squared: None,
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
            idle_after_2q: None,
            p2_scale: None,
            // measurement noise
            p_meas_0: None,
            p_meas_1: None,
            meas_scale: None,
            p_meas_crosstalk_global: None,
            p_meas_crosstalk_local: None,
            p_meas_crosstalk_model: None,
            p_meas_crosstalk_scale: None,
        }
    }

    /// Fill unset parameters with the legacy demonstration preset.
    ///
    /// This preset reproduces the general noise model's historical defaults. It is intended for
    /// demonstrations and is not a calibrated device model. Parameters already set by the caller
    /// are preserved, so explicit setters win whether they appear before or after `auto()`.
    ///
    /// The preset uses preparation, measurement, one-qubit, two-qubit, and linear-idle error rates
    /// of 0.01, 0.01/0.01, 0.001, 0.01, and 0.001 respectively. Its other effects are:
    ///
    /// - `p_prep_leak_ratio = 0.5`: half of preparation faults leak the qubit out of the
    ///   computational subspace.
    /// - `p1_emission_ratio = p2_emission_ratio = 0.5`: half of gate errors take the spontaneous-
    ///   emission branch, which removes the original gate and substitutes a sample from the
    ///   emission model. The preset's uniform emission models contain Pauli keys only, so these
    ///   branches do not cause leakage.
    /// - `p1_seepage_prob = p2_seepage_prob = 0.5`: seepage is attempted only for qubits that are
    ///   already leaked.
    /// - `p_idle_coherent_to_incoherent_factor = 1.5`: stochastic quadratic idle dephasing is
    ///   inflated by 50% relative to the exact Pauli twirl.
    #[must_use]
    pub fn auto(mut self) -> Self {
        self.p_prep.get_or_insert(0.01);
        self.p_meas_0.get_or_insert(0.01);
        self.p_meas_1.get_or_insert(0.01);
        self.p1.get_or_insert(0.001);
        self.p2.get_or_insert(0.01);
        self.p_idle_linear_rate.get_or_insert(0.001);
        self.p1_emission_ratio.get_or_insert(0.5);
        self.p2_emission_ratio.get_or_insert(0.5);
        self.p_prep_leak_ratio.get_or_insert(0.5);
        self.p1_seepage_prob.get_or_insert(0.5);
        self.p2_seepage_prob.get_or_insert(0.5);
        self.p_idle_coherent_to_incoherent_factor.get_or_insert(1.5);
        self
    }

    /// Build the general noise model
    ///
    /// # Returns
    /// A `GeneralNoiseModel`
    ///
    /// # Panics
    /// Panics if any probabilities are not set or are not between 0 and 1.
    #[must_use]
    pub fn build(mut self) -> GeneralNoiseModel {
        self.validate_configuration()
            .unwrap_or_else(|message| panic!("{message}"));

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

        if let Some((rate, sine_model)) = self.p_idle_sin_squared.clone() {
            model.p_idle_sin_squared_rate = rate;
            model.p_idle_sin_squared_model = sine_model;
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

        if let Some(idle_after_2q) = self.idle_after_2q {
            model.idle_after_2q = idle_after_2q;
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

        if let Some(prob) = self.p_meas_crosstalk_global {
            model.p_meas_crosstalk_global = prob;
        }

        if let Some(prob) = self.p_meas_crosstalk_local {
            model.p_meas_crosstalk_local = prob;
        }

        if let Some(model_map) = self.p_meas_crosstalk_model.clone() {
            model.p_meas_crosstalk_model = model_map;
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
            self.noiseless_gates = Some(BTreeSet::new());
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

    /// Set the DEM-style linear idle-noise family.
    ///
    /// `rate` is the total event rate per time unit. For an idle of duration `d`, one event is
    /// sampled with probability `rate * d`, then its X, Y, Z, or leakage axis is drawn from
    /// `model`. The model must therefore be a normalized distribution: this linear family splits
    /// one total rate across its axes. This is exactly the pairing convenience
    /// `with_p_idle_linear_rate(rate).with_p_idle_linear_model(model)`.
    ///
    /// In contrast, [`Self::with_p_idle_sin_squared`] takes radians per time unit and an
    /// unnormalized model because sine laws do not add linearly: each axis carries its own
    /// independent rate. That setter applies no `2*pi` conversion and no
    /// `coherent_to_incoherent_factor`, unlike [`Self::with_p_idle_quadratic_rate`]. With neutral
    /// global and idle scales, `with_p_idle_quadratic_rate(r)` equals
    /// `with_p_idle_sin_squared(r * factor/2 * 2*pi, {"Z": 1.0})`, or `r * pi` at the
    /// default factor.
    ///
    /// All engines idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families.
    ///
    /// The linear sampling structure deliberately remains different from the DEM: engines emits
    /// at most one linear event followed by a categorical axis choice, while the DEM emits
    /// independent per-axis mechanisms. The difference is second order in the rates; this setter
    /// aligns the units and axis alphabet, not that sampling structure.
    #[must_use]
    pub fn with_p_idle_linear(self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        self.with_p_idle_linear_rate(rate)
            .with_p_idle_linear_model(model)
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
        let rate: f64 = rate * (3.0 / 2.0_f64).sqrt();
        self.p_idle_quadratic_rate = Some(rate);
        self
    }

    /// Set the DEM-style stochastic sine-squared idle-noise family.
    ///
    /// `rate` is in radians per time unit. No `2*pi` conversion and no
    /// `coherent_to_incoherent_factor` is applied, unlike
    /// [`Self::with_p_idle_quadratic_rate`]. For each axis P with multiplier `n_P` and an idle of
    /// duration `d`, engines independently samples `P(P) = sin^2(rate * n_P * d)`.
    ///
    /// The model accepts X, Y, Z, and L and is intentionally unnormalized: sine laws do not add
    /// linearly, so every axis carries its own independent rate. By comparison,
    /// [`Self::with_p_idle_linear`] requires a normalized distribution because its one total
    /// linear event rate is split across axes.
    ///
    /// With neutral global and idle scales, `with_p_idle_quadratic_rate(r)` equals
    /// `with_p_idle_sin_squared(r * factor/2 * 2*pi, {"Z": 1.0})`, or `r * pi` at the
    /// default factor. The legacy spelling is in cycles per time unit and also folds the factor
    /// into its runtime rate; this setter is the radians-per-time-unit spelling.
    ///
    /// All engines idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families.
    ///
    /// The linear sampling structure deliberately remains different from the DEM: engines emits
    /// at most one linear event followed by a categorical axis choice, while the DEM emits
    /// independent per-axis mechanisms. The difference is second order in the rates; this setter
    /// aligns the units and axis alphabet, not that sampling structure.
    ///
    /// # Panics
    ///
    /// Panics if `rate` or a multiplier is not finite and non-negative, or if `model` contains a
    /// key other than X, Y, Z, or L.
    #[must_use]
    pub fn with_p_idle_sin_squared(mut self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        let rate = Self::validate_finite_non_negative(rate, "sine-squared idling rate");
        Self::validate_sine_model(model);
        self.p_idle_sin_squared = Some((rate, model.clone()));
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
    pub fn with_p_prep(mut self, probability: f64) -> Self {
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

    // TODO: See if we should put a average scaling...
    /// Set the average prep crosstalk
    #[must_use]
    pub fn with_average_p_prep_crosstalk(mut self, prob: f64) -> Self {
        let prob: f64 = prob * 18.0 / 5.0;
        self.p_prep_crosstalk = Some(prob);
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
    pub fn with_p1(mut self, probability: f64) -> Self {
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
    pub fn with_average_p1(mut self, probability: f64) -> Self {
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
    pub fn with_p2(mut self, probability: f64) -> Self {
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
    pub fn with_average_p2(mut self, probability: f64) -> Self {
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

    /// Set the duration of the idle-noise site applied to each qubit after a two-qubit gate.
    ///
    /// A duration of `0.0` disables these sites. Nonzero sites receive all configured idle
    /// mechanisms over the given duration: linear stochastic noise from `p_idle_linear_rate` and
    /// `p_idle_linear_model`, quadratic dephasing from `p_idle_quadratic_rate` honoring
    /// `p_idle_coherent`, and the independent per-axis sine-squared family.
    ///
    /// Anyone who previously wrote `with_p2_idle(0.01)` and no linear rate now gets no after-2q
    /// idle noise; the equivalent is
    /// `with_p_idle_linear_rate(0.01).with_idle_after_2q(1.0)`.
    #[must_use]
    pub fn with_idle_after_2q(mut self, duration: f64) -> Self {
        self.idle_after_2q = Some(Self::validate_duration(duration));
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
    pub fn with_p_meas_0(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of flipping 1 to 0 during measurement
    #[must_use]
    pub fn with_p_meas_1(mut self, probability: f64) -> Self {
        self.p_meas_1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of bit flipping the measurement result
    #[must_use]
    pub fn with_p_meas(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(Self::validate_probability(probability));
        self.p_meas_1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of global crosstalk during measurement operations
    #[must_use]
    pub fn with_p_meas_crosstalk_global(mut self, prob: f64) -> Self {
        self.p_meas_crosstalk_global = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability of local crosstalk during measurement operations
    #[must_use]
    pub fn with_p_meas_crosstalk_local(mut self, prob: f64) -> Self {
        self.p_meas_crosstalk_local = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability of crosstalk during measurement operations
    /// This is a shorthand that sets both global and local to the given value
    #[must_use]
    pub fn with_p_meas_crosstalk(mut self, prob: f64) -> Self {
        self.p_meas_crosstalk_global = Some(Self::validate_probability(prob));
        self.p_meas_crosstalk_local = Some(Self::validate_probability(prob));
        self
    }

    /// Set the transition model for measurement crosstalk
    #[must_use]
    pub fn with_p_meas_crosstalk_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p_meas_crosstalk_model = Some(CrosstalkWeightedSampler::new(model));
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

    /// Validate that a value is finite and non-negative.
    fn validate_finite_non_negative(value: f64, name: &str) -> f64 {
        assert!(
            value.is_finite() && value >= 0.0,
            "{name} must be finite and non-negative, got {value}"
        );
        value
    }

    /// Validate an unnormalized sine-family multiplier model.
    fn validate_sine_model(model: &BTreeMap<String, f64>) {
        for (axis, multiplier) in model {
            assert!(
                matches!(axis.as_str(), "X" | "Y" | "Z" | "L"),
                "p_idle_sin_squared model has invalid key '{axis}'; expected X, Y, Z, or L"
            );
            Self::validate_finite_non_negative(
                *multiplier,
                &format!("p_idle_sin_squared multiplier for '{axis}'"),
            );
        }
    }

    /// Validate combinations whose interpretation would otherwise depend on silent precedence.
    ///
    /// # Errors
    ///
    /// Returns a description of the conflicting spellings and their incompatible semantics.
    pub fn validate_configuration(&self) -> Result<(), &'static str> {
        if self.p_idle_sin_squared.is_some() && self.p_idle_quadratic_rate.is_some() {
            return Err("with_p_idle_sin_squared cannot be combined with \
                 with_p_idle_quadratic_rate/with_average_p_idle_quadratic_rate: \
                 with_p_idle_sin_squared uses radians per time unit, while the legacy quadratic \
                 spellings use cycles per time unit and apply coherent_to_incoherent_factor");
        }
        if self.p_idle_sin_squared.is_some() && self.p_idle_coherent == Some(true) {
            return Err(
                "with_p_idle_sin_squared cannot be combined with with_p_idle_coherent(true): \
                 with_p_idle_sin_squared is stochastic by definition, while \
                 with_p_idle_coherent(true) selects the legacy coherent path",
            );
        }
        Ok(())
    }

    /// Validate that a duration is finite and non-negative
    fn validate_duration(duration: f64) -> f64 {
        assert!(
            duration.is_finite() && duration >= 0.0,
            "Duration must be finite and non-negative, got {duration}"
        );
        duration
    }

    // ========================================================================================== //
    /// The simple Pauli-probability subset of this configuration, if the
    /// physics reduces to it.
    ///
    /// Returns `(p_prep, p_meas_0, p_meas_1, p1, p2)`. `p1`/`p2` are in the
    /// standard depolarizing convention the builder stores internally (the
    /// `with_average_*` setters convert on the way in). Unset probabilities
    /// take their no-effect `GeneralNoiseModel::default()` values.
    ///
    /// Returns `Some` only when the noise shape is plain Pauli noise:
    ///
    /// - Emission ratios, preparation leakage, idle noise, crosstalk, and other optional
    ///   mechanisms may be unset or set to their neutral value.
    /// - Scales may be unset or set to one, and the noiseless-gate set must be empty.
    /// - Custom Pauli/emission/crosstalk models and angle-dependent
    ///   two-qubit noise must be unset.
    ///
    /// A configured seed is ignored (it selects a random stream, not
    /// physics). This exists so other simulation stacks can translate the
    /// common configuration without re-deriving probability conventions.
    #[must_use]
    pub fn simple_probabilities(&self) -> Option<(f64, f64, f64, f64, f64)> {
        let zero_or_unset = |v: Option<f64>| v.is_none() || v == Some(0.0);
        let emission_off =
            zero_or_unset(self.p1_emission_ratio) && zero_or_unset(self.p2_emission_ratio);
        if self.is_plain_pauli_except_angle_and_emission()
            && self.resolved_angle_scaling().is_none()
            && emission_off
        {
            Some(self.resolved_base_probabilities())
        } else {
            None
        }
    }

    /// Like [`Self::simple_probabilities`], but ALSO permits angle-dependent
    /// two-qubit scaling and returns it alongside the base probabilities.
    ///
    /// Returns `(p_prep, p_meas_0, p_meas_1, p1, p2, angle, p1_emission,
    /// p2_emission)` where `angle` is `Some((a, b, c, d, power))` when any
    /// `p2_angle_*` parameter is configured (the unset components take their
    /// model defaults), and `None` otherwise. This lets other simulation stacks
    /// translate the common "plain Pauli, plus optional angle-dependent
    /// two-qubit gate noise" configuration — the angle-dependent error rate is
    /// `p2 * (coeff * |theta/pi|^power + offset)` with separate
    /// `(a, b)` for negative and `(c, d)` for positive angles
    /// (see [`GeneralNoiseModel::p2_angle_error_rate`]).
    ///
    /// The two `*_emission` ratios are the resolved spontaneous-emission
    /// fractions (unset components take the model default). Emission is
    /// gate-removing in both engines and neo, so a downstream stack can
    /// reproduce it exactly by carrying these ratios with the default uniform
    /// emission distribution.
    ///
    /// All the OTHER non-angle feature requirements of
    /// [`Self::simple_probabilities`] still apply (leakage, idle, crosstalk,
    /// scales, custom samplers including custom emission distributions, and
    /// noiseless gates must be off).
    #[must_use]
    pub fn pauli_with_angle_scaling(&self) -> Option<PauliWithAngleScaling> {
        if self.is_plain_pauli_except_angle_and_emission() {
            let (p_prep, p_meas_0, p_meas_1, p1, p2) = self.resolved_base_probabilities();
            let (p1_emission, p2_emission) = self.resolved_emission_ratios();
            Some((
                p_prep,
                p_meas_0,
                p_meas_1,
                p1,
                p2,
                self.resolved_angle_scaling(),
                p1_emission,
                p2_emission,
            ))
        } else {
            None
        }
    }

    /// True when every non-Pauli feature is off EXCEPT possibly the
    /// angle-dependent two-qubit scaling and the spontaneous-emission ratios.
    /// Shared by `simple_probabilities` (which additionally requires both the
    /// angle scaling unset and emission off) and
    /// `pauli_with_angle_scaling` (which extracts them). The emission DISTRIBUTION
    /// must still be the default uniform model (`p1/p2_emission_model` unset) --
    /// custom emission samplers are NOT in this subset.
    fn is_plain_pauli_except_angle_and_emission(&self) -> bool {
        let zero_or_unset = |v: Option<f64>| v.is_none() || v == Some(0.0);
        let one_or_unset = |v: Option<f64>| v.is_none() || v == Some(1.0);

        // Emission ratios are intentionally NOT required off here: they are handled separately,
        // since neo matches engines' gate-removing emission with the default uniform distribution.
        let optional_features_off = zero_or_unset(self.p_prep_leak_ratio)
            && zero_or_unset(self.p_idle_linear_rate)
            && zero_or_unset(self.p_idle_quadratic_rate)
            && self.p_idle_sin_squared.is_none()
            && zero_or_unset(self.p_prep_crosstalk)
            && zero_or_unset(self.idle_after_2q)
            && zero_or_unset(self.p_meas_crosstalk_global)
            && zero_or_unset(self.p_meas_crosstalk_local);

        // Custom samplers/models could change the Pauli distribution; the
        // model defaults are uniform, so unset is standard. (Angle scaling is
        // intentionally NOT required here — it is handled separately.)
        let custom_models_off = self.p_idle_linear_model.is_none()
            && self.p1_emission_model.is_none()
            && self.p1_pauli_model.is_none()
            && self.p2_emission_model.is_none()
            && self.p2_pauli_model.is_none()
            && self.p_meas_crosstalk_model.is_none();

        let scales_neutral = one_or_unset(self.scale)
            && one_or_unset(self.idle_scale)
            && one_or_unset(self.prep_scale)
            && one_or_unset(self.meas_scale)
            && one_or_unset(self.p1_scale)
            && one_or_unset(self.p2_scale)
            && one_or_unset(self.p_prep_crosstalk_scale)
            && one_or_unset(self.p_meas_crosstalk_scale)
            // `emission_scale` multiplies the emission ratios in `build()`
            // (`scale_parameters`), but the resolved ratios surfaced here are
            // RAW. Require it neutral so a non-unit scale falls out of the
            // subset and is rejected, rather than silently mapping the
            // un-scaled ratio to neo (cross-stack mismatch).
            && one_or_unset(self.emission_scale);

        let gates_default = self.noiseless_gates.as_ref().is_none_or(BTreeSet::is_empty);

        optional_features_off && custom_models_off && scales_neutral && gates_default
    }

    /// Resolve the base Pauli probabilities `(p_prep, p_meas_0, p_meas_1, p1,
    /// p2)`, filling unset values from `GeneralNoiseModel::default()` so they
    /// cannot drift from the model's own defaults.
    fn resolved_base_probabilities(&self) -> (f64, f64, f64, f64, f64) {
        let (d_prep, d_meas_0, d_meas_1, d_p1, d_p2, _) =
            GeneralNoiseModel::default().probabilities();
        (
            self.p_prep.unwrap_or(d_prep),
            self.p_meas_0.unwrap_or(d_meas_0),
            self.p_meas_1.unwrap_or(d_meas_1),
            self.p1.unwrap_or(d_p1),
            self.p2.unwrap_or(d_p2),
        )
    }

    /// The configured angle-dependent two-qubit scaling `(a, b, c, d, power)`,
    /// or `None` when no `p2_angle_*` parameter is set. Unset components take
    /// the model default (read from `GeneralNoiseModel::default()` so they
    /// cannot drift).
    fn resolved_angle_scaling(&self) -> Option<(f64, f64, f64, f64, f64)> {
        if self.p2_angle_params.is_none() && self.p2_angle_power.is_none() {
            return None;
        }
        let default = GeneralNoiseModel::default();
        let (a, b, c, d) = self
            .p2_angle_params
            .unwrap_or_else(|| default.p2_angle_params());
        let power = self
            .p2_angle_power
            .unwrap_or_else(|| default.p2_angle_power());
        Some((a, b, c, d, power))
    }

    /// The resolved `(p1, p2)` spontaneous-emission ratios, taking the model
    /// default (read from `GeneralNoiseModel::default()`) for unset values.
    /// Only meaningful alongside the default uniform emission distribution
    /// (enforced by [`Self::is_plain_pauli_except_angle_and_emission`]).
    fn resolved_emission_ratios(&self) -> (f64, f64) {
        let default = GeneralNoiseModel::default();
        (
            self.p1_emission_ratio
                .unwrap_or_else(|| default.p1_emission_ratio()),
            self.p2_emission_ratio
                .unwrap_or_else(|| default.p2_emission_ratio()),
        )
    }

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
        model.p_meas_crosstalk_global *= p_meas_crosstalk_scale;
        model.p_meas_crosstalk_local *= p_meas_crosstalk_scale;
        model.p_prep_crosstalk *= p_prep_crosstalk_scale;

        // Then apply the regular scaling to crosstalks
        model.p_meas_crosstalk_global *= meas_scale * scale;
        model.p_meas_crosstalk_local *= meas_scale * scale;
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
    }
}

impl crate::noise::IntoNoiseModel for GeneralNoiseModelBuilder {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(self.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < f64::EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn auto_reproduces_legacy_demonstration_preset() {
        let model = GeneralNoiseModelBuilder::new().auto().build();

        assert_float_eq(model.p_prep, 0.01);
        assert_float_eq(model.p_meas_0, 0.01);
        assert_float_eq(model.p_meas_1, 0.01);
        assert_float_eq(model.p1, 0.001);
        assert_float_eq(model.p2, 0.01);
        assert_float_eq(model.p_idle_linear_rate, 0.001);
        assert_float_eq(model.p1_emission_ratio, 0.5);
        assert_float_eq(model.p2_emission_ratio, 0.5);
        assert_float_eq(model.p_prep_leak_ratio, 0.5);
        assert_float_eq(model.p1_seepage_prob, 0.5);
        assert_float_eq(model.p2_seepage_prob, 0.5);
        assert_float_eq(model.p_idle_coherent_to_incoherent_factor, 1.5);
    }

    #[test]
    fn explicit_setter_beats_auto_in_both_orders() {
        let set_explicit = |builder: GeneralNoiseModelBuilder| {
            builder
                .with_p_prep(0.11)
                .with_p_meas_0(0.12)
                .with_p_meas_1(0.13)
                .with_p1(0.14)
                .with_p2(0.15)
                .with_p_idle_linear_rate(0.16)
                .with_p1_emission_ratio(0.17)
                .with_p2_emission_ratio(0.18)
                .with_prep_leak_ratio(0.19)
                .with_p1_seepage_prob(0.20)
                .with_p2_seepage_prob(0.21)
                .with_p_idle_coherent_to_incoherent_factor(1.25)
        };
        let models = [
            set_explicit(GeneralNoiseModelBuilder::new().auto()).build(),
            set_explicit(GeneralNoiseModelBuilder::new()).auto().build(),
        ];

        for model in models {
            assert_float_eq(model.p_prep, 0.11);
            assert_float_eq(model.p_meas_0, 0.12);
            assert_float_eq(model.p_meas_1, 0.13);
            assert_float_eq(model.p1, 0.14);
            assert_float_eq(model.p2, 0.15);
            assert_float_eq(model.p_idle_linear_rate, 0.16);
            assert_float_eq(model.p1_emission_ratio, 0.17);
            assert_float_eq(model.p2_emission_ratio, 0.18);
            assert_float_eq(model.p_prep_leak_ratio, 0.19);
            assert_float_eq(model.p1_seepage_prob, 0.20);
            assert_float_eq(model.p2_seepage_prob, 0.21);
            assert_float_eq(model.p_idle_coherent_to_incoherent_factor, 1.25);
        }
    }

    #[test]
    fn auto_does_not_overwrite_explicit_zero() {
        let model = GeneralNoiseModelBuilder::new().with_p2(0.0).auto().build();

        assert_float_eq(model.p2, 0.0);
        assert_float_eq(model.p1, 0.001);
    }

    #[test]
    fn simple_probabilities_accepts_neutral_defaults_and_rejects_auto_features() {
        assert_eq!(
            GeneralNoiseModelBuilder::new().simple_probabilities(),
            Some((0.0, 0.0, 0.0, 0.0, 0.0))
        );
        assert!(
            GeneralNoiseModelBuilder::new()
                .with_average_p1(0.2)
                .simple_probabilities()
                .is_some()
        );
        assert!(
            GeneralNoiseModelBuilder::new()
                .auto()
                .simple_probabilities()
                .is_none()
        );
    }

    #[test]
    fn simple_probabilities_returns_stored_convention_values() {
        let simple = GeneralNoiseModelBuilder::new()
            .with_average_p1(0.2)
            .with_average_p2(0.4)
            .with_p_prep(0.01)
            .with_p_meas_0(0.02)
            .with_p_meas_1(0.03)
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0)
            .simple_probabilities()
            .expect("plain Pauli config is simple");

        let (p_prep, p_meas_0, p_meas_1, p1, p2) = simple;
        assert!((p_prep - 0.01).abs() < 1e-12);
        assert!((p_meas_0 - 0.02).abs() < 1e-12);
        assert!((p_meas_1 - 0.03).abs() < 1e-12);
        // Stored in standard depolarizing convention: average x 1.5 / x 1.25.
        assert!((p1 - 0.3).abs() < 1e-12);
        assert!((p2 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn simple_probabilities_unset_probabilities_take_model_defaults() {
        let simple = GeneralNoiseModelBuilder::new()
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0)
            .simple_probabilities()
            .expect("neutral defaults are simple");

        let (d_prep, d_meas_0, d_meas_1, d_p1, d_p2, _) =
            GeneralNoiseModel::default().probabilities();
        assert_eq!(simple, (d_prep, d_meas_0, d_meas_1, d_p1, d_p2));
    }

    /// The angle-aware extractor returns the same base probabilities as
    /// `simple_probabilities` with `None` angle when no `p2_angle_*` is set.
    #[test]
    fn pauli_with_angle_scaling_matches_simple_when_no_angle() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_average_p1(0.2)
            .with_average_p2(0.4)
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0);

        let simple = builder.simple_probabilities().expect("simple config");
        let (p_prep, p_meas_0, p_meas_1, p1, p2, angle, p1_emission, p2_emission) = builder
            .pauli_with_angle_scaling()
            .expect("simple config is also pauli-with-angle");
        assert_eq!((p_prep, p_meas_0, p_meas_1, p1, p2), simple);
        assert!(angle.is_none());
        // Emission is zero in the strict simple subset.
        assert_eq!((p1_emission, p2_emission), (0.0, 0.0));
    }

    /// Setting angle parameters keeps the config out of the strict simple
    /// subset but inside the angle-aware subset, with the coefficients and
    /// power surfaced verbatim.
    #[test]
    fn pauli_with_angle_scaling_extracts_configured_angle() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p2(0.3)
            .with_p2_angle_params(1.5, 0.0, 1.0, 0.0)
            .with_p2_angle_power(2.0)
            .with_average_p1(0.0)
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0)
            .with_p_prep(0.0)
            .with_p_meas_0(0.0)
            .with_p_meas_1(0.0);

        // Angle scaling is outside the STRICT simple subset.
        assert!(builder.simple_probabilities().is_none());

        let (_, _, _, _, p2, angle, _, _) = builder
            .pauli_with_angle_scaling()
            .expect("plain Pauli plus angle is in the angle-aware subset");
        assert!((p2 - 0.3).abs() < 1e-12);
        assert_eq!(angle, Some((1.5, 0.0, 1.0, 0.0, 2.0)));
    }

    /// An unset power takes the model default rather than dropping the angle.
    #[test]
    fn pauli_with_angle_scaling_fills_unset_power_from_default() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p2(0.3)
            .with_p2_angle_params(1.5, 0.0, 1.0, 0.0)
            .with_average_p1(0.0)
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0)
            .with_p_prep(0.0)
            .with_p_meas_0(0.0)
            .with_p_meas_1(0.0);

        let (_, _, _, _, _, angle, _, _) = builder
            .pauli_with_angle_scaling()
            .expect("angle-aware subset");
        let default_power = GeneralNoiseModel::default().p2_angle_power();
        assert_eq!(angle, Some((1.5, 0.0, 1.0, 0.0, default_power)));
    }

    /// Non-angle features beyond the subset still force `None`, even with an
    /// angle configured.
    #[test]
    fn pauli_with_angle_scaling_rejects_non_angle_features() {
        // Explicit preparation leakage is beyond the subset. (Emission ratios are NOT a
        // blocker -- they are part of the subset.)
        let builder = GeneralNoiseModelBuilder::new()
            .with_p2(0.3)
            .with_prep_leak_ratio(0.5)
            .with_p2_angle_params(1.5, 0.0, 1.0, 0.0);
        assert!(builder.pauli_with_angle_scaling().is_none());
    }

    /// Emission ratios are surfaced verbatim when they are in the subset, so a
    /// downstream stack can reproduce engines' gate-removing emission channel.
    #[test]
    fn pauli_with_angle_scaling_extracts_emission_ratios() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_average_p1(0.2)
            .with_average_p2(0.4)
            .with_p1_emission_ratio(0.25)
            .with_p2_emission_ratio(0.75)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0);

        // Non-zero emission is OUTSIDE the strict simple subset...
        assert!(builder.simple_probabilities().is_none());
        // ...but inside the angle-aware subset, with the ratios surfaced.
        let (.., angle, p1_emission, p2_emission) = builder
            .pauli_with_angle_scaling()
            .expect("plain Pauli plus emission is in the subset");
        assert!(angle.is_none());
        assert!((p1_emission - 0.25).abs() < 1e-12);
        assert!((p2_emission - 0.75).abs() < 1e-12);
    }

    /// A non-unit `emission_scale` multiplies the emission ratios at `build()`,
    /// but the subset surfaces the RAW ratios. It must therefore be rejected
    /// from the subset rather than silently mapping the un-scaled ratio.
    #[test]
    fn pauli_with_angle_scaling_rejects_emission_scale() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_average_p1(0.2)
            .with_p1_emission_ratio(0.25)
            .with_emission_scale(2.0)
            .with_prep_leak_ratio(0.0)
            .with_p_idle_linear_rate(0.0);

        // The built model applies the scale (0.25 * 2.0 = 0.5)...
        let built = builder.clone().build();
        assert!((built.p1_emission_ratio() - 0.5).abs() < 1e-12);
        // ...so surfacing the raw 0.25 would be a cross-stack mismatch: the
        // subset must refuse it.
        assert!(builder.pauli_with_angle_scaling().is_none());
    }
}
