use super::HookChannel;
use crate::GateType;
use crate::noise::weighted_sampler::{P2PauliLeakageWeightedStep, P2TransitionWeightedStep};
use crate::noise::{
    CrosstalkWeightedSampler, GeneralNoiseModel, NoiseRng, P2PauliLeakageStep, P2TransitionStep,
    PauliLeakageChannel, PauliLeakageWeightedSampler, QubitTransitionWeightedSampler,
    SingleQubitWeightedSampler, StateTransitionChannel, TwoQubitWeightedSampler,
};
use pecos_core::errors::PecosError;
use std::collections::{BTreeMap, BTreeSet};

fn validate_hook_probabilities(
    p1_transitions: [&[QubitTransitionWeightedSampler]; 2],
    p1_events: [&[PauliLeakageWeightedSampler]; 2],
    p2_transitions: [&[P2TransitionWeightedStep]; 2],
    p2_events: [&[P2PauliLeakageWeightedStep]; 2],
    p1_scale: f64,
    p2_scale: f64,
) -> Result<(), &'static str> {
    let valid = |p: f64| p.is_finite() && (0.0..=1.0).contains(&p);
    if !p1_transitions
        .into_iter()
        .flatten()
        .map(QubitTransitionWeightedSampler::probability)
        .chain(
            p1_events
                .into_iter()
                .flatten()
                .map(PauliLeakageWeightedSampler::probability),
        )
        .all(|p| valid(p * p1_scale))
    {
        return Err("scaled one-qubit hook probabilities must be finite and between zero and one");
    }
    if !p2_transitions
        .into_iter()
        .flatten()
        .flat_map(P2TransitionWeightedStep::probabilities)
        .chain(
            p2_events
                .into_iter()
                .flatten()
                .flat_map(P2PauliLeakageWeightedStep::probabilities),
        )
        .all(|p| valid(p * p2_scale))
    {
        return Err("scaled two-qubit hook probabilities must be finite and between zero and one");
    }
    Ok(())
}

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

/// Builder for creating general noise models.
///
/// Invalid setters preserve their previous fields and record the first failure.
/// Use [`Self::try_build`] or [`Self::validate_configuration`] to report it;
/// [`Self::build`] panics with the same validation message.
#[derive(Debug, Clone)]
pub struct GeneralNoiseModelBuilder {
    setter_error: Option<String>,
    p1_hook_order_before_gate: Vec<HookChannel>,
    p1_hook_order_after_gate: Vec<HookChannel>,
    p2_hook_order_before_gate: Vec<HookChannel>,
    p2_hook_order_after_gate: Vec<HookChannel>,
    // global params
    noiseless_gates: Option<BTreeSet<GateType>>,
    seed: Option<u64>,
    scale: Option<f64>,
    leakage_scale: Option<f64>,
    emission_scale: Option<f64>,
    // idle noise
    p_idle_linear_rate: Option<f64>,
    p_idle_linear_model: Option<SingleQubitWeightedSampler>,
    p_idle_sin_squared: Option<(f64, BTreeMap<String, f64>)>,
    p_idle_coherent: Option<(f64, BTreeMap<String, f64>)>,
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
    p1_transition_channels_before_gate: Vec<QubitTransitionWeightedSampler>,
    p1_transition_channels_after_gate: Vec<QubitTransitionWeightedSampler>,
    p1_pauli_leakage_channels_before_gate: Vec<PauliLeakageWeightedSampler>,
    p1_pauli_leakage_channels_after_gate: Vec<PauliLeakageWeightedSampler>,
    p1_scale: Option<f64>,
    // two-qubit gate noise
    p2: Option<f64>,
    p2_angle_params: Option<(f64, f64, f64, f64)>,
    p2_angle_power: Option<f64>,
    p2_emission_ratio: Option<f64>,
    p2_emission_model: Option<TwoQubitWeightedSampler>,
    p2_seepage_prob: Option<f64>,
    p2_pauli_model: Option<TwoQubitWeightedSampler>,
    p2_transition_steps_before_gate: Vec<P2TransitionWeightedStep>,
    p2_transition_steps_after_gate: Vec<P2TransitionWeightedStep>,
    p2_pauli_leakage_steps_before_gate: Vec<P2PauliLeakageWeightedStep>,
    p2_pauli_leakage_steps_after_gate: Vec<P2PauliLeakageWeightedStep>,
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
            setter_error: None,
            p1_hook_order_before_gate: Vec::new(),
            p1_hook_order_after_gate: Vec::new(),
            p2_hook_order_before_gate: Vec::new(),
            p2_hook_order_after_gate: Vec::new(),
            // global params
            noiseless_gates: None,
            seed: None,
            scale: None,
            leakage_scale: None,
            emission_scale: None,
            // idle noise
            p_idle_linear_rate: None,
            p_idle_linear_model: None,
            p_idle_sin_squared: None,
            p_idle_coherent: None,
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
            p1_transition_channels_before_gate: Vec::new(),
            p1_transition_channels_after_gate: Vec::new(),
            p1_pauli_leakage_channels_before_gate: Vec::new(),
            p1_pauli_leakage_channels_after_gate: Vec::new(),
            p1_scale: None,
            // two-qubit gate noise
            p2: None,
            p2_angle_params: None,
            p2_angle_power: None,
            p2_emission_ratio: None,
            p2_emission_model: None,
            p2_seepage_prob: None,
            p2_pauli_model: None,
            p2_transition_steps_before_gate: Vec::new(),
            p2_transition_steps_after_gate: Vec::new(),
            p2_pauli_leakage_steps_before_gate: Vec::new(),
            p2_pauli_leakage_steps_after_gate: Vec::new(),
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
        self
    }

    /// Build the general noise model
    ///
    /// # Returns
    /// A `GeneralNoiseModel`
    ///
    /// # Panics
    /// Panics if the configuration contains invalid probabilities, scale factors,
    /// or overflowing idle-rate products at `idle_after_2q`. Use [`Self::try_build`]
    /// to report invalid caller configuration without panicking.
    #[must_use]
    pub fn build(self) -> GeneralNoiseModel {
        self.validate_configuration()
            .unwrap_or_else(|message| panic!("{message}"));
        self.build_validated()
    }

    /// Build the model, reporting invalid setters or cross-field configuration.
    ///
    /// # Errors
    /// Returns [`PecosError::Input`] with the first validation failure.
    pub fn try_build(self) -> Result<GeneralNoiseModel, PecosError> {
        self.validate_configuration().map_err(PecosError::Input)?;
        Ok(self.build_validated())
    }

    fn build_validated(mut self) -> GeneralNoiseModel {
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
        if let Some(p_idle_linear_rate) = self.p_idle_linear_rate {
            model.p_idle_linear_rate = p_idle_linear_rate;
        }

        if let Some(model_map) = self.p_idle_linear_model.clone() {
            model.p_idle_linear_model = model_map;
        }

        if let Some((rate, sine_model)) = self.p_idle_sin_squared.clone() {
            model.p_idle_sin_squared_rate = rate;
            model.p_idle_sin_squared_model = sine_model;
        }

        if let Some((rate, coherent_model)) = self.p_idle_coherent.clone() {
            model.p_idle_coherent_rate = rate;
            model.p_idle_coherent_model = coherent_model;
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

        model.p1_hook_order_before_gate = self.p1_hook_order_before_gate.clone();
        model.p1_hook_order_after_gate = self.p1_hook_order_after_gate.clone();
        model.p2_hook_order_before_gate = self.p2_hook_order_before_gate.clone();
        model.p2_hook_order_after_gate = self.p2_hook_order_after_gate.clone();
        model.p1_transition_channels_before_gate = self.p1_transition_channels_before_gate.clone();
        model.p1_transition_channels_after_gate = self.p1_transition_channels_after_gate.clone();
        model.p1_pauli_leakage_channels_before_gate =
            self.p1_pauli_leakage_channels_before_gate.clone();
        model.p1_pauli_leakage_channels_after_gate =
            self.p1_pauli_leakage_channels_after_gate.clone();

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

        model.p2_transition_steps_before_gate = self.p2_transition_steps_before_gate.clone();
        model.p2_transition_steps_after_gate = self.p2_transition_steps_after_gate.clone();
        model.p2_pauli_leakage_steps_before_gate = self.p2_pauli_leakage_steps_before_gate.clone();
        model.p2_pauli_leakage_steps_after_gate = self.p2_pauli_leakage_steps_after_gate.clone();

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
        if let Some(value) = self.record_result(Self::validate_probability(scale)) {
            self.leakage_scale = Some(value);
        }
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
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p1_seepage_prob = Some(value);
            self.p2_seepage_prob = Some(value);
        }
        self
    }

    // --- idle noise --- //

    /// Set the DEM-style linear idle-noise family.
    ///
    /// `rate` is the total event rate per time unit. For an idle of duration `d`, one event is
    /// sampled with probability `rate * d`, then its X, Y, Z, or leakage axis is drawn from
    /// `model`. The model must therefore be a normalized distribution: this linear family splits
    /// one total rate across its axes.
    ///
    /// In contrast, [`Self::with_p_idle_sin_squared`] takes radians per time unit and an
    /// unnormalized model because sine laws do not add linearly: each axis carries its own
    /// independent rate. That setter applies no unit conversion.
    ///
    /// All engines idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families.
    ///
    /// The linear sampling structure deliberately remains different from the DEM: engines emits
    /// at most one linear event followed by a categorical axis choice, while the DEM emits
    /// independent per-axis mechanisms. The difference is second order in the rates; this setter
    /// aligns the units and axis alphabet, not that sampling structure.
    #[must_use]
    pub fn with_p_idle_linear(mut self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        let update = Self::validate_non_negative(rate, "linear idling rate")
            .and_then(|rate| SingleQubitWeightedSampler::try_new(model).map(|model| (rate, model)));
        if let Some((rate, model)) = self.record_result(update) {
            self.p_idle_linear_rate = Some(rate);
            self.p_idle_linear_model = Some(model);
        }
        self
    }

    /// Set the DEM-style stochastic sine-squared idle-noise family.
    ///
    /// `rate` is in radians per time unit. No unit conversion is applied. For each axis P with
    /// multiplier `n_P` and an idle of duration `d`, engines independently samples
    /// `P(P) = sin^2(rate * n_P * d)`.
    ///
    /// The model accepts X, Y, Z, and L and is intentionally unnormalized: sine laws do not add
    /// linearly, so every axis carries its own independent rate. By comparison,
    /// [`Self::with_p_idle_linear`] requires a normalized distribution because its one total
    /// linear event rate is split across axes.
    ///
    /// The removed cycles-per-time spelling migrates exactly as follows at its former default
    /// factor of one:
    ///
    /// `with_p_idle_quadratic_rate(r)  ==  with_p_idle_sin_squared(r * PI, {"Z": 1.0})`
    ///
    /// All engines idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families.
    ///
    /// The linear sampling structure deliberately remains different from the DEM: engines emits
    /// at most one linear event followed by a categorical axis choice, while the DEM emits
    /// independent per-axis mechanisms. The difference is second order in the rates; this setter
    /// aligns the units and axis alphabet, not that sampling structure.
    ///
    /// Invalid input is recorded and reported by [`Self::try_build`] or
    /// [`Self::validate_configuration`]: `rate` and multipliers must be finite and non-negative;
    /// the model must not contain a key other than X, Y, Z, or L.
    #[must_use]
    pub fn with_p_idle_sin_squared(mut self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        let update = Self::validate_finite_non_negative(rate, "sine-squared idling rate")
            .and_then(|rate| Self::validate_sine_model(model).map(|()| rate));
        if let Some(rate) = self.record_result(update) {
            self.p_idle_sin_squared = Some((rate, model.clone()));
        }
        self
    }

    /// Set the DEM-style coherent idle-noise family.
    ///
    /// `rate` is in radians per time unit. No unit conversion is applied, just as for
    /// [`Self::with_p_idle_sin_squared`]. For each RX/RY/RZ generator with multiplier `n_P` and an
    /// idle of duration `d`, engines applies a deterministic rotation with angle `rate * n_P * d`;
    /// coherent evolution is not sampled and consumes no random draw.
    ///
    /// The model is intentionally unnormalized because its values are relative rate multipliers,
    /// not probabilities to be split from one total event rate. The symmetric default model is
    /// `{"RX": 1.0, "RY": 1.0, "RZ": 1.0}`. Leakage and all keys other than RX, RY, and RZ
    /// are rejected because leakage is not a rotation.
    ///
    /// Whether these rotations can be consumed depends on the downstream consumer. The standard
    /// DEM builder rejects coherent idle noise; the EEG route in `exp/pecos-eeg` represents it
    /// with an RZ generator; and a simulator applies it only when its rotation executor is
    /// installed. PECOS #437 documents how a missing executor could otherwise silently drop it.
    ///
    /// Invalid input is recorded and reported by [`Self::try_build`] or
    /// [`Self::validate_configuration`]: `rate` and multipliers must be finite and non-negative;
    /// the model must not contain a key other than RX, RY, or RZ.
    #[must_use]
    pub fn with_p_idle_coherent(mut self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        let update = Self::validate_finite_non_negative(rate, "coherent idling rate")
            .and_then(|rate| Self::validate_coherent_model(model).map(|()| rate));
        if let Some(rate) = self.record_result(update) {
            self.p_idle_coherent = Some((rate, model.clone()));
        }
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
        if let Some(value) = self.record_result(Self::validate_probability(probability)) {
            self.p_prep = Some(value);
        }
        self
    }

    /// Set the preparation leakage ratio
    #[must_use]
    pub fn with_prep_leak_ratio(mut self, ratio: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(ratio)) {
            self.p_prep_leak_ratio = Some(value);
        }
        self
    }

    /// Set the probability of crosstalk during initialization operations
    #[must_use]
    pub fn with_p_prep_crosstalk(mut self, prob: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p_prep_crosstalk = Some(value);
        }
        self
    }

    // TODO: See if we should put a average scaling...
    /// Set the average prep crosstalk
    #[must_use]
    pub fn with_average_p_prep_crosstalk(mut self, prob: f64) -> Self {
        let value = if prob.is_finite() && (0.0..=5.0 / 18.0).contains(&prob) {
            Ok(prob * 18.0 / 5.0)
        } else {
            Err(format!(
                "average preparation crosstalk probability must be between 0 and 5/18, got {prob}"
            ))
        };
        if let Some(value) = self.record_result(value) {
            self.p_prep_crosstalk = Some(value);
        }
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
        if let Some(value) = self.record_result(Self::validate_non_negative(
            scale,
            "Preparation crosstalk rescale factor",
        )) {
            self.p_prep_crosstalk_scale = Some(value);
        }
        self
    }

    // ========================================================================================== //
    // single-qubit gate noise
    // ========================================================================================== //

    /// Set the probability of error after single-qubit gates
    #[must_use]
    pub fn with_p1(mut self, probability: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(probability)) {
            self.p1 = Some(value);
        }
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
        if let Some(value) = self.record_result(Self::validate_probability(probability * 3.0 / 2.0))
        {
            self.p1 = Some(value);
        }
        self
    }

    /// Set the emission ratio for single-qubit gate errors
    #[must_use]
    pub fn with_p1_emission_ratio(mut self, ratio: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(ratio)) {
            self.p1_emission_ratio = Some(value);
        }
        self
    }

    /// Set the emission error model for single-qubit gates
    #[must_use]
    pub fn with_p1_emission_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        if let Some(value) = self.record_result(SingleQubitWeightedSampler::try_new(model)) {
            self.p1_emission_model = Some(value);
        }
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_p1_seepage_prob(mut self, prob: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p1_seepage_prob = Some(value);
        }
        self
    }

    /// Set the Pauli error model for single-qubit gates
    #[must_use]
    pub fn with_p1_pauli_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        if let Some(value) = self.record_result(SingleQubitWeightedSampler::try_new(model)) {
            self.p1_pauli_model = Some(value);
        }
        self
    }

    /// Replace the ordered Pauli-plus-leakage channel stack before single-qubit gates.
    ///
    /// This stack is appended in builder insertion order across both channel families.
    /// A sampled `L` can suppress the ideal gate,
    /// while a later before-gate transition channel can recover it. Outer probabilities follow
    /// the global and p1 scale factors.
    #[must_use]
    pub fn with_p1_pauli_leakage_channels_before_gate(
        mut self,
        channels: &[PauliLeakageChannel],
    ) -> Self {
        self.p1_pauli_leakage_channels_before_gate = channels
            .iter()
            .map(PauliLeakageWeightedSampler::new)
            .collect();
        self.p1_hook_order_before_gate
            .retain(|entry| !matches!(entry, HookChannel::PauliLeakage(_)));
        self.p1_hook_order_before_gate.extend(
            (0..self.p1_pauli_leakage_channels_before_gate.len()).map(HookChannel::PauliLeakage),
        );
        self
    }

    /// Append one Pauli-plus-leakage channel before single-qubit gates.
    #[must_use]
    pub fn add_p1_pauli_leakage_channel_before_gate(
        mut self,
        channel: &PauliLeakageChannel,
    ) -> Self {
        self.p1_pauli_leakage_channels_before_gate
            .push(PauliLeakageWeightedSampler::new(channel));
        self.p1_hook_order_before_gate
            .push(HookChannel::PauliLeakage(
                self.p1_pauli_leakage_channels_before_gate.len() - 1,
            ));
        self
    }

    /// Replace the ordered Pauli-plus-leakage channel stack after single-qubit noise sites.
    ///
    /// This stack runs after ordinary p1 noise, in insertion order across both families.
    #[must_use]
    pub fn with_p1_pauli_leakage_channels_after_gate(
        mut self,
        channels: &[PauliLeakageChannel],
    ) -> Self {
        self.p1_pauli_leakage_channels_after_gate = channels
            .iter()
            .map(PauliLeakageWeightedSampler::new)
            .collect();
        self.p1_hook_order_after_gate
            .retain(|entry| !matches!(entry, HookChannel::PauliLeakage(_)));
        self.p1_hook_order_after_gate.extend(
            (0..self.p1_pauli_leakage_channels_after_gate.len()).map(HookChannel::PauliLeakage),
        );
        self
    }

    /// Append one Pauli-plus-leakage channel after single-qubit noise sites.
    #[must_use]
    pub fn add_p1_pauli_leakage_channel_after_gate(
        mut self,
        channel: &PauliLeakageChannel,
    ) -> Self {
        self.p1_pauli_leakage_channels_after_gate
            .push(PauliLeakageWeightedSampler::new(channel));
        self.p1_hook_order_after_gate
            .push(HookChannel::PauliLeakage(
                self.p1_pauli_leakage_channels_after_gate.len() - 1,
            ));
        self
    }

    /// Replace the ordered transition-channel stack before every single-qubit gate.
    ///
    /// Channels execute in slice order. State changes from one channel are visible to the next.
    /// Recovery can allow an incoming leaked qubit to participate in the ideal gate, while leakage
    /// can suppress it. Channel probabilities are affected by global and p1 scale factors.
    #[must_use]
    pub fn with_p1_transition_channels_before_gate(
        mut self,
        channels: &[StateTransitionChannel],
    ) -> Self {
        self.p1_transition_channels_before_gate = channels
            .iter()
            .map(QubitTransitionWeightedSampler::new)
            .collect();
        self.p1_hook_order_before_gate
            .retain(|entry| !matches!(entry, HookChannel::Transition(_)));
        self.p1_hook_order_before_gate.extend(
            (0..self.p1_transition_channels_before_gate.len()).map(HookChannel::Transition),
        );
        self
    }

    /// Append one transition channel before every single-qubit gate.
    #[must_use]
    pub fn add_p1_transition_channel_before_gate(
        mut self,
        channel: &StateTransitionChannel,
    ) -> Self {
        self.p1_transition_channels_before_gate
            .push(QubitTransitionWeightedSampler::new(channel));
        self.p1_hook_order_before_gate.push(HookChannel::Transition(
            self.p1_transition_channels_before_gate.len() - 1,
        ));
        self
    }

    /// Replace the ordered transition-channel stack after every single-qubit noise site.
    ///
    /// These channels run after the ideal gate or its suppression and after ordinary p1
    /// Pauli/emission noise. Recovery does not retroactively execute a suppressed gate. Prefer
    /// [`Self::with_p1_pauli_model`] for ordinary gate-local Pauli noise.
    #[must_use]
    pub fn with_p1_transition_channels_after_gate(
        mut self,
        channels: &[StateTransitionChannel],
    ) -> Self {
        self.p1_transition_channels_after_gate = channels
            .iter()
            .map(QubitTransitionWeightedSampler::new)
            .collect();
        self.p1_hook_order_after_gate
            .retain(|entry| !matches!(entry, HookChannel::Transition(_)));
        self.p1_hook_order_after_gate
            .extend((0..self.p1_transition_channels_after_gate.len()).map(HookChannel::Transition));
        self
    }

    /// Append one transition channel after every single-qubit noise site.
    #[must_use]
    pub fn add_p1_transition_channel_after_gate(
        mut self,
        channel: &StateTransitionChannel,
    ) -> Self {
        self.p1_transition_channels_after_gate
            .push(QubitTransitionWeightedSampler::new(channel));
        self.p1_hook_order_after_gate.push(HookChannel::Transition(
            self.p1_transition_channels_after_gate.len() - 1,
        ));
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
        if let Some(value) = self.record_result(Self::validate_probability(probability)) {
            self.p2 = Some(value);
        }
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
        if let Some(value) = self.record_result(Self::validate_probability(probability * 5.0 / 4.0))
        {
            self.p2 = Some(value);
        }
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
        if let Some(value) =
            self.record_result(Self::validate_positive(power, "RZZ power parameter"))
        {
            self.p2_angle_power = Some(value);
        }
        self
    }

    /// Set the two-qubit emission ratio
    #[must_use]
    pub fn with_p2_emission_ratio(mut self, ratio: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(ratio)) {
            self.p2_emission_ratio = Some(value);
        }
        self
    }

    /// Set the probability model for two-qubit emission errors
    #[must_use]
    pub fn with_p2_emission_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        if let Some(value) = self.record_result(TwoQubitWeightedSampler::try_new(model)) {
            self.p2_emission_model = Some(value);
        }
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_p2_seepage_prob(mut self, prob: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p2_seepage_prob = Some(value);
        }
        self
    }

    /// Set the probability model for two-qubit Pauli errors
    #[must_use]
    pub fn with_p2_pauli_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        if let Some(value) = self.record_result(TwoQubitWeightedSampler::try_new(model)) {
            self.p2_pauli_model = Some(value);
        }
        self
    }

    /// Replace the ordered Pauli-plus-leakage step stack before two-qubit gates.
    ///
    /// This stack is appended in builder insertion order across both channel families.
    /// Each step is either two independently drawn
    /// single-qubit channels or one correlated joint event channel.
    #[must_use]
    pub fn with_p2_pauli_leakage_steps_before_gate(mut self, steps: &[P2PauliLeakageStep]) -> Self {
        self.p2_pauli_leakage_steps_before_gate =
            steps.iter().map(P2PauliLeakageWeightedStep::new).collect();
        self.p2_hook_order_before_gate
            .retain(|entry| !matches!(entry, HookChannel::PauliLeakage(_)));
        self.p2_hook_order_before_gate.extend(
            (0..self.p2_pauli_leakage_steps_before_gate.len()).map(HookChannel::PauliLeakage),
        );
        self
    }

    /// Append one Pauli-plus-leakage step before two-qubit gates.
    #[must_use]
    pub fn add_p2_pauli_leakage_step_before_gate(mut self, step: &P2PauliLeakageStep) -> Self {
        self.p2_pauli_leakage_steps_before_gate
            .push(P2PauliLeakageWeightedStep::new(step));
        self.p2_hook_order_before_gate
            .push(HookChannel::PauliLeakage(
                self.p2_pauli_leakage_steps_before_gate.len() - 1,
            ));
        self
    }

    /// Append the same independently drawn single-qubit channel to both two-qubit gate legs.
    #[must_use]
    pub fn add_p2_pauli_leakage_channel_before_gate(self, channel: &PauliLeakageChannel) -> Self {
        self.add_p2_pauli_leakage_step_before_gate(&P2PauliLeakageStep::same_on_each(
            channel.clone(),
        ))
    }

    /// Replace the ordered Pauli-plus-leakage step stack after two-qubit noise sites.
    ///
    /// This stack runs after ordinary p2 noise, in insertion order across both families.
    #[must_use]
    pub fn with_p2_pauli_leakage_steps_after_gate(mut self, steps: &[P2PauliLeakageStep]) -> Self {
        self.p2_pauli_leakage_steps_after_gate =
            steps.iter().map(P2PauliLeakageWeightedStep::new).collect();
        self.p2_hook_order_after_gate
            .retain(|entry| !matches!(entry, HookChannel::PauliLeakage(_)));
        self.p2_hook_order_after_gate.extend(
            (0..self.p2_pauli_leakage_steps_after_gate.len()).map(HookChannel::PauliLeakage),
        );
        self
    }

    /// Append one Pauli-plus-leakage step after two-qubit noise sites.
    #[must_use]
    pub fn add_p2_pauli_leakage_step_after_gate(mut self, step: &P2PauliLeakageStep) -> Self {
        self.p2_pauli_leakage_steps_after_gate
            .push(P2PauliLeakageWeightedStep::new(step));
        self.p2_hook_order_after_gate
            .push(HookChannel::PauliLeakage(
                self.p2_pauli_leakage_steps_after_gate.len() - 1,
            ));
        self
    }

    /// Append the same independently drawn single-qubit channel after both gate legs.
    #[must_use]
    pub fn add_p2_pauli_leakage_channel_after_gate(self, channel: &PauliLeakageChannel) -> Self {
        self.add_p2_pauli_leakage_step_after_gate(&P2PauliLeakageStep::same_on_each(
            channel.clone(),
        ))
    }

    /// Replace the ordered transition-step stack before every two-qubit gate.
    ///
    /// A step is either two single-qubit channels applied independently to their respective gate
    /// legs, or one joint conditional matrix on the nine pair states from `00` through `LL`.
    /// Steps execute in slice order before PECOS checks the pair for leakage and before ordinary
    /// p2 noise. Recovery can allow the ideal gate to execute; leakage can suppress it.
    /// Probabilities are affected by global and p2 scale factors.
    #[must_use]
    pub fn with_p2_transition_steps_before_gate(mut self, steps: &[P2TransitionStep]) -> Self {
        self.p2_transition_steps_before_gate =
            steps.iter().map(P2TransitionWeightedStep::new).collect();
        self.p2_hook_order_before_gate
            .retain(|entry| !matches!(entry, HookChannel::Transition(_)));
        self.p2_hook_order_before_gate
            .extend((0..self.p2_transition_steps_before_gate.len()).map(HookChannel::Transition));
        self
    }

    /// Append one independent-leg or joint transition step before every two-qubit gate.
    #[must_use]
    pub fn add_p2_transition_step_before_gate(mut self, step: &P2TransitionStep) -> Self {
        self.p2_transition_steps_before_gate
            .push(P2TransitionWeightedStep::new(step));
        self.p2_hook_order_before_gate.push(HookChannel::Transition(
            self.p2_transition_steps_before_gate.len() - 1,
        ));
        self
    }

    /// Append a transition channel before every two-qubit gate.
    ///
    /// A [`StateTransitionChannel`] is applied independently to both gate legs. A
    /// [`TwoQubitStateTransitionChannel`](crate::noise::TwoQubitStateTransitionChannel) is applied as one
    /// joint map on the pair state. Use [`Self::add_p2_transition_step_before_gate`] when the two
    /// legs need different single-qubit channels.
    #[must_use]
    pub fn add_p2_transition_channel_before_gate(
        self,
        channel: impl Into<P2TransitionStep>,
    ) -> Self {
        self.add_p2_transition_step_before_gate(&channel.into())
    }

    /// Replace the ordered transition-step stack after every two-qubit noise site.
    ///
    /// Each independent-leg or joint step runs after the ideal gate (or suppression) and ordinary
    /// p2 Pauli/emission noise, but before after-2Q idle noise. A later step observes leakage or
    /// recovery produced by every earlier step. Prefer [`Self::with_p2_pauli_model`] for ordinary
    /// correlated two-qubit Pauli noise.
    #[must_use]
    pub fn with_p2_transition_steps_after_gate(mut self, steps: &[P2TransitionStep]) -> Self {
        self.p2_transition_steps_after_gate =
            steps.iter().map(P2TransitionWeightedStep::new).collect();
        self.p2_hook_order_after_gate
            .retain(|entry| !matches!(entry, HookChannel::Transition(_)));
        self.p2_hook_order_after_gate
            .extend((0..self.p2_transition_steps_after_gate.len()).map(HookChannel::Transition));
        self
    }

    /// Append one independent-leg or joint transition step after every two-qubit noise site.
    #[must_use]
    pub fn add_p2_transition_step_after_gate(mut self, step: &P2TransitionStep) -> Self {
        self.p2_transition_steps_after_gate
            .push(P2TransitionWeightedStep::new(step));
        self.p2_hook_order_after_gate.push(HookChannel::Transition(
            self.p2_transition_steps_after_gate.len() - 1,
        ));
        self
    }

    /// Append a transition channel after every two-qubit noise site.
    ///
    /// A [`StateTransitionChannel`] is applied independently to both gate legs. A
    /// [`TwoQubitStateTransitionChannel`](crate::noise::TwoQubitStateTransitionChannel) is applied as one
    /// joint map on the pair state. Use [`Self::add_p2_transition_step_after_gate`] when the two
    /// legs need different single-qubit channels.
    #[must_use]
    pub fn add_p2_transition_channel_after_gate(
        self,
        channel: impl Into<P2TransitionStep>,
    ) -> Self {
        self.add_p2_transition_step_after_gate(&channel.into())
    }

    /// Set the duration of the idle-noise site applied to each qubit after a two-qubit gate.
    ///
    /// A duration of `0.0` disables these sites. Nonzero sites receive every configured linear,
    /// sine-squared, and coherent idle family over the given duration.
    ///
    /// Anyone who previously wrote `with_p2_idle(0.01)` and no linear rate now gets no after-2q
    /// idle noise; the equivalent is
    /// `with_p_idle_linear(0.01, model).with_idle_after_2q(1.0)`.
    #[must_use]
    pub fn with_idle_after_2q(mut self, duration: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_duration(duration)) {
            self.idle_after_2q = Some(value);
        }
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
        if let Some(value) = self.record_result(Self::validate_probability(probability)) {
            self.p_meas_0 = Some(value);
        }
        self
    }

    /// Set the probability of flipping 1 to 0 during measurement
    #[must_use]
    pub fn with_p_meas_1(mut self, probability: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(probability)) {
            self.p_meas_1 = Some(value);
        }
        self
    }

    /// Set the probability of bit flipping the measurement result
    #[must_use]
    pub fn with_p_meas(mut self, probability: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(probability)) {
            self.p_meas_0 = Some(value);
            self.p_meas_1 = Some(value);
        }
        self
    }

    /// Set the probability of global crosstalk during measurement operations
    #[must_use]
    pub fn with_p_meas_crosstalk_global(mut self, prob: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p_meas_crosstalk_global = Some(value);
        }
        self
    }

    /// Set the probability of local crosstalk during measurement operations
    #[must_use]
    pub fn with_p_meas_crosstalk_local(mut self, prob: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p_meas_crosstalk_local = Some(value);
        }
        self
    }

    /// Set the probability of crosstalk during measurement operations
    /// This is a shorthand that sets both global and local to the given value
    #[must_use]
    pub fn with_p_meas_crosstalk(mut self, prob: f64) -> Self {
        if let Some(value) = self.record_result(Self::validate_probability(prob)) {
            self.p_meas_crosstalk_global = Some(value);
            self.p_meas_crosstalk_local = Some(value);
        }
        self
    }

    /// Set the transition model for measurement crosstalk
    #[must_use]
    pub fn with_p_meas_crosstalk_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        if let Some(value) = self.record_result(CrosstalkWeightedSampler::try_new(model)) {
            self.p_meas_crosstalk_model = Some(value);
        }
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
        if let Some(value) = self.record_result(Self::validate_non_negative(
            scale,
            "Measurement crosstalk rescale factor",
        )) {
            self.p_meas_crosstalk_scale = Some(value);
        }
        self
    }

    // ========================================================================================== //
    // validation
    // ========================================================================================== //

    /// Return the first recorded setter failure, if any.
    #[must_use]
    pub fn setter_error(&self) -> Option<&str> {
        self.setter_error.as_deref()
    }

    fn record_result<T>(&mut self, result: Result<T, String>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(message) => {
                self.setter_error.get_or_insert(message);
                None
            }
        }
    }

    /// Validate that a value is a valid probability (between 0 and 1)
    fn validate_probability(prob: f64) -> Result<f64, String> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(format!("Probability must be between 0 and 1, got {prob}"));
        }
        Ok(prob)
    }

    /// Validate that a value is positive
    fn validate_positive(value: f64, name: &str) -> Result<f64, String> {
        if value.is_nan() || value <= 0.0 {
            return Err(format!("{name} must be positive, got {value}"));
        }
        Ok(value)
    }

    /// Validate that a value is non-negative
    fn validate_non_negative(value: f64, name: &str) -> Result<f64, String> {
        if value.is_nan() || value < 0.0 {
            return Err(format!("{name} must be non-negative, got {value}"));
        }
        Ok(value)
    }

    /// Validate that a value is finite and non-negative.
    fn validate_finite_non_negative(value: f64, name: &str) -> Result<f64, String> {
        if !(value.is_finite() && value >= 0.0) {
            return Err(format!(
                "{name} must be finite and non-negative, got {value}"
            ));
        }
        Ok(value)
    }

    /// Validate an unnormalized sine-family multiplier model.
    fn validate_sine_model(model: &BTreeMap<String, f64>) -> Result<(), String> {
        for (axis, multiplier) in model {
            if !matches!(axis.as_str(), "X" | "Y" | "Z" | "L") {
                return Err(format!(
                    "p_idle_sin_squared model has invalid key '{axis}'; expected X, Y, Z, or L"
                ));
            }
            Self::validate_finite_non_negative(
                *multiplier,
                &format!("p_idle_sin_squared multiplier for '{axis}'"),
            )?;
        }
        Ok(())
    }

    /// Validate an unnormalized coherent-family multiplier model.
    fn validate_coherent_model(model: &BTreeMap<String, f64>) -> Result<(), String> {
        for (axis, multiplier) in model {
            if !matches!(axis.as_str(), "RX" | "RY" | "RZ") {
                return Err(format!(
                    "p_idle_coherent model has invalid key '{axis}'; expected RX, RY, or RZ"
                ));
            }
            Self::validate_finite_non_negative(
                *multiplier,
                &format!("p_idle_coherent multiplier for '{axis}'"),
            )?;
        }
        Ok(())
    }

    /// Validate cross-field configuration invariants.
    ///
    /// # Errors
    ///
    /// Returns the first recorded setter error, then the first invalid combination.
    pub fn validate_configuration(&self) -> Result<(), String> {
        if let Some(message) = self.setter_error() {
            return Err(message.to_owned());
        }
        let finite_non_negative = |value: f64| value.is_finite() && value >= 0.0;
        let probability = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
        let scale = self.scale.unwrap_or(1.0);
        let idle_scale = self.idle_scale.unwrap_or(1.0);
        let prep_scale = self.prep_scale.unwrap_or(1.0);
        let meas_scale = self.meas_scale.unwrap_or(1.0);
        let p1_scale = self.p1_scale.unwrap_or(1.0);
        let p2_scale = self.p2_scale.unwrap_or(1.0);
        let emission_scale = self.emission_scale.unwrap_or(1.0);
        let prep_crosstalk_scale = self.p_prep_crosstalk_scale.unwrap_or(1.0);
        let meas_crosstalk_scale = self.p_meas_crosstalk_scale.unwrap_or(1.0);
        if ![
            scale,
            idle_scale,
            prep_scale,
            meas_scale,
            p1_scale,
            p2_scale,
            emission_scale,
            prep_crosstalk_scale,
            meas_crosstalk_scale,
        ]
        .into_iter()
        .all(finite_non_negative)
        {
            return Err("general-noise scale factors must be finite and non-negative".to_string());
        }

        let (p_prep, p_meas_0, p_meas_1, p1, p2) = self.resolved_base_probabilities();
        if !probability(p_prep * prep_scale * scale) {
            return Err("scaled preparation probability must be between zero and one".to_string());
        }
        if !probability(p_meas_0 * meas_scale * scale)
            || !probability(p_meas_1 * meas_scale * scale)
        {
            return Err(
                "scaled measurement probabilities must be between zero and one".to_string(),
            );
        }
        if !probability(p1 * p1_scale * scale) {
            return Err("scaled one-qubit probability must be between zero and one".to_string());
        }
        let scaled_p2 = p2 * p2_scale * scale;
        if !probability(scaled_p2) {
            return Err("scaled two-qubit probability must be between zero and one".to_string());
        }

        validate_hook_probabilities(
            [
                &self.p1_transition_channels_before_gate,
                &self.p1_transition_channels_after_gate,
            ],
            [
                &self.p1_pauli_leakage_channels_before_gate,
                &self.p1_pauli_leakage_channels_after_gate,
            ],
            [
                &self.p2_transition_steps_before_gate,
                &self.p2_transition_steps_after_gate,
            ],
            [
                &self.p2_pauli_leakage_steps_before_gate,
                &self.p2_pauli_leakage_steps_after_gate,
            ],
            p1_scale * scale,
            p2_scale * scale,
        )?;

        let default = GeneralNoiseModel::default();
        let prep_crosstalk = self.p_prep_crosstalk.unwrap_or(default.p_prep_crosstalk);
        if !probability(prep_crosstalk * prep_crosstalk_scale * prep_scale * scale) {
            return Err(
                "scaled preparation-crosstalk probability must be between zero and one".to_string(),
            );
        }
        let meas_crosstalk_global = self
            .p_meas_crosstalk_global
            .unwrap_or(default.p_meas_crosstalk_global);
        let meas_crosstalk_local = self
            .p_meas_crosstalk_local
            .unwrap_or(default.p_meas_crosstalk_local);
        if !probability(meas_crosstalk_global * meas_crosstalk_scale * meas_scale * scale)
            || !probability(meas_crosstalk_local * meas_crosstalk_scale * meas_scale * scale)
        {
            return Err(
                "scaled measurement-crosstalk probabilities must be between zero and one"
                    .to_string(),
            );
        }

        let (a, b, c, d) = self
            .p2_angle_params
            .unwrap_or_else(|| default.p2_angle_params());
        if ![b, a + b, d, c + d, (b + d) * 0.5]
            .into_iter()
            .map(|multiplier| scaled_p2 * multiplier)
            .all(probability)
        {
            return Err(
                "angle-dependent two-qubit probabilities must remain between zero and one"
                    .to_string(),
            );
        }
        let angle_power = self
            .p2_angle_power
            .unwrap_or_else(|| default.p2_angle_power());
        if !angle_power.is_finite() || angle_power <= 0.0 {
            return Err("two-qubit angle power must be finite and positive".to_string());
        }

        let linear_rate = self
            .p_idle_linear_rate
            .unwrap_or(default.p_idle_linear_rate)
            * (scale * idle_scale);
        let sine_rate = self
            .p_idle_sin_squared
            .as_ref()
            .map_or(default.p_idle_sin_squared_rate, |(rate, _)| *rate)
            * (scale * idle_scale);
        let coherent_rate = self
            .p_idle_coherent
            .as_ref()
            .map_or(default.p_idle_coherent_rate, |(rate, _)| *rate)
            * (scale * idle_scale);
        if ![linear_rate, sine_rate, coherent_rate]
            .into_iter()
            .all(finite_non_negative)
        {
            return Err("scaled idle rates must be finite and non-negative".to_string());
        }
        let duration = self.idle_after_2q.unwrap_or(default.idle_after_2q);
        let coherent_model = self
            .p_idle_coherent
            .as_ref()
            .map_or(&default.p_idle_coherent_model, |(_, model)| model);
        let sine_model = self
            .p_idle_sin_squared
            .as_ref()
            .map_or(&default.p_idle_sin_squared_model, |(_, model)| model);
        GeneralNoiseModel::validate_idle_products(
            duration,
            "idle_after_2q",
            linear_rate,
            sine_rate,
            sine_model,
            coherent_rate,
            coherent_model,
        )?;
        Ok(())
    }

    /// Validate that a duration is finite and non-negative
    fn validate_duration(duration: f64) -> Result<f64, String> {
        if !(duration.is_finite() && duration >= 0.0) {
            return Err(format!(
                "Duration must be finite and non-negative, got {duration}"
            ));
        }
        Ok(duration)
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
    /// physics). Returns `None` if a setter error is recorded. This exists so
    /// other simulation stacks can translate the common configuration without
    /// re-deriving probability conventions.
    #[must_use]
    pub fn simple_probabilities(&self) -> Option<(f64, f64, f64, f64, f64)> {
        let zero_or_unset = |v: Option<f64>| v.is_none() || v == Some(0.0);
        let emission_off =
            zero_or_unset(self.p1_emission_ratio) && zero_or_unset(self.p2_emission_ratio);
        if self.is_plain_pauli_except_angle_and_emission(false)
            && self.resolved_angle_scaling().is_none()
            && emission_off
        {
            Some(self.resolved_base_probabilities())
        } else {
            None
        }
    }

    // Frame events execute as unnoised physical commands. Hooks may coexist
    // with them: a FLIP on a hook-leaked qubit only changes the simulator's
    // placeholder state, which no readout in this profile can observe (MZ reads
    // 1, MeasureLeaked reads 2, and every exit from leakage resets the qubit).
    // Other callers of simple_probabilities must still reject hooks they
    // cannot translate.
    // Returns false if a setter error is recorded.
    pub(crate) fn runtime_frame_supports_physical_events(&self) -> bool {
        self.is_plain_pauli_except_angle_and_emission(true)
            && self.resolved_angle_scaling().is_none()
            && self.p1_emission_ratio.is_none_or(|p| p == 0.0)
            && self.p2_emission_ratio.is_none_or(|p| p == 0.0)
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
    /// noiseless gates must be off). Returns `None` if a setter error is recorded.
    #[must_use]
    pub fn pauli_with_angle_scaling(&self) -> Option<PauliWithAngleScaling> {
        if self.is_plain_pauli_except_angle_and_emission(false) {
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
    fn is_plain_pauli_except_angle_and_emission(&self, allow_hooks: bool) -> bool {
        if self.setter_error().is_some() {
            return false;
        }
        let zero_or_unset = |v: Option<f64>| v.is_none() || v == Some(0.0);
        let one_or_unset = |v: Option<f64>| v.is_none() || v == Some(1.0);

        // Emission ratios are intentionally NOT required off here: they are handled separately,
        // since neo matches engines' gate-removing emission with the default uniform distribution.
        let optional_features_off = zero_or_unset(self.p_prep_leak_ratio)
            && zero_or_unset(self.p_idle_linear_rate)
            && self.p_idle_sin_squared.is_none()
            && self.p_idle_coherent.is_none()
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

        let hooks_off = self.p1_pauli_leakage_channels_before_gate.is_empty()
            && self.p1_pauli_leakage_channels_after_gate.is_empty()
            && self.p1_transition_channels_before_gate.is_empty()
            && self.p1_transition_channels_after_gate.is_empty()
            && self.p2_pauli_leakage_steps_before_gate.is_empty()
            && self.p2_pauli_leakage_steps_after_gate.is_empty()
            && self.p2_transition_steps_before_gate.is_empty()
            && self.p2_transition_steps_after_gate.is_empty();

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

        optional_features_off
            && custom_models_off
            && (allow_hooks || hooks_off)
            && scales_neutral
            && gates_default
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
    ///
    /// # Panics
    /// Panics if a setter error is recorded, or if the model's scaled hook probabilities
    /// are not finite or lie outside `[0, 1]`.
    pub fn scale_parameters(&mut self, model: &mut GeneralNoiseModel) {
        if let Some(message) = self.setter_error() {
            panic!("{message}");
        }
        validate_hook_probabilities(
            [
                &model.p1_transition_channels_before_gate,
                &model.p1_transition_channels_after_gate,
            ],
            [
                &model.p1_pauli_leakage_channels_before_gate,
                &model.p1_pauli_leakage_channels_after_gate,
            ],
            [
                &model.p2_transition_steps_before_gate,
                &model.p2_transition_steps_after_gate,
            ],
            [
                &model.p2_pauli_leakage_steps_before_gate,
                &model.p2_pauli_leakage_steps_after_gate,
            ],
            self.p1_scale.unwrap_or(1.0) * self.scale.unwrap_or(1.0),
            self.p2_scale.unwrap_or(1.0) * self.scale.unwrap_or(1.0),
        )
        .expect("invalid scaled gate-hook probabilities");
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
        for channel in &mut model.p1_transition_channels_before_gate {
            channel.scale_probability(p1_scale * scale);
        }
        for channel in &mut model.p1_transition_channels_after_gate {
            channel.scale_probability(p1_scale * scale);
        }
        for channel in &mut model.p1_pauli_leakage_channels_before_gate {
            channel.scale_probability(p1_scale * scale);
        }
        for channel in &mut model.p1_pauli_leakage_channels_after_gate {
            channel.scale_probability(p1_scale * scale);
        }

        // Scale two-qubit gate error probability
        model.p2 *= p2_scale * scale;
        for step in &mut model.p2_transition_steps_before_gate {
            step.scale_probability(p2_scale * scale);
        }
        for step in &mut model.p2_transition_steps_after_gate {
            step.scale_probability(p2_scale * scale);
        }
        for step in &mut model.p2_pauli_leakage_steps_before_gate {
            step.scale_probability(p2_scale * scale);
        }
        for step in &mut model.p2_pauli_leakage_steps_after_gate {
            step.scale_probability(p2_scale * scale);
        }

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

        model.p_idle_linear_rate *= scale * idle_scale;
        model.p_idle_sin_squared_rate *= scale * idle_scale;
        model.p_idle_coherent_rate *= scale * idle_scale;
    }
}

impl crate::noise::IntoNoiseModel for GeneralNoiseModelBuilder {
    fn into_noise_model(self) -> Result<Box<dyn crate::noise::NoiseModel>, PecosError> {
        Ok(Box::new(self.try_build()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::TwoQubitStateTransitionChannel;

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
    }

    #[test]
    fn explicit_setter_beats_auto_in_both_orders() {
        let linear_model = BTreeMap::from([
            ("X".to_string(), 1.0 / 3.0),
            ("Y".to_string(), 1.0 / 3.0),
            ("Z".to_string(), 1.0 / 3.0),
        ]);
        let set_explicit = |builder: GeneralNoiseModelBuilder| {
            builder
                .with_p_prep(0.11)
                .with_p_meas_0(0.12)
                .with_p_meas_1(0.13)
                .with_p1(0.14)
                .with_p2(0.15)
                .with_p_idle_linear(0.16, &linear_model)
                .with_p1_emission_ratio(0.17)
                .with_p2_emission_ratio(0.18)
                .with_prep_leak_ratio(0.19)
                .with_p1_seepage_prob(0.20)
                .with_p2_seepage_prob(0.21)
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
        }
    }

    #[test]
    fn retired_idle_builder_state_and_setters_are_absent_from_rust_source() {
        let source = include_str!("builder.rs");
        let removed_setters = [
            "with_p_idle_linear_rate",
            "with_p_idle_linear_model",
            "with_p_idle_quadratic_rate",
            "with_p_idle_quadratic_coherent",
            "with_p_idle_coherent_to_incoherent_factor",
            "with_average_p_idle_linear_rate",
            "with_average_p_idle_quadratic_rate",
        ];

        for setter in removed_setters {
            assert!(
                !source.contains(&format!("pub fn {setter}")),
                "{setter} must not remain on GeneralNoiseModelBuilder"
            );
        }

        let removed_factor = ["p_idle_coherent_to_", "incoherent_factor"].concat();
        assert!(
            !source.contains(&format!("{removed_factor}: Option"))
                && !source.contains(&format!("self.{removed_factor}"))
                && !source.contains(&format!("model.{removed_factor}")),
            "the orphaned coherent-to-incoherent factor must not remain in builder state or auto()"
        );
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
            .with_prep_leak_ratio(0.0);

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
            .with_prep_leak_ratio(0.0);

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
            .with_prep_leak_ratio(0.0);

        // The built model applies the scale (0.25 * 2.0 = 0.5)...
        let built = builder.clone().build();
        assert!((built.p1_emission_ratio() - 0.5).abs() < 1e-12);
        // ...so surfacing the raw 0.25 would be a cross-stack mismatch: the
        // subset must refuse it.
        assert!(builder.pauli_with_angle_scaling().is_none());
    }

    #[test]
    fn simple_subsets_reject_transition_channel() {
        let channel = StateTransitionChannel::leak_recovery(0.9, 0.5).unwrap();
        let builder =
            GeneralNoiseModelBuilder::new().add_p2_transition_channel_after_gate(&channel);

        assert!(builder.simple_probabilities().is_none());
        assert!(builder.pauli_with_angle_scaling().is_none());
    }

    #[test]
    fn p2_transition_channel_methods_accept_joint_channels_directly() {
        let transitions = BTreeMap::from([(
            "LL".to_string(),
            BTreeMap::from([("00".to_string(), 0.9), ("LL".to_string(), 0.1)]),
        )]);
        let channel = TwoQubitStateTransitionChannel::new(1.0, &transitions).unwrap();
        let model = GeneralNoiseModelBuilder::new()
            .add_p2_transition_channel_before_gate(&channel)
            .add_p2_transition_channel_after_gate(&channel)
            .build();

        assert_eq!(model.p2_transition_steps_before_gate.len(), 1);
        assert_eq!(model.p2_transition_steps_after_gate.len(), 1);
    }

    #[test]
    fn transition_application_probability_uses_gate_and_global_scales() {
        let channel = StateTransitionChannel::leak_recovery(0.2, 0.5).unwrap();
        let model = GeneralNoiseModelBuilder::new()
            .add_p1_transition_channel_after_gate(&channel)
            .add_p2_transition_channel_after_gate(&channel)
            .with_scale(2.0)
            .with_p1_scale(2.0)
            .with_p2_scale(0.5)
            .build();

        assert!(
            (model
                .p1_transition_channels_after_gate
                .first()
                .unwrap()
                .probability()
                - 0.8)
                .abs()
                < f64::EPSILON
        );
        match model.p2_transition_steps_after_gate.first().unwrap() {
            P2TransitionWeightedStep::Independent { first, second } => {
                assert!((first.probability() - 0.2).abs() < f64::EPSILON);
                assert!((second.probability() - 0.2).abs() < f64::EPSILON);
            }
            P2TransitionWeightedStep::Joint { .. } => panic!("expected independent step"),
        }
    }

    #[test]
    fn pauli_leakage_application_probability_uses_gate_and_global_scales() {
        let channel =
            PauliLeakageChannel::new(0.2, &BTreeMap::from([("L".to_string(), 1.0)])).unwrap();
        let model = GeneralNoiseModelBuilder::new()
            .add_p1_pauli_leakage_channel_after_gate(&channel)
            .add_p2_pauli_leakage_channel_after_gate(&channel)
            .with_scale(2.0)
            .with_p1_scale(2.0)
            .with_p2_scale(0.5)
            .build();

        assert!(
            (model
                .p1_pauli_leakage_channels_after_gate
                .first()
                .unwrap()
                .probability()
                - 0.8)
                .abs()
                < f64::EPSILON
        );
        match model.p2_pauli_leakage_steps_after_gate.first().unwrap() {
            P2PauliLeakageWeightedStep::Independent { first, second } => {
                assert!((first.probability() - 0.2).abs() < f64::EPSILON);
                assert!((second.probability() - 0.2).abs() < f64::EPSILON);
            }
            P2PauliLeakageWeightedStep::Joint(_) => panic!("expected independent step"),
        }
    }

    #[test]
    fn simple_subsets_reject_pauli_leakage_channels() {
        let channel =
            PauliLeakageChannel::new(0.2, &BTreeMap::from([("L".to_string(), 1.0)])).unwrap();
        let builder =
            GeneralNoiseModelBuilder::new().add_p2_pauli_leakage_channel_after_gate(&channel);

        assert!(builder.simple_probabilities().is_none());
        assert!(builder.pauli_with_angle_scaling().is_none());
    }

    #[test]
    fn cross_field_validation_rejects_invalid_effective_probabilities() {
        assert_eq!(
            GeneralNoiseModelBuilder::new()
                .with_p1(0.8)
                .with_scale(2.0)
                .validate_configuration(),
            Err("scaled one-qubit probability must be between zero and one".to_string())
        );
        assert_eq!(
            GeneralNoiseModelBuilder::new()
                .with_p2(0.8)
                .with_p2_angle_params(1.0, 1.0, 0.0, 1.0)
                .validate_configuration(),
            Err(
                "angle-dependent two-qubit probabilities must remain between zero and one"
                    .to_string()
            )
        );
        assert!(
            GeneralNoiseModelBuilder::new()
                .with_p1(0.4)
                .with_scale(2.0)
                .validate_configuration()
                .is_ok()
        );
    }

    #[test]
    fn idle_scale_applies_to_every_idle_family() {
        let linear = BTreeMap::from([("X".to_string(), 1.0)]);
        let sine = BTreeMap::from([("Z".to_string(), 1.0)]);
        let coherent = BTreeMap::from([("RZ".to_string(), 1.0)]);
        let model = GeneralNoiseModelBuilder::new()
            .with_p_idle_linear(0.2, &linear)
            .with_p_idle_sin_squared(0.3, &sine)
            .with_p_idle_coherent(0.4, &coherent)
            .with_idle_scale(0.5)
            .build();

        assert_float_eq(model.p_idle_linear_rate, 0.1);
        assert_float_eq(model.p_idle_sin_squared_rate, 0.15);
        assert_float_eq(model.p_idle_coherent_rate, 0.2);
    }

    #[test]
    #[should_panic(expected = "invalid scaled gate-hook probabilities")]
    fn direct_scaling_checks_the_supplied_models_hooks() {
        let channel = StateTransitionChannel::leak_recovery(0.8, 1.0).unwrap();
        let mut model = GeneralNoiseModelBuilder::new()
            .add_p1_transition_channel_before_gate(&channel)
            .build();
        GeneralNoiseModelBuilder::new()
            .with_scale(2.0)
            .scale_parameters(&mut model);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0 and 1, got 1.5")]
    fn scale_parameters_rejects_a_recorded_setter_error() {
        GeneralNoiseModelBuilder::new()
            .with_p1(1.5)
            .scale_parameters(&mut GeneralNoiseModel::default());
    }

    fn assert_setter_error(builder: GeneralNoiseModelBuilder, expected: &str) {
        assert_eq!(builder.setter_error(), Some(expected));
        assert_eq!(builder.validate_configuration(), Err(expected.to_string()));
        assert!(builder.simple_probabilities().is_none());
        assert!(builder.pauli_with_angle_scaling().is_none());
        assert!(!builder.runtime_frame_supports_physical_events());
        let error = builder.try_build().expect_err("invalid builder must fail");
        assert!(matches!(error, PecosError::Input(ref message) if message == expected));
    }

    #[test]
    fn all_probability_setters_record_errors() {
        for value in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let expected = format!("Probability must be between 0 and 1, got {value}");
            for setter in [
                GeneralNoiseModelBuilder::with_leakage_scale,
                GeneralNoiseModelBuilder::with_seepage_prob,
                GeneralNoiseModelBuilder::with_p_prep,
                GeneralNoiseModelBuilder::with_prep_leak_ratio,
                GeneralNoiseModelBuilder::with_p_prep_crosstalk,
                GeneralNoiseModelBuilder::with_p1,
                GeneralNoiseModelBuilder::with_p1_emission_ratio,
                GeneralNoiseModelBuilder::with_p1_seepage_prob,
                GeneralNoiseModelBuilder::with_p2,
                GeneralNoiseModelBuilder::with_p2_emission_ratio,
                GeneralNoiseModelBuilder::with_p2_seepage_prob,
                GeneralNoiseModelBuilder::with_p_meas_0,
                GeneralNoiseModelBuilder::with_p_meas_1,
                GeneralNoiseModelBuilder::with_p_meas,
                GeneralNoiseModelBuilder::with_p_meas_crosstalk_global,
                GeneralNoiseModelBuilder::with_p_meas_crosstalk_local,
                GeneralNoiseModelBuilder::with_p_meas_crosstalk,
            ] {
                assert_setter_error(setter(GeneralNoiseModelBuilder::new(), value), &expected);
            }
        }
        assert_setter_error(
            GeneralNoiseModelBuilder::new().with_average_p1(1.0),
            "Probability must be between 0 and 1, got 1.5",
        );
        assert_setter_error(
            GeneralNoiseModelBuilder::new().with_average_p2(1.0),
            "Probability must be between 0 and 1, got 1.25",
        );
    }

    #[test]
    fn all_scalar_validation_helpers_record_errors() {
        for value in [-1.0, f64::NAN] {
            for (builder, message) in [
                (
                    GeneralNoiseModelBuilder::new().with_p_prep_crosstalk_scale(value),
                    format!(
                        "Preparation crosstalk rescale factor must be non-negative, got {value}"
                    ),
                ),
                (
                    GeneralNoiseModelBuilder::new().with_p_meas_crosstalk_scale(value),
                    format!(
                        "Measurement crosstalk rescale factor must be non-negative, got {value}"
                    ),
                ),
                (
                    GeneralNoiseModelBuilder::new().with_p2_angle_power(value),
                    format!("RZZ power parameter must be positive, got {value}"),
                ),
                (
                    GeneralNoiseModelBuilder::new()
                        .with_p_idle_linear(value, &BTreeMap::from([("X".to_string(), 1.0)])),
                    format!("linear idling rate must be non-negative, got {value}"),
                ),
            ] {
                assert_setter_error(builder, &message);
            }
        }
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            assert_setter_error(
                GeneralNoiseModelBuilder::new().with_idle_after_2q(value),
                &format!("Duration must be finite and non-negative, got {value}"),
            );
            assert_setter_error(
                GeneralNoiseModelBuilder::new().with_p_idle_sin_squared(value, &BTreeMap::new()),
                &format!("sine-squared idling rate must be finite and non-negative, got {value}"),
            );
            assert_setter_error(
                GeneralNoiseModelBuilder::new().with_p_idle_coherent(value, &BTreeMap::new()),
                &format!("coherent idling rate must be finite and non-negative, got {value}"),
            );
        }
        for value in [-0.1, 0.3, f64::NAN, f64::INFINITY] {
            assert_setter_error(
                GeneralNoiseModelBuilder::new()
                    .with_average_p_prep_crosstalk(value)
                    .with_prep_scale(0.01),
                &format!(
                    "average preparation crosstalk probability must be between 0 and 5/18, got {value}"
                ),
            );
        }
        for value in [0.0, 5.0 / 18.0] {
            assert!(
                GeneralNoiseModelBuilder::new()
                    .with_average_p_prep_crosstalk(value)
                    .try_build()
                    .is_ok()
            );
        }
    }

    #[test]
    fn sampler_setters_record_errors_without_replacing_fields() {
        let valid_one = BTreeMap::from([("X".to_string(), 1.0)]);
        let valid_two = BTreeMap::from([("XI".to_string(), 1.0)]);
        let valid_cross = BTreeMap::from([("0->1".to_string(), 1.0), ("1->0".to_string(), 1.0)]);
        let initial = GeneralNoiseModelBuilder::new()
            .with_p1_pauli_model(&valid_one)
            .with_p1_emission_model(&valid_one)
            .with_p2_pauli_model(&valid_two)
            .with_p2_emission_model(&valid_two)
            .with_p_meas_crosstalk_model(&valid_cross);
        for invalid in [
            BTreeMap::new(),
            BTreeMap::from([("bad".to_string(), 1.0)]),
            BTreeMap::from([("X".to_string(), -1.0)]),
            BTreeMap::from([("X".to_string(), 2.0)]),
        ] {
            let builder = initial
                .clone()
                .with_p1_pauli_model(&invalid)
                .with_p1_emission_model(&invalid)
                .with_p2_pauli_model(&invalid)
                .with_p2_emission_model(&invalid)
                .with_p_meas_crosstalk_model(&invalid);
            assert!(builder.setter_error().is_some());
            assert_eq!(
                builder.p1_pauli_model.as_ref().unwrap().get_weighted_map(),
                &valid_one
            );
            assert_eq!(
                builder
                    .p1_emission_model
                    .as_ref()
                    .unwrap()
                    .get_weighted_map(),
                &valid_one
            );
            assert_eq!(
                builder.p2_pauli_model.as_ref().unwrap().get_weighted_map(),
                &valid_two
            );
            assert_eq!(
                builder
                    .p2_emission_model
                    .as_ref()
                    .unwrap()
                    .get_weighted_map(),
                &valid_two
            );
            assert_eq!(
                builder
                    .p_meas_crosstalk_model
                    .as_ref()
                    .unwrap()
                    .get_weighted_map(0),
                &BTreeMap::from([("0->1".to_string(), 1.0)])
            );
            assert_setter_error(builder.clone(), builder.setter_error().unwrap());
        }
    }

    #[test]
    fn multi_field_setters_are_atomic_and_first_error_survives() {
        let initial = GeneralNoiseModelBuilder::new()
            .with_p1_seepage_prob(0.1)
            .with_p2_seepage_prob(0.2)
            .with_p_meas_0(0.3)
            .with_p_meas_1(0.4)
            .with_p_meas_crosstalk_global(0.5)
            .with_p_meas_crosstalk_local(0.6);
        let rejected = initial
            .with_seepage_prob(2.0)
            .with_p_meas(-1.0)
            .with_p_meas_crosstalk(3.0);
        assert_eq!(
            (rejected.p1_seepage_prob, rejected.p2_seepage_prob),
            (Some(0.1), Some(0.2))
        );
        assert_eq!(
            (rejected.p_meas_0, rejected.p_meas_1),
            (Some(0.3), Some(0.4))
        );
        assert_eq!(
            (
                rejected.p_meas_crosstalk_global,
                rejected.p_meas_crosstalk_local
            ),
            (Some(0.5), Some(0.6))
        );
        assert_setter_error(
            rejected.with_p1(0.2).auto(),
            "Probability must be between 0 and 1, got 2",
        );

        let valid = BTreeMap::from([("X".to_string(), 1.0)]);
        let invalid = BTreeMap::from([("bad".to_string(), 1.0)]);
        let initial = GeneralNoiseModelBuilder::new()
            .with_p_idle_linear(0.1, &valid)
            .with_p_idle_sin_squared(0.2, &valid)
            .with_p_idle_coherent(0.3, &BTreeMap::from([("RX".to_string(), 1.0)]));
        for rejected in [
            initial.clone().with_p_idle_linear(0.9, &invalid),
            initial
                .clone()
                .with_p_idle_linear(-1.0, &BTreeMap::from([("Z".to_string(), 1.0)])),
        ] {
            assert_eq!(rejected.p_idle_linear_rate, initial.p_idle_linear_rate);
            assert_eq!(
                rejected
                    .p_idle_linear_model
                    .as_ref()
                    .unwrap()
                    .get_weighted_map(),
                &valid
            );
            assert!(rejected.setter_error().is_some());
        }
        for rejected in [
            initial.clone().with_p_idle_sin_squared(0.9, &invalid),
            initial.clone().with_p_idle_sin_squared(-1.0, &valid),
        ] {
            assert_eq!(rejected.p_idle_sin_squared, initial.p_idle_sin_squared);
            assert!(rejected.setter_error().is_some());
        }
        for rejected in [
            initial.clone().with_p_idle_coherent(0.9, &invalid),
            initial.clone().with_p_idle_coherent(-1.0, &BTreeMap::new()),
        ] {
            assert_eq!(rejected.p_idle_coherent, initial.p_idle_coherent);
            assert!(rejected.setter_error().is_some());
        }
        let rejected = GeneralNoiseModelBuilder::new().with_p1(0.2).with_p1(1.5);
        assert_eq!(rejected.p1, Some(0.2));
        assert_setter_error(rejected, "Probability must be between 0 and 1, got 1.5");
    }

    #[test]
    fn setter_reports_probability_message() {
        let builder = GeneralNoiseModelBuilder::new().with_p1(1.5);
        assert_setter_error(builder, "Probability must be between 0 and 1, got 1.5");
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0 and 1, got 1.5")]
    fn build_panics_with_raw_probability_message() {
        let builder = GeneralNoiseModelBuilder::new().with_p1(1.5);
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_scale_message() {
        let builder = GeneralNoiseModelBuilder::new().with_p_prep_crosstalk_scale(-1.0);
        assert_setter_error(
            builder,
            "Preparation crosstalk rescale factor must be non-negative, got -1",
        );
    }

    #[test]
    #[should_panic(expected = "Preparation crosstalk rescale factor must be non-negative, got -1")]
    fn build_panics_with_raw_scale_message() {
        let builder = GeneralNoiseModelBuilder::new().with_p_prep_crosstalk_scale(-1.0);
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_sampler_key_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p1_pauli_model(&BTreeMap::from([("bad".to_string(), 1.0)]));
        assert_setter_error(
            builder,
            "SingleQubitWeightedSampler: invalid key 'bad' - must be one of X, Y, Z, or L",
        );
    }

    #[test]
    #[should_panic(
        expected = "SingleQubitWeightedSampler: invalid key 'bad' - must be one of X, Y, Z, or L"
    )]
    fn build_panics_with_raw_sampler_key_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p1_pauli_model(&BTreeMap::from([("bad".to_string(), 1.0)]));
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_sampler_weights_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p2_pauli_model(&BTreeMap::from([("XI".to_string(), 2.0)]));
        assert_setter_error(
            builder,
            "WeightedSampler: total weight 2 deviates from 1.0 by more than tolerance 0.00001",
        );
    }

    #[test]
    #[should_panic(
        expected = "WeightedSampler: total weight 2 deviates from 1.0 by more than tolerance 0.00001"
    )]
    fn build_panics_with_raw_sampler_weights_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p2_pauli_model(&BTreeMap::from([("XI".to_string(), 2.0)]));
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_sine_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(0.1, &BTreeMap::from([("bad".to_string(), 1.0)]));
        assert_setter_error(
            builder,
            "p_idle_sin_squared model has invalid key 'bad'; expected X, Y, Z, or L",
        );
    }

    #[test]
    #[should_panic(
        expected = "p_idle_sin_squared model has invalid key 'bad'; expected X, Y, Z, or L"
    )]
    fn build_panics_with_raw_sine_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(0.1, &BTreeMap::from([("bad".to_string(), 1.0)]));
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_coherent_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_coherent(0.1, &BTreeMap::from([("bad".to_string(), 1.0)]));
        assert_setter_error(
            builder,
            "p_idle_coherent model has invalid key 'bad'; expected RX, RY, or RZ",
        );
    }

    #[test]
    #[should_panic(
        expected = "p_idle_coherent model has invalid key 'bad'; expected RX, RY, or RZ"
    )]
    fn build_panics_with_raw_coherent_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_coherent(0.1, &BTreeMap::from([("bad".to_string(), 1.0)]));
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_duration_message() {
        let builder = GeneralNoiseModelBuilder::new().with_idle_after_2q(-1.0);
        assert_setter_error(builder, "Duration must be finite and non-negative, got -1");
    }

    #[test]
    #[should_panic(expected = "Duration must be finite and non-negative, got -1")]
    fn build_panics_with_raw_duration_message() {
        let builder = GeneralNoiseModelBuilder::new().with_idle_after_2q(-1.0);
        let _ = builder.build();
    }

    #[test]
    fn setter_reports_average_crosstalk_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_average_p_prep_crosstalk(0.3)
            .with_prep_scale(0.01);
        assert_setter_error(
            builder,
            "average preparation crosstalk probability must be between 0 and 5/18, got 0.3",
        );
    }

    #[test]
    #[should_panic(
        expected = "average preparation crosstalk probability must be between 0 and 5/18, got 0.3"
    )]
    fn build_panics_with_raw_average_crosstalk_message() {
        let builder = GeneralNoiseModelBuilder::new()
            .with_average_p_prep_crosstalk(0.3)
            .with_prep_scale(0.01);
        let _ = builder.build();
    }
}
