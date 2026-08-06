use crate::byte_message::ByteMessage;
use crate::noise::{
    CrosstalkWeightedSampler, GeneralNoiseModel, NoiseRng, SingleQubitWeightedSampler,
    TwoQubitWeightedSampler,
};
use std::collections::{BTreeMap, BTreeSet};

impl Default for GeneralNoiseModel {
    /// Create a noiseless general noise model.
    ///
    /// All rates and probabilities use their no-effect value. Multipliers and angle-scaling
    /// parameters use their identity values, and the sampler distributions remain uniform so
    /// that explicitly enabling a rate without replacing its sampler remains well-defined.
    ///
    /// # Example
    /// ```
    /// use pecos_engines::noise::GeneralNoiseModel;
    ///
    /// // The default model adds no noise.
    /// let mut model = GeneralNoiseModel::default();
    /// ```
    fn default() -> Self {
        // Initialize default models
        let mut p1_pauli_model = BTreeMap::new();
        p1_pauli_model.insert("X".to_string(), 1.0 / 3.0);
        p1_pauli_model.insert("Y".to_string(), 1.0 / 3.0);
        p1_pauli_model.insert("Z".to_string(), 1.0 / 3.0);

        let mut p1_emission_model = BTreeMap::new();
        p1_emission_model.insert("X".to_string(), 1.0 / 3.0);
        p1_emission_model.insert("Y".to_string(), 1.0 / 3.0);
        p1_emission_model.insert("Z".to_string(), 1.0 / 3.0);

        let mut p2_pauli_model = BTreeMap::new();
        p2_pauli_model.insert("XX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("XY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("XZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("IX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("IY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("IZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("XI".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YI".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZI".to_string(), 1.0 / 15.0);

        let mut p2_emission_model = BTreeMap::new();
        p2_emission_model.insert("XX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("XY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("XZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("IX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("IY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("IZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("XI".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YI".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZI".to_string(), 1.0 / 15.0);

        let p_meas_0: f64 = 0.0;
        let p_meas_1: f64 = 0.0;

        let mut p_meas_crosstalk_model = BTreeMap::new();
        p_meas_crosstalk_model.insert("0->0".to_string(), 1.0);
        p_meas_crosstalk_model.insert("1->1".to_string(), 1.0);

        let p_idle_coherent_model = BTreeMap::from([
            ("RX".to_string(), 1.0),
            ("RY".to_string(), 1.0),
            ("RZ".to_string(), 1.0),
        ]);

        // No-effect defaults
        Self {
            p_prep: 0.0,
            p_idle_linear_rate: 0.0,
            p_idle_linear_model: SingleQubitWeightedSampler::new(&p1_pauli_model),
            p_idle_sin_squared_rate: 0.0,
            p_idle_sin_squared_model: BTreeMap::new(),
            p_idle_coherent_rate: 0.0,
            p_idle_coherent_model,
            p_meas_0,
            p_meas_1,
            p1: 0.0,
            p2: 0.0,
            p1_emission_ratio: 0.0,
            p_prep_leak_ratio: 0.0,
            p2_emission_ratio: 0.0,
            p1_pauli_model: SingleQubitWeightedSampler::new(&p1_pauli_model),
            p1_emission_model: SingleQubitWeightedSampler::new(&p1_emission_model),
            p2_pauli_model: TwoQubitWeightedSampler::new(&p2_pauli_model),
            p2_emission_model: TwoQubitWeightedSampler::new(&p2_emission_model),
            p1_seepage_prob: 0.0,
            p2_seepage_prob: 0.0,
            p2_angle_a: 0.0,
            p2_angle_b: 1.0,
            p2_angle_c: 0.0,
            p2_angle_d: 1.0,
            p2_angle_power: 1.0,
            idle_after_2q: 0.0,
            leaked_qubits: BTreeSet::new(),
            rng: NoiseRng::default(),
            prepared_qubits: BTreeSet::new(),
            measured_qubits: Vec::new(),
            p_meas_crosstalk_global: 0.0,
            p_meas_crosstalk_local: 0.0,
            p_meas_crosstalk_model: CrosstalkWeightedSampler::new(&p_meas_crosstalk_model),
            p_prep_crosstalk: 0.0,
            noiseless_gates: BTreeSet::new(),
            p_meas_max: p_meas_0.max(p_meas_1),
            leakage_scale: 1.0,
            results_builder: ByteMessage::outcomes_builder(),
        }
    }
}
