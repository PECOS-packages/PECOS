use crate::byte_message::ByteMessage;
use crate::noise::{
    CrosstalkWeightedSampler, GeneralNoiseModel, NoiseRng, SingleQubitWeightedSampler,
    TwoQubitWeightedSampler,
};
use std::collections::{BTreeMap, BTreeSet};

impl Default for GeneralNoiseModel {
    /// Create a new noise model with default error parameters
    ///
    /// Creates a `GeneralNoiseModel` with sensible default error probabilities:
    /// * `p_prep` - Preparation (initialization) error probability: 0.01
    /// * `p_meas_0` - Probability of measuring 1 when the state is |0⟩: 0.01
    /// * `p_meas_1` - Probability of measuring 0 when the state is |1⟩: 0.01
    /// * `p1` - Single-qubit gate error probability (average error rate): 0.001
    /// * `p2` - Two-qubit gate error probability (average error rate): 0.01
    ///
    /// Other parameters are initialized with sensible defaults, including uniform
    /// distributions for Pauli errors and emission errors.
    ///
    /// # Example
    /// ```
    /// use pecos_engines::noise::GeneralNoiseModel;
    ///
    /// // Create model with default error probabilities
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

        let p_meas_0: f64 = 0.01; // 1% probability of measuring 1 when state is |0⟩
        let p_meas_1: f64 = 0.01; // 1% probability of measuring 0 when state is |1⟩

        let mut p_meas_crosstalk_model = BTreeMap::new();
        p_meas_crosstalk_model.insert("0->0".to_string(), 1.0);
        p_meas_crosstalk_model.insert("1->1".to_string(), 1.0);

        // Default error probabilities
        Self {
            p_prep: 0.01,
            p_idle_coherent: false,
            p_idle_linear_rate: 0.001,
            p_idle_linear_model: SingleQubitWeightedSampler::new(&p1_pauli_model),
            p_idle_quadratic_rate: 0.0,
            p_meas_0,
            p_meas_1,
            p1: 0.001,
            p2: 0.01,
            p1_emission_ratio: 0.5,
            p_prep_leak_ratio: 0.5,
            p2_emission_ratio: 0.5,
            p1_pauli_model: SingleQubitWeightedSampler::new(&p1_pauli_model),
            p1_emission_model: SingleQubitWeightedSampler::new(&p1_emission_model),
            p2_pauli_model: TwoQubitWeightedSampler::new(&p2_pauli_model),
            p2_emission_model: TwoQubitWeightedSampler::new(&p2_emission_model),
            p1_seepage_prob: 0.5,
            p2_seepage_prob: 0.5,
            p2_angle_a: 0.0,
            p2_angle_b: 1.0,
            p2_angle_c: 0.0,
            p2_angle_d: 1.0,
            p2_angle_power: 1.0,
            p2_idle: 0.0,
            leaked_qubits: BTreeSet::new(),
            rng: NoiseRng::default(),
            prepared_qubits: BTreeSet::new(),
            measured_qubits: Vec::new(),
            p_meas_crosstalk_global: 0.0,
            p_meas_crosstalk_local: 0.0,
            p_meas_crosstalk_model: CrosstalkWeightedSampler::new(&p_meas_crosstalk_model),
            p_prep_crosstalk: 0.0,

            p_idle_coherent_to_incoherent_factor: 1.5,
            noiseless_gates: BTreeSet::new(),
            p_meas_max: p_meas_0.max(p_meas_1),
            leakage_scale: 1.0,
            results_builder: ByteMessage::outcomes_builder(),
        }
    }
}
