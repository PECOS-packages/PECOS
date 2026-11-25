//! Read-only snapshots of built, scaled noise settings. Shot state and RNG state are excluded.
use super::loss::LossDescription;
use super::{GeneralNoiseModel, HookChannel};
use crate::GateType;
use std::collections::{BTreeMap, BTreeSet};

/// An ordered hook entry; the index addresses the corresponding family vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDescription {
    /// Population transition channel.
    Transition(usize),
    /// Pauli/leakage channel.
    PauliLeakage(usize),
    /// Loss channel.
    Loss(usize),
}

/// Built noise settings, copied without exposing mutable access to the model.
/// All rates include builder scales. Hook vectors retain scaled outer probabilities;
/// two-qubit entries distinguish independent legs from a joint channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralNoiseDescription {
    /// Compiled `noiseless_gates` setting.
    pub noiseless_gates: BTreeSet<GateType>,
    /// Compiled `readout` setting.
    pub readout: BTreeMap<GateType, BTreeMap<String, BTreeMap<u8, f64>>>,
    /// Compiled `loss` setting.
    pub loss: Option<LossDescription>,
    /// Compiled `leakage_scale` setting.
    pub leakage_scale: f64,
    /// Compiled `p_idle_linear_rate` setting.
    pub p_idle_linear_rate: f64,
    /// Compiled `p_idle_linear_model` setting.
    pub p_idle_linear_model: BTreeMap<String, f64>,
    /// Compiled `p_idle_sin_squared_rate` setting.
    pub p_idle_sin_squared_rate: f64,
    /// Compiled `p_idle_sin_squared_model` setting.
    pub p_idle_sin_squared_model: BTreeMap<String, f64>,
    /// Compiled `p_idle_coherent_rate` setting.
    pub p_idle_coherent_rate: f64,
    /// Compiled `p_idle_coherent_model` setting.
    pub p_idle_coherent_model: BTreeMap<String, f64>,
    /// Compiled `p_prep` setting.
    pub p_prep: f64,
    /// Compiled `p_prep_leak_ratio` setting.
    pub p_prep_leak_ratio: f64,
    /// Compiled `p_prep_crosstalk` setting.
    pub p_prep_crosstalk: f64,
    /// Compiled `p1` setting.
    pub p1: f64,
    /// Compiled `p1_emission_ratio` setting.
    pub p1_emission_ratio: f64,
    /// Compiled `p1_emission_model` setting.
    pub p1_emission_model: BTreeMap<String, f64>,
    /// Compiled `p1_seepage_prob` setting.
    pub p1_seepage_prob: f64,
    /// Compiled `p1_pauli_model` setting.
    pub p1_pauli_model: BTreeMap<String, f64>,
    /// Compiled `p1_transition_channels_before_gate` setting.
    pub p1_transition_channels_before_gate: Vec<f64>,
    /// Compiled `p1_transition_channels_after_gate` setting.
    pub p1_transition_channels_after_gate: Vec<f64>,
    /// Compiled `p1_pauli_leakage_channels_before_gate` setting.
    pub p1_pauli_leakage_channels_before_gate: Vec<f64>,
    /// Compiled `p1_pauli_leakage_channels_after_gate` setting.
    pub p1_pauli_leakage_channels_after_gate: Vec<f64>,
    /// Compiled `p2` setting.
    pub p2: f64,
    /// Compiled `p2_angle_a` setting.
    pub p2_angle_a: f64,
    /// Compiled `p2_angle_b` setting.
    pub p2_angle_b: f64,
    /// Compiled `p2_angle_c` setting.
    pub p2_angle_c: f64,
    /// Compiled `p2_angle_d` setting.
    pub p2_angle_d: f64,
    /// Compiled `p2_angle_power` setting.
    pub p2_angle_power: f64,
    /// Compiled `p2_emission_ratio` setting.
    pub p2_emission_ratio: f64,
    /// Compiled `p2_emission_model` setting.
    pub p2_emission_model: BTreeMap<String, f64>,
    /// Compiled `p2_seepage_prob` setting.
    pub p2_seepage_prob: f64,
    /// Compiled `p2_pauli_model` setting.
    pub p2_pauli_model: BTreeMap<String, f64>,
    /// Compiled `p2_transition_steps_before_gate` setting.
    pub p2_transition_steps_before_gate: Vec<HookProbabilities>,
    /// Compiled `p2_transition_steps_after_gate` setting.
    pub p2_transition_steps_after_gate: Vec<HookProbabilities>,
    /// Compiled `p2_pauli_leakage_steps_before_gate` setting.
    pub p2_pauli_leakage_steps_before_gate: Vec<HookProbabilities>,
    /// Compiled `p2_pauli_leakage_steps_after_gate` setting.
    pub p2_pauli_leakage_steps_after_gate: Vec<HookProbabilities>,
    /// Compiled `idle_after_2q` setting.
    pub idle_after_2q: f64,
    /// Compiled `p_meas_0` setting.
    pub p_meas_0: f64,
    /// Compiled `p_meas_1` setting.
    pub p_meas_1: f64,
    /// Compiled `p_meas_crosstalk_global` setting.
    pub p_meas_crosstalk_global: f64,
    /// Compiled `p_meas_crosstalk_local` setting.
    pub p_meas_crosstalk_local: f64,
    /// Compiled `p_meas_crosstalk_model` setting.
    pub p_meas_crosstalk_model: [BTreeMap<String, f64>; 2],
    /// Compiled `p1_hook_order_before_gate` setting.
    pub p1_hook_order_before_gate: Vec<HookDescription>,
    /// Compiled `p1_hook_order_after_gate` setting.
    pub p1_hook_order_after_gate: Vec<HookDescription>,
    /// Compiled `p2_hook_order_before_gate` setting.
    pub p2_hook_order_before_gate: Vec<HookDescription>,
    /// Compiled `p2_hook_order_after_gate` setting.
    pub p2_hook_order_after_gate: Vec<HookDescription>,
}

/// Outer application probabilities for a two-qubit hook.
#[derive(Debug, Clone, PartialEq)]
pub enum HookProbabilities {
    /// Separate coins on the two operands.
    Independent([f64; 2]),
    /// One coin for the joint channel.
    Joint(f64),
}

pub(super) fn is_noiseless_operation(gates: &BTreeSet<GateType>, gate: &crate::Gate) -> bool {
    gates.contains(&gate.gate_type)
        || (gate.phase_angle().is_some() && gates.contains(&GateType::RZ))
}

impl GeneralNoiseDescription {
    /// Apply the engine's scheduled-operation exemption, including phase-shaped U.
    #[must_use]
    pub fn is_noiseless_operation(&self, gate: &crate::Gate) -> bool {
        is_noiseless_operation(&self.noiseless_gates, gate)
    }
}

impl GeneralNoiseModel {
    /// Describe every compiled setting without copying shot-local state.
    /// The exhaustive destructure intentionally fails compilation when a model field is added.
    #[must_use]
    pub fn description(&self) -> GeneralNoiseDescription {
        let Self {
            noiseless_gates,
            readout,
            loss,
            leakage_scale,
            p_idle_linear_rate,
            p_idle_linear_model,
            p_idle_sin_squared_rate,
            p_idle_sin_squared_model,
            p_idle_coherent_rate,
            p_idle_coherent_model,
            p_prep,
            p_prep_leak_ratio,
            p_prep_crosstalk,
            p1,
            p1_emission_ratio,
            p1_emission_model,
            p1_seepage_prob,
            p1_pauli_model,
            p1_transition_channels_before_gate,
            p1_transition_channels_after_gate,
            p1_pauli_leakage_channels_before_gate,
            p1_pauli_leakage_channels_after_gate,
            p2,
            p2_angle_a,
            p2_angle_b,
            p2_angle_c,
            p2_angle_d,
            p2_angle_power,
            p2_emission_ratio,
            p2_emission_model,
            p2_seepage_prob,
            p2_pauli_model,
            p2_transition_steps_before_gate,
            p2_transition_steps_after_gate,
            p2_pauli_leakage_steps_before_gate,
            p2_pauli_leakage_steps_after_gate,
            idle_after_2q,
            p_meas_0,
            p_meas_1,
            p_meas_crosstalk_global,
            p_meas_crosstalk_local,
            p_meas_crosstalk_model,
            p_meas_max: _,
            leaked_qubits: _,
            rng: _,
            prepared_qubits: _,
            measured_qubits: _,
            scheduled_batch_active: _,
            pending_crosstalk: _,
            continuation_failed: _,
            crosstalk_active: _,
            pending_transition_batches: _,
            transition_measurements: _,
            pending_gate_continuation: _,
            pending_hook: _,
            p1_hook_order_before_gate,
            p1_hook_order_after_gate,
            p2_hook_order_before_gate,
            p2_hook_order_after_gate,
            pending_input: _,
            results_builder: _,
        } = self;
        GeneralNoiseDescription {
            noiseless_gates: noiseless_gates.clone(),
            readout: readout
                .iter()
                .map(|(&gate, rows)| (gate, rows.description()))
                .collect(),
            loss: loss.as_ref().map(super::loss::QubitLoss::description),
            leakage_scale: *leakage_scale,
            p_idle_linear_rate: *p_idle_linear_rate,
            p_idle_linear_model: p_idle_linear_model.get_weighted_map().clone(),
            p_idle_sin_squared_rate: *p_idle_sin_squared_rate,
            p_idle_sin_squared_model: p_idle_sin_squared_model.clone(),
            p_idle_coherent_rate: *p_idle_coherent_rate,
            p_idle_coherent_model: p_idle_coherent_model.clone(),
            p_prep: *p_prep,
            p_prep_leak_ratio: *p_prep_leak_ratio,
            p_prep_crosstalk: *p_prep_crosstalk,
            p1: *p1,
            p1_emission_ratio: *p1_emission_ratio,
            p1_emission_model: p1_emission_model.get_weighted_map().clone(),
            p1_seepage_prob: *p1_seepage_prob,
            p1_pauli_model: p1_pauli_model.get_weighted_map().clone(),
            p1_transition_channels_before_gate: p1_transition_channels_before_gate
                .iter()
                .map(crate::noise::weighted_sampler::QubitTransitionWeightedSampler::probability)
                .collect(),
            p1_transition_channels_after_gate: p1_transition_channels_after_gate
                .iter()
                .map(crate::noise::weighted_sampler::QubitTransitionWeightedSampler::probability)
                .collect(),
            p1_pauli_leakage_channels_before_gate: p1_pauli_leakage_channels_before_gate
                .iter()
                .map(crate::noise::weighted_sampler::PauliLeakageWeightedSampler::probability)
                .collect(),
            p1_pauli_leakage_channels_after_gate: p1_pauli_leakage_channels_after_gate
                .iter()
                .map(crate::noise::weighted_sampler::PauliLeakageWeightedSampler::probability)
                .collect(),
            p2: *p2,
            p2_angle_a: *p2_angle_a,
            p2_angle_b: *p2_angle_b,
            p2_angle_c: *p2_angle_c,
            p2_angle_d: *p2_angle_d,
            p2_angle_power: *p2_angle_power,
            p2_emission_ratio: *p2_emission_ratio,
            p2_emission_model: p2_emission_model.get_weighted_map().clone(),
            p2_seepage_prob: *p2_seepage_prob,
            p2_pauli_model: p2_pauli_model.get_weighted_map().clone(),
            p2_transition_steps_before_gate: p2_transition_steps_before_gate
                .iter()
                .map(crate::noise::weighted_sampler::P2TransitionWeightedStep::description)
                .collect(),
            p2_transition_steps_after_gate: p2_transition_steps_after_gate
                .iter()
                .map(crate::noise::weighted_sampler::P2TransitionWeightedStep::description)
                .collect(),
            p2_pauli_leakage_steps_before_gate: p2_pauli_leakage_steps_before_gate
                .iter()
                .map(crate::noise::weighted_sampler::P2PauliLeakageWeightedStep::description)
                .collect(),
            p2_pauli_leakage_steps_after_gate: p2_pauli_leakage_steps_after_gate
                .iter()
                .map(crate::noise::weighted_sampler::P2PauliLeakageWeightedStep::description)
                .collect(),
            idle_after_2q: *idle_after_2q,
            p_meas_0: *p_meas_0,
            p_meas_1: *p_meas_1,
            p_meas_crosstalk_global: *p_meas_crosstalk_global,
            p_meas_crosstalk_local: *p_meas_crosstalk_local,
            p_meas_crosstalk_model: [
                p_meas_crosstalk_model.get_weighted_map(0).clone(),
                p_meas_crosstalk_model.get_weighted_map(1).clone(),
            ],
            p1_hook_order_before_gate: p1_hook_order_before_gate
                .iter()
                .map(|hook| match hook {
                    HookChannel::Transition(i) => HookDescription::Transition(*i),
                    HookChannel::PauliLeakage(i) => HookDescription::PauliLeakage(*i),
                    HookChannel::Loss(i) => HookDescription::Loss(*i),
                })
                .collect(),
            p1_hook_order_after_gate: p1_hook_order_after_gate
                .iter()
                .map(|hook| match hook {
                    HookChannel::Transition(i) => HookDescription::Transition(*i),
                    HookChannel::PauliLeakage(i) => HookDescription::PauliLeakage(*i),
                    HookChannel::Loss(i) => HookDescription::Loss(*i),
                })
                .collect(),
            p2_hook_order_before_gate: p2_hook_order_before_gate
                .iter()
                .map(|hook| match hook {
                    HookChannel::Transition(i) => HookDescription::Transition(*i),
                    HookChannel::PauliLeakage(i) => HookDescription::PauliLeakage(*i),
                    HookChannel::Loss(i) => HookDescription::Loss(*i),
                })
                .collect(),
            p2_hook_order_after_gate: p2_hook_order_after_gate
                .iter()
                .map(|hook| match hook {
                    HookChannel::Transition(i) => HookDescription::Transition(*i),
                    HookChannel::PauliLeakage(i) => HookDescription::PauliLeakage(*i),
                    HookChannel::Loss(i) => HookDescription::Loss(*i),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::general::{LossChannelKind, LossSite, QubitLoss};

    #[test]
    fn description_scaled_loss_sources_and_channels() {
        let sites = [
            LossSite::BeforeSingleQubit,
            LossSite::AfterSingleQubit,
            LossSite::BeforeTwoQubit,
            LossSite::AfterTwoQubit,
        ];
        let mut loss = QubitLoss::builder()
            .with_measurement(0.01)
            .with_prep(0.02)
            .with_idle_rate(0.03)
            .with_reload_on_prep(true)
            .with_lost_partner_loss_probability(0.4)
            .with_lost_partner_pauli_model(&BTreeMap::from([("Z".into(), 1.0)]));
        for (i, site) in sites.into_iter().enumerate() {
            let label = if i < 2 { "L" } else { "LZ" };
            loss = loss.add_events_channel(site, 0.04, &BTreeMap::from([(label.into(), 1.0)]));
        }
        loss = loss.add_recovery_channel(LossSite::AfterSingleQubit, 0.17, 0.23);
        let model = GeneralNoiseModel::builder()
            .with_loss(loss.build().unwrap())
            .with_scale(2.0)
            .with_meas_scale(3.0)
            .with_prep_scale(4.0)
            .with_idle_scale(5.0)
            .with_p1_scale(6.0)
            .with_p2_scale(7.0)
            .build();
        let d = model.description().loss.unwrap();
        assert!((d.measurement - 0.06).abs() < 1e-14);
        assert!((d.prep - 0.16).abs() < 1e-14);
        assert!((d.idle_rate - 0.30).abs() < 1e-14);
        assert!(d.reload_on_prep);
        assert!((d.lost_partner_probability - 0.4).abs() < 1e-14);
        assert_eq!(
            d.lost_partner_pauli.unwrap(),
            BTreeMap::from([("Z".into(), 1.0)])
        );
        for (i, channel) in d.channels[..4].iter().enumerate() {
            assert_eq!(channel.site, sites[i]);
            let LossChannelKind::Events {
                probability,
                events,
            } = &channel.kind
            else {
                panic!("expected events");
            };
            assert!((*probability - 0.04 * 2.0 * if i < 2 { 6.0 } else { 7.0 }).abs() < 1e-14);
            assert_eq!(
                events,
                &BTreeMap::from([(if i < 2 { "L" } else { "LZ" }.into(), 1.0)])
            );
        }
        assert_eq!(
            d.channels[4].kind,
            LossChannelKind::Recovery {
                false_negative: 0.17,
                false_positive: 0.23
            }
        );
        assert!(!d.default_recovery_added);
        let d = QubitLoss::builder().build().unwrap().description();
        assert!(d.default_recovery_added);
        assert_eq!(d.channels.len(), 1);
        let d = QubitLoss::builder()
            .with_default_recovery(false)
            .build()
            .unwrap()
            .description();
        assert!(!d.default_recovery_added);
        assert_eq!(d.channels.len(), 0);
        assert!(GeneralNoiseModel::default().description().loss.is_none());
    }
}
