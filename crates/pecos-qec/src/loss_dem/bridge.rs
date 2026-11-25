use super::{LossDemError, circuit_error, invalid};
use crate::fault_tolerance::dem_builder::PauliWeights;
use crate::{IdleNoiseFamily, NoiseConfig};
use pecos_core::gate_type::GateType;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::noise::general::GeneralNoiseDescription;
use pecos_quantum::TickCircuit;
use std::collections::BTreeMap;

fn unsupported(name: &str, value: impl std::fmt::Debug) -> LossDemError {
    LossDemError::UnsupportedSetting {
        name: name.into(),
        value: format!("{value:?}"),
        fix: "restore this setting to its default".into(),
    }
}

// Configuration equality is exact, including inert parameters; no tolerance is appropriate.
fn reject_nondefault<T: PartialEq + std::fmt::Debug>(
    name: &str,
    value: &T,
    default: &T,
) -> Result<(), LossDemError> {
    if value != default {
        return Err(unsupported(name, value));
    }
    Ok(())
}

fn weights(
    name: &str,
    arity: usize,
    model: &BTreeMap<String, f64>,
) -> Result<PauliWeights, LossDemError> {
    let total: f64 = model.values().sum();
    if !total.is_finite()
        || (total - 1.0).abs() >= 1e-6
        || model.values().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err(unsupported(name, model));
    }
    let entries = model
        .iter()
        .map(|(label, &p)| {
            if label.len() != arity || !label.chars().all(|c| "IXYZ".contains(c)) {
                return Err(unsupported(name, model));
            }
            label
                .parse()
                .map(|pauli| (pauli, p))
                .map_err(|_| unsupported(name, model))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PauliWeights::new(entries))
}

/// Convert built settings to the ordinary DEM Pauli configuration for this circuit.
///
/// Inert nondefault unsupported settings are rejected deliberately. Detect readout
/// rows are ignored because the decoder trusts flags; Z-herald rows are ignored
/// because those outcomes never enter detectors. Builder scaling is already applied.
///
/// # Errors
/// Names unsupported settings, repeated unprepared records with measurement noise,
/// and incompatible readout lifecycles. Categorical fits retain ordinary DEM residuals.
#[allow(clippy::float_cmp)] // Readout symmetry is a configuration identity, not a numerical tolerance.
pub fn pauli_noise_config(
    circuit: &TickCircuit,
    description: &GeneralNoiseDescription,
) -> Result<NoiseConfig, LossDemError> {
    super::runner::validate_circuit(circuit)?;
    let default = GeneralNoiseModel::default().description();
    macro_rules! reject_changed {
        ($($field:ident),* $(,)?) => {
            $(
                reject_nondefault(
                    stringify!($field),
                    &description.$field,
                    &default.$field,
                )?;
            )*
        };
    }
    // Every description setting outside the allow-list is checked, even if its rate is zero.
    reject_changed!(
        leakage_scale,
        p_idle_sin_squared_rate,
        p_idle_sin_squared_model,
        p_idle_coherent_rate,
        p_idle_coherent_model,
        p_prep_leak_ratio,
        p_prep_crosstalk,
        p1_emission_ratio,
        p1_emission_model,
        p1_seepage_prob,
        p1_transition_channels_before_gate,
        p1_transition_channels_after_gate,
        p1_pauli_leakage_channels_before_gate,
        p1_pauli_leakage_channels_after_gate,
        p2_angle_a,
        p2_angle_b,
        p2_angle_c,
        p2_angle_d,
        p2_angle_power,
        p2_emission_ratio,
        p2_emission_model,
        p2_seepage_prob,
        p2_transition_steps_before_gate,
        p2_transition_steps_after_gate,
        p2_pauli_leakage_steps_before_gate,
        p2_pauli_leakage_steps_after_gate,
        p_meas_crosstalk_global,
        p_meas_crosstalk_local,
        p_meas_crosstalk_model,
    );
    // Hook-order entries referring to loss are handled by the location-channel path.
    for (name, order) in [
        (
            "p1_hook_order_before_gate",
            &description.p1_hook_order_before_gate,
        ),
        (
            "p1_hook_order_after_gate",
            &description.p1_hook_order_after_gate,
        ),
        (
            "p2_hook_order_before_gate",
            &description.p2_hook_order_before_gate,
        ),
        (
            "p2_hook_order_after_gate",
            &description.p2_hook_order_after_gate,
        ),
    ] {
        if order.iter().any(|entry| {
            !matches!(
                entry,
                pecos_engines::noise::general::HookDescription::Loss(_)
            )
        }) {
            return Err(unsupported(name, order));
        }
    }
    if description.p_meas_0 != description.p_meas_1 {
        return Err(unsupported(
            "p_meas_0/p_meas_1",
            (description.p_meas_0, description.p_meas_1),
        ));
    }
    for (&flavor, rows) in &description.readout {
        if flavor.consumes_measurement_record() {
            return Err(unsupported(&format!("readout.{flavor}"), rows));
        }
    }
    // Match GeneralNoiseModel::apply_idle_noise's rate cutoff before duration scaling.
    let idle_rate = if description.p_idle_linear_rate <= f64::EPSILON {
        0.0
    } else {
        description.p_idle_linear_rate
    };
    if description.idle_after_2q > 0.0 && idle_rate > 0.0 {
        return Err(unsupported(
            "idle_after_2q with p_idle_linear_rate",
            description.idle_after_2q,
        ));
    }
    let mut config = NoiseConfig::new(
        description.p1,
        description.p2,
        description.p_meas_0,
        description.p_prep,
    );
    config.p1_weights = (description.p1_pauli_model != default.p1_pauli_model)
        .then(|| weights("p1_pauli_model", 1, &description.p1_pauli_model))
        .transpose()?;
    config.p2_weights = (description.p2_pauli_model != default.p2_pauli_model)
        .then(|| weights("p2_pauli_model", 2, &description.p2_pauli_model))
        .transpose()?;
    weights("p_idle_linear_model", 1, &description.p_idle_linear_model)?;
    config.p_idle_linear = IdleNoiseFamily::new(idle_rate, description.p_idle_linear_model.clone());
    for &kind in &description.noiseless_gates {
        let rate = match kind {
            GateType::MZ | GateType::MPZ => description.p_meas_0,
            GateType::PZ => description.p_prep,
            GateType::Idle => idle_rate,
            _ => 0.0,
        };
        if rate != 0.0 {
            return Err(unsupported(
                &format!("noiseless_gates.{kind} with matching rate"),
                rate,
            ));
        }
        match super::channels::site_arity(kind) {
            Some(1) => {
                config.p1_gate_rates.insert(kind, 0.0);
            }
            Some(2) => {
                config.p2_gate_rates.insert(kind, 0.0);
            }
            _ => {}
        }
    }
    let mut last_record = BTreeMap::new();
    let mut record = 0;
    for gate in circuit.iter_gate_batches() {
        let g = gate.as_gate();
        let exempt = description.is_noiseless_operation(g);
        for &q in &g.qubits {
            if g.gate_type == GateType::PZ {
                last_record.remove(&q);
            }
            if g.gate_type.consumes_measurement_record() {
                if description.p_meas_0 > 0.0
                    && let Some(previous) = last_record.insert(q, record)
                {
                    return Err(invalid(format!(
                        "p_meas = {} is not supported by the loss-conditioned DEM for records {previous} and {record} on qubit {q} without preparation; insert a preparation or disable measurement noise",
                        description.p_meas_0
                    )));
                }
                record += 1;
            }
            if g.gate_type == GateType::MPZ {
                last_record.remove(&q);
            }
        }
        if super::channels::site_arity(g.gate_type).is_some() && exempt {
            if g.gate_type.is_two_qubit() {
                config.p2_gate_rates.insert(g.gate_type, 0.0);
            } else {
                config.p1_gate_rates.insert(g.gate_type, 0.0);
            }
        }
    }
    if description.p_meas_0 > 0.0 {
        if let Some(error) =
            crate::fault_tolerance::propagator::first_unsupported_tick_gate(circuit)
        {
            return Err(circuit_error("measurement-X propagation", error));
        }
        let (flat_gates, measurements) =
            crate::fault_tolerance::fault_sampler::flatten_tick_circuit(circuit);
        let mut source = 0;
        for (tick, gates) in circuit.iter_ticks() {
            for (batch, gate) in gates.iter_gate_batches().enumerate() {
                if !gate.gate_type.consumes_measurement_record() {
                    continue;
                }
                for operand in 0..gate.qubits.len() {
                    if gate.gate_type == GateType::MZ {
                        let flips = super::symbolic::located_pauli_flips_prepared(
                            &flat_gates,
                            &measurements,
                            pecos_core::Pauli::X,
                            super::Location {
                                tick,
                                batch,
                                operand,
                                side: super::Side::After,
                            },
                        )?;
                        if let Some(target) = flips.iter().next() {
                            return Err(invalid(format!(
                                "p_meas = {} is not supported by the loss-conditioned DEM: the pre-measurement X for record {source} propagates to record {target}; prepare the measured qubit before reusing it or disable measurement noise",
                                description.p_meas_0
                            )));
                        }
                    }
                    source += 1;
                }
            }
        }
    }
    Ok(config)
}
