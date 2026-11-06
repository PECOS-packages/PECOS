// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Shared gate-noise metadata resolution at circuit-to-source boundaries.

use pecos_core::gate_type::GateType;
use pecos_quantum::Attribute;

/// Resolve per-gate suppression: only boolean true removes noise sources.
/// Ideal operations, record indices and configuration validation are unchanged.
/// Non-boolean values report the scheduled gate and its original location.
pub(crate) fn resolve_noiseless(
    attribute: Option<&Attribute>,
    gate_type: GateType,
    location: impl std::fmt::Display,
) -> Result<bool, String> {
    match attribute {
        None | Some(Attribute::Bool(false)) => Ok(false),
        Some(Attribute::Bool(true)) => Ok(true),
        Some(_) => Err(format!(
            "attribute 'noiseless' on gate {gate_type:?} at {location} must be a boolean"
        )),
    }
}

/// Resolve stored `TickCircuit` batch metadata before conversion loses tick provenance.
pub(crate) fn noiseless_tick_batches(
    circuit: &pecos_quantum::TickCircuit,
) -> Result<std::collections::BTreeSet<(usize, usize)>, String> {
    let mut batches = std::collections::BTreeSet::new();
    for (tick, slice) in circuit.iter_ticks() {
        for batch in slice.iter_gate_batches() {
            if resolve_noiseless(
                batch.get_attr("noiseless"),
                batch.gate_type,
                super::propagator::UnsupportedGateLocation::Tick {
                    tick,
                    gate_in_tick: batch.batch_index(),
                },
            )? {
                batches.insert((tick, batch.batch_index()));
            }
        }
    }
    Ok(batches)
}

/// Validate stored batch attributes before converting a `TickCircuit` to a DAG.
pub(crate) fn validate_tick_noise_attributes(
    circuit: &pecos_quantum::TickCircuit,
) -> Result<(), String> {
    noiseless_tick_batches(circuit).map(|_| ())
}
