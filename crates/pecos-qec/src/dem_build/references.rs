// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::DemBuildError;
use crate::fault_tolerance::circuit_definitions::CircuitDefinitions;
use crate::fault_tolerance::dem_builder::{
    DemBuilderError, measurement_parity_expression, parse_detectors_json, parse_observables_json,
};
use crate::fault_tolerance::fault_sampler::{
    MeasurementHistoryError, symbolic_measurement_history,
};
use pecos_quantum::{Attribute, TickCircuit};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Exact noiseless values keyed by definition ID; tracked Paulis are not readouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoiselessReferences {
    pub detectors: BTreeMap<usize, u8>,
    pub observables: BTreeMap<usize, u8>,
}

/// Compute every definition's reference with one symbolic evaluation.
/// Resolved supports index tick emission order, as does the symbolic history.
/// Replay uses `is_supported_noop_or_metadata_gate` for transparent gates.
///
/// # Errors
/// Rejects unsupported circuits, random parities, and disagreeing declarations.
pub fn noiseless_references(
    circuit: &TickCircuit,
    definitions: &CircuitDefinitions,
) -> Result<NoiselessReferences, DemBuildError> {
    let history = symbolic_measurement_history(circuit).map_err(|error| match error {
        MeasurementHistoryError::UnsupportedGate(error) => {
            DemBuildError::Propagation(DemBuilderError::UnsupportedGate(error))
        }
    })?;
    if history.len() != definitions.num_measurements {
        return Err(DemBuildError::Configuration(format!(
            "symbolic measurement history has {} records; definitions expect {}",
            history.len(),
            definitions.num_measurements
        )));
    }
    let positions: Vec<_> = (0..definitions.num_measurements).collect();
    let reference = |role, id, support: &[usize], declared| {
        let expression = measurement_parity_expression(&history, &positions, support, "noiseless")?;
        if !expression.dependencies.is_empty() {
            return Err(DemBuildError::Configuration(format!(
                "{role} {id} has a random noiseless parity; detectors and observables must be deterministic in the noiseless circuit"
            )));
        }
        let computed = u8::from(expression.flip);
        if let Some(declared) = declared
            && declared != computed
        {
            return Err(DemBuildError::Configuration(format!(
                "{role} {id} declares reference {declared}, but its noiseless parity is {computed}"
            )));
        }
        Ok((id as usize, computed))
    };
    Ok(NoiselessReferences {
        detectors: definitions
            .detectors
            .iter()
            .map(|d| reference("detector", d.id, &d.measurements, d.reference))
            .collect::<Result<_, _>>()?,
        observables: definitions
            .observables
            .iter()
            .map(|d| reference("observable", d.id, &d.measurements, d.reference))
            .collect::<Result<_, _>>()?,
    })
}

pub(super) fn write_references(
    circuit: &mut TickCircuit,
    definitions: &mut CircuitDefinitions,
    references: &NoiselessReferences,
    measurement_ids: &[usize],
) -> Result<(Value, Value), DemBuildError> {
    let mut detectors = metadata_entries(circuit, "detectors", |text| {
        Ok(parse_detectors_json(text)?
            .into_iter()
            .map(|d| d.id)
            .collect())
    })?;
    for definition in &mut definitions.detectors {
        let reference = *references
            .detectors
            .get(&(definition.id as usize))
            .ok_or_else(|| {
                DemBuildError::Configuration(format!(
                    "missing noiseless reference for detector {}",
                    definition.id
                ))
            })?;
        definition.reference = Some(reference);
        let entry = definition_entry(
            &mut detectors,
            definition.id,
            &definition.measurements,
            definition.label.as_deref(),
            reference,
            measurement_ids,
        );
        if let Some(coords) = &definition.coords {
            entry["coords"] = json!(coords);
        }
        if let Some(sector) = &definition.fault_sector {
            entry["fault_sector"] = json!(sector);
        }
    }
    let mut observables = metadata_entries(circuit, "observables", |text| {
        Ok(parse_observables_json(text)?
            .into_iter()
            .map(|d| d.id)
            .collect())
    })?;
    for definition in &mut definitions.observables {
        let reference = *references
            .observables
            .get(&(definition.id as usize))
            .ok_or_else(|| {
                DemBuildError::Configuration(format!(
                    "missing noiseless reference for observable {}",
                    definition.id
                ))
            })?;
        definition.reference = Some(reference);
        definition_entry(
            &mut observables,
            definition.id,
            &definition.measurements,
            definition.label.as_deref(),
            reference,
            measurement_ids,
        );
    }
    let detectors = Value::Array(detectors.into_values().collect());
    let observables = Value::Array(observables.into_values().collect());
    circuit.set_meta("detectors", Attribute::String(detectors.to_string()));
    circuit.set_meta("observables", Attribute::String(observables.to_string()));
    Ok((detectors, observables))
}

fn metadata_entries(
    circuit: &TickCircuit,
    key: &str,
    parse_ids: impl FnOnce(&str) -> Result<Vec<u32>, DemBuilderError>,
) -> Result<BTreeMap<u32, Value>, DemBuildError> {
    let mut entries = BTreeMap::new();
    if let Some(Attribute::String(text)) = circuit.get_meta(key) {
        // The shared reader already validated aliases, IDs and entry shapes.
        let ids = parse_ids(text)?;
        if !ids.is_empty() {
            let values: Vec<Value> = serde_json::from_str(text)
                .map_err(|error| DemBuildError::Configuration(error.to_string()))?;
            entries.extend(ids.into_iter().zip(values));
        }
    }
    Ok(entries)
}

fn definition_entry<'a>(
    entries: &'a mut BTreeMap<u32, Value>,
    id: u32,
    support: &[usize],
    label: Option<&str>,
    reference: u8,
    measurement_ids: &[usize],
) -> &'a mut Value {
    let entry = entries.entry(id).or_insert_with(|| json!({
        "id": id,
        "meas_ids": support.iter().map(|&position| measurement_ids[position]).collect::<Vec<_>>()
    }));
    entry["reference"] = json!(reference);
    if let Some(label) = label {
        entry["label"] = json!(label);
    }
    entry
}

#[cfg(test)]
mod tests;
