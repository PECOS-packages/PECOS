// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::bindings::{MeasurementLedgerEntry, named_binding_classification};
use super::{DemAnnotationOrigin, DemBuild, DemBuildError, DemDefinitionOrigin};
use crate::fault_tolerance::dem_builder::NoiseChannelResidual;
use pecos_quantum::Attribute;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Empirical inference evidence; absent runtime evidence is represented by None.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemInferenceAudit {
    pub method: &'static str,
    pub raw_binding: String,
    pub probe_shots: usize,
    pub provenance_shots: Option<usize>,
    pub validation_rows: usize,
    pub validation_kind: &'static str,
    pub schedule_validation: Option<String>,
    pub schedule_validation_shots: Option<usize>,
    pub raw_measurement_ids: Vec<usize>,
}

impl DemInferenceAudit {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "method": self.method, "raw_binding": self.raw_binding,
            "probe_shots": self.probe_shots, "provenance_shots": self.provenance_shots,
            "validation_rows": self.validation_rows, "validation_kind": self.validation_kind,
            "schedule_validation": self.schedule_validation,
            "schedule_validation_shots": self.schedule_validation_shots,
            "raw_measurement_ids": self.raw_measurement_ids,
        })
    }
}

/// Build evidence using the Python audit keys. The later Python/two-phase-builder
/// packet supplies compiler/generator binding classification and full result-call
/// provenance. Today named bindings describe the inferred raw tag, or none;
/// caller-supplied probes use `caller_bound_measurement_ids`. Runtime evidence
/// absent from the current input is None, never a claim of certification.
#[derive(Clone, Debug, PartialEq)]
pub struct DemBuildAudit {
    pub schema_fingerprint: String,
    pub soundness_scope: &'static str,
    pub inference: Option<DemInferenceAudit>,
    pub definition_origins: Vec<DemDefinitionOrigin>,
    pub named_result_binding: String,
    pub measurement_count: usize,
    pub runtime_measurement_order: Vec<usize>,
    pub runtime_order_is_canonical: bool,
    pub runtime_order_mismatch_count: usize,
    pub measurement_ledger: Vec<MeasurementLedgerEntry>,
    pub idle_noise_residuals: Vec<NoiseChannelResidual>,
}

impl DemBuildAudit {
    #[must_use]
    pub fn to_json(&self) -> Value {
        let origins: Vec<_> = self
            .definition_origins
            .iter()
            .map(|origin| {
                let mut value = json!({"role": origin.role, "id": origin.id});
                match &origin.source {
                    DemAnnotationOrigin::Explicit => {
                        value["source"] = json!("explicit");
                    }
                    DemAnnotationOrigin::Inferred(output) => {
                        value["source"] = json!("inferred");
                        value["tag"] = json!(output.tag);
                        value["element"] = json!(output.element);
                    }
                    DemAnnotationOrigin::InferredDefaults { tags, column } => {
                        value["source"] = json!("inferred");
                        value["tags"] = json!(tags);
                        value["column"] = json!(column);
                    }
                }
                value
            })
            .collect();
        json!({
            "schema_fingerprint": self.schema_fingerprint,
            "soundness_scope": self.soundness_scope,
            "inference": self.inference.as_ref().map(DemInferenceAudit::to_json),
            "definition_origins": origins,
            "named_result_binding": self.named_result_binding,
            "measurement_count": self.measurement_count,
            "runtime_measurement_order": self.runtime_measurement_order,
            "runtime_order_is_canonical": self.runtime_order_is_canonical,
            "runtime_order_mismatch_count": self.runtime_order_mismatch_count,
            "measurement_ledger": self.measurement_ledger.iter().map(MeasurementLedgerEntry::to_json).collect::<Vec<_>>(),
            "idle_noise_residuals": self.idle_noise_residuals.iter().map(|residual| json!({
                "location_index": residual.location_index,
                "channel_kind": residual.channel_kind.as_str(),
                "detectors": residual.effect.detectors.to_vec(),
                "dem_outputs": residual.effect.dem_outputs.to_vec(),
                "tracked_paulis": residual.effect.tracked_paulis.to_vec(),
                "magnitude": residual.magnitude,
                "channel_weight": residual.channel_weight,
                "relative_magnitude": residual.relative_magnitude(),
            })).collect::<Vec<_>>(),
        })
    }
}

impl DemBuild {
    #[must_use]
    pub fn audit(&self) -> DemBuildAudit {
        let meta = |key| match self.schema.circuit.get_meta(key) {
            Some(Attribute::String(value)) => Some(value.clone()),
            _ => None,
        };
        let inference = self.inference.as_ref().map(|report| {
            let schedule_shots =
                meta("dem_schedule_validation_shots").and_then(|value| value.parse::<usize>().ok());
            DemInferenceAudit {
                method: "coin_toss_affine_inference_v2",
                raw_binding: meta("dem_raw_measurement_binding")
                    .unwrap_or_else(|| "caller_bound_measurement_ids".into()),
                probe_shots: report.probe_shots,
                provenance_shots: schedule_shots
                    .and_then(|shots| shots.checked_sub(report.probe_shots)),
                validation_rows: report.validation_rows,
                validation_kind: "additional_consistency_equations",
                schedule_validation: meta("dem_schedule_validation"),
                schedule_validation_shots: schedule_shots,
                raw_measurement_ids: report.raw_measurement_ids.clone(),
            }
        });
        let mismatch_count = self
            .schema
            .measurement_ids
            .iter()
            .enumerate()
            .filter(|(position, id)| position != *id)
            .count();
        DemBuildAudit {
            schema_fingerprint: self.schema.schema_fingerprint.clone(),
            soundness_scope: if inference.is_some() {
                "empirical_static_schedule"
            } else {
                "static_schedule_only"
            },
            named_result_binding: named_binding_classification(
                inference.as_ref().map(|audit| audit.raw_binding.as_str()),
                false,
                &self.schema.named_result_measurements,
            )
            .into(),
            inference,
            definition_origins: self.definition_origins.clone(),
            measurement_count: self.schema.measurement_ids.len(),
            runtime_measurement_order: self.schema.measurement_ids.clone(),
            runtime_order_is_canonical: mismatch_count == 0,
            runtime_order_mismatch_count: mismatch_count,
            measurement_ledger: self.schema.measurement_ledger.clone(),
            idle_noise_residuals: self.dem.idle_noise_residuals().to_vec(),
        }
    }
}

/// Numeric summary of retained residuals for one channel kind.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseResidualSummary {
    pub channel_kind: &'static str,
    pub count: usize,
    pub max_relative_magnitude: f64,
    pub max_tv_magnitude: f64,
}

/// Group residuals strictly above the relative threshold, sorted by kind label.
/// No warnings are emitted; presentation belongs to Python.
/// # Errors
/// Rejects non-finite thresholds and thresholds outside [0, 1].
pub fn summarize_noise_residuals(
    residuals: &[NoiseChannelResidual],
    relative_threshold: f64,
) -> Result<Vec<NoiseResidualSummary>, DemBuildError> {
    if !relative_threshold.is_finite() || !(0.0..=1.0).contains(&relative_threshold) {
        return Err(DemBuildError::Configuration(
            "residual_warning_threshold must be finite and between 0 and 1".into(),
        ));
    }
    let mut groups = BTreeMap::new();
    for residual in residuals {
        let relative = residual.relative_magnitude();
        if relative > relative_threshold {
            let kind = residual.channel_kind.as_str();
            let group = groups.entry(kind).or_insert(NoiseResidualSummary {
                channel_kind: kind,
                count: 0,
                max_relative_magnitude: 0.0,
                max_tv_magnitude: 0.0,
            });
            group.count += 1;
            group.max_relative_magnitude = group.max_relative_magnitude.max(relative);
            group.max_tv_magnitude = group.max_tv_magnitude.max(residual.magnitude);
        }
    }
    Ok(groups.into_values().collect())
}
