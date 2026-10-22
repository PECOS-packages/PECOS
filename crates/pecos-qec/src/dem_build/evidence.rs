// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::bindings::{self, DemBindingError, ResultTagMeasurements, ResultTrace};
use super::{
    DemBuildError, DemProgram, check_generator_certificate_digest, generator_certificate_digest,
};
use crate::{DemInferenceReport, DemInferenceSettings};

/// Independent static and sampled schedule evidence. Zero shots means no comparison ran.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DemScheduleEvidence {
    pub static_source: Option<String>,
    pub validation_shots: usize,
}

/// Adapter-reported evidence, never inferred from a successful parity fit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DemExecutionEvidence {
    pub schedule: DemScheduleEvidence,
    pub raw_binding: Option<String>,
    pub provenance_shots: Option<usize>,
}

/// A generator-owned layout bound to the exact HUGR envelope used for execution.
/// The adapter asserts that a trusted built-in generator stamped this certificate
/// on the definition compiled to these exact executed bytes. The digest binds the
/// layout to the bytes; it does not establish the generator's identity.
#[derive(Clone, Debug)]
pub struct GeneratorLayout {
    pub hugr_bytes: Vec<u8>,
    pub layout: Vec<(String, usize)>,
    pub expected_digest: String,
}

impl GeneratorLayout {
    /// Verify the byte/layout binding before using the adapter's trust assertion.
    /// # Errors
    /// Rejects changed program bytes, layout, or digest.
    pub fn verify(&self) -> Result<(), DemBindingError> {
        check_generator_certificate_digest(
            &generator_certificate_digest(&self.hugr_bytes, &self.layout),
            &self.expected_digest,
        )
    }
}

pub(super) const JSON_CERTIFICATION_ERROR: &str = "Cannot bind result_tags: the program's lowered form could not be certified; use records, rec[-k], or meas_ids instead.";

pub(crate) fn require_static_schedule(
    program: &dyn DemProgram,
    required: bool,
    source: Option<&str>,
) -> Result<(), DemBuildError> {
    if required && source.is_none_or(str::is_empty) {
        return Err(program.missing_static_schedule_error());
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct TraceCertification<'a> {
    pub analysis: Option<&'a Result<ResultTagMeasurements, DemBindingError>>,
    pub generator: Option<&'a GeneratorLayout>,
}

pub(super) fn certify_traces(
    runtime_traces: &[ResultTrace],
    source: &[usize],
    runtime: &[usize],
    certification: Option<TraceCertification<'_>>,
    required_tags: &[String],
    inference: Option<(&DemInferenceReport, &DemInferenceSettings)>,
) -> Result<Vec<ResultTrace>, DemBuildError> {
    let binding_error = |error: DemBindingError| DemBuildError::Configuration(error.to_string());
    let generator = certification.as_ref().and_then(|c| c.generator);
    let analysis = certification.as_ref().and_then(|c| c.analysis);
    let certified = if certification.is_none() {
        Ok(Vec::new())
    } else if let Some(generator) = generator {
        generator.verify().map_err(binding_error)?;
        bindings::generator_certified_traces(
            &generator.layout,
            source,
            runtime,
            runtime_traces,
            required_tags,
        )
    } else if let Some(analysis) = analysis {
        bindings::compiler_certified_traces(source, runtime, runtime_traces, required_tags, || {
            analysis.clone()
        })
    } else {
        // Custom adapters own their reported bindings and may omit occurrence numbers.
        runtime_traces
            .iter()
            .zip(bindings::numbered_occurrences(runtime_traces))
            .map(|(call, occurrence)| {
                Ok(ResultTrace {
                    occurrence: Some(occurrence?),
                    ..call.clone()
                })
            })
            .collect()
    }
    .map_err(binding_error)?;
    let Some((report, settings)) = inference else {
        return Ok(certified);
    };
    let mut offset = 0usize;
    let mut bound = Vec::new();
    for call in runtime_traces
        .iter()
        .filter(|call| call.name == settings.raw_tag)
    {
        let end = offset.checked_add(call.width).ok_or_else(|| {
            DemBuildError::Configuration("raw result call widths overflow".into())
        })?;
        let ids = report.raw_measurement_ids.get(offset..end).ok_or_else(|| {
            DemBuildError::Configuration(
                "raw result call widths differ from inference binding".into(),
            )
        })?;
        bound.push(ResultTrace {
            measurement_ids: Some(ids.to_vec()),
            ..call.clone()
        });
        offset = end;
    }
    if !bound.is_empty() && offset != report.raw_measurement_ids.len() {
        return Err(DemBuildError::Configuration(
            "raw result call widths differ from inference binding".into(),
        ));
    }
    bindings::check_certified_raw_binding(
        &certified,
        &bound,
        &settings.raw_tag,
        generator.is_some(),
    )
    .map_err(binding_error)?;
    bound.extend(
        certified
            .into_iter()
            .filter(|call| call.name != settings.raw_tag),
    );
    Ok(bound)
}

#[cfg(test)]
mod tests;
