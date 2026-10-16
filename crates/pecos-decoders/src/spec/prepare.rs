//! Model preparation shared by direct leaf construction and batch execution.

use super::build::family_name;
use super::{BeliefMatchingMode, DecodeModel, DecoderSpec, PecosUfConfig, PecosUfPreset};
use crate::batch::PreparedModel;
use pecos_decoder_core::DecoderError;
use pecos_decoder_core::graphlike_projection::{
    GraphlikeProjectionError, IrreduciblePolicy, project_graphlike,
};
use pecos_decoder_core::window::StructuredDem;
use std::fmt::Write;

impl DecoderSpec {
    pub(crate) fn irreducible_policy(&self) -> Option<IrreduciblePolicy> {
        match self {
            Self::PyMatching(c) => Some(c.irreducible),
            Self::FusionBlossom(c) => Some(c.irreducible),
            Self::PerturbedFusionBlossomCorrelated(c) => Some(c.irreducible),
            Self::KMwpm(c) => Some(c.irreducible),
            Self::AStar(c) => Some(c.irreducible),
            Self::PecosUf(c) => Some(c.irreducible),
            Self::BeliefMatching(c) if c.mode != BeliefMatchingMode::Hybrid => Some(c.irreducible),
            Self::Perturbed(c) => c.inner.irreducible_policy(),
            Self::Windowed(c) => c.inner.irreducible_policy(),
            Self::Ensemble(c) => {
                let policy = c.members.first()?.irreducible_policy()?;
                c.members
                    .iter()
                    .all(|member| member.irreducible_policy() == Some(policy))
                    .then_some(policy)
            }
            _ => None,
        }
    }
}

/// Prepare before graph construction, probability perturbation, or windowing.
pub(crate) fn prepare_model(
    spec: &DecoderSpec,
    model: &DecodeModel,
) -> Result<PreparedModel, DecoderError> {
    prepare_model_in_context(spec, model, &ProjectionContext::default())
}

pub(super) fn prepare_model_in_context(
    spec: &DecoderSpec,
    model: &DecodeModel,
    context: &ProjectionContext,
) -> Result<PreparedModel, DecoderError> {
    validate_preparation_specs(spec)?;
    let policy = spec.irreducible_policy();
    let unchanged = || PreparedModel {
        model: model.clone(),
        projection: None,
    };
    if matches!(model, DecodeModel::StructuredDem(_)) {
        reject_structured_hybrid(spec)?;
    }
    let Some(policy) = policy else {
        reject_unreported_drop(spec, context)?;
        return Ok(unchanged());
    };
    let parsed;
    let structured = match model {
        DecodeModel::SingleDem(text) => {
            parsed = StructuredDem::from_dem_str(text)?;
            &parsed
        }
        DecodeModel::StructuredDem(model) => model,
        DecodeModel::HybridDem { .. } => return Ok(unchanged()),
    };
    // Inspect written components, including zero probabilities and support that
    // cancels across components. Matching-graph parsing can discard either.
    let decomposed = structured
        .errors
        .iter()
        .any(|error| error.components.len() > 1);
    let raw_hyperedges = structured
        .errors
        .iter()
        .any(|error| error.components.len() == 1 && error.components[0].detectors.len() > 2);
    if decomposed {
        if raw_hyperedges {
            return Err(DecoderError::InvalidConfiguration(format!(
                "{}: model mixes decomposed and raw hyperedges; supply either a raw model or a fully graphlike (^ decomposed) model",
                context.decoder_names(spec)
            )));
        }
        ensure_graphlike(
            structured,
            &format!("{} supplied decomposition", context.decoder_names(spec)),
        )?;
        return Ok(unchanged());
    }
    if !raw_hyperedges {
        return Ok(unchanged());
    }
    #[cfg(test)]
    PROJECTION_INPUTS.lock().unwrap().push(structured.clone());
    let projected = project_graphlike(structured, policy).map_err(|error| {
        let mut message = format!("{}: {error}", context.decoder_names(spec));
        if matches!(error, GraphlikeProjectionError::Irreducible { .. }) {
            let _ = write!(message, ". {} Dropping removes those faults from the decoder's model; alternatively use a hyperedge-capable decoder such as bp_osd or tesseract.", context.drop_remedy(spec));
        }
        DecoderError::InvalidConfiguration(message)
    })?;
    Ok(PreparedModel {
        model: DecodeModel::StructuredDem(projected.model),
        projection: Some(projected.report),
    })
}

// Compositions without a shared projection cannot report individual losses.
fn reject_unreported_drop(
    spec: &DecoderSpec,
    context: &ProjectionContext,
) -> Result<(), DecoderError> {
    match spec {
        DecoderSpec::Ensemble(c) => {
            for (index, member) in c.members.iter().enumerate() {
                reject_unreported_drop(member, &context.member(index))?;
            }
        }
        DecoderSpec::Perturbed(c) => reject_unreported_drop(&c.inner, &context.inner("perturbed"))?,
        DecoderSpec::Windowed(c) => reject_unreported_drop(&c.inner, &context.inner("windowed"))?,
        DecoderSpec::BeamSearch(c) => reject_unreported_drop(&c.phase2, &context.phase2())?,
        _ if spec.irreducible_policy() == Some(IrreduciblePolicy::Drop) => {
            return Err(DecoderError::InvalidConfiguration(format!(
                "{}: drop inside such a composition needs every graph member to share the policy; a shared projection policy is required to report dropped faults",
                context.decoder_names(spec)
            )));
        }
        _ => {}
    }
    Ok(())
}

fn reject_structured_hybrid(spec: &DecoderSpec) -> Result<(), DecoderError> {
    match spec {
        DecoderSpec::BeliefMatching(c) if c.mode == BeliefMatchingMode::Hybrid => {
            Err(DecoderError::InvalidConfiguration(
                "belief_matching_hybrid requires its full and decomposed DEM text; DetectorErrorModel objects are not supported".into(),
            ))
        }
        DecoderSpec::Perturbed(c) => reject_structured_hybrid(&c.inner),
        DecoderSpec::Windowed(c) => reject_structured_hybrid(&c.inner),
        DecoderSpec::Ensemble(c) => {
            for member in &c.members { reject_structured_hybrid(member)?; }
            Ok(())
        }
        DecoderSpec::BeamSearch(c) => reject_structured_hybrid(&c.phase2),
        _ => Ok(()),
    }
}

/// Contexts without a loss-policy option must receive graphlike components.
pub(super) fn ensure_graphlike(model: &StructuredDem, context: &str) -> Result<(), DecoderError> {
    let count = model
        .errors
        .iter()
        .flat_map(|error| &error.components)
        .filter(|component| component.detectors.len() > 2)
        .count();
    if count > 0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "{context} needs a graphlike model, but {count} component(s) touch more than two detectors. Pass a graphlike (^ decomposed) model or use a hyperedge-capable decoder such as bp_osd or tesseract."
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) static PROJECTION_INPUTS: std::sync::Mutex<Vec<StructuredDem>> =
    std::sync::Mutex::new(Vec::new());

/// Check nested window support before a parent can project on its behalf.
fn validate_preparation_specs(spec: &DecoderSpec) -> Result<(), DecoderError> {
    match spec {
        DecoderSpec::Windowed(c) => {
            match &*c.inner {
                DecoderSpec::PecosUf(PecosUfConfig { preset: PecosUfPreset::Fast, .. })
                | DecoderSpec::PyMatching(_) => {}
                inner => return Err(DecoderError::InvalidConfiguration(format!(
                    "windowed does not support inner {inner:?}; use pecos_uf or pymatching"
                ))),
            }
            validate_preparation_specs(&c.inner)
        }
        DecoderSpec::Perturbed(c) => validate_preparation_specs(&c.inner),
        DecoderSpec::Ensemble(c) => {
            for member in &c.members { validate_preparation_specs(member)?; }
            Ok(())
        }
        DecoderSpec::BeliefMatching(c)
            if c.mode == BeliefMatchingMode::Hybrid && c.irreducible != IrreduciblePolicy::Refuse =>
        {
            Err(DecoderError::InvalidConfiguration(
                "belief_matching_hybrid does not support an irreducible policy; supply its full and graphlike decomposed models".into(),
            ))
        }
        _ => Ok(()),
    }
}

#[derive(Clone)]
enum ProjectionStep {
    Inner(&'static str),
    Member(usize),
    Phase2,
}

/// Construction context survives composites that cannot share a projection.
#[derive(Clone, Default)]
pub(super) struct ProjectionContext(Vec<ProjectionStep>);

impl ProjectionContext {
    pub(super) fn inner(&self, family: &'static str) -> Self {
        let mut child = self.clone();
        child.0.push(ProjectionStep::Inner(family));
        child
    }

    pub(super) fn member(&self, index: usize) -> Self {
        let mut child = self.clone();
        child.0.push(ProjectionStep::Member(index));
        child
    }

    pub(super) fn phase2(&self) -> Self {
        let mut child = self.clone();
        child.0.push(ProjectionStep::Phase2);
        child
    }

    fn targets(&self, spec: &DecoderSpec) -> Vec<(String, String)> {
        match spec {
            DecoderSpec::Perturbed(c) => self.inner("perturbed").targets(&c.inner),
            DecoderSpec::Windowed(c) => self.inner("windowed").targets(&c.inner),
            DecoderSpec::Ensemble(c) => c
                .members
                .iter()
                .enumerate()
                .flat_map(|(index, member)| self.member(index).targets(member))
                .collect(),
            _ => {
                let name = family_name(spec);
                let location = self
                    .0
                    .iter()
                    .map(|step| match step {
                        ProjectionStep::Inner(family) => format!("{family} inner decoder"),
                        ProjectionStep::Member(index) => format!("ensemble member {index}"),
                        ProjectionStep::Phase2 => "beamsearch phase-2 decoder".into(),
                    })
                    .collect::<Vec<_>>()
                    .join(" -> ");
                let label = if location.is_empty() {
                    name.into()
                } else {
                    format!("{name} ({location})")
                };
                let remedy = if self
                    .0
                    .iter()
                    .any(|step| matches!(step, ProjectionStep::Phase2))
                {
                    format!(
                        "Choose a hyperedge-capable decoder for {label} through a typed spec: Rust BeamSearchConfig.phase2, or Python pecos_rslib.decoders.beamsearch(phase2=...); the beamsearch string grammar has no phase2 key. Drop is unavailable because beamsearch has no shared projection report."
                    )
                } else if self
                    .0
                    .iter()
                    .any(|step| matches!(step, ProjectionStep::Member(_)))
                {
                    format!(
                        "For {label}, drop needs every graph member to share one policy and the composition to resolve to a shared projection policy; mixing graph and hypergraph members cannot do this. Use an all-graph composition with irreducible=drop on every graph leaf, for example ensemble:astar:irreducible=drop,pecos_uf:irreducible=drop. For nested members use typed EnsembleConfig.members or Python pecos_rslib.decoders.ensemble(...), for example ensemble(perturbed(inner=astar(irreducible=\"drop\")), astar(irreducible=\"drop\")). Alternatively, use a hyperedge-capable decoder for {label}."
                    )
                } else {
                    let mut form = format!("{name}:irreducible=drop");
                    for step in self.0.iter().rev() {
                        match step {
                            ProjectionStep::Inner(family) => {
                                form = format!("{family}:...,inner={form}");
                            }
                            ProjectionStep::Member(_) => form = format!("ensemble:{form}"),
                            ProjectionStep::Phase2 => unreachable!(),
                        }
                    }
                    format!("Set irreducible=drop on {label}, for example {form}.")
                };
                vec![(label, remedy)]
            }
        }
    }

    fn decoder_names(&self, spec: &DecoderSpec) -> String {
        self.targets(spec)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn drop_remedy(&self, spec: &DecoderSpec) -> String {
        self.targets(spec)
            .into_iter()
            .map(|(_, remedy)| remedy)
            .collect::<Vec<_>>()
            .join(" ")
    }
}
