use super::config::{
    BeamSearchConfig, BeliefMatchingConfig, BeliefMatchingMode, BpLsdConfig, BpOsdConfig,
    EnsembleConfig, FusionBlossomConfig, KMwpmConfig, MinSumBpConfig, MwpfConfig, PecosUfPreset,
    PerturbedConfig, PerturbedFusionBlossomConfig, PyMatchingConfig, RelayBpConfig,
    TesseractConfig, TesseractTrellisConfig, WindowedConfig,
};
use super::{DecodeModel, DecoderSpec};
use pecos_decoder_core::window::StructuredDem;
use pecos_decoder_core::{DecoderError, ObservableDecoder};

#[cfg(feature = "ldpc")]
use super::config::BpSchedule as SpecBpSchedule;
#[cfg(feature = "fusion-blossom")]
use super::config::FusionBlossomSolverType;
#[cfg(feature = "mwpf")]
use super::config::MwpfSolverType as SpecMwpfSolverType;
#[cfg(feature = "relay-bp")]
use super::config::RelayStoppingCriterion;
#[cfg(feature = "tesseract")]
use super::config::TesseractPreset;
#[cfg(feature = "ldpc")]
use crate::BpSchedule as LdpcBpSchedule;
#[cfg(feature = "tesseract")]
use crate::TesseractConfig as TesseractEngineConfig;
#[cfg(feature = "uf")]
use crate::{
    BeamSearchConfig as BeamSearchEngineConfig, BeamWindowConfig,
    WindowedConfig as WindowedEngineConfig,
};
#[cfg(feature = "fusion-blossom")]
use crate::{
    FusionBlossomConfig as FusionBlossomEngineConfig, SolverType as FusionBlossomEngineSolverType,
};
#[cfg(feature = "mwpf")]
use crate::{MwpfConfig as MwpfEngineConfig, MwpfSolverType as MwpfEngineSolverType};

#[cfg(any(feature = "uf", test))]
use super::config::PecosUfConfig;
#[cfg(test)]
use super::prepare::prepare_model;
use super::prepare::{ProjectionContext, prepare_model_in_context};
#[cfg(test)]
use pecos_decoder_core::graphlike_projection::IrreduciblePolicy;

/// Preserve the existing matching-graph checks for Hybrid and beam phase 1.
fn ensure_graphlike_matching_model(context: &str, dem: &str) -> Result<(), DecoderError> {
    let count = pecos_decoder_core::DemMatchingGraph::from_dem_str(dem)?.skipped_hyperedges;
    if count > 0 {
        return Err(DecoderError::InvalidConfiguration(format!(
            "{context} needs a graphlike model, but this DEM has {count} mechanism(s) touching three or more detectors. Pass a graphlike (^ decomposed) model or use a hyperedge-capable decoder such as bp_osd or tesseract."
        )));
    }
    Ok(())
}

pub(super) fn build(
    spec: &DecoderSpec,
    model: &DecodeModel,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    build_in_context(spec, model, &ProjectionContext::default())
}

fn build_in_context(
    spec: &DecoderSpec,
    model: &DecodeModel,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let prepared = prepare_model_in_context(spec, model, context)?;
    let model = &prepared.model;
    #[cfg(test)]
    if spec.requires_graphlike_model() && spec.irreducible_policy().is_some() {
        LEAF_MODELS.with(|models| models.borrow_mut().push(model.clone()));
    }
    let decoder = match (spec, model) {
        (
            DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                ..
            }),
            DecodeModel::HybridDem { full, decomposed },
        ) => {
            ensure_graphlike_matching_model("belief_matching_hybrid decomposed model", decomposed)?;
            build_belief_matching_hybrid(full, decomposed)
        }
        (
            DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                ..
            }),
            DecodeModel::SingleDem(_),
        )
        | (
            DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                ..
            }),
            DecodeModel::StructuredDem(_),
        ) => Err(DecoderError::InvalidConfiguration(
            "belief_matching_hybrid requires DecodeModel::HybridDem".to_string(),
        )),
        (_, DecodeModel::SingleDem(dem)) => build_single(spec, dem, context),
        (_, DecodeModel::StructuredDem(dem)) => build_structured_single(spec, dem, context),
        (_, DecodeModel::HybridDem { .. }) => Err(DecoderError::InvalidConfiguration(format!(
            "{} requires DecodeModel::SingleDem or DecodeModel::StructuredDem",
            family_name(spec)
        ))),
    }?;
    let num_detectors = match model {
        DecodeModel::SingleDem(dem) => pecos_decoder_core::dem::utils::parse_dem_metadata(dem)?.0,
        DecodeModel::StructuredDem(dem) => dem.num_detectors,
        DecodeModel::HybridDem { decomposed, .. } => {
            pecos_decoder_core::dem::utils::parse_dem_metadata(decomposed)?.0
        }
    };
    Ok(Box::new(ModelDimensionDecoder {
        inner: decoder,
        num_detectors,
    }))
}

struct ModelDimensionDecoder {
    inner: Box<dyn ObservableDecoder>,
    num_detectors: usize,
}

impl ObservableDecoder for ModelDimensionDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.num_detectors)
    }

    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, DecoderError> {
        self.inner.decode_obs(syndrome)
    }

    fn decode_batch_to_observables(
        &mut self,
        shots: &[u8],
        num_shots: usize,
        num_detectors: usize,
    ) -> Result<Vec<pecos_decoder_core::obs_mask::ObsMask>, DecoderError> {
        self.inner
            .decode_batch_to_observables(shots, num_shots, num_detectors)
    }

    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        // Forward rather than inherit the trait default: an inner decoder that
        // overrides this method must keep its own semantics when wrapped.
        self.inner.decode_to_observables(syndrome)
    }
}

fn build_single(
    spec: &DecoderSpec,
    dem: &str,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    match spec {
        DecoderSpec::PyMatching(config) => build_pymatching(dem, config),
        DecoderSpec::Tesseract(config) => build_tesseract(dem, config),
        DecoderSpec::TesseractTrellis(config) => build_tesseract_trellis(dem, config),
        DecoderSpec::KMwpm(config) => build_k_mwpm(dem, *config),
        DecoderSpec::AStar(_) => build_astar(dem, false),
        DecoderSpec::AStarFull => build_astar(dem, true),
        DecoderSpec::FusionBlossom(config) => build_fusion_blossom(dem, *config),
        DecoderSpec::PerturbedFusionBlossomCorrelated(config) => {
            build_perturbed_fusion_blossom(dem, config)
        }
        DecoderSpec::BpOsd(config) => build_bp_osd(dem, config),
        DecoderSpec::BpLsd(config) => build_bp_lsd(dem, config),
        DecoderSpec::BeliefFind => build_belief_find(dem),
        DecoderSpec::UnionFind => build_union_find(dem),
        DecoderSpec::RelayBp(config) => build_relay_bp(dem, config),
        DecoderSpec::MinSumBp(config) => build_min_sum_bp(dem, config),
        DecoderSpec::PecosUf(config) => build_pecos_uf(dem, config.preset),
        DecoderSpec::BeliefMatching(config) => build_belief_matching(dem, config),
        DecoderSpec::Windowed(config) => build_windowed(dem, config),
        DecoderSpec::Mwpf(config) => build_mwpf(dem, config),
        DecoderSpec::Perturbed(config) => build_perturbed(dem, config, &context.inner("perturbed")),
        DecoderSpec::BeamSearch(config) => build_beamsearch(dem, config, &context.phase2()),
        DecoderSpec::Ensemble(config) => {
            build_ensemble(&DecodeModel::SingleDem(dem.to_string()), config, context)
        }
    }
}

fn build_structured_single(
    spec: &DecoderSpec,
    dem: &StructuredDem,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    match spec {
        DecoderSpec::Windowed(config) => build_windowed_structured(dem, config),
        DecoderSpec::BeamSearch(config) => {
            build_beamsearch_structured(dem, config, &context.phase2())
        }
        DecoderSpec::Ensemble(config) => {
            build_ensemble(&DecodeModel::StructuredDem(dem.clone()), config, context)
        }
        DecoderSpec::Perturbed(config) => {
            build_perturbed_structured(dem, config, &context.inner("perturbed"))
        }
        _ => build_single(spec, &dem.to_dem_string(), context),
    }
}

pub(super) fn family_name(spec: &DecoderSpec) -> &'static str {
    match spec {
        DecoderSpec::PyMatching(_) => "pymatching",
        DecoderSpec::Tesseract(_) => "tesseract",
        DecoderSpec::TesseractTrellis(_) => "tesseract_trellis",
        DecoderSpec::KMwpm(_) => "k_mwpm",
        DecoderSpec::AStar(_) => "astar",
        DecoderSpec::AStarFull => "astar_full",
        DecoderSpec::FusionBlossom(_) => "fusion_blossom",
        DecoderSpec::PerturbedFusionBlossomCorrelated(_) => "perturbed_fb_corr",
        DecoderSpec::BpOsd(_) => "bp_osd",
        DecoderSpec::BpLsd(_) => "bp_lsd",
        DecoderSpec::BeliefFind => "belief_find",
        DecoderSpec::UnionFind => "union_find",
        DecoderSpec::RelayBp(_) => "relay_bp",
        DecoderSpec::MinSumBp(_) => "min_sum_bp",
        DecoderSpec::PecosUf(_) => "pecos_uf",
        DecoderSpec::BeliefMatching(_) => "belief_matching",
        DecoderSpec::Windowed(_) => "windowed",
        DecoderSpec::Mwpf(_) => "mwpf",
        DecoderSpec::Perturbed(_) => "perturbed",
        DecoderSpec::BeamSearch(_) => "beamsearch",
        DecoderSpec::Ensemble(_) => "ensemble",
    }
}

#[cfg(not(all(
    feature = "pymatching",
    feature = "tesseract",
    feature = "fusion-blossom",
    feature = "ldpc",
    feature = "relay-bp",
    feature = "uf",
    feature = "mwpf"
)))]
fn unavailable<T>(family: &'static str, required_feature: &'static str) -> Result<T, DecoderError> {
    Err(DecoderError::BackendUnavailable {
        family,
        required_feature,
    })
}

#[cfg(any(
    feature = "pymatching",
    feature = "tesseract",
    feature = "fusion-blossom",
    feature = "ldpc",
    feature = "relay-bp",
    feature = "mwpf"
))]
fn internal(error: impl std::fmt::Display) -> DecoderError {
    DecoderError::InternalError(error.to_string())
}

#[cfg(any(feature = "uf", test))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedBeamSearchConfig {
    buffer_size: usize,
    commit_weight_max: f64,
}

#[cfg(any(feature = "uf", test))]
fn resolve_beamsearch_config(config: &BeamSearchConfig) -> ResolvedBeamSearchConfig {
    let buffer_size = if config.buffer_size == 0 {
        if config.step_size > 0 {
            config.step_size
        } else {
            5
        }
    } else {
        config.buffer_size
    };
    let commit_weight_max = if config.commit_weight_max == 0.0 {
        2.5
    } else {
        config.commit_weight_max
    };
    ResolvedBeamSearchConfig {
        buffer_size,
        commit_weight_max,
    }
}

#[cfg(feature = "pymatching")]
fn build_pymatching(
    dem: &str,
    config: &PyMatchingConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let mut decoder = if config.correlated {
        crate::PyMatchingDecoder::from_dem_with_correlations(dem, true)
    } else {
        crate::PyMatchingDecoder::from_dem(dem)
    }
    .map_err(DecoderError::from)?;
    if let Some(error_probability) = config.error_probability {
        decoder
            .set_all_error_probabilities(error_probability)
            .map_err(internal)?;
    }
    Ok(Box::new(decoder))
}

#[cfg(not(feature = "pymatching"))]
fn build_pymatching(
    _dem: &str,
    _config: &PyMatchingConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("pymatching", "pymatching")
}

#[cfg(feature = "tesseract")]
fn trellis_engine_config(config: &TesseractTrellisConfig) -> crate::TesseractTrellisConfig {
    use super::config::TesseractTrellisRankingMode as SpecRanking;
    use crate::TesseractTrellisRankingMode as EngineRanking;
    crate::TesseractTrellisConfig {
        beam_width: config.beam_width,
        beam_eps: config.beam_eps,
        future_detcost_scale: config.future_detcost_scale,
        verbose: config.verbose,
        merge_errors: config.merge_errors,
        ranking_mode: match config.ranking_mode {
            SpecRanking::MassOnly => EngineRanking::MassOnly,
            SpecRanking::FutureDetcostRanked => EngineRanking::FutureDetcostRanked,
            SpecRanking::FutureActiveDetcostRanked => EngineRanking::FutureActiveDetcostRanked,
        },
    }
}

#[cfg(feature = "tesseract")]
fn build_tesseract_trellis(
    dem: &str,
    config: &TesseractTrellisConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    Ok(Box::new(
        crate::TesseractTrellisDecoder::new(dem, trellis_engine_config(config))
            .map_err(DecoderError::from)?,
    ))
}

#[cfg(not(feature = "tesseract"))]
fn build_tesseract_trellis(
    _dem: &str,
    _config: &TesseractTrellisConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("tesseract_trellis", "tesseract")
}

#[cfg(feature = "tesseract")]
fn build_tesseract(
    dem: &str,
    config: &TesseractConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let mut engine_config = match config.preset {
        TesseractPreset::Default => TesseractEngineConfig::default(),
        TesseractPreset::Fast => TesseractEngineConfig::fast(),
        TesseractPreset::Accurate => TesseractEngineConfig::accurate(),
    };
    if let Some(merge_errors) = config.merge_errors {
        engine_config.merge_errors = merge_errors;
    }
    if let Some(det_beam) = config.det_beam {
        engine_config.det_beam = det_beam;
    }
    if let Some(beam_climbing) = config.beam_climbing {
        engine_config.beam_climbing = beam_climbing;
    }
    if let Some(verbose) = config.verbose {
        engine_config.verbose = verbose;
    }
    if let Some(no_revisit_dets) = config.no_revisit_dets {
        engine_config.no_revisit_dets = no_revisit_dets;
    }
    if let Some(pqlimit) = config.pqlimit {
        engine_config.pqlimit = pqlimit;
    }
    if let Some(det_penalty) = config.det_penalty {
        engine_config.det_penalty = det_penalty;
    }
    Ok(Box::new(
        crate::TesseractDecoder::new(dem, engine_config).map_err(DecoderError::from)?,
    ))
}

#[cfg(not(feature = "tesseract"))]
fn build_tesseract(
    _dem: &str,
    _config: &TesseractConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("tesseract", "tesseract")
}

#[cfg(feature = "fusion-blossom")]
fn build_k_mwpm(
    dem: &str,
    config: KMwpmConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::k_mwpm::{KMwpmConfig as CoreConfig, KMwpmDecoder};

    let inner = crate::FusionBlossomDecoder::from_dem(dem).map_err(internal)?;
    Ok(Box::new(KMwpmDecoder::new(
        inner,
        CoreConfig { k: config.k },
    )))
}

#[cfg(not(feature = "fusion-blossom"))]
fn build_k_mwpm(
    _dem: &str,
    _config: KMwpmConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("k_mwpm", "fusion-blossom")
}

#[cfg(feature = "uf")]
fn build_astar(dem: &str, full: bool) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let config = crate::AStarConfig::default();
    let decoder = if full {
        crate::AStarDecoder::from_dem_full(dem, config)
    } else {
        crate::AStarDecoder::from_dem(dem, config)
    }?;
    Ok(Box::new(decoder))
}

#[cfg(not(feature = "uf"))]
fn build_astar(_dem: &str, full: bool) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable(if full { "astar_full" } else { "astar" }, "uf")
}

#[cfg(feature = "fusion-blossom")]
fn build_fusion_blossom(
    dem: &str,
    config: FusionBlossomConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    match (config.solver_type, config.correlated) {
        (FusionBlossomSolverType::Auto, false) => build_fusion_auto(dem),
        (FusionBlossomSolverType::Parallel, false) => build_fusion_parallel(dem),
        (FusionBlossomSolverType::Serial, false) => Ok(Box::new(
            crate::FusionBlossomDecoder::from_dem(dem).map_err(internal)?,
        )),
        (FusionBlossomSolverType::Legacy, false) => Ok(Box::new(
            crate::FusionBlossomDecoder::from_dem_with_solver_type(
                dem,
                FusionBlossomEngineSolverType::Legacy,
            )
            .map_err(internal)?,
        )),
        (FusionBlossomSolverType::Auto | FusionBlossomSolverType::Serial, true) => {
            build_fusion_correlated(dem)
        }
        (FusionBlossomSolverType::Legacy, true) => Ok(Box::new(
            crate::FusionBlossomDecoder::from_dem_correlated_with_solver_type(
                dem,
                FusionBlossomEngineSolverType::Legacy,
            )
            .map_err(internal)?,
        )),
        (FusionBlossomSolverType::Parallel, true) => Err(DecoderError::InvalidConfiguration(
            "correlated Fusion Blossom does not support the parallel solver".to_string(),
        )),
    }
}

#[cfg(feature = "fusion-blossom")]
fn build_fusion_auto(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let graph = pecos_decoder_core::DemMatchingGraph::from_dem_str(dem)?;
    let has_coords = graph.detector_coords.iter().any(Option::is_some);
    if graph.num_detectors >= 500 && has_coords {
        build_fusion_parallel_from_graph(dem, &graph)
    } else {
        Ok(Box::new(
            crate::FusionBlossomDecoder::from_dem(dem).map_err(internal)?,
        ))
    }
}

#[cfg(feature = "fusion-blossom")]
fn build_fusion_correlated(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::correlation_table::CorrelationTable;
    use pecos_decoder_core::two_pass_decoder::TwoPassDecoder;
    use std::collections::BTreeMap;

    let graph = pecos_decoder_core::DemMatchingGraph::from_dem_str(dem)?;
    graph.ensure_observables_fit_u64()?;
    let engine_config = FusionBlossomEngineConfig {
        num_nodes: Some(graph.num_detectors),
        num_observables: graph.num_observables,
        ..Default::default()
    };
    let mut decoder = crate::FusionBlossomDecoder::new(engine_config).map_err(internal)?;
    let mut edge_index_map = BTreeMap::new();
    let mut base_weights = Vec::with_capacity(graph.edges.len());
    for (index, edge) in graph.edges.iter().enumerate() {
        base_weights.push(edge.weight);
        let observables: Vec<usize> = edge
            .observables
            .iter()
            .map(|&value| value as usize)
            .collect();
        let key = if let Some(node2) = edge.node2 {
            decoder
                .add_edge(
                    edge.node1 as usize,
                    node2 as usize,
                    &observables,
                    Some(edge.weight),
                )
                .map_err(internal)?;
            ordered_edge(edge.node1, node2)
        } else {
            decoder
                .add_boundary_edge(edge.node1 as usize, &observables, Some(edge.weight))
                .map_err(internal)?;
            (edge.node1, u32::MAX)
        };
        edge_index_map.insert(key, index);
    }
    let correlations = CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len())?;
    Ok(Box::new(TwoPassDecoder::new(
        decoder,
        base_weights,
        correlations,
    )))
}

#[cfg(feature = "fusion-blossom")]
fn build_fusion_parallel(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let graph = pecos_decoder_core::DemMatchingGraph::from_dem_str(dem)?;
    build_fusion_parallel_from_graph(dem, &graph)
}

#[cfg(feature = "fusion-blossom")]
fn build_fusion_parallel_from_graph(
    dem: &str,
    graph: &pecos_decoder_core::DemMatchingGraph,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use std::collections::BTreeMap;

    graph.ensure_observables_fit_u64()?;
    let mut round_groups: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for (id, coord) in graph.detector_coords.iter().enumerate() {
        let time = coord
            .as_ref()
            .and_then(|values| values.get(2))
            .copied()
            .unwrap_or(0.0);
        let scaled_time = ordered_time_key(time);
        let detector = u32::try_from(id).map_err(|_| {
            DecoderError::InvalidConfiguration("too many detectors for Fusion Blossom".to_string())
        })?;
        round_groups.entry(scaled_time).or_default().push(detector);
    }
    let num_rounds = round_groups.len();
    if num_rounds < 2 {
        return Ok(Box::new(
            crate::FusionBlossomDecoder::from_dem(dem).map_err(internal)?,
        ));
    }

    let num_detectors = graph.num_detectors;
    let mut old_to_new = vec![0; num_detectors];
    let mut detector_to_round = vec![0; num_detectors];
    let mut next_id = 0;
    let mut round_starts = Vec::new();
    let mut round_ends = Vec::new();
    let mut partition_boundary = Vec::new();
    for (round_index, ids) in round_groups.values().enumerate() {
        round_starts.push(next_id);
        for &old_id in ids {
            old_to_new[old_id as usize] = next_id;
            detector_to_round[old_id as usize] = round_index;
            next_id += 1;
        }
        partition_boundary.push(next_id);
        next_id += 1;
        round_ends.push(next_id);
    }
    let total_vertices = next_id;
    let engine_config = FusionBlossomEngineConfig {
        num_nodes: Some(total_vertices),
        num_observables: graph.num_observables,
        ..Default::default()
    };
    let mut decoder = crate::FusionBlossomDecoder::new(engine_config).map_err(internal)?;
    for &boundary in &partition_boundary {
        decoder.virtual_vertices.push(boundary);
    }
    for edge in &graph.edges {
        let observables: Vec<usize> = edge
            .observables
            .iter()
            .map(|&value| value as usize)
            .collect();
        let node1 = old_to_new[edge.node1 as usize];
        let node2 = edge.node2.map_or_else(
            || partition_boundary[detector_to_round[edge.node1 as usize]],
            |node2| old_to_new[node2 as usize],
        );
        decoder
            .add_edge(node1, node2, &observables, Some(edge.weight))
            .map_err(internal)?;
    }

    let partition_count = num_rounds.clamp(2, 4);
    let mut partitions = crate::PartitionConfig::new(total_vertices);
    partitions.partitions.clear();
    for partition_index in 0..partition_count {
        let start_round = partition_index * num_rounds / partition_count;
        let end_round = (partition_index + 1) * num_rounds / partition_count;
        let start_vertex = if partition_index == 0 {
            round_starts[start_round]
        } else {
            round_starts[(start_round + 1).min(num_rounds - 1)]
        };
        let end_vertex = round_ends[end_round - 1];
        if start_vertex < end_vertex {
            partitions
                .partitions
                .push(crate::VertexRange::new(start_vertex, end_vertex));
        }
    }
    let number_of_partitions = partitions.partitions.len();
    partitions.fusions.clear();
    if number_of_partitions > 1 {
        let mut active: Vec<usize> = (0..number_of_partitions).collect();
        while active.len() > 1 {
            let mut next_active = Vec::new();
            let mut index = 0;
            while index + 1 < active.len() {
                partitions.fusions.push((active[index], active[index + 1]));
                next_active.push(number_of_partitions + partitions.fusions.len() - 1);
                index += 2;
            }
            if index < active.len() {
                next_active.push(active[index]);
            }
            active = next_active;
        }
    }
    decoder.set_partition_config(partitions);
    Ok(Box::new(RelabeledFusionBlossomDecoder {
        decoder,
        old_to_new,
    }))
}

#[cfg(feature = "fusion-blossom")]
struct RelabeledFusionBlossomDecoder {
    decoder: crate::FusionBlossomDecoder,
    old_to_new: Vec<usize>,
}

#[cfg(feature = "fusion-blossom")]
impl ObservableDecoder for RelabeledFusionBlossomDecoder {
    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, DecoderError> {
        let mut relabeled = vec![0; self.decoder.num_nodes()];
        for (old_id, &value) in syndrome.iter().enumerate() {
            if let Some(&new_id) = self.old_to_new.get(old_id)
                && new_id < relabeled.len()
            {
                relabeled[new_id] = value;
            }
        }
        self.decoder.decode_obs(&relabeled)
    }
}

#[cfg(not(feature = "fusion-blossom"))]
fn build_fusion_blossom(
    _dem: &str,
    _config: FusionBlossomConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("fusion_blossom", "fusion-blossom")
}

#[cfg(feature = "fusion-blossom")]
fn ordered_time_key(time: f64) -> u64 {
    const SIGN_MASK: u64 = 1 << 63;

    let truncated = (time * 1000.0).trunc();
    let normalized = if truncated == 0.0 || truncated.is_nan() {
        0.0
    } else {
        truncated
    };
    let bits = normalized.to_bits();
    if bits & SIGN_MASK == 0 {
        bits ^ SIGN_MASK
    } else {
        !bits
    }
}

#[cfg(feature = "fusion-blossom")]
fn build_perturbed_fusion_blossom(
    dem: &str,
    config: &PerturbedFusionBlossomConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::perturbed::{PerturbedConfig as CoreConfig, build_perturbed_ensemble};

    let core_config = CoreConfig {
        k: config.k,
        sigma: config.sigma,
        seed: config.seed,
    };
    let decoder = build_perturbed_ensemble(dem, &core_config, |member_dem| {
        Ok(Box::new(
            crate::FusionBlossomDecoder::from_dem_correlated(member_dem).map_err(internal)?,
        ))
    })?;
    Ok(Box::new(decoder))
}

#[cfg(not(feature = "fusion-blossom"))]
fn build_perturbed_fusion_blossom(
    _dem: &str,
    _config: &PerturbedFusionBlossomConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("perturbed_fb_corr", "fusion-blossom")
}

#[cfg(feature = "ldpc")]
fn dem_matrix(
    dem: &str,
) -> Result<(pecos_decoder_core::DemCheckMatrix, crate::SparseMatrix), DecoderError> {
    let matrix = pecos_decoder_core::DemCheckMatrix::from_dem_str(dem)?;
    let sparse = crate::SparseMatrix::from_dense(&matrix.check_matrix.view());
    Ok((matrix, sparse))
}

#[cfg(feature = "ldpc")]
fn ldpc_bp_schedule(schedule: SpecBpSchedule) -> LdpcBpSchedule {
    match schedule {
        SpecBpSchedule::Serial => LdpcBpSchedule::Serial,
        SpecBpSchedule::Parallel => LdpcBpSchedule::Parallel,
        SpecBpSchedule::SerialRelative => LdpcBpSchedule::SerialRelative,
    }
}

#[cfg(feature = "ldpc")]
fn build_bp_osd(
    dem: &str,
    config: &BpOsdConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let (matrix, sparse) = dem_matrix(dem)?;
    let priors = config.error_rate.map_or_else(
        || matrix.error_priors.clone(),
        |rate| vec![rate; matrix.num_mechanisms],
    );
    let bp_method = if config.ms_scaling_factor.is_some() {
        crate::BpMethod::MinimumSum
    } else {
        crate::BpMethod::ProductSum
    };
    let osd_method = if config.osd_order == 0 {
        crate::OsdMethod::Osd0
    } else {
        crate::OsdMethod::OsdCs
    };
    let decoder = crate::BpOsdDecoder::new(
        &sparse,
        None,
        Some(&priors),
        config.max_iter,
        bp_method,
        ldpc_bp_schedule(config.bp_schedule),
        config.ms_scaling_factor.unwrap_or(1.0),
        osd_method,
        config.osd_order,
        crate::InputVectorType::Syndrome,
        None,
        None,
        config.random_schedule_seed,
    )
    .map_err(internal)?;
    Ok(Box::new(
        pecos_decoder_core::CheckMatrixObservableDecoder::new(decoder, matrix),
    ))
}

#[cfg(not(feature = "ldpc"))]
fn build_bp_osd(
    _dem: &str,
    _config: &BpOsdConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("bp_osd", "ldpc")
}

#[cfg(feature = "ldpc")]
fn build_bp_lsd(
    dem: &str,
    config: &BpLsdConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let (matrix, sparse) = dem_matrix(dem)?;
    let priors = config.error_rate.map_or_else(
        || matrix.error_priors.clone(),
        |rate| vec![rate; matrix.num_mechanisms],
    );
    let bp_method = if config.ms_scaling_factor.is_some() {
        crate::BpMethod::MinimumSum
    } else {
        crate::BpMethod::ProductSum
    };
    let lsd_method = if config.lsd_order == 0 {
        crate::OsdMethod::Off
    } else {
        crate::OsdMethod::OsdCs
    };
    let decoder = crate::BpLsdDecoder::new(
        &sparse,
        None,
        Some(&priors),
        config.max_iter,
        bp_method,
        ldpc_bp_schedule(config.bp_schedule),
        config.ms_scaling_factor.unwrap_or(1.0),
        lsd_method,
        config.lsd_order,
        config.bits_per_step,
        crate::InputVectorType::Syndrome,
        None,
        None,
        config.random_schedule_seed,
    )
    .map_err(internal)?;
    Ok(Box::new(
        pecos_decoder_core::CheckMatrixObservableDecoder::new(decoder, matrix),
    ))
}

#[cfg(not(feature = "ldpc"))]
fn build_bp_lsd(
    _dem: &str,
    _config: &BpLsdConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("bp_lsd", "ldpc")
}

#[cfg(feature = "ldpc")]
fn build_belief_find(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let (matrix, sparse) = dem_matrix(dem)?;
    let decoder = crate::BeliefFindDecoder::new(
        &sparse,
        None,
        Some(&matrix.error_priors),
        100,
        crate::BpMethod::ProductSum,
        1.0,
        LdpcBpSchedule::Parallel,
        None,
        None,
        None,
        crate::UfMethod::Inversion,
        0,
    )
    .map_err(internal)?;
    Ok(Box::new(
        pecos_decoder_core::CheckMatrixObservableDecoder::new(decoder, matrix),
    ))
}

#[cfg(not(feature = "ldpc"))]
fn build_belief_find(_dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("belief_find", "ldpc")
}

#[cfg(feature = "ldpc")]
fn build_union_find(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let (matrix, sparse) = dem_matrix(dem)?;
    let decoder =
        crate::UnionFindDecoder::new(&sparse, crate::UfMethod::Inversion).map_err(internal)?;
    let llrs = matrix
        .error_priors
        .iter()
        .map(|&probability| {
            if probability > 0.0 && probability < 1.0 {
                ((1.0 - probability) / probability).ln()
            } else {
                0.0
            }
        })
        .collect();
    Ok(Box::new(crate::WeightedUnionFindDecoder::new(
        decoder, matrix, llrs,
    )))
}

#[cfg(not(feature = "ldpc"))]
fn build_union_find(_dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("union_find", "ldpc")
}

#[cfg(feature = "relay-bp")]
fn relay_priors(matrix: &pecos_decoder_core::DemCheckMatrix, error_rate: Option<f64>) -> Vec<f64> {
    error_rate.map_or_else(
        || matrix.error_priors.clone(),
        |rate| vec![rate; matrix.num_mechanisms],
    )
}

#[cfg(feature = "relay-bp")]
fn build_relay_bp(
    dem: &str,
    config: &RelayBpConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let matrix = pecos_decoder_core::DemCheckMatrix::from_dem_str(dem)?;
    let priors = relay_priors(&matrix, config.error_rate);
    let stopping = match config.stopping_criterion {
        RelayStoppingCriterion::PreIter => crate::StoppingCriterion::PreIter,
        RelayStoppingCriterion::All => crate::StoppingCriterion::All,
        RelayStoppingCriterion::FirstConvergence => {
            crate::StoppingCriterion::NConv { stop_after: 1 }
        }
        RelayStoppingCriterion::NConvergences(stop_after) => {
            crate::StoppingCriterion::NConv { stop_after }
        }
    };
    let decoder = crate::RelayBpBuilder::new(&matrix.check_matrix.view())
        .error_priors(&priors)
        .max_iter(config.max_iter)
        .alpha(config.alpha)
        .alpha_iteration_scaling_factor(config.alpha_iteration_scaling_factor)
        .gamma0(config.gamma0)
        .pre_iter(config.pre_iter)
        .num_sets(config.num_sets)
        .set_max_iter(config.set_max_iter)
        .gamma_dist_interval(config.gamma_dist_interval)
        .stopping_criterion(stopping)
        .seed(config.seed)
        .build()
        .map_err(internal)?;
    Ok(Box::new(
        pecos_decoder_core::CheckMatrixObservableDecoder::new(decoder, matrix),
    ))
}

#[cfg(not(feature = "relay-bp"))]
fn build_relay_bp(
    _dem: &str,
    _config: &RelayBpConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("relay_bp", "relay-bp")
}

#[cfg(feature = "relay-bp")]
fn build_min_sum_bp(
    dem: &str,
    config: &MinSumBpConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let matrix = pecos_decoder_core::DemCheckMatrix::from_dem_str(dem)?;
    let priors = relay_priors(&matrix, config.error_rate);
    let decoder = crate::MinSumBpBuilder::new(&matrix.check_matrix.view())
        .error_priors(&priors)
        .max_iter(config.max_iter)
        .alpha(config.alpha)
        .build()
        .map_err(internal)?;
    Ok(Box::new(
        pecos_decoder_core::CheckMatrixObservableDecoder::new(decoder, matrix),
    ))
}

#[cfg(not(feature = "relay-bp"))]
fn build_min_sum_bp(
    _dem: &str,
    _config: &MinSumBpConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("min_sum_bp", "relay-bp")
}

#[cfg(feature = "uf")]
fn build_pecos_uf(
    dem: &str,
    preset: PecosUfPreset,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    match preset {
        PecosUfPreset::Fast => Ok(Box::new(crate::UfDecoder::from_dem(
            dem,
            crate::UfDecoderConfig::fast(),
        )?)),
        PecosUfPreset::Balanced | PecosUfPreset::Accurate => build_correlated_uf(dem),
        PecosUfPreset::Bp => Ok(Box::new(crate::BpUfDecoder::from_dem(
            dem,
            crate::BpUfConfig::balanced(),
        )?)),
        PecosUfPreset::BpSerial => Ok(Box::new(crate::BpUfDecoder::from_dem(
            dem,
            crate::BpUfConfig::accurate(),
        )?)),
    }
}

#[cfg(feature = "uf")]
fn build_correlated_uf(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::correlation_table::CorrelationTable;
    use pecos_decoder_core::two_pass_decoder::TwoPassDecoder;
    use std::collections::BTreeMap;

    let graph = pecos_decoder_core::DemMatchingGraph::from_dem_str(dem)?;
    let mut edge_index_map = BTreeMap::new();
    let mut base_weights = Vec::with_capacity(graph.edges.len());
    for (index, edge) in graph.edges.iter().enumerate() {
        base_weights.push(edge.weight);
        edge_index_map.insert(
            edge.node2.map_or((edge.node1, u32::MAX), |node2| {
                ordered_edge(edge.node1, node2)
            }),
            index,
        );
    }
    let correlations = CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len())?;
    crate::UfDecoder::check_non_negative_weights(&graph)?;
    let decoder =
        crate::UfDecoder::from_matching_graph(&graph, crate::UfDecoderConfig::balanced())?;
    Ok(Box::new(TwoPassDecoder::new(
        decoder,
        base_weights,
        correlations,
    )))
}

#[cfg(not(feature = "uf"))]
fn build_pecos_uf(
    _dem: &str,
    _preset: PecosUfPreset,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("pecos_uf", "uf")
}

#[cfg(all(feature = "uf", feature = "fusion-blossom"))]
fn build_belief_matching(
    dem: &str,
    config: &BeliefMatchingConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    match config.mode {
        BeliefMatchingMode::Hybrid => Err(DecoderError::InvalidConfiguration(
            "belief_matching_hybrid requires DecodeModel::HybridDem".to_string(),
        )),
        mode => build_belief_matching_with_dems(dem, dem, mode),
    }
}

#[cfg(all(feature = "uf", feature = "fusion-blossom"))]
fn build_belief_matching_hybrid(
    full: &str,
    decomposed: &str,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    build_belief_matching_with_dems(full, decomposed, BeliefMatchingMode::Hybrid)
}

#[cfg(all(feature = "uf", feature = "fusion-blossom"))]
fn build_belief_matching_with_dems(
    bp_dem: &str,
    matching_dem: &str,
    mode: BeliefMatchingMode,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::bp_matching::BpMatchingDecoder;
    use pecos_decoder_core::correlation_table::CorrelationTable;
    use std::collections::BTreeMap;

    let bp = match mode {
        BeliefMatchingMode::MatchingGraphBp => {
            crate::BpUfDecoder::from_dem(bp_dem, crate::BpUfConfig::matching_bp())?
        }
        BeliefMatchingMode::Hybrid => {
            crate::BpUfDecoder::from_dual_dem(bp_dem, matching_dem, crate::BpUfConfig::balanced())?
        }
        BeliefMatchingMode::Standard | BeliefMatchingMode::Correlated => {
            crate::BpUfDecoder::from_dem(bp_dem, crate::BpUfConfig::balanced())?
        }
    };
    let graph = pecos_decoder_core::DemMatchingGraph::from_dem_str(matching_dem)?;
    graph.ensure_observables_fit_u64()?;
    let engine_config = FusionBlossomEngineConfig {
        num_nodes: Some(graph.num_detectors),
        num_observables: graph.num_observables,
        ..Default::default()
    };
    let mut matching = crate::FusionBlossomDecoder::new(engine_config).map_err(internal)?;
    let correlated = !matches!(mode, BeliefMatchingMode::Standard);
    if correlated {
        let mut edge_index_map = BTreeMap::new();
        for (index, edge) in graph.edges.iter().enumerate() {
            let key = add_fusion_edge(&mut matching, edge)?;
            edge_index_map.insert(key, index);
        }
        let correlations =
            CorrelationTable::from_dem_str(matching_dem, &edge_index_map, graph.edges.len())?;
        Ok(Box::new(BpMatchingDecoder::with_correlations(
            matching,
            bp,
            correlations,
        )))
    } else {
        for edge in &graph.edges {
            add_fusion_edge(&mut matching, edge)?;
        }
        Ok(Box::new(BpMatchingDecoder::new(matching, bp)))
    }
}

#[cfg(all(feature = "uf", feature = "fusion-blossom"))]
fn add_fusion_edge(
    decoder: &mut crate::FusionBlossomDecoder,
    edge: &pecos_decoder_core::dem::MatchingEdge,
) -> Result<(u32, u32), DecoderError> {
    let observables: Vec<usize> = edge
        .observables
        .iter()
        .map(|&value| value as usize)
        .collect();
    if let Some(node2) = edge.node2 {
        decoder
            .add_edge(
                edge.node1 as usize,
                node2 as usize,
                &observables,
                Some(edge.weight),
            )
            .map_err(internal)?;
        Ok(ordered_edge(edge.node1, node2))
    } else {
        decoder
            .add_boundary_edge(edge.node1 as usize, &observables, Some(edge.weight))
            .map_err(internal)?;
        Ok((edge.node1, u32::MAX))
    }
}

#[cfg(any(feature = "uf", feature = "fusion-blossom"))]
fn ordered_edge(node1: u32, node2: u32) -> (u32, u32) {
    if node1 <= node2 {
        (node1, node2)
    } else {
        (node2, node1)
    }
}

#[cfg(not(feature = "uf"))]
fn build_belief_matching(
    _dem: &str,
    _config: &BeliefMatchingConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("belief_matching", "uf")
}

#[cfg(all(feature = "uf", not(feature = "fusion-blossom")))]
fn build_belief_matching(
    _dem: &str,
    _config: &BeliefMatchingConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("belief_matching", "fusion-blossom")
}

#[cfg(not(feature = "uf"))]
fn build_belief_matching_hybrid(
    _full: &str,
    _decomposed: &str,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("belief_matching_hybrid", "uf")
}

#[cfg(all(feature = "uf", not(feature = "fusion-blossom")))]
fn build_belief_matching_hybrid(
    _full: &str,
    _decomposed: &str,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("belief_matching_hybrid", "fusion-blossom")
}

#[cfg(feature = "uf")]
fn build_windowed(
    dem: &str,
    config: &WindowedConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let dem = StructuredDem::from_dem_str(dem)?;
    build_windowed_structured(&dem, config)
}

#[cfg(feature = "uf")]
fn build_windowed_structured(
    dem: &StructuredDem,
    config: &WindowedConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let decoder = crate::StreamingWindowedDecoder::from_structured_dem(
        dem,
        WindowedEngineConfig {
            step: config.step_size,
            buffer: config.buffer_size,
        },
        |window| build_window_inner(window, &config.inner),
    )?;
    Ok(Box::new(decoder))
}

#[cfg(feature = "uf")]
fn build_window_inner(
    window: &pecos_decoder_core::window::CommitWindow,
    inner: &DecoderSpec,
) -> Result<Box<dyn pecos_decoder_core::EdgeDecoder>, DecoderError> {
    match inner {
        DecoderSpec::PecosUf(PecosUfConfig {
            preset: PecosUfPreset::Fast,
            ..
        }) => Ok(Box::new(crate::UfDecoder::from_commit_window(
            window,
            crate::UfDecoderConfig::fast(),
        )?)),
        DecoderSpec::PyMatching(config) => build_window_matching(window, config),
        _ => Err(DecoderError::InvalidConfiguration(format!(
            "windowed does not support inner {inner:?}"
        ))),
    }
}

#[cfg(all(feature = "uf", feature = "pymatching"))]
fn build_window_matching(
    window: &pecos_decoder_core::window::CommitWindow,
    config: &PyMatchingConfig,
) -> Result<Box<dyn pecos_decoder_core::EdgeDecoder>, DecoderError> {
    let mut decoder =
        pecos_pymatching::PyMatchingEdgeDecoder::from_commit_window(window, config.correlated)?;
    if let Some(probability) = config.error_probability {
        decoder.set_all_error_probabilities(probability)?;
    }
    Ok(Box::new(decoder))
}

#[cfg(all(feature = "uf", not(feature = "pymatching")))]
fn build_window_matching(
    _window: &pecos_decoder_core::window::CommitWindow,
    _config: &PyMatchingConfig,
) -> Result<Box<dyn pecos_decoder_core::EdgeDecoder>, DecoderError> {
    Err(DecoderError::BackendUnavailable {
        family: "pymatching",
        required_feature: "pymatching",
    })
}

#[cfg(not(feature = "uf"))]
fn build_windowed(
    _dem: &str,
    _config: &WindowedConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("windowed", "uf")
}

#[cfg(not(feature = "uf"))]
fn build_windowed_structured(
    _dem: &StructuredDem,
    _config: &WindowedConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("windowed", "uf")
}

#[cfg(feature = "mwpf")]
fn build_mwpf(dem: &str, config: &MwpfConfig) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let solver_type = match config.solver_type {
        SpecMwpfSolverType::UnionFind => MwpfEngineSolverType::UnionFind,
        SpecMwpfSolverType::SingleHair => MwpfEngineSolverType::SingleHair,
        SpecMwpfSolverType::BpHybrid => MwpfEngineSolverType::BpHybrid,
        SpecMwpfSolverType::JointSingleHair => MwpfEngineSolverType::JointSingleHair,
    };
    Ok(Box::new(
        crate::MwpfDecoder::from_dem(
            dem,
            MwpfEngineConfig {
                solver_type,
                cluster_node_limit: config.cluster_node_limit,
                timeout: config.timeout,
                only_solve_primal_once: config.only_solve_primal_once,
            },
        )
        .map_err(internal)?,
    ))
}

#[cfg(not(feature = "mwpf"))]
fn build_mwpf(
    _dem: &str,
    _config: &MwpfConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("mwpf", "mwpf")
}

fn build_perturbed(
    dem: &str,
    config: &PerturbedConfig,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::perturbed::{PerturbedConfig as CoreConfig, build_perturbed_ensemble};

    let decoder = build_perturbed_ensemble(
        dem,
        &CoreConfig {
            k: config.k,
            sigma: config.sigma,
            seed: config.seed,
        },
        |member_dem| {
            build_in_context(
                &config.inner,
                &DecodeModel::SingleDem(member_dem.to_string()),
                context,
            )
        },
    )?;
    Ok(Box::new(decoder))
}

fn build_perturbed_structured(
    dem: &StructuredDem,
    config: &PerturbedConfig,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::perturbed::{PerturbedConfig as CoreConfig, build_perturbed_ensemble};

    let core_config = CoreConfig {
        k: config.k,
        sigma: config.sigma,
        seed: config.seed,
    };
    let inner = config.inner.clone();
    let decoder = build_perturbed_ensemble(&dem.to_dem_string(), &core_config, |member_dem| {
        let mut member = StructuredDem::from_dem_str(member_dem)?;
        member
            .detector_fault_sectors
            .clone_from(&dem.detector_fault_sectors);
        build_in_context(&inner, &DecodeModel::StructuredDem(member), context)
    })?;
    Ok(Box::new(decoder))
}

#[cfg(feature = "uf")]
fn build_beamsearch(
    dem: &str,
    config: &BeamSearchConfig,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let dem = StructuredDem::from_dem_str(dem)?;
    build_beamsearch_structured(&dem, config, context)
}

#[cfg(feature = "uf")]
fn build_beamsearch_structured(
    dem: &StructuredDem,
    config: &BeamSearchConfig,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let resolved = resolve_beamsearch_config(config);
    let engine_config = BeamSearchEngineConfig {
        beam_width: config.beam_width,
        perturbation_sigma: config.perturbation_sigma,
        seed: config.seed,
        window: BeamWindowConfig {
            step_size: config.step_size,
            buffer_size: resolved.buffer_size,
            seam_half_width: 0,
            core_extend: 0,
            commit_weight_max: resolved.commit_weight_max,
        },
    };
    let phase2 = config.phase2.clone();
    let decoder = crate::BeamSearchWindowedDecoder::from_structured_dem(
        dem,
        engine_config,
        |sub_dem| {
            ensure_graphlike_matching_model("beamsearch phase-1 window", sub_dem)?;
            crate::UfDecoder::from_dem(sub_dem, crate::UfDecoderConfig::windowed())
        },
        Some(|model: &StructuredDem| {
            build_in_context(&phase2, &DecodeModel::StructuredDem(model.clone()), context)
        }),
    )?;
    Ok(Box::new(decoder))
}

#[cfg(not(feature = "uf"))]
fn build_beamsearch(
    _dem: &str,
    _config: &BeamSearchConfig,
    _context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("beamsearch", "uf")
}

#[cfg(not(feature = "uf"))]
fn build_beamsearch_structured(
    _dem: &StructuredDem,
    _config: &BeamSearchConfig,
    _context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("beamsearch", "uf")
}

fn build_ensemble(
    model: &DecodeModel,
    config: &EnsembleConfig,
    context: &ProjectionContext,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    if config.members.is_empty() {
        return Err(DecoderError::InvalidConfiguration(
            "ensemble needs at least one decoder".to_string(),
        ));
    }
    let members = config
        .members
        .iter()
        .enumerate()
        .map(|(index, member)| build_in_context(member, model, &context.member(index)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Box::new(
        pecos_decoder_core::ensemble::EnsembleDecoder::new(members),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "uf")]
    const HYPEREDGE_DEM: &str = "error(0.1) D0 D1 D2 L0\nerror(0.05) D0\n";

    #[cfg(feature = "uf")]
    #[test]
    fn manually_constructed_windowed_spec_rejects_zero_step() {
        let spec = DecoderSpec::Windowed(WindowedConfig {
            step_size: 0,
            buffer_size: 1,
            inner: Box::new(DecoderSpec::PecosUf(PecosUfConfig {
                preset: PecosUfPreset::Fast,
                ..Default::default()
            })),
        });
        for model in [
            DecodeModel::SingleDem(String::new()),
            DecodeModel::StructuredDem(StructuredDem::from_dem_str("").unwrap()),
        ] {
            assert!(
                spec.build(&model)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("step must be at least 1")
            );
        }
    }

    #[cfg(feature = "uf")]
    #[test]
    fn windowed_spec_builds_from_structured_model() {
        let model = StructuredDem::from_dem_str(
            "error(0.1) D0 D1 L0\nerror(0.01) D0\ndetector(0, 0, 0) D0\ndetector(0, 0, 1) D1\nlogical_observable L0\n",
        )
        .unwrap();
        let spec = DecoderSpec::parse("windowed:step=1,buf=1,inner=pecos_uf").unwrap();
        let decoder = spec.build(&DecodeModel::StructuredDem(model)).unwrap();
        assert_eq!(decoder.num_detectors(), Some(2));
    }

    #[test]
    fn hyperedge_capable_decoders_are_not_classified_graphlike_only() {
        for spec in [
            DecoderSpec::BpOsd(BpOsdConfig::default()),
            DecoderSpec::Tesseract(TesseractConfig::default()),
            DecoderSpec::TesseractTrellis(TesseractTrellisConfig::default()),
            DecoderSpec::AStarFull,
        ] {
            assert!(
                !spec.requires_graphlike_model(),
                "{spec:?} must not be treated as graphlike-only",
            );
        }
    }

    #[test]
    fn composite_specs_are_not_classified_at_the_top_level() {
        // Windowing slices mechanisms per time window, so a top-level
        // hyperedge can become graphlike in every sub-model; classification
        // is leaf-only and the guard runs where the parser actually runs.
        for spec in [
            DecoderSpec::Windowed(WindowedConfig {
                step_size: 1,
                buffer_size: 1,
                inner: Box::new(DecoderSpec::PecosUf(PecosUfConfig {
                    preset: PecosUfPreset::Fast,
                    ..Default::default()
                })),
            }),
            DecoderSpec::BeamSearch(BeamSearchConfig::default()),
            DecoderSpec::Perturbed(PerturbedConfig {
                inner: Box::new(DecoderSpec::PecosUf(PecosUfConfig {
                    preset: PecosUfPreset::Fast,
                    ..Default::default()
                })),
                ..PerturbedConfig::default()
            }),
            DecoderSpec::Ensemble(EnsembleConfig {
                members: vec![DecoderSpec::PecosUf(PecosUfConfig {
                    preset: PecosUfPreset::Fast,
                    ..Default::default()
                })],
            }),
        ] {
            assert!(
                !spec.requires_graphlike_model(),
                "{spec:?} must defer graphlike classification to member builds",
            );
        }
    }

    #[cfg(feature = "uf")]
    #[test]
    fn windowed_refuses_irreducible_hyperedges_and_unsupported_inner() {
        let spec = DecoderSpec::parse("windowed:step=1,buf=1,inner=pecos_uf").unwrap();
        let dem =
            format!("{HYPEREDGE_DEM}detector(0,0,0) D0\ndetector(1,0,0) D1\ndetector(2,0,0) D2\n");
        let error = spec.build(&DecodeModel::SingleDem(dem)).err().unwrap();
        // contract: full-model projection now refuses before constructing commit windows.
        assert!(error.to_string().contains("1 irreducible errors"));
        assert!(error.to_string().contains("irreducible=drop"));
        let unsupported = DecoderSpec::parse("windowed:step=1,buf=1,inner=bp_osd").unwrap();
        let error = unsupported
            .build(&DecodeModel::SingleDem(DEM.to_string()))
            .err()
            .unwrap();
        assert!(error.to_string().contains("BpOsd"));
    }

    const DEM: &str = "error(0.1) D0 D1 L0\nerror(0.05) D1\n";

    // The wide-observable helpers below serve only feature-gated tests, so
    // they carry the union of their users' gates; a default-feature build
    // would otherwise see dead code.
    #[cfg(any(
        feature = "pymatching",
        feature = "ldpc",
        feature = "relay-bp",
        feature = "tesseract",
        feature = "fusion-blossom",
        feature = "uf",
        feature = "mwpf"
    ))]
    fn width_dem(highest_observable: usize) -> String {
        format!("error(0.1) D0 L{highest_observable}\ndetector(0, 0, 0) D0\n")
    }

    #[cfg(any(feature = "pymatching", feature = "ldpc", feature = "relay-bp"))]
    fn assert_wide_correct(spec: &DecoderSpec) {
        for observable in [63, 64] {
            let mut decoder = spec
                .clone()
                .build(&DecodeModel::SingleDem(width_dem(observable)))
                .unwrap();
            assert!(
                decoder.decode_obs(&[1]).unwrap().get(observable),
                "{spec:?} did not preserve observable {observable}"
            );
        }
    }

    #[cfg(any(
        feature = "tesseract",
        feature = "fusion-blossom",
        feature = "uf",
        feature = "mwpf"
    ))]
    fn assert_64_then_65_rejected(spec: &DecoderSpec) {
        let mut decoder = spec
            .clone()
            .build(&DecodeModel::SingleDem(width_dem(63)))
            .unwrap();
        assert!(decoder.decode_obs(&[0]).unwrap().is_zero());
        let error = spec
            .build(&DecodeModel::SingleDem(width_dem(64)))
            .err()
            .expect("65-observable model must fail during construction");
        assert!(
            error.to_string().contains("64"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "pymatching")]
    #[test]
    fn pymatching_and_wide_composites_preserve_64_and_65_observables() {
        let pymatching = DecoderSpec::PyMatching(PyMatchingConfig::default());
        assert_wide_correct(&pymatching);
        assert_wide_correct(&DecoderSpec::Perturbed(PerturbedConfig {
            inner: Box::new(pymatching.clone()),
            k: 1,
            ..PerturbedConfig::default()
        }));
        assert_wide_correct(&DecoderSpec::Ensemble(EnsembleConfig {
            members: vec![pymatching],
        }));
    }

    #[cfg(feature = "ldpc")]
    #[test]
    fn ldpc_family_preserves_64_and_65_observables() {
        for spec in [
            DecoderSpec::BpOsd(BpOsdConfig::default()),
            DecoderSpec::BpLsd(BpLsdConfig::default()),
            DecoderSpec::BeliefFind,
            DecoderSpec::UnionFind,
        ] {
            assert_wide_correct(&spec);
        }
    }

    #[cfg(feature = "relay-bp")]
    #[test]
    fn relay_family_preserves_64_and_65_observables() {
        for spec in [
            DecoderSpec::RelayBp(RelayBpConfig::default()),
            DecoderSpec::MinSumBp(MinSumBpConfig::default()),
        ] {
            assert_wide_correct(&spec);
        }
    }

    #[cfg(feature = "tesseract")]
    #[test]
    fn tesseract_accepts_64_and_rejects_65_observables() {
        assert_64_then_65_rejected(&DecoderSpec::Tesseract(TesseractConfig::default()));
    }

    #[cfg(feature = "fusion-blossom")]
    #[test]
    fn fusion_family_accepts_64_and_rejects_65_observables() {
        for spec in [
            DecoderSpec::FusionBlossom(FusionBlossomConfig::default()),
            DecoderSpec::KMwpm(KMwpmConfig::default()),
            DecoderSpec::PerturbedFusionBlossomCorrelated(PerturbedFusionBlossomConfig {
                k: 1,
                ..PerturbedFusionBlossomConfig::default()
            }),
        ] {
            assert_64_then_65_rejected(&spec);
        }
    }

    #[cfg(feature = "uf")]
    #[test]
    fn uf_and_windowed_families_accept_64_and_reject_65_observables() {
        for spec in [
            DecoderSpec::AStar(crate::spec::AStarConfig::default()),
            DecoderSpec::AStarFull,
            DecoderSpec::PecosUf(PecosUfConfig {
                preset: PecosUfPreset::Fast,
                ..Default::default()
            }),
            DecoderSpec::PecosUf(PecosUfConfig {
                preset: PecosUfPreset::Bp,
                ..Default::default()
            }),
            DecoderSpec::Windowed(WindowedConfig {
                step_size: 1,
                buffer_size: 1,
                inner: Box::new(DecoderSpec::PecosUf(PecosUfConfig {
                    preset: PecosUfPreset::Fast,
                    ..Default::default()
                })),
            }),
            DecoderSpec::BeamSearch(BeamSearchConfig {
                beam_width: 1,
                // compatibility: keep this UF-only test independent of PyMatching.
                phase2: Box::new(DecoderSpec::PecosUf(PecosUfConfig::default())),
                ..BeamSearchConfig::default()
            }),
        ] {
            assert_64_then_65_rejected(&spec);
        }
    }

    #[cfg(all(feature = "uf", feature = "fusion-blossom"))]
    #[test]
    fn belief_matching_family_accepts_64_and_rejects_65_observables() {
        for mode in [
            BeliefMatchingMode::Standard,
            BeliefMatchingMode::Correlated,
            BeliefMatchingMode::MatchingGraphBp,
        ] {
            assert_64_then_65_rejected(&DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                irreducible: IrreduciblePolicy::default(),
                mode,
                embedded_full_dem: None,
            }));
        }

        let hybrid = DecoderSpec::BeliefMatching(BeliefMatchingConfig {
            irreducible: IrreduciblePolicy::default(),
            mode: BeliefMatchingMode::Hybrid,
            embedded_full_dem: None,
        });
        let dem64 = width_dem(63);
        let mut decoder = hybrid
            .build(&DecodeModel::HybridDem {
                full: dem64.clone(),
                decomposed: dem64,
            })
            .unwrap();
        assert!(decoder.decode_obs(&[0]).unwrap().is_zero());
        let dem65 = width_dem(64);
        let error = hybrid
            .build(&DecodeModel::HybridDem {
                full: dem65.clone(),
                decomposed: dem65,
            })
            .err()
            .expect("65-observable hybrid model must fail during construction");
        assert!(
            error.to_string().contains("64"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "mwpf")]
    #[test]
    fn mwpf_accepts_64_and_rejects_65_observables() {
        assert_64_then_65_rejected(&DecoderSpec::Mwpf(MwpfConfig::default()));
    }

    #[test]
    fn resolves_beamsearch_defaults_from_step_size() {
        let DecoderSpec::BeamSearch(bare_config) = DecoderSpec::parse("beamsearch").unwrap() else {
            panic!("expected beamsearch spec");
        };
        let bare = resolve_beamsearch_config(&bare_config);
        assert_eq!(bare.buffer_size, 5);
        assert!((bare.commit_weight_max - 2.5).abs() < f64::EPSILON);

        let DecoderSpec::BeamSearch(step_config) = DecoderSpec::parse("beamsearch:step=3").unwrap()
        else {
            panic!("expected beamsearch spec");
        };
        assert_eq!(resolve_beamsearch_config(&step_config).buffer_size, 3);
    }

    #[test]
    fn model_family_mismatches_are_configuration_errors() {
        let hybrid = DecoderSpec::BeliefMatching(BeliefMatchingConfig {
            irreducible: IrreduciblePolicy::default(),
            mode: BeliefMatchingMode::Hybrid,
            embedded_full_dem: None,
        });
        assert!(matches!(
            hybrid.build(&DecodeModel::SingleDem(DEM.to_string())),
            Err(DecoderError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            DecoderSpec::PyMatching(PyMatchingConfig::default()).build(&DecodeModel::HybridDem {
                full: DEM.to_string(),
                decomposed: DEM.to_string(),
            }),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }

    #[cfg(not(feature = "mwpf"))]
    #[test]
    fn unavailable_backend_is_typed() {
        let error = DecoderSpec::Mwpf(MwpfConfig::default())
            .build(&DecodeModel::SingleDem(DEM.to_string()))
            .err()
            .expect("mwpf should be unavailable");
        assert!(matches!(
            error,
            DecoderError::BackendUnavailable {
                family: "mwpf",
                required_feature: "mwpf"
            }
        ));
    }

    #[cfg(not(any(
        feature = "pymatching",
        feature = "tesseract",
        feature = "fusion-blossom",
        feature = "ldpc",
        feature = "relay-bp",
        feature = "uf"
    )))]
    #[test]
    fn all_absent_backend_families_report_their_required_feature() {
        let cases = [
            (
                DecoderSpec::PyMatching(PyMatchingConfig::default()),
                "pymatching",
            ),
            (
                DecoderSpec::Tesseract(TesseractConfig::default()),
                "tesseract",
            ),
            (
                DecoderSpec::TesseractTrellis(TesseractTrellisConfig::default()),
                "tesseract",
            ),
            (
                DecoderSpec::FusionBlossom(FusionBlossomConfig::default()),
                "fusion-blossom",
            ),
            (DecoderSpec::BpOsd(BpOsdConfig::default()), "ldpc"),
            (DecoderSpec::RelayBp(RelayBpConfig::default()), "relay-bp"),
            (
                DecoderSpec::PecosUf(PecosUfConfig {
                    preset: PecosUfPreset::Fast,
                    ..Default::default()
                }),
                "uf",
            ),
        ];
        for (spec, expected_feature) in cases {
            let error = spec
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .err()
                .expect("backend should be unavailable");
            assert!(matches!(
                error,
                DecoderError::BackendUnavailable {
                    family,
                    required_feature,
                } if required_feature == expected_feature && family == family_name(&spec)
            ));
        }
    }

    #[cfg(feature = "tesseract")]
    #[test]
    fn trellis_spec_config_reaches_the_engine_field_by_field() {
        use super::super::config::TesseractTrellisRankingMode as SpecRanking;
        use crate::TesseractTrellisRankingMode as EngineRanking;

        // The spec defaults are the engine defaults, so an omitted option
        // builds the same decoder either way.
        assert_eq!(
            trellis_engine_config(&TesseractTrellisConfig::default()),
            crate::TesseractTrellisConfig::default()
        );

        let engine = trellis_engine_config(&TesseractTrellisConfig {
            beam_width: 7,
            beam_eps: 0.25,
            future_detcost_scale: 0.5,
            verbose: true,
            merge_errors: false,
            ranking_mode: SpecRanking::FutureActiveDetcostRanked,
        });
        assert_eq!(engine.beam_width, 7);
        assert_eq!(engine.beam_eps.to_bits(), 0.25_f64.to_bits());
        assert_eq!(engine.future_detcost_scale.to_bits(), 0.5_f64.to_bits());
        assert!(engine.verbose);
        assert!(!engine.merge_errors);
        assert_eq!(
            engine.ranking_mode,
            EngineRanking::FutureActiveDetcostRanked
        );

        for (spec, expected) in [
            (SpecRanking::MassOnly, EngineRanking::MassOnly),
            (
                SpecRanking::FutureDetcostRanked,
                EngineRanking::FutureDetcostRanked,
            ),
            (
                SpecRanking::FutureActiveDetcostRanked,
                EngineRanking::FutureActiveDetcostRanked,
            ),
        ] {
            let config = TesseractTrellisConfig {
                ranking_mode: spec,
                ..TesseractTrellisConfig::default()
            };
            assert_eq!(trellis_engine_config(&config).ranking_mode, expected);
        }
    }

    #[cfg(feature = "pymatching")]
    #[test]
    fn parse_build_and_decode_pymatching() {
        let mut decoder = DecoderSpec::parse("pymatching")
            .unwrap()
            .build(&DecodeModel::SingleDem(DEM.to_string()))
            .unwrap();
        assert!(decoder.decode_obs(&[0, 0]).is_ok());
        assert!(decoder.decode_obs(&[1, 1]).is_ok());

        for spec in [
            DecoderSpec::Perturbed(PerturbedConfig {
                k: 1,
                ..PerturbedConfig::default()
            }),
            DecoderSpec::Ensemble(EnsembleConfig {
                members: vec![
                    DecoderSpec::parse("pymatching").unwrap(),
                    DecoderSpec::parse("pymatching_uncorrelated").unwrap(),
                ],
            }),
        ] {
            let mut decoder = spec
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert!(decoder.decode_obs(&[0, 0]).is_ok());
        }
    }

    #[cfg(feature = "uf")]
    #[test]
    fn parse_build_and_decode_uf_family() {
        for family in ["pecos_uf", "pecos_uf:bp", "astar", "astar_full"] {
            let mut decoder = DecoderSpec::parse(family)
                .unwrap()
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert!(decoder.decode_obs(&[0, 0]).is_ok(), "failed {family}");
        }
    }

    #[cfg(feature = "fusion-blossom")]
    #[test]
    fn parse_build_and_decode_fusion_family() {
        for family in [
            "fusion_blossom_serial",
            "fusion_blossom_correlated",
            "k_mwpm:K=2",
            "perturbed_fb_corr:K=1",
        ] {
            let mut decoder = DecoderSpec::parse(family)
                .unwrap()
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert!(decoder.decode_obs(&[0, 0]).is_ok(), "failed {family}");
        }
    }

    #[cfg(feature = "fusion-blossom")]
    #[test]
    fn parse_build_and_decode_parallel_fusion_blossom() {
        const PARALLEL_DEM: &str = "error(0.05) D0 L0\n\
             error(0.05) D1 L0\n\
             detector(0, 0, 0) D0\n\
             detector(0, 0, 1) D1\n";
        let mut decoder = DecoderSpec::parse("fusion_blossom_parallel")
            .unwrap()
            .build(&DecodeModel::SingleDem(PARALLEL_DEM.to_string()))
            .unwrap();
        assert!(decoder.decode_obs(&[0, 0]).is_ok());
        assert!(decoder.decode_obs(&[1, 0]).is_ok());
    }

    #[cfg(feature = "tesseract")]
    #[test]
    fn parse_build_and_decode_tesseract() {
        for family in ["tesseract", "tesseract_trellis"] {
            let spec = DecoderSpec::parse(family).unwrap();
            assert_eq!(
                spec.execution_traits(),
                super::super::ExecutionTraits::default()
            );
            assert!(!spec.native_batch_capable());
            let mut decoder = spec
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert_eq!(decoder.decode_obs(&[0, 0]).unwrap().to_u64(), Some(0));
        }
    }

    #[cfg(feature = "ldpc")]
    #[test]
    fn parse_build_and_decode_ldpc_family() {
        for family in ["bp_osd", "bp_lsd", "belief_find", "union_find"] {
            let mut decoder = DecoderSpec::parse(family)
                .unwrap()
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert!(decoder.decode_obs(&[0, 0]).is_ok(), "failed {family}");
        }
    }

    #[cfg(feature = "relay-bp")]
    #[test]
    fn parse_build_and_decode_relay_family() {
        for family in ["relay_bp", "min_sum_bp"] {
            let mut decoder = DecoderSpec::parse(family)
                .unwrap()
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert!(decoder.decode_obs(&[0, 0]).is_ok(), "failed {family}");
        }
    }

    #[cfg(feature = "mwpf")]
    #[test]
    fn parse_build_and_decode_mwpf() {
        let mut decoder = DecoderSpec::parse("mwpf")
            .unwrap()
            .build(&DecodeModel::SingleDem(DEM.to_string()))
            .unwrap();
        assert!(decoder.decode_obs(&[0, 0]).is_ok());
    }

    #[cfg(all(feature = "uf", feature = "fusion-blossom"))]
    #[test]
    fn parse_build_and_decode_belief_matching_family() {
        for family in [
            "belief_matching",
            "belief_matching_correlated",
            "belief_matching_mgbp",
        ] {
            let mut decoder = DecoderSpec::parse(family)
                .unwrap()
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .unwrap();
            assert!(decoder.decode_obs(&[0, 0]).is_ok(), "failed {family}");
        }

        let hybrid = DecoderSpec::parse(&format!("belief_matching_hybrid:{DEM}")).unwrap();
        let mut decoder = hybrid
            .build(&DecodeModel::HybridDem {
                full: DEM.to_string(),
                decomposed: DEM.to_string(),
            })
            .unwrap();
        assert!(decoder.decode_obs(&[0, 0]).is_ok());
    }

    #[cfg(all(feature = "uf", feature = "fusion-blossom"))]
    #[test]
    fn hybrid_model_keeps_full_and_decomposed_projections_in_order() {
        const FULL: &str = "error(0.1) D0 L0\n";
        const DECOMPOSED: &str = "error(0.1) D0\n";
        let spec = DecoderSpec::parse(&format!("belief_matching_hybrid:{FULL}")).unwrap();

        let mut ordered = spec
            .build(&DecodeModel::HybridDem {
                full: FULL.to_string(),
                decomposed: DECOMPOSED.to_string(),
            })
            .unwrap();
        let mut swapped = spec
            .build(&DecodeModel::HybridDem {
                full: DECOMPOSED.to_string(),
                decomposed: FULL.to_string(),
            })
            .unwrap();

        assert_eq!(ordered.decode_obs(&[1]).unwrap().to_u64(), Some(0));
        assert_eq!(swapped.decode_obs(&[1]).unwrap().to_u64(), Some(1));
    }
}

#[cfg(test)]
std::thread_local! {
    static LEAF_MODELS: std::cell::RefCell<Vec<DecodeModel>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod projection_tests {
    use super::*;
    use crate::batch::DecoderFactory;
    #[cfg(feature = "uf")]
    use crate::batch::{DecodeOptions, ExecutionPath, SampleBatch};
    use pecos_decoder_core::graphlike_projection::project_graphlike;
    #[cfg(feature = "uf")]
    use pecos_decoder_core::obs_mask::ObsMask;

    const GRAPH_NAMES: &[&str] = &[
        "pymatching",
        "pymatching_correlated",
        "pymatching_uncorrelated",
        "fusion_blossom",
        "fusion_blossom_serial",
        "fusion_blossom_parallel",
        "fusion_blossom_correlated",
        "perturbed_fb_corr",
        "astar",
        "k_mwpm",
        "pecos_uf",
        "pecos_uf:balanced",
        "pecos_uf:accurate",
        "pecos_uf:bp",
        "pecos_uf:bp_serial",
        "belief_matching",
        "belief_matching_correlated",
        "belief_matching_mgbp",
    ];
    const RAW: &str = "error(0.12345678912345678) D0 D1 D2 L0\nerror(0.02) D0 D1 L0\nerror(0.03) D2\nerror(0.04) D0\nerror(0.05) D1\n";
    const IRREDUCIBLE: &str =
        "error(0.1) D0 D1 D2 L0\nerror(0.02) D0\nerror(0.03) D1\nerror(0.04) D2\n";
    const DECOMPOSED: &str = "error(0.12345678912345678) D0 D1 L0 ^ D2\nerror(0.02) D0 D1 L0\nerror(0.03) D2\nerror(0.04) D0\nerror(0.05) D1\n";

    fn with_drop(name: &str) -> DecoderSpec {
        DecoderSpec::parse(&format!(
            "{name}{}irreducible=drop",
            if name.contains(':') { ',' } else { ':' }
        ))
        .unwrap()
    }

    fn built(spec: &DecoderSpec, model: &DecodeModel) -> Option<Box<dyn ObservableDecoder>> {
        match spec.build(model) {
            Ok(decoder) => Some(decoder),
            Err(DecoderError::BackendUnavailable { .. }) => None,
            Err(error) => panic!("{spec:?}: {error}"),
        }
    }

    fn predictions_equal(
        mut left: Box<dyn ObservableDecoder>,
        mut right: Box<dyn ObservableDecoder>,
        width: usize,
    ) {
        assert_eq!(left.num_detectors(), right.num_detectors());
        for bits in 0..1usize << width {
            let syndrome: Vec<_> = (0..width)
                .map(|bit| u8::from(bits & (1 << bit) != 0))
                .collect();
            assert_eq!(
                left.decode_obs(&syndrome).unwrap(),
                right.decode_obs(&syndrome).unwrap(),
                "syndrome {syndrome:?}"
            );
        }
    }

    // contract: every graph leaf projects reducible raw input and honors explicit loss policy.
    #[test]
    fn every_graph_leaf_projects_or_refuses() {
        let raw = StructuredDem::from_dem_str(RAW).unwrap();
        let projected = project_graphlike(&raw, IrreduciblePolicy::Refuse)
            .unwrap()
            .model;
        let explicit = DecodeModel::SingleDem(projected.to_dem_string());
        for name in GRAPH_NAMES {
            let spec = DecoderSpec::parse(name).unwrap();
            assert!(spec.requires_graphlike_model());
            for input in [
                DecodeModel::SingleDem(RAW.into()),
                DecodeModel::StructuredDem(raw.clone()),
            ] {
                LEAF_MODELS.with(|models| models.borrow_mut().clear());
                let decoder = built(&spec, &input);
                // Inspect the actual leaf boundary before any backend text rendering.
                let prepared = LEAF_MODELS.with(|models| models.borrow().last().unwrap().clone());
                assert_eq!(prepared, DecodeModel::StructuredDem(projected.clone()));
                let DecodeModel::StructuredDem(model) = prepared else {
                    unreachable!()
                };
                for (retained, original) in model.errors.iter().zip(&raw.errors) {
                    assert_eq!(
                        retained.probability.to_bits(),
                        original.probability.to_bits()
                    );
                }
                assert_eq!(model.errors[0].components.len(), 2);
                assert_eq!((model.num_detectors, model.num_observables), (3, 1));
                assert_eq!(model.detector_fault_sectors, raw.detector_fault_sectors);
                assert_eq!(
                    StructuredDem::from_dem_str(&model.to_dem_string()).unwrap(),
                    model
                );
                if let Some(decoder) = decoder {
                    predictions_equal(decoder, built(&spec, &explicit).unwrap(), 3);
                }
            }
            let irreducible = DecodeModel::SingleDem(IRREDUCIBLE.into());
            let error = spec.build(&irreducible).err().unwrap().to_string();
            assert!(
                error.starts_with(&format!("Invalid configuration: {}:", family_name(&spec))),
                "{error}"
            );
            for message in [
                "1 irreducible errors",
                "[0]",
                "irreducible=drop",
                "bp_osd",
                "tesseract",
            ] {
                assert!(error.contains(message), "{name}: {error}");
            }
            let drop = with_drop(name);
            let prepared = drop.decode_model(IRREDUCIBLE).unwrap();
            assert_eq!(prepared.projection.unwrap().dropped_indices, vec![0]);
            let mut original = StructuredDem::from_dem_str(IRREDUCIBLE).unwrap();
            original.detector_fault_sectors = vec![Some("X".into()), Some("Z".into()), None];
            built(&drop, &DecodeModel::StructuredDem(original.clone()));
            let actual = LEAF_MODELS.with(|models| models.borrow().last().unwrap().clone());
            let DecodeModel::StructuredDem(actual) = actual else {
                unreachable!()
            };
            assert_eq!(actual.errors, original.errors[1..]);
            assert_eq!(actual.num_detectors, original.num_detectors);
            assert_eq!(actual.num_observables, original.num_observables);
            assert_eq!(
                actual.detector_fault_sectors,
                original.detector_fault_sectors
            );
            for (retained, original) in actual.errors.iter().zip(&original.errors[1..]) {
                assert_eq!(
                    retained.probability.to_bits(),
                    original.probability.to_bits()
                );
            }
            let supplied = DecodeModel::SingleDem(DECOMPOSED.into());
            let prepared = prepare_model(&spec, &supplied).unwrap();
            assert_eq!(prepared.model, supplied);
            assert_eq!(prepared.projection, None);
            let decoder = built(&spec, &supplied);
            assert_eq!(
                LEAF_MODELS.with(|models| models.borrow().last().unwrap().clone()),
                supplied
            );
            if let Some(decoder) = decoder {
                predictions_equal(decoder, built(&spec, &explicit).unwrap(), 3);
            }
            let mixed = DecodeModel::SingleDem(format!("{DECOMPOSED}error(0.01) D0 D1 D2\n"));
            assert!(
                spec.build(&mixed)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("mixes decomposed and raw hyperedges")
            );
        }
    }

    // contract: classification is structural and identical for text and structured input.
    #[test]
    fn structural_classification_parity() {
        let spec = DecoderSpec::parse("astar").unwrap();
        for text in [
            RAW,
            DECOMPOSED,
            IRREDUCIBLE,
            "error(0) D0 D1 D2",
            "error(0.1) D0 D1 D2 ^ D0 D1 ^ D2",
            "error(0.1) D0 ^ D1\nerror(0) D0 D1 D2",
            "error(0.1) D0 D1 D2 ^ D3",
            "error(0.01) D0 L0",
        ] {
            let text_model = DecodeModel::SingleDem(text.into());
            let structured_model =
                DecodeModel::StructuredDem(StructuredDem::from_dem_str(text).unwrap());
            match (
                prepare_model(&spec, &text_model),
                prepare_model(&spec, &structured_model),
            ) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.projection, b.projection);
                    if a.projection.is_some() {
                        assert_eq!(a.model, b.model);
                    } else {
                        assert_eq!(a.model, text_model);
                        assert_eq!(b.model, structured_model);
                    }
                }
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                other => panic!("classification differs: {other:?}"),
            }
        }
        let zero = spec.decode_model("error(0) D0 D1 D2").unwrap();
        assert_eq!(zero.projection.unwrap().zero_probability_omitted, vec![0]);
        let error = spec
            .decode_model("error(0.1) D0 D1 D2 ^ D0 D1 ^ D2")
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("component(s) touch more than two detectors")
        );
    }

    // compatibility: Hybrid requires two models and never applies a raw projection policy.
    #[test]
    fn hybrid_is_exempt() {
        let mut spec = DecoderSpec::parse("belief_matching_hybrid:error(0.1) D0 D1 D2 L0").unwrap();
        assert!(
            spec.build(&DecodeModel::SingleDem(RAW.into()))
                .err()
                .unwrap()
                .to_string()
                .contains("requires DecodeModel::HybridDem")
        );
        let model = DecodeModel::HybridDem {
            full: IRREDUCIBLE.into(),
            decomposed: DECOMPOSED.into(),
        };
        assert_eq!(prepare_model(&spec, &model).unwrap().projection, None);
        built(&spec, &model);
        let bad = DecodeModel::HybridDem {
            full: RAW.into(),
            decomposed: RAW.into(),
        };
        let error = spec.build(&bad).err().unwrap().to_string();
        assert!(error.contains("graphlike"));
        assert!(!error.contains("irreducible=drop"));
        let DecoderSpec::BeliefMatching(config) = &mut spec else {
            unreachable!()
        };
        config.irreducible = IrreduciblePolicy::Drop;
        assert!(
            spec.build(&model)
                .err()
                .unwrap()
                .to_string()
                .contains("does not support an irreducible policy")
        );
    }

    // contract: projection error variants retain their precise diagnostics, even under Drop.
    #[test]
    fn projection_failures_keep_their_messages() {
        for text in [
            "error(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D1 D3\nerror(0.2) D0 D1 D2 D3",
            "error(0.1) D0 D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12",
        ] {
            let model = StructuredDem::from_dem_str(text).unwrap();
            let expected = project_graphlike(&model, IrreduciblePolicy::Drop)
                .unwrap_err()
                .to_string();
            let actual = with_drop("astar")
                .decode_model(text)
                .unwrap_err()
                .to_string();
            assert!(
                actual.starts_with("Invalid configuration: astar:"),
                "{actual}"
            );
            assert!(actual.contains(&expected), "{actual}: {expected}");
            for spec in [DecoderSpec::parse("astar").unwrap(), with_drop("astar")] {
                let error = spec
                    .build(&DecodeModel::SingleDem(text.into()))
                    .err()
                    .unwrap()
                    .to_string();
                for remedy in [
                    "irreducible=drop does not apply to this effect",
                    "bp_osd",
                    "tesseract",
                ] {
                    assert!(error.contains(remedy), "{error}");
                }
            }
        }
        let mut invalid = StructuredDem::from_dem_str(RAW).unwrap();
        invalid.errors[0].probability = f64::NAN;
        let expected = project_graphlike(&invalid, IrreduciblePolicy::Drop)
            .unwrap_err()
            .to_string();
        assert!(
            prepare_model(&with_drop("astar"), &DecodeModel::StructuredDem(invalid))
                .unwrap_err()
                .to_string()
                .contains(&expected)
        );
    }

    #[cfg(all(feature = "uf", feature = "ldpc"))]
    #[test]
    fn perturbed_mixed_ensemble_preserves_zero_hyperedges() {
        let spec = DecoderSpec::Perturbed(PerturbedConfig {
            k: 3,
            inner: Box::new(DecoderSpec::parse("ensemble:astar,bp_osd").unwrap()),
            ..Default::default()
        });
        assert_eq!(spec.irreducible_policy(), None);
        let text = IRREDUCIBLE.replace("error(0.1)", "error(0)");
        for model in [
            DecodeModel::SingleDem(text.clone()),
            DecodeModel::StructuredDem(StructuredDem::from_dem_str(&text).unwrap()),
        ] {
            let mut decoder = spec.build(&model).unwrap();
            for bits in 0..8 {
                let syndrome: Vec<_> = (0..3).map(|bit| u8::from(bits & (1 << bit) != 0)).collect();
                assert_eq!(
                    decoder.decode_obs(&syndrome).unwrap(),
                    pecos_decoder_core::obs_mask::ObsMask::new()
                );
            }
        }
    }

    // contract: sector-sensitive choices survive Ensemble and probability perturbation.
    #[cfg(feature = "uf")]
    #[test]
    fn composites_preserve_sector_choices() {
        let mut model = StructuredDem::from_dem_str("error(0.01) D0 D1 L0\nerror(0.01) D2 D3\nerror(0.4) D0 D2 L0\nerror(0.4) D1 D3\nerror(0.123456789) D3 D1 D0 D2 L0\nerror(0.001) D0\nerror(0.001) D1\nerror(0.001) D2\nerror(0.001) D3").unwrap();
        let without_sectors = project_graphlike(&model, IrreduciblePolicy::Refuse)
            .unwrap()
            .model;
        model.detector_fault_sectors = vec![
            Some("X".into()),
            Some("X".into()),
            Some("Z".into()),
            Some("Z".into()),
        ];
        let projected = project_graphlike(&model, IrreduciblePolicy::Refuse)
            .unwrap()
            .model;
        assert_ne!(
            projected.errors[4].components,
            without_sectors.errors[4].components
        );
        let leaf = DecoderSpec::parse("astar").unwrap();
        for (composite, expected_count) in [
            ("ensemble:astar", 1),
            ("perturbed:k=3,sigma=1.5,inner=astar", 3),
        ] {
            LEAF_MODELS.with(|models| models.borrow_mut().clear());
            let spec = DecoderSpec::parse(composite).unwrap();
            let actual = built(&spec, &DecodeModel::StructuredDem(model.clone()));
            let members = LEAF_MODELS.with(|models| models.borrow().clone());
            assert_eq!(members.len(), expected_count, "{composite}");
            for member in members {
                let DecodeModel::StructuredDem(member) = member else {
                    panic!("lost structured transport")
                };
                assert_eq!(member.detector_fault_sectors, model.detector_fault_sectors);
                assert_eq!(member.errors[4].components, projected.errors[4].components);
            }
            if let Some(actual) = actual {
                let (expected_spec, expected_model) = if composite.starts_with("ensemble") {
                    (&leaf, &model)
                } else {
                    (&spec, &projected)
                };
                let expected = built(
                    expected_spec,
                    &DecodeModel::StructuredDem(expected_model.clone()),
                )
                .unwrap();
                predictions_equal(actual, expected, 4);
            }
        }
    }

    // contract: one projection before parallel workers, with report and direct/batch agreement.
    #[cfg(feature = "uf")]
    #[test]
    fn parallel_preparation_projects_once_per_call() {
        use crate::spec::prepare::PROJECTION_INPUTS;
        for (name, probability) in [
            ("astar", "0.123456780001"),
            ("perturbed:k=3,sigma=0.9,inner=astar", "0.123456780002"),
        ] {
            let text = RAW.replace("0.12345678912345678", probability);
            let raw = StructuredDem::from_dem_str(&text).unwrap();
            let count = || {
                PROJECTION_INPUTS
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|model| **model == raw)
                    .count()
            };
            let spec = DecoderSpec::parse(name).unwrap();
            let batch = SampleBatch::from_row_major(
                &vec![vec![1u8, 1, 1]; 16],
                &vec![ObsMask::new(); 16],
                1,
            )
            .unwrap();
            let before = count();
            let result = batch
                .decode_with(
                    &text,
                    &spec,
                    &DecodeOptions::default().workers(2).predictions(true),
                )
                .unwrap();
            assert_eq!(result.execution_path, ExecutionPath::Parallel);
            assert_eq!(result.workers_used, 2);
            assert_eq!(count() - before, 1);
            assert_eq!(result.projection.unwrap().projected, 1);
            let mut direct = spec.build(&DecodeModel::SingleDem(text)).unwrap();
            assert_eq!(
                result.predictions.unwrap(),
                vec![direct.decode_obs(&[1, 1, 1]).unwrap(); 16]
            );
        }
    }

    // contract: the inner policy applies to the full model before window construction.
    #[cfg(all(feature = "uf", feature = "pymatching"))]
    #[test]
    fn windowed_projects_full_model_with_inner_policy() {
        let coords = "detector(0,0,0) D0\ndetector(1,0,0) D1\ndetector(2,0,0) D2\n";
        let spec = DecoderSpec::parse("windowed:step=1,buf=1,inner=pymatching").unwrap();
        let text = format!("{RAW}{coords}");
        let prepared = spec.decode_model(&text).unwrap();
        assert_eq!(prepared.projection.unwrap().projected, 1);
        predictions_equal(
            spec.build(&DecodeModel::SingleDem(text)).unwrap(),
            spec.build(&prepared.model).unwrap(),
            3,
        );
        let text = format!("{IRREDUCIBLE}{coords}");
        assert!(
            spec.build(&DecodeModel::SingleDem(text.clone()))
                .err()
                .unwrap()
                .to_string()
                .contains("irreducible=drop")
        );
        let drop =
            DecoderSpec::parse("windowed:step=1,buf=1,inner=pymatching:irreducible=drop").unwrap();
        let prepared = drop.decode_model(&text).unwrap();
        assert_eq!(prepared.projection.unwrap().dropped_indices, vec![0]);
        predictions_equal(
            drop.build(&DecodeModel::SingleDem(text)).unwrap(),
            drop.build(&prepared.model).unwrap(),
            3,
        );
    }

    // compatibility: beam phase 1 still refuses raw hyperedges and offers no policy option.
    #[cfg(feature = "uf")]
    #[test]
    fn beam_phase_one_still_refuses() {
        let text = format!("{RAW}detector(0,0,0) D0\ndetector(1,0,0) D1\ndetector(2,0,0) D2\n");
        let spec = DecoderSpec::parse("beamsearch:step=1,buf=1").unwrap();
        let error = spec
            .build(&DecodeModel::SingleDem(text))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("graphlike"), "{error}");
        assert!(!error.contains("irreducible=drop"));
    }
    const TIE: &str = "error(0.1) D0 D1 L0\nerror(0.1) D2 D3\nerror(0.1) D0 D2 L0\nerror(0.1) D1 D3\nerror(0.123456781234) D0 D1 D2 D3 L0\nerror(0.001) D0\nerror(0.001) D1\nerror(0.001) D2\nerror(0.001) D3\n";

    // contract: recursive consensus includes nested Perturbed, Windowed, and graph ensembles.
    #[test]
    fn recursive_policy_resolution() {
        for name in [
            "perturbed:inner=ensemble:astar,pecos_uf",
            "perturbed:inner=perturbed:inner=astar",
            "perturbed:inner=windowed:step=1,buf=1,inner=pecos_uf",
            "ensemble:astar,perturbed:inner=astar",
        ] {
            let spec = DecoderSpec::parse(name).unwrap();
            assert_eq!(spec.irreducible_policy(), Some(IrreduciblePolicy::Refuse));
            assert_eq!(
                spec.decode_model(TIE)
                    .unwrap()
                    .projection
                    .unwrap()
                    .projected,
                1
            );
        }
        for name in ["ensemble:astar,astar_full", "ensemble:astar_full,astar"] {
            let spec = DecoderSpec::parse(name).unwrap();
            assert_eq!(spec.irreducible_policy(), None);
            assert_eq!(spec.decode_model(TIE).unwrap().projection, None);
        }
        assert_eq!(
            DecoderSpec::Ensemble(EnsembleConfig::default()).irreducible_policy(),
            None
        );
        let drop = DecoderSpec::parse(
            "perturbed:inner=ensemble:astar:irreducible=drop,pecos_uf:irreducible=drop",
        )
        .unwrap();
        assert_eq!(drop.irreducible_policy(), Some(IrreduciblePolicy::Drop));
        assert_eq!(
            drop.decode_model(IRREDUCIBLE)
                .unwrap()
                .projection
                .unwrap()
                .dropped_indices,
            vec![0]
        );
    }

    // contract: compositions without a shared report must never hide dropped faults.
    #[test]
    fn mixed_compositions_refuse_drop() {
        for name in [
            "ensemble:astar,astar:irreducible=drop",
            "ensemble:astar_full,astar:irreducible=drop",
            "perturbed:inner=ensemble:astar_full,astar:irreducible=drop",
            "ensemble:astar_full,perturbed:inner=astar:irreducible=drop",
            "ensemble:astar_full,ensemble:astar:irreducible=drop,astar:irreducible=drop",
        ] {
            let spec = DecoderSpec::parse(name).unwrap();
            let message = spec.decode_model(TIE).unwrap_err().to_string();
            assert!(message.contains("astar ("), "{message}");
            assert!(message.contains("ensemble member"), "{message}");
            assert!(
                message.contains(
                    "drop inside such a composition needs every graph member to share the policy"
                ),
                "{message}"
            );
        }
    }

    // compatibility: Refuse compositions remain lossless and may prepare members individually.
    #[test]
    fn mixed_compositions_accept_refuse() {
        let spec = DecoderSpec::parse("ensemble:astar,astar_full").unwrap();
        assert_eq!(spec.decode_model(TIE).unwrap().projection, None);
        spec.build(&DecodeModel::SingleDem(TIE.into())).unwrap();
    }

    // contract: shared policies choose one decomposition before any nested copy is perturbed.
    #[cfg(feature = "uf")]
    #[test]
    fn nested_perturbed_shares_one_projection() {
        for inner in [
            "ensemble:astar,astar",
            "perturbed:k=2,sigma=1.5,inner=astar",
        ] {
            let spec =
                DecoderSpec::parse(&format!("perturbed:k=8,sigma=1.5,inner={inner}")).unwrap();
            let prepared = spec.decode_model(TIE).unwrap();
            assert_eq!(prepared.projection.unwrap().projected, 1);
            let DecodeModel::StructuredDem(projected) = prepared.model else {
                unreachable!()
            };
            LEAF_MODELS.with(|models| models.borrow_mut().clear());
            spec.build(&DecodeModel::SingleDem(TIE.into())).unwrap();
            let members = LEAF_MODELS.with(|models| models.borrow().clone());
            assert_eq!(members.len(), 16);
            for member in members {
                let DecodeModel::StructuredDem(member) = member else {
                    unreachable!()
                };
                assert_eq!(member.errors[4].components, projected.errors[4].components);
            }
        }
    }

    // contract: lossless mixed ensembles project each graph member after perturbation.
    #[cfg(feature = "uf")]
    #[test]
    fn mixed_perturbed_ensemble_projects_individually() {
        for (inner, count) in [
            ("ensemble:astar,astar_full", 8),
            ("ensemble:astar_full,astar", 8),
        ] {
            let spec =
                DecoderSpec::parse(&format!("perturbed:k=8,sigma=1.5,inner={inner}")).unwrap();
            assert_eq!(spec.decode_model(TIE).unwrap().projection, None);
            LEAF_MODELS.with(|models| models.borrow_mut().clear());
            spec.build(&DecodeModel::SingleDem(TIE.into())).unwrap();
            let members = LEAF_MODELS.with(|models| models.borrow().clone());
            assert_eq!(members.len(), count);
            let choices: std::collections::BTreeSet<_> = members
                .into_iter()
                .map(|member| {
                    let DecodeModel::StructuredDem(member) = member else {
                        unreachable!()
                    };
                    member.errors[4]
                        .components
                        .iter()
                        .map(|component| component.detectors.clone())
                        .collect::<Vec<_>>()
                })
                .collect();
            assert!(
                choices.len() > 1,
                "fixture must expose probability-dependent choices"
            );
        }
    }

    // contract: remedies identify the leaf, its member/inner location, and where to set policy.
    #[test]
    fn projection_errors_name_decoder_and_location() {
        for (name, leaf, location, example) in [
            (
                "windowed:step=1,buf=1,inner=pymatching",
                "pymatching",
                "windowed inner decoder",
                "windowed:...,inner=pymatching:irreducible=drop",
            ),
            (
                "perturbed:inner=astar",
                "astar",
                "perturbed inner decoder",
                "perturbed:...,inner=astar:irreducible=drop",
            ),
            (
                "ensemble:astar,pecos_uf",
                "astar",
                "ensemble member 0",
                "drop needs every graph member to share one policy",
            ),
            (
                "ensemble:astar,astar_full",
                "astar",
                "ensemble member 0",
                "drop needs every graph member to share one policy",
            ),
            (
                "perturbed:inner=ensemble:astar,astar_full",
                "astar",
                "ensemble member 0",
                "drop needs every graph member to share one policy",
            ),
        ] {
            let spec = DecoderSpec::parse(name).unwrap();
            let error = spec
                .build(&DecodeModel::SingleDem(IRREDUCIBLE.into()))
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.starts_with(&format!("Invalid configuration: {leaf}")),
                "{error}"
            );
            for text in [location, example, "1 irreducible errors", "[0]"] {
                assert!(error.contains(text), "{name}: {error}");
            }
        }
    }

    // contract: a composite ensemble member needs typed advice when its parameters contain commas.
    #[test]
    fn nested_member_remedy_uses_typed_ensemble() {
        let spec = DecoderSpec::Ensemble(EnsembleConfig {
            members: vec![
                DecoderSpec::parse("perturbed:k=8,inner=astar").unwrap(),
                DecoderSpec::parse("astar_full").unwrap(),
            ],
        });
        let message = spec
            .build(&DecodeModel::SingleDem(IRREDUCIBLE.into()))
            .err()
            .unwrap()
            .to_string();
        for part in [
            "astar",
            "ensemble member 0",
            "perturbed inner decoder",
            "drop needs every graph member to share one policy",
            "mixing graph and hypergraph members cannot do this",
            "ensemble:astar:irreducible=drop,pecos_uf:irreducible=drop",
            "ensemble(perturbed(inner=astar(irreducible=\"drop\")), astar(irreducible=\"drop\"))",
            "hyperedge-capable decoder",
            "typed EnsembleConfig.members",
            "pecos_rslib.decoders.ensemble(...)",
            "irreducible=drop",
        ] {
            assert!(message.contains(part), "{message}");
        }
    }

    // contract: unsupported Windowed inners fail before projection, even beneath composites.
    #[test]
    fn windowed_support_precedes_projection() {
        for prefix in ["", "perturbed:inner=", "ensemble:"] {
            for policy in ["", ",irreducible=drop"] {
                let spec = if prefix == "ensemble:" {
                    DecoderSpec::Ensemble(EnsembleConfig {
                        members: vec![
                            DecoderSpec::parse(&format!(
                                "windowed:step=1,buf=1,inner=pecos_uf:balanced{policy}"
                            ))
                            .unwrap(),
                        ],
                    })
                } else {
                    DecoderSpec::parse(&format!(
                        "{prefix}windowed:step=1,buf=1,inner=pecos_uf:balanced{policy}"
                    ))
                    .unwrap()
                };
                let errors = [
                    spec.decode_model(IRREDUCIBLE).unwrap_err(),
                    spec.build(&DecodeModel::SingleDem(IRREDUCIBLE.into()))
                        .err()
                        .unwrap(),
                ];
                for error in errors {
                    let message = error.to_string();
                    assert!(
                        message.contains("windowed does not support inner"),
                        "{message}"
                    );
                    assert!(!message.contains("1 irreducible errors"));
                    assert!(!message.contains("irreducible=drop"));
                }
            }
        }
    }

    // contract: graphlike phase-1 slices can leave raw hyperedges for phase 2 to project.
    #[cfg(feature = "uf")]
    #[test]
    fn beam_phase_two_projects_and_names_typed_policy() {
        let coords = "detector(0,0,0) D0\ndetector(0,0,3) D1\ndetector(0,0,6) D2\n";
        let mut spec = DecoderSpec::parse("beamsearch:k=1,step=1,buf=1").unwrap();
        let DecoderSpec::BeamSearch(config) = &mut spec else {
            unreachable!()
        };
        *config.phase2 = DecoderSpec::parse("astar").unwrap();
        let mut decoder = spec
            .build(&DecodeModel::SingleDem(format!("{RAW}{coords}")))
            .unwrap();
        assert!(decoder.decode_obs(&[0, 0, 0]).unwrap().is_zero());
        let text = format!("{IRREDUCIBLE}{coords}");
        let error = spec
            .build(&DecodeModel::SingleDem(text.clone()))
            .err()
            .unwrap()
            .to_string();
        for part in [
            "astar",
            "beamsearch phase-2 decoder",
            "typed spec",
            "BeamSearchConfig.phase2",
            "pecos_rslib.decoders.beamsearch(phase2=...)",
            "no phase2 key",
            "1 irreducible errors",
        ] {
            assert!(error.contains(part), "{error}");
        }
        let DecoderSpec::BeamSearch(config) = &mut spec else {
            unreachable!()
        };
        *config.phase2 = with_drop("astar");
        let error = spec
            .build(&DecodeModel::SingleDem(text))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("beamsearch phase-2 decoder"));
        assert!(error.contains("every graph member to share the policy"));
        assert!(DecoderSpec::parse("beamsearch:phase2=astar").is_err());
    }
}
