use super::config::{
    BeamSearchConfig, BeliefMatchingConfig, BeliefMatchingMode, BpLsdConfig, BpOsdConfig,
    EnsembleConfig, FusionBlossomConfig, KMwpmConfig, MinSumBpConfig, MwpfConfig, PecosUfPreset,
    PerturbedConfig, PerturbedFusionBlossomConfig, PyMatchingConfig, RelayBpConfig,
    TesseractConfig, WindowedConfig,
};
use super::{DecodeModel, DecoderSpec};
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
#[cfg(any(feature = "uf", test))]
use super::config::WindowedMode;
#[cfg(feature = "ldpc")]
use crate::BpSchedule as LdpcBpSchedule;
#[cfg(feature = "tesseract")]
use crate::TesseractConfig as TesseractEngineConfig;
#[cfg(feature = "uf")]
use crate::{BeamSearchConfig as BeamSearchEngineConfig, WindowedConfig as WindowedEngineConfig};
#[cfg(feature = "fusion-blossom")]
use crate::{
    FusionBlossomConfig as FusionBlossomEngineConfig, SolverType as FusionBlossomEngineSolverType,
};
#[cfg(feature = "mwpf")]
use crate::{MwpfConfig as MwpfEngineConfig, MwpfSolverType as MwpfEngineSolverType};

pub(super) fn build(
    spec: &DecoderSpec,
    model: &DecodeModel,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let decoder = match (spec, model) {
        (
            DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                ..
            }),
            DecodeModel::HybridDem { full, decomposed },
        ) => build_belief_matching_hybrid(full, decomposed),
        (
            DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                ..
            }),
            DecodeModel::SingleDem(_),
        ) => Err(DecoderError::InvalidConfiguration(
            "belief_matching_hybrid requires DecodeModel::HybridDem".to_string(),
        )),
        (_, DecodeModel::SingleDem(dem)) => build_single(spec, dem),
        (_, DecodeModel::HybridDem { .. }) => Err(DecoderError::InvalidConfiguration(format!(
            "{} requires DecodeModel::SingleDem",
            family_name(spec)
        ))),
    }?;
    let dimension_dem = match model {
        DecodeModel::SingleDem(dem) => dem,
        DecodeModel::HybridDem { decomposed, .. } => decomposed,
    };
    let (num_detectors, _) = pecos_decoder_core::dem::utils::parse_dem_metadata(dimension_dem)?;
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

fn build_single(spec: &DecoderSpec, dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    match spec {
        DecoderSpec::PyMatching(config) => build_pymatching(dem, config),
        DecoderSpec::Tesseract(config) => build_tesseract(dem, config),
        DecoderSpec::KMwpm(config) => build_k_mwpm(dem, *config),
        DecoderSpec::AStar => build_astar(dem, false),
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
        DecoderSpec::PecosUf(preset) => build_pecos_uf(dem, *preset),
        DecoderSpec::BeliefMatching(config) => build_belief_matching(dem, config),
        DecoderSpec::Windowed(config) => build_windowed(dem, config),
        DecoderSpec::Mwpf(config) => build_mwpf(dem, config),
        DecoderSpec::Perturbed(config) => build_perturbed(dem, config),
        DecoderSpec::BeamSearch(config) => build_beamsearch(dem, config),
        DecoderSpec::Ensemble(config) => build_ensemble(dem, config),
    }
}

fn family_name(spec: &DecoderSpec) -> &'static str {
    match spec {
        DecoderSpec::PyMatching(_) => "pymatching",
        DecoderSpec::Tesseract(_) => "tesseract",
        DecoderSpec::KMwpm(_) => "k_mwpm",
        DecoderSpec::AStar => "astar",
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedWindowedMode {
    Sandwich,
    Overlap,
    NonOverlapping,
}

#[cfg(any(feature = "uf", test))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedWindowedConfig {
    mode: ResolvedWindowedMode,
    buffer_size: usize,
    commit_weight_max: f64,
}

#[cfg(any(feature = "uf", test))]
fn resolve_windowed_config(config: &WindowedConfig) -> ResolvedWindowedConfig {
    let mode = match config.mode {
        WindowedMode::Auto if config.buffer_size > 0 => ResolvedWindowedMode::Sandwich,
        WindowedMode::Auto | WindowedMode::NonOverlapping => ResolvedWindowedMode::NonOverlapping,
        WindowedMode::Sandwich => ResolvedWindowedMode::Sandwich,
        WindowedMode::Overlap => ResolvedWindowedMode::Overlap,
    };
    let (buffer_size, commit_weight_max) = if mode == ResolvedWindowedMode::Sandwich {
        (
            if config.buffer_size == 0 {
                config.step_size
            } else {
                config.buffer_size
            },
            if config.commit_weight_max == 0.0 {
                2.5
            } else {
                config.commit_weight_max
            },
        )
    } else {
        (config.buffer_size, config.commit_weight_max)
    };
    ResolvedWindowedConfig {
        mode,
        buffer_size,
        commit_weight_max,
    }
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
    .map_err(internal)?;
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
fn build_tesseract(
    dem: &str,
    config: &TesseractConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let mut engine_config = match config.preset {
        TesseractPreset::Default => TesseractEngineConfig::default(),
        TesseractPreset::Fast => TesseractEngineConfig::fast(),
        TesseractPreset::Accurate => TesseractEngineConfig::accurate(),
    };
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
        crate::TesseractDecoder::new(dem, engine_config).map_err(internal)?,
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
    let resolved = resolve_windowed_config(config);
    let window = WindowedEngineConfig {
        step_size: config.step_size,
        buffer_size: resolved.buffer_size,
        seam_half_width: config.seam_half_width,
        core_extend: config.core_extend,
        commit_weight_max: resolved.commit_weight_max,
    };
    match resolved.mode {
        ResolvedWindowedMode::Sandwich => {
            let phase2 = (*config.sandwich_phase2).clone();
            let decoder = crate::SandwichWindowedDecoder::from_dem(
                dem,
                window,
                |sub_dem| crate::UfDecoder::from_dem(sub_dem, crate::UfDecoderConfig::windowed()),
                |sub_dem| phase2.build(&DecodeModel::SingleDem(sub_dem.to_string())),
            )?;
            Ok(Box::new(decoder))
        }
        ResolvedWindowedMode::Overlap => Ok(Box::new(crate::OverlappingWindowedDecoder::from_dem(
            dem,
            window,
            |sub_dem| crate::UfDecoder::from_dem(sub_dem, crate::UfDecoderConfig::windowed()),
        )?)),
        ResolvedWindowedMode::NonOverlapping => {
            let inner = config.inner.clone();
            Ok(Box::new(crate::WindowedDecoder::from_dem(
                dem,
                window,
                |sub_dem| inner.build(&DecodeModel::SingleDem(sub_dem.to_string())),
            )?))
        }
    }
}

#[cfg(not(feature = "uf"))]
fn build_windowed(
    _dem: &str,
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
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    use pecos_decoder_core::perturbed::{PerturbedConfig as CoreConfig, build_perturbed_ensemble};

    let core_config = CoreConfig {
        k: config.k,
        sigma: config.sigma,
        seed: config.seed,
    };
    let inner = config.inner.clone();
    let decoder = build_perturbed_ensemble(dem, &core_config, |member_dem| {
        inner.build(&DecodeModel::SingleDem(member_dem.to_string()))
    })?;
    Ok(Box::new(decoder))
}

#[cfg(feature = "uf")]
fn build_beamsearch(
    dem: &str,
    config: &BeamSearchConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    let resolved = resolve_beamsearch_config(config);
    let engine_config = BeamSearchEngineConfig {
        beam_width: config.beam_width,
        perturbation_sigma: config.perturbation_sigma,
        seed: config.seed,
        window: WindowedEngineConfig {
            step_size: config.step_size,
            buffer_size: resolved.buffer_size,
            seam_half_width: 0,
            core_extend: 0,
            commit_weight_max: resolved.commit_weight_max,
        },
    };
    let phase2 = config.phase2.clone();
    let decoder = crate::BeamSearchWindowedDecoder::from_dem(
        dem,
        engine_config,
        |sub_dem| crate::UfDecoder::from_dem(sub_dem, crate::UfDecoderConfig::windowed()),
        Some(|sub_dem: &str| phase2.build(&DecodeModel::SingleDem(sub_dem.to_string()))),
    )?;
    Ok(Box::new(decoder))
}

#[cfg(not(feature = "uf"))]
fn build_beamsearch(
    _dem: &str,
    _config: &BeamSearchConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    unavailable("beamsearch", "uf")
}

fn build_ensemble(
    dem: &str,
    config: &EnsembleConfig,
) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
    if config.members.is_empty() {
        return Err(DecoderError::InvalidConfiguration(
            "ensemble needs at least one decoder".to_string(),
        ));
    }
    let members = config
        .members
        .iter()
        .map(|member| member.build(&DecodeModel::SingleDem(dem.to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Box::new(
        pecos_decoder_core::ensemble::EnsembleDecoder::new(members),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEM: &str = "error(0.1) D0 D1 L0\nerror(0.05) D1\n";

    fn width_dem(highest_observable: usize) -> String {
        format!("error(0.1) D0 L{highest_observable}\ndetector(0, 0, 0) D0\n")
    }

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
            DecoderSpec::AStar,
            DecoderSpec::AStarFull,
            DecoderSpec::PecosUf(PecosUfPreset::Fast),
            DecoderSpec::PecosUf(PecosUfPreset::Bp),
            DecoderSpec::Windowed(WindowedConfig::default()),
            DecoderSpec::BeamSearch(BeamSearchConfig {
                beam_width: 1,
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
                mode,
                embedded_full_dem: None,
            }));
        }

        let hybrid = DecoderSpec::BeliefMatching(BeliefMatchingConfig {
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
    fn resolves_windowed_modes_and_sandwich_defaults() {
        let DecoderSpec::Windowed(auto_config) =
            DecoderSpec::parse("windowed:step=5,buf=5").unwrap()
        else {
            panic!("expected windowed spec");
        };
        let resolved = resolve_windowed_config(&auto_config);
        assert_eq!(resolved.mode, ResolvedWindowedMode::Sandwich);

        let DecoderSpec::Windowed(sandwich_config) =
            DecoderSpec::parse("windowed:mode=sandwich,step=5").unwrap()
        else {
            panic!("expected windowed spec");
        };
        let resolved = resolve_windowed_config(&sandwich_config);
        assert_eq!(resolved.mode, ResolvedWindowedMode::Sandwich);
        assert_eq!(resolved.buffer_size, 5);
        assert!((resolved.commit_weight_max - 2.5).abs() < f64::EPSILON);
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
                DecoderSpec::FusionBlossom(FusionBlossomConfig::default()),
                "fusion-blossom",
            ),
            (DecoderSpec::BpOsd(BpOsdConfig::default()), "ldpc"),
            (DecoderSpec::RelayBp(RelayBpConfig::default()), "relay-bp"),
            (DecoderSpec::PecosUf(PecosUfPreset::Fast), "uf"),
        ];
        for (spec, expected_feature) in cases {
            let error = spec
                .build(&DecodeModel::SingleDem(DEM.to_string()))
                .err()
                .expect("backend should be unavailable");
            assert!(matches!(
                error,
                DecoderError::BackendUnavailable {
                    required_feature,
                    ..
                } if required_feature == expected_feature
            ));
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
        let mut decoder = DecoderSpec::parse("tesseract")
            .unwrap()
            .build(&DecodeModel::SingleDem(DEM.to_string()))
            .unwrap();
        assert!(decoder.decode_obs(&[0, 0]).is_ok());
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
