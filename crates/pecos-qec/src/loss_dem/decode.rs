//! Cached loss-aware and unconditional decoding of the same engine shots.
use super::branches::emit_normal_form;
use super::shot::normalise;
use super::{
    Approximation, AssignmentError, LossCatalog, LossDemError, ProbeId, Shot, ShotBuildError,
    ShotDem, ShotDemBuilder, ShotLimits, TickRunner, invalid,
};
use pecos_core::BitSet;
use pecos_decoder_core::{DecoderError, ObservableDecoder, obs_mask::ObsMask};
use pecos_decoders::{DecodeModel, DecoderSpec};
use std::collections::{BTreeMap, BTreeSet};

/// Cacheable failure, with a diagnostic retained for each distinct reason.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ShotError {
    /// Invalid flags or failed assignment search.
    #[error(transparent)]
    Assignment(#[from] AssignmentError),
    /// Affine/model construction failure.
    #[error("shot DEM: {0}")]
    Dem(String),
    /// Caller-selected backend or hyperedge policy rejected the DEM.
    #[error("decoder build: {0}")]
    DecoderBuild(String),
    /// A built decoder failed to decode this syndrome.
    #[error("decode: {0}")]
    Decode(String),
}

impl From<LossDemError> for ShotError {
    fn from(error: LossDemError) -> Self {
        Self::Dem(error.to_string())
    }
}

impl From<ShotBuildError> for ShotError {
    fn from(error: ShotBuildError) -> Self {
        match error {
            ShotBuildError::Assignment(error) => Self::Assignment(error),
            ShotBuildError::Dem(error) => Self::Dem(error.to_string()),
        }
    }
}

/// Stable categories for per-mode error counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErroredShotReason {
    /// A probe was outside the catalog.
    InvalidProbe,
    /// No cover with a distinct-draw assignment.
    NoAssignment,
    /// An assignment limit was exceeded.
    TooManyAssignments,
    /// Cover and candidate traversal exceeded the per-region node budget.
    SearchNodeLimit,
    /// Invalid DEM inputs.
    Dem,
    /// Backend construction or graphlike projection failed.
    DecoderBuild,
    /// Runtime decoder error.
    Decode,
}

impl ShotError {
    fn reason(&self) -> ErroredShotReason {
        match self {
            Self::Assignment(AssignmentError::InvalidProbe { .. }) => {
                ErroredShotReason::InvalidProbe
            }
            Self::Assignment(AssignmentError::NoAssignment { .. }) => {
                ErroredShotReason::NoAssignment
            }
            Self::Assignment(AssignmentError::TooManyAssignments { .. }) => {
                ErroredShotReason::TooManyAssignments
            }
            Self::Assignment(AssignmentError::SearchNodeLimit { .. }) => {
                ErroredShotReason::SearchNodeLimit
            }
            Self::Dem(_) => ErroredShotReason::Dem,
            Self::DecoderBuild(_) => ErroredShotReason::DecoderBuild,
            Self::Decode(_) => ErroredShotReason::Decode,
        }
    }
}

/// Backend-independent bias wrapper. Standard text is passed to `DecoderSpec`,
/// which owns backend dispatch and the caller's irreducible-hyperedge policy.
pub struct BiasedDecoder {
    decoder: Option<Box<dyn ObservableDecoder>>,
    detector_bias: Vec<u8>,
    observable_bias: ObsMask,
}

impl BiasedDecoder {
    /// Construct using `DecodeModel::SingleDem(shot.dem.to_string())`.
    /// A normalised model with no nonempty effects bypasses the factory: its prediction is
    /// the observable bias, including when the shifted syndrome is nonzero.
    ///
    /// # Errors
    /// Forwards backend and projection errors as `DecoderBuild`.
    pub fn build(shot: &ShotDem, spec: &DecoderSpec) -> Result<Self, ShotError> {
        let decoder = if shot.dem.contributions().iter().all(|c| c.effect.is_empty()) {
            None
        } else {
            Some(
                spec.build(&DecodeModel::SingleDem(shot.dem.to_string()))
                    .map_err(|e| ShotError::DecoderBuild(e.to_string()))?,
            )
        };
        Ok(Self {
            decoder,
            detector_bias: shot.detector_bias.clone(),
            observable_bias: shot.observable_bias.clone(),
        })
    }
}

impl ObservableDecoder for BiasedDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.detector_bias.len())
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        if syndrome.len() != self.detector_bias.len() || syndrome.iter().any(|&b| b > 1) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "syndrome length {} must match shot DEM width {}; nonbinary entries: {:?}",
                syndrome.len(),
                self.detector_bias.len(),
                syndrome
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| **b > 1)
                    .collect::<Vec<_>>()
            )));
        }
        let shifted: Vec<_> = syndrome
            .iter()
            .zip(&self.detector_bias)
            .map(|(bit, bias)| bit ^ bias)
            .collect();
        let mut prediction = match &mut self.decoder {
            Some(decoder) => decoder.decode_obs(&shifted)?,
            None => ObsMask::new(),
        };
        prediction ^= &self.observable_bias;
        Ok(prediction)
    }
}

/// Counts include errored shots in both the denominator and logical failures.
#[derive(Debug, Clone, Default)]
pub struct ModeCounts {
    /// Number of attempted shots.
    pub shots: usize,
    /// Wrong logical predictions plus all errored shots.
    pub logical_errors: usize,
    /// Errored shots by reason.
    pub errored_shots: BTreeMap<ErroredShotReason, usize>,
    /// First full diagnostic for each reason.
    pub error_examples: BTreeMap<ErroredShotReason, String>,
    /// Reuses of a cached decoder or cached construction error.
    pub cache_hits: usize,
    /// Number of shots whose model carries each approximation.
    pub approximation_counts: BTreeMap<Approximation, usize>,
}

impl ModeCounts {
    fn record(&mut self, result: Result<ObsMask, ShotError>, actual: &ObsMask) {
        self.shots += 1;
        match result {
            Ok(prediction) => self.logical_errors += usize::from(prediction != *actual),
            Err(error) => {
                self.logical_errors += 1;
                *self.errored_shots.entry(error.reason()).or_default() += 1;
                self.error_examples
                    .entry(error.reason())
                    .or_insert_with(|| error.to_string());
            }
        }
    }
}

/// Side-by-side counts on identical shots.
#[derive(Debug, Clone, Default)]
pub struct DecodeReport {
    /// Flag-conditioned mode.
    pub loss_aware: ModeCounts,
    /// One unconditional model at absolute branch weights.
    pub loss_blind: ModeCounts,
}

struct CacheEntry {
    decoder: Result<BiasedDecoder, ShotError>,
    approximations: BTreeSet<Approximation>,
}

impl CacheEntry {
    fn new(model: Result<ShotDem, ShotError>, spec: &DecoderSpec) -> Self {
        match model {
            Ok(model) => Self {
                decoder: BiasedDecoder::build(&model, spec),
                approximations: model.approximations,
            },
            Err(error) => Self {
                decoder: Err(error),
                approximations: BTreeSet::new(),
            },
        }
    }

    fn decode(&mut self, shot: &Shot, counts: &mut ModeCounts) {
        for &approximation in &self.approximations {
            *counts
                .approximation_counts
                .entry(approximation)
                .or_default() += 1;
        }
        let result = match &mut self.decoder {
            Ok(decoder) => decoder
                .decode_obs(&shot.detectors)
                .map_err(|e| ShotError::Decode(e.to_string())),
            Err(error) => Err(error.clone()),
        };
        counts.record(result, &shot.observables);
    }
}

impl LossCatalog {
    /// Unconditional first-order DEM. Unflagged branches stay at prior weight,
    /// without posterior rescaling or removal (`UnflaggedBackground`). Surviving
    /// gate Pauli noise is the lossless background, as in E2.
    ///
    /// # Errors
    /// Invalid normal forms or correlated ranks that cannot be enumerated.
    pub fn loss_blind_dem(&self, max_correlated_rank: usize) -> Result<ShotDem, LossDemError> {
        let mut dem = self.background.clone();
        let mut approximations: BTreeSet<_> = self
            .report
            .approximated
            .values()
            .flatten()
            .copied()
            .collect();
        approximations.insert(Approximation::UnflaggedBackground);
        for root in &self.roots {
            for branch in &root.branches {
                if !branch.flags.is_empty() {
                    approximations.extend(emit_normal_form(
                        &mut dem,
                        &branch.normal_form,
                        branch.weight,
                        max_correlated_rank,
                        self.num_detectors,
                    )?);
                }
            }
        }
        let mut result = normalise(&dem, &BitSet::new())?;
        result.approximations = approximations;
        result.pruned_mass_upper_bound = self.report.dropped_mass;
        Ok(result)
    }
}

/// Run-scoped decoder cache. Both successes and construction failures are cached
/// by flag set, unbounded within a run. The runner noise model is NOT checked
/// against the catalog's; the caller must use matching built noise settings.
/// The library enables no decoder backend features; callers choose `DecoderSpec`
/// and enable their backend on `pecos-decoders`.
pub struct LossDecoder;

impl LossDecoder {
    /// Decode both modes on each shot. Output layouts must agree before sampling.
    ///
    /// # Errors
    /// Mismatched output tables, invalid catalog draw masses, or engine execution failure. Assignment and decoder
    /// errors are counted as logical failures and retained in the returned report.
    pub fn run(
        catalog: &LossCatalog,
        runner: &mut TickRunner,
        spec: &DecoderSpec,
        shots: usize,
        limits: ShotLimits,
    ) -> Result<DecodeReport, LossDemError> {
        if catalog.outputs != *runner.table() {
            let actual = runner.table();
            let mut differing = Vec::new();
            if catalog.outputs.slots != actual.slots {
                differing.push("slots");
            }
            if catalog.outputs.probes != actual.probes {
                differing.push("probes");
            }
            if catalog.outputs.records != actual.records {
                differing.push("records");
            }
            if catalog.outputs.record_locations != actual.record_locations {
                differing.push("record_locations");
            }
            return Err(invalid(format!(
                "catalog.outputs differs from runner.table() in fields {differing:?}; use the same circuit layout"
            )));
        }
        let builder = ShotDemBuilder::new(catalog)?;
        let mut cache = BTreeMap::new();
        let mut blind = CacheEntry::new(
            catalog
                .loss_blind_dem(limits.max_correlated_rank)
                .map_err(Into::into),
            spec,
        );
        let mut report = DecodeReport::default();
        for index in 0..shots {
            let shot = runner.shot()?;
            let flags: BTreeSet<_> = shot
                .flags
                .iter()
                .enumerate()
                .filter_map(|(i, &flag)| flag.then_some(ProbeId(i)))
                .collect();
            let entry = match cache.entry(flags) {
                std::collections::btree_map::Entry::Occupied(entry) => {
                    report.loss_aware.cache_hits += 1;
                    entry.into_mut()
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let model = builder.build(entry.key(), limits).map_err(Into::into);
                    entry.insert(CacheEntry::new(model, spec))
                }
            };
            entry.decode(&shot, &mut report.loss_aware);
            report.loss_blind.cache_hits += usize::from(index > 0);
            blind.decode(&shot, &mut report.loss_blind);
        }
        Ok(report)
    }
}
