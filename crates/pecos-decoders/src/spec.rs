//! Typed, feature-independent decoder construction specifications.

pub mod config;

mod build;
mod parse;

use pecos_decoder_core::{DecoderError, ObservableDecoder};

pub use config::*;

/// Detector-error-model inputs accepted by decoder specifications.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodeModel {
    SingleDem(String),
    HybridDem { full: String, decomposed: String },
}

/// Cross-shot execution properties relevant to batching.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionTraits {
    pub history_dependent: bool,
    pub wall_clock_dependent: bool,
}

impl ExecutionTraits {
    fn or(self, other: Self) -> Self {
        Self {
            history_dependent: self.history_dependent || other.history_dependent,
            wall_clock_dependent: self.wall_clock_dependent || other.wall_clock_dependent,
        }
    }
}

/// Typed construction recipe for every decoder family supported by the legacy factory.
#[derive(Clone, Debug, PartialEq)]
pub enum DecoderSpec {
    PyMatching(PyMatchingConfig),
    Tesseract(TesseractConfig),
    KMwpm(KMwpmConfig),
    AStar,
    AStarFull,
    FusionBlossom(FusionBlossomConfig),
    PerturbedFusionBlossomCorrelated(PerturbedFusionBlossomConfig),
    BpOsd(BpOsdConfig),
    BpLsd(BpLsdConfig),
    BeliefFind,
    UnionFind,
    RelayBp(RelayBpConfig),
    MinSumBp(MinSumBpConfig),
    PecosUf(PecosUfPreset),
    BeliefMatching(BeliefMatchingConfig),
    Windowed(WindowedConfig),
    Mwpf(MwpfConfig),
    Perturbed(PerturbedConfig),
    BeamSearch(BeamSearchConfig),
    Ensemble(EnsembleConfig),
}

impl DecoderSpec {
    /// Parse the strict, one-way legacy type-string grammar.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for malformed parameters,
    /// unknown keys, or unsupported decoder names.
    pub fn parse(type_string: &str) -> Result<Self, DecoderError> {
        parse::parse(type_string)
    }

    /// Build a decoder from a typed model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] for a model mismatch,
    /// [`DecoderError::BackendUnavailable`] when a required feature is absent,
    /// or a backend construction error.
    pub fn build(&self, model: &DecodeModel) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
        build::build(self, model)
    }

    /// Return execution properties, recursively OR-ing composite members.
    #[must_use]
    pub fn execution_traits(&self) -> ExecutionTraits {
        match self {
            Self::RelayBp(_) => ExecutionTraits {
                history_dependent: true,
                wall_clock_dependent: false,
            },
            Self::BpOsd(config) => ExecutionTraits {
                history_dependent: config.bp_schedule.uses_random_serial_order(),
                wall_clock_dependent: false,
            },
            Self::BpLsd(config) => ExecutionTraits {
                history_dependent: config.bp_schedule.uses_random_serial_order(),
                wall_clock_dependent: false,
            },
            Self::Mwpf(config) => ExecutionTraits {
                history_dependent: false,
                wall_clock_dependent: config.timeout.is_some(),
            },
            Self::Windowed(config) => config
                .inner
                .execution_traits()
                .or(config.sandwich_phase2.execution_traits()),
            Self::Perturbed(config) => config.inner.execution_traits(),
            Self::BeamSearch(config) => config.phase2.execution_traits(),
            Self::Ensemble(config) => config
                .members
                .iter()
                .fold(ExecutionTraits::default(), |traits, member| {
                    traits.or(member.execution_traits())
                }),
            _ => ExecutionTraits::default(),
        }
    }

    /// Whether this specification has a backend-native observable batch path.
    #[must_use]
    pub const fn native_batch_capable(&self) -> bool {
        matches!(self, Self::PyMatching(_))
    }

    /// Return the full DEM embedded by the legacy hybrid grammar, if present.
    #[must_use]
    pub fn embedded_hybrid_full_dem(&self) -> Option<&str> {
        match self {
            Self::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                embedded_full_dem,
            }) => embedded_full_dem.as_deref(),
            _ => None,
        }
    }
}
