// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Detector Error Model (DEM) generation from fault influence maps.
//!
//! This module provides Rust-native DEM generation that produces output
//! compatible with Stim's format. It uses the per-qubit fault model for
//! accurate depolarizing noise analysis.
//!
//! # Architecture
//!
//! The DEM builder takes a [`DagFaultInfluenceMap`] (which maps fault locations
//! to their effects on measurements) and detector/observable metadata to produce
//! a complete DEM.
//!
//! # Example
//!
//! ```
//! use pecos_qec::DemBuilder;
//! use pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Normally `influence_map` comes from `DagFaultAnalyzer::build_influence_map()`;
//! // here we use an empty map to keep the doctest self-contained.
//! let influence_map = DagFaultInfluenceMap::with_capacity(0);
//!
//! let dem = DemBuilder::new(&influence_map)
//!     .with_noise(0.01, 0.01, 0.01, 0.01)
//!     .with_detectors_json("[]")?
//!     .with_observables_json("[]")?
//!     .build();
//!
//! // Output in Stim format (non-decomposed).
//! let _ = dem.to_string();
//! # Ok(())
//! # }
//! ```
//!
//! # Error Decomposition
//!
//! When using `to_stim_format_decomposed()`, hyperedge errors (affecting 3+
//! detectors) are decomposed into combinations of graphlike errors (affecting
//! 1-2 detectors). This is necessary for MWPM decoders which only work on
//! graphs, not hypergraphs.
//!
//! # Comparison with Python Implementation
//!
//! This Rust implementation mirrors the Python `generate_dem_from_tick_circuit`
//! function but runs entirely in Rust for better performance. Key features:
//!
//! - **Per-qubit fault model**: Each fault location has exactly one qubit,
//!   enabling proper analysis of correlated two-qubit gate errors.
//!
//! - **15 Pauli combinations for 2Q gates**: All non-identity two-qubit
//!   Pauli combinations (IX, IY, IZ, XI, ..., ZZ) are considered.
//!
//! - **XOR effect combining**: Correlated errors are properly combined
//!   by XOR-ing detector/observable effects.
//!
//! - **Independent probability combination**: When the same fault mechanism
//!   is triggered by multiple error sources, probabilities are combined
//!   using p1*(1-p2) + p2*(1-p1).
//!
//! # Measurement Noise Model (MNM)
//!
//! In addition to the DEM, this module provides a Measurement Noise Model (MNM)
//! for fast approximate sampling. Unlike the DEM which maps to detectors, the
//! MNM maps directly to raw measurement effects.
//!
//! ```
//! use pecos_qec::fault_tolerance::dem_builder::MemBuilder;
//! use pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Normally `influence_map` comes from `DagFaultAnalyzer::build_influence_map()`;
//! // here we use an empty map to keep the doctest self-contained.
//! let influence_map = DagFaultInfluenceMap::with_capacity(0);
//!
//! let mnm = MemBuilder::new(&influence_map)
//!     .with_noise(0.01, 0.01, 0.01, 0.01)
//!     .build();
//!
//! let mut rng = StdRng::seed_from_u64(0);
//! let _outcomes = mnm.sample(&mut rng);
//! ```
//!
//! The MNM aggregates fault locations by their measurement effects (which
//! measurements flip together), enabling faster sampling with fewer random
//! draws compared to per-fault-location sampling.

mod builder;
mod dem_sampler;
mod equivalence;
mod mem_builder;
mod types;

pub use builder::{DemBuilder, DemBuilderError};
pub use dem_sampler::{DemSampler, DemSamplerBuilder, SamplingStatistics};
pub use equivalence::{
    ComparisonDetails, ComparisonMethod, DemParseError, EffectKey, EquivalenceResult,
    MechanismComponent, ParsedDem, ParsedMechanism, ProbabilityMismatch, compare_dems_exact,
    compare_dems_statistical, verify_dem_equivalence,
};
pub use mem_builder::MemBuilder;
pub use types::{
    ContributionEffectSummary, ContributionRenderRecord, ContributionRenderStrategy,
    ContributionRenderSummary, DecomposedFault, DetectorDef, DetectorErrorModel,
    DirectSourceFamily, FaultContribution, FaultMechanism, FaultSourceType, LogicalObservable,
    MeasurementMechanism, MeasurementNoiseModel, NoiseConfig, TwoDetectorDirectRenderPolicy,
    combine_probabilities, record_offset_to_absolute_index,
};
