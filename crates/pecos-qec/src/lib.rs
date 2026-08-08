// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Quantum error correction utilities for PECOS.
//!
//! This crate provides tools for defining, verifying, and analyzing stabilizer
//! quantum error correcting codes, with a focus on fault tolerance analysis.
//!
//! # Architecture
//!
//! The crate is organized into three levels:
//!
//! 1. **Abstract level** ([`stabilizer_code`], [`distance`]): Stabilizer algebra, code
//!    verification, distance calculation. Works with the mathematical structure of codes.
//!
//! 2. **Geometry level** ([`geometry`], [`surface`]): Physical layout of codes - where qubits
//!    go, how stabilizers are arranged. Bridges abstract and circuit levels.
//!
//! 3. **Circuit level** ([`fault_tolerance`]): Syndrome extraction circuits, fault tolerance
//!    testing. Integrates with pecos-simulators for simulation.
//!
//! # Fault Tolerance Analysis
//!
//! The [`fault_tolerance`] module provides multiple analysis approaches:
//!
//! - [`StabilizerFlipChecker`]: Code-level analysis using anti-commutation. Works without
//!   a circuit and handles dynamic circuits naturally.
//!
//! - [`PauliPropChecker`]: Circuit-level analysis using Pauli propagation. Verifies specific
//!   circuit implementations.
//!
//! - Syndrome history analysis: Multi-round QEC analysis tracking syndromes across rounds.
//!
//! # Quick Example
//!
//! ```
//! use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker};
//! use pecos_core::{Xs, Zs};
//!
//! // Define a 3-qubit bit flip code
//! let code = StabilizerCodeSpec::builder(3)
//!     .check(Zs([0, 1]))
//!     .check(Zs([1, 2]))
//!     .logical_z(Zs([0, 1, 2]))
//!     .logical_x(Xs([0]))
//!     .build()
//!     .unwrap();
//!
//! // Analyze fault tolerance
//! let checker = StabilizerFlipChecker::new(&code);
//!
//! // X-distance is 3 (protects against 1 X error)
//! let analysis = checker.analyze_weight_with_types(1, true, false, false);
//! assert_eq!(analysis.undetectable_logical, 0);
//! ```

pub mod bivariate_bicycle;
pub mod bounded_enumeration_distance;
pub mod code_distance;
pub mod coloration;
pub mod dem_stab;
pub mod distance;
pub mod distance_problem;
pub mod fault_tolerance;
pub mod geometry;
pub mod hypergraph_product;
pub mod logical_discovery;
pub mod mem_stab;
mod memory_circuit;
pub mod parity_check_matrix;
pub mod stabilizer_code;
pub mod stabilizer_code_spec;
pub mod surface;

pub use bivariate_bicycle::{
    BbMemoryBasis, BbMonomial, BivariateBicycleCode, BivariateBicycleError, bb_memory_circuit,
};
pub use bounded_enumeration_distance::{
    BoundedEnumerationBackendError, BoundedEnumerationDistance, CpuLevelEnumerationBackend,
    LevelEnumerationBackend, LevelEnumerationInput, LevelEnumerationMinimum,
    PackedSystematicGenerator, bounded_enumeration_code_distance,
    bounded_enumeration_code_distance_with_backend, bounded_enumeration_stabilizer_distance,
    bounded_enumeration_stabilizer_distance_with_backend, bounded_enumeration_x_distance,
    bounded_enumeration_x_distance_with_backend, bounded_enumeration_z_distance,
    bounded_enumeration_z_distance_with_backend,
};
pub use coloration::{ColorationMemoryError, coloration_memory_circuit};
pub use dem_stab::{DemStabError, DemStabShotBatch, DemStabSim, DemStabSimBuilder};
pub use hypergraph_product::{HypergraphProductCode, HypergraphProductError};
pub use mem_stab::{MemStabError, MemStabSim, MemStabSimBuilder};
pub use memory_circuit::MemoryBasis;
pub use parity_check_matrix::{ParityCheckMatrix, ParityCheckMatrixError};

pub use code_distance::{
    connected_cluster_code_distance, stabilizer_code_distance, x_distance, z_distance,
};

pub use distance::{
    DistanceResult, DistanceSearchConfig, LogicalOperatorInfo, WeightedPauliIterator,
    calculate_distance, find_min_weight_logicals, find_min_weight_logicals_with_info,
    find_shortest_logicals, has_logical_error_at_weight,
};
pub use distance_problem::{
    CertifiedDistance, CosetWeightError, DistanceCertificationError, DistanceProblem,
    DistanceProblemError, SolverAnswer, WitnessError, certified_classical_distance,
    certified_coset_weight, certified_distance, certified_stabilizer_coset_weight,
    logical_coset_weight_profile,
};
pub use fault_tolerance::dem_builder::{
    DecomposedFault, DemBuilder, DemBuilderError, DemOutput, DetectorDef, DetectorErrorModel,
    FaultMechanism, NoiseConfig, PecosDemMetadataError, combine_probabilities,
};
pub use fault_tolerance::{
    CircuitDistanceResult, CorrectionResult, DecoderAnalysis, DemOutputKind, DemOutputMetadata,
    ErrorClass, ErrorCorrectionChecker, ErrorCorrectionConfig, ErrorCorrectionResult,
    FaultCheckConfig, FaultCheckResult, FaultChecker, FaultClass, FaultConfiguration,
    FaultDistanceError, FaultDistanceResult, FaultToleranceAnalysis, FaultToleranceFailure,
    FlagFaultToleranceReport, FlagViolation, HookError, HookErrorReport, LookupTableDecoder,
    MeasurementRound, PauliFault, PauliFaultIterator, PauliPropChecker, PropagationResult,
    SpacetimeLocation, StabilizerFlipAnalysis, StabilizerFlipChecker, StabilizerFlips,
    SyndromeAnalysis, SyndromeClass, SyndromeHistory, SyndromeHistoryAnalysis,
    SyndromeHistoryResult, anticommutes_with_logical, apply_recovery, classify_fault,
    connected_cluster_fault_distance, exhaustive_fault_distance, extract_measurement_rounds,
    extract_spacetime_locations, extract_syndrome, get_syndrome_flips, graphlike_fault_distance,
    has_syndrome, per_observable_fault_distances, propagate_fault, propagate_faults,
    run_circuit_with_faults, run_correction_cycle,
};
pub use geometry::{CheckSchedule, LogicalOperator, PauliOp, StabilizerCheck, StabilizerColor};
pub use logical_discovery::{
    LogicalDiscoveryError, LogicalDiscoveryResult, discover_logical_operators,
};
pub use stabilizer_code::StabilizerCode;
pub use stabilizer_code_spec::{
    StabilizerCodeSpec, StabilizerCodeSpecBuilder, StabilizerCodeSpecError,
};
pub use surface::{SurfaceCode, SurfaceCodeBuilder};
