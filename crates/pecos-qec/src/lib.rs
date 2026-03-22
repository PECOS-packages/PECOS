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

pub mod distance;
pub mod fault_tolerance;
pub mod geometry;
pub mod logical_discovery;
pub mod stabilizer_code;
pub mod stabilizer_code_spec;
pub mod surface;

pub use distance::{
    DistanceResult, DistanceSearchConfig, LogicalOperatorInfo, WeightedPauliIterator,
    calculate_distance, find_min_weight_logicals, find_min_weight_logicals_with_info,
};
pub use fault_tolerance::dem_builder::{
    DecomposedError, DemBuilder, DemBuilderError, DetectorDef, DetectorErrorModel, ErrorMechanism,
    LogicalObservable, NoiseConfig, combine_probabilities,
};
pub use fault_tolerance::{
    CorrectionResult, DecoderAnalysis, ErrorClass, ErrorCorrectionChecker, ErrorCorrectionConfig,
    ErrorCorrectionResult, FaultCheckConfig, FaultCheckResult, FaultChecker, FaultClass,
    FaultConfiguration, FaultToleranceAnalysis, FaultToleranceFailure, LookupTableDecoder,
    MeasurementRound, PauliFault, PauliFaultIterator, PauliPropChecker, PropagationResult,
    SpacetimeLocation, StabilizerFlipAnalysis, StabilizerFlipChecker, StabilizerFlips,
    SyndromeAnalysis, SyndromeClass, SyndromeHistory, SyndromeHistoryAnalysis,
    SyndromeHistoryResult, anticommutes_with_logical, apply_recovery, classify_fault,
    extract_measurement_rounds, extract_spacetime_locations, extract_syndrome, get_syndrome_flips,
    has_syndrome, propagate_fault, propagate_faults, run_circuit_with_faults, run_correction_cycle,
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
