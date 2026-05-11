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

pub mod angle;
pub mod bit;
pub mod bit_int;
pub mod bit_uint;
pub mod bitset;
pub mod bitvec;
pub mod circuit_diagram;
pub mod classical_bit_id;
pub mod clifford_rep;
pub mod clifford_simplify;
pub mod duration;
pub mod element;
pub mod errors;
pub mod gate_registry;
pub mod gate_type;
pub mod gates;
pub mod index_set;
pub mod meas_id;
pub mod pauli;
pub mod phase;
pub mod prelude;
pub mod qubit_id;
mod qubit_support;
pub mod rng;
pub mod sets;
pub mod signal;
pub mod sorted_vec_set;
pub mod unitary_rep;
pub mod value;

pub use angle::{Angle, Angle8, Angle16, Angle32, Angle64, Angle128, LossyInto};
pub use bit::{Bit, Bits};
pub use bit_int::BitInt;
pub use bit_uint::BitUInt;
pub use bitset::BitSet;
pub use duration::{TimeScale, TimeUnits};
pub use element::Element;
pub use index_set::IndexSet;
pub use meas_id::MeasId;
pub use phase::GlobalPhase;
pub use phase::quarter_phase::QuarterPhase;
pub use phase::sign::Sign;
pub use qubit_id::{QubitId, QubitIdSet, qid, qid2, qids, qids2};
pub use rng::{RngManageable, derive_seed};
pub use sets::set::Set;
pub use sets::vec_set::VecSet;
pub use sorted_vec_set::SortedVecSet;

// Utility functions for random number generation
pub use rng::{choose_weighted, coin_flip, gen_bools};

// Random utilities struct for improved RNG API
pub use rng::RandomUtils;

pub use classical_bit_id::ClassicalBitId;
pub use clifford_simplify::{
    half_turn_decomposition, is_rzz_z_tensor_z, try_simplify_r1xy, try_simplify_rotation,
};
pub use gate_registry::{
    AngleSource, ConcreteStep, DecompStep, GateDefinition, GateDefinitionBuilder, GateRegistry,
    GateSignature,
};
pub use gates::{Gate, GateAngles, GateMeasIds, GateParams, GateQubits};
pub use pauli::pauli_bitmap::PauliBitmap;
pub use pauli::pauli_bitmask::{
    BitmaskStorage, Conjugated, PauliBitmask, PauliBitmaskGeneric, PauliBitmaskSmall,
    PauliBitmaskVec,
};
pub use pauli::pauli_sparse::PauliSparse;
pub use pauli::pauli_string::{ParsePauliStringError, PauliString};
pub use pauli::{Pauli, PauliOperator};
pub use phase::Phase;
pub use rng::choices::Choices;
pub use value::Value;

// Circuit diagram styling
pub use circuit_diagram::{
    AngleUnit, ColorPalette, ColorTriplet, CosetPatterns, DiagramRenderer, DiagramStyle,
    DiagramStyleBuilder, FamilyPalette, FillPattern, GraphStyle, GraphStyleBuilder, blend_hex,
};

// --- Algebraic-level namespaces ---
//
// Each level is a module whose glob import gives the user the constructors and
// types for that algebraic level:
//
//   use pecos_core::pauli::*;     // I, X, Y, Z, Xs, Ys, Zs -> PauliString
//   use pecos_core::clifford::*;  // H, CX, CZ, SWAP, ... -> CliffordRep
//   use pecos_core::unitary::*;   // T, RZ, CCX, ... -> UnitaryRep
//   use pecos_core::gate::*;      // MZ, PZ, Reset, ... -> GateExpr
//   use pecos_core::channel::*;   // Depolarizing, PauliChannel, ... -> ChannelExpr
//   use pecos_core::op::*;        // MZ, PZ, Depolarizing, ... -> Op (promoted)

pub mod unitary;
pub use unitary_rep::{Is, Unitary, UnitaryRep};

pub use pauli::constructors::{I, X, Xs, Y, Ys, Z, Zs};

pub mod clifford;
pub use clifford::Clifford;

pub mod channel;
pub mod gate;
pub mod gate_algebra;

pub mod op;
pub use op::{Basis, ChannelExpr, GateExpr, Level, Op};

// Signals
pub use signal::Signal;
