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
pub mod bitset;
pub mod bitvec;
pub mod clifford_rep;
pub mod duration;
pub mod element;
pub mod errors;
pub mod gate_type;
pub mod gates;
pub mod index_set;
pub mod operator;
pub mod pauli;
pub mod phase;
pub mod prelude;
pub mod qubit_id;
pub mod rng;
pub mod sets;
pub mod sorted_vec_set;

pub use angle::{Angle, Angle8, Angle16, Angle32, Angle64, Angle128, LossyInto};
pub use bit::{Bit, Bits};
pub use bit_int::BitInt;
pub use bitset::BitSet;
pub use duration::{Nanoseconds, TimeUnits};
pub use element::Element;
pub use index_set::IndexSet;
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

pub use gates::{Gate, GateAngles, GateParams, GateQubits};
pub use pauli::pauli_bitmap::PauliBitmap;
pub use pauli::pauli_sparse::PauliSparse;
pub use pauli::pauli_string::{ParsePauliStringError, PauliString};
pub use pauli::{Pauli, PauliOperator};
pub use phase::Phase;
pub use rng::choices::Choices;

// Operator algebra
pub use operator::{I, Is, Operator, X, Xs, Y, Ys, Z, Zs};
