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

//! Default state vector simulator.
//!
//! `StateVec` is the recommended state vector simulator for most use cases.
//! It is currently backed by `StateVecSoA`, which uses a Structure of Arrays
//! layout for optimal performance.
//!
//! This module provides a stable API - the underlying implementation may change
//! in future versions without breaking user code.

use crate::state_vec_soa::StateVecSoA;
use pecos_rng::PecosRng;

/// The default state vector simulator.
///
/// This is a type alias to the current recommended implementation (`StateVecSoA`).
/// Using this type ensures you always get the best-performing implementation
/// without needing to update your code when the underlying implementation changes.
///
/// # Examples
/// ```rust
/// use pecos_qsim::{StateVec, CliffordGateable, qid};
///
/// let mut sim = StateVec::new(2);
/// sim.h(&qid(0));
/// sim.cx(&[pecos_core::QubitId(0), pecos_core::QubitId(1)]);
/// ```
pub type StateVec<R = PecosRng> = StateVecSoA<R>;
