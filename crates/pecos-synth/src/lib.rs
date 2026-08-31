// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Exact arithmetic and one-qubit Clifford+T synthesis.
//!
//! This crate provides exact arithmetic in [`ZSqrt2`], [`ZOmega`], and
//! [`DOmega`], exact two-by-two [`Matrix`] operations, and Matsumoto--Amano
//! exact synthesis. No floating-point representations or operations are used.

pub mod matrix;
pub mod ring;
pub mod synthesis;

pub use matrix::{GateToken, Matrix, OmegaExponent};
pub use ring::{DOmega, ZOmega, ZSqrt2};
pub use synthesis::{NormalForm, SynthError, exact_synthesize};
