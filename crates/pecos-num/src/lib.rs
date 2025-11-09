// Copyright 2024 The PECOS Developers
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

//! # pecos-num: Numerical Computing for PECOS
//!
//! This crate provides numerical computing functionality for PECOS, serving as a
//! Rust-based replacement for scipy.optimize functions. It offers:
//!
//! - Root finding algorithms (Brent's method, Newton-Raphson)
//! - Curve fitting (Levenberg-Marquardt, polynomial fitting)
//! - Performance improvements over scipy
//! - Better cross-platform support
//!
//! ## Usage
//!
//! This crate is typically accessed through the `pecos::prelude`. Python bindings
//! are provided separately in `pecos-rslib`.

pub mod curve_fit;
pub mod optimize;
pub mod polynomial;
pub mod prelude;

pub use curve_fit::{CurveFitError, CurveFitOptions, CurveFitResult, curve_fit};
pub use optimize::{BrentqOptions, NewtonOptions, OptimizeError, brentq, newton};
pub use polynomial::{Poly1d, PolynomialError, polyfit};
