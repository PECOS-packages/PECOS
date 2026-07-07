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
//! Numerical computing functionality for PECOS, including:
//!
//! - Statistical functions (mean, std)
//! - Mathematical functions (cos, sin, exp, sqrt, power, etc.)
//! - Comparison utilities (isnan, isclose)
//! - Array operations (diag, linspace)
//! - Random number generation (numpy.random drop-in replacements)
//! - Root finding algorithms (Brent's method, Newton-Raphson)
//! - Curve fitting (damped least-squares, polynomial fitting)
//! - Graph data structures ([`Graph`](graph::Graph), [`DiGraph`](digraph::DiGraph), [`DAG`](dag::DAG))
//! - Graph algorithms (MWPM matching, shortest paths, topological sort)
//! - Performance improvements over scipy/numpy
//! - Better cross-platform support
//!
//! ## Usage
//!
//! This crate is typically accessed through the `pecos::prelude`. Python bindings
//! are provided separately in `pecos-rslib`.

pub mod array;
pub mod compare;
pub mod curve_fit;
pub mod dag;
pub mod digraph;
pub mod graph;
pub mod linalg;
pub mod math;
pub mod optimize;
pub mod polynomial;
pub mod prelude;
pub mod random;
pub mod special;
pub mod stats;
pub mod z2_linalg;

pub use compare::{allclose, relative_eq};
pub use curve_fit::{CurveFitError, CurveFitOptions, CurveFitResult, curve_fit};
pub use dag::{DAG, DAGHasCycleError, DagWouldCycleError};
pub use digraph::DiGraph;
pub use graph::Graph;
pub use linalg::{matrix_exp, matrix_log};
pub use optimize::{BrentqOptions, NewtonOptions, OptimizeError, brentq, newton};
pub use polynomial::{Poly1d, PolynomialError, polyfit};
pub use special::{betainc_inv, betainc_reg, ln_gamma};
pub use stats::{jeffreys_interval, mean};
