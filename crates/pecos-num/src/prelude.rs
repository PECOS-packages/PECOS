// Copyright 2025 The PECOS Developers
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

//! A prelude for users of the `pecos-num` crate.
//!
//! This prelude re-exports numerical computing functions that replace scipy.optimize.

// Re-export curve fitting
pub use crate::curve_fit::{CurveFitError, CurveFitOptions, CurveFitResult, curve_fit};

// Re-export optimization algorithms
pub use crate::optimize::{BrentqOptions, NewtonOptions, OptimizeError, brentq, newton};

// Re-export polynomial fitting
pub use crate::polynomial::{Poly1d, PolynomialError, polyfit};
