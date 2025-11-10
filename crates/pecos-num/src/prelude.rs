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
//! This prelude re-exports numerical computing functions that replace scipy and numpy functionality.

// Re-export curve fitting
pub use crate::curve_fit::{CurveFitError, CurveFitOptions, CurveFitResult, curve_fit};

// Re-export optimization algorithms
pub use crate::optimize::{BrentqOptions, NewtonOptions, OptimizeError, brentq, newton};

// Re-export polynomial fitting
pub use crate::polynomial::{Poly1d, PolynomialError, polyfit, polyfit_with_cov};

// Re-export random number generation
pub use crate::random;

// Re-export statistical functions
pub use crate::stats::{mean, mean_axis, std, std_axis};

// Re-export mathematical functions
pub use crate::math::{
    ceil, cos, cos_array, exp, exp_array, exp_complex, exp_complex_array, floor, power,
    power_array, round, sin, sin_array, sqrt, sqrt_array,
};

// Re-export mathematical traits
pub use crate::math::{Cos, Exp, Power, Sin, Sqrt};

// Re-export comparison traits
pub use crate::compare::{IsClose, IsNan};

// Re-export ndarray for convenience
pub use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Ix1, Ix2, IxDyn, array};
pub use num_complex::Complex64;

// Re-export array operations
pub use crate::array::{diag, linspace};

// Re-export mathematical constants
pub use crate::math::{
    E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4,
    FRAC_PI_6, FRAC_PI_8, LN_2, LN_10, LOG2_E, LOG10_E, PI, SQRT_2, TAU,
};
