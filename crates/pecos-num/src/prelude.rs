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

// Re-export linear algebra
pub use crate::linalg::{norm, norm_complex};

// Re-export optimization algorithms
pub use crate::optimize::{BrentqOptions, NewtonOptions, OptimizeError, brentq, newton};

// Re-export polynomial fitting
pub use crate::polynomial::{Poly1d, PolynomialError, polyfit, polyfit_with_cov};

// Re-export random number generation
pub use crate::random;

// Re-export statistical functions
pub use crate::stats::{
    jackknife_resamples, jackknife_stats, jackknife_stats_axis, jackknife_weighted, mean,
    mean_axis, std, std_axis, weighted_mean,
};

// Re-export mathematical traits (use these for polymorphism!)
pub use crate::math::{
    Abs, Acos, Acosh, Asin, Asinh, Atan, Atan2, Atanh, Ceil, Cos, Cosh, Exp, Floor, Ln, LogBase,
    Power, RoundTiesEven, Sin, Sinh, Sqrt, Tan, Tanh,
};

// Re-export mathematical functions
// Note: floor/ceil are simple wrappers around stdlib for convenience
// For NumPy-compatible rounding:
//   - f32/f64 scalars: use .round_ties_even() directly (stdlib method)
//   - Complex scalars: use .round_ties_even() (trait extension from this crate)
//   - f32/f64 arrays: use .mapv(|x| x.round_ties_even())
//   - Complex arrays: use .mapv(|x| x.round_ties_even())
pub use crate::math::{atan2, ceil, floor};

// Re-export comparison functions and traits
pub use crate::compare::{IsClose, IsNan, Where, allclose, array_equal, assert_allclose, where_};

// Re-export ndarray for convenience (expanded for better ergonomics)
// Core array types
pub use ndarray::{
    Array,
    Array1,
    Array2,
    Array3,
    ArrayBase,
    ArrayD,
    ArrayView,
    ArrayView1,
    ArrayView2,
    ArrayView3,
    ArrayViewMut,
    ArrayViewMut1,
    ArrayViewMut2,
    ArrayViewMut3,
    Axis,
    Dim,
    Dimension,
    Ix1,
    Ix2,
    Ix3,
    Ix4,
    Ix5,
    Ix6,
    IxDyn,
    ScalarOperand,
    Slice,
    SliceInfo,
    SliceInfoElem,
    // Constructors and macros
    array,
    aview1,
    aview2,
    s,
};

// Re-export num-complex
pub use num_complex::{Complex, Complex32, Complex64};

// Re-export array operations
// Note: sum() for slices removed - use .iter().sum() directly (idiomatic Rust)
pub use crate::array::{arange, delete, diag, linspace, ones, sum_axis, zeros};

// Re-export graph types and algorithms
pub use crate::dag::{DAG, DAGHasCycleError, DagWouldCycleError};
pub use crate::digraph::DiGraph;
pub use crate::graph::{self, Graph};

// Re-export mathematical constants (f64)
pub use crate::math::{
    E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4,
    FRAC_PI_6, FRAC_PI_8, LN_2, LN_10, LOG2_E, LOG10_E, PI, SQRT_2, TAU,
};

// Re-export mathematical constants (f32)
pub use crate::math::{
    E_F32, FRAC_1_PI_F32, FRAC_1_SQRT_2_F32, FRAC_2_PI_F32, FRAC_2_SQRT_PI_F32, FRAC_PI_2_F32,
    FRAC_PI_3_F32, FRAC_PI_4_F32, FRAC_PI_6_F32, FRAC_PI_8_F32, LN_2_F32, LN_10_F32, LOG2_E_F32,
    LOG10_E_F32, PI_F32, SQRT_2_F32, TAU_F32,
};
