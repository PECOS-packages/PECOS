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

//! `BitVec` arithmetic operations for arbitrary-precision integer arithmetic
//!
//! This module provides common arithmetic, bitwise, and comparison operations
//! for `BitVec` values used throughout the PECOS quantum computing stack.
//! All operations use two's complement representation for signed arithmetic.

pub mod arithmetic;
pub mod bitwise;
pub mod comparison;
pub mod conversion;
pub mod display;
pub mod utils;

// Re-export all public functions for convenience
pub use arithmetic::{add, divide, multiply, negate, subtract};
pub use bitwise::{shift_left, shift_left_extend, shift_right};
pub use comparison::{compare, compare_unsigned};
pub use conversion::{from_u32, parse_decimal_string, to_i32, to_i64, to_i128, to_u32, to_u64};
pub use display::{
    from_bitstring, to_binary_string, to_bitstring, to_bool_array, to_decimal_string, to_hex_string,
};
pub use utils::resize_to_same_width;
