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

//! Arithmetic operations (`arithmetic.float`, `arithmetic.int`, `arithmetic.conversions`).
//!
//! This module handles arithmetic extension operations:
//! - Float operations: transcendental functions, comparisons, special value checks
//! - Integer operations: arithmetic, bitwise, shifts, comparisons
//! - Conversions: int<->float type conversions

use log::debug;
use tket::hugr::{Hugr, Node};

use crate::engine::HugrEngine;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle `arithmetic.float` operations (transcendental functions, etc.).
    #[allow(clippy::too_many_lines)]
    pub(crate) fn handle_float_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing arithmetic.float operation: {op_name} at {node:?}");

        // Get input values
        let a = self
            .get_input_value(hugr, node, 0)
            .and_then(|v| v.as_float());
        let b = self
            .get_input_value(hugr, node, 1)
            .and_then(|v| v.as_float());

        let result = match op_name {
            // Basic arithmetic (may also be handled elsewhere, but include for completeness)
            "fadd" => a.zip(b).map(|(x, y)| x + y),
            "fsub" => a.zip(b).map(|(x, y)| x - y),
            "fmul" => a.zip(b).map(|(x, y)| x * y),
            "fdiv" => a.zip(b).map(|(x, y)| x / y),
            "fneg" => a.map(|x| -x),
            "fabs" => a.map(f64::abs),

            // Rounding operations
            "ffloor" => a.map(f64::floor),
            "fceil" => a.map(f64::ceil),
            "fround" => a.map(f64::round),
            "ftrunc" => a.map(f64::trunc),

            // Transcendental functions
            "fsqrt" | "sqrt" => a.map(f64::sqrt),
            "fexp" | "exp" => a.map(f64::exp),
            "fexp2" | "exp2" => a.map(f64::exp2),
            "flog" | "log" | "fln" | "ln" => a.map(f64::ln),
            "flog2" | "log2" => a.map(f64::log2),
            "flog10" | "log10" => a.map(f64::log10),

            // Trigonometric functions
            "fsin" | "sin" => a.map(f64::sin),
            "fcos" | "cos" => a.map(f64::cos),
            "ftan" | "tan" => a.map(f64::tan),
            "fasin" | "asin" => a.map(f64::asin),
            "facos" | "acos" => a.map(f64::acos),
            "fatan" | "atan" => a.map(f64::atan),
            "fatan2" | "atan2" => a.zip(b).map(|(y, x)| y.atan2(x)),

            // Hyperbolic functions
            "fsinh" | "sinh" => a.map(f64::sinh),
            "fcosh" | "cosh" => a.map(f64::cosh),
            "ftanh" | "tanh" => a.map(f64::tanh),
            "fasinh" | "asinh" => a.map(f64::asinh),
            "facosh" | "acosh" => a.map(f64::acosh),
            "fatanh" | "atanh" => a.map(f64::atanh),

            // Power and special functions
            "fpow" | "pow" => a.zip(b).map(|(x, y)| x.powf(y)),
            "fpowi" | "powi" => {
                let exp = self.get_input_value(hugr, node, 1).and_then(|v| v.as_int());
                // Clamp exponent to i32 range for powi
                #[allow(clippy::cast_possible_truncation)]
                a.zip(exp)
                    .map(|(x, n)| x.powi(n.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32))
            }
            "fhypot" | "hypot" => a.zip(b).map(|(x, y)| x.hypot(y)),

            // Comparison/special
            "fmin" | "min" => a.zip(b).map(|(x, y)| x.min(y)),
            "fmax" | "max" => a.zip(b).map(|(x, y)| x.max(y)),
            "fcopysign" | "copysign" => a.zip(b).map(|(x, y)| x.copysign(y)),

            // Fused multiply-add
            "ffma" | "fma" => {
                let c = self
                    .get_input_value(hugr, node, 2)
                    .and_then(|v| v.as_float());
                a.zip(b).zip(c).map(|((x, y), z)| x.mul_add(y, z))
            }

            // Float comparisons - exact comparison is intentional per HUGR semantics
            #[allow(clippy::float_cmp)]
            "feq" => a.zip(b).map(|(x, y)| if x == y { 1.0 } else { 0.0 }),
            #[allow(clippy::float_cmp)]
            "fne" => a.zip(b).map(|(x, y)| if x == y { 0.0 } else { 1.0 }),
            "flt" => a.zip(b).map(|(x, y)| if x < y { 1.0 } else { 0.0 }),
            "fle" => a.zip(b).map(|(x, y)| if x <= y { 1.0 } else { 0.0 }),
            "fgt" => a.zip(b).map(|(x, y)| if x > y { 1.0 } else { 0.0 }),
            "fge" => a.zip(b).map(|(x, y)| if x >= y { 1.0 } else { 0.0 }),

            // Check for special values
            "fis_nan" | "is_nan" => a.map(|x| if x.is_nan() { 1.0 } else { 0.0 }),
            "fis_inf" | "is_inf" => a.map(|x| if x.is_infinite() { 1.0 } else { 0.0 }),
            "fis_finite" | "is_finite" => a.map(|x| if x.is_finite() { 1.0 } else { 0.0 }),

            _ => {
                debug!("Unknown arithmetic.float operation: {op_name}");
                return false;
            }
        };

        if let Some(value) = result {
            self.wire_state
                .classical_values
                .insert((node, 0), ClassicalValue::Float(value));
            debug!("arithmetic.float.{op_name}: result = {value}");
        }

        true
    }

    /// Handle `arithmetic.int` operations (extended integer operations).
    #[allow(
        clippy::too_many_lines, // Large dispatch function with many integer operations
        clippy::cast_sign_loss, // shift amounts are clamped to 0-63 before cast to u32
        clippy::cast_possible_truncation // shift amounts are clamped before cast
    )]
    pub(crate) fn handle_int_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing arithmetic.int operation: {op_name} at {node:?}");

        // Get input values
        let a = self.get_input_value(hugr, node, 0).and_then(|v| v.as_int());
        let b = self.get_input_value(hugr, node, 1).and_then(|v| v.as_int());

        let result: Option<i64> = match op_name {
            // Basic arithmetic (may also be handled elsewhere)
            "iadd" => a.zip(b).map(|(x, y)| x.wrapping_add(y)),
            "isub" => a.zip(b).map(|(x, y)| x.wrapping_sub(y)),
            "imul" => a.zip(b).map(|(x, y)| x.wrapping_mul(y)),
            "idiv_s" | "idiv" => a.zip(b).map(|(x, y)| if y != 0 { x / y } else { 0 }),
            // Cast u64 result to i64 for unified storage - wrap is acceptable for large values
            #[allow(clippy::cast_possible_wrap)]
            "idiv_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu)
                    .map(|(x, y)| if y != 0 { (x / y) as i64 } else { 0 })
            }
            "imod_s" | "imod" => a.zip(b).map(|(x, y)| if y != 0 { x % y } else { 0 }),
            #[allow(clippy::cast_possible_wrap)]
            "imod_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu)
                    .map(|(x, y)| if y != 0 { (x % y) as i64 } else { 0 })
            }
            "ineg" => a.map(i64::wrapping_neg),
            "iabs" => a.map(i64::abs),

            // Bitwise operations
            "iand" => a.zip(b).map(|(x, y)| x & y),
            "ior" => a.zip(b).map(|(x, y)| x | y),
            "ixor" => a.zip(b).map(|(x, y)| x ^ y),
            "inot" => a.map(|x| !x),

            // Shift operations - clamp shift amount to valid range (0-63 for i64)
            "ishl" => a.zip(b).map(|(x, y)| x.wrapping_shl(y.clamp(0, 63) as u32)),
            "ishr_s" | "ishr" => a.zip(b).map(|(x, y)| x.wrapping_shr(y.clamp(0, 63) as u32)),
            #[allow(clippy::cast_possible_wrap)]
            "ishr_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                au.zip(b).map(|(x, y)| (x >> y.clamp(0, 63) as u32) as i64)
            }
            "irotl" | "rotl" => a.zip(b).map(|(x, y)| x.rotate_left(y.clamp(0, 63) as u32)),
            "irotr" | "rotr" => a.zip(b).map(|(x, y)| x.rotate_right(y.clamp(0, 63) as u32)),

            // Bit counting
            "ipopcnt" | "popcnt" | "popcount" => a.map(|x| i64::from(x.count_ones())),
            "iclz" | "clz" => a.map(|x| i64::from(x.leading_zeros())),
            "ictz" | "ctz" => a.map(|x| i64::from(x.trailing_zeros())),

            // Min/max
            "imin_s" | "imin" => a.zip(b).map(|(x, y)| x.min(y)),
            "imax_s" | "imax" => a.zip(b).map(|(x, y)| x.max(y)),
            #[allow(clippy::cast_possible_wrap)]
            "imin_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu).map(|(x, y)| x.min(y) as i64)
            }
            #[allow(clippy::cast_possible_wrap)]
            "imax_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu).map(|(x, y)| x.max(y) as i64)
            }

            // Sign extension / truncation - all no-ops for i64 unified storage
            #[allow(clippy::match_same_arms)] // Intentionally separate for clarity
            "iwiden_s" | "widen_s" => a, // Sign-extend (no-op for i64)
            #[allow(clippy::cast_possible_wrap)]
            "iwiden_u" | "widen_u" => self
                .get_input_value(hugr, node, 0)
                .and_then(|v| v.as_uint())
                .map(|x| x as i64),
            #[allow(clippy::match_same_arms)]
            "inarrow_s" | "narrow_s" => a, // Truncate (no-op for now)
            #[allow(clippy::match_same_arms)]
            "inarrow_u" | "narrow_u" => a, // Truncate (no-op for now)

            // Comparisons (return 0 or 1)
            "ieq" => a.zip(b).map(|(x, y)| i64::from(x == y)),
            "ine" => a.zip(b).map(|(x, y)| i64::from(x != y)),
            "ilt_s" | "ilt" => a.zip(b).map(|(x, y)| i64::from(x < y)),
            "ile_s" | "ile" => a.zip(b).map(|(x, y)| i64::from(x <= y)),
            "igt_s" | "igt" => a.zip(b).map(|(x, y)| i64::from(x > y)),
            "ige_s" | "ige" => a.zip(b).map(|(x, y)| i64::from(x >= y)),
            "ilt_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu).map(|(x, y)| i64::from(x < y))
            }
            "ile_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu).map(|(x, y)| i64::from(x <= y))
            }
            "igt_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu).map(|(x, y)| i64::from(x > y))
            }
            "ige_u" => {
                let au = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint());
                let bu = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint());
                au.zip(bu).map(|(x, y)| i64::from(x >= y))
            }

            _ => {
                debug!("Unknown arithmetic.int operation: {op_name}");
                return false;
            }
        };

        if let Some(value) = result {
            self.wire_state
                .classical_values
                .insert((node, 0), ClassicalValue::Int(value));
            debug!("arithmetic.int.{op_name}: result = {value}");
        }

        true
    }

    /// Handle `arithmetic.conversions` operations (int/float conversions).
    ///
    /// Type conversion casts are intentional and match HUGR/Guppy semantics:
    /// - `cast_precision_loss`: i64/u64 to f64 conversion may lose precision for large integers
    /// - `cast_possible_truncation`: f64 to integer conversion truncates fractional part
    /// - `cast_sign_loss`: f64 to u64 is safe because we clamp to non-negative first
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(crate) fn handle_conversions_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing arithmetic.conversions operation: {op_name} at {node:?}");

        match op_name {
            // Integer to float conversions
            "convert_s" | "itof_s" => {
                // Signed integer to float
                if let Some(value) = self.get_input_value(hugr, node, 0).and_then(|v| v.as_int()) {
                    let result = value as f64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Float(result));
                    debug!("convert_s: {value} -> {result}");
                }
            }
            "convert_u" | "itof_u" => {
                // Unsigned integer to float
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint())
                {
                    let result = value as f64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Float(result));
                    debug!("convert_u: {value} -> {result}");
                }
            }

            // Float to integer conversions (truncate toward zero)
            "trunc_s" | "ftoi_s" => {
                // Float to signed integer (truncate)
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let result = value.trunc() as i64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Int(result));
                    debug!("trunc_s: {value} -> {result}");
                }
            }
            "trunc_u" | "ftoi_u" => {
                // Float to unsigned integer (truncate)
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    // Clamp to non-negative before converting
                    let clamped = value.max(0.0).trunc();
                    let result = clamped as u64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::UInt(result));
                    debug!("trunc_u: {value} -> {result}");
                }
            }

            // Ceiling/floor variants
            "ceil_s" => {
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let result = value.ceil() as i64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Int(result));
                    debug!("ceil_s: {value} -> {result}");
                }
            }
            "ceil_u" => {
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let clamped = value.max(0.0).ceil();
                    let result = clamped as u64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::UInt(result));
                    debug!("ceil_u: {value} -> {result}");
                }
            }
            "floor_s" => {
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let result = value.floor() as i64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Int(result));
                    debug!("floor_s: {value} -> {result}");
                }
            }
            "floor_u" => {
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let clamped = value.max(0.0).floor();
                    let result = clamped as u64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::UInt(result));
                    debug!("floor_u: {value} -> {result}");
                }
            }

            // Rounding
            "round_s" => {
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let result = value.round() as i64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Int(result));
                    debug!("round_s: {value} -> {result}");
                }
            }
            "round_u" => {
                if let Some(value) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                {
                    let clamped = value.max(0.0).round();
                    let result = clamped as u64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::UInt(result));
                    debug!("round_u: {value} -> {result}");
                }
            }

            _ => {
                debug!("Unknown arithmetic.conversions operation: {op_name}");
                return false;
            }
        }

        true
    }
}
