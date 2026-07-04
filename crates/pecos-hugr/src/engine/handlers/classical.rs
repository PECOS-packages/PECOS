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

//! Classical computation operations.
//!
//! This module handles classical operations extracted from HUGR nodes:
//! - Logic operations (and, or, not, xor, eq)
//! - Integer arithmetic (iadd, isub, imul, idiv, imod, ineg, iabs)
//! - Integer comparisons (ieq, ine, ilt, ile, igt, ige)
//! - Integer bitwise operations (iand, ior, ixor, inot, ishl, ishr)
//! - Float arithmetic (fadd, fsub, fmul, fdiv, fneg, fabs, ffloor, fceil)
//! - Float comparisons (feq, fne, flt, fle, fgt, fge)
//! - Conversions (int<->float)
//! - Tuple operations (`make_tuple`, `unpack_tuple`)
//!
//! Also handles `tket.bool` extension operations.

use log::debug;
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::HugrEngine;
use crate::engine::handlers::{ClassicalOutcome, HandlerOutcome};
use crate::engine::types::{ClassicalOp, ClassicalOpType, ClassicalValue};

/// Mask for the low `2^log_width` bits (`log_width` >= 6 means full `i64`).
pub(crate) fn width_mask(log_width: u8) -> u64 {
    if log_width >= 6 {
        u64::MAX
    } else {
        (1u64 << (1u32 << log_width)) - 1
    }
}

/// Canonicalize a value to the op's width: mask to `2^log_width` bits and
/// sign-extend, so the stored i64 is the two's-complement value the width
/// implies (e.g. `int<5>` `0xFFFF_FFFF` stores as -1). Matches `ConstInt`
/// parsing, which stores `value_s()` (sign-extended) for every width.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub(crate) fn canonicalize_width(value: i64, log_width: u8) -> i64 {
    if log_width >= 6 {
        return value;
    }
    let bits = 1u32 << log_width;
    let masked = (value as u64) & width_mask(log_width);
    ((masked << (64 - bits)) as i64) >> (64 - bits)
}

impl HugrEngine {
    /// Execute a classical operation.
    ///
    /// Returns [`ClassicalOutcome::Outputs`] with (`port_index`, value)
    /// pairs on success, `Defer` when an input is missing or
    /// unconvertible, and `Fault` for unrecoverable spec-defined errors
    /// (e.g. unchecked division by zero).
    #[allow(
        clippy::too_many_lines,
        clippy::float_cmp, // Exact float comparison is intentional for feq/fne operations
        clippy::cast_precision_loss, // int->float conversion precision loss is expected
        clippy::cast_possible_truncation, // float->int truncation is intentional
        clippy::cast_sign_loss // shift amounts are clamped to 0-63 before cast to u32
    )]
    pub(crate) fn handle_classical_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op: &ClassicalOp,
    ) -> ClassicalOutcome {
        // Collect input values
        let mut inputs = Vec::with_capacity(op.num_inputs);
        for port_idx in 0..op.num_inputs {
            let in_port = IncomingPort::from(port_idx);
            if let Some((src_node, src_port)) = hugr.single_linked_output(node, in_port) {
                let wire_key = (src_node, src_port.index());
                if let Some(value) = self.wire_state.classical_values.get(&wire_key) {
                    inputs.push(value.clone());
                } else if matches!(op.op_type, ClassicalOpType::TagSum)
                    && let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&wire_key)
                {
                    // A Tag may carry linear payload elements (e.g. an
                    // iterator's Option over (qubit, state)): represent the
                    // qubit as a QubitRef so the Sum value materializes.
                    // Scoped to TagSum only -- a qubit input to an
                    // arithmetic op is a semantic error, not a value.
                    inputs.push(ClassicalValue::QubitRef(qubit_id));
                } else {
                    debug!(
                        "Classical op {node:?}: missing input value for port {port_idx} from {wire_key:?}"
                    );
                    return ClassicalOutcome::Defer;
                }
            } else {
                debug!("Classical op {node:?}: no source for input port {port_idx}");
                return ClassicalOutcome::Defer;
            }
        }

        // Multi-output / special arms first (they return whole port lists).
        match op.op_type {
            // Constants (shouldn't be processed as operations, but handle anyway)
            ClassicalOpType::ConstInt
            | ClassicalOpType::ConstFloat
            | ClassicalOpType::ConstBool => {
                return op
                    .const_value
                    .as_ref()
                    .map_or(ClassicalOutcome::Defer, |value| {
                        ClassicalOutcome::Outputs(vec![(0, value.clone())])
                    });
            }
            ClassicalOpType::MakeTuple => {
                // MakeTuple combines all inputs into a single tuple
                return ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Tuple(inputs))]);
            }
            ClassicalOpType::UnpackTuple => {
                // UnpackTuple takes a single tuple input and produces multiple
                // outputs. A tuple is a 1-variant sum, so accept a Sum payload
                // the same way (e.g. a tuple that crossed a CFG/Call boundary
                // as a tagged value).
                let tuple_value = inputs.into_iter().next();
                match tuple_value {
                    Some(
                        ClassicalValue::Tuple(elements)
                        | ClassicalValue::Sum {
                            values: elements, ..
                        },
                    ) => {
                        // Return each element on its respective output port
                        return ClassicalOutcome::Outputs(
                            elements.into_iter().enumerate().collect(),
                        );
                    }
                    Some(value) => {
                        // If it's a single non-tuple value, just pass it through on port 0
                        return ClassicalOutcome::Outputs(vec![(0, value)]);
                    }
                    None => return ClassicalOutcome::Defer,
                }
            }
            ClassicalOpType::LoadFunc => {
                // Resolve the static FuncDefn target into a function value.
                let Some(func_defn) = hugr.static_source(node) else {
                    return ClassicalOutcome::Fault(format!(
                        "LoadFunction at {node:?} has no static source"
                    ));
                };
                return ClassicalOutcome::Outputs(vec![(0, ClassicalValue::FuncRef(func_defn))]);
            }
            ClassicalOpType::TagSum => {
                // Tag wraps its inputs into the given variant of a sum.
                let OpType::Tag(tag_op) = hugr.get_optype(node) else {
                    // Classified as TagSum but not a Tag op: an engine
                    // invariant violation that no retry can repair.
                    return ClassicalOutcome::Fault(format!(
                        "node {node:?} classified as TagSum is not a Tag op"
                    ));
                };
                return ClassicalOutcome::Outputs(vec![(
                    0,
                    ClassicalValue::Sum {
                        tag: tag_op.tag,
                        values: inputs,
                    },
                )]);
            }
            ClassicalOpType::Idivmod => {
                // Combined Euclidean division+remainder, TWO outputs (q, r);
                // m=0 panics per the spec, like Idiv/Imod.
                let int_at = |i: usize| inputs.get(i).and_then(ClassicalValue::as_int);
                let uint_at = |i: usize| match inputs.get(i) {
                    Some(ClassicalValue::UInt(u)) => Some(*u),
                    #[allow(clippy::cast_sign_loss)]
                    other => other.and_then(ClassicalValue::as_int).map(|v| v as u64),
                };
                let signed = op.int_info.is_none_or(|(_, is_signed)| is_signed);
                let qr = if signed {
                    let (Some(n), Some(m)) = (int_at(0), uint_at(1)) else {
                        debug!("idivmod at {node:?}: inputs not ready, deferring");
                        return ClassicalOutcome::Defer;
                    };
                    let (n, m) = (i128::from(n), i128::from(m));
                    if m == 0 {
                        None
                    } else {
                        #[allow(clippy::cast_possible_truncation)]
                        Some((n.div_euclid(m) as i64, n.rem_euclid(m) as i64))
                    }
                } else {
                    let (Some(a), Some(b)) = (uint_at(0), uint_at(1)) else {
                        debug!("idivmod at {node:?}: inputs not ready, deferring");
                        return ClassicalOutcome::Defer;
                    };
                    #[allow(clippy::cast_possible_wrap)]
                    (b != 0).then(|| ((a / b) as i64, (a % b) as i64))
                };
                let Some((q, r)) = qr else {
                    return ClassicalOutcome::Fault(format!(
                        "division by zero at {node:?} (the HUGR spec defines m=0 as a panic)"
                    ));
                };
                return ClassicalOutcome::Outputs(vec![
                    (0, ClassicalValue::Int(q)),
                    (1, ClassicalValue::Int(r)),
                ]);
            }
            ClassicalOpType::IdivmodChecked => {
                // sum_with_error(tuple(q, r)): error = tag 0, value = tag 1.
                let int_at = |i: usize| inputs.get(i).and_then(ClassicalValue::as_int);
                let uint_at = |i: usize| match inputs.get(i) {
                    Some(ClassicalValue::UInt(u)) => Some(*u),
                    #[allow(clippy::cast_sign_loss)]
                    other => other.and_then(ClassicalValue::as_int).map(|v| v as u64),
                };
                let signed = op.int_info.is_none_or(|(_, is_signed)| is_signed);
                let qr = if signed {
                    let (Some(n), Some(m)) = (int_at(0), uint_at(1)) else {
                        debug!("idivmod_checked at {node:?}: inputs not ready, deferring");
                        return ClassicalOutcome::Defer;
                    };
                    let (n, m) = (i128::from(n), i128::from(m));
                    #[allow(clippy::cast_possible_truncation)]
                    (m != 0).then(|| (n.div_euclid(m) as i64, n.rem_euclid(m) as i64))
                } else {
                    let (Some(a), Some(b)) = (uint_at(0), uint_at(1)) else {
                        debug!("idivmod_checked at {node:?}: inputs not ready, deferring");
                        return ClassicalOutcome::Defer;
                    };
                    #[allow(clippy::cast_possible_wrap)]
                    (b != 0).then(|| ((a / b) as i64, (a % b) as i64))
                };
                let value = match qr {
                    Some((q, r)) => ClassicalValue::Sum {
                        tag: 1,
                        values: vec![ClassicalValue::Tuple(vec![
                            ClassicalValue::Int(q),
                            ClassicalValue::Int(r),
                        ])],
                    },
                    None => ClassicalValue::Sum {
                        tag: 0,
                        values: vec![],
                    },
                };
                return ClassicalOutcome::Outputs(vec![(0, value)]);
            }
            _ => {}
        }

        // Typed extraction for the scalar arms below: a PRESENT but
        // unconvertible input is the same hazard as a missing one --
        // defaulting (the old unwrap_or(0/false/0.0)) silently computes on
        // fabricated operands, so extraction failure defers the whole op.
        // Extraction is width-aware: the op's declared width (int_info)
        // masks unsigned reads and canonicalizes signed ones, so narrow
        // ints (int<5> = 32-bit etc.) compute exactly.
        let log_width = op.int_info.map_or(6, |(lw, _)| lw);
        let int = |i: usize| {
            inputs
                .get(i)
                .and_then(ClassicalValue::as_int)
                .map(|v| canonicalize_width(v, log_width))
        };
        // Unsigned ops reinterpret the stored i64 bit pattern: wrapping
        // arithmetic stores results through Int, so as_uint (which rejects
        // negatives) would spuriously defer on e.g. u64::MAX.
        #[allow(clippy::cast_sign_loss)]
        let uint = |i: usize| {
            match inputs.get(i) {
                Some(ClassicalValue::UInt(u)) => Some(*u),
                other => other.and_then(ClassicalValue::as_int).map(|v| v as u64),
            }
            .map(|v| v & width_mask(log_width))
        };
        let boolean = |i: usize| inputs.get(i).and_then(ClassicalValue::as_bool);
        let float = |i: usize| inputs.get(i).and_then(ClassicalValue::as_float);
        // Classified arithmetic.int ops carry their signedness; ops without
        // int_info (logic/float/etc.) never consult it.
        let signed = op.int_info.is_none_or(|(_, is_signed)| is_signed);

        // Execute the operation
        let mut div_by_zero = false;
        #[allow(clippy::cast_possible_wrap)]
        let result: Option<ClassicalValue> = (|| {
            Some(match op.op_type {
                // Logic operations
                ClassicalOpType::And => ClassicalValue::Bool(boolean(0)? && boolean(1)?),
                ClassicalOpType::Or => ClassicalValue::Bool(boolean(0)? || boolean(1)?),
                ClassicalOpType::Not => ClassicalValue::Bool(!boolean(0)?),
                ClassicalOpType::Xor => ClassicalValue::Bool(boolean(0)? ^ boolean(1)?),
                ClassicalOpType::Eq => ClassicalValue::Bool(boolean(0)? == boolean(1)?),

                // Integer arithmetic (add/sub/mul are sign-agnostic modulo 2^64)
                ClassicalOpType::Iadd => ClassicalValue::Int(int(0)?.wrapping_add(int(1)?)),
                ClassicalOpType::Isub => ClassicalValue::Int(int(0)?.wrapping_sub(int(1)?)),
                ClassicalOpType::Imul => ClassicalValue::Int(int(0)?.wrapping_mul(int(1)?)),
                ClassicalOpType::Idiv => {
                    // Division by zero panics per the spec ("m=0 will call
                    // panic"); raised as a fatal execution fault. Signed
                    // division is EUCLIDEAN per the HUGR spec (idivmod_s:
                    // q*m+r=n with 0<=r<m, unsigned divisor) -- computed in
                    // i128 so a divisor above i64::MAX is exact.
                    if signed {
                        let (n, m) = (i128::from(int(0)?), i128::from(uint(1)?));
                        if m == 0 {
                            div_by_zero = true;
                            return None;
                        }
                        ClassicalValue::Int(n.div_euclid(m) as i64)
                    } else {
                        let (a, b) = (uint(0)?, uint(1)?);
                        let Some(q) = a.checked_div(b) else {
                            div_by_zero = true;
                            return None;
                        };
                        ClassicalValue::Int(q as i64)
                    }
                }
                ClassicalOpType::Imod => {
                    // Euclidean remainder for signed (0 <= r < m), see Idiv;
                    // modulo by zero panics per the spec.
                    if signed {
                        let (n, m) = (i128::from(int(0)?), i128::from(uint(1)?));
                        if m == 0 {
                            div_by_zero = true;
                            return None;
                        }
                        ClassicalValue::Int(n.rem_euclid(m) as i64)
                    } else {
                        let (a, b) = (uint(0)?, uint(1)?);
                        let Some(r) = a.checked_rem(b) else {
                            div_by_zero = true;
                            return None;
                        };
                        ClassicalValue::Int(r as i64)
                    }
                }
                // Checked variants return sum_with_error(int): tag 1 wraps
                // the value, tag 0 is the error variant. The error payload
                // (a prelude error value) is not modeled -- correct programs
                // never take that branch, and one that does stalls loudly on
                // the missing payload instead of computing on a fabricated
                // value.
                ClassicalOpType::IdivChecked => {
                    // Signed checked division is Euclidean, like Idiv.
                    let ok = if signed {
                        let (n, m) = (i128::from(int(0)?), i128::from(uint(1)?));
                        (m != 0).then(|| n.div_euclid(m) as i64)
                    } else {
                        let (a, b) = (uint(0)?, uint(1)?);
                        a.checked_div(b).map(|q| q as i64)
                    };
                    match ok {
                        Some(q) => ClassicalValue::Sum {
                            tag: 1,
                            values: vec![ClassicalValue::Int(q)],
                        },
                        None => ClassicalValue::Sum {
                            tag: 0,
                            values: vec![],
                        },
                    }
                }
                ClassicalOpType::ImodChecked => {
                    // Euclidean remainder (0 <= r < m), like Imod.
                    let ok = if signed {
                        let (n, m) = (i128::from(int(0)?), i128::from(uint(1)?));
                        (m != 0).then(|| n.rem_euclid(m) as i64)
                    } else {
                        let (a, b) = (uint(0)?, uint(1)?);
                        a.checked_rem(b).map(|r| r as i64)
                    };
                    match ok {
                        Some(r) => ClassicalValue::Sum {
                            tag: 1,
                            values: vec![ClassicalValue::Int(r)],
                        },
                        None => ClassicalValue::Sum {
                            tag: 0,
                            values: vec![],
                        },
                    }
                }
                ClassicalOpType::Ineg => ClassicalValue::Int(int(0)?.wrapping_neg()),
                ClassicalOpType::Iabs => ClassicalValue::Int(int(0)?.wrapping_abs()),

                // Integer comparisons (ordering is signedness-sensitive)
                ClassicalOpType::Ieq => ClassicalValue::Bool(int(0)? == int(1)?),
                ClassicalOpType::Ine => ClassicalValue::Bool(int(0)? != int(1)?),
                ClassicalOpType::Ilt => ClassicalValue::Bool(if signed {
                    int(0)? < int(1)?
                } else {
                    uint(0)? < uint(1)?
                }),
                ClassicalOpType::Ile => ClassicalValue::Bool(if signed {
                    int(0)? <= int(1)?
                } else {
                    uint(0)? <= uint(1)?
                }),
                ClassicalOpType::Igt => ClassicalValue::Bool(if signed {
                    int(0)? > int(1)?
                } else {
                    uint(0)? > uint(1)?
                }),
                ClassicalOpType::Ige => ClassicalValue::Bool(if signed {
                    int(0)? >= int(1)?
                } else {
                    uint(0)? >= uint(1)?
                }),

                // Integer bitwise operations (sign-agnostic)
                ClassicalOpType::Iand => ClassicalValue::Int(int(0)? & int(1)?),
                ClassicalOpType::Ior => ClassicalValue::Int(int(0)? | int(1)?),
                ClassicalOpType::Ixor => ClassicalValue::Int(int(0)? ^ int(1)?),
                ClassicalOpType::Inot => ClassicalValue::Int(!int(0)?),
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                ClassicalOpType::Ishl => {
                    // "leftmost bits dropped, rightmost bits set to zero":
                    // shifting by k >= N drops every bit.
                    let k = uint(1)?;
                    let bits = u64::from(1u32 << log_width);
                    if k >= bits {
                        return Some(ClassicalValue::Int(0));
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    ClassicalValue::Int((uint(0)? << (k as u32)) as i64)
                }
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                ClassicalOpType::Ishr => {
                    // LOGICAL shift per the spec ("rightmost bits dropped,
                    // leftmost bits set to zero") -- ishr has no signed
                    // variant, and shifting by k >= N drops every bit.
                    let k = uint(1)?;
                    let bits = u64::from(1u32 << log_width);
                    if k >= bits {
                        return Some(ClassicalValue::Int(0));
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    ClassicalValue::Int((uint(0)? >> (k as u32)) as i64)
                }
                ClassicalOpType::Ipow => {
                    // "raise first input to the power of second input, the
                    // exponent is treated as an unsigned integer"; wrapping
                    // square-and-multiply so huge exponents stay exact under
                    // two's-complement wrap.
                    let (mut base, mut exp) = (int(0)?, uint(1)?);
                    let mut result: i64 = 1;
                    while exp > 0 {
                        if exp & 1 == 1 {
                            result = result.wrapping_mul(base);
                        }
                        base = base.wrapping_mul(base);
                        exp >>= 1;
                    }
                    ClassicalValue::Int(result)
                }
                ClassicalOpType::ItoBool => {
                    // itobool: int<1> -> bool (1 is true, 0 is false)
                    ClassicalValue::Bool(int(0)? != 0)
                }
                ClassicalOpType::IfromBool => {
                    // ifrombool: bool -> int<1>
                    ClassicalValue::Int(i64::from(boolean(0)?))
                }

                // Float arithmetic
                ClassicalOpType::Fadd => ClassicalValue::Float(float(0)? + float(1)?),
                ClassicalOpType::Fsub => ClassicalValue::Float(float(0)? - float(1)?),
                ClassicalOpType::Fmul => ClassicalValue::Float(float(0)? * float(1)?),
                ClassicalOpType::Fdiv => ClassicalValue::Float(float(0)? / float(1)?),
                ClassicalOpType::Fneg => ClassicalValue::Float(-float(0)?),
                ClassicalOpType::Fabs => ClassicalValue::Float(float(0)?.abs()),
                ClassicalOpType::Ffloor => ClassicalValue::Float(float(0)?.floor()),
                ClassicalOpType::Fceil => ClassicalValue::Float(float(0)?.ceil()),

                // Float comparisons (exact comparison is intentional)
                ClassicalOpType::Feq => ClassicalValue::Bool(float(0)? == float(1)?),
                ClassicalOpType::Fne => ClassicalValue::Bool(float(0)? != float(1)?),
                ClassicalOpType::Flt => ClassicalValue::Bool(float(0)? < float(1)?),
                ClassicalOpType::Fle => ClassicalValue::Bool(float(0)? <= float(1)?),
                ClassicalOpType::Fgt => ClassicalValue::Bool(float(0)? > float(1)?),
                ClassicalOpType::Fge => ClassicalValue::Bool(float(0)? >= float(1)?),

                #[allow(clippy::cast_precision_loss)]
                ClassicalOpType::ConvertIntToFloat => ClassicalValue::Float(if signed {
                    int(0)? as f64
                } else {
                    uint(0)? as f64
                }),
                #[allow(clippy::cast_possible_truncation)]
                ClassicalOpType::ConvertFloatToInt => {
                    // Truncate toward zero, matching standard float-to-int semantics
                    ClassicalValue::Int(float(0)?.trunc() as i64)
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                ClassicalOpType::ConvertFloatToIntChecked => {
                    // trunc_s/trunc_u: sum_with_error(int) -- error (tag 0)
                    // for NaN/infinite/out-of-range, value (tag 1) otherwise.
                    let f = float(0)?;
                    let t = f.trunc();
                    // Strict upper bounds: i64::MAX as f64 rounds UP to
                    // 2^63 (and u64::MAX to 2^64), so `<= MAX as f64` would
                    // accept one out-of-range value and saturate instead of
                    // taking the error branch. i64::MIN (-2^63) is exactly
                    // representable, so >= is correct there.
                    let in_range = if signed {
                        t.is_finite()
                            && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
                                .contains(&t)
                    } else {
                        t.is_finite() && (0.0..18_446_744_073_709_551_616.0).contains(&t)
                    };
                    if in_range {
                        let bits = if signed { t as i64 } else { (t as u64) as i64 };
                        ClassicalValue::Sum {
                            tag: 1,
                            values: vec![ClassicalValue::Int(bits)],
                        }
                    } else {
                        ClassicalValue::Sum {
                            tag: 0,
                            values: vec![],
                        }
                    }
                }

                // Handled by the early match above
                ClassicalOpType::ConstInt
                | ClassicalOpType::ConstFloat
                | ClassicalOpType::ConstBool
                | ClassicalOpType::MakeTuple
                | ClassicalOpType::UnpackTuple
                | ClassicalOpType::TagSum
                | ClassicalOpType::Idivmod
                | ClassicalOpType::IdivmodChecked
                | ClassicalOpType::LoadFunc => return None,
            })
        })();

        if let Some(value) = result {
            // Results of width-carrying ops store CANONICAL: masked to the
            // op's width and sign-extended (so wrapping at int<5> etc. is
            // exact, and every consumer sees the value the width implies).
            let value = match value {
                ClassicalValue::Int(v) if op.int_info.is_some() => {
                    ClassicalValue::Int(canonicalize_width(v, log_width))
                }
                other => other,
            };
            ClassicalOutcome::Outputs(vec![(0, value)])
        } else if div_by_zero {
            // The spec says unchecked division/modulo by zero panics.
            ClassicalOutcome::Fault(format!(
                "division by zero at {node:?} (the HUGR spec defines m=0 as a panic)"
            ))
        } else {
            debug!("Classical op {node:?}: input type mismatch, deferring");
            ClassicalOutcome::Defer
        }
    }

    /// Handle `tket.bool` operations.
    #[allow(clippy::too_many_lines)] // Boolean operation dispatch is inherently large
    pub(crate) fn handle_bool_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.bool operation: {op_name} at {node:?}");

        match op_name {
            // Binary/unary bool ops defer on missing or non-bool inputs
            // instead of fabricating `false`: a fabricated operand commits a
            // wrong branch value downstream (silent misexecution).
            "and" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool());
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool());
                let (Some(a), Some(b)) = (a, b) else {
                    debug!("tket.bool.and at {node:?}: deferring - input not ready");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a && b));
                debug!("tket.bool.and: {a} && {b} = {}", a && b);
                HandlerOutcome::Processed
            }
            "or" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool());
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool());
                let (Some(a), Some(b)) = (a, b) else {
                    debug!("tket.bool.or at {node:?}: deferring - input not ready");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a || b));
                debug!("tket.bool.or: {a} || {b} = {}", a || b);
                HandlerOutcome::Processed
            }
            "xor" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool());
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool());
                let (Some(a), Some(b)) = (a, b) else {
                    debug!("tket.bool.xor at {node:?}: deferring - input not ready");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a ^ b));
                debug!("tket.bool.xor: {a} ^ {b} = {}", a ^ b);
                HandlerOutcome::Processed
            }
            "not" => {
                let Some(a) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                else {
                    debug!("tket.bool.not at {node:?}: deferring - input not ready");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(!a));
                debug!("tket.bool.not: !{a} = {}", !a);
                HandlerOutcome::Processed
            }
            "eq" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool());
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool());
                let (Some(a), Some(b)) = (a, b) else {
                    debug!("tket.bool.eq at {node:?}: deferring - input not ready");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a == b));
                debug!("tket.bool.eq: {a} == {b} = {}", a == b);
                HandlerOutcome::Processed
            }
            "make_opaque" => {
                // make_opaque: Sum<bool> -> tket.bool
                // Convert Sum type to opaque bool
                let input_value = self.get_input_value(hugr, node, 0);
                debug!("tket.bool.make_opaque at {node:?}: input_value={input_value:?}");

                // If the input value is not available, defer this operation
                let Some(input_val) = input_value else {
                    debug!("tket.bool.make_opaque at {node:?}: deferring - input not ready");
                    // Track this node so it can be retried when input becomes available
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };

                // A present but non-bool value is the same hazard as a
                // missing one: fabricating `false` commits a wrong value.
                let Some(value) = input_val.as_bool() else {
                    debug!("tket.bool.make_opaque at {node:?}: deferring - input not a bool");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(value));
                debug!("tket.bool.make_opaque: {value}");
                HandlerOutcome::Processed
            }
            "read" => {
                // read: tket.bool -> Sum<bool>
                // Convert opaque bool to Sum type
                let input_value = self.get_input_value(hugr, node, 0);
                debug!("tket.bool.read at {node:?}: input_value={input_value:?}");

                // If the input value is not available (e.g., measurement result pending),
                // defer this operation by returning false. It will be retried later
                // when the measurement result is available.
                let Some(input_val) = input_value else {
                    debug!("tket.bool.read at {node:?}: deferring - input not ready");
                    // Track this node so it can be retried when measurement results arrive
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };

                let Some(value) = input_val.as_bool() else {
                    debug!("tket.bool.read at {node:?}: deferring - input not a bool");
                    self.pending_bool_reads.insert(node);
                    return HandlerOutcome::Defer;
                };

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(value));
                debug!("tket.bool.read: {value}");
                HandlerOutcome::Processed
            }
            _ => {
                debug!("Unknown tket.bool operation: {op_name}");
                HandlerOutcome::Defer
            }
        }
    }
}
