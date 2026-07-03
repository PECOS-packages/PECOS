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
use crate::engine::types::{ClassicalOp, ClassicalOpType, ClassicalValue};

impl HugrEngine {
    /// Execute a classical operation and return the output values.
    ///
    /// Returns a vector of (`port_index`, value) pairs for output ports.
    #[allow(
        clippy::too_many_lines,
        clippy::float_cmp, // Exact float comparison is intentional for feq/fne operations
        clippy::cast_precision_loss, // int->float conversion precision loss is expected
        clippy::cast_possible_truncation, // float->int truncation is intentional
        clippy::cast_sign_loss // shift amounts are clamped to 0-63 before cast to u32
    )]
    pub(crate) fn handle_classical_op(
        &self,
        hugr: &Hugr,
        node: Node,
        op: &ClassicalOp,
    ) -> Vec<(usize, ClassicalValue)> {
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
                    return vec![];
                }
            } else {
                debug!("Classical op {node:?}: no source for input port {port_idx}");
                return vec![];
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
                    .map(|value| vec![(0, value.clone())])
                    .unwrap_or_default();
            }
            ClassicalOpType::MakeTuple => {
                // MakeTuple combines all inputs into a single tuple
                return vec![(0, ClassicalValue::Tuple(inputs))];
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
                        return elements.into_iter().enumerate().collect();
                    }
                    Some(value) => {
                        // If it's a single non-tuple value, just pass it through on port 0
                        return vec![(0, value)];
                    }
                    None => return vec![],
                }
            }
            ClassicalOpType::TagSum => {
                // Tag wraps its inputs into the given variant of a sum.
                let OpType::Tag(tag_op) = hugr.get_optype(node) else {
                    debug!("TagSum at {node:?}: node is not a Tag op");
                    return vec![];
                };
                return vec![(
                    0,
                    ClassicalValue::Sum {
                        tag: tag_op.tag,
                        values: inputs,
                    },
                )];
            }
            _ => {}
        }

        // Typed extraction for the scalar arms below: a PRESENT but
        // unconvertible input is the same hazard as a missing one --
        // defaulting (the old unwrap_or(0/false/0.0)) silently computes on
        // fabricated operands, so extraction failure defers the whole op.
        let int = |i: usize| inputs.get(i).and_then(ClassicalValue::as_int);
        // Unsigned ops reinterpret the stored i64 bit pattern: wrapping
        // arithmetic stores results through Int, so as_uint (which rejects
        // negatives) would spuriously defer on e.g. u64::MAX.
        #[allow(clippy::cast_sign_loss)]
        let uint = |i: usize| {
            inputs
                .get(i)
                .and_then(ClassicalValue::as_int)
                .map(|v| v as u64)
        };
        let boolean = |i: usize| inputs.get(i).and_then(ClassicalValue::as_bool);
        let float = |i: usize| inputs.get(i).and_then(ClassicalValue::as_float);
        // Classified arithmetic.int ops carry their signedness; ops without
        // int_info (logic/float/etc.) never consult it.
        let signed = op.int_info.is_none_or(|(_, is_signed)| is_signed);

        // Execute the operation
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
                    // Division by zero yields 0 (the unchecked op's legacy
                    // behavior; proper error-Sum modeling for the _checked
                    // variants is tracked separately).
                    if signed {
                        let (a, b) = (int(0)?, int(1)?);
                        ClassicalValue::Int(if b == 0 { 0 } else { a.wrapping_div(b) })
                    } else {
                        let (a, b) = (uint(0)?, uint(1)?);
                        ClassicalValue::Int(a.checked_div(b).unwrap_or(0) as i64)
                    }
                }
                ClassicalOpType::Imod => {
                    if signed {
                        let (a, b) = (int(0)?, int(1)?);
                        ClassicalValue::Int(if b == 0 { 0 } else { a.wrapping_rem(b) })
                    } else {
                        let (a, b) = (uint(0)?, uint(1)?);
                        ClassicalValue::Int(a.checked_rem(b).unwrap_or(0) as i64)
                    }
                }
                // Checked variants return sum_with_error(int): tag 1 wraps
                // the value, tag 0 is the error variant. The error payload
                // (a prelude error value) is not modeled -- correct programs
                // never take that branch, and one that does stalls loudly on
                // the missing payload instead of computing on a fabricated
                // value.
                ClassicalOpType::IdivChecked => {
                    let ok = if signed {
                        let (a, b) = (int(0)?, int(1)?);
                        (b != 0).then(|| a.wrapping_div(b))
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
                    let ok = if signed {
                        let (a, b) = (int(0)?, int(1)?);
                        (b != 0).then(|| a.wrapping_rem(b))
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
                #[allow(clippy::cast_sign_loss)]
                ClassicalOpType::Ishl => {
                    let shift = int(1)?.clamp(0, 63) as u32;
                    ClassicalValue::Int(int(0)?.wrapping_shl(shift))
                }
                #[allow(clippy::cast_sign_loss)]
                ClassicalOpType::Ishr => {
                    // Arithmetic shift for signed, logical for unsigned
                    let shift = int(1)?.clamp(0, 63) as u32;
                    if signed {
                        ClassicalValue::Int(int(0)?.wrapping_shr(shift))
                    } else {
                        ClassicalValue::Int((uint(0)? >> shift) as i64)
                    }
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
                ClassicalOpType::ConvertIntToFloat => ClassicalValue::Float(int(0)? as f64),
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
                    let in_range = if signed {
                        t.is_finite() && t >= i64::MIN as f64 && t <= i64::MAX as f64
                    } else {
                        t.is_finite() && t >= 0.0 && t <= u64::MAX as f64
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
                | ClassicalOpType::TagSum => return None,
            })
        })();

        if let Some(value) = result {
            vec![(0, value)]
        } else {
            debug!("Classical op {node:?}: input type mismatch, deferring");
            vec![]
        }
    }

    /// Handle `tket.bool` operations.
    #[allow(clippy::too_many_lines)] // Boolean operation dispatch is inherently large
    pub(crate) fn handle_bool_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
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
                    return false;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a && b));
                debug!("tket.bool.and: {a} && {b} = {}", a && b);
                true
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
                    return false;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a || b));
                debug!("tket.bool.or: {a} || {b} = {}", a || b);
                true
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
                    return false;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a ^ b));
                debug!("tket.bool.xor: {a} ^ {b} = {}", a ^ b);
                true
            }
            "not" => {
                let Some(a) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                else {
                    debug!("tket.bool.not at {node:?}: deferring - input not ready");
                    self.pending_bool_reads.insert(node);
                    return false;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(!a));
                debug!("tket.bool.not: !{a} = {}", !a);
                true
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
                    return false;
                };
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a == b));
                debug!("tket.bool.eq: {a} == {b} = {}", a == b);
                true
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
                    return false;
                };

                // A present but non-bool value is the same hazard as a
                // missing one: fabricating `false` commits a wrong value.
                let Some(value) = input_val.as_bool() else {
                    debug!("tket.bool.make_opaque at {node:?}: deferring - input not a bool");
                    self.pending_bool_reads.insert(node);
                    return false;
                };

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(value));
                debug!("tket.bool.make_opaque: {value}");
                true
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
                    return false;
                };

                let Some(value) = input_val.as_bool() else {
                    debug!("tket.bool.read at {node:?}: deferring - input not a bool");
                    self.pending_bool_reads.insert(node);
                    return false;
                };

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&node);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(value));
                debug!("tket.bool.read: {value}");
                true
            }
            _ => {
                debug!("Unknown tket.bool operation: {op_name}");
                false
            }
        }
    }
}
