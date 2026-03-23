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

        // Execute the operation
        let result = match op.op_type {
            // Logic operations
            ClassicalOpType::And => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                ClassicalValue::Bool(a && b)
            }
            ClassicalOpType::Or => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                ClassicalValue::Bool(a || b)
            }
            ClassicalOpType::Not => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                ClassicalValue::Bool(!a)
            }
            ClassicalOpType::Xor => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                ClassicalValue::Bool(a ^ b)
            }
            ClassicalOpType::Eq => {
                // Eq can work on bools
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_bool)
                    .unwrap_or(false);
                ClassicalValue::Bool(a == b)
            }

            // Integer arithmetic
            ClassicalOpType::Iadd => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a.wrapping_add(b))
            }
            ClassicalOpType::Isub => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a.wrapping_sub(b))
            }
            ClassicalOpType::Imul => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a.wrapping_mul(b))
            }
            ClassicalOpType::Idiv => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(1);
                if b == 0 {
                    ClassicalValue::Int(0) // Avoid division by zero
                } else {
                    ClassicalValue::Int(a.wrapping_div(b))
                }
            }
            ClassicalOpType::Imod => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(1);
                if b == 0 {
                    ClassicalValue::Int(0)
                } else {
                    ClassicalValue::Int(a.wrapping_rem(b))
                }
            }
            ClassicalOpType::Ineg => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a.wrapping_neg())
            }
            ClassicalOpType::Iabs => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a.wrapping_abs())
            }

            // Integer comparisons
            ClassicalOpType::Ieq => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Bool(a == b)
            }
            ClassicalOpType::Ine => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Bool(a != b)
            }
            ClassicalOpType::Ilt => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Bool(a < b)
            }
            ClassicalOpType::Ile => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Bool(a <= b)
            }
            ClassicalOpType::Igt => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Bool(a > b)
            }
            ClassicalOpType::Ige => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Bool(a >= b)
            }

            // Integer bitwise operations
            ClassicalOpType::Iand => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a & b)
            }
            ClassicalOpType::Ior => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a | b)
            }
            ClassicalOpType::Ixor => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(a ^ b)
            }
            ClassicalOpType::Inot => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Int(!a)
            }
            ClassicalOpType::Ishl => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                // Clamp shift amount to valid range (0-63 for i64)
                let shift = b.clamp(0, 63) as u32;
                ClassicalValue::Int(a.wrapping_shl(shift))
            }
            ClassicalOpType::Ishr => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                let b = inputs.get(1).and_then(ClassicalValue::as_int).unwrap_or(0);
                // Clamp shift amount to valid range (0-63 for i64)
                let shift = b.clamp(0, 63) as u32;
                ClassicalValue::Int(a.wrapping_shr(shift))
            }

            // Float arithmetic
            ClassicalOpType::Fadd => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(a + b)
            }
            ClassicalOpType::Fsub => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(a - b)
            }
            ClassicalOpType::Fmul => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(a * b)
            }
            ClassicalOpType::Fdiv => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(1.0);
                ClassicalValue::Float(a / b)
            }
            ClassicalOpType::Fneg => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(-a)
            }
            ClassicalOpType::Fabs => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(a.abs())
            }
            ClassicalOpType::Ffloor => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(a.floor())
            }
            ClassicalOpType::Fceil => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Float(a.ceil())
            }

            // Float comparisons
            ClassicalOpType::Feq => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Bool(a == b)
            }
            ClassicalOpType::Fne => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Bool(a != b)
            }
            ClassicalOpType::Flt => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Bool(a < b)
            }
            ClassicalOpType::Fle => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Bool(a <= b)
            }
            ClassicalOpType::Fgt => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Bool(a > b)
            }
            ClassicalOpType::Fge => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                let b = inputs
                    .get(1)
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                ClassicalValue::Bool(a >= b)
            }

            // Conversions
            ClassicalOpType::ConvertIntToFloat => {
                let a = inputs.first().and_then(ClassicalValue::as_int).unwrap_or(0);
                ClassicalValue::Float(a as f64)
            }
            ClassicalOpType::ConvertFloatToInt => {
                let a = inputs
                    .first()
                    .and_then(ClassicalValue::as_float)
                    .unwrap_or(0.0);
                // Truncate toward zero, matching standard float-to-int semantics
                ClassicalValue::Int(a.trunc() as i64)
            }

            // Constants (shouldn't be processed as operations, but handle anyway)
            ClassicalOpType::ConstInt
            | ClassicalOpType::ConstFloat
            | ClassicalOpType::ConstBool => {
                if let Some(value) = &op.const_value {
                    value.clone()
                } else {
                    return vec![];
                }
            }

            // Tuple operations - these have special return handling
            ClassicalOpType::MakeTuple => {
                // MakeTuple combines all inputs into a single tuple
                // inputs already collected above
                return vec![(0, ClassicalValue::Tuple(inputs))];
            }
            ClassicalOpType::UnpackTuple => {
                // UnpackTuple takes a single tuple input and produces multiple outputs
                let tuple_value = inputs.into_iter().next();
                if let Some(ClassicalValue::Tuple(elements)) = tuple_value {
                    // Return each element on its respective output port
                    return elements.into_iter().enumerate().collect();
                } else if let Some(value) = tuple_value {
                    // If it's a single non-tuple value, just pass it through on port 0
                    return vec![(0, value)];
                }
                return vec![];
            }
        };

        // Return output on port 0
        vec![(0, result)]
    }

    /// Handle `tket.bool` operations.
    #[allow(clippy::too_many_lines)] // Boolean operation dispatch is inherently large
    pub(crate) fn handle_bool_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.bool operation: {op_name} at {node:?}");

        match op_name {
            "and" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a && b));
                debug!("tket.bool.and: {a} && {b} = {}", a && b);
                true
            }
            "or" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a || b));
                debug!("tket.bool.or: {a} || {b} = {}", a || b);
                true
            }
            "xor" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(a ^ b));
                debug!("tket.bool.xor: {a} ^ {b} = {}", a ^ b);
                true
            }
            "not" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(!a));
                debug!("tket.bool.not: !{a} = {}", !a);
                true
            }
            "eq" => {
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
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

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&node);

                let value = input_val.as_bool().unwrap_or(false);
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

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&node);

                let value = input_val.as_bool().unwrap_or(false);
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
