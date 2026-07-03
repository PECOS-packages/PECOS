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

//! Borrow array operations (`collections.borrow_arr`).
//!
//! This module handles borrow-checked array operations emitted by Guppy for
//! array element access with ownership tracking (`qs[i]`, array
//! comprehensions, `discard_array`). Arrays are represented as
//! [`ClassicalValue::Array`] whose slots hold element values (qubits as
//! [`ClassicalValue::QubitRef`]) or [`ClassicalValue::Borrowed`] holes.
//!
//! Operations (signatures per the HUGR std extension):
//! - `new_all_borrowed<n, T>`: `[] -> [array]` -- n slots, all holes
//! - `borrow<n, T>`: `[array, usize] -> [array, elem]` -- take slot, leave hole
//! - `return<n, T>`: `[array, usize, elem] -> [array]` -- fill hole
//! - `is_borrowed<n, T>`: `[array, usize] -> [array, bool]`
//! - `discard_all_borrowed<n, T>`: `[array] -> []` -- consume the array
//!
//! Handlers defer (return false) on missing inputs so consumers never see
//! fabricated arrays or elements; a permanently missing input surfaces via
//! completion-time stall detection.

use log::debug;
use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle `collections.borrow_arr` operations.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn handle_borrow_arr_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing collections.borrow_arr operation: {op_name} at {node:?}");

        match op_name {
            "new_all_borrowed" => {
                // [] -> [array]: n slots, all borrowed-out (holes). The size
                // comes from the op's first type arg: a concrete BoundedNat,
                // or (inside a generic function) a type variable resolved
                // through the active call chain. Defer if unresolvable -- a
                // fabricated 0-slot array makes every later return/borrow
                // out of bounds.
                let op = hugr.get_optype(node);
                let size = op.as_extension_op().and_then(|ext| {
                    ext.args().iter().find_map(|arg| match arg {
                        tket::hugr::types::TypeArg::BoundedNat(n) => Some(*n),
                        tket::hugr::types::TypeArg::Variable(var) => {
                            self.resolve_call_type_arg(hugr, node, var.index())
                        }
                        _ => None,
                    })
                });
                let Some(size) = size else {
                    debug!("new_all_borrowed at {node:?}: size unresolved, deferring");
                    return false;
                };
                #[allow(clippy::cast_possible_truncation)] // Array sizes fit in usize
                let elements = vec![ClassicalValue::Borrowed; size as usize];
                debug!("new_all_borrowed: created {size}-slot all-borrowed array");
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements));
                true
            }
            "borrow" => {
                // [array, usize] -> [array, elem]: take the element at the
                // index, leaving a hole.
                let Some(ClassicalValue::Array(mut elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("borrow at {node:?}: array not ready, deferring");
                    return false;
                };
                #[allow(clippy::cast_possible_truncation)] // Array indices fit in usize
                let Some(index) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("borrow at {node:?}: index not ready, deferring");
                    return false;
                };

                let Some(slot) = elements.get_mut(index) else {
                    debug!(
                        "borrow at {node:?}: index {index} out of bounds (len={}), deferring",
                        elements.len()
                    );
                    return false;
                };
                let element = std::mem::replace(slot, ClassicalValue::Borrowed);
                if matches!(element, ClassicalValue::Borrowed) {
                    // Guppy guards accesses with is_borrowed + panic, so a
                    // hole here means an upstream value was wrong; defer so
                    // stall detection names this node instead of handing a
                    // fabricated element downstream.
                    debug!("borrow at {node:?}: slot {index} already borrowed, deferring");
                    return false;
                }

                if let ClassicalValue::QubitRef(qubit_id) = &element {
                    self.wire_state.wire_to_qubit.insert((node, 1), *qubit_id);
                }
                self.wire_state.classical_values.insert((node, 1), element);
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements));
                debug!("borrow[{index}]: extracted element");
                true
            }
            "return" => {
                // [array, usize, elem] -> [array]: fill the hole at the index.
                let Some(ClassicalValue::Array(mut elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("return at {node:?}: array not ready, deferring");
                    return false;
                };
                #[allow(clippy::cast_possible_truncation)] // Array indices fit in usize
                let Some(index) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("return at {node:?}: index not ready, deferring");
                    return false;
                };
                // A qubit element arrives on the qubit wire; other elements
                // as classical values.
                let element = if let Some(qubit_id) = self.get_input_qubit(hugr, node, 2) {
                    ClassicalValue::QubitRef(qubit_id)
                } else if let Some(value) = self.get_input_value(hugr, node, 2) {
                    value
                } else {
                    debug!("return at {node:?}: element not ready, deferring");
                    return false;
                };

                let Some(slot) = elements.get_mut(index) else {
                    debug!(
                        "return at {node:?}: index {index} out of bounds (len={}), deferring",
                        elements.len()
                    );
                    return false;
                };
                *slot = element;
                debug!("return[{index}]: element returned to borrow array");
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements));
                true
            }
            "is_borrowed" => {
                // [array, usize] -> [array, bool]
                let Some(ClassicalValue::Array(elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("is_borrowed at {node:?}: array not ready, deferring");
                    return false;
                };
                #[allow(clippy::cast_possible_truncation)] // Array indices fit in usize
                let Some(index) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("is_borrowed at {node:?}: index not ready, deferring");
                    return false;
                };

                let borrowed = elements
                    .get(index)
                    .is_none_or(|slot| matches!(slot, ClassicalValue::Borrowed));
                debug!("is_borrowed[{index}]: {borrowed}");
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements));
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::Bool(borrowed));
                true
            }
            "get" => {
                // [array, usize] -> [Sum([[], [T]]), array]: copy the element
                // at the index (copyable element types only -- no hole is
                // left). Out-of-bounds or borrowed slots yield the None
                // variant, matching the std extension's option result.
                let Some(ClassicalValue::Array(elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("get at {node:?}: array not ready, deferring");
                    return false;
                };
                #[allow(clippy::cast_possible_truncation)] // Array indices fit in usize
                let Some(index) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("get at {node:?}: index not ready, deferring");
                    return false;
                };

                let result = match elements.get(index) {
                    Some(slot) if !matches!(slot, ClassicalValue::Borrowed) => {
                        ClassicalValue::Sum {
                            tag: 1,
                            values: vec![slot.clone()],
                        }
                    }
                    _ => ClassicalValue::Sum {
                        tag: 0,
                        values: vec![],
                    },
                };
                debug!("get[{index}]: {result:?}");
                self.wire_state.classical_values.insert((node, 0), result);
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::Array(elements));
                true
            }
            "new_array" => {
                // [T; n] -> [array]: construct from elements. A missing
                // element defers (skipping would shorten the array and shift
                // every index); qubit elements ride as QubitRef.
                use tket::hugr::ops::OpTrait;
                let op = hugr.get_optype(node);
                let num_inputs = op.dataflow_signature().map_or(0, |sig| sig.input_count());
                let mut elements = Vec::with_capacity(num_inputs);
                for port in 0..num_inputs {
                    if let Some(qubit_id) = self.get_input_qubit(hugr, node, port) {
                        elements.push(ClassicalValue::QubitRef(qubit_id));
                    } else if let Some(value) = self.get_input_value(hugr, node, port) {
                        elements.push(value);
                    } else {
                        debug!(
                            "borrow_arr.new_array at {node:?}: element {port} not ready, deferring"
                        );
                        return false;
                    }
                }
                debug!("borrow_arr.new_array: created {} elements", elements.len());
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements));
                true
            }
            "pop_left" => {
                // [array<n,T>] -> [Sum([[], [T, array<n-1,T>]])]: take the
                // first element; the empty variant (tag 0) when nothing is
                // left, else (element, rest) in the value variant (tag 1).
                let Some(ClassicalValue::Array(mut elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("pop_left at {node:?}: array not ready, deferring");
                    return false;
                };
                let result = if elements.is_empty() {
                    debug!("pop_left at {node:?}: empty array -> None variant");
                    ClassicalValue::Sum {
                        tag: 0,
                        values: vec![],
                    }
                } else {
                    let element = elements.remove(0);
                    debug!(
                        "pop_left at {node:?}: popped element, {} remain",
                        elements.len()
                    );
                    ClassicalValue::Sum {
                        tag: 1,
                        values: vec![element, ClassicalValue::Array(elements)],
                    }
                };
                self.wire_state.classical_values.insert((node, 0), result);
                true
            }
            "discard_empty" => {
                // [array<0,T>] -> []: consume an empty array. Defer until
                // the value exists so the op is not marked done while its
                // producer is pending.
                if self.get_input_value(hugr, node, 0).is_none() {
                    debug!("discard_empty at {node:?}: array not ready, deferring");
                    return false;
                }
                debug!("discard_empty: array consumed");
                true
            }
            "discard_all_borrowed" => {
                // [array] -> []: consumes the (all-borrowed) array; nothing
                // to produce. Defer until the array value exists so the op
                // is not marked done while its producer is still pending.
                if self.get_input_value(hugr, node, 0).is_none() {
                    debug!("discard_all_borrowed at {node:?}: array not ready, deferring");
                    return false;
                }
                debug!("discard_all_borrowed: array consumed");
                true
            }
            _ => {
                // For unknown borrow_arr operations, try pass-through
                debug!(
                    "Unknown collections.borrow_arr operation: {op_name} - attempting pass-through"
                );
                self.propagate_all_inputs(hugr, node);
                true
            }
        }
    }
}
