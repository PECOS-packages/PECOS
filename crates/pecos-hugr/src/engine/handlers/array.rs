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

//! Array operations (`collections.array`).
//!
//! Implements the HUGR std array ops with their spec signatures
//! (`hugr-core` `std_extensions/collections/array/array_op.rs`):
//! - `new_array<n, T>`: `[T; n] -> [array]`
//! - `unpack<n, T>`: `[array] -> [T; n]`
//! - `get<n, T>`: `[array, usize] -> [option<T>, array]`
//! - `set<n, T>`: `[array, usize, T] -> [either([T, array], [T, array])]`
//!   (tag 1 = success carrying the displaced element; tag 0 = out-of-bounds
//!   carrying the given element back so linear values are not lost)
//! - `swap<n, T>`: `[array, usize, usize] -> [either([array], [array])]`
//! - `pop_left`/`pop_right<n, T>`: `[array] -> [option<(T, array<n-1>)>]`
//! - `discard_empty<T>`: `[array<0, T>] -> []`
//!
//! Handlers defer (return false) on missing inputs so consumers never see
//! fabricated arrays or elements; unknown ops also defer so they surface in
//! the completion-time stall report instead of silently passing through.

use log::debug;
use tket::hugr::ops::OpTrait;
use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;
use crate::engine::handlers::HandlerOutcome;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle `collections.array` operations.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation // Array indices in simulation context won't exceed usize
    )]
    pub(crate) fn handle_array_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing collections.array operation: {op_name} at {node:?}");

        match op_name {
            "new_array" | "NewArray" => {
                // [T; n] -> [array]. A missing element defers the whole
                // construction: silently skipping it would shorten the array
                // and shift every index. Qubit elements ride as QubitRef.
                let op = hugr.get_optype(node);
                let num_inputs = op.dataflow_signature().map_or(0, |sig| sig.input_count());
                let mut elements = Vec::with_capacity(num_inputs);
                for port in 0..num_inputs {
                    if let Some(value) = self.get_input_value(hugr, node, port) {
                        elements.push(value);
                    } else if let Some(qubit_id) = self.get_input_qubit(hugr, node, port) {
                        elements.push(ClassicalValue::QubitRef(qubit_id));
                    } else {
                        debug!("new_array at {node:?}: element {port} not ready, deferring");
                        return HandlerOutcome::Defer;
                    }
                }

                debug!("new_array: created array with {} elements", elements.len());
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements));
                HandlerOutcome::Processed
            }
            "unpack" => {
                // [array] -> [T; n]: each element on its own output port.
                let Some(ClassicalValue::Array(elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("array.unpack at {node:?}: array not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                for (port, value) in elements.into_iter().enumerate() {
                    if let ClassicalValue::QubitRef(qubit_id) = &value {
                        self.wire_state
                            .wire_to_qubit
                            .insert((node, port), *qubit_id);
                    }
                    self.wire_state.classical_values.insert((node, port), value);
                }
                debug!("array.unpack at {node:?}: unpacked");
                HandlerOutcome::Processed
            }
            "get" | "Get" | "index" | "Index" => {
                // [array, usize] -> [option<T>, array]: option on port 0
                // (None = tag 0 for out-of-bounds), the array back on port 1.
                let Some(ClassicalValue::Array(elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("array.get at {node:?}: array not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(index) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("array.get at {node:?}: index not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                let result = match elements.get(index) {
                    Some(element) => ClassicalValue::Sum {
                        tag: 1,
                        values: vec![element.clone()],
                    },
                    None => ClassicalValue::Sum {
                        tag: 0,
                        values: vec![],
                    },
                };
                debug!("array.get[{index}]: {result:?}");
                self.wire_state.classical_values.insert((node, 0), result);
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::Array(elements));
                HandlerOutcome::Processed
            }
            "set" | "Set" => {
                // [array, usize, T] -> [either([T, array], [T, array])]:
                // tag 1 = success carrying the DISPLACED element and the
                // updated array; tag 0 = out-of-bounds carrying the given
                // element and the unchanged array (linear values survive).
                let Some(ClassicalValue::Array(mut elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("array.set at {node:?}: array not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(index) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("array.set at {node:?}: index not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let new_value = if let Some(qubit_id) = self.get_input_qubit(hugr, node, 2) {
                    ClassicalValue::QubitRef(qubit_id)
                } else if let Some(value) = self.get_input_value(hugr, node, 2) {
                    value
                } else {
                    debug!("array.set at {node:?}: value not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                let result = if let Some(slot) = elements.get_mut(index) {
                    let displaced = std::mem::replace(slot, new_value);
                    debug!("array.set[{index}]: element replaced");
                    ClassicalValue::Sum {
                        tag: 1,
                        values: vec![displaced, ClassicalValue::Array(elements)],
                    }
                } else {
                    debug!("array.set[{index}]: out of bounds (len={})", elements.len());
                    ClassicalValue::Sum {
                        tag: 0,
                        values: vec![new_value, ClassicalValue::Array(elements)],
                    }
                };
                self.wire_state.classical_values.insert((node, 0), result);
                HandlerOutcome::Processed
            }
            "swap" | "Swap" => {
                // [array, usize, usize] -> [either([array], [array])]:
                // tag 1 = success with the swapped array; tag 0 =
                // out-of-bounds with the unchanged array.
                let Some(ClassicalValue::Array(mut elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("array.swap at {node:?}: array not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(i) = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("array.swap at {node:?}: first index not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(j) = self
                    .get_input_value(hugr, node, 2)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize)
                else {
                    debug!("array.swap at {node:?}: second index not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                let result = if i < elements.len() && j < elements.len() {
                    elements.swap(i, j);
                    debug!("array.swap[{i},{j}]: swapped");
                    ClassicalValue::Sum {
                        tag: 1,
                        values: vec![ClassicalValue::Array(elements)],
                    }
                } else {
                    debug!(
                        "array.swap[{i},{j}]: out of bounds (len={})",
                        elements.len()
                    );
                    ClassicalValue::Sum {
                        tag: 0,
                        values: vec![ClassicalValue::Array(elements)],
                    }
                };
                self.wire_state.classical_values.insert((node, 0), result);
                HandlerOutcome::Processed
            }
            "pop_left" | "pop_right" => {
                // [array<n, T>] -> [option<(T, array<n-1, T>)>]: the empty
                // variant (tag 0) when nothing is left, else (element, rest)
                // in the value variant (tag 1).
                let Some(ClassicalValue::Array(mut elements)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("array.{op_name} at {node:?}: array not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let result = if elements.is_empty() {
                    debug!("array.{op_name} at {node:?}: empty array -> None variant");
                    ClassicalValue::Sum {
                        tag: 0,
                        values: vec![],
                    }
                } else {
                    let element = if op_name == "pop_left" {
                        elements.remove(0)
                    } else {
                        elements.pop().expect("non-empty checked above")
                    };
                    debug!(
                        "array.{op_name} at {node:?}: popped element, {} remain",
                        elements.len()
                    );
                    ClassicalValue::Sum {
                        tag: 1,
                        values: vec![element, ClassicalValue::Array(elements)],
                    }
                };
                self.wire_state.classical_values.insert((node, 0), result);
                HandlerOutcome::Processed
            }
            "discard_empty" => {
                // [array<0, T>] -> []: consume an empty array. Defer until
                // the value exists so the op is not marked done while its
                // producer is pending.
                if self.get_input_value(hugr, node, 0).is_none() {
                    debug!("array.discard_empty at {node:?}: array not ready, deferring");
                    return HandlerOutcome::Defer;
                }
                debug!("array.discard_empty: array consumed");
                HandlerOutcome::Processed
            }
            "scan" => self.handle_scan_op(hugr, node),
            _ => {
                // Unknown/unimplemented array op (e.g. `repeat`, whose real
                // signature takes a function value the engine cannot
                // execute): defer so it surfaces in the completion-time
                // stall report instead of silently passing values through
                // as if the op were an identity wire.
                debug!("Unknown collections.array operation: {op_name} at {node:?}, deferring");
                HandlerOutcome::Defer
            }
        }
    }
}
