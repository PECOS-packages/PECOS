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
//! This module handles array collection operations:
//! - Creation: `new_array`, `repeat`
//! - Access: `get`, `set`, `len`
//! - Modification: `push`, `pop`, `swap`

use log::debug;
use tket::hugr::ops::OpTrait;
use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle `collections.array` operations.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation // Array indices in simulation context won't exceed usize
    )]
    pub(crate) fn handle_array_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing collections.array operation: {op_name} at {node:?}");

        match op_name {
            "new_array" | "NewArray" => {
                // new_array: (T, ...) -> Array<T>
                // Collect all inputs into an array
                let op = hugr.get_optype(node);
                let num_inputs = op.dataflow_signature().map_or(0, |sig| sig.input_count());

                let mut elements = Vec::with_capacity(num_inputs);
                for port in 0..num_inputs {
                    if let Some(value) = self.get_input_value(hugr, node, port) {
                        elements.push(value);
                    }
                }

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Array(elements.clone()));

                debug!("new_array: created array with {} elements", elements.len());
                true
            }
            "get" | "Get" | "index" | "Index" => {
                // get: (Array<T>, int) -> T
                // Get element at index
                let array = self.get_input_value(hugr, node, 0);
                let index = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0) as usize;

                if let Some(ClassicalValue::Array(elements)) = array {
                    if let Some(element) = elements.get(index) {
                        self.wire_state
                            .classical_values
                            .insert((node, 0), element.clone());
                        debug!("array.get[{index}]: retrieved element");
                    } else {
                        debug!("array.get[{index}]: index out of bounds");
                    }
                }
                true
            }
            "set" | "Set" => {
                // set: (Array<T>, int, T) -> Array<T>
                // Set element at index
                let array = self.get_input_value(hugr, node, 0);
                let index = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0) as usize;
                let value = self.get_input_value(hugr, node, 2);

                if let (Some(ClassicalValue::Array(mut elements)), Some(new_value)) = (array, value)
                {
                    if index < elements.len() {
                        elements[index] = new_value;
                    }
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements));
                    debug!("array.set[{index}]: updated element");
                }
                true
            }
            "len" | "Len" | "length" | "Length" => {
                // len: Array<T> -> int
                // Get array length
                let array = self.get_input_value(hugr, node, 0);

                if let Some(ClassicalValue::Array(elements)) = array {
                    let len = elements.len() as u64;
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::UInt(len));
                    debug!("array.len: {len}");
                }
                true
            }
            "pop" | "Pop" => {
                // pop: Array<T> -> (Array<T>, T)
                // Remove and return the last element
                let array = self.get_input_value(hugr, node, 0);

                if let Some(ClassicalValue::Array(mut elements)) = array
                    && let Some(last) = elements.pop()
                {
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements));
                    self.wire_state.classical_values.insert((node, 1), last);
                    debug!("array.pop: removed last element");
                }
                true
            }
            "push" | "Push" => {
                // push: (Array<T>, T) -> Array<T>
                // Append element to array
                let array = self.get_input_value(hugr, node, 0);
                let value = self.get_input_value(hugr, node, 1);

                if let (Some(ClassicalValue::Array(mut elements)), Some(new_value)) = (array, value)
                {
                    elements.push(new_value);
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements));
                    debug!("array.push: appended element");
                }
                true
            }
            "repeat" | "Repeat" => {
                // repeat: (T, int) -> Array<T>
                // Create array with n copies of value
                let value = self.get_input_value(hugr, node, 0);
                let count = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0) as usize;

                if let Some(val) = value {
                    let elements = vec![val; count];
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements));
                    debug!("array.repeat: created array with {count} copies");
                }
                true
            }
            "swap" | "Swap" => {
                // swap: (Array<T>, int, int) -> Array<T>
                // Swap elements at two indices
                let array = self.get_input_value(hugr, node, 0);
                let i = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0) as usize;
                let j = self
                    .get_input_value(hugr, node, 2)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0) as usize;

                if let Some(ClassicalValue::Array(mut elements)) = array {
                    if i < elements.len() && j < elements.len() {
                        elements.swap(i, j);
                    }
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements));
                    debug!("array.swap[{i}, {j}]");
                }
                true
            }
            _ => {
                // For unknown array operations, try pass-through
                debug!("Unknown collections.array operation: {op_name} - attempting pass-through");
                self.propagate_all_inputs(hugr, node);
                true
            }
        }
    }
}
