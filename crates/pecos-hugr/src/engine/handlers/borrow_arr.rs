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
//! This module handles borrow-checked array operations emitted by Guppy
//! for array element access with ownership tracking. At simulation time,
//! borrow tracking is irrelevant -- we just need array access semantics.
//!
//! Operations:
//! - `new_all_borrowed`: Create an empty borrow array
//! - `borrow`: Extract element at index from array
//! - `return`: Put element back into array
//! - `discard_all_borrowed`: Finalize/cleanup (no-op for simulation)

use log::debug;
use tket::hugr::ops::OpTrait;
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
                // new_all_borrowed: create an empty borrow array.
                // The actual elements get populated via subsequent `return` operations.
                // Input port 0 = the original array to borrow from.
                // Output port 0 = borrow array (initially clone the input array).
                let op = hugr.get_optype(node);
                let num_inputs = op.dataflow_signature().map_or(0, |sig| sig.input_count());

                if num_inputs > 0 {
                    // If there's an input array, use it as the borrow array
                    if let Some(array_val) = self.get_input_value(hugr, node, 0) {
                        self.wire_state
                            .classical_values
                            .insert((node, 0), array_val);
                        debug!("new_all_borrowed: cloned input array as borrow array");
                    } else {
                        // No input value found; create empty array
                        self.wire_state
                            .classical_values
                            .insert((node, 0), ClassicalValue::Array(vec![]));
                        debug!("new_all_borrowed: created empty borrow array (no input)");
                    }
                } else {
                    // No inputs; create empty array
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(vec![]));
                    debug!("new_all_borrowed: created empty borrow array");
                }

                // Propagate qubit wires if present
                for port in 0..num_inputs {
                    if let Some(qubit) = self.get_input_qubit(hugr, node, port) {
                        self.wire_state.wire_to_qubit.insert((node, port), qubit);
                    }
                }

                true
            }
            "borrow" => {
                // borrow: extract element at index from array.
                // Input port 0 = borrow_array, port 1 = int index
                // Output port 0 = borrow_array (unchanged for simulation), port 1 = element
                let array = self.get_input_value(hugr, node, 0);
                #[allow(clippy::cast_possible_truncation)] // Array indices fit in usize
                let index = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0) as usize;

                if let Some(ClassicalValue::Array(elements)) = array {
                    // Output port 0 = the array (unchanged for simulation purposes)
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements.clone()));

                    if let Some(element) = elements.get(index).cloned() {
                        // Output port 1 = the borrowed element
                        // If element is a QubitRef, also propagate to wire_to_qubit
                        if let ClassicalValue::QubitRef(qubit_id) = &element {
                            self.wire_state.wire_to_qubit.insert((node, 1), *qubit_id);
                        }
                        self.wire_state.classical_values.insert((node, 1), element);

                        debug!("borrow[{index}]: extracted element");
                    } else {
                        debug!(
                            "borrow[{index}]: index out of bounds (len={})",
                            elements.len()
                        );
                    }
                } else {
                    debug!("borrow: no array found on input port 0");
                    // Try pass-through
                    self.propagate_all_inputs(hugr, node);
                }
                true
            }
            "return" => {
                // return: put element back into array.
                // Input port 0 = borrow_array, port 1 = index (usize), port 2 = element
                // Output port 0 = updated array
                let array = self.get_input_value(hugr, node, 0);
                #[allow(clippy::cast_possible_truncation)] // Array indices fit in usize
                let index = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_uint())
                    .map(|v| v as usize);
                let element = self.get_input_value(hugr, node, 2);

                // Also check if the element is a qubit
                let element_qubit = self.get_input_qubit(hugr, node, 2);

                // If the element isn't available yet (e.g., waiting for measurement result),
                // defer this operation. Return false so the main loop adds us to pending.
                if element.is_none() && element_qubit.is_none() {
                    debug!("return: element not available yet, deferring");
                    return false;
                }

                if let Some(ClassicalValue::Array(mut elements)) = array {
                    if let Some(val) = element {
                        // The element might need to be wrapped as QubitRef
                        let val = if let Some(qubit_id) = element_qubit {
                            ClassicalValue::QubitRef(qubit_id)
                        } else {
                            val
                        };
                        // Put the element at the specified index
                        if let Some(idx) = index {
                            // Extend array if needed
                            while elements.len() <= idx {
                                elements.push(ClassicalValue::Bool(false));
                            }
                            elements[idx] = val;
                            debug!("return[{idx}]: element returned to borrow array");
                        } else {
                            // No index -- append to array
                            elements.push(val);
                            debug!("return: element appended to borrow array");
                        }
                    } else if let Some(qubit_id) = element_qubit {
                        // Element is a qubit (no classical value, just qubit wire)
                        let val = ClassicalValue::QubitRef(qubit_id);
                        if let Some(idx) = index {
                            while elements.len() <= idx {
                                elements.push(ClassicalValue::Bool(false));
                            }
                            elements[idx] = val;
                            debug!("return[{idx}]: qubit returned to borrow array");
                        } else {
                            elements.push(val);
                            debug!("return: qubit appended to borrow array");
                        }
                    }

                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Array(elements));
                } else {
                    // Array not available - might need to defer
                    if array.is_none() {
                        debug!("return: array not available yet, deferring");
                        return false;
                    }
                    debug!("return: no array found on input port 0, passing through");
                    self.propagate_all_inputs(hugr, node);
                }
                true
            }
            "discard_all_borrowed" => {
                // discard_all_borrowed: finalize/cleanup.
                // Input port 0 = borrow_array, Output port 0 = the original array
                // For simulation, we just pass through the array value.
                let array = self.get_input_value(hugr, node, 0);
                // eprintln!("[BORROW_ARR] discard_all_borrowed at {node:?}: array={array:?}");
                if let Some(arr) = array {
                    // eprintln!("[BORROW_ARR] discard_all_borrowed: propagating {:?}", arr);
                    debug!("discard_all_borrowed: propagating array value {arr:?}");
                    self.wire_state.classical_values.insert((node, 0), arr);
                } else {
                    // Array not available yet - defer until it's ready
                    // eprintln!("[BORROW_ARR] discard_all_borrowed: deferring (array not available)");
                    debug!("discard_all_borrowed: array not available yet, deferring");
                    return false;
                }
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
