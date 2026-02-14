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

//! Future/lazy measurement operations (`tket.futures`).
//!
//! This module handles operations on Future types, which represent
//! deferred measurement results:
//! - `Read`: Resolve a Future to its value
//! - `Dup`: Duplicate a Future handle
//! - `Free`: Discard a Future without reading

use log::debug;
use tket::hugr::{Hugr, Node};

use crate::engine::HugrEngine;
use crate::engine::types::{ClassicalValue, FutureState};

impl HugrEngine {
    /// Handle tket.futures operations.
    pub(crate) fn handle_futures_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.futures operation: {op_name} at {node:?}");

        match op_name {
            "Read" => {
                // Read: Future<T> -> T
                // Resolve the Future to its value
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let ClassicalValue::Future(future_id) = value
                    && let Some(state) = self.extension_state.futures.get(&future_id)
                {
                    match state {
                        FutureState::Resolved(outcome) => {
                            // Future is resolved, output the value
                            self.wire_state
                                .classical_values
                                .insert((node, 0), ClassicalValue::Bool(*outcome != 0));
                            debug!("Read future {future_id} -> {outcome}");
                        }
                        FutureState::Pending {
                            measurement_index, ..
                        } => {
                            // Check if measurement result is available
                            if let Some((_, qubit)) =
                                self.measurement_state.mappings.get(*measurement_index)
                            {
                                if let Some(&result) = self.measurement_state.results.get(qubit) {
                                    self.wire_state
                                        .classical_values
                                        .insert((node, 0), ClassicalValue::Bool(result != 0));
                                    debug!("Read future {future_id} from measurement -> {result}");
                                } else {
                                    // Result not yet available - use default
                                    self.wire_state
                                        .classical_values
                                        .insert((node, 0), ClassicalValue::Bool(false));
                                    debug!("Read future {future_id} pending, using default");
                                }
                            }
                        }
                    }
                }
                true
            }
            "Dup" => {
                // Dup: Future<T> -> (Future<T>, Future<T>)
                // Create two new Futures pointing to the same result
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let ClassicalValue::Future(original_id) = value
                {
                    // Create two new Future IDs that share the same state
                    let new_id1 = self.extension_state.next_future_id;
                    self.extension_state.next_future_id += 1;
                    let new_id2 = self.extension_state.next_future_id;
                    self.extension_state.next_future_id += 1;

                    // Copy the state to both new Futures
                    if let Some(state) = self.extension_state.futures.get(&original_id).cloned() {
                        self.extension_state.futures.insert(new_id1, state.clone());
                        self.extension_state.futures.insert(new_id2, state);
                    }

                    // Output both Futures
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Future(new_id1));
                    self.wire_state
                        .classical_values
                        .insert((node, 1), ClassicalValue::Future(new_id2));

                    debug!("Dup future {original_id} -> {new_id1}, {new_id2}");
                }
                true
            }
            "Free" => {
                // Free: Future<T> -> ()
                // Discard the Future without reading
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let ClassicalValue::Future(future_id) = value
                {
                    self.extension_state.futures.remove(&future_id);
                    debug!("Free future {future_id}");
                }
                true
            }
            _ => {
                debug!("Unknown tket.futures operation: {op_name}");
                false
            }
        }
    }
}
