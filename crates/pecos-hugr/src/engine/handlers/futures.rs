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
use crate::engine::handlers::HandlerOutcome;
use crate::engine::types::{ClassicalValue, FutureState};

impl HugrEngine {
    /// Handle `tket.measurement` operations emitted by current Guppy versions.
    pub(crate) fn handle_measurement_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        match op_name {
            "Read" => {
                let Some(input) = self.get_input_value(hugr, node, 0) else {
                    debug!("measurement.Read at {node:?}: result not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let value = match input {
                    ClassicalValue::Bool(value) => value,
                    ClassicalValue::Future(future_id) => {
                        let Some(state) = self.extension_state.futures.get(&future_id).cloned()
                        else {
                            debug!(
                                "measurement.Read at {node:?}: unknown future {future_id}, deferring"
                            );
                            return HandlerOutcome::Defer;
                        };
                        match state {
                            FutureState::Resolved {
                                outcome,
                                int_valued: false,
                            } => outcome != 0,
                            FutureState::Resolved {
                                int_valued: true, ..
                            } => {
                                debug!(
                                    "measurement.Read at {node:?}: future has non-Boolean output, deferring"
                                );
                                return HandlerOutcome::Defer;
                            }
                            FutureState::Pending {
                                measurement_index,
                                int_valued: false,
                                ..
                            } => {
                                let Some(&outcome) =
                                    self.measurement_state.outcomes.get(&measurement_index)
                                else {
                                    debug!(
                                        "measurement.Read at {node:?}: future result not ready, deferring"
                                    );
                                    return HandlerOutcome::Defer;
                                };
                                outcome != 0
                            }
                            FutureState::Pending {
                                int_valued: true, ..
                            } => {
                                debug!(
                                    "measurement.Read at {node:?}: future has non-Boolean output, deferring"
                                );
                                return HandlerOutcome::Defer;
                            }
                        }
                    }
                    _ => {
                        debug!("measurement.Read at {node:?}: input is not a Boolean, deferring");
                        return HandlerOutcome::Defer;
                    }
                };
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Bool(value));
                // A Conditional may expand as soon as its tag is available,
                // before its other measurement-derived inputs resolve. Copy
                // this newly available value into any such active Case.
                self.repropagate_active_case_inputs(hugr);
                HandlerOutcome::Processed
            }
            _ => HandlerOutcome::Defer,
        }
    }

    /// Handle tket.futures operations.
    pub(crate) fn handle_futures_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.futures operation: {op_name} at {node:?}");

        match op_name {
            "Read" => {
                // Read: Future<T> -> T. Every unresolved shape defers: a
                // missing input, a non-Future value, an unknown future id,
                // or a pending measurement all mean the value does not exist
                // yet -- returning handled would mark the node processed
                // with no output and strand every consumer.
                let Some(ClassicalValue::Future(future_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("futures.Read at {node:?}: future not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(state) = self.extension_state.futures.get(&future_id).cloned() else {
                    debug!("futures.Read at {node:?}: unknown future {future_id}, deferring");
                    return HandlerOutcome::Defer;
                };
                // Produce the future's DECLARED type: Future<bool> reads a
                // Bool; Future<int> (LazyMeasureLeaked, 0/1/2-leaked) reads
                // an Int -- a Bool there loses the leak value.
                let to_value = |outcome: u32, int_valued: bool| {
                    if int_valued {
                        ClassicalValue::Int(i64::from(outcome))
                    } else {
                        ClassicalValue::Bool(outcome != 0)
                    }
                };
                match state {
                    FutureState::Resolved {
                        outcome,
                        int_valued,
                    } => {
                        let value = to_value(outcome, int_valued);
                        self.wire_state.classical_values.insert((node, 0), value);
                        self.repropagate_active_case_inputs(hugr);
                        debug!("Read future {future_id} -> {outcome}");
                        HandlerOutcome::Processed
                    }
                    FutureState::Pending {
                        measurement_index,
                        int_valued,
                        ..
                    } => {
                        if let Some(&result) =
                            self.measurement_state.outcomes.get(&measurement_index)
                        {
                            let value = to_value(result, int_valued);
                            self.wire_state.classical_values.insert((node, 0), value);
                            self.repropagate_active_case_inputs(hugr);
                            debug!("Read future {future_id} from measurement -> {result}");
                            HandlerOutcome::Processed
                        } else {
                            // Result not yet available: defer -- retried
                            // when measurement results arrive.
                            debug!("Read future {future_id} pending, deferring");
                            HandlerOutcome::Defer
                        }
                    }
                }
            }
            "Dup" => {
                // Dup: Future<T> -> (Future<T>, Future<T>)
                // Create two new Futures pointing to the same result
                let Some(ClassicalValue::Future(original_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("futures.Dup at {node:?}: future not ready, deferring");
                    return HandlerOutcome::Defer;
                };
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
                HandlerOutcome::Processed
            }
            "Free" => {
                // Free: Future<T> -> ()
                // Discard the Future without reading. Defer until the input
                // resolves to an actual Future: succeeding without one marks
                // the node processed while its producer never ran.
                let Some(ClassicalValue::Future(future_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("Free at {node:?}: future not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                self.extension_state.futures.remove(&future_id);
                debug!("Free future {future_id}");
                HandlerOutcome::Processed
            }
            _ => {
                debug!("Unknown tket.futures operation: {op_name}");
                HandlerOutcome::Defer
            }
        }
    }
}
