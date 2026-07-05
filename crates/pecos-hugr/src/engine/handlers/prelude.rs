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

//! Prelude extension operations.
//!
//! This module handles HUGR prelude extension operations:
//! - `load_nat`: Load a bounded nat parameter into a usize runtime value
//! - `panic`: Trigger a panic (error condition)
//! - `print`: Print a value (for debugging)
//! - `MakeTuple` / `UnpackTuple`: Handled via classical ops, but included for completeness

use log::debug;
use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;
use crate::engine::handlers::HandlerOutcome;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle prelude extension operations.
    ///
    /// The prelude extension provides fundamental operations used across all HUGR programs.
    pub(crate) fn handle_prelude_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing prelude operation: {op_name} at {node:?}");

        match op_name {
            "load_nat" => {
                // load_nat loads a bounded nat parameter into a usize runtime
                // value. In a monomorphic context the value is a concrete
                // BoundedNat type arg on the op itself; inside a generic
                // function body it is a type VARIABLE that must be resolved
                // through the calling Call's instantiation type args.
                let op = hugr.get_optype(node);
                if let Some(ext_op) = op.as_extension_op() {
                    for arg in ext_op.args() {
                        match arg {
                            tket::hugr::types::TypeArg::BoundedNat(n) => {
                                debug!("load_nat: found bounded nat value {n}");
                                self.wire_state
                                    .classical_values
                                    .insert((node, 0), ClassicalValue::UInt(*n));
                                return HandlerOutcome::Processed;
                            }
                            tket::hugr::types::TypeArg::Variable(var) => {
                                if let Some(n) = self.resolve_call_type_arg(hugr, node, var.index())
                                {
                                    debug!(
                                        "load_nat: resolved type variable {} to {n} via active call",
                                        var.index()
                                    );
                                    self.wire_state
                                        .classical_values
                                        .insert((node, 0), ClassicalValue::UInt(n));
                                    return HandlerOutcome::Processed;
                                }
                            }
                            _ => {}
                        }
                    }
                    debug!("load_nat: couldn't extract bounded nat value from args");
                }
                // No fabricated default: a wrong nat here silently corrupts
                // whatever consumes it (e.g. a loop bound). Defer so the
                // engine's pending/retry mechanism (or, at completion, the
                // stall accounting) surfaces the problem instead.
                debug!("load_nat at {node:?}: value unresolved, deferring");
                HandlerOutcome::Defer
            }

            "exit" => {
                // prelude.exit "immediately halts a single shot's
                // execution" -- a NORMAL termination the engine's
                // drain-to-completion model cannot express mid-shot yet.
                // Fault with a clear message instead of stalling with a
                // starved-node report.
                HandlerOutcome::Fault(format!(
                    "prelude.exit executed at {node:?}: mid-shot halt is not \
                     supported by the engine yet"
                ))
            }
            "panic" => {
                // An EXECUTED panic is a real runtime error on the taken
                // path (guppy routes bounds/borrow/arithmetic failures
                // here): raise a fatal fault instead of continuing with the
                // panic's outputs unproduced, which either stalls with a
                // misleading message or completes with corrupt results.
                HandlerOutcome::Fault(format!(
                    "program panicked (prelude.panic executed at {node:?})"
                ))
            }

            "print" => {
                // Print operation - for simulation, we just pass through
                debug!("prelude::print at {node:?}");
                self.propagate_all_inputs(hugr, node);
                HandlerOutcome::Processed
            }

            "MakeTuple" => {
                // MakeTuple: N inputs -> 1 output (a tuple/sum containing all inputs)
                // Collect all input values into a tuple
                use tket::hugr::ops::OpTrait;
                let op = hugr.get_optype(node);
                let num_inputs = op.dataflow_signature().map_or(0, |sig| sig.input_count());

                // A missing input means the value is not ready yet (or is a
                // linear qubit handled structurally): defer instead of
                // fabricating a default element, which would mark this node
                // processed and let consumers (calls, blocks) fire with a
                // garbage tuple.
                let mut elements = Vec::with_capacity(num_inputs);
                for port in 0..num_inputs {
                    if let Some(value) = self.get_input_value(hugr, node, port) {
                        elements.push(value);
                    } else if let Some(qubit_id) = self.get_input_qubit(hugr, node, port) {
                        // Linear payload: qubit flow is resolved structurally.
                        elements.push(ClassicalValue::QubitRef(qubit_id));
                    } else {
                        debug!("MakeTuple at {node:?}: input {port} not ready, deferring");
                        return HandlerOutcome::Defer;
                    }
                }

                debug!(
                    "MakeTuple at {node:?}: created tuple with {} elements",
                    elements.len()
                );
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Tuple(elements));
                HandlerOutcome::Processed
            }
            "UnpackTuple" => {
                // UnpackTuple: 1 input (a tuple) -> N outputs (the elements)
                use tket::hugr::ops::OpTrait;
                let op = hugr.get_optype(node);
                let num_outputs = op.dataflow_signature().map_or(0, |sig| sig.output_count());

                // A tuple is a 1-variant sum, so accept a Sum payload the
                // same way (e.g. a tuple that crossed a CFG/Call boundary as
                // a tagged value). A missing input defers (marking the node
                // processed without output would let consumers fire early).
                match self.get_input_value(hugr, node, 0) {
                    Some(
                        ClassicalValue::Tuple(elements)
                        | ClassicalValue::Sum {
                            values: elements, ..
                        },
                    ) => {
                        for (port, value) in elements.into_iter().enumerate() {
                            if port < num_outputs {
                                if let ClassicalValue::QubitRef(qubit_id) = &value {
                                    self.wire_state
                                        .wire_to_qubit
                                        .insert((node, port), *qubit_id);
                                }
                                self.wire_state.classical_values.insert((node, port), value);
                            }
                        }
                        debug!("UnpackTuple at {node:?}: unpacked to {num_outputs} outputs");
                        HandlerOutcome::Processed
                    }
                    Some(_) => {
                        // Single non-tuple value - pass through
                        debug!(
                            "UnpackTuple at {node:?}: input not a tuple, attempting pass-through"
                        );
                        self.propagate_all_inputs(hugr, node);
                        HandlerOutcome::Processed
                    }
                    None if self.get_input_qubit(hugr, node, 0).is_some() => {
                        // Linear (qubit) tuple: flow is resolved structurally.
                        self.propagate_all_inputs(hugr, node);
                        HandlerOutcome::Processed
                    }
                    None => {
                        debug!("UnpackTuple at {node:?}: input not ready, deferring");
                        HandlerOutcome::Defer
                    }
                }
            }

            "Noop" | "Lift" | "Barrier" => {
                // Genuine identity/annotation ops: pass values through.
                self.propagate_all_inputs(hugr, node);
                HandlerOutcome::Processed
            }
            _ => {
                // Unknown op: defer so it surfaces in the completion-time
                // stall report -- treating an op the engine knows nothing
                // about as an identity wire fabricates semantics.
                debug!("Unknown prelude operation: {op_name} at {node:?}, deferring");
                HandlerOutcome::Defer
            }
        }
    }
}
