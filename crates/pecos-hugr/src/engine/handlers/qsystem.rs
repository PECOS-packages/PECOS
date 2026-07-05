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

//! Quantum system operations (`tket.qsystem`, `tket.qsystem.random`, `tket.qsystem.utils`).
//!
//! This module handles quantum system operations including:
//! - Lazy measurements (`LazyMeasure`, `LazyMeasureReset`, `LazyMeasureLeaked`)
//! - Measurement with reset (`MeasureReset`)
//! - Qubit allocation (`TryQAlloc`)
//! - Barriers and state operations (`RuntimeBarrier`, `StateResult`)
//! - Random number generation (`NewRNGContext`, `RandomFloat`, `RandomInt`, etc.)
//! - Utility operations (`GetCurrentShot`)

use log::debug;
use pecos_core::QubitId;
use tket::hugr::{Hugr, Node};

use crate::engine::HugrEngine;
use crate::engine::handlers::HandlerOutcome;
use crate::engine::types::{ClassicalValue, FutureState, RngContextId, RngContextState};

impl HugrEngine {
    /// Handle tket.qsystem operations (lazy measurements, barriers, etc.).
    #[allow(clippy::too_many_lines)] // Operation dispatch is inherently large
    pub(crate) fn handle_qsystem_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.qsystem operation: {op_name} at {node:?}");

        match op_name {
            "LazyMeasure" => {
                // LazyMeasure: Qubit -> Future<bool>
                // Queue the measurement and create a Future handle
                let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) else {
                    debug!("LazyMeasure at {node:?}: qubit not resolved, deferring");
                    return HandlerOutcome::Defer;
                };
                // Queue measurement
                self.message_builder.mz(&[qubit_id.0]);
                let measurement_index = self.measurement_state.mappings.len();
                self.measurement_state.mappings.push((node, qubit_id));

                // Create a Future
                let future_id = self.extension_state.next_future_id;
                self.extension_state.next_future_id += 1;
                self.extension_state.futures.insert(
                    future_id,
                    FutureState::Pending {
                        measurement_node: node,
                        qubit: qubit_id,
                        measurement_index,
                    },
                );

                // Store Future value on output port 0
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Future(future_id));

                debug!("LazyMeasure on qubit {qubit_id:?}, created future {future_id}");
                HandlerOutcome::Processed
            }
            "LazyMeasureReset" => {
                // LazyMeasureReset: Qubit -> (Qubit, Future<bool>)
                let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) else {
                    debug!("LazyMeasureReset at {node:?}: qubit not resolved, deferring");
                    return HandlerOutcome::Defer;
                };
                // Queue measurement
                self.message_builder.mz(&[qubit_id.0]);
                let measurement_index = self.measurement_state.mappings.len();
                self.measurement_state.mappings.push((node, qubit_id));

                // Queue reset
                self.message_builder.pz(&[qubit_id.0]);

                // Create a Future
                let future_id = self.extension_state.next_future_id;
                self.extension_state.next_future_id += 1;
                self.extension_state.futures.insert(
                    future_id,
                    FutureState::Pending {
                        measurement_node: node,
                        qubit: qubit_id,
                        measurement_index,
                    },
                );

                // Output port 0: qubit, Output port 1: Future
                self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::Future(future_id));

                debug!("LazyMeasureReset on qubit {qubit_id:?}, created future {future_id}");
                HandlerOutcome::Processed
            }
            "LazyMeasureLeaked" => {
                // LazyMeasureLeaked: Qubit -> Future<int[6]>
                // Same as LazyMeasure but result can be 0, 1, or 2 (leaked)
                let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) else {
                    debug!("LazyMeasureLeaked at {node:?}: qubit not resolved, deferring");
                    return HandlerOutcome::Defer;
                };
                self.message_builder.mz(&[qubit_id.0]);
                let measurement_index = self.measurement_state.mappings.len();
                self.measurement_state.mappings.push((node, qubit_id));

                let future_id = self.extension_state.next_future_id;
                self.extension_state.next_future_id += 1;
                self.extension_state.futures.insert(
                    future_id,
                    FutureState::Pending {
                        measurement_node: node,
                        qubit: qubit_id,
                        measurement_index,
                    },
                );

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Future(future_id));

                debug!("LazyMeasureLeaked on qubit {qubit_id:?}, created future {future_id}");
                HandlerOutcome::Processed
            }
            "MeasureReset" => {
                // MeasureReset: Qubit -> (Qubit, bool)
                // Atomic measure + reset (not lazy)
                let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) else {
                    debug!("MeasureReset at {node:?}: qubit not resolved, deferring");
                    return HandlerOutcome::Defer;
                };
                self.message_builder.mz(&[qubit_id.0]);
                self.measurement_state.mappings.push((node, qubit_id));

                // Queue reset
                self.message_builder.pz(&[qubit_id.0]);

                // Track measurement output wire
                self.measurement_state.output_wires.insert(node, (node, 1));

                // Output port 0: qubit
                self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);

                debug!("MeasureReset on qubit {qubit_id:?}");
                HandlerOutcome::Processed
            }
            "RuntimeBarrier" | "StateResult" => {
                // Pass-through operations: input array = output array
                // For simulation, these are no-ops
                // Propagate qubit arrays if present
                self.propagate_qubit_array(hugr, node);
                debug!("{op_name} at {node:?} (no-op for simulation)");
                HandlerOutcome::Processed
            }
            "TryQAlloc" => {
                // TryQAlloc: () -> Option<Qubit>
                // For simulation, always succeed and allocate a qubit. The
                // value must be a REAL Sum carrying the qubit payload:
                // case-input propagation unpacks payloads from Sum values,
                // and a bare scalar loses the allocated qubit (falling into
                // implicit re-allocation downstream).
                let qubit_id = QubitId::from(self.wire_state.next_qubit_id);
                self.wire_state.next_qubit_id += 1;

                self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);
                self.wire_state.classical_values.insert(
                    (node, 0),
                    ClassicalValue::Sum {
                        tag: 1,
                        values: vec![ClassicalValue::QubitRef(qubit_id)],
                    },
                );

                debug!("TryQAlloc created qubit {qubit_id:?}");
                HandlerOutcome::Processed
            }
            "Reset" | "Rz" | "PhasedX" | "ZZPhase" | "Measure" | "QFree" => {
                // These are handled as quantum ops (via hugr_op_to_gate_type)
                // Return false to let the quantum op handler process them
                HandlerOutcome::Defer
            }
            _ => {
                debug!("Unknown tket.qsystem operation: {op_name}");
                HandlerOutcome::Defer
            }
        }
    }

    /// Handle `tket.qsystem.random` operations for random number generation.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub(crate) fn handle_random_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.qsystem.random operation: {op_name} at {node:?}");

        match op_name {
            "NewRNGContext" => {
                // NewRNGContext: int<64> -> Option<RNGContext>
                // Create a new RNG context with the given seed. The
                // signature returns an option (None on a second call); this
                // engine has no global-context restriction, so it always
                // produces Some.
                let Some(seed) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint())
                else {
                    debug!("NewRNGContext at {node:?}: seed not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                let ctx_id = self.extension_state.next_rng_context_id;
                self.extension_state.next_rng_context_id += 1;

                self.extension_state
                    .rng_contexts
                    .insert(ctx_id, RngContextState::new(seed));

                self.wire_state.classical_values.insert(
                    (node, 0),
                    ClassicalValue::Sum {
                        tag: 1,
                        values: vec![ClassicalValue::RngContext(ctx_id)],
                    },
                );

                debug!("NewRNGContext with seed {seed} -> Some(context {ctx_id})");
                HandlerOutcome::Processed
            }
            "DeleteRNGContext" => {
                // DeleteRNGContext: RNGContext -> ()
                // Clean up an RNG context
                let Some(ClassicalValue::RngContext(ctx_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("DeleteRNGContext at {node:?}: context not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                self.extension_state.rng_contexts.remove(&ctx_id);
                debug!("DeleteRNGContext: removed context {ctx_id}");
                HandlerOutcome::Processed
            }
            "RandomFloat" => {
                // RandomFloat: RNGContext -> (float64, RNGContext)
                // Generate a random float in [0, 1)
                let Some(ClassicalValue::RngContext(ctx_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("RandomFloat at {node:?}: context not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let random_float = self.generate_random_float(ctx_id);

                // Value first, context second, per the extension signature
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Float(random_float));
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::RngContext(ctx_id));

                debug!("RandomFloat: generated {random_float}");
                HandlerOutcome::Processed
            }
            "RandomInt" => {
                // RandomInt: RNGContext -> (int<32>, RNGContext)
                // Generate a random 32-bit integer
                let Some(ClassicalValue::RngContext(ctx_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("RandomInt at {node:?}: context not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                // The output is int<5> (32-bit): canonical storage is the
                // sign-extended low 32 bits, per the engine-wide width
                // convention (a raw zero-extended u32 would misread in every
                // signed consumer).
                #[allow(clippy::cast_possible_truncation)] // intentional 32-bit mask
                let random_int = i64::from((self.generate_random_u64(ctx_id) as u32).cast_signed());

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Int(random_int));
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::RngContext(ctx_id));

                debug!("RandomInt: generated {random_int}");
                HandlerOutcome::Processed
            }
            "RandomIntBounded" => {
                // RandomIntBounded: (RNGContext, int<32>) -> (int<32>, RNGContext)
                // Generate a random integer in [0, bound)
                let Some(ClassicalValue::RngContext(ctx_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("RandomIntBounded at {node:?}: context not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(bound) = self.get_input_value(hugr, node, 1).and_then(|v| v.as_int())
                else {
                    debug!("RandomIntBounded at {node:?}: bound not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                // The bound is UNSIGNED int<5>: reinterpret the canonical
                // (sign-extended) storage as its 32-bit pattern -- a bound
                // >= 2^31 stores negative but names a valid nonempty range.
                #[allow(clippy::cast_sign_loss)]
                let bound = (bound as u64) & 0xFFFF_FFFF;
                if bound == 0 {
                    // [0, 0) is empty: there is no value this op could
                    // produce, so clamping would fabricate a result.
                    return HandlerOutcome::Fault(format!(
                        "RandomIntBounded at {node:?}: bound 0 names an empty range"
                    ));
                }
                let random_val = self.generate_random_u64(ctx_id) % bound;

                #[allow(clippy::cast_possible_truncation)]
                self.wire_state.classical_values.insert(
                    (node, 0),
                    ClassicalValue::Int(i64::from((random_val as u32).cast_signed())),
                );
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::RngContext(ctx_id));

                debug!("RandomIntBounded({bound}): generated {random_val}");
                HandlerOutcome::Processed
            }
            "RandomAdvance" => {
                // RandomAdvance: (RNGContext, int<64>) -> RNGContext
                // Advance the RNG state by delta steps (can be negative for backtracking)
                let Some(ClassicalValue::RngContext(ctx_id)) = self.get_input_value(hugr, node, 0)
                else {
                    debug!("RandomAdvance at {node:?}: context not ready, deferring");
                    return HandlerOutcome::Defer;
                };
                let Some(delta) = self.get_input_value(hugr, node, 1).and_then(|v| v.as_int())
                else {
                    debug!("RandomAdvance at {node:?}: delta not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                {
                    const MAX_ADVANCE_STEPS: u64 = 1 << 24;
                    // The spec advances OR BACKTRACKS by delta; this
                    // step-based implementation can only go forward, and a
                    // huge forward delta would hang the host. Fail loud on
                    // both instead of silently advancing the wrong way.
                    if delta < 0 {
                        return HandlerOutcome::Fault(format!(
                            "RandomAdvance at {node:?}: backtracking (delta {delta}) is \
                             not supported by the step-based RNG implementation"
                        ));
                    }
                    let steps = delta.unsigned_abs();
                    if steps > MAX_ADVANCE_STEPS {
                        return HandlerOutcome::Fault(format!(
                            "RandomAdvance at {node:?}: delta {steps} exceeds the \
                             step-based implementation's ceiling ({MAX_ADVANCE_STEPS})"
                        ));
                    }
                    for _ in 0..steps {
                        self.generate_random_u64(ctx_id);
                    }

                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::RngContext(ctx_id));

                    debug!("RandomAdvance: advanced by {delta} steps");
                }
                HandlerOutcome::Processed
            }
            _ => {
                debug!("Unknown tket.qsystem.random operation: {op_name}");
                HandlerOutcome::Defer
            }
        }
    }

    /// Generate a random float in [0, 1) using xorshift64.
    pub(crate) fn generate_random_float(&mut self, ctx_id: RngContextId) -> f64 {
        if let Some(ctx) = self.extension_state.rng_contexts.get_mut(&ctx_id) {
            ctx.next_f64()
        } else {
            0.0
        }
    }

    /// Generate a random u64 using xorshift64.
    pub(crate) fn generate_random_u64(&mut self, ctx_id: RngContextId) -> u64 {
        if let Some(ctx) = self.extension_state.rng_contexts.get_mut(&ctx_id) {
            ctx.next_u64()
        } else {
            0
        }
    }

    /// Handle `tket.qsystem.utils` operations.
    pub(crate) fn handle_utils_op(
        &mut self,
        _hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.qsystem.utils operation: {op_name} at {node:?}");

        if op_name == "GetCurrentShot" {
            // GetCurrentShot: () -> int<64>
            // Return the current shot number
            self.wire_state.classical_values.insert(
                (node, 0),
                ClassicalValue::UInt(self.extension_state.current_shot),
            );

            debug!("GetCurrentShot: {}", self.extension_state.current_shot);
            HandlerOutcome::Processed
        } else {
            debug!("Unknown tket.qsystem.utils operation: {op_name}");
            HandlerOutcome::Defer
        }
    }
}
