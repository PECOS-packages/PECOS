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
use crate::engine::types::{ClassicalValue, FutureState, RngContextId, RngContextState};

impl HugrEngine {
    /// Handle tket.qsystem operations (lazy measurements, barriers, etc.).
    #[allow(clippy::too_many_lines)] // Operation dispatch is inherently large
    pub(crate) fn handle_qsystem_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.qsystem operation: {op_name} at {node:?}");

        match op_name {
            "LazyMeasure" => {
                // LazyMeasure: Qubit -> Future<bool>
                // Queue the measurement and create a Future handle
                if let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) {
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
                }
                true
            }
            "LazyMeasureReset" => {
                // LazyMeasureReset: Qubit -> (Qubit, Future<bool>)
                if let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) {
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
                }
                true
            }
            "LazyMeasureLeaked" => {
                // LazyMeasureLeaked: Qubit -> Future<int[6]>
                // Same as LazyMeasure but result can be 0, 1, or 2 (leaked)
                if let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) {
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
                }
                true
            }
            "MeasureReset" => {
                // MeasureReset: Qubit -> (Qubit, bool)
                // Atomic measure + reset (not lazy)
                if let Some(qubit_id) = self.get_input_qubit(hugr, node, 0) {
                    self.message_builder.mz(&[qubit_id.0]);
                    self.measurement_state.mappings.push((node, qubit_id));

                    // Queue reset
                    self.message_builder.pz(&[qubit_id.0]);

                    // Track measurement output wire
                    self.measurement_state.output_wires.insert(node, (node, 1));

                    // Output port 0: qubit
                    self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);

                    debug!("MeasureReset on qubit {qubit_id:?}");
                }
                true
            }
            "RuntimeBarrier" | "StateResult" => {
                // Pass-through operations: input array = output array
                // For simulation, these are no-ops
                // Propagate qubit arrays if present
                self.propagate_qubit_array(hugr, node);
                debug!("{op_name} at {node:?} (no-op for simulation)");
                true
            }
            "TryQAlloc" => {
                // TryQAlloc: () -> Sum<(), Qubit>
                // For simulation, always succeed and allocate a qubit
                let qubit_id = QubitId::from(self.wire_state.next_qubit_id);
                self.wire_state.next_qubit_id += 1;

                // Output on port 0 (Sum type, tag 1 = success with qubit)
                self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);
                // Store Sum tag = 1 (success) for control flow
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::UInt(1));

                debug!("TryQAlloc created qubit {qubit_id:?}");
                true
            }
            "Reset" | "Rz" | "PhasedX" | "ZZPhase" | "Measure" | "QFree" => {
                // These are handled as quantum ops (via hugr_op_to_gate_type)
                // Return false to let the quantum op handler process them
                false
            }
            _ => {
                debug!("Unknown tket.qsystem operation: {op_name}");
                false
            }
        }
    }

    /// Handle `tket.qsystem.random` operations for random number generation.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub(crate) fn handle_random_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.qsystem.random operation: {op_name} at {node:?}");

        match op_name {
            "NewRNGContext" => {
                // NewRNGContext: int<64> -> RNGContext
                // Create a new RNG context with the given seed
                let seed = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_uint())
                    .unwrap_or(0);

                let ctx_id = self.extension_state.next_rng_context_id;
                self.extension_state.next_rng_context_id += 1;

                self.extension_state
                    .rng_contexts
                    .insert(ctx_id, RngContextState::new(seed));

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::RngContext(ctx_id));

                debug!("NewRNGContext with seed {seed} -> context {ctx_id}");
                true
            }
            "DeleteRNGContext" => {
                // DeleteRNGContext: RNGContext -> ()
                // Clean up an RNG context
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let ClassicalValue::RngContext(ctx_id) = value
                {
                    self.extension_state.rng_contexts.remove(&ctx_id);
                    debug!("DeleteRNGContext: removed context {ctx_id}");
                }
                true
            }
            "RandomFloat" => {
                // RandomFloat: RNGContext -> (RNGContext, float64)
                // Generate a random float in [0, 1)
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let ClassicalValue::RngContext(ctx_id) = value
                {
                    let random_float = self.generate_random_float(ctx_id);

                    // Output port 0: RNGContext (pass through)
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::RngContext(ctx_id));
                    // Output port 1: random float
                    self.wire_state
                        .classical_values
                        .insert((node, 1), ClassicalValue::Float(random_float));

                    debug!("RandomFloat: generated {random_float}");
                }
                true
            }
            "RandomInt" => {
                // RandomInt: RNGContext -> (RNGContext, int<32>)
                // Generate a random 32-bit integer
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let ClassicalValue::RngContext(ctx_id) = value
                {
                    let random_int = self.generate_random_u64(ctx_id) as i64;

                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::RngContext(ctx_id));
                    self.wire_state
                        .classical_values
                        .insert((node, 1), ClassicalValue::Int(random_int));

                    debug!("RandomInt: generated {random_int}");
                }
                true
            }
            "RandomIntBounded" => {
                // RandomIntBounded: (RNGContext, int<32>) -> (RNGContext, int<32>)
                // Generate a random integer in [0, bound)
                let ctx_value = self.get_input_value(hugr, node, 0);
                let bound = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_int())
                    .unwrap_or(1)
                    .max(1) as u64;

                if let Some(ClassicalValue::RngContext(ctx_id)) = ctx_value {
                    let random_val = self.generate_random_u64(ctx_id) % bound;

                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::RngContext(ctx_id));
                    self.wire_state
                        .classical_values
                        .insert((node, 1), ClassicalValue::Int(random_val as i64));

                    debug!("RandomIntBounded({bound}): generated {random_val}");
                }
                true
            }
            "RandomAdvance" => {
                // RandomAdvance: (RNGContext, int<64>) -> RNGContext
                // Advance the RNG state by delta steps (can be negative for backtracking)
                let ctx_value = self.get_input_value(hugr, node, 0);
                let delta = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_int())
                    .unwrap_or(0);

                if let Some(ClassicalValue::RngContext(ctx_id)) = ctx_value {
                    // Advance the RNG state by |delta| steps
                    // Note: For simplicity, we only support forward advancement
                    // Negative delta would require storing history which we don't do
                    let steps = delta.unsigned_abs();
                    for _ in 0..steps {
                        self.generate_random_u64(ctx_id);
                    }

                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::RngContext(ctx_id));

                    debug!("RandomAdvance: advanced by {delta} steps");
                }
                true
            }
            _ => {
                debug!("Unknown tket.qsystem.random operation: {op_name}");
                false
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
    pub(crate) fn handle_utils_op(&mut self, _hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.qsystem.utils operation: {op_name} at {node:?}");

        if op_name == "GetCurrentShot" {
            // GetCurrentShot: () -> int<64>
            // Return the current shot number
            self.wire_state.classical_values.insert(
                (node, 0),
                ClassicalValue::UInt(self.extension_state.current_shot),
            );

            debug!("GetCurrentShot: {}", self.extension_state.current_shot);
            true
        } else {
            debug!("Unknown tket.qsystem.utils operation: {op_name}");
            false
        }
    }
}
