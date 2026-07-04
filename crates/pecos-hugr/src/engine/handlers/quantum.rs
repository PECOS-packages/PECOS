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

//! Quantum extension operations (`tket.quantum`, `tket.rotation`, `tket.modifier`, `tket.global_phase`).
//!
//! This module handles quantum-related extension operations:
//! - `tket.quantum`: Non-gate operations like `symbolic_angle`
//! - `tket.rotation`: Rotation type operations (`from_halfturns`, `to_halfturns`, radd)
//! - `tket.modifier`: Gate modifiers (`ControlModifier`, `DaggerModifier`, `PowerModifier`)
//! - `tket.global_phase`: Global phase accumulation
//!
//! Note: Quantum gate operations are handled via the quantum ops extraction path,
//! not through these extension handlers.

use log::debug;
use pecos_quantum::hugr_convert::try_extract_rotation_angle;
use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;
use crate::engine::handlers::HandlerOutcome;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle `tket.quantum` non-gate operations (e.g., `symbolic_angle`).
    ///
    /// Note: Quantum gate operations from tket.quantum are handled via the
    /// quantum ops extraction path. This handler is for non-gate operations
    /// like `symbolic_angle` that create classical values (rotations).
    pub(crate) fn handle_quantum_extension_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.quantum non-gate operation: {op_name} at {node:?}");

        match op_name {
            "symbolic_angle" => {
                // symbolic_angle: () -> rotation
                // Creates a rotation from a symbolic expression (sympy string parameter)
                // For simulation, we try to parse simple numeric expressions
                let op = hugr.get_optype(node);
                if let Some(ext_op) = op.as_extension_op() {
                    let debug_str = format!("{ext_op:?}");
                    // Try to extract the symbolic expression from parameters
                    let angle = Self::parse_symbolic_angle(&debug_str);
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Rotation(angle));
                    debug!("symbolic_angle: parsed angle = {angle} half-turns");
                } else {
                    // Default to 0 if we can't parse
                    self.wire_state
                        .classical_values
                        .insert((node, 0), ClassicalValue::Rotation(0.0));
                    debug!("symbolic_angle: defaulting to 0");
                }
                HandlerOutcome::Processed
            }
            // Quantum gates are handled via the quantum ops path, not here
            // -- defer so they fall through to the gate handling
            _ => HandlerOutcome::Defer,
        }
    }

    /// Parse a symbolic angle expression from debug representation.
    ///
    /// Attempts to parse simple expressions like:
    /// - Numeric literals: "0.5", "1.0", "-0.25"
    /// - Pi expressions: "pi", "pi/2", "pi/4", "2*pi"
    /// - Fractions: "1/2", "1/4"
    pub(crate) fn parse_symbolic_angle(debug_str: &str) -> f64 {
        // Look for quoted string content that might contain the expression
        if let Some(expr) = Self::extract_string_from_debug(debug_str) {
            let expr = expr.trim().to_lowercase();

            // Try parsing as a simple float
            if let Ok(val) = expr.parse::<f64>() {
                return val;
            }

            // Handle pi expressions (angles in half-turns, so pi = 1.0 half-turn)
            if expr == "pi" {
                return 1.0;
            }
            if expr == "-pi" {
                return -1.0;
            }
            if expr == "2*pi" || expr == "2pi" {
                return 2.0;
            }

            // Handle pi/n expressions
            if let Some(rest) = expr.strip_prefix("pi/")
                && let Ok(divisor) = rest.parse::<f64>()
            {
                return 1.0 / divisor;
            }
            if let Some(rest) = expr.strip_prefix("-pi/")
                && let Ok(divisor) = rest.parse::<f64>()
            {
                return -1.0 / divisor;
            }

            // Handle n*pi expressions
            if let Some(rest) = expr.strip_suffix("*pi")
                && let Ok(multiplier) = rest.parse::<f64>()
            {
                return multiplier;
            }

            // Handle simple fractions like 1/2, 1/4
            if let Some((num_str, denom_str)) = expr.split_once('/')
                && let (Ok(num), Ok(denom)) = (num_str.parse::<f64>(), denom_str.parse::<f64>())
                && denom != 0.0
            {
                return num / denom;
            }

            debug!("Could not parse symbolic angle expression: '{expr}', defaulting to 0");
        }

        0.0
    }

    /// Handle `tket.rotation` operations.
    ///
    /// Missing inputs DEFER the node (return `false` so the engine's
    /// pending/retry mechanism picks it up) instead of fabricating a
    /// silent 0.0 rotation. `from_halfturns` additionally falls back to
    /// the static angle tracer: some angle chains are only evaluable
    /// statically (e.g. guppy expression angles like `pi / 4` lower to
    /// Call nodes the runtime does not execute). If neither source can
    /// produce the value, the node stays deferred and the gate consumer
    /// (`resolve_rotation_angle`) fails loud if the statically extracted
    /// gate angle is also missing.
    pub(crate) fn handle_rotation_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.rotation operation: {op_name} at {node:?}");

        match op_name {
            "from_halfturns" | "from_halfturns_unchecked" => {
                // from_halfturns: float64 -> Rotation
                // Convert a float (in half-turns) to a Rotation type.
                // try_extract_rotation_angle traces this node's input 0 (0
                // qubit inputs) and returns full turns; halfturns = 2 * turns.
                let Some(halfturns) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_float())
                    .or_else(|| try_extract_rotation_angle(hugr, node, 0).map(|turns| turns * 2.0))
                else {
                    debug!("tket.rotation.{op_name} at {node:?}: input not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Rotation(halfturns));

                debug!("tket.rotation.from_halfturns: {halfturns}");
                HandlerOutcome::Processed
            }
            "to_halfturns" => {
                // to_halfturns: Rotation -> float64
                // Convert a Rotation to a float (in half-turns)
                let Some(halfturns) = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_rotation())
                else {
                    debug!("tket.rotation.to_halfturns at {node:?}: input not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Float(halfturns));

                debug!("tket.rotation.to_halfturns: {halfturns}");
                HandlerOutcome::Processed
            }
            "radd" => {
                // radd: (Rotation, Rotation) -> Rotation
                // Add two rotations
                let a = self
                    .get_input_value(hugr, node, 0)
                    .and_then(|v| v.as_rotation());
                let b = self
                    .get_input_value(hugr, node, 1)
                    .and_then(|v| v.as_rotation());
                let (Some(a), Some(b)) = (a, b) else {
                    debug!("tket.rotation.radd at {node:?}: inputs not ready, deferring");
                    return HandlerOutcome::Defer;
                };

                // Rotation addition, normalized to [0, 2) half-turns
                let sum = (a + b).rem_euclid(2.0);

                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Rotation(sum));

                debug!("tket.rotation.radd: {a} + {b} = {sum}");
                HandlerOutcome::Processed
            }
            _ => {
                debug!("Unknown tket.rotation operation: {op_name}");
                HandlerOutcome::Defer
            }
        }
    }

    /// Handle `tket.modifier` operations for gate modifiers.
    pub(crate) fn handle_modifier_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.modifier operation: {op_name} at {node:?}");

        // Gate modifiers change how gates are applied.
        // For simulation, we track these as metadata but the actual gate
        // application happens in the quantum backend.
        match op_name {
            "ControlModifier" => {
                // ControlModifier adds quantum control to an operation
                // Input: control qubit(s) + operation
                // For simulation, this is handled by the quantum backend
                self.propagate_qubit_array(hugr, node);
                debug!("ControlModifier at {node:?} (handled by quantum backend)");
                HandlerOutcome::Processed
            }
            "DaggerModifier" => {
                // DaggerModifier applies the inverse/adjoint of an operation
                // For simulation, this is handled by the quantum backend
                self.propagate_qubit_array(hugr, node);
                debug!("DaggerModifier at {node:?} (handled by quantum backend)");
                HandlerOutcome::Processed
            }
            "PowerModifier" => {
                // PowerModifier raises an operation to a power
                // For simulation, this is handled by the quantum backend
                self.propagate_qubit_array(hugr, node);
                debug!("PowerModifier at {node:?} (handled by quantum backend)");
                HandlerOutcome::Processed
            }
            _ => {
                debug!("Unknown tket.modifier operation: {op_name}");
                HandlerOutcome::Defer
            }
        }
    }

    /// Handle `tket.global_phase` operations.
    pub(crate) fn handle_global_phase_op(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op_name: &str,
    ) -> HandlerOutcome {
        debug!("Processing tket.global_phase operation: {op_name} at {node:?}");

        if op_name == "global_phase" {
            // global_phase: Rotation -> ()
            // Add global phase to the circuit
            let phase = self
                .get_input_value(hugr, node, 0)
                .and_then(|v| v.as_rotation())
                .unwrap_or(0.0);

            // Accumulate global phase (normalized to [0, 2))
            self.extension_state.global_phase =
                (self.extension_state.global_phase + phase).rem_euclid(2.0);

            debug!(
                "tket.global_phase: added {phase}, total = {}",
                self.extension_state.global_phase
            );
            HandlerOutcome::Processed
        } else {
            debug!("Unknown tket.global_phase operation: {op_name}");
            HandlerOutcome::Defer
        }
    }
}
