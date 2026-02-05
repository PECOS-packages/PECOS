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

use crate::engine::types::ClassicalValue;
use crate::engine::HugrEngine;

impl HugrEngine {
    /// Handle prelude extension operations.
    ///
    /// The prelude extension provides fundamental operations used across all HUGR programs.
    pub(crate) fn handle_prelude_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing prelude operation: {op_name} at {node:?}");

        match op_name {
            "load_nat" => {
                // load_nat loads a bounded nat parameter into a usize runtime value.
                // The value comes from the type arguments of the polymorphic instantiation.
                // For now, we try to extract it from the extension op's args.
                let op = hugr.get_optype(node);
                if let Some(ext_op) = op.as_extension_op() {
                    let args = ext_op.args();
                    for arg in args {
                        // Look for BoundedNat type arg
                        if let tket::hugr::types::TypeArg::BoundedNat(n) = arg {
                            debug!("load_nat: found bounded nat value {n}");
                            self.wire_state
                                .classical_values
                                .insert((node, 0), ClassicalValue::UInt(*n));
                            return true;
                        }
                    }
                    // If we can't find the value, log and return false
                    debug!("load_nat: couldn't extract bounded nat value from args");
                }
                // Fallback: set a default value of 0
                debug!("load_nat: using default value 0");
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::UInt(0));
                true
            }

            "panic" => {
                // Panic operation - for simulation, we log it and continue
                // In a real execution, this would halt the program
                debug!("prelude::panic encountered at {node:?}");
                // Mark as handled but don't crash the simulation
                true
            }

            "print" => {
                // Print operation - for simulation, we just pass through
                debug!("prelude::print at {node:?}");
                self.propagate_all_inputs(hugr, node);
                true
            }

            "MakeTuple" | "UnpackTuple" => {
                // These are handled by the classical ops handler
                // But if we get here, just propagate
                debug!("prelude::{op_name} at {node:?} - propagating");
                self.propagate_all_inputs(hugr, node);
                true
            }

            _ => {
                debug!("Unknown prelude operation: {op_name}");
                // For unknown ops, try to propagate inputs as a pass-through
                self.propagate_all_inputs(hugr, node);
                true
            }
        }
    }
}
