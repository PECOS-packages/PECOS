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

//! Guppy-specific operations (`tket.guppy`, `guppylang`).
//!
//! This module handles operations specific to the Guppy language frontend,
//! including type system operations like `drop` and `partial`.

use log::debug;
use tket::hugr::{Hugr, Node};

use crate::engine::GuppyHugrEngine;

impl GuppyHugrEngine {
    /// Handle `tket.guppy` operations.
    #[allow(clippy::unused_self)] // Consistent with other handler methods; may use self in future
    pub(crate) fn handle_guppy_op(&mut self, _hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.guppy operation: {op_name} at {node:?}");

        if op_name == "drop" {
            // drop: T -> ()
            // Drop an affine type value (opposite of move semantics)
            // No-op for simulation - just consumes the value
            debug!("tket.guppy.drop at {node:?} (value consumed)");
            true
        } else {
            debug!("Unknown tket.guppy operation: {op_name}");
            false
        }
    }

    /// Handle `guppylang` extension operations.
    pub(crate) fn handle_guppylang_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing guppylang operation: {op_name} at {node:?}");

        match op_name {
            "unsupported" => {
                // unsupported: stub for operations that can't be compiled
                // Log a warning but allow execution to continue
                debug!("guppylang.unsupported at {node:?} - operation not supported");
                // Pass through any inputs to outputs
                self.propagate_all_inputs(hugr, node);
                true
            }
            "partial" => {
                // partial: partial function application
                // For simulation, treat as identity/pass-through
                debug!("guppylang.partial at {node:?} - pass-through");
                self.propagate_all_inputs(hugr, node);
                true
            }
            _ => {
                debug!("Unknown guppylang operation: {op_name}");
                false
            }
        }
    }
}
