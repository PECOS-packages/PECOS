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

//! Debug operations (`tket.debug`).
//!
//! This module handles debug-related operations that may be useful
//! for testing and debugging quantum programs.

use log::debug;
use tket::hugr::{Hugr, Node};

use crate::engine::GuppyHugrEngine;

impl GuppyHugrEngine {
    /// Handle tket.debug operations.
    pub(crate) fn handle_debug_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.debug operation: {op_name} at {node:?}");

        if op_name == "StateResult" {
            // StateResult: array<N, Qubit> -> array<N, Qubit>
            // Pass-through for simulation; optionally log state info
            self.propagate_qubit_array(hugr, node);
            debug!("StateResult at {node:?} (no-op for simulation)");
            true
        } else {
            debug!("Unknown tket.debug operation: {op_name}");
            false
        }
    }
}
