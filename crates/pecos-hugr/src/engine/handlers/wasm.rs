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

//! WebAssembly operations (`tket.wasm`).
//!
//! This module handles WASM integration operations for hybrid
//! classical-quantum computation. Currently provides stub implementations.

use log::debug;
use tket::hugr::{Hugr, Node};

use crate::engine::HugrEngine;
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Handle `tket.wasm` operations for WebAssembly integration.
    pub(crate) fn handle_wasm_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.wasm operation: {op_name} at {node:?}");

        // WASM operations are for hybrid classical-quantum computation.
        // For now, we provide stub implementations that allow programs to run
        // without full WASM support.
        match op_name {
            "get_context" | "GetContext" => {
                // get_context: () -> WasmContext
                // Create or get WASM execution context
                // Stub: output a placeholder value
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::UInt(0));
                debug!("tket.wasm.get_context: stub (no WASM support)");
                true
            }
            "dispose_context" | "DisposeContext" => {
                // dispose_context: WasmContext -> ()
                // Clean up WASM context (no-op for stub)
                debug!("tket.wasm.dispose_context: stub (no WASM support)");
                true
            }
            "call" | "Call" => {
                // call: (WasmContext, ...) -> (WasmContext, ...)
                // Call a WASM function
                // Stub: pass through inputs to outputs
                self.propagate_all_inputs(hugr, node);
                debug!("tket.wasm.call: stub (no WASM support)");
                true
            }
            "lookup_by_id" | "LookupById" => {
                // lookup_by_id: (WasmContext, int) -> (WasmContext, WasmFunc)
                // Stub: output placeholder
                if let Some(ctx) = self.get_input_value(hugr, node, 0) {
                    self.wire_state.classical_values.insert((node, 0), ctx);
                }
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::UInt(0));
                debug!("tket.wasm.lookup_by_id: stub (no WASM support)");
                true
            }
            "lookup_by_name" | "LookupByName" => {
                // lookup_by_name: (WasmContext, String) -> (WasmContext, WasmFunc)
                // Stub: output placeholder
                if let Some(ctx) = self.get_input_value(hugr, node, 0) {
                    self.wire_state.classical_values.insert((node, 0), ctx);
                }
                self.wire_state
                    .classical_values
                    .insert((node, 1), ClassicalValue::UInt(0));
                debug!("tket.wasm.lookup_by_name: stub (no WASM support)");
                true
            }
            "read_result" | "ReadResult" => {
                // read_result: WasmResult -> value
                // Stub: output zero
                self.wire_state
                    .classical_values
                    .insert((node, 0), ClassicalValue::Int(0));
                debug!("tket.wasm.read_result: stub (no WASM support)");
                true
            }
            _ => {
                debug!("Unknown tket.wasm operation: {op_name}");
                false
            }
        }
    }
}
