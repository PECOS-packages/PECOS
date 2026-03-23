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

//! Extension operation handlers for the HUGR engine.
//!
//! This module contains handlers for various HUGR extension operations:
//!
//! - [`result`]: Result capture operations (`tket.result`)
//! - [`qsystem`]: Quantum system operations (`tket.qsystem`, `tket.qsystem.random`, `tket.qsystem.utils`)
//! - [`futures`]: Future/lazy measurement operations (`tket.futures`)
//! - [`classical`]: Classical logic operations (`tket.bool`)
//! - [`arithmetic`]: Arithmetic operations (`arithmetic.float`, `arithmetic.int`, `arithmetic.conversions`)
//! - [`array`]: Array operations (`collections.array`)
//! - [`borrow_arr`]: Borrow array operations (`collections.borrow_arr`)
//! - [`prelude`]: Prelude operations (`prelude`)
//! - [`quantum`]: Quantum extension operations (`tket.quantum`, `tket.rotation`, `tket.modifier`, `tket.global_phase`)
//! - [`wasm`]: WebAssembly operations (`tket.wasm`)
//! - [`guppy`]: Guppy-specific operations (`tket.guppy`, `guppylang`)
//! - [`debug`]: Debug operations (`tket.debug`)
//!
//! Each handler module provides methods on [`HugrEngine`](super::HugrEngine) for processing
//! specific extension operations. The main dispatcher [`HugrEngine::handle_extension_op`]
//! routes operations to the appropriate handler based on extension ID.
//!
//! # Architecture Notes
//!
//! Handlers are implemented as `impl HugrEngine` blocks in separate files to:
//! - Keep related code together for easier maintenance
//! - Allow future refactoring toward ECS-like state separation
//! - Reduce the size of the main engine module
//!
//! All handlers follow a similar pattern:
//! 1. Extract input values using `get_input_value()` or `get_input_qubit()`
//! 2. Perform the operation
//! 3. Store results in `classical_values` or `wire_to_qubit`
//! 4. Return `true` if handled, `false` otherwise

mod arithmetic;
mod array;
mod borrow_arr;
mod classical;
mod debug;
mod futures;
mod guppy;
mod prelude;
mod qsystem;
mod quantum;
mod result;
mod wasm;

use tket::hugr::{Hugr, HugrView, Node};

use super::HugrEngine;

impl HugrEngine {
    /// Handle extension operations from various tket extensions.
    ///
    /// This is the main dispatcher for HUGR extension operations. It routes
    /// operations to the appropriate handler based on the extension ID.
    ///
    /// # Supported Extensions
    ///
    /// - `tket.result`: Result capture operations
    /// - `tket.qsystem`: Quantum system operations (measurements, barriers)
    /// - `tket.qsystem.random`: Random number generation
    /// - `tket.qsystem.utils`: Utility operations (shot tracking)
    /// - `tket.futures`: Future/lazy measurement operations
    /// - `tket.debug`: Debug operations
    /// - `tket.bool`: Boolean logic operations
    /// - `tket.rotation`: Rotation type operations
    /// - `tket.modifier`: Gate modifier operations
    /// - `tket.wasm`: WebAssembly integration
    /// - `tket.guppy`: Guppy-specific operations
    /// - `tket.global_phase`: Global phase operations
    /// - `tket.quantum`: Quantum non-gate operations
    /// - `guppylang`: Guppy language operations
    /// - `prelude`: Prelude operations
    /// - `collections.array`: Array operations
    /// - `collections.borrow_arr`: Borrow array operations
    /// - `arithmetic.float`: Float arithmetic
    /// - `arithmetic.int`: Integer arithmetic
    /// - `arithmetic.conversions`: Type conversions
    ///
    /// # Returns
    ///
    /// Returns `true` if the operation was handled, `false` otherwise.
    pub(crate) fn handle_extension_op(&mut self, hugr: &Hugr, node: Node) -> bool {
        let op = hugr.get_optype(node);
        let Some(ext_op) = op.as_extension_op() else {
            return false;
        };

        let ext_id = ext_op.extension_id();
        let ext_name = ext_id.as_ref() as &str;
        let op_name = ext_op.unqualified_id().to_string();

        match ext_name {
            "tket.result" => self.handle_result_op(hugr, node, &op_name),
            "tket.qsystem" => self.handle_qsystem_op(hugr, node, &op_name),
            "tket.qsystem.random" => self.handle_random_op(hugr, node, &op_name),
            "tket.qsystem.utils" => self.handle_utils_op(hugr, node, &op_name),
            "tket.futures" => self.handle_futures_op(hugr, node, &op_name),
            "tket.debug" => self.handle_debug_op(hugr, node, &op_name),
            "tket.bool" => self.handle_bool_op(hugr, node, &op_name),
            "tket.rotation" => self.handle_rotation_op(hugr, node, &op_name),
            "tket.modifier" => self.handle_modifier_op(hugr, node, &op_name),
            "tket.wasm" => self.handle_wasm_op(hugr, node, &op_name),
            "tket.guppy" => self.handle_guppy_op(hugr, node, &op_name),
            "tket.global_phase" => self.handle_global_phase_op(hugr, node, &op_name),
            "tket.quantum" => self.handle_quantum_extension_op(hugr, node, &op_name),
            "guppylang" => self.handle_guppylang_op(hugr, node, &op_name),
            "prelude" => self.handle_prelude_op(hugr, node, &op_name),
            "collections.array" => self.handle_array_op(hugr, node, &op_name),
            "collections.borrow_arr" => self.handle_borrow_arr_op(hugr, node, &op_name),
            "arithmetic.float" => self.handle_float_op(hugr, node, &op_name),
            "arithmetic.int" => self.handle_int_op(hugr, node, &op_name),
            "arithmetic.conversions" => self.handle_conversions_op(hugr, node, &op_name),
            _ => false,
        }
    }
}
