// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Types submodule for `pecos_rslib`.
//!
//! This module provides a `pecos_rslib.types` submodule containing core data types:
//!
//! Core types:
//! - `Array` - N-dimensional array type
//! - `BitInt` - Fixed-width bit integer type
//! - `Pauli` - Single Pauli operator
//! - `PauliString` - String of Pauli operators
//!
//! Result types:
//! - `ShotVec` - Vector of shot results
//! - `ShotMap` - Map of shot results
//!
//! Message types:
//! - `ByteMessage` - Binary message type
//! - `ByteMessageBuilder` - Builder for binary messages
//!
//! Foreign object types (when `wasm` feature enabled):
//! - `WasmForeignObject` - WASM foreign object wrapper

use pyo3::prelude::*;

/// Register the 'types' submodule containing core data types.
///
/// This creates `pecos_rslib.types` with all type classes, enabling:
/// ```python
/// from pecos_rslib.types import Array, BitInt, Pauli
/// # or
/// import pecos_rslib.types as types
/// arr = types.Array([1, 2, 3])
/// ```
pub fn register_types_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let types = PyModule::new(py, "types")?;

    // Core types
    types.add("Array", parent.getattr("Array")?)?;
    types.add("BitInt", parent.getattr("BitInt")?)?;
    types.add("Pauli", parent.getattr("Pauli")?)?;
    types.add("PauliString", parent.getattr("PauliString")?)?;

    // Result types
    types.add("ShotVec", parent.getattr("ShotVec")?)?;
    types.add("ShotMap", parent.getattr("ShotMap")?)?;

    // Message types
    types.add("ByteMessage", parent.getattr("ByteMessage")?)?;
    types.add("ByteMessageBuilder", parent.getattr("ByteMessageBuilder")?)?;

    // Gate registry types
    types.add("GateRegistry", parent.getattr("GateRegistry")?)?;
    types.add("GateDefBuilder", parent.getattr("GateDefBuilder")?)?;
    types.add("AngleSource", parent.getattr("AngleSource")?)?;

    // Foreign object types (conditionally compiled)
    #[cfg(feature = "wasm")]
    types.add("WasmForeignObject", parent.getattr("WasmForeignObject")?)?;

    // Register in sys.modules for import statement support
    // This allows: `from pecos_rslib.types import Array`
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.types", &types)?;

    parent.add_submodule(&types)?;
    Ok(())
}
