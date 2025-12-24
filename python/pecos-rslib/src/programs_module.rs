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

//! Programs submodule for `pecos_rslib`.
//!
//! This module provides a `pecos_rslib.programs` submodule containing all
//! program representation types:
//!
//! - `Qasm` - `OpenQASM` program representation
//! - `Qis` - QIS (Quantum Instruction Set) program representation
//! - `PhirJson` - PHIR JSON program representation
//! - `Hugr` - HUGR (Hierarchical Unified Graph Representation) program
//! - `Wasm` - WebAssembly bytecode program
//! - `Wat` - WebAssembly text format program

use pyo3::prelude::*;

/// Register the 'programs' submodule containing all program types.
///
/// This creates `pecos_rslib.programs` with all program classes, enabling:
/// ```python
/// from pecos_rslib.programs import Qasm, Wasm
/// # or
/// import pecos_rslib.programs as progs
/// prog = progs.Qasm(source)
/// ```
pub fn register_programs_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let programs = PyModule::new(py, "programs")?;

    // Add all program classes from the parent module
    // These are already registered at the top level, so we reference them via getattr

    // QASM/QIS programs
    programs.add("Qasm", parent.getattr("Qasm")?)?;
    programs.add("Qis", parent.getattr("Qis")?)?;

    // PHIR/HUGR programs
    programs.add("PhirJson", parent.getattr("PhirJson")?)?;
    programs.add("Hugr", parent.getattr("Hugr")?)?;

    // WebAssembly programs
    programs.add("Wasm", parent.getattr("Wasm")?)?;
    programs.add("Wat", parent.getattr("Wat")?)?;

    // Register in sys.modules for import statement support
    // This allows: `from pecos_rslib.programs import Qasm`
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.programs", &programs)?;

    parent.add_submodule(&programs)?;
    Ok(())
}
