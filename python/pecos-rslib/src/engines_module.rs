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

//! Engines submodule for `pecos_rslib`.
//!
//! This module provides a `pecos_rslib.engines` submodule containing all
//! execution engines and engine builders:
//!
//! Engine classes:
//! - `StateVecEngine` - State vector execution engine
//! - `SparseStabEngine` - Sparse stabilizer execution engine
//! - `PhirJsonEngine` - PHIR JSON execution engine
//!
//! Builder classes:
//! - `StateVectorEngineBuilder` - Builder for state vector engines
//! - `SparseStabEngineBuilder` - Builder for sparse stabilizer engines
//! - `QasmEngineBuilder` - Builder for QASM engines
//! - `QisEngineBuilder` - Builder for QIS engines
//! - `PhirJsonEngineBuilder` - Builder for PHIR JSON engines
//!
//! Factory functions:
//! - `qasm_engine()` - Create a QASM engine builder
//! - `qis_engine()` - Create a QIS engine builder
//! - `phir_json_engine()` - Create a PHIR JSON engine builder

use pyo3::prelude::*;

/// Register the 'engines' submodule containing all execution engines and builders.
///
/// This creates `pecos_rslib.engines` with all engine classes, enabling:
/// ```python
/// from pecos_rslib.engines import StateVecEngine, QasmEngineBuilder
/// # or
/// import pecos_rslib.engines as engines
/// builder = engines.qasm_engine()
/// ```
pub fn register_engines_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let engines = PyModule::new(py, "engines")?;

    // Engine classes
    engines.add("StateVecEngine", parent.getattr("StateVecEngine")?)?;
    engines.add("SparseStabEngine", parent.getattr("SparseStabEngine")?)?;
    engines.add("PhirJsonEngine", parent.getattr("PhirJsonEngine")?)?;

    // Builder classes
    engines.add(
        "StateVectorEngineBuilder",
        parent.getattr("StateVectorEngineBuilder")?,
    )?;
    engines.add(
        "SparseStabEngineBuilder",
        parent.getattr("SparseStabEngineBuilder")?,
    )?;
    engines.add(
        "StabilizerEngineBuilder",
        parent.getattr("StabilizerEngineBuilder")?,
    )?;
    engines.add(
        "StabVecEngineBuilder",
        parent.getattr("StabVecEngineBuilder")?,
    )?;
    engines.add(
        "DensityMatrixEngineBuilder",
        parent.getattr("DensityMatrixEngineBuilder")?,
    )?;
    engines.add(
        "CoinTossEngineBuilder",
        parent.getattr("CoinTossEngineBuilder")?,
    )?;
    engines.add("QasmEngineBuilder", parent.getattr("QasmEngineBuilder")?)?;
    engines.add("QisEngineBuilder", parent.getattr("QisEngineBuilder")?)?;
    engines.add(
        "PhirJsonEngineBuilder",
        parent.getattr("PhirJsonEngineBuilder")?,
    )?;
    engines.add("PhirEngineBuilder", parent.getattr("PhirEngineBuilder")?)?;

    // Factory functions
    engines.add_function(parent.getattr("qasm_engine")?.extract()?)?;
    engines.add_function(parent.getattr("qis_engine")?.extract()?)?;
    engines.add_function(parent.getattr("phir_json_engine")?.extract()?)?;
    engines.add_function(parent.getattr("phir_engine")?.extract()?)?;

    // Register in sys.modules for import statement support
    // This allows: `from pecos_rslib.engines import StateVecEngine`
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.engines", &engines)?;

    parent.add_submodule(&engines)?;
    Ok(())
}
