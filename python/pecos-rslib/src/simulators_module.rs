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

//! Simulators submodule for `pecos_rslib`.
//!
//! This module provides a `pecos_rslib.simulators` submodule containing all
//! quantum simulator backends:
//!
//! - `SparseSim` - Rust sparse stabilizer simulator
//! - `SparseSimCpp` - C++ sparse stabilizer simulator (via FFI)
//! - `StateVec` - State vector simulator
//! - `Qulacs` - Qulacs-based state vector simulator
//! - `CoinToss` - Random measurement simulator for testing
//! - `PauliProp` - Pauli propagation/fault tracking simulator
//! - `QuestStateVec` - `QuEST` state vector simulator
//! - `QuestDensityMatrix` - `QuEST` density matrix simulator

use pyo3::prelude::*;

/// Register the 'simulators' submodule containing all quantum simulator backends.
///
/// This creates `pecos_rslib.simulators` with all simulator classes, enabling:
/// ```python
/// from pecos_rslib.simulators import SparseSim, StateVec
/// # or
/// import pecos_rslib.simulators as sims
/// sim = sims.SparseSim(10)
/// ```
pub fn register_simulators_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let simulators = PyModule::new(py, "simulators")?;

    // Add all simulator classes from the parent module
    // These are already registered at the top level, so we reference them via getattr

    // Stabilizer simulators
    simulators.add("SparseSim", parent.getattr("SparseSim")?)?;
    simulators.add("SparseSimCpp", parent.getattr("SparseSimCpp")?)?;

    // State vector simulators
    simulators.add("StateVec", parent.getattr("StateVec")?)?;
    simulators.add("Qulacs", parent.getattr("Qulacs")?)?;

    // QuEST simulators
    simulators.add("QuestStateVec", parent.getattr("QuestStateVec")?)?;
    simulators.add("QuestDensityMatrix", parent.getattr("QuestDensityMatrix")?)?;

    // Other simulators
    simulators.add("CoinToss", parent.getattr("CoinToss")?)?;
    simulators.add("PauliProp", parent.getattr("PauliProp")?)?;

    // Register in sys.modules for import statement support
    // This allows: `from pecos_rslib.simulators import SparseSim`
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.simulators", &simulators)?;

    parent.add_submodule(&simulators)?;
    Ok(())
}
