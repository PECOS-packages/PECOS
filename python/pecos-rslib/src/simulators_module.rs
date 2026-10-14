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
//! - `SparseStab` - Rust sparse stabilizer simulator
//! - `Stabilizer` - Generic stabilizer simulator (recommended)
//! - `StateVec` - State vector simulator
//! - `QuditStateVec` / `QutritStateVec` - Multilevel state-vector simulators
//! - `QuditDensityMatrix` / `QutritDensityMatrix` - Exact multilevel simulators
//! - `CoinToss` - Random measurement simulator for testing
//! - `PauliProp` - Pauli propagation/fault tracking simulator

use pyo3::prelude::*;

/// Register the 'simulators' submodule containing all quantum simulator backends.
///
/// This creates `pecos_rslib.simulators` with all simulator classes, enabling:
/// ```python
/// from pecos_rslib.simulators import SparseStab, StateVec
/// # or
/// import pecos_rslib.simulators as sims
/// sim = sims.SparseStab(10)
/// ```
pub fn register_simulators_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let simulators = PyModule::new(py, "simulators")?;

    // Add all simulator classes from the parent module
    // These are already registered at the top level, so we reference them via getattr

    // Stabilizer simulators
    simulators.add("SparseStab", parent.getattr("SparseStab")?)?;
    simulators.add("Stabilizer", parent.getattr("Stabilizer")?)?;

    // StabVec simulator (Clifford+RZ)
    simulators.add("StabVec", parent.getattr("StabVec")?)?;

    // State vector simulators
    simulators.add("StateVec", parent.getattr("StateVec")?)?;
    simulators.add("QuditStateVec", parent.getattr("QuditStateVec")?)?;
    simulators.add("QutritStateVec", parent.getattr("QutritStateVec")?)?;
    simulators.add("QuditDensityMatrix", parent.getattr("QuditDensityMatrix")?)?;
    simulators.add(
        "QutritDensityMatrix",
        parent.getattr("QutritDensityMatrix")?,
    )?;

    // Structured result and diagnostic types returned by multilevel simulators.
    simulators.add("KrausSample", parent.getattr("KrausSample")?)?;
    simulators.add("MeasurementSample", parent.getattr("MeasurementSample")?)?;
    simulators.add("InstrumentSample", parent.getattr("InstrumentSample")?)?;
    simulators.add(
        "DensityMatrixDiagnostics",
        parent.getattr("DensityMatrixDiagnostics")?,
    )?;
    simulators.add("basis_swap", parent.getattr("basis_swap")?)?;
    simulators.add(
        "embedded_qubit_unitary",
        parent.getattr("embedded_qubit_unitary")?,
    )?;
    simulators.add(
        "qutrit_leakage_channel",
        parent.getattr("qutrit_leakage_channel")?,
    )?;
    simulators.add(
        "qutrit_seepage_channel",
        parent.getattr("qutrit_seepage_channel")?,
    )?;

    // Exception family raised by the multilevel simulators.
    for name in [
        "QuditError",
        "QuditValueError",
        "QuditIndexError",
        "QuditMemoryError",
    ] {
        simulators.add(name, parent.getattr(name)?)?;
    }

    // Other simulators
    simulators.add("CoinToss", parent.getattr("CoinToss")?)?;
    simulators.add("PauliProp", parent.getattr("PauliProp")?)?;

    // Register in sys.modules for import statement support
    // This allows: `from pecos_rslib.simulators import SparseStab`
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.simulators", &simulators)?;

    parent.add_submodule(&simulators)?;
    Ok(())
}
