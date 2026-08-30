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

//! Simulator utilities implemented in Rust.
//!
//! This module provides `GateBindingsDict` and `TableauWrapper` classes
//! that were previously implemented in Python.

use pyo3::exceptions::PyKeyError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyModule};
use std::collections::HashMap;

use crate::dtypes::AngleParam;
use crate::sparse_stab_bindings::adjust_tableau_string;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Symbol {
    I,
    X,
    Y,
    Z,
    H,
    H2,
    H3,
    H4,
    H5,
    H6,
    F,
    Fdg,
    F2,
    F2dg,
    F3,
    F3dg,
    F4,
    F4dg,
    Sx,
    Sxdg,
    Sy,
    Sydg,
    Sz,
    Szdg,
    T,
    Tdg,
    Pz,
    PzForced,
    Pnz,
    Px,
    Pnx,
    Py,
    Pny,
    InitZ,
    InitNz,
    InitX,
    InitNx,
    InitY,
    InitNy,
    Mz,
    MzForced,
    MeasureZ,
    Mx,
    My,
    Rx,
    Ry,
    Rz,
    Rxy1q,
    U,
    Cx,
    Cy,
    Cz,
    Sxx,
    Msxx,
    Sxxdg,
    Syy,
    Syydg,
    Szz,
    Szzdg,
    Swap,
    G,
    Gdg,
    Iswap,
    Iswapdg,
    Rxx,
    Ryy,
    Rzz,
    RxxRyyRzz,
    Ii,
    Crx,
    Cry,
    Crz,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ParameterMode {
    None,
    Angle,
    Angles2,
    Angles3,
    ForcedOutcome,
    OptionalForcedOutcome,
}

impl ParameterMode {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Angle => "angle",
            Self::Angles2 => "angles:2",
            Self::Angles3 => "angles:3",
            Self::ForcedOutcome => "forced_outcome",
            Self::OptionalForcedOutcome => "optional_forced_outcome",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SymbolEntry {
    pub(crate) spelling: &'static str,
    pub(crate) symbol: Symbol,
    pub(crate) parameter_mode: ParameterMode,
}

impl SymbolEntry {
    pub(crate) const fn is_measurement(self) -> bool {
        matches!(
            self.symbol,
            Symbol::Mz | Symbol::MzForced | Symbol::MeasureZ | Symbol::Mx | Symbol::My
        )
    }

    pub(crate) const fn qubit_count(self) -> u8 {
        if matches!(
            self.symbol,
            Symbol::Cx
                | Symbol::Cy
                | Symbol::Cz
                | Symbol::Sxx
                | Symbol::Msxx
                | Symbol::Sxxdg
                | Symbol::Syy
                | Symbol::Syydg
                | Symbol::Szz
                | Symbol::Szzdg
                | Symbol::Swap
                | Symbol::G
                | Symbol::Gdg
                | Symbol::Iswap
                | Symbol::Iswapdg
                | Symbol::Rxx
                | Symbol::Ryy
                | Symbol::Rzz
                | Symbol::RxxRyyRzz
                | Symbol::Ii
                | Symbol::Crx
                | Symbol::Cry
                | Symbol::Crz
        ) {
            2
        } else {
            1
        }
    }
}

macro_rules! symbol_entries {
    ($($symbol:ident, $mode:ident, [$($spelling:literal),+ $(,)?];)+) => {
        &[
            $($(SymbolEntry {
                spelling: $spelling,
                symbol: Symbol::$symbol,
                parameter_mode: ParameterMode::$mode,
            }),+),+
        ]
    };
}

macro_rules! supports_exact {
    ($entry:ident; $($symbol:ident => [$($spelling:literal),+ $(,)?];)+) => {
        match $entry.symbol {
            $($crate::simulator_utils::Symbol::$symbol => matches!($entry.spelling, $($spelling)|+),)+
            _ => false,
        }
    };
}

pub(crate) use supports_exact;

#[cfg(test)]
macro_rules! direct_surface_test {
    ($name:ident, $simulator:expr) => {
        #[test]
        fn $name() {
            pyo3::Python::initialize();
            pyo3::Python::attach(|py| {
                for entry in $crate::simulator_utils::SYMBOL_ENTRIES {
                    let mut simulator = $simulator;
                    let params = pyo3::types::PyDict::new(py);
                    match entry.parameter_mode {
                        $crate::simulator_utils::ParameterMode::None => {}
                        $crate::simulator_utils::ParameterMode::Angle => {
                            params
                                .set_item("angle", std::f64::consts::FRAC_PI_2)
                                .unwrap();
                        }
                        $crate::simulator_utils::ParameterMode::Angles2 => {
                            params
                                .set_item("angles", (std::f64::consts::FRAC_PI_2, 0.0))
                                .unwrap();
                        }
                        $crate::simulator_utils::ParameterMode::Angles3 => {
                            params.set_item("angles", (0.0, 0.0, 0.0)).unwrap();
                        }
                        $crate::simulator_utils::ParameterMode::ForcedOutcome
                        | $crate::simulator_utils::ParameterMode::OptionalForcedOutcome => {
                            params.set_item("forced_outcome", 1).unwrap();
                        }
                    }
                    let params = (!params.is_empty()).then_some(&params);
                    let accepted = if entry.qubit_count() == 1 {
                        simulator.run_1q_gate(entry.spelling, 0, params).is_ok()
                    } else {
                        let location = pyo3::types::PyTuple::new(py, [0_usize, 1_usize]).unwrap();
                        simulator
                            .run_2q_gate(entry.spelling, &location, params)
                            .is_ok()
                    };

                    assert_eq!(
                        accepted,
                        supports(entry),
                        "direct dispatch and support predicate disagree for {}",
                        entry.spelling
                    );
                }

                let empty_locations = pyo3::types::PySet::empty(py).unwrap();
                let mut unsupported_simulator = $simulator;
                let unsupported_error = unsupported_simulator
                    .run_gate_highlevel("NOPE", empty_locations.as_any(), None, py)
                    .unwrap_err();
                assert!(
                    unsupported_error
                        .to_string()
                        .contains("Unsupported single-qubit gate: NOPE")
                );

                let supported_spelling = $crate::simulator_utils::SYMBOL_ENTRIES
                    .iter()
                    .find(|entry| supports(entry))
                    .unwrap()
                    .spelling;
                let mut supported_simulator = $simulator;
                let output = supported_simulator
                    .run_gate_highlevel(supported_spelling, empty_locations.as_any(), None, py)
                    .unwrap();
                assert!(output.bind(py).is_empty());
            });
        }
    };
}

#[cfg(test)]
pub(crate) use direct_surface_test;

/// Exact gate spellings accepted by at least one Rust-backed simulator.
pub(crate) const SYMBOL_ENTRIES: &[SymbolEntry] = symbol_entries! {
    I, None, ["I"];
    X, None, ["X"];
    Y, None, ["Y"];
    Z, None, ["Z"];
    H, None, ["H", "H1", "H+z+x"];
    H2, None, ["H2", "H-z-x"];
    H3, None, ["H3", "H+y-z"];
    H4, None, ["H4", "H-y-z"];
    H5, None, ["H5", "H-x+y"];
    H6, None, ["H6", "H-x-y"];
    F, None, ["F", "F1"];
    Fdg, None, ["Fdg", "F1d", "F1dg"];
    F2, None, ["F2"];
    F2dg, None, ["F2dg", "F2d"];
    F3, None, ["F3"];
    F3dg, None, ["F3dg", "F3d"];
    F4, None, ["F4"];
    F4dg, None, ["F4dg", "F4d"];
    Sx, None, ["Q", "SX", "SqrtX"];
    Sxdg, None, ["Qd", "SXdg", "SqrtXd", "SqrtXdg"];
    Sy, None, ["R", "SY", "SqrtY"];
    Sydg, None, ["Rd", "SYdg", "SqrtYd", "SqrtYdg"];
    Sz, None, ["S", "SZ", "SqrtZ"];
    Szdg, None, ["Sd", "SZdg", "SqrtZd", "SqrtZdg"];
    T, None, ["T"];
    Tdg, None, ["Tdg"];
    Pz, None, ["PZ"];
    PzForced, ForcedOutcome, ["PZForced"];
    Pnz, None, ["PNZ"];
    Px, None, ["PX"];
    Pnx, None, ["PNX"];
    Py, None, ["PY"];
    Pny, None, ["PNY"];
    InitZ, OptionalForcedOutcome,
        ["Init", "Init +Z", "init |0>", "leak", "leak |0>", "unleak |0>"];
    InitNz, None, ["Init -Z", "init |1>", "leak |1>", "unleak |1>"];
    InitX, None, ["Init +X", "init |+>"];
    InitNx, None, ["Init -X", "init |->"];
    InitY, None, ["Init +Y", "init |+i>"];
    InitNy, None, ["Init -Y", "init |-i>"];
    Mz, None, ["MZ"];
    MzForced, ForcedOutcome, ["MZForced"];
    MeasureZ, OptionalForcedOutcome, ["Measure", "measure Z", "Measure +Z"];
    Mx, None, ["MX", "Measure +X", "measure X"];
    My, None, ["MY", "Measure +Y", "measure Y"];
    Rx, Angle, ["RX"];
    Ry, Angle, ["RY"];
    Rz, Angle, ["RZ"];
    Rxy1q, Angles2, ["RXY1Q", "R1XY"];
    U, Angles3, ["U"];
    Cx, None, ["CX", "CNOT"];
    Cy, None, ["CY"];
    Cz, None, ["CZ"];
    Sxx, None, ["SXX", "SqrtXX"];
    Msxx, None, ["MS", "MSXX"];
    Sxxdg, None, ["SXXdg", "SqrtXXd", "SqrtXXdg"];
    Syy, None, ["SYY", "SqrtYY"];
    Syydg, None, ["SYYdg", "SqrtYYd", "SqrtYYdg"];
    Szz, None, ["SZZ", "SqrtZZ"];
    Szzdg, None, ["SZZdg", "SqrtZZd", "SqrtZZdg"];
    Swap, None, ["SWAP"];
    G, None, ["G", "G2"];
    Gdg, None, ["Gdg"];
    Iswap, None, ["ISWAP"];
    Iswapdg, None, ["ISWAPdg"];
    Rxx, Angle, ["RXX"];
    Ryy, Angle, ["RYY"];
    Rzz, Angle, ["RZZ"];
    RxxRyyRzz, Angles3, ["RXXRYYRZZ", "RZZRYYRXX", "R2XXYYZZ", "RXXYYZZ"];
    Ii, None, ["II"];
    Crx, Angle, ["CRX"];
    Cry, Angle, ["CRY"];
    Crz, Angle, ["CRZ"];
};

pub(crate) fn resolve_symbol(spelling: &str) -> Option<&'static SymbolEntry> {
    SYMBOL_ENTRIES
        .iter()
        .find(|entry| entry.spelling == spelling)
}

/// Raw generators data: `(col_x, col_z, row_x, row_z)`.
pub type GensData = (
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
);

/// Special dict that delegates all gate lookups to Rust's `run_gate()`.
///
/// This provides backwards compatibility for code that accesses sim.bindings[`gate_name`].
/// Instead of storing lambdas for every gate, we create them on-demand.
#[pyclass(mapping)]
pub struct GateBindingsDict {
    sim: Py<PyAny>,
    cache: HashMap<String, Py<PyAny>>,
    supports: fn(&SymbolEntry) -> bool,
}

static GATE_LAMBDA_FACTORY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

impl GateBindingsDict {
    /// Create a new `GateBindingsDict` from Rust code.
    pub(crate) fn new(sim: Py<PyAny>, supports: fn(&SymbolEntry) -> bool) -> Self {
        Self {
            sim,
            cache: HashMap::new(),
            supports,
        }
    }
}

#[pymethods]
impl GateBindingsDict {
    fn __getitem__(&mut self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.cache.get(key) {
            return Ok(cached.clone_ref(py));
        }

        let entry = resolve_symbol(key)
            .filter(|entry| (self.supports)(entry))
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))?;
        let gate_name = key.to_string();

        let factory = GATE_LAMBDA_FACTORY.get_or_try_init(py, || {
            let code = c_str!(
                r#"
def make_gate_lambda(sim, gate_name, is_measurement):
    def gate_lambda(simulator, location, **params):
        if isinstance(location, (tuple, list)):
            loc_tuple = tuple(location)
        else:
            loc_tuple = (location,)

        loc_set = {loc_tuple}
        result_dict = sim.run_gate(gate_name, loc_set, **params)

        if result_dict:
            if len(loc_tuple) == 1 and loc_tuple[0] in result_dict:
                return result_dict[loc_tuple[0]]
            return result_dict.get(loc_tuple)
        return 0 if is_measurement else None

    return gate_lambda
"#
            );
            let module = PyModule::from_code(
                py,
                code,
                c_str!("_pecos_gate_bindings"),
                c_str!("_pecos_gate_bindings"),
            )?;
            Ok::<_, PyErr>(module.getattr("make_gate_lambda")?.unbind())
        })?;
        let gate_lambda = factory
            .bind(py)
            .call1((self.sim.clone_ref(py), &gate_name, entry.is_measurement()))?
            .unbind();

        // Cache the lambda
        self.cache
            .insert(key.to_string(), gate_lambda.clone_ref(py));

        Ok(gate_lambda)
    }

    fn __setitem__(&mut self, _py: Python<'_>, key: &str, value: Py<PyAny>) {
        // Store the value in the cache (allows overriding gate lambdas)
        self.cache.insert(key.to_string(), value);
    }

    fn __contains__(&self, key: &str) -> bool {
        self.cache.contains_key(key)
            || resolve_symbol(key).is_some_and(|entry| (self.supports)(entry))
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &mut self,
        py: Python<'_>,
        key: &str,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if self.__contains__(key) {
            self.__getitem__(py, key)
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }

    /// Return the number of cached generated callables and overrides.
    fn __len__(&self) -> usize {
        self.cache.len()
    }

    /// Return cached generated-callable and override keys, not every supported symbol.
    fn keys(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
}

#[pyfunction(name = "_gate_bindings_symbols")]
fn gate_bindings_symbols() -> Vec<(String, bool, String, u8)> {
    SYMBOL_ENTRIES
        .iter()
        .map(|entry| {
            (
                entry.spelling.to_string(),
                entry.is_measurement(),
                entry.parameter_mode.name().to_string(),
                entry.qubit_count(),
            )
        })
        .collect()
}

/// Wrapper for accessing stabilizer/destabilizer tableaus from simulators.
#[pyclass]
pub struct TableauWrapper {
    sim: Py<PyAny>,
    is_stab: bool,
}

impl TableauWrapper {
    /// Create a new `TableauWrapper` from Rust code.
    pub fn new(sim: Py<PyAny>, is_stab: bool) -> Self {
        Self { sim, is_stab }
    }
}

#[pymethods]
impl TableauWrapper {
    #[new]
    #[pyo3(signature = (sim, *, is_stab))]
    fn py_new(sim: Py<PyAny>, is_stab: bool) -> Self {
        Self::new(sim, is_stab)
    }

    #[pyo3(signature = (*, verbose = false))]
    fn print_tableau(&self, py: Python<'_>, verbose: bool) -> PyResult<Vec<String>> {
        // Get the tableau from the simulator
        let tableau: String = if self.is_stab {
            self.sim.call_method0(py, "stab_tableau")?.extract(py)?
        } else {
            self.sim.call_method0(py, "destab_tableau")?.extract(py)?
        };

        // Split into lines and adjust each
        let lines: Vec<String> = tableau
            .lines()
            .map(|line| adjust_tableau_string(line, self.is_stab, false))
            .collect();

        // Print if verbose
        if verbose {
            for line in &lines {
                println!("{line}");
            }
        }

        Ok(lines)
    }

    /// Helper to get raw gens data from the simulator.
    fn get_gens_data(&self, py: Python<'_>) -> PyResult<GensData> {
        self.sim
            .call_method1(py, "_gens_data", (self.is_stab,))?
            .extract(py)
    }

    #[getter]
    fn col_x(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.0)
    }

    #[getter]
    fn col_z(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.1)
    }

    #[getter]
    fn row_x(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.2)
    }

    #[getter]
    fn row_z(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.get_gens_data(py)?.3)
    }
}

/// Register the simulator utils module
pub fn register_simulator_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GateBindingsDict>()?;
    m.add_class::<TableauWrapper>()?;
    m.add_function(wrap_pyfunction!(gate_bindings_symbols, m)?)?;
    Ok(())
}

// --- Shared batch dispatch for simulator bindings ---

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{CliffordGateable, MeasurementResult};
use pyo3::types::{PySet, PyTuple};

/// Reject a symbol outside a simulator's declared surface before execution can be skipped.
pub(crate) fn validate_supported_symbol(
    symbol: &str,
    locations: &Bound<'_, PySet>,
    supports: fn(&SymbolEntry) -> bool,
) -> PyResult<()> {
    if resolve_symbol(symbol).is_some_and(supports) {
        return Ok(());
    }

    let arity = locations.iter().next().map_or(1, |location| {
        location
            .cast::<PyTuple>()
            .map_or(1, pyo3::types::PyTupleMethods::len)
    });
    let gate_kind = if arity == 2 { "two" } else { "single" };
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "Unsupported {gate_kind}-qubit gate: {symbol}"
    )))
}

/// Extract a single qubit index from a Python location.
/// Handles both bare ints and 1-tuples like `(0,)` (the `GateBindingsDict` wraps ints in tuples).
pub fn extract_single_qubit(location: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(q) = location.extract::<usize>() {
        return Ok(q);
    }
    if let Ok(tuple) = location.cast::<PyTuple>()
        && tuple.len() == 1
    {
        return tuple.get_item(0)?.extract::<usize>();
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected int or 1-tuple for single-qubit location, got {:?}",
        location.get_type().name()?
    )))
}

/// Extract one angle from the `"angle"` parameter.
pub fn extract_angle(params: Option<&Bound<'_, PyDict>>, gate_name: &str) -> PyResult<Angle64> {
    let params = params.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{gate_name} requires params with 'angle'"))
    })?;
    let value = params.get_item("angle")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{gate_name} requires an 'angle' parameter"
        ))
    })?;
    let angle = value.extract::<AngleParam>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Expected a valid angle parameter for {gate_name}"
        ))
    })?;
    Ok(angle.0)
}

/// Extract exactly `count` angles from the `"angles"` parameter.
pub fn extract_angles(
    params: Option<&Bound<'_, PyDict>>,
    gate_name: &str,
    count: usize,
) -> PyResult<Vec<Angle64>> {
    let params = params.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{gate_name} requires params with 'angles'"
        ))
    })?;
    let value = params.get_item("angles")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{gate_name} requires an 'angles' parameter"
        ))
    })?;
    let angles = value.extract::<Vec<AngleParam>>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Expected valid angle parameters for {gate_name}"
        ))
    })?;
    if angles.len() != count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Gate {gate_name} expected {count} angle parameters, got {}",
            angles.len()
        )));
    }
    Ok(angles.into_iter().map(|angle| angle.0).collect())
}

/// Collect single-qubit locations from a Python set into a Vec of `QubitIds`.
fn collect_single_qubits(locations: &Bound<'_, PySet>) -> PyResult<Vec<QubitId>> {
    locations
        .iter()
        .map(|l| Ok(QubitId(extract_single_qubit(&l)?)))
        .collect()
}

/// Collect single-qubit locations as raw usize values.
fn collect_single_qubit_indices(locations: &Bound<'_, PySet>) -> PyResult<Vec<usize>> {
    locations.iter().map(|l| extract_single_qubit(&l)).collect()
}

/// Collect two-qubit pair locations from a Python set.
fn collect_pairs(locations: &Bound<'_, PySet>) -> PyResult<Vec<(QubitId, QubitId)>> {
    locations
        .iter()
        .map(|l| {
            let t: (usize, usize) = l.extract()?;
            Ok((QubitId(t.0), QubitId(t.1)))
        })
        .collect()
}

/// Build a measurement output dict from qubit indices and results.
fn build_meas_output(
    py: Python<'_>,
    qubits: &[usize],
    results: Vec<MeasurementResult>,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    for (&q, r) in qubits.iter().zip(results) {
        if r.outcome {
            output.set_item(q, 1u8)?;
        }
    }
    Ok(output.into())
}

/// Try to dispatch a gate in batch mode for any `CliffordGateable` simulator.
///
/// Returns `Some(output_dict)` if the gate was handled, `None` to fall back to
/// per-location dispatch (for parameterized gates, unknown symbols, etc.).
pub fn try_clifford_batch_dispatch<S: CliffordGateable>(
    sim: &mut S,
    symbol: &str,
    locations: &Bound<'_, PySet>,
    py: Python<'_>,
) -> PyResult<Option<Py<PyDict>>> {
    match symbol {
        // Identity
        "I" => return Ok(Some(PyDict::new(py).into())),

        // Single-qubit Clifford gates (no return value)
        "X" | "Y" | "Z" | "H" | "H1" | "H+z+x" | "H2" | "H-z-x" | "H3" | "H+y-z" | "H4"
        | "H-y-z" | "H5" | "H-x+y" | "H6" | "H-x-y" | "F" | "F1" | "Fdg" | "F1d" | "F1dg"
        | "F2" | "F2dg" | "F2d" | "F3" | "F3dg" | "F3d" | "F4" | "F4dg" | "F4d" | "Q" | "SX"
        | "SqrtX" | "Qd" | "SXdg" | "SqrtXd" | "SqrtXdg" | "R" | "SY" | "SqrtY" | "Rd" | "SYdg"
        | "SqrtYd" | "SqrtYdg" | "S" | "SZ" | "SqrtZ" | "Sd" | "SZdg" | "SqrtZd" | "SqrtZdg" => {
            let qubits = collect_single_qubits(locations)?;
            match symbol {
                "X" => {
                    sim.x(&qubits);
                }
                "Y" => {
                    sim.y(&qubits);
                }
                "Z" => {
                    sim.z(&qubits);
                }
                "H" | "H1" | "H+z+x" => {
                    sim.h(&qubits);
                }
                "H2" | "H-z-x" => {
                    sim.h2(&qubits);
                }
                "H3" | "H+y-z" => {
                    sim.h3(&qubits);
                }
                "H4" | "H-y-z" => {
                    sim.h4(&qubits);
                }
                "H5" | "H-x+y" => {
                    sim.h5(&qubits);
                }
                "H6" | "H-x-y" => {
                    sim.h6(&qubits);
                }
                "F" | "F1" => {
                    sim.f(&qubits);
                }
                "Fdg" | "F1d" | "F1dg" => {
                    sim.fdg(&qubits);
                }
                "F2" => {
                    sim.f2(&qubits);
                }
                "F2dg" | "F2d" => {
                    sim.f2dg(&qubits);
                }
                "F3" => {
                    sim.f3(&qubits);
                }
                "F3dg" | "F3d" => {
                    sim.f3dg(&qubits);
                }
                "F4" => {
                    sim.f4(&qubits);
                }
                "F4dg" | "F4d" => {
                    sim.f4dg(&qubits);
                }
                "Q" | "SX" | "SqrtX" => {
                    sim.sx(&qubits);
                }
                "Qd" | "SXdg" | "SqrtXd" | "SqrtXdg" => {
                    sim.sxdg(&qubits);
                }
                "R" | "SY" | "SqrtY" => {
                    sim.sy(&qubits);
                }
                "Rd" | "SYdg" | "SqrtYd" | "SqrtYdg" => {
                    sim.sydg(&qubits);
                }
                "S" | "SZ" | "SqrtZ" => {
                    sim.sz(&qubits);
                }
                "Sd" | "SZdg" | "SqrtZd" | "SqrtZdg" => {
                    sim.szdg(&qubits);
                }
                _ => unreachable!(),
            }
            return Ok(Some(PyDict::new(py).into()));
        }

        // Preparations (no return value)
        "PZ" | "Init" | "Init +Z" | "init |0>" | "leak" | "leak |0>" | "unleak |0>" => {
            sim.pz(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PNZ" | "Init -Z" | "init |1>" | "leak |1>" | "unleak |1>" => {
            sim.pnz(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PX" | "Init +X" | "init |+>" => {
            sim.px(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PNX" | "Init -X" | "init |->" => {
            sim.pnx(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PY" | "Init +Y" | "init |+i>" => {
            sim.py(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }
        "PNY" | "Init -Y" | "init |-i>" => {
            sim.pny(&collect_single_qubits(locations)?);
            return Ok(Some(PyDict::new(py).into()));
        }

        // Measurements (return outcomes)
        "MZ" | "Measure" | "measure Z" | "Measure +Z" => {
            let qubits = collect_single_qubit_indices(locations)?;
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
            let results = sim.mz(&qubit_ids);
            return Ok(Some(build_meas_output(py, &qubits, results)?));
        }
        "MX" | "Measure +X" => {
            let qubits = collect_single_qubit_indices(locations)?;
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
            let results = sim.mx(&qubit_ids);
            return Ok(Some(build_meas_output(py, &qubits, results)?));
        }
        "MY" | "Measure +Y" => {
            let qubits = collect_single_qubit_indices(locations)?;
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
            let results = sim.my(&qubit_ids);
            return Ok(Some(build_meas_output(py, &qubits, results)?));
        }

        // Two-qubit Clifford gates (no return value)
        "CX" | "CNOT" | "CY" | "CZ" | "SZZ" | "SZZdg" | "SXX" | "SXXdg" | "SYY" | "SYYdg"
        | "SqrtZZ" | "SqrtZZd" | "SqrtXX" | "SqrtXXd" | "SqrtYY" | "SqrtYYd" | "SWAP" | "G"
        | "G2" | "Gdg" | "ISWAP" | "ISWAPdg" => {
            let pairs = collect_pairs(locations)?;
            match symbol {
                "CX" | "CNOT" => {
                    sim.cx(&pairs);
                }
                "CY" => {
                    sim.cy(&pairs);
                }
                "CZ" => {
                    sim.cz(&pairs);
                }
                "SZZ" | "SqrtZZ" => {
                    sim.szz(&pairs);
                }
                "SZZdg" | "SqrtZZd" => {
                    sim.szzdg(&pairs);
                }
                "SXX" | "SqrtXX" => {
                    sim.sxx(&pairs);
                }
                "SXXdg" | "SqrtXXd" => {
                    sim.sxxdg(&pairs);
                }
                "SYY" | "SqrtYY" => {
                    sim.syy(&pairs);
                }
                "SYYdg" | "SqrtYYd" => {
                    sim.syydg(&pairs);
                }
                "SWAP" => {
                    sim.swap(&pairs);
                }
                "G" | "G2" => {
                    sim.g(&pairs);
                }
                "Gdg" => {
                    sim.gdg(&pairs);
                }
                "ISWAP" => {
                    sim.iswap(&pairs);
                }
                "ISWAPdg" => {
                    sim.iswapdg(&pairs);
                }
                _ => unreachable!(),
            }
            return Ok(Some(PyDict::new(py).into()));
        }

        _ => {}
    }

    Ok(None)
}
