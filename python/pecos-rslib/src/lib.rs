#![allow(clippy::needless_pass_by_value)] // PyO3 requires owned values from Python
#![doc(html_root_url = "https://docs.rs/pecos-rslib")]
// Disable doctests since they don't work with our workspace setup
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(test(no_crate_inject))]
#![doc(test(attr(deny(warnings))))]

// Copyright 2024 The PECOS Developers
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

mod prelude;

mod array_buffer;
mod bit_conversion;
mod bit_int_bindings;
mod bit_uint_bindings;
mod byte_message_bindings;
mod clifford_rep_bindings;
mod clifford_rz_bindings;
mod coin_toss_bindings;
mod dag_circuit_bindings;
mod decoder_bindings;
mod dtypes;
mod engine_bindings;
mod engine_builders;
mod engines_module;
mod experimental_bindings;
mod fault_tolerance_bindings;
mod gate_registry_bindings;
mod graph_bindings;
mod hugr_compilation_bindings;
mod namespace_modules;
mod num_bindings;
mod pauli_bindings;
mod pauli_prop_bindings;
mod pauli_sequence_bindings;
mod pecos_array;
mod pecos_random_bindings;
mod phir_json_bridge;
mod programs_module;
mod py_foreign_decoder;
mod py_foreign_simulator;
mod quest_bindings;
mod qulacs_bindings;
mod shot_results_bindings;
mod sim;
mod simulator_utils;
mod simulators_module;
mod sparse_sim;
mod sparse_stab_bindings;
mod sparse_stab_engine_bindings;
mod stab_bindings;
mod stabilizer_code_bindings;
mod stabilizer_group_bindings;
mod state_vec_bindings;
mod state_vec_engine_bindings;
mod types_module;
#[cfg(feature = "wasm")]
mod wasm_foreign_object_bindings;
mod wasm_program_bindings;

// Note: hugr_bindings module is currently disabled - conflicts with pecos-qis-interface due to duplicate symbols

use bit_int_bindings::PyBitInt;
use bit_uint_bindings::PyBitUInt;
use byte_message_bindings::{PyByteMessage, PyByteMessageBuilder};
use clifford_rz_bindings::PyCliffordRz;
use coin_toss_bindings::PyCoinToss;
use engine_builders::{PyHugr, PyPhirJson, PyQasm, PyQis};
use pauli_prop_bindings::PyPauliProp;
use pecos_array::Array;
use pecos_random_bindings::RngPcg;
use pyo3::prelude::*;
use quest_bindings::{QuestDensityMatrix, QuestStateVec};
use qulacs_bindings::PyQulacs;
use sparse_stab_bindings::PySparseStab;
use sparse_stab_engine_bindings::PySparseStabEngine;
use stab_bindings::PyStabilizer;
use state_vec_bindings::PyStateVec;
use state_vec_engine_bindings::PyStateVecEngine;
#[cfg(feature = "wasm")]
use wasm_foreign_object_bindings::PyWasmForeignObject;

/// Find an LLVM tool by name (e.g., "llvm-as", "llc", "opt").
///
/// This searches for the tool in the LLVM 14 installation using the same
/// logic as the pecos-build crate:
/// 1. ~/.pecos/llvm/ (PECOS managed installation)
/// 2. Project-local llvm/ directory
/// 3. System installations (Homebrew on macOS, package manager on Linux)
///
/// Returns None if the tool is not found.
#[pyfunction]
fn find_llvm_tool(tool_name: &str) -> Option<String> {
    pecos_build::llvm::find_tool(tool_name).map(|p| p.to_string_lossy().into_owned())
}

/// Set up the `QuEST` CUDA backend path environment variable for runtime loading.
/// This allows the Rust code to find and load the CUDA-accelerated `QuEST` backend
/// via dlopen when CUDA acceleration is requested.
fn setup_cuda_library_path() {
    // Only set if not already configured by the user
    if std::env::var("PECOS_QUEST_CUDA_LIB").is_ok() {
        log::debug!("PECOS_QUEST_CUDA_LIB already set, skipping auto-detection");
        return;
    }

    // Determine the QuEST CUDA backend filename based on platform
    #[cfg(target_os = "linux")]
    let cuda_backend_name = "libpecos_quest_cuda.so";
    #[cfg(target_os = "macos")]
    let cuda_backend_name = "libpecos_quest_cuda.dylib";
    #[cfg(target_os = "windows")]
    let cuda_backend_name = "pecos_quest_cuda.dll";
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    return;

    // Try to find the QuEST CUDA backend in common locations
    let search_paths = [
        // 1. Same directory as the current executable/library
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(cuda_backend_name))),
        // 2. ~/.pecos/lib/
        dirs::home_dir().map(|h| h.join(".pecos").join("lib").join(cuda_backend_name)),
        // 3. Cargo target directory (for development)
        Some(std::path::PathBuf::from("target/release").join(cuda_backend_name)),
    ];

    for path_opt in search_paths.into_iter().flatten() {
        if path_opt.exists() {
            log::info!("Found QuEST CUDA backend at: {}", path_opt.display());
            // SAFETY: Setting environment variables is safe in single-threaded context
            // during module initialization. This is called once before any other code runs.
            unsafe {
                std::env::set_var("PECOS_QUEST_CUDA_LIB", &path_opt);
            }
            return;
        }
    }

    log::debug!("QuEST CUDA backend not found in standard locations");
}

/// A Python module implemented in Rust.
/// Users should import from `pecos` (quantum-pecos) which re-exports these types
/// with additional Python-native enhancements.
#[pymodule]
#[allow(clippy::too_many_lines)] // Module initialization legitimately needs many lines
fn pecos_rslib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Note: Rust logging is controlled via RUST_LOG environment variable (e.g., RUST_LOG=debug)
    // We don't use pyo3-log because it interferes with Python's logging.basicConfig() in tests
    log::debug!("pecos_rslib module initializing...");

    // Set up QuEST CUDA backend path for runtime loading (before any QuEST usage)
    setup_cuda_library_path();

    // CRITICAL: Preload libselene_simple_runtime.so with RTLD_GLOBAL BEFORE anything else
    // This prevents conflicts with LLVM-14 when the Selene runtime is loaded later
    #[cfg(unix)]
    {
        use std::ffi::CString;

        const RTLD_LAZY: i32 = 0x00001;
        const RTLD_GLOBAL: i32 = 0x00100;

        log::debug!("Unix detected, attempting Selene runtime preload...");

        // Build search paths for libselene_simple_runtime.so:
        // 1. PECOS_SELENE_PRELOAD env var (explicit override)
        // 2. ~/.pecos/lib/
        // 3. Relative development paths (target/debug, target/release)
        let mut possible_paths: Vec<std::path::PathBuf> = Vec::new();

        if let Ok(path) = std::env::var("PECOS_SELENE_PRELOAD") {
            possible_paths.push(std::path::PathBuf::from(path));
        }

        if let Some(home) = dirs::home_dir() {
            possible_paths.push(home.join(".pecos/lib/libselene_simple_runtime.so"));
        }

        possible_paths.push("target/debug/libselene_simple_runtime.so".into());
        possible_paths.push("target/release/libselene_simple_runtime.so".into());

        log::debug!("Checking for Selene runtime libraries...");
        for path in &possible_paths {
            let path_str = path.to_string_lossy();
            log::trace!("Checking path: {path_str}");
            if path.exists() {
                log::debug!("Found Selene runtime! Attempting to preload: {path_str}");

                unsafe {
                    let path_cstr =
                        CString::new(path_str.as_bytes()).expect("path contains null byte");
                    let handle = libc::dlopen(path_cstr.as_ptr(), RTLD_LAZY | RTLD_GLOBAL);
                    if handle.is_null() {
                        let error_ptr = libc::dlerror();
                        if !error_ptr.is_null() {
                            let error = std::ffi::CStr::from_ptr(error_ptr).to_string_lossy();
                            log::warn!("Failed to preload {path_str}: {error}");
                        }
                    } else {
                        log::info!(
                            "Successfully preloaded Selene runtime with RTLD_GLOBAL from: {path_str}"
                        );
                        break;
                    }
                }
            }
        }
    }

    m.add_class::<PyCliffordRz>()?;
    m.add_class::<PySparseStab>()?;
    m.add_class::<PyStabilizer>()?;
    m.add_class::<phir_json_bridge::PhirJsonEngine>()?;
    m.add_class::<PyStateVec>()?;
    m.add_class::<PyQulacs>()?;
    m.add_class::<PyCoinToss>()?;
    m.add_class::<PyPauliProp>()?;
    m.add_class::<PyByteMessage>()?;
    m.add_class::<PyByteMessageBuilder>()?;
    m.add_class::<shot_results_bindings::PyShotVec>()?;
    m.add_class::<shot_results_bindings::PyShotMap>()?;
    m.add_class::<PyStateVecEngine>()?;
    m.add_class::<PySparseStabEngine>()?;
    m.add_class::<RngPcg>()?;
    m.add_class::<QuestStateVec>()?;
    m.add_class::<QuestDensityMatrix>()?;
    m.add_class::<Array>()?;
    m.add_class::<PyBitInt>()?;
    m.add_class::<PyBitUInt>()?;

    // Register simulator utilities (GateBindingsDict, TableauWrapper)
    simulator_utils::register_simulator_utils(m)?;

    // Register array buffer view types (for NumPy interop)
    m.add_class::<array_buffer::F64ArrayView>()?;
    m.add_class::<array_buffer::F32ArrayView>()?;
    m.add_class::<array_buffer::I64ArrayView>()?;
    m.add_class::<array_buffer::I32ArrayView>()?;
    m.add_class::<array_buffer::I16ArrayView>()?;
    m.add_class::<array_buffer::I8ArrayView>()?;
    m.add_class::<array_buffer::U64ArrayView>()?;
    m.add_class::<array_buffer::U32ArrayView>()?;
    m.add_class::<array_buffer::U16ArrayView>()?;
    m.add_class::<array_buffer::U8ArrayView>()?;
    m.add_class::<array_buffer::BoolArrayView>()?;
    m.add_class::<array_buffer::Complex64ArrayView>()?;
    m.add_class::<array_buffer::Complex32ArrayView>()?;

    // Register the unified sim() function
    sim::register_sim_module(m)?;

    // Register engine builders (QasmEngineBuilder, etc.)
    engine_builders::register_engine_builders(m)?;

    // Register HUGR compilation functions
    hugr_compilation_bindings::register_hugr_compilation_functions(m)?;

    // Register numerical computing module (scipy.optimize replacements)
    num_bindings::register_num_module(m)?;

    // Register dtypes module (Rust-backed dtype system)
    dtypes::register_dtypes_module(m)?;

    // Register Pauli types (quantum operators)
    pauli_bindings::register_pauli_types(m)?;

    // Register stabilizer group, Pauli sequence, and Clifford types
    stabilizer_group_bindings::register_stabilizer_group_types(m)?;
    stabilizer_code_bindings::register_stabilizer_code_types(m)?;
    pauli_sequence_bindings::register_pauli_sequence_types(m)?;
    clifford_rep_bindings::register_clifford_types(m)?;

    // Register graph module (graph algorithms for MWPM)
    graph_bindings::register_graph_module(m)?;

    // Register decoders module (QEC decoders: PyMatching, Fusion Blossom, LDPC, etc.)
    decoder_bindings::register_decoders_module(m)?;

    // Register quantum circuit types (DagCircuit, Gate, GateType, QubitId)
    dag_circuit_bindings::register_quantum_circuit_types(m)?;

    // Register gate registry types (GateRegistry, GateDefBuilder, AngleSource)
    gate_registry_bindings::register_gate_registry_types(m)?;

    // Register time unit types at top level (Nanoseconds, TimeUnits)
    dag_circuit_bindings::register_time_unit_types(m)?;

    // Register program types
    m.add_class::<PyQasm>()?;
    m.add_class::<PyQis>()?;
    m.add_class::<PyHugr>()?;
    m.add_class::<PyPhirJson>()?;
    wasm_program_bindings::register_wasm_programs(m)?;

    // Register engine builder functions
    m.add_function(wrap_pyfunction!(engine_builders::qasm_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::qis_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::selene_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::phir_json_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::phir_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::sim_builder, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::general_noise, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::depolarizing_noise, m)?)?;
    m.add_function(wrap_pyfunction!(
        engine_builders::biased_depolarizing_noise,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(engine_builders::state_vector, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::sparse_stab, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::stabilizer, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::clifford_rz, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::coin_toss, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(
        sparse_stab_bindings::adjust_tableau_string,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(find_llvm_tool, m)?)?;

    // Array creation function (NumPy-like interface, no NumPy dependency)
    m.add_function(wrap_pyfunction!(pecos_array::array, m)?)?;

    // Python foreign plugin types (Python-implemented decoders and simulators)
    m.add_class::<py_foreign_decoder::PyForeignDecoder>()?;
    m.add_class::<py_foreign_simulator::PyForeignSimulator>()?;

    // WebAssembly foreign object (optional)
    #[cfg(feature = "wasm")]
    m.add_class::<PyWasmForeignObject>()?;
    #[cfg(feature = "wasm")]
    wasm_foreign_object_bindings::register_wasm_types(m)?;

    // Register namespace modules (quantum, noise, llvm) for organizational structure
    // Note: This must come after all the factory functions and classes are registered
    namespace_modules::register_namespace_modules(m)?;

    // Register simulators submodule containing all quantum simulator backends
    simulators_module::register_simulators_module(m)?;

    // Register programs submodule containing all program types
    programs_module::register_programs_module(m)?;

    // Register engines submodule containing all execution engines and builders
    engines_module::register_engines_module(m)?;

    // Register types submodule containing core data types
    types_module::register_types_module(m)?;

    // Register experimental submodule (symbolic HUGR execution)
    experimental_bindings::register_experimental_module(m)?;

    // Register QEC fault tolerance submodule
    fault_tolerance_bindings::register_qec_module(m)?;

    // =========================================================================
    // Top-level numerical function exports (NumPy-like API)
    // These are convenience aliases for pecos_rslib.mean instead of pecos_rslib.num.mean
    // =========================================================================
    let num = m.getattr("num")?;

    // Statistical functions
    m.add("mean", num.getattr("mean")?)?;
    m.add("std", num.getattr("std")?)?;

    // Array reduction functions
    m.add("sum", num.getattr("sum")?)?;
    m.add("max", num.getattr("max")?)?;
    m.add("min", num.getattr("min")?)?;

    // Math functions (from num.math)
    let num_math = num.getattr("math")?;
    m.add("power", num_math.getattr("power")?)?;
    m.add("sqrt", num_math.getattr("sqrt")?)?;
    m.add("exp", num_math.getattr("exp")?)?;
    m.add("ln", num.getattr("ln")?)?;
    m.add("log", num.getattr("log")?)?;
    m.add("abs", num_math.getattr("abs")?)?;
    m.add("cos", num_math.getattr("cos")?)?;
    m.add("sin", num_math.getattr("sin")?)?;
    m.add("tan", num_math.getattr("tan")?)?;
    m.add("sinh", num_math.getattr("sinh")?)?;
    m.add("cosh", num_math.getattr("cosh")?)?;
    m.add("tanh", num_math.getattr("tanh")?)?;
    m.add("asin", num_math.getattr("asin")?)?;
    m.add("acos", num_math.getattr("acos")?)?;
    m.add("atan", num_math.getattr("atan")?)?;
    m.add("asinh", num_math.getattr("asinh")?)?;
    m.add("acosh", num_math.getattr("acosh")?)?;
    m.add("atanh", num_math.getattr("atanh")?)?;
    m.add("atan2", num_math.getattr("atan2")?)?;
    m.add("floor", num.getattr("floor")?)?;
    m.add("ceil", num.getattr("ceil")?)?;
    m.add("round", num.getattr("round")?)?;

    // Comparison functions (from num.compare)
    let num_compare = num.getattr("compare")?;
    m.add("isnan", num_compare.getattr("isnan")?)?;
    m.add("isclose", num_compare.getattr("isclose")?)?;
    m.add("allclose", num_compare.getattr("allclose")?)?;
    m.add("array_equal", num_compare.getattr("array_equal")?)?;
    m.add("all", num.getattr("all")?)?;
    m.add("any", num.getattr("any")?)?;
    m.add("where", num.getattr("where_array")?)?;

    // Optimization functions
    m.add("brentq", num.getattr("brentq")?)?;
    m.add("newton", num.getattr("newton")?)?;

    // Polynomial functions
    m.add("polyfit", num.getattr("polyfit")?)?;
    m.add("Poly1d", num.getattr("Poly1d")?)?;

    // Curve fitting
    m.add("curve_fit", num.getattr("curve_fit")?)?;

    // Array creation functions
    m.add("diag", num.getattr("diag")?)?;
    m.add("linspace", num.getattr("linspace")?)?;
    m.add("arange", num.getattr("arange")?)?;
    m.add("zeros", num.getattr("zeros")?)?;
    m.add("ones", num.getattr("ones")?)?;
    m.add("delete", num.getattr("delete")?)?;
    m.add("kron", num.getattr("kron")?)?;

    // Constants
    m.add("inf", num.getattr("inf")?)?;
    m.add("nan", num.getattr("nan")?)?;

    // Submodules as top-level exports
    m.add("random", num.getattr("random")?)?;
    m.add("stats", num.getattr("stats")?)?;

    // =========================================================================
    // Scalar type shortcuts (i8, i16, etc.)
    // These are convenience aliases for dtypes.i8.type
    // =========================================================================
    let dtypes = m.getattr("dtypes")?;
    m.add("i8", dtypes.getattr("i8")?.getattr("type")?)?;
    m.add("i16", dtypes.getattr("i16")?.getattr("type")?)?;
    m.add("i32", dtypes.getattr("i32")?.getattr("type")?)?;
    m.add("i64", dtypes.getattr("i64")?.getattr("type")?)?;
    m.add("u8", dtypes.getattr("u8")?.getattr("type")?)?;
    m.add("u16", dtypes.getattr("u16")?.getattr("type")?)?;
    m.add("u32", dtypes.getattr("u32")?.getattr("type")?)?;
    m.add("u64", dtypes.getattr("u64")?.getattr("type")?)?;
    m.add("f32", dtypes.getattr("f32")?.getattr("type")?)?;
    m.add("f64", dtypes.getattr("f64")?.getattr("type")?)?;
    m.add("complex64", dtypes.getattr("complex64")?.getattr("type")?)?;
    m.add("complex128", dtypes.getattr("complex128")?.getattr("type")?)?;
    m.add("angle64", m.py().get_type::<dtypes::ScalarAngle64>())?;

    // Note: Type aliases (Integer, Float, Complex, etc.) are now defined in quantum-pecos
    // (pecos.typing module) as they are Python TypeAlias constructs, not Rust types.
    // The .pyi stub file provides type information for static type checkers.

    // Add __version__ attribute
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
