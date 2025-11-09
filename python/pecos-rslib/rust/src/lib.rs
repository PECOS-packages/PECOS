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

mod byte_message_bindings;
mod coin_toss_bindings;
mod cpp_sparse_sim_bindings;
mod engine_bindings;
mod engine_builders;
mod noise_helpers;
mod pauli_prop_bindings;
// mod pcg_bindings;
mod hugr_compilation_bindings;
mod pecos_rng_bindings;
mod phir_json_bridge;
// mod qir_bindings;  // Removed - replaced by llvm_bindings
mod llvm_bindings;
mod quest_bindings;
mod qulacs_bindings;
mod shot_results_bindings;
mod sim;
mod sparse_sim;
mod sparse_stab_bindings;
mod sparse_stab_engine_bindings;
mod state_vec_bindings;
mod state_vec_engine_bindings;
#[cfg(feature = "wasm")]
mod wasm_foreign_object_bindings;

// Note: hugr_bindings module is currently disabled - conflicts with pecos-qis-interface due to duplicate symbols

use byte_message_bindings::{PyByteMessage, PyByteMessageBuilder};
use coin_toss_bindings::RsCoinToss;
use cpp_sparse_sim_bindings::CppSparseSim;
use engine_builders::{PyHugrProgram, PyPhirJsonProgram, PyQasmProgram, PyQisProgram};
use pauli_prop_bindings::PyPauliProp;
use pecos_rng_bindings::RngPcg;
use pyo3::prelude::*;
use quest_bindings::{QuestDensityMatrix, QuestStateVec};
use qulacs_bindings::RsQulacs;
use sparse_stab_bindings::SparseSim;
use sparse_stab_engine_bindings::PySparseStabEngine;
use state_vec_bindings::RsStateVec;
use state_vec_engine_bindings::PyStateVecEngine;
#[cfg(feature = "wasm")]
use wasm_foreign_object_bindings::PyWasmForeignObject;

/// Clear the global JIT compilation cache (deprecated - JIT is no longer available)
#[pyfunction]
fn clear_jit_cache() {
    // JIT has been removed - this function is now a no-op for compatibility
    log::warn!("clear_jit_cache() is deprecated - JIT has been removed from PECOS");
}

/// A Python module implemented in Rust.
#[pymodule]
fn _pecos_rslib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Note: Rust logging is controlled via RUST_LOG environment variable (e.g., RUST_LOG=debug)
    // We don't use pyo3-log because it interferes with Python's logging.basicConfig() in tests
    log::debug!("_pecos_rslib module initializing...");

    // CRITICAL: Preload libselene_simple_runtime.so with RTLD_GLOBAL BEFORE anything else
    // This prevents conflicts with LLVM-14 when the Selene runtime is loaded later
    #[cfg(unix)]
    {
        use std::ffi::CString;

        const RTLD_LAZY: i32 = 0x00001;
        const RTLD_GLOBAL: i32 = 0x00100;

        log::debug!("Unix detected, attempting Selene runtime preload...");

        // Try to find libselene_simple_runtime.so
        let possible_paths = [
            "/home/ciaranra/Repos/cl_projects/gup/selene/target/debug/libselene_simple_runtime.so",
            "/home/ciaranra/Repos/cl_projects/gup/selene/target/release/libselene_simple_runtime.so",
            "../selene/target/debug/libselene_simple_runtime.so",
            "../selene/target/release/libselene_simple_runtime.so",
        ];

        log::debug!("Checking for Selene runtime libraries...");
        for path in &possible_paths {
            log::trace!("Checking path: {path}");
            if std::path::Path::new(path).exists() {
                log::debug!("Found Selene runtime! Attempting to preload: {path}");

                unsafe {
                    let path_cstr = CString::new(path.as_bytes()).unwrap();
                    let handle = libc::dlopen(path_cstr.as_ptr(), RTLD_LAZY | RTLD_GLOBAL);
                    if handle.is_null() {
                        let error_ptr = libc::dlerror();
                        if !error_ptr.is_null() {
                            let error = std::ffi::CStr::from_ptr(error_ptr).to_string_lossy();
                            log::warn!("Failed to preload {path}: {error}");
                        }
                    } else {
                        log::info!(
                            "Successfully preloaded Selene runtime with RTLD_GLOBAL from: {path}"
                        );
                        break;
                    }
                }
            }
        }
    }

    m.add_class::<SparseSim>()?;
    m.add_class::<phir_json_bridge::PhirJsonEngine>()?;
    m.add_class::<CppSparseSim>()?;
    m.add_class::<RsStateVec>()?;
    m.add_class::<RsQulacs>()?;
    m.add_class::<RsCoinToss>()?;
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

    // Register the unified sim() function
    sim::register_sim_module(m)?;

    // Register engine builders (QasmEngineBuilder, etc.)
    engine_builders::register_engine_builders(m)?;

    // Register HUGR compilation functions
    hugr_compilation_bindings::register_hugr_compilation_functions(m)?;

    // Register LLVM IR generation module (compatible with Python's llvmlite API)
    llvm_bindings::register_llvm_module(m)?;

    // Register binding module for LLVM bitcode generation
    llvm_bindings::register_binding_module(m)?;

    // Register program types
    m.add_class::<PyQasmProgram>()?;
    m.add_class::<PyQisProgram>()?;
    m.add_class::<PyHugrProgram>()?;
    m.add_class::<PyPhirJsonProgram>()?;

    // Register engine builder functions
    m.add_function(wrap_pyfunction!(engine_builders::qasm_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::qis_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::selene_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::phir_json_engine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::sim_builder, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::general_noise, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::depolarizing_noise, m)?)?;
    m.add_function(wrap_pyfunction!(
        engine_builders::biased_depolarizing_noise,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(engine_builders::state_vector, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::sparse_stabilizer, m)?)?;
    m.add_function(wrap_pyfunction!(engine_builders::sparse_stab, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(clear_jit_cache, m)?)?;

    // WebAssembly foreign object (optional)
    #[cfg(feature = "wasm")]
    m.add_class::<PyWasmForeignObject>()?;

    Ok(())
}
