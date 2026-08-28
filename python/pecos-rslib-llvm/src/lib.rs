#![doc(html_root_url = "https://docs.rs/pecos-rslib-llvm")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(test(no_crate_inject))]
#![doc(test(attr(deny(warnings))))]

// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

mod hugr_compilation_bindings;
mod llvm_bindings;

use pyo3::prelude::*;

/// LLVM IR generation Python bindings for PECOS (llvmlite-compatible API).
#[pymodule]
fn pecos_rslib_llvm(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The Python distribution version from pyproject.toml, injected by build.rs. The crate
    // version (CARGO_PKG_VERSION) is a different number -- it rides the Rust workspace train.
    m.add("__version__", env!("PECOS_PYTHON_VERSION"))?;

    // Register LLVM IR module (pecos_rslib_llvm.ir)
    llvm_bindings::register_llvm_module(m)?;

    // Register binding module (pecos_rslib_llvm.binding)
    llvm_bindings::register_binding_module(m)?;

    // HUGR -> QIS (LLVM IR) compilation entry points. These live in the
    // LLVM wheel, not base pecos-rslib: they are what forces an LLVM link,
    // and the base wheel must import cleanly on machines without LLVM's
    // runtime dependencies (LLVM 21's Windows build pulls zlib.dll).
    hugr_compilation_bindings::register_hugr_compilation_functions(m)?;

    Ok(())
}
