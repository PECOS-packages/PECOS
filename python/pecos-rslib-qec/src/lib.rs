#![doc(html_root_url = "https://docs.rs/pecos-rslib-qec")]
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

mod decoder_bindings;

use pyo3::prelude::*;

/// QEC decoder Python bindings for PECOS.
#[pymodule]
fn pecos_rslib_qec(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register decoders submodule
    decoder_bindings::register_decoders_module(m)?;

    Ok(())
}
