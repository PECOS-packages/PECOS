// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! Re-export foreign object trait from pecos-wasm crate
//!
//! This module previously defined its own `ForeignObject` trait,
//! but now uses the unified trait from pecos-wasm.

// Re-export from pecos-wasm crate
#[cfg(feature = "wasm")]
pub use pecos_wasm::{DummyForeignObject, ForeignObject};

// For when wasm feature is disabled, provide minimal trait
#[cfg(not(feature = "wasm"))]
pub use pecos_wasm::{DummyForeignObject, ForeignObject};
