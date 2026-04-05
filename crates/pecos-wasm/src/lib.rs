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

//! WebAssembly foreign object support for PECOS
//!
//! Unified WebAssembly foreign object implementation that can be used
//! across different PECOS crates (pecos-qasm, pecos-phir-json, etc.) and exposed to Python
//! via `PyO3`.
//!
//! # Features
//!
//! - Thread-safe execution with RwLock/Mutex synchronization
//! - Configurable timeout support via epoch interruption
//! - Type conversion between i32/i64 with bounds checking
//! - Function discovery and caching
//! - Shot reinitialization support
//! - Resource cleanup via Drop trait
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "wasm")] {
//! use pecos_wasm::{WasmForeignObject, ForeignObject};
//!
//! let mut wasm = WasmForeignObject::new("module.wasm").unwrap();
//! wasm.init().unwrap();
//!
//! let result = wasm.exec("add", &[5, 3]).unwrap();
//! println!("Result: {:?}", result);
//! # }
//! ```

pub mod foreign_object;

#[cfg(feature = "wasm")]
pub mod wasmtime_foreign_object;

// Re-export main types
pub use foreign_object::{DummyForeignObject, ForeignObject};

#[cfg(feature = "wasm")]
pub use wasmtime_foreign_object::WasmForeignObject;
