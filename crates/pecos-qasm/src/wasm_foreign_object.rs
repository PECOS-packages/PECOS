//! WebAssembly Foreign Object Implementation
//!
//! This module provides WebAssembly support for QASM simulations, allowing you to call
//! WASM functions from within QASM programs.
//!
//! # Example
//!
//! ## QASM Usage
//!
//! ```text
//! OPENQASM 2.0;
//! creg a[10];
//! creg b[10];
//! creg result[10];
//!
//! a = 5;
//! b = 3;
//! result = add(a, b);      // Call WASM function
//! void_func(a, b);         // Call void WASM function
//! a = get_value();         // Call WASM function with no args
//! ```
//!
//! ## Rust Usage
//!
//! ```no_run
//! # #[cfg(feature = "wasm")] {
//! use pecos_qasm::qasm_engine;
//! use pecos_engines::ClassicalControlEngineBuilder;
//! use pecos_programs::Qasm;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     creg a[10];
//!     creg b[10];
//!     creg result[10];
//!
//!     a = 5;
//!     b = 3;
//!     result = add(a, b);
//! "#;
//!
//! // Run simulation with WASM module
//! let results = qasm_engine()
//!     .program(Qasm::from_string(qasm))
//!     .wasm("math.wasm")
//!     .to_sim()
//!     .run(100)
//!     .expect("Failed to run simulation");
//!
//! // Process results
//! for shot in &results.shots {
//!     let result_value = shot.data.get("result").unwrap();
//!     println!("Result: {:?}", result_value);
//! }
//! # }
//! ```
//!
//! # Requirements
//!
//! - WASM modules must export an `init()` function that is called at the start of each shot
//! - Functions can accept i32/i64 parameters and return i32/i64 values
//! - Built-in functions (sin, cos, tan, exp, ln, sqrt) cannot be overridden
//!
//! # Build-time Validation
//!
//! All function calls are validated at build time to ensure they exist in the WASM module.
//! This eliminates runtime errors for missing functions.

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

//! Re-export WebAssembly foreign object from pecos-wasm crate
//!
//! This module previously contained the `WasmtimeForeignObject` implementation,
//! but it has been moved to the unified pecos-wasm crate to avoid duplication
//! across different PECOS crates.

#[cfg(feature = "wasm")]
pub use pecos_wasm::WasmForeignObject as WasmtimeForeignObject;
