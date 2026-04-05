// Copyright 2024 The PECOS Developers
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

//! LLVM IR generation using inkwell
//!
//! Rust types and functions for generating LLVM IR,
//! designed to be compatible with Python's llvmlite usage patterns.
//!
//! The main module is `llvm_compat`, which provides types for LLVM IR generation
//! that are compatible with Python's llvmlite API.

pub mod llvm_compat;
pub mod prelude;

// Re-export main types at crate root for convenience
pub use llvm_compat::{
    LLConstant, LLContext, LLFunction, LLFunctionType, LLIRBuilder, LLModule, LLResult, LLType,
    LLValue,
};
