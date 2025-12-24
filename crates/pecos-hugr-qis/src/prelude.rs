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

//! A prelude for users of the `pecos-hugr-qis` crate.
//!
//! This prelude re-exports the HUGR compilation functionality.

// Re-export main compiler functions
pub use crate::{
    check_hugr, compile_hugr_bytes_to_bitcode, compile_hugr_bytes_to_bitcode_with_options,
    compile_hugr_bytes_to_string, compile_hugr_bytes_to_string_with_options,
    compile_hugr_to_bitcode, compile_hugr_to_llvm,
};

// Re-export types
pub use crate::{CompileArgs, HugrCompiler, HugrCompilerConfig, OptimizationLevel};

// Re-export helper functions
pub use crate::{
    get_native_target_machine, get_opt_level, get_target_machine_from_triple, read_hugr_envelope,
};

// Re-export common error type
pub use pecos_core::errors::PecosError;
