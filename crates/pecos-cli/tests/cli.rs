// Copyright 2025 The PECOS Developers
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

//! CLI integration tests for the pecos binary.
//!
//! These tests verify the behavior of the `pecos` CLI tool by running it
//! as a subprocess and checking its output.

// Include all test modules from the cli/ directory.
// Note: llvm_test_lock is included directly by llvm_tests.rs via #[path] attribute.
mod cli {
    pub mod basic_determinism_tests;
    pub mod bell_state_tests;
    pub mod llvm;
    pub mod llvm_tests;
    pub mod seed;
    pub mod simple_determinism_test;
    pub mod worker_count_tests;
}
