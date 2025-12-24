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

//! A prelude for users of the `pecos-qis-selene` crate.
//!
//! This prelude re-exports the most commonly used types, traits, and functions
//! needed for working with Selene-based QIS interfaces and runtimes in PECOS.

// Re-export builder types
pub use crate::builder::{HeliosInterfaceBuilder, helios_interface_builder};

// Re-export main interface type
pub use crate::executor::QisHeliosInterface;

// Re-export runtime types
pub use crate::selene_runtime::SeleneRuntime;
pub use crate::selene_runtimes::{
    RuntimeFetchError, find_selene_runtime, selene_runtime_auto, selene_simple_runtime,
    selene_soft_rz_runtime,
};
