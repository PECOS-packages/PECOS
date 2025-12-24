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

//! A prelude for users of the `pecos-qis-core` crate.
//!
//! This prelude re-exports the most commonly used types, traits, and functions
//! needed for working with QIS control engines in PECOS.

// Re-export main engine types
pub use crate::builder::{QisEngineBuilder, qis_engine};
pub use crate::ccengine::QisEngine;

// Re-export QisInterface trait and related types
pub use crate::interface_impl::SimpleQisInterface;
pub use crate::qis_interface::{BoxedInterface, InterfaceError, ProgramFormat, QisInterface};

// Re-export program types
pub use crate::program::{
    InterfaceChoice, IntoQisInterface, ProgramType, QisEngineProgram, QisInterfaceBuilder,
    QisInterfaceProvider,
};

// Re-export runtime trait and types
// Note: Shot and Value are internal implementation details and not re-exported
pub use crate::runtime::{
    CallFrame, ClassicalState, QisRuntime, Result as RuntimeResult, RuntimeError,
};
