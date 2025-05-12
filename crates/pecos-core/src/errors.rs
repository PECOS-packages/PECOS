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

use std::error::Error;
use std::io;
use thiserror::Error;

/// The main error type for PECOS
#[derive(Error, Debug)]
pub enum PecosError {
    /// Input/output related error
    #[error("IO error: {0}")]
    IO(#[from] io::Error),

    /// Generic error when a more specific category doesn't apply
    #[error("{0}")]
    Generic(String),

    /// Error with context information
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },

    /// Error from an external source
    #[error(transparent)]
    External(#[from] Box<dyn Error + Send + Sync>),

    /// Error related to invalid input parameters, arguments, or configuration
    #[error("Input error: {0}")]
    Input(String),

    /// Error related to failures during command or operation processing
    #[error("Processing error: {0}")]
    Processing(String),

    /// Error related to resource handling (files, libraries, etc.)
    #[error("Resource error: {0}")]
    Resource(String),

    /// Error related to the compilation process
    #[error("Compilation error: {0}")]
    Compilation(String),

    /// Error related to an unsupported or invalid quantum gate
    #[error("Gate error: {0}")]
    Gate(String),

    /// Error related to expression evaluation or computation
    /// This covers arithmetic errors, variable access, and general expression evaluation
    #[error("Computation error: {0}")]
    Computation(String),
}

impl PecosError {
    /// Adds context to any error
    pub fn with_context<E, S>(error: E, context: S) -> Self
    where
        E: Error + Send + Sync + 'static,
        S: Into<String>,
    {
        Self::WithContext {
            context: context.into(),
            source: Box::new(error),
        }
    }
}
