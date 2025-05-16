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

    /// Error related to missing or disabled features
    #[error("Feature error: {0}")]
    Feature(String),

    // Parse errors
    /// Language syntax error
    #[error("{language} syntax error: {message}")]
    ParseSyntax { language: String, message: String },

    /// Invalid version for a language
    #[error("Invalid version for {language}: {version}")]
    ParseInvalidVersion { language: String, version: String },

    /// Invalid number format
    #[error("Invalid number: {0}")]
    ParseInvalidNumber(String),

    /// Invalid identifier
    #[error("Invalid identifier: {0}")]
    ParseInvalidIdentifier(String),

    /// Invalid expression
    #[error("Invalid expression: {0}")]
    ParseInvalidExpression(String),

    // Compilation errors
    /// Invalid operation during compilation
    #[error("Invalid {operation}: {reason}")]
    CompileInvalidOperation { operation: String, reason: String },

    /// Circular dependency detected
    #[error("Circular dependency: {0}")]
    CompileCircularDependency(String),

    /// Undefined reference
    #[error("Undefined {kind} '{name}'")]
    CompileUndefinedReference { kind: String, name: String },

    /// Invalid register size
    #[error("Invalid register size: {0}")]
    CompileInvalidRegisterSize(String),

    // Runtime errors
    /// Division by zero
    #[error("Division by zero")]
    RuntimeDivisionByZero,

    /// Stack overflow
    #[error("Stack overflow")]
    RuntimeStackOverflow,

    /// Index out of bounds
    #[error("Index out of bounds: {index} not in 0..{length}")]
    RuntimeIndexOutOfBounds { index: usize, length: usize },

    // Validation errors
    /// Invalid circuit structure
    #[error("Invalid circuit structure: {0}")]
    ValidationInvalidCircuitStructure(String),

    /// Invalid gate parameters
    #[error("Invalid gate parameters: {0}")]
    ValidationInvalidGateParameters(String),

    /// Invalid qubit reference
    #[error("Invalid qubit reference: {0}")]
    ValidationInvalidQubitReference(String),
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
