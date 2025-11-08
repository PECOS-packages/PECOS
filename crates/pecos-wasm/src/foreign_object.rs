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

//! Unified `ForeignObject` trait for PECOS
//!
//! This trait defines the interface for foreign object implementations (like WebAssembly modules)
//! that can be called from PECOS quantum simulations.

use pecos_core::errors::PecosError;
use std::any::Any;
use std::fmt::Debug;

/// Trait for foreign object implementations
///
/// This trait provides a unified interface for foreign objects (like WebAssembly modules)
/// that can be executed from quantum simulation programs. Implementations must be thread-safe
/// (`Send + Sync`) to support parallel execution.
///
/// # Required Methods
///
/// - `clone_box`: Create a boxed clone of the object
/// - `init`: Initialize the object before a series of simulations
/// - `new_instance`: Create a new instance/reset internal state
/// - `get_funcs`: Get list of available function names
/// - `exec`: Execute a named function with arguments
///
/// # Optional Methods
///
/// - `teardown`: Cleanup resources (default: no-op)
/// - `as_any`: Downcast to concrete type (for type inspection)
/// - `as_any_mut`: Mutable downcast to concrete type
pub trait ForeignObject: Debug + Send + Sync {
    /// Clone the foreign object
    ///
    /// Returns a boxed clone that can be used independently.
    fn clone_box(&self) -> Box<dyn ForeignObject>;

    /// Initialize object before running a series of simulations
    ///
    /// This is typically called once before running multiple shots. It should:
    /// - Create a new instance
    /// - Call the `init` function in the foreign object if it exists
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    fn init(&mut self) -> Result<(), PecosError>;

    /// Create new instance/internal state
    ///
    /// This resets the internal state of the foreign object, typically called
    /// at the start of each simulation shot.
    ///
    /// # Errors
    ///
    /// Returns an error if instance creation fails.
    fn new_instance(&mut self) -> Result<(), PecosError>;

    /// Get a list of function names available from the object
    ///
    /// Returns all exported function names that can be called via `exec()`.
    fn get_funcs(&self) -> Vec<String>;

    /// Execute a function given a list of arguments
    ///
    /// # Parameters
    ///
    /// - `func_name`: Name of the function to execute
    /// - `args`: Slice of i64 arguments to pass to the function
    ///
    /// # Returns
    ///
    /// Vector of i64 return values from the function
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The function does not exist
    /// - Execution fails
    /// - Timeout occurs (if supported)
    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError>;

    /// Cleanup resources
    ///
    /// Called when the foreign object is no longer needed. Default implementation
    /// does nothing, but implementations with background threads or other resources
    /// should override this.
    fn teardown(&mut self) {}

    /// Get as Any for downcasting
    ///
    /// Allows downcasting to the concrete type for type-specific operations.
    fn as_any(&self) -> &dyn Any;

    /// Get as Any for downcasting (mutable)
    ///
    /// Allows mutable downcasting to the concrete type.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Dummy foreign object for when no foreign object is needed
///
/// This is a no-op implementation that returns errors for all `exec()` calls.
/// Useful as a placeholder or default value.
#[derive(Debug, Clone)]
pub struct DummyForeignObject {}

impl DummyForeignObject {
    /// Create a new dummy foreign object
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DummyForeignObject {
    fn default() -> Self {
        Self::new()
    }
}

impl ForeignObject for DummyForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        Box::new(Self::default())
    }

    fn init(&mut self) -> Result<(), PecosError> {
        Ok(())
    }

    fn new_instance(&mut self) -> Result<(), PecosError> {
        Ok(())
    }

    fn get_funcs(&self) -> Vec<String> {
        vec![]
    }

    fn exec(&mut self, func_name: &str, _args: &[i64]) -> Result<Vec<i64>, PecosError> {
        Err(PecosError::Input(format!(
            "Dummy foreign object cannot execute function: {func_name}"
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
