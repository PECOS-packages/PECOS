//! Trait for building classical control engines and converting to simulation builders
//!
//! This module provides the core trait that all engine builders must implement
//! to participate in the unified simulation API.

use crate::ClassicalControlEngine;
use pecos_core::errors::PecosError;

/// Trait for building classical control engines
///
/// This trait must be implemented by all engine builders (QASM, LLVM, Selene, etc.)
/// to enable the unified simulation API. The preferred pattern is to use `sim_builder()`
/// instead of the deprecated `.to_sim()` method.
pub trait ClassicalControlEngineBuilder {
    /// The type of engine this builder creates
    type Engine: ClassicalControlEngine + Clone + 'static;

    /// Build the classical control engine
    ///
    /// This method is called internally by `SimBuilder` when `.build()` or `.run()` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine cannot be built due to missing configuration,
    /// invalid program, or resource allocation failure
    fn build(self) -> Result<Self::Engine, PecosError>;

    /// Convert this engine builder to a simulation builder
    ///
    /// **Deprecated**: Use `sim_builder()` instead for the preferred API pattern:
    ///
    /// ```text
    /// // Preferred pattern:
    /// let results = sim_builder()
    ///     .classical(qasm_engine()
    ///         .qasm("H q[0];"))
    ///     .seed(42)
    ///     .run(1000)?;
    /// ```
    fn to_sim(self) -> crate::sim_builder::SimBuilder
    where
        Self: Sized + Send + 'static,
        Self::Engine: 'static,
    {
        crate::sim_builder::SimBuilder::new().classical(self)
    }
}

/// Trait for types that can be converted into a simulation builder
///
/// This trait enables the `sim()` function to accept various input types
/// like engine builders, programs, or other simulation configurations.
///
/// # Example
/// ```text
/// // Any type implementing SimInput can be used with sim()
/// let results = sim(MyInput).run(100)?;
/// ```
pub trait SimInput {
    /// Convert this input into a `SimBuilder`
    fn into_sim_builder(self) -> crate::sim_builder::SimBuilder;
}

/// Implement `SimInput` for any `ClassicalControlEngineBuilder`
impl<B> SimInput for B
where
    B: ClassicalControlEngineBuilder + Send + 'static,
    B::Engine: 'static,
{
    fn into_sim_builder(self) -> crate::sim_builder::SimBuilder {
        self.to_sim()
    }
}
