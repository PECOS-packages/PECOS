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
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # // This is a conceptual example showing the API pattern
    /// # // In practice, you would use the actual engine builders from specific crates
    /// # use pecos_engines::{ClassicalControlEngineBuilder, sim_builder};
    /// #
    /// # // Example using a hypothetical qasm_engine function
    /// # // In real code, use: use pecos_qasm::qasm_engine;
    /// # mod example {
    /// #     use pecos_engines::{ClassicalControlEngineBuilder, SimBuilder};
    /// #     use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    /// #
    /// #     pub struct QasmEngineBuilder;
    /// #
    /// #     impl ClassicalControlEngineBuilder for QasmEngineBuilder {
    /// #         type Engine = ExternalClassicalEngine;
    /// #
    /// #         fn build(self) -> Result<Self::Engine, pecos_core::errors::PecosError> {
    /// #             Ok(ExternalClassicalEngine::new())
    /// #         }
    /// #     }
    /// #
    /// #     impl QasmEngineBuilder {
    /// #         pub fn qasm(self, _qasm: &str) -> Self { self }
    /// #     }
    /// #
    /// #     pub fn qasm_engine() -> QasmEngineBuilder { QasmEngineBuilder }
    /// # }
    /// # use example::qasm_engine;
    /// #
    /// // Preferred pattern:
    /// let results = sim_builder()
    ///     .classical(qasm_engine()
    ///         .qasm("H q[0];"))
    ///     .seed(42)
    ///     .run(1000)?;
    /// # Ok(())
    /// # }
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
/// ```no_run
/// # use pecos_engines::{sim, SimInput};
/// # struct MyInput;
/// # impl SimInput for MyInput {
/// #     fn into_sim_builder(self) -> pecos_engines::SimBuilder {
/// #         pecos_engines::sim_builder()
/// #     }
/// # }
/// // Any type implementing SimInput can be used with sim()
/// let results = sim(MyInput).run(100)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
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
