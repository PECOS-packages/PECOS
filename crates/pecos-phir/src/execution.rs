/*!
PHIR Execution Engine

This module provides the `PhirEngine` - a `ClassicalEngine` implementation that can execute
PHIR programs directly, matching the capabilities of `PhirJsonEngine` but operating on
PHIR modules instead of JSON.

The `PhirEngine` handles:
- Classical computation and variable management
- Quantum operation generation via `ByteMessage` protocol
- Measurement result processing
- Integration with PECOS quantum simulation infrastructure
*/

pub mod engine;
pub mod environment;
pub mod expression;
pub mod processor;

#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use engine::PhirEngine;
pub use environment::{DataType, Environment, TypedValue};
pub use expression::ExpressionEvaluator;
pub use processor::PhirProcessor;
