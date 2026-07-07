//! Guppy IR to Zlup compiler module.
//!
//! Transforms validated Guppy IR into Zlup source code.

pub mod parser;
pub mod transform;

pub use parser::{ParseError, parse_ir};
pub use transform::{TransformError, transform};
