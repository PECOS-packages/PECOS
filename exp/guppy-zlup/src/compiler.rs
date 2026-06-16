//! Guppy IR to Zlup compiler module.
//!
//! Transforms validated Guppy IR into Zlup source code.

pub mod parser;
pub mod transform;

pub use parser::{parse_ir, ParseError};
pub use transform::{transform, TransformError};
