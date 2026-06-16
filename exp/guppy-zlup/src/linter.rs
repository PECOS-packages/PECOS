//! Guppy linter module.
//!
//! Validates Guppy quantum programs against NASA Power of 10 rules.

pub mod ast;
pub mod config;
pub mod diagnostic;
pub mod engine;
pub mod lower;
pub mod noqa;
pub mod output;
pub mod rules;

pub use config::Config;
pub use diagnostic::{Diagnostic, Severity};
pub use engine::{LintResult, Linter};
pub use lower::{lower_source, LowerError};
pub use output::OutputFormat;
