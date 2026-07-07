//! Logging infrastructure for the Zlup compiler.
//!
//! Uses the standard `RUST_LOG` environment variable to control compiler log output.
//! This follows Rust conventions and integrates well with other Rust tooling.
//!
//! Note: For logging within Zluppy *programs* (the language feature), see the
//! `log` builtin which is controlled by `ZLUP_LOG` at runtime.
//!
//! # Usage
//!
//! Set the `RUST_LOG` environment variable to control compiler logging:
//!
//! ```bash
//! # Show all debug messages
//! RUST_LOG=debug zlup build
//!
//! # Show only warnings and errors
//! RUST_LOG=warn zlup build
//!
//! # Target specific modules
//! RUST_LOG=zlup::parser=trace zlup build
//! RUST_LOG=zlup::semantic=debug,zlup::codegen=info zlup build
//!
//! # Trace everything
//! RUST_LOG=trace zlup build
//! ```
//!
//! # Log Levels
//!
//! - `error` - Unrecoverable errors
//! - `warn` - Recoverable issues or deprecation warnings
//! - `info` - High-level progress information
//! - `debug` - Detailed debugging information
//! - `trace` - Very verbose tracing (e.g., AST dumps)

use std::io::Write;

/// Initialize the Zlup compiler logger.
///
/// Reads the standard `RUST_LOG` environment variable to configure log levels.
/// If `RUST_LOG` is not set, logging is disabled (no output).
///
/// This should be called once at the start of the program, typically
/// in `main()`.
///
/// # Example
///
/// ```ignore
/// fn main() {
///     zlup::logging::init();
///     // ... rest of program
/// }
/// ```
pub fn init() {
    env_logger::Builder::from_env(env_logger::Env::default())
        .format(|buf, record| {
            let level_style = buf.default_level_style(record.level());
            writeln!(
                buf,
                "{level_style}[{level}]{level_style:#} {target}: {args}",
                level = record.level(),
                target = record.target(),
                args = record.args(),
            )
        })
        .init();
}

/// Initialize the Zlup compiler logger with a default level.
///
/// If `RUST_LOG` is not set, uses the provided default level.
/// This is useful for development builds where you want some
/// output by default.
///
/// # Example
///
/// ```ignore
/// fn main() {
///     // Default to info level if RUST_LOG not set
///     zlup::logging::init_with_default("info");
/// }
/// ```
pub fn init_with_default(default_level: &str) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .format(|buf, record| {
            let level_style = buf.default_level_style(record.level());
            writeln!(
                buf,
                "{level_style}[{level}]{level_style:#} {target}: {args}",
                level = record.level(),
                target = record.target(),
                args = record.args(),
            )
        })
        .init();
}

/// Check if logging is enabled at the given level.
///
/// Useful for avoiding expensive formatting when logging is disabled.
///
/// # Example
///
/// ```ignore
/// if zlup::logging::enabled(log::Level::Debug) {
///     log::debug!("Expensive debug info: {:?}", compute_debug_info());
/// }
/// ```
pub fn enabled(level: log::Level) -> bool {
    log::log_enabled!(level)
}
