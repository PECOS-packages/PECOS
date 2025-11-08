// Re-export from pecos-wasm crate
#[cfg(feature = "wasm")]
pub use pecos_wasm::{DummyForeignObject, ForeignObject};

// For when wasm feature is disabled, provide minimal trait
#[cfg(not(feature = "wasm"))]
pub use pecos_wasm::{DummyForeignObject, ForeignObject};
