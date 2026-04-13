//! Universal C-ABI shared library for PECOS.
//!
//! This crate produces `libpecos_ffi.so` (Linux), `libpecos_ffi.dylib` (macOS),
//! or `pecos_ffi.dll` (Windows). Any language that can call C functions can link
//! against this library and use the full PECOS foreign plugin API.
//!
//! # What's included
//!
//! All `#[no_mangle] extern "C"` functions from `pecos-foreign` are automatically
//! exported. This includes:
//!
//! - Decoder plugin functions (`pecos_foreign_decoder_*`)
//! - Simulator plugin functions (`pecos_foreign_simulator_*`)
//! - Engine functions (`pecos_engine_*`, `pecos_circuit_*`)
//! - Result parsing (`pecos_parse_outcomes`)
//! - Version queries (`pecos_decoder_vtable_version`, `pecos_simulator_vtable_version`)
//!
//! # For plugin authors
//!
//! 1. Build this crate: `cargo build -p pecos-ffi --release`
//! 2. Find the library: `target/release/libpecos_ffi.so`
//! 3. Include the C header: `crates/pecos-foreign/include/pecos_foreign.h`
//! 4. Link and call functions from your language
//!
//! No Rust knowledge required.

// Re-export pecos_foreign so the linker includes all its #[no_mangle] symbols.
pub use pecos_foreign;
