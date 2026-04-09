//! ABI version constants for foreign plugin vtables.
//!
//! Every vtable struct has a `version: u32` as its first field. The Rust side
//! checks this on construction and rejects incompatible plugins with a clear error.
//!
//! Bump the version when the vtable layout changes (fields added, removed, or reordered).

/// Current version of the decoder vtable ABI.
pub const DECODER_VTABLE_VERSION: u32 = 1;

/// Current version of the simulator vtable ABI.
pub const SIMULATOR_VTABLE_VERSION: u32 = 1;
