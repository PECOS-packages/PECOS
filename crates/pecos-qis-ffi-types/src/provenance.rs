//! Compiler/runtime byte-shadow protocol. Numeric values are part of QIS FFI ABI 8.
//! Unlisted operation numbers are invalid; removed operations are not reusable aliases.

/// Uniform constant-byte handle.
pub const CONSTANT: u64 = 0;
/// Uniform unknown-byte handle.
pub const UNKNOWN: u64 = 1;
/// Uniform derived-byte handle.
pub const DERIVED: u64 = 2;
/// Immutable vector handles are this offset plus their zero-based index.
pub const VECTOR_BASE: u64 = 3;

/// SSA and call-frame operations.
pub mod op {
    /// Join operand classes, preferring derived over unknown over constant.
    pub const MERGE: u64 = 1;
    /// Copy c bytes from handle a at byte offset b.
    pub const EXTRACT: u64 = 2;
    /// Move bytes of handle a by signed byte count b within extent c.
    pub const SHIFT: u64 = 4;
    /// Condition a selects b; nonconstant conditions merge arm classes c and d and raise control dependence.
    pub const SELECT: u64 = 5;
    /// Raise the sticky control flag if handle a depends on measurements.
    pub const CONTROL: u64 = 6;
    /// Return derived and raise the flag for measurement-dependent address or range handle a; otherwise constant.
    pub const ADDRESS: u64 = 7;
    /// Push an internal-call shadow frame.
    pub const PUSH_FRAME: u64 = 8;
    /// Set shadow argument a to handle b in the current frame.
    pub const SET_ARGUMENT: u64 = 9;
    /// Get shadow argument a, or constant for the program entry.
    pub const GET_ARGUMENT: u64 = 10;
    /// Set the current frame return to handle a.
    pub const SET_RETURN: u64 = 11;
    /// Pop the completed frame and return its result handle.
    pub const POP_FRAME: u64 = 12;
    /// Insert sized member b into aggregate a at offset c, with total extent d.
    pub const INSERT: u64 = 13;
    /// Materialize exactly b bytes of handle a, retaining extent even for a constant member.
    pub const SIZED: u64 = 14;
}
/// Byte-memory operations. All raise control dependence for dependent address/range shadows.
pub mod memory {
    /// Read size bytes from pointer, or derived bytes for a dependent address/range.
    pub const LOAD: u64 = 0;
    /// Write size bytes from handle, or derived bytes for a dependent address/range.
    pub const STORE: u64 = 1;
    /// Make a fresh allocation unknown, or derived for a dependent allocation range.
    pub const CLEAR: u64 = 2;
    /// Copy size source-shadow bytes, snapshotting first for overlapping memmove.
    pub const COPY: u64 = 3;
    /// Repeat the fill handle byte across size destination bytes.
    pub const FILL: u64 = 4;
}
