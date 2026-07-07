//! FFI-safe types for Zlup/Rust interop.
//!
//! These types are designed to cross the FFI boundary safely and map
//! directly to Zlup types.

use std::marker::PhantomData;

/// A qubit identifier.
///
/// This is an opaque handle that identifies a qubit in the quantum state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QubitId(pub u32);

impl QubitId {
    /// Create a new qubit ID.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub fn raw(&self) -> u32 {
        self.0
    }
}

/// Gate types supported by Zlup.
///
/// This enum maps directly to Zlup's gate operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    // Single-qubit Pauli gates
    /// Pauli X gate
    X,
    /// Pauli Y gate
    Y,
    /// Pauli Z gate
    Z,

    // Single-qubit Clifford gates
    /// Hadamard gate
    H,
    /// S gate (sqrt Z)
    S,
    /// S-dagger gate
    Sdg,
    /// T gate (fourth root Z)
    T,
    /// T-dagger gate
    Tdg,

    // Square root gates
    /// sqrt(X) gate
    Sx,
    /// sqrt(Y) gate
    Sy,
    /// sqrt(Z) gate (same as S)
    Sz,

    // Rotation gates (angle in radians)
    /// Rotation around X axis
    Rx(f64),
    /// Rotation around Y axis
    Ry(f64),
    /// Rotation around Z axis
    Rz(f64),

    // Two-qubit gates
    /// Controlled-X (CNOT)
    Cx,
    /// Controlled-Y
    Cy,
    /// Controlled-Z
    Cz,
    /// Controlled-H
    Ch,

    // Swap gates
    /// SWAP gate
    Swap,
    /// iSWAP gate
    Iswap,

    // Ising gates
    /// sqrt(XX)
    Sxx,
    /// sqrt(YY)
    Syy,
    /// sqrt(ZZ)
    Szz,

    // Parameterized two-qubit gates
    /// ZZ rotation
    Rzz(f64),
    /// XX rotation
    Rxx(f64),
    /// YY rotation
    Ryy(f64),

    // Three-qubit gates
    /// Toffoli (CCX)
    Ccx,
}

// =============================================================================
// Angle64 - Fixed-point angle representation
// =============================================================================

/// A 64-bit fixed-point angle representation.
///
/// Angles are stored as fractions of a full turn using fixed-point arithmetic.
/// The internal representation uses a u64 where [0, 2^64) maps to [0, 1) turns.
/// This provides exact representation for all dyadic fractions (denominators
/// that are powers of 2), which covers all common quantum gate angles:
///
/// - 1/2 turn (π rad) = 2^63
/// - 1/4 turn (π/2 rad) = 2^62
/// - 1/8 turn (π/4 rad, T-gate) = 2^61
/// - etc.
///
/// # Design Rationale
///
/// This representation is compatible with PECOS's angle handling and avoids
/// floating-point precision issues that can cause bugs in quantum circuits
/// (similar to the Mars Climate Orbiter unit conversion bug).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Angle64 {
    /// Internal fixed-point representation.
    /// The full range [0, 2^64) represents [0, 1) turns.
    raw: u64,
}

impl Angle64 {
    /// The constant 2^64 as f64 for conversions.
    const SCALE: f64 = (1u64 << 63) as f64 * 2.0;

    /// Zero angle.
    pub const ZERO: Angle64 = Angle64 { raw: 0 };

    /// One full turn (wraps to zero in modular arithmetic).
    pub const FULL_TURN: Angle64 = Angle64 { raw: 0 };

    /// Half turn (π radians, 180°).
    pub const HALF_TURN: Angle64 = Angle64 { raw: 1 << 63 };

    /// Quarter turn (π/2 radians, 90°).
    pub const QUARTER_TURN: Angle64 = Angle64 { raw: 1 << 62 };

    /// Eighth turn (π/4 radians, 45°, T-gate angle).
    pub const EIGHTH_TURN: Angle64 = Angle64 { raw: 1 << 61 };

    /// Create an angle from a fraction of a turn.
    ///
    /// The value should be in [0, 1) for angles less than a full turn,
    /// but values outside this range will wrap correctly.
    ///
    /// # Example
    /// ```
    /// use zlup_ffi::types::Angle64;
    /// let quarter = Angle64::from_turns(0.25);
    /// assert_eq!(quarter, Angle64::QUARTER_TURN);
    /// ```
    pub fn from_turns(turns: f64) -> Self {
        // Normalize to [0, 1) range
        let normalized = turns.rem_euclid(1.0);
        let raw = (normalized * Self::SCALE) as u64;
        Self { raw }
    }

    /// Create an angle from radians.
    ///
    /// # Example
    /// ```
    /// use zlup_ffi::types::Angle64;
    /// use std::f64::consts::PI;
    /// let quarter = Angle64::from_radians(PI / 2.0);
    /// assert_eq!(quarter, Angle64::QUARTER_TURN);
    /// ```
    pub fn from_radians(radians: f64) -> Self {
        Self::from_turns(radians / (2.0 * std::f64::consts::PI))
    }

    /// Create an angle from an exact fraction of a turn.
    ///
    /// This method provides exact representation for dyadic fractions
    /// (where the denominator is a power of 2).
    ///
    /// # Example
    /// ```
    /// use zlup_ffi::types::Angle64;
    /// let quarter = Angle64::from_turn_fraction(1, 4);
    /// assert_eq!(quarter, Angle64::QUARTER_TURN);
    /// ```
    pub fn from_turn_fraction(numerator: u64, denominator: u64) -> Self {
        if denominator == 0 {
            return Self::ZERO;
        }
        // For exact representation, we compute: (numerator * 2^64) / denominator
        // Using 128-bit arithmetic for precision
        let num_scaled = (numerator as u128) << 64;
        let raw = (num_scaled / denominator as u128) as u64;
        Self { raw }
    }

    /// Convert to turns (fraction of a full rotation).
    pub fn to_turns(&self) -> f64 {
        self.raw as f64 / Self::SCALE
    }

    /// Convert to radians.
    pub fn to_radians(&self) -> f64 {
        self.to_turns() * 2.0 * std::f64::consts::PI
    }

    /// Get the raw fixed-point value.
    pub fn raw(&self) -> u64 {
        self.raw
    }

    /// Create from raw fixed-point value.
    pub fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// Add two angles (wraps at full turn).
    pub fn add(self, other: Self) -> Self {
        Self {
            raw: self.raw.wrapping_add(other.raw),
        }
    }

    /// Subtract two angles (wraps at full turn).
    pub fn sub(self, other: Self) -> Self {
        Self {
            raw: self.raw.wrapping_sub(other.raw),
        }
    }

    /// Negate the angle.
    pub fn neg(self) -> Self {
        Self {
            raw: self.raw.wrapping_neg(),
        }
    }

    /// Multiply the angle by an integer.
    pub fn mul(self, n: u64) -> Self {
        Self {
            raw: self.raw.wrapping_mul(n),
        }
    }

    /// Divide the angle by an integer.
    pub fn div(self, n: u64) -> Self {
        if n == 0 {
            return Self::ZERO;
        }
        Self { raw: self.raw / n }
    }

    /// Check if this is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.raw == 0
    }

    /// Check if this is exactly a half turn.
    pub fn is_half_turn(&self) -> bool {
        self.raw == Self::HALF_TURN.raw
    }

    /// Check if this is a Clifford angle (multiple of 1/4 turn).
    pub fn is_clifford(&self) -> bool {
        // Clifford angles are multiples of 1/4 turn
        // In our representation, that means the lower 62 bits are zero
        (self.raw & ((1 << 62) - 1)) == 0
    }
}

impl std::ops::Add for Angle64 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}

impl std::ops::Sub for Angle64 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::sub(self, other)
    }
}

impl std::ops::Neg for Angle64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::neg(self)
    }
}

impl std::fmt::Display for Angle64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Try to display as a nice fraction if possible
        let turns = self.to_turns();
        if turns == 0.0 {
            write!(f, "0 turns")
        } else if turns == 0.5 {
            write!(f, "1/2 turns")
        } else if turns == 0.25 {
            write!(f, "1/4 turns")
        } else if turns == 0.125 {
            write!(f, "1/8 turns")
        } else if turns == 0.75 {
            write!(f, "3/4 turns")
        } else {
            write!(f, "{:.6} turns", turns)
        }
    }
}

/// Packed bit representation for syndromes and corrections (64 bits).
///
/// For larger syndromes, use `Syndrome128` or `Syndrome256`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Syndrome64 {
    data: u64,
}

impl Syndrome64 {
    /// Create a new Syndrome64 with all zeros.
    pub fn zeros() -> Self {
        Self { data: 0 }
    }

    /// Create from a u64.
    pub fn from_u64(value: u64) -> Self {
        Self { data: value }
    }

    /// Get a single bit.
    pub fn get(&self, index: usize) -> bool {
        if index >= 64 {
            return false;
        }
        (self.data >> index) & 1 == 1
    }

    /// Set a single bit.
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= 64 {
            return;
        }
        if value {
            self.data |= 1 << index;
        } else {
            self.data &= !(1 << index);
        }
    }

    /// Get the number of bits (always 64).
    pub fn len(&self) -> usize {
        64
    }

    /// Check if empty (never true for fixed-size).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Convert to u64.
    pub fn to_u64(&self) -> u64 {
        self.data
    }

    /// Count the number of set bits (popcount).
    pub fn popcount(&self) -> usize {
        self.data.count_ones() as usize
    }

    /// Compute parity (XOR of all bits).
    pub fn parity(&self) -> bool {
        self.data.count_ones() % 2 == 1
    }
}

/// 128-bit syndrome storage.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Syndrome128 {
    low: u64,
    high: u64,
}

impl Syndrome128 {
    /// Create a new Syndrome128 with all zeros.
    pub fn zeros() -> Self {
        Self { low: 0, high: 0 }
    }

    /// Get a single bit.
    pub fn get(&self, index: usize) -> bool {
        if index >= 128 {
            return false;
        }
        if index < 64 {
            (self.low >> index) & 1 == 1
        } else {
            (self.high >> (index - 64)) & 1 == 1
        }
    }

    /// Set a single bit.
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= 128 {
            return;
        }
        if index < 64 {
            if value {
                self.low |= 1 << index;
            } else {
                self.low &= !(1 << index);
            }
        } else {
            let bit = index - 64;
            if value {
                self.high |= 1 << bit;
            } else {
                self.high &= !(1 << bit);
            }
        }
    }

    /// Count the number of set bits (popcount).
    pub fn popcount(&self) -> usize {
        (self.low.count_ones() + self.high.count_ones()) as usize
    }
}

/// 256-bit syndrome storage.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Syndrome256 {
    words: [u64; 4],
}

impl Syndrome256 {
    /// Create a new Syndrome256 with all zeros.
    pub fn zeros() -> Self {
        Self { words: [0; 4] }
    }

    /// Get a single bit.
    pub fn get(&self, index: usize) -> bool {
        if index >= 256 {
            return false;
        }
        let word = index / 64;
        let bit = index % 64;
        (self.words[word] >> bit) & 1 == 1
    }

    /// Set a single bit.
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= 256 {
            return;
        }
        let word = index / 64;
        let bit = index % 64;
        if value {
            self.words[word] |= 1 << bit;
        } else {
            self.words[word] &= !(1 << bit);
        }
    }

    /// Count the number of set bits (popcount).
    pub fn popcount(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// FFI-safe result type.
///
/// Use this for functions that can fail across FFI boundaries.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiResult<T> {
    /// Whether the operation succeeded.
    pub ok: bool,
    /// The value (only valid if ok is true).
    pub value: T,
    /// Error code (only valid if ok is false).
    pub error_code: u32,
}

impl<T: Default> FfiResult<T> {
    /// Create a success result.
    pub fn ok(value: T) -> Self {
        Self {
            ok: true,
            value,
            error_code: 0,
        }
    }

    /// Create an error result.
    pub fn err(code: u32) -> Self {
        Self {
            ok: false,
            value: T::default(),
            error_code: code,
        }
    }
}

impl<T> FfiResult<T> {
    /// Convert to a Rust Result.
    pub fn into_result(self) -> Result<T, u32> {
        if self.ok {
            Ok(self.value)
        } else {
            Err(self.error_code)
        }
    }
}

/// Common FFI error codes.
pub mod error_codes {
    /// Success (no error).
    pub const SUCCESS: u32 = 0;
    /// Null pointer passed.
    pub const NULL_POINTER: u32 = 1;
    /// Invalid argument.
    pub const INVALID_ARGUMENT: u32 = 2;
    /// Out of memory.
    pub const OUT_OF_MEMORY: u32 = 3;
    /// Decoder failed.
    pub const DECODE_FAILED: u32 = 4;
    /// Internal error.
    pub const INTERNAL_ERROR: u32 = 5;
}

/// Marker trait for types that are safe to use as syndrome data.
pub trait SyndromeData: Copy + Send + Sync {}

impl SyndromeData for u8 {}
impl SyndromeData for u16 {}
impl SyndromeData for u32 {}
impl SyndromeData for u64 {}
impl SyndromeData for Syndrome64 {}
impl SyndromeData for Syndrome128 {}
impl SyndromeData for Syndrome256 {}

/// Marker trait for types that are safe to use as correction data.
pub trait CorrectionData: Copy + Send + Sync {}

impl CorrectionData for u8 {}
impl CorrectionData for u16 {}
impl CorrectionData for u32 {}
impl CorrectionData for u64 {}
impl CorrectionData for Syndrome64 {}
impl CorrectionData for Syndrome128 {}
impl CorrectionData for Syndrome256 {}

/// A slice of qubits for FFI.
#[repr(C)]
pub struct QubitSlice<'a> {
    ptr: *const QubitId,
    len: usize,
    _marker: PhantomData<&'a [QubitId]>,
}

impl<'a> QubitSlice<'a> {
    /// Create from a Rust slice.
    pub fn from_slice(slice: &'a [QubitId]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    /// Convert back to a Rust slice.
    ///
    /// # Safety
    /// The pointer must still be valid.
    pub unsafe fn as_slice(&self) -> &'a [QubitId] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
