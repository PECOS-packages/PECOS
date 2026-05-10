// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Compact Pauli operator using bitmasks with Clifford conjugation.
//!
//! Provides allocation-free Pauli arithmetic and Clifford conjugation
//! for use in error propagation, fault analysis, and EEG algorithms.
//!
//! The default `PauliBitmask` uses `u128` (up to 128 qubits). For
//! larger qubit counts, use `PauliBitmaskVec` which heap-allocates.

use smallvec::SmallVec;
use std::fmt;

/// Trait for bitmask storage backends.
///
/// Enables `PauliBitmaskGeneric<B>` to work with different widths:
/// `u64` (64 qubits), `u128` (128 qubits), or `Vec<u64>` (unlimited).
pub trait BitmaskStorage: Clone + PartialEq + Eq + std::hash::Hash + Default + fmt::Debug {
    fn zero() -> Self;
    fn set_bit(&mut self, bit: usize);
    fn clear_bit(&mut self, bit: usize);
    fn get_bit(&self, bit: usize) -> bool;
    fn xor_assign(&mut self, other: &Self);
    fn xor_bit(&mut self, bit: usize);
    fn and_count_ones_xor(&self, other_z: &Self, self_z: &Self, other_x: &Self) -> u32;
    fn is_zero(&self) -> bool;
    fn or_count_ones(&self, other: &Self) -> u32;
    fn highest_set_bit(&self) -> Option<usize>;
}

impl BitmaskStorage for u128 {
    fn zero() -> Self {
        0
    }
    fn set_bit(&mut self, bit: usize) {
        *self |= 1u128 << bit;
    }
    fn clear_bit(&mut self, bit: usize) {
        *self &= !(1u128 << bit);
    }
    fn get_bit(&self, bit: usize) -> bool {
        *self & (1u128 << bit) != 0
    }
    fn xor_assign(&mut self, other: &Self) {
        *self ^= other;
    }
    fn xor_bit(&mut self, bit: usize) {
        *self ^= 1u128 << bit;
    }
    fn and_count_ones_xor(&self, other_z: &Self, self_z: &Self, other_x: &Self) -> u32 {
        ((*self & other_z) ^ (self_z & other_x)).count_ones()
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn or_count_ones(&self, other: &Self) -> u32 {
        (*self | other).count_ones()
    }
    fn highest_set_bit(&self) -> Option<usize> {
        if *self == 0 {
            None
        } else {
            Some(127 - self.leading_zeros() as usize)
        }
    }
}

impl BitmaskStorage for Vec<u64> {
    fn zero() -> Self {
        Vec::new()
    }
    fn set_bit(&mut self, bit: usize) {
        let word = bit / 64;
        if word >= self.len() {
            self.resize(word + 1, 0);
        }
        self[word] |= 1u64 << (bit % 64);
    }
    fn clear_bit(&mut self, bit: usize) {
        let word = bit / 64;
        if word < self.len() {
            self[word] &= !(1u64 << (bit % 64));
        }
    }
    fn get_bit(&self, bit: usize) -> bool {
        let word = bit / 64;
        word < self.len() && self[word] & (1u64 << (bit % 64)) != 0
    }
    fn xor_assign(&mut self, other: &Self) {
        if self.len() < other.len() {
            self.resize(other.len(), 0);
        }
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a ^= b;
        }
    }
    fn xor_bit(&mut self, bit: usize) {
        let word = bit / 64;
        if word >= self.len() {
            self.resize(word + 1, 0);
        }
        self[word] ^= 1u64 << (bit % 64);
    }
    fn and_count_ones_xor(&self, other_z: &Self, self_z: &Self, other_x: &Self) -> u32 {
        let max = self
            .len()
            .max(other_z.len())
            .max(self_z.len())
            .max(other_x.len());
        let mut count = 0u32;
        for i in 0..max {
            let sx = self.get(i).copied().unwrap_or(0);
            let oz = other_z.get(i).copied().unwrap_or(0);
            let sz = self_z.get(i).copied().unwrap_or(0);
            let ox = other_x.get(i).copied().unwrap_or(0);
            count += ((sx & oz) ^ (sz & ox)).count_ones();
        }
        count
    }
    fn is_zero(&self) -> bool {
        self.iter().all(|&w| w == 0)
    }
    fn or_count_ones(&self, other: &Self) -> u32 {
        let max = self.len().max(other.len());
        let mut count = 0u32;
        for i in 0..max {
            let a = self.get(i).copied().unwrap_or(0);
            let b = other.get(i).copied().unwrap_or(0);
            count += (a | b).count_ones();
        }
        count
    }
    fn highest_set_bit(&self) -> Option<usize> {
        for (i, &w) in self.iter().enumerate().rev() {
            if w != 0 {
                return Some(i * 64 + 63 - w.leading_zeros() as usize);
            }
        }
        None
    }
}

/// `SmallVec<[u64; 8]>` backend: 512 bits inline (covers d≤9 surface codes),
/// spills to heap for larger circuits. Zero allocation for typical QEC.
impl BitmaskStorage for SmallVec<[u64; 8]> {
    fn zero() -> Self {
        SmallVec::new()
    }
    fn set_bit(&mut self, bit: usize) {
        let word = bit / 64;
        if word >= self.len() {
            self.resize(word + 1, 0);
        }
        self[word] |= 1u64 << (bit % 64);
    }
    fn clear_bit(&mut self, bit: usize) {
        let word = bit / 64;
        if word < self.len() {
            self[word] &= !(1u64 << (bit % 64));
        }
    }
    fn get_bit(&self, bit: usize) -> bool {
        let word = bit / 64;
        word < self.len() && self[word] & (1u64 << (bit % 64)) != 0
    }
    fn xor_assign(&mut self, other: &Self) {
        if self.len() < other.len() {
            self.resize(other.len(), 0);
        }
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a ^= b;
        }
    }
    fn xor_bit(&mut self, bit: usize) {
        let word = bit / 64;
        if word >= self.len() {
            self.resize(word + 1, 0);
        }
        self[word] ^= 1u64 << (bit % 64);
    }
    fn and_count_ones_xor(&self, other_z: &Self, self_z: &Self, other_x: &Self) -> u32 {
        let max = self
            .len()
            .max(other_z.len())
            .max(self_z.len())
            .max(other_x.len());
        let mut count = 0u32;
        for i in 0..max {
            let sx = self.get(i).copied().unwrap_or(0);
            let oz = other_z.get(i).copied().unwrap_or(0);
            let sz = self_z.get(i).copied().unwrap_or(0);
            let ox = other_x.get(i).copied().unwrap_or(0);
            count += ((sx & oz) ^ (sz & ox)).count_ones();
        }
        count
    }
    fn is_zero(&self) -> bool {
        self.iter().all(|&w| w == 0)
    }
    fn or_count_ones(&self, other: &Self) -> u32 {
        let max = self.len().max(other.len());
        let mut count = 0u32;
        for i in 0..max {
            let a = self.get(i).copied().unwrap_or(0);
            let b = other.get(i).copied().unwrap_or(0);
            count += (a | b).count_ones();
        }
        count
    }
    fn highest_set_bit(&self) -> Option<usize> {
        for (i, &w) in self.iter().enumerate().rev() {
            if w != 0 {
                return Some(i * 64 + 63 - w.leading_zeros() as usize);
            }
        }
        None
    }
}

/// N-qubit Pauli operator in symplectic binary representation.
///
/// Phase is NOT tracked internally — the caller tracks signs from
/// multiplication and conjugation separately.
///
/// Type parameter `B` selects the bitmask backend:
/// - `u128` (default): up to 128 qubits, stack-allocated, `Copy`
/// - `Vec<u64>`: unlimited qubits, heap-allocated
#[derive(Clone, Default)]
pub struct PauliBitmaskGeneric<B: BitmaskStorage = u128> {
    pub x_bits: B,
    pub z_bits: B,
}

// --- PartialEq, Eq, Hash for u128 backend ---

impl PartialEq for PauliBitmaskGeneric<u128> {
    fn eq(&self, other: &Self) -> bool {
        self.x_bits == other.x_bits && self.z_bits == other.z_bits
    }
}

impl Eq for PauliBitmaskGeneric<u128> {}

impl std::hash::Hash for PauliBitmaskGeneric<u128> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x_bits.hash(state);
        self.z_bits.hash(state);
    }
}

// --- PartialEq, Eq, Hash for Vec<u64> backend ---
// Trailing zero words are ignored so that Vecs of different lengths
// representing the same logical value compare equal and hash identically.

impl PartialEq for PauliBitmaskGeneric<Vec<u64>> {
    fn eq(&self, other: &Self) -> bool {
        vecs_eq_ignoring_trailing_zeros(&self.x_bits, &other.x_bits)
            && vecs_eq_ignoring_trailing_zeros(&self.z_bits, &other.z_bits)
    }
}

impl Eq for PauliBitmaskGeneric<Vec<u64>> {}

impl std::hash::Hash for PauliBitmaskGeneric<Vec<u64>> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash_vec_ignoring_trailing_zeros(&self.x_bits, state);
        hash_vec_ignoring_trailing_zeros(&self.z_bits, state);
    }
}

fn vecs_eq_ignoring_trailing_zeros(a: &[u64], b: &[u64]) -> bool {
    let max = a.len().max(b.len());
    for i in 0..max {
        let aw = a.get(i).copied().unwrap_or(0);
        let bw = b.get(i).copied().unwrap_or(0);
        if aw != bw {
            return false;
        }
    }
    true
}

fn hash_vec_ignoring_trailing_zeros<H: std::hash::Hasher>(v: &[u64], state: &mut H) {
    use std::hash::Hash;
    // Find the last non-zero word and hash only up to that point.
    let effective_len = v.iter().rposition(|&w| w != 0).map_or(0, |i| i + 1);
    effective_len.hash(state);
    for &w in &v[..effective_len] {
        w.hash(state);
    }
}

/// Fixed-size Pauli bitmask for up to 128 qubits (stack-allocated, Copy).
pub type PauliBitmask = PauliBitmaskGeneric<u128>;

/// Dynamically-sized Pauli bitmask for unlimited qubits (heap-allocated).
pub type PauliBitmaskVec = PauliBitmaskGeneric<Vec<u64>>;

/// SmallVec-backed Pauli bitmask: 512 bits inline (d≤9 surface codes),
/// spills to heap only for larger circuits. Best of both worlds.
pub type PauliBitmaskSmall = PauliBitmaskGeneric<SmallVec<[u64; 8]>>;

// --- PartialEq, Eq, Hash, Ord for SmallVec backend ---

impl PartialEq for PauliBitmaskSmall {
    fn eq(&self, other: &Self) -> bool {
        vecs_eq_ignoring_trailing_zeros(&self.x_bits, &other.x_bits)
            && vecs_eq_ignoring_trailing_zeros(&self.z_bits, &other.z_bits)
    }
}

impl Eq for PauliBitmaskSmall {}

impl std::hash::Hash for PauliBitmaskSmall {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash_vec_ignoring_trailing_zeros(&self.x_bits, state);
        hash_vec_ignoring_trailing_zeros(&self.z_bits, state);
    }
}

impl PartialOrd for PauliBitmaskSmall {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PauliBitmaskSmall {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let max_len = self
            .x_bits
            .len()
            .max(other.x_bits.len())
            .max(self.z_bits.len())
            .max(other.z_bits.len());
        for i in (0..max_len).rev() {
            let sx = self.x_bits.get(i).copied().unwrap_or(0);
            let ox = other.x_bits.get(i).copied().unwrap_or(0);
            match sx.cmp(&ox) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        for i in (0..max_len).rev() {
            let sz = self.z_bits.get(i).copied().unwrap_or(0);
            let oz = other.z_bits.get(i).copied().unwrap_or(0);
            match sz.cmp(&oz) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}

// Copy only for fixed-size backends
impl Copy for PauliBitmask {}
impl PartialOrd for PauliBitmask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PauliBitmask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.x_bits
            .cmp(&other.x_bits)
            .then(self.z_bits.cmp(&other.z_bits))
    }
}

impl PartialOrd for PauliBitmaskVec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PauliBitmaskVec {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lexicographic comparison of the word vectors (most-significant word first)
        let max_len = self
            .x_bits
            .len()
            .max(other.x_bits.len())
            .max(self.z_bits.len())
            .max(other.z_bits.len());
        for i in (0..max_len).rev() {
            let sx = self.x_bits.get(i).copied().unwrap_or(0);
            let ox = other.x_bits.get(i).copied().unwrap_or(0);
            match sx.cmp(&ox) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        for i in (0..max_len).rev() {
            let sz = self.z_bits.get(i).copied().unwrap_or(0);
            let oz = other.z_bits.get(i).copied().unwrap_or(0);
            match sz.cmp(&oz) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl<B: BitmaskStorage> PauliBitmaskGeneric<B> {
    /// Single-qubit X on qubit q.
    #[must_use]
    pub fn x(q: usize) -> Self {
        let mut x = B::zero();
        x.set_bit(q);
        Self {
            x_bits: x,
            z_bits: B::zero(),
        }
    }

    /// Single-qubit Z on qubit q.
    #[must_use]
    pub fn z(q: usize) -> Self {
        let mut z = B::zero();
        z.set_bit(q);
        Self {
            x_bits: B::zero(),
            z_bits: z,
        }
    }

    /// Single-qubit Y on qubit q.
    #[must_use]
    pub fn y(q: usize) -> Self {
        let mut x = B::zero();
        let mut z = B::zero();
        x.set_bit(q);
        z.set_bit(q);
        Self {
            x_bits: x,
            z_bits: z,
        }
    }

    /// Product of two Pauli labels (XOR of symplectic vectors, phase ignored).
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let mut x = self.x_bits.clone();
        x.xor_assign(&other.x_bits);
        let mut z = self.z_bits.clone();
        z.xor_assign(&other.z_bits);
        Self {
            x_bits: x,
            z_bits: z,
        }
    }

    /// Product of two Paulis with phase tracking.
    ///
    /// Returns `(product, phase_exponent)` where the full product is i^phase · product.
    /// Phase exponent is in 0..4.
    #[must_use]
    pub fn multiply_with_phase(&self, other: &Self) -> (Self, u8) {
        // Per-qubit phase from Pauli multiplication.
        // Pauli types: I=0, X=1, Z=2, Y=3 (encoding: type = x + 2*z)
        // Phase lookup: A*B = i^{phase[A][B]} * C
        // I  X  Z  Y
        // 0  0  0  0   (I * anything)
        // 0  0  3  1   (X * I,X,Z,Y)
        // 0  1  0  3   (Z * I,X,Z,Y)
        // 0  3  1  0   (Y * I,X,Z,Y)
        const PHASE_TABLE: [[u8; 4]; 4] = [
            [0, 0, 0, 0], // I
            [0, 0, 3, 1], // X
            [0, 1, 0, 3], // Z
            [0, 3, 1, 0], // Y
        ];

        let product = self.multiply(other);
        let mut total_phase = 0u32;
        let max_q = [
            self.x_bits.highest_set_bit(),
            other.x_bits.highest_set_bit(),
            self.z_bits.highest_set_bit(),
            other.z_bits.highest_set_bit(),
        ]
        .into_iter()
        .flatten()
        .max()
        .map_or(0, |q| q + 1);

        for q in 0..max_q {
            let xa = usize::from(self.x_bits.get_bit(q));
            let za = usize::from(self.z_bits.get_bit(q));
            let xb = usize::from(other.x_bits.get_bit(q));
            let zb = usize::from(other.z_bits.get_bit(q));
            let type_a = xa + 2 * za; // I=0, X=1, Z=2, Y=3
            let type_b = xb + 2 * zb;
            total_phase += u32::from(PHASE_TABLE[type_a][type_b]);
        }

        (product, (total_phase % 4) as u8)
    }

    /// True if the two Paulis commute (symplectic inner product = 0 mod 2).
    #[must_use]
    pub fn commutes_with(&self, other: &Self) -> bool {
        self.x_bits
            .and_count_ones_xor(&other.z_bits, &self.z_bits, &other.x_bits)
            .is_multiple_of(2)
    }

    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.x_bits.is_zero() && self.z_bits.is_zero()
    }

    /// Number of non-identity single-qubit factors.
    #[must_use]
    pub fn weight(&self) -> u32 {
        self.x_bits.or_count_ones(&self.z_bits)
    }

    #[must_use]
    pub fn has_x(&self, q: usize) -> bool {
        self.x_bits.get_bit(q)
    }

    #[must_use]
    pub fn has_z(&self, q: usize) -> bool {
        self.z_bits.get_bit(q)
    }
}

impl PauliBitmask {
    pub const IDENTITY: Self = Self {
        x_bits: 0,
        z_bits: 0,
    };
}

impl<B: BitmaskStorage> PauliBitmaskGeneric<B> {
    /// Identity Pauli (all qubits I).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            x_bits: B::zero(),
            z_bits: B::zero(),
        }
    }
}

/// Convert from u128 (fixed-size) to Vec<u64> (unlimited) backend.
impl From<PauliBitmask> for PauliBitmaskVec {
    fn from(p: PauliBitmask) -> Self {
        let x_lo = u64::try_from(p.x_bits & u128::from(u64::MAX)).expect("masked low word fits");
        let x_hi = u64::try_from(p.x_bits >> 64).expect("shifted high word fits");
        let z_lo = u64::try_from(p.z_bits & u128::from(u64::MAX)).expect("masked low word fits");
        let z_hi = u64::try_from(p.z_bits >> 64).expect("shifted high word fits");
        Self {
            x_bits: if x_hi != 0 {
                vec![x_lo, x_hi]
            } else if x_lo != 0 {
                vec![x_lo]
            } else {
                vec![]
            },
            z_bits: if z_hi != 0 {
                vec![z_lo, z_hi]
            } else if z_lo != 0 {
                vec![z_lo]
            } else {
                vec![]
            },
        }
    }
}

impl From<PauliBitmask> for PauliBitmaskSmall {
    fn from(p: PauliBitmask) -> Self {
        let x_lo = u64::try_from(p.x_bits & u128::from(u64::MAX)).expect("masked low word fits");
        let x_hi = u64::try_from(p.x_bits >> 64).expect("shifted high word fits");
        let z_lo = u64::try_from(p.z_bits & u128::from(u64::MAX)).expect("masked low word fits");
        let z_hi = u64::try_from(p.z_bits >> 64).expect("shifted high word fits");
        let mut x = SmallVec::new();
        if x_hi != 0 {
            x.push(x_lo);
            x.push(x_hi);
        } else if x_lo != 0 {
            x.push(x_lo);
        }
        let mut z = SmallVec::new();
        if z_hi != 0 {
            z.push(z_lo);
            z.push(z_hi);
        } else if z_lo != 0 {
            z.push(z_lo);
        }
        Self {
            x_bits: x,
            z_bits: z,
        }
    }
}

impl<B: BitmaskStorage> fmt::Debug for PauliBitmaskGeneric<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "I");
        }
        let max_q = match self.x_bits.highest_set_bit() {
            Some(a) => match self.z_bits.highest_set_bit() {
                Some(b) => a.max(b) + 1,
                None => a + 1,
            },
            None => match self.z_bits.highest_set_bit() {
                Some(b) => b + 1,
                None => return write!(f, "I"),
            },
        };
        for q in 0..max_q {
            match (self.has_x(q), self.has_z(q)) {
                (false, false) => write!(f, "I")?,
                (true, false) => write!(f, "X")?,
                (false, true) => write!(f, "Z")?,
                (true, true) => write!(f, "Y")?,
            }
        }
        Ok(())
    }
}

// ============================================================
// Clifford conjugation: U†PU = sign * P'
// ============================================================

/// Result of Clifford conjugation U†PU.
#[derive(Clone, Debug)]
pub struct Conjugated<B: BitmaskStorage = u128> {
    pub label: PauliBitmaskGeneric<B>,
    /// True if the sign is negative (U†PU = -P').
    pub sign_negative: bool,
}

impl Copy for Conjugated<u128> {}

/// Hadamard on qubit q: X↔Z, Y→-Y.
#[must_use]
pub fn conjugate_h<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_h_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place Hadamard conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_h_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    let has_x = p.x_bits.get_bit(q);
    let has_z = p.z_bits.get_bit(q);
    if has_x != has_z {
        p.x_bits.xor_bit(q);
        p.z_bits.xor_bit(q);
    }
    has_x && has_z
}

/// SZ gate on qubit q: X→Y, Y→-X, Z→Z.
#[must_use]
pub fn conjugate_sz<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sz_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place SZ conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sz_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    if !p.x_bits.get_bit(q) {
        return false;
    }
    let was_y = p.z_bits.get_bit(q);
    p.z_bits.xor_bit(q);
    was_y
}

/// `SZdg` gate on qubit q: X→-Y, Y→X, Z→Z.
#[must_use]
pub fn conjugate_szdg<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_szdg_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place `SZdg` conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_szdg_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q: usize,
) -> bool {
    if !p.x_bits.get_bit(q) {
        return false;
    }
    let was_y = p.z_bits.get_bit(q);
    p.z_bits.xor_bit(q);
    !was_y
}

/// CX (CNOT) with control c, target t: XI→XX, IZ→ZZ.
///
/// The sign comes from Pauli multiplication phases when the control's
/// Pauli multiplies Z (from target Z spreading) and the target's Pauli
/// multiplies X (from control X spreading):
///   `phase_c` = phase(Pc · Z)  if target has Z, else 0
///   `phase_t` = phase(X · Pt)  if control has X, else 0
///   `sign_negative` = (`phase_c` + `phase_t`) % 4 == 2
#[must_use]
pub fn conjugate_cx<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    c: usize,
    t: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_cx_in_place(&mut label, c, t);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place CX conjugation with control c and target t.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_cx_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    c: usize,
    t: usize,
) -> bool {
    const PHASE: [[u8; 4]; 4] = [
        [0, 0, 0, 0], // I·{I,X,Z,Y}
        [0, 0, 3, 1], // X·{I,X,Z,Y}
        [0, 1, 0, 3], // Z·{I,X,Z,Y}
        [0, 3, 1, 0], // Y·{I,X,Z,Y}
    ];

    let cx = p.x_bits.get_bit(c);
    let cz = p.z_bits.get_bit(c);
    let tx = p.x_bits.get_bit(t);
    let tz = p.z_bits.get_bit(t);
    if cx {
        p.x_bits.xor_bit(t);
    }
    if tz {
        p.z_bits.xor_bit(c);
    }
    // Pauli type encoding: I=0, X=1, Z=2, Y=3 (x + 2*z)
    // Phase from Pauli multiplication table:
    //   Pc·Z at control (if tz), X·Pt at target (if cx)
    let pc = u8::from(cx) + 2 * u8::from(cz);
    let pt = u8::from(tx) + 2 * u8::from(tz);
    let phase_c = if tz { PHASE[pc as usize][2] } else { 0 };
    let phase_t = if cx { PHASE[1][pt as usize] } else { 0 };
    (phase_c + phase_t) % 4 == 2
}

/// CZ on qubits a, b: XI→XZ, IX→ZX, ZI→ZI, IZ→IZ.
#[must_use]
pub fn conjugate_cz<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    a: usize,
    b: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_cz_in_place(&mut label, a, b);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place CZ conjugation on qubits a and b.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_cz_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    a: usize,
    b: usize,
) -> bool {
    let ax = p.x_bits.get_bit(a);
    let az = p.z_bits.get_bit(a);
    let bx = p.x_bits.get_bit(b);
    let bz = p.z_bits.get_bit(b);
    if bx {
        p.z_bits.xor_bit(a);
    }
    if ax {
        p.z_bits.xor_bit(b);
    }
    ax && bx && (az != bz)
}

/// Pauli X gate on qubit q: Z→-Z, Y→-Y.
#[must_use]
pub fn conjugate_x<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_x_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place Pauli X conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_x_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    p.z_bits.get_bit(q)
}

/// Pauli Y gate on qubit q: X→-X, Z→-Z.
#[must_use]
pub fn conjugate_y<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_y_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place Pauli Y conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_y_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    p.x_bits.get_bit(q) != p.z_bits.get_bit(q)
}

/// Pauli Z gate on qubit q: X→-X, Y→-Y.
#[must_use]
pub fn conjugate_z<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_z_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place Pauli Z conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_z_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    p.x_bits.get_bit(q)
}

/// SWAP on qubits a, b: exchanges the Pauli at both sites.
#[must_use]
pub fn conjugate_swap<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    a: usize,
    b: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_swap_in_place(&mut label, a, b);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place SWAP conjugation on qubits a and b.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_swap_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    a: usize,
    b: usize,
) -> bool {
    let ax = p.x_bits.get_bit(a);
    let az = p.z_bits.get_bit(a);
    let bx = p.x_bits.get_bit(b);
    let bz = p.z_bits.get_bit(b);
    // Clear both positions
    p.x_bits.clear_bit(a);
    p.x_bits.clear_bit(b);
    if az {
        p.z_bits.clear_bit(a);
    }
    if bz {
        p.z_bits.clear_bit(b);
    }
    // Set swapped
    if bx {
        p.x_bits.set_bit(a);
    }
    if ax {
        p.x_bits.set_bit(b);
    }
    if bz {
        p.z_bits.set_bit(a);
    }
    if az {
        p.z_bits.set_bit(b);
    }
    false
}

/// SX gate on qubit q: X→X, Z→-Y, Y→Z.
#[must_use]
pub fn conjugate_sx<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sx_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place SX conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sx_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    let xq = p.x_bits.get_bit(q);
    let zq = p.z_bits.get_bit(q);
    if zq {
        p.x_bits.xor_bit(q);
    }
    !xq && zq
}

/// `SXdg` gate on qubit q: X→X, Z→Y, Y→-Z.
#[must_use]
pub fn conjugate_sxdg<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sxdg_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place `SXdg` conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sxdg_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q: usize,
) -> bool {
    let xq = p.x_bits.get_bit(q);
    let zq = p.z_bits.get_bit(q);
    if zq {
        p.x_bits.xor_bit(q);
    }
    xq && zq
}

/// SY gate on qubit q: X→-Z, Y→Y, Z→X.
#[must_use]
pub fn conjugate_sy<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sy_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place SY conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sy_in_place<B: BitmaskStorage>(p: &mut PauliBitmaskGeneric<B>, q: usize) -> bool {
    let xq = p.x_bits.get_bit(q);
    let zq = p.z_bits.get_bit(q);
    if xq != zq {
        p.x_bits.xor_bit(q);
        p.z_bits.xor_bit(q);
    }
    xq && !zq
}

/// `SYdg` gate on qubit q: X→Z, Y→Y, Z→-X.
#[must_use]
pub fn conjugate_sydg<B: BitmaskStorage>(p: &PauliBitmaskGeneric<B>, q: usize) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sydg_in_place(&mut label, q);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place `SYdg` conjugation on qubit q.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sydg_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q: usize,
) -> bool {
    let xq = p.x_bits.get_bit(q);
    let zq = p.z_bits.get_bit(q);
    if xq != zq {
        p.x_bits.xor_bit(q);
        p.z_bits.xor_bit(q);
    }
    !xq && zq
}

/// CY (controlled-Y) with control c, target t.
///
/// Decomposed as CY = (I⊗SZ) · CX · (I⊗SZdg), so conjugation is:
/// 1. conjugate by `SZdg` on target
/// 2. conjugate by CX
/// 3. conjugate by SZ on target
#[must_use]
pub fn conjugate_cy<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    c: usize,
    t: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_cy_in_place(&mut label, c, t);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place CY conjugation with control c and target t.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_cy_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    c: usize,
    t: usize,
) -> bool {
    conjugate_szdg_in_place(p, t) ^ conjugate_cx_in_place(p, c, t) ^ conjugate_sz_in_place(p, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PauliBitmask basics ---

    #[test]
    fn test_commutation() {
        assert!(!PauliBitmask::x(0).commutes_with(&PauliBitmask::z(0)));
        assert!(PauliBitmask::x(0).commutes_with(&PauliBitmask::x(1)));
        assert!(!PauliBitmask::x(0).commutes_with(&PauliBitmask::y(0)));

        let a = PauliBitmask {
            x_bits: 1,
            z_bits: 2,
        };
        let b = PauliBitmask {
            x_bits: 2,
            z_bits: 1,
        };
        assert!(a.commutes_with(&b));
    }

    #[test]
    fn test_multiply() {
        assert_eq!(
            PauliBitmask::x(0).multiply(&PauliBitmask::z(0)),
            PauliBitmask::y(0)
        );
    }

    #[test]
    fn test_h() {
        let r = conjugate_h(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(!r.sign_negative);

        let r = conjugate_h(&PauliBitmask::z(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(!r.sign_negative);

        let r = conjugate_h(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(r.sign_negative);
    }

    #[test]
    fn test_sz() {
        let r = conjugate_sz(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(!r.sign_negative);

        let r = conjugate_sz(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(r.sign_negative);

        let r = conjugate_sz(&PauliBitmask::z(0), 0);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_szdg() {
        let r = conjugate_szdg(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(r.sign_negative);

        let r = conjugate_szdg(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_cx() {
        let r = conjugate_cx(&PauliBitmask::x(0), 0, 1);
        assert_eq!(
            r.label,
            PauliBitmask {
                x_bits: 0b11,
                z_bits: 0
            }
        );
        assert!(!r.sign_negative);

        let r = conjugate_cx(&PauliBitmask::z(1), 0, 1);
        assert_eq!(
            r.label,
            PauliBitmask {
                x_bits: 0,
                z_bits: 0b11
            }
        );
        assert!(!r.sign_negative);

        let r = conjugate_cx(&PauliBitmask::x(1), 0, 1);
        assert_eq!(r.label, PauliBitmask::x(1));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_cz() {
        let r = conjugate_cz(&PauliBitmask::x(0), 0, 1);
        assert_eq!(
            r.label,
            PauliBitmask {
                x_bits: 1,
                z_bits: 2
            }
        );
        assert!(!r.sign_negative);

        let r = conjugate_cz(&PauliBitmask::z(0), 0, 1);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_swap() {
        let r = conjugate_swap(&PauliBitmask::x(0), 0, 1);
        assert_eq!(r.label, PauliBitmask::x(1));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_h_involution() {
        for p in [PauliBitmask::x(0), PauliBitmask::z(0), PauliBitmask::y(0)] {
            let r1 = conjugate_h(&p, 0);
            let r2 = conjugate_h(&r1.label, 0);
            assert_eq!(r2.label, p);
            assert!(!(r1.sign_negative ^ r2.sign_negative));
        }
    }

    #[test]
    fn test_sz_fourth_power() {
        let p = PauliBitmask::x(0);
        let mut label = p;
        let mut sign = false;
        for _ in 0..4 {
            let r = conjugate_sz(&label, 0);
            label = r.label;
            sign ^= r.sign_negative;
        }
        assert_eq!(label, p);
        assert!(!sign);
    }

    #[test]
    fn test_sx() {
        // X→X (no sign)
        let r = conjugate_sx(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(!r.sign_negative);

        // Z→-Y
        let r = conjugate_sx(&PauliBitmask::z(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(r.sign_negative);

        // Y→Z (no sign)
        let r = conjugate_sx(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_sxdg() {
        // X→X
        let r = conjugate_sxdg(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(!r.sign_negative);

        // Z→Y
        let r = conjugate_sxdg(&PauliBitmask::z(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(!r.sign_negative);

        // Y→-Z
        let r = conjugate_sxdg(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(r.sign_negative);
    }

    #[test]
    fn test_sy() {
        // X→-Z
        let r = conjugate_sy(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(r.sign_negative);

        // Z→X
        let r = conjugate_sy(&PauliBitmask::z(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(!r.sign_negative);

        // Y→Y
        let r = conjugate_sy(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_sydg() {
        // X→Z
        let r = conjugate_sydg(&PauliBitmask::x(0), 0);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(!r.sign_negative);

        // Z→-X
        let r = conjugate_sydg(&PauliBitmask::z(0), 0);
        assert_eq!(r.label, PauliBitmask::x(0));
        assert!(r.sign_negative);

        // Y→Y
        let r = conjugate_sydg(&PauliBitmask::y(0), 0);
        assert_eq!(r.label, PauliBitmask::y(0));
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_sx_fourth_power() {
        let p = PauliBitmask::z(0);
        let mut label = p;
        let mut sign = false;
        for _ in 0..4 {
            let r = conjugate_sx(&label, 0);
            label = r.label;
            sign ^= r.sign_negative;
        }
        assert_eq!(label, p);
        assert!(!sign);
    }

    #[test]
    fn test_sy_fourth_power() {
        let p = PauliBitmask::x(0);
        let mut label = p;
        let mut sign = false;
        for _ in 0..4 {
            let r = conjugate_sy(&label, 0);
            label = r.label;
            sign ^= r.sign_negative;
        }
        assert_eq!(label, p);
        assert!(!sign);
    }

    #[test]
    fn test_sx_sxdg_inverse() {
        for p in [PauliBitmask::x(0), PauliBitmask::z(0), PauliBitmask::y(0)] {
            let r1 = conjugate_sx(&p, 0);
            let r2 = conjugate_sxdg(&r1.label, 0);
            assert_eq!(r2.label, p);
            assert!(!(r1.sign_negative ^ r2.sign_negative));
        }
    }

    #[test]
    fn test_sy_sydg_inverse() {
        for p in [PauliBitmask::x(0), PauliBitmask::z(0), PauliBitmask::y(0)] {
            let r1 = conjugate_sy(&p, 0);
            let r2 = conjugate_sydg(&r1.label, 0);
            assert_eq!(r2.label, p);
            assert!(!(r1.sign_negative ^ r2.sign_negative));
        }
    }

    #[test]
    fn test_cy() {
        // X_c → X_c Y_t
        let r = conjugate_cy(&PauliBitmask::x(0), 0, 1);
        assert_eq!(
            r.label,
            PauliBitmask {
                x_bits: 0b11,
                z_bits: 0b10
            }
        );
        assert!(!r.sign_negative);

        // Z_c → Z_c
        let r = conjugate_cy(&PauliBitmask::z(0), 0, 1);
        assert_eq!(r.label, PauliBitmask::z(0));
        assert!(!r.sign_negative);

        // X_t → Z_c X_t
        let r = conjugate_cy(&PauliBitmask::x(1), 0, 1);
        assert_eq!(
            r.label,
            PauliBitmask {
                x_bits: 0b10,
                z_bits: 0b01
            }
        );
        assert!(!r.sign_negative);

        // Z_t → Z_c Z_t
        let r = conjugate_cy(&PauliBitmask::z(1), 0, 1);
        assert_eq!(
            r.label,
            PauliBitmask {
                x_bits: 0,
                z_bits: 0b11
            }
        );
        assert!(!r.sign_negative);
    }

    #[test]
    fn test_multiply_with_phase() {
        // X * Z = -iY (phase = 3, i.e., i^3 = -i)
        let (prod, phase) = PauliBitmask::x(0).multiply_with_phase(&PauliBitmask::z(0));
        assert_eq!(prod, PauliBitmask::y(0));
        assert_eq!(phase, 3); // i^3 = -i

        // Z * X = iY (phase = 1)
        let (prod, phase) = PauliBitmask::z(0).multiply_with_phase(&PauliBitmask::x(0));
        assert_eq!(prod, PauliBitmask::y(0));
        assert_eq!(phase, 1); // i^1 = i

        // X * X = I (phase = 0)
        let (prod, phase) = PauliBitmask::x(0).multiply_with_phase(&PauliBitmask::x(0));
        assert!(prod.is_identity());
        assert_eq!(phase, 0);

        // Y * Y = I (phase = 0)
        let (prod, phase) = PauliBitmask::y(0).multiply_with_phase(&PauliBitmask::y(0));
        assert!(prod.is_identity());
        assert_eq!(phase, 0);

        // Multi-qubit: (X⊗Z) * (Z⊗X) = (XZ)⊗(ZX) = (-iY)⊗(iY) = (-i·i)(Y⊗Y) = Y⊗Y
        let a = PauliBitmask {
            x_bits: 0b01,
            z_bits: 0b10,
        }; // XZ
        let b = PauliBitmask {
            x_bits: 0b10,
            z_bits: 0b01,
        }; // ZX
        let (prod, phase) = a.multiply_with_phase(&b);
        assert_eq!(
            prod,
            PauliBitmask {
                x_bits: 0b11,
                z_bits: 0b11
            }
        ); // YY
        assert_eq!(phase, 0); // (-i)(i) = 1, phase = 3+1 = 4 mod 4 = 0
    }

    #[test]
    fn test_cy_involution() {
        // CY is hermitian (CY² = I), so double conjugation should be identity
        for p in [
            PauliBitmask::x(0),
            PauliBitmask::z(0),
            PauliBitmask::x(1),
            PauliBitmask::z(1),
        ] {
            let r1 = conjugate_cy(&p, 0, 1);
            let r2 = conjugate_cy(&r1.label, 0, 1);
            assert_eq!(r2.label, p);
            assert!(!(r1.sign_negative ^ r2.sign_negative));
        }
    }
}
