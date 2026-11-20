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

/// Per-word quarter-phase contribution of a Pauli product, for the
/// `u64`-word backends. `lx`/`lz` are the lhs X/Z plane words, `rx`/`rz`
/// the rhs. Un-negated masks from both operands suppress zero-padding bits.
#[inline]
fn phase_terms_u64(lx: u64, lz: u64, rx: u64, rz: u64) -> u64 {
    u64::from((lx & !lz & rx & rz).count_ones()) // X*Y = iZ   (+1)
        + 3 * u64::from((lx & lz & rx & !rz).count_ones()) // Y*X = -iZ  (+3)
        + 3 * u64::from((lx & !lz & !rx & rz).count_ones()) // X*Z = -iY  (+3)
        + u64::from((!lx & lz & rx & !rz).count_ones()) // Z*X = iY   (+1)
        + u64::from((lx & lz & !rx & rz).count_ones()) // Y*Z = iX   (+1)
        + 3 * u64::from((!lx & lz & rx & rz).count_ones()) // Z*Y = -iX  (+3)
}

/// Per-word quarter-phase contribution of a Pauli acting on a computational
/// basis state, for the `u64`-word backends. `x`/`z` are the Pauli X/Z plane
/// words, `r` the basis-state word (set bit = that qubit is in state one).
#[inline]
fn basis_state_phase_terms_u64(x: u64, z: u64, r: u64) -> u64 {
    u64::from((x & z & !r).count_ones()) // Y|0> = i|1>   (+1)
        + 3 * u64::from((x & z & r).count_ones()) // Y|1> = -i|0>  (+3)
        + 2 * u64::from((!x & z & r).count_ones()) // Z|1> = -|1>   (+2)
}

/// Symplectic inner-product population count across zero-padded `u64` words.
fn and_count_ones_xor_words(
    self_x: &[u64],
    other_z: &[u64],
    self_z: &[u64],
    other_x: &[u64],
) -> u32 {
    let max = self_x
        .len()
        .max(other_z.len())
        .max(self_z.len())
        .max(other_x.len());
    let mut count = 0u32;
    for i in 0..max {
        let sx = self_x.get(i).copied().unwrap_or(0);
        let oz = other_z.get(i).copied().unwrap_or(0);
        let sz = self_z.get(i).copied().unwrap_or(0);
        let ox = other_x.get(i).copied().unwrap_or(0);
        count += ((sx & oz) ^ (sz & ox)).count_ones();
    }
    count
}

/// Quarter-phase exponent of a Pauli product across `u64` words.
fn mul_phase_exponent_words(lx: &[u64], lz: &[u64], rx: &[u64], rz: &[u64]) -> u8 {
    // A word only contributes where BOTH operands have stored bits, so
    // bound the walk by the shorter operand's plane length.
    let lhs_len = lx.len().max(lz.len());
    let rhs_len = rx.len().max(rz.len());
    let common_len = lhs_len.min(rhs_len);
    let mut total = 0u64;
    for i in 0..common_len {
        let lx = lx.get(i).copied().unwrap_or(0);
        let lz = lz.get(i).copied().unwrap_or(0);
        let rx = rx.get(i).copied().unwrap_or(0);
        let rz = rz.get(i).copied().unwrap_or(0);
        // Every term needs a set bit from each operand, so a word where
        // either side is all identity contributes nothing. Sparse Paulis
        // on a wide register are mostly such words.
        if (lx | lz) == 0 || (rx | rz) == 0 {
            continue;
        }
        // Reduce mod 4 each word so the accumulator stays small
        // regardless of input length.
        total = (total + phase_terms_u64(lx, lz, rx, rz)) % 4;
    }
    (total % 4) as u8
}

/// Quarter-phase exponent of a Pauli acting on a basis state across `u64` words.
fn basis_state_phase_exponent_words(x: &[u64], z: &[u64], state: &[u64]) -> u8 {
    // Every term needs a Z bit, and each needs either an X bit or a set
    // state bit, so words past those lengths cannot contribute. Padding
    // within the bound is meaningful: an absent state word means those
    // qubits are in state zero, and an absent X word means Z rather than Y.
    let contributing_len = z.len().min(x.len().max(state.len()));
    let mut total = 0u64;
    for i in 0..contributing_len {
        let x = x.get(i).copied().unwrap_or(0);
        let z = z.get(i).copied().unwrap_or(0);
        let r = state.get(i).copied().unwrap_or(0);
        // Reduce mod 4 each word so the accumulator stays small
        // regardless of input length.
        total = (total + basis_state_phase_terms_u64(x, z, r)) % 4;
    }
    (total % 4) as u8
}

/// Whether every `u64` word is zero.
fn is_zero_words(words: &[u64]) -> bool {
    words.iter().all(|&w| w == 0)
}

/// Population count of the bitwise OR across zero-padded `u64` words.
fn or_count_ones_words(a: &[u64], b: &[u64]) -> u32 {
    let max = a.len().max(b.len());
    let mut count = 0u32;
    for i in 0..max {
        let a = a.get(i).copied().unwrap_or(0);
        let b = b.get(i).copied().unwrap_or(0);
        count += (a | b).count_ones();
    }
    count
}

/// Index of the highest set bit across `u64` words.
fn highest_set_bit_words(words: &[u64]) -> Option<usize> {
    for (i, &w) in words.iter().enumerate().rev() {
        if w != 0 {
            return Some(i * 64 + 63 - w.leading_zeros() as usize);
        }
    }
    None
}

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
    /// Quarter-phase exponent of the Pauli product lhs * rhs, where
    /// lhs = (`self`, `self_z`) and rhs = (`other_x`, `other_z`) are X/Z bit planes.
    /// Returns k in 0..4 such that the full product is i^k times the
    /// phaseless symplectic product.
    fn mul_phase_exponent(&self, self_z: &Self, other_x: &Self, other_z: &Self) -> u8;
    /// Quarter-phase exponent of the Pauli (`self`, `self_z`) acting on the
    /// computational basis state `state`, whose set bits mark qubits in state
    /// one. Returns k in 0..4 such that `P|state> = i^k |state XOR x_bits>`.
    fn basis_state_phase_exponent(&self, self_z: &Self, state: &Self) -> u8;
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
    fn mul_phase_exponent(&self, self_z: &Self, other_x: &Self, other_z: &Self) -> u8 {
        // Same six terms as `phase_terms_u64`, on the full u128 width.
        let lx = *self;
        let lz = *self_z;
        let rx = *other_x;
        let rz = *other_z;
        let mut total = 0u64;
        total += u64::from((lx & !lz & rx & rz).count_ones());
        total += 3 * u64::from((lx & lz & rx & !rz).count_ones());
        total += 3 * u64::from((lx & !lz & !rx & rz).count_ones());
        total += u64::from((!lx & lz & rx & !rz).count_ones());
        total += u64::from((lx & lz & !rx & rz).count_ones());
        total += 3 * u64::from((!lx & lz & rx & rz).count_ones());
        (total % 4) as u8
    }
    fn basis_state_phase_exponent(&self, self_z: &Self, state: &Self) -> u8 {
        // Same three terms as `basis_state_phase_terms_u64`, on the full u128 width.
        let x = *self;
        let z = *self_z;
        let r = *state;
        let mut total = 0u64;
        total += u64::from((x & z & !r).count_ones());
        total += 3 * u64::from((x & z & r).count_ones());
        total += 2 * u64::from((!x & z & r).count_ones());
        (total % 4) as u8
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
        and_count_ones_xor_words(self, other_z, self_z, other_x)
    }
    fn mul_phase_exponent(&self, self_z: &Self, other_x: &Self, other_z: &Self) -> u8 {
        mul_phase_exponent_words(self, self_z, other_x, other_z)
    }
    fn basis_state_phase_exponent(&self, self_z: &Self, state: &Self) -> u8 {
        basis_state_phase_exponent_words(self, self_z, state)
    }
    fn is_zero(&self) -> bool {
        is_zero_words(self)
    }
    fn or_count_ones(&self, other: &Self) -> u32 {
        or_count_ones_words(self, other)
    }
    fn highest_set_bit(&self) -> Option<usize> {
        highest_set_bit_words(self)
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
        and_count_ones_xor_words(self, other_z, self_z, other_x)
    }
    fn mul_phase_exponent(&self, self_z: &Self, other_x: &Self, other_z: &Self) -> u8 {
        mul_phase_exponent_words(self, self_z, other_x, other_z)
    }
    fn basis_state_phase_exponent(&self, self_z: &Self, state: &Self) -> u8 {
        basis_state_phase_exponent_words(self, self_z, state)
    }
    fn is_zero(&self) -> bool {
        is_zero_words(self)
    }
    fn or_count_ones(&self, other: &Self) -> u32 {
        or_count_ones_words(self, other)
    }
    fn highest_set_bit(&self) -> Option<usize> {
        highest_set_bit_words(self)
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
        let product = self.multiply(other);
        let phase = self
            .x_bits
            .mul_phase_exponent(&self.z_bits, &other.x_bits, &other.z_bits);
        debug_assert_eq!(
            phase % 2,
            u8::from(!self.commutes_with(other)),
            "phase exponent parity must equal the symplectic product parity"
        );
        (product, phase)
    }

    /// Action of this Pauli on a computational basis state.
    ///
    /// `state` marks with a set bit each qubit in state one. Returns
    /// `(image, phase_exponent)` where `P|state> = i^phase · |image>`.
    /// Phase exponent is in 0..4.
    ///
    /// A Pauli maps a basis state to a single basis state, so the image is
    /// exact: `X` and `Y` flip their qubits, `Z` and `Y` contribute phase.
    #[must_use]
    pub fn apply_to_basis_state(&self, state: &B) -> (B, u8) {
        let mut image = state.clone();
        image.xor_assign(&self.x_bits);
        let phase = self.x_bits.basis_state_phase_exponent(&self.z_bits, state);
        (image, phase)
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
// Clifford conjugation: U P U† = sign * P'
// ============================================================

/// Result of Clifford conjugation U P U†.
#[derive(Clone, Debug)]
pub struct Conjugated<B: BitmaskStorage = u128> {
    pub label: PauliBitmaskGeneric<B>,
    /// True if the sign is negative (U P U† = -P').
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

/// Forward SZZ conjugation: P -> U P U†.
///
/// CX, SZ on q1, then CX.
#[must_use]
pub fn conjugate_szz<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_szz_in_place(&mut label, q0, q1);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place forward SZZ conjugation on q0 and q1.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_szz_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> bool {
    conjugate_cx_in_place(p, q0, q1)
        ^ conjugate_sz_in_place(p, q1)
        ^ conjugate_cx_in_place(p, q0, q1)
}

/// Forward `SZZdg` conjugation: P -> U P U†.
///
/// CX, `SZdg` on q1, then CX.
#[must_use]
pub fn conjugate_szzdg<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_szzdg_in_place(&mut label, q0, q1);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place forward `SZZdg` conjugation on q0 and q1.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_szzdg_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> bool {
    conjugate_cx_in_place(p, q0, q1)
        ^ conjugate_szdg_in_place(p, q1)
        ^ conjugate_cx_in_place(p, q0, q1)
}

/// Forward SXX conjugation: P -> U P U†.
///
/// Change basis with H on both qubits, conjugate by SZZ, then undo with H.
#[must_use]
pub fn conjugate_sxx<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sxx_in_place(&mut label, q0, q1);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place forward SXX conjugation on q0 and q1.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sxx_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> bool {
    conjugate_h_in_place(p, q0)
        ^ conjugate_h_in_place(p, q1)
        ^ conjugate_szz_in_place(p, q0, q1)
        ^ conjugate_h_in_place(p, q0)
        ^ conjugate_h_in_place(p, q1)
}

/// Forward `SXXdg` conjugation: P -> U P U†.
///
/// Change basis with H on both qubits, conjugate by `SZZdg`, then undo with H.
#[must_use]
pub fn conjugate_sxxdg<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_sxxdg_in_place(&mut label, q0, q1);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place forward `SXXdg` conjugation on q0 and q1.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_sxxdg_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> bool {
    conjugate_h_in_place(p, q0)
        ^ conjugate_h_in_place(p, q1)
        ^ conjugate_szzdg_in_place(p, q0, q1)
        ^ conjugate_h_in_place(p, q0)
        ^ conjugate_h_in_place(p, q1)
}

/// Forward SYY conjugation: P -> U P U†.
///
/// Change basis with SX on both qubits, conjugate by SZZ, then undo with `SXdg`.
#[must_use]
pub fn conjugate_syy<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_syy_in_place(&mut label, q0, q1);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place forward SYY conjugation on q0 and q1.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_syy_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> bool {
    conjugate_sx_in_place(p, q0)
        ^ conjugate_sx_in_place(p, q1)
        ^ conjugate_szz_in_place(p, q0, q1)
        ^ conjugate_sxdg_in_place(p, q0)
        ^ conjugate_sxdg_in_place(p, q1)
}

/// Forward `SYYdg` conjugation: P -> U P U†.
///
/// Change basis with SX on both qubits, conjugate by `SZZdg`, then undo with `SXdg`.
#[must_use]
pub fn conjugate_syydg<B: BitmaskStorage>(
    p: &PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> Conjugated<B> {
    let mut label = p.clone();
    let sign_negative = conjugate_syydg_in_place(&mut label, q0, q1);
    Conjugated {
        label,
        sign_negative,
    }
}

/// In-place forward `SYYdg` conjugation on q0 and q1.
///
/// Returns `true` when the conjugation contributes a negative sign.
#[must_use]
pub fn conjugate_syydg_in_place<B: BitmaskStorage>(
    p: &mut PauliBitmaskGeneric<B>,
    q0: usize,
    q1: usize,
) -> bool {
    conjugate_sx_in_place(p, q0)
        ^ conjugate_sx_in_place(p, q1)
        ^ conjugate_szzdg_in_place(p, q0, q1)
        ^ conjugate_sxdg_in_place(p, q0)
        ^ conjugate_sxdg_in_place(p, q1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Entry `(row, col)` of the 2x2 Pauli matrix on qubit `q` of `p`, with
    /// `Y = [[0, -i], [i, 0]]`. Uses explicit complex arithmetic,
    /// independently of the conjugation decompositions.
    fn local_pauli_entry(
        p: &PauliBitmask,
        q: usize,
        row: usize,
        col: usize,
    ) -> num_complex::Complex64 {
        use num_complex::Complex64;
        match (p.x_bits.get_bit(q), p.z_bits.get_bit(q)) {
            (false, false) => Complex64::new(f64::from(row == col), 0.0),
            (true, false) => Complex64::new(f64::from(row != col), 0.0),
            (false, true) => Complex64::new(
                if row == col {
                    if row == 0 { 1.0 } else { -1.0 }
                } else {
                    0.0
                },
                0.0,
            ),
            (true, true) => Complex64::new(
                0.0,
                if row == col {
                    0.0
                } else if row == 0 {
                    -1.0
                } else {
                    1.0
                },
            ),
        }
    }

    /// The 4x4 matrix of `p` restricted to `qubits`, in `|q0 q1>` order.
    fn two_qubit_pauli_matrix(
        p: &PauliBitmask,
        qubits: [usize; 2],
    ) -> [[num_complex::Complex64; 4]; 4] {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                local_pauli_entry(p, qubits[0], r / 2, c / 2)
                    * local_pauli_entry(p, qubits[1], r % 2, c % 2)
            })
        })
    }

    /// Complex matrix from the interleaved real/imaginary canonical layout.
    fn complex_matrix<const N: usize>(flat: &[f64]) -> [[num_complex::Complex64; N]; N] {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                num_complex::Complex64::new(flat[2 * (N * r + c)], flat[2 * (N * r + c) + 1])
            })
        })
    }

    /// `u m u^dagger` for square complex matrices.
    fn conjugate_matrix<const N: usize>(
        u: &[[num_complex::Complex64; N]; N],
        m: &[[num_complex::Complex64; N]; N],
    ) -> [[num_complex::Complex64; N]; N] {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                (0..N)
                    .flat_map(|i| (0..N).map(move |j| u[r][i] * m[i][j] * u[c][j].conj()))
                    .sum()
            })
        })
    }

    /// The unique `(candidate, negative)` with `image == +-matrix_of(candidate)`.
    fn unique_signed_match<'a, const N: usize>(
        image: &[[num_complex::Complex64; N]; N],
        candidates: &'a [PauliBitmask],
        matrix_of: impl Fn(&PauliBitmask) -> [[num_complex::Complex64; N]; N],
    ) -> (&'a PauliBitmask, bool) {
        let matches: Vec<_> = candidates
            .iter()
            .flat_map(|candidate| [false, true].map(move |negative| (candidate, negative)))
            .filter(|(candidate, negative)| {
                let expected = matrix_of(candidate);
                let sign = if *negative { -1.0 } else { 1.0 };
                image
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .all(|(a, b)| (*a - sign * b).norm() < 1e-12)
            })
            .collect();
        assert_eq!(matches.len(), 1, "image must be exactly one signed Pauli");
        matches[0]
    }

    #[test]
    fn single_qubit_conjugations_match_canonical_matrices() {
        use crate::Clifford;
        type Forward = fn(&PauliBitmask, usize) -> Conjugated;
        type InPlace = fn(&mut PauliBitmask, usize) -> bool;
        let cases: [(Clifford, Forward, InPlace); 10] = [
            (
                Clifford::X,
                conjugate_x::<u128>,
                conjugate_x_in_place::<u128>,
            ),
            (
                Clifford::Y,
                conjugate_y::<u128>,
                conjugate_y_in_place::<u128>,
            ),
            (
                Clifford::Z,
                conjugate_z::<u128>,
                conjugate_z_in_place::<u128>,
            ),
            (
                Clifford::H,
                conjugate_h::<u128>,
                conjugate_h_in_place::<u128>,
            ),
            (
                Clifford::SX,
                conjugate_sx::<u128>,
                conjugate_sx_in_place::<u128>,
            ),
            (
                Clifford::SXdg,
                conjugate_sxdg::<u128>,
                conjugate_sxdg_in_place::<u128>,
            ),
            (
                Clifford::SY,
                conjugate_sy::<u128>,
                conjugate_sy_in_place::<u128>,
            ),
            (
                Clifford::SYdg,
                conjugate_sydg::<u128>,
                conjugate_sydg_in_place::<u128>,
            ),
            (
                Clifford::SZ,
                conjugate_sz::<u128>,
                conjugate_sz_in_place::<u128>,
            ),
            (
                Clifford::SZdg,
                conjugate_szdg::<u128>,
                conjugate_szdg_in_place::<u128>,
            ),
        ];
        for (gate, conjugate, in_place) in cases {
            let u = complex_matrix::<2>(&gate.canonical_1q_matrix().unwrap());
            for q in [0, 65, 127] {
                let paulis = [
                    PauliBitmask::default(),
                    PauliBitmask::x(q),
                    PauliBitmask::y(q),
                    PauliBitmask::z(q),
                ];
                let matrix_of = |p: &PauliBitmask| -> [[num_complex::Complex64; 2]; 2] {
                    std::array::from_fn(|r| std::array::from_fn(|c| local_pauli_entry(p, q, r, c)))
                };
                for p in &paulis {
                    let image = conjugate_matrix(&u, &matrix_of(p));
                    let expected = unique_signed_match(&image, &paulis, matrix_of);
                    let result = conjugate(p, q);
                    assert_eq!(
                        (&result.label, result.sign_negative),
                        expected,
                        "{gate:?} q={q} {p:?}"
                    );
                    let mut label = *p;
                    let sign = in_place(&mut label, q);
                    assert_eq!((&label, sign), expected, "in-place {gate:?} q={q} {p:?}");
                }
            }
        }
    }

    #[test]
    fn two_qubit_conjugations_match_canonical_matrices() {
        use crate::Clifford;
        type Forward = fn(&PauliBitmask, usize, usize) -> Conjugated;
        type InPlace = fn(&mut PauliBitmask, usize, usize) -> bool;
        let cases: [(Clifford, Forward, InPlace); 10] = [
            (
                Clifford::CX,
                conjugate_cx::<u128>,
                conjugate_cx_in_place::<u128>,
            ),
            (
                Clifford::CY,
                conjugate_cy::<u128>,
                conjugate_cy_in_place::<u128>,
            ),
            (
                Clifford::CZ,
                conjugate_cz::<u128>,
                conjugate_cz_in_place::<u128>,
            ),
            (
                Clifford::SWAP,
                conjugate_swap::<u128>,
                conjugate_swap_in_place::<u128>,
            ),
            (
                Clifford::SZZ,
                conjugate_szz::<u128>,
                conjugate_szz_in_place::<u128>,
            ),
            (
                Clifford::SZZdg,
                conjugate_szzdg::<u128>,
                conjugate_szzdg_in_place::<u128>,
            ),
            (
                Clifford::SXX,
                conjugate_sxx::<u128>,
                conjugate_sxx_in_place::<u128>,
            ),
            (
                Clifford::SXXdg,
                conjugate_sxxdg::<u128>,
                conjugate_sxxdg_in_place::<u128>,
            ),
            (
                Clifford::SYY,
                conjugate_syy::<u128>,
                conjugate_syy_in_place::<u128>,
            ),
            (
                Clifford::SYYdg,
                conjugate_syydg::<u128>,
                conjugate_syydg_in_place::<u128>,
            ),
        ];
        for (gate, conjugate, in_place) in cases {
            let u = complex_matrix::<4>(&gate.canonical_2q_matrix().unwrap());
            for qubits in [[0, 1], [2, 65], [65, 2], [127, 0]] {
                let local = |q| {
                    [
                        PauliBitmask::default(),
                        PauliBitmask::x(q),
                        PauliBitmask::y(q),
                        PauliBitmask::z(q),
                    ]
                };
                let paulis: Vec<_> = local(qubits[0])
                    .into_iter()
                    .flat_map(|a| local(qubits[1]).map(|b| a.multiply(&b)))
                    .collect();
                let matrix_of = |p: &PauliBitmask| two_qubit_pauli_matrix(p, qubits);
                for p in &paulis {
                    let image = conjugate_matrix(&u, &matrix_of(p));
                    let expected = unique_signed_match(&image, &paulis, matrix_of);
                    let result = conjugate(p, qubits[0], qubits[1]);
                    assert_eq!(
                        (&result.label, result.sign_negative),
                        expected,
                        "{gate:?} {qubits:?} {p:?}"
                    );
                    let mut label = *p;
                    let sign = in_place(&mut label, qubits[0], qubits[1]);
                    assert_eq!((&label, sign), expected, "in-place {gate:?} {p:?}");
                }
            }
        }
    }

    fn reference_phase_exponent<B: BitmaskStorage>(
        a: &PauliBitmaskGeneric<B>,
        b: &PauliBitmaskGeneric<B>,
    ) -> u8 {
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

        let mut total_phase = 0u32;
        let max_q = [
            a.x_bits.highest_set_bit(),
            b.x_bits.highest_set_bit(),
            a.z_bits.highest_set_bit(),
            b.z_bits.highest_set_bit(),
        ]
        .into_iter()
        .flatten()
        .max()
        .map_or(0, |q| q + 1);

        for q in 0..max_q {
            let xa = usize::from(a.x_bits.get_bit(q));
            let za = usize::from(a.z_bits.get_bit(q));
            let xb = usize::from(b.x_bits.get_bit(q));
            let zb = usize::from(b.z_bits.get_bit(q));
            let type_a = xa + 2 * za; // I=0, X=1, Z=2, Y=3
            let type_b = xb + 2 * zb;
            total_phase += u32::from(PHASE_TABLE[type_a][type_b]);
        }

        (total_phase % 4) as u8
    }

    fn random_pauli<B: BitmaskStorage>(
        rng: &mut StdRng,
        num_qubits: usize,
        dense: bool,
    ) -> PauliBitmaskGeneric<B> {
        let mut pauli = PauliBitmaskGeneric::<B>::identity();
        let samples = if dense {
            num_qubits
        } else {
            rng.random_range(0..=num_qubits.min(8))
        };
        for sample in 0..samples {
            let q = if dense {
                sample
            } else {
                rng.random_range(0..num_qubits)
            };
            let first_pauli_type = usize::from(!dense);
            let pauli_type = rng.random_range(first_pauli_type..4);
            if pauli_type & 1 != 0 {
                pauli.x_bits.set_bit(q);
            }
            if pauli_type & 2 != 0 {
                pauli.z_bits.set_bit(q);
            }
        }
        pauli
    }

    fn assert_random_phase_matches_reference<B: BitmaskStorage>(
        rng: &mut StdRng,
        qubit_counts: &[usize],
        cases_per_count: usize,
    ) where
        PauliBitmaskGeneric<B>: PartialEq,
    {
        for &num_qubits in qubit_counts {
            for case in 0..cases_per_count {
                let a = random_pauli(rng, num_qubits, case % 2 == 0);
                let b = random_pauli(rng, num_qubits, case % 4 < 2);
                let expected_product = a.multiply(&b);
                let expected_phase = reference_phase_exponent(&a, &b);
                let (product, phase) = a.multiply_with_phase(&b);
                assert_eq!(
                    product, expected_product,
                    "product mismatch for {num_qubits} qubits in case {case}"
                );
                assert_eq!(
                    phase, expected_phase,
                    "phase mismatch for {num_qubits} qubits in case {case}"
                );
            }
        }
    }

    fn pauli_from_sites<B: BitmaskStorage>(sites: &[(usize, u8)]) -> PauliBitmaskGeneric<B> {
        let mut pauli = PauliBitmaskGeneric::<B>::identity();
        for &(q, pauli_type) in sites {
            if pauli_type & 1 != 0 {
                pauli.x_bits.set_bit(q);
            }
            if pauli_type & 2 != 0 {
                pauli.z_bits.set_bit(q);
            }
        }
        pauli
    }

    /// Per-qubit oracle for `apply_to_basis_state`, independent of the
    /// word-level formula: I and X carry no phase, Z contributes -1 on a
    /// qubit in state one, Y contributes i on state zero and -i on state one.
    fn reference_basis_state_phase<B: BitmaskStorage>(
        pauli: &PauliBitmaskGeneric<B>,
        state: &B,
    ) -> u8 {
        let max_q = [
            pauli.x_bits.highest_set_bit(),
            pauli.z_bits.highest_set_bit(),
            state.highest_set_bit(),
        ]
        .into_iter()
        .flatten()
        .max()
        .map_or(0, |q| q + 1);

        let mut total = 0u32;
        for q in 0..max_q {
            let x = pauli.x_bits.get_bit(q);
            let z = pauli.z_bits.get_bit(q);
            let one = state.get_bit(q);
            total += match (x, z, one) {
                (true, true, false) => 1, // Y|0> = i|1>
                (true, true, true) => 3,  // Y|1> = -i|0>
                (false, true, true) => 2, // Z|1> = -|1>
                _ => 0,                   // I, X, and Z|0>
            };
        }
        (total % 4) as u8
    }

    fn basis_state_from_sites<B: BitmaskStorage>(sites: &[usize]) -> B {
        let mut state = B::zero();
        for &q in sites {
            state.set_bit(q);
        }
        state
    }

    fn assert_unequal_length_phase_matches_reference<B: BitmaskStorage>()
    where
        PauliBitmaskGeneric<B>: PartialEq,
    {
        let short = pauli_from_sites::<B>(&[(2, 1), (17, 3), (63, 2)]);
        let long = pauli_from_sites::<B>(&[(2, 2), (17, 1), (63, 3), (129, 1), (191, 3)]);
        // X-only vs Z-only supports give the four planes independently
        // different stored lengths (x long / z short and vice versa).
        let x_heavy = pauli_from_sites::<B>(&[(3, 2), (530, 1), (600, 1)]);
        let z_heavy = pauli_from_sites::<B>(&[(3, 1), (530, 2), (600, 2)]);
        // Four pairwise-distinct stored plane lengths (2, 5, 8, 11 words),
        // with anticommuting overlap at qubits 70 and 300 and support past
        // word 9 on one side only. The two overlaps contribute 3 each, so the
        // expected phase is 2 rather than 0: a phase-zeroing defect fails here.
        let distinct_a = pauli_from_sites::<B>(&[(70, 1), (300, 2)]);
        let distinct_b = pauli_from_sites::<B>(&[(70, 2), (300, 3), (450, 1), (640, 2)]);
        // The only phase-contributing overlap sits in word 10, so a word walk
        // that stops early still has to reach it. Every other fixture here
        // contributes solely from words 0 through 9.
        let high_word_a = pauli_from_sites::<B>(&[(700, 1)]);
        let high_word_b = pauli_from_sites::<B>(&[(700, 2)]);
        for (a, b) in [
            (&short, &long),
            (&long, &short),
            (&x_heavy, &z_heavy),
            (&z_heavy, &x_heavy),
            (&distinct_a, &distinct_b),
            (&distinct_b, &distinct_a),
            (&high_word_a, &high_word_b),
            (&high_word_b, &high_word_a),
        ] {
            let expected_product = a.multiply(b);
            let expected_phase = reference_phase_exponent(a, b);
            let (product, phase) = a.multiply_with_phase(b);
            assert_eq!(product, expected_product);
            assert_eq!(phase, expected_phase);
        }
    }

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

    fn assert_exhaustive_single_qubit_phases<B: BitmaskStorage>()
    where
        PauliBitmaskGeneric<B>: PartialEq,
    {
        let paulis = [
            PauliBitmaskGeneric::<B>::identity(),
            PauliBitmaskGeneric::<B>::x(0),
            PauliBitmaskGeneric::<B>::z(0),
            PauliBitmaskGeneric::<B>::y(0),
        ];
        let expected = [[0, 0, 0, 0], [0, 0, 3, 1], [0, 1, 0, 3], [0, 3, 1, 0]];

        for (a_index, a) in paulis.iter().enumerate() {
            for (b_index, b) in paulis.iter().enumerate() {
                let (product, phase) = a.multiply_with_phase(b);
                assert_eq!(product, a.multiply(b));
                assert_eq!(phase, expected[a_index][b_index]);
            }
        }
    }

    #[test]
    fn multiply_with_phase_exhaustive_single_qubit() {
        assert_exhaustive_single_qubit_phases::<u128>();
        assert_exhaustive_single_qubit_phases::<Vec<u64>>();
        assert_exhaustive_single_qubit_phases::<SmallVec<[u64; 8]>>();
    }

    #[test]
    fn multiply_with_phase_matches_reference_for_random_backends() {
        let mut rng = StdRng::seed_from_u64(0x5eed_51a5_e123_0a11);
        assert_random_phase_matches_reference::<u128>(&mut rng, &[1, 5, 63, 64, 65, 127, 128], 40);
        assert_random_phase_matches_reference::<Vec<u64>>(
            &mut rng,
            &[
                1, 5, 63, 64, 65, 127, 128, 129, 191, 192, 193, 300, 511, 512, 520,
            ],
            24,
        );
        // 511/512/520 cross the SmallVec inline capacity (8 words = 512 bits),
        // exercising the heap-spilled representation.
        assert_random_phase_matches_reference::<SmallVec<[u64; 8]>>(
            &mut rng,
            &[
                1, 5, 63, 64, 65, 127, 128, 129, 191, 192, 193, 300, 511, 512, 520,
            ],
            24,
        );
    }

    /// Bits covered by `assert_bit_primitives_match_model`: every word the
    /// starting lengths and qubits below can reach, plus one more.
    const MODEL_BITS: usize = 11 * 64;

    /// Asserts `storage` holds exactly the bits of `model`, reading every
    /// position in range and checking that no bit is set beyond it.
    fn assert_storage_matches<S: BitmaskStorage>(storage: &S, model: &[bool], context: &str) {
        for (q, &expected) in model.iter().enumerate() {
            assert_eq!(storage.get_bit(q), expected, "{context}: bit {q}");
        }
        assert!(
            storage.highest_set_bit().is_none_or(|q| q < MODEL_BITS),
            "{context}: bit set beyond the model"
        );
    }

    /// Each bit primitive against a `Vec<bool>` model, from every starting
    /// length up to past the `SmallVec` inline capacity of 8 words, at qubits
    /// on word edges and past the spill point. Every bit is compared, so a
    /// write to the wrong position anywhere in the storage fails.
    fn assert_bit_primitives_match_model<S: BitmaskStorage + From<Vec<u64>>>() {
        const QUBITS: [usize; 10] = [0, 1, 63, 64, 127, 128, 511, 512, 520, 600];
        for len in 0..=9 {
            let words: Vec<u64> = (0..len)
                .map(|i| 0x9e37_79b9_7f4a_7c15_u64.rotate_left(u32::try_from(i).unwrap()))
                .collect();
            let model: Vec<bool> = (0..MODEL_BITS)
                .map(|q| words.get(q / 64).is_some_and(|w| (w >> (q % 64)) & 1 == 1))
                .collect();
            assert_storage_matches(&S::from(words.clone()), &model, &format!("len={len}"));
            for q in QUBITS {
                let mut storage = S::from(words.clone());
                let mut expected = model.clone();
                storage.xor_bit(q);
                expected[q] = !expected[q];
                assert_storage_matches(&storage, &expected, &format!("xor_bit len={len} q={q}"));

                let mut storage = S::from(words.clone());
                let mut expected = model.clone();
                storage.set_bit(q);
                expected[q] = true;
                assert_storage_matches(&storage, &expected, &format!("set_bit len={len} q={q}"));

                let mut storage = S::from(words.clone());
                let mut expected = model.clone();
                storage.clear_bit(q);
                expected[q] = false;
                assert_storage_matches(&storage, &expected, &format!("clear_bit len={len} q={q}"));
            }
        }
    }

    #[test]
    fn bit_primitives_match_model_across_growth() {
        assert_bit_primitives_match_model::<Vec<u64>>();
        assert_bit_primitives_match_model::<SmallVec<[u64; 8]>>();
    }

    #[test]
    fn multiply_with_phase_handles_unequal_vec_lengths() {
        assert_unequal_length_phase_matches_reference::<Vec<u64>>();
    }

    #[test]
    fn multiply_with_phase_handles_unequal_smallvec_lengths() {
        assert_unequal_length_phase_matches_reference::<SmallVec<[u64; 8]>>();
    }

    fn assert_exhaustive_single_qubit_basis_action<B: BitmaskStorage>() {
        // (pauli, qubit is one) -> (qubit flips, phase exponent)
        let cases = [
            (PauliBitmaskGeneric::<B>::identity(), false, false, 0),
            (PauliBitmaskGeneric::<B>::identity(), true, false, 0),
            (PauliBitmaskGeneric::<B>::x(0), false, true, 0),
            (PauliBitmaskGeneric::<B>::x(0), true, true, 0),
            (PauliBitmaskGeneric::<B>::z(0), false, false, 0),
            (PauliBitmaskGeneric::<B>::z(0), true, false, 2),
            (PauliBitmaskGeneric::<B>::y(0), false, true, 1),
            (PauliBitmaskGeneric::<B>::y(0), true, true, 3),
        ];

        for (pauli, one, flips, expected_phase) in cases {
            let state = basis_state_from_sites::<B>(if one { &[0] } else { &[] });
            let (image, phase) = pauli.apply_to_basis_state(&state);
            assert_eq!(phase, expected_phase);
            assert_eq!(image.get_bit(0), one != flips);
        }
    }

    #[test]
    fn apply_to_basis_state_exhaustive_single_qubit() {
        assert_exhaustive_single_qubit_basis_action::<u128>();
        assert_exhaustive_single_qubit_basis_action::<Vec<u64>>();
        assert_exhaustive_single_qubit_basis_action::<SmallVec<[u64; 8]>>();
    }

    fn assert_basis_action_matches_reference<B: BitmaskStorage>(
        rng: &mut StdRng,
        qubit_counts: &[usize],
        cases_per_count: usize,
    ) {
        for &num_qubits in qubit_counts {
            for case in 0..cases_per_count {
                let pauli = random_pauli::<B>(rng, num_qubits, case % 2 == 0);
                let mut state = B::zero();
                for q in 0..num_qubits {
                    if rng.random_bool(if case % 4 < 2 { 0.5 } else { 0.1 }) {
                        state.set_bit(q);
                    }
                }
                let (image, phase) = pauli.apply_to_basis_state(&state);
                assert_eq!(
                    phase,
                    reference_basis_state_phase(&pauli, &state),
                    "phase mismatch for {num_qubits} qubits in case {case}"
                );
                for q in 0..num_qubits {
                    assert_eq!(
                        image.get_bit(q),
                        state.get_bit(q) != pauli.x_bits.get_bit(q),
                        "image bit {q} wrong for {num_qubits} qubits in case {case}"
                    );
                }
            }
        }
    }

    #[test]
    fn apply_to_basis_state_matches_reference_for_random_backends() {
        let mut rng = StdRng::seed_from_u64(0x0006_a515_7a7e_0bad);
        assert_basis_action_matches_reference::<u128>(&mut rng, &[1, 5, 63, 64, 65, 127, 128], 40);
        assert_basis_action_matches_reference::<Vec<u64>>(
            &mut rng,
            &[
                1, 5, 63, 64, 65, 127, 128, 129, 191, 192, 193, 300, 511, 512, 520,
            ],
            24,
        );
        assert_basis_action_matches_reference::<SmallVec<[u64; 8]>>(
            &mut rng,
            &[
                1, 5, 63, 64, 65, 127, 128, 129, 191, 192, 193, 300, 511, 512, 520,
            ],
            24,
        );
    }

    fn assert_basis_action_handles_unequal_lengths<B: BitmaskStorage>() {
        // Pauli support far past the state's stored words, and vice versa,
        // so both zero-padding directions are exercised. The set state bit at
        // qubit 640 puts a pure-Z contribution in a word past the X plane's
        // stored length, so a walk bounded by the X plane alone misses it.
        let wide_pauli = pauli_from_sites::<B>(&[(3, 3), (70, 2), (300, 3), (640, 2)]);
        let narrow_pauli = pauli_from_sites::<B>(&[(3, 2), (5, 3)]);
        let narrow_state = basis_state_from_sites::<B>(&[3, 5]);
        let wide_state = basis_state_from_sites::<B>(&[3, 70, 300, 640, 700]);
        for (pauli, state) in [
            (&wide_pauli, &narrow_state),
            (&wide_pauli, &wide_state),
            (&narrow_pauli, &wide_state),
            (&narrow_pauli, &narrow_state),
        ] {
            let (image, phase) = pauli.apply_to_basis_state(state);
            assert_eq!(phase, reference_basis_state_phase(pauli, state));
            for q in [3, 5, 70, 300, 450, 640, 700] {
                assert_eq!(
                    image.get_bit(q),
                    state.get_bit(q) != pauli.x_bits.get_bit(q),
                    "image bit {q} wrong"
                );
            }
        }
    }

    #[test]
    fn apply_to_basis_state_handles_unequal_vec_lengths() {
        assert_basis_action_handles_unequal_lengths::<Vec<u64>>();
    }

    #[test]
    fn apply_to_basis_state_handles_unequal_smallvec_lengths() {
        assert_basis_action_handles_unequal_lengths::<SmallVec<[u64; 8]>>();
    }

    #[test]
    fn apply_to_basis_state_composes_with_multiply_with_phase() {
        // Applying p then q must agree with applying the product q*p, tying
        // the basis-state phase convention to the operator-product one
        // without sharing an oracle between them.
        let mut rng = StdRng::seed_from_u64(0xc0d9_05e1_2345_6789);
        for case in 0..200 {
            let num_qubits = 130;
            let p = random_pauli::<SmallVec<[u64; 8]>>(&mut rng, num_qubits, case % 2 == 0);
            let q = random_pauli::<SmallVec<[u64; 8]>>(&mut rng, num_qubits, case % 3 == 0);
            let mut state = SmallVec::<[u64; 8]>::zero();
            for site in 0..num_qubits {
                if rng.random_bool(0.5) {
                    state.set_bit(site);
                }
            }

            let (after_p, phase_p) = p.apply_to_basis_state(&state);
            let (after_q, phase_q) = q.apply_to_basis_state(&after_p);

            let (product, product_phase) = q.multiply_with_phase(&p);
            let (direct, phase_direct) = product.apply_to_basis_state(&state);

            for site in 0..num_qubits {
                assert_eq!(after_q.get_bit(site), direct.get_bit(site), "case {case}");
            }
            assert_eq!(
                (phase_p + phase_q) % 4,
                (product_phase + phase_direct) % 4,
                "sequential and product phases disagree in case {case}"
            );
        }
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

/// Kani proofs for the bit-parallel phase arithmetic.
///
/// The oracle is derived from the Y = iXZ convention rather than from the
/// six-case product table the implementation uses. Writing a Pauli as
/// `P(x, z) = i^(x z) X^x Z^z` and commuting `Z^za` past `X^xb` gives, per
/// qubit, `P(a) P(b) = i^e P(a xor b)` with
/// `e = xa za + xb zb + 2 za xb - xc zc` (mod 4), where `xc = xa xor xb`
/// and `zc = za xor zb`. Acting on a basis state gives
/// `P(x, z) |r> = i^(x z + 2 z r) |r xor x>`. Summing over qubits turns each
/// product of bits into a population count.
///
/// Run with `cargo kani -p pecos-core`.
#[cfg(kani)]
mod verification {
    use super::*;

    /// Product phase exponent from the Y = iXZ derivation, over 128 qubits.
    fn spec_mul_exponent(lx: u128, lz: u128, rx: u128, rz: u128) -> u8 {
        let xc = lx ^ rx;
        let zc = lz ^ rz;
        // Subtracting the xc zc term is adding 3 copies of it mod 4.
        let e = (lx & lz).count_ones()
            + (rx & rz).count_ones()
            + 2 * (lz & rx).count_ones()
            + 3 * (xc & zc).count_ones();
        (e % 4) as u8
    }

    /// Basis-state phase exponent from the Y = iXZ derivation, over 128 qubits.
    fn spec_basis_exponent(x: u128, z: u128, r: u128) -> u8 {
        let e = (x & z).count_ones() + 2 * (z & r).count_ones();
        (e % 4) as u8
    }

    /// A word plane of symbolic length 0..=2, plus its zero-padded u128 value.
    fn any_plane() -> (Vec<u64>, u128) {
        let lo: u64 = kani::any();
        let hi: u64 = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 2);
        let mut words = vec![lo, hi];
        words.truncate(len);
        let value = match len {
            0 => 0,
            1 => u128::from(lo),
            _ => u128::from(lo) | (u128::from(hi) << 64),
        };
        (words, value)
    }

    // Proving the product phase against the spec for all 64 or 128 qubits at
    // once does not finish: the solver must show two different popcount
    // circuits agree, and the cost grows from 30 s at 8 bits to over 5 minutes
    // at 16. The proofs below check every single-qubit product at every bit
    // position. Full width then follows from the code's shape, which is not
    // machine-checked: each term is a popcount of a bitwise (same-position)
    // mask, so qubits contribute independently and the sums cannot overflow.
    // The 8-qubit window proofs are a bounded check of that shape; they would
    // miss a term that mixes positions further apart, such as a shifted mask.

    /// One symbolic qubit `(x, z)` at symbolic position `pos`, other bits clear.
    fn one_qubit_u128(pos: u32, x: bool, z: bool) -> (u128, u128) {
        (u128::from(x) << pos, u128::from(z) << pos)
    }

    #[kani::proof]
    fn phase_terms_u64_matches_spec_on_each_qubit() {
        let pos: u32 = kani::any();
        kani::assume(pos < 64);
        let (lx, lz) = one_qubit_u128(pos, kani::any(), kani::any());
        let (rx, rz) = one_qubit_u128(pos, kani::any(), kani::any());
        let word = |v: u128| u64::try_from(v).expect("pos < 64 keeps the qubit in the low word");
        let got = (phase_terms_u64(word(lx), word(lz), word(rx), word(rz)) % 4) as u8;
        assert_eq!(got, spec_mul_exponent(lx, lz, rx, rz));
    }

    #[kani::proof]
    fn phase_terms_u64_matches_spec_on_8_qubits() {
        let (lx, lz, rx, rz): (u8, u8, u8, u8) = kani::any();
        let (lx, lz, rx, rz) = (u64::from(lx), u64::from(lz), u64::from(rx), u64::from(rz));
        let got = (phase_terms_u64(lx, lz, rx, rz) % 4) as u8;
        assert_eq!(
            got,
            spec_mul_exponent(lx.into(), lz.into(), rx.into(), rz.into())
        );
    }

    #[kani::proof]
    fn u128_mul_phase_exponent_matches_spec_on_each_qubit() {
        let pos: u32 = kani::any();
        kani::assume(pos < 128);
        let (lx, lz) = one_qubit_u128(pos, kani::any(), kani::any());
        let (rx, rz) = one_qubit_u128(pos, kani::any(), kani::any());
        assert_eq!(
            lx.mul_phase_exponent(&lz, &rx, &rz),
            spec_mul_exponent(lx, lz, rx, rz)
        );
    }

    /// The window straddles the two 64-bit halves. A symbolic window position
    /// did not finish in 20 minutes.
    #[kani::proof]
    fn u128_mul_phase_exponent_matches_spec_on_8_qubits() {
        let (lx, lz, rx, rz): (u8, u8, u8, u8) = kani::any();
        let place = |v: u8| u128::from(v) << 60;
        let (lx, lz, rx, rz) = (place(lx), place(lz), place(rx), place(rz));
        assert_eq!(
            lx.mul_phase_exponent(&lz, &rx, &rz),
            spec_mul_exponent(lx, lz, rx, rz)
        );
    }

    /// The word backend is the mod-4 sum of `phase_terms_u64` over zero-padded
    /// words, for planes of up to two words. With the `phase_terms_u64` proofs
    /// this checks the backend against the spec on each qubit; checking it
    /// directly did not finish in 30 minutes. Longer planes rely on the loop
    /// treating every word alike, which is not machine-checked.
    #[kani::proof]
    #[kani::unwind(3)]
    fn vec_mul_phase_exponent_is_the_padded_word_sum() {
        let (lx, _) = any_plane();
        let (lz, _) = any_plane();
        let (rx, _) = any_plane();
        let (rz, _) = any_plane();
        let word = |plane: &Vec<u64>, i: usize| plane.get(i).copied().unwrap_or(0);
        let mut expected = 0;
        for i in 0..2 {
            expected += phase_terms_u64(word(&lx, i), word(&lz, i), word(&rx, i), word(&rz, i));
        }
        assert_eq!(
            u64::from(lx.mul_phase_exponent(&lz, &rx, &rz)),
            expected % 4
        );
    }

    #[kani::proof]
    fn basis_state_phase_terms_u64_matches_spec() {
        let (x, z, r): (u64, u64, u64) = kani::any();
        let got = (basis_state_phase_terms_u64(x, z, r) % 4) as u8;
        assert_eq!(got, spec_basis_exponent(x.into(), z.into(), r.into()));
    }

    #[kani::proof]
    fn u128_basis_state_phase_exponent_matches_spec() {
        let (x, z, r): (u128, u128, u128) = kani::any();
        assert_eq!(
            x.basis_state_phase_exponent(&z, &r),
            spec_basis_exponent(x, z, r)
        );
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn vec_basis_state_phase_exponent_matches_spec() {
        let (x, x_v) = any_plane();
        let (z, z_v) = any_plane();
        let (r, r_v) = any_plane();
        assert_eq!(
            x.basis_state_phase_exponent(&z, &r),
            spec_basis_exponent(x_v, z_v, r_v)
        );
    }

    // Clifford conjugation. The unit tests `*_conjugations_match_canonical_matrices`
    // pin each gate's local rule against its canonical matrix at fixed
    // positions. The `*_is_local` proofs extend that to every u128 Pauli and
    // every valid position (qubits below 128, and distinct for two-qubit
    // gates): only the target qubits change, they change exactly as at the
    // canonical positions, and the forward and in-place forms agree. Qubits
    // at or above 128 and coinciding two-qubit operands are excluded inputs,
    // not proven behavior.
    //
    // The conjugation functions are generic over the storage backend and touch
    // it only through `get_bit`, `xor_bit`, `set_bit` and `clear_bit`. The
    // `*_bit_refines_u128` proofs show those four behave on `Vec<u64>` as on
    // u128 for planes of up to two words and qubits below 128, so within those
    // bounds the u128 proofs carry over (see below for `SmallVec`). Beyond them
    // the unit test `bit_primitives_match_model_across_growth` covers longer
    // storage. Proving each gate on `Vec<u64>` directly took 99 s and 10 GB for
    // X and ran out of memory at 76 GB for H.

    fn bit(v: u128, q: usize) -> u128 {
        (v >> q) & 1
    }

    /// Value of a word plane as u128 (planes here hold at most two words).
    fn plane_value(words: &[u64]) -> u128 {
        let word = |i: usize| u128::from(words.get(i).copied().unwrap_or(0));
        word(0) | (word(1) << 64)
    }

    /// Checks one single-qubit conjugation on `(x, z)` at `q` against the
    /// same conjugation of the local Pauli at qubit 0.
    fn check_1q(
        x: u128,
        z: u128,
        out_x: u128,
        out_z: u128,
        sign: bool,
        q: usize,
        in_place: fn(&mut PauliBitmask, usize) -> bool,
    ) {
        let others = !(1_u128 << q);
        assert_eq!(out_x & others, x & others);
        assert_eq!(out_z & others, z & others);
        let mut local = PauliBitmask {
            x_bits: bit(x, q),
            z_bits: bit(z, q),
        };
        let local_sign = in_place(&mut local, 0);
        assert_eq!(
            (bit(out_x, q), bit(out_z, q), sign),
            (local.x_bits, local.z_bits, local_sign)
        );
    }

    /// Two-qubit counterpart of `check_1q`, against qubits 0 and 1.
    #[allow(clippy::too_many_arguments)]
    fn check_2q(
        x: u128,
        z: u128,
        out_x: u128,
        out_z: u128,
        sign: bool,
        a: usize,
        b: usize,
        in_place: fn(&mut PauliBitmask, usize, usize) -> bool,
    ) {
        let others = !((1_u128 << a) | (1_u128 << b));
        assert_eq!(out_x & others, x & others);
        assert_eq!(out_z & others, z & others);
        let mut local = PauliBitmask {
            x_bits: bit(x, a) | (bit(x, b) << 1),
            z_bits: bit(z, a) | (bit(z, b) << 1),
        };
        let local_sign = in_place(&mut local, 0, 1);
        assert_eq!(
            (
                bit(out_x, a),
                bit(out_z, a),
                bit(out_x, b),
                bit(out_z, b),
                sign
            ),
            (
                bit(local.x_bits, 0),
                bit(local.z_bits, 0),
                bit(local.x_bits, 1),
                bit(local.z_bits, 1),
                local_sign
            )
        );
    }

    fn any_qubit() -> usize {
        let q: usize = kani::any();
        kani::assume(q < 128);
        q
    }

    macro_rules! conjugation_1q_proofs {
        ($name:ident, $forward:ident, $in_place:ident) => {
            #[kani::proof]
            fn $name() {
                let p = PauliBitmask {
                    x_bits: kani::any(),
                    z_bits: kani::any(),
                };
                let q = any_qubit();
                let forward = $forward(&p, q);
                let mut out = p;
                let sign = $in_place(&mut out, q);
                assert_eq!((&forward.label, forward.sign_negative), (&out, sign));
                check_1q(
                    p.x_bits,
                    p.z_bits,
                    out.x_bits,
                    out.z_bits,
                    sign,
                    q,
                    $in_place::<u128>,
                );
            }
        };
    }

    macro_rules! conjugation_2q_proofs {
        ($name:ident, $forward:ident, $in_place:ident) => {
            #[kani::proof]
            fn $name() {
                let p = PauliBitmask {
                    x_bits: kani::any(),
                    z_bits: kani::any(),
                };
                let (a, b) = (any_qubit(), any_qubit());
                kani::assume(a != b);
                let forward = $forward(&p, a, b);
                let mut out = p;
                let sign = $in_place(&mut out, a, b);
                assert_eq!((&forward.label, forward.sign_negative), (&out, sign));
                check_2q(
                    p.x_bits,
                    p.z_bits,
                    out.x_bits,
                    out.z_bits,
                    sign,
                    a,
                    b,
                    $in_place::<u128>,
                );
            }
        };
    }

    conjugation_1q_proofs!(conjugate_x_is_local, conjugate_x, conjugate_x_in_place);
    conjugation_1q_proofs!(conjugate_y_is_local, conjugate_y, conjugate_y_in_place);
    conjugation_1q_proofs!(conjugate_z_is_local, conjugate_z, conjugate_z_in_place);
    conjugation_1q_proofs!(conjugate_h_is_local, conjugate_h, conjugate_h_in_place);
    conjugation_1q_proofs!(conjugate_sx_is_local, conjugate_sx, conjugate_sx_in_place);
    conjugation_1q_proofs!(
        conjugate_sxdg_is_local,
        conjugate_sxdg,
        conjugate_sxdg_in_place
    );
    conjugation_1q_proofs!(conjugate_sy_is_local, conjugate_sy, conjugate_sy_in_place);
    conjugation_1q_proofs!(
        conjugate_sydg_is_local,
        conjugate_sydg,
        conjugate_sydg_in_place
    );
    conjugation_1q_proofs!(conjugate_sz_is_local, conjugate_sz, conjugate_sz_in_place);
    conjugation_1q_proofs!(
        conjugate_szdg_is_local,
        conjugate_szdg,
        conjugate_szdg_in_place
    );
    conjugation_2q_proofs!(conjugate_cx_is_local, conjugate_cx, conjugate_cx_in_place);
    conjugation_2q_proofs!(conjugate_cy_is_local, conjugate_cy, conjugate_cy_in_place);
    conjugation_2q_proofs!(conjugate_cz_is_local, conjugate_cz, conjugate_cz_in_place);
    conjugation_2q_proofs!(
        conjugate_swap_is_local,
        conjugate_swap,
        conjugate_swap_in_place
    );
    conjugation_2q_proofs!(
        conjugate_szz_is_local,
        conjugate_szz,
        conjugate_szz_in_place
    );
    conjugation_2q_proofs!(
        conjugate_szzdg_is_local,
        conjugate_szzdg,
        conjugate_szzdg_in_place
    );
    conjugation_2q_proofs!(
        conjugate_sxx_is_local,
        conjugate_sxx,
        conjugate_sxx_in_place
    );
    conjugation_2q_proofs!(
        conjugate_sxxdg_is_local,
        conjugate_sxxdg,
        conjugate_sxxdg_in_place
    );
    conjugation_2q_proofs!(
        conjugate_syy_is_local,
        conjugate_syy,
        conjugate_syy_in_place
    );
    conjugation_2q_proofs!(
        conjugate_syydg_is_local,
        conjugate_syydg,
        conjugate_syydg_in_place
    );

    /// A word storage built from `words`.
    trait FromWords: BitmaskStorage {
        fn from_words(words: Vec<u64>) -> Self;
        fn word_len(&self) -> usize;
        fn plane(&self) -> u128;
    }

    impl FromWords for Vec<u64> {
        fn from_words(words: Vec<u64>) -> Self {
            words
        }
        fn word_len(&self) -> usize {
            self.len()
        }
        fn plane(&self) -> u128 {
            plane_value(self)
        }
    }

    impl FromWords for SmallVec<[u64; 8]> {
        fn from_words(words: Vec<u64>) -> Self {
            SmallVec::from_vec(words)
        }
        fn word_len(&self) -> usize {
            self.len()
        }
        fn plane(&self) -> u128 {
            plane_value(self)
        }
    }

    /// A storage of symbolic length 0..=2 and a symbolic qubit in its range.
    fn any_storage<S: FromWords>() -> (S, u128, usize) {
        let (words, value) = any_plane();
        (S::from_words(words), value, any_qubit())
    }

    /// Applies `update` and checks the result against `expected`, and that the
    /// storage did not grow past the two words that `plane` reads.
    fn check_update<S: FromWords>(update: fn(&mut S, usize), expected: fn(u128, u128) -> u128) {
        let (mut storage, value, q) = any_storage::<S>();
        update(&mut storage, q);
        assert!(storage.word_len() <= 2);
        assert_eq!(storage.plane(), expected(value, 1_u128 << q));
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn vec_get_bit_refines_u128() {
        let (storage, value, q) = any_storage::<Vec<u64>>();
        assert_eq!(storage.get_bit(q), bit(value, q) == 1);
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn vec_xor_bit_refines_u128() {
        check_update::<Vec<u64>>(|s, q| s.xor_bit(q), |v, m| v ^ m);
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn vec_set_bit_refines_u128() {
        check_update::<Vec<u64>>(|s, q| s.set_bit(q), |v, m| v | m);
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn vec_clear_bit_refines_u128() {
        check_update::<Vec<u64>>(|s, q| s.clear_bit(q), |v, m| v & !m);
    }

    // `SmallVec` has the same `get_bit`, `xor_bit`, `set_bit` and `clear_bit`
    // source as `Vec<u64>`. Only the two that never grow the storage are proven
    // here: CBMC runs out of memory (76 GB) encoding `SmallVec::resize`, even
    // when an assumption makes it unreachable. The unit test
    // `bit_primitives_match_model_across_growth` covers growth for both
    // backends, including the spill past the inline capacity.

    #[kani::proof]
    #[kani::unwind(3)]
    fn smallvec_get_bit_refines_u128() {
        let (storage, value, q) = any_storage::<SmallVec<[u64; 8]>>();
        assert_eq!(storage.get_bit(q), bit(value, q) == 1);
    }

    #[kani::proof]
    #[kani::unwind(3)]
    fn smallvec_clear_bit_refines_u128() {
        check_update::<SmallVec<[u64; 8]>>(|s, q| s.clear_bit(q), |v, m| v & !m);
    }
}
