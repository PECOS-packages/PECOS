// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Quadratic form over GF(2) with exponential sum evaluation.
//!
//! Computes `Z = sum_x exp(i * pi/4 * q(x))` where `q(x)` is a quadratic form
//! over n-bit binary strings, in O(n^3) time via Gaussian elimination.
//!
//! The quadratic form is:
//! ```text
//! q(x) = Q + sum_a D_a * x_a + 4 * sum_{a<b} J_{a,b} * x_a * x_b  (mod 8)
//! ```
//!
//! where Q is a constant (mod 8), `D_a` takes values 0,2,4,6, and J is a symmetric
//! binary matrix with zero diagonal.
//!
//! # References
//!
//! Bravyi et al. "Simulation of quantum circuits by low-rank stabilizer
//! decompositions." arXiv:1808.00128, Section V.

use super::exact_scalar::ExactScalar;

/// A quadratic form over GF(2) for exponential sum evaluation.
///
/// Represents `q(x) = Q + sum D_a x_a + 4 sum_{a<b} J_{ab} x_a x_b (mod 8)`
/// where D is encoded as two bitstrings (D1, D2) with `D_a = 2*D2_a + D1_a * 2`...
/// D is encoded as two bitstrings (D1, D2) with `D_a = 2*(2*D2_a + D1_a)`,
/// giving `D_a` in {0, 2, 4, 6}.
pub struct QuadraticForm {
    /// Number of variables.
    pub n: usize,
    /// Words per row in the J matrix.
    pub row_words: usize,
    /// Constant term, mod 8.
    pub q_const: i32,
    /// Linear diagonal part, packed as Vec<u64> bitstrings (64 bits per word).
    /// D1 bit i = bit 0 of `D_i` coefficient.
    pub d1: Vec<u64>,
    /// D2 bit i = bit 1 of `D_i` coefficient.
    pub d2: Vec<u64>,
    /// Quadratic part: symmetric binary matrix, flat storage.
    /// Row i is at `j_flat`[i * `row_words` .. (i+1) * `row_words`].
    pub j_flat: Vec<u64>,
}

/// Number of u64 words needed for n bits.
#[inline]
#[must_use]
pub fn words(n: usize) -> usize {
    n.div_ceil(64)
}

/// Get bit i from a packed word array.
#[inline]
#[must_use]
pub fn get_bit(words: &[u64], i: usize) -> bool {
    words[i / 64] & (1u64 << (i % 64)) != 0
}

/// Set bit i in a packed word array.
#[inline]
pub fn set_bit(words: &mut [u64], i: usize) {
    words[i / 64] |= 1u64 << (i % 64);
}

/// XOR two packed word arrays.
#[inline]
pub fn xor_words(a: &mut [u64], b: &[u64]) {
    for (aw, bw) in a.iter_mut().zip(b.iter()) {
        *aw ^= *bw;
    }
}

/// Check if all bits are zero.
#[inline]
#[must_use]
pub fn is_zero(words: &[u64]) -> bool {
    words.iter().all(|&w| w == 0)
}

/// Find the lowest set bit index, or None.
#[inline]
#[must_use]
pub fn lowest_set_bit(words: &[u64]) -> Option<usize> {
    for (wi, &w) in words.iter().enumerate() {
        if w != 0 {
            return Some(wi * 64 + w.trailing_zeros() as usize);
        }
    }
    None
}

/// Clear bit i.
#[inline]
pub fn clear_bit(words: &mut [u64], i: usize) {
    words[i / 64] &= !(1u64 << (i % 64));
}

/// Clear all occurrences of bit i across a Vec of packed rows (column clear).
#[inline]
pub fn clear_column(rows: &mut [Vec<u64>], bit: usize) {
    let word_idx = bit / 64;
    let mask = !(1u64 << (bit % 64));
    for row in rows.iter_mut() {
        if word_idx < row.len() {
            row[word_idx] &= mask;
        }
    }
}

/// Macro generating the `ExponentialSum` for a fixed-width packed integer type.
/// Instantiated for u64 (<=62 qubits) and u128 (63-126 qubits).
macro_rules! exp_sum_fixed {
    ($self:expr, $T:ty) => {{
        let slf = $self;
        let n = slf.n;
        let m = n + 1;

        let mut pow2_real: i32 = 0;
        let mut sigma_real = false;
        let mut is_zero_real = false;
        let mut pow2_imag: i32 = 0;
        let mut sigma_imag = false;
        let mut is_zero_imag = false;

        let one: $T = 1;
        let zero: $T = 0;

        // Pack D1 into the fixed type (only first word needed for u64, first two for u128)
        let mut c: $T = 0;
        for i in 0..n {
            if get_bit(&slf.d1, i) {
                c |= one << i;
            }
        }

        let mut big_m: Vec<$T> = vec![zero; m];
        big_m[n] = c;

        let mut l_real: $T = 0;
        let mut l_imag: $T = 0;
        for i in 0..n {
            if get_bit(&slf.d2, i) {
                l_real |= one << i;
                l_imag |= one << i;
            }
        }
        l_imag |= one << n;

        for i in 0..n {
            for jj in (i + 1)..n {
                let j_bit = get_bit(slf.j_row(i), jj);
                let c1 = c & (one << i) != zero;
                let c2 = c & (one << jj) != zero;
                if j_bit ^ (c1 & c2) {
                    big_m[jj] |= one << i;
                }
            }
        }

        let mut active: $T = if m < (std::mem::size_of::<$T>() * 8) {
            (one << m) - one
        } else {
            !zero
        };
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        // m = n+1 where n is qubit count, fits in i32
        let mut n_active = m as i32;

        while n_active >= 1 {
            let i1 = active.trailing_zeros() as usize;
            if i1 >= m {
                break;
            }

            let mut row_i1: $T = zero;
            for k in 0..m {
                if big_m[k] & (one << i1) != zero {
                    row_i1 |= one << k;
                }
            }
            let asymmetric = row_i1 ^ big_m[i1];
            let i2 = if asymmetric != zero {
                asymmetric.trailing_zeros() as usize
            } else {
                m
            };

            let diag_i1 = (big_m[i1] >> i1) & one != zero;
            let l1_real = ((l_real >> i1) & one != zero) ^ diag_i1;
            let l1_imag = ((l_imag >> i1) & one != zero) ^ diag_i1;

            if i2 >= m {
                if l1_real {
                    is_zero_real = true;
                }
                pow2_real += 1;
                if l1_imag {
                    is_zero_imag = true;
                }
                pow2_imag += 1;
                n_active -= 1;
                active &= !(one << i1);
                big_m[i1] = zero;
                let not_i1 = !(one << i1);
                for k in 0..m {
                    big_m[k] &= not_i1;
                }
                l_real &= not_i1;
                l_imag &= not_i1;
                if is_zero_real && is_zero_imag {
                    return ExactScalar::zero();
                }
                continue;
            }

            let diag_i2 = (big_m[i2] >> i2) & one != zero;
            let l2_real = ((l_real >> i2) & one != zero) ^ diag_i2;
            let l2_imag = ((l_imag >> i2) & one != zero) ^ diag_i2;
            l_real &= !((one << i1) | (one << i2));
            l_imag &= !((one << i1) | (one << i2));

            let mut m1: $T = zero;
            let mut m2: $T = zero;
            for k in 0..m {
                if big_m[k] & (one << i1) != zero {
                    m1 |= one << k;
                }
                if big_m[k] & (one << i2) != zero {
                    m2 |= one << k;
                }
            }
            m1 ^= big_m[i1];
            m2 ^= big_m[i2];
            let clear = !((one << i1) | (one << i2));
            m1 &= clear;
            m2 &= clear;

            big_m[i1] = zero;
            big_m[i2] = zero;
            for k in 0..m {
                big_m[k] &= clear;
            }

            if !is_zero_real {
                pow2_real += 1;
                sigma_real ^= l1_real && l2_real;
                if l1_real {
                    l_real ^= m2;
                }
                if l2_real {
                    l_real ^= m1;
                }
            }
            if !is_zero_imag {
                pow2_imag += 1;
                sigma_imag ^= l1_imag && l2_imag;
                if l1_imag {
                    l_imag ^= m2;
                }
                if l2_imag {
                    l_imag ^= m1;
                }
            }

            let mut bits = m2;
            while bits != zero {
                let k = bits.trailing_zeros() as usize;
                big_m[k] ^= m1;
                bits &= bits.wrapping_sub(one);
            }

            active &= clear;
            n_active -= 2;
        }

        combine_result(
            slf.q_const,
            pow2_real,
            sigma_real,
            is_zero_real,
            pow2_imag,
            sigma_imag,
            is_zero_imag,
        )
    }};
}

impl QuadraticForm {
    /// Create a zero quadratic form with n variables.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let w = words(n);
        Self {
            n,
            row_words: w,
            q_const: 0,
            d1: vec![0u64; w],
            d2: vec![0u64; w],
            j_flat: vec![0u64; n * w],
        }
    }

    /// Get row i of the J matrix.
    #[inline]
    #[must_use]
    pub fn j_row(&self, i: usize) -> &[u64] {
        let w = self.row_words;
        &self.j_flat[i * w..(i + 1) * w]
    }

    /// Get mutable row i of the J matrix.
    #[inline]
    pub fn j_row_mut(&mut self, i: usize) -> &mut [u64] {
        let w = self.row_words;
        &mut self.j_flat[i * w..(i + 1) * w]
    }

    /// Evaluate the exponential sum: `sum_x exp(i*pi/4 * q(x))`.
    ///
    /// Returns an `ExactScalar` of the form `eps * 2^{p/2} * exp(i*pi*e/4)`.
    /// Time complexity: O(n^3) via Gaussian elimination.
    ///
    /// Based on Bravyi et al. arXiv:1808.00128, Section V.
    #[must_use]
    pub fn exponential_sum(&self) -> ExactScalar {
        let n = self.n;
        if n == 0 {
            let e = ((self.q_const % 8) + 8) % 8;
            #[allow(clippy::cast_sign_loss)] // (x % 8 + 8) % 8 is in [0,7]
            return ExactScalar::from_phase(e as u8);
        }

        // Fast path for purely linear form (J = 0): the sum factors as a product.
        // sum_x exp(i*pi/4 * (Q + sum D_a x_a)) = exp(i*pi/4*Q) * prod_a (1 + exp(i*pi/4*D_a))
        // D_a = 2*(2*D2_a + D1_a), so exp(i*pi/4*D_a) = i^{D_a/2} = i^{2*D2_a + D1_a}.
        // Factors: D_a=0 -> 2, D_a=2 -> 1+i, D_a=4 -> 0, D_a=6 -> 1-i.
        if is_zero(&self.j_flat) {
            // Purely linear: O(n) product formula.
            let mut pow2: i32 = 0;
            let mut phase8: i32 = ((self.q_const % 8) + 8) % 8;
            for i in 0..n {
                let d1_bit = get_bit(&self.d1, i);
                let d2_bit = get_bit(&self.d2, i);
                let d_idx = u8::from(d2_bit) * 2 + u8::from(d1_bit); // 0,1,2,3 -> D=0,2,4,6
                match d_idx {
                    0 => {
                        pow2 += 2;
                    } // factor = 2 = 2^{2/2}
                    1 => {
                        pow2 += 1;
                        phase8 = (phase8 + 1) % 8;
                    } // factor = 1+i = sqrt(2)*e^{i*pi/4}
                    2 => {
                        return ExactScalar::zero();
                    } // factor = 0
                    3 => {
                        pow2 += 1;
                        phase8 = (phase8 + 7) % 8;
                    } // factor = 1-i = sqrt(2)*e^{-i*pi/4}
                    _ => unreachable!(),
                }
            }
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            // phase8 is in [0,7] via % 8
            let mut result = ExactScalar::from_phase(phase8 as u8);
            result.mul_sqrt2_pow(pow2);
            return result;
        }

        // Three-tier dispatch: u64 for <=62, u128 for 63-126, Vec<u64> for larger.
        if n <= 62 {
            return self.exponential_sum_u64();
        }
        if n <= 126 {
            return self.exponential_sum_u128();
        }

        let mut pow2_real: i32 = 0;
        let mut sigma_real = false;
        let mut is_zero_real = false;
        let mut pow2_imag: i32 = 0;
        let mut sigma_imag = false;
        let mut is_zero_imag = false;

        let m = n + 1;
        let w = words(m);

        // M is (n+1) columns, each a packed bitstring of m bits.
        // Flat storage: column k is at bm[k*w..(k+1)*w].
        let mut bm = vec![0u64; m * w];

        // Set M column n = D1 bits.
        let col_n = n * w;
        bm[col_n..(words(n) + col_n)].copy_from_slice(&self.d1[..words(n)]);

        // L vectors as packed bitstrings
        let mut l_real = vec![0u64; w];
        let mut l_imag = vec![0u64; w];
        l_real[..words(n)].copy_from_slice(&self.d2[..words(n)]);
        l_imag[..words(n)].copy_from_slice(&self.d2[..words(n)]);
        set_bit(&mut l_imag, n);

        // Fill M from J adjusted by D1.
        for i in 0..n {
            let c1 = get_bit(&self.d1, i);
            let ji_row = self.j_row(i);
            let i_word = i / 64;
            let i_bit = 1u64 << (i % 64);
            let start_jj = i + 1;
            let start_w = start_jj / 64;
            for (wi, &j_word) in ji_row.iter().enumerate().take(words(n)).skip(start_w) {
                let adj = if c1 { j_word ^ self.d1[wi] } else { j_word };
                let mask = if wi == start_w && start_jj % 64 != 0 {
                    adj & !((1u64 << (start_jj % 64)) - 1)
                } else {
                    adj
                };
                let mut bits = mask;
                while bits != 0 {
                    let jj = wi * 64 + bits.trailing_zeros() as usize;
                    if jj >= n {
                        break;
                    }
                    bm[jj * w + i_word] |= i_bit;
                    bits &= bits.wrapping_sub(1);
                }
            }
        }

        // Active tracking
        let mut active = vec![0u64; w];
        for i in 0..m {
            set_bit(&mut active, i);
        }
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        // m = n+1 where n is qubit count, fits in i32
        let mut n_active = m as i32;

        // Pre-allocate scratch buffers.
        let mut row_i1 = vec![0u64; w];
        let mut asymmetric = vec![0u64; w];
        let mut m1 = vec![0u64; w];
        let mut m2 = vec![0u64; w];

        while n_active >= 1 {
            let Some(i1) = lowest_set_bit(&active) else {
                break;
            };
            if i1 >= m {
                break;
            }
            let i1_off = i1 * w;

            // Build row i1 and find asymmetric entry
            row_i1.fill(0);
            for k in 0..m {
                if get_bit(&bm[k * w..(k + 1) * w], i1) {
                    set_bit(&mut row_i1, k);
                }
            }
            for wi in 0..w {
                asymmetric[wi] = row_i1[wi] ^ bm[i1_off + wi];
            }
            let i2 = lowest_set_bit(&asymmetric).unwrap_or(m);

            let diag_i1 = get_bit(&bm[i1_off..i1_off + w], i1);
            let l1_real = get_bit(&l_real, i1) ^ diag_i1;
            let l1_imag = get_bit(&l_imag, i1) ^ diag_i1;

            if i2 >= m {
                if l1_real {
                    is_zero_real = true;
                }
                pow2_real += 1;
                if l1_imag {
                    is_zero_imag = true;
                }
                pow2_imag += 1;
                n_active -= 1;
                clear_bit(&mut active, i1);
                bm[i1_off..i1_off + w].fill(0);
                // Clear column i1 across all rows
                let cw = i1 / 64;
                let cmask = !(1u64 << (i1 % 64));
                for k in 0..m {
                    bm[k * w + cw] &= cmask;
                }
                clear_bit(&mut l_real, i1);
                clear_bit(&mut l_imag, i1);
                if is_zero_real && is_zero_imag {
                    return ExactScalar::zero();
                }
                continue;
            }

            let i2_off = i2 * w;
            let diag_i2 = get_bit(&bm[i2_off..i2_off + w], i2);
            let l2_real = get_bit(&l_real, i2) ^ diag_i2;
            let l2_imag = get_bit(&l_imag, i2) ^ diag_i2;

            clear_bit(&mut l_real, i1);
            clear_bit(&mut l_real, i2);
            clear_bit(&mut l_imag, i1);
            clear_bit(&mut l_imag, i2);

            // Extract symmetrized rows
            m1.fill(0);
            m2.fill(0);
            for k in 0..m {
                let ko = k * w;
                if get_bit(&bm[ko..ko + w], i1) {
                    set_bit(&mut m1, k);
                }
                if get_bit(&bm[ko..ko + w], i2) {
                    set_bit(&mut m2, k);
                }
            }
            xor_words(&mut m1, &bm[i1_off..i1_off + w]);
            xor_words(&mut m2, &bm[i2_off..i2_off + w]);
            clear_bit(&mut m1, i1);
            clear_bit(&mut m1, i2);
            clear_bit(&mut m2, i1);
            clear_bit(&mut m2, i2);

            // Zero columns i1 and i2
            bm[i1_off..i1_off + w].fill(0);
            bm[i2_off..i2_off + w].fill(0);
            let cw1 = i1 / 64;
            let cmask1 = !(1u64 << (i1 % 64));
            let cw2 = i2 / 64;
            let cmask2 = !(1u64 << (i2 % 64));
            for k in 0..m {
                let ko = k * w;
                bm[ko + cw1] &= cmask1;
                bm[ko + cw2] &= cmask2;
            }

            // Update L vectors
            if !is_zero_real {
                pow2_real += 1;
                sigma_real ^= l1_real && l2_real;
                if l1_real {
                    xor_words(&mut l_real, &m2);
                }
                if l2_real {
                    xor_words(&mut l_real, &m1);
                }
            }
            if !is_zero_imag {
                pow2_imag += 1;
                sigma_imag ^= l1_imag && l2_imag;
                if l1_imag {
                    xor_words(&mut l_imag, &m2);
                }
                if l2_imag {
                    xor_words(&mut l_imag, &m1);
                }
            }

            // Update M: for each k where m2[k]=1, XOR m1 into column k
            for k in 0..m {
                if get_bit(&m2, k) {
                    let ko = k * w;
                    xor_words(&mut bm[ko..ko + w], &m1);
                }
            }

            clear_bit(&mut active, i1);
            clear_bit(&mut active, i2);
            n_active -= 2;
        }

        combine_result(
            self.q_const,
            pow2_real,
            sigma_real,
            is_zero_real,
            pow2_imag,
            sigma_imag,
            is_zero_imag,
        )
    }

    /// u128 fast path for 63 <= n <= 126 (m = n+1 fits in u128).
    fn exponential_sum_u128(&self) -> ExactScalar {
        exp_sum_fixed!(self, u128)
    }

    /// u64 fast path for n <= 62 (m = n+1 fits in u64).
    fn exponential_sum_u64(&self) -> ExactScalar {
        exp_sum_fixed!(self, u64)
    }
}

/// Shared result combination for `ExponentialSum`.
#[allow(clippy::fn_params_excessive_bools)]
fn combine_result(
    q_const: i32,
    pow2_real: i32,
    sigma_real: bool,
    is_zero_real: bool,
    pow2_imag: i32,
    sigma_imag: bool,
    is_zero_imag: bool,
) -> ExactScalar {
    let q_mod8 = q_const.rem_euclid(8);

    if is_zero_imag {
        let mut result = ExactScalar::one();
        result.mul_sqrt2_pow(2 * pow2_real - 2);
        #[allow(clippy::cast_sign_loss)] // all terms non-negative, result in [0,7]
        let e = ((4 * i32::from(sigma_real) + q_mod8) % 8) as u8;
        result.mul_phase(e);
        result
    } else if is_zero_real {
        let mut result = ExactScalar::one();
        result.mul_sqrt2_pow(2 * pow2_imag - 2);
        #[allow(clippy::cast_sign_loss)] // all terms non-negative, result in [0,7]
        let e = ((2 + 4 * i32::from(sigma_imag) + q_mod8) % 8) as u8;
        result.mul_phase(e);
        result
    } else {
        let mut result = ExactScalar::one();
        result.mul_sqrt2_pow(2 * pow2_real - 1);
        let e = if !sigma_real {
            if sigma_imag { 7 } else { 1 }
        } else if !sigma_imag {
            3
        } else {
            5
        };
        #[allow(clippy::cast_sign_loss)] // all terms non-negative, result in [0,7]
        let e = ((e + q_mod8) % 8) as u8;
        result.mul_phase(e);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    const EPS: f64 = 1e-10;

    /// Brute-force exponential sum for testing.
    fn brute_force_exp_sum(qf: &QuadraticForm) -> Complex64 {
        let n = qf.n;
        let dim = 1usize << n;
        let mut result = Complex64::new(0.0, 0.0);
        for x in 0..dim {
            let mut q_val = qf.q_const;
            for a in 0..n {
                if (x >> a) & 1 == 1 {
                    // D_a = 2*(2*D2_a + D1_a)
                    let d1_a = get_bit(&qf.d1, a);
                    let d2_a = get_bit(&qf.d2, a);
                    let d_a = 2 * (2 * i32::from(d2_a) + i32::from(d1_a));
                    q_val += d_a;
                    for b in (a + 1)..n {
                        if (x >> b) & 1 == 1 && get_bit(qf.j_row(a), b) {
                            q_val += 4;
                        }
                    }
                }
            }
            let angle = std::f64::consts::FRAC_PI_4 * f64::from(q_val % 8);
            result += Complex64::new(angle.cos(), angle.sin());
        }
        result
    }

    #[test]
    fn test_exponential_sum_zero_variables() {
        // n=0: sum over empty set = exp(i*pi/4 * Q)
        let mut qf = QuadraticForm::new(0);
        qf.q_const = 0;
        let result = qf.exponential_sum().to_complex();
        assert!((result - Complex64::new(1.0, 0.0)).norm() < EPS);

        qf.q_const = 2;
        let result = qf.exponential_sum().to_complex();
        let expected = Complex64::new(0.0, 1.0); // exp(i*pi/2) = i
        assert!((result - expected).norm() < EPS);
    }

    #[test]
    fn test_exponential_sum_one_variable() {
        // n=1, Q=0, D=0, J=0: sum = exp(0) + exp(0) = 2
        let qf = QuadraticForm::new(1);
        let result = qf.exponential_sum().to_complex();
        let expected = brute_force_exp_sum(&qf);
        assert!(
            (result - expected).norm() < EPS,
            "got {result:.4}, expected {expected:.4}"
        );
    }

    #[test]
    fn test_exponential_sum_vs_brute_force() {
        // Test several random quadratic forms against brute force.
        for n in 1..=5 {
            for seed in 0..10u64 {
                let mut qf = QuadraticForm::new(n);
                qf.q_const = ((seed * 3) % 8) as i32;
                for i in 0..n {
                    if ((seed >> i) & 1) == 1 {
                        set_bit(&mut qf.d1, i);
                    }
                    if ((seed >> (i + 3)) & 1) == 1 {
                        set_bit(&mut qf.d2, i);
                    }
                }
                for i in 0..n {
                    for j in (i + 1)..n {
                        if ((seed.wrapping_mul(7) >> (i + j)) & 1) == 1 {
                            set_bit(qf.j_row_mut(i), j);
                            set_bit(qf.j_row_mut(j), i);
                        }
                    }
                }

                let fast = qf.exponential_sum().to_complex();
                let brute = brute_force_exp_sum(&qf);
                assert!(
                    (fast - brute).norm() < EPS,
                    "n={n} seed={seed}: fast={fast:.6}, brute={brute:.6}, diff={:.2e}",
                    (fast - brute).norm()
                );
            }
        }
    }

    /// Test `ExponentialSum` at the u64/u128 boundary (n=62,63).
    #[test]
    fn test_exponential_sum_u64_u128_boundary() {
        for n in [62, 63] {
            let mut qf = QuadraticForm::new(n);
            qf.q_const = 3;
            // Set some bits near the boundary
            set_bit(&mut qf.d1, 0);
            set_bit(&mut qf.d1, n - 1);
            set_bit(&mut qf.d2, n / 2);
            if n > 1 {
                set_bit(qf.j_row_mut(0), n - 1);
                set_bit(qf.j_row_mut(n - 1), 0);
            }
            // Just verify it doesn't panic and returns a valid scalar
            let result = qf.exponential_sum();
            let c = result.to_complex();
            assert!(c.norm().is_finite(), "n={n}: result should be finite");
        }
    }

    /// Test `ExponentialSum` at the u128/Vec boundary (n=126,127).
    #[test]
    fn test_exponential_sum_u128_vec_boundary() {
        for n in [126, 127] {
            let mut qf = QuadraticForm::new(n);
            qf.q_const = 5;
            set_bit(&mut qf.d1, 0);
            set_bit(&mut qf.d1, n - 1);
            set_bit(&mut qf.d2, n / 2);
            set_bit(qf.j_row_mut(0), 1);
            set_bit(qf.j_row_mut(1), 0);
            let result = qf.exponential_sum();
            let c = result.to_complex();
            assert!(c.norm().is_finite(), "n={n}: result should be finite");
        }
    }

    /// Test `ExponentialSum` at large n (Vec<u64> path).
    #[test]
    fn test_exponential_sum_large_n() {
        let n = 150;
        let mut qf = QuadraticForm::new(n);
        qf.q_const = 2;
        set_bit(&mut qf.d1, 0);
        set_bit(&mut qf.d1, 149);
        set_bit(&mut qf.d2, 75);
        set_bit(qf.j_row_mut(0), 149);
        set_bit(qf.j_row_mut(149), 0);
        let result = qf.exponential_sum();
        let c = result.to_complex();
        assert!(c.norm().is_finite(), "n=150: result should be finite");
    }
}
