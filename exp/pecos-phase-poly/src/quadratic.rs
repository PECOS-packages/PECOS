// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Weighted Clifford forms and exact quadratic Gauss sums.

use crate::binary::Mask;
use num_complex::Complex64;

/// `k + 2 sum s_i y_i + 4 sum_{i<j} m_ij y_i y_j` modulo eight.
#[derive(Clone, Debug)]
pub(crate) struct Clifford {
    pub(crate) k: u8,
    pub(crate) s: Vec<u8>,
    pub(crate) m: Vec<Mask>,
}

impl Clifford {
    pub(crate) fn zero(width: usize, capacity: usize) -> Self {
        Self {
            k: 0,
            s: vec![0; width],
            m: vec![Mask::zero(capacity); width],
        }
    }

    pub(crate) fn negate(&mut self) {
        self.k = (8 - self.k) % 8;
        for s in &mut self.s {
            *s = (4 - *s) % 4;
        }
    }

    /// Add an even multiple of an affine parity, retaining its constant.
    pub(crate) fn add_affine(&mut self, coefficient: u8, constant: bool, mask: &Mask) {
        let coefficient = if constant {
            self.k = (self.k + coefficient) % 8;
            (8 - coefficient) % 8
        } else {
            coefficient
        };
        let bits: Vec<_> = mask.ones().collect();
        for (position, &i) in bits.iter().enumerate() {
            self.s[i] = (self.s[i] + coefficient / 2) % 4;
            if coefficient % 4 != 0 {
                for &j in &bits[position + 1..] {
                    self.toggle_edge(i, j);
                }
            }
        }
    }

    fn toggle_edge(&mut self, i: usize, j: usize) {
        if i == j {
            self.s[i] = (self.s[i] + 2) % 4;
        } else {
            self.m[i].toggle(j);
            self.m[j].toggle(i);
        }
    }

    fn remove(&mut self, j: usize) {
        let width = self.s.len();
        self.s.remove(j);
        self.m.remove(j);
        for row in &mut self.m {
            row.remove(j, width);
        }
    }

    /// Substitute an affine constraint in weighted form, including constants.
    /// See arXiv:2610.06811, appendix on efficient simulation (Gauss sums).
    fn restrict(&mut self, mask: &Mask, constant: bool) {
        let j = mask.ones().next().expect("nonzero constraint");
        let mut form = mask.clone();
        form.toggle(j);
        let coefficient = 2 * self.s[j];
        self.s[j] = 0;
        let neighbors: Vec<_> = self.m[j].ones().collect();
        for &i in &neighbors {
            self.toggle_edge(i, j);
        }
        self.add_affine(coefficient, constant, &form);
        for i in neighbors {
            if constant {
                self.s[i] = (self.s[i] + 2) % 4;
            }
            for k in form.ones() {
                self.toggle_edge(i, k);
            }
        }
        self.remove(j);
    }

    /// Eliminate variables without enumerating assignments. The result stays
    /// an eighth root of unity times an integer power of sqrt(2), or zero.
    /// Implements the quadratic sum in arXiv:2610.06811, efficient-simulation
    /// appendix, equation `pauli-expectation-gauss-sum`.
    pub(crate) fn gauss_sum(mut self) -> Option<GaussSum> {
        let mut power = 0;
        while let Some(&s) = self.s.last() {
            let j = self.s.len() - 1;
            let mask = self.m[j].clone();
            self.remove(j);
            if s % 2 == 0 {
                power += 2;
                if mask.is_zero() {
                    if s == 2 {
                        return None;
                    }
                } else {
                    self.restrict(&mask, s == 2);
                }
            } else {
                power += 1;
                let sigma = if s == 1 { 1 } else { 7 };
                self.k = (self.k + sigma) % 8;
                self.add_affine((8 - 2 * sigma % 8) % 8, false, &mask);
            }
        }
        Some(GaussSum {
            phase: self.k,
            power,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GaussSum {
    pub(crate) phase: u8,
    pub(crate) power: i64,
}

impl GaussSum {
    pub(crate) fn value(self, normalization_power: i64) -> Complex64 {
        let power = self.power - normalization_power;
        let exponent = i32::try_from(power.div_euclid(2)).unwrap_or(if power < 0 {
            i32::MIN
        } else {
            i32::MAX
        });
        let magnitude = 2.0_f64.powi(exponent)
            * if power.rem_euclid(2) == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
            };
        root(self.phase) * magnitude
    }
}

pub(crate) fn root(phase: u8) -> Complex64 {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let (real, imag) = match phase % 8 {
        0 => (1.0, 0.0),
        1 => (h, h),
        2 => (0.0, 1.0),
        3 => (-h, h),
        4 => (-1.0, 0.0),
        5 => (-h, -h),
        6 => (0.0, -1.0),
        _ => (h, -h),
    };
    Complex64::new(real, imag)
}
