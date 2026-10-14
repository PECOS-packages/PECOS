// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Bravyi–Haah distillation (arXiv:1209.2426), from a validated triorthogonal matrix.
//!
//! Rows are independent; odd logical rows precede even syndrome rows. Circuit
//! construction uses the transversal-T lemma, with integer weight expansions
//! understood **modulo eight**. Oracles enumerate only the even-row span and its
//! logical cosets, exponentially in the number of even rows, not physical qubits.

use crate::binary::{Mask, rank};
use crate::{IncompatibleMeasurement, PhasePoly};
use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
use pecos_random::PecosRng;
use pecos_simulators::{MeasurementResult, QuantumSimulator};

mod circuit;
pub use circuit::{Op, run_shot, run_shot_sampled, to_stim};

#[cfg(test)]
mod tests;

/// Validated binary matrix with the first `k` rows odd and all remaining rows even.
#[derive(Clone, Debug)]
pub struct TriorthogonalMatrix {
    rows: Vec<Mask>,
    n: usize,
    k: usize,
}

impl TriorthogonalMatrix {
    /// Validate dimensions, row parity, pair/triple overlaps, and independence.
    ///
    /// # Errors
    /// Returns a description of the first failed condition. An empty row list
    /// represents the 0-by-0 matrix; zero logical or zero even rows are allowed.
    pub fn new(rows: &[Vec<bool>], k: usize) -> Result<Self, String> {
        let n = rows.first().map_or(0, Vec::len);
        if k > rows.len() {
            return Err("logical row count exceeds total row count".into());
        }
        let mut masks = Vec::with_capacity(rows.len());
        for (a, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(format!("row {a} has length {}, expected {n}", row.len()));
            }
            let mut mask = Mask::zero(n);
            for (j, &bit) in row.iter().enumerate() {
                if bit {
                    mask.toggle(j);
                }
            }
            if (mask.ones().count() % 2 == 1) != (a < k) {
                return Err(format!(
                    "row {a} must have {} weight",
                    if a < k { "odd" } else { "even" }
                ));
            }
            masks.push(mask);
        }
        for a in 0..masks.len() {
            for b in a + 1..masks.len() {
                if masks[a].dot(&masks[b]) {
                    return Err(format!("rows {a}, {b} have odd pair overlap"));
                }
                for c in b + 1..masks.len() {
                    if masks[a]
                        .ones()
                        .filter(|&j| masks[b].get(j) && masks[c].get(j))
                        .count()
                        % 2
                        == 1
                    {
                        return Err(format!("rows {a}, {b}, {c} have odd triple overlap"));
                    }
                }
            }
        }
        if rank(&masks, n) != masks.len() {
            return Err("matrix rows are linearly dependent over F2".into());
        }
        Ok(Self { rows: masks, n, k })
    }

    /// Number of physical qubits (columns).
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of logical qubits (odd rows).
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of independent rows.
    #[must_use]
    pub fn m(&self) -> usize {
        self.rows.len()
    }

    /// Original matrix in row order; each row contains bits in qubit order.
    #[must_use]
    pub fn rows(&self) -> Vec<Vec<bool>> {
        self.rows
            .iter()
            .map(|row| (0..self.n).map(|j| row.get(j)).collect())
            .collect()
    }

    /// Construct the paper's `G(k)`, of size `(k+3) × (3k+8)`.
    ///
    /// # Errors
    /// Requires even `k >= 2` and dimensions representable by `usize`.
    pub fn bravyi_haah(k: usize) -> Result<Self, String> {
        if k < 2 || !k.is_multiple_of(2) {
            return Err("G(k) requires even k >= 2".into());
        }
        let n = k
            .checked_mul(3)
            .and_then(|v| v.checked_add(8))
            .ok_or("G(k) column count overflows usize")?;
        let mut rows = vec![vec![false; n]; k + 3];
        for (a, row) in rows[..k].iter_mut().enumerate() {
            row[4..8].fill(true); // L: both rows are 1111.
            let start = 8 + 6 * (a / 2) + 3 * (a % 2);
            row[start..start + 3].fill(true); // M: 111000 / 000111.
        }
        let s1 = [
            [false, true, false, true],
            [false, false, true, true],
            [true; 4],
        ];
        let s2 = [
            [true, false, true, true, false, true],
            [false, true, true, false, true, true],
            [false; 6],
        ];
        for a in 0..3 {
            rows[k + a][..4].copy_from_slice(&s1[a]);
            rows[k + a][4..8].copy_from_slice(&s1[a]);
            for block in rows[k + a][8..].as_chunks_mut::<6>().0 {
                block.copy_from_slice(&s2[a]);
            }
        }
        Self::new(&rows, k)
    }

    /// The 15-to-1 matrix: all-ones logical row and four simplex rows.
    ///
    /// # Panics
    /// Panics only if the fixed construction fails triorthogonality validation.
    #[must_use]
    pub fn rm15() -> Self {
        let mut rows = vec![vec![true; 15]];
        for bit in 0..4 {
            rows.push((1..=15).map(|j| j & (1 << bit) != 0).collect());
        }
        Self::new(&rows, 1).expect("the simplex construction is triorthogonal")
    }

    /// Enumerate exact integer weight multiplicities for `G0` and each `G0 + f^a`.
    /// The latter is a coset, excluding `G0`; add the two enumerators for qdual.
    /// Runtime is exponential in `m-k`; memory is `O(kn + mn)`.
    ///
    /// # Errors
    /// Returns an error if a multiplicity cannot fit in `u64`.
    pub fn weight_enumerators(&self) -> Result<WeightEnumerators, String> {
        let mut weights = WeightEnumerators {
            even: vec![0; self.n + 1],
            cosets: vec![vec![0; self.n + 1]; self.k],
        };
        self.enumerate(0, &mut Mask::zero(self.n), &mut weights)?;
        Ok(weights)
    }

    fn enumerate(
        &self,
        depth: usize,
        word: &mut Mask,
        weights: &mut WeightEnumerators,
    ) -> Result<(), String> {
        if self.k + depth < self.m() {
            self.enumerate(depth + 1, word, weights)?;
            word.xor(&self.rows[self.k + depth]);
            self.enumerate(depth + 1, word, weights)?;
            word.xor(&self.rows[self.k + depth]);
        } else {
            increment(&mut weights.even[word.ones().count()])?;
            for (row, coset) in self.rows.iter().zip(&mut weights.cosets) {
                let weight = (0..self.n).filter(|&j| word.get(j) ^ row.get(j)).count();
                increment(&mut coset[weight])?;
            }
        }
        Ok(())
    }

    /// Exact per-pattern syndrome and logical signs from the subroutine analysis.
    ///
    /// # Errors
    /// Requires one error bit per column. Logical values are absent on rejection.
    pub fn pattern_oracle(&self, errors: &[bool]) -> Result<PatternPrediction, String> {
        if errors.len() != self.n {
            return Err("error pattern length must equal n".into());
        }
        let parity = |row: &Mask| row.ones().fold(false, |value, j| value ^ errors[j]);
        let syndrome: Vec<_> = self.rows[self.k..].iter().map(parity).collect();
        let accepted = !syndrome.iter().any(|&bit| bit);
        let logical_w = accepted.then(|| {
            self.rows[..self.k]
                .iter()
                .map(|row| if parity(row) { -1.0 } else { 1.0 })
                .collect()
        });
        Ok(PatternPrediction {
            syndrome,
            accepted,
            logical_w,
        })
    }
}

fn increment(value: &mut u64) -> Result<(), String> {
    *value = value
        .checked_add(1)
        .ok_or("weight multiplicity overflows u64")?;
    Ok(())
}

/// Integer coefficients indexed by Hamming weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightEnumerators {
    /// Weight enumerator of the even-row span.
    pub even: Vec<u64>,
    /// Weight enumerator of each logical coset, excluding the even-row span.
    pub cosets: Vec<Vec<u64>>,
}

impl WeightEnumerators {
    /// Evaluate paper equations `P_s` and qdual, using finite sums, not a series.
    /// Results have ordinary `f64` rounding (including cancellation near p=0).
    ///
    /// # Errors
    /// Requires finite `p` in `[0,1]` and nonempty, equally sized enumerators
    /// with a nonzero even enumerator and nonzero evaluated denominator.
    pub fn oracles(&self, p: f64) -> Result<Oracles, String> {
        validate_probability(p)?;
        if self.even.is_empty() || self.cosets.iter().any(|c| c.len() != self.even.len()) {
            return Err("weight enumerators must be nonempty and equally sized".into());
        }
        let evaluate = |coefficients: &[u64], x: f64| {
            coefficients
                .iter()
                .rev()
                .fold(0.0_f64, |sum, &c| sum.mul_add(x, count_as_f64(c)))
        };
        let x = 1.0 - 2.0 * p;
        let denominator = evaluate(&self.even, x);
        let size = evaluate(&self.even, 1.0);
        if size == 0.0 || denominator == 0.0 {
            return Err("oracle denominator is zero".into());
        }
        Ok(Oracles {
            p_s: denominator / size,
            q_a: self
                .cosets
                .iter()
                .map(|coset| 0.5 * (1.0 - evaluate(coset, x) / denominator))
                .collect(),
        })
    }
}

// Convert counts without a lossy integer cast; ordinary f64 rounding still applies.
fn count_as_f64(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).expect("high word fits u32");
    let low = u32::try_from(value & u64::from(u32::MAX)).expect("low word fits u32");
    f64::from(high).mul_add(4_294_967_296.0, f64::from(low))
}

fn validate_probability(p: f64) -> Result<(), String> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err("p must be finite and in [0,1]".into());
    }
    Ok(())
}

/// Exact finite-sum acceptance and conditional marginal error probabilities.
#[derive(Clone, Debug)]
pub struct Oracles {
    /// Acceptance probability.
    pub p_s: f64,
    /// Conditional error probability per logical output.
    pub q_a: Vec<f64>,
}

/// Algebraic prediction for one specified physical Z-error pattern.
#[derive(Clone, Debug)]
pub struct PatternPrediction {
    /// Syndrome bits, in original even-row order.
    pub syndrome: Vec<bool>,
    /// Whether the syndrome is zero.
    pub accepted: bool,
    /// Conditional logical W expectations, absent for rejected patterns.
    pub logical_w: Option<Vec<f64>>,
}

/// A simulated shot, retaining measurement determinism as well as outcomes.
pub struct ShotResult {
    /// Syndrome results, in circuit order; false denotes eigenvalue +1.
    pub syndrome: Vec<MeasurementResult>,
    /// Whether all syndrome outcomes were false.
    pub accepted: bool,
    /// Logical W expectations on acceptance, absent on rejection.
    pub logical_w: Option<Vec<f64>>,
}

fn logical_paulis(support: &[QubitId]) -> (PauliString, PauliString) {
    let x = PauliString::with_phase_and_paulis(
        QuarterPhase::PlusOne,
        support.iter().map(|&q| (Pauli::X, q)).collect(),
    );
    // Ybar = i Xbar Zbar = i (-i)^weight Y(support), for odd weight.
    let sign = if support.len() % 4 == 1 {
        QuarterPhase::PlusOne
    } else {
        QuarterPhase::MinusOne
    };
    let y =
        PauliString::with_phase_and_paulis(sign, support.iter().map(|&q| (Pauli::Y, q)).collect());
    (x, y)
}
