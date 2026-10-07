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

//! Compatible X measurements and affine Z measurements.

use crate::{
    PhasePoly,
    binary::{Mask, rank, solve},
    quadratic::{Clifford, root},
};
use pecos_core::QubitId;
use pecos_simulators::MeasurementResult;
use std::fmt;

/// Compatible cases of arXiv:2610.06811, Section 3, Theorem 3.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XMeasurementCase {
    /// The X translation lies outside the support; its dimension increases.
    Case1,
    /// The derivative is four times an affine Boolean function.
    /// Includes deterministic outcomes and the empty identity string.
    Case2a,
    /// The derivative is `2 + 4g`, with quadratic Boolean g of rank at most two.
    Case2b,
    /// The derivative is an odd constant times the sign of an affine parity.
    Case2c,
}

/// An X measurement whose nonzero branches do not all remain 3PP.
/// The diagnostic identifies the failed coefficient or rank condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncompatibleMeasurement {
    reason: String,
}

impl fmt::Display for IncompatibleMeasurement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason)
    }
}
impl std::error::Error for IncompatibleMeasurement {}

struct Plan {
    case: XMeasurementCase,
    derivative: Clifford,
    deterministic: Option<bool>,
}

impl Plan {
    fn probabilities(&self) -> [f64; 2] {
        if let Some(outcome) = self.deterministic {
            return if outcome { [0.0, 1.0] } else { [1.0, 0.0] };
        }
        if self.case == XMeasurementCase::Case2c {
            let cosine = root(self.derivative.k).re;
            [(1.0 + cosine) / 2.0, (1.0 - cosine) / 2.0]
        } else {
            [0.5, 0.5]
        }
    }
}

impl PhasePoly {
    fn z_form(&self, qubits: &[QubitId]) -> (Mask, bool) {
        self.validate(qubits.iter().copied());
        let mut mask = Mask::zero(self.x0.len());
        let mut constant = false;
        for q in qubits {
            mask.xor(&self.rows[q.0]);
            constant ^= self.x0[q.0];
        }
        (mask, constant)
    }

    /// Probabilities `[Pr(0), Pr(1)]` for a Z string, without mutation.
    /// Empty strings are identity, with probabilities `[1,0]`.
    #[must_use]
    pub fn z_probabilities(&self, qubits: &[QubitId]) -> [f64; 2] {
        let (mask, constant) = self.z_form(qubits);
        if mask.is_zero() {
            if constant { [0.0, 1.0] } else { [1.0, 0.0] }
        } else {
            [0.5, 0.5]
        }
    }

    pub(crate) fn measure_z(
        &mut self,
        qubits: &[QubitId],
        forced: Option<bool>,
    ) -> MeasurementResult {
        let (mask, constant) = self.z_form(qubits);
        let is_deterministic = mask.is_zero();
        let outcome = if is_deterministic {
            constant
        } else {
            forced.unwrap_or_else(|| self.rng.next_bool_fast())
        };
        if !is_deterministic {
            self.restrict(&mask, outcome ^ constant);
        }
        MeasurementResult {
            outcome,
            is_deterministic,
        }
    }

    /// Measure Z on distinct qubits in order, returning one result per qubit.
    pub fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.validate(qubits.iter().copied());
        qubits.iter().map(|&q| self.measure_z(&[q], None)).collect()
    }

    /// Measure the product of Z operators; false/true denotes eigenvalue +1/-1.
    pub fn mz_string(&mut self, qubits: &[QubitId]) -> MeasurementResult {
        self.measure_z(qubits, None)
    }

    /// Force a Z-string outcome only if nondeterministic, without consuming RNG.
    pub fn mz_string_forced(&mut self, qubits: &[QubitId], outcome: bool) -> MeasurementResult {
        self.measure_z(qubits, Some(outcome))
    }

    fn reduce_x(&mut self, qubits: &[QubitId]) {
        if let Some((a, rest)) = qubits.split_first() {
            for j in rest {
                self.cx_one(a.0, j.0);
            }
        }
    }

    /// Coefficient decision in the proposition of arXiv:2610.06811,
    /// X-measurements appendix, following Theorem 3.2.
    fn classify_x(&self, qubit: Option<QubitId>) -> Result<Plan, IncompatibleMeasurement> {
        let mut shift = vec![false; self.x0.len()];
        if let Some(q) = qubit {
            shift[q.0] = true;
        }
        let Some(t) = solve(&self.rows, &shift, self.r, self.x0.len()) else {
            return Ok(Plan {
                case: XMeasurementCase::Case1,
                derivative: Clifford::zero(0, self.x0.len()),
                deterministic: None,
            });
        };
        let derivative = self.derivative(&t);
        let Clifford { k, s, m } = &derivative;
        let case;
        let mut deterministic = None;
        let fail = |reason| Err(IncompatibleMeasurement { reason });
        if k % 2 == 0 {
            if let Some((i, coefficient)) = s.iter().enumerate().find(|(_, s)| *s % 2 != 0) {
                return fail(format!(
                    "Derivative constant {k} is even, but s[{i}]={coefficient} is odd; cases 2a/2b require even linear coefficients."
                ));
            }
            if k % 4 == 0 {
                if m.iter().any(|row| !row.is_zero()) {
                    return fail(format!(
                        "Derivative constant {k} selects case 2a, but its quadratic matrix is nonzero."
                    ));
                }
                case = XMeasurementCase::Case2a;
                if s.iter().all(|&s| s == 0) {
                    deterministic = Some(*k == 4);
                }
            } else {
                let matrix_rank = rank(m, self.r);
                if matrix_rank > 2 {
                    return fail(format!(
                        "Derivative constant {k} selects case 2b, but its quadratic matrix has rank {matrix_rank}, exceeding two."
                    ));
                }
                case = XMeasurementCase::Case2b;
            }
        } else {
            let required = (4 - k % 4) % 4;
            if let Some((i, coefficient)) = s
                .iter()
                .enumerate()
                .find(|(_, s)| **s != 0 && **s != required)
            {
                return fail(format!(
                    "Odd derivative constant {k} selects case 2c, but s[{i}]={coefficient} is neither zero nor {required}."
                ));
            }
            for i in 0..self.r {
                for j in i + 1..self.r {
                    if m[i].get(j) != (s[i] * s[j] % 2 != 0) {
                        return fail(format!(
                            "Odd derivative constant {k} selects case 2c, but M[{i},{j}] differs from s[{i}]*s[{j}] modulo two."
                        ));
                    }
                }
            }
            case = XMeasurementCase::Case2c;
        }
        Ok(Plan {
            case,
            derivative,
            deterministic,
        })
    }

    /// Non-mutating X-string probabilities `[Pr(0), Pr(1)]` and compatible case.
    /// Empty strings are deterministic identity, classified as case 2a.
    ///
    /// # Errors
    /// Returns the failed compatibility condition if any nonzero branch is not 3PP.
    pub fn x_probabilities(
        &self,
        qubits: &[QubitId],
    ) -> Result<([f64; 2], XMeasurementCase), IncompatibleMeasurement> {
        self.validate(qubits.iter().copied());
        let mut reduced = self.clone();
        reduced.reduce_x(qubits);
        let plan = reduced.classify_x(qubits.first().copied())?;
        Ok((plan.probabilities(), plan.case))
    }

    /// Measure a single X. This deliberately accepts one qubit, not a batch.
    ///
    /// # Errors
    /// Returns incompatibility without changing the state or RNG stream.
    pub fn mx(&mut self, qubit: QubitId) -> Result<MeasurementResult, IncompatibleMeasurement> {
        self.mx_string(&[qubit])
    }

    /// Force a single X outcome if nondeterministic; consumes no RNG.
    ///
    /// # Errors
    /// Returns incompatibility without changing the state or RNG stream.
    pub fn mx_forced(
        &mut self,
        qubit: QubitId,
        outcome: bool,
    ) -> Result<MeasurementResult, IncompatibleMeasurement> {
        self.mx_string_forced(&[qubit], outcome)
    }

    /// Measure one X string, using CX reduction and Theorem 3.2 of
    /// arXiv:2610.06811 (Section 3 and X-measurements appendix).
    ///
    /// # Errors
    /// Returns incompatibility without changing the state or RNG stream.
    pub fn mx_string(
        &mut self,
        qubits: &[QubitId],
    ) -> Result<MeasurementResult, IncompatibleMeasurement> {
        self.measure_x(qubits, None)
    }

    /// Force one X-string outcome if nondeterministic; consumes no RNG.
    ///
    /// # Errors
    /// Returns incompatibility without changing the state or RNG stream.
    pub fn mx_string_forced(
        &mut self,
        qubits: &[QubitId],
        outcome: bool,
    ) -> Result<MeasurementResult, IncompatibleMeasurement> {
        self.measure_x(qubits, Some(outcome))
    }

    fn measure_x(
        &mut self,
        qubits: &[QubitId],
        forced: Option<bool>,
    ) -> Result<MeasurementResult, IncompatibleMeasurement> {
        self.validate(qubits.iter().copied());
        let mut reduced = self.clone();
        reduced.reduce_x(qubits);
        let plan = reduced.classify_x(qubits.first().copied())?;
        let outcome = plan
            .deterministic
            .or(forced)
            .unwrap_or_else(|| reduced.rng.random_bool(plan.probabilities()[1]));
        let is_deterministic = plan.deterministic.is_some();
        if !is_deterministic {
            reduced.apply_x_plan(qubits[0], &plan, outcome);
        }
        reduced.reduce_x(qubits);
        *self = reduced;
        Ok(MeasurementResult {
            outcome,
            is_deterministic,
        })
    }

    fn linear_mask(&self, coefficients: &[u8], divisor: u8) -> Mask {
        let mut mask = Mask::zero(self.x0.len());
        for (i, &s) in coefficients.iter().enumerate() {
            if !(s / divisor).is_multiple_of(2) {
                mask.toggle(i);
            }
        }
        mask
    }

    fn apply_x_plan(&mut self, qubit: QubitId, plan: &Plan, outcome: bool) {
        let d = &plan.derivative;
        match plan.case {
            XMeasurementCase::Case1 => {
                self.rows[qubit.0].toggle(self.r);
                let mut mask = Mask::zero(self.x0.len());
                mask.toggle(self.r);
                self.add_term(mask, 4 * u8::from(outcome));
                self.r += 1;
            }
            XMeasurementCase::Case2a => {
                self.restrict(&self.linear_mask(&d.s, 2), outcome ^ (d.k == 4));
            }
            XMeasurementCase::Case2b => self.update_case_2b(d, outcome),
            XMeasurementCase::Case2c => {
                self.add_term(
                    self.linear_mask(&d.s, 1),
                    (8 + 4 * u8::from(outcome) - d.k) % 8,
                );
            }
        }
    }

    /// Rank-two factorization and `2g = 2l + 2uv - 4luv` from the
    /// efficient-simulation appendix of arXiv:2610.06811.
    fn update_case_2b(&mut self, d: &Clifford, outcome: bool) {
        let mut linear = self.linear_mask(&d.s, 2);
        let constant = d.k == 6;
        let coefficient = if outcome { 2 } else { 6 };
        if let Some((i, row)) = d.m.iter().enumerate().find(|(_, row)| !row.is_zero()) {
            let j = row.ones().next().expect("nonzero row");
            let u = d.m[i].clone();
            let v = d.m[j].clone();
            for bit in u.ones() {
                if v.get(bit) {
                    linear.toggle(bit);
                }
            }
            self.add_product(&[(u.clone(), false), (v.clone(), false)], coefficient);
            self.add_product(&[(linear.clone(), constant), (u, false), (v, false)], 4);
        }
        self.add_affine(linear, constant, coefficient);
    }
}
