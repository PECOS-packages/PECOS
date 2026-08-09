// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Small dense linear-programming solver: two-phase tableau simplex with
//! Bland's rule.
//!
//! Built for the tiny LPs that arise in decoder relaxations (tens to ~200
//! columns). Deterministic by construction — fixed iteration order, no
//! randomization — so repeated solves are byte-identical. Every accepted
//! solution passes a final primal-feasibility audit; numerical trouble
//! surfaces as [`LpOutcome::InternalError`], never as a silently wrong
//! optimum.

/// Whether the objective is maximized or minimized.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Direction {
    /// Maximize the objective.
    Maximize,
    /// Minimize the objective.
    Minimize,
}

const EPSILON: f64 = 1e-9;

/// Minimum pivot magnitude. Dividing a row by a pivot near `EPSILON` amplifies
/// roundoff into order-one errors that silently corrupt the tableau, so the
/// ratio test only falls back below this floor when no better pivot exists.
const PIVOT_TOLERANCE: f64 = 1e-7;

/// Primal feasibility tolerance for the final solution audit, scaled by the
/// row bound's magnitude. Matches the order of `HiGHS`'s default (1e-7).
const FEASIBILITY_TOLERANCE: f64 = 1e-7;

/// A variable: objective coefficient and inclusive bounds (infinities allowed).
#[derive(Clone, Copy)]
pub struct ColumnData {
    /// Objective coefficient.
    pub objective: f64,
    /// Lower bound (`f64::NEG_INFINITY` for unbounded below).
    pub lower: f64,
    /// Upper bound (`f64::INFINITY` for unbounded above).
    pub upper: f64,
}

/// A constraint row: inclusive bounds on a sparse linear form over columns.
pub struct RowData {
    /// Lower bound (`f64::NEG_INFINITY` for unbounded below).
    pub lower: f64,
    /// Upper bound (`f64::INFINITY` for unbounded above).
    pub upper: f64,
    /// Sparse coefficients as `(column index, factor)` pairs.
    pub factors: Vec<(usize, f64)>,
}

/// Result of an LP solve.
pub enum LpOutcome {
    /// Optimal variable values in column order.
    Optimal(Vec<f64>),
    /// No feasible point exists.
    Infeasible,
    /// The objective improves without bound.
    Unbounded,
    /// Numerical failure (audit rejection or iteration cap); never a silently
    /// degraded solution.
    InternalError,
}

struct VariableTransform {
    offset: f64,
    terms: Vec<(usize, f64)>,
}

struct Inequality {
    coefficients: Vec<f64>,
    bound: f64,
}

struct StandardProblem {
    objective: Vec<f64>,
    inequalities: Vec<Inequality>,
    transforms: Vec<VariableTransform>,
}

enum PreprocessError {
    Infeasible,
    Internal,
}

struct Tableau {
    rows: Vec<Vec<f64>>,
    objective: Vec<f64>,
    basic: Vec<usize>,
    variables: usize,
}

enum SimplexResult {
    Optimal,
    Unbounded,
    IterationLimit,
}

/// Solve the LP defined by `columns` and `rows`.
#[must_use]
pub fn solve(direction: Direction, columns: &[ColumnData], rows: &[RowData]) -> LpOutcome {
    let problem = match preprocess(direction, columns, rows) {
        Ok(problem) => problem,
        Err(PreprocessError::Infeasible) => return LpOutcome::Infeasible,
        Err(PreprocessError::Internal) => return LpOutcome::InternalError,
    };

    solve_standard(problem)
}

fn preprocess(
    direction: Direction,
    columns: &[ColumnData],
    rows: &[RowData],
) -> Result<StandardProblem, PreprocessError> {
    let sign = if direction == Direction::Maximize {
        1.0
    } else {
        -1.0
    };
    let mut objective = Vec::new();
    let mut transforms = Vec::with_capacity(columns.len());
    let mut upper_bounds = Vec::new();

    for column in columns {
        validate_bounds(column.lower, column.upper)?;
        if !column.objective.is_finite() {
            return Err(PreprocessError::Internal);
        }
        let coefficient = sign * column.objective;

        if column.lower.is_finite() {
            let variable = objective.len();
            objective.push(coefficient);
            transforms.push(VariableTransform {
                offset: column.lower,
                terms: vec![(variable, 1.0)],
            });
            if column.upper.is_finite() {
                let width = column.upper - column.lower;
                if !width.is_finite() {
                    return Err(PreprocessError::Internal);
                }
                upper_bounds.push((variable, width));
            }
        } else if column.upper.is_finite() {
            let variable = objective.len();
            objective.push(-coefficient);
            transforms.push(VariableTransform {
                offset: column.upper,
                terms: vec![(variable, -1.0)],
            });
        } else {
            let positive = objective.len();
            objective.push(coefficient);
            let negative = objective.len();
            objective.push(-coefficient);
            transforms.push(VariableTransform {
                offset: 0.0,
                terms: vec![(positive, 1.0), (negative, -1.0)],
            });
        }
    }

    let variable_count = objective.len();
    let mut inequalities = Vec::new();
    for (variable, bound) in upper_bounds {
        let mut coefficients = vec![0.0; variable_count];
        coefficients[variable] = 1.0;
        inequalities.push(Inequality {
            coefficients,
            bound,
        });
    }

    for row in rows {
        validate_bounds(row.lower, row.upper)?;
        if row.factors.is_empty() {
            if zero_satisfies(row.lower, row.upper) {
                continue;
            }
            return Err(PreprocessError::Infeasible);
        }

        let mut coefficients = vec![0.0; variable_count];
        let mut offset = 0.0;
        for &(column, factor) in &row.factors {
            if column >= transforms.len() || !factor.is_finite() {
                return Err(PreprocessError::Internal);
            }
            let transform = &transforms[column];
            offset += factor * transform.offset;
            for &(variable, transform_factor) in &transform.terms {
                coefficients[variable] += factor * transform_factor;
            }
        }
        if !offset.is_finite() || coefficients.iter().any(|value| !value.is_finite()) {
            return Err(PreprocessError::Internal);
        }

        if row.upper.is_finite() {
            let bound = row.upper - offset;
            if !bound.is_finite() {
                return Err(PreprocessError::Internal);
            }
            inequalities.push(Inequality {
                coefficients: coefficients.clone(),
                bound,
            });
        }
        if row.lower.is_finite() {
            let bound = offset - row.lower;
            if !bound.is_finite() {
                return Err(PreprocessError::Internal);
            }
            inequalities.push(Inequality {
                coefficients: coefficients.into_iter().map(|value| -value).collect(),
                bound,
            });
        }
    }

    Ok(StandardProblem {
        objective,
        inequalities,
        transforms,
    })
}

fn validate_bounds(lower: f64, upper: f64) -> Result<(), PreprocessError> {
    if lower.is_nan() || upper.is_nan() {
        return Err(PreprocessError::Internal);
    }
    if (lower.is_infinite() && lower.is_sign_positive())
        || (upper.is_infinite() && upper.is_sign_negative())
    {
        return Err(PreprocessError::Infeasible);
    }
    // Exact: tolerating crossed bounds here would silently rewrite the LP
    // (clamped widths admit points above the declared upper bound).
    if lower > upper {
        return Err(PreprocessError::Infeasible);
    }
    Ok(())
}

// Exact on purpose: an empty row violated by any margin is a constraint the
// caller wrote; dropping it within a tolerance would rewrite the LP.
fn zero_satisfies(lower: f64, upper: f64) -> bool {
    lower <= 0.0 && upper >= 0.0
}

fn solve_standard(problem: StandardProblem) -> LpOutcome {
    let real_variables = problem.objective.len();
    let row_count = problem.inequalities.len();
    let non_artificial_variables = real_variables + row_count;
    // Any strictly negative RHS goes through phase 1. Treating tiny negatives
    // as zero here would rewrite the constraint; infeasibility-within-noise is
    // for the phase-1 verdict to decide, in exactly one place.
    let artificial_count = problem
        .inequalities
        .iter()
        .filter(|inequality| inequality.bound < 0.0)
        .count();
    let total_variables = non_artificial_variables + artificial_count;
    let mut rows = Vec::with_capacity(row_count);
    let mut basic = Vec::with_capacity(row_count);
    let mut next_artificial = non_artificial_variables;

    for (row_index, inequality) in problem.inequalities.iter().enumerate() {
        let mut row = vec![0.0; total_variables + 1];
        if inequality.bound < 0.0 {
            for (target, source) in row.iter_mut().zip(&inequality.coefficients) {
                *target = -*source;
            }
            row[real_variables + row_index] = -1.0;
            row[next_artificial] = 1.0;
            row[total_variables] = -inequality.bound;
            basic.push(next_artificial);
            next_artificial += 1;
        } else {
            row[..real_variables].copy_from_slice(&inequality.coefficients);
            row[real_variables + row_index] = 1.0;
            row[total_variables] = inequality.bound;
            basic.push(real_variables + row_index);
        }
        rows.push(row);
    }

    let mut tableau = Tableau {
        rows,
        objective: vec![0.0; total_variables + 1],
        basic,
        variables: total_variables,
    };
    // Observed pivot counts are ~7x the row count; 100x (rows + variables) is
    // generous headroom while still failing in well under a second if the
    // ratio test ever degenerates into a cycle.
    let iteration_cap = 100usize.saturating_mul(row_count.saturating_add(total_variables).max(1));
    let mut iterations = 0;

    if artificial_count > 0 {
        let mut phase_one_objective = vec![0.0; total_variables];
        for coefficient in &mut phase_one_objective[non_artificial_variables..] {
            *coefficient = -1.0;
        }
        tableau.set_objective(&phase_one_objective);
        match tableau.optimize(total_variables, &mut iterations, iteration_cap) {
            SimplexResult::Optimal => {}
            SimplexResult::Unbounded | SimplexResult::IterationLimit => {
                return LpOutcome::InternalError;
            }
        }

        for (row, &basic_variable) in tableau.rows.iter().zip(&tableau.basic) {
            if basic_variable >= non_artificial_variables && row[total_variables] > EPSILON {
                return LpOutcome::Infeasible;
            }
        }
        if !tableau.remove_artificial_variables(non_artificial_variables) {
            return LpOutcome::InternalError;
        }
    }

    let mut phase_two_objective = vec![0.0; tableau.variables];
    phase_two_objective[..real_variables].copy_from_slice(&problem.objective);
    tableau.set_objective(&phase_two_objective);
    match tableau.optimize(tableau.variables, &mut iterations, iteration_cap) {
        SimplexResult::Optimal => {}
        SimplexResult::Unbounded => return LpOutcome::Unbounded,
        SimplexResult::IterationLimit => return LpOutcome::InternalError,
    }

    // No epsilon-snapping of solution values: consumers may act on magnitudes
    // below any snapping threshold (mwpf branches on positivity at 1e-10), so
    // values are returned as computed. Only sign dust from the tableau's
    // nonnegativity invariant is clamped.
    let mut standard_values = vec![0.0; real_variables];
    for (row, &basic_variable) in tableau.rows.iter().zip(&tableau.basic) {
        if basic_variable < real_variables {
            let value = row[tableau.variables];
            if value < -EPSILON || !value.is_finite() {
                return LpOutcome::InternalError;
            }
            standard_values[basic_variable] = value.max(0.0);
        }
    }

    // Final primal-feasibility audit: substitute the recovered point back into
    // the original inequalities so any tableau corruption surfaces as a loud
    // error instead of a silently wrong "Optimal".
    for inequality in &problem.inequalities {
        let activity: f64 = inequality
            .coefficients
            .iter()
            .zip(&standard_values)
            .map(|(coefficient, value)| coefficient * value)
            .sum();
        // A NaN activity must fail here: `NaN > tolerance` is false, so the
        // comparison alone would wave a non-finite solution through.
        if !activity.is_finite()
            || activity - inequality.bound > FEASIBILITY_TOLERANCE * (1.0 + inequality.bound.abs())
        {
            return LpOutcome::InternalError;
        }
    }

    let mut original_values = Vec::with_capacity(problem.transforms.len());
    for transform in problem.transforms {
        let mut value = transform.offset;
        for (variable, factor) in transform.terms {
            value += factor * standard_values[variable];
        }
        if !value.is_finite() {
            return LpOutcome::InternalError;
        }
        original_values.push(value);
    }
    LpOutcome::Optimal(original_values)
}

impl Tableau {
    fn set_objective(&mut self, coefficients: &[f64]) {
        self.objective.fill(0.0);
        for (target, &coefficient) in self.objective.iter_mut().zip(coefficients) {
            *target = -coefficient;
        }

        for (row_index, &basic_variable) in self.basic.iter().enumerate() {
            let factor = self.objective[basic_variable];
            if factor != 0.0 {
                for column in 0..=self.variables {
                    self.objective[column] -= factor * self.rows[row_index][column];
                }
            }
        }
    }

    fn optimize(
        &mut self,
        eligible_variables: usize,
        iterations: &mut usize,
        iteration_cap: usize,
    ) -> SimplexResult {
        loop {
            let Some(entering) =
                (0..eligible_variables).find(|&variable| self.objective[variable] < -EPSILON)
            else {
                return SimplexResult::Optimal;
            };

            let Some(leaving) = self.leaving_row(entering) else {
                return SimplexResult::Unbounded;
            };
            if *iterations >= iteration_cap {
                return SimplexResult::IterationLimit;
            }
            *iterations += 1;
            self.pivot(leaving, entering);
        }
    }

    // The minimum ratio is taken over every pivot above EPSILON: skipping a
    // small-pivot row in favor of a larger pivot can skip the binding
    // constraint and change the answer, not just the numerics. Entries at or
    // below EPSILON are treated as zero coefficients — cancellation dust —
    // matching reference HiGHS, whose small_matrix_value default is also
    // 1e-9. Within an exact ratio tie, a pivot above PIVOT_TOLERANCE is
    // preferred so ill-conditioned divisions happen only when a small pivot
    // is truly binding; the final feasibility audit catches any corruption
    // those rare divisions cause.
    fn leaving_row(&self, entering: usize) -> Option<usize> {
        let mut best: Option<(usize, f64, f64, usize)> = None;
        for (row_index, row) in self.rows.iter().enumerate() {
            let pivot = row[entering];
            if !pivot.is_finite() || pivot <= EPSILON {
                continue;
            }
            let rhs = row[self.variables];
            debug_assert!(rhs >= -EPSILON, "tableau rhs went negative: {rhs}");
            let ratio = rhs.max(0.0) / pivot;
            if !ratio.is_finite() {
                continue;
            }
            let basic_variable = self.basic[row_index];
            // Exact minimum ratio; ties prefer safe pivots, then lowest basic
            // index (Bland's leaving rule). A fuzzy ratio comparison is not
            // transitive and would forfeit the anti-cycling guarantee.
            let replace = best.is_none_or(|(_, best_ratio, best_pivot, best_basic)| {
                match ratio.partial_cmp(&best_ratio) {
                    Some(std::cmp::Ordering::Less) => true,
                    Some(std::cmp::Ordering::Equal) => {
                        let candidate_safe = pivot > PIVOT_TOLERANCE;
                        let incumbent_safe = best_pivot > PIVOT_TOLERANCE;
                        (candidate_safe && !incumbent_safe)
                            || (candidate_safe == incumbent_safe && basic_variable < best_basic)
                    }
                    _ => false,
                }
            });
            if replace {
                best = Some((row_index, ratio, pivot, basic_variable));
            }
        }
        best.map(|(row_index, _, _, _)| row_index)
    }

    // No epsilon-snapping inside tableau arithmetic: zeroing an entry without a
    // compensating row operation shifts the row space, and a later division by
    // a small pivot amplifies that into an order-one error that every internal
    // consistency check still accepts. Elimination runs for every nonzero
    // factor for the same reason.
    fn pivot(&mut self, leaving: usize, entering: usize) {
        let pivot = self.rows[leaving][entering];
        for value in &mut self.rows[leaving] {
            *value /= pivot;
        }
        self.rows[leaving][entering] = 1.0;

        let pivot_row = self.rows[leaving].clone();
        for row_index in 0..self.rows.len() {
            if row_index == leaving {
                continue;
            }
            let factor = self.rows[row_index][entering];
            if factor != 0.0 {
                for (value, &pivot_value) in self.rows[row_index].iter_mut().zip(&pivot_row) {
                    *value -= factor * pivot_value;
                }
            }
            self.rows[row_index][entering] = 0.0;
        }

        let factor = self.objective[entering];
        if factor != 0.0 {
            for (value, &pivot_value) in self.objective.iter_mut().zip(&pivot_row) {
                *value -= factor * pivot_value;
            }
        }
        self.objective[entering] = 0.0;
        self.basic[leaving] = entering;
    }

    fn remove_artificial_variables(&mut self, non_artificial_variables: usize) -> bool {
        let mut redundant = vec![false; self.rows.len()];
        for (row_index, redundant_row) in redundant.iter_mut().enumerate() {
            if self.basic[row_index] < non_artificial_variables {
                continue;
            }

            let mut is_basic = vec![false; non_artificial_variables];
            for &variable in &self.basic {
                if variable < non_artificial_variables {
                    is_basic[variable] = true;
                }
            }
            // Pivot on the largest-magnitude eligible entry: dividing the row
            // by a near-epsilon coefficient here would amplify roundoff just
            // like a degenerate ratio-test pivot.
            let entering = (0..non_artificial_variables)
                .filter(|&variable| {
                    !is_basic[variable] && self.rows[row_index][variable].abs() > EPSILON
                })
                .max_by(|&a, &b| {
                    self.rows[row_index][a]
                        .abs()
                        .partial_cmp(&self.rows[row_index][b].abs())
                        .expect("tableau entries are finite")
                });
            if let Some(entering) = entering {
                self.pivot(row_index, entering);
            } else if self.rows[row_index][self.variables].abs() <= EPSILON {
                *redundant_row = true;
            } else {
                return false;
            }
        }

        let old_variables = self.variables;
        let mut new_rows = Vec::with_capacity(self.rows.len());
        let mut new_basic = Vec::with_capacity(self.basic.len());
        for (row_index, row) in self.rows.drain(..).enumerate() {
            if redundant[row_index] {
                continue;
            }
            let mut new_row = row[..non_artificial_variables].to_vec();
            new_row.push(row[old_variables]);
            new_rows.push(new_row);
            new_basic.push(self.basic[row_index]);
        }
        self.rows = new_rows;
        self.basic = new_basic;
        self.variables = non_artificial_variables;
        self.objective = vec![0.0; non_artificial_variables + 1];
        true
    }
}
