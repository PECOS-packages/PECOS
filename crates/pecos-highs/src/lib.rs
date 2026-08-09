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

//! Pure-Rust compatibility layer for the subset of the `highs` API used by mwpf.
//!
//! Covers exactly what mwpf's `float_lp` (non-`incr_lp`) path calls. The rest
//! of the vendored wrapper's surface (`ColProblem`, dual accessors, `try_*`
//! methods, `From<SolvedModel> for Model`) is intentionally omitted so that an
//! mwpf feature or revision bump that starts needing it fails loudly at
//! compile time instead of silently changing solver behavior.

use std::ops::{Bound, RangeBounds};

use pecos_lp::{ColumnData, Direction, LpOutcome, RowData};

/// Whether an objective is maximized or minimized.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// Maximize the objective.
    Maximise,
    /// Minimize the objective.
    Minimise,
}

/// A problem whose variables and constraints are added after optimization is selected.
#[derive(Default)]
pub struct RowProblem;

impl RowProblem {
    /// Creates a mutable model with the requested objective sense.
    #[must_use]
    pub const fn optimise(self, sense: Sense) -> Model {
        Model {
            sense,
            columns: Vec::new(),
            rows: Vec::new(),
        }
    }
}

/// An index identifying a model column.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Col(usize);

/// An index identifying a model row.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Row(usize);

/// A linear program under construction.
pub struct Model {
    sense: Sense,
    columns: Vec<ColumnData>,
    rows: Vec<RowData>,
}

impl Model {
    /// Accepts a `HiGHS` option for API compatibility.
    ///
    /// The pure-Rust solver is single-threaded and has no runtime options.
    pub fn set_option<S: Into<Vec<u8>>, V>(&mut self, _option: S, _value: V) {}

    /// Adds a variable and returns its index.
    ///
    /// # Panics
    ///
    /// Panics if a factor references a row that has not been added.
    pub fn add_col(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Col {
        let col = Col(self.columns.len());
        self.columns.push(ColumnData {
            objective: col_factor,
            lower: lower_bound(&bounds),
            upper: upper_bound(&bounds),
        });

        for (row, factor) in row_factors {
            let existing = self
                .rows
                .get_mut(row.0)
                .expect("column factor references an unknown row");
            existing.factors.push((col.0, factor));
        }
        col
    }

    /// Adds a constraint and returns its index.
    ///
    /// # Panics
    ///
    /// Panics if a factor references a column that has not been added.
    pub fn add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Row {
        let mut factors = Vec::new();
        for (col, factor) in row_factors {
            assert!(
                col.0 < self.columns.len(),
                "row factor references an unknown column"
            );
            factors.push((col.0, factor));
        }

        let row = Row(self.rows.len());
        self.rows.push(RowData {
            lower: lower_bound(&bounds),
            upper: upper_bound(&bounds),
            factors,
        });
        row
    }

    /// Solves the model.
    #[must_use]
    pub fn solve(self) -> SolvedModel {
        if self.columns.is_empty() {
            return SolvedModel {
                status: HighsModelStatus::ModelEmpty,
                solution: Solution {
                    columns: Vec::new(),
                },
            };
        }

        let direction = match self.sense {
            Sense::Maximise => Direction::Maximize,
            Sense::Minimise => Direction::Minimize,
        };
        let outcome = pecos_lp::solve(direction, &self.columns, &self.rows);
        match outcome {
            LpOutcome::Optimal(columns) => SolvedModel {
                status: HighsModelStatus::Optimal,
                solution: Solution { columns },
            },
            LpOutcome::Infeasible => SolvedModel {
                status: HighsModelStatus::Infeasible,
                solution: Solution {
                    columns: Vec::new(),
                },
            },
            LpOutcome::Unbounded => SolvedModel {
                status: HighsModelStatus::Unbounded,
                solution: Solution {
                    columns: Vec::new(),
                },
            },
            LpOutcome::InternalError => SolvedModel {
                status: HighsModelStatus::SolveError,
                solution: Solution {
                    columns: Vec::new(),
                },
            },
        }
    }
}

/// A solved linear program.
pub struct SolvedModel {
    status: HighsModelStatus,
    solution: Solution,
}

impl SolvedModel {
    /// Returns the model status.
    #[must_use]
    pub const fn status(&self) -> HighsModelStatus {
        self.status
    }

    /// Returns the primal solution.
    #[must_use]
    pub fn get_solution(&self) -> Solution {
        self.solution.clone()
    }
}

/// Primal variable values from a solved model.
#[derive(Clone)]
pub struct Solution {
    columns: Vec<f64>,
}

impl Solution {
    /// Returns variable values in column-addition order.
    #[must_use]
    pub fn columns(&self) -> &[f64] {
        &self.columns
    }
}

/// The result category produced by the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HighsModelStatus {
    /// The model has not been solved.
    NotSet,
    /// The solver failed internally.
    SolveError,
    /// The model contains no columns.
    ModelEmpty,
    /// No feasible solution exists.
    Infeasible,
    /// The solver could not distinguish unboundedness from infeasibility.
    /// Declared for API parity with `HiGHS` presolve; never produced here.
    UnboundedOrInfeasible,
    /// The objective is unbounded.
    Unbounded,
    /// An optimal solution was found.
    Optimal,
    /// The model status is not recognized.
    Unknown,
}

fn lower_bound(bounds: &impl RangeBounds<f64>) -> f64 {
    match bounds.start_bound() {
        Bound::Included(value) | Bound::Excluded(value) => *value,
        Bound::Unbounded => f64::NEG_INFINITY,
    }
}

fn upper_bound(bounds: &impl RangeBounds<f64>) -> f64 {
    match bounds.end_bound() {
        Bound::Included(value) | Bound::Excluded(value) => *value,
        Bound::Unbounded => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::{HighsModelStatus, Model, RowProblem, Sense};

    const TEST_EPSILON: f64 = 1e-8;

    fn assert_columns(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= TEST_EPSILON,
                "column {index}: expected {expected}, got {actual}"
            );
        }
    }

    fn objective(columns: &[f64], factors: &[f64]) -> f64 {
        columns
            .iter()
            .zip(factors)
            .map(|(value, factor)| value * factor)
            .sum()
    }

    #[test]
    fn solves_known_lp() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x = model.add_col(1.0, 0.0.., []);
        let y = model.add_col(1.0, 0.0.., []);
        model.add_row(..=2.0, [(x, 1.0)]);
        model.add_row(..=3.0, [(y, 1.0)]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert_columns(solution.columns(), &[2.0, 3.0]);
        assert!((objective(solution.columns(), &[1.0, 1.0]) - 5.0).abs() <= TEST_EPSILON);
    }

    #[test]
    fn solves_mwpf_shaped_lp() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x0 = model.add_col(1.0, 0.0.., []);
        let y0 = model.add_col(-1.0, 0.0.., []);
        let x1 = model.add_col(1.0, 0.0.., []);
        let y1 = model.add_col(-1.0, 0.0.., []);
        model.add_row(..=0.0, [(x0, -1.0), (y0, 1.0)]);
        model.add_row(..=1.0, [(x1, -1.0), (y1, 1.0)]);
        model.add_row(..=1.0, [(x0, 1.0), (y0, -1.0)]);
        model.add_row(..=2.0, [(x1, 1.0), (y1, -1.0)]);
        model.add_row(..=3.0, [(x0, 1.0), (y0, -1.0), (x1, 1.0), (y1, -1.0)]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert_columns(solution.columns(), &[1.0, 0.0, 2.0, 0.0]);
        assert!(
            (objective(solution.columns(), &[1.0, -1.0, 1.0, -1.0]) - 3.0).abs() <= TEST_EPSILON
        );
    }

    #[test]
    fn detects_infeasible_empty_row() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        model.add_col(0.0, 0.0.., []);
        model.add_row(..=-1.0, []);
        assert_eq!(model.solve().status(), HighsModelStatus::Infeasible);
    }

    #[test]
    fn detects_infeasible_phase_one() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x = model.add_col(0.0, 0.0.., []);
        model.add_row(..=1.0, [(x, 1.0)]);
        model.add_row(2.0.., [(x, 1.0)]);
        assert_eq!(model.solve().status(), HighsModelStatus::Infeasible);
    }

    #[test]
    fn detects_unbounded_objective() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        model.add_col(1.0, 0.0.., []);
        assert_eq!(model.solve().status(), HighsModelStatus::Unbounded);
    }

    #[test]
    fn minimizes() {
        let mut model = RowProblem.optimise(Sense::Minimise);
        model.add_col(1.0, 3.0.., []);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        assert_columns(solved.get_solution().columns(), &[3.0]);
    }

    #[test]
    fn supports_all_bound_forms_and_equality_rows() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let free = model.add_col(0.0, .., []);
        let _excluded_upper = model.add_col(1.0, ..2.0, []);
        let _included_upper = model.add_col(1.0, ..=3.0, []);
        let _lower = model.add_col(-1.0, 4.0.., []);
        model.add_row(1.0..=1.0, [(free, 1.0)]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        assert_columns(solved.get_solution().columns(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transforms_all_variable_bound_shapes() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let shifted = model.add_col(1.0, 2.0..=5.0, []);
        let negated = model.add_col(1.0, ..=-2.0, []);
        let free = model.add_col(-1.0, .., []);
        model.add_row(-4.0.., [(free, 1.0)]);
        model.add_row(..=5.0, [(shifted, 1.0)]);
        model.add_row(..=-2.0, [(negated, 1.0)]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        assert_columns(solved.get_solution().columns(), &[5.0, -2.0, -4.0]);
    }

    #[test]
    fn enforces_implicit_upper_bound_without_explicit_row() {
        // The variable's upper bound is the only thing capping the objective,
        // so this fails if the internal width row is dropped or relaxed.
        let mut model = RowProblem.optimise(Sense::Maximise);
        model.add_col(1.0, 2.0..=5.0, []);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        assert_columns(solved.get_solution().columns(), &[5.0]);
    }

    #[test]
    fn preserves_sub_epsilon_scale_coefficients() {
        // Optima far below any coarse zero-snapping threshold must survive;
        // this fails if the solver's epsilon hygiene grows beyond ~1e-8.
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x = model.add_col(1.0, 0.0.., []);
        model.add_row(..=5e-4, [(x, 1.0)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let columns = solved.get_solution().columns().to_vec();
        assert!(
            (columns[0] - 5e-4).abs() <= 1e-12,
            "expected 5e-4, got {}",
            columns[0]
        );
    }

    #[test]
    fn terminates_at_degenerate_vertex() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x = model.add_col(1.0, 0.0.., []);
        let y = model.add_col(1.0, 0.0.., []);
        model.add_row(..=1.0, [(x, 1.0)]);
        model.add_row(..=1.0, [(y, 1.0)]);
        model.add_row(..=2.0, [(x, 1.0), (y, 1.0)]);
        model.add_row(..=2.0, [(x, 1.0), (y, 1.0)]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert!((objective(solution.columns(), &[1.0, 1.0]) - 2.0).abs() <= TEST_EPSILON);
    }

    #[test]
    fn reports_empty_model() {
        let model = RowProblem.optimise(Sense::Maximise);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::ModelEmpty);
        assert!(solved.get_solution().columns().is_empty());
    }

    fn solve_model_built_by_rows() -> Vec<f64> {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x = model.add_col(1.0, 0.0.., []);
        model.add_row(..=2.0, [(x, 1.0)]);
        model.solve().get_solution().columns().to_vec()
    }

    fn solve_model_built_by_columns() -> Vec<f64> {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let row = model.add_row(..=2.0, []);
        model.add_col(1.0, 0.0.., [(row, 1.0)]);
        model.solve().get_solution().columns().to_vec()
    }

    #[test]
    fn add_col_row_factors_affect_solution() {
        let by_rows = solve_model_built_by_rows();
        let by_columns = solve_model_built_by_columns();
        assert_columns(&by_rows, &[2.0]);
        assert_eq!(by_rows, by_columns);
    }

    #[test]
    fn preserves_column_addition_order() {
        let mut model = RowProblem.optimise(Sense::Maximise);
        model.add_col(0.0, 3.0..=3.0, []);
        model.add_col(0.0, 1.0..=1.0, []);
        model.add_col(0.0, 4.0..=4.0, []);
        model.add_col(0.0, -2.0..=-2.0, []);
        assert_columns(
            model.solve().get_solution().columns(),
            &[3.0, 1.0, 4.0, -2.0],
        );
    }

    fn deterministic_model() -> Model {
        let mut model = RowProblem.optimise(Sense::Maximise);
        let x = model.add_col(2.0, 0.0.., []);
        let y = model.add_col(1.0, 0.0.., []);
        let z = model.add_col(3.0, 0.0.., []);
        model.add_row(..=4.0, [(x, 1.0), (y, 1.0)]);
        model.add_row(..=6.0, [(y, 1.0), (z, 2.0)]);
        model.add_row(..=5.0, [(x, 1.0), (z, 1.0)]);
        model
    }

    #[test]
    fn solving_is_byte_deterministic() {
        let first = deterministic_model()
            .solve()
            .get_solution()
            .columns()
            .to_vec();
        let second = deterministic_model()
            .solve()
            .get_solution()
            .columns()
            .to_vec();
        let first_bytes: Vec<_> = first.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let second_bytes: Vec<_> = second
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(first_bytes, second_bytes);
    }
}
