// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Solver-independent SAT and `MaxSAT` encodings of code- and fault-distance problems.
//!
//! The encoding choices follow the qLDPC distance study in
//! [arXiv:2606.12445](https://arxiv.org/abs/2606.12445): parity constraints use Tseitin XOR
//! chains, while weight bounds use a Sinz sequential counter. The module emits standard text
//! formats for external solvers and can also feed the internal clauses directly to batsat.
//!
//! Certification has an asymmetric trust boundary. A solver's SAT witness is checked here using
//! native GF(2) arithmetic, so the SAT half needs no solver trust. UNSAT answers cannot be checked
//! without a proof checker; exactness therefore rests on trusting every solver UNSAT answer below
//! the returned distance.

use crate::{DetectorErrorModel, ParityCheckMatrix, StabilizerCodeSpec};
use batsat::{BasicSolver, Lit, SolverInterface, Var, lbool};
use pecos_core::PauliOperator;
use pecos_quantum::F2Matrix;
use std::fmt::Write as _;
use thiserror::Error;

/// A binary distance problem: find a nonzero logical effect in the kernel of the checks.
///
/// Columns are candidate qubits or fault mechanisms. `H e = 0` enforces undetectability and
/// `L e != 0` enforces a nontrivial logical or observable effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistanceProblem {
    h: F2Matrix,
    l: F2Matrix,
    num_vars: usize,
}

/// Errors constructing a [`DistanceProblem`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DistanceProblemError {
    /// The check and logical matrices describe different numbers of variables.
    #[error("distance matrices have different widths: H has {h_width}, L has {l_width}")]
    MatrixWidthMismatch {
        /// Number of columns in the check matrix.
        h_width: usize,
        /// Number of columns in the logical matrix.
        l_width: usize,
    },
    /// A stabilizer or logical operator is not in the required CSS form.
    #[error("stabilizer code spec is not CSS: {component} {index} contains both X and Z support")]
    NonCssOperator {
        /// Collection containing the mixed operator.
        component: &'static str,
        /// Index within that collection.
        index: usize,
    },
    /// An operator addresses a qubit outside the spec width.
    #[error(
        "stabilizer code spec {component} {index} addresses qubit {qubit}, but the code has {num_qubits} qubits"
    )]
    QubitOutOfRange {
        /// Collection containing the invalid operator.
        component: &'static str,
        /// Index within that collection.
        index: usize,
        /// Invalid qubit index.
        qubit: usize,
        /// Declared number of qubits.
        num_qubits: usize,
    },
    /// A named logical collection has the wrong Pauli type for a CSS-form spec.
    #[error("stabilizer code spec is not CSS: {component} {index} has the wrong Pauli type")]
    WrongCssLogicalType {
        /// Logical collection with the wrong type.
        component: &'static str,
        /// Index within that collection.
        index: usize,
    },
}

/// A solver's answer for one bounded SAT decision problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolverAnswer {
    /// A satisfying assignment for the original problem variables (not DIMACS auxiliaries).
    Sat(Vec<bool>),
    /// The bounded problem is unsatisfiable.
    Unsat,
    /// The solver could not decide the bounded problem.
    Unknown,
}

/// A natively verified SAT witness plus the trusted UNSAT prefix establishing exactness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedDistance {
    /// Weight of the natively verified witness.
    pub distance: usize,
    /// Assignment of the original problem variables.
    pub witness: Vec<bool>,
    /// Always true: the returned SAT half was checked natively.
    pub sat_certified: bool,
    /// All smaller queried bounds were reported UNSAT by the solver.
    pub unsat_trusted_below: usize,
}

/// Reasons a proposed SAT witness fails native verification.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum WitnessError {
    /// The assignment does not contain exactly one bit per problem column.
    #[error("witness length is {actual}, expected {expected}")]
    LengthMismatch {
        /// Required assignment length.
        expected: usize,
        /// Supplied assignment length.
        actual: usize,
    },
    /// A check row has odd overlap with the assignment.
    #[error("witness violates H row {row}: overlap is odd")]
    OddCheck {
        /// Index of the failed check row.
        row: usize,
    },
    /// Every logical/observable row has even overlap with the assignment.
    #[error("witness has zero logical effect: L e = 0")]
    ZeroLogicalEffect,
}

/// Errors in the incremental distance-certification loop.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DistanceCertificationError {
    /// The solver returned an assignment rejected by native verification.
    #[error("solver returned invalid witness at weight {weight}: {reason}")]
    InvalidWitness {
        /// Bound at which the solver returned SAT.
        weight: usize,
        /// Native verification failure.
        #[source]
        reason: WitnessError,
    },
    /// The assignment is valid but exceeds the bound used for that SAT call.
    #[error(
        "solver returned invalid witness at weight {weight}: verified weight {actual} exceeds the bound"
    )]
    WitnessExceedsBound {
        /// Bound at which the solver returned SAT.
        weight: usize,
        /// Native weight of the assignment.
        actual: usize,
    },
    /// The solver stopped without deciding a bound.
    #[error("solver returned unknown at weight {weight}")]
    Unknown {
        /// First undecided weight.
        weight: usize,
    },
}

#[derive(Clone, Debug)]
struct ClauseGroup {
    description: &'static str,
    clauses: Vec<Vec<i32>>,
}

#[derive(Clone, Debug)]
struct Encoding {
    num_vars: usize,
    aux_ranges: Vec<(usize, usize, &'static str)>,
    groups: Vec<ClauseGroup>,
}

#[derive(Clone, Copy, Debug)]
enum CounterOutput {
    FixedBound,
    Assumptions,
}

#[derive(Debug, Default)]
struct FinalCounterRow {
    /// Positive literals for s[n,1], s[n,2], ... in threshold order.
    threshold_literals: Vec<i32>,
}

impl FinalCounterRow {
    fn assumption_for_bound(&self, max_weight: usize) -> Option<i32> {
        // Index max_weight is s[n,max_weight+1], meaning the count exceeds max_weight.
        self.threshold_literals
            .get(max_weight)
            .map(|&literal| -literal)
    }
}

#[derive(Debug)]
struct EncodingBuilder {
    next_var: usize,
    aux_ranges: Vec<(usize, usize, &'static str)>,
    parity_clauses: Vec<Vec<i32>>,
    nontriviality_clauses: Vec<Vec<i32>>,
    cardinality_clauses: Vec<Vec<i32>>,
}

impl EncodingBuilder {
    fn new(num_primary_vars: usize) -> Self {
        Self {
            next_var: num_primary_vars + 1,
            aux_ranges: Vec::new(),
            parity_clauses: Vec::new(),
            nontriviality_clauses: Vec::new(),
            cardinality_clauses: Vec::new(),
        }
    }

    fn allocate_range(&mut self, count: usize, role: &'static str) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let start = self.next_var;
        let end = start + count - 1;
        self.next_var = end + 1;
        self.aux_ranges.push((start, end, role));
        (start..=end).collect()
    }

    fn finish(self, include_cardinality: bool) -> Encoding {
        let mut groups = vec![ClauseGroup {
            description: "parity constraints (Tseitin XOR chains)",
            clauses: self.parity_clauses,
        }];
        groups.push(ClauseGroup {
            description: "logical nontriviality",
            clauses: self.nontriviality_clauses,
        });
        if include_cardinality {
            groups.push(ClauseGroup {
                description: "weight bound (Sinz sequential counter)",
                clauses: self.cardinality_clauses,
            });
        }
        Encoding {
            num_vars: self.next_var - 1,
            aux_ranges: self.aux_ranges,
            groups,
        }
    }
}

impl DistanceProblem {
    /// Constructs a problem from check and logical matrices with matching widths.
    ///
    /// # Errors
    ///
    /// Returns [`DistanceProblemError::MatrixWidthMismatch`] when the matrices have different
    /// numbers of columns.
    pub fn from_css_checks(
        h: &ParityCheckMatrix,
        l: &ParityCheckMatrix,
    ) -> Result<Self, DistanceProblemError> {
        if h.num_qubits() != l.num_qubits() {
            return Err(DistanceProblemError::MatrixWidthMismatch {
                h_width: h.num_qubits(),
                l_width: l.num_qubits(),
            });
        }
        Ok(Self {
            h: h.matrix().clone(),
            l: l.matrix().clone(),
            num_vars: h.num_qubits(),
        })
    }

    /// Constructs the pure-X distance problem for a CSS-form stabilizer code spec.
    ///
    /// Z stabilizers form `H` and logical Z operators form `L`, since those are the operators
    /// whose overlap detects a pure-X support vector.
    ///
    /// # Errors
    ///
    /// Returns an error instead of projecting a non-CSS or incorrectly typed spec.
    pub fn from_css_code_x_distance(
        code: &StabilizerCodeSpec,
    ) -> Result<Self, DistanceProblemError> {
        Self::from_css_code(code, false)
    }

    /// Constructs the pure-Z distance problem for a CSS-form stabilizer code spec.
    ///
    /// X stabilizers form `H` and logical X operators form `L`, since those are the operators
    /// whose overlap detects a pure-Z support vector.
    ///
    /// # Errors
    ///
    /// Returns an error instead of projecting a non-CSS or incorrectly typed spec.
    pub fn from_css_code_z_distance(
        code: &StabilizerCodeSpec,
    ) -> Result<Self, DistanceProblemError> {
        Self::from_css_code(code, true)
    }

    fn from_css_code(
        code: &StabilizerCodeSpec,
        use_x_operators: bool,
    ) -> Result<Self, DistanceProblemError> {
        let num_qubits = code.num_qubits();
        let mut x_checks = Vec::new();
        let mut z_checks = Vec::new();
        for (index, operator) in code.stabilizers().iter().enumerate() {
            let x = operator.x_positions();
            let z = operator.z_positions();
            Self::validate_positions(&x, "stabilizer", index, num_qubits)?;
            Self::validate_positions(&z, "stabilizer", index, num_qubits)?;
            if !x.is_empty() && !z.is_empty() {
                return Err(DistanceProblemError::NonCssOperator {
                    component: "stabilizer",
                    index,
                });
            }
            if !x.is_empty() {
                x_checks.push(Self::support_row(num_qubits, &x));
            } else if !z.is_empty() {
                z_checks.push(Self::support_row(num_qubits, &z));
            }
        }

        let logical_x = Self::css_logical_rows(code.logical_xs(), "logical X", true, num_qubits)?;
        let logical_z = Self::css_logical_rows(code.logical_zs(), "logical Z", false, num_qubits)?;
        let (checks, logicals) = if use_x_operators {
            (x_checks, logical_x)
        } else {
            (z_checks, logical_z)
        };
        Ok(Self {
            h: Self::matrix_from_rows(checks, num_qubits),
            l: Self::matrix_from_rows(logicals, num_qubits),
            num_vars: num_qubits,
        })
    }

    fn css_logical_rows(
        operators: &[pecos_core::PauliString],
        component: &'static str,
        expect_x: bool,
        num_qubits: usize,
    ) -> Result<Vec<Vec<u8>>, DistanceProblemError> {
        operators
            .iter()
            .enumerate()
            .map(|(index, operator)| {
                let x = operator.x_positions();
                let z = operator.z_positions();
                Self::validate_positions(&x, component, index, num_qubits)?;
                Self::validate_positions(&z, component, index, num_qubits)?;
                if !x.is_empty() && !z.is_empty() {
                    return Err(DistanceProblemError::NonCssOperator { component, index });
                }
                let wrong_type = if expect_x {
                    !z.is_empty()
                } else {
                    !x.is_empty()
                };
                if wrong_type {
                    return Err(DistanceProblemError::WrongCssLogicalType { component, index });
                }
                Ok(Self::support_row(
                    num_qubits,
                    if expect_x { &x } else { &z },
                ))
            })
            .collect()
    }

    fn validate_positions(
        positions: &[usize],
        component: &'static str,
        index: usize,
        num_qubits: usize,
    ) -> Result<(), DistanceProblemError> {
        if let Some(&qubit) = positions.iter().find(|&&qubit| qubit >= num_qubits) {
            return Err(DistanceProblemError::QubitOutOfRange {
                component,
                index,
                qubit,
                num_qubits,
            });
        }
        Ok(())
    }

    fn support_row(num_qubits: usize, support: &[usize]) -> Vec<u8> {
        let mut row = vec![0; num_qubits];
        for &column in support {
            row[column] = 1;
        }
        row
    }

    fn matrix_from_rows(rows: Vec<Vec<u8>>, num_cols: usize) -> F2Matrix {
        if rows.is_empty() {
            F2Matrix::zeros(0, num_cols)
        } else {
            F2Matrix::from_rows(rows)
        }
    }

    /// Constructs a fault-distance problem from a detector error model.
    ///
    /// Columns follow [`DetectorErrorModel::to_mechanisms`] order. Detector incidence forms `H`
    /// and observable incidence forms `L`; mechanism probabilities are deliberately ignored.
    #[must_use]
    pub fn from_dem(dem: &DetectorErrorModel) -> Self {
        let (mechanisms, _coordinates) = dem.to_mechanisms();
        let num_vars = mechanisms.len();
        let detector_rows = mechanisms
            .iter()
            .flat_map(|(_, detectors, _)| detectors)
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0)
            .max(dem.num_detectors());
        let logical_rows = mechanisms
            .iter()
            .flat_map(|(_, _, observables)| observables)
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0)
            .max(dem.num_observables());
        let mut h = F2Matrix::zeros(detector_rows, num_vars);
        let mut l = F2Matrix::zeros(logical_rows, num_vars);
        for (column, (_, detectors, observables)) in mechanisms.iter().enumerate() {
            for &detector in detectors {
                h.set(detector as usize, column, 1);
            }
            for &observable in observables {
                l.set(observable as usize, column, 1);
            }
        }
        Self { h, l, num_vars }
    }

    /// Returns the number of original decision variables.
    #[must_use]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Emits the DIMACS CNF decision problem with `|e| <= max_weight`.
    ///
    /// Comment lines identify primary and auxiliary ranges and separate clause groups for audit.
    #[must_use]
    pub fn to_dimacs(&self, max_weight: usize) -> String {
        let encoding = self.encode(Some(max_weight));
        let clause_count: usize = encoding
            .groups
            .iter()
            .map(|group| group.clauses.len())
            .sum();
        let mut output = String::new();
        writeln!(output, "c PECOS distance decision encoding")
            .expect("writing to String cannot fail");
        Self::write_range_comment(&mut output, 1, self.num_vars, "primary support variables");
        for &(start, end, role) in &encoding.aux_ranges {
            Self::write_range_comment(&mut output, start, end, role);
        }
        writeln!(output, "p cnf {} {clause_count}", encoding.num_vars)
            .expect("writing to String cannot fail");
        for group in &encoding.groups {
            writeln!(output, "c clause group: {}", group.description)
                .expect("writing to String cannot fail");
            Self::write_clauses(&mut output, &group.clauses, None);
        }
        output
    }

    /// Emits a new-format WCNF `MaxSAT` problem minimizing `|e|`.
    ///
    /// Hard clauses enforce parity and nontriviality. There is one weight-1 soft clause `not x_i`
    /// per original variable. The output uses the modern `h` hard-clause prefix rather than a
    /// synthetic top weight, keeping hard/soft intent explicit and avoiding top-weight overflow.
    #[must_use]
    pub fn to_wcnf(&self) -> String {
        let encoding = self.encode(None);
        let hard_count: usize = encoding
            .groups
            .iter()
            .map(|group| group.clauses.len())
            .sum();
        let clause_count = hard_count + self.num_vars;
        let mut output = String::new();
        writeln!(
            output,
            "c new-format WCNF: hard clauses use the h prefix (no synthetic top weight)"
        )
        .expect("writing to String cannot fail");
        Self::write_range_comment(&mut output, 1, self.num_vars, "primary support variables");
        for &(start, end, role) in &encoding.aux_ranges {
            Self::write_range_comment(&mut output, start, end, role);
        }
        writeln!(output, "p wcnf {} {clause_count}", encoding.num_vars)
            .expect("writing to String cannot fail");
        for group in &encoding.groups {
            writeln!(output, "c hard clause group: {}", group.description)
                .expect("writing to String cannot fail");
            Self::write_clauses(&mut output, &group.clauses, Some("h"));
        }
        writeln!(
            output,
            "c soft clause group: unit penalties for selected variables"
        )
        .expect("writing to String cannot fail");
        for variable in 1..=self.num_vars {
            writeln!(output, "1 -{variable} 0").expect("writing to String cannot fail");
        }
        output
    }

    fn write_range_comment(output: &mut String, start: usize, end: usize, role: &str) {
        if start <= end {
            writeln!(output, "c variables {start}..{end}: {role}")
                .expect("writing to String cannot fail");
        } else {
            writeln!(output, "c variables none: {role}").expect("writing to String cannot fail");
        }
    }

    fn write_clauses(output: &mut String, clauses: &[Vec<i32>], prefix: Option<&str>) {
        for clause in clauses {
            if let Some(prefix) = prefix {
                write!(output, "{prefix} ").expect("writing to String cannot fail");
            }
            for literal in clause {
                write!(output, "{literal} ").expect("writing to String cannot fail");
            }
            writeln!(output, "0").expect("writing to String cannot fail");
        }
    }

    fn encode(&self, max_weight: Option<usize>) -> Encoding {
        let mut builder = EncodingBuilder::new(self.num_vars);
        self.encode_checks(&mut builder);
        self.encode_logical_nontriviality(&mut builder);
        if let Some(max_weight) = max_weight {
            self.encode_sequential_counter(&mut builder, max_weight, CounterOutput::FixedBound);
        }
        builder.finish(max_weight.is_some())
    }

    fn encode_for_assumptions(&self, max_weight: usize) -> (Encoding, FinalCounterRow) {
        let mut builder = EncodingBuilder::new(self.num_vars);
        self.encode_checks(&mut builder);
        self.encode_logical_nontriviality(&mut builder);
        let final_row =
            self.encode_sequential_counter(&mut builder, max_weight, CounterOutput::Assumptions);
        (builder.finish(true), final_row)
    }

    fn row_support(matrix: &F2Matrix, row: usize) -> Vec<usize> {
        (0..matrix.num_cols())
            .filter(|&column| matrix.get(row, column) == 1)
            .map(|column| column + 1)
            .collect()
    }

    fn literal(variable: usize) -> i32 {
        i32::try_from(variable).expect("DIMACS variable count exceeds i32::MAX")
    }

    fn encode_checks(&self, builder: &mut EncodingBuilder) {
        let aux_count: usize = (0..self.h.num_rows())
            .map(|row| Self::row_support(&self.h, row).len().saturating_sub(1))
            .sum();
        let auxiliaries = builder.allocate_range(aux_count, "H-row XOR-chain auxiliaries");
        let mut aux_iter = auxiliaries.into_iter();
        for row in 0..self.h.num_rows() {
            let support = Self::row_support(&self.h, row);
            match support.as_slice() {
                [] => {}
                &[variable] => builder.parity_clauses.push(vec![-Self::literal(variable)]),
                &[first, second, ref rest @ ..] => {
                    let mut output = aux_iter.next().expect("pre-counted XOR auxiliary");
                    Self::push_xor(&mut builder.parity_clauses, first, second, output);
                    for &variable in rest {
                        let next = aux_iter.next().expect("pre-counted XOR auxiliary");
                        Self::push_xor(&mut builder.parity_clauses, output, variable, next);
                        output = next;
                    }
                    builder.parity_clauses.push(vec![-Self::literal(output)]);
                }
            }
        }
    }

    fn encode_logical_nontriviality(&self, builder: &mut EncodingBuilder) {
        let outputs = builder.allocate_range(self.l.num_rows(), "logical XOR outputs y_j");
        let intermediate_count: usize = (0..self.l.num_rows())
            .map(|row| Self::row_support(&self.l, row).len().saturating_sub(2))
            .sum();
        let intermediates =
            builder.allocate_range(intermediate_count, "logical XOR-chain intermediates");
        let mut intermediate_iter = intermediates.into_iter();
        for (row, &output) in outputs.iter().enumerate() {
            let support = Self::row_support(&self.l, row);
            match support.as_slice() {
                [] => builder.parity_clauses.push(vec![-Self::literal(output)]),
                &[variable] => {
                    builder
                        .parity_clauses
                        .push(vec![-Self::literal(variable), Self::literal(output)]);
                    builder
                        .parity_clauses
                        .push(vec![Self::literal(variable), -Self::literal(output)]);
                }
                &[first, second] => {
                    Self::push_xor(&mut builder.parity_clauses, first, second, output);
                }
                &[first, second, ref rest @ ..] => {
                    let mut previous = intermediate_iter
                        .next()
                        .expect("pre-counted logical XOR auxiliary");
                    Self::push_xor(&mut builder.parity_clauses, first, second, previous);
                    for (offset, &variable) in rest.iter().enumerate() {
                        let next = if offset + 1 == rest.len() {
                            output
                        } else {
                            intermediate_iter
                                .next()
                                .expect("pre-counted logical XOR auxiliary")
                        };
                        Self::push_xor(&mut builder.parity_clauses, previous, variable, next);
                        previous = next;
                    }
                }
            }
        }
        builder
            .nontriviality_clauses
            .push(outputs.into_iter().map(Self::literal).collect());
    }

    fn push_xor(clauses: &mut Vec<Vec<i32>>, left: usize, right: usize, output: usize) {
        let left = Self::literal(left);
        let right = Self::literal(right);
        let output = Self::literal(output);
        clauses.push(vec![left, right, -output]);
        clauses.push(vec![-left, -right, -output]);
        clauses.push(vec![left, -right, output]);
        clauses.push(vec![-left, right, output]);
    }

    fn encode_sequential_counter(
        &self,
        builder: &mut EncodingBuilder,
        max_weight: usize,
        output: CounterOutput,
    ) -> FinalCounterRow {
        match output {
            CounterOutput::FixedBound if max_weight >= self.num_vars => {
                return FinalCounterRow::default();
            }
            CounterOutput::FixedBound if max_weight == 0 => {
                for variable in 1..=self.num_vars {
                    builder
                        .cardinality_clauses
                        .push(vec![-Self::literal(variable)]);
                }
                return FinalCounterRow::default();
            }
            CounterOutput::Assumptions if max_weight == 0 || self.num_vars <= 1 => {
                return FinalCounterRow::default();
            }
            CounterOutput::FixedBound | CounterOutput::Assumptions => {}
        }

        // s[i, j] means that at least j of x_1..=x_i are true. These are the sequential unary
        // counter variables of Sinz (2005). Both directions of each recurrence are emitted, making
        // every auxiliary functionally determined by the primary variables. Assumption mode needs
        // s[n,w+1] for every queried w, including max_weight, but thresholds above n are impossible.
        let threshold = max_weight.saturating_add(1).min(self.num_vars);
        let variables = builder.allocate_range(
            self.num_vars * threshold,
            "Sinz sequential-counter prefix thresholds",
        );
        let counter = |i: usize, j: usize| variables[(i - 1) * threshold + (j - 1)];

        // First prefix: s[1,1] <-> x_1; all unreachable higher thresholds are false.
        let first = counter(1, 1);
        builder
            .cardinality_clauses
            .push(vec![-1, Self::literal(first)]);
        builder
            .cardinality_clauses
            .push(vec![1, -Self::literal(first)]);
        for j in 2..=threshold {
            builder
                .cardinality_clauses
                .push(vec![-Self::literal(counter(1, j))]);
        }

        for i in 2..=self.num_vars {
            let x = Self::literal(i);
            // s[i,1] <-> (s[i-1,1] OR x_i).
            let previous = Self::literal(counter(i - 1, 1));
            let current = Self::literal(counter(i, 1));
            builder.cardinality_clauses.push(vec![-previous, current]);
            builder.cardinality_clauses.push(vec![-x, current]);
            builder
                .cardinality_clauses
                .push(vec![-current, previous, x]);

            for j in 2..=threshold {
                // s[i,j] <-> (s[i-1,j] OR (x_i AND s[i-1,j-1])).
                let same = Self::literal(counter(i - 1, j));
                let lower = Self::literal(counter(i - 1, j - 1));
                let current = Self::literal(counter(i, j));
                builder.cardinality_clauses.push(vec![-same, current]);
                builder.cardinality_clauses.push(vec![-x, -lower, current]);
                builder.cardinality_clauses.push(vec![-current, same, x]);
                builder
                    .cardinality_clauses
                    .push(vec![-current, same, lower]);
            }
        }
        let final_row = FinalCounterRow {
            threshold_literals: (1..=threshold)
                .map(|j| Self::literal(counter(self.num_vars, j)))
                .collect(),
        };
        if matches!(output, CounterOutput::FixedBound) {
            builder
                .cardinality_clauses
                .push(vec![-final_row.threshold_literals[threshold - 1]]);
        }
        final_row
    }

    /// Checks a candidate witness with native GF(2) arithmetic and returns its Hamming weight.
    ///
    /// This check does not trust the SAT solver: it independently verifies every `H` row and the
    /// complete `L e != 0` predicate.
    ///
    /// # Errors
    ///
    /// Returns a length error, identifies the first odd `H` row, or reports that `L e` is zero.
    pub fn verify_witness(&self, assignment: &[bool]) -> Result<usize, WitnessError> {
        if assignment.len() != self.num_vars {
            return Err(WitnessError::LengthMismatch {
                expected: self.num_vars,
                actual: assignment.len(),
            });
        }
        for row in 0..self.h.num_rows() {
            let odd = assignment
                .iter()
                .enumerate()
                .filter(|&(column, selected)| *selected && self.h.get(row, column) == 1)
                .count()
                % 2
                == 1;
            if odd {
                return Err(WitnessError::OddCheck { row });
            }
        }
        let logical_nonzero = (0..self.l.num_rows()).any(|row| {
            assignment
                .iter()
                .enumerate()
                .filter(|&(column, selected)| *selected && self.l.get(row, column) == 1)
                .count()
                % 2
                == 1
        });
        if !logical_nonzero {
            return Err(WitnessError::ZeroLogicalEffect);
        }
        Ok(assignment.iter().filter(|&&selected| selected).count())
    }

    /// Incrementally certifies distance through `max_weight` using a pluggable SAT solver.
    ///
    /// The solver is called once per bound from 1 upward. Its SAT assignment must contain only
    /// the original variables and is checked natively, so SAT soundness does not rely on the
    /// solver. Every preceding UNSAT result is trusted; that trust is what turns the verified upper
    /// bound into an exact distance. `Ok(None)` means all bounds through `max_weight` were reported
    /// UNSAT, establishing only a solver-trusted lower bound greater than `max_weight`.
    ///
    /// # Errors
    ///
    /// Returns immediately for an invalid or overweight SAT witness or an `Unknown` answer.
    pub fn certify_distance_with<S>(
        &self,
        max_weight: usize,
        mut solver: S,
    ) -> Result<Option<CertifiedDistance>, DistanceCertificationError>
    where
        S: FnMut(&str, usize) -> SolverAnswer,
    {
        self.certify_distance_by(max_weight, |problem, weight| {
            let dimacs = problem.to_dimacs(weight);
            solver(&dimacs, weight)
        })
    }

    fn certify_distance_by<S>(
        &self,
        max_weight: usize,
        mut solver: S,
    ) -> Result<Option<CertifiedDistance>, DistanceCertificationError>
    where
        S: FnMut(&Self, usize) -> SolverAnswer,
    {
        for weight in 1..=max_weight {
            match solver(self, weight) {
                SolverAnswer::Unsat => {}
                SolverAnswer::Unknown => {
                    return Err(DistanceCertificationError::Unknown { weight });
                }
                SolverAnswer::Sat(witness) => {
                    let actual = self.verify_witness(&witness).map_err(|reason| {
                        DistanceCertificationError::InvalidWitness { weight, reason }
                    })?;
                    if actual > weight {
                        return Err(DistanceCertificationError::WitnessExceedsBound {
                            weight,
                            actual,
                        });
                    }
                    return Ok(Some(CertifiedDistance {
                        distance: actual,
                        witness,
                        sat_certified: true,
                        unsat_trusted_below: weight,
                    }));
                }
            }
        }
        Ok(None)
    }
}

/// Certifies distance through `max_weight` using the in-process batsat SAT solver.
///
/// One deterministic solver instance contains the parity, nontriviality, and sequential-counter
/// clauses. Each weight is selected by an assumption on the counter's final row, retaining learned
/// clauses between bounds. SAT answers are certified natively with
/// [`DistanceProblem::verify_witness`] before they are accepted. UNSAT answers, and therefore the
/// exactness of a returned distance, rest on trusting the solver. `Ok(None)` means batsat reported
/// every weight through `max_weight` UNSAT.
///
/// # Errors
///
/// Returns an error if batsat produces an invalid or overweight model, or does not decide a bound.
pub fn certified_distance(
    problem: &DistanceProblem,
    max_weight: usize,
) -> Result<Option<CertifiedDistance>, DistanceCertificationError> {
    let (encoding, final_row) = problem.encode_for_assumptions(max_weight);
    let mut solver = BatsatDistanceSolver::new(&encoding, problem.num_vars);
    problem.certify_distance_by(max_weight, |_, weight| {
        solver.solve(final_row.assumption_for_bound(weight))
    })
}

#[cfg(test)] // reference fresh-per-weight path, retained for cross-path tests and probes
fn solve_with_batsat(encoding: &Encoding, num_primary_vars: usize) -> SolverAnswer {
    BatsatDistanceSolver::new(encoding, num_primary_vars).solve(None)
}

struct BatsatDistanceSolver {
    solver: BasicSolver,
    variables: Vec<Var>,
    num_primary_vars: usize,
    clauses_consistent: bool,
}

impl BatsatDistanceSolver {
    fn new(encoding: &Encoding, num_primary_vars: usize) -> Self {
        let mut solver = BasicSolver::default();
        let variables: Vec<_> = (0..encoding.num_vars)
            .map(|_| solver.new_var_default())
            .collect();
        let mut clauses_consistent = true;

        for clause in encoding.groups.iter().flat_map(|group| &group.clauses) {
            let mut literals: Vec<_> = clause
                .iter()
                .map(|&literal| Self::to_batsat_literal(&variables, literal))
                .collect();
            if !solver.add_clause_reuse(&mut literals) {
                clauses_consistent = false;
                break;
            }
        }

        Self {
            solver,
            variables,
            num_primary_vars,
            clauses_consistent,
        }
    }

    fn to_batsat_literal(variables: &[Var], literal: i32) -> Lit {
        let index = literal.unsigned_abs() as usize - 1;
        Lit::new(variables[index], literal > 0)
    }

    fn solve(&mut self, assumption: Option<i32>) -> SolverAnswer {
        if !self.clauses_consistent {
            return SolverAnswer::Unsat;
        }
        let assumptions: Vec<_> = assumption
            .map(|literal| Self::to_batsat_literal(&self.variables, literal))
            .into_iter()
            .collect();
        let answer = self.solver.solve_limited(&assumptions);
        if answer == lbool::TRUE {
            SolverAnswer::Sat(
                self.variables[..self.num_primary_vars]
                    .iter()
                    .map(|&variable| self.solver.value_var(variable) == lbool::TRUE)
                    .collect(),
            )
        } else if answer == lbool::FALSE {
            SolverAnswer::Unsat
        } else {
            SolverAnswer::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DistanceSearchConfig, FaultMechanism, StabilizerCode, calculate_distance,
        connected_cluster_fault_distance, exhaustive_fault_distance,
    };
    use pecos_core::pauli::{Xs, Zs};
    use std::time::Instant;

    #[derive(Debug)]
    struct ParsedCnf {
        num_vars: usize,
        clauses: Vec<Vec<i32>>,
    }

    fn parse_dimacs(text: &str) -> ParsedCnf {
        let mut num_vars = None;
        let mut declared_clauses = None;
        let mut clauses = Vec::new();
        for line in text.lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            match fields.as_slice() {
                [] | ["c", ..] => {}
                ["p", "cnf", variables, count] => {
                    num_vars = Some(variables.parse().unwrap());
                    declared_clauses = Some(count.parse().unwrap());
                }
                _ => {
                    let mut clause: Vec<i32> =
                        fields.iter().map(|field| field.parse().unwrap()).collect();
                    assert_eq!(clause.pop(), Some(0));
                    assert!(!clause.contains(&0));
                    clauses.push(clause);
                }
            }
        }
        assert_eq!(declared_clauses, Some(clauses.len()));
        ParsedCnf {
            num_vars: num_vars.unwrap(),
            clauses,
        }
    }

    fn cnf_satisfied_with_primary(cnf: &ParsedCnf, primary: &[bool]) -> bool {
        assert!(primary.len() <= cnf.num_vars);
        let mut values = vec![None; cnf.num_vars + 1];
        for (index, &value) in primary.iter().enumerate() {
            values[index + 1] = Some(value);
        }

        // Every emitted auxiliary is the output of an equivalence encoding. Unit propagation
        // therefore computes it from the fixed primary assignment without searching.
        loop {
            let mut changed = false;
            for clause in &cnf.clauses {
                let mut unresolved = None;
                let mut unresolved_count = 0;
                let mut satisfied = false;
                for &literal in clause {
                    let variable = literal.unsigned_abs() as usize;
                    if let Some(value) = values[variable] {
                        if value == (literal > 0) {
                            satisfied = true;
                            break;
                        }
                    } else {
                        unresolved = Some(literal);
                        unresolved_count += 1;
                    }
                }
                if satisfied {
                    continue;
                }
                if unresolved_count == 0 {
                    return false;
                }
                if unresolved_count == 1 {
                    let literal = unresolved.unwrap();
                    let variable = literal.unsigned_abs() as usize;
                    let required = literal > 0;
                    match values[variable] {
                        Some(value) if value != required => return false,
                        Some(_) => {}
                        None => {
                            values[variable] = Some(required);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        assert!(
            values[1..].iter().all(Option::is_some),
            "encoding left an auxiliary variable underdetermined"
        );
        cnf.clauses.iter().all(|clause| {
            clause
                .iter()
                .any(|&literal| values[literal.unsigned_abs() as usize].unwrap() == (literal > 0))
        })
    }

    fn assignment(mask: usize, num_vars: usize) -> Vec<bool> {
        (0..num_vars)
            .map(|column| mask & (1 << column) != 0)
            .collect()
    }

    fn exhaustive_minimum(problem: &DistanceProblem) -> Option<usize> {
        (0..1 << problem.num_vars())
            .filter_map(|mask| {
                problem
                    .verify_witness(&assignment(mask, problem.num_vars()))
                    .ok()
            })
            .min()
    }

    fn exhaustive_dimacs_minimum(problem: &DistanceProblem) -> Option<usize> {
        (0..=problem.num_vars()).find(|&bound| {
            let cnf = parse_dimacs(&problem.to_dimacs(bound));
            (0..1 << problem.num_vars())
                .any(|mask| cnf_satisfied_with_primary(&cnf, &assignment(mask, problem.num_vars())))
        })
    }

    fn exhaustive_dimacs_answer(problem: &DistanceProblem, dimacs: &str) -> SolverAnswer {
        let cnf = parse_dimacs(dimacs);
        (0..1 << problem.num_vars())
            .map(|mask| assignment(mask, problem.num_vars()))
            .find(|candidate| cnf_satisfied_with_primary(&cnf, candidate))
            .map_or(SolverAnswer::Unsat, SolverAnswer::Sat)
    }

    fn repetition_triad_dem() -> DetectorErrorModel {
        let mut dem = DetectorErrorModel::new();
        for (detectors, observables) in
            [(vec![0, 1], vec![0]), (vec![0], vec![]), (vec![1], vec![])]
        {
            dem.add_direct_contribution(
                FaultMechanism::from_unsorted(detectors, observables),
                0.01,
            );
        }
        dem
    }

    fn steane_hamming_matrix() -> ParityCheckMatrix {
        ParityCheckMatrix::from_dense(vec![
            vec![1, 0, 1, 0, 1, 0, 1],
            vec![0, 1, 1, 0, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 1, 1],
        ])
        .unwrap()
    }

    fn steane_distance_problem() -> DistanceProblem {
        let h = steane_hamming_matrix();
        let logical = ParityCheckMatrix::from_dense(vec![vec![1; 7]]).unwrap();
        DistanceProblem::from_css_checks(&h, &logical).unwrap()
    }

    #[test]
    fn dimacs_encoding_matches_native_predicate_for_every_small_assignment() {
        let cases = [
            DistanceProblem::from_css_checks(
                &ParityCheckMatrix::from_dense(vec![vec![1, 1, 0, 0]]).unwrap(),
                &ParityCheckMatrix::from_dense(vec![vec![0, 1, 1, 0]]).unwrap(),
            )
            .unwrap(),
            DistanceProblem::from_css_checks(
                &ParityCheckMatrix::from_dense(vec![vec![1, 1, 1, 1, 0]]).unwrap(),
                &ParityCheckMatrix::from_dense(vec![vec![1, 0, 1, 0, 1], vec![0, 1, 0, 1, 0]])
                    .unwrap(),
            )
            .unwrap(),
            DistanceProblem::from_css_checks(
                &ParityCheckMatrix::zeros(0, 6),
                &ParityCheckMatrix::from_dense(vec![vec![0, 0, 0, 0, 0, 0]]).unwrap(),
            )
            .unwrap(),
        ];

        for problem in &cases {
            assert!(problem.num_vars() <= 12);
            for bound in 0..=problem.num_vars() {
                let cnf = parse_dimacs(&problem.to_dimacs(bound));
                for mask in 0..1 << problem.num_vars() {
                    let candidate = assignment(mask, problem.num_vars());
                    let direct = problem
                        .verify_witness(&candidate)
                        .is_ok_and(|weight| weight <= bound);
                    assert_eq!(
                        cnf_satisfied_with_primary(&cnf, &candidate),
                        direct,
                        "assignment {mask:#b} at bound {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn steane_hamming_problem_matches_existing_distance_search() {
        let h = steane_hamming_matrix();
        let spec = StabilizerCodeSpec::builder(7)
            .checks_from_css(&h, &h)
            .unwrap()
            .logical_x(Xs([0, 1, 2, 3, 4, 5, 6]))
            .logical_z(Zs([0, 1, 2, 3, 4, 5, 6]))
            .build_verified()
            .unwrap();
        let oracle = calculate_distance(&spec, &DistanceSearchConfig::css()).unwrap();
        let problem = steane_distance_problem();

        assert_eq!(oracle.distance, 3);
        assert_eq!(exhaustive_minimum(&problem), Some(oracle.distance));
        assert_eq!(exhaustive_dimacs_minimum(&problem), Some(oracle.distance));
        assert_eq!(
            exhaustive_minimum(&DistanceProblem::from_css_code_x_distance(&spec).unwrap()),
            Some(oracle.distance)
        );
        assert_eq!(
            exhaustive_minimum(&DistanceProblem::from_css_code_z_distance(&spec).unwrap()),
            Some(oracle.distance)
        );
    }

    #[test]
    fn non_css_spec_is_rejected_without_projection() {
        let spec = StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        assert!(matches!(
            DistanceProblem::from_css_code_x_distance(&spec),
            Err(DistanceProblemError::NonCssOperator { .. })
        ));
        assert!(matches!(
            DistanceProblem::from_css_code_z_distance(&spec),
            Err(DistanceProblemError::NonCssOperator { .. })
        ));
    }

    #[test]
    fn dem_problem_matches_both_existing_fault_distance_searches() {
        let dem = repetition_triad_dem();
        let problem = DistanceProblem::from_dem(&dem);
        assert_eq!(exhaustive_minimum(&problem), Some(3));
        assert_eq!(exhaustive_dimacs_minimum(&problem), Some(3));
        assert_eq!(exhaustive_fault_distance(&dem, 3).unwrap().distance, 3);
        assert_eq!(
            connected_cluster_fault_distance(&dem, 3).unwrap().distance,
            3
        );
    }

    #[test]
    fn sequential_counter_is_exact_on_both_sides_of_distance() {
        let problem = DistanceProblem::from_dem(&repetition_triad_dem());
        for (bound, expected) in [(2, false), (3, true)] {
            let cnf = parse_dimacs(&problem.to_dimacs(bound));
            let any_satisfying = (0..1 << problem.num_vars()).any(|mask| {
                cnf_satisfied_with_primary(&cnf, &assignment(mask, problem.num_vars()))
            });
            assert_eq!(any_satisfying, expected, "bound {bound}");
        }
    }

    #[test]
    fn certification_rejects_witness_violating_a_check() {
        let problem = DistanceProblem::from_dem(&repetition_triad_dem());
        let result =
            problem.certify_distance_with(3, |_, _| SolverAnswer::Sat(vec![true, false, false]));
        assert_eq!(
            result,
            Err(DistanceCertificationError::InvalidWitness {
                weight: 1,
                reason: WitnessError::OddCheck { row: 0 },
            })
        );
    }

    #[test]
    fn certification_rejects_valid_witness_above_solver_bound() {
        let problem = DistanceProblem::from_css_checks(
            &ParityCheckMatrix::zeros(0, 2),
            &ParityCheckMatrix::from_dense(vec![vec![1, 0]]).unwrap(),
        )
        .unwrap();
        let result = problem.certify_distance_with(1, |_, _| SolverAnswer::Sat(vec![true, true]));
        assert_eq!(
            result,
            Err(DistanceCertificationError::WitnessExceedsBound {
                weight: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn certification_returns_none_after_trusted_unsat_bound() {
        let problem = steane_distance_problem();
        let mut weights = Vec::new();
        let result = problem
            .certify_distance_with(4, |_, weight| {
                weights.push(weight);
                SolverAnswer::Unsat
            })
            .unwrap();
        assert_eq!(result, None);
        assert_eq!(weights, vec![1, 2, 3, 4]);
    }

    #[test]
    fn honest_mock_certifies_steane_and_repetition_triad() {
        for (problem, expected) in [
            (steane_distance_problem(), 3),
            (DistanceProblem::from_dem(&repetition_triad_dem()), 3),
        ] {
            let certified = problem
                .certify_distance_with(expected, |dimacs, _| {
                    exhaustive_dimacs_answer(&problem, dimacs)
                })
                .unwrap()
                .unwrap();
            assert_eq!(certified.distance, expected);
            assert_eq!(certified.unsat_trusted_below, expected);
            assert!(certified.sat_certified);
            assert_eq!(problem.verify_witness(&certified.witness), Ok(expected));
        }
    }

    #[test]
    fn batsat_certifies_steane_x_and_z_against_existing_oracle() {
        let mut spec = StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::steane()).unwrap();
        let oracle = spec.calculate_distance().unwrap().distance;
        assert_eq!(spec.distance(), Some(oracle));

        for problem in [
            DistanceProblem::from_css_code_x_distance(&spec).unwrap(),
            DistanceProblem::from_css_code_z_distance(&spec).unwrap(),
        ] {
            let certified = certified_distance(&problem, oracle).unwrap().unwrap();
            assert_eq!(certified.distance, oracle);
            assert_eq!(certified.unsat_trusted_below, oracle);
            assert!(certified.sat_certified);
            assert_eq!(problem.verify_witness(&certified.witness), Ok(oracle));
        }
    }

    #[test]
    fn batsat_certifies_repetition_triad_against_exhaustive_oracle() {
        let dem = repetition_triad_dem();
        let oracle = exhaustive_fault_distance(&dem, 3).unwrap().distance;
        let problem = DistanceProblem::from_dem(&dem);
        let certified = certified_distance(&problem, oracle).unwrap().unwrap();

        assert_eq!(certified.distance, oracle);
        assert_eq!(certified.unsat_trusted_below, oracle);
        assert!(certified.sat_certified);
        assert_eq!(problem.verify_witness(&certified.witness), Ok(oracle));
    }

    #[test]
    fn batsat_is_deterministic_and_respects_max_weight() {
        let problem = steane_distance_problem();
        assert_eq!(certified_distance(&problem, 2), Ok(None));

        let first = certified_distance(&problem, 3).unwrap().unwrap();
        let second = certified_distance(&problem, 3).unwrap().unwrap();
        assert_eq!(first.witness, second.witness);
        assert_eq!(problem.verify_witness(&first.witness), Ok(3));
        assert_eq!(problem.verify_witness(&second.witness), Ok(3));
    }

    #[test]
    fn incremental_matches_reference_path_on_seeded_small_problems() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        const NUM_CASES: usize = 128;
        const NUM_VARS: usize = 8;

        let mut rng = SmallRng::seed_from_u64(0x19C0_5EED);
        for case_index in 0..NUM_CASES {
            let random_rows = |rng: &mut SmallRng, count: usize| -> Vec<Vec<u8>> {
                (0..count)
                    .map(|_| {
                        (0..NUM_VARS)
                            .map(|_| u8::from(rng.random_bool(0.4)))
                            .collect()
                    })
                    .collect()
            };
            let h_count = rng.random_range(1..=3);
            let l_count = rng.random_range(1..=2);
            let h_rows = random_rows(&mut rng, h_count);
            let l_rows = random_rows(&mut rng, l_count);
            let (Ok(h), Ok(l)) = (
                ParityCheckMatrix::from_dense(h_rows.clone()),
                ParityCheckMatrix::from_dense(l_rows.clone()),
            ) else {
                continue;
            };
            let problem = DistanceProblem::from_css_checks(&h, &l).unwrap();

            for max_weight in [2usize, NUM_VARS] {
                let incremental = certified_distance(&problem, max_weight).unwrap();
                let reference = problem
                    .certify_distance_with(max_weight, |dimacs, _| {
                        exhaustive_dimacs_answer(&problem, dimacs)
                    })
                    .unwrap();
                assert_eq!(
                    incremental.as_ref().map(|c| c.distance),
                    reference.as_ref().map(|c| c.distance),
                    "distance diverged for seeded case {case_index} at max_weight {max_weight}: H {h_rows:?} L {l_rows:?}"
                );
                if let Some(certified) = &incremental {
                    assert_eq!(
                        problem.verify_witness(&certified.witness),
                        Ok(certified.distance),
                        "incremental witness failed native verification for case {case_index}"
                    );
                }
            }
        }
    }

    fn bb_circulant(l: usize, m: usize, terms: &[(usize, usize)]) -> F2Matrix {
        let size = l * m;
        let mut matrix = F2Matrix::zeros(size, size);
        for row_x in 0..l {
            for row_y in 0..m {
                let row = row_x * m + row_y;
                for &(x_power, y_power) in terms {
                    let column = ((row_x + x_power) % l) * m + (row_y + y_power) % m;
                    matrix.set(row, column, matrix.get(row, column) ^ 1);
                }
            }
        }
        matrix
    }

    #[test]
    #[ignore = "timing probe for the batsat backend"]
    fn batsat_bivariate_bicycle_72_12_6_timing_probe() {
        let (l, m) = (6, 6);
        let block_size = l * m;
        let n = 2 * block_size;
        let a = bb_circulant(l, m, &[(3, 0), (0, 1), (0, 2)]);
        let b = bb_circulant(l, m, &[(0, 3), (1, 0), (2, 0)]);
        let mut hx = F2Matrix::zeros(block_size, n);
        let mut hz = F2Matrix::zeros(block_size, n);
        for row in 0..block_size {
            for column in 0..block_size {
                hx.set(row, column, a.get(row, column));
                hx.set(row, block_size + column, b.get(row, column));
                hz.set(row, column, b.get(column, row));
                hz.set(row, block_size + column, a.get(column, row));
            }
        }

        assert_eq!(n, 72);
        assert_eq!(
            hx.mul(&hz.transpose()),
            F2Matrix::zeros(block_size, block_size)
        );
        let hx_rank = hx.row_reduce().1.len();
        let hz_rank = hz.row_reduce().1.len();
        assert_eq!(n - hx_rank - hz_rank, 12);

        let (hx_rref, hx_pivots) = hx.row_reduce();
        let logical_candidates = hz.kernel().into_iter().filter_map(|mut vector| {
            for (row, &pivot) in hx_pivots.iter().enumerate() {
                if vector[pivot] == 1 {
                    for (column, bit) in vector.iter_mut().enumerate() {
                        *bit ^= hx_rref.get(row, column);
                    }
                }
            }
            vector.iter().any(|&bit| bit != 0).then_some(vector)
        });
        let (logical_rref, _) = F2Matrix::from_rows(logical_candidates.collect()).row_reduce();
        let logical_rows: Vec<_> = logical_rref
            .rows()
            .into_iter()
            .filter(|row| row.iter().any(|&bit| bit != 0))
            .collect();
        assert_eq!(logical_rows.len(), 12);

        let hx_checks = ParityCheckMatrix::from_dense(hx.rows()).unwrap();
        let logical_checks = ParityCheckMatrix::from_dense(logical_rows).unwrap();
        let problem = DistanceProblem::from_css_checks(&hx_checks, &logical_checks).unwrap();
        let total_started = Instant::now();
        let certified = problem
            .certify_distance_by(6, |problem, weight| {
                let started = Instant::now();
                let answer = solve_with_batsat(&problem.encode(Some(weight)), problem.num_vars);
                println!(
                    "BB [[72,12,6]] weight {weight}: {:?} ({answer:?})",
                    started.elapsed()
                );
                answer
            })
            .unwrap()
            .unwrap();
        println!(
            "BB [[72,12,6]] fresh-path total: {:?}",
            total_started.elapsed()
        );
        assert_eq!(certified.distance, 6);
        assert_eq!(problem.verify_witness(&certified.witness), Ok(6));

        let incremental_started = Instant::now();
        let incremental = certified_distance(&problem, 6).unwrap().unwrap();
        println!(
            "BB [[72,12,6]] incremental total: {:?}",
            incremental_started.elapsed()
        );
        assert_eq!(incremental.distance, 6);
        assert_eq!(problem.verify_witness(&incremental.witness), Ok(6));
    }

    #[test]
    fn unknown_answer_names_the_reached_weight() {
        let problem = steane_distance_problem();
        let result = problem.certify_distance_with(3, |_, weight| {
            if weight < 2 {
                SolverAnswer::Unsat
            } else {
                SolverAnswer::Unknown
            }
        });
        assert_eq!(
            result,
            Err(DistanceCertificationError::Unknown { weight: 2 })
        );
    }

    #[test]
    fn wcnf_has_exact_hard_encoding_and_one_soft_unit_per_primary() {
        let problem = steane_distance_problem();
        let text = problem.to_wcnf();
        let mut header = None;
        let mut hard = Vec::new();
        let mut soft = Vec::new();
        for line in text.lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            match fields.as_slice() {
                [] | ["c", ..] => {}
                ["p", "wcnf", variables, clauses] => {
                    header = Some((
                        variables.parse::<usize>().unwrap(),
                        clauses.parse::<usize>().unwrap(),
                    ));
                }
                ["h", literals @ .., "0"] => hard.push(
                    literals
                        .iter()
                        .map(|literal| literal.parse::<i32>().unwrap())
                        .collect::<Vec<_>>(),
                ),
                ["1", literal, "0"] => soft.push(literal.parse::<i32>().unwrap()),
                _ => panic!("unrecognized WCNF line: {line}"),
            }
        }

        let encoding = problem.encode(None);
        let expected_hard: Vec<_> = encoding
            .groups
            .into_iter()
            .flat_map(|group| group.clauses)
            .collect();
        assert_eq!(hard, expected_hard);
        assert_eq!(
            soft,
            (1..=problem.num_vars())
                .map(|x| -i32::try_from(x).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(header, Some((encoding.num_vars, hard.len() + soft.len())));
    }

    #[test]
    fn matrix_width_mismatch_and_witness_length_are_explicit() {
        assert_eq!(
            DistanceProblem::from_css_checks(
                &ParityCheckMatrix::zeros(0, 2),
                &ParityCheckMatrix::zeros(0, 3),
            ),
            Err(DistanceProblemError::MatrixWidthMismatch {
                h_width: 2,
                l_width: 3,
            })
        );
        assert_eq!(
            steane_distance_problem().verify_witness(&[false; 6]),
            Err(WitnessError::LengthMismatch {
                expected: 7,
                actual: 6,
            })
        );
    }
}
