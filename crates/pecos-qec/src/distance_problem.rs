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

use crate::{DetectorErrorModel, ParityCheckMatrix, StabilizerCodeSpec, StabilizerCodeSpecError};
use batsat::{BasicSolver, Lit, SolverInterface, lbool};
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
    weight_mode: WeightMode,
    /// Target parity bit per H row. All-zero for homogeneous (distance) problems;
    /// nonzero targets arise from affine coset-membership constraints.
    parity_targets: Vec<u8>,
    /// Whether a witness must flip at least one L row. Distance problems require
    /// it; coset-weight problems have no nontriviality requirement (weight 0
    /// legitimately means the representative lies in the group).
    require_logical_effect: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightMode {
    Bit,
    QubitSupport { num_qubits: usize },
}

/// Errors constructing a [`DistanceProblem`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DistanceProblemError {
    /// The stabilizer specification is not valid for an ordinary-code distance problem.
    #[error(transparent)]
    StabilizerSpec(#[from] StabilizerCodeSpecError),

    /// The check and logical matrices describe different numbers of variables.
    #[error("distance matrices have different widths: H has {h_width}, L has {l_width}")]
    MatrixWidthMismatch {
        /// Number of columns in the check matrix.
        h_width: usize,
        /// Number of columns in the logical matrix.
        l_width: usize,
    },
    /// A coset representative contains an entry outside the binary alphabet.
    #[error("coset representative entry at index {index} is {value}, expected 0 or 1")]
    NonBinaryRepresentative {
        /// Index of the invalid entry.
        index: usize,
        /// Invalid value.
        value: u8,
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

/// Outcome of a budgeted classical-code distance certification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassicalDistanceSearchOutcome {
    /// The exact minimum weight and a nonzero kernel witness were certified.
    Certified(CertifiedDistance),
    /// No nonzero kernel element exists because the parity-check matrix has full column rank.
    NoNonzeroCodeword,
    /// Every weight through the requested bound was certified absent.
    BudgetExhausted {
        /// Largest Hamming weight included in the search.
        max_weight: usize,
    },
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
    objective_variables: Vec<usize>,
}

#[derive(Debug)]
struct EncodingBuilder {
    next_var: usize,
    aux_ranges: Vec<(usize, usize, &'static str)>,
    parity_clauses: Vec<Vec<i32>>,
    nontriviality_clauses: Vec<Vec<i32>>,
    support_clauses: Vec<Vec<i32>>,
    cardinality_clauses: Vec<Vec<i32>>,
}

impl EncodingBuilder {
    fn new(num_primary_vars: usize) -> Self {
        Self {
            next_var: num_primary_vars + 1,
            aux_ranges: Vec::new(),
            parity_clauses: Vec::new(),
            nontriviality_clauses: Vec::new(),
            support_clauses: Vec::new(),
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

    fn finish(self, include_cardinality: bool, objective_variables: Vec<usize>) -> Encoding {
        let mut groups = vec![ClauseGroup {
            description: "parity constraints (Tseitin XOR chains)",
            clauses: self.parity_clauses,
        }];
        groups.push(ClauseGroup {
            description: "logical nontriviality",
            clauses: self.nontriviality_clauses,
        });
        if !self.support_clauses.is_empty() {
            groups.push(ClauseGroup {
                description: "qubit-support indicators (Tseitin OR gates)",
                clauses: self.support_clauses,
            });
        }
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
            objective_variables,
        }
    }
}

impl DistanceProblem {
    pub(crate) fn matrices(&self) -> (&F2Matrix, &F2Matrix) {
        (&self.h, &self.l)
    }

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
            parity_targets: vec![0; h.num_checks()],
            require_logical_effect: true,
            weight_mode: WeightMode::Bit,
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
        code.verify_logical_completeness()?;
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
            parity_targets: vec![0; checks.len()],
            h: Self::matrix_from_rows(checks, num_qubits),
            l: Self::matrix_from_rows(logicals, num_qubits),
            num_vars: num_qubits,
            require_logical_effect: true,
            weight_mode: WeightMode::Bit,
        })
    }

    /// Constructs the full symplectic distance problem for an arbitrary stabilizer code spec.
    ///
    /// The `2n` primary variables use `[X|Z]` order. Each `H` row is the symplectic product with
    /// one stabilizer, and the `L` rows are the symplectic products with every logical Z followed
    /// by every logical X. Weight is physical-qubit support: a qubit contributes once when either
    /// or both of its X and Z variables are selected.
    ///
    /// # Errors
    ///
    /// Returns [`DistanceProblemError::QubitOutOfRange`] if a stabilizer or logical operator acts
    /// outside the code width.
    pub fn from_stabilizer_spec(code: &StabilizerCodeSpec) -> Result<Self, DistanceProblemError> {
        code.verify_logical_completeness()?;
        Self::from_stabilizer_spec_without_logical_completeness(code)
    }

    /// Constructs the symplectic problem after a caller has applied its own logical-count rule.
    ///
    /// Subsystem codes use this only after validating the gauge-aware counting relation, since
    /// their protected logical count is smaller than `n - rank(S)` by the gauge-qubit count.
    pub(crate) fn from_stabilizer_spec_without_logical_completeness(
        code: &StabilizerCodeSpec,
    ) -> Result<Self, DistanceProblemError> {
        let num_qubits = code.num_qubits();
        let checks = code
            .stabilizers()
            .iter()
            .enumerate()
            .map(|(index, operator)| {
                Self::symplectic_commutation_row(operator, "stabilizer", index, num_qubits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let logicals = code
            .logical_zs()
            .iter()
            .enumerate()
            .map(|(index, operator)| {
                Self::symplectic_commutation_row(operator, "logical Z", index, num_qubits)
            })
            .chain(
                code.logical_xs()
                    .iter()
                    .enumerate()
                    .map(|(index, operator)| {
                        Self::symplectic_commutation_row(operator, "logical X", index, num_qubits)
                    }),
            )
            .collect::<Result<Vec<_>, _>>()?;
        let num_vars = 2 * num_qubits;
        Ok(Self {
            parity_targets: vec![0; checks.len()],
            h: Self::matrix_from_rows(checks, num_vars),
            l: Self::matrix_from_rows(logicals, num_vars),
            num_vars,
            require_logical_effect: true,
            weight_mode: WeightMode::QubitSupport { num_qubits },
        })
    }

    fn symplectic_commutation_row(
        operator: &pecos_core::PauliString,
        component: &'static str,
        index: usize,
        num_qubits: usize,
    ) -> Result<Vec<u8>, DistanceProblemError> {
        let x = operator.x_positions();
        let z = operator.z_positions();
        Self::validate_positions(&x, component, index, num_qubits)?;
        Self::validate_positions(&z, component, index, num_qubits)?;
        let mut row = vec![0; 2 * num_qubits];
        for qubit in z {
            row[qubit] = 1;
        }
        for qubit in x {
            row[num_qubits + qubit] = 1;
        }
        Ok(row)
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
        Self {
            parity_targets: vec![0; h.num_rows()],
            h,
            l,
            num_vars,
            require_logical_effect: true,
            weight_mode: WeightMode::Bit,
        }
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
        let clause_count = hard_count + encoding.objective_variables.len();
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
        for &variable in &encoding.objective_variables {
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
        if self.require_logical_effect {
            self.encode_logical_nontriviality(&mut builder);
        }
        let objective_variables = self.encode_weight_variables(&mut builder);
        if let Some(max_weight) = max_weight {
            Self::encode_sequential_counter(&mut builder, max_weight, &objective_variables);
        }
        builder.finish(max_weight.is_some(), objective_variables)
    }

    fn encode_weight_variables(&self, builder: &mut EncodingBuilder) -> Vec<usize> {
        match self.weight_mode {
            WeightMode::Bit => (1..=self.num_vars).collect(),
            WeightMode::QubitSupport { num_qubits } => {
                let indicators =
                    builder.allocate_range(num_qubits, "physical-qubit support indicators");
                for (qubit, &indicator) in indicators.iter().enumerate() {
                    let x = qubit + 1;
                    let z = num_qubits + qubit + 1;
                    let indicator = Self::literal(indicator);
                    builder
                        .support_clauses
                        .push(vec![-Self::literal(x), indicator]);
                    builder
                        .support_clauses
                        .push(vec![-Self::literal(z), indicator]);
                    builder.support_clauses.push(vec![
                        Self::literal(x),
                        Self::literal(z),
                        -indicator,
                    ]);
                }
                indicators
            }
        }
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
                [] => {
                    if self.parity_targets[row] == 1 {
                        builder.parity_clauses.push(Vec::new());
                    }
                }
                &[variable] => {
                    let sign = if self.parity_targets[row] == 1 { 1 } else { -1 };
                    builder
                        .parity_clauses
                        .push(vec![sign * Self::literal(variable)]);
                }
                &[first, second, ref rest @ ..] => {
                    let mut output = aux_iter.next().expect("pre-counted XOR auxiliary");
                    Self::push_xor(&mut builder.parity_clauses, first, second, output);
                    for &variable in rest {
                        let next = aux_iter.next().expect("pre-counted XOR auxiliary");
                        Self::push_xor(&mut builder.parity_clauses, output, variable, next);
                        output = next;
                    }
                    // The chain output must equal the row's target parity.
                    let sign = if self.parity_targets[row] == 1 { 1 } else { -1 };
                    builder
                        .parity_clauses
                        .push(vec![sign * Self::literal(output)]);
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
        builder: &mut EncodingBuilder,
        max_weight: usize,
        inputs: &[usize],
    ) {
        if max_weight >= inputs.len() {
            return;
        }
        if max_weight == 0 {
            for &variable in inputs {
                builder
                    .cardinality_clauses
                    .push(vec![-Self::literal(variable)]);
            }
            return;
        }

        // s[i, j] means that at least j of x_1..=x_i are true. These are the sequential unary
        // counter variables of Sinz (2005). Both directions of each recurrence are emitted, making
        // every auxiliary functionally determined by the primary variables.
        let threshold = max_weight + 1;
        let variables = builder.allocate_range(
            inputs.len() * threshold,
            "Sinz sequential-counter prefix thresholds",
        );
        let counter = |i: usize, j: usize| variables[(i - 1) * threshold + (j - 1)];

        // First prefix: s[1,1] <-> x_1; all unreachable higher thresholds are false.
        let first = counter(1, 1);
        let first_input = Self::literal(inputs[0]);
        builder
            .cardinality_clauses
            .push(vec![-first_input, Self::literal(first)]);
        builder
            .cardinality_clauses
            .push(vec![first_input, -Self::literal(first)]);
        for j in 2..=threshold {
            builder
                .cardinality_clauses
                .push(vec![-Self::literal(counter(1, j))]);
        }

        for i in 2..=inputs.len() {
            let x = Self::literal(inputs[i - 1]);
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
        builder
            .cardinality_clauses
            .push(vec![-Self::literal(counter(inputs.len(), threshold))]);
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
            if odd != (self.parity_targets[row] == 1) {
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
        if self.require_logical_effect && !logical_nonzero {
            return Err(WitnessError::ZeroLogicalEffect);
        }
        Ok(match self.weight_mode {
            WeightMode::Bit => assignment.iter().filter(|&&selected| selected).count(),
            WeightMode::QubitSupport { num_qubits } => (0..num_qubits)
                .filter(|&qubit| assignment[qubit] || assignment[num_qubits + qubit])
                .count(),
        })
    }

    /// Incrementally certifies distance through `max_weight` using a pluggable SAT solver.
    ///
    /// A valid all-zero assignment is returned directly. Otherwise, the solver is called once per
    /// bound from 1 upward. Its SAT assignment must contain only the original variables and is
    /// checked natively, so SAT soundness does not rely on the solver. Every preceding UNSAT result
    /// is trusted; that trust is what turns the verified upper bound into an exact distance.
    /// `Ok(None)` means all bounds through `max_weight` were reported UNSAT, establishing only a
    /// solver-trusted lower bound greater than `max_weight`.
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
        if self.zero_assignment_satisfies() {
            return Ok(Some(CertifiedDistance {
                distance: 0,
                witness: vec![false; self.num_vars],
                sat_certified: true,
                unsat_trusted_below: 0,
            }));
        }
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

impl DistanceProblem {
    /// Builds the affine problem "minimum weight of `representative + rowspan(group)`".
    ///
    /// Membership in the coset is expressed through the orthogonal complement: with `D` a basis
    /// of `rowspan(G)^perp` (the kernel of `G`), `e` lies in `p + rowspan(G)` iff `D e = D p`.
    /// There is no nontriviality requirement: weight 0 means the representative is in the group.
    ///
    /// # Errors
    ///
    /// Returns an error if the representative width does not match the group or if an entry is not
    /// binary.
    pub fn coset_weight_problem(
        group: &ParityCheckMatrix,
        representative: &[u8],
    ) -> Result<Self, DistanceProblemError> {
        let num_vars = group.num_qubits();
        if representative.len() != num_vars {
            return Err(DistanceProblemError::MatrixWidthMismatch {
                h_width: num_vars,
                l_width: representative.len(),
            });
        }
        if let Some((index, &value)) = representative
            .iter()
            .enumerate()
            .find(|&(_, &value)| value > 1)
        {
            return Err(DistanceProblemError::NonBinaryRepresentative { index, value });
        }
        let dual_rows = group.matrix().kernel();
        let parity_targets: Vec<u8> = dual_rows
            .iter()
            .map(|dual| {
                u8::from(
                    dual.iter()
                        .zip(representative)
                        .filter(|&(&d, &r)| d == 1 && r == 1)
                        .count()
                        % 2
                        == 1,
                )
            })
            .collect();
        Ok(Self {
            parity_targets,
            h: Self::matrix_from_rows(dual_rows, num_vars),
            l: F2Matrix::zeros(0, num_vars),
            num_vars,
            require_logical_effect: false,
            weight_mode: WeightMode::Bit,
        })
    }

    /// True when every affine target is zero, i.e. the zero assignment is a valid witness.
    fn zero_assignment_satisfies(&self) -> bool {
        !self.require_logical_effect && self.parity_targets.iter().all(|&target| target == 0)
    }
}

/// Certified minimum weight of `representative + rowspan(group)` over the binary alphabet.
///
/// Weight 0 (the representative lies in the group) is certified without any solver call.
///
/// # Errors
///
/// Returns an error on width mismatch or if the solver misbehaves.
pub fn certified_coset_weight(
    group: &ParityCheckMatrix,
    representative: &[u8],
    max_weight: usize,
) -> Result<Option<CertifiedDistance>, CosetWeightError> {
    let problem = DistanceProblem::coset_weight_problem(group, representative)?;
    certified_distance(&problem, max_weight).map_err(CosetWeightError::Certification)
}

/// Certified minimum qubit-support weight of `operator * stabilizer group` for any code.
///
/// Uses the plain symplectic representation `[X | Z]` (phases are irrelevant to weight and to
/// GF(2) span membership) with the per-qubit-support weight mode, so a `Y` costs one.
/// This deliberately does not require a complete logical basis: the operation measures one
/// supplied representative against the stabilizer group and does not use the code's logicals.
///
/// # Errors
///
/// Returns an error on width mismatch or solver misbehavior.
pub fn certified_stabilizer_coset_weight(
    code: &StabilizerCodeSpec,
    operator: &pecos_core::PauliString,
    max_weight: usize,
) -> Result<Option<CertifiedDistance>, CosetWeightError> {
    let num_qubits = code.num_qubits();
    let group = pecos_quantum::SymplecticMatrix::from_pauli_sequence_ignoring_phase(
        &pecos_quantum::PauliSequence::new(code.stabilizers().to_vec()),
        num_qubits,
    )
    .map_err(|error| CosetWeightError::Symplectic(error.to_string()))?;
    let representative_rows = pecos_quantum::SymplecticMatrix::from_pauli_sequence_ignoring_phase(
        &pecos_quantum::PauliSequence::new(vec![operator.clone()]),
        num_qubits,
    )
    .map_err(|error| CosetWeightError::Symplectic(error.to_string()))?;
    let Some(representative) = representative_rows.rows().into_iter().next() else {
        return Err(CosetWeightError::Symplectic(
            "operator produced no symplectic row".to_string(),
        ));
    };
    let group_rows = group.rows();
    let num_vars = 2 * num_qubits;
    let dual_rows = DistanceProblem::matrix_from_rows(group_rows, num_vars).kernel();
    let parity_targets: Vec<u8> = dual_rows
        .iter()
        .map(|dual| {
            u8::from(
                dual.iter()
                    .zip(&representative)
                    .filter(|&(&d, &r)| d == 1 && r == 1)
                    .count()
                    % 2
                    == 1,
            )
        })
        .collect();
    let problem = DistanceProblem {
        parity_targets,
        h: DistanceProblem::matrix_from_rows(dual_rows, num_vars),
        l: F2Matrix::zeros(0, num_vars),
        num_vars,
        require_logical_effect: false,
        weight_mode: WeightMode::QubitSupport { num_qubits },
    };
    certified_distance(&problem, max_weight).map_err(CosetWeightError::Certification)
}

/// Certified coset weight of every supplied logical generator (Z generators, then X generators).
///
/// The minimum of this list is **not** the code distance: for example, two weight-two supplied
/// generators can have a weight-one product in a distinct logical coset.
/// This intentionally operates on the supplied generators and does not require them to form a
/// complete logical basis.
///
/// # Errors
///
/// Propagates the first failure from [`certified_stabilizer_coset_weight`].
pub fn logical_generator_coset_weights(
    code: &StabilizerCodeSpec,
    max_weight: usize,
) -> Result<Vec<Option<CertifiedDistance>>, CosetWeightError> {
    code.logical_zs()
        .iter()
        .chain(code.logical_xs())
        .map(|logical| certified_stabilizer_coset_weight(code, logical, max_weight))
        .collect()
}

/// Certified minimum weight of a nonzero kernel element of a classical parity-check matrix.
///
/// Nontriviality is expressed as "some coordinate is set" via an identity logical block.
///
/// # Errors
///
/// Returns an error on solver misbehavior.
pub fn certified_classical_distance(
    h: &ParityCheckMatrix,
    max_weight: usize,
) -> Result<ClassicalDistanceSearchOutcome, CosetWeightError> {
    let n = h.num_qubits();
    if h.rank() == n {
        return Ok(ClassicalDistanceSearchOutcome::NoNonzeroCodeword);
    }
    let identity_rows = (0..n)
        .map(|column| {
            let mut row = vec![0u8; n];
            row[column] = 1;
            row
        })
        .collect();
    let identity = ParityCheckMatrix::from_dense(identity_rows)
        .map_err(|error| CosetWeightError::Symplectic(error.to_string()))?;
    let problem = DistanceProblem::from_css_checks(h, &identity)?;
    certified_distance(&problem, max_weight)
        .map(|result| match result {
            Some(certified) => ClassicalDistanceSearchOutcome::Certified(certified),
            None => ClassicalDistanceSearchOutcome::BudgetExhausted { max_weight },
        })
        .map_err(CosetWeightError::Certification)
}

/// Errors from the coset-weight and classical-distance entry points.
#[derive(Debug, Error)]
pub enum CosetWeightError {
    /// Problem construction failed.
    #[error(transparent)]
    Problem(#[from] DistanceProblemError),
    /// Symplectic conversion failed.
    #[error("symplectic conversion failed: {0}")]
    Symplectic(String),
    /// The certification loop failed.
    #[error(transparent)]
    Certification(DistanceCertificationError),
}

/// Certifies distance through `max_weight` using the in-process batsat SAT solver.
///
/// A fresh deterministic solver instance is built for each weight from the internal clause
/// encoding. SAT answers are certified natively with [`DistanceProblem::verify_witness`] before
/// they are accepted. UNSAT answers, and therefore the exactness of a returned distance, rest on
/// trusting the solver. `Ok(None)` means batsat reported every weight through `max_weight` UNSAT.
///
/// Incremental assumptions could avoid rebuilding the solver in a future implementation, but are
/// deliberately not used here.
///
/// # Errors
///
/// Returns an error if batsat produces an invalid or overweight model, or does not decide a bound.
pub fn certified_distance(
    problem: &DistanceProblem,
    max_weight: usize,
) -> Result<Option<CertifiedDistance>, DistanceCertificationError> {
    problem.certify_distance_by(max_weight, |problem, weight| {
        solve_with_batsat(&problem.encode(Some(weight)), problem.num_vars)
    })
}

fn solve_with_batsat(encoding: &Encoding, num_primary_vars: usize) -> SolverAnswer {
    // The solver is an external dependency; an internal panic (observed: an
    // arithmetic overflow under debug assertions on instances with thousands of
    // variables) must not cross the FFI boundary. A fresh solver is built per
    // call and discarded on unwind, so no shared state can be poisoned.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve_with_batsat_inner(encoding, num_primary_vars)
    }))
    .unwrap_or(SolverAnswer::Unknown)
}

fn solve_with_batsat_inner(encoding: &Encoding, num_primary_vars: usize) -> SolverAnswer {
    let mut solver = BasicSolver::default();
    let variables: Vec<_> = (0..encoding.num_vars)
        .map(|_| solver.new_var_default())
        .collect();

    for clause in encoding.groups.iter().flat_map(|group| &group.clauses) {
        let mut literals: Vec<_> = clause
            .iter()
            .map(|&literal| {
                let index = literal.unsigned_abs() as usize - 1;
                Lit::new(variables[index], literal > 0)
            })
            .collect();
        if !solver.add_clause_reuse(&mut literals) {
            return SolverAnswer::Unsat;
        }
    }

    let answer = solver.solve_limited(&[]);
    if answer == lbool::TRUE {
        SolverAnswer::Sat(
            variables[..num_primary_vars]
                .iter()
                .map(|&variable| solver.value_var(variable) == lbool::TRUE)
                .collect(),
        )
    } else if answer == lbool::FALSE {
        SolverAnswer::Unsat
    } else {
        SolverAnswer::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DemOutput, DistanceSearchConfig, FaultMechanism, StabilizerCode,
        bounded_enumeration_code_distance, calculate_distance, connected_cluster_code_distance,
        connected_cluster_fault_distance, exhaustive_fault_distance,
    };
    use pecos_core::pauli::{X, Xs, Y, Ys, Z, Zs};
    use pecos_quantum::SymplecticMatrix;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
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

    #[test]
    fn seeded_matrix_pairs_match_exhaustive_dimacs_minimum() {
        let mut rng = SmallRng::seed_from_u64(0xC0DE_D157_A11C_E5E0);
        for case in 0..64 {
            let num_qubits = rng.random_range(1..=8);
            let num_checks = rng.random_range(1..=4);
            let num_logicals = rng.random_range(1..=3);
            let h = ParityCheckMatrix::from_dense(
                (0..num_checks)
                    .map(|_| {
                        (0..num_qubits)
                            .map(|_| u8::from(rng.random_bool(0.5)))
                            .collect()
                    })
                    .collect(),
            )
            .unwrap();
            let l = ParityCheckMatrix::from_dense(
                (0..num_logicals)
                    .map(|_| {
                        (0..num_qubits)
                            .map(|_| u8::from(rng.random_bool(0.5)))
                            .collect()
                    })
                    .collect(),
            )
            .unwrap();
            let problem = DistanceProblem::from_css_checks(&h, &l).unwrap();
            let exhaustive = exhaustive_dimacs_minimum(&problem);
            let connected = connected_cluster_code_distance(&h, &l, num_qubits);
            let bounded = bounded_enumeration_code_distance(&h, &l, num_qubits);

            assert_eq!(
                connected.as_ref().map(|result| result.distance),
                exhaustive,
                "distance mismatch in seeded case {case}: H={:?}, L={:?}",
                h.rows(),
                l.rows()
            );
            assert_eq!(
                bounded.as_ref().map(super::super::bounded_enumeration_distance::BoundedEnumerationDistance::upper_bound),
                exhaustive,
                "bounded-enumeration mismatch in seeded case {case}: H={:?}, L={:?}",
                h.rows(),
                l.rows()
            );
            if let Some(result) = bounded {
                assert!(result.is_certified());
                assert_eq!(
                    problem.verify_witness(result.witness()),
                    Ok(exhaustive.unwrap())
                );
            }
            assert_eq!(connected.is_some(), exhaustive.is_some());
        }
    }

    fn repetition_triad_dem() -> DetectorErrorModel {
        let mut dem = DetectorErrorModel::new();
        dem.add_observable(DemOutput::new(0));
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

    fn tiny_non_css_spec() -> StabilizerCodeSpec {
        StabilizerCodeSpec::builder(2)
            .check(Ys([0, 1]))
            .logical_z(Zs([0, 1]))
            .logical_x(X(0) & Z(1))
            .build_verified()
            .unwrap()
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
    fn empty_affine_row_with_target_one_is_encoded_unsatisfiable() {
        let problem = DistanceProblem {
            h: F2Matrix::zeros(1, 1),
            l: F2Matrix::zeros(0, 1),
            num_vars: 1,
            weight_mode: WeightMode::Bit,
            parity_targets: vec![1],
            require_logical_effect: false,
        };
        let encoding = problem.encode(Some(1));

        assert_eq!(encoding.groups[0].clauses, vec![Vec::<i32>::new()]);
        assert_eq!(
            solve_with_batsat(&encoding, problem.num_vars),
            SolverAnswer::Unsat
        );
        assert_eq!(
            problem.verify_witness(&[false]),
            Err(WitnessError::OddCheck { row: 0 })
        );
    }

    #[test]
    fn symplectic_dimacs_encoding_matches_qubit_support_predicate() {
        let problem = DistanceProblem::from_stabilizer_spec(&tiny_non_css_spec()).unwrap();
        assert_eq!(problem.num_vars(), 4);

        for bound in 0..=2 {
            let cnf = parse_dimacs(&problem.to_dimacs(bound));
            for mask in 0..1 << problem.num_vars() {
                let candidate = assignment(mask, problem.num_vars());
                let direct = problem
                    .verify_witness(&candidate)
                    .is_ok_and(|weight| weight <= bound);
                assert_eq!(
                    cnf_satisfied_with_primary(&cnf, &candidate),
                    direct,
                    "assignment {mask:#b} at qubit-support bound {bound}"
                );
            }
        }

        // Y on qubit 0 selects both symplectic bits but has physical support weight one.
        assert_eq!(problem.verify_witness(&[true, false, true, false]), Ok(1));
        assert_eq!(
            problem
                .to_wcnf()
                .lines()
                .filter(|line| line.starts_with("1 -"))
                .count(),
            2
        );
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
        let oracle = calculate_distance(&spec, &DistanceSearchConfig::css())
            .unwrap()
            .unwrap();
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
    fn css_projection_constructors_reject_non_css_spec() {
        let spec = StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        // Full non-CSS codes are supported by `from_stabilizer_spec`; only the CSS projections
        // reject mixed operators instead of silently dropping half of their support.
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
    fn batsat_certifies_five_qubit_symplectic_distance_and_logical_witness() {
        let mut spec =
            StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        let calculated = spec.calculate_distance().unwrap().unwrap().distance;
        let oracle = spec.distance().unwrap();
        assert_eq!(calculated, oracle);
        assert_eq!(oracle, 3);

        let problem = DistanceProblem::from_stabilizer_spec(&spec).unwrap();
        let certified = certified_distance(&problem, oracle).unwrap().unwrap();
        assert_eq!(certified.distance, oracle);
        assert_eq!(problem.verify_witness(&certified.witness), Ok(oracle));

        let mut witness_paulis = SymplecticMatrix::from_dense(vec![
            certified.witness.iter().map(|&bit| u8::from(bit)).collect(),
        ])
        .unwrap()
        .to_positive_paulis();
        let witness = witness_paulis.pop().unwrap();
        assert!(
            spec.stabilizers()
                .iter()
                .all(|stabilizer| witness.commutes_with(stabilizer))
        );
        assert!(
            spec.logical_zs()
                .iter()
                .chain(spec.logical_xs())
                .any(|logical| !witness.commutes_with(logical))
        );
        assert!(spec.is_logical_error(&witness));
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
    fn certification_handles_valid_zero_weight_problem_at_zero_bound() {
        let problem =
            DistanceProblem::coset_weight_problem(&ParityCheckMatrix::zeros(0, 1), &[0]).unwrap();
        let expected = CertifiedDistance {
            distance: 0,
            witness: vec![false],
            sat_certified: true,
            unsat_trusted_below: 0,
        };

        assert_eq!(certified_distance(&problem, 0), Ok(Some(expected.clone())));
        assert_eq!(
            problem.certify_distance_with(0, |_, _| panic!("zero-weight problem called solver")),
            Ok(Some(expected))
        );
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
        let oracle = spec.calculate_distance().unwrap().unwrap().distance;
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
    #[ignore = "timing probe for the batsat backend"]
    fn batsat_bivariate_bicycle_72_12_6_timing_probe() {
        let code = crate::BivariateBicycleCode::new(
            6,
            6,
            &[(3, 0), (0, 1), (0, 2)],
            &[(0, 3), (1, 0), (2, 0)],
        )
        .expect("the paper's [[72,12,6]] code is valid");
        assert_eq!(code.num_qubits(), 72);
        assert_eq!(code.num_logical_qubits(), 12);

        let problem = DistanceProblem::from_css_checks(code.hx(), code.logical_x()).unwrap();
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
        println!("BB [[72,12,6]] total: {:?}", total_started.elapsed());
        assert_eq!(certified.distance, 6);
        assert_eq!(problem.verify_witness(&certified.witness), Ok(6));
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

    fn five_qubit_spec() -> StabilizerCodeSpec {
        use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
        let pauli = |terms: &[(Pauli, usize)]| {
            PauliString::with_phase_and_paulis(
                QuarterPhase::PlusOne,
                terms.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
            )
        };
        StabilizerCodeSpec::new(
            5,
            vec![
                pauli(&[(Pauli::X, 0), (Pauli::Z, 1), (Pauli::Z, 2), (Pauli::X, 3)]),
                pauli(&[(Pauli::X, 1), (Pauli::Z, 2), (Pauli::Z, 3), (Pauli::X, 4)]),
                pauli(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::Z, 3), (Pauli::Z, 4)]),
                pauli(&[(Pauli::Z, 0), (Pauli::X, 1), (Pauli::X, 3), (Pauli::Z, 4)]),
            ],
            vec![pauli(&[
                (Pauli::Z, 0),
                (Pauli::Z, 1),
                (Pauli::Z, 2),
                (Pauli::Z, 3),
                (Pauli::Z, 4),
            ])],
            vec![pauli(&[
                (Pauli::X, 0),
                (Pauli::X, 1),
                (Pauli::X, 2),
                (Pauli::X, 3),
                (Pauli::X, 4),
            ])],
        )
        .unwrap()
    }

    #[test]
    fn classical_distance_matches_known_codes() {
        let hamming = steane_hamming_matrix();
        let certified = match certified_classical_distance(&hamming, 4).unwrap() {
            ClassicalDistanceSearchOutcome::Certified(certified) => certified,
            other => panic!("expected certified Hamming distance, got {other:?}"),
        };
        assert_eq!(certified.distance, 3);

        let repetition = ParityCheckMatrix::from_dense(vec![vec![1, 1, 0], vec![0, 1, 1]]).unwrap();
        let certified = match certified_classical_distance(&repetition, 3).unwrap() {
            ClassicalDistanceSearchOutcome::Certified(certified) => certified,
            other => panic!("expected certified repetition distance, got {other:?}"),
        };
        assert_eq!(certified.distance, 3);

        let enumerated = bounded_enumeration_classical_agreement(&hamming, 3);
        assert_eq!(enumerated, 3);
    }

    #[test]
    fn classical_distance_distinguishes_full_rank_from_budget_exhaustion() {
        let full_rank =
            ParityCheckMatrix::from_dense(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]])
                .unwrap();
        assert_eq!(
            certified_classical_distance(&full_rank, 2).unwrap(),
            ClassicalDistanceSearchOutcome::NoNonzeroCodeword
        );

        let repetition = ParityCheckMatrix::from_dense(vec![vec![1, 1, 0], vec![0, 1, 1]]).unwrap();
        assert_eq!(
            certified_classical_distance(&repetition, 2).unwrap(),
            ClassicalDistanceSearchOutcome::BudgetExhausted { max_weight: 2 }
        );
    }

    fn bounded_enumeration_classical_agreement(h: &ParityCheckMatrix, expected: usize) -> usize {
        let n = h.num_qubits();
        let identity = ParityCheckMatrix::from_dense(
            (0..n)
                .map(|column| {
                    let mut row = vec![0u8; n];
                    row[column] = 1;
                    row
                })
                .collect(),
        )
        .unwrap();
        match bounded_enumeration_code_distance(h, &identity, n).unwrap() {
            crate::BoundedEnumerationDistance::CertifiedByBounds { distance, .. } => {
                assert_eq!(distance, expected);
                distance
            }
            other @ crate::BoundedEnumerationDistance::LevelLimitReached { .. } => {
                panic!("expected certified classical distance, got {other:?}")
            }
        }
    }

    #[test]
    fn coset_weight_of_group_element_is_zero_and_of_all_ones_is_three() {
        let hamming = steane_hamming_matrix();
        let member = hamming.rows()[0].clone();
        let zero = certified_coset_weight(&hamming, &member, 7)
            .unwrap()
            .unwrap();
        assert_eq!(zero.distance, 0);

        let all_ones = vec![1u8; 7];
        let three = certified_coset_weight(&hamming, &all_ones, 7)
            .unwrap()
            .unwrap();
        assert_eq!(three.distance, 3);
        // Native re-verification of the witness against the coset condition.
        let problem = DistanceProblem::coset_weight_problem(&hamming, &all_ones).unwrap();
        assert_eq!(problem.verify_witness(&three.witness), Ok(3));
    }

    #[test]
    fn coset_weight_rejects_non_binary_representative() {
        let error = certified_coset_weight(&ParityCheckMatrix::zeros(0, 1), &[3], 1).unwrap_err();

        assert!(
            matches!(
                &error,
                CosetWeightError::Problem(DistanceProblemError::NonBinaryRepresentative {
                    index: 0,
                    value: 3
                })
            ),
            "unexpected error: {error}"
        );
        assert_eq!(
            error.to_string(),
            "coset representative entry at index 0 is 3, expected 0 or 1"
        );
    }

    #[test]
    fn coset_weight_agrees_with_brute_force_on_seeded_groups() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(0xC05E_75EE_D000_0001);
        for _ in 0..48 {
            let n = rng.random_range(3..=9);
            let row_count = rng.random_range(1..=3);
            let rows: Vec<Vec<u8>> = (0..row_count)
                .map(|_| (0..n).map(|_| u8::from(rng.random_bool(0.5))).collect())
                .collect();
            let Ok(group) = ParityCheckMatrix::from_dense(rows.clone()) else {
                continue;
            };
            let representative: Vec<u8> = (0..n).map(|_| u8::from(rng.random_bool(0.5))).collect();
            let certified = certified_coset_weight(&group, &representative, n)
                .unwrap()
                .expect("coset always has an element within weight n");
            let mut brute = usize::MAX;
            for mask in 0..(1usize << row_count) {
                let mut candidate = representative.clone();
                for (index, row) in rows.iter().enumerate() {
                    if mask & (1 << index) != 0 {
                        for (bit, r) in candidate.iter_mut().zip(row) {
                            *bit ^= r;
                        }
                    }
                }
                brute = brute.min(candidate.iter().map(|&bit| usize::from(bit)).sum::<usize>());
            }
            assert_eq!(certified.distance, brute);
        }
    }

    #[test]
    fn five_qubit_logical_x_coset_weight_is_three_by_qubit_support() {
        use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
        let spec = five_qubit_spec();
        let logical_x = PauliString::with_phase_and_paulis(
            QuarterPhase::PlusOne,
            (0..5).map(|q| (Pauli::X, QubitId::new(q))).collect(),
        );
        let certified = certified_stabilizer_coset_weight(&spec, &logical_x, 5)
            .unwrap()
            .unwrap();
        // Raw weight is 5; XXXXX times XZZXI equals IYYIX of qubit-support weight 3.
        assert_eq!(certified.distance, 3);
    }

    #[test]
    fn stabilizer_coset_weight_does_not_require_a_complete_logical_basis() {
        let spec = StabilizerCodeSpec::new(2, Vec::new(), vec![Z(0)], vec![X(0)]).unwrap();
        assert_eq!(
            spec.verify_logical_completeness(),
            Err(StabilizerCodeSpecError::IncompleteLogicalBasis {
                supplied_logical_pairs: 1,
                num_logical_qubits: 2,
            })
        );

        let certified = certified_stabilizer_coset_weight(&spec, &X(0), 2)
            .unwrap()
            .unwrap();
        assert_eq!(certified.distance, 1);
    }

    #[test]
    fn empty_stabilizer_group_preserves_symplectic_width() {
        let spec = StabilizerCodeSpec::new(1, Vec::new(), vec![Z(0)], vec![X(0)]).unwrap();

        let certified = certified_stabilizer_coset_weight(&spec, &X(0), 1)
            .unwrap()
            .unwrap();
        assert_eq!(certified.distance, 1);

        let profile = logical_generator_coset_weights(&spec, 1).unwrap();
        assert_eq!(
            profile
                .into_iter()
                .map(|entry| entry.unwrap().distance)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn generator_coset_minimum_can_exceed_code_distance() {
        let spec = StabilizerCodeSpec::new(
            2,
            Vec::new(),
            vec![Xs([0, 1]), Y(0) & Z(1)],
            vec![X(0) & Y(1), Zs([0, 1])],
        )
        .unwrap();
        spec.verify().unwrap();

        let profile = logical_generator_coset_weights(&spec, 2).unwrap();
        assert_eq!(
            profile
                .into_iter()
                .map(|entry| entry.unwrap().distance)
                .min(),
            Some(2)
        );
        let distance = crate::stabilizer_code_distance(&spec, 1).unwrap();
        match distance {
            crate::StabilizerDistanceSearchOutcome::Certified(result) => {
                assert_eq!(result.distance, 1);
            }
            other @ crate::StabilizerDistanceSearchOutcome::BudgetExhausted { .. } => {
                panic!("expected certified code distance, got {other:?}")
            }
        }
    }

    #[test]
    fn steane_logical_profile_is_all_threes_and_stabilizer_costs_zero() {
        let hamming = steane_hamming_matrix();
        let mut builder = crate::StabilizerCodeSpecBuilder::new(7);
        builder = builder.checks_from_css(&hamming, &hamming).unwrap();
        let spec = builder.build_with_discovered_logicals().unwrap();

        let profile = logical_generator_coset_weights(&spec, 7).unwrap();
        assert_eq!(profile.len(), 2);
        for entry in profile {
            assert_eq!(entry.unwrap().distance, 3);
        }
        let stabilizer = spec.stabilizers()[0].clone();
        let zero = certified_stabilizer_coset_weight(&spec, &stabilizer, 7)
            .unwrap()
            .unwrap();
        assert_eq!(zero.distance, 0);
    }

    #[test]
    fn yy_code_y_logical_coset_weight_is_one() {
        use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
        let pauli = |terms: &[(Pauli, usize)]| {
            PauliString::with_phase_and_paulis(
                QuarterPhase::PlusOne,
                terms.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
            )
        };
        let spec = StabilizerCodeSpec::new(
            2,
            vec![pauli(&[(Pauli::Y, 0), (Pauli::Y, 1)])],
            vec![pauli(&[(Pauli::Y, 0)])],
            vec![pauli(&[(Pauli::X, 0), (Pauli::Z, 1)])],
        )
        .unwrap();
        let certified = certified_stabilizer_coset_weight(&spec, &pauli(&[(Pauli::Y, 0)]), 2)
            .unwrap()
            .unwrap();
        assert_eq!(certified.distance, 1);
    }
}
