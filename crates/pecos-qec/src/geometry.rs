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

//! QEC code geometry framework.
//!
//! This module provides code-agnostic abstractions for stabilizer checks
//! that can be used across different QEC codes (surface codes, color codes, etc.).
//!
//! The framework bridges the abstract stabilizer algebra level and the circuit level:
//! - Abstract level: Stabilizer definitions feed into `StabilizerCode` for verification
//! - Circuit level: Check schedules define measurement order for syndrome extraction

use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
use std::collections::HashMap;

/// A Pauli operator on a specific qubit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PauliOp {
    /// The qubit index this operator acts on.
    pub qubit: usize,
    /// The Pauli type (X, Y, or Z).
    pub pauli: Pauli,
}

impl PauliOp {
    /// Create a new Pauli operator.
    #[inline]
    #[must_use]
    pub fn new(qubit: usize, pauli: Pauli) -> Self {
        Self { qubit, pauli }
    }

    /// Create an X operator on the given qubit.
    #[inline]
    #[must_use]
    pub fn x(qubit: usize) -> Self {
        Self::new(qubit, Pauli::X)
    }

    /// Create a Y operator on the given qubit.
    #[inline]
    #[must_use]
    pub fn y(qubit: usize) -> Self {
        Self::new(qubit, Pauli::Y)
    }

    /// Create a Z operator on the given qubit.
    #[inline]
    #[must_use]
    pub fn z(qubit: usize) -> Self {
        Self::new(qubit, Pauli::Z)
    }
}

impl std::fmt::Display for PauliOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}{}", self.pauli, self.qubit)
    }
}

/// Color label for color codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StabilizerColor {
    Red,
    Green,
    Blue,
}

impl std::fmt::Display for StabilizerColor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Red => write!(f, "red"),
            Self::Green => write!(f, "green"),
            Self::Blue => write!(f, "blue"),
        }
    }
}

/// A generic stabilizer check.
///
/// This represents a stabilizer measurement that can be used in any
/// CSS or non-CSS quantum error correction code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabilizerCheck {
    /// Unique identifier for this check.
    pub index: usize,
    /// Sequence of Pauli operators defining the stabilizer.
    pub paulis: Vec<PauliOp>,
    /// Optional color for color codes.
    pub color: Option<StabilizerColor>,
    /// Whether this is a boundary stabilizer.
    pub is_boundary: bool,
    /// Optional 2D position for visualization/layout.
    pub position: Option<(i32, i32)>,
}

impl StabilizerCheck {
    /// Create a new stabilizer check.
    #[must_use]
    pub fn new(index: usize, paulis: Vec<PauliOp>) -> Self {
        Self {
            index,
            paulis,
            color: None,
            is_boundary: false,
            position: None,
        }
    }

    /// Create an X-type stabilizer check on the given qubits.
    #[must_use]
    pub fn x_check(index: usize, qubits: &[usize]) -> Self {
        Self::new(index, qubits.iter().map(|&q| PauliOp::x(q)).collect())
    }

    /// Create a Z-type stabilizer check on the given qubits.
    #[must_use]
    pub fn z_check(index: usize, qubits: &[usize]) -> Self {
        Self::new(index, qubits.iter().map(|&q| PauliOp::z(q)).collect())
    }

    /// Create a stabilizer check from a Pauli string and qubit list.
    ///
    /// If `pauli_str` is a single character, it's applied to all qubits.
    ///
    /// # Errors
    /// Returns an error if the Pauli string length doesn't match the qubit count,
    /// or if the Pauli string is empty or contains invalid characters.
    #[allow(clippy::missing_panics_doc)] // Panic unreachable due to empty check
    pub fn from_string(index: usize, pauli_str: &str, qubits: &[usize]) -> Result<Self, String> {
        if pauli_str.is_empty() {
            return Err("Pauli string cannot be empty".to_string());
        }
        let paulis_chars: Vec<char> = if pauli_str.len() == 1 {
            vec![pauli_str.chars().next().expect("checked non-empty above"); qubits.len()]
        } else {
            pauli_str.chars().collect()
        };

        if paulis_chars.len() != qubits.len() {
            return Err(format!(
                "Pauli string length ({}) must match number of qubits ({})",
                paulis_chars.len(),
                qubits.len()
            ));
        }

        let paulis = paulis_chars
            .iter()
            .zip(qubits.iter())
            .map(|(&c, &q)| {
                let pauli = match c {
                    'X' | 'x' => Pauli::X,
                    'Y' | 'y' => Pauli::Y,
                    'Z' | 'z' => Pauli::Z,
                    'I' | 'i' => Pauli::I,
                    _ => return Err(format!("Invalid Pauli character: {c}")),
                };
                Ok(PauliOp::new(q, pauli))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self::new(index, paulis))
    }

    /// Set the color of this check.
    #[must_use]
    pub fn with_color(mut self, color: StabilizerColor) -> Self {
        self.color = Some(color);
        self
    }

    /// Mark this check as a boundary stabilizer.
    #[must_use]
    pub fn as_boundary(mut self) -> Self {
        self.is_boundary = true;
        self
    }

    /// Set the 2D position of this check.
    #[must_use]
    pub fn at_position(mut self, row: i32, col: i32) -> Self {
        self.position = Some((row, col));
        self
    }

    /// Number of qubits this check acts on (weight).
    #[inline]
    #[must_use]
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|p| p.pauli != Pauli::I).count()
    }

    /// Qubit indices this check acts on.
    #[must_use]
    pub fn qubits(&self) -> Vec<usize> {
        self.paulis.iter().map(|p| p.qubit).collect()
    }

    /// Get the Pauli string representation (e.g., "XXXX", "ZZZZ").
    #[must_use]
    pub fn pauli_string(&self) -> String {
        self.paulis
            .iter()
            .map(|p| match p.pauli {
                Pauli::I => 'I',
                Pauli::X => 'X',
                Pauli::Y => 'Y',
                Pauli::Z => 'Z',
            })
            .collect()
    }

    /// Check if this is a CSS stabilizer (all same Pauli type, excluding identity).
    #[must_use]
    pub fn is_css(&self) -> bool {
        let non_identity: Vec<_> = self.paulis.iter().filter(|p| p.pauli != Pauli::I).collect();
        if non_identity.is_empty() {
            return true;
        }
        let first = non_identity[0].pauli;
        non_identity.iter().all(|p| p.pauli == first)
    }

    /// Convert this check to a [`PauliString`] for use with [`StabilizerCode`].
    #[must_use]
    pub fn to_pauli_string(&self) -> PauliString {
        let paulis: Vec<(Pauli, QubitId)> = self
            .paulis
            .iter()
            .filter(|p| p.pauli != Pauli::I)
            .map(|p| (p.pauli, QubitId::new(p.qubit)))
            .collect();
        PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
    }
}

impl std::fmt::Display for StabilizerCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Check[{}]: {}", self.index, self.pauli_string())?;
        if let Some(color) = &self.color {
            write!(f, " ({color})")?;
        }
        if self.is_boundary {
            write!(f, " [boundary]")?;
        }
        Ok(())
    }
}

/// A schedule for measuring multiple stabilizer checks.
///
/// Organizes checks into rounds that can be measured in parallel,
/// respecting qubit constraints.
#[derive(Clone, Debug)]
pub struct CheckSchedule {
    /// List of rounds, each containing checks that can run in parallel.
    pub rounds: Vec<Vec<StabilizerCheck>>,
}

impl CheckSchedule {
    /// Create a new empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self { rounds: Vec::new() }
    }

    /// Create a sequential schedule (one check per round).
    #[must_use]
    pub fn sequential(checks: Vec<StabilizerCheck>) -> Self {
        Self {
            rounds: checks.into_iter().map(|c| vec![c]).collect(),
        }
    }

    /// Create a schedule that parallelizes checks by color.
    ///
    /// Checks with the same color are placed in different rounds,
    /// allowing checks of different colors to run in parallel.
    #[must_use]
    pub fn parallel_by_color(checks: Vec<StabilizerCheck>) -> Self {
        let mut by_color: HashMap<Option<StabilizerColor>, Vec<StabilizerCheck>> = HashMap::new();

        for check in checks {
            by_color.entry(check.color).or_default().push(check);
        }

        let max_len = by_color.values().map(Vec::len).max().unwrap_or(0);

        let mut rounds = Vec::new();
        for i in 0..max_len {
            let round_checks: Vec<StabilizerCheck> = by_color
                .values()
                .filter_map(|color_checks| color_checks.get(i).cloned())
                .collect();
            if !round_checks.is_empty() {
                rounds.push(round_checks);
            }
        }

        Self { rounds }
    }

    /// Create a schedule that parallelizes X and Z checks.
    ///
    /// All X checks run in the first round, all Z checks in the second.
    #[must_use]
    pub fn parallel_xz(checks: Vec<StabilizerCheck>) -> Self {
        let mut x_checks = Vec::new();
        let mut z_checks = Vec::new();
        let mut other_checks = Vec::new();

        for check in checks {
            if check.pauli_string().chars().all(|c| c == 'X' || c == 'I') {
                x_checks.push(check);
            } else if check.pauli_string().chars().all(|c| c == 'Z' || c == 'I') {
                z_checks.push(check);
            } else {
                other_checks.push(check);
            }
        }

        let mut rounds = Vec::new();
        if !x_checks.is_empty() {
            rounds.push(x_checks);
        }
        if !z_checks.is_empty() {
            rounds.push(z_checks);
        }
        for check in other_checks {
            rounds.push(vec![check]);
        }

        Self { rounds }
    }

    /// Add a round of checks to the schedule.
    pub fn add_round(&mut self, checks: Vec<StabilizerCheck>) {
        self.rounds.push(checks);
    }

    /// Total number of checks in the schedule.
    #[must_use]
    pub fn total_checks(&self) -> usize {
        self.rounds.iter().map(Vec::len).sum()
    }

    /// Number of rounds in the schedule.
    #[must_use]
    pub fn num_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Get all checks as a flat list.
    #[must_use]
    pub fn all_checks(&self) -> Vec<&StabilizerCheck> {
        self.rounds.iter().flatten().collect()
    }

    /// Convert all checks to [`PauliString`] for use with [`StabilizerCode`].
    #[must_use]
    pub fn to_pauli_strings(&self) -> Vec<PauliString> {
        self.all_checks()
            .into_iter()
            .map(StabilizerCheck::to_pauli_string)
            .collect()
    }
}

impl Default for CheckSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// A logical operator definition for a QEC code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalOperator {
    /// The type of logical operator ("X" or "Z").
    pub op_type: String,
    /// Data qubits the operator acts on.
    pub data_qubits: Vec<usize>,
    /// The Pauli type applied to each qubit (if CSS, all same type).
    pub pauli_type: Pauli,
}

impl LogicalOperator {
    /// Create a logical X operator.
    #[must_use]
    pub fn x(data_qubits: Vec<usize>) -> Self {
        Self {
            op_type: "X".to_string(),
            data_qubits,
            pauli_type: Pauli::X,
        }
    }

    /// Create a logical Z operator.
    #[must_use]
    pub fn z(data_qubits: Vec<usize>) -> Self {
        Self {
            op_type: "Z".to_string(),
            data_qubits,
            pauli_type: Pauli::Z,
        }
    }

    /// Convert to a [`PauliString`].
    #[must_use]
    pub fn to_pauli_string(&self) -> PauliString {
        let paulis: Vec<(Pauli, QubitId)> = self
            .data_qubits
            .iter()
            .map(|&q| (self.pauli_type, QubitId::new(q)))
            .collect();
        PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::PauliOperator;

    #[test]
    fn test_stabilizer_check_creation() {
        let check = StabilizerCheck::x_check(0, &[0, 1, 2, 3]);
        assert_eq!(check.weight(), 4);
        assert_eq!(check.pauli_string(), "XXXX");
        assert!(check.is_css());
    }

    #[test]
    fn test_stabilizer_check_from_string() {
        let check = StabilizerCheck::from_string(0, "XYZZ", &[0, 1, 2, 3]).unwrap();
        assert_eq!(check.pauli_string(), "XYZZ");
        assert!(!check.is_css());
    }

    #[test]
    fn test_stabilizer_check_single_char() {
        let check = StabilizerCheck::from_string(0, "Z", &[0, 1, 2, 3]).unwrap();
        assert_eq!(check.pauli_string(), "ZZZZ");
        assert!(check.is_css());
    }

    #[test]
    fn test_check_schedule_sequential() {
        let checks = vec![
            StabilizerCheck::x_check(0, &[0, 1]),
            StabilizerCheck::z_check(1, &[1, 2]),
        ];
        let schedule = CheckSchedule::sequential(checks);
        assert_eq!(schedule.num_rounds(), 2);
        assert_eq!(schedule.total_checks(), 2);
    }

    #[test]
    fn test_check_schedule_parallel_xz() {
        let checks = vec![
            StabilizerCheck::x_check(0, &[0, 1]),
            StabilizerCheck::x_check(1, &[2, 3]),
            StabilizerCheck::z_check(2, &[0, 2]),
            StabilizerCheck::z_check(3, &[1, 3]),
        ];
        let schedule = CheckSchedule::parallel_xz(checks);
        assert_eq!(schedule.num_rounds(), 2);
        assert_eq!(schedule.rounds[0].len(), 2); // X checks
        assert_eq!(schedule.rounds[1].len(), 2); // Z checks
    }

    #[test]
    fn test_stabilizer_check_to_pauli_string() {
        let check = StabilizerCheck::x_check(0, &[0, 2, 4]);
        let ps = check.to_pauli_string();
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_logical_operator() {
        let logical_x = LogicalOperator::x(vec![0, 1, 2]);
        let ps = logical_x.to_pauli_string();
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_stabilizer_check_with_color() {
        let check = StabilizerCheck::x_check(0, &[0, 1, 2])
            .with_color(StabilizerColor::Red)
            .as_boundary();
        assert_eq!(check.color, Some(StabilizerColor::Red));
        assert!(check.is_boundary);
    }
}
