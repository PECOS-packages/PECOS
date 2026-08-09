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

//! Connected-cluster distance searches for stabilizer codes.
//!
//! The reduction uses the Connected Cluster method of
//! [arXiv:2603.22532](https://arxiv.org/abs/2603.22532). A column of a binary check/logical pair
//! becomes a unit-weight mechanism: its check-row support is the detector set and its logical-row
//! support is the output set. The fault-distance engine can therefore search code distance without
//! duplicating the connected-cluster enumeration.

use crate::fault_tolerance::dem_builder::FaultMechanism;
use crate::fault_tolerance::fault_distance::connected_cluster_mechanism_distance;
#[cfg(test)]
use crate::fault_tolerance::fault_distance::connected_cluster_mechanism_distance_at_weight;
use crate::{
    DistanceProblem, DistanceProblemError, DistanceResult, FaultDistanceResult, ParityCheckMatrix,
    StabilizerCodeSpec,
};
use pecos_core::{Pauli, PauliOperator, PauliString, QuarterPhase, QubitId};
use pecos_quantum::F2Matrix;

/// Outcome of a budgeted connected-cluster stabilizer-code distance search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StabilizerDistanceSearchOutcome {
    /// The exact distance and a minimum-weight logical operator were found.
    Certified(DistanceResult),
    /// No logical operator was found through the requested physical weight.
    BudgetExhausted {
        /// Largest physical weight included in the search.
        max_weight: usize,
    },
}

fn mechanisms_from_matrices(h: &F2Matrix, l: &F2Matrix) -> Vec<FaultMechanism> {
    assert_eq!(
        h.num_cols(),
        l.num_cols(),
        "code-distance matrices must have matching widths"
    );
    let num_detectors =
        u32::try_from(h.num_rows()).expect("check row count fits in the u32 id space");
    let num_outputs =
        u32::try_from(l.num_rows()).expect("logical row count fits in the u32 id space");

    (0..h.num_cols())
        .map(|column| {
            FaultMechanism::from_unsorted(
                (0..num_detectors).filter(|&row| h.get(row as usize, column) == 1),
                (0..num_outputs).filter(|&row| l.get(row as usize, column) == 1),
            )
        })
        .collect()
}

fn matrix_distance(h: &F2Matrix, l: &F2Matrix, max_weight: usize) -> Option<FaultDistanceResult> {
    let mechanisms = mechanisms_from_matrices(h, l);
    let num_outputs =
        u32::try_from(l.num_rows()).expect("logical row count fits in the u32 id space");
    connected_cluster_mechanism_distance(&mechanisms, num_outputs, max_weight)
}

/// Computes connected-cluster distance for a binary `(H, L)` pair.
///
/// Column `j` is one unit-weight mechanism, so returned mechanism indices are qubit indices.
/// `H e = 0` enforces an undetectable support and `L e != 0` enforces a nontrivial logical effect.
///
/// # Panics
///
/// Panics if the matrices have different widths or either row count exceeds the `u32` id space.
#[must_use]
pub fn connected_cluster_code_distance(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    max_weight: usize,
) -> Option<FaultDistanceResult> {
    matrix_distance(h.matrix(), l.matrix(), max_weight)
}

/// Computes pure-X distance for a CSS-form stabilizer code.
///
/// # Errors
///
/// Returns the same CSS-form and bounds errors as
/// [`DistanceProblem::from_css_code_x_distance`].
pub fn x_distance(
    code: &StabilizerCodeSpec,
    max_weight: usize,
) -> Result<Option<FaultDistanceResult>, DistanceProblemError> {
    let problem = DistanceProblem::from_css_code_x_distance(code)?;
    let (h, l) = problem.matrices();
    Ok(matrix_distance(h, l, max_weight))
}

/// Computes pure-Z distance for a CSS-form stabilizer code.
///
/// # Errors
///
/// Returns the same CSS-form and bounds errors as
/// [`DistanceProblem::from_css_code_z_distance`].
pub fn z_distance(
    code: &StabilizerCodeSpec,
    max_weight: usize,
) -> Result<Option<FaultDistanceResult>, DistanceProblemError> {
    let problem = DistanceProblem::from_css_code_z_distance(code)?;
    let (h, l) = problem.matrices();
    Ok(matrix_distance(h, l, max_weight))
}

fn single_qubit_pauli(pauli: Pauli, qubit: usize) -> PauliString {
    PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, vec![(pauli, QubitId::new(qubit))])
}

pub(crate) fn mechanisms_from_stabilizer_code(
    code: &StabilizerCodeSpec,
) -> Result<Vec<FaultMechanism>, DistanceProblemError> {
    // Reuse the symplectic constructor's established out-of-range validation. Logical
    // completeness is enforced by each ordinary-code entry point; subsystem distance applies
    // its gauge-aware count before reaching this shared mechanism construction.
    DistanceProblem::from_stabilizer_spec_without_logical_completeness(code)?;
    let logicals: Vec<_> = code.logical_zs().iter().chain(code.logical_xs()).collect();

    Ok((0..code.num_qubits())
        .flat_map(|qubit| {
            [Pauli::X, Pauli::Y, Pauli::Z].map(|pauli| {
                let error = single_qubit_pauli(pauli, qubit);
                FaultMechanism::from_unsorted(
                    code.stabilizers()
                        .iter()
                        .enumerate()
                        .filter(|(_, stabilizer)| error.anticommutes_with(stabilizer))
                        .map(|(index, _)| {
                            u32::try_from(index).expect("stabilizer count fits in the u32 id space")
                        }),
                    logicals
                        .iter()
                        .enumerate()
                        .filter(|(_, logical)| error.anticommutes_with(logical))
                        .map(|(index, _)| {
                            u32::try_from(index).expect("logical count fits in the u32 id space")
                        }),
                )
            })
        })
        .collect())
}

fn xor_pauli(left: Pauli, right: Pauli) -> Pauli {
    match (left, right) {
        (Pauli::I, pauli) | (pauli, Pauli::I) => pauli,
        (Pauli::X, Pauli::X) | (Pauli::Y, Pauli::Y) | (Pauli::Z, Pauli::Z) => Pauli::I,
        (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::X) => Pauli::Z,
        (Pauli::X, Pauli::Z) | (Pauli::Z, Pauli::X) => Pauli::Y,
        (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::Y) => Pauli::X,
    }
}

fn mechanism_witness_to_pauli(num_qubits: usize, mechanism_indices: &[usize]) -> PauliString {
    let mut paulis = vec![Pauli::I; num_qubits];
    for &mechanism_index in mechanism_indices {
        let qubit = mechanism_index / 3;
        let pauli = [Pauli::X, Pauli::Y, Pauli::Z][mechanism_index % 3];
        paulis[qubit] = xor_pauli(paulis[qubit], pauli);
    }
    PauliString::with_phase_and_paulis(
        QuarterPhase::PlusOne,
        paulis
            .into_iter()
            .enumerate()
            .filter(|(_, pauli)| *pauli != Pauli::I)
            .map(|(qubit, pauli)| (pauli, QubitId::new(qubit)))
            .collect(),
    )
}

/// Computes connected-cluster distance for any stabilizer code specification.
///
/// Each qubit supplies X, Y, and Z mechanisms whose detector and output sets are their
/// anticommuting stabilizers and logicals. Although the engine counts mechanisms, a minimum
/// solution never needs two mechanisms on one qubit: the binary effect rows for X and Z XOR
/// exactly to the Y row (and likewise for the other pairs), so replacing either pair by the third
/// mechanism gives the same effect at strictly lower weight. Thus a physical-support minimum
/// exists among the mechanism-count minima, and the connected-component theorem of
/// [arXiv:2603.22532](https://arxiv.org/abs/2603.22532) applies.
///
/// # Errors
///
/// Returns a stabilizer-spec error if the logical basis is incomplete or the code encodes no
/// logical qubits. Returns [`DistanceProblemError::QubitOutOfRange`] if an operator addresses a
/// qubit outside the declared code width.
///
/// # Panics
///
/// Panics if the stabilizer or logical count exceeds the `u32` id space.
pub fn stabilizer_code_distance(
    code: &StabilizerCodeSpec,
    max_weight: usize,
) -> Result<StabilizerDistanceSearchOutcome, DistanceProblemError> {
    code.verify_as_complete_code()?;
    // No qubit-support witness can weigh more than the code's qubit count, so a larger
    // budget is semantically exhaustive: clamp it. A verified complete spec encodes at
    // least one logical qubit, and its logical operators weigh at most `num_qubits`, so
    // a completed search at the clamped budget always certifies — `BudgetExhausted`
    // therefore always carries a budget below `num_qubits`, keeping the
    // `lower_bound = max_weight + 1` invariant free of overflow.
    let max_weight = max_weight.min(code.num_qubits());
    Ok(
        match stabilizer_code_distance_without_logical_completeness(code, max_weight)? {
            Some(result) => StabilizerDistanceSearchOutcome::Certified(result),
            None => StabilizerDistanceSearchOutcome::BudgetExhausted { max_weight },
        },
    )
}

/// Searches after a caller has validated either ordinary or subsystem logical completeness.
pub(crate) fn stabilizer_code_distance_without_logical_completeness(
    code: &StabilizerCodeSpec,
    max_weight: usize,
) -> Result<Option<DistanceResult>, DistanceProblemError> {
    let mechanisms = mechanisms_from_stabilizer_code(code)?;
    let num_outputs = u32::try_from(code.logical_zs().len() + code.logical_xs().len())
        .expect("logical count fits in the u32 id space");
    let Some(result) = connected_cluster_mechanism_distance(&mechanisms, num_outputs, max_weight)
    else {
        return Ok(None);
    };
    let min_weight_operator =
        mechanism_witness_to_pauli(code.num_qubits(), &result.mechanism_indices);
    debug_assert_eq!(min_weight_operator.weight(), result.distance);
    Ok(Some(DistanceResult {
        distance: result.distance,
        min_weight_operator,
    }))
}

#[cfg(test)]
fn matrix_distance_at_weight(
    h: &F2Matrix,
    l: &F2Matrix,
    weight: usize,
) -> Option<FaultDistanceResult> {
    let mechanisms = mechanisms_from_matrices(h, l);
    let num_outputs =
        u32::try_from(l.num_rows()).expect("logical row count fits in the u32 id space");
    connected_cluster_mechanism_distance_at_weight(&mechanisms, num_outputs, weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        StabilizerCode, StabilizerCodeSpecError, bounded_enumeration_stabilizer_distance,
        bounded_enumeration_x_distance, bounded_enumeration_z_distance, certified_distance,
    };
    use pecos_core::{PauliOperator, X, Xs, Y, Ys, Z, Zs};
    use std::time::Instant;

    fn steane_spec() -> StabilizerCodeSpec {
        StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::steane()).unwrap()
    }

    fn yy_code() -> StabilizerCodeSpec {
        // Y0 commutes with Y0Y1. X0Z1 also commutes because it anticommutes once with Y on
        // each qubit, while Y0 and X0Z1 anticommute on qubit 0. Thus these are a valid logical
        // pair. Every single-qubit X or Z anticommutes with Y0Y1, whereas Y0 and Y1 commute and
        // anticommute with X0Z1, so every minimum logical is a single-qubit Y.
        StabilizerCodeSpec::builder(2)
            .check(Ys([0, 1]))
            .logical_z(Y(0))
            .logical_x(X(0) & Z(1))
            .build_verified()
            .unwrap()
    }

    fn incomplete_five_qubit_spec() -> StabilizerCodeSpec {
        StabilizerCodeSpec::new(
            5,
            vec![Xs(1..=4), Zs(1..=4)],
            vec![Zs([1, 2])],
            vec![Xs([2, 3])],
        )
        .unwrap()
    }

    fn incomplete_basis_error() -> DistanceProblemError {
        DistanceProblemError::StabilizerSpec(StabilizerCodeSpecError::IncompleteLogicalBasis {
            supplied_logical_pairs: 1,
            num_logical_qubits: 3,
        })
    }

    fn certified(outcome: StabilizerDistanceSearchOutcome) -> DistanceResult {
        match outcome {
            StabilizerDistanceSearchOutcome::Certified(result) => result,
            StabilizerDistanceSearchOutcome::BudgetExhausted { max_weight } => {
                panic!("expected certified distance, exhausted weight {max_weight}")
            }
        }
    }

    #[test]
    fn mismatched_logical_lists_are_rejected_not_searched_one_sided() {
        // A logical Z added through the mutator without its X partner: the pair checks
        // iterate over the shorter list, so nothing pairs, completeness counts only the
        // Z side, and an unchecked search would certify distance 3 via the X coset while
        // the weight-1 logical Z0 stays invisible to the Z-only searches.
        let mut spec =
            StabilizerCodeSpec::from_stabilizers(3, vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        spec.add_logical_z(Zs([0]));
        assert_eq!(
            stabilizer_code_distance(&spec, 3).unwrap_err(),
            DistanceProblemError::StabilizerSpec(StabilizerCodeSpecError::MismatchedLogicalLists {
                num_logical_zs: 1,
                num_logical_xs: 0,
            })
        );
    }

    #[test]
    fn logical_anticommuting_with_a_stabilizer_is_rejected_not_certified() {
        // Frozen qubit 0 tensor the Steane code (true distance 3). The supplied count is
        // complete (k = 1), but the logical X anticommutes with the frozen-qubit stabilizer
        // Z0, so an unchecked search certifies the stabilizer Z0 as a weight-1 "logical".
        let spec = StabilizerCodeSpec::new(
            8,
            vec![
                Zs([0]),
                Xs([1, 3, 5, 7]),
                Xs([2, 3, 6, 7]),
                Xs([4, 5, 6, 7]),
                Zs([1, 3, 5, 7]),
                Zs([2, 3, 6, 7]),
                Zs([4, 5, 6, 7]),
            ],
            vec![Zs(1..=7)],
            vec![Xs([0])],
        )
        .unwrap();
        assert_eq!(
            stabilizer_code_distance(&spec, 3).unwrap_err(),
            DistanceProblemError::StabilizerSpec(
                StabilizerCodeSpecError::LogicalAnticommutesWithStabilizer {
                    logical: "X0".to_string(),
                    stabilizer: 0,
                }
            )
        );
    }

    #[test]
    fn duplicate_logical_pairs_are_rejected_not_overstated() {
        // Bare qubit 0 direct-summed with the five-qubit code on qubits 1-5 (true distance
        // 1 via the bare qubit). Supplying the five-qubit logical pair twice satisfies the
        // count (k = 2) while hiding the bare logical qubit; an unchecked search reports 3.
        // X of pair 0 anticommutes with Z of pair 1 (the same coset), violating delta_ij.
        let five_qubit_stabilizers = vec![
            Xs([1, 4]) * Zs([2, 3]),
            Xs([2, 5]) * Zs([3, 4]),
            Xs([1, 3]) * Zs([4, 5]),
            Xs([2, 4]) * Zs([1, 5]),
        ];
        let spec = StabilizerCodeSpec::new(
            6,
            five_qubit_stabilizers,
            vec![Zs(1..=5), Zs(1..=5)],
            vec![Xs(1..=5), Xs(1..=5)],
        )
        .unwrap();
        assert_eq!(
            stabilizer_code_distance(&spec, 3).unwrap_err(),
            DistanceProblemError::StabilizerSpec(StabilizerCodeSpecError::CrossLogicalAnticommute(
                0, 1
            ))
        );
    }

    #[test]
    fn anticommuting_stabilizers_are_rejected_not_certified() {
        // X0 and Z0 anticommute: there is no common code space, yet independence and the
        // logical count both pass. An unchecked search certifies distance 1 for a spec
        // that describes nothing.
        let spec = StabilizerCodeSpec::new(3, vec![Xs([0]), Zs([0])], vec![Zs([1])], vec![Xs([1])])
            .unwrap();
        assert_eq!(
            stabilizer_code_distance(&spec, 1).unwrap_err(),
            DistanceProblemError::StabilizerSpec(StabilizerCodeSpecError::StabilizersAnticommute(
                0, 1
            ))
        );
    }

    #[test]
    fn maximum_budget_is_clamped_and_always_certifies_a_valid_spec() {
        // A budget beyond the qubit count is semantically exhaustive, and a verified
        // complete spec always has a logical of weight <= n, so the search certifies
        // instead of reporting a budget exhaustion whose lower bound would overflow.
        let outcome = stabilizer_code_distance(&steane_spec(), usize::MAX).unwrap();
        assert_eq!(certified(outcome).distance, 3);
    }

    #[test]
    fn incomplete_logical_basis_is_rejected_by_all_spec_distance_entry_points() {
        let spec = incomplete_five_qubit_spec();

        assert_eq!(
            stabilizer_code_distance(&spec, 5).unwrap_err(),
            incomplete_basis_error()
        );
        assert_eq!(
            DistanceProblem::from_stabilizer_spec(&spec).unwrap_err(),
            incomplete_basis_error()
        );
        assert_eq!(
            DistanceProblem::from_css_code_x_distance(&spec).unwrap_err(),
            incomplete_basis_error()
        );
        assert_eq!(
            DistanceProblem::from_css_code_z_distance(&spec).unwrap_err(),
            incomplete_basis_error()
        );
        assert_eq!(x_distance(&spec, 5).unwrap_err(), incomplete_basis_error());
        assert_eq!(z_distance(&spec, 5).unwrap_err(), incomplete_basis_error());
        assert_eq!(
            bounded_enumeration_x_distance(&spec, 5).unwrap_err(),
            incomplete_basis_error()
        );
        assert_eq!(
            bounded_enumeration_z_distance(&spec, 5).unwrap_err(),
            incomplete_basis_error()
        );
        assert_eq!(
            bounded_enumeration_stabilizer_distance(&spec, 5).unwrap_err(),
            incomplete_basis_error()
        );
    }

    #[test]
    fn stabilizer_distance_distinguishes_budget_exhaustion_and_no_logical_qubits() {
        let five_qubit =
            StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        assert_eq!(
            stabilizer_code_distance(&five_qubit, 2).unwrap(),
            StabilizerDistanceSearchOutcome::BudgetExhausted { max_weight: 2 }
        );

        let stabilizer_state = StabilizerCodeSpec::builder(1).check(Z(0)).build().unwrap();
        assert_eq!(
            stabilizer_code_distance(&stabilizer_state, 1).unwrap_err(),
            DistanceProblemError::StabilizerSpec(StabilizerCodeSpecError::NoLogicalQubits)
        );
    }

    #[test]
    fn steane_css_distances_agree_with_sat_and_weight_search() {
        let mut spec = steane_spec();
        let searched = spec.calculate_distance().unwrap().unwrap();
        assert_eq!(searched.distance, 3);
        assert_eq!(spec.distance(), Some(3));

        let cases = [
            (
                x_distance(&spec, 3).unwrap(),
                DistanceProblem::from_css_code_x_distance(&spec).unwrap(),
            ),
            (
                z_distance(&spec, 3).unwrap(),
                DistanceProblem::from_css_code_z_distance(&spec).unwrap(),
            ),
        ];
        for (connected, problem) in cases {
            let connected = connected.unwrap();
            let certified = certified_distance(&problem, 3).unwrap().unwrap();
            assert_eq!(connected.distance, 3);
            assert_eq!(certified.distance, connected.distance);
        }
    }

    #[test]
    fn css_conveniences_reject_non_css_codes() {
        let spec = StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        assert!(matches!(
            x_distance(&spec, 3),
            Err(DistanceProblemError::NonCssOperator { .. })
        ));
        assert!(matches!(
            z_distance(&spec, 3),
            Err(DistanceProblemError::NonCssOperator { .. })
        ));
    }

    #[test]
    fn five_qubit_distance_agrees_with_sat_and_weight_search() {
        let mut spec =
            StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::five_qubit()).unwrap();
        let searched = spec.calculate_distance().unwrap().unwrap();
        let connected = certified(stabilizer_code_distance(&spec, 3).unwrap());
        let problem = DistanceProblem::from_stabilizer_spec(&spec).unwrap();
        let certified = certified_distance(&problem, 3).unwrap().unwrap();

        assert_eq!(searched.distance, 3);
        assert_eq!(spec.distance(), Some(3));
        assert_eq!(connected.distance, searched.distance);
        assert_eq!(certified.distance, connected.distance);
        assert_eq!(connected.min_weight_operator.weight(), 3);
        assert!(spec.commutes_with_all_stabilizers(&connected.min_weight_operator));
        assert!(spec.anticommutes_with_logical(&connected.min_weight_operator));
        assert!(spec.is_logical_error(&connected.min_weight_operator));
    }

    #[test]
    fn yy_code_requires_a_y_mechanism_at_distance_one() {
        let spec = yy_code();
        let connected = certified(stabilizer_code_distance(&spec, 1).unwrap());

        assert_eq!(connected.distance, 1);
        assert!(
            connected.min_weight_operator == Y(0) || connected.min_weight_operator == Y(1),
            "minimum witness must be Y on one qubit: {:?}",
            connected.min_weight_operator
        );
        assert!(spec.is_logical_error(&connected.min_weight_operator));
    }

    #[test]
    fn y_mechanism_effects_are_exact_for_each_qubit() {
        let mechanisms = mechanisms_from_stabilizer_code(&yy_code()).unwrap();
        assert_eq!(mechanisms.len(), 6);

        let y0 = &mechanisms[1];
        assert!(y0.detectors.is_empty());
        assert_eq!(y0.dem_outputs.as_slice(), &[1]);

        let y1 = &mechanisms[4];
        assert!(y1.detectors.is_empty());
        assert_eq!(y1.dem_outputs.as_slice(), &[1]);
    }

    #[test]
    fn matrix_search_is_deterministic_bounded_and_peels_unique_detectors() {
        let h = ParityCheckMatrix::from_dense(vec![vec![1, 0, 0], vec![0, 1, 1]]).unwrap();
        let l = ParityCheckMatrix::from_dense(vec![vec![1, 1, 0]]).unwrap();

        assert_eq!(connected_cluster_code_distance(&h, &l, 1), None);
        let first = connected_cluster_code_distance(&h, &l, 2).unwrap();
        let second = connected_cluster_code_distance(&h, &l, 2).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.distance, 2);
        assert_eq!(first.mechanism_indices, vec![1, 2]);
        assert!(!first.mechanism_indices.contains(&0));
    }

    fn bb_code_pair(l: usize, m: usize) -> (ParityCheckMatrix, ParityCheckMatrix, usize) {
        let code = crate::BivariateBicycleCode::new(
            l,
            m,
            &[(3, 0), (0, 1), (0, 2)],
            &[(0, 3), (1, 0), (2, 0)],
        )
        .unwrap();
        (
            code.hx().clone(),
            code.logical_x().clone(),
            code.num_logical_qubits(),
        )
    }

    #[test]
    fn bb_72_distance_agrees_with_sat() {
        let (h, logicals, num_logical_qubits) = bb_code_pair(6, 6);
        assert_eq!(h.num_qubits(), 72);
        assert_eq!(num_logical_qubits, 12);

        let connected = connected_cluster_code_distance(&h, &logicals, 6).unwrap();
        let problem = DistanceProblem::from_css_checks(&h, &logicals).unwrap();
        let certified = certified_distance(&problem, 6).unwrap().unwrap();
        assert_eq!(connected.distance, 6);
        assert_eq!(certified.distance, connected.distance);
    }

    fn run_bb_timing_probe(l: usize, m: usize, expected_distance: usize, label: &str) {
        let (h, logicals, num_logical_qubits) = bb_code_pair(l, m);
        assert_eq!(num_logical_qubits, 12);
        let total_started = Instant::now();
        for weight in 1..=expected_distance {
            let started = Instant::now();
            let result = matrix_distance_at_weight(h.matrix(), logicals.matrix(), weight);
            println!("{label} CC weight {weight}: {:?}", started.elapsed());
            if weight < expected_distance {
                assert_eq!(result, None);
            } else {
                assert_eq!(result.unwrap().distance, expected_distance);
            }
        }
        println!("{label} CC total: {:?}", total_started.elapsed());
    }

    #[test]
    #[ignore = "timing probe for connected-cluster code distance"]
    fn connected_cluster_bb_72_12_6_timing_probe() {
        run_bb_timing_probe(6, 6, 6, "BB [[72,12,6]]");
    }

    #[test]
    #[ignore = "timing probe for connected-cluster code distance"]
    fn connected_cluster_gross_144_12_12_timing_probe() {
        run_bb_timing_probe(12, 6, 12, "gross [[144,12,12]]");
    }
}
