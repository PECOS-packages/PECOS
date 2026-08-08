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

//! Dressed distance of subsystem (gauge) codes.
//!
//! A subsystem code's dressed distance is the minimum weight of an error that
//! commutes with every STABILIZER while acting nontrivially on the logical
//! subsystem — gauge operators are free. In the `(H, L)` formulation this is
//! exactly the stabilizer-code distance problem with `H` built from the
//! stabilizer generators only and `L` from the bare logical representatives:
//! gauge-group elements commute with the stabilizers and with the bare
//! logicals, so they are excluded from witnesses automatically, and dressed
//! representatives (bare logicals times gauge operators) are reachable because
//! gauge factors cost weight but violate nothing.

use crate::code_distance::stabilizer_code_distance;
use crate::distance::DistanceResult;
use crate::distance_problem::DistanceProblemError;
use crate::stabilizer_code_spec::{StabilizerCodeSpec, StabilizerCodeSpecError};
use pecos_core::{PauliOperator, PauliString};
use thiserror::Error;

/// Errors validating a subsystem-code specification.
#[derive(Debug, Error)]
pub enum SubsystemCodeError {
    /// A gauge generator anticommutes with a stabilizer.
    #[error("gauge generator {gauge} anticommutes with stabilizer {stabilizer}")]
    GaugeAnticommutesWithStabilizer {
        /// Index of the offending gauge generator.
        gauge: usize,
        /// Index of the stabilizer it fails against.
        stabilizer: usize,
    },
    /// A bare logical anticommutes with a gauge generator.
    #[error("bare logical {logical} anticommutes with gauge generator {gauge}")]
    LogicalAnticommutesWithGauge {
        /// Index of the offending logical (Z basis first, then X basis).
        logical: usize,
        /// Index of the gauge generator it fails against.
        gauge: usize,
    },
    /// The underlying stabilizer specification was rejected.
    #[error(transparent)]
    Spec(#[from] StabilizerCodeSpecError),
    /// The distance search rejected the specification.
    #[error(transparent)]
    Distance(DistanceProblemError),
}

/// Computes the dressed distance of a subsystem code by qubit-support weight.
///
/// `stabilizers` must be the stabilizer generators (the center of the gauge
/// group up to phases), `gauge_generators` the remaining gauge generators, and
/// the logicals BARE representatives (commuting with the full gauge group).
/// Validation enforces both commutation families before searching; the search
/// itself is the check-driven cluster engine over the stabilizer-only
/// specification.
///
/// # Errors
///
/// Returns an error if validation fails or the specification is rejected.
pub fn subsystem_dressed_distance(
    num_qubits: usize,
    stabilizers: Vec<PauliString>,
    gauge_generators: &[PauliString],
    logical_zs: Vec<PauliString>,
    logical_xs: Vec<PauliString>,
    max_weight: usize,
) -> Result<Option<DistanceResult>, SubsystemCodeError> {
    for (g, gauge) in gauge_generators.iter().enumerate() {
        for (s, stabilizer) in stabilizers.iter().enumerate() {
            if !gauge.commutes_with(stabilizer) {
                return Err(SubsystemCodeError::GaugeAnticommutesWithStabilizer {
                    gauge: g,
                    stabilizer: s,
                });
            }
        }
    }
    for (index, logical) in logical_zs.iter().chain(&logical_xs).enumerate() {
        for (g, gauge) in gauge_generators.iter().enumerate() {
            if !logical.commutes_with(gauge) {
                return Err(SubsystemCodeError::LogicalAnticommutesWithGauge {
                    logical: index,
                    gauge: g,
                });
            }
        }
    }
    let spec = StabilizerCodeSpec::new(num_qubits, stabilizers, logical_zs, logical_xs)?;
    stabilizer_code_distance(&spec, max_weight).map_err(SubsystemCodeError::Distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{Pauli, QuarterPhase, QubitId};

    fn pauli(terms: &[(Pauli, usize)]) -> PauliString {
        PauliString::with_phase_and_paulis(
            QuarterPhase::PlusOne,
            terms.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
        )
    }

    /// Bacon-Shor on an `m x n` grid: qubit (r, c) at index `r * n + c`.
    fn bacon_shor(
        m: usize,
        n: usize,
    ) -> (Vec<PauliString>, Vec<PauliString>, PauliString, PauliString) {
        let q = |r: usize, c: usize| r * n + c;
        let mut gauges = Vec::new();
        for r in 0..m {
            for c in 0..n - 1 {
                gauges.push(pauli(&[(Pauli::Z, q(r, c)), (Pauli::Z, q(r, c + 1))]));
            }
        }
        for r in 0..m - 1 {
            for c in 0..n {
                gauges.push(pauli(&[(Pauli::X, q(r, c)), (Pauli::X, q(r + 1, c))]));
            }
        }
        let mut stabilizers = Vec::new();
        for r in 0..m - 1 {
            let terms: Vec<_> = (0..n)
                .flat_map(|c| [(Pauli::X, q(r, c)), (Pauli::X, q(r + 1, c))])
                .collect();
            stabilizers.push(pauli(&terms));
        }
        for c in 0..n - 1 {
            let terms: Vec<_> = (0..m)
                .flat_map(|r| [(Pauli::Z, q(r, c)), (Pauli::Z, q(r, c + 1))])
                .collect();
            stabilizers.push(pauli(&terms));
        }
        let logical_x = pauli(&(0..n).map(|c| (Pauli::X, q(0, c))).collect::<Vec<_>>());
        let logical_z = pauli(&(0..m).map(|r| (Pauli::Z, q(r, 0))).collect::<Vec<_>>());
        (stabilizers, gauges, logical_z, logical_x)
    }

    #[test]
    fn square_bacon_shor_dressed_distance_is_three() {
        let (stabilizers, gauges, lz, lx) = bacon_shor(3, 3);
        let result = subsystem_dressed_distance(
            9,
            stabilizers.clone(),
            &gauges,
            vec![lz.clone()],
            vec![lx.clone()],
            9,
        )
        .unwrap()
        .expect("distance within budget");
        assert_eq!(result.distance, 3);

        // Independent engine: the certified SAT path over the same stabilizer-only spec.
        let spec = StabilizerCodeSpec::new(9, stabilizers, vec![lz], vec![lx]).unwrap();
        let problem = crate::DistanceProblem::from_stabilizer_spec(&spec).unwrap();
        let certified = crate::certified_distance(&problem, 3).unwrap().unwrap();
        assert_eq!(certified.distance, 3);
    }

    #[test]
    fn rectangular_bacon_shor_dressed_distance_is_the_short_side() {
        let (stabilizers, gauges, lz, lx) = bacon_shor(2, 3);
        let result = subsystem_dressed_distance(6, stabilizers, &gauges, vec![lz], vec![lx], 6)
            .unwrap()
            .expect("distance within budget");
        assert_eq!(result.distance, 2);
    }

    #[test]
    fn validation_rejects_a_logical_that_anticommutes_with_a_gauge() {
        let (stabilizers, gauges, lz, _lx) = bacon_shor(3, 3);
        // X on a single column anticommutes with a horizontal ZZ gauge pair.
        let bad_logical_x = pauli(&[(Pauli::X, 0), (Pauli::X, 3), (Pauli::X, 6)]);
        let error =
            subsystem_dressed_distance(9, stabilizers, &gauges, vec![lz], vec![bad_logical_x], 9)
                .unwrap_err();
        assert!(matches!(
            error,
            SubsystemCodeError::LogicalAnticommutesWithGauge { .. }
        ));
    }

    #[test]
    fn validation_rejects_a_gauge_that_anticommutes_with_a_stabilizer() {
        let (stabilizers, mut gauges, lz, lx) = bacon_shor(3, 3);
        gauges.push(pauli(&[(Pauli::Z, 0)]));
        let error =
            subsystem_dressed_distance(9, stabilizers, &gauges, vec![lz], vec![lx], 9).unwrap_err();
        assert!(matches!(
            error,
            SubsystemCodeError::GaugeAnticommutesWithStabilizer { .. }
        ));
    }
}
