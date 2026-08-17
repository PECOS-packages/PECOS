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

//! Verification of the propagated-fault condition for flag circuits.
//!
//! Chao and Reichardt define a t-flag circuit by requiring that every set of `v <= t` faults
//! producing a data error `E` with `min(wt(E), wt(E * P)) > v` raises a flag, and additionally
//! requiring that a fault-free run does not flag; see arXiv:1708.02246. This module verifies the
//! first requirement by Pauli-frame propagation. It does **not** check the fault-free requirement:
//! that is a property of the ideal measurement outcomes, not of the propagated deviation frame,
//! and must be established separately.

use super::{FaultConfiguration, PauliPropChecker, has_syndrome};
use pecos_simulators::PauliProp;
use std::collections::BTreeSet;

/// A fault configuration that violates the propagated-fault condition for a flag circuit.
#[derive(Debug, Clone)]
pub struct FlagViolation {
    /// The counterexample fault configuration.
    pub faults: FaultConfiguration,
    /// Number of faulty circuit locations in `faults`.
    pub num_faults: usize,
    /// `min(wt(E), wt(E * P))`, restricted to the caller-supplied data qubits.
    pub error_weight: usize,
}

/// Result of checking the propagated-fault condition through fault weight `t`.
#[derive(Debug, Clone)]
pub struct FlagFaultToleranceReport {
    /// True when no fault configuration of weight at most `t` violates the t-flag condition.
    ///
    /// The full definition in arXiv:1708.02246 additionally requires a fault-free run not to flag.
    /// Pauli propagation tracks deviations from an ideal run and cannot check that ideal-outcome
    /// property, so callers must establish it separately.
    pub fault_condition_satisfied: bool,
    /// Maximum number of faulty circuit locations checked.
    pub t: usize,
    /// Counterexamples in ascending fault weight and deterministic iterator order.
    pub violations: Vec<FlagViolation>,
    /// Number of nonempty fault configurations propagated.
    pub total_configurations_tested: usize,
}

fn stabilizer_equivalent_error_weight(
    prop: &PauliProp,
    data_qubits: &[usize],
    measured_stabilizer: (&[usize], &[usize]),
) -> usize {
    let data_qubits: BTreeSet<_> = data_qubits.iter().copied().collect();
    let (stabilizer_xs, stabilizer_zs) = measured_stabilizer;
    let mut error_weight = 0;
    let mut equivalent_error_weight = 0;

    for qubit in data_qubits {
        let has_x = prop.contains_x(qubit);
        let has_z = prop.contains_z(qubit);
        error_weight += usize::from(has_x || has_z);

        let equivalent_has_x = has_x ^ stabilizer_xs.contains(&qubit);
        let equivalent_has_z = has_z ^ stabilizer_zs.contains(&qubit);
        equivalent_error_weight += usize::from(equivalent_has_x || equivalent_has_z);
    }

    error_weight.min(equivalent_error_weight)
}

impl PauliPropChecker<'_> {
    /// Checks the propagated-fault part of the Chao-Reichardt t-flag condition.
    ///
    /// For every exact fault-location weight `v` from 1 through `t`, this method propagates each
    /// configured Pauli fault assignment. A configuration violates the condition when no supplied
    /// flag qubit's Z-basis measurement is flipped and
    /// `min(wt(E), wt(E * P)) > v`, with both weights restricted to `data_qubits`.
    ///
    /// Fault-free flag behavior is outside this Pauli-frame check and must be verified separately.
    #[must_use]
    pub fn verify_flag_fault_tolerance(
        &self,
        data_qubits: &[usize],
        flag_qubits: &[usize],
        measured_stabilizer: (&[usize], &[usize]),
        t: usize,
    ) -> FlagFaultToleranceReport {
        let mut violations = Vec::new();
        let mut total_configurations_tested = 0;

        for weight in 1..=t {
            let fault_iter = self.fault_iterator_for_weight(weight);

            for faults in fault_iter {
                total_configurations_tested += 1;
                let prop = self.propagate_fault_configuration(&faults);
                if has_syndrome(&prop, flag_qubits, &[]) {
                    continue;
                }

                let error_weight =
                    stabilizer_equivalent_error_weight(&prop, data_qubits, measured_stabilizer);
                let num_faults = faults.len();
                if error_weight > num_faults {
                    violations.push(FlagViolation {
                        faults,
                        num_faults,
                        error_weight,
                    });
                }
            }
        }

        FlagFaultToleranceReport {
            fault_condition_satisfied: violations.is_empty(),
            t,
            violations,
            total_configurations_tested,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::propagate_faults;
    use pecos_quantum::TickCircuit;

    const DATA_QUBITS: &[usize] = &[0, 1, 2, 3];
    const STABILIZER_XS: &[usize] = &[0, 1, 2, 3];
    const MEASUREMENT_ANCILLA: usize = 4;
    const FLAG_ANCILLA: usize = 5;

    fn weight_four_x_measurement(with_flag: bool) -> TickCircuit {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[MEASUREMENT_ANCILLA]);
        circuit.tick().h(&[MEASUREMENT_ANCILLA]);
        if with_flag {
            circuit.tick().pz(&[FLAG_ANCILLA]);
        }
        circuit.tick().cx(&[(MEASUREMENT_ANCILLA, 0)]);
        if with_flag {
            circuit.tick().cx(&[(MEASUREMENT_ANCILLA, FLAG_ANCILLA)]);
        }
        circuit.tick().cx(&[(MEASUREMENT_ANCILLA, 1)]);
        circuit.tick().cx(&[(MEASUREMENT_ANCILLA, 2)]);
        if with_flag {
            circuit.tick().cx(&[(MEASUREMENT_ANCILLA, FLAG_ANCILLA)]);
        }
        circuit.tick().cx(&[(MEASUREMENT_ANCILLA, 3)]);
        circuit.tick().h(&[MEASUREMENT_ANCILLA]);
        circuit.tick().mz(&[MEASUREMENT_ANCILLA]);
        if with_flag {
            circuit.tick().mz(&[FLAG_ANCILLA]);
        }
        circuit
    }

    fn weight_six_single_flag_x_measurement() -> TickCircuit {
        const MEASUREMENT: usize = 6;
        const FLAG: usize = 7;

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[MEASUREMENT]);
        circuit.tick().h(&[MEASUREMENT]);
        circuit.tick().pz(&[FLAG]);
        circuit.tick().cx(&[(MEASUREMENT, 0)]);
        circuit.tick().cx(&[(MEASUREMENT, FLAG)]);
        for data_qubit in 1..=4 {
            circuit.tick().cx(&[(MEASUREMENT, data_qubit)]);
        }
        circuit.tick().cx(&[(MEASUREMENT, FLAG)]);
        circuit.tick().cx(&[(MEASUREMENT, 5)]);
        circuit.tick().h(&[MEASUREMENT]);
        circuit.tick().mz(&[MEASUREMENT]);
        circuit.tick().mz(&[FLAG]);
        circuit
    }

    fn verify(
        circuit: &TickCircuit,
        data_qubits: &[usize],
        flag_qubits: &[usize],
        t: usize,
    ) -> FlagFaultToleranceReport {
        let checker = PauliPropChecker::new(circuit);
        checker.verify_flag_fault_tolerance(data_qubits, flag_qubits, (STABILIZER_XS, &[]), t)
    }

    #[test]
    fn standard_single_flag_weight_four_measurement_satisfies_one_fault_condition() {
        let circuit = weight_four_x_measurement(true);
        let report = verify(&circuit, DATA_QUBITS, &[FLAG_ANCILLA], 1);

        assert!(report.fault_condition_satisfied);
        assert!(report.violations.is_empty());
        assert!(report.total_configurations_tested > 0);
        assert_eq!(report.t, 1);
    }

    #[test]
    fn unflagged_weight_four_measurement_has_weight_two_hook() {
        let circuit = weight_four_x_measurement(false);
        let report = verify(&circuit, DATA_QUBITS, &[], 1);

        assert!(!report.fault_condition_satisfied);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.num_faults == 1 && violation.error_weight == 2)
        );
    }

    #[test]
    fn stabilizer_equivalent_weight_prevents_false_violation() {
        let circuit = weight_four_x_measurement(false);
        let checker = PauliPropChecker::new(&circuit);
        let fault = checker
            .locations()
            .iter()
            .find(|location| location.tick == 2)
            .expect("the first data CX should be at tick 2")
            .clone();
        let faults =
            FaultConfiguration::with_faults(vec![super::super::PauliFault::new(fault, vec![1, 0])]);
        let prop = propagate_faults(&circuit, &faults);

        // The X fault on the measurement ancilla after CX(a, 0) propagates through the remaining
        // data couplings, so E = X1 X2 X3 and wt(E) = 3. For P = X0 X1 X2 X3,
        // E * P = X0 and wt(E * P) = 1. Thus min(3, 1) = v = 1: this is not a violation.
        assert_eq!(
            DATA_QUBITS
                .iter()
                .filter(|&&qubit| prop.contains_x(qubit) || prop.contains_z(qubit))
                .count(),
            3
        );
        assert_eq!(
            stabilizer_equivalent_error_weight(&prop, DATA_QUBITS, (STABILIZER_XS, &[])),
            1
        );

        let report = verify(&circuit, DATA_QUBITS, &[], 1);
        assert!(report.violations.iter().all(|violation| {
            let [reported_fault] = violation.faults.faults.as_slice() else {
                return true;
            };
            reported_fault.location.tick != 2 || reported_fault.paulis != [1, 0]
        }));
    }

    #[test]
    fn single_flag_circuit_does_not_satisfy_two_fault_condition() {
        const WEIGHT_SIX_DATA: &[usize] = &[0, 1, 2, 3, 4, 5];
        const WEIGHT_SIX_STABILIZER: &[usize] = &[0, 1, 2, 3, 4, 5];
        const WEIGHT_SIX_FLAG: &[usize] = &[7];

        let circuit = weight_six_single_flag_x_measurement();
        let checker = PauliPropChecker::new(&circuit);
        let one_fault_report = checker.verify_flag_fault_tolerance(
            WEIGHT_SIX_DATA,
            WEIGHT_SIX_FLAG,
            (WEIGHT_SIX_STABILIZER, &[]),
            1,
        );
        let report = checker.verify_flag_fault_tolerance(
            WEIGHT_SIX_DATA,
            WEIGHT_SIX_FLAG,
            (WEIGHT_SIX_STABILIZER, &[]),
            2,
        );

        assert!(one_fault_report.fault_condition_satisfied);
        assert!(!report.fault_condition_satisfied);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.num_faults == 2)
        );
    }

    #[test]
    fn verification_is_deterministic() {
        let circuit = weight_four_x_measurement(false);
        let first = verify(&circuit, DATA_QUBITS, &[], 2);
        let second = verify(&circuit, DATA_QUBITS, &[], 2);

        assert_eq!(
            first.fault_condition_satisfied,
            second.fault_condition_satisfied
        );
        assert_eq!(first.t, second.t);
        assert_eq!(
            first.total_configurations_tested,
            second.total_configurations_tested
        );
        assert_eq!(first.violations.len(), second.violations.len());
        for (left, right) in first.violations.iter().zip(&second.violations) {
            assert_eq!(left.num_faults, right.num_faults);
            assert_eq!(left.error_weight, right.error_weight);
            assert_eq!(left.faults.faults, right.faults.faults);
        }
    }

    #[test]
    fn error_weight_is_restricted_to_caller_supplied_data_qubits() {
        let circuit = weight_four_x_measurement(false);
        let report = verify(&circuit, &[0], &[], 1);

        assert!(report.fault_condition_satisfied);
        assert!(report.violations.is_empty());
    }
}
