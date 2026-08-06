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

//! Diagnosis of single-qubit faults that amplify into multi-qubit data errors.

use super::{PauliPropChecker, SpacetimeLocation, anticommutes_with_logical, has_syndrome};
use pecos_simulators::PauliProp;

/// A single-qubit fault that amplifies into an error on multiple data qubits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookError {
    /// The gate location responsible for the amplified error.
    pub location: SpacetimeLocation,
    /// The injected Pauli operators, with one non-identity entry.
    pub fault_paulis: Vec<u8>,
    /// Sorted support of the propagated error on the caller-supplied data qubits.
    pub data_support: Vec<usize>,
    /// Number of data qubits in [`Self::data_support`].
    pub data_weight: usize,
    /// Whether the propagated error triggers a syndrome on the supplied ancillas.
    pub detected: bool,
    /// Whether the propagated error anticommutes with any supplied logical operator.
    pub causes_logical_error: bool,
}

/// Summary of hook-error diagnosis across the checker's configured fault set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookErrorReport {
    /// Single-qubit faults whose propagated data weight reaches the requested threshold.
    pub hook_errors: Vec<HookError>,
    /// Number of fault configurations returned by the existing fault analyzer.
    pub total_faults_examined: usize,
    /// Largest propagated data weight among the single-qubit faults examined.
    pub max_data_weight: usize,
}

fn propagated_data_support(prop: &PauliProp, data_qubits: &[usize]) -> Vec<usize> {
    let mut support: Vec<usize> = data_qubits
        .iter()
        .copied()
        .filter(|&qubit| prop.contains_x(qubit) || prop.contains_z(qubit))
        .collect();
    support.sort_unstable();
    support.dedup();
    support
}

impl PauliPropChecker<'_> {
    /// Finds single-qubit faults that amplify on the caller-supplied data block.
    ///
    /// A fault is reported only when its injected non-identity Pauli weight is exactly one and
    /// its propagated support on `data_qubits` has weight at least `min_data_weight`. A threshold
    /// of 2 is standard for hook-error diagnosis; it is explicit here so callers can request
    /// higher-weight amplification.
    ///
    /// The returned hook errors are sorted by location tick, gate index, qubits, and injected
    /// Paulis. Syndrome and logical classifications use the same helpers as the existing fault
    /// analysis machinery.
    #[must_use]
    pub fn diagnose_hook_errors(
        &self,
        data_qubits: &[usize],
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        min_data_weight: usize,
    ) -> HookErrorReport {
        let analyses = self.analyze_all_faults(z_ancillas, x_ancillas, logicals);
        let total_faults_examined = analyses.len();
        let mut hook_errors = Vec::new();
        let mut max_data_weight = 0;

        for (fault_configuration, result) in analyses {
            let [fault] = fault_configuration.faults.as_slice() else {
                continue;
            };
            if fault.weight() != 1 {
                continue;
            }

            let data_support = propagated_data_support(&result.propagated_error, data_qubits);
            let data_weight = data_support.len();
            max_data_weight = max_data_weight.max(data_weight);

            if data_weight < min_data_weight {
                continue;
            }

            hook_errors.push(HookError {
                location: fault.location.clone(),
                fault_paulis: fault.paulis.clone(),
                data_support,
                data_weight,
                detected: has_syndrome(&result.propagated_error, z_ancillas, x_ancillas),
                causes_logical_error: logicals
                    .iter()
                    .any(|(xs, zs)| anticommutes_with_logical(&result.propagated_error, xs, zs)),
            });
        }

        hook_errors.sort_by(|left, right| {
            left.location
                .tick
                .cmp(&right.location.tick)
                .then_with(|| left.location.gate_index.cmp(&right.location.gate_index))
                .then_with(|| left.location.qubits.cmp(&right.location.qubits))
                .then_with(|| left.fault_paulis.cmp(&right.fault_paulis))
        });

        HookErrorReport {
            hook_errors,
            total_faults_examined,
            max_data_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::FaultCheckConfig;
    use pecos_core::QubitId;
    use pecos_core::gate_type::GateType;
    use pecos_quantum::TickCircuit;

    fn control_ancilla_ladder() -> TickCircuit {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3]);
        circuit.tick().cx(&[(3, 0)]);
        circuit.tick().cx(&[(3, 1)]);
        circuit.tick().cx(&[(3, 2)]);
        circuit.tick().mz(&[3]);
        circuit
    }

    fn find_hook<'a>(
        report: &'a HookErrorReport,
        tick: usize,
        fault_paulis: &[u8],
    ) -> Option<&'a HookError> {
        report
            .hook_errors
            .iter()
            .find(|hook| hook.location.tick == tick && hook.fault_paulis == fault_paulis)
    }

    #[test]
    fn amplifying_ancilla_fault_reports_responsible_cx_and_data_support() {
        let circuit = control_ancilla_ladder();
        let checker = PauliPropChecker::new(&circuit);
        let report = checker.diagnose_hook_errors(&[2, 0, 1], &[3], &[], &[], 2);

        let hook = find_hook(&report, 1, &[1, 0])
            .expect("X on the ancilla after the first CX should amplify");
        assert_eq!(hook.location.gate_type, GateType::CX);
        assert_eq!(hook.location.gate_index, 0);
        assert_eq!(hook.location.qubits, [QubitId(3), QubitId(0)]);
        assert_eq!(hook.data_support, [1, 2]);
        assert_eq!(hook.data_weight, 2);
        assert_eq!(hook.data_weight, hook.data_support.len());
        assert_eq!(report.max_data_weight, 3);
        assert_eq!(
            report.total_faults_examined,
            checker.analyze_all_faults(&[3], &[], &[]).len()
        );
    }

    #[test]
    fn ancilla_fault_after_final_cx_is_not_reported() {
        let circuit = control_ancilla_ladder();
        let checker = PauliPropChecker::new(&circuit);
        let report = checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], &[], 2);

        assert!(find_hook(&report, 3, &[1, 0]).is_none());
    }

    #[test]
    fn weight_two_fault_with_weight_two_data_support_is_not_a_hook() {
        let circuit = control_ancilla_ladder();
        let checker = PauliPropChecker::new(&circuit);

        let analyzed_xx = checker
            .analyze_all_faults(&[3], &[], &[])
            .into_iter()
            .find(|(configuration, _)| {
                configuration.faults.len() == 1
                    && configuration.faults[0].location.tick == 2
                    && configuration.faults[0].paulis == [1, 1]
            })
            .expect("the existing enumerator should include XX at the second CX");
        assert_eq!(analyzed_xx.0.total_weight(), 2);
        assert_eq!(
            propagated_data_support(&analyzed_xx.1.propagated_error, &[0, 1, 2]),
            [1, 2]
        );

        let report = checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], &[], 2);
        assert!(find_hook(&report, 2, &[1, 1]).is_none());
    }

    #[test]
    fn detected_and_logical_fields_use_propagated_error() {
        let detected_circuit = control_ancilla_ladder();
        let detected_checker = PauliPropChecker::new(&detected_circuit);
        let detected_logicals: &[(&[usize], &[usize])] = &[(&[], &[1, 2])];
        let detected_report =
            detected_checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], detected_logicals, 2);
        let detected = find_hook(&detected_report, 1, &[1, 0]).unwrap();
        assert!(detected.detected);
        assert!(!detected.causes_logical_error);

        let mut undetected_circuit = TickCircuit::new();
        undetected_circuit.tick().pz(&[3]);
        undetected_circuit.tick().cx(&[(0, 3)]);
        undetected_circuit.tick().cx(&[(1, 3)]);
        undetected_circuit.tick().cx(&[(2, 3)]);
        undetected_circuit.tick().mz(&[3]);
        let undetected_checker = PauliPropChecker::new(&undetected_circuit);
        let undetected_logicals: &[(&[usize], &[usize])] = &[(&[1], &[])];
        let undetected_report =
            undetected_checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], undetected_logicals, 2);
        let undetected = find_hook(&undetected_report, 1, &[0, 3]).unwrap();
        assert!(!undetected.detected);
        assert!(undetected.causes_logical_error);
        assert_eq!(undetected.data_support, [1, 2]);
    }

    #[test]
    fn diagnosis_order_is_deterministic() {
        let circuit = control_ancilla_ladder();
        let checker = PauliPropChecker::new(&circuit);

        let first = checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], &[], 2);
        let second = checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], &[], 2);

        assert_eq!(first, second);
        assert!(first.hook_errors.windows(2).all(|pair| {
            let left = &pair[0];
            let right = &pair[1];
            (
                left.location.tick,
                left.location.gate_index,
                &left.location.qubits,
                &left.fault_paulis,
            ) <= (
                right.location.tick,
                right.location.gate_index,
                &right.location.qubits,
                &right.fault_paulis,
            )
        }));
    }

    #[test]
    fn higher_minimum_data_weight_reports_strictly_fewer_hooks() {
        let circuit = control_ancilla_ladder();
        let checker = PauliPropChecker::new(&circuit)
            .with_config(FaultCheckConfig::new().with_weight(1).all_paulis());

        let weight_two = checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], &[], 2);
        let weight_three = checker.diagnose_hook_errors(&[0, 1, 2], &[3], &[], &[], 3);

        assert!(weight_three.hook_errors.len() < weight_two.hook_errors.len());
        assert!(
            weight_three
                .hook_errors
                .iter()
                .all(|hook| hook.data_weight >= 3)
        );
    }
}
