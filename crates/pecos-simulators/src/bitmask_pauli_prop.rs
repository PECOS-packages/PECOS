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

//! Bitmask-backed Pauli propagation for hot Clifford fault-analysis paths.
//!
//! This type tracks only the binary X/Z support of a propagating Pauli. It
//! intentionally ignores global phase, matching the fault-catalog use case
//! where only measurement flips and anticommutation with tracked Paulis
//! matter.

use crate::clifford_gateable::{CliffordGateable, MeasurementResult};
use crate::quantum_simulator::QuantumSimulator;
use pecos_core::{BitmaskStorage, PauliBitmaskSmall, QubitId};

/// Internal phase-free Pauli propagator backed by `PauliBitmaskSmall`.
///
/// This is a performance helper for fault analysis and other hot internal
/// propagation paths. User-facing code should prefer Pauli strings and the
/// standard simulator APIs.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct BitmaskPauliProp {
    label: PauliBitmaskSmall,
    num_qubits: usize,
}

impl Default for BitmaskPauliProp {
    fn default() -> Self {
        Self::new()
    }
}

impl BitmaskPauliProp {
    /// Create an empty propagating Pauli with no fixed qubit count.
    #[must_use]
    pub fn new() -> Self {
        Self {
            label: PauliBitmaskSmall::identity(),
            num_qubits: 0,
        }
    }

    /// Create an empty propagating Pauli with a fixed qubit count for display
    /// and tests.
    #[must_use]
    pub fn with_num_qubits(num_qubits: usize) -> Self {
        Self {
            label: PauliBitmaskSmall::identity(),
            num_qubits,
        }
    }

    /// Checks whether the specified qubit has an X component.
    #[inline]
    #[must_use]
    pub fn contains_x(&self, qubit: usize) -> bool {
        self.label.x_bits.get_bit(qubit)
    }

    /// Checks whether the specified qubit has a Z component.
    #[inline]
    #[must_use]
    pub fn contains_z(&self, qubit: usize) -> bool {
        self.label.z_bits.get_bit(qubit)
    }

    /// Checks whether the specified qubit has a Y component.
    #[inline]
    #[must_use]
    pub fn contains_y(&self, qubit: usize) -> bool {
        self.contains_x(qubit) && self.contains_z(qubit)
    }

    /// Toggle X components on the given qubits.
    #[inline]
    pub fn track_x(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.label.x_bits.xor_bit(q);
            self.num_qubits = self.num_qubits.max(q + 1);
        }
    }

    /// Toggle Z components on the given qubits.
    #[inline]
    pub fn track_z(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.label.z_bits.xor_bit(q);
            self.num_qubits = self.num_qubits.max(q + 1);
        }
    }

    /// Toggle Y components on the given qubits.
    #[inline]
    pub fn track_y(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.label.x_bits.xor_bit(q);
            self.label.z_bits.xor_bit(q);
            self.num_qubits = self.num_qubits.max(q + 1);
        }
    }

    /// Remove all Pauli components from one qubit.
    #[inline]
    pub fn clear_qubit(&mut self, qubit: usize) {
        self.label.x_bits.clear_bit(qubit);
        self.label.z_bits.clear_bit(qubit);
    }

    /// True when no X/Z components remain.
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.label.is_identity()
    }

    /// Number of non-identity single-qubit factors.
    #[must_use]
    pub fn weight(&self) -> usize {
        self.label.weight() as usize
    }

    /// Dense string representation in qubit-index order.
    #[must_use]
    pub fn dense_string(&self) -> String {
        let mut result = String::with_capacity(self.num_qubits);
        for q in 0..self.num_qubits {
            match (self.contains_x(q), self.contains_z(q)) {
                (false, false) => result.push('I'),
                (true, false) => result.push('X'),
                (false, true) => result.push('Z'),
                (true, true) => result.push('Y'),
            }
        }
        result
    }

    #[inline]
    fn set_x_component(&mut self, q: usize, value: bool) {
        if value {
            self.label.x_bits.set_bit(q);
        } else {
            self.label.x_bits.clear_bit(q);
        }
        self.num_qubits = self.num_qubits.max(q + 1);
    }

    #[inline]
    fn set_z_component(&mut self, q: usize, value: bool) {
        if value {
            self.label.z_bits.set_bit(q);
        } else {
            self.label.z_bits.clear_bit(q);
        }
        self.num_qubits = self.num_qubits.max(q + 1);
    }

    #[inline]
    fn set_components(&mut self, q: usize, x: bool, z: bool) {
        self.set_x_component(q, x);
        self.set_z_component(q, z);
    }
}

impl QuantumSimulator for BitmaskPauliProp {
    #[inline]
    fn reset(&mut self) -> &mut Self {
        self.label = PauliBitmaskSmall::identity();
        self
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

impl CliffordGateable for BitmaskPauliProp {
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let q = q.index();
            if self.contains_x(q) {
                self.label.z_bits.xor_bit(q);
            }
            self.num_qubits = self.num_qubits.max(q + 1);
        }
        self
    }

    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sz(qubits)
    }

    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let q = q.index();
            let x = self.contains_x(q);
            let z = self.contains_z(q);
            self.set_components(q, z, x);
        }
        self
    }

    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let q = q.index();
            if self.contains_z(q) {
                self.label.x_bits.xor_bit(q);
            }
            self.num_qubits = self.num_qubits.max(q + 1);
        }
        self
    }

    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sx(qubits)
    }

    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let q = q.index();
            let x = self.contains_x(q);
            let z = self.contains_z(q);
            self.set_components(q, z, x);
        }
        self
    }

    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sy(qubits)
    }

    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            let control = control.index();
            let target = target.index();
            let control_x = self.contains_x(control);
            let target_z = self.contains_z(target);
            if control_x {
                self.label.x_bits.xor_bit(target);
            }
            if target_z {
                self.label.z_bits.xor_bit(control);
            }
            self.num_qubits = self.num_qubits.max(control.max(target) + 1);
        }
        self
    }

    #[inline]
    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x1, z1 ^ x2 ^ z2);
            self.set_components(q2, x2 ^ x1, z2 ^ x1);
        }
        self
    }

    #[inline]
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x1, z1 ^ x2);
            self.set_components(q2, x2, z2 ^ x1);
        }
        self
    }

    #[inline]
    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            let affected = z1 ^ z2;
            self.set_components(q1, x1 ^ affected, z1);
            self.set_components(q2, x2 ^ affected, z2);
        }
        self
    }

    #[inline]
    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.sxx(pairs)
    }

    #[inline]
    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x2 ^ z1 ^ z2, x1 ^ x2 ^ z2);
            self.set_components(q2, x1 ^ z1 ^ z2, x1 ^ x2 ^ z1);
        }
        self
    }

    #[inline]
    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.syy(pairs)
    }

    #[inline]
    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            let affected = x1 ^ x2;
            self.set_components(q1, x1, z1 ^ affected);
            self.set_components(q2, x2, z2 ^ affected);
        }
        self
    }

    #[inline]
    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.szz(pairs)
    }

    #[inline]
    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let x1 = self.contains_x(q1);
            let z1 = self.contains_z(q1);
            let x2 = self.contains_x(q2);
            let z2 = self.contains_z(q2);
            self.set_components(q1, x2, z2);
            self.set_components(q2, x1, z1);
        }
        self
    }

    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| MeasurementResult {
                outcome: self.contains_x(q.index()),
                is_deterministic: true,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli_prop::PauliProp;
    use pecos_core::QubitId;

    fn all_paulis(num_qubits: usize) -> Vec<String> {
        let labels = ['I', 'X', 'Y', 'Z'];
        let mut out = Vec::new();
        let total = 4usize.pow(num_qubits.try_into().expect("test qubit count fits"));
        for mut value in 0..total {
            let mut s = String::with_capacity(num_qubits);
            for _ in 0..num_qubits {
                s.push(labels[value % 4]);
                value /= 4;
            }
            out.push(s);
        }
        out
    }

    fn sparse_prop_from_dense(input: &str) -> PauliProp {
        let mut prop = PauliProp::with_sign_tracking(input.len());
        for (q, p) in input.chars().enumerate() {
            match p {
                'I' => {}
                'X' => prop.track_x(&[q]),
                'Y' => prop.track_y(&[q]),
                'Z' => prop.track_z(&[q]),
                _ => panic!("invalid Pauli label {p}"),
            }
        }
        prop
    }

    fn bitmask_prop_from_dense(input: &str) -> BitmaskPauliProp {
        let mut prop = BitmaskPauliProp::with_num_qubits(input.len());
        for (q, p) in input.chars().enumerate() {
            match p {
                'I' => {}
                'X' => prop.track_x(&[q]),
                'Y' => prop.track_y(&[q]),
                'Z' => prop.track_z(&[q]),
                _ => panic!("invalid Pauli label {p}"),
            }
        }
        prop
    }

    fn assert_matches_sparse_1q<F, G>(name: &str, mut apply_sparse: F, mut apply_bitmask: G)
    where
        F: FnMut(&mut PauliProp),
        G: FnMut(&mut BitmaskPauliProp),
    {
        for input in all_paulis(1) {
            let mut sparse = sparse_prop_from_dense(&input);
            let mut bitmask = bitmask_prop_from_dense(&input);
            apply_sparse(&mut sparse);
            apply_bitmask(&mut bitmask);
            assert_eq!(
                bitmask.dense_string(),
                sparse.dense_string(),
                "{name}: {input}"
            );
        }
    }

    fn assert_matches_sparse_2q<F, G>(name: &str, mut apply_sparse: F, mut apply_bitmask: G)
    where
        F: FnMut(&mut PauliProp, &[(QubitId, QubitId)]),
        G: FnMut(&mut BitmaskPauliProp, &[(QubitId, QubitId)]),
    {
        let pair = [(QubitId(0), QubitId(1))];
        for input in all_paulis(2) {
            let mut sparse = sparse_prop_from_dense(&input);
            let mut bitmask = bitmask_prop_from_dense(&input);
            apply_sparse(&mut sparse, &pair);
            apply_bitmask(&mut bitmask, &pair);
            assert_eq!(
                bitmask.dense_string(),
                sparse.dense_string(),
                "{name}: {input}"
            );
        }
    }

    fn assert_matches_sparse_2q_at_pair<F, G>(
        name: &str,
        pair: (QubitId, QubitId),
        num_qubits: usize,
        mut apply_sparse: F,
        mut apply_bitmask: G,
    ) where
        F: FnMut(&mut PauliProp, &[(QubitId, QubitId)]),
        G: FnMut(&mut BitmaskPauliProp, &[(QubitId, QubitId)]),
    {
        let labels = ['I', 'X', 'Y', 'Z'];
        let pairs = [pair];
        for lhs in labels {
            for rhs in labels {
                let mut input = vec!['I'; num_qubits];
                input[pair.0.0] = lhs;
                input[pair.1.0] = rhs;
                let input = input.into_iter().collect::<String>();

                let mut sparse = sparse_prop_from_dense(&input);
                let mut bitmask = bitmask_prop_from_dense(&input);
                apply_sparse(&mut sparse, &pairs);
                apply_bitmask(&mut bitmask, &pairs);
                assert_eq!(
                    bitmask.dense_string(),
                    sparse.dense_string(),
                    "{name}: pair {pair:?}, input {input}"
                );
            }
        }
    }

    fn assert_phase_free_1q_table<F>(name: &str, mut apply: F, table: &[(&str, &str)])
    where
        F: FnMut(&mut BitmaskPauliProp),
    {
        for &(input, expected) in table {
            let mut prop = bitmask_prop_from_dense(input);
            apply(&mut prop);
            assert_eq!(prop.dense_string(), expected, "{name}: {input}");
        }
    }

    fn assert_phase_free_2q_table<F>(name: &str, mut apply: F, table: &[(&str, &str)])
    where
        F: FnMut(&mut BitmaskPauliProp),
    {
        for &(input, expected) in table {
            let mut prop = bitmask_prop_from_dense(input);
            apply(&mut prop);
            assert_eq!(prop.dense_string(), expected, "{name}: {input}");
        }
    }

    #[test]
    fn single_qubit_cliffords_match_sparse_pauli_prop() {
        assert_matches_sparse_1q(
            "H",
            |p| {
                p.h(&[QubitId(0)]);
            },
            |p| {
                p.h(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "SZ",
            |p| {
                p.sz(&[QubitId(0)]);
            },
            |p| {
                p.sz(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "SZdg",
            |p| {
                p.szdg(&[QubitId(0)]);
            },
            |p| {
                p.szdg(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "SX",
            |p| {
                p.sx(&[QubitId(0)]);
            },
            |p| {
                p.sx(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "SXdg",
            |p| {
                p.sxdg(&[QubitId(0)]);
            },
            |p| {
                p.sxdg(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "SY",
            |p| {
                p.sy(&[QubitId(0)]);
            },
            |p| {
                p.sy(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "SYdg",
            |p| {
                p.sydg(&[QubitId(0)]);
            },
            |p| {
                p.sydg(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "F",
            |p| {
                p.f(&[QubitId(0)]);
            },
            |p| {
                p.f(&[QubitId(0)]);
            },
        );
        assert_matches_sparse_1q(
            "Fdg",
            |p| {
                p.fdg(&[QubitId(0)]);
            },
            |p| {
                p.fdg(&[QubitId(0)]);
            },
        );
    }

    #[test]
    fn standard_cliffords_match_phase_free_pauli_tables() {
        const PAULI_SELF: &[(&str, &str)] = &[("I", "I"), ("X", "X"), ("Y", "Y"), ("Z", "Z")];
        const H_SY: &[(&str, &str)] = &[("I", "I"), ("X", "Z"), ("Y", "Y"), ("Z", "X")];
        const SZ: &[(&str, &str)] = &[("I", "I"), ("X", "Y"), ("Y", "X"), ("Z", "Z")];
        const SX: &[(&str, &str)] = &[("I", "I"), ("X", "X"), ("Y", "Z"), ("Z", "Y")];
        const F: &[(&str, &str)] = &[("I", "I"), ("X", "Y"), ("Y", "Z"), ("Z", "X")];
        const FDG: &[(&str, &str)] = &[("I", "I"), ("X", "Z"), ("Y", "X"), ("Z", "Y")];
        const CX: &[(&str, &str)] = &[
            ("XI", "XX"),
            ("YI", "YX"),
            ("ZI", "ZI"),
            ("IX", "IX"),
            ("IY", "ZY"),
            ("IZ", "ZZ"),
        ];
        const CY: &[(&str, &str)] = &[
            ("XI", "XY"),
            ("YI", "YY"),
            ("ZI", "ZI"),
            ("IX", "ZX"),
            ("IY", "IY"),
            ("IZ", "ZZ"),
        ];
        const CZ: &[(&str, &str)] = &[
            ("XI", "XZ"),
            ("YI", "YZ"),
            ("ZI", "ZI"),
            ("IX", "ZX"),
            ("IY", "ZY"),
            ("IZ", "IZ"),
        ];
        const SXX: &[(&str, &str)] = &[
            ("XI", "XI"),
            ("YI", "ZX"),
            ("ZI", "YX"),
            ("IX", "IX"),
            ("IY", "XZ"),
            ("IZ", "XY"),
        ];
        const SYY: &[(&str, &str)] = &[
            ("XI", "ZY"),
            ("YI", "YI"),
            ("ZI", "XY"),
            ("IX", "YZ"),
            ("IY", "IY"),
            ("IZ", "YX"),
        ];
        const SZZ: &[(&str, &str)] = &[
            ("XI", "YZ"),
            ("YI", "XZ"),
            ("ZI", "ZI"),
            ("IX", "ZY"),
            ("IY", "ZX"),
            ("IZ", "IZ"),
        ];
        const SWAP: &[(&str, &str)] = &[
            ("XI", "IX"),
            ("YI", "IY"),
            ("ZI", "IZ"),
            ("IX", "XI"),
            ("IY", "YI"),
            ("IZ", "ZI"),
        ];

        assert_phase_free_1q_table(
            "X",
            |p| {
                p.x(&[QubitId(0)]);
            },
            PAULI_SELF,
        );
        assert_phase_free_1q_table(
            "Y",
            |p| {
                p.y(&[QubitId(0)]);
            },
            PAULI_SELF,
        );
        assert_phase_free_1q_table(
            "Z",
            |p| {
                p.z(&[QubitId(0)]);
            },
            PAULI_SELF,
        );
        assert_phase_free_1q_table(
            "H",
            |p| {
                p.h(&[QubitId(0)]);
            },
            H_SY,
        );
        assert_phase_free_1q_table(
            "SZ",
            |p| {
                p.sz(&[QubitId(0)]);
            },
            SZ,
        );
        assert_phase_free_1q_table(
            "SZdg",
            |p| {
                p.szdg(&[QubitId(0)]);
            },
            SZ,
        );
        assert_phase_free_1q_table(
            "SX",
            |p| {
                p.sx(&[QubitId(0)]);
            },
            SX,
        );
        assert_phase_free_1q_table(
            "SXdg",
            |p| {
                p.sxdg(&[QubitId(0)]);
            },
            SX,
        );
        assert_phase_free_1q_table(
            "SY",
            |p| {
                p.sy(&[QubitId(0)]);
            },
            H_SY,
        );
        assert_phase_free_1q_table(
            "SYdg",
            |p| {
                p.sydg(&[QubitId(0)]);
            },
            H_SY,
        );
        assert_phase_free_1q_table(
            "F",
            |p| {
                p.f(&[QubitId(0)]);
            },
            F,
        );
        assert_phase_free_1q_table(
            "Fdg",
            |p| {
                p.fdg(&[QubitId(0)]);
            },
            FDG,
        );

        let pair = [(QubitId(0), QubitId(1))];
        assert_phase_free_2q_table(
            "CX",
            |p| {
                p.cx(&pair);
            },
            CX,
        );
        assert_phase_free_2q_table(
            "CY",
            |p| {
                p.cy(&pair);
            },
            CY,
        );
        assert_phase_free_2q_table(
            "CZ",
            |p| {
                p.cz(&pair);
            },
            CZ,
        );
        assert_phase_free_2q_table(
            "SXX",
            |p| {
                p.sxx(&pair);
            },
            SXX,
        );
        assert_phase_free_2q_table(
            "SXXdg",
            |p| {
                p.sxxdg(&pair);
            },
            SXX,
        );
        assert_phase_free_2q_table(
            "SYY",
            |p| {
                p.syy(&pair);
            },
            SYY,
        );
        assert_phase_free_2q_table(
            "SYYdg",
            |p| {
                p.syydg(&pair);
            },
            SYY,
        );
        assert_phase_free_2q_table(
            "SZZ",
            |p| {
                p.szz(&pair);
            },
            SZZ,
        );
        assert_phase_free_2q_table(
            "SZZdg",
            |p| {
                p.szzdg(&pair);
            },
            SZZ,
        );
        assert_phase_free_2q_table(
            "SWAP",
            |p| {
                p.swap(&pair);
            },
            SWAP,
        );
    }

    #[test]
    fn two_qubit_cliffords_match_sparse_pauli_prop() {
        assert_matches_sparse_2q(
            "CX",
            |p, qs| {
                p.cx(qs);
            },
            |p, qs| {
                p.cx(qs);
            },
        );
        assert_matches_sparse_2q(
            "CY",
            |p, qs| {
                p.cy(qs);
            },
            |p, qs| {
                p.cy(qs);
            },
        );
        assert_matches_sparse_2q(
            "CZ",
            |p, qs| {
                p.cz(qs);
            },
            |p, qs| {
                p.cz(qs);
            },
        );
        assert_matches_sparse_2q(
            "SXX",
            |p, qs| {
                p.sxx(qs);
            },
            |p, qs| {
                p.sxx(qs);
            },
        );
        assert_matches_sparse_2q(
            "SXXdg",
            |p, qs| {
                p.sxxdg(qs);
            },
            |p, qs| {
                p.sxxdg(qs);
            },
        );
        assert_matches_sparse_2q(
            "SYY",
            |p, qs| {
                p.syy(qs);
            },
            |p, qs| {
                p.syy(qs);
            },
        );
        assert_matches_sparse_2q(
            "SYYdg",
            |p, qs| {
                p.syydg(qs);
            },
            |p, qs| {
                p.syydg(qs);
            },
        );
        assert_matches_sparse_2q(
            "SZZ",
            |p, qs| {
                p.szz(qs);
            },
            |p, qs| {
                p.szz(qs);
            },
        );
        assert_matches_sparse_2q(
            "SZZdg",
            |p, qs| {
                p.szzdg(qs);
            },
            |p, qs| {
                p.szzdg(qs);
            },
        );
        assert_matches_sparse_2q(
            "SWAP",
            |p, qs| {
                p.swap(qs);
            },
            |p, qs| {
                p.swap(qs);
            },
        );
        assert_matches_sparse_2q(
            "ISWAP",
            |p, qs| {
                p.iswap(qs);
            },
            |p, qs| {
                p.iswap(qs);
            },
        );
        assert_matches_sparse_2q(
            "ISWAPdg",
            |p, qs| {
                p.iswapdg(qs);
            },
            |p, qs| {
                p.iswapdg(qs);
            },
        );
    }

    #[test]
    fn two_qubit_cliffords_match_sparse_pauli_prop_across_word_boundaries() {
        for pair in [
            (QubitId(63), QubitId(64)),
            (QubitId(64), QubitId(63)),
            (QubitId(64), QubitId(65)),
        ] {
            assert_matches_sparse_2q_at_pair(
                "CX",
                pair,
                66,
                |p, qs| {
                    p.cx(qs);
                },
                |p, qs| {
                    p.cx(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "CY",
                pair,
                66,
                |p, qs| {
                    p.cy(qs);
                },
                |p, qs| {
                    p.cy(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "CZ",
                pair,
                66,
                |p, qs| {
                    p.cz(qs);
                },
                |p, qs| {
                    p.cz(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SXX",
                pair,
                66,
                |p, qs| {
                    p.sxx(qs);
                },
                |p, qs| {
                    p.sxx(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SXXdg",
                pair,
                66,
                |p, qs| {
                    p.sxxdg(qs);
                },
                |p, qs| {
                    p.sxxdg(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SYY",
                pair,
                66,
                |p, qs| {
                    p.syy(qs);
                },
                |p, qs| {
                    p.syy(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SYYdg",
                pair,
                66,
                |p, qs| {
                    p.syydg(qs);
                },
                |p, qs| {
                    p.syydg(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SZZ",
                pair,
                66,
                |p, qs| {
                    p.szz(qs);
                },
                |p, qs| {
                    p.szz(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SZZdg",
                pair,
                66,
                |p, qs| {
                    p.szzdg(qs);
                },
                |p, qs| {
                    p.szzdg(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "SWAP",
                pair,
                66,
                |p, qs| {
                    p.swap(qs);
                },
                |p, qs| {
                    p.swap(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "ISWAP",
                pair,
                66,
                |p, qs| {
                    p.iswap(qs);
                },
                |p, qs| {
                    p.iswap(qs);
                },
            );
            assert_matches_sparse_2q_at_pair(
                "ISWAPdg",
                pair,
                66,
                |p, qs| {
                    p.iswapdg(qs);
                },
                |p, qs| {
                    p.iswapdg(qs);
                },
            );
        }
    }

    #[test]
    fn word_boundary_propagation_matches_sparse_pauli_prop() {
        let qubits = [63, 64, 65];
        let qids = qubits.map(QubitId);
        let mut sparse = PauliProp::with_sign_tracking(66);
        let mut bitmask = BitmaskPauliProp::with_num_qubits(66);

        sparse.track_z(&qubits);
        bitmask.track_z(&qubits);
        sparse.h(&qids);
        bitmask.h(&qids);

        assert_eq!(bitmask.dense_string(), sparse.dense_string());

        let sparse_meas = sparse.mz(&qids);
        let bitmask_meas = bitmask.mz(&qids);
        assert_eq!(
            bitmask_meas.iter().map(|m| m.outcome).collect::<Vec<_>>(),
            sparse_meas.iter().map(|m| m.outcome).collect::<Vec<_>>()
        );
        assert_eq!(
            bitmask_meas
                .iter()
                .map(|m| m.is_deterministic)
                .collect::<Vec<_>>(),
            sparse_meas
                .iter()
                .map(|m| m.is_deterministic)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sequential_gate_composition_matches_sparse_pauli_prop() {
        let sequence = |sparse: &mut PauliProp, bitmask: &mut BitmaskPauliProp| {
            sparse
                .h(&[QubitId(0)])
                .sz(&[QubitId(1)])
                .cx(&[(QubitId(0), QubitId(1))])
                .sxx(&[(QubitId(1), QubitId(2))])
                .iswap(&[(QubitId(0), QubitId(2))])
                .sydg(&[QubitId(2)])
                .cz(&[(QubitId(2), QubitId(1))])
                .swap(&[(QubitId(0), QubitId(1))]);
            bitmask
                .h(&[QubitId(0)])
                .sz(&[QubitId(1)])
                .cx(&[(QubitId(0), QubitId(1))])
                .sxx(&[(QubitId(1), QubitId(2))])
                .iswap(&[(QubitId(0), QubitId(2))])
                .sydg(&[QubitId(2)])
                .cz(&[(QubitId(2), QubitId(1))])
                .swap(&[(QubitId(0), QubitId(1))]);
        };

        for input in all_paulis(3) {
            let mut sparse = sparse_prop_from_dense(&input);
            let mut bitmask = bitmask_prop_from_dense(&input);
            sequence(&mut sparse, &mut bitmask);
            assert_eq!(
                bitmask.dense_string(),
                sparse.dense_string(),
                "sequential composition: {input}"
            );
        }
    }

    #[test]
    fn measurement_reset_and_identity_match_fault_catalog_semantics() {
        let mut prop = BitmaskPauliProp::with_num_qubits(3);
        prop.track_y(&[1]);

        let meas = prop.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        assert_eq!(
            meas.iter().map(|m| m.outcome).collect::<Vec<_>>(),
            vec![false, true, false]
        );
        assert!(meas.iter().all(|m| m.is_deterministic));

        prop.clear_qubit(1);
        assert!(prop.is_identity());

        prop.track_z(&[2]);
        assert_eq!(prop.weight(), 1);
        prop.reset();
        assert!(prop.is_identity());
    }
}
