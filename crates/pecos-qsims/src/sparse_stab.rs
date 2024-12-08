// Copyright 2024 The PECOS Developers
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

use crate::{CliffordSimulator, Gens, QuantumSimulator};
use core::fmt::Debug;
use core::mem;
use pecos_core::SimRng;
use pecos_core::{IndexableElement, Set};
use rand_chacha::ChaCha8Rng;
// TODO: Look into seeing if a dense bool for signs_minus and signs_i is more efficient

#[derive(Clone, Debug)]
pub struct SparseStab<T, E, R = ChaCha8Rng>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
    R: SimRng,
{
    num_qubits: usize,
    stabs: Gens<T, E>,
    destabs: Gens<T, E>,
    rng: R,
}
impl<T, E, R> SparseStab<T, E, R>
where
    E: IndexableElement,
    R: SimRng,
    T: for<'a> Set<'a, Element = E>,
{
    #[inline]
    #[must_use]
    fn new(num_qubits: usize) -> Self {
        let rng = SimRng::from_entropy();
        Self::with_rng(num_qubits, rng)
    }

    #[inline]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let mut stab = Self {
            num_qubits,
            stabs: Gens::<T, E>::new(num_qubits),
            destabs: Gens::<T, E>::new(num_qubits),
            rng,
        };
        stab.reset();
        stab
    }

    #[expect(clippy::single_call_fn)]
    #[inline]
    fn reset(&mut self) -> &mut Self {
        self.stabs.init_all_z();
        self.destabs.init_all_x();
        self
    }

    #[inline]
    pub fn verify_matrix(&self) {
        Self::check_row_eq_col(&self.stabs);
        Self::check_row_eq_col(&self.destabs);

        // TODO: Check that stabilizers commute.
        // TODO: Check destabilizers commute.
        // TODO: Check that only stab[i] anti-commutes with destab[j] only iff i == j;
        todo!()
    }

    #[inline]
    fn check_row_eq_col(gens: &Gens<T, E>) {
        // TODO: Verify that this is doing what is intended...
        for (i, row) in gens.row_x.iter().enumerate() {
            for j in row.iter() {
                assert!(
                    gens.col_x[j.to_usize()].contains(&E::from_usize(i)),
                    "Column-wise sparse matrix doesn't match row-wise spare matrix"
                );
            }
        }
    }

    /// Utility that creates a string for the Pauli generates of a `Gens`.
    fn tableau_string(num_qubits: usize, gens: &Gens<T, E>) -> String {
        // TODO: calculate signs so we are really doing Y and not W
        let mut result =
            String::with_capacity(num_qubits * gens.row_x.len() + gens.row_x.len() + 2);
        for i in 0..gens.row_x.len() {
            if gens.signs_minus.contains(&(E::from_usize(i))) {
                result.push('-');
            } else {
                result.push('+');
            }
            if gens.signs_i.contains(&(E::from_usize(i))) {
                result.push('i');
            }

            for qubit in 0..num_qubits {
                let qubit_u = E::from_usize(qubit);
                let in_row_x = gens.row_x[i].contains(&qubit_u);
                let in_row_z = gens.row_z[i].contains(&qubit_u);

                let char = match (in_row_x, in_row_z) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                };
                result.push(char);
            }
            result.push('\n');
        }

        result
    }

    /// Produces a textual representation of the stabilizer in tableau form.
    #[inline]
    pub fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    /// Produces a textual representation of the destabilizer in tableau form.
    #[inline]
    pub fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    /// Negate the sign of a stabilizer generator.
    #[inline]
    pub fn neg(&mut self, s: E) {
        self.stabs.signs_minus ^= &s;
    }

    #[inline]
    pub fn signs_minus(&self) -> &T {
        &self.stabs.signs_minus
    }

    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn deterministic_meas(&mut self, q: E) -> bool {
        let qu = q.to_usize();

        let mut num_minuses = self.destabs.col_x[qu]
            .intersection(&self.stabs.signs_minus)
            .count();

        let num_is = &self.destabs.col_x[qu]
            .intersection(&self.stabs.signs_i)
            .count();

        let mut cumulative_x = T::new();
        for row in self.destabs.col_x[qu].iter() {
            let rowu = row.to_usize();
            num_minuses += &self.stabs.row_z[rowu].intersection(&cumulative_x).count();
            cumulative_x ^= &self.stabs.row_x[rowu];
        }
        if num_is & 3 != 0 {
            // num_is % 4 != 0
            num_minuses += 1;
        }
        num_minuses & 1 != 0 // num_minuses % 2 != 0 (is odd)
    }

    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn nondeterministic_meas(&mut self, q: E) -> E {
        let qu = q.to_usize();

        let mut anticom_stabs_col = self.stabs.col_x[qu].clone();
        let mut anticom_destabs_col = self.destabs.col_x[qu].clone();

        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<E> = None;

        for stab_id in anticom_stabs_col.iter() {
            let stab_usize = stab_id.to_usize();
            let weight = self.stabs.row_x[stab_usize].len() + self.stabs.row_z[stab_usize].len();

            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(*stab_id);
                // break // TODO: Should it exit early? // If we do... it avoids smallest weight
                // TODO: Does the smallest weight matter? Maybe at least break if smallest weight == 1
                // TODO: Does it always exist? If so, can we avoid Some()?
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");

        anticom_stabs_col.remove(&id);
        let id_usize = id.to_usize(); // Convert `id` to `usize`
        let removed_row_x = self.stabs.row_x[id_usize].clone();
        let removed_row_z = self.stabs.row_z[id_usize].clone();

        if self.stabs.signs_minus.contains(&id) {
            self.stabs.signs_minus ^= &anticom_stabs_col;
        }

        if self.stabs.signs_i.contains(&id) {
            self.stabs.signs_i.remove(&id);

            let gens_common = self
                .stabs
                .signs_i
                .intersection(&anticom_stabs_col)
                .copied()
                .collect::<Vec<_>>();
            let gens_only_stabs = anticom_stabs_col
                .difference(&self.stabs.signs_i)
                .copied()
                .collect::<Vec<_>>();

            for i in gens_common {
                self.stabs.signs_minus ^= &i;
                self.stabs.signs_i.remove(&i);
            }

            for i in gens_only_stabs {
                self.stabs.signs_i.insert(i);
            }
        }

        for gen in anticom_stabs_col.iter() {
            let gen_usize = gen.to_usize(); // Convert `gen` to `usize`
            let num_minuses = removed_row_z
                .intersection(&self.stabs.row_x[gen_usize])
                .count();

            if num_minuses & 1 != 0 {
                // num_minuses % 2 != 0 (is odd)
                self.stabs.signs_minus ^= gen;
            }

            self.stabs.row_x[gen_usize] ^= &removed_row_x;
            self.stabs.row_z[gen_usize] ^= &removed_row_z;
            // Use `num_minuses` as needed
        }

        for i in removed_row_x.iter() {
            let iu = i.to_usize();
            self.stabs.col_x[iu] ^= &anticom_stabs_col;
        }

        for i in removed_row_z.iter() {
            let iu = i.to_usize();
            self.stabs.col_z[iu] ^= &anticom_stabs_col;
        }

        for i in self.stabs.row_x[id_usize].iter() {
            let iu = i.to_usize();
            self.stabs.col_x[iu].remove(&id);
        }

        for i in self.stabs.row_z[id_usize].iter() {
            let iu = i.to_usize();
            self.stabs.col_z[iu].remove(&id);
        }

        // Remove replaced stabilizer with the measured stabilizer
        self.stabs.col_z[qu].insert(id);

        // Row update
        self.stabs.row_x[id_usize].clear();
        self.stabs.row_z[id_usize].clear();
        self.stabs.row_z[id_usize].insert(q);

        for i in self.destabs.row_x[id_usize].iter() {
            let iu = i.to_usize();
            self.destabs.col_x[iu].remove(&id);
        }

        for i in self.destabs.row_z[id_usize].iter() {
            let iu = i.to_usize();
            self.destabs.col_z[iu].remove(&id);
        }

        anticom_destabs_col.remove(&id);

        for i in removed_row_x.iter() {
            let iu = i.to_usize();
            self.destabs.col_x[iu].insert(id);
            self.destabs.col_x[iu] ^= &anticom_destabs_col;
        }

        for i in removed_row_z.iter() {
            let iu = i.to_usize();
            self.destabs.col_z[iu].insert(id);
            self.destabs.col_z[iu] ^= &anticom_destabs_col;
        }

        for row in anticom_destabs_col.iter() {
            let ru = row.to_usize();
            self.destabs.row_x[ru] ^= &removed_row_x;
            self.destabs.row_z[ru] ^= &removed_row_z;
        }

        self.destabs.row_x[id_usize] = removed_row_x;
        self.destabs.row_z[id_usize] = removed_row_z;

        id
    }

    /// Measurement of the +`Z_q` operator where random outcomes are forced to a particular value.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    pub fn mz_forced(&mut self, q: E, forced_outcome: bool) -> (bool, bool) {
        let qu = q.to_usize();

        let deterministic = self.stabs.col_x[qu].is_empty();

        // There are no stabilizers that anti-commute with Z_q
        let meas_out = if deterministic {
            self.deterministic_meas(q)
        } else {
            let id = self.nondeterministic_meas(q);

            self.apply_outcome(id, forced_outcome)
        };
        (meas_out, deterministic)
    }

    /// Preparation of the +`Z_q` operator where random outcomes are forced to a particular value.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    pub fn pz_forced(&mut self, q: E, forced_outcome: bool) -> (bool, bool) {
        let (meas, deter) = self.mz_forced(q, forced_outcome);
        if meas {
            self.x(q);
        }
        (meas, deter)
    }

    /// Apply measurement outcome
    #[inline]
    fn apply_outcome(&mut self, id: E, meas_outcome: bool) -> bool {
        if meas_outcome {
            self.stabs.signs_minus.insert(id);
        } else {
            self.stabs.signs_minus.remove(&id);
        }
        meas_outcome
    }
}

impl<T, E, R> QuantumSimulator for SparseStab<T, E, R>
where
    E: IndexableElement,
    R: SimRng,
    T: for<'a> Set<'a, Element = E>,
{
    #[inline]
    #[must_use]
    fn new(num_qubits: usize) -> Self {
        Self::new(num_qubits)
    }

    #[inline]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[inline]
    fn reset(&mut self) -> &mut Self {
        Self::reset(self)
    }
}

impl<T, E, R> CliffordSimulator<E> for SparseStab<T, E, R>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
    R: SimRng,
{
    // TODO: pub fun p(&mut self, pauli: &pauli, q: U) { todo!() }
    // TODO: pub fun m(&mut self, pauli: &pauli, q: U) -> bool { todo!() }

    /// Measurement of the +`Z_q` operator.
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn mz(&mut self, q: E) -> (bool, bool) {
        let qu = q.to_usize();

        let deterministic = self.stabs.col_x[qu].is_empty();

        // There are no stabilizers that anti-commute with Z_q
        let meas_out = if deterministic {
            self.deterministic_meas(q)
        } else {
            let id = self.nondeterministic_meas(q);

            let meas_outcome = self.rng.gen_bool(0.5);

            self.apply_outcome(id, meas_outcome)
        };
        (meas_out, deterministic)
    }

    /// Pauli X gate. X -> X, Z -> -Z
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn x(&mut self, q: E) {
        let qu = q.to_usize();
        self.stabs.signs_minus ^= &self.stabs.col_z[qu];
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn y(&mut self, q: E) {
        // TODO: Add test
        let qu = q.to_usize();
        // stabs.signs_minus ^= stabs.col_x[qubit] ^ stabs.col_z[qubit]
        for i in self.stabs.col_x[qu].symmetric_difference(&self.stabs.col_z[qu]) {
            self.stabs.signs_minus ^= i;
        }
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn z(&mut self, q: E) {
        // TODO: Add test
        self.stabs.signs_minus ^= &self.stabs.col_x[q.to_usize()];
    }

    /// Sqrt of Z gate.
    ///     X -> iW = Y
    ///     Z -> Z
    ///     W -> iX
    ///     Y -> -X
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn sz(&mut self, q: E) {
        let qu = q.to_usize();

        // X -> i
        // ---------------------
        // i * i = -1
        // stabs.signs_minus ^= stabs.signs_i & stabs.col_x[qubit]
        // For each X add an i unless there is already an i there then delete it.
        // stabs.signs_i ^= stabs.col_x[qubit]
        for i in self.stabs.signs_i.intersection(&self.stabs.col_x[qu]) {
            self.stabs.signs_minus ^= i;
        }
        self.stabs.signs_i ^= &self.stabs.col_x[qu];

        for g in [&mut self.stabs, &mut self.destabs] {
            g.col_z[qu] ^= &g.col_x[qu];

            for &i in g.col_x[qu].iter() {
                let iu = i.to_usize();
                g.row_z[iu] ^= &q;
            }
        }
    }

    /// Hadamard gate. X -> Z, Z -> X
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn h(&mut self, q: E) {
        let qu = q.to_usize();

        // self.stabs.signs_minus.symmetric_difference_update(self.stabs.col_x[qu].intersection())
        // self.stabs.signs_minus ^= &self.stabs.col_x[qu] & &self.stabs.col_z[qu];
        for i in self.stabs.col_x[qu].intersection(&self.stabs.col_z[qu]) {
            self.stabs.signs_minus ^= i;
        }

        for g in [&mut self.stabs, &mut self.destabs] {
            for i in g.col_x[qu].difference(&g.col_z[qu]) {
                let iu = i.to_usize();
                g.row_x[iu].remove(&q);
                g.row_z[iu].insert(q);
            }

            for i in g.col_z[qu].difference(&g.col_x[qu]) {
                let iu = i.to_usize();
                g.row_z[iu].remove(&q);
                g.row_x[iu].insert(q);
            }

            mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
        }
    }

    /// CX: +IX -> +IX; +IZ -> +ZZ; +XI -> +XX; +ZI -> +ZI
    /// # Panics
    /// Will panic if qubit ids don't convert to usize.
    #[inline]
    fn cx(&mut self, q1: E, q2: E) {
        let qu1 = q1.to_usize();
        let qu2 = q2.to_usize();

        for g in &mut [&mut self.stabs, &mut self.destabs] {
            let (qu_min, qu_max) = if qu1 < qu2 { (qu1, qu2) } else { (qu2, qu1) };

            // Handle col_x
            {
                let (_left, right) = g.col_x.split_at_mut(qu_min);
                let (mid, right) = right.split_at_mut(qu_max - qu_min);
                let col_x_min = &mut mid[0];
                let col_x_max = &mut right[0];

                let (col_x_qu1, col_x_qu2) = if qu1 < qu2 {
                    (col_x_min, col_x_max)
                } else {
                    (col_x_max, col_x_min)
                };

                let mut q2_set = T::new();
                q2_set.insert(q2);

                for i in col_x_qu1.iter() {
                    let iu = i.to_usize();
                    g.row_x[iu].symmetric_difference_update(&q2_set);
                }
                col_x_qu2.symmetric_difference_update(col_x_qu1);
            }

            // Handle col_z
            {
                let (_left, right) = g.col_z.split_at_mut(qu_min);
                let (mid, right) = right.split_at_mut(qu_max - qu_min);
                let col_z_min = &mut mid[0];
                let col_z_max = &mut right[0];

                let (col_z_qu1, col_z_qu2) = if qu1 < qu2 {
                    (col_z_min, col_z_max)
                } else {
                    (col_z_max, col_z_min)
                };

                let mut q1_set = T::new();
                q1_set.insert(q1);

                for i in col_z_qu2.iter() {
                    let iu = i.to_usize();
                    g.row_z[iu].symmetric_difference_update(&q1_set);
                }
                col_z_qu1.symmetric_difference_update(col_z_qu2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::VecSet;

    #[allow(clippy::cast_possible_truncation)]
    fn check_matrix(m: &[&str], gens: &Gens<VecSet<u32>, u32>) {
        for (r, v) in m.iter().enumerate() {
            let ru32 = &(r as u32);

            let (_, phase, v) = split_pauli(v);

            // TODO: Allow +Y in place of +iW
            // TODO: Return bools instead of doing the asserts here...

            match phase {
                "+" => {
                    assert!(!gens.signs_minus.contains(ru32));
                    assert!(!gens.signs_i.contains(ru32));
                }
                "-" => {
                    assert!(gens.signs_minus.contains(ru32));
                    assert!(!gens.signs_i.contains(ru32));
                }
                "+i" => {
                    assert!(!gens.signs_minus.contains(ru32));
                    assert!(gens.signs_i.contains(ru32));
                }
                "-i" => {
                    assert!(gens.signs_minus.contains(ru32));
                    assert!(gens.signs_i.contains(ru32));
                }
                _ => unreachable!(),
            }

            for (c, val) in v.chars().enumerate() {
                let cu32 = &(c as u32);
                match val {
                    'I' => {
                        assert!(!gens.col_x[c].contains(ru32));
                        assert!(!gens.col_z[c].contains(ru32));
                        assert!(!gens.row_x[r].contains(cu32));
                        assert!(!gens.row_z[r].contains(cu32));
                    }
                    'X' => {
                        assert!(gens.col_x[c].contains(ru32));
                        assert!(!gens.col_z[c].contains(ru32));
                        assert!(gens.row_x[r].contains(cu32));
                        assert!(!gens.row_z[r].contains(cu32));
                    }
                    'Z' => {
                        assert!(!gens.col_x[c].contains(ru32));
                        assert!(gens.col_z[c].contains(ru32));
                        assert!(!gens.row_x[r].contains(cu32));
                        assert!(gens.row_z[r].contains(cu32));
                    }
                    'W' => {
                        assert!(gens.col_x[c].contains(ru32));
                        assert!(gens.col_z[c].contains(ru32));
                        assert!(gens.row_x[r].contains(cu32));
                        assert!(gens.row_z[r].contains(cu32));
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn check_state(state: &SparseStab<VecSet<u32>, u32>, stabs: &[&str], destabs: &[&str]) {
        check_matrix(stabs, &state.stabs);
        check_matrix(destabs, &state.destabs);
        // SparseStab::verify_matrix(&state);
        // TODO: Add matrix verification func
    }

    fn split_pauli(pauli_str: &str) -> (usize, &str, &str) {
        let (phase, pauli_str) = if pauli_str.contains("+i") || pauli_str.contains("-i") {
            pauli_str.split_at(2)
        } else if pauli_str.contains('+') || pauli_str.contains('-') || pauli_str.contains('i') {
            pauli_str.split_at(1)
        } else {
            ("+", pauli_str)
        };
        let n = pauli_str.chars().count();

        let phase = if phase == "i" { "+i" } else { phase };

        (n, phase, pauli_str)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn prep_pauli_gens(pauli_vec: &[&str], gens: &mut Gens<VecSet<u32>, u32>) {
        // TODO: Think about how to automatically determine the destabilizers you need so you can optionally only provide stabilizers...

        gens.signs_i.clear();
        gens.signs_minus.clear();

        let (n, _, _) = split_pauli(pauli_vec[0]);

        for u in 0..n {
            gens.col_x[u].clear();
            gens.col_z[u].clear();
            gens.row_x[u].clear();
            gens.row_z[u].clear();
        }

        for (ru, pauli_str) in pauli_vec.iter().enumerate() {
            let (n_, phase, pauli_str) = split_pauli(pauli_str);

            assert_eq!(
                n, n_,
                "The number of qubits differs between the first generator and another!"
            );

            match phase {
                "+" => {}
                "-" => {
                    gens.signs_minus.insert(ru as u32);
                }
                "+i" => {
                    gens.signs_i.insert(ru as u32);
                }
                "-i" => {
                    gens.signs_minus.insert(ru as u32);
                    gens.signs_i.insert(ru as u32);
                }
                _ => unreachable!(),
            }

            for (cu, p) in pauli_str.chars().enumerate() {
                match p {
                    'I' => {}
                    'X' => {
                        gens.col_x[cu].insert(ru as u32);
                        gens.row_x[ru].insert(cu as u32);
                    }
                    'W' => {
                        gens.col_x[cu].insert(ru as u32);
                        gens.col_z[cu].insert(ru as u32);
                        gens.row_x[ru].insert(cu as u32);
                        gens.row_z[ru].insert(cu as u32);
                    }
                    'Z' => {
                        gens.col_z[cu].insert(ru as u32);
                        gens.row_z[ru].insert(cu as u32);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn prep_state(stabs: &[&str], destabs: &[&str]) -> SparseStab<VecSet<u32>, u32> {
        let mut state = SparseStab::<VecSet<u32>, u32>::new(3);
        prep_pauli_gens(stabs, &mut state.stabs);
        prep_pauli_gens(destabs, &mut state.destabs);

        state
    }

    #[test]
    fn test_setting_up_stab_state() {
        let tab_stab = vec!["XII", "iIWI", "IIZ"];
        let tab_destab = vec!["ZII", "IXI", "IIX"];

        let state = prep_state(&tab_stab, &tab_destab);
        check_state(&state, &tab_stab, &tab_destab);
    }

    #[test]
    fn test_setting_up_neg_stab_state() {
        let tab_stab = vec!["-XII", "-iIWI", "-IIZ"];
        let tab_destab = vec!["ZII", "IXI", "IIX"];

        let state = prep_state(&tab_stab, &tab_destab);
        check_state(&state, &tab_stab, &tab_destab);
    }

    #[test]
    fn test_nondeterministic_px() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let (_m0, d0) = state.px(0);
            let (m1, d1) = state.mx(0);
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 0); // |+X>
            assert!(!d0); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_px() {
        let mut state = prep_state(&["X"], &["Z"]);
        let (m0, d0) = state.px(0);
        let m0_int = u8::from(m0);

        assert!(d0); // Deterministic
        assert_eq!(m0_int, 0); // |+X>
    }

    #[test]
    fn test_nondeterministic_pnx() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let (_m0, d0) = state.pnx(0);
            let (m1, d1) = state.mx(0);
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 1); // |-X>
            assert!(!d0); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pnx() {
        let mut state = prep_state(&["-X"], &["Z"]);
        let (m0, d0) = state.pnx(0);
        let m0_int = u8::from(m0);

        assert!(d0); // Deterministic
        assert_eq!(m0_int, 0); // |-X>
    }

    #[test]
    fn test_nondeterministic_py() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let (_m0, d0) = state.py(0);
            let (m1, d1) = state.my(0);
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 0); // |+Y>
            assert!(!d0); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_py() {
        let mut state = prep_state(&["iW"], &["Z"]);
        let (m0, d0) = state.py(0);
        let m0_int = u8::from(m0);

        assert!(d0); // Deterministic
        assert_eq!(m0_int, 0); // |+Y>
    }

    #[test]
    fn test_nondeterministic_pny() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let (_m0, d0) = state.pny(0);
            let (m1, d1) = state.my(0);
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 1); // |-Y>
            assert!(!d0); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pny() {
        let mut state = prep_state(&["-iW"], &["Z"]);
        let (m0, d0) = state.pny(0);
        let m0_int = u8::from(m0);

        assert!(d0); // Deterministic
        assert_eq!(m0_int, 0); // |-Y>
    }

    #[test]
    fn test_nondeterministic_pz() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["X"], &["Z"]);
            let (_m0, d0) = state.pz(0);
            let (m1, d1) = state.mz(0);
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 0); // |0>
            assert!(!d0); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pz() {
        let mut state = prep_state(&["Z"], &["X"]);
        let (m0, d0) = state.pz(0);
        let m0_int = u8::from(m0);

        assert!(d0); // Deterministic
        assert_eq!(m0_int, 0); // |+Z>
    }

    #[test]
    fn test_nondeterministic_pnz() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["X"], &["Z"]);
            let (_m0, d0) = state.pnz(0);
            let (m1, d1) = state.mz(0);
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 1); // |1>
            assert!(!d0); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pnz() {
        let mut state = prep_state(&["-Z"], &["X"]);
        let (m0, d0) = state.pnz(0);
        let m0_int = u8::from(m0);

        assert!(d0); // Deterministic
        assert_eq!(m0_int, 0); // |-Z>
    }

    #[test]
    fn test_nondeterministic_mx() {
        let mut state = prep_state(&["Z"], &["X"]);
        let (_meas, determined) = state.mx(0);
        assert!(!determined);
    }

    #[test]
    fn test_deterministic_mx() {
        let mut state0 = prep_state(&["X"], &["Z"]);
        let (meas0, determined0) = state0.mx(0);
        assert!(determined0);
        assert!(!meas0);

        let mut state1 = prep_state(&["-X"], &["Z"]);
        let (meas1, determined1) = state1.mx(0);
        assert!(determined1);
        assert!(meas1);
    }

    #[test]
    fn test_nondeterministic_mnx() {
        let mut state = prep_state(&["Z"], &["X"]);
        let (_meas, determined) = state.mnx(0);
        assert!(!determined);
    }

    #[test]
    fn test_deterministic_mnx() {
        let mut state0 = prep_state(&["-X"], &["Z"]);
        let (meas0, determined0) = state0.mnx(0);
        assert!(determined0);
        assert!(!meas0);

        let mut state1 = prep_state(&["X"], &["Z"]);
        let (meas1, determined1) = state1.mnx(0);
        assert!(determined1);
        assert!(meas1);
    }

    #[test]
    fn test_nondeterministic_my() {
        let mut state = prep_state(&["Z"], &["X"]);
        let (_meas, determined) = state.my(0);
        assert!(!determined);
    }

    #[test]
    fn test_deterministic_my() {
        let mut state0 = prep_state(&["iW"], &["Z"]);
        let (meas0, determined0) = state0.my(0);
        assert!(determined0);
        assert!(!meas0);

        let mut state1 = prep_state(&["-iW"], &["Z"]);
        let (meas1, determined1) = state1.my(0);
        assert!(determined1);
        assert!(meas1);
    }

    #[test]
    fn test_nondeterministic_mny() {
        let mut state = prep_state(&["Z"], &["X"]);
        let (_meas, determined) = state.mny(0);
        assert!(!determined);
    }

    #[test]
    fn test_deterministic_mny() {
        let mut state0 = prep_state(&["-iW"], &["Z"]);
        let (meas0, determined0) = state0.mny(0);
        assert!(determined0);
        assert!(!meas0);

        let mut state1 = prep_state(&["iW"], &["Z"]);
        let (meas1, determined1) = state1.mny(0);
        assert!(determined1);
        assert!(meas1);
    }

    #[test]
    fn test_nondeterministic_mz() {
        let mut state = prep_state(&["X"], &["Z"]);
        let (_meas, determined) = state.mz(0);
        assert!(!determined);
    }

    #[test]
    fn test_deterministic_mz() {
        let mut state0 = prep_state(&["Z"], &["X"]);
        let (meas0, determined0) = state0.mz(0);
        assert!(determined0);
        assert!(!meas0);

        let mut state1 = prep_state(&["-Z"], &["X"]);
        let (meas1, determined1) = state1.mz(0);
        assert!(determined1);
        assert!(meas1);
    }

    #[test]
    fn test_nondeterministic_mnz() {
        let mut state = prep_state(&["X"], &["Z"]);
        let (_meas, determined) = state.mnz(0);
        assert!(!determined);
    }

    #[test]
    fn test_deterministic_mnz() {
        let mut state0 = prep_state(&["Z"], &["X"]);
        let (meas0, determined0) = state0.mnz(0);
        assert!(determined0);
        assert!(meas0);

        let mut state1 = prep_state(&["-Z"], &["X"]);
        let (meas1, determined1) = state1.mnz(0);
        assert!(determined1);
        assert!(!meas1);
    }

    #[test]
    fn test_identity() {
        // I: +X -> +X; +Z -> +Z; +Y -> +Y;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.identity(0);
        check_state(&state, &["X"], &["Z"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.identity(0);
        check_state(&state, &["iW"], &["X"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.identity(0);
        check_state(&state, &["Z"], &["X"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.identity(1);
        check_state(&state, &["-iIWI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_x() {
        // X: +X -> +X; +Z -> -Z; +Y -> -Y;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.x(0);
        check_state(&state, &["X"], &["Z"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.x(0);
        check_state(&state, &["-iW"], &["X"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.x(0);
        check_state(&state, &["-Z"], &["X"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.x(1);
        check_state(&state, &["iIWI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_y() {
        // Y: +X -> -X; +Z -> -Z; +Y -> +Y;

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.y(0);
        check_state(&state, &["-X"], &["Z"]);

        // +Y -> +Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.y(0);
        check_state(&state, &["iW"], &["X"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.y(0);
        check_state(&state, &["-Z"], &["X"]);

        // -IXI -> +IXI
        let mut state = prep_state(&["-IXI"], &["IZI"]);
        state.y(1);
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_z() {
        // Z: +X -> -X; +Z -> +Z; +Y -> -Y;

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.z(0);
        check_state(&state, &["-X"], &["Z"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.z(0);
        check_state(&state, &["-iW"], &["X"]);

        // +Z -> +Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.z(0);
        check_state(&state, &["Z"], &["X"]);

        // -IXI -> +IXI
        let mut state = prep_state(&["-IXI"], &["IZI"]);
        state.z(1);
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sx() {
        // SX: +X -> +X; +Z -> -Y; +Y -> +Z;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.sx(0);
        check_state(&state, &["X"], &["W"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.sx(0);
        check_state(&state, &["Z"], &["X"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.sx(0);
        check_state(&state, &["-iW"], &["X"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sx(1);
        check_state(&state, &["-IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sxdg() {
        // SXdg: +X -> +X; +Z -> +Y; +Y -> -Z;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.sxdg(0);
        check_state(&state, &["X"], &["W"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.sxdg(0);
        check_state(&state, &["-Z"], &["X"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.sxdg(0);
        check_state(&state, &["iW"], &["X"]);

        // -IYI -> +IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sxdg(1);
        check_state(&state, &["IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sy() {
        // SY: +X -> -Z; +Z -> +X; +Y -> +Y;

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.sy(0);
        check_state(&state, &["-Z"], &["X"]);

        // +Y -> +Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.sy(0);
        check_state(&state, &["iW"], &["Z"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.sy(0);
        check_state(&state, &["X"], &["Z"]);

        // -IYI -> -IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sy(1);
        check_state(&state, &["-iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sydg() {
        // SYdg: +X -> +Z; +Z -> -X; +Y -> +Y;

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.sydg(0);
        check_state(&state, &["Z"], &["X"]);

        // +Y -> +Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.sydg(0);
        check_state(&state, &["iW"], &["Z"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.sydg(0);
        check_state(&state, &["-X"], &["Z"]);

        // -IYI -> -IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sydg(1);
        check_state(&state, &["-iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sz() {
        // SZ: +X -> +Y; +Z -> +Z; +Y -> -X;

        // +X -> +Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.sz(0);
        check_state(&state, &["iW"], &["Z"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.sz(0);
        check_state(&state, &["-X"], &["W"]);

        // +Z -> +Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.sz(0);
        check_state(&state, &["Z"], &["W"]);

        // -IYI -> +IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sz(1);
        check_state(&state, &["IXI"], &["IWI"]);
    }

    #[test]
    fn test_szdg() {
        // SZdg: +X -> -Y; +Z -> +Z; +Y -> +X;

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.szdg(0);
        check_state(&state, &["-iW"], &["Z"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.szdg(0);
        check_state(&state, &["X"], &["W"]);

        // +Z -> +Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.szdg(0);
        check_state(&state, &["Z"], &["W"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.szdg(1);
        check_state(&state, &["-IXI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h() {
        // H: X -> Z; Z -> X; Y -> -Y;

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.h(0);
        check_state(&state, &["Z"], &["X"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.h(0);
        check_state(&state, &["-iW"], &["Z"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.h(0);
        check_state(&state, &["X"], &["Z"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h(1);
        check_state(&state, &["iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h2() {
        // H2: X -> -Z, Z -> -X, Y -> -Y

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.h2(0);
        check_state(&state, &["-Z"], &["X"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.h2(0);
        check_state(&state, &["-iW"], &["Z"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.h2(0);
        check_state(&state, &["-X"], &["Z"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h2(1);
        check_state(&state, &["iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h3() {
        // H3: X -> Y, Z -> -Z, Y -> X

        // +X -> Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.h3(0);
        check_state(&state, &["iW"], &["Z"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.h3(0);
        check_state(&state, &["X"], &["W"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.h3(0);
        check_state(&state, &["-Z"], &["W"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h3(1);
        check_state(&state, &["-IXI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h4() {
        // H4: X -> -Y, Z -> -Z, Y -> -X

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.h4(0);
        check_state(&state, &["-iW"], &["Z"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.h4(0);
        check_state(&state, &["-X"], &["W"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.h4(0);
        check_state(&state, &["-Z"], &["W"]);

        // -IYI -> IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h4(1);
        check_state(&state, &["IXI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h5() {
        // H5: X -> -X, Z -> Y, Y -> Z

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.h5(0);
        check_state(&state, &["-X"], &["W"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.h5(0);
        check_state(&state, &["Z"], &["X"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.h5(0);
        check_state(&state, &["iW"], &["X"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h5(1);
        check_state(&state, &["-IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h6() {
        // H6: X -> -X, Z -> -Y, Y -> -Z

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.h6(0);
        check_state(&state, &["-X"], &["W"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.h6(0);
        check_state(&state, &["-Z"], &["X"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.h6(0);
        check_state(&state, &["-iW"], &["X"]);

        // -IYI -> IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h6(1);
        check_state(&state, &["IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f() {
        // F: X -> Y, Z -> X, Y -> Z

        // +X -> +Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f(0);
        check_state(&state, &["iW"], &["X"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f(0);
        check_state(&state, &["Z"], &["W"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f(0);
        check_state(&state, &["X"], &["W"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f(1);
        check_state(&state, &["-IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_fdg() {
        // Fdg: X -> Z, Z -> Y, Y -> X

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.fdg(0);
        check_state(&state, &["Z"], &["W"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.fdg(0);
        check_state(&state, &["X"], &["Z"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.fdg(0);
        check_state(&state, &["iW"], &["Z"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.fdg(1);
        check_state(&state, &["-IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f2() {
        // F2: X -> -Z, Z -> Y, Y -> -X

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.f2(0);
        check_state(&state, &["-Z"], &["W"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.f2(0);
        check_state(&state, &["-X"], &["Z"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.f2(0);
        check_state(&state, &["iW"], &["Z"]);

        // -IYI -> IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f2(1);
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f2dg() {
        // F2dg: X -> -Y, Z -> -X, Y -> Z

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f2dg(0);
        check_state(&state, &["-iW"], &["X"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f2dg(0);
        check_state(&state, &["Z"], &["W"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f2dg(0);
        check_state(&state, &["-X"], &["W"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f2dg(1);
        check_state(&state, &["-IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f3() {
        // F3: X -> Y, Z -> -X, Y -> -Z

        // +X -> +Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f3(0);
        check_state(&state, &["iW"], &["X"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f3(0);
        check_state(&state, &["-Z"], &["W"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f3(0);
        check_state(&state, &["-X"], &["W"]);

        // -IYI -> IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f3(1);
        check_state(&state, &["IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f3dg() {
        // F3dg: X -> -Z, Z -> -Y, Y -> X

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.f3dg(0);
        check_state(&state, &["-Z"], &["W"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.f3dg(0);
        check_state(&state, &["X"], &["Z"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.f3dg(0);
        check_state(&state, &["-iW"], &["Z"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f3dg(1);
        check_state(&state, &["-IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f4() {
        // F4: X -> Z, Z -> -Y, Y -> -X

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.f4(0);
        check_state(&state, &["Z"], &["W"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.f4(0);
        check_state(&state, &["-X"], &["Z"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.f4(0);
        check_state(&state, &["-iW"], &["Z"]);

        // -IYI -> IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f4(1);
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f4dg() {
        // F4dg: X -> -Y, Z -> X, Y -> -Z

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f4dg(0);
        check_state(&state, &["-iW"], &["X"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f4dg(0);
        check_state(&state, &["-Z"], &["W"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f4dg(0);
        check_state(&state, &["X"], &["W"]);

        // -IYI -> +IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f4dg(1);
        check_state(&state, &["IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_cx() {
        // CX: +IX -> +IX; +IZ -> +ZZ; +XI -> +XX; +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +IX
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.cx(0, 1);
        check_state(&state, &["IX"], &["ZZ"]);

        // +IZ -> +ZZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.cx(0, 1);
        check_state(&state, &["ZZ"], &["IX"]);

        // +XI -> +XX
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.cx(0, 1);
        check_state(&state, &["XX"], &["ZI"]);

        // +ZI -> +ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.cx(0, 1);
        check_state(&state, &["ZI"], &["XX"]);
    }

    #[test]
    fn test_cy() {
        // CY: +IX -> +ZX; +IZ -> +ZZ; +XI -> -XY; +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +ZX
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.cy(0, 1);
        check_state(&state, &["ZX"], &["ZZ"]);

        // +IZ -> +ZZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.cy(0, 1);
        check_state(&state, &["ZZ"], &["ZX"]);

        // +XI -> -XY
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.cy(0, 1);
        check_state(&state, &["-iXW"], &["ZI"]);

        // +ZI -> +ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.cy(0, 1);
        check_state(&state, &["ZI"], &["XW"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_cz() {
        // CZ: +IX -> +ZX; +IZ -> +IZ; +XI -> +XZ; +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +ZX
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.cz(0, 1);
        check_state(&state, &["ZX"], &["IZ"]);

        // +IZ -> +IZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.cz(0, 1);
        check_state(&state, &["IZ"], &["ZX"]);

        // +XI -> +XZ
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.cz(0, 1);
        check_state(&state, &["XZ"], &["ZI"]);

        // +ZI -> +ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.cz(0, 1);
        check_state(&state, &["ZI"], &["XZ"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sxx() {
        // SXX: XI -> XI
        //      IX -> IX
        //      ZI -> -YX
        //      IZ -> -XY

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.sxx(0, 1);
        check_state(&state, &["IX"], &["XW"]);

        // +IZ -> -XY
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.sxx(0, 1);
        check_state(&state, &["-iXW"], &["IX"]);

        // +XI -> +XI
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.sxx(0, 1);
        check_state(&state, &["XI"], &["WX"]);

        // +ZI -> -YX
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.sxx(0, 1);
        check_state(&state, &["-iWX"], &["XI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sxxdg() {
        // SXXdg: XI -> XI
        //        IX -> IX
        //        ZI -> YX
        //        IZ -> XY

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.sxxdg(0, 1);
        check_state(&state, &["IX"], &["XW"]);

        // +IZ -> +XY
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.sxxdg(0, 1);
        check_state(&state, &["iXW"], &["IX"]);

        // +XI -> +XI
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.sxxdg(0, 1);
        check_state(&state, &["XI"], &["WX"]);

        // +ZI -> +YX
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.sxxdg(0, 1);
        check_state(&state, &["iWX"], &["XI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_syy() {
        // SYY: XI -> -ZY
        //      IX -> -YZ
        //      ZI -> XY
        //      IZ -> YX

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> -YZ
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.syy(0, 1);
        check_state(&state, &["-iWZ"], &["WX"]);

        // +IZ -> +YX
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.syy(0, 1);
        check_state(&state, &["iWX"], &["WZ"]);

        // +XI -> -ZY
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.syy(0, 1);
        check_state(&state, &["-iZW"], &["XW"]);

        // +ZI -> +XY
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.syy(0, 1);
        check_state(&state, &["iXW"], &["ZW"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_syydg() {
        // SYYdg: XI -> ZY
        //        IX -> YZ
        //        ZI -> -XY
        //        IZ -> -YX

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> YZ
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.syydg(0, 1);
        check_state(&state, &["iWZ"], &["WX"]);

        // +IZ -> -YX
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.syydg(0, 1);
        check_state(&state, &["-iWX"], &["WZ"]);

        // +XI -> ZY
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.syydg(0, 1);
        check_state(&state, &["iZW"], &["XW"]);

        // +ZI -> +XY
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.syydg(0, 1);
        check_state(&state, &["-iXW"], &["ZW"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_szz() {
        // SZZ: +IX -> +ZY;
        //      +IZ -> +IZ;
        //      +XI -> +ZY;
        //      +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> ZY
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.szz(0, 1);
        check_state(&state, &["iZW"], &["IZ"]);

        // +IZ -> IZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.szz(0, 1);
        check_state(&state, &["IZ"], &["ZW"]);

        // +XI -> YZ
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.szz(0, 1);
        check_state(&state, &["iWZ"], &["ZI"]);

        // +ZI -> ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.szz(0, 1);
        check_state(&state, &["ZI"], &["WZ"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_szzdg() {
        // SZZ: +IX -> -ZY;
        //      +IZ -> +IZ;
        //      +XI -> -ZY;
        //      +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> -ZY
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.szzdg(0, 1);
        check_state(&state, &["-iZW"], &["IZ"]);

        // +IZ -> IZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.szzdg(0, 1);
        check_state(&state, &["IZ"], &["ZW"]);

        // +XI -> -YZ
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.szzdg(0, 1);
        check_state(&state, &["-iWZ"], &["ZI"]);

        // +ZI -> ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.szzdg(0, 1);
        check_state(&state, &["ZI"], &["WZ"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_swap() {
        // SWAP: +IX -> +XI;
        //       +IZ -> +ZI;
        //       +XI -> +IX;
        //       +ZI -> +IZ;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.swap(0, 1);
        check_state(&state, &["XI"], &["ZI"]);

        // +IZ -> +ZI
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.swap(0, 1);
        check_state(&state, &["ZI"], &["XI"]);

        // +XI -> +IX
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.swap(0, 1);
        check_state(&state, &["IX"], &["IZ"]);

        // +ZI -> +IZ
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.swap(0, 1);
        check_state(&state, &["IZ"], &["IX"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_g2() {
        // G2: +XI -> +IX
        //     +IX -> +XI
        //     +ZI -> +XZ
        //     +IZ -> +ZX

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.g2(0, 1);
        check_state(&state, &["XI"], &["ZX"]);

        // +IZ -> +ZX
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.g2(0, 1);
        check_state(&state, &["ZX"], &["XI"]);

        // +XI -> +IX
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.g2(0, 1);
        check_state(&state, &["IX"], &["XZ"]);

        // +ZI -> +XZ
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.g2(0, 1);
        check_state(&state, &["XZ"], &["IX"]);
    }

    fn one_bit_z_teleport(
        mut state: SparseStab<VecSet<u32>, u32>,
    ) -> (SparseStab<VecSet<u32>, u32>, bool) {
        state.cx(1, 0);
        state.h(1);
        let (m1, d1) = state.mz(1);
        if m1 {
            state.z(0);
        }
        (state, d1)
    }

    /// Test one-bit Z teleportation of |+X>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_x() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(2);
            state.h(1); // Set input to |+>
            (state, d1) = one_bit_z_teleport(state);
            // X basis meas
            state.h(0);
            let (m0, d0) = state.mz(0);
            let m0_int = u8::from(m0);
            assert_eq!(m0_int, 0); // |+> -> 0 == false
            assert!(!d1); // Not deterministic
            assert!(d0); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |-X>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_nx() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(2);
            state.x(1);
            state.h(1); // Set input to |->
            (state, d1) = one_bit_z_teleport(state);
            // X basis meas
            state.h(0);
            let (m0, d0) = state.mz(0);
            let m0_int = u8::from(m0);
            assert_eq!(m0_int, 1); // |-> -> 1 == true
            assert!(!d1); // Not deterministic
            assert!(d0); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |+Y>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_y() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(2);
            state.sxdg(1); // Set input to |+i>
            (state, d1) = one_bit_z_teleport(state);
            // Y basis meas
            state.sx(0); // Y -> Z
            let (m0, d0) = state.mz(0);
            let m0_int = u8::from(m0);
            assert_eq!(m0_int, 0); // |+X> -> 0 == false
            assert!(!d1); // Not deterministic
            assert!(d0); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |-Y>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_ny() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(2);
            state.x(1);
            state.sxdg(1); // Set input to |-i>
            (state, d1) = one_bit_z_teleport(state);
            // Y basis meas
            state.sx(0); // Y -> Z
            let (m0, d0) = state.mz(0);
            let m0_int = u8::from(m0);
            assert_eq!(m0_int, 1); // |-Y> -> 1 == true
            assert!(!d1); // Not deterministic
            assert!(d0); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |+Z>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_z() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(2);
            // Set input to |0>
            (state, d1) = one_bit_z_teleport(state);
            let (m0, d0) = state.mz(0);
            let m0_int = u8::from(m0);
            assert_eq!(m0_int, 0); // |0>
            assert!(!d1); // Not deterministic
            assert!(d0); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |-Z>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_nz() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(2);
            state.x(1); // Set input to |1>
            (state, d1) = one_bit_z_teleport(state);
            let (m0, d0) = state.mz(0);
            let m0_int = u8::from(m0);
            assert_eq!(m0_int, 1); // |1> -> 1 == true
            assert!(!d1); // Not deterministic
            assert!(d0); // Deterministic
        }
    }

    fn teleport(
        mut state: SparseStab<VecSet<u32>, u32>,
    ) -> (SparseStab<VecSet<u32>, u32>, bool, bool) {
        // |psi> -----.-H-MZ=m0
        //            |
        // |0>   -H-.-X---MZ=m1
        //          |
        // |0>   ---X------------X^m1-Z^m0-MZ=m2

        state.h(1);
        state.cx(1, 2);
        state.cx(0, 1);
        state.h(0);
        let (m0, d0) = state.mz(0);
        let (m1, d1) = state.mz(1);
        if m1 {
            state.x(2);
        }
        if m0 {
            state.z(2);
        }
        (state, d0, d1)
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_x() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(3);
            state.h(0);
            (state, d0, d1) = teleport(state);
            state.h(2);
            let (m2, d2) = state.mz(2);
            let m2_int = u8::from(m2);
            assert_eq!(m2_int, 0);
            assert!(!d0);
            assert!(!d1);
            assert!(d2);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_nx() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(3);
            state.x(0);
            state.h(0);
            (state, d0, d1) = teleport(state);
            state.h(2);
            let (m2, d2) = state.mz(2);
            let m2_int = u8::from(m2);

            assert_eq!(m2_int, 1);
            assert!(!d0);
            assert!(!d1);
            assert!(d2);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_y() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(3);
            state.sxdg(0);
            (state, d0, d1) = teleport(state);
            state.sx(2);
            let (m2, d2) = state.mz(2);
            let m2_int = u8::from(m2);
            assert_eq!(m2_int, 0);
            assert!(!d0);
            assert!(!d1);
            assert!(d2);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_ny() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(3);
            state.x(0);
            state.sxdg(0);
            (state, d0, d1) = teleport(state);
            state.sx(2);
            let (m2, d2) = state.mz(2);
            let m2_int = u8::from(m2);
            assert_eq!(m2_int, 1);
            assert!(!d0);
            assert!(!d1);
            assert!(d2);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_z() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(3);
            (state, d0, d1) = teleport(state);
            let (m2, d2) = state.mz(2);
            let m2_int = u8::from(m2);

            assert_eq!(m2_int, 0);
            assert!(!d0);
            assert!(!d1);
            assert!(d2);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_nz() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab<VecSet<u32>, u32> = SparseStab::new(3);
            state.x(0); // input state |-Z>
            (state, d0, d1) = teleport(state);
            let (m2, d2) = state.mz(2);
            let m2_int = u8::from(m2);

            assert_eq!(m2_int, 1);
            assert!(!d0);
            assert!(!d1);
            assert!(d2);
        }
    }

    // TODO: Consider "forcing" the random number for cleaner testing.
    // TODO: Consider a seed to still have random numbers but make them predictable
}
