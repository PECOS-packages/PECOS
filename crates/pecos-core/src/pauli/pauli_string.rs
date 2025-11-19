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

use crate::{
    IndexableElement, Pauli, PauliBitmap, PauliOperator, PauliSparse, QuarterPhase, QubitId, VecSet,
};

/// A string of Pauli operators acting on multiple qubits
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliString {
    phase: QuarterPhase,
    paulis: Vec<(Pauli, QubitId)>,
}

impl Default for PauliString {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: make sure all these conversions are fast and safe. especially all this get stuff...

impl PauliString {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            phase: QuarterPhase::PlusOne,
            paulis: Vec::new(),
        }
    }

    /// Create a `PauliString` with the given phase and paulis
    #[inline]
    #[must_use]
    pub fn with_phase_and_paulis(phase: QuarterPhase, paulis: Vec<(Pauli, QubitId)>) -> Self {
        Self { phase, paulis }
    }

    #[inline]
    #[must_use]
    pub fn get_phase(&self) -> QuarterPhase {
        self.phase
    }

    #[inline]
    #[must_use]
    pub fn get_paulis(&self) -> &Vec<(Pauli, QubitId)> {
        &self.paulis
    }

    // Conversion to efficient representations
    /// # Errors
    ///
    /// Results in an error if failed to create a valid `PauliSparse`
    pub fn into_pauli_sparse(self) -> Result<PauliSparse<VecSet<usize>>, String> {
        // Convert to SetPauli representation
        let mut x_positions = Vec::new();
        let mut y_positions = Vec::new();
        let mut z_positions = Vec::new();

        for (pauli, qubit) in self.paulis {
            let idx = qubit.to_index();
            match pauli {
                Pauli::X => x_positions.push(idx),
                Pauli::Z => z_positions.push(idx),
                Pauli::Y => y_positions.push(idx),
                Pauli::I => {}
            }
        }

        PauliSparse::with_operators(self.phase, &x_positions, &y_positions, &z_positions)
    }

    /// # Errors
    ///
    /// Results in an error if `QubitId`s are larger than 64 bits or if failed to create a valid `PauliBitmap`
    pub fn into_pauli_bitmap(self) -> Result<PauliBitmap, String> {
        // Convert to BitSetPauli if all qubits are < 64
        if self.paulis.iter().any(|(_, q)| q.to_index() >= 64) {
            return Err("QubitId larger than 64 bits".to_string());
        }

        let mut x_positions = Vec::new();
        let mut y_positions = Vec::new();
        let mut z_positions = Vec::new();

        for (pauli, qubit) in self.paulis {
            let idx = qubit.to_index() as u64;
            match pauli {
                Pauli::X => x_positions.push(idx),
                Pauli::Z => z_positions.push(idx),
                Pauli::Y => y_positions.push(idx),
                Pauli::I => {}
            }
        }

        PauliBitmap::with_operators(self.phase, &x_positions, &y_positions, &z_positions)
    }
}

impl From<PauliSparse<VecSet<usize>>> for PauliString {
    fn from(pauli_sparse: PauliSparse<VecSet<usize>>) -> Self {
        let mut paulis = Vec::new();

        // Collect all qubit positions
        let mut all_positions: Vec<_> = pauli_sparse
            .x_positions()
            .iter()
            .chain(pauli_sparse.z_positions().iter())
            .copied()
            .collect();
        all_positions.sort_unstable();
        all_positions.dedup();

        // Determine Pauli operator for each position
        for pos in all_positions {
            let qubit = QubitId::from_index(pos);
            let pauli = match (
                pauli_sparse.x_positions().contains(&pos),
                pauli_sparse.z_positions().contains(&pos),
            ) {
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
                (false, false) => continue,
            };
            paulis.push((pauli, qubit));
        }

        Self {
            phase: pauli_sparse.phase(),
            paulis,
        }
    }
}

impl TryFrom<PauliBitmap> for PauliString {
    type Error = &'static str;

    fn try_from(pauli_bit: PauliBitmap) -> Result<Self, Self::Error> {
        let mut paulis = Vec::new();

        // Iterate through set bits in both x_bits and z_bits
        for i in 0..64 {
            let x_set = (pauli_bit.get_x_bits() >> i) & 1 == 1;
            let z_set = (pauli_bit.get_z_bits() >> i) & 1 == 1;

            let pauli = match (x_set, z_set) {
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
                (false, false) => continue,
            };

            paulis.push((pauli, QubitId::from_index(i)));
        }

        Ok(Self {
            phase: pauli_bit.phase(),
            paulis,
        })
    }
}
