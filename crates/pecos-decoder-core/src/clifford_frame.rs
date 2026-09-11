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

//! Shared sign-free Clifford transformations for logical Pauli frames.

use std::ops::BitXor;

/// Apply a Hadamard to paired X/Z frame components.
#[must_use]
pub const fn conjugate_h(x: bool, z: bool) -> (bool, bool) {
    (z, x)
}

/// Apply an S gate to paired X/Z frame components, ignoring Pauli sign.
#[must_use]
pub const fn conjugate_s(x: bool, z: bool) -> (bool, bool) {
    (x, z ^ x)
}

/// Apply a control-to-target CNOT to paired control and target X/Z components.
#[must_use]
pub fn conjugate_cnot<T>(control_x: T, control_z: T, target_x: T, target_z: T) -> (T, T, T, T)
where
    T: Copy + BitXor<Output = T>,
{
    (
        control_x,
        control_z ^ target_z,
        target_x ^ control_x,
        target_z,
    )
}

/// Real Clifford gates supported by the two-patch canonical-state machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwoPatchClifford {
    /// Hadamard on the first patch.
    HadamardFirst,
    /// Hadamard on the second patch.
    HadamardSecond,
    /// CNOT from the first patch to the second.
    Cnot,
}

/// Transform an encoded sign-free `(x0, z0, x1, z1)` Pauli mask.
///
/// Bits zero through three carry the four components in that order. Higher
/// bits are preserved so callers cannot accidentally discard adjacent state.
#[must_use]
pub fn transform_two_patch_pauli(pauli: u8, gate: TwoPatchClifford) -> u8 {
    let mut x0 = pauli & 1 != 0;
    let mut z0 = pauli & 2 != 0;
    let mut x1 = pauli & 4 != 0;
    let mut z1 = pauli & 8 != 0;
    match gate {
        TwoPatchClifford::HadamardFirst => (x0, z0) = conjugate_h(x0, z0),
        TwoPatchClifford::HadamardSecond => (x1, z1) = conjugate_h(x1, z1),
        TwoPatchClifford::Cnot => (x0, z0, x1, z1) = conjugate_cnot(x0, z0, x1, z1),
    }
    (pauli & !0x0f) | u8::from(x0) | (u8::from(z0) << 1) | (u8::from(x1) << 2) | (u8::from(z1) << 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_paulis_follow_the_clifford_conjugation_rules() {
        assert_eq!(
            transform_two_patch_pauli(0b0001, TwoPatchClifford::HadamardFirst),
            0b0010
        );
        assert_eq!(
            transform_two_patch_pauli(0b0010, TwoPatchClifford::HadamardFirst),
            0b0001
        );
        assert_eq!(
            transform_two_patch_pauli(0b0100, TwoPatchClifford::HadamardSecond),
            0b1000
        );
        assert_eq!(
            transform_two_patch_pauli(0b1000, TwoPatchClifford::HadamardSecond),
            0b0100
        );
        assert_eq!(
            transform_two_patch_pauli(0b0001, TwoPatchClifford::Cnot),
            0b0101
        );
        assert_eq!(
            transform_two_patch_pauli(0b1000, TwoPatchClifford::Cnot),
            0b1010
        );
    }

    #[test]
    fn h_and_cnot_are_involutions_for_every_two_patch_pauli() {
        for gate in [
            TwoPatchClifford::HadamardFirst,
            TwoPatchClifford::HadamardSecond,
            TwoPatchClifford::Cnot,
        ] {
            for pauli in 0..16 {
                assert_eq!(
                    transform_two_patch_pauli(transform_two_patch_pauli(pauli, gate), gate),
                    pauli
                );
            }
        }
    }

    #[test]
    fn generic_cnot_supports_parallel_bit_masks() {
        assert_eq!(
            conjugate_cnot(0b0101_u64, 0b0010, 0b1000, 0b1010),
            (0b0101, 0b1000, 0b1101, 0b1010)
        );
    }
}
