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

//! Pauli basis types for Lindblad -> Pauli-Lindblad synthesis.

use std::fmt;
use std::str::FromStr;

/// Single-qubit Pauli operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum Pauli1 {
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

impl Pauli1 {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'I' | 'i' => Some(Pauli1::I),
            'X' | 'x' => Some(Pauli1::X),
            'Y' | 'y' => Some(Pauli1::Y),
            'Z' | 'z' => Some(Pauli1::Z),
            _ => None,
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Pauli1::I => 'I',
            Pauli1::X => 'X',
            Pauli1::Y => 'Y',
            Pauli1::Z => 'Z',
        }
    }

    /// Pauli multiplication ignoring global phase. Returns the Hermitian
    /// Pauli factor: XY -> Z (phase `i` dropped), etc. Safe for our use in
    /// `PauliLindbladModel::sample`, where rates are carried by commuting
    /// structure and phases cancel in `P rho P^dag` style actions.
    pub fn multiply(self, other: Pauli1) -> Pauli1 {
        use Pauli1::*;
        match (self, other) {
            (I, x) | (x, I) => x,
            (X, X) | (Y, Y) | (Z, Z) => I,
            (X, Y) | (Y, X) => Z,
            (Y, Z) | (Z, Y) => X,
            (X, Z) | (Z, X) => Y,
        }
    }

    /// 1 if two single-qubit Paulis anticommute, 0 if they commute.
    pub fn anticommutes_with(self, other: Pauli1) -> u8 {
        use Pauli1::*;
        match (self, other) {
            (I, _) | (_, I) => 0,
            (a, b) if a == b => 0,
            _ => 1,
        }
    }
}

/// Multi-qubit Pauli string. Index 0 = leftmost factor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PauliString(pub Vec<Pauli1>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParsePauliStringError {
    invalid_char: char,
}

impl fmt::Display for ParsePauliStringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid Pauli character {:?}", self.invalid_char)
    }
}

impl std::error::Error for ParsePauliStringError {}

impl PauliString {
    pub fn single(p: Pauli1) -> Self {
        PauliString(vec![p])
    }

    pub fn from_label(s: &str) -> Option<Self> {
        s.chars()
            .map(Pauli1::from_char)
            .collect::<Option<Vec<_>>>()
            .map(PauliString)
    }

    pub fn num_qubits(&self) -> usize {
        self.0.len()
    }

    /// Weight (number of non-identity factors).
    pub fn weight(&self) -> usize {
        self.0.iter().filter(|&&p| p != Pauli1::I).count()
    }

    /// Is this the identity string?
    pub fn is_identity(&self) -> bool {
        self.weight() == 0
    }

    /// Elementwise product with global phase dropped. See
    /// [`Pauli1::multiply`].
    pub fn multiply(&self, other: &PauliString) -> PauliString {
        assert_eq!(self.num_qubits(), other.num_qubits(), "ragged multiply");
        PauliString(
            self.0
                .iter()
                .zip(&other.0)
                .map(|(a, b)| a.multiply(*b))
                .collect(),
        )
    }

    /// Symplectic product `<self, other>_sp`: 1 if the two strings
    /// anticommute, 0 if they commute. Equal to (sum of pairwise
    /// anticommutes) mod 2.
    pub fn symplectic_product(&self, other: &PauliString) -> u8 {
        assert_eq!(self.num_qubits(), other.num_qubits(), "ragged symplectic");
        self.0
            .iter()
            .zip(&other.0)
            .map(|(a, b)| a.anticommutes_with(*b))
            .sum::<u8>()
            & 1
    }

    /// Enumerate all non-identity Pauli strings on `n` qubits. Length
    /// 4^n - 1 = 3, 15, 63, ...
    pub fn enumerate_nonidentity(n: usize) -> Vec<PauliString> {
        let total = 1usize << (2 * n);
        (1..total)
            .map(|idx| {
                // idx in base 4: two bits per qubit, low bits = rightmost factor
                let mut qs = Vec::with_capacity(n);
                for q in 0..n {
                    let shift = 2 * (n - 1 - q);
                    let bits = (idx >> shift) & 0b11;
                    let p = match bits {
                        0 => Pauli1::I,
                        1 => Pauli1::X,
                        2 => Pauli1::Y,
                        3 => Pauli1::Z,
                        _ => unreachable!(),
                    };
                    qs.push(p);
                }
                PauliString(qs)
            })
            .collect()
    }
}

impl FromStr for PauliString {
    type Err = ParsePauliStringError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut paulis = Vec::with_capacity(s.len());
        for c in s.chars() {
            let Some(pauli) = Pauli1::from_char(c) else {
                return Err(ParsePauliStringError { invalid_char: c });
            };
            paulis.push(pauli);
        }
        Ok(PauliString(paulis))
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for p in &self.0 {
            write!(f, "{}", p.to_char())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_string() {
        let s = PauliString::from_label("XYZ").unwrap();
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.weight(), 3);
        assert_eq!(format!("{}", s), "XYZ");
    }

    #[test]
    fn identity_weight() {
        let s = PauliString::from_label("III").unwrap();
        assert_eq!(s.weight(), 0);
    }

    #[test]
    fn mixed_weight() {
        let s = PauliString::from_label("IXI").unwrap();
        assert_eq!(s.weight(), 1);
    }

    #[test]
    fn symplectic_product_2q() {
        let ix = PauliString::from_label("IX").unwrap();
        let iz = PauliString::from_label("IZ").unwrap();
        let zx = PauliString::from_label("ZX").unwrap();
        assert_eq!(ix.symplectic_product(&iz), 1); // X,Z anticommute on right
        assert_eq!(ix.symplectic_product(&ix), 0);
        assert_eq!(zx.symplectic_product(&iz), 1); // X,Z on right anticommute
        assert_eq!(zx.symplectic_product(&zx), 0);
    }

    #[test]
    fn enumerate_1q_gives_xyz() {
        let all = PauliString::enumerate_nonidentity(1);
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], PauliString::from_label("X").unwrap());
        assert_eq!(all[1], PauliString::from_label("Y").unwrap());
        assert_eq!(all[2], PauliString::from_label("Z").unwrap());
    }

    #[test]
    fn enumerate_2q_gives_15() {
        let all = PauliString::enumerate_nonidentity(2);
        assert_eq!(all.len(), 15);
        // First should be IX (idx 1 = 0b01 = 0|X).
        assert_eq!(all[0], PauliString::from_label("IX").unwrap());
        // Last should be ZZ (idx 15 = 0b1111 = Z|Z).
        assert_eq!(all[14], PauliString::from_label("ZZ").unwrap());
    }
}
