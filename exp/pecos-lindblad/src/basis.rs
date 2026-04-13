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

/// Single-qubit Pauli operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
}

/// Multi-qubit Pauli string. Index 0 = leftmost factor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PauliString(pub Vec<Pauli1>);

impl PauliString {
    pub fn single(p: Pauli1) -> Self {
        PauliString(vec![p])
    }

    pub fn from_str(s: &str) -> Option<Self> {
        s.chars().map(Pauli1::from_char).collect::<Option<Vec<_>>>().map(PauliString)
    }

    pub fn num_qubits(&self) -> usize {
        self.0.len()
    }

    /// Weight (number of non-identity factors).
    pub fn weight(&self) -> usize {
        self.0.iter().filter(|&&p| p != Pauli1::I).count()
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
        let s = PauliString::from_str("XYZ").unwrap();
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.weight(), 3);
        assert_eq!(format!("{}", s), "XYZ");
    }

    #[test]
    fn identity_weight() {
        let s = PauliString::from_str("III").unwrap();
        assert_eq!(s.weight(), 0);
    }

    #[test]
    fn mixed_weight() {
        let s = PauliString::from_str("IXI").unwrap();
        assert_eq!(s.weight(), 1);
    }
}
