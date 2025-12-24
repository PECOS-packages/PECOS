//! Gate type enumeration for quantum operations
//!
//! This module provides the `GateType` enum which represents the different
//! types of quantum gates supported by the byte protocol.

use std::fmt;

/// FFI-friendly representation of quantum gate types
///
/// This enum is designed to be FFI-friendly with a C-compatible memory layout.
/// It represents the same gate types as the core `GateType` enum but with a more
/// predictable memory layout.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateType {
    I = 0b00,
    X = 0b01,
    Z = 0b10,
    Y = 0b11,
    // SX = 4,
    // SXdg = 5,
    // SY = 6
    // SYdg = 7
    SZ = 8,
    SZdg = 9,
    H = 10,
    // H2 = 11
    // H3 = 12
    // H4 = 13
    // H5 = 14
    // H6 = 15
    // F = 16
    // Fdg = 17
    // F2 = 18
    // F2dg = 19
    // F3 = 20
    // F3dg = 21
    // F4 = 22
    // F4dg = 23
    RX = 30,
    RY = 31,
    RZ = 32,
    T = 33,
    Tdg = 34,
    // Other T-like gates?
    U = 35,
    R1XY = 36,

    CX = 50,
    // CY = 51
    // CZ = 52
    // SXX = 53
    // SXXdg = 54
    // SYY = 55
    // SYYdg = 56
    SZZ = 57,
    SZZdg = 58,
    // SWAP = 59
    // iSWAP = 60
    // G = 61

    // RXX = 80
    // RYY = 81
    RZZ = 82,
    // RXXYYZZ

    // MX = 100
    // MnX = 101
    // MY = 102
    // MnY = 103
    // MZ = 104
    Measure = 104,
    // MnZ = 105
    MeasureLeaked = 105,
    /// Measure and free the qubit (destructive measurement)
    MeasureFree = 106,
    // TODO: MPauli instead of the other variants?

    // PX = 130
    // PnX = 131
    // PY = 132
    // PnY = 133
    // PZ = 134
    Prep = 134,
    // PnZ
    /// Allocate a qubit in the |0⟩ state
    QAlloc = 135,
    /// Free/deallocate a qubit
    QFree = 136,
    Idle = 200,
    MeasCrosstalkGlobalPayload = 218,
    MeasCrosstalkLocalPayload = 219,
}

impl From<u8> for GateType {
    fn from(value: u8) -> Self {
        match value {
            0 => GateType::I,
            1 => GateType::X,
            2 => GateType::Z,
            3 => GateType::Y,
            8 => GateType::SZ,
            9 => GateType::SZdg,
            10 => GateType::H,
            30 => GateType::RX,
            31 => GateType::RY,
            32 => GateType::RZ,
            33 => GateType::T,
            34 => GateType::Tdg,
            35 => GateType::U,
            36 => GateType::R1XY,
            50 => GateType::CX,
            57 => GateType::SZZ,
            58 => GateType::SZZdg,
            82 => GateType::RZZ,
            104 => GateType::Measure,
            105 => GateType::MeasureLeaked,
            106 => GateType::MeasureFree,
            134 => GateType::Prep,
            135 => GateType::QAlloc,
            136 => GateType::QFree,
            200 => GateType::Idle,
            218 => GateType::MeasCrosstalkGlobalPayload,
            219 => GateType::MeasCrosstalkLocalPayload,
            _ => panic!("Invalid gate type ID: {value}"),
        }
    }
}

impl GateType {
    /// Returns the number of angle parameters this gate type requires
    ///
    /// # Returns
    ///
    /// The number of floating-point angle parameters needed for this gate type
    #[must_use]
    pub const fn classical_arity(self) -> usize {
        match self {
            // Gates with no parameters
            GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::SZ
            | GateType::SZdg
            | GateType::H
            | GateType::T
            | GateType::Tdg
            | GateType::CX
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::Measure
            | GateType::MeasureLeaked
            | GateType::MeasureFree
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::Prep
            | GateType::QAlloc
            | GateType::QFree => 0,

            // Gates with one parameter
            GateType::RX | GateType::RY | GateType::RZ | GateType::RZZ | GateType::Idle => 1,

            // Gates with two parameters
            GateType::R1XY => 2,

            // Gates with three parameters
            GateType::U => 3,
        }
    }

    /// Returns the number of qubits this gate type operates on
    ///
    /// # Returns
    ///
    /// The number of qubits this gate type requires. All current gate types
    /// have a fixed number of qubits (1 or 2).
    #[must_use]
    pub const fn quantum_arity(self) -> usize {
        match self {
            // Single-qubit gates
            GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::SZ
            | GateType::SZdg
            | GateType::H
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::T
            | GateType::Tdg
            | GateType::R1XY
            | GateType::U
            | GateType::Measure
            | GateType::MeasureLeaked
            | GateType::MeasureFree
            | GateType::Prep
            | GateType::QAlloc
            | GateType::QFree
            | GateType::Idle
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload => 1,

            // Two-qubit gates
            GateType::CX | GateType::SZZ | GateType::SZZdg | GateType::RZZ => 2,
        }
    }

    /// Returns the number of angle parameters this gate type requires.
    ///
    /// This is separate from `classical_arity()` which includes all classical parameters.
    /// For example, `Idle` has `classical_arity() = 1` (duration) but `angle_arity() = 0`.
    #[must_use]
    pub const fn angle_arity(self) -> usize {
        match self {
            // Rotation gates with angle parameters
            GateType::RX | GateType::RY | GateType::RZ | GateType::RZZ => 1,
            GateType::R1XY => 2,
            GateType::U => 3,
            // All other gates have no angle parameters
            _ => 0,
        }
    }

    /// Returns whether this gate type requires angle parameters
    #[must_use]
    pub const fn is_parameterized(self) -> bool {
        self.classical_arity() > 0
    }

    /// Returns whether this gate type operates on a single qubit
    #[must_use]
    pub const fn is_single_qubit(self) -> bool {
        self.quantum_arity() == 1
    }

    /// Returns whether this gate type operates on two qubits
    #[must_use]
    pub const fn is_two_qubit(self) -> bool {
        self.quantum_arity() == 2
    }

    /// Returns whether this gate is a crosstalk payload gate
    #[must_use]
    pub const fn is_crosstalk_payload(self) -> bool {
        matches!(
            self,
            GateType::MeasCrosstalkGlobalPayload | GateType::MeasCrosstalkLocalPayload
        )
    }
}

impl fmt::Display for GateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateType::I => write!(f, "I"),
            GateType::X => write!(f, "X"),
            GateType::Y => write!(f, "Y"),
            GateType::Z => write!(f, "Z"),
            GateType::SZ => write!(f, "SZ"),
            GateType::SZdg => write!(f, "SZdg"),
            GateType::H => write!(f, "H"),
            GateType::RX => write!(f, "RX"),
            GateType::RY => write!(f, "RY"),
            GateType::RZ => write!(f, "RZ"),
            GateType::T => write!(f, "T"),
            GateType::Tdg => write!(f, "Tdg"),
            GateType::U => write!(f, "U"),
            GateType::R1XY => write!(f, "R1XY"),
            GateType::CX => write!(f, "CX"),
            GateType::SZZ => write!(f, "SZZ"),
            GateType::SZZdg => write!(f, "SZZdg"),
            GateType::RZZ => write!(f, "RZZ"),
            GateType::Measure => write!(f, "Measure"),
            GateType::MeasureLeaked => write!(f, "MeasureLeaked"),
            GateType::MeasureFree => write!(f, "MeasureFree"),
            GateType::Prep => write!(f, "Prep"),
            GateType::QAlloc => write!(f, "QAlloc"),
            GateType::QFree => write!(f, "QFree"),
            GateType::Idle => write!(f, "Idle"),
            GateType::MeasCrosstalkGlobalPayload => write!(f, "MeasCrosstalkGlobalPayload"),
            GateType::MeasCrosstalkLocalPayload => write!(f, "MeasCrosstalkLocalPayload"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_type_id_conversion() {
        assert_eq!(GateType::I as u8, 0);
        assert_eq!(GateType::X as u8, 1);
        assert_eq!(GateType::Z as u8, 2);
        assert_eq!(GateType::Y as u8, 3);
        assert_eq!(GateType::H as u8, 10);
        assert_eq!(GateType::CX as u8, 50);
        assert_eq!(GateType::SZZ as u8, 57);
        assert_eq!(GateType::RZ as u8, 32);
        assert_eq!(GateType::R1XY as u8, 36);
        assert_eq!(GateType::Measure as u8, 104);
        assert_eq!(GateType::MeasureLeaked as u8, 105);
        assert_eq!(GateType::MeasureFree as u8, 106);
        assert_eq!(GateType::Prep as u8, 134);
        assert_eq!(GateType::QAlloc as u8, 135);
        assert_eq!(GateType::QFree as u8, 136);
        assert_eq!(GateType::Idle as u8, 200);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload as u8, 218);
        assert_eq!(GateType::MeasCrosstalkLocalPayload as u8, 219);

        assert_eq!(GateType::from(0u8), GateType::I);
        assert_eq!(GateType::from(1u8), GateType::X);
        assert_eq!(GateType::from(2u8), GateType::Z);
        assert_eq!(GateType::from(3u8), GateType::Y);
        assert_eq!(GateType::from(10u8), GateType::H);
        assert_eq!(GateType::from(50u8), GateType::CX);
        assert_eq!(GateType::from(57u8), GateType::SZZ);
        assert_eq!(GateType::from(32u8), GateType::RZ);
        assert_eq!(GateType::from(36u8), GateType::R1XY);
        assert_eq!(GateType::from(104u8), GateType::Measure);
        assert_eq!(GateType::from(105u8), GateType::MeasureLeaked);
        assert_eq!(GateType::from(106u8), GateType::MeasureFree);
        assert_eq!(GateType::from(134u8), GateType::Prep);
        assert_eq!(GateType::from(135u8), GateType::QAlloc);
        assert_eq!(GateType::from(136u8), GateType::QFree);
        assert_eq!(GateType::from(200u8), GateType::Idle);
        assert_eq!(GateType::from(218u8), GateType::MeasCrosstalkGlobalPayload);
        assert_eq!(GateType::from(219u8), GateType::MeasCrosstalkLocalPayload);
    }

    #[test]
    fn test_classical_arity() {
        // Gates with no parameters
        assert_eq!(GateType::I.classical_arity(), 0);
        assert_eq!(GateType::X.classical_arity(), 0);
        assert_eq!(GateType::Y.classical_arity(), 0);
        assert_eq!(GateType::Z.classical_arity(), 0);
        assert_eq!(GateType::H.classical_arity(), 0);
        assert_eq!(GateType::CX.classical_arity(), 0);
        assert_eq!(GateType::SZZ.classical_arity(), 0);
        assert_eq!(GateType::SZZdg.classical_arity(), 0);
        assert_eq!(GateType::Measure.classical_arity(), 0);
        assert_eq!(GateType::MeasureLeaked.classical_arity(), 0);
        assert_eq!(GateType::MeasureFree.classical_arity(), 0);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload.classical_arity(), 0);
        assert_eq!(GateType::MeasCrosstalkLocalPayload.classical_arity(), 0);
        assert_eq!(GateType::Prep.classical_arity(), 0);
        assert_eq!(GateType::QAlloc.classical_arity(), 0);
        assert_eq!(GateType::QFree.classical_arity(), 0);

        // Gates with one parameter
        assert_eq!(GateType::RZ.classical_arity(), 1);
        assert_eq!(GateType::RZZ.classical_arity(), 1);
        assert_eq!(GateType::Idle.classical_arity(), 1);

        // Gates with two parameters
        assert_eq!(GateType::R1XY.classical_arity(), 2);

        // Gates with three parameters
        assert_eq!(GateType::U.classical_arity(), 3);
    }

    #[test]
    fn test_quantum_arity() {
        // Single-qubit gates
        assert_eq!(GateType::I.quantum_arity(), 1);
        assert_eq!(GateType::X.quantum_arity(), 1);
        assert_eq!(GateType::Y.quantum_arity(), 1);
        assert_eq!(GateType::Z.quantum_arity(), 1);
        assert_eq!(GateType::H.quantum_arity(), 1);
        assert_eq!(GateType::RZ.quantum_arity(), 1);
        assert_eq!(GateType::R1XY.quantum_arity(), 1);
        assert_eq!(GateType::U.quantum_arity(), 1);
        assert_eq!(GateType::Measure.quantum_arity(), 1);
        assert_eq!(GateType::MeasureLeaked.quantum_arity(), 1);
        assert_eq!(GateType::MeasureFree.quantum_arity(), 1);
        assert_eq!(GateType::Prep.quantum_arity(), 1);
        assert_eq!(GateType::QAlloc.quantum_arity(), 1);
        assert_eq!(GateType::QFree.quantum_arity(), 1);
        assert_eq!(GateType::Idle.quantum_arity(), 1);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload.quantum_arity(), 1);
        assert_eq!(GateType::MeasCrosstalkLocalPayload.quantum_arity(), 1);

        // Two-qubit gates
        assert_eq!(GateType::CX.quantum_arity(), 2);
        assert_eq!(GateType::SZZ.quantum_arity(), 2);
        assert_eq!(GateType::SZZdg.quantum_arity(), 2);
        assert_eq!(GateType::RZZ.quantum_arity(), 2);
    }

    #[test]
    fn test_is_parameterized() {
        // Non-parameterized gates
        assert!(!GateType::I.is_parameterized());
        assert!(!GateType::X.is_parameterized());
        assert!(!GateType::Y.is_parameterized());
        assert!(!GateType::Z.is_parameterized());
        assert!(!GateType::H.is_parameterized());
        assert!(!GateType::CX.is_parameterized());
        assert!(!GateType::SZZ.is_parameterized());
        assert!(!GateType::SZZdg.is_parameterized());
        assert!(!GateType::Measure.is_parameterized());
        assert!(!GateType::MeasureLeaked.is_parameterized());
        assert!(!GateType::MeasureFree.is_parameterized());
        assert!(!GateType::MeasCrosstalkGlobalPayload.is_parameterized());
        assert!(!GateType::MeasCrosstalkLocalPayload.is_parameterized());
        assert!(!GateType::Prep.is_parameterized());
        assert!(!GateType::QAlloc.is_parameterized());
        assert!(!GateType::QFree.is_parameterized());

        // Parameterized gates
        assert!(GateType::RZ.is_parameterized());
        assert!(GateType::RZZ.is_parameterized());
        assert!(GateType::R1XY.is_parameterized());
        assert!(GateType::U.is_parameterized());
        assert!(GateType::Idle.is_parameterized());
    }

    #[test]
    fn test_is_single_qubit() {
        // Single-qubit gates
        assert!(GateType::I.is_single_qubit());
        assert!(GateType::X.is_single_qubit());
        assert!(GateType::Y.is_single_qubit());
        assert!(GateType::Z.is_single_qubit());
        assert!(GateType::H.is_single_qubit());
        assert!(GateType::RZ.is_single_qubit());
        assert!(GateType::R1XY.is_single_qubit());
        assert!(GateType::U.is_single_qubit());
        assert!(GateType::Measure.is_single_qubit());
        assert!(GateType::MeasureLeaked.is_single_qubit());
        assert!(GateType::MeasureFree.is_single_qubit());
        assert!(GateType::Prep.is_single_qubit());
        assert!(GateType::QAlloc.is_single_qubit());
        assert!(GateType::QFree.is_single_qubit());
        assert!(GateType::Idle.is_single_qubit());
        assert!(GateType::MeasCrosstalkGlobalPayload.is_single_qubit());
        assert!(GateType::MeasCrosstalkLocalPayload.is_single_qubit());

        // Two-qubit gates
        assert!(!GateType::CX.is_single_qubit());
        assert!(!GateType::SZZ.is_single_qubit());
        assert!(!GateType::SZZdg.is_single_qubit());
        assert!(!GateType::RZZ.is_single_qubit());
    }

    #[test]
    fn test_is_two_qubit() {
        // Single-qubit gates
        assert!(!GateType::I.is_two_qubit());
        assert!(!GateType::X.is_two_qubit());
        assert!(!GateType::Y.is_two_qubit());
        assert!(!GateType::Z.is_two_qubit());
        assert!(!GateType::H.is_two_qubit());
        assert!(!GateType::RZ.is_two_qubit());
        assert!(!GateType::R1XY.is_two_qubit());
        assert!(!GateType::U.is_two_qubit());
        assert!(!GateType::Measure.is_two_qubit());
        assert!(!GateType::MeasureLeaked.is_two_qubit());
        assert!(!GateType::MeasureFree.is_two_qubit());
        assert!(!GateType::Prep.is_two_qubit());
        assert!(!GateType::QAlloc.is_two_qubit());
        assert!(!GateType::QFree.is_two_qubit());
        assert!(!GateType::Idle.is_two_qubit());
        assert!(!GateType::MeasCrosstalkGlobalPayload.is_two_qubit());
        assert!(!GateType::MeasCrosstalkLocalPayload.is_two_qubit());

        // Two-qubit gates
        assert!(GateType::CX.is_two_qubit());
        assert!(GateType::SZZ.is_two_qubit());
        assert!(GateType::SZZdg.is_two_qubit());
        assert!(GateType::RZZ.is_two_qubit());
    }

    #[test]
    fn test_arity_usage_examples() {
        // Example usage of arity methods for validation
        let gate_type = GateType::RZZ;

        // Check parameter requirements
        assert_eq!(
            gate_type.classical_arity(),
            1,
            "RZZ requires 1 angle parameter"
        );
        assert!(gate_type.is_parameterized(), "RZZ is parameterized");

        // Check qubit requirements
        assert_eq!(gate_type.quantum_arity(), 2, "RZZ operates on 2 qubits");
        assert!(gate_type.is_two_qubit(), "RZZ is a two-qubit gate");
        assert!(
            !gate_type.is_single_qubit(),
            "RZZ is not a single-qubit gate"
        );

        // Example of using arity for validation
        let params = [1.57]; // One angle parameter
        let qubits = [0, 1]; // Two qubits

        // Validate parameter count
        assert_eq!(params.len(), gate_type.classical_arity());

        // Validate qubit count
        assert_eq!(qubits.len(), gate_type.quantum_arity());
    }
}
