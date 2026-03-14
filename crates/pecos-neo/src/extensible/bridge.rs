//! Bridge between legacy `GateType` enum and extensible `GateId` system.
//!
//! This module provides conversion between the existing `GateType` enum
//! and the new extensible `GateId` system, enabling gradual migration.

use super::{GateId, gates};
use crate::command::GateType;

impl GateType {
    /// Convert a `GateType` to the corresponding `GateId`.
    ///
    /// All core gate types have a corresponding `GateId`.
    #[must_use]
    pub const fn to_gate_id(self) -> GateId {
        match self {
            // Single-qubit Paulis
            Self::I => gates::I,
            Self::X => gates::X,
            Self::Y => gates::Y,
            Self::Z => gates::Z,

            // Single-qubit Cliffords
            Self::H => gates::H,
            Self::SX => gates::SX,
            Self::SXdg => gates::SXdg,
            Self::SY => gates::SY,
            Self::SYdg => gates::SYdg,
            Self::SZ => gates::SZ,
            Self::SZdg => gates::SZdg,
            Self::T => gates::T,
            Self::Tdg => gates::Tdg,

            // Single-qubit rotations
            Self::RX => gates::RX,
            Self::RY => gates::RY,
            Self::RZ => gates::RZ,
            Self::U => gates::U,
            Self::R1XY => gates::R1XY,

            // Two-qubit gates
            Self::CX => gates::CX,
            Self::CY => gates::CY,
            Self::CZ => gates::CZ,
            Self::SZZ => gates::SZZ,
            Self::SZZdg => gates::SZZdg,
            Self::SWAP => gates::SWAP,
            Self::CRZ => gates::CRZ,
            Self::RXX => gates::RXX,
            Self::RYY => gates::RYY,
            Self::RZZ => gates::RZZ,

            // Three-qubit gates
            Self::CCX => gates::CCX,

            // Measurement and preparation
            Self::MZ => gates::MZ,
            Self::MeasureLeaked => gates::MEASURE_LEAKED,
            Self::MeasureFree => gates::MEASURE_FREE,
            Self::PZ => gates::PZ,
            Self::QAlloc => gates::QALLOC,
            Self::QFree => gates::QFREE,

            // Idle
            Self::Idle => gates::IDLE,
        }
    }
}

impl GateId {
    /// Try to convert a `GateId` to the corresponding `GateType`.
    ///
    /// Returns `None` for user-defined gates (ID >= 256) or
    /// core gates not represented in the `GateType` enum.
    #[must_use]
    pub const fn try_to_gate_type(self) -> Option<GateType> {
        // User-defined gates have no GateType equivalent
        if self.0 >= 256 {
            return None;
        }

        // Match against known gate IDs (must match constants in gate_id.rs)
        Some(match self.0 {
            // Single-qubit Paulis
            0 => GateType::I,
            1 => GateType::X,
            2 => GateType::Y,
            3 => GateType::Z,

            // Single-qubit Cliffords
            10 => GateType::H,
            11 => GateType::SX,
            12 => GateType::SXdg,
            13 => GateType::SY,
            14 => GateType::SYdg,
            15 => GateType::SZ,
            16 => GateType::SZdg,

            // T gates
            20 => GateType::T,
            21 => GateType::Tdg,

            // Single-qubit rotations
            30 => GateType::RX,
            31 => GateType::RY,
            32 => GateType::RZ,
            33 => GateType::U,
            34 => GateType::R1XY,

            // Two-qubit gates
            50 => GateType::CX,
            51 => GateType::CY,
            52 => GateType::CZ,
            53 => GateType::SWAP,

            // Two-qubit Clifford rotations
            64 => GateType::SZZ,
            65 => GateType::SZZdg,

            // Two-qubit parameterized gates
            70 => GateType::CRZ,
            71 => GateType::RXX,
            72 => GateType::RYY,
            73 => GateType::RZZ,

            // Three-qubit gates
            80 => GateType::CCX,

            // Measurement
            100 => GateType::MZ,
            101 => GateType::MeasureLeaked,
            102 => GateType::MeasureFree,

            // State preparation
            110 => GateType::PZ,

            // Qubit management
            120 => GateType::QAlloc,
            121 => GateType::QFree,

            // Idle
            130 => GateType::Idle,

            // Unknown core gate ID
            _ => return None,
        })
    }
}

impl From<GateType> for GateId {
    fn from(gate_type: GateType) -> Self {
        gate_type.to_gate_id()
    }
}

impl TryFrom<GateId> for GateType {
    type Error = GateIdConversionError;

    fn try_from(gate_id: GateId) -> Result<Self, Self::Error> {
        gate_id
            .try_to_gate_type()
            .ok_or(GateIdConversionError { gate_id })
    }
}

/// Error when converting a `GateId` to `GateType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateIdConversionError {
    /// The `GateId` that could not be converted.
    pub gate_id: GateId,
}

impl std::fmt::Display for GateIdConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.gate_id.is_user_defined() {
            write!(
                f,
                "User-defined gate ID {} has no `GateType` equivalent",
                self.gate_id.0
            )
        } else {
            write!(
                f,
                "Core gate ID {} is not represented in `GateType` enum",
                self.gate_id.0
            )
        }
    }
}

impl std::error::Error for GateIdConversionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_type_to_gate_id_roundtrip() {
        // Test all GateType variants can convert to GateId and back
        let gate_types = [
            GateType::I,
            GateType::X,
            GateType::Y,
            GateType::Z,
            GateType::H,
            GateType::SX,
            GateType::SXdg,
            GateType::SY,
            GateType::SYdg,
            GateType::SZ,
            GateType::SZdg,
            GateType::T,
            GateType::Tdg,
            GateType::RX,
            GateType::RY,
            GateType::RZ,
            GateType::U,
            GateType::R1XY,
            GateType::CX,
            GateType::CY,
            GateType::CZ,
            GateType::SZZ,
            GateType::SZZdg,
            GateType::SWAP,
            GateType::CRZ,
            GateType::RXX,
            GateType::RYY,
            GateType::RZZ,
            GateType::CCX,
            GateType::MZ,
            GateType::MeasureLeaked,
            GateType::MeasureFree,
            GateType::PZ,
            GateType::QAlloc,
            GateType::QFree,
            GateType::Idle,
        ];

        for gate_type in gate_types {
            let gate_id = gate_type.to_gate_id();
            let roundtrip = gate_id.try_to_gate_type();
            assert_eq!(
                roundtrip,
                Some(gate_type),
                "Roundtrip failed for {gate_type:?}"
            );
        }
    }

    #[test]
    fn test_gate_id_core_check() {
        // All GateType conversions should produce core gate IDs
        assert!(GateType::X.to_gate_id().is_core());
        assert!(GateType::CX.to_gate_id().is_core());
        assert!(GateType::CCX.to_gate_id().is_core());
        assert!(GateType::MZ.to_gate_id().is_core());
    }

    #[test]
    fn test_user_defined_gate_conversion() {
        let user_gate = GateId(256);
        assert!(user_gate.is_user_defined());
        assert!(user_gate.try_to_gate_type().is_none());

        let result: Result<GateType, _> = user_gate.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn test_from_impl() {
        let gate_id: GateId = GateType::H.into();
        assert_eq!(gate_id, gates::H);
    }

    #[test]
    fn test_try_from_impl() {
        let gate_type: Result<GateType, _> = gates::H.try_into();
        assert_eq!(gate_type.unwrap(), GateType::H);

        let user_gate: Result<GateType, _> = GateId(300).try_into();
        assert!(user_gate.is_err());
    }

    #[test]
    fn test_specific_gate_mappings() {
        // Verify specific mappings are correct
        assert_eq!(GateType::I.to_gate_id(), gates::I);
        assert_eq!(GateType::X.to_gate_id(), gates::X);
        assert_eq!(GateType::H.to_gate_id(), gates::H);
        assert_eq!(GateType::RZ.to_gate_id(), gates::RZ);
        assert_eq!(GateType::CX.to_gate_id(), gates::CX);
        assert_eq!(GateType::CCX.to_gate_id(), gates::CCX);
        assert_eq!(GateType::MZ.to_gate_id(), gates::MZ);
        assert_eq!(GateType::PZ.to_gate_id(), gates::PZ);
        assert_eq!(GateType::Idle.to_gate_id(), gates::IDLE);
    }
}
