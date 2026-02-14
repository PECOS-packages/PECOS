//! Compact gate type identifier.

/// Compact gate type identifier.
///
/// Core gates have reserved IDs in range 0-255.
/// User-defined gates are assigned IDs >= 256.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct GateId(pub u16);

impl GateId {
    /// Check if this is a core (built-in) gate.
    #[inline]
    #[must_use]
    pub const fn is_core(self) -> bool {
        self.0 < 256
    }

    /// Check if this is a user-defined gate.
    #[inline]
    #[must_use]
    pub const fn is_user_defined(self) -> bool {
        self.0 >= 256
    }

    /// Get the raw ID value.
    #[inline]
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

impl From<GateId> for usize {
    #[inline]
    fn from(id: GateId) -> usize {
        id.0 as usize
    }
}

impl From<u16> for GateId {
    #[inline]
    fn from(id: u16) -> GateId {
        GateId(id)
    }
}

/// Core gate ID constants.
///
/// These match the discriminant values in the existing `GateType` enum
/// to ensure compatibility during migration.
#[allow(non_upper_case_globals)]
pub mod gates {
    use super::GateId;

    // Single-qubit Paulis
    pub const I: GateId = GateId(0);
    pub const X: GateId = GateId(1);
    pub const Y: GateId = GateId(2);
    pub const Z: GateId = GateId(3);

    // Single-qubit Cliffords
    pub const H: GateId = GateId(10);
    pub const SX: GateId = GateId(11);
    pub const SXdg: GateId = GateId(12);
    pub const SY: GateId = GateId(13);
    pub const SYdg: GateId = GateId(14);
    pub const SZ: GateId = GateId(15);
    pub const SZdg: GateId = GateId(16);

    // T gates
    pub const T: GateId = GateId(20);
    pub const Tdg: GateId = GateId(21);

    // Single-qubit rotations
    pub const RX: GateId = GateId(30);
    pub const RY: GateId = GateId(31);
    pub const RZ: GateId = GateId(32);
    pub const U: GateId = GateId(33);
    pub const R1XY: GateId = GateId(34);

    // Two-qubit gates
    pub const CX: GateId = GateId(50);
    pub const CY: GateId = GateId(51);
    pub const CZ: GateId = GateId(52);
    pub const SWAP: GateId = GateId(53);
    pub const ISWAP: GateId = GateId(54);

    // Two-qubit Clifford rotations
    pub const SXX: GateId = GateId(60);
    pub const SXXdg: GateId = GateId(61);
    pub const SYY: GateId = GateId(62);
    pub const SYYdg: GateId = GateId(63);
    pub const SZZ: GateId = GateId(64);
    pub const SZZdg: GateId = GateId(65);

    // Two-qubit parameterized gates
    pub const CRZ: GateId = GateId(70);
    pub const RXX: GateId = GateId(71);
    pub const RYY: GateId = GateId(72);
    pub const RZZ: GateId = GateId(73);

    // Three-qubit gates
    pub const CCX: GateId = GateId(80);
    pub const CCZ: GateId = GateId(81);
    pub const CSWAP: GateId = GateId(82);

    // Measurement
    pub const MZ: GateId = GateId(100);
    pub const MEASURE_LEAKED: GateId = GateId(101);
    pub const MEASURE_FREE: GateId = GateId(102);

    // State preparation
    pub const PZ: GateId = GateId(110);
    pub const PX: GateId = GateId(111);
    pub const PY: GateId = GateId(112);

    // Qubit management
    pub const QALLOC: GateId = GateId(120);
    pub const QFREE: GateId = GateId(121);

    // Idle
    pub const IDLE: GateId = GateId(130);

    // User-defined gate ID range
    /// First ID available for user-defined gates.
    pub const USER_GATE_START: u16 = 256;
}
