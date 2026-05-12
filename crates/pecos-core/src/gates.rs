//! Gate command representation for quantum operations
//!
//! This module provides the `GateCommand` struct which represents a quantum
//! gate operation with its type, qubits, and parameters.

use crate::Angle64;
use crate::ChannelExpr;
use crate::MeasId;
use crate::QubitId;
use crate::gate_type::GateType;
use crate::qubit_support::duplicate_qubits;
use smallvec::SmallVec;

/// Stack-allocated qubit buffer for gates (up to 4 qubits inline).
/// Most gates operate on 1-2 qubits, so this avoids heap allocation.
pub type GateQubits = SmallVec<[QubitId; 4]>;

/// Stack-allocated angle buffer for gates (up to 3 angles inline).
/// Most gates have 0-2 angle parameters.
pub type GateAngles = SmallVec<[Angle64; 3]>;

/// Stack-allocated parameter buffer for gates (up to 2 params inline).
/// Most gates have 0-1 non-angle parameters.
pub type GateParams = SmallVec<[f64; 2]>;

/// Measurement result identities for measurement gates.
/// Empty for non-measurement gates. One entry per qubit for MZ/MX/MY.
pub type GateMeasIds = SmallVec<[MeasId; 1]>;

/// Flat gate command representation for quantum operations
///
/// Clean, flat representation of quantum gate commands
/// without unnecessary nesting. Serves as the primary interface for gate
/// operations in the `ByteMessage` system.
///
/// # Design
/// - Uses `QubitId` for type-safe qubit representation
/// - Uses `Angle64` for rotation angles (in full turns)
/// - Flat structure for easy access to gate data
/// - Compatible with binary protocol serialization
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// The type of the gate
    pub gate_type: GateType,
    /// Rotation angles for parameterized gates (in full turns).
    /// Use `Angle64::from_turns()` or `Angle64::from_radians()` to create.
    /// Stack-allocated for up to 3 angles.
    pub angles: GateAngles,
    /// Other non-angle parameters (e.g., duration for Idle gate)
    /// Stack-allocated for up to 2 parameters.
    pub params: GateParams,
    /// The qubits the gate acts on.
    /// Stack-allocated for up to 4 qubits.
    pub qubits: GateQubits,
    /// Measurement result identities (one per qubit for measurement gates).
    ///
    /// Assigned at circuit construction time, carried through all
    /// transformations. Empty for non-measurement gates.
    /// Follows the MLIR SSA pattern: defined once, referenced everywhere.
    pub meas_ids: GateMeasIds,
    /// Typed channel payload for `GateType::Channel`.
    ///
    /// This is `None` for ideal circuit gates. It is populated only for
    /// annotated/noisy circuits that explicitly carry channel operations.
    pub channel: Option<ChannelExpr>,
}

/// Legacy quantum gate representation for `ByteMessageBuilder` compatibility
///
/// Designed to replace `QuantumCommand` with a more FFI-friendly
/// representation. Contains all the information needed to represent a quantum
/// gate operation.
///
impl Gate {
    /// Create a new gate command with angles and params
    #[must_use]
    pub fn new(
        gate_type: GateType,
        angles: impl Into<GateAngles>,
        params: impl Into<GateParams>,
        qubits: impl Into<GateQubits>,
    ) -> Self {
        Self {
            gate_type,
            angles: angles.into(),
            params: params.into(),
            qubits: qubits.into(),
            meas_ids: GateMeasIds::new(),
            channel: None,
        }
    }

    /// Create a new gate command with angles only (no other params)
    #[must_use]
    pub fn with_angles(
        gate_type: GateType,
        angles: impl Into<GateAngles>,
        qubits: impl Into<GateQubits>,
    ) -> Self {
        Self::new(gate_type, angles, GateParams::new(), qubits)
    }

    /// Create a new gate command with no angles or params
    #[must_use]
    pub fn simple(gate_type: GateType, qubits: impl Into<GateQubits>) -> Self {
        Self::new(gate_type, GateAngles::new(), GateParams::new(), qubits)
    }

    /// Total number of qubits being gated
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// The number of individual gates represented by this `Gate`
    #[inline]
    #[must_use]
    pub fn num_gates(&self) -> usize {
        self.gate_type.num_gates(self.num_qubits())
    }

    /// Returns true if `self` and `other` can be represented as one batched gate
    /// command by concatenating qubit and measurement-id payloads.
    ///
    /// Batch-compatible gates are identical except for disjoint qubit support
    /// and, for measurement gates, their corresponding measurement ids.
    #[must_use]
    pub fn can_batch_with(&self, other: &Self) -> bool {
        if self.gate_type != other.gate_type
            || self.angles != other.angles
            || self.params != other.params
            || self.channel != other.channel
        {
            return false;
        }

        if matches!(
            self.gate_type,
            GateType::Custom
                | GateType::Channel
                | GateType::TrackedPauliMeta
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::MeasCrosstalkLocalPayload
        ) {
            return false;
        }

        if self.qubits.iter().any(|q| other.qubits.contains(q)) {
            return false;
        }

        let self_has_meas_ids = !self.meas_ids.is_empty();
        let other_has_meas_ids = !other.meas_ids.is_empty();
        if self_has_meas_ids != other_has_meas_ids {
            return false;
        }
        if self_has_meas_ids
            && (self.meas_ids.len() != self.qubits.len()
                || other.meas_ids.len() != other.qubits.len())
        {
            return false;
        }

        true
    }

    /// Appends a compatible gate command into this batch.
    ///
    /// # Panics
    ///
    /// Panics if `other` is not batch-compatible with `self`.
    pub fn append_batch(&mut self, other: Self) {
        assert!(
            self.can_batch_with(&other),
            "cannot batch incompatible gate commands"
        );
        self.qubits.extend(other.qubits);
        self.meas_ids.extend(other.meas_ids);
    }

    /// Helper function to flatten qubit pairs into a `GateQubits` buffer
    fn flatten_qubit_pairs(
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> GateQubits {
        qubit_pairs
            .iter()
            .flat_map(|&(q1, q2)| [q1.into(), q2.into()])
            .collect()
    }

    /// Create a Custom gate on the given qubits
    #[must_use]
    pub fn custom(qubits: impl Into<GateQubits>) -> Self {
        Self::simple(GateType::Custom, qubits)
    }

    /// Create a typed channel operation for an annotated/noisy circuit.
    #[must_use]
    pub fn channel(channel: ChannelExpr) -> Self {
        let qubits = channel
            .qubits()
            .into_iter()
            .map(QubitId)
            .collect::<GateQubits>();
        Self {
            gate_type: GateType::Channel,
            angles: GateAngles::new(),
            params: GateParams::new(),
            qubits,
            meas_ids: GateMeasIds::new(),
            channel: Some(channel),
        }
    }

    /// Returns the typed channel payload when this is a channel operation.
    #[must_use]
    pub fn channel_expr(&self) -> Option<&ChannelExpr> {
        self.channel.as_ref()
    }

    /// Returns true when this gate carries a channel payload.
    #[must_use]
    pub fn is_channel(&self) -> bool {
        self.gate_type == GateType::Channel
    }

    /// Create Identity gate on multiple qubits
    #[must_use]
    pub fn i(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::I,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create X gate on multiple qubits
    #[must_use]
    pub fn x(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::X,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create Y gate on multiple qubits
    #[must_use]
    pub fn y(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::Y,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create Z gate on multiple qubits
    #[must_use]
    pub fn z(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::Z,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create H gate on multiple qubits
    #[must_use]
    pub fn h(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::H,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SX gate (sqrt-X) on multiple qubits
    #[must_use]
    pub fn sx(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::SX,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `SXdg` gate (sqrt-X dagger) on multiple qubits
    #[must_use]
    pub fn sxdg(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::SXdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SY gate (sqrt-Y) on multiple qubits
    #[must_use]
    pub fn sy(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::SY,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `SYdg` gate (sqrt-Y dagger) on multiple qubits
    #[must_use]
    pub fn sydg(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::SYdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SZ gate (sqrt-Z) on multiple qubits
    #[must_use]
    pub fn sz(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::SZ,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `SZdg` gate (sqrt-Z dagger) on multiple qubits
    #[must_use]
    pub fn szdg(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::SZdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create F gate on multiple qubits
    #[must_use]
    pub fn f(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::F,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create Fdg gate on multiple qubits
    #[must_use]
    pub fn fdg(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::Fdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create T gate on multiple qubits
    #[must_use]
    pub fn t(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::T,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create Tdg gate on multiple qubits
    #[must_use]
    pub fn tdg(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::Tdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create CX gate from flat qubit list (control1, target1, control2, target2, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `CX` gates require pairs of qubits.
    #[must_use]
    pub fn cx_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "CX gate requires an even number of qubits"
        );
        Self::simple(
            GateType::CX,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create CX gate on multiple qubit pairs
    #[must_use]
    pub fn cx(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::cx_vec(&flat_qubits)
    }

    /// Create CY gate from flat qubit list (control1, target1, control2, target2, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `CY` gates require pairs of qubits.
    #[must_use]
    pub fn cy_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "CY gate requires an even number of qubits"
        );
        Self::simple(
            GateType::CY,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create CY gate on multiple qubit pairs
    #[must_use]
    pub fn cy(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::cy_vec(&flat_qubits)
    }

    /// Create CZ gate from flat qubit list (control1, target1, control2, target2, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `CZ` gates require pairs of qubits.
    #[must_use]
    pub fn cz_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "CZ gate requires an even number of qubits"
        );
        Self::simple(
            GateType::CZ,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create CZ gate on multiple qubit pairs
    #[must_use]
    pub fn cz(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::cz_vec(&flat_qubits)
    }

    /// Create SZZ gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `SZZ` gates require pairs of qubits.
    #[must_use]
    pub fn szz_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SZZ gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SZZ,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SZZ gate on multiple qubit pairs
    #[must_use]
    pub fn szz(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::szz_vec(&flat_qubits)
    }

    /// Create `SZZdg` gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `SZZdg` gates require pairs of qubits.
    #[must_use]
    pub fn szzdg_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SZZdg gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SZZdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `SZZdg` gate on multiple qubit pairs
    #[must_use]
    pub fn szzdg(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::szzdg_vec(&flat_qubits)
    }

    /// Create SXX gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn sxx_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SXX gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SXX,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SXX gate on multiple qubit pairs
    #[must_use]
    pub fn sxx(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::sxx_vec(&flat_qubits)
    }

    /// Create `SXXdg` gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn sxxdg_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SXXdg gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SXXdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `SXXdg` gate on multiple qubit pairs
    #[must_use]
    pub fn sxxdg(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::sxxdg_vec(&flat_qubits)
    }

    /// Create SYY gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn syy_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SYY gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SYY,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SYY gate on multiple qubit pairs
    #[must_use]
    pub fn syy(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::syy_vec(&flat_qubits)
    }

    /// Create `SYYdg` gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn syydg_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SYYdg gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SYYdg,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `SYYdg` gate on multiple qubit pairs
    #[must_use]
    pub fn syydg(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::syydg_vec(&flat_qubits)
    }

    /// Create SWAP gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn swap_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "SWAP gate requires an even number of qubits"
        );
        Self::simple(
            GateType::SWAP,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create SWAP gate on multiple qubit pairs
    #[must_use]
    pub fn swap(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::swap_vec(&flat_qubits)
    }

    /// Create CH gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn ch_vec(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "CH gate requires an even number of qubits"
        );
        Self::simple(
            GateType::CH,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create CH gate on multiple qubit pairs
    #[must_use]
    pub fn ch(qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::ch_vec(&flat_qubits)
    }

    /// Create CRZ gate from flat qubit list
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn crz_vec(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "CRZ gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::CRZ,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create CRZ gate on multiple qubit pairs
    #[must_use]
    pub fn crz(
        theta: Angle64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::crz_vec(theta, &flat_qubits)
    }

    /// Create CCX (Toffoli) gate on qubit triples
    #[must_use]
    pub fn ccx(
        triples: &[(
            impl Into<QubitId> + Copy,
            impl Into<QubitId> + Copy,
            impl Into<QubitId> + Copy,
        )],
    ) -> Self {
        let qubits: GateQubits = triples
            .iter()
            .flat_map(|&(c1, c2, t)| [c1.into(), c2.into(), t.into()])
            .collect();
        Self::simple(GateType::CCX, qubits)
    }

    /// Create RXX gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `RXX` gates require pairs of qubits.
    #[must_use]
    pub fn rxx_vec(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "RXX gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::RXX,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create RXX gate on multiple qubit pairs
    #[must_use]
    pub fn rxx(
        theta: Angle64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::rxx_vec(theta, &flat_qubits)
    }

    /// Create RYY gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `RYY` gates require pairs of qubits.
    #[must_use]
    pub fn ryy_vec(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "RYY gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::RYY,
            vec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create RYY gate on multiple qubit pairs
    #[must_use]
    pub fn ryy(
        theta: Angle64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::ryy_vec(theta, &flat_qubits)
    }

    /// Create RZZ gate from flat qubit list (`qubit1_1`, `qubit2_1`, `qubit1_2`, `qubit2_2`, ...)
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even, as `RZZ` gates require pairs of qubits.
    #[must_use]
    pub fn rzz_vec(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "RZZ gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::RZZ,
            smallvec::smallvec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create RZZ gate on multiple qubit pairs
    #[must_use]
    pub fn rzz(
        theta: Angle64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::rzz_vec(theta, &flat_qubits)
    }

    /// Create RX gate on multiple qubits
    #[must_use]
    pub fn rx(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::RX,
            smallvec::smallvec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create RY gate on multiple qubits
    #[must_use]
    pub fn ry(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::RY,
            smallvec::smallvec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create RZ gate on multiple qubits
    #[must_use]
    pub fn rz(theta: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::RZ,
            smallvec::smallvec![theta],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create R1XY gate on multiple qubits
    #[must_use]
    pub fn r1xy(theta: Angle64, phi: Angle64, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::with_angles(
            GateType::R1XY,
            smallvec::smallvec![theta, phi],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create U gate on multiple qubits
    #[must_use]
    pub fn u(
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> Self {
        Self::with_angles(
            GateType::U,
            smallvec::smallvec![theta, phi, lambda],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create RXXRYYRZZ gate on multiple qubit pairs
    #[must_use]
    pub fn rxxryyrzz(
        alpha: Angle64,
        beta: Angle64,
        gamma: Angle64,
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::rxxryyrzz_vec(alpha, beta, gamma, &flat_qubits)
    }

    /// Create RXXRYYRZZ gate from a flat qubit slice
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn rxxryyrzz_vec(
        alpha: Angle64,
        beta: Angle64,
        gamma: Angle64,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "RXXRYYRZZ gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::RXXRYYRZZ,
            smallvec::smallvec![alpha, beta, gamma],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create U2q gate on multiple qubit pairs
    ///
    /// Angles are packed as: before[0](3) + before[1](3) + interaction(3) + after[0](3) + after[1](3)
    #[must_use]
    pub fn u2q(
        before: [[Angle64; 3]; 2],
        interaction: [Angle64; 3],
        after: [[Angle64; 3]; 2],
        qubit_pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        let flat_qubits = Self::flatten_qubit_pairs(qubit_pairs);
        Self::u2q_vec(before, interaction, after, &flat_qubits)
    }

    /// Create U2q gate from a flat qubit slice
    ///
    /// # Panics
    ///
    /// Panics if the number of qubits is not even.
    #[must_use]
    pub fn u2q_vec(
        before: [[Angle64; 3]; 2],
        interaction: [Angle64; 3],
        after: [[Angle64; 3]; 2],
        qubits: &[impl Into<QubitId> + Copy],
    ) -> Self {
        assert!(
            qubits.len().is_multiple_of(2),
            "U2q gate requires an even number of qubits"
        );
        Self::with_angles(
            GateType::U2q,
            smallvec::smallvec![
                before[0][0],
                before[0][1],
                before[0][2],
                before[1][0],
                before[1][1],
                before[1][2],
                interaction[0],
                interaction[1],
                interaction[2],
                after[0][0],
                after[0][1],
                after[0][2],
                after[1][0],
                after[1][1],
                after[1][2],
            ],
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create Measure gate on multiple qubits
    #[must_use]
    pub fn mz(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MZ,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `MeasureLeaked` gate on multiple qubits
    #[must_use]
    pub fn measure_leaked(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasureLeaked,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create PZ (prep) gate on multiple qubits
    #[must_use]
    pub fn pz(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::PZ,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `QAlloc` gate to allocate qubits in the |0⟩ state
    #[must_use]
    pub fn qalloc(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::QAlloc,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `QFree` gate to deallocate qubits
    #[must_use]
    pub fn qfree(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::QFree,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create `MeasureFree` gate (measure and deallocate) on multiple qubits
    #[must_use]
    pub fn mz_free(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasureFree,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create a new Idle gate for qubits idling for a specific duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the idle period in seconds
    /// * `qubits` - The qubits that are idling
    ///
    /// # Returns
    ///
    /// A new Idle gate with the specified parameters
    #[must_use]
    pub fn idle(duration: f64, qubits: impl Into<GateQubits>) -> Self {
        Self::new(
            GateType::Idle,
            GateAngles::new(),
            smallvec::smallvec![duration],
            qubits,
        )
    }

    /// Returns the duration of an idle gate, or 0.0 if not an idle gate
    #[must_use]
    pub fn idle_duration(&self) -> f64 {
        if self.gate_type == GateType::Idle && !self.params.is_empty() {
            self.params[0]
        } else {
            0.0
        }
    }

    /// Create a new `MeasCrosstalkGlobalPayload` with the data from runtime.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The qubits that are guaranteed *not* to be affected by the
    ///   global crosstalk event.
    ///
    /// NOTE: it seems unintuitive to give the complement of the list of victim qubits.
    /// It fits better with the previous version of crosstalk, but we might want to
    /// refactor this.
    ///
    /// # Returns
    ///
    /// A new `MeasCrosstalkGlobalPayload` gate with the specified parameters
    #[must_use]
    pub fn meas_crosstalk_global_payload(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasCrosstalkGlobalPayload,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Create a new `MeasCrosstalkLocalPayload` with the data from runtime.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The qubits that are potential victims of the local crosstalk event.
    ///
    /// # Returns
    ///
    /// A new `MeasCrosstalkLocalPayload` gate with the specified parameters
    #[must_use]
    pub fn meas_crosstalk_local_payload(qubits: &[impl Into<QubitId> + Copy]) -> Self {
        Self::simple(
            GateType::MeasCrosstalkLocalPayload,
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        )
    }

    /// Returns the number of angle parameters this gate requires
    ///
    /// # Returns
    ///
    /// The number of floating-point angle parameters needed for this gate type
    #[inline]
    #[must_use]
    pub fn classical_arity(&self) -> usize {
        if self.is_channel() {
            return 0;
        }
        self.gate_type.classical_arity()
    }

    /// Returns the number of qubits this gate operates on
    ///
    /// # Returns
    ///
    /// The number of qubits this gate type requires (1 or 2)
    #[inline]
    #[must_use]
    pub fn quantum_arity(&self) -> usize {
        if self.is_channel() {
            return self.qubits.len().max(1);
        }
        self.gate_type.quantum_arity()
    }

    /// Returns whether this gate requires angle parameters
    #[inline]
    #[must_use]
    pub fn is_parameterized(&self) -> bool {
        if self.is_channel() {
            return false;
        }
        self.gate_type.is_parameterized()
    }

    /// Returns whether this gate operates on a single qubit
    #[inline]
    #[must_use]
    pub fn is_single_qubit(&self) -> bool {
        if self.is_channel() {
            return self.qubits.len() == 1;
        }
        self.gate_type.is_single_qubit()
    }

    /// Returns whether this gate operates on two qubits
    #[inline]
    #[must_use]
    pub fn is_two_qubit(&self) -> bool {
        if self.is_channel() {
            return self.qubits.len() == 2;
        }
        self.gate_type.is_two_qubit()
    }

    /// Returns the number of angle parameters this gate requires
    #[inline]
    #[must_use]
    pub fn angle_arity(&self) -> usize {
        if self.is_channel() {
            return 0;
        }
        self.gate_type.angle_arity()
    }

    /// Validates that this gate has the correct number of parameters and qubits
    ///
    /// # Returns
    ///
    /// `Ok(())` if the gate is valid, or an error message describing the issue
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of angles doesn't match the gate's angle arity
    /// - The number of qubits is not a multiple of the gate's quantum arity
    /// - Any qubit is repeated within the gate command
    pub fn validate(&self) -> Result<(), String> {
        if self.is_channel() {
            let Some(channel) = &self.channel else {
                return Err("GateType::Channel requires a channel payload".to_string());
            };
            if !self.angles.is_empty() || !self.params.is_empty() || !self.meas_ids.is_empty() {
                return Err(
                    "Channel gates cannot carry angle, parameter, or measurement-id payloads"
                        .to_string(),
                );
            }
            let expected = channel
                .qubits()
                .into_iter()
                .map(QubitId)
                .collect::<GateQubits>();
            if self.qubits != expected {
                return Err(format!(
                    "Channel gate qubits {:?} do not match channel payload qubits {:?}",
                    self.qubits, expected
                ));
            }
            return Ok(());
        }
        if self.channel.is_some() {
            return Err("Only GateType::Channel can carry a channel payload".to_string());
        }
        if self.gate_type == GateType::Custom {
            let duplicates = duplicate_qubits(self.qubits.iter().map(|q| q.0));
            if !duplicates.is_empty() {
                return Err(format!(
                    "Gate {:?} requires distinct qubits within one gate command; duplicated qubits: {:?}",
                    self.gate_type, duplicates
                ));
            }
            if !self.meas_ids.is_empty() {
                return Err("Custom gates cannot carry measurement-id payloads".to_string());
            }
            return Ok(());
        }
        // Check angle parameters
        if self.angles.len() != self.angle_arity() {
            return Err(format!(
                "Gate {:?} expected {} angle parameters, got {}",
                self.gate_type,
                self.angle_arity(),
                self.angles.len()
            ));
        }
        // Check qubit count
        if !self.qubits.len().is_multiple_of(self.quantum_arity()) {
            return Err(format!(
                "Gate {:?} requires a multiple of {} qubits, got {}",
                self.gate_type,
                self.quantum_arity(),
                self.qubits.len()
            ));
        }
        let expected_params = self.classical_arity() - self.angle_arity();
        if self.params.len() != expected_params {
            return Err(format!(
                "Gate {:?} expected {} non-angle parameters, got {}",
                self.gate_type,
                expected_params,
                self.params.len()
            ));
        }
        let duplicates = duplicate_qubits(self.qubits.iter().map(|q| q.0));
        if !duplicates.is_empty() {
            return Err(format!(
                "Gate {:?} requires distinct qubits within one gate command; duplicated qubits: {:?}",
                self.gate_type, duplicates
            ));
        }
        let is_measurement = matches!(
            self.gate_type,
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree
        );
        if is_measurement {
            if !self.meas_ids.is_empty() && self.meas_ids.len() != self.qubits.len() {
                return Err(format!(
                    "Measurement gate {:?} expected measurement-id count to be 0 or {}, got {}",
                    self.gate_type,
                    self.qubits.len(),
                    self.meas_ids.len()
                ));
            }
        } else if !self.meas_ids.is_empty() {
            return Err(format!(
                "Gate {:?} cannot carry measurement-id payloads",
                self.gate_type
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_command_creation() {
        // Single qubit gates
        let x_gate = Gate::x(&[0, 1, 2]);
        assert_eq!(x_gate.gate_type, GateType::X);
        assert_eq!(
            x_gate.qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1), QubitId::from(2)]
        );
        assert!(x_gate.angles.is_empty());

        // Parameterized single qubit gates
        let rz_gate = Gate::rz(Angle64::from_turns(0.5), &[1, 2]);
        assert_eq!(rz_gate.gate_type, GateType::RZ);
        assert_eq!(
            rz_gate.qubits.as_slice(),
            &[QubitId::from(1), QubitId::from(2)]
        );
        assert_eq!(rz_gate.angles.as_slice(), &[Angle64::from_turns(0.5)]);

        // Two qubit gates
        let cx_gate = Gate::cx(&[(0, 1), (2, 3)]);
        assert_eq!(cx_gate.gate_type, GateType::CX);
        assert_eq!(
            cx_gate.qubits.as_slice(),
            &[
                QubitId::from(0),
                QubitId::from(1),
                QubitId::from(2),
                QubitId::from(3)
            ]
        );
        assert!(cx_gate.angles.is_empty());

        // Measure gates
        let measure_gate = Gate::mz(&[2, 3]);
        assert_eq!(measure_gate.gate_type, GateType::MZ);
        assert_eq!(
            measure_gate.qubits.as_slice(),
            &[QubitId::from(2), QubitId::from(3)]
        );
        assert!(measure_gate.angles.is_empty());
    }

    #[test]
    fn test_channel_gate_creation_and_validation() {
        use crate::channel::{Dephasing, Depolarizing};

        let gate = Gate::channel(Depolarizing(0.25, 0));
        assert_eq!(gate.gate_type, GateType::Channel);
        assert_eq!(gate.qubits.as_slice(), &[QubitId::from(0)]);
        assert!(gate.channel_expr().is_some());
        assert!(gate.validate().is_ok());

        let two_qubit_channel = Depolarizing(0.1, 0) & Dephasing(0.2, 1);
        let two_qubit_gate = Gate::channel(two_qubit_channel);
        assert_eq!(
            two_qubit_gate.qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(two_qubit_gate.quantum_arity(), 2);
        assert_eq!(two_qubit_gate.num_gates(), 1);
        assert!(two_qubit_gate.is_two_qubit());
        assert!(two_qubit_gate.validate().is_ok());
    }

    #[test]
    fn test_num_gates_counts_batched_gates() {
        assert_eq!(Gate::h(&[0, 1, 2, 3]).num_gates(), 4);
        assert_eq!(Gate::cx(&[(0, 1), (2, 3)]).num_gates(), 2);
        assert_eq!(Gate::ccx(&[(0, 1, 2), (3, 4, 5)]).num_gates(), 2);

        assert_eq!(
            Gate::custom(vec![QubitId::from(0), QubitId::from(1)]).num_gates(),
            1
        );
        assert_eq!(
            Gate::simple(
                GateType::TrackedPauliMeta,
                vec![QubitId::from(0), QubitId::from(1)]
            )
            .num_gates(),
            0
        );
        assert_eq!(Gate::meas_crosstalk_global_payload(&[0, 1]).num_gates(), 0);
    }

    #[test]
    fn test_gate_batch_compatibility_and_append() {
        let mut h0 = Gate::h(&[0]);
        let h1 = Gate::h(&[1]);
        assert!(h0.can_batch_with(&h1));
        h0.append_batch(h1);
        assert_eq!(h0.qubits.as_slice(), &[QubitId::from(0), QubitId::from(1)]);
        assert_eq!(h0.num_gates(), 2);

        assert!(!Gate::h(&[0]).can_batch_with(&Gate::h(&[0])));
        assert!(
            !Gate::rz(Angle64::from_turns(0.25), &[0])
                .can_batch_with(&Gate::rz(Angle64::from_turns(0.5), &[1]))
        );
        assert!(
            !Gate::custom(vec![QubitId::from(0)])
                .can_batch_with(&Gate::custom(vec![QubitId::from(1)]))
        );
    }

    #[test]
    fn test_measurement_batch_compatibility_preserves_measurement_ids() {
        let mut m0 = Gate::mz(&[0]);
        m0.meas_ids.push(MeasId(4));
        let mut m1 = Gate::mz(&[1]);
        m1.meas_ids.push(MeasId(5));

        assert!(m0.can_batch_with(&m1));
        m0.append_batch(m1);

        assert_eq!(m0.qubits.as_slice(), &[QubitId::from(0), QubitId::from(1)]);
        assert_eq!(m0.meas_ids.as_slice(), &[MeasId(4), MeasId(5)]);
    }

    #[test]
    fn test_two_qubit_gate_vec_variants() {
        // Test CX with _vec variant - much more convenient when you have a flat list
        let cx_pairs = Gate::cx(&[(0, 1), (2, 3)]);
        let cx_vec = Gate::cx_vec(&[0, 1, 2, 3]);
        assert_eq!(cx_pairs.gate_type, cx_vec.gate_type);
        assert_eq!(cx_pairs.qubits, cx_vec.qubits);
        assert_eq!(cx_pairs.angles, cx_vec.angles);

        // Test SZZ with _vec variant
        let szz_pairs = Gate::szz(&[(1, 2), (3, 4)]);
        let szz_vec = Gate::szz_vec(&[1, 2, 3, 4]);
        assert_eq!(szz_pairs.gate_type, szz_vec.gate_type);
        assert_eq!(szz_pairs.qubits, szz_vec.qubits);
        assert_eq!(szz_pairs.angles, szz_vec.angles);

        // Test SZZdg with _vec variant
        let szzdg_pairs = Gate::szzdg(&[(0, 2), (1, 3)]);
        let szzdg_vec = Gate::szzdg_vec(&[0, 2, 1, 3]);
        assert_eq!(szzdg_pairs.gate_type, szzdg_vec.gate_type);
        assert_eq!(szzdg_pairs.qubits, szzdg_vec.qubits);
        assert_eq!(szzdg_pairs.angles, szzdg_vec.angles);

        // Test RZZ with _vec variant
        let angle = Angle64::from_turns(0.25);
        let rzz_pairs = Gate::rzz(angle, &[(0, 1), (2, 3)]);
        let rzz_vec = Gate::rzz_vec(angle, &[0, 1, 2, 3]);
        assert_eq!(rzz_pairs.gate_type, rzz_vec.gate_type);
        assert_eq!(rzz_pairs.qubits, rzz_vec.qubits);
        assert_eq!(rzz_pairs.angles, rzz_vec.angles);
    }

    #[test]
    #[should_panic(expected = "CX gate requires an even number of qubits")]
    fn test_cx_vec_odd_qubits() {
        let _ = Gate::cx_vec(&[0, 1, 2]);
    }

    #[test]
    fn test_gate_validate_rejects_repeated_qubits_within_pair() {
        let err = Gate::cx(&[(0, 0)]).validate().unwrap_err();
        assert!(err.contains("requires distinct qubits"));
        assert!(err.contains("[0]"));
    }

    #[test]
    fn test_gate_validate_rejects_repeated_qubits_across_batched_pairs() {
        let err = Gate::swap(&[(0, 1), (1, 2)]).validate().unwrap_err();
        assert!(err.contains("requires distinct qubits"));
        assert!(err.contains("[1]"));
    }

    #[test]
    fn test_gate_validate_rejects_repeated_qubits_in_three_qubit_gate() {
        let err = Gate::ccx(&[(0, 1, 1)]).validate().unwrap_err();
        assert!(err.contains("requires distinct qubits"));
        assert!(err.contains("[1]"));
    }

    #[test]
    fn test_gate_validate_rejects_repeated_qubits_in_parameterized_two_qubit_gates() {
        let angle = Angle64::from_turns(0.25);
        let kak_angles = [[Angle64::ZERO; 3]; 2];
        let interaction = [Angle64::ZERO; 3];

        let cases = [
            Gate::rzz(angle, &[(2, 2)]),
            Gate::rxxryyrzz(angle, angle, angle, &[(3, 3)]),
            Gate::u2q(kak_angles, interaction, kak_angles, &[(4, 4)]),
        ];

        for gate in cases {
            let err = gate.validate().unwrap_err();
            assert!(err.contains("requires distinct qubits"), "{err}");
        }
    }

    #[test]
    #[should_panic(expected = "SZZ gate requires an even number of qubits")]
    fn test_szz_vec_odd_qubits() {
        let _ = Gate::szz_vec(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "SZZdg gate requires an even number of qubits")]
    fn test_szzdg_vec_odd_qubits() {
        let _ = Gate::szzdg_vec(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "RZZ gate requires an even number of qubits")]
    fn test_rzz_vec_odd_qubits() {
        let _ = Gate::rzz_vec(Angle64::from_turns(0.5), &[0, 1, 2]);
    }

    #[test]
    fn test_flatten_qubit_pairs_helper() {
        // Test the helper function directly
        let pairs = [(0usize, 1usize), (2usize, 3usize), (4usize, 5usize)];
        let flattened = Gate::flatten_qubit_pairs(&pairs);
        let expected: GateQubits = vec![0, 1, 2, 3, 4, 5]
            .into_iter()
            .map(QubitId::from)
            .collect();
        assert_eq!(flattened, expected);

        // Test empty case
        let empty_pairs: &[(usize, usize)] = &[];
        let flattened_empty = Gate::flatten_qubit_pairs(empty_pairs);
        assert!(flattened_empty.is_empty());
    }

    #[test]
    fn test_gate_arity_methods() {
        // Test single-qubit gates
        let x_gate = Gate::x(&[0]);
        assert_eq!(x_gate.classical_arity(), 0);
        assert_eq!(x_gate.angle_arity(), 0);
        assert_eq!(x_gate.quantum_arity(), 1);
        assert!(!x_gate.is_parameterized());
        assert!(x_gate.is_single_qubit());
        assert!(!x_gate.is_two_qubit());

        // Test parameterized single-qubit gates
        let rz_gate = Gate::rz(Angle64::from_turns(0.25), &[0]);
        assert_eq!(rz_gate.classical_arity(), 1);
        assert_eq!(rz_gate.angle_arity(), 1);
        assert_eq!(rz_gate.quantum_arity(), 1);
        assert!(rz_gate.is_parameterized());
        assert!(rz_gate.is_single_qubit());
        assert!(!rz_gate.is_two_qubit());

        // Test two-parameter single-qubit gates
        let r1xy_gate = Gate::r1xy(Angle64::from_turns(0.5), Angle64::from_turns(0.25), &[1]);
        assert_eq!(r1xy_gate.classical_arity(), 2);
        assert_eq!(r1xy_gate.angle_arity(), 2);
        assert_eq!(r1xy_gate.quantum_arity(), 1);
        assert!(r1xy_gate.is_parameterized());
        assert!(r1xy_gate.is_single_qubit());
        assert!(!r1xy_gate.is_two_qubit());

        // Test three-parameter single-qubit gates
        let u_gate = Gate::u(
            Angle64::from_turns(0.5),
            Angle64::from_turns(0.25),
            Angle64::from_turns(0.125),
            &[2],
        );
        assert_eq!(u_gate.classical_arity(), 3);
        assert_eq!(u_gate.angle_arity(), 3);
        assert_eq!(u_gate.quantum_arity(), 1);
        assert!(u_gate.is_parameterized());
        assert!(u_gate.is_single_qubit());
        assert!(!u_gate.is_two_qubit());

        // Test two-qubit gates
        let cx_gate = Gate::cx(&[(0, 1)]);
        assert_eq!(cx_gate.classical_arity(), 0);
        assert_eq!(cx_gate.angle_arity(), 0);
        assert_eq!(cx_gate.quantum_arity(), 2);
        assert!(!cx_gate.is_parameterized());
        assert!(!cx_gate.is_single_qubit());
        assert!(cx_gate.is_two_qubit());

        // Test parameterized two-qubit gates
        let rzz_two_qubit = Gate::rzz(Angle64::from_turns(0.25), &[(0, 1)]);
        assert_eq!(rzz_two_qubit.classical_arity(), 1);
        assert_eq!(rzz_two_qubit.angle_arity(), 1);
        assert_eq!(rzz_two_qubit.quantum_arity(), 2);
        assert!(rzz_two_qubit.is_parameterized());
        assert!(!rzz_two_qubit.is_single_qubit());
        assert!(rzz_two_qubit.is_two_qubit());

        // Test idle gate (single-qubit, parameterized but not with angles)
        let idle_gate = Gate::idle(1.0, vec![QubitId::from(0)]);
        assert_eq!(idle_gate.classical_arity(), 1);
        assert_eq!(idle_gate.angle_arity(), 0); // Idle uses params, not angles
        assert_eq!(idle_gate.quantum_arity(), 1);
        assert!(idle_gate.is_parameterized());
        assert!(idle_gate.is_single_qubit());
        assert!(!idle_gate.is_two_qubit());
    }

    #[test]
    fn test_gate_validation() {
        // Test valid gates
        let valid_x = Gate::x(&[0]);
        assert!(valid_x.validate().is_ok());

        let valid_rz = Gate::rz(Angle64::from_turns(0.25), &[1]);
        assert!(valid_rz.validate().is_ok());

        let valid_r1xy = Gate::r1xy(Angle64::from_turns(0.5), Angle64::from_turns(0.25), &[2]);
        assert!(valid_r1xy.validate().is_ok());

        let valid_u = Gate::u(
            Angle64::from_turns(0.5),
            Angle64::from_turns(0.25),
            Angle64::from_turns(0.125),
            &[3],
        );
        assert!(valid_u.validate().is_ok());

        let valid_cx_gate = Gate::cx(&[(0, 1)]);
        assert!(valid_cx_gate.validate().is_ok());

        let valid_rzz = Gate::rzz(Angle64::from_turns(0.25), &[(2, 3)]);
        assert!(valid_rzz.validate().is_ok());

        // Test invalid gates - wrong angle count
        let invalid_angles = Gate::new(
            GateType::RZ,
            vec![Angle64::from_turns(0.25), Angle64::from_turns(0.5)],
            Vec::<f64>::new(),
            vec![QubitId::from(0)],
        );
        assert!(invalid_angles.validate().is_err());
        assert!(
            invalid_angles
                .validate()
                .unwrap_err()
                .contains("expected 1 angle parameters, got 2")
        );

        let missing_angles = Gate::new(
            GateType::U,
            vec![Angle64::from_turns(0.25)],
            Vec::<f64>::new(),
            vec![QubitId::from(0)],
        );
        assert!(missing_angles.validate().is_err());
        assert!(
            missing_angles
                .validate()
                .unwrap_err()
                .contains("expected 3 angle parameters, got 1")
        );

        // Test invalid gates - wrong qubit count (not a multiple of quantum arity)
        let invalid_qubits = Gate::new(
            GateType::CX,
            Vec::<Angle64>::new(),
            Vec::<f64>::new(),
            vec![QubitId::from(0)],
        );
        assert!(invalid_qubits.validate().is_err());
        assert!(
            invalid_qubits
                .validate()
                .unwrap_err()
                .contains("requires a multiple of 2 qubits, got 1")
        );

        let odd_cx_qubits = Gate::new(
            GateType::CX,
            Vec::<Angle64>::new(),
            Vec::<f64>::new(),
            vec![QubitId::from(0), QubitId::from(1), QubitId::from(2)],
        );
        assert!(odd_cx_qubits.validate().is_err());
        assert!(
            odd_cx_qubits
                .validate()
                .unwrap_err()
                .contains("requires a multiple of 2 qubits, got 3")
        );

        // Test valid multi-qubit gates
        let multi_x = Gate::new(
            GateType::X,
            Vec::<Angle64>::new(),
            Vec::<f64>::new(),
            vec![QubitId::from(0), QubitId::from(1), QubitId::from(2)],
        );
        assert!(multi_x.validate().is_ok()); // Multiple X gates on different qubits

        let multi_cx_gates = Gate::new(
            GateType::CX,
            Vec::<Angle64>::new(),
            Vec::<f64>::new(),
            vec![
                QubitId::from(0),
                QubitId::from(1),
                QubitId::from(2),
                QubitId::from(3),
            ],
        );
        assert!(multi_cx_gates.validate().is_ok()); // Multiple CX gates
    }

    #[test]
    fn test_gate_validation_rejects_duplicate_qubits_for_batched_commands() {
        let duplicate_x = Gate::x(&[0, 0]);
        let err = duplicate_x.validate().unwrap_err();
        assert!(err.contains("requires distinct qubits"));
        assert!(err.contains("[0]"));

        let duplicate_mz = Gate::mz(&[1, 1]);
        let err = duplicate_mz.validate().unwrap_err();
        assert!(err.contains("requires distinct qubits"));
        assert!(err.contains("[1]"));
    }

    #[test]
    fn test_gate_validation_checks_non_angle_parameters_and_measurement_ids() {
        let missing_idle_duration = Gate::new(
            GateType::Idle,
            Vec::<Angle64>::new(),
            Vec::<f64>::new(),
            vec![QubitId::from(0)],
        );
        assert!(
            missing_idle_duration
                .validate()
                .unwrap_err()
                .contains("expected 1 non-angle parameters, got 0")
        );

        let mut measured = Gate::mz(&[0, 1]);
        measured.meas_ids.push(MeasId(0));
        assert!(
            measured
                .validate()
                .unwrap_err()
                .contains("expected measurement-id count to be 0 or 2, got 1")
        );

        let mut non_measurement = Gate::x(&[0]);
        non_measurement.meas_ids.push(MeasId(0));
        assert!(
            non_measurement
                .validate()
                .unwrap_err()
                .contains("cannot carry measurement-id payloads")
        );
    }

    #[test]
    fn test_channel_gate_validation_rejects_stale_payloads() {
        use crate::channel::{BitFlip, Depolarizing};

        let mut stale_qubits = Gate::channel(Depolarizing(0.25, 0));
        stale_qubits.qubits = vec![QubitId::from(1)].into();
        assert!(
            stale_qubits
                .validate()
                .unwrap_err()
                .contains("do not match channel payload qubits")
        );

        let mut stale_angles = Gate::channel(BitFlip(0.1, 0));
        stale_angles.angles.push(Angle64::from_turns(0.25));
        assert!(
            stale_angles
                .validate()
                .unwrap_err()
                .contains("cannot carry angle, parameter, or measurement-id payloads")
        );

        let mut channel_payload_on_ideal_gate = Gate::x(&[0]);
        channel_payload_on_ideal_gate.channel = Some(BitFlip(0.1, 0));
        assert!(
            channel_payload_on_ideal_gate
                .validate()
                .unwrap_err()
                .contains("Only GateType::Channel can carry a channel payload")
        );
    }
}
