//! Gate canonicalization - maps parameterized gates to fixed gates at exact angles.

use super::{GateId, gates};
use pecos_core::Angle64;

/// A canonical form mapping: parameterized gate at exact angle → fixed gate.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalForm {
    /// The parameterized gate (e.g., RZ)
    pub from_gate: GateId,
    /// The exact angle value (fixed-point, no tolerance needed)
    pub angle: Angle64,
    /// The equivalent fixed gate (e.g., SZ)
    pub to_gate: GateId,
}

/// Canonicalizer for mapping parameterized gates to fixed gates.
///
/// Uses exact Angle64 comparison - no floating-point tolerance needed
/// because Angle64 is fixed-point and standard angles are exactly representable.
pub struct GateCanonicalizer {
    /// Known rules, sorted by `(from_gate, angle)` for binary search
    rules: Vec<CanonicalForm>,
}

impl Default for GateCanonicalizer {
    fn default() -> Self {
        Self::standard()
    }
}

impl GateCanonicalizer {
    /// Create an empty canonicalizer.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create a canonicalizer with standard gate mappings.
    #[must_use]
    pub fn standard() -> Self {
        use Angle64 as A;

        let mut canon = Self::new();

        // RZ rules
        canon.add(gates::RZ, A::ZERO, gates::I);
        canon.add(gates::RZ, A::HALF_TURN / 4, gates::T); // π/4
        canon.add(gates::RZ, A::ZERO - A::HALF_TURN / 4, gates::Tdg); // -π/4
        canon.add(gates::RZ, A::QUARTER_TURN, gates::SZ); // π/2
        canon.add(gates::RZ, A::ZERO - A::QUARTER_TURN, gates::SZdg); // -π/2
        canon.add(gates::RZ, A::HALF_TURN, gates::Z); // π

        // RX rules
        canon.add(gates::RX, A::ZERO, gates::I);
        canon.add(gates::RX, A::QUARTER_TURN, gates::SX); // π/2
        canon.add(gates::RX, A::ZERO - A::QUARTER_TURN, gates::SXdg); // -π/2
        canon.add(gates::RX, A::HALF_TURN, gates::X); // π

        // RY rules
        canon.add(gates::RY, A::ZERO, gates::I);
        canon.add(gates::RY, A::QUARTER_TURN, gates::SY); // π/2
        canon.add(gates::RY, A::ZERO - A::QUARTER_TURN, gates::SYdg); // -π/2
        canon.add(gates::RY, A::HALF_TURN, gates::Y); // π

        // RZZ rules
        canon.add(gates::RZZ, A::QUARTER_TURN, gates::SZZ); // π/2
        canon.add(gates::RZZ, A::ZERO - A::QUARTER_TURN, gates::SZZdg); // -π/2

        // RXX rules
        canon.add(gates::RXX, A::QUARTER_TURN, gates::SXX); // π/2
        canon.add(gates::RXX, A::ZERO - A::QUARTER_TURN, gates::SXXdg); // -π/2

        // RYY rules
        canon.add(gates::RYY, A::QUARTER_TURN, gates::SYY); // π/2
        canon.add(gates::RYY, A::ZERO - A::QUARTER_TURN, gates::SYYdg); // -π/2

        // Sort for efficient lookup
        canon.rules.sort_by(|a, b| {
            a.from_gate
                .cmp(&b.from_gate)
                .then_with(|| a.angle.cmp(&b.angle))
        });

        canon
    }

    /// Add a canonicalization rule.
    pub fn add(&mut self, from_gate: GateId, angle: Angle64, to_gate: GateId) {
        self.rules.push(CanonicalForm {
            from_gate,
            angle,
            to_gate,
        });
    }

    /// Try to canonicalize a gate.
    ///
    /// Returns the canonical fixed gate if the parameterized gate with exact angle
    /// has a known canonical form. Uses exact `Angle64` comparison.
    ///
    /// Only handles single-angle gates currently.
    #[must_use]
    pub fn canonicalize(&self, gate_id: GateId, angles: &[Angle64]) -> Option<GateId> {
        // Only canonicalize single-angle gates
        if angles.len() != 1 {
            return None;
        }

        let angle = angles[0];

        // Linear search through rules for matching gate
        // (could use binary search if list gets large)
        for canon in &self.rules {
            if canon.from_gate == gate_id && canon.angle == angle {
                return Some(canon.to_gate);
            }
        }

        None
    }

    /// Try to expand a fixed gate to its parameterized form.
    ///
    /// This is the reverse of canonicalization.
    #[must_use]
    pub fn expand(&self, gate_id: GateId) -> Option<(GateId, Angle64)> {
        for canon in &self.rules {
            if canon.to_gate == gate_id {
                return Some((canon.from_gate, canon.angle));
            }
        }
        None
    }

    /// Check if a gate can be canonicalized at any angle.
    #[must_use]
    pub fn can_canonicalize(&self, gate_id: GateId) -> bool {
        self.rules.iter().any(|c| c.from_gate == gate_id)
    }

    /// Get all canonical forms for a given parameterized gate.
    #[must_use]
    pub fn get_forms_for(&self, gate_id: GateId) -> Vec<&CanonicalForm> {
        self.rules
            .iter()
            .filter(|c| c.from_gate == gate_id)
            .collect()
    }
}
