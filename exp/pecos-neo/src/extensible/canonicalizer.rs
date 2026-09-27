//! Gate canonicalization - maps parameterized gates to fixed gates at exact angles.

use super::GateId;
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
    shared_policy: bool,
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
        Self {
            rules: Vec::new(),
            shared_policy: false,
        }
    }

    /// Create a canonicalizer with standard gate mappings.
    #[must_use]
    pub fn standard() -> Self {
        use Angle64 as A;

        let mut canon = Self::new();

        canon.shared_policy = true;
        for &gate_type in crate::GateType::ALL {
            let gate_id = gate_type.to_gate_id();
            for angle in [
                A::ZERO,
                A::QUARTER_TURN,
                A::HALF_TURN,
                A::THREE_QUARTERS_TURN,
            ] {
                if let Some(pecos_core::CliffordLowering::Named(named)) =
                    lower_rotation(gate_id, &[angle])
                {
                    canon.add(gate_id, angle, crate::GateType::from(named).to_gate_id());
                }
            }
        }

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
    /// A per-qubit decomposition cannot be represented by a single gate ID.
    #[must_use]
    pub fn canonicalize(&self, gate_id: GateId, angles: &[Angle64]) -> Option<GateId> {
        if self.shared_policy
            && let Some(pecos_core::CliffordLowering::Named(named)) =
                lower_rotation(gate_id, angles)
        {
            return Some(crate::GateType::from(named).to_gate_id());
        }
        // Custom rules describe single-angle gates.
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
        (self.shared_policy
            && gate_id
                .try_to_gate_type()
                .is_some_and(|gate| pecos_core::is_lowerable_rotation(gate.into())))
            || self.rules.iter().any(|c| c.from_gate == gate_id)
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

/// Resolve built-in rotations through the core policy without inventing qubits.
pub(super) fn lower_rotation(
    gate_id: GateId,
    angles: &[Angle64],
) -> Option<pecos_core::CliffordLowering> {
    let gate_type = gate_id.try_to_gate_type()?;
    let gate = pecos_core::Gate::try_with_angles(
        gate_type.into(),
        angles.iter().copied().collect::<pecos_core::GateAngles>(),
        smallvec::SmallVec::new(),
    )
    .ok()?;
    pecos_core::try_lower_rotation_to_clifford(&gate)
}
