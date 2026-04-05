//! Circuit validators for enforcing gate and angle constraints.
//!
//! Validators check circuits at build time to ensure they meet
//! specific requirements (e.g., Clifford-only, exact angles).

use super::{GateCanonicalizer, GateId, GateRegistry, GateSupportSet, gates};
use pecos_core::Angle64;

/// Convert angle to turns (0.0 to 1.0).
fn angle_to_turns(angle: Angle64) -> f64 {
    angle.to_radians() / std::f64::consts::TAU
}

/// Error from circuit validation.
#[derive(Clone, Debug)]
pub enum ValidationError {
    /// A gate is not allowed by this validator
    ForbiddenGate {
        gate_id: GateId,
        gate_name: String,
        position: usize,
    },
    /// An angle value is not allowed
    ForbiddenAngle {
        gate_id: GateId,
        gate_name: String,
        angle: Angle64,
        position: usize,
        allowed: Vec<Angle64>,
    },
    /// An angle cannot be canonicalized to a fixed gate
    NonCanonicalAngle {
        gate_id: GateId,
        gate_name: String,
        angle: Angle64,
        position: usize,
    },
    /// Gate ID is not registered
    UnknownGate { gate_id: GateId, position: usize },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForbiddenGate {
                gate_name,
                position,
                ..
            } => {
                write!(f, "Forbidden gate '{gate_name}' at position {position}")
            }
            Self::ForbiddenAngle {
                gate_name,
                angle,
                position,
                ..
            } => {
                write!(
                    f,
                    "Forbidden angle {:.6} turns for gate '{}' at position {}",
                    angle_to_turns(*angle),
                    gate_name,
                    position
                )
            }
            Self::NonCanonicalAngle {
                gate_name,
                angle,
                position,
                ..
            } => {
                write!(
                    f,
                    "Non-canonical angle {:.6} turns for gate '{}' at position {} (must be a standard angle like pi/4, pi/2, pi)",
                    angle_to_turns(*angle),
                    gate_name,
                    position
                )
            }
            Self::UnknownGate { gate_id, position } => {
                write!(f, "Unknown gate ID {} at position {}", gate_id.0, position)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// A gate with its angles for validation purposes.
#[derive(Clone, Debug)]
pub struct GateForValidation {
    pub gate_id: GateId,
    pub angles: Vec<Angle64>,
}

/// Trait for circuit validators.
pub trait CircuitValidator {
    /// Validate a sequence of gates.
    ///
    /// # Errors
    /// Returns `ValidationError` if any gate is not permitted by this validator.
    fn validate(
        &self,
        gates: &[GateForValidation],
        registry: &GateRegistry,
    ) -> Result<(), ValidationError>;

    /// Check if a single gate is allowed (without position info).
    fn is_gate_allowed(&self, gate_id: GateId, angles: &[Angle64], registry: &GateRegistry)
    -> bool;
}

/// Validator for Clifford-only circuits.
///
/// Only allows Clifford gates and parameterized gates at Clifford angles.
pub struct CliffordValidator {
    /// Allowed non-parameterized gates
    allowed_gates: GateSupportSet,
    /// For parameterized gates, allowed exact angles
    allowed_angles: Vec<Angle64>,
}

impl Default for CliffordValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CliffordValidator {
    /// Create a new Clifford validator.
    #[must_use]
    pub fn new() -> Self {
        use Angle64 as A;
        let mut allowed_gates = GateSupportSet::new();

        // Paulis
        allowed_gates.insert(gates::I);
        allowed_gates.insert(gates::X);
        allowed_gates.insert(gates::Y);
        allowed_gates.insert(gates::Z);

        // Cliffords
        allowed_gates.insert(gates::H);
        allowed_gates.insert(gates::SX);
        allowed_gates.insert(gates::SXdg);
        allowed_gates.insert(gates::SY);
        allowed_gates.insert(gates::SYdg);
        allowed_gates.insert(gates::SZ);
        allowed_gates.insert(gates::SZdg);

        // Two-qubit Cliffords
        allowed_gates.insert(gates::CX);
        allowed_gates.insert(gates::CY);
        allowed_gates.insert(gates::CZ);
        allowed_gates.insert(gates::SWAP);
        allowed_gates.insert(gates::ISWAP);
        allowed_gates.insert(gates::SZZ);
        allowed_gates.insert(gates::SZZdg);

        // Measurement and prep
        allowed_gates.insert(gates::MZ);
        allowed_gates.insert(gates::PZ);

        // Parameterized gates are allowed only at Clifford angles
        allowed_gates.insert(gates::RX);
        allowed_gates.insert(gates::RY);
        allowed_gates.insert(gates::RZ);
        allowed_gates.insert(gates::RZZ);
        let allowed_angles = vec![
            A::ZERO,
            A::QUARTER_TURN,        // pi/2
            A::HALF_TURN,           // pi
            A::THREE_QUARTERS_TURN, // 3pi/2 (same as -pi/2)
        ];

        Self {
            allowed_gates,
            allowed_angles,
        }
    }

    /// Check if an angle is a Clifford angle.
    #[must_use]
    pub fn is_clifford_angle(&self, angle: Angle64) -> bool {
        self.allowed_angles.contains(&angle)
    }
}

impl CircuitValidator for CliffordValidator {
    fn validate(
        &self,
        gates: &[GateForValidation],
        registry: &GateRegistry,
    ) -> Result<(), ValidationError> {
        for (idx, gate) in gates.iter().enumerate() {
            // Check if gate is registered
            let spec = registry.get(gate.gate_id).ok_or({
                ValidationError::UnknownGate {
                    gate_id: gate.gate_id,
                    position: idx,
                }
            })?;

            // Check if gate type is allowed
            if !self.allowed_gates.contains(gate.gate_id) {
                return Err(ValidationError::ForbiddenGate {
                    gate_id: gate.gate_id,
                    gate_name: spec.name.to_string(),
                    position: idx,
                });
            }

            // For parameterized gates, check angles
            if spec.angle_arity > 0 && !gate.angles.is_empty() {
                for angle in &gate.angles {
                    if !self.is_clifford_angle(*angle) {
                        return Err(ValidationError::ForbiddenAngle {
                            gate_id: gate.gate_id,
                            gate_name: spec.name.to_string(),
                            angle: *angle,
                            position: idx,
                            allowed: self.allowed_angles.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn is_gate_allowed(
        &self,
        gate_id: GateId,
        angles: &[Angle64],
        registry: &GateRegistry,
    ) -> bool {
        if !self.allowed_gates.contains(gate_id) {
            return false;
        }

        if let Some(spec) = registry.get(gate_id)
            && spec.angle_arity > 0
        {
            return angles.iter().all(|a| self.is_clifford_angle(*a));
        }

        true
    }
}

/// Validator for Clifford+T circuits.
///
/// Allows Clifford gates plus T and Tdg gates.
pub struct CliffordTValidator {
    inner: CliffordValidator,
}

impl Default for CliffordTValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CliffordTValidator {
    /// Create a new Clifford+T validator.
    #[must_use]
    pub fn new() -> Self {
        use Angle64 as A;
        let mut inner = CliffordValidator::new();

        // Add T gates
        inner.allowed_gates.insert(gates::T);
        inner.allowed_gates.insert(gates::Tdg);

        // Add T angle (pi/4)
        inner.allowed_angles.push(A::HALF_TURN / 4); // pi/4
        inner.allowed_angles.push(A::ZERO - A::HALF_TURN / 4); // -pi/4

        Self { inner }
    }
}

impl CircuitValidator for CliffordTValidator {
    fn validate(
        &self,
        gates: &[GateForValidation],
        registry: &GateRegistry,
    ) -> Result<(), ValidationError> {
        self.inner.validate(gates, registry)
    }

    fn is_gate_allowed(
        &self,
        gate_id: GateId,
        angles: &[Angle64],
        registry: &GateRegistry,
    ) -> bool {
        self.inner.is_gate_allowed(gate_id, angles, registry)
    }
}

/// Validator that requires all angles to be canonicalizable.
///
/// Ensures that every parameterized gate with an angle can be
/// converted to an equivalent fixed gate (e.g., RZ(pi/2) -> SZ).
pub struct ExactAngleValidator {
    canonicalizer: GateCanonicalizer,
}

impl Default for ExactAngleValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactAngleValidator {
    /// Create a new exact angle validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            canonicalizer: GateCanonicalizer::standard(),
        }
    }

    /// Create with a custom canonicalizer.
    #[must_use]
    pub fn with_canonicalizer(canonicalizer: GateCanonicalizer) -> Self {
        Self { canonicalizer }
    }
}

impl CircuitValidator for ExactAngleValidator {
    fn validate(
        &self,
        gates: &[GateForValidation],
        registry: &GateRegistry,
    ) -> Result<(), ValidationError> {
        for (idx, gate) in gates.iter().enumerate() {
            let spec = registry.get(gate.gate_id).ok_or({
                ValidationError::UnknownGate {
                    gate_id: gate.gate_id,
                    position: idx,
                }
            })?;

            // Skip gates without angles
            if gate.angles.is_empty() {
                continue;
            }

            // For single-angle gates, check if canonicalizable
            if gate.angles.len() == 1
                && self
                    .canonicalizer
                    .canonicalize(gate.gate_id, &gate.angles)
                    .is_none()
            {
                // Check if this is a gate that can be canonicalized at all
                if self.canonicalizer.can_canonicalize(gate.gate_id) {
                    return Err(ValidationError::NonCanonicalAngle {
                        gate_id: gate.gate_id,
                        gate_name: spec.name.to_string(),
                        angle: gate.angles[0],
                        position: idx,
                    });
                }
            }
        }

        Ok(())
    }

    fn is_gate_allowed(
        &self,
        gate_id: GateId,
        angles: &[Angle64],
        _registry: &GateRegistry,
    ) -> bool {
        if angles.is_empty() {
            return true;
        }

        if angles.len() == 1 {
            // If this gate can be canonicalized, check if this angle works
            if self.canonicalizer.can_canonicalize(gate_id) {
                return self.canonicalizer.canonicalize(gate_id, angles).is_some();
            }
        }

        // For multi-angle gates or non-canonicalizable gates, allow
        true
    }
}

/// Validator that allows only specific gates.
pub struct AllowListValidator {
    allowed: GateSupportSet,
}

impl AllowListValidator {
    /// Create a new allow-list validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allowed: GateSupportSet::new(),
        }
    }

    /// Create from a list of allowed gate IDs.
    #[must_use]
    pub fn from_gates(gates: &[GateId]) -> Self {
        let mut allowed = GateSupportSet::new();
        for &id in gates {
            allowed.insert(id);
        }
        Self { allowed }
    }

    /// Add a gate to the allow list.
    pub fn allow(&mut self, gate_id: GateId) {
        self.allowed.insert(gate_id);
    }

    /// Check if a gate is allowed.
    #[must_use]
    pub fn is_allowed(&self, gate_id: GateId) -> bool {
        self.allowed.contains(gate_id)
    }
}

impl Default for AllowListValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitValidator for AllowListValidator {
    fn validate(
        &self,
        gates: &[GateForValidation],
        registry: &GateRegistry,
    ) -> Result<(), ValidationError> {
        for (idx, gate) in gates.iter().enumerate() {
            if !self.allowed.contains(gate.gate_id) {
                let gate_name = registry
                    .get(gate.gate_id)
                    .map_or_else(|| format!("ID({})", gate.gate_id.0), |s| s.name.to_string());

                return Err(ValidationError::ForbiddenGate {
                    gate_id: gate.gate_id,
                    gate_name,
                    position: idx,
                });
            }
        }

        Ok(())
    }

    fn is_gate_allowed(
        &self,
        gate_id: GateId,
        _angles: &[Angle64],
        _registry: &GateRegistry,
    ) -> bool {
        self.allowed.contains(gate_id)
    }
}

/// Composite validator that runs multiple validators.
pub struct CompositeValidator {
    validators: Vec<Box<dyn CircuitValidator>>,
}

impl CompositeValidator {
    /// Create a new composite validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to the chain.
    pub fn add<V: CircuitValidator + 'static>(&mut self, validator: V) {
        self.validators.push(Box::new(validator));
    }

    /// Builder pattern for adding validators.
    #[must_use]
    pub fn with<V: CircuitValidator + 'static>(mut self, validator: V) -> Self {
        self.add(validator);
        self
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitValidator for CompositeValidator {
    fn validate(
        &self,
        gates: &[GateForValidation],
        registry: &GateRegistry,
    ) -> Result<(), ValidationError> {
        for validator in &self.validators {
            validator.validate(gates, registry)?;
        }
        Ok(())
    }

    fn is_gate_allowed(
        &self,
        gate_id: GateId,
        angles: &[Angle64],
        registry: &GateRegistry,
    ) -> bool {
        self.validators
            .iter()
            .all(|v| v.is_gate_allowed(gate_id, angles, registry))
    }
}
