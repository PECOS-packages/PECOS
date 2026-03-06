//! Validation utilities for `CommandQueue`.
//!
//! This module provides utilities to validate `CommandQueue` circuits
//! using the extensible gate validators.

use super::{
    AngleSnapper, CircuitValidator, GateForValidation, GateRegistry, SnapError, SnapPolicy,
    ValidationError,
};
use crate::command::{CommandQueue, GateCommand, GateType};
use pecos_core::Angle64;

/// Extension trait for validating `CommandQueue` circuits.
pub trait CommandQueueValidation {
    /// Validate this command queue using the given validator.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_neo::extensible::CommandQueueValidation;
    ///
    /// let commands = CommandBuilder::new()
    ///     .pz(0)
    ///     .h(0)
    ///     .mz(0)
    ///     .build();
    ///
    /// let registry = GateRegistry::new();
    /// let validator = CliffordValidator::new();
    /// commands.validate(&validator, &registry).unwrap();
    /// ```
    fn validate(
        &self,
        validator: &dyn CircuitValidator,
        registry: &GateRegistry,
    ) -> Result<(), ValidationError>;

    /// Convert to a vector of `GateForValidation` for use with validators.
    fn to_gate_validations(&self) -> Vec<GateForValidation>;
}

impl CommandQueueValidation for CommandQueue {
    fn validate(
        &self,
        validator: &dyn CircuitValidator,
        registry: &GateRegistry,
    ) -> Result<(), ValidationError> {
        let gates = self.to_gate_validations();
        validator.validate(&gates, registry)
    }

    fn to_gate_validations(&self) -> Vec<GateForValidation> {
        self.iter()
            .map(|cmd| GateForValidation {
                gate_id: cmd.gate_type.to_gate_id(),
                angles: cmd.angles.iter().copied().collect(),
            })
            .collect()
    }
}

/// Snap all angles in a `CommandQueue` according to a policy.
///
/// This is useful for normalizing floating-point angles to exact values
/// before validation or execution.
///
/// # Arguments
///
/// * `commands` - The command queue to snap
/// * `policy` - The snapping policy to use
/// * `snapper` - The angle snapper with target angles
///
/// # Returns
///
/// A new `CommandQueue` with snapped angles, or an error if snapping fails.
pub fn snap_command_queue(
    commands: &CommandQueue,
    policy: SnapPolicy,
    snapper: &AngleSnapper,
) -> Result<CommandQueue, (usize, SnapError)> {
    let mut result = CommandQueue::with_capacity(commands.len());

    for (idx, cmd) in commands.iter().enumerate() {
        if cmd.angles.is_empty() {
            result.push(cmd.clone());
        } else {
            let mut snapped_angles = smallvec::SmallVec::<[Angle64; 2]>::new();

            for angle in &cmd.angles {
                match policy {
                    SnapPolicy::Exact => {
                        snapped_angles.push(*angle);
                    }
                    SnapPolicy::SnapOrFail { tolerance: _ } => match snapper.snap(*angle) {
                        Ok(result) => snapped_angles.push(result.snapped),
                        Err(e) => return Err((idx, e)),
                    },
                    SnapPolicy::SnapOrKeep { tolerance: _ } => match snapper.snap(*angle) {
                        Ok(result) => snapped_angles.push(result.snapped),
                        Err(_) => snapped_angles.push(*angle),
                    },
                }
            }

            result.push(GateCommand::with_angles(
                cmd.gate_type,
                cmd.qubits.clone(),
                snapped_angles,
            ));
        }
    }

    Ok(result)
}

/// Check if a `CommandQueue` contains only Clifford gates.
///
/// This is a quick check that doesn't require a registry.
#[must_use]
pub fn is_clifford_circuit(commands: &CommandQueue) -> bool {
    commands.iter().all(|cmd| {
        // Check gate type
        if !is_clifford_gate_type(cmd.gate_type) {
            return false;
        }

        // For parameterized gates, check if angles are Clifford angles
        if cmd.gate_type.angle_arity() > 0 && !cmd.angles.is_empty() {
            return cmd.angles.iter().all(|a| is_clifford_angle(*a));
        }

        true
    })
}

/// Check if a gate type is a Clifford gate.
#[must_use]
pub fn is_clifford_gate_type(gate_type: GateType) -> bool {
    matches!(
        gate_type,
        GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::H
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
            | GateType::MZ
            | GateType::MeasureLeaked
            | GateType::MeasureFree
            | GateType::PZ
            | GateType::QAlloc
            | GateType::QFree
            | GateType::Idle
            // Parameterized gates are Clifford only at specific angles
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::RZZ
    )
}

/// Check if an angle is a Clifford angle (multiple of pi/2).
#[must_use]
pub fn is_clifford_angle(angle: Angle64) -> bool {
    angle == Angle64::ZERO
        || angle == Angle64::QUARTER_TURN
        || angle == Angle64::HALF_TURN
        || angle == Angle64::THREE_QUARTERS_TURN
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::extensible::{CliffordValidator, GateRegistry};

    #[test]
    fn test_validate_clifford_circuit() {
        let commands = CommandBuilder::new()
            .pz(0)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        let registry = GateRegistry::new();
        let validator = CliffordValidator::new();

        assert!(commands.validate(&validator, &registry).is_ok());
    }

    #[test]
    fn test_validate_rejects_t_gate() {
        let commands = CommandBuilder::new()
            .pz(0)
            .t(0) // T gate is not Clifford
            .mz(0)
            .build();

        let registry = GateRegistry::new();
        let validator = CliffordValidator::new();

        assert!(commands.validate(&validator, &registry).is_err());
    }

    #[test]
    fn test_is_clifford_circuit() {
        let clifford = CommandBuilder::new().pz(0).h(0).sz(0).mz(0).build();

        assert!(is_clifford_circuit(&clifford));

        let non_clifford = CommandBuilder::new().pz(0).t(0).mz(0).build();

        assert!(!is_clifford_circuit(&non_clifford));
    }

    #[test]
    fn test_is_clifford_angle() {
        assert!(is_clifford_angle(Angle64::ZERO));
        assert!(is_clifford_angle(Angle64::QUARTER_TURN));
        assert!(is_clifford_angle(Angle64::HALF_TURN));
        assert!(is_clifford_angle(Angle64::THREE_QUARTERS_TURN));

        // T angle (pi/4) is not Clifford
        let t_angle = Angle64::HALF_TURN / 4;
        assert!(!is_clifford_angle(t_angle));
    }

    #[test]
    fn test_rz_at_clifford_angle_is_clifford() {
        let commands = CommandBuilder::new()
            .pz(0)
            .rz(0, Angle64::QUARTER_TURN) // RZ(pi/2) = SZ, Clifford
            .mz(0)
            .build();

        assert!(is_clifford_circuit(&commands));
    }

    #[test]
    fn test_rz_at_non_clifford_angle() {
        let commands = CommandBuilder::new()
            .pz(0)
            .rz(0, Angle64::HALF_TURN / 4) // RZ(pi/4) = T, not Clifford
            .mz(0)
            .build();

        assert!(!is_clifford_circuit(&commands));
    }

    #[test]
    fn test_snap_command_queue_exact() {
        let commands = CommandBuilder::new()
            .pz(0)
            .rz(0, Angle64::QUARTER_TURN)
            .mz(0)
            .build();

        let snapper = AngleSnapper::clifford(1e-9);
        let snapped = snap_command_queue(&commands, SnapPolicy::Exact, &snapper).unwrap();

        // Exact policy doesn't change anything
        assert_eq!(snapped.len(), commands.len());
    }

    #[test]
    fn test_to_gate_validations() {
        let commands = CommandBuilder::new()
            .pz(0)
            .h(0)
            .rz(0, Angle64::QUARTER_TURN)
            .mz(0)
            .build();

        let validations = commands.to_gate_validations();

        assert_eq!(validations.len(), 4);
        assert_eq!(validations[0].gate_id, GateType::PZ.to_gate_id());
        assert_eq!(validations[1].gate_id, GateType::H.to_gate_id());
        assert_eq!(validations[2].gate_id, GateType::RZ.to_gate_id());
        assert_eq!(validations[2].angles.len(), 1);
        assert_eq!(validations[3].gate_id, GateType::MZ.to_gate_id());
    }
}
