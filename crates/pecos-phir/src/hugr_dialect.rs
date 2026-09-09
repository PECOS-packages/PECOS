/*!
HUGR Dialect for PHIR

This dialect provides operations that correspond to HUGR's quantum and classical operations,
allowing PHIR to parse and represent HUGR programs without depending on external libraries.

The dialect follows HUGR's operation model but represents them as PHIR operations.
*/

use crate::dialect::{Dialect, DialectRegistry, OperationDef};
use crate::error::Result;
use crate::ops::CustomOp;
use crate::traits::OpTrait;

/// HUGR dialect implementation
pub struct HugrDialect;

// This is the single source of truth for fixed named gates accepted by the
// HUGR-to-QIS converter and registered by this dialect.
const HUGR_NAMED_GATE_DEFS: &[(&str, &str, i32, i32)] = &[
    ("h", "Hadamard gate", 1, 1),
    ("x", "Pauli-X gate", 1, 1),
    ("y", "Pauli-Y gate", 1, 1),
    ("z", "Pauli-Z gate", 1, 1),
    ("s", "S gate", 1, 1),
    ("sdg", "S-dagger gate", 1, 1),
    ("t", "T gate", 1, 1),
    ("tdg", "T-dagger gate", 1, 1),
    ("sx", "Square-root-of-X gate", 1, 1),
    ("sxdg", "Adjoint square-root-of-X gate", 1, 1),
    ("cx", "Controlled-X (CNOT) gate", 2, 2),
];

pub(crate) fn is_hugr_named_gate(name: &str) -> bool {
    HUGR_NAMED_GATE_DEFS
        .iter()
        .any(|(registered, _, _, _)| *registered == name)
}

impl Dialect for HugrDialect {
    fn namespace(&self) -> &'static str {
        "hugr"
    }

    fn description(&self) -> &'static str {
        "HUGR (Hierarchical Unified Graph Representation) operations for quantum programs"
    }

    #[allow(clippy::too_many_lines)] // Dialect initialization is inherently a long list of operation registrations
    fn initialize(&self, registry: &mut DialectRegistry) -> Result<()> {
        // Register HUGR quantum operations
        for &(name, description, num_operands, num_results) in HUGR_NAMED_GATE_DEFS {
            registry.register_operation(
                self.namespace(),
                OperationDef {
                    name: name.to_string(),
                    description: description.to_string(),
                    num_operands,
                    num_results,
                    num_regions: 0,
                    traits: vec![OpTrait::NoSideEffect],
                },
            )?;
        }

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "rz".to_string(),
                description: "RZ rotation gate".to_string(),
                num_operands: 2, // qubit + angle
                num_results: 1,
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "rx".to_string(),
                description: "RX rotation gate".to_string(),
                num_operands: 2, // qubit + angle
                num_results: 1,
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "ry".to_string(),
                description: "RY rotation gate".to_string(),
                num_operands: 2, // qubit + angle
                num_results: 1,
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        // Measurement operations
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "measure".to_string(),
                description: "Measurement in computational basis".to_string(),
                num_operands: 1,
                num_results: 1,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        // Quantum allocation
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "qalloc".to_string(),
                description: "Allocate a qubit".to_string(),
                num_operands: 0,
                num_results: 1,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "qfree".to_string(),
                description: "Free a qubit".to_string(),
                num_operands: 1,
                num_results: 0,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        // Control flow
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "conditional".to_string(),
                description: "Conditional branching".to_string(),
                num_operands: 1, // condition
                num_results: -1, // variadic
                num_regions: 2,  // then and else regions
                traits: vec![OpTrait::RegionBranch],
            },
        )?;

        // Function operations
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "funcdefn".to_string(),
                description: "Function definition".to_string(),
                num_operands: 0,
                num_results: 0,
                num_regions: 1, // body region
                traits: vec![OpTrait::FunctionLike],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "call".to_string(),
                description: "Function call".to_string(),
                num_operands: -1, // variadic arguments
                num_results: -1,  // variadic results
                num_regions: 0,
                traits: vec![],
            },
        )?;

        Ok(())
    }

    fn verify_operation(&self, op: &CustomOp) -> Result<()> {
        // Verify HUGR-specific constraints
        match op.name() {
            name if is_hugr_named_gate(name) || matches!(name, "rx" | "ry" | "rz") => {
                // Gate operand/result counts are declared by the operation definition.
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn get_operation_traits(&self, op_name: &str) -> Vec<OpTrait> {
        match op_name {
            name if is_hugr_named_gate(name) || matches!(name, "rx" | "ry" | "rz") => {
                vec![OpTrait::NoSideEffect]
            }
            "funcdefn" => vec![OpTrait::FunctionLike],
            "conditional" => vec![OpTrait::RegionBranch],
            _ => vec![],
        }
    }
}

/// Register the HUGR dialect
///
/// # Errors
/// Returns an error if the dialect cannot be registered with the registry.
pub fn register_dialect(registry: &mut DialectRegistry) -> Result<()> {
    let dialect = HugrDialect;
    registry.register_dialect(dialect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn converter_named_gates_are_registered_and_validate() {
        let mut registry = DialectRegistry::new();
        register_dialect(&mut registry).unwrap();

        for &(name, _, num_operands, num_results) in HUGR_NAMED_GATE_DEFS {
            assert!(is_hugr_named_gate(name));
            let definition = registry
                .get_operation("hugr", name)
                .unwrap_or_else(|| panic!("converter accepts unregistered hugr.{name}"));
            assert_eq!(definition.num_operands, num_operands);
            assert_eq!(definition.num_results, num_results);
            assert_eq!(definition.traits, [OpTrait::NoSideEffect]);

            let operation = CustomOp::new("hugr", name, vec![], BTreeMap::new());
            registry.verify_custom_operation(&operation).unwrap();
        }
    }
}
