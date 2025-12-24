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
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "h".to_string(),
                description: "Hadamard gate".to_string(),
                num_operands: 1,
                num_results: 1,
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "cx".to_string(),
                description: "Controlled-X (CNOT) gate".to_string(),
                num_operands: 2,
                num_results: 2,
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

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
            "h" | "rx" | "ry" | "rz" => {
                // Single qubit gates should have correct operand/result counts
                // This is handled by the operation definition
                Ok(())
            }
            "cx" => {
                // Two-qubit gate constraints
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn get_operation_traits(&self, op_name: &str) -> Vec<OpTrait> {
        match op_name {
            "h" | "rx" | "ry" | "rz" | "cx" => vec![OpTrait::NoSideEffect],
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
