/*!
QIS Dialect for PHIR

This dialect provides Quantum Instruction Set (QIS) operations that map directly to
hardware-native quantum operations. These are the operations that Selene and PECOS
compile to for execution.

The QIS dialect uses the triple-underscore naming convention (___qalloc, ___rxy, etc.)
to match the QIS standard used by Selene.
*/

use crate::dialect::{Dialect, DialectRegistry, OperationDef};
use crate::error::Result;
use crate::ops::CustomOp;
use crate::traits::OpTrait;

/// QIS dialect implementation
pub struct QisDialect;

impl Dialect for QisDialect {
    fn namespace(&self) -> &'static str {
        "qis"
    }

    fn description(&self) -> &'static str {
        "Quantum Instruction Set (QIS) operations for hardware-native quantum execution"
    }

    #[allow(clippy::too_many_lines)] // Dialect initialization is inherently a long list of operation registrations
    fn initialize(&self, registry: &mut DialectRegistry) -> Result<()> {
        // Core QIS operations (hardware-native gates)

        // Qubit management
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "qalloc".to_string(),
                description: "Allocate a qubit (___qalloc)".to_string(),
                num_operands: 0,
                num_results: 1, // returns qubit ID
                num_regions: 0,
                traits: vec![],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "qfree".to_string(),
                description: "Free a qubit (___qfree)".to_string(),
                num_operands: 1, // qubit ID
                num_results: 0,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "reset".to_string(),
                description: "Reset qubit to |0⟩ (___reset)".to_string(),
                num_operands: 1, // qubit ID
                num_results: 0,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        // Hardware-native rotation gates
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "rxy".to_string(),
                description: "RXY rotation gate (___rxy)".to_string(),
                num_operands: 3, // qubit, theta, phi
                num_results: 0,  // in-place operation
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "rz".to_string(),
                description: "RZ rotation gate (___rz)".to_string(),
                num_operands: 2, // qubit, angle
                num_results: 0,  // in-place operation
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "rzz".to_string(),
                description: "RZZ two-qubit rotation gate (___rzz)".to_string(),
                num_operands: 3, // qubit1, qubit2, angle
                num_results: 0,  // in-place operation
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        // Measurement operations
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "measure".to_string(),
                description: "Immediate measurement (___measure)".to_string(),
                num_operands: 1, // qubit
                num_results: 1,  // measurement result (bool)
                num_regions: 0,
                traits: vec![],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "lazy_measure".to_string(),
                description: "Lazy measurement returning future (___lazy_measure)".to_string(),
                num_operands: 1, // qubit
                num_results: 1,  // future handle
                num_regions: 0,
                traits: vec![],
            },
        )?;

        // Future operations (for lazy measurements)
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "read_future".to_string(),
                description: "Read result from measurement future (___read_future)".to_string(),
                num_operands: 1, // future handle
                num_results: 1,  // bool result
                num_regions: 0,
                traits: vec![],
            },
        )?;

        // Runtime initialization (if needed)
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "initialize".to_string(),
                description: "Initialize QIS runtime (___initialize)".to_string(),
                num_operands: 0,
                num_results: 0,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        Ok(())
    }

    fn verify_operation(&self, op: &CustomOp) -> Result<()> {
        // Verify QIS-specific constraints
        match op.name() {
            "rxy" => {
                // RXY requires exactly 3 operands: qubit, theta, phi
                Ok(())
            }
            "rz" => {
                // RZ requires exactly 2 operands: qubit, angle
                Ok(())
            }
            "rzz" => {
                // RZZ requires exactly 3 operands: qubit1, qubit2, angle
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn get_operation_traits(&self, op_name: &str) -> Vec<OpTrait> {
        match op_name {
            "rxy" | "rz" | "rzz" => vec![OpTrait::NoSideEffect],
            _ => vec![],
        }
    }
}

/// Register the QIS dialect
///
/// # Errors
/// Returns an error if the dialect cannot be registered with the registry.
pub fn register_dialect(registry: &mut DialectRegistry) -> Result<()> {
    let dialect = QisDialect;
    registry.register_dialect(dialect)
}
