//! Classical name resolution shared by parsing and AST consumers.

use pecos_core::errors::PecosError;

use crate::ast::{Expression, Operation};
use crate::parser::Program;
use crate::parser::errors::{undeclared_classical_bit, unknown_register};

impl Program {
    /// Validate classical references, including unused gate definitions and
    /// operations inside conditions that may never execute.
    ///
    /// # Errors
    /// Returns an error naming any undeclared classical register or bit.
    pub fn validate_classical_references(&self) -> Result<(), PecosError> {
        for definition in self.gate_definitions.values() {
            for operation in &definition.body {
                for expression in &operation.params {
                    self.validate_expression(expression, &definition.params)?;
                }
            }
        }
        for operation in &self.operations {
            self.validate_classical_operation(operation)?;
        }
        Ok(())
    }

    fn validate_classical_reference(
        &self,
        name: &str,
        index: Option<usize>,
    ) -> Result<(), PecosError> {
        let size = self
            .classical_registers
            .get(name)
            .ok_or_else(|| unknown_register("classical", name))?;
        if let Some(index) = index
            && index >= *size
        {
            return Err(undeclared_classical_bit(name, index, *size));
        }
        Ok(())
    }

    pub(super) fn validate_expression(
        &self,
        expression: &Expression,
        parameters: &[String],
    ) -> Result<(), PecosError> {
        match expression {
            Expression::Variable(name) if parameters.contains(name) => Ok(()),
            Expression::Variable(name) => self.validate_classical_reference(name, None),
            Expression::BitId(name, index) => self.validate_classical_reference(name, Some(*index)),
            Expression::BinaryOp { left, right, .. } => {
                self.validate_expression(left, parameters)?;
                self.validate_expression(right, parameters)
            }
            Expression::UnaryOp { expr, .. } => self.validate_expression(expr, parameters),
            Expression::FunctionCall { args, .. } => {
                for argument in args {
                    self.validate_expression(argument, parameters)?;
                }
                Ok(())
            }
            Expression::Integer(_) | Expression::Float(_) | Expression::Pi => Ok(()),
        }
    }

    fn validate_classical_operation(&self, operation: &Operation) -> Result<(), PecosError> {
        match operation {
            Operation::ClassicalAssignment {
                target,
                index,
                expression,
                ..
            } => {
                self.validate_classical_reference(target, *index)?;
                self.validate_expression(expression, &[])
            }
            Operation::MeasureWithMapping { c_reg, c_index, .. } => {
                self.validate_classical_reference(c_reg, Some(*c_index))
            }
            Operation::RegMeasure { c_reg, .. } => self.validate_classical_reference(c_reg, None),
            Operation::If {
                condition,
                operation,
            } => {
                self.validate_expression(condition, &[])?;
                self.validate_classical_operation(operation)
            }
            Operation::VoidFunctionCall { expression } => self.validate_expression(expression, &[]),
            // Opaque parameters and quantum arguments are declarations, not values.
            Operation::OpaqueGate { .. }
            | Operation::Gate { .. }
            | Operation::NativeGate(_)
            | Operation::Barrier { .. } => Ok(()),
        }
    }
}
