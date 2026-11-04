//! Classical and gate-parameter name resolution shared by parsing and AST consumers.

use pecos_core::errors::PecosError;

use crate::ast::{Expression, GateDefinition, Operation};
use crate::parser::Program;
use crate::parser::errors::{
    classical_bit_out_of_bounds, classical_register_shrunk, invalid_operation,
    unknown_classical_register,
};

impl Program {
    /// Validate classical references and gate parameter scopes, including unused
    /// gate definitions and operations inside conditions that may never execute.
    ///
    /// # Errors
    /// Returns an error naming an undeclared classical reference, a bit outside its
    /// register, or an invalid gate parameter.
    pub fn validate_classical_references(&self) -> Result<(), PecosError> {
        self.validate_classical_references_with_bounds_error(classical_bit_out_of_bounds)
    }

    pub(super) fn validate_classical_references_after_parsing(&self) -> Result<(), PecosError> {
        // Each statement already checked its current width. This final pass catches
        // later redeclarations that shrink the storage used by earlier operations.
        self.validate_classical_references_with_bounds_error(classical_register_shrunk)
    }

    fn validate_classical_references_with_bounds_error(
        &self,
        bounds_error: fn(&str, usize, usize) -> PecosError,
    ) -> Result<(), PecosError> {
        for definition in self.gate_definitions.values() {
            definition.validate_parameter_references()?;
        }
        for operation in &self.operations {
            self.validate_classical_operation_with_bounds_error(operation, bounds_error)?;
        }
        Ok(())
    }

    fn validate_classical_reference(
        &self,
        name: &str,
        index: Option<usize>,
        bounds_error: fn(&str, usize, usize) -> PecosError,
    ) -> Result<(), PecosError> {
        let size = self
            .classical_registers
            .get(name)
            .ok_or_else(|| unknown_classical_register(name))?;
        if let Some(index) = index
            && index >= *size
        {
            return Err(bounds_error(name, index, *size));
        }
        Ok(())
    }

    fn validate_classical_expression(
        &self,
        expression: &Expression,
        bounds_error: fn(&str, usize, usize) -> PecosError,
    ) -> Result<(), PecosError> {
        match expression {
            Expression::Variable(name) => {
                self.validate_classical_reference(name, None, bounds_error)
            }
            Expression::BitId(name, index) => {
                self.validate_classical_reference(name, Some(*index), bounds_error)
            }
            Expression::BinaryOp { left, right, .. } => {
                self.validate_classical_expression(left, bounds_error)?;
                self.validate_classical_expression(right, bounds_error)
            }
            Expression::UnaryOp { expr, .. } => {
                self.validate_classical_expression(expr, bounds_error)
            }
            Expression::FunctionCall { args, .. } => {
                for argument in args {
                    self.validate_classical_expression(argument, bounds_error)?;
                }
                Ok(())
            }
            Expression::Integer(_) | Expression::Float(_) | Expression::Pi => Ok(()),
        }
    }

    pub(super) fn validate_classical_operation(
        &self,
        operation: &Operation,
    ) -> Result<(), PecosError> {
        self.validate_classical_operation_with_bounds_error(operation, classical_bit_out_of_bounds)
    }

    fn validate_classical_operation_with_bounds_error(
        &self,
        operation: &Operation,
        bounds_error: fn(&str, usize, usize) -> PecosError,
    ) -> Result<(), PecosError> {
        match operation {
            Operation::ClassicalAssignment {
                target,
                index,
                expression,
                ..
            } => {
                self.validate_classical_reference(target, *index, bounds_error)?;
                self.validate_classical_expression(expression, bounds_error)
            }
            Operation::MeasureWithMapping { c_reg, c_index, .. } => {
                self.validate_classical_reference(c_reg, Some(*c_index), bounds_error)
            }
            Operation::RegMeasure { c_reg, .. } => {
                self.validate_classical_reference(c_reg, None, bounds_error)
            }
            Operation::If {
                condition,
                operation,
            } => {
                self.validate_classical_expression(condition, bounds_error)?;
                self.validate_classical_operation_with_bounds_error(operation, bounds_error)
            }
            Operation::VoidFunctionCall { expression } => {
                self.validate_classical_expression(expression, bounds_error)
            }
            // Opaque parameters and quantum arguments are declarations, not values.
            Operation::OpaqueGate { .. }
            | Operation::Gate { .. }
            | Operation::NativeGate(_)
            | Operation::Barrier { .. } => Ok(()),
        }
    }
}

impl GateDefinition {
    /// Validate parameter expressions in this gate's own scope, including unused gates.
    pub(crate) fn validate_parameter_references(&self) -> Result<(), PecosError> {
        for operation in &self.body {
            for expression in &operation.params {
                self.validate_parameter_expression(expression)?;
            }
        }
        Ok(())
    }

    fn validate_parameter_expression(&self, expression: &Expression) -> Result<(), PecosError> {
        match expression {
            Expression::Variable(name) => {
                if self.params.contains(name) {
                    Ok(())
                } else {
                    Err(invalid_operation(format!(
                        "Gate '{}' references unknown parameter '{name}'; declared parameters: [{}]. \
                         Add '{name}' to the gate's parameter list or use a declared parameter",
                        self.name,
                        self.params.join(", ")
                    )))
                }
            }
            Expression::BitId(name, index) => Err(invalid_operation(format!(
                "Gate '{}': bit references are not allowed in gate bodies (found '{name}[{index}]'). \
                     Use a gate parameter or constant expression",
                self.name
            ))),
            Expression::BinaryOp { left, right, .. } => {
                self.validate_parameter_expression(left)?;
                self.validate_parameter_expression(right)
            }
            Expression::UnaryOp { expr, .. } => self.validate_parameter_expression(expr),
            Expression::FunctionCall { args, .. } => {
                for argument in args {
                    self.validate_parameter_expression(argument)?;
                }
                Ok(())
            }
            Expression::Integer(_) | Expression::Float(_) | Expression::Pi => Ok(()),
        }
    }
}
