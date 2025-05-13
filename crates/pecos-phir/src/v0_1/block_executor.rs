use pecos_core::{errors::PecosError, QubitId};
use pecos_engines::byte_message::quantum_command::QuantumCommand;
use pecos_engines::core::result_id::ResultId;

use crate::v0_1::ast::{Operation, QubitArg};
use crate::v0_1::environment::Environment;
use crate::v0_1::expression::ExpressionEvaluator;

/// Represents a block of operations in PHIR
#[derive(Debug, Clone)]
pub enum Block {
    /// A sequence of operations to be executed sequentially
    Sequence(Vec<Operation>),
    
    /// A conditional block with a condition, true branch, and optional false branch
    Conditional {
        /// Condition expression
        condition: crate::v0_1::ast::Expression,
        /// Operations to execute if condition is true
        true_branch: Vec<Operation>,
        /// Optional operations to execute if condition is false
        false_branch: Option<Vec<Operation>>,
    },
    
    /// A parallel block of quantum operations
    Parallel(Vec<Operation>),
    
    /// A single operation
    Single(Operation),
}

/// Handles execution of operation blocks
pub struct BlockExecutor<'a> {
    /// Environment for variable access
    environment: &'a mut Environment,
    /// Current operation index for tracking progress
    current_index: usize,
    /// Measurement mappings (result_id -> variable name)
    measurement_mappings: Vec<(u64, String)>,
    /// Operations that produced quantum commands
    quantum_ops: Vec<usize>,
    /// Exported values (for Result operations)
    exported_values: std::collections::HashMap<String, u64>,
}

impl<'a> BlockExecutor<'a> {
    /// Creates a new block executor with the given environment
    pub fn new(environment: &'a mut Environment) -> Self {
        Self {
            environment,
            current_index: 0,
            measurement_mappings: Vec::new(),
            quantum_ops: Vec::new(),
            exported_values: std::collections::HashMap::new(),
        }
    }

    /// Processes a block of operations and returns generated quantum commands
    pub fn process_block(&mut self, block: &Block) -> Result<Vec<QuantumCommand>, PecosError> {
        match block {
            Block::Sequence(operations) => self.process_sequence(operations),
            Block::Conditional { condition, true_branch, false_branch } => {
                self.process_conditional(condition, true_branch, false_branch)
            },
            Block::Parallel(operations) => self.process_parallel(operations),
            Block::Single(operation) => self.process_operation(operation),
        }
    }

    /// Processes a sequence of operations
    fn process_sequence(&mut self, operations: &[Operation]) -> Result<Vec<QuantumCommand>, PecosError> {
        let mut commands = Vec::new();
        
        for (index, op) in operations.iter().enumerate() {
            self.current_index = index;
            let mut op_commands = self.process_operation(op)?;
            commands.append(&mut op_commands);
        }
        
        Ok(commands)
    }

    /// Processes a conditional block
    fn process_conditional(
        &mut self,
        condition: &crate::v0_1::ast::Expression,
        true_branch: &[Operation],
        false_branch: &Option<Vec<Operation>>,
    ) -> Result<Vec<QuantumCommand>, PecosError> {
        // Evaluate the condition
        let evaluator = ExpressionEvaluator::new(self.environment);
        let condition_value = evaluator.eval_expr(condition)?;
        
        if condition_value != 0 {
            // Execute true branch
            self.process_sequence(true_branch)
        } else if let Some(else_branch) = false_branch {
            // Execute false branch if available
            self.process_sequence(else_branch)
        } else {
            // No false branch, return empty commands
            Ok(Vec::new())
        }
    }

    /// Processes operations in parallel (for quantum operations)
    fn process_parallel(&mut self, operations: &[Operation]) -> Result<Vec<QuantumCommand>, PecosError> {
        let mut commands = Vec::new();
        
        // First validate that all operations are quantum operations
        for op in operations {
            match op {
                Operation::QuantumOp { .. } => {
                    // Quantum operations are allowed
                },
                _ => {
                    return Err(PecosError::Input(format!(
                        "Only quantum operations are allowed in parallel blocks, found: {:?}", op
                    )));
                }
            }
        }
        
        // Then process all operations
        for (index, op) in operations.iter().enumerate() {
            self.current_index = index;
            let mut op_commands = self.process_operation(op)?;
            commands.append(&mut op_commands);
        }
        
        Ok(commands)
    }

    /// Processes a single operation
    fn process_operation(&mut self, operation: &Operation) -> Result<Vec<QuantumCommand>, PecosError> {
        match operation {
            Operation::QuantumOp { qop, args, returns, angles, .. } => {
                // Process quantum operation
                let commands = self.process_quantum_op(qop, args, &returns, angles)?;
                self.quantum_ops.push(self.current_index);
                Ok(commands)
            },
            Operation::ClassicalOp { cop, args, returns, .. } => {
                // Process classical operation
                self.process_classical_op(cop, args, &returns)?;
                Ok(Vec::new()) // Classical operations don't generate quantum commands
            },
            Operation::Block { block, ops, condition, true_branch, false_branch, .. } => {
                // Process block operation
                match block.as_str() {
                    "sequence" => {
                        self.process_sequence(ops)
                    },
                    "qparallel" => {
                        self.process_parallel(ops)
                    },
                    "if" => {
                        if let (Some(cond), Some(true_br)) = (condition, true_branch) {
                            self.process_conditional(cond, true_br, false_branch)
                        } else {
                            Err(PecosError::Input(
                                "If block missing required condition or true_branch".into()
                            ))
                        }
                    },
                    _ => Err(PecosError::Input(format!(
                        "Unsupported block type: {}", block
                    ))),
                }
            },
            Operation::VariableDefinition { .. } => {
                // Variable definitions are handled separately during initialization
                Ok(Vec::new())
            },
            Operation::MachineOp { .. } => {
                // Machine operations are not implemented yet
                Err(PecosError::Input("Machine operations not implemented".into()))
            },
            Operation::MetaInstruction { .. } => {
                // Meta instructions are not implemented yet
                Ok(Vec::new()) // For now, treat as no-ops
            },
            Operation::Comment { .. } => {
                // Comments don't generate any commands
                Ok(Vec::new())
            },
        }
    }

    /// Processes a quantum operation
    fn process_quantum_op(
        &mut self,
        qop: &str,
        _args: &[QubitArg],
        returns: &Vec<(String, usize)>,
        _angles: &Option<Vec<f64>>,
    ) -> Result<Vec<QuantumCommand>, PecosError> {
        // This is a placeholder for actual quantum operation processing
        // In a real implementation, this would create the appropriate QuantumCommand
        // based on the operation type, arguments, etc.

        // Create a simple placeholder command - in a real implementation this would
        // map to specific gate types based on the operation name
        let command = match qop {
            "H" => QuantumCommand::H(QubitId(0)),
            "CNOT" => QuantumCommand::CX(QubitId(0), QubitId(1)),
            "Measure" => QuantumCommand::Measure(QubitId(0), ResultId(0)),
            _ => return Ok(vec![]), // Skip unsupported operations for now
        };
        
        // Handle measurement operations
        if qop == "Measure" || qop == "measure Z" || qop == "Measure +Z" {
            if !returns.is_empty() {
                // Map measurement result to variable
                let result_id = self.current_index as u64; // Use op index as result ID
                let (var_name, _) = &returns[0];

                // Store mapping for later use
                self.measurement_mappings.push((result_id, var_name.clone()));
            }
        }
        
        Ok(vec![command])
    }

    /// Processes a classical operation
    fn process_classical_op(
        &mut self,
        cop: &str,
        args: &[crate::v0_1::ast::ArgItem],
        returns: &Vec<crate::v0_1::ast::ArgItem>,
    ) -> Result<(), PecosError> {
        let evaluator = ExpressionEvaluator::new(self.environment);
        
        match cop {
            "=" => {
                // Assignment operation
                // Evaluate arguments
                let mut values = Vec::new();
                for arg in args {
                    values.push(evaluator.eval_arg(arg)?);
                }

                // Assign to return variables
                for (i, ret_var) in returns.iter().enumerate() {
                    if i < values.len() {
                        match ret_var {
                            crate::v0_1::ast::ArgItem::Simple(name) => {
                                self.environment.set(name, values[i])?;
                            },
                            crate::v0_1::ast::ArgItem::Indexed((name, idx)) => {
                                self.environment.set_bit(name, *idx, values[i])?;
                            },
                            _ => {
                                return Err(PecosError::Input(format!(
                                    "Invalid assignment target: {:?}", ret_var
                                )));
                            }
                        }
                    }
                }
                Ok(())
            },
            "Result" => {
                // Result operation (exports values)
                self.process_result_op(args, returns)?;
                Ok(())
            },
            _ => {
                Err(PecosError::Input(format!(
                    "Unsupported classical operation: {}", cop
                )))
            }
        }
    }

    /// Processes a Result operation (for variable export)
    fn process_result_op(
        &mut self,
        args: &[crate::v0_1::ast::ArgItem],
        returns: &Vec<crate::v0_1::ast::ArgItem>,
    ) -> Result<(), PecosError> {
        let _evaluator = ExpressionEvaluator::new(self.environment);

        for (i, src) in args.iter().enumerate() {
            if i < returns.len() {
                let dst = &returns[i];
                // Extract source variable name
                let src_name = match src {
                    crate::v0_1::ast::ArgItem::Simple(name) => name.clone(),
                    crate::v0_1::ast::ArgItem::Indexed((name, _)) => name.clone(),
                    _ => {
                        return Err(PecosError::Input(format!(
                            "Invalid Result source: {:?}", src
                        )));
                    }
                };

                // Extract destination variable name
                let dst_name = match dst {
                    crate::v0_1::ast::ArgItem::Simple(name) => name.clone(),
                    crate::v0_1::ast::ArgItem::Indexed((name, _)) => name.clone(),
                    _ => {
                        return Err(PecosError::Input(format!(
                            "Invalid Result destination: {:?}", dst
                        )));
                    }
                };

                // Get source value
                let src_value = self.environment.get(&src_name)
                    .ok_or_else(|| PecosError::Input(format!(
                        "Source variable not found: {}", src_name
                    )))?;

                // If destination doesn't exist, create it with same type as source
                if !self.environment.has_variable(&dst_name) {
                    let src_info = self.environment.get_variable_info(&src_name)?;
                    self.environment.add_variable(
                        &dst_name,
                        src_info.data_type.clone(),
                        src_info.size
                    )?;
                }

                // Set destination value
                self.environment.set(&dst_name, src_value)?;

                // Add to exported values
                self.exported_values.insert(dst_name, src_value);
            }
        }

        Ok(())
    }

    /// Gets the measurement mappings
    pub fn get_measurement_mappings(&self) -> &[(u64, String)] {
        &self.measurement_mappings
    }

    /// Gets the exported values
    pub fn get_exported_values(&self) -> &std::collections::HashMap<String, u64> {
        &self.exported_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::ast::{ArgItem, Expression};
    use crate::v0_1::environment::{Environment, DataType};

    #[test]
    fn test_sequence_execution() {
        let mut env = Environment::new();
        env.add_variable("x", DataType::I32, 32).unwrap();

        let operations = vec![
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(42)],
                returns: vec![ArgItem::Simple("x".to_string())],
                metadata: None,
                function: None,
            },
        ];

        {
            let mut executor = BlockExecutor::new(&mut env);
            let commands = executor.process_sequence(&operations).unwrap();

            // Sequence should execute without errors
            assert_eq!(commands.len(), 0); // No quantum commands generated
        }

        // After executor goes out of scope, we can access env directly
        assert_eq!(env.get("x"), Some(42)); // Variable should be updated
    }

    #[test]
    fn test_conditional_execution() {
        let mut env = Environment::new();
        env.add_variable("condition", DataType::I32, 32).unwrap();
        env.add_variable("result", DataType::I32, 32).unwrap();

        let true_branch = vec![
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(1)],
                returns: vec![ArgItem::Simple("result".to_string())],
                metadata: None,
                function: None,
            },
        ];

        let false_branch = vec![
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(0)],
                returns: vec![ArgItem::Simple("result".to_string())],
                metadata: None,
                function: None,
            },
        ];

        // Test with true condition
        env.set("condition", 1).unwrap(); // true condition
        {
            let mut executor = BlockExecutor::new(&mut env);
            let condition = Expression::Variable("condition".to_string());
            executor.process_conditional(&condition, &true_branch, &Some(false_branch.clone())).unwrap();
        }
        assert_eq!(env.get("result"), Some(1)); // True branch executed

        // Test with false condition
        env.set("condition", 0).unwrap(); // false condition
        {
            let mut executor = BlockExecutor::new(&mut env);
            let condition = Expression::Variable("condition".to_string());
            executor.process_conditional(&condition, &true_branch, &Some(false_branch)).unwrap();
        }
        assert_eq!(env.get("result"), Some(0)); // False branch executed
    }

    #[test]
    fn test_result_operation() {
        let mut env = Environment::new();
        env.add_variable("internal", DataType::I32, 32).unwrap();
        env.set("internal", 42).unwrap();

        let operations = vec![
            Operation::ClassicalOp {
                cop: "Result".to_string(),
                args: vec![ArgItem::Simple("internal".to_string())],
                returns: vec![ArgItem::Simple("output".to_string())],
                metadata: None,
                function: None,
            },
        ];

        let exported_values = {
            let mut executor = BlockExecutor::new(&mut env);
            executor.process_sequence(&operations).unwrap();
            // Clone exported values before executor is dropped
            executor.get_exported_values().clone()
        };

        // Result operation should create a new variable
        assert!(env.has_variable("output"));
        assert_eq!(env.get("output"), Some(42));

        // The value should be in exported_values
        assert_eq!(exported_values.get("output"), Some(&42));
    }
}