use crate::v0_1::ast::{Expression, Operation, QubitArg};
use crate::v0_1::environment::Environment;
use crate::v0_1::expression::ExpressionEvaluator;
use crate::v0_1::foreign_objects::ForeignObject;
use crate::v0_1::operations::OperationProcessor;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use std::collections::{BTreeMap, BTreeSet};

/// Block executor for processing and executing blocks of operations in PHIR programs.
/// The `BlockExecutor` manages:
/// 1. Execution flow through different block types (sequence, conditional, parallel)
/// 2. Operation processing and execution
/// 3. Quantum and classical operation handling
/// 4. Measurement result processing
pub struct BlockExecutor {
    /// The operation processor for handling individual operations
    pub processor: OperationProcessor,
    /// Tracks the current byte message builder for collecting quantum operations
    pub builder: Option<ByteMessageBuilder>,
}

impl BlockExecutor {
    /// Creates a new block executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            processor: OperationProcessor::new(),
            builder: None,
        }
    }

    /// Creates a new block executor with a foreign object
    #[must_use]
    pub fn with_foreign_object(foreign_object: Box<dyn ForeignObject>) -> Self {
        Self {
            processor: OperationProcessor::with_foreign_object(foreign_object),
            builder: None,
        }
    }

    /// Resets the block executor state
    pub fn reset(&mut self) {
        self.processor.reset();
        if let Some(builder) = &mut self.builder {
            builder.clear();
        }
    }

    /// Sets the foreign object for the processor
    pub fn set_foreign_object(&mut self, foreign_object: Box<dyn ForeignObject>) {
        self.processor.set_foreign_object(foreign_object);
    }

    /// Gets a reference to the environment from the processor
    #[must_use]
    pub fn get_environment(&self) -> &Environment {
        &self.processor.environment
    }

    /// Gets a mutable reference to the environment
    pub fn get_environment_mut(&mut self) -> &mut Environment {
        &mut self.processor.environment
    }

    /// Gets the operation processor for direct access
    #[must_use]
    pub fn get_processor(&self) -> &OperationProcessor {
        &self.processor
    }

    /// Gets a mutable reference to the operation processor for direct access
    pub fn get_processor_mut(&mut self) -> &mut OperationProcessor {
        &mut self.processor
    }

    /// Add a quantum variable to the processor
    ///
    /// # Errors
    /// Returns an error if the variable already exists or cannot be added.
    pub fn add_quantum_variable(&mut self, variable: &str, size: usize) -> Result<(), PecosError> {
        self.processor.add_quantum_variable(variable, size)
    }

    /// Add a classical variable to the processor
    ///
    /// # Errors
    /// Returns an error if the data type is invalid or the variable already exists.
    pub fn add_classical_variable(
        &mut self,
        variable: &str,
        data_type: &str,
        size: usize,
    ) -> Result<(), PecosError> {
        self.processor
            .add_classical_variable(variable, data_type, size)
    }

    /// Sets the byte message builder
    pub fn set_builder(&mut self, builder: ByteMessageBuilder) {
        self.builder = Some(builder);
    }

    /// Gets the current byte message builder or creates a new one
    ///
    /// # Panics
    ///
    /// This function will not panic under normal circumstances as it creates a builder
    /// if none exists. However, it could theoretically panic if memory allocation fails
    /// when creating a new builder.
    pub fn get_builder(&mut self) -> &mut ByteMessageBuilder {
        if self.builder.is_none() {
            self.builder = Some(ByteMessageBuilder::new());
        }
        self.builder.as_mut().expect("builder is initialized above")
    }

    /// Handle variable definition operations
    ///
    /// # Errors
    /// Returns an error if the variable definition type is unknown.
    pub fn handle_variable_definition(
        &mut self,
        data: &str,
        data_type: &str,
        variable: &str,
        size: usize,
    ) -> Result<(), PecosError> {
        self.processor
            .handle_variable_definition(data, data_type, variable, size)
    }

    /// Processes a single operation
    ///
    /// # Errors
    /// Returns an error if the operation cannot be processed.
    pub fn process_operation(&mut self, op: &Operation) -> Result<(), PecosError> {
        match op {
            Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } => {
                debug!("Processing variable definition: {data_type} {variable}");
                self.processor
                    .handle_variable_definition(data, data_type, variable, *size)?;
            }
            Operation::QuantumOp {
                qop, angles, args, ..
            } => {
                debug!("Processing quantum operation: {qop}");
                let (gate_type, qubit_args, angle_args) =
                    self.processor
                        .process_quantum_op(qop, angles.as_ref(), args)?;

                // Add to byte message builder if we have one
                if let Some(builder) = &mut self.builder {
                    self.processor.add_quantum_operation_to_builder(
                        builder,
                        &gate_type,
                        &qubit_args,
                        &angle_args,
                    )?;
                }
            }
            Operation::ClassicalOp {
                cop, args, returns, ..
            } => {
                debug!("Processing classical operation: {cop}");
                let result = self.processor.handle_classical_op(
                    cop,
                    args,
                    returns,
                    std::slice::from_ref(op),
                    0,
                )?;
                if !result {
                    debug!("Classical operation handled as expression or skipped: {cop}");
                }
            }
            Operation::MachineOp {
                mop,
                args,
                duration,
                metadata,
                ..
            } => {
                debug!("Processing machine operation: {mop}");
                let mop_result = self.processor.process_machine_op(
                    mop,
                    args.as_ref(),
                    duration.as_ref(),
                    metadata.as_ref(),
                )?;

                // Add to byte message builder if we have one
                if let Some(builder) = &mut self.builder {
                    self.processor
                        .add_machine_operation_to_builder(builder, &mop_result)?;
                }
            }
            Operation::MetaInstruction { meta, args, .. } => {
                debug!("Processing meta instruction: {meta}");
                let meta_result = self.processor.process_meta_instruction(meta, args)?;

                // Add to byte message builder if we have one
                if let Some(builder) = &mut self.builder {
                    self.processor
                        .add_meta_instruction_to_builder(builder, &meta_result)?;
                }
            }
            Operation::Block { .. } => {
                // Process nested blocks
                self.process_block_operation(op)?;
            }
            Operation::Comment { comment } => {
                debug!("Skipping comment: {comment}");
                // Comments are no-ops
            }
        }

        Ok(())
    }

    /// Executes a block of operations in sequence (previously `execute_block`)
    ///
    /// # Errors
    /// Returns an error if any operation in the sequence fails.
    pub fn execute_sequence(&mut self, operations: &[Operation]) -> Result<(), PecosError> {
        debug!(
            "Executing sequence block with {} operations",
            operations.len()
        );

        for op in operations {
            self.process_operation(op)?;
        }

        Ok(())
    }

    /// Evaluates a conditional expression using the environment
    ///
    /// # Errors
    /// Returns an error if the expression cannot be evaluated.
    pub fn evaluate_condition(&self, condition: &Expression) -> Result<bool, PecosError> {
        debug!("Evaluating condition: {condition:?}");

        // Create an evaluator with the current environment
        let mut evaluator = ExpressionEvaluator::new(&self.processor.environment);

        // Evaluate the condition
        let result = evaluator.eval_expr(condition)?;

        // Convert to boolean
        Ok(result.as_bool())
    }

    /// Executes a conditional (if/else) block
    ///
    /// # Errors
    /// Returns an error if the condition cannot be evaluated or branch execution fails.
    pub fn execute_conditional(
        &mut self,
        condition: &Expression,
        true_branch: &[Operation],
        false_branch: Option<&[Operation]>,
    ) -> Result<(), PecosError> {
        debug!("Executing conditional block");

        // Evaluate the condition
        let condition_result = self.evaluate_condition(condition)?;
        debug!("Condition evaluated to: {condition_result}");

        if condition_result {
            // Execute the true branch
            debug!(
                "Executing true branch with {} operations",
                true_branch.len()
            );
            self.execute_sequence(true_branch)?;
        } else if let Some(branch) = false_branch {
            // Execute the false branch
            debug!("Executing false branch with {} operations", branch.len());
            self.execute_sequence(branch)?;
        } else {
            debug!("Condition is false and no false branch exists");
        }

        Ok(())
    }

    /// Executes a quantum parallel block
    ///
    /// # Errors
    /// Returns an error if:
    /// - Non-quantum operations are found in the block
    /// - A qubit is used more than once
    /// - Any operation fails to process
    pub fn execute_qparallel(&mut self, operations: &[Operation]) -> Result<(), PecosError> {
        debug!(
            "Executing quantum parallel block with {} operations",
            operations.len()
        );

        // Verify all operations are quantum operations or meta instructions
        for op in operations {
            match op {
                Operation::QuantumOp { .. } | Operation::MetaInstruction { .. } => {
                    // These are allowed in qparallel
                }
                _ => {
                    return Err(PecosError::Input(format!(
                        "Invalid operation in qparallel block: {op:?}"
                    )));
                }
            }
        }

        // Verify no qubit is used more than once
        let mut used_qubits = BTreeSet::new();

        for op in operations {
            if let Operation::QuantumOp { args, .. } = op {
                for qubit_arg in args {
                    match qubit_arg {
                        QubitArg::SingleQubit((var, idx)) => {
                            let qubit_id = format!("{var}_{idx}");
                            if !used_qubits.insert(qubit_id) {
                                return Err(PecosError::Input(format!(
                                    "Qubit {var}[{idx}] used more than once in qparallel block"
                                )));
                            }
                        }
                        QubitArg::MultipleQubits(qubits) => {
                            for (var, idx) in qubits {
                                let qubit_id = format!("{var}_{idx}");
                                if !used_qubits.insert(qubit_id) {
                                    return Err(PecosError::Input(format!(
                                        "Qubit {var}[{idx}] used more than once in qparallel block"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Now execute all operations in the block
        for op in operations {
            self.process_operation(op)?;
        }

        Ok(())
    }

    /// Process a block operation (sequence, qparallel, conditional)
    ///
    /// # Errors
    /// Returns an error if:
    /// - The block type is unknown
    /// - Required block components are missing
    /// - Block execution fails
    fn process_block_operation(&mut self, op: &Operation) -> Result<(), PecosError> {
        if let Operation::Block {
            block,
            ops,
            condition,
            true_branch,
            false_branch,
            ..
        } = op
        {
            match block.as_str() {
                "sequence" => {
                    debug!("Processing sequence block with {} operations", ops.len());
                    self.execute_sequence(ops)?;
                }
                "qparallel" => {
                    debug!("Processing qparallel block with {} operations", ops.len());
                    self.execute_qparallel(ops)?;
                }
                "if" => {
                    debug!("Processing conditional block");
                    if let Some(cond) = condition {
                        if let Some(true_ops) = true_branch {
                            self.execute_conditional(cond, true_ops, false_branch.as_deref())?;
                        } else {
                            return Err(PecosError::Input(
                                "Conditional block missing true branch".to_string(),
                            ));
                        }
                    } else {
                        return Err(PecosError::Input(
                            "Conditional block missing condition".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(PecosError::Input(format!("Unknown block type: {block}")));
                }
            }
        } else {
            return Err(PecosError::Input("Expected block operation".to_string()));
        }

        Ok(())
    }

    /// Process a block with the appropriate handler (wraps `process_block_operation`)
    ///
    /// # Errors
    /// Returns an error if the block type is unknown or block execution fails.
    pub fn process_block(
        &mut self,
        block_type: &str,
        operations: &[Operation],
        condition: Option<&Expression>,
        true_branch: Option<&[Operation]>,
        false_branch: Option<&[Operation]>,
    ) -> Result<(), PecosError> {
        match block_type {
            "sequence" => {
                self.execute_sequence(operations)?;
            }
            "qparallel" => {
                self.execute_qparallel(operations)?;
            }
            "if" => {
                if let Some(condition) = condition {
                    self.execute_conditional(condition, true_branch.unwrap_or(&[]), false_branch)?;
                } else {
                    return Err(PecosError::Input(
                        "Conditional block missing condition".to_string(),
                    ));
                }
            }
            _ => {
                return Err(PecosError::Input(format!(
                    "Unknown block type: {block_type}"
                )));
            }
        }

        Ok(())
    }

    /// Handles measurement results from the quantum backend
    ///
    /// # Errors
    /// Returns an error if variable creation or value setting fails.
    pub fn handle_measurements(
        &mut self,
        measurements: &[u32],
        ops: &[Operation],
    ) -> Result<(), PecosError> {
        self.processor.handle_measurements(measurements, ops)
    }

    /// Gets the measurement results from the processor
    #[must_use]
    pub fn get_measurement_results(&self) -> BTreeMap<String, u32> {
        self.processor.get_measurement_results()
    }

    /// Process export mappings to determine values to return from simulations
    #[must_use]
    pub fn process_export_mappings(&self) -> BTreeMap<String, u32> {
        self.processor.process_export_mappings()
    }

    /// Get mapped results for output (alias for `process_export_mappings`)
    #[must_use]
    pub fn get_mapped_results(&self) -> BTreeMap<String, u32> {
        self.processor.process_export_mappings()
    }

    /// Execute a complete PHIR program
    ///
    /// # Errors
    /// Returns an error if any operation in the program fails.
    pub fn execute_program(
        &mut self,
        program: &[Operation],
    ) -> Result<BTreeMap<String, u32>, PecosError> {
        debug!("Executing PHIR program with {} operations", program.len());

        // Reset state before execution
        self.reset();

        // Initialize a new builder if none exists
        if self.builder.is_none() {
            self.builder = Some(ByteMessageBuilder::new());
        }

        // Execute all operations in sequence
        self.execute_sequence(program)?;

        // Return the exported values
        Ok(self.process_export_mappings())
    }
}

impl Default for BlockExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::ast::{ArgItem, Operation};

    #[test]
    fn test_block_executor_basic() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_quantum_variable("q", 2).unwrap();
        executor.add_classical_variable("c", "i32", 32).unwrap();

        // Execute a simple assignment operation
        let op = Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(42)],
            returns: vec![ArgItem::Simple("c".to_string())],
            function: None,
            metadata: None,
        };

        let result = executor.process_operation(&op);
        assert!(result.is_ok());

        // Verify the value was set
        let env = executor.get_environment();
        assert_eq!(env.get_raw("c"), Some(42));
    }

    #[test]
    fn test_execute_conditional() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_classical_variable("x", "i32", 32).unwrap();
        executor.add_classical_variable("y", "i32", 32).unwrap();

        // Set initial values
        executor.get_environment_mut().set_raw("x", 10).unwrap();

        // Create a condition: x > 5
        let condition = Expression::Operation {
            cop: ">".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(5)],
        };

        // Create true branch: y = 20
        let true_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(20)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        // Create false branch: y = 30
        let false_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(30)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        // Execute conditional with the branches
        let result = executor.execute_conditional(&condition, &true_branch, Some(&false_branch));
        assert!(result.is_ok());

        // Since x = 10, which is > 5, the true branch should have executed
        let env = executor.get_environment();
        assert_eq!(env.get_raw("y"), Some(20));

        // Change x to make the condition false
        executor.get_environment_mut().set_raw("x", 2).unwrap();

        // Execute again
        let result = executor.execute_conditional(&condition, &true_branch, Some(&false_branch));
        assert!(result.is_ok());

        // Now the false branch should have executed
        let env = executor.get_environment();
        assert_eq!(env.get_raw("y"), Some(30));
    }

    #[test]
    fn test_execute_sequence() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_classical_variable("a", "i32", 32).unwrap();
        executor.add_classical_variable("b", "i32", 32).unwrap();

        // Create a sequence of operations
        let operations = vec![
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(10)],
                returns: vec![ArgItem::Simple("a".to_string())],
                function: None,
                metadata: None,
            },
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(20)],
                returns: vec![ArgItem::Simple("b".to_string())],
                function: None,
                metadata: None,
            },
        ];

        // Execute the block
        let result = executor.execute_sequence(&operations);
        assert!(result.is_ok());

        // Verify both operations executed correctly
        let env = executor.get_environment();
        assert_eq!(env.get_raw("a"), Some(10));
        assert_eq!(env.get_raw("b"), Some(20));
    }

    #[test]
    fn test_execute_qparallel() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_quantum_variable("q", 2).unwrap();

        // Create a parallel block of quantum operations
        let operations = vec![
            Operation::QuantumOp {
                qop: "H".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
                returns: vec![],
                angles: None,
                metadata: None,
            },
            Operation::QuantumOp {
                qop: "X".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 1))],
                returns: vec![],
                angles: None,
                metadata: None,
            },
        ];

        // Execute the parallel block
        let result = executor.execute_qparallel(&operations);
        assert!(result.is_ok());

        // Test that invalid parallel blocks are rejected
        let invalid_operations = vec![
            // Same qubit used twice
            Operation::QuantumOp {
                qop: "H".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
                returns: vec![],
                angles: None,
                metadata: None,
            },
            Operation::QuantumOp {
                qop: "X".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
                returns: vec![],
                angles: None,
                metadata: None,
            },
        ];

        // This should fail because the same qubit is used twice
        let result = executor.execute_qparallel(&invalid_operations);
        assert!(result.is_err());

        // Test that non-quantum operations are rejected
        let invalid_operations = vec![
            Operation::QuantumOp {
                qop: "H".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
                returns: vec![],
                angles: None,
                metadata: None,
            },
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(10)],
                returns: vec![ArgItem::Simple("a".to_string())],
                function: None,
                metadata: None,
            },
        ];

        // This should fail because a classical op is included in a qparallel block
        let result = executor.execute_qparallel(&invalid_operations);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_block() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_classical_variable("x", "i32", 32).unwrap();
        executor.add_classical_variable("y", "i32", 32).unwrap();

        // Set initial value
        executor.get_environment_mut().set_raw("x", 10).unwrap();

        // Test sequence block
        let operations = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(20)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        let result = executor.process_block("sequence", &operations, None, None, None);
        assert!(result.is_ok());
        assert_eq!(executor.get_environment().get_raw("y"), Some(20));

        // Test conditional block
        let condition = Expression::Operation {
            cop: "<".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(15)],
        };

        let true_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(30)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        let false_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(40)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        let result = executor.process_block(
            "if",
            &[],
            Some(&condition),
            Some(&true_branch),
            Some(&false_branch),
        );
        assert!(result.is_ok());

        // x = 10, which is < 15, so true branch should have executed
        assert_eq!(executor.get_environment().get_raw("y"), Some(30));
    }

    #[test]
    fn test_handle_measurements() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_quantum_variable("q", 2).unwrap();
        executor.add_classical_variable("m", "i32", 32).unwrap();

        // Create measurement operations for testing
        let operations = vec![Operation::QuantumOp {
            qop: "Measure".to_string(),
            args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
            returns: vec![("m".to_string(), 0)],
            angles: None,
            metadata: None,
        }];

        // Define measurement results
        let measurements = vec![1]; // Index 0, value 1

        // Handle measurements
        let result = executor.handle_measurements(&measurements, &operations);
        assert!(result.is_ok());

        // Verify the measurement was stored
        let env = executor.get_environment();

        // The bit should be set in the m variable
        assert_eq!(env.get_bit("m", 0).unwrap(), true);
    }

    #[test]
    fn test_get_mapped_results() {
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_classical_variable("a", "i32", 32).unwrap();
        executor.add_classical_variable("b", "i32", 32).unwrap();

        // Set values
        executor.get_environment_mut().set_raw("a", 10).unwrap();
        executor.get_environment_mut().set_raw("b", 20).unwrap();

        // Add a mapping
        executor
            .get_environment_mut()
            .add_mapping("a", "result_a")
            .unwrap();

        // Get mapped results
        let results = executor.get_mapped_results();

        // Verify the mapped value is present
        assert_eq!(results.get("result_a"), Some(&10));
    }

    #[test]
    fn test_execute_program() {
        let mut executor = BlockExecutor::new();

        // Create a simple program
        let program = vec![
            Operation::VariableDefinition {
                data: "cvar_define".to_string(),
                data_type: "i32".to_string(),
                variable: "x".to_string(),
                size: 32,
            },
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(42)],
                returns: vec![ArgItem::Simple("x".to_string())],
                function: None,
                metadata: None,
            },
        ];

        // Execute the program
        let results = executor.execute_program(&program).unwrap();

        // Verify the results
        assert_eq!(results.get("x"), Some(&42));
    }
}
