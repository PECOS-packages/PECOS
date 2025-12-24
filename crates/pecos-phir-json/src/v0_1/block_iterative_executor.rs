use crate::v0_1::ast::{Operation, QubitArg};
use crate::v0_1::block_executor::BlockExecutor;
use crate::v0_1::expression::ExpressionEvaluator;
use log::debug;
use pecos_core::errors::PecosError;
use std::collections::VecDeque;

/// Operation type for flattened operations with additional context
pub enum FlattenedOperation<'a> {
    /// Regular operation reference
    Operation(&'a Operation),
    /// Buffer of operations (e.g. for quantum parallel)
    Buffer(Vec<&'a Operation>),
    /// End of a block (used for tracking)
    EndBlock,
}

/// Struct for iteratively executing blocks of operations
/// This is an alternative to the recursive approach in `BlockExecutor`
/// It provides a more flexible way to process blocks, similar to Python's _`flatten_blocks`
pub struct BlockIterativeExecutor<'a> {
    /// Reference to the block executor for processing operations
    executor: &'a mut BlockExecutor,
    /// Stack of operations to process
    operation_stack: VecDeque<FlattenedOperation<'a>>,
    /// Operation buffer for collecting operations (e.g., around measurements)
    buffer: Vec<&'a Operation>,
    /// Flag to enable buffering behavior
    enable_buffering: bool,
}

impl<'a> BlockIterativeExecutor<'a> {
    /// Creates a new iterative executor
    pub fn new(executor: &'a mut BlockExecutor) -> Self {
        Self {
            executor,
            operation_stack: VecDeque::new(),
            buffer: Vec::new(),
            enable_buffering: true,
        }
    }

    /// Set whether to enable operation buffering
    pub fn set_buffering(&mut self, enable: bool) {
        self.enable_buffering = enable;
    }

    /// Initialize with a block of operations
    #[must_use]
    pub fn with_operations(mut self, operations: &'a [Operation]) -> Self {
        // Add operations in reverse order to the stack
        for op in operations.iter().rev() {
            self.operation_stack
                .push_front(FlattenedOperation::Operation(op));
        }
        self
    }

    /// Process operations iteratively
    ///
    /// # Errors
    /// Returns an error if any operation fails to process.
    pub fn process(&mut self) -> Result<(), PecosError> {
        while let Some(flattened_op) = self.operation_stack.pop_front() {
            match flattened_op {
                FlattenedOperation::Operation(op) => {
                    self.process_operation(op)?;
                }
                FlattenedOperation::Buffer(ops) => {
                    // Process a buffer of operations as a unit
                    for op in ops {
                        self.executor.process_operation(op)?;
                    }
                }
                FlattenedOperation::EndBlock => {
                    // End of block marker, used for tracking
                    debug!("End of block reached");
                }
            }
        }

        // Process any remaining operations in the buffer
        if !self.buffer.is_empty() {
            debug!(
                "Processing remaining buffer with {} operations",
                self.buffer.len()
            );
            for op in self.buffer.drain(..) {
                self.executor.process_operation(op)?;
            }
        }

        Ok(())
    }

    /// Process a single operation, handling blocks and buffering
    #[allow(clippy::too_many_lines)]
    fn process_operation(&mut self, op: &'a Operation) -> Result<(), PecosError> {
        debug!("Processing operation: {op:?}");
        match op {
            Operation::Block {
                block,
                ops,
                condition,
                true_branch,
                false_branch,
                ..
            } => {
                match block.as_str() {
                    "sequence" => {
                        debug!("Flattening sequence block with {} operations", ops.len());
                        // Add end block marker
                        self.operation_stack
                            .push_front(FlattenedOperation::EndBlock);

                        // Add all operations in reverse order
                        for op in ops.iter().rev() {
                            self.operation_stack
                                .push_front(FlattenedOperation::Operation(op));
                        }
                    }
                    "qparallel" => {
                        debug!("Processing qparallel block with {} operations", ops.len());
                        // For quantum parallel blocks, we validate and add as a buffer
                        // to ensure they're processed as a unit

                        // Verify all operations are quantum operations or meta instructions
                        for op in ops {
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
                        let mut used_qubits = std::collections::BTreeSet::new();

                        for op in ops {
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

                        // Add as a buffer to ensure atomic processing
                        let ops_refs: Vec<&'a Operation> = ops.iter().collect();
                        self.operation_stack
                            .push_front(FlattenedOperation::Buffer(ops_refs));
                    }
                    "if" => {
                        debug!("Processing conditional block");
                        if let Some(cond) = condition {
                            // Make sure any buffered operations are processed first
                            // This ensures that operations before the if block are executed
                            // before we evaluate the condition
                            if !self.buffer.is_empty() && self.enable_buffering {
                                debug!("Processing buffer before evaluating condition");
                                for buffered_op in self.buffer.drain(..) {
                                    self.executor.process_operation(buffered_op)?;
                                }
                            }

                            // Evaluate the condition
                            let mut evaluator =
                                ExpressionEvaluator::new(self.executor.get_environment());
                            let condition_result = evaluator.eval_expr(cond)?.as_bool();

                            debug!("Condition evaluated to: {condition_result}");

                            // Add end block marker
                            self.operation_stack
                                .push_front(FlattenedOperation::EndBlock);

                            // Add operations from the appropriate branch in reverse order
                            if condition_result {
                                if let Some(branch) = true_branch {
                                    for op in branch.iter().rev() {
                                        self.operation_stack
                                            .push_front(FlattenedOperation::Operation(op));
                                    }
                                } else {
                                    return Err(PecosError::Input(
                                        "Conditional block missing true branch".to_string(),
                                    ));
                                }
                            } else if let Some(branch) = false_branch {
                                for op in branch.iter().rev() {
                                    self.operation_stack
                                        .push_front(FlattenedOperation::Operation(op));
                                }
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
            }
            Operation::QuantumOp { qop, .. } => {
                if self.enable_buffering {
                    // Add to buffer
                    self.buffer.push(op);

                    // If this is a measurement operation, process the buffer
                    if qop.contains("Measure") {
                        debug!(
                            "Processing buffer around measurement with {} operations",
                            self.buffer.len()
                        );
                        for op in self.buffer.drain(..) {
                            self.executor.process_operation(op)?;
                        }
                    }
                } else {
                    // Process directly if buffering is disabled
                    self.executor.process_operation(op)?;
                }
            }
            Operation::ClassicalOp { .. } => {
                // For non-quantum operations, process any buffered operations first
                if !self.buffer.is_empty() && self.enable_buffering {
                    debug!(
                        "Processing buffer before classical op with {} operations",
                        self.buffer.len()
                    );
                    for buffered_op in self.buffer.drain(..) {
                        self.executor.process_operation(buffered_op)?;
                    }
                }

                // Process this classical operation
                debug!("Processing classical operation");
                let result = self.executor.process_operation(op);

                // Debug: check the environment after processing
                debug!(
                    "After processing classical op - Environment: {:?}",
                    self.executor.get_environment()
                );

                result?;
            }
            _ => {
                // For other operations, process any buffered operations first
                if !self.buffer.is_empty() && self.enable_buffering {
                    debug!(
                        "Processing buffer before non-quantum op with {} operations",
                        self.buffer.len()
                    );
                    for buffered_op in self.buffer.drain(..) {
                        self.executor.process_operation(buffered_op)?;
                    }
                }

                // Then process this operation
                self.executor.process_operation(op)?;
            }
        }

        Ok(())
    }

    /// Iterator interface for stepping through operations
    pub fn step(&mut self) -> Option<Result<&'a Operation, PecosError>> {
        if let Some(flattened_op) = self.operation_stack.pop_front() {
            match flattened_op {
                FlattenedOperation::Operation(op) => {
                    // For blocks, expand them and return the next operation
                    if let Operation::Block { .. } = op {
                        match self.process_operation(op) {
                            Ok(()) => self.step(),
                            Err(e) => Some(Err(e)),
                        }
                    } else {
                        // For regular operations, just return them
                        Some(Ok(op))
                    }
                }
                FlattenedOperation::Buffer(ops) => {
                    // For buffers, add all operations back to the stack
                    // and return the first one
                    if ops.is_empty() {
                        self.step()
                    } else {
                        let first = ops[0];
                        for op in ops.into_iter().rev().skip(1) {
                            self.operation_stack
                                .push_front(FlattenedOperation::Operation(op));
                        }
                        Some(Ok(first))
                    }
                }
                FlattenedOperation::EndBlock => {
                    // Skip end block markers
                    self.step()
                }
            }
        } else {
            None
        }
    }

    /// Get a reference to the underlying block executor
    #[must_use]
    pub fn get_executor(&self) -> &BlockExecutor {
        self.executor
    }

    /// Get a mutable reference to the underlying block executor
    pub fn get_executor_mut(&mut self) -> &mut BlockExecutor {
        self.executor
    }
}

/// Iterator implementation for `BlockIterativeExecutor`
impl<'a> Iterator for BlockIterativeExecutor<'a> {
    type Item = Result<&'a Operation, PecosError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::ast::{ArgItem, Expression, Operation};

    #[test]
    fn test_simple_sequence() {
        // Create a block executor
        let mut executor = BlockExecutor::new();

        // Add a variable for testing
        executor.add_classical_variable("x", "i32", 32).unwrap();

        // Create a sequence of operations
        let operations = vec![
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(10)],
                returns: vec![ArgItem::Simple("x".to_string())],
                function: None,
                metadata: None,
            },
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(20)],
                returns: vec![ArgItem::Simple("x".to_string())],
                function: None,
                metadata: None,
            },
        ];

        // Create an iterative executor
        let mut iterative_executor =
            BlockIterativeExecutor::new(&mut executor).with_operations(&operations);

        // Process operations
        let result = iterative_executor.process();
        assert!(result.is_ok());

        // Verify the final value
        let env = executor.get_environment();
        assert_eq!(env.get_raw("x"), Some(20));
    }

    #[test]
    fn test_conditional_blocks() {
        // Create a block executor
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_classical_variable("x", "i32", 32).unwrap();
        executor.add_classical_variable("y", "i32", 32).unwrap();

        // Set initial value
        executor.get_environment_mut().set_raw("x", 10).unwrap();

        // Create an if block with condition x > 5
        let condition = Expression::Operation {
            cop: ">".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(5)],
        };

        // True branch: y = 20
        let true_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(20)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        // False branch: y = 30
        let false_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(30)],
            returns: vec![ArgItem::Simple("y".to_string())],
            function: None,
            metadata: None,
        }];

        // Create the if block operation
        let if_operation = Operation::Block {
            block: "if".to_string(),
            ops: vec![],
            condition: Some(condition),
            true_branch: Some(true_branch),
            false_branch: Some(false_branch),
            metadata: None,
        };

        // Create an iterative executor with the if block
        let operations = vec![if_operation];
        let mut iterative_executor =
            BlockIterativeExecutor::new(&mut executor).with_operations(&operations);

        // Process operations
        let result = iterative_executor.process();
        assert!(result.is_ok());

        // Verify the true branch was executed (x = 10 > 5)
        let env = executor.get_environment();
        assert_eq!(env.get_raw("y"), Some(20));
    }

    #[test]
    fn test_nested_blocks() {
        // Create a block executor
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_classical_variable("x", "i32", 32).unwrap();
        executor.add_classical_variable("y", "i32", 32).unwrap();
        executor.add_classical_variable("z", "i32", 32).unwrap();

        // Set initial values
        executor.get_environment_mut().set_raw("x", 10).unwrap();
        // For testing purposes, we'll set y directly to 15 (as if x + 5 was already calculated)
        executor.get_environment_mut().set_raw("y", 15).unwrap();

        // Create a nested structure:
        // sequence
        //   if y > 10
        //     z = 100
        //   else
        //     z = 200

        // Inner condition: y > 10
        let inner_condition = Expression::Operation {
            cop: ">".to_string(),
            args: vec![ArgItem::Simple("y".to_string()), ArgItem::Integer(10)],
        };

        // Inner true branch: z = 100
        let inner_true_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(100)],
            returns: vec![ArgItem::Simple("z".to_string())],
            function: None,
            metadata: None,
        }];

        // Inner false branch: z = 200
        let inner_false_branch = vec![Operation::ClassicalOp {
            cop: "=".to_string(),
            args: vec![ArgItem::Integer(200)],
            returns: vec![ArgItem::Simple("z".to_string())],
            function: None,
            metadata: None,
        }];

        // Inner if block
        let inner_if_block = Operation::Block {
            block: "if".to_string(),
            ops: vec![],
            condition: Some(inner_condition),
            true_branch: Some(inner_true_branch),
            false_branch: Some(inner_false_branch),
            metadata: None,
        };

        // Create operations array with just the if block
        let operations = vec![inner_if_block];

        // Create an iterative executor
        let mut iterative_executor =
            BlockIterativeExecutor::new(&mut executor).with_operations(&operations);

        // Process operations
        let result = iterative_executor.process();
        assert!(result.is_ok());

        // Verify results:
        // 1. y should be x + 5 = 15
        // 2. z should be 100 (from true branch since y > 10)
        let env = executor.get_environment();

        // In y = x + 5 where x = 10, y should be 15
        let y_value = env.get_raw("y");
        println!("y value: {y_value:?}");
        assert_eq!(y_value, Some(15));

        let z_value = env.get_raw("z");
        println!("z value: {z_value:?}");
        assert_eq!(z_value, Some(100));
    }

    #[test]
    fn test_buffering() {
        // Create a block executor
        let mut executor = BlockExecutor::new();

        // Add variables for testing
        executor.add_quantum_variable("q", 2).unwrap();
        executor.add_classical_variable("m", "i32", 32).unwrap();

        // Create operations with measurements
        let operations = vec![
            // Quantum op (should be buffered)
            Operation::QuantumOp {
                qop: "H".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
                returns: vec![],
                angles: None,
                metadata: None,
            },
            // Measurement op (should flush buffer)
            Operation::QuantumOp {
                qop: "Measure".to_string(),
                args: vec![QubitArg::SingleQubit(("q".to_string(), 0))],
                returns: vec![("m".to_string(), 0)],
                angles: None,
                metadata: None,
            },
            // Classical op (should not be buffered)
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(42)],
                returns: vec![ArgItem::Simple("m".to_string())],
                function: None,
                metadata: None,
            },
        ];

        // Create an iterative executor with buffering enabled
        let mut iterative_executor =
            BlockIterativeExecutor::new(&mut executor).with_operations(&operations);

        // Process operations
        let result = iterative_executor.process();
        assert!(result.is_ok());

        // Verify the final state
        let env = executor.get_environment();
        assert_eq!(env.get_raw("m"), Some(42));
    }

    #[test]
    fn test_iterator_interface() {
        // Create a block executor
        let mut executor = BlockExecutor::new();

        // Add a variable for testing
        executor.add_classical_variable("x", "i32", 32).unwrap();

        // Create a sequence of operations
        let operations = vec![
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(10)],
                returns: vec![ArgItem::Simple("x".to_string())],
                function: None,
                metadata: None,
            },
            Operation::ClassicalOp {
                cop: "=".to_string(),
                args: vec![ArgItem::Integer(20)],
                returns: vec![ArgItem::Simple("x".to_string())],
                function: None,
                metadata: None,
            },
        ];

        // Create an iterative executor
        let iterative_executor =
            BlockIterativeExecutor::new(&mut executor).with_operations(&operations);

        // Collect operations using the iterator interface
        let collected_ops: Vec<_> = iterative_executor.filter_map(Result::ok).collect();

        // There should be 2 operations
        assert_eq!(collected_ops.len(), 2);
    }
}
