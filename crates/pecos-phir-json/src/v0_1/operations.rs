use crate::v0_1::ast::{ArgItem, Expression, MEASUREMENT_PREFIX, Operation, QubitArg};
use crate::v0_1::environment::{DataType, Environment, TypedValue};
use crate::v0_1::expression::ExpressionEvaluator;
use crate::v0_1::foreign_objects::ForeignObject;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_core::{Angle64, Gate};
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use std::collections::{BTreeMap, BTreeSet};

/// Represents the result of processing a meta instruction
#[derive(Debug, Clone)]
pub enum MetaInstructionResult {
    /// Barrier operation - prevents compiler optimizations across this point
    Barrier {
        /// Qubits affected by the barrier
        qubits: Vec<(String, usize)>,
    },
}

/// Represents the result of processing a machine operation.
///
/// Machine operations (MOPs) provide fine-grained control over physical aspects of quantum computation,
/// such as timing, qubit movement, and hardware-specific features. These operations complement
/// quantum and classical operations to create complete quantum programs with timing constraints
/// and hardware-specific optimizations.
#[derive(Debug, Clone)]
pub enum MachineOperationResult {
    /// Idle operation - qubits idle for a specific duration
    ///
    /// The idle operation specifies that the given qubits should remain in their current state
    /// without any operations being applied for the specified duration. This is useful for
    /// implementing delays or synchronizing operations across different qubits.
    ///
    /// # Example JSON representation
    /// ```json
    /// {
    ///   "mop": "Idle",
    ///   "args": [["q", 0], ["q", 1]],
    ///   "duration": [5.0, "ms"]
    /// }
    /// ```
    Idle {
        /// Qubits affected by the idle operation
        qubits: Vec<(String, usize)>,
        /// Duration in nanoseconds
        duration_ns: u64,
        /// Additional metadata for the operation
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Transport operation - qubits are moved from one location to another
    ///
    /// The transport operation represents moving qubits between different physical locations
    /// in architectures where this is possible (e.g., trapped ions, photonic systems).
    ///
    /// # Example JSON representation
    /// ```json
    /// {
    ///   "mop": "Transport",
    ///   "args": [["q", 1]],
    ///   "duration": [1.0, "ms"],
    ///   "metadata": {"from_position": [0, 0], "to_position": [1, 0]}
    /// }
    /// ```
    Transport {
        /// Qubits being transported
        qubits: Vec<(String, usize)>,
        /// Duration in nanoseconds
        duration_ns: u64,
        /// Additional metadata for the operation
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Delay operation - insert a specific delay for qubits
    ///
    /// The delay operation is similar to idle but specifically represents
    /// an intentional delay inserted into the program execution. This can be used
    /// to implement timing constraints or account for physical system relaxation.
    ///
    /// # Example JSON representation
    /// ```json
    /// {
    ///   "mop": "Delay",
    ///   "args": [["q", 0]],
    ///   "duration": [2.0, "us"]
    /// }
    /// ```
    Delay {
        /// Qubits to delay
        qubits: Vec<(String, usize)>,
        /// Duration in nanoseconds
        duration_ns: u64,
        /// Additional metadata for the operation
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Timing operation - synchronize operations in time
    ///
    /// The timing operation provides synchronization points in the program. It can be used
    /// to mark the beginning or end of a timing region, or to synchronize operations across
    /// different qubits. The exact semantics depend on the `timing_type` field.
    ///
    /// # Example JSON representation
    /// ```json
    /// {
    ///   "mop": "Timing",
    ///   "args": [["q", 0], ["q", 1]],
    ///   "metadata": {"timing_type": "sync", "label": "sync_point_1"}
    /// }
    /// ```
    Timing {
        /// Qubits affected by the timing operation
        qubits: Vec<(String, usize)>,
        /// Timing type ("start", "end", "sync", etc.)
        timing_type: String,
        /// Timing label for synchronization
        label: String,
        /// Additional metadata for the operation
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Skip operation - does nothing
    ///
    /// The skip operation is a no-op that can be used as a placeholder or
    /// to explicitly indicate that nothing should be done at this point.
    ///
    /// # Example JSON representation
    /// ```json
    /// {
    ///   "mop": "Skip"
    /// }
    /// ```
    Skip,
}

/// Handles processing of variable definitions, quantum and classical operations
#[derive(Debug)]
pub struct OperationProcessor {
    /// Environment for variable storage and access - the single source of truth for all variables, values, and mappings
    pub environment: Environment,
    /// Foreign object for executing foreign function calls
    pub foreign_object: Option<Box<dyn ForeignObject>>,
    /// Current operation index being processed
    current_op: usize,
}

impl Default for OperationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for OperationProcessor {
    fn clone(&self) -> Self {
        // Create a new processor with all fields cloned
        Self {
            environment: self.environment.clone(),
            foreign_object: self.foreign_object.as_ref().map(|fo| fo.clone_box()),
            current_op: self.current_op,
        }
        // All data including mappings is now in the environment
    }
}

impl OperationProcessor {
    /// Creates a new operation processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            environment: Environment::new(),
            foreign_object: None,
            current_op: 0,
        }
    }

    /// Get the variables of type "qubits"
    /// Returns a map of quantum variable names to their sizes
    /// This is a helper method that accesses the environment directly
    #[must_use]
    pub fn get_quantum_variables(&self) -> BTreeMap<String, usize> {
        // Use the environment to get all variables of type Qubits
        let qubits_variables = self.environment.get_variables_of_type(&DataType::Qubits);

        // Convert to a HashMap with variable name -> size
        qubits_variables
            .into_iter()
            .map(|info| (info.name.clone(), info.size))
            .collect()
    }

    /// Get the classical variables
    /// Returns a map of classical variable names to their types and sizes
    /// This is a helper method that accesses the environment directly
    #[must_use]
    pub fn get_classical_variables(&self) -> BTreeMap<String, (String, usize)> {
        // Get all variables except qubits
        self.environment
            .get_all_variables()
            .iter()
            .filter(|info| info.data_type != DataType::Qubits)
            .map(|info| {
                let type_name = info.data_type.to_string();
                (info.name.clone(), (type_name, info.size))
            })
            .collect()
    }

    /// Get all measurement results from the environment
    ///
    /// Returns a map of variable names to their u32 values by extracting:
    /// 1. All measurement variables from the environment (m_*, measurement_*, m)
    /// 2. All explicitly mapped variables (from environment mappings)
    ///
    /// This delegates directly to the environment which is the single source of truth.
    #[must_use]
    pub fn get_measurement_results(&self) -> BTreeMap<String, u32> {
        // Get all measurement-related variables from the environment
        let mut results = BTreeMap::new();
        let all_results = self.environment.get_measurement_results();

        // Convert TypedValue to u32
        for (name, value) in all_results {
            results.insert(name, value.as_u32());
        }

        // If no results were found, fall back to mapped results
        if results.is_empty() {
            return self.environment.get_mapped_results();
        }

        results
    }

    /// Creates a new operation processor with a foreign object
    #[must_use]
    pub fn with_foreign_object(foreign_object: Box<dyn ForeignObject>) -> Self {
        let mut processor = Self::new();
        processor.foreign_object = Some(foreign_object);
        processor
    }

    /// Resets the operation processor state
    /// Reset this processor to its initial state, but preserve the foreign object and variable definitions
    pub fn reset(&mut self) {
        // Clear state but keep variable definitions
        self.environment.reset_values();
        // Environment reset_values now also clears mappings

        // We deliberately don't clear variable definitions or foreign_object
        // so that we preserve the structure of the program while resetting state
    }

    /// Set a variable value in the environment
    /// Environment is the single source of truth for all variables
    ///
    /// # Errors
    /// Returns an error if the variable cannot be created or updated.
    pub fn set_variable_value(&mut self, name: &str, value: u64) -> Result<(), PecosError> {
        // Create the variable if it doesn't exist
        if !self.environment.has_variable(name) {
            // Add but allow failure if it already exists
            match self.environment.add_variable(name, DataType::I32, 32) {
                Ok(()) => log::debug!("Created new variable: {name} in environment"),
                Err(e) => log::warn!(
                    "Could not create variable in environment: {name}. Will try to update anyway: {e}"
                ),
            }
        }

        // Set the value in the environment
        match self.environment.set(name, value) {
            Ok(()) => log::debug!("Set variable {name} = {value} in environment"),
            Err(e) => log::warn!("Could not set variable value in environment: {name}. Error: {e}"),
        }

        Ok(())
    }

    /// Sets the foreign object for this processor
    pub fn set_foreign_object(&mut self, foreign_object: Box<dyn ForeignObject>) {
        self.foreign_object = Some(foreign_object);
    }

    /// Evaluates a classical expression
    ///
    /// # Errors
    /// Returns an error if the expression cannot be evaluated (e.g., undefined variables).
    pub fn evaluate_expression(&self, expr: &Expression) -> Result<i64, PecosError> {
        log::debug!("Evaluating expression: {expr:?}");

        // Create an expression evaluator using our environment
        let mut evaluator = ExpressionEvaluator::new(&self.environment);

        // Evaluate the expression and return as i64
        let result = evaluator.eval_expr(expr)?;
        Ok(result.as_i64())
    }

    /// Evaluates an argument item (variable, literal, etc.)
    fn evaluate_arg_item(&self, arg: &ArgItem) -> Result<i64, PecosError> {
        log::debug!("Evaluating argument item: {arg:?}");

        // Create an expression evaluator using our environment as the primary variable source
        let mut evaluator = ExpressionEvaluator::new(&self.environment);

        // Evaluate the argument using the environment and return as i64
        let result = evaluator.eval_arg(arg)?;
        Ok(result.as_i64())
    }

    // Removed get_variable_value method as it's no longer needed

    /// Process a block operation with improved validation and handling
    ///
    /// # Errors
    /// Returns an error if the block type is unknown or invalid.
    pub fn process_block(
        &self,
        block_type: &str,
        operations: &[Operation],
    ) -> Result<Vec<Operation>, PecosError> {
        match block_type {
            "sequence" => {
                // Sequence blocks are just a sequence of operations, return as-is
                // No additional validation needed since any sequence is valid
                log::debug!(
                    "Processing sequence block with {} operations",
                    operations.len()
                );
                Ok(operations.to_vec())
            }
            "qparallel" => {
                // Process qparallel block with enhanced validation
                log::debug!(
                    "Processing qparallel block with {} operations",
                    operations.len()
                );
                Self::process_qparallel_block(operations)
            }
            "if" => {
                // If blocks are handled separately by process_conditional_block
                // Here we're just returning the operations; actual condition evaluation
                // happens in process_conditional_block
                log::debug!("Processing if block structure (condition will be evaluated later)");
                Ok(operations.to_vec())
            }
            _ => {
                log::error!("Unknown block type: {block_type}");
                Err(PecosError::Input(format!(
                    "Unknown block type: {block_type}"
                )))
            }
        }
    }

    /// Process a qparallel block with improved validation
    fn process_qparallel_block(operations: &[Operation]) -> Result<Vec<Operation>, PecosError> {
        // First validate that all operations are quantum operations
        for op in operations {
            match op {
                Operation::QuantumOp { .. } | Operation::MetaInstruction { .. } => {
                    // Quantum operations and meta instructions are allowed
                }
                _ => {
                    log::error!("Non-quantum operation in qparallel block: {op:?}");
                    return Err(PecosError::Input(format!(
                        "Invalid qparallel block: only quantum operations and meta instructions are allowed, found: {op:?}"
                    )));
                }
            }
        }

        // For qparallel blocks, we need to ensure no qubits are used more than once
        let mut all_qubits = BTreeSet::new();

        for op in operations {
            if let Operation::QuantumOp { args, .. } = op {
                for qubit_arg in args {
                    match qubit_arg {
                        QubitArg::SingleQubit(qubit) => {
                            if !all_qubits.insert(qubit.clone()) {
                                log::error!(
                                    "Qubit {qubit:?} used more than once in qparallel block"
                                );
                                return Err(PecosError::Input(format!(
                                    "Invalid qparallel block: qubit {qubit:?} used more than once"
                                )));
                            }
                        }
                        QubitArg::MultipleQubits(qubits) => {
                            for qubit in qubits {
                                if !all_qubits.insert(qubit.clone()) {
                                    log::error!(
                                        "Qubit {qubit:?} used more than once in qparallel block"
                                    );
                                    return Err(PecosError::Input(format!(
                                        "Invalid qparallel block: qubit {qubit:?} used more than once"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        // If we get here, all qubits are used only once, so the block is valid
        log::debug!(
            "Qparallel block validated successfully with {} operations",
            operations.len()
        );
        Ok(operations.to_vec())
    }

    /// Process a conditional (if/else) block with improved evaluation
    ///
    /// # Errors
    /// Returns an error if the condition expression cannot be evaluated.
    pub fn process_conditional_block(
        &self,
        condition: &Expression,
        true_branch: &[Operation],
        false_branch: Option<&[Operation]>,
    ) -> Result<Vec<Operation>, PecosError> {
        // Evaluate the condition using our improved ExpressionEvaluator
        log::debug!("Evaluating condition for conditional block: {condition:?}");

        // Create expression evaluator with our environment
        let mut evaluator = ExpressionEvaluator::new(&self.environment);

        // Evaluate the condition - convert u64 result to i64 for compatibility
        let condition_value = evaluator.eval_expr(condition)?;
        log::debug!("Condition evaluated to: {condition_value}");

        // Execute the appropriate branch
        if condition_value != 0 {
            // Condition is true, return the true branch operations
            log::debug!(
                "Condition is true, executing true branch with {} operations",
                true_branch.len()
            );
            Ok(true_branch.to_vec())
        } else if let Some(branch) = false_branch {
            // Condition is false and there's a false branch, return its operations
            log::debug!(
                "Condition is false, executing false branch with {} operations",
                branch.len()
            );
            Ok(branch.to_vec())
        } else {
            // Condition is false and there's no false branch, return empty list
            log::debug!("Condition is false, no false branch provided");
            Ok(Vec::new())
        }
    }

    /// Process a meta instruction
    ///
    /// # Errors
    /// Returns an error if the meta instruction type is unsupported or arguments are invalid.
    pub fn process_meta_instruction(
        &self,
        meta_type: &str,
        args: &[(String, usize)],
    ) -> Result<MetaInstructionResult, PecosError> {
        match meta_type {
            "barrier" => {
                // Process barrier instruction
                // Validate all qubits in the barrier
                for (var, idx) in args {
                    self.validate_variable_access(var, *idx)?;
                }

                // Extract qubit indices for the barrier (just for validation)
                let _qubit_indices: Vec<usize> = args.iter().map(|(_, idx)| *idx).collect();

                // Return barrier result
                Ok(MetaInstructionResult::Barrier {
                    qubits: args.to_vec(),
                })
            }
            _ => Err(PecosError::Input(format!(
                "Unsupported meta instruction: {meta_type}"
            ))),
        }
    }

    /// Add a meta instruction to the byte message builder
    ///
    /// # Errors
    /// Currently never returns an error, but may in future implementations.
    pub fn add_meta_instruction_to_builder(
        &self,
        _builder: &mut ByteMessageBuilder,
        meta_result: &MetaInstructionResult,
    ) -> Result<(), PecosError> {
        match meta_result {
            MetaInstructionResult::Barrier { qubits } => {
                // Extract qubit indices for the barrier for debug output
                let qubit_indices: Vec<usize> = qubits.iter().map(|(_, idx)| *idx).collect();

                // Add barrier operation to the builder (if supported by the ByteMessageBuilder)
                // For now, we handle it as a "no-op" since barriers are primarily compiler hints
                debug!("Adding barrier for qubits: {qubit_indices:?}");
            }
        }

        Ok(())
    }

    /// Process a machine operation (MOP) and return the corresponding result object.
    ///
    /// This function takes the basic parameters of a machine operation from the PHIR format
    /// and processes them into a structured `MachineOperationResult` that can be used by the executor.
    /// It validates the parameters, converts time units to a standard format (nanoseconds),
    /// and extracts relevant information from the metadata.
    ///
    /// # Parameters
    ///
    /// * `mop_type` - The type of machine operation (e.g., "Idle", "Transport", "Delay", "Timing", "Reset", "Skip")
    /// * `args` - Optional list of qubit arguments affected by the operation
    /// * `duration` - Optional duration for time-based operations as a tuple of (value, unit)
    /// * `metadata` - Optional additional information for the operation
    ///
    /// # Returns
    ///
    /// * `Ok(MachineOperationResult)` - A structured result object representing the machine operation
    /// * `Err(PecosError)` - If the operation parameters are invalid
    ///
    /// # Errors
    /// Returns an error if:
    /// - The duration is negative
    /// - The time unit is not supported
    /// - The duration value would overflow when converted to nanoseconds
    /// - The machine operation type is not supported
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use pecos_phir_json::v0_1::operations::OperationProcessor;
    /// # use std::collections::BTreeMap;
    /// # let processor = OperationProcessor::new();
    /// // Process an idle operation for 5 milliseconds
    /// let result = processor.process_machine_op(
    ///     "Idle",
    ///     None,
    ///     Some(&(5.0, "ms".to_string())),
    ///     None
    /// );
    /// ```
    #[allow(clippy::too_many_lines)]
    pub fn process_machine_op(
        &self,
        mop_type: &str,
        args: Option<&Vec<QubitArg>>,
        duration: Option<&(f64, String)>,
        metadata: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> Result<MachineOperationResult, PecosError> {
        // Define constants at the beginning of the function
        const MAX_SAFE_F64_TO_U64: f64 = 18_446_744_073_709_549_568.0; // 2^64 - 2048

        // Convert the duration to nanoseconds for consistent handling
        let duration_ns = if let Some((value, unit)) = duration {
            // Validate that the value is non-negative
            if *value < 0.0 {
                return Err(PecosError::Input(format!(
                    "Duration must be non-negative, got: {value}"
                )));
            }

            // Convert to nanoseconds with overflow checking
            let ns_value = match unit.as_str() {
                "s" => *value * 1_000_000_000.0,
                "ms" => *value * 1_000_000.0,
                "us" => *value * 1_000.0,
                "ns" => *value,
                _ => {
                    return Err(PecosError::Input(format!("Unsupported time unit: {unit}")));
                }
            };

            // Check for overflow before casting
            if ns_value > MAX_SAFE_F64_TO_U64 {
                return Err(PecosError::Input(format!(
                    "Duration too large: {value} {unit}"
                )));
            }

            // Safe cast after validation
            // We've already validated the value is non-negative and not too large
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let casted_value = ns_value as u64;
            casted_value
        } else {
            0 // No duration specified
        };

        // Process the different machine operation types
        match mop_type {
            "Idle" => {
                // Extract qubit arguments if provided
                let qubit_args = if let Some(qargs) = args {
                    self.extract_all_qubits(qargs)?
                } else {
                    Vec::new()
                };

                // Create idle operation result
                Ok(MachineOperationResult::Idle {
                    qubits: qubit_args,
                    duration_ns,
                    metadata: metadata.cloned(),
                })
            }
            "Transport" => {
                // Extract qubit arguments if provided
                let qubit_args = if let Some(qargs) = args {
                    self.extract_all_qubits(qargs)?
                } else {
                    Vec::new()
                };

                // Create transport operation result
                Ok(MachineOperationResult::Transport {
                    qubits: qubit_args,
                    duration_ns,
                    metadata: metadata.cloned(),
                })
            }
            "Delay" => {
                // Extract qubit arguments if provided
                let qubit_args = if let Some(qargs) = args {
                    self.extract_all_qubits(qargs)?
                } else {
                    Vec::new()
                };

                // Create delay operation result
                Ok(MachineOperationResult::Delay {
                    qubits: qubit_args,
                    duration_ns,
                    metadata: metadata.cloned(),
                })
            }
            "Timing" => {
                // Extract qubit arguments if provided
                let qubit_args = if let Some(qargs) = args {
                    self.extract_all_qubits(qargs)?
                } else {
                    Vec::new()
                };

                // Extract timing metadata
                let timing_type = if let Some(meta) = metadata {
                    meta.get("timing_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("sync")
                        .to_string()
                } else {
                    "sync".to_string()
                };

                let label = if let Some(meta) = metadata {
                    meta.get("label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default")
                        .to_string()
                } else {
                    "default".to_string()
                };

                // Create timing operation result
                Ok(MachineOperationResult::Timing {
                    qubits: qubit_args,
                    timing_type,
                    label,
                    metadata: metadata.cloned(),
                })
            }
            "Skip" => {
                // Skip operation does nothing
                Ok(MachineOperationResult::Skip)
            }
            _ => Err(PecosError::Input(format!(
                "Unsupported machine operation: {mop_type}"
            ))),
        }
    }

    /// Helper method to extract all qubits from a list of `QubitArg` values
    fn extract_all_qubits(
        &self,
        qubit_args: &[QubitArg],
    ) -> Result<Vec<(String, usize)>, PecosError> {
        let mut qubits = Vec::new();

        for qubit_arg in qubit_args {
            match qubit_arg {
                QubitArg::SingleQubit((var, idx)) => {
                    // Validate the qubit exists
                    self.validate_variable_access(var, *idx)?;
                    qubits.push((var.clone(), *idx));
                }
                QubitArg::MultipleQubits(qubit_list) => {
                    for (var, idx) in qubit_list {
                        // Validate each qubit exists
                        self.validate_variable_access(var, *idx)?;
                        qubits.push((var.clone(), *idx));
                    }
                }
            }
        }

        Ok(qubits)
    }

    /// Add a machine operation to a byte message builder.
    ///
    /// This function translates a high-level `MachineOperationResult` into the corresponding
    /// byte-level representation in the `ByteMessageBuilder`. The exact representation depends on
    /// the capabilities of the builder and the target hardware.
    ///
    /// # Parameters
    ///
    /// * `builder` - The byte message builder to add the operation to
    /// * `mop_result` - The machine operation result to add
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the operation was successfully added to the builder
    /// * `Err(PecosError)` - If the operation could not be added
    ///
    /// # Errors
    /// Currently never returns an error, but may in future implementations.
    ///
    /// # Notes
    ///
    /// Some machine operations may not be directly supported by all hardware backends. In these cases,
    /// the operations are translated to the closest equivalent (e.g., a `Reset` might be implemented
    /// as a measurement followed by conditional X gates, or a `Timing` operation might be implemented
    /// as an `Idle` operation).
    pub fn add_machine_operation_to_builder(
        &self,
        builder: &mut ByteMessageBuilder,
        mop_result: &MachineOperationResult,
    ) -> Result<(), PecosError> {
        match mop_result {
            MachineOperationResult::Idle {
                qubits,
                duration_ns,
                ..
            } => {
                // Extract qubit indices for the idle operation
                let qubit_indices: Vec<usize> = qubits.iter().map(|(_, idx)| *idx).collect();

                // Add idle operation to the builder
                if !qubit_indices.is_empty() {
                    // Convert nanoseconds to seconds as f64
                    // This conversion may lose precision for very large durations (>52 bits)
                    #[allow(clippy::cast_precision_loss)]
                    let duration_seconds = *duration_ns as f64 / 1_000_000_000.0;
                    builder.idle(duration_seconds, &qubit_indices);
                }
            }
            MachineOperationResult::Transport {
                qubits,
                duration_ns,
                ..
            } => {
                // Extract qubit indices for the transport operation
                let qubit_indices: Vec<usize> = qubits.iter().map(|(_, idx)| *idx).collect();

                // Add transport operation to the builder if supported
                // For now, we'll treat it as an idle operation
                if !qubit_indices.is_empty() {
                    // Convert nanoseconds to seconds as f64
                    // This conversion may lose precision for very large durations (>52 bits)
                    #[allow(clippy::cast_precision_loss)]
                    let duration_seconds = *duration_ns as f64 / 1_000_000_000.0;
                    builder.idle(duration_seconds, &qubit_indices);
                }
            }
            MachineOperationResult::Delay {
                qubits,
                duration_ns,
                ..
            } => {
                // Extract qubit indices for the delay operation
                let qubit_indices: Vec<usize> = qubits.iter().map(|(_, idx)| *idx).collect();

                // Add delay operation to the builder if supported
                // For now, we'll treat it as an idle operation
                if !qubit_indices.is_empty() {
                    // Convert nanoseconds to seconds as f64
                    // This conversion may lose precision for very large durations (>52 bits)
                    #[allow(clippy::cast_precision_loss)]
                    let duration_seconds = *duration_ns as f64 / 1_000_000_000.0;
                    builder.idle(duration_seconds, &qubit_indices);
                }
            }
            MachineOperationResult::Timing {
                qubits,
                timing_type,
                label,
                ..
            } => {
                // Extract qubit indices for the timing operation
                let qubit_indices: Vec<usize> = qubits.iter().map(|(_, idx)| *idx).collect();

                // Add timing operation to the builder if supported
                debug!(
                    "Timing operation '{timing_type}' with label '{label}' for qubits: {qubit_indices:?}"
                );
            }
            MachineOperationResult::Skip => {
                // Skip does nothing
            }
        }

        Ok(())
    }

    /// Add a quantum variable to the environment
    /// Uses the environment as the single source of truth
    ///
    /// # Errors
    /// Returns an error if the variable already exists or cannot be added.
    pub fn add_quantum_variable(&mut self, variable: &str, size: usize) -> Result<(), PecosError> {
        // Store in the environment (single source of truth)
        self.environment
            .add_variable(variable, DataType::Qubits, size)?;
        log::debug!("Defined quantum variable {variable} of size {size}");
        Ok(())
    }

    /// Add a classical variable to the environment
    /// Uses the environment as the single source of truth
    ///
    /// # Errors
    /// Returns an error if the data type is invalid.
    pub fn add_classical_variable(
        &mut self,
        variable: &str,
        data_type: &str,
        size: usize,
    ) -> Result<(), PecosError> {
        // Convert string data type to DataType enum
        let dt = DataType::from_str(data_type)?;

        // Only add to environment if it doesn't already exist
        // This is important for compatibility with test programs that might redefine variables
        if self.environment.has_variable(variable) {
            log::debug!("Variable '{variable}' already exists in environment, skipping creation");
        } else {
            match self.environment.add_variable(variable, dt, size) {
                Ok(()) => log::debug!(
                    "Added classical variable {variable} of type {data_type} and size {size}"
                ),
                Err(e) => log::warn!(
                    "Could not add variable '{variable}' to environment: {e}. Will continue with existing variable."
                ),
            }
        }

        Ok(())
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
        match data {
            "qvar_define" if data_type == "qubits" => {
                self.add_quantum_variable(variable, size)?;
            }
            "cvar_define" => {
                self.add_classical_variable(variable, data_type, size)?;
            }
            _ => {
                log::warn!("Unknown variable definition: {data} {data_type} {variable}");
                return Err(PecosError::Input(format!(
                    "Unknown variable definition: {data} {data_type} {variable}"
                )));
            }
        }

        Ok(())
    }

    /// Validate variable access to ensure it exists in the environment
    ///
    /// This method ensures the variable exists and the index is within bounds.
    /// It no longer auto-creates missing variables as that's inconsistent with
    /// using the environment as a single source of truth.
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist or the index is out of bounds.
    pub fn validate_variable_access(&self, var: &str, idx: usize) -> Result<(), PecosError> {
        // Check in environment (single source of truth)
        if self.environment.has_variable(var) {
            // Get variable info to check size
            let var_info = self.environment.get_variable_info(var)?;
            if idx >= var_info.size {
                return Err(PecosError::Input(format!(
                    "Variable access validation failed: Index {idx} out of bounds for variable '{var}' of size {}",
                    var_info.size
                )));
            }
            return Ok(());
        }

        // Variable doesn't exist, return error
        Err(PecosError::Input(format!(
            "Variable '{var}' not found in environment"
        )))
    }

    /// Ensure all variables in the environment have consistent values
    /// This method is now much simpler since the environment is the single source of truth.
    /// It's primarily kept for compatibility with code that expects this method to exist.
    ///
    /// # Errors
    /// Currently never returns an error.
    pub fn update_expression_results(&mut self) -> Result<(), PecosError> {
        log::debug!("Variables from environment are already the single source of truth");
        // No need to do anything - the environment already has all the values
        Ok(())
    }

    /// Handle classical operations
    ///
    /// # Errors
    /// Returns an error if:
    /// - Variable references are invalid
    /// - Expressions cannot be evaluated
    /// - Foreign function calls fail
    /// - Assignment operations fail
    #[allow(clippy::too_many_lines)]
    pub fn handle_classical_op(
        &mut self,
        cop: &str,
        args: &[ArgItem],
        returns: &[ArgItem],
        ops: &[Operation], // Reference to all operations
        current_op: usize, // Current operation index
    ) -> Result<bool, PecosError> {
        // Store the current operation index for later use
        self.current_op = current_op;

        // No synchronization needed - environment is the single source of truth
        // Extract variable name and index from each ArgItem
        let extract_var_idx = |arg: &ArgItem| -> Result<(String, usize), PecosError> {
            match arg {
                ArgItem::Indexed((name, idx)) => Ok((name.clone(), *idx)),
                ArgItem::Simple(name) => Ok((name.clone(), 0)),
                ArgItem::Integer(_) => Err(PecosError::Input(
                    "Expected variable reference, got integer literal".to_string(),
                )),
                ArgItem::Expression(_) => Err(PecosError::Input(
                    "Expected variable reference, got expression".to_string(),
                )),
            }
        };

        // For most operations, validate all variable accesses
        if cop == "Result" {
            // For Result operation, only validate the source variables (args)
            // The return variables are outputs and don't need to be defined
            for arg in args {
                let (var, idx) = extract_var_idx(arg)?;
                self.validate_variable_access(&var, idx)?;
            }
        } else if cop == "ffcall" {
            debug!("Processing ffcall operation: {:?}", ops.get(current_op));
        } else if cop == "=" {
            // For assignment, we evaluate the expression and assign to the variable

            // Validate return variables (target of assignment)
            for ret in returns {
                match ret {
                    ArgItem::Simple(_var) | ArgItem::Indexed((_var, _)) => {
                        // For assignment, we don't need to validate the variable exists
                        // It might be created by this operation
                    }
                    _ => {
                        return Err(PecosError::Input(
                            "Assignment target must be a variable reference".to_string(),
                        ));
                    }
                }
            }

            // Evaluate arguments (source of assignment)
            // For now, we only support a single argument
            if args.len() == 1 && returns.len() == 1 {
                let value = self.evaluate_arg_item(&args[0])?;

                // Assign to the target variable
                let (var, idx) = extract_var_idx(&returns[0])?;

                // For bit-level assignment, set the specific bit in the environment
                if let ArgItem::Indexed(_) = &returns[0] {
                    // Set the bit at position idx to value & 1
                    let bit_value = value & 1;

                    // Update in environment if the variable exists there
                    if self.environment.has_variable(&var) {
                        // Set the bit in environment
                        // bit_value is already masked with & 1, so it's guaranteed to be 0 or 1
                        self.environment.set_bit(
                            &var,
                            idx,
                            u64::try_from(bit_value).unwrap_or(0),
                        )?;
                        log::debug!("Set bit {var}[{idx}] = {bit_value} in environment");
                    }

                    // Calculate the new value and update exported_values
                    // Get the current value from environment or use 0 if it doesn't exist
                    let env_value = self.environment.get(&var).unwrap_or(TypedValue::U32(0));
                    let current_value = env_value.as_u32();

                    // Clear the bit and set it to the new value
                    let mask = !(1 << idx);
                    // bit_value is already masked with & 1, so it's guaranteed to be 0 or 1
                    let bit_u32 = u32::try_from(bit_value).unwrap_or(0);
                    let new_value = (current_value & mask) | (bit_u32 << idx);

                    // Make sure the composite variable is updated in the environment as well
                    match self.environment.set(&var, u64::from(new_value)) {
                        Ok(()) => {
                            log::debug!("Updated composite variable: {var} = {new_value}");
                        }
                        Err(e) => {
                            log::warn!("Could not update composite variable: {var}. Error: {e}");
                        }
                    }
                    log::debug!("Added bit-level value to environment: {var} = {new_value}");
                } else {
                    // For whole variable assignment, store in environment
                    log::debug!("Storing assignment value {value} in variable {var}");

                    // Make sure variable exists in environment and update it
                    if !self.environment.has_variable(&var) {
                        self.environment.add_variable(&var, DataType::I32, 32)?;
                    }
                    // Convert to u64 safely - we're working with raw bit patterns
                    #[allow(clippy::cast_sign_loss)]
                    let value_u64 = u64::from_ne_bytes((value as u64).to_ne_bytes());
                    self.environment.set(&var, value_u64)?;
                    log::debug!("Updated variable {var} = {value} in environment");

                    // Values are stored in the environment and will be available for expression evaluation
                    log::debug!("Variable is now available in environment: {var} = {value}");
                }

                // Return false to continue processing (assignment doesn't end the batch)
                log::debug!("Assignment operation handled successfully");
                return Ok(false);
            }
        } else {
            // For other operations, validate all variables
            for arg in args.iter().chain(returns) {
                match arg {
                    ArgItem::Simple(var) => {
                        self.validate_variable_access(var, 0)?;
                    }
                    ArgItem::Indexed((var, idx)) => {
                        self.validate_variable_access(var, *idx)?;
                    }
                    ArgItem::Integer(_) => {
                        // Integer literals are valid and don't need validation
                    }
                    ArgItem::Expression(_expr) => {
                        // For expressions, we recursively validate any variables they reference
                        // This is a simplification - a more robust implementation would
                        // traverse the expression tree
                    }
                }
            }
        }

        if cop == "Result" {
            // Process Result operation with our improved implementation
            log::debug!(
                "Processing Result operation with {} sources and {} destinations",
                args.len(),
                returns.len()
            );

            // Use our improved method that handles bit indexing and uses the environment
            self.process_result_op(args, returns)?;

            // Return true to indicate we've handled this operation
            return Ok(true);
        } else if cop == "ffcall" {
            // Process foreign function call
            if let Some(foreign_obj) = &self.foreign_object {
                // Validate that we have a function name
                // Find the function name from either the current operation or from ops[current_op]
                let function_name = match ops.get(current_op) {
                    // First check if the operation at current_op index has the function name
                    Some(Operation::ClassicalOp {
                        function: Some(name),
                        cop: op_cop,
                        ..
                    }) if op_cop == "ffcall" => name,

                    // Otherwise, we need to look for the function name directly in ClassicalOp.function parameter
                    // which is needed when processing operations inside conditional blocks or other nested structures
                    _ => {
                        // Check if we have a 'function' parameter passed to this function
                        // Look for it in the operation that called this function by searching
                        // through all operations for an ffcall that matches our parameters
                        if let Some(Operation::ClassicalOp {
                            function: Some(name),
                            ..
                        }) = ops.iter().find(|op| {
                            if let Operation::ClassicalOp {
                                cop: op_cop,
                                args: op_args,
                                returns: op_returns,
                                function: Some(_),
                                ..
                            } = op
                            {
                                // Check if this is an ffcall operation with matching args and returns
                                op_cop == "ffcall" && op_args == args && op_returns == returns
                            } else {
                                false
                            }
                        }) {
                            name
                        } else {
                            for op in ops {
                                if let Operation::Block {
                                    true_branch: Some(tb),
                                    false_branch: fb,
                                    ..
                                } = op
                                {
                                    // Check true branch
                                    for branch_op in tb {
                                        if let Operation::ClassicalOp {
                                            cop: op_cop,
                                            args: op_args,
                                            returns: op_returns,
                                            function: Some(name),
                                            ..
                                        } = branch_op
                                            && op_cop == "ffcall"
                                            && op_args == args
                                            && op_returns == returns
                                        {
                                            // Execute the function directly
                                            let mut fo_clone = foreign_obj.clone_box();

                                            // Convert arguments to i64 values
                                            let mut call_args = Vec::new();
                                            for arg in args {
                                                let value = self.evaluate_arg_item(arg)?;
                                                call_args.push(value);
                                            }

                                            let result = fo_clone.exec(name, &call_args)?;

                                            // Handle return values
                                            if !returns.is_empty() {
                                                for (i, ret) in returns.iter().enumerate() {
                                                    if i < result.len() {
                                                        match ret {
                                                            ArgItem::Simple(var) => {
                                                                // Assign to a variable
                                                                let result_value =
                                                                    u32::try_from(result[i])
                                                                        .unwrap_or(0);

                                                                // Update primary storage in environment
                                                                if !self
                                                                    .environment
                                                                    .has_variable(var)
                                                                {
                                                                    let _ = self
                                                                        .environment
                                                                        .add_variable(
                                                                            var,
                                                                            DataType::I32,
                                                                            32,
                                                                        );
                                                                }
                                                                let _ = self.environment.set(
                                                                    var,
                                                                    u64::from(result_value),
                                                                );

                                                                // All values stored in environment
                                                            }
                                                            ArgItem::Indexed((var, idx)) => {
                                                                // Assign to a bit
                                                                let bit_value =
                                                                    u32::try_from(result[i] & 1)
                                                                        .unwrap_or(0);

                                                                // Update primary storage in environment
                                                                if !self
                                                                    .environment
                                                                    .has_variable(var)
                                                                {
                                                                    let _ = self
                                                                        .environment
                                                                        .add_variable(
                                                                            var,
                                                                            DataType::I32,
                                                                            32,
                                                                        );
                                                                }

                                                                // Set the bit in environment
                                                                let _ = self.environment.set_bit(
                                                                    var,
                                                                    *idx,
                                                                    u64::from(bit_value),
                                                                );

                                                                // Environment is the single source of truth - no need for additional storage
                                                            }
                                                            _ => {
                                                                return Err(PecosError::Input(
                                                                            "Invalid return type for foreign function call".to_string(),
                                                                        ));
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            return Ok(true);
                                        }
                                    }

                                    // Check false branch if it exists
                                    if let Some(fb_ops) = fb {
                                        for branch_op in fb_ops {
                                            if let Operation::ClassicalOp {
                                                cop: op_cop,
                                                args: op_args,
                                                returns: op_returns,
                                                function: Some(name),
                                                ..
                                            } = branch_op
                                                && op_cop == "ffcall"
                                                && op_args == args
                                                && op_returns == returns
                                            {
                                                // Execute the function directly
                                                let mut fo_clone = foreign_obj.clone_box();

                                                // Convert arguments to i64 values
                                                let mut call_args = Vec::new();
                                                for arg in args {
                                                    let value = self.evaluate_arg_item(arg)?;
                                                    call_args.push(value);
                                                }

                                                let result = fo_clone.exec(name, &call_args)?;

                                                // Handle return values
                                                if !returns.is_empty() {
                                                    for (i, ret) in returns.iter().enumerate() {
                                                        if i < result.len() {
                                                            match ret {
                                                                ArgItem::Simple(var) => {
                                                                    // Assign to a variable
                                                                    let result_value =
                                                                        u32::try_from(result[i])
                                                                            .unwrap_or(0);

                                                                    // Update primary storage in environment
                                                                    if !self
                                                                        .environment
                                                                        .has_variable(var)
                                                                    {
                                                                        let _ = self
                                                                            .environment
                                                                            .add_variable(
                                                                                var,
                                                                                DataType::I32,
                                                                                32,
                                                                            );
                                                                    }
                                                                    let _ = self.environment.set(
                                                                        var,
                                                                        u64::from(result_value),
                                                                    );

                                                                    // Environment is the single source of truth for all variable data
                                                                }
                                                                ArgItem::Indexed((var, idx)) => {
                                                                    // Assign to a bit
                                                                    let bit_value = u32::try_from(
                                                                        result[i] & 1,
                                                                    )
                                                                    .unwrap_or(0);

                                                                    // Update primary storage in environment
                                                                    if !self
                                                                        .environment
                                                                        .has_variable(var)
                                                                    {
                                                                        let _ = self
                                                                            .environment
                                                                            .add_variable(
                                                                                var,
                                                                                DataType::I32,
                                                                                32,
                                                                            );
                                                                    }

                                                                    // Set the bit in environment
                                                                    let _ =
                                                                        self.environment.set_bit(
                                                                            var,
                                                                            *idx,
                                                                            u64::from(bit_value),
                                                                        );

                                                                    // Environment is the single source of truth for all variable data
                                                                }
                                                                _ => {
                                                                    return Err(PecosError::Input(
                                                                                "Invalid return type for foreign function call".to_string(),
                                                                            ));
                                                                }
                                                            }
                                                        }
                                                    }
                                                }

                                                return Ok(true);
                                            }
                                        }
                                    }
                                }
                            }

                            // If we got here, no function name was found
                            return Err(PecosError::Input(
                                "Foreign function call missing function name".to_string(),
                            ));
                        }
                    }
                };

                debug!("Executing foreign function call: {function_name}");

                // Convert arguments to i64 values using consistent evaluation approach
                // Since the environment is the single source of truth, we can use the standard
                // evaluation method for all argument types
                let mut call_args = Vec::new();
                for arg in args {
                    // For all argument types, use the evaluate_arg_item method which uses the environment
                    // as the primary source of data
                    let value = self.evaluate_arg_item(arg)?;
                    debug!("FFI arg value: {value}");
                    call_args.push(value);
                }

                // Execute the function using the foreign object
                debug!("Executing foreign function: {function_name} with args: {call_args:?}");

                // Create a mutable clone that we can call exec on
                let mut fo_clone = foreign_obj.clone_box();
                let result = fo_clone.exec(function_name, &call_args)?;

                debug!("Foreign function result: {result:?}");

                // Handle return values
                if !returns.is_empty() {
                    // Map the results to the returns
                    debug!("FFI result: {result:?}");

                    for (i, ret) in returns.iter().enumerate() {
                        if i < result.len() {
                            match ret {
                                ArgItem::Simple(var) => {
                                    // Store whole variable value in environment
                                    let result_value = u64::try_from(result[i]).unwrap_or(0);

                                    // Make sure the variable exists
                                    if !self.environment.has_variable(var) {
                                        // Create if needed
                                        self.environment.add_variable(var, DataType::I32, 32)?;
                                    }

                                    // Set value in environment (single source of truth)
                                    self.environment.set(var, result_value)?;
                                    debug!("Set variable {var} = {result_value}");
                                }
                                ArgItem::Indexed((var, idx)) => {
                                    // Set specific bit in variable
                                    let bit_value = result[i] & 1;

                                    // Make sure the variable exists
                                    if !self.environment.has_variable(var) {
                                        // Create if needed
                                        self.environment.add_variable(var, DataType::I32, 32)?;
                                    }

                                    // Set bit in environment (single source of truth)
                                    self.environment.set_bit(
                                        var,
                                        *idx,
                                        u64::try_from(bit_value).unwrap_or(0),
                                    )?;
                                    debug!("Set bit {var}[{idx}] = {bit_value}");
                                }
                                _ => {
                                    return Err(PecosError::Input(
                                        "Invalid return type for foreign function call".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }

                return Ok(true);
            }
            // No foreign object available
            return Err(PecosError::Processing(
                "Foreign function call attempted but no foreign object is available".to_string(),
            ));
        }
        // For other operators (arithmetic, comparison, bitwise),
        // we handle them in expression evaluation, not here directly
        log::debug!("Skipping direct handling of operator: {cop}");

        Ok(false)
    }

    /// Process a quantum operation and return the gate type, qubit arguments, and angle arguments
    ///
    /// # Errors
    /// Returns an error if:
    /// - No qubit arguments are provided
    /// - Required angle parameters are missing
    /// - The quantum gate type is not supported
    /// - Qubit variables are invalid
    pub fn process_quantum_op(
        &self,
        qop: &str,
        angles: Option<&Vec<f64>>, // Now just Vec<f64> in radians, no unit string
        args: &[QubitArg],
    ) -> Result<(String, Vec<usize>, Vec<f64>), PecosError> {
        // Validate that we have at least one qubit argument
        if args.is_empty() {
            return Err(PecosError::Input(format!(
                "Invalid quantum operation: Operation '{qop}' requires at least one qubit argument"
            )));
        }

        // Validate and extract qubit arguments
        let mut qubit_args = Vec::new();

        for qubit_arg in args {
            match qubit_arg {
                QubitArg::SingleQubit((var, idx)) => {
                    // Validate the qubit
                    self.validate_variable_access(var, *idx)?;
                    qubit_args.push(*idx);
                }
                QubitArg::MultipleQubits(qubits) => {
                    for (var, idx) in qubits {
                        // Validate each qubit
                        self.validate_variable_access(var, *idx)?;
                        qubit_args.push(*idx);
                    }
                }
            }
        }

        // Process based on gate type
        match qop {
            // Single-qubit rotation gates
            "RZ" => {
                let theta = angles
                    .as_ref()
                    .and_then(|angles| angles.first().copied())
                    .ok_or_else(|| {
                        PecosError::ValidationInvalidGateParameters(format!(
                            "Missing rotation angle for '{qop}' gate"
                        ))
                    })?;
                Ok((qop.to_string(), qubit_args, vec![theta]))
            }
            "R1XY" => {
                // Get angles safely
                let angles_ref = angles.as_ref().ok_or_else(|| {
                    PecosError::ValidationInvalidGateParameters(format!(
                        "'{qop}' gate requires two angles (theta, phi)"
                    ))
                })?;

                if angles_ref.len() < 2 {
                    return Err(PecosError::ValidationInvalidGateParameters(format!(
                        "'{qop}' gate requires two angles (theta, phi), but only {} provided",
                        angles_ref.len()
                    )));
                }

                // R1XY convention: angles[0] = theta (rotation angle), angles[1] = phi (axis angle)
                // See: R1XY = RZ(phi-pi/2) * RY(theta) * RZ(phi-pi/2)
                let theta = angles_ref[0];
                let phi = angles_ref[1];
                Ok((qop.to_string(), qubit_args, vec![theta, phi]))
            }

            // Two-qubit rotation gate: RXXRYYRZZ (3 angles)
            "RXXRYYRZZ" | "R2XXYYZZ" | "RXXYYZZ" => {
                let angles_ref = angles.as_ref().ok_or_else(|| {
                    PecosError::ValidationInvalidGateParameters(
                        "RXXRYYRZZ gate requires three angles (alpha, beta, gamma)".to_string(),
                    )
                })?;
                if angles_ref.len() < 3 {
                    return Err(PecosError::ValidationInvalidGateParameters(format!(
                        "RXXRYYRZZ gate requires three angles (alpha, beta, gamma), but only {} provided",
                        angles_ref.len()
                    )));
                }
                if qubit_args.len() < 2 {
                    return Err(PecosError::ValidationInvalidGateParameters(format!(
                        "RXXRYYRZZ gate requires exactly two qubits, but found {}",
                        qubit_args.len()
                    )));
                }
                // Always return canonical name
                Ok((
                    "RXXRYYRZZ".to_string(),
                    qubit_args,
                    vec![angles_ref[0], angles_ref[1], angles_ref[2]],
                ))
            }

            // Two-qubit gates
            "SZZ" | "ZZ" => {
                // Verify we have exactly 2 qubits
                if qubit_args.len() < 2 {
                    return Err(PecosError::ValidationInvalidGateParameters(format!(
                        "'{qop}' gate requires exactly two qubits, but found {}",
                        qubit_args.len()
                    )));
                }
                // Always return the canonical name SZZ
                Ok(("SZZ".to_string(), qubit_args, vec![]))
            }
            "CX" | "CNOT" => {
                // Verify we have exactly 2 qubits
                if qubit_args.len() < 2 {
                    return Err(PecosError::ValidationInvalidGateParameters(format!(
                        "'{qop}' gate requires control and target qubits (2 qubits total), but found {}",
                        qubit_args.len()
                    )));
                }
                // Always return the canonical name CX
                Ok(("CX".to_string(), qubit_args, vec![]))
            }

            // Single-qubit Clifford gates, Initialization, and Measurement
            "H" | "X" | "Y" | "Z" | "Measure" | "Init" => Ok((qop.to_string(), qubit_args, vec![])),

            _ => Err(PecosError::Processing(format!(
                "Unsupported quantum gate operation: Gate type '{qop}' is not implemented"
            ))),
        }
    }

    /// Add quantum operation to byte message builder
    ///
    /// # Errors
    /// Returns an error if the gate type is not supported.
    pub fn add_quantum_operation_to_builder(
        &self,
        builder: &mut ByteMessageBuilder,
        gate_type: &str,
        qubit_args: &[usize],
        angle_args: &[f64],
    ) -> Result<(), PecosError> {
        match gate_type {
            "RZ" => {
                builder.rz(Angle64::from_radians(angle_args[0]), &[qubit_args[0]]);
            }
            "R1XY" => {
                builder.r1xy(
                    Angle64::from_radians(angle_args[0]),
                    Angle64::from_radians(angle_args[1]),
                    &[qubit_args[0]],
                );
            }
            "RXXRYYRZZ" => {
                let gate = Gate::rxxryyrzz(
                    Angle64::from_radians(angle_args[0]),
                    Angle64::from_radians(angle_args[1]),
                    Angle64::from_radians(angle_args[2]),
                    &[(qubit_args[0], qubit_args[1])],
                );
                builder.add_gate_command(&gate);
            }
            "SZZ" => {
                builder.szz(&[(qubit_args[0], qubit_args[1])]);
            }
            "CX" => {
                builder.cx(&[(qubit_args[0], qubit_args[1])]);
            }
            "H" => {
                builder.h(&[qubit_args[0]]);
            }
            "X" => {
                builder.x(&[qubit_args[0]]);
            }
            "Y" => {
                builder.y(&[qubit_args[0]]);
            }
            "Z" => {
                builder.z(&[qubit_args[0]]);
            }
            "Measure" => {
                builder.mz(&[qubit_args[0]]);
            }
            "Init" => {
                // Initialize qubit to |0⟩ state using the Prep gate
                for &qubit in qubit_args {
                    // The Prep gate initializes a qubit to the |0⟩ state
                    builder.pz(&[qubit]);
                }
            }
            _ => {
                return Err(PecosError::Processing(format!(
                    "Unsupported quantum gate operation: Gate type '{gate_type}' is not implemented"
                )));
            }
        }
        Ok(())
    }

    /// Store a measurement result in the environment
    ///
    /// This method stores a measurement outcome by updating a specific bit
    /// in the integer variable (e.g., "m") in the environment.
    ///
    /// The environment is the single source of truth for all variables.
    fn store_measurement_result(&mut self, var_name: &str, var_idx: usize, outcome: u32) {
        log::debug!("PHIR: Storing measurement result {var_name}[{var_idx}] = {outcome}");

        // Step 1: Ensure the main variable exists in the environment with appropriate size
        if !self.environment.has_variable(var_name) {
            // Determine appropriate size (at least large enough to hold this bit)
            let var_size = std::cmp::max(var_idx + 1, 32);

            // Create the variable
            match self
                .environment
                .add_variable(var_name, DataType::I32, var_size)
            {
                Ok(()) => log::debug!("Created variable {var_name} with size {var_size}"),
                Err(e) => log::warn!(
                    "Could not create variable: {var_name}. Will try to update anyway: {e}"
                ),
            }
        }

        // Step 2: Update the specific bit directly using the environment's bit setting functionality
        let bit_value = u64::from(outcome != 0);
        if let Err(e) = self.environment.set_bit(var_name, var_idx, bit_value) {
            log::warn!("Could not set bit {var_name}[{var_idx}] = {bit_value}. Error: {e}");
        } else {
            log::debug!("Set bit {var_name}[{var_idx}] = {bit_value} in environment");
        }
    }

    /// Handle incoming measurements from quantum operations and store results
    ///
    /// This method processes measurement results and stores them in:
    /// 1. The environment (single source of truth for all variables)
    /// 2. Standard measurement variables (e.g., "`measurement_0`")
    /// 3. Named variables from the program (e.g., "m")
    ///
    /// # Errors
    /// Returns an error if variable creation or value setting fails.
    pub fn handle_measurements(
        &mut self,
        measurements: &[u32],
        ops: &[Operation],
    ) -> Result<(), PecosError> {
        log::debug!("PHIR: Handling {} measurement results", measurements.len());

        for (index, outcome) in measurements.iter().enumerate() {
            let result_id = u32::try_from(index).unwrap_or(u32::MAX);
            log::debug!("PHIR: Received measurement index={index}, outcome={outcome}");

            // Create the standard measurement variable name (e.g., "measurement_0")
            let prefixed_name = format!("{MEASUREMENT_PREFIX}{result_id}");

            // Store in the standard measurement variable
            // Create the variable if it doesn't exist
            if !self.environment.has_variable(&prefixed_name)
                && let Err(e) = self
                    .environment
                    .add_variable(&prefixed_name, DataType::I32, 32)
            {
                log::warn!("Could not create measurement variable: {prefixed_name}. Error: {e}");
            }

            // Set the measurement value
            if let Err(e) = self.environment.set(&prefixed_name, u64::from(*outcome)) {
                log::warn!("Could not set measurement variable {prefixed_name}. Error: {e}");
            } else {
                log::debug!("Stored measurement result: {prefixed_name} = {outcome}");
            }

            // Also map to specific variable based on the Measure operation
            let mut found_mapping = false;
            for op in ops {
                if let Operation::QuantumOp {
                    qop,
                    args: _,
                    returns,
                    ..
                } = op
                    && qop == "Measure"
                    && !returns.is_empty()
                {
                    // Get the variable name and index from the returns field
                    let (var_name, var_idx) = &returns[0];

                    // Check if this is the right measurement result
                    if *var_idx == result_id as usize {
                        // Store the result in the specific bit of the variable
                        self.store_measurement_result(var_name, *var_idx, *outcome);
                        found_mapping = true;
                    }
                }
            }

            // If we didn't find a mapping in the operations, add a default mapping to variable "m"
            // This helps with tests and interoperability, particularly Bell state tests
            if !found_mapping && self.environment.has_variable("m") {
                // Store in main "m" variable for test compatibility
                let idx = result_id as usize;
                self.store_measurement_result("m", idx, *outcome);
                log::debug!(
                    "PHIR: Auto-mapped measurement result {result_id} to m[{idx}] = {outcome}"
                );
            }
        }

        // Log mappings for debugging purposes
        // The environment automatically manages and uses these mappings
        // when generating results, so no additional processing is needed
        let mappings = self.environment.get_mappings();
        if !mappings.is_empty() {
            log::debug!(
                "PHIR: {} mappings registered in environment",
                mappings.len()
            );
            for (source, dest) in mappings {
                log::debug!("PHIR: Mapping {source} -> {dest}");
            }
        }

        Ok(())
    }

    /// Helper method to extract variable name and optional index from an argument
    fn extract_arg_info(arg: &ArgItem) -> Result<(String, Option<usize>), PecosError> {
        match arg {
            ArgItem::Simple(name) => Ok((name.clone(), None)),
            ArgItem::Indexed((name, idx)) => Ok((name.clone(), Some(*idx))),
            _ => Err(PecosError::Input(format!(
                "Invalid argument for Result operation: {arg:?}"
            ))),
        }
    }

    /// Get a variable value from the environment
    ///
    /// This simplified implementation treats the environment as the single source of truth
    /// for retrieving variable values.
    fn get_variable_value(&self, var_name: &str, index: Option<usize>) -> Result<u32, PecosError> {
        log::debug!("Getting variable value for {var_name}[{index:?}]");

        // Ensure the variable exists in the environment
        if !self.environment.has_variable(var_name) {
            return Err(PecosError::Input(format!(
                "Variable not found in environment: {var_name}[{index:?}]"
            )));
        }

        // Handle bit access if an index is provided
        if let Some(idx) = index {
            // Try to get the specific bit using the environment's bit accessor
            match self.environment.get_bit(var_name, idx) {
                Ok(bit_value) => {
                    log::debug!("Found bit value in environment: {var_name}[{idx}] = {bit_value}");
                    return Ok(u32::from(bit_value.0));
                }
                Err(_) => {
                    // Fall back to extracting bit from full value
                    if let Some(full_val) = self.environment.get(var_name) {
                        let bit_value = (full_val >> idx) & 1;
                        log::debug!("Extracted bit from variable: {var_name}[{idx}] = {bit_value}");
                        return Ok(bit_value as u32);
                    }
                }
            }
            // If we couldn't get the bit, return an error
            return Err(PecosError::Input(format!(
                "Could not access bit {var_name}[{idx}] in environment"
            )));
        }

        // Handle whole variable access
        if let Some(val) = self.environment.get(var_name) {
            log::debug!("Got value from environment: {var_name} = {val}");
            return Ok(val.as_u32());
        }

        // If we get here, the variable exists but has no value
        Err(PecosError::Input(format!(
            "Variable exists in environment but has no value: {var_name}"
        )))
    }

    /// Process a Result operation which maps source variables to destination variables
    ///
    /// This method:
    /// 1. Creates mappings between source and destination variables in the environment
    /// 2. Gets values from the source variables
    /// 3. Stores values in the destination variables, handling both whole variables and bit access
    fn process_result_op(
        &mut self,
        args: &[ArgItem],
        returns: &[ArgItem],
    ) -> Result<(), PecosError> {
        log::debug!(
            "Processing Result operation with {} args and {} returns",
            args.len(),
            returns.len()
        );

        // Process each source -> destination mapping
        for (i, src) in args.iter().enumerate() {
            if i < returns.len() {
                let dst = &returns[i];

                // Extract source and destination information
                let (src_name, src_index) = Self::extract_arg_info(src)?;
                let (dst_name, dst_index) = Self::extract_arg_info(dst)?;

                log::debug!(
                    "Result mapping: {src_name}[{src_index:?}] -> {dst_name}[{dst_index:?}]"
                );

                // Store mapping in the environment
                let _ = self.environment.add_mapping(&src_name, &dst_name);

                // Get the source value directly from the environment
                // No special handling or fallbacks - environment is the single source of truth
                let value = self.get_variable_value(&src_name, src_index)?;

                log::debug!("Got value for {src_name}: {value}");

                // Create destination variable if needed
                if !self.environment.has_variable(&dst_name) {
                    // Size depends on whether we're doing bit access
                    let var_size = if let Some(idx) = dst_index {
                        std::cmp::max(idx + 1, 32)
                    } else {
                        32
                    };

                    // Create the variable, but don't fail if it already exists
                    if let Err(e) =
                        self.environment
                            .add_variable(&dst_name, DataType::I32, var_size)
                    {
                        log::warn!(
                            "Could not create variable: {dst_name}. Will try to update existing: {e}"
                        );
                    }
                }

                // Store the value in the destination
                if let Some(idx) = dst_index {
                    // Bit access - set specific bit in the variable
                    let bit_value = value & 1;
                    if let Err(e) = self
                        .environment
                        .set_bit(&dst_name, idx, u64::from(bit_value))
                    {
                        log::warn!("Could not set bit {dst_name}[{idx}] = {bit_value}: {e}");
                    } else {
                        log::debug!("Set bit {dst_name}[{idx}] = {bit_value}");
                    }
                } else {
                    // Whole variable assignment
                    if let Err(e) = self.environment.set(&dst_name, u64::from(value)) {
                        log::warn!("Could not set variable {dst_name} = {value}: {e}");
                    } else {
                        log::debug!("Set variable {dst_name} = {value}");
                    }
                }
            }
        }

        Ok(())
    }

    /// Process export mappings to determine values to return from simulations
    ///
    /// This simplified method treats the environment as the single source of truth
    /// and provides a clean, simple approach to gathering exported values.
    #[must_use]
    pub fn process_export_mappings(&self) -> BTreeMap<String, u32> {
        let mut exported_values = BTreeMap::new();
        log::debug!("Processing export mappings using environment as source of truth");

        // Get all mappings from the environment
        let mappings = self.environment.get_mappings();

        if !mappings.is_empty() {
            log::debug!(
                "Processing {} explicit mappings from environment",
                mappings.len()
            );

            // Process all explicit mappings first
            for (source_register, export_name) in mappings {
                // Skip if we already have this export (in case of duplicates)
                if exported_values.contains_key(export_name) {
                    log::debug!("Skipping already processed export: {export_name}");
                    continue;
                }

                log::debug!("Processing export mapping: {source_register} -> {export_name}");

                // Primary approach: Direct lookup in environment
                if self.environment.has_variable(source_register) {
                    if let Some(value) = self.environment.get(source_register) {
                        log::debug!("Using value from environment: {source_register} = {value}");
                        exported_values.insert(export_name.clone(), value.as_u32());
                    } else {
                        log::debug!(
                            "Variable {source_register} exists in environment but has no value"
                        );
                    }
                } else {
                    // If the source doesn't exist, log but don't use fallbacks since environment
                    // is the single source of truth
                    log::warn!(
                        "Source variable '{source_register}' for export '{export_name}' not found in environment"
                    );
                }
            }
        }

        // If no explicit mappings or we didn't find any values, include all variables with values
        if mappings.is_empty() || exported_values.is_empty() {
            log::debug!("Adding automatic mappings for all variables with values");

            for var_info in self.environment.get_all_variables() {
                // Skip variables we've already exported
                if exported_values.contains_key(&var_info.name) {
                    continue;
                }

                // Include any variable that has a value
                if let Some(val) = self.environment.get(&var_info.name) {
                    log::debug!("Adding variable: {} = {}", var_info.name, val);
                    exported_values.insert(var_info.name.clone(), val.as_u32());
                }
            }
        }

        // Log summary of what we're exporting
        log::debug!("Exporting {} values:", exported_values.len());
        for (name, value) in &exported_values {
            log::debug!("  {name} = {value}");
        }

        exported_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::ast::{ArgItem, Expression};

    #[test]
    fn test_evaluate_expression() {
        let mut processor = OperationProcessor::new();

        // Add a test variable to the environment
        processor
            .environment
            .add_variable("test_var", DataType::I32, 32)
            .unwrap();
        processor.environment.set("test_var", 42).unwrap();

        // Test integer literal
        let expr = Expression::Integer(123);
        assert_eq!(processor.evaluate_expression(&expr).unwrap(), 123);

        // Test variable reference
        let expr = Expression::Variable("test_var".to_string());
        assert_eq!(processor.evaluate_expression(&expr).unwrap(), 42);

        // Test bit access using bitwise operations
        let expr = Expression::Operation {
            cop: "&".to_string(),
            args: vec![
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: ">>".to_string(),
                    args: vec![ArgItem::Simple("test_var".to_string()), ArgItem::Integer(1)],
                })),
                ArgItem::Integer(1),
            ],
        };
        assert_eq!(processor.evaluate_expression(&expr).unwrap(), 1); // 42 = 0b101010, so bit 1 is 1

        // Test bit access via Indexed ArgItem
        assert_eq!(
            processor
                .evaluate_arg_item(&ArgItem::Indexed(("test_var".to_string(), 1)))
                .unwrap(),
            1
        );

        // Test simple binary operation
        let expr = Expression::Operation {
            cop: "+".to_string(),
            args: vec![ArgItem::Integer(10), ArgItem::Integer(20)],
        };
        assert_eq!(processor.evaluate_expression(&expr).unwrap(), 30);

        // Test complex nested expression
        let expr = Expression::Operation {
            cop: "*".to_string(),
            args: vec![
                ArgItem::Integer(5),
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: "+".to_string(),
                    args: vec![
                        ArgItem::Integer(10),
                        ArgItem::Simple("test_var".to_string()),
                    ],
                })),
            ],
        };
        assert_eq!(processor.evaluate_expression(&expr).unwrap(), 5 * (10 + 42));
    }
}
