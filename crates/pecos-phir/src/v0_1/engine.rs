use crate::v0_1::ast::{Operation, PHIRProgram};
use crate::v0_1::foreign_objects::ForeignObject;
use crate::v0_1::operations::OperationProcessor;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::{ByteMessage, builder::ByteMessageBuilder};
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage};
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// `PHIREngine` processes PHIR programs and generates quantum operations
#[derive(Debug)]
pub struct PHIREngine {
    /// The loaded PHIR program
    program: Option<PHIRProgram>,
    /// Current operation index being processed
    current_op: usize,
    /// Operation processor for handling different operation types
    pub processor: OperationProcessor,
    /// Builder for constructing `ByteMessages`
    message_builder: ByteMessageBuilder,
}

impl PHIREngine {
    /// Sets a foreign object for executing foreign function calls
    pub fn set_foreign_object(&mut self, foreign_object: Box<dyn ForeignObject>) {
        self.processor.set_foreign_object(foreign_object);
    }

    /// Creates a new instance of `PHIREngine` by loading a PHIR program JSON file.
    ///
    /// # Parameters
    /// - `path`: A reference to the path of the PHIR program JSON file to load.
    ///
    /// # Returns
    /// - `Ok(Self)`: If the PHIR program file is successfully loaded and validated.
    /// - `Err(PecosError)`: If any errors occur during file reading,
    ///   parsing, or if the format/version is not compatible.
    ///
    /// # Errors
    /// - Returns an error if the file cannot be read.
    /// - Returns an error if the JSON parsing fails.
    /// - Returns an error if the format is not "PHIR/JSON".
    /// - Returns an error if the version is not "0.1.0".
    ///
    /// # Examples
    /// ```rust
    /// use pecos_phir::v0_1::engine::PHIREngine;
    ///
    /// let engine = PHIREngine::new("path_to_program.json");
    /// match engine {
    ///     Ok(engine) => println!("PHIREngine loaded successfully!"),
    ///     Err(e) => eprintln!("Error loading PHIREngine: {}", e),
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        let content = std::fs::read_to_string(path).map_err(PecosError::IO)?;
        Self::from_json(&content)
    }

    /// Creates a new instance of `PHIREngine` from a JSON string.
    ///
    /// # Parameters
    /// - `json_str`: A string containing the PHIR program in JSON format.
    ///
    /// # Returns
    /// - `Ok(Self)`: If the PHIR program is successfully parsed and validated.
    /// - `Err(PecosError)`: If any errors occur during parsing,
    ///   or if the format/version is not compatible.
    ///
    /// # Errors
    /// - Returns an error if the JSON parsing fails.
    /// - Returns an error if the format is not "PHIR/JSON".
    /// - Returns an error if the version is not "0.1.0".
    ///
    /// # Examples
    /// ```rust
    /// use pecos_phir::v0_1::engine::PHIREngine;
    ///
    /// let json = r#"{"format":"PHIR/JSON","version":"0.1.0","metadata":{},"ops":[]}"#;
    /// let engine = PHIREngine::from_json(json);
    /// match engine {
    ///     Ok(engine) => println!("PHIREngine loaded successfully!"),
    ///     Err(e) => eprintln!("Error loading PHIREngine: {}", e),
    /// }
    /// ```
    pub fn from_json(json_str: &str) -> Result<Self, PecosError> {
        let program: PHIRProgram = serde_json::from_str(json_str).map_err(|e| {
            PecosError::Input(format!(
                "Failed to parse PHIR program: Invalid JSON format: {e}"
            ))
        })?;

        if program.format != "PHIR/JSON" {
            return Err(PecosError::Input(format!(
                "Invalid PHIR program format: found '{}', expected 'PHIR/JSON'",
                program.format
            )));
        }

        if program.version != "0.1.0" {
            return Err(PecosError::Input(format!(
                "Unsupported PHIR version: found '{}', only version '0.1.0' is supported",
                program.version
            )));
        }

        // Validate that at least one Result command exists
        let has_result_command = program.ops.iter().any(|op| {
            if let Operation::ClassicalOp { cop, .. } = op {
                cop == "Result"
            } else {
                false
            }
        });

        if !has_result_command {
            return Err(PecosError::Input(
                "Invalid PHIR program structure: Program must contain at least one Result command to specify outputs"
                    .to_string(),
            ));
        }

        log::debug!("Loading PHIR program with metadata: {:?}", program.metadata);

        // Initialize operation processor and extract variable definitions
        let mut processor = OperationProcessor::new();

        // Process variable definitions
        for op in &program.ops {
            if let Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } = op
            {
                let _ = processor.handle_variable_definition(data, data_type, variable, *size);
            }
        }

        Ok(Self {
            program: Some(program),
            current_op: 0,
            processor,
            message_builder: ByteMessageBuilder::new(),
        })
    }

    /// Creates a new instance of `PHIREngine` from a parsed `PHIRProgram`.
    ///
    /// # Parameters
    /// - `program`: A `PHIRProgram` instance.
    ///
    /// # Returns
    /// - Returns a new `PHIREngine` initialized with the provided program.
    ///
    /// # Errors
    /// - Returns an error if variable definitions cannot be processed.
    pub fn from_program(program: PHIRProgram) -> Result<Self, PecosError> {
        let mut processor = OperationProcessor::new();

        // Process variable definitions
        for op in &program.ops {
            if let Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } = op
            {
                processor.handle_variable_definition(data, data_type, variable, *size)?;
            }
        }

        Ok(Self {
            program: Some(program),
            current_op: 0,
            processor,
            message_builder: ByteMessageBuilder::new(),
        })
    }

    /// Resets the engine state
    ///
    /// Simplified reset that treats the environment as the single source of truth.
    /// This no longer preserves and restores variable values during reset, as they
    /// should be recomputed during program execution.
    fn reset_state(&mut self) {
        debug!(
            "INTERNAL RESET: PHIREngine reset, current_op={}",
            self.current_op
        );

        // Reset the operation index to start from the beginning
        self.current_op = 0;

        // Log operations for debugging if needed
        if log::log_enabled!(log::Level::Debug) && self.program.is_some() {
            let program = self.program.as_ref().unwrap();
            debug!("Operations to process after reset: {}", program.ops.len());
        }

        // Reset the processor state (maintains variable definitions but clears values)
        // This is now a clean reset without preserving values, since the environment
        // is the single source of truth and values should be recomputed as needed
        self.processor.reset();

        // Reset the message builder to reuse allocated memory
        self.message_builder.reset();

        debug!("PHIREngine reset complete, ready for next execution");
    }

    // Create an empty engine without any program
    fn empty() -> Self {
        Self {
            program: None,
            current_op: 0,
            processor: OperationProcessor::new(),
            message_builder: ByteMessageBuilder::new(),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn generate_commands_impl(&mut self) -> Result<Option<ByteMessage>, PecosError> {
        // Define a maximum batch size for better performance
        // This helps avoid creating excessively large messages
        const MAX_BATCH_SIZE: usize = 100;

        debug!("generate_commands called, current_op: {}", self.current_op);

        debug!(
            "Generating commands - thread {:?}, current_op: {}",
            std::thread::current().id(),
            self.current_op
        );

        // Get program reference and clone ops to avoid borrow issues
        let prog = self.program.as_ref().ok_or_else(|| {
            PecosError::Resource("Cannot generate commands: No PHIR program loaded".to_string())
        })?;
        let ops = prog.ops.clone();

        // If we've processed all ops, return None to signal completion
        if self.current_op >= ops.len() {
            debug!(
                "End of program reached at op {}, no more commands to generate",
                self.current_op
            );

            // With our updated HybridEngine and ControlEngine implementations,
            // we can now consistently return None when there are no more commands,
            // even for the first batch.
            return Ok(None);
        }

        debug!(
            "Current operation to process: {} - {:?}",
            self.current_op, ops[self.current_op]
        );

        // Reset and configure the reusable message builder for quantum operations
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();
        let mut operation_count = 0;

        while self.current_op < ops.len() && operation_count < MAX_BATCH_SIZE {
            match &ops[self.current_op] {
                Operation::VariableDefinition {
                    data,
                    data_type,
                    variable,
                    size,
                } => {
                    debug!("Processing variable definition: {data} {data_type} {variable}");
                    let _ = self
                        .processor
                        .handle_variable_definition(data, data_type, variable, *size);
                    self.current_op += 1;
                    return self.generate_commands_impl();
                }
                Operation::QuantumOp {
                    qop,
                    angles,
                    args,
                    returns: _,
                    metadata: _,
                } => {
                    debug!("Processing quantum operation: {qop}");

                    // Clone the operation parameters to avoid borrow issues
                    let qop_str = qop.clone();
                    let args_clone = args.clone();
                    let angles_clone = angles.clone();

                    // Process the quantum operation with angles in radians
                    match self.processor.process_quantum_op(
                        &qop_str,
                        angles_clone.as_ref(),
                        &args_clone,
                    ) {
                        Ok((gate_type, qubit_args, angle_args)) => {
                            // Add the gate to the builder
                            self.processor.add_quantum_operation_to_builder(
                                &mut self.message_builder,
                                &gate_type,
                                &qubit_args,
                                &angle_args,
                            )?;

                            operation_count += 1;
                            debug!("Added quantum operation to builder");
                        }
                        Err(e) => return Err(e),
                    }
                }
                Operation::ClassicalOp {
                    cop,
                    args,
                    returns,
                    metadata: _,
                    function,
                } => {
                    debug!("Processing classical operation: {cop}");

                    // Debug log specially for ffcall operations
                    if cop == "ffcall" {
                        debug!(
                            "Found ffcall operation: function={function:?}, args={args:?}, returns={returns:?}"
                        );
                    }

                    if self.processor.handle_classical_op(
                        cop,
                        args,
                        returns,
                        &ops,
                        self.current_op,
                    )? {
                        debug!("Finishing batch due to classical operation completion");
                        self.current_op += 1;

                        // Build and return the message
                        if operation_count > 0 {
                            debug!("Returning batch with {operation_count} operations");
                            return Ok(Some(self.message_builder.build()));
                        }

                        // Create an empty message if no operations were added
                        debug!("Returning empty batch after classical operation");
                        return Ok(Some(ByteMessage::builder().build()));
                    }
                }
                Operation::Block {
                    block,
                    ops,
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => {
                    debug!("Processing block operation: {block}");

                    match block.as_str() {
                        "if" => {
                            // Process if/else block
                            if let Some(_cond) = condition
                                && let (Some(tb), fb) = (true_branch, false_branch)
                            {
                                // Get operations based on condition
                                let branch_ops = self.processor.process_conditional_block(
                                    condition.as_ref().unwrap(),
                                    tb,
                                    fb.as_deref(),
                                )?;

                                // Replace the current op with the branch operations
                                // This is a simplification - a more robust implementation would
                                // involve temporarily changing the ops list
                                for branch_op in &branch_ops {
                                    match branch_op {
                                        Operation::QuantumOp {
                                            qop, angles, args, ..
                                        } => {
                                            // Process each quantum operation in the branch
                                            let qop_str = qop.clone();
                                            let args_clone = args.clone();
                                            let angles_clone = angles.clone();

                                            match self.processor.process_quantum_op(
                                                &qop_str,
                                                angles_clone.as_ref(),
                                                &args_clone,
                                            ) {
                                                Ok((gate_type, qubit_args, angle_args)) => {
                                                    self.processor
                                                        .add_quantum_operation_to_builder(
                                                            &mut self.message_builder,
                                                            &gate_type,
                                                            &qubit_args,
                                                            &angle_args,
                                                        )?;
                                                    operation_count += 1;
                                                }
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        Operation::ClassicalOp {
                                            cop,
                                            args,
                                            returns,
                                            metadata: _,
                                            function,
                                        } => {
                                            debug!(
                                                "Processing classical operation in branch: {cop}"
                                            );
                                            // Handle classical operations from conditional branches
                                            if cop == "ffcall" {
                                                debug!(
                                                    "Processing ffcall in branch: function={function:?}, args={args:?}, returns={returns:?}"
                                                );
                                            }
                                            // For ffcall operations from branches, we need to handle them specially
                                            // because they have the function name directly in the operation
                                            if cop == "ffcall" {
                                                // Create a temporary operation list with just this operation
                                                // This ensures handle_classical_op can find the function name
                                                let temp_ops = vec![branch_op.clone()];
                                                if self.processor.handle_classical_op(
                                                    cop, args, returns, &temp_ops,
                                                    0, // The operation is at index 0 in temp_ops
                                                )? {
                                                    debug!(
                                                        "Classical ffcall operation in branch completed"
                                                    );
                                                }
                                            } else {
                                                // For other classical operations, use the original ops
                                                if self.processor.handle_classical_op(
                                                    cop,
                                                    args,
                                                    returns,
                                                    &branch_ops,
                                                    0, // Index within branch ops
                                                )? {
                                                    debug!(
                                                        "Classical operation in branch completed"
                                                    );
                                                }
                                            }
                                        }
                                        _ => {
                                            // For other operation types, we'll handle them later
                                            debug!(
                                                "Skipping other operation type in branch: {branch_op:?}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        "qparallel" => {
                            // Process qparallel block
                            let parallel_ops = self.processor.process_block(block, ops)?;

                            for parallel_op in parallel_ops {
                                match parallel_op {
                                    Operation::QuantumOp {
                                        qop, angles, args, ..
                                    } => {
                                        // Process each quantum operation in the parallel block
                                        let qop_str = qop.clone();
                                        let args_clone = args.clone();
                                        let angles_clone = angles.clone();

                                        match self.processor.process_quantum_op(
                                            &qop_str,
                                            angles_clone.as_ref(),
                                            &args_clone,
                                        ) {
                                            Ok((gate_type, qubit_args, angle_args)) => {
                                                self.processor.add_quantum_operation_to_builder(
                                                    &mut self.message_builder,
                                                    &gate_type,
                                                    &qubit_args,
                                                    &angle_args,
                                                )?;
                                                operation_count += 1;
                                            }
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    _ => {
                                        // For other operation types, we'll handle them later
                                        debug!("Skipping non-quantum operation in qparallel block");
                                    }
                                }
                            }
                        }
                        "sequence" => {
                            // Process sequence block by recursively processing all operations
                            debug!("Processing sequence block");

                            // Process each operation in the sequence block
                            for op in ops {
                                match op {
                                    Operation::QuantumOp {
                                        qop, angles, args, ..
                                    } => {
                                        // Process each quantum operation
                                        let qop_str = qop.clone();
                                        let args_clone = args.clone();
                                        let angles_clone = angles.clone();

                                        match self.processor.process_quantum_op(
                                            &qop_str,
                                            angles_clone.as_ref(),
                                            &args_clone,
                                        ) {
                                            Ok((gate_type, qubit_args, angle_args)) => {
                                                self.processor.add_quantum_operation_to_builder(
                                                    &mut self.message_builder,
                                                    &gate_type,
                                                    &qubit_args,
                                                    &angle_args,
                                                )?;
                                                operation_count += 1;
                                                debug!(
                                                    "Added quantum operation from sequence block to builder"
                                                );
                                            }
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    Operation::ClassicalOp {
                                        cop,
                                        args,
                                        returns,
                                        function: _,
                                        metadata: _,
                                    } => {
                                        // Process classical operations in the sequence
                                        if self.processor.handle_classical_op(
                                            cop,
                                            args,
                                            returns,
                                            ops,
                                            self.current_op,
                                        )? {
                                            debug!(
                                                "Processed classical operation from sequence block"
                                            );
                                            operation_count += 1;
                                        }
                                    }
                                    Operation::MachineOp {
                                        mop,
                                        args,
                                        duration,
                                        metadata,
                                    } => {
                                        // Process machine operations in the sequence
                                        match self.processor.process_machine_op(
                                            mop,
                                            args.as_ref(),
                                            duration.as_ref(),
                                            metadata.as_ref(),
                                        ) {
                                            Ok(mop_result) => {
                                                self.processor.add_machine_operation_to_builder(
                                                    &mut self.message_builder,
                                                    &mop_result,
                                                )?;
                                                operation_count += 1;
                                                debug!(
                                                    "Added machine operation from sequence block to builder"
                                                );
                                            }
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    // We don't process nested blocks here to avoid excessive recursion
                                    // If needed, we could add a recursion limit
                                    _ => debug!("Skipping complex operation in sequence block"),
                                }
                            }
                        }
                        _ => {
                            return Err(PecosError::Input(format!("Unknown block type: {block}")));
                        }
                    }
                }
                Operation::MachineOp {
                    mop,
                    args,
                    duration,
                    metadata,
                } => {
                    debug!("Processing machine operation: {mop}");

                    // Process the machine operation
                    match self.processor.process_machine_op(
                        mop,
                        args.as_ref(),
                        duration.as_ref(),
                        metadata.as_ref(),
                    ) {
                        Ok(mop_result) => {
                            // Add the machine operation to the builder
                            self.processor.add_machine_operation_to_builder(
                                &mut self.message_builder,
                                &mop_result,
                            )?;
                            operation_count += 1;
                            debug!("Added machine operation to builder");
                        }
                        Err(e) => return Err(e),
                    }
                }
                Operation::MetaInstruction {
                    meta,
                    args,
                    metadata: _,
                } => {
                    debug!("Processing meta instruction: {meta}");

                    // Process meta instructions like barrier
                    match self.processor.process_meta_instruction(meta, args) {
                        Ok(meta_result) => {
                            // Add the meta instruction to the builder
                            self.processor.add_meta_instruction_to_builder(
                                &mut self.message_builder,
                                &meta_result,
                            )?;
                            operation_count += 1;
                            debug!("Added meta instruction to builder");
                        }
                        Err(e) => return Err(e),
                    }
                }
                Operation::Comment { comment } => {
                    debug!("Processing comment: {comment}");
                    // Comments are ignored during execution
                }
            }
            self.current_op += 1;

            // If we've reached the maximum batch size, break out of the loop
            // This ensures we don't create excessively large messages
            if operation_count >= MAX_BATCH_SIZE {
                debug!("Reached maximum batch size ({MAX_BATCH_SIZE}), returning current batch");
                break;
            }
        }

        debug!("PHIR engine generated {operation_count} operations for shot");

        // Build and return the message
        Ok(Some(self.message_builder.build()))
    }

    /// Gets the results in a specific format
    ///
    /// # Returns
    ///
    /// A compact JSON string containing the results
    ///
    /// # Errors
    ///
    /// Returns an error if there was a problem getting the results
    pub fn get_formatted_results(&self) -> Result<String, PecosError> {
        let shot_result = self.get_results()?;

        // Convert single Shot to ShotVec for better formatting
        let shot_results = pecos_engines::shot_results::ShotVec {
            shots: vec![shot_result],
        };

        Ok(shot_results.to_compact_json())
    }
}

impl Default for PHIREngine {
    fn default() -> Self {
        Self::empty()
    }
}

impl ControlEngine for PHIREngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!(
            "PHIR: start() called with current_op={}, beginning new shot",
            self.current_op
        );
        self.current_op = 0; // Force reset here too
        self.processor.reset();

        debug!("start() called, generating commands");
        if let Some(commands) = self.generate_commands_impl()? {
            debug!("start() - Returning commands for processing");
            Ok(EngineStage::NeedsProcessing(commands))
        } else {
            debug!("start() - No commands to process, returning results immediately");
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!(
            "continue_processing called with current_op={}",
            self.current_op
        );

        // Handle received measurements
        let measurement_results = measurements.outcomes()?;
        log::debug!("PHIREngine: Measurement results received: {measurement_results:?}");

        // For Bell state debugging - check if we have 2 qubits and get result patterns
        if let Some(prog) = &self.program
            && prog.ops.iter().any(|op| {
                if let Operation::VariableDefinition { variable, size, .. } = op {
                    variable == "q" && *size == 2
                } else {
                    false
                }
            })
        {
            log::debug!(
                "Bell state program detected - measurement results: {measurement_results:?}"
            );
        }

        let ops = match &self.program {
            Some(program) => program.ops.clone(),
            None => vec![],
        };
        self.processor
            .handle_measurements(&measurement_results, &ops)?;

        // Get next batch of commands if any
        debug!("Getting next batch of commands");
        if let Some(commands) = self.generate_commands_impl()? {
            debug!("Returning more commands for processing");
            Ok(EngineStage::NeedsProcessing(commands))
        } else {
            debug!("No more commands, returning results");
            // Make sure to process any remaining Result operations
            if self.current_op < self.program.as_ref().map_or(0, |prog| prog.ops.len()) {
                let ops = self.program.as_ref().unwrap().ops.clone();
                if let Operation::ClassicalOp {
                    cop, args, returns, ..
                } = &ops[self.current_op]
                    && cop == "Result"
                {
                    debug!("Processing Result operation: {cop}");
                    self.processor.handle_classical_op(
                        cop,
                        args,
                        returns,
                        &ops,
                        self.current_op,
                    )?;
                }
            }

            let results = self.get_results()?;
            debug!("Completed processing, returning results");
            Ok(EngineStage::Complete(results))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("PHIREngine::reset() implementation for ControlEngine being called!");
        self.reset_state();
        Ok(())
    }
}

impl ClassicalEngine for PHIREngine {
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        // When no commands are left to generate, create an empty message
        Ok(self
            .generate_commands_impl()?
            .unwrap_or_else(ByteMessage::create_empty))
    }

    fn num_qubits(&self) -> usize {
        // First check if environment has quantum variables
        let sum = self.processor.environment.count_qubits();
        if sum > 0 {
            return sum;
        }

        // If no quantum variables in environment, directly scan the program ops
        if let Some(program) = &self.program {
            let mut total = 0;
            for op in &program.ops {
                if let Operation::VariableDefinition {
                    data,
                    data_type,
                    variable: _,
                    size,
                } = op
                    && data == "qvar_define"
                    && data_type == "qubits"
                {
                    total += size;
                }
            }
            return total;
        }

        0 // If no program is loaded, return 0
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        let measurement_outcomes = message.outcomes()?;
        let ops = match &self.program {
            Some(program) => program.ops.clone(),
            None => vec![],
        };
        self.processor
            .handle_measurements(&measurement_outcomes, &ops)
    }

    #[allow(clippy::too_many_lines)]
    fn get_results(&self) -> Result<Shot, PecosError> {
        let mut results = Shot::default();

        // First process all export mappings to get properly processed values
        let mut exported_values = self.processor.process_export_mappings();

        // Determine which registers to include in the results based on environment mappings
        let mappings = self.processor.environment.get_mappings();
        if mappings.is_empty() {
            // No explicit export mappings - include all environment variables
            log::debug!(
                "PHIR: No explicit export mappings - adding all variables from environment"
            );

            for info in self.processor.environment.get_all_variables() {
                if let Some(value) = self.processor.environment.get(&info.name) {
                    // Add to exported_values if not already there
                    exported_values
                        .entry(info.name.clone())
                        .or_insert(value.as_u32());

                    log::debug!(
                        "PHIR: Added direct variable from environment {} = {}",
                        info.name,
                        value
                    );

                    // Simply add all variables from environment without any special transformations
                    // No assumptions about variable naming conventions
                }
            }
        } else {
            log::debug!("PHIR: Using environment mappings to determine which registers to include");

            // Keep only the registers that are explicitly mapped as destinations
            // This provides a general approach that works for all tests including Bell state tests
            let destination_registers: HashSet<String> =
                mappings.iter().map(|(_, dest)| dest.clone()).collect();

            // Keep only the explicitly mapped destination registers if we have any
            if !destination_registers.is_empty() {
                let mut filtered_values = HashMap::new();

                for dest in destination_registers {
                    if exported_values.contains_key(&dest) {
                        let value = exported_values[&dest];
                        log::debug!("PHIR: Keeping explicitly mapped register: {dest} = {value}");
                        filtered_values.insert(dest, value);
                    }
                }

                // Replace with filtered values
                exported_values = filtered_values;
            }
        }

        // Add the processed values to the results
        log::debug!(
            "PHIR: Adding {} exported values to results",
            exported_values.len()
        );

        for (key, value) in &exported_values {
            results.data.insert(key.clone(), Data::U32(*value));
            log::debug!("PHIR: Adding mapped register {key} = {value}");
        }

        // If nothing has been exported so far, use all available variables
        // This general approach works for all types of programs
        if results.data.is_empty() {
            log::debug!("PHIR: No exported values found - using all available variables");

            // Add all variables from environment
            for info in self.processor.environment.get_all_variables() {
                if let Some(value) = self.processor.environment.get(&info.name) {
                    log::debug!("PHIR: Adding variable {} = {} to results", info.name, value);
                    results
                        .data
                        .insert(info.name.clone(), Data::U32(value.as_u32()));
                }
            }

            // Process all mappings from environment for any variables not previously handled
            for (source, dest) in self.processor.environment.get_mappings() {
                // Skip if this destination is already in the results
                if results.data.contains_key(dest) {
                    continue;
                }

                // Try to get the value from the environment
                if let Some(value) = self.processor.environment.get(source) {
                    log::debug!("PHIR: Exporting {source} -> {dest} = {value}");
                    results.data.insert(dest.clone(), Data::U32(value.as_u32()));
                } else {
                    // If not found in environment, try the exported_values directly
                    // Try to get the value directly from environment if not already found
                    if let Some(value) = self.processor.environment.get(source) {
                        log::debug!(
                            "PHIR: Exporting from environment {source} -> {dest} = {value}"
                        );
                        results.data.insert(dest.clone(), Data::U32(value.as_u32()));
                    }
                    // Note: We no longer fall back to measurement_results as primary source
                }
            }

            // If there are no registers in the results, add all variables from environment
            if results.data.is_empty() {
                for info in self.processor.environment.get_all_variables() {
                    if let Some(value) = self.processor.environment.get(&info.name) {
                        log::debug!("PHIR: Adding all variables: {} = {}", info.name, value);
                        results
                            .data
                            .insert(info.name.clone(), Data::U32(value.as_u32()));
                    }
                }
            }

            // No legacy fallback needed anymore since the environment is the single source of truth
            if results.data.is_empty() {
                log::debug!(
                    "PHIR: No register values found in environment, returning empty results"
                );
            }
        }

        // Since the environment is now the single source of truth for all variable data,
        // we don't need to maintain consistency between bit-indexed variables and composite variables.
        // All variables should already have the correct values directly from the environment.
        //
        // We're removing the complex bit variable reconstruction code since:
        // 1. We no longer create or manage separate bit-indexed variables
        // 2. All bit values are stored directly in integer variables
        // 3. The environment handles all bit operations transparently

        // Just log the final state of the registers for debugging
        log::debug!("PHIR: Final register values from environment - no reconstruction needed");
        for (key, value) in &results.data {
            log::debug!("PHIR: Register {key} = {value:?}");
        }

        log::debug!("PHIR: Exported {} registers", results.data.len());
        log::debug!("PHIR: Final registers: {:?}", results.data);
        Ok(results)
    }

    fn compile(&self) -> Result<(), PecosError> {
        // No compilation needed for PHIR/JSON
        Ok(())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("PHIREngine::reset() implementation for ClassicalEngine being called!");
        self.reset_state();
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Clone for PHIREngine {
    fn clone(&self) -> Self {
        // Create a new instance with the same program
        match &self.program {
            Some(program) => {
                // Clone the processor with all its state
                // This includes the foreign object, variable definitions, and any results
                let processor = self.processor.clone();

                Self {
                    program: Some(program.clone()),
                    current_op: self.current_op, // Preserve the current operation position
                    processor, // Use the fully cloned processor with preserved state
                    message_builder: ByteMessageBuilder::new(),
                }
            }
            None => Self::empty(),
        }
    }
}

impl Engine for PHIREngine {
    type Input = ();
    type Output = Shot;

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        // Print out operations for debugging
        if let Some(program) = &self.program {
            log::debug!(
                "Process() called, processing {} operations",
                program.ops.len()
            );
            for (i, op) in program.ops.iter().enumerate() {
                log::debug!("Process: Operation {i}: {op:?}");
            }
        }

        // Reset state to ensure we start fresh
        self.reset_state();

        // Start the engine and check its state
        match self.start(())? {
            EngineStage::Complete(result) => {
                log::debug!("Shot completed directly in start()");
                Ok(result)
            }
            EngineStage::NeedsProcessing(_cmds) => {
                log::debug!("PHIREngine cannot process quantum operations directly");
                log::debug!("Falling back to manual direct execution for integration testing");

                // For integration tests, manually execute the operations
                if let Some(program) = &self.program {
                    log::debug!("Process: processing all operations in order");

                    // Process operations in order (like a real execution)
                    for (i, op) in program.ops.iter().enumerate() {
                        log::debug!("Processing operation {i}: {op:?}");

                        match op {
                            Operation::VariableDefinition {
                                data,
                                data_type,
                                variable,
                                size,
                            } => {
                                log::debug!(
                                    "Processing variable definition: {data_type} {variable}"
                                );
                                let _ = self
                                    .processor
                                    .handle_variable_definition(data, data_type, variable, *size);
                            }
                            Operation::ClassicalOp {
                                cop,
                                args,
                                returns,
                                function: _,
                                metadata: _,
                            } => {
                                log::debug!("Processing classical operation {i}: {cop}");
                                if let Err(e) = self.processor.handle_classical_op(
                                    cop,
                                    args,
                                    returns,
                                    &program.ops,
                                    i,
                                ) {
                                    log::error!("Failed to process classical operation: {e}");
                                    return Err(e);
                                }
                            }
                            Operation::QuantumOp {
                                qop,
                                args,
                                returns: _,
                                angles: _,
                                metadata: _,
                            } => {
                                log::debug!("Processing quantum operation {i}: {qop}");
                                log::debug!("Simulating quantum gate: {qop} on qubits: {args:?}");
                            }
                            // Handle other operation types as needed
                            _ => log::debug!("Skipping operation type for direct execution"),
                        }
                    }

                    // Process all Result commands to ensure outputs are generated
                    let mut result_ops = Vec::new();
                    for (i, op) in program.ops.iter().enumerate() {
                        if let Operation::ClassicalOp {
                            cop, args, returns, ..
                        } = op
                            && cop == "Result"
                        {
                            result_ops.push((i, args.clone(), returns.clone()));
                        }
                    }

                    log::debug!("Processing {} Result commands", result_ops.len());
                    for (i, args, returns) in result_ops {
                        self.processor.handle_classical_op(
                            "Result",
                            &args,
                            &returns,
                            &program.ops,
                            i,
                        )?;
                    }
                }

                // Return results from the processed state
                Ok(self.get_results()?)
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Call our internal reset method
        self.reset_state();
        Ok(())
    }
}
