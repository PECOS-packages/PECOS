use crate::v0_1::ast::{Operation, PHIRProgram};
use crate::v0_1::foreign_objects::ForeignObject;
use crate::v0_1::operations::OperationProcessor;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::{ByteMessage, builder::ByteMessageBuilder};
use pecos_engines::core::shot_results::ShotResult;
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
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
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

        // If we've processed all ops, return empty batch to signal completion
        if self.current_op >= ops.len() {
            debug!(
                "End of program reached at op {}, sending flush",
                self.current_op
            );
            return Ok(ByteMessage::create_flush());
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
                    debug!(
                        "Processing variable definition: {} {} {}",
                        data, data_type, variable
                    );
                    let _ = self
                        .processor
                        .handle_variable_definition(data, data_type, variable, *size);
                    self.current_op += 1;
                    return self.generate_commands();
                }
                Operation::QuantumOp {
                    qop,
                    angles,
                    args,
                    returns: _,
                    metadata: _,
                } => {
                    debug!("Processing quantum operation: {}", qop);

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
                    debug!("Processing classical operation: {}", cop);

                    // Debug log specially for ffcall operations
                    if cop == "ffcall" {
                        debug!(
                            "Found ffcall operation: function={:?}, args={:?}, returns={:?}",
                            function, args, returns
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
                            debug!("Returning batch with {} operations", operation_count);
                            return Ok(self.message_builder.build());
                        }

                        // Create an empty message if no operations were added
                        debug!("Returning empty batch after classical operation");
                        return Ok(ByteMessage::builder().build());
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
                    debug!("Processing block operation: {}", block);

                    match block.as_str() {
                        "if" => {
                            // Process if/else block
                            if let Some(_cond) = condition {
                                if let (Some(tb), fb) = (true_branch, false_branch) {
                                    // Get operations based on condition
                                    let branch_ops = self.processor.process_conditional_block(
                                        condition.as_ref().unwrap(),
                                        tb,
                                        fb.as_deref(),
                                    )?;

                                    // Replace the current op with the branch operations
                                    // This is a simplification - a more robust implementation would
                                    // involve temporarily changing the ops list
                                    for branch_op in branch_ops {
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
                                            _ => {
                                                // For other operation types, we'll handle them later
                                                debug!("Skipping non-quantum operation in branch");
                                            }
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
                    debug!("Processing machine operation: {}", mop);

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
                    debug!("Processing meta instruction: {}", meta);

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
                    debug!("Processing comment: {}", comment);
                    // Comments are ignored during execution
                }
            }
            self.current_op += 1;

            // If we've reached the maximum batch size, break out of the loop
            // This ensures we don't create excessively large messages
            if operation_count >= MAX_BATCH_SIZE {
                debug!(
                    "Reached maximum batch size ({}), returning current batch",
                    MAX_BATCH_SIZE
                );
                break;
            }
        }

        debug!(
            "PHIR engine generated {} operations for shot",
            operation_count
        );

        // Build and return the message
        Ok(self.message_builder.build())
    }

    /// Gets the results in a specific format
    ///
    /// # Parameters
    ///
    /// * `format` - The output format to use (`PrettyJson`, `CompactJson`, or Tabular)
    ///
    /// # Returns
    ///
    /// A string containing the results in the specified format
    ///
    /// # Errors
    ///
    /// Returns an error if there was a problem getting the results
    pub fn get_formatted_results(
        &self,
        format: pecos_engines::core::shot_results::OutputFormat,
    ) -> Result<String, PecosError> {
        let shot_result = self.get_results()?;

        // Convert single ShotResult to ShotResults for better formatting
        let mut shot_results = pecos_engines::core::shot_results::ShotResults::new();

        // Add each register to the ShotResults
        for (key, &value) in &shot_result.registers {
            shot_results.register_shots.insert(key.clone(), vec![value]);
        }

        for (key, &value) in &shot_result.registers_u64 {
            shot_results
                .register_shots_u64
                .insert(key.clone(), vec![value]);
        }

        for (key, &value) in &shot_result.registers_i64 {
            shot_results
                .register_shots_i64
                .insert(key.clone(), vec![value]);
        }

        Ok(shot_results.to_string_with_format(format))
    }
}

impl Default for PHIREngine {
    fn default() -> Self {
        Self::empty()
    }
}

impl ControlEngine for PHIREngine {
    type Input = ();
    type Output = ShotResult;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, ShotResult>, PecosError> {
        debug!(
            "PHIR: start() called with current_op={}, beginning new shot",
            self.current_op
        );
        self.current_op = 0; // Force reset here too
        self.processor.reset();

        debug!("start() called, generating commands");
        let commands = self.generate_commands()?;

        if commands.is_empty().unwrap_or(false) {
            debug!("start() - No commands to process, returning results immediately");
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            debug!("start() - Returning commands for processing");
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ShotResult>, PecosError> {
        debug!(
            "continue_processing called with current_op={}",
            self.current_op
        );

        // Handle received measurements
        let measurement_results = measurements.parse_measurements()?;
        log::info!(
            "PHIREngine: Measurement results received: {:?}",
            measurement_results
        );

        // For Bell state debugging - check if we have 2 qubits and get result patterns
        if let Some(prog) = &self.program {
            if prog.ops.iter().any(|op| {
                if let Operation::VariableDefinition { variable, size, .. } = op {
                    variable == "q" && *size == 2
                } else {
                    false
                }
            }) {
                log::info!(
                    "Bell state program detected - measurement results: {:?}",
                    measurement_results
                );
            }
        }

        let ops = match &self.program {
            Some(program) => program.ops.clone(),
            None => vec![],
        };
        self.processor
            .handle_measurements(&measurement_results, &ops)?;

        // Get next batch of commands if any
        debug!("Getting next batch of commands");
        let commands = self.generate_commands()?;

        if commands.is_empty().unwrap_or(false) {
            debug!("No more commands, returning results");
            // Make sure to process any remaining Result operations
            if self.current_op < self.program.as_ref().map_or(0, |prog| prog.ops.len()) {
                let ops = self.program.as_ref().unwrap().ops.clone();
                if let Operation::ClassicalOp {
                    cop, args, returns, ..
                } = &ops[self.current_op]
                {
                    if cop == "Result" {
                        debug!("Processing Result operation: {}", cop);
                        self.processor.handle_classical_op(
                            cop,
                            args,
                            returns,
                            &ops,
                            self.current_op,
                        )?;
                    }
                }
            }

            let results = self.get_results()?;
            debug!("Completed processing, returning results");
            Ok(EngineStage::Complete(results))
        } else {
            debug!("Returning more commands for processing");
            Ok(EngineStage::NeedsProcessing(commands))
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
        self.generate_commands()
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
                {
                    if data == "qvar_define" && data_type == "qubits" {
                        total += size;
                    }
                }
            }
            return total;
        }

        0 // If no program is loaded, return 0
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        let measurements = message.parse_measurements()?;
        let ops = match &self.program {
            Some(program) => program.ops.clone(),
            None => vec![],
        };
        self.processor.handle_measurements(&measurements, &ops)
    }

    #[allow(clippy::too_many_lines)]
    fn get_results(&self) -> Result<ShotResult, PecosError> {
        let mut results = ShotResult::default();

        // First process all export mappings to get properly processed values
        let mut exported_values = self.processor.process_export_mappings();

        // Determine which registers to include in the results based on environment mappings
        let mappings = self.processor.environment.get_mappings();
        if mappings.is_empty() {
            // No explicit export mappings - include all environment variables
            log::info!("PHIR: No explicit export mappings - adding all variables from environment");

            for info in self.processor.environment.get_all_variables() {
                if let Some(value) = self.processor.environment.get(&info.name) {
                    // Add to exported_values if not already there
                    exported_values
                        .entry(info.name.clone())
                        .or_insert(value.as_u32());

                    log::info!(
                        "PHIR: Added direct variable from environment {} = {}",
                        info.name,
                        value
                    );

                    // Simply add all variables from environment without any special transformations
                    // No assumptions about variable naming conventions
                }
            }
        } else {
            log::info!("PHIR: Using environment mappings to determine which registers to include");

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
                        log::info!(
                            "PHIR: Keeping explicitly mapped register: {} = {}",
                            dest,
                            value
                        );
                        filtered_values.insert(dest, value);
                    }
                }

                // Replace with filtered values
                exported_values = filtered_values;
            }
        }

        // Add the processed values to the results
        log::info!(
            "PHIR: Adding {} exported values to results",
            exported_values.len()
        );

        for (key, value) in &exported_values {
            results.registers.insert(key.clone(), *value);
            results.registers_u64.insert(key.clone(), u64::from(*value));
            results.registers_i64.insert(key.clone(), i64::from(*value));
            log::info!("PHIR: Adding mapped register {} = {}", key, value);
        }

        // If nothing has been exported so far, use all available variables
        // This general approach works for all types of programs
        if results.registers.is_empty() {
            log::info!("PHIR: No exported values found - using all available variables");

            // Add all variables from environment
            for info in self.processor.environment.get_all_variables() {
                if let Some(value) = self.processor.environment.get(&info.name) {
                    log::info!("PHIR: Adding variable {} = {} to results", info.name, value);
                    results.registers.insert(info.name.clone(), value.as_u32());
                    results
                        .registers_u64
                        .insert(info.name.clone(), value.as_u64());
                    results
                        .registers_i64
                        .insert(info.name.clone(), value.as_i64());
                }
            }

            // Process all mappings from environment for any variables not previously handled
            for (source, dest) in self.processor.environment.get_mappings() {
                // Skip if this destination is already in the results
                if results.registers.contains_key(dest) {
                    continue;
                }

                // Try to get the value from the environment
                if let Some(value) = self.processor.environment.get(source) {
                    log::info!("PHIR: Exporting {} -> {} = {}", source, dest, value);
                    results.registers.insert(dest.clone(), value.as_u32());
                    results.registers_u64.insert(dest.clone(), value.as_u64());
                    results.registers_i64.insert(dest.clone(), value.as_i64());
                } else {
                    // If not found in environment, try the exported_values directly
                    // Try to get the value directly from environment if not already found
                    if let Some(value) = self.processor.environment.get(source) {
                        log::info!(
                            "PHIR: Exporting from environment {} -> {} = {}",
                            source,
                            dest,
                            value
                        );
                        results.registers.insert(dest.clone(), value.as_u32());
                        results.registers_u64.insert(dest.clone(), value.as_u64());
                        results.registers_i64.insert(dest.clone(), value.as_i64());
                    }
                    // Note: We no longer fall back to measurement_results as primary source
                }
            }

            // If there are no registers in the results, add all variables from environment
            if results.registers.is_empty() {
                for info in self.processor.environment.get_all_variables() {
                    if let Some(value) = self.processor.environment.get(&info.name) {
                        log::info!("PHIR: Adding all variables: {} = {}", info.name, value);
                        results.registers.insert(info.name.clone(), value.as_u32());
                        results
                            .registers_u64
                            .insert(info.name.clone(), value.as_u64());
                        results
                            .registers_i64
                            .insert(info.name.clone(), value.as_i64());
                    }
                }
            }

            // No legacy fallback needed anymore since the environment is the single source of truth
            if results.registers.is_empty() {
                log::info!(
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
        log::info!("PHIR: Final register values from environment - no reconstruction needed");
        for (key, value) in &results.registers {
            log::debug!("PHIR: Register {} = {}", key, value);
        }

        log::info!("PHIR: Exported {} registers", results.registers.len());
        log::info!("PHIR: Final registers: {:?}", results.registers);
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
    type Output = ShotResult;

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        // Print out operations for debugging
        if let Some(program) = &self.program {
            log::info!(
                "Process() called, processing {} operations",
                program.ops.len()
            );
            for (i, op) in program.ops.iter().enumerate() {
                log::info!("Process: Operation {}: {:?}", i, op);
            }
        }

        // For integration tests, we want to manually execute the operations
        // to ensure expression tests work correctly - they depend on variable values
        // being properly set and expressions being properly evaluated
        log::info!("INTEGRATION TEST HELPER - Enabling direct execution mode");

        // Reset state to ensure we start fresh
        self.reset_state();

        // Process all operations sequentially as they would be in a real program
        if let Some(program) = &self.program {
            log::info!("Process: processing all operations in order");

            // Process operations in order (like a real execution)
            for (i, op) in program.ops.iter().enumerate() {
                log::info!("Processing operation {}: {:?}", i, op);

                match op {
                    Operation::VariableDefinition {
                        data,
                        data_type,
                        variable,
                        size,
                    } => {
                        log::info!("Processing variable definition: {} {}", data_type, variable);
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
                        log::info!("Processing classical operation {}: {}", i, cop);
                        if let Err(e) =
                            self.processor
                                .handle_classical_op(cop, args, returns, &program.ops, i)
                        {
                            log::error!("Failed to process classical operation: {}", e);
                            return Err(e);
                        }

                        // Log state after each classical operation
                        log::info!(
                            "After classical operation {}, environment: {:?}",
                            i,
                            self.processor.environment.get_all_variables()
                        );
                    }
                    Operation::QuantumOp {
                        qop,
                        args,
                        returns: _, // Unused variable
                        angles: _,
                        metadata: _,
                    } => {
                        log::info!("Processing quantum operation {}: {}", i, qop);

                        // When using process() method directly, we DO NOT simulate quantum operations
                        // Quantum operations (including measurements) should be simulated by a quantum simulator
                        if qop == "Init" {
                            // For initialization, nothing needs to be done in simulation
                            log::info!("Simulated initialization of qubits: {:?}", args);
                        } else {
                            // For other gates, nothing needs to be done in simulation
                            log::info!("Simulated quantum gate: {} on qubits: {:?}", qop, args);
                        }
                    }
                    Operation::Block {
                        block,
                        ops: block_ops,
                        condition,
                        true_branch,
                        false_branch,
                        metadata: _,
                    } => {
                        log::info!("Processing block operation {}: {}", i, block);

                        // For direct execution, recursively process operations in blocks
                        match block.as_str() {
                            "if" => {
                                // For conditional blocks, evaluate condition and process appropriate branch
                                if let Some(cond) = condition {
                                    if let (Some(tb), fb) = (true_branch, false_branch) {
                                        // Actually evaluate the condition using ExpressionEvaluator
                                        let condition_value =
                                            self.processor.evaluate_expression(cond)? != 0;

                                        // Select branch based on condition
                                        let branch_ops = if condition_value {
                                            log::info!(
                                                "Condition evaluated to true, executing true branch"
                                            );
                                            tb
                                        } else if let Some(fb_ops) = fb {
                                            log::info!(
                                                "Condition evaluated to false, executing false branch"
                                            );
                                            fb_ops
                                        } else {
                                            log::info!(
                                                "Condition evaluated to false, no false branch"
                                            );
                                            &Vec::new()
                                        };

                                        // Process all operations in the selected branch
                                        for branch_op in branch_ops {
                                            // Recursively process this operation
                                            log::info!(
                                                "Processing operation in branch: {:?}",
                                                branch_op
                                            );
                                            match branch_op {
                                                Operation::QuantumOp {
                                                    qop, args: _, returns, .. // Marking args as unused since we don't use it here
                                                } => {
                                                    if qop == "Measure" && !returns.is_empty() {
                                                        // Quantum operations including measurements are handled by the quantum simulator
                                                        log::info!("Processing quantum operation in branch: {}", qop);
                                                    }
                                                }
                                                Operation::ClassicalOp {
                                                    cop, args, returns, function: _, metadata: _
                                                } => {
                                                    // Actually process the classical operation
                                                    log::info!("Processing classical operation in branch: {}", cop);
                                                    if let Err(e) = self.processor.handle_classical_op(
                                                        cop, args, returns, &program.ops, i
                                                    ) {
                                                        log::error!("Failed to process classical operation in branch: {}", e);
                                                        return Err(e);
                                                    }
                                                }
                                                // Handle other operations if needed
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                            "qparallel" => {
                                // For parallel blocks, process all operations
                                for parallel_op in block_ops {
                                    if let Operation::QuantumOp {
                                        qop, args: _, returns, .. // Marking args as unused since we don't use it here
                                    } = parallel_op {
                                        if qop == "Measure" && !returns.is_empty() {
                                            // Quantum operations including measurements are handled by the quantum simulator
                                            log::info!("Processing quantum operation in qparallel block: {}", qop);
                                        }
                                    }
                                }
                            }
                            "sequence" => {
                                // Process all operations sequentially
                                for seq_op in block_ops {
                                    match seq_op {
                                        Operation::QuantumOp {
                                            qop, args: _, returns, .. // Marking args as unused since we don't use it here
                                        } => {
                                            if qop == "Measure" && !returns.is_empty() {
                                                // Quantum operations including measurements are handled by the quantum simulator
                                                log::info!("Processing quantum operation in sequence block: {}", qop);
                                            }
                                        }
                                        Operation::ClassicalOp {
                                            cop, args, returns, ..
                                        } => {
                                            if let Err(e) = self.processor.handle_classical_op(
                                                cop,
                                                args,
                                                returns,
                                                &program.ops,
                                                i,
                                            ) {
                                                log::error!(
                                                    "Failed to process classical operation in sequence: {}",
                                                    e
                                                );
                                                return Err(e);
                                            }
                                        }
                                        // Handle other operations if needed
                                        _ => {}
                                    }
                                }
                            }
                            _ => {
                                log::warn!("Unknown block type: {}", block);
                            }
                        }
                    }
                    Operation::MachineOp {
                        mop,
                        args,
                        duration,
                        metadata,
                    } => {
                        log::info!("Processing machine operation {}: {}", i, mop);

                        // For machine operations, record that we're simulating them
                        match mop.as_str() {
                            "Idle" => {
                                // Use trace level for verification - zero cost in production
                                log::trace!(
                                    "VERIFICATION: mop_idle:{} args:{:?} duration:{:?}",
                                    self.current_op,
                                    args,
                                    duration
                                );

                                // Log additional details at debug level
                                log::debug!("Simulating Idle operation with args: {:?}", args);
                            }
                            "Delay" => {
                                // Use trace level for verification - zero cost in production
                                log::trace!(
                                    "VERIFICATION: mop_delay:{} args:{:?} duration:{:?}",
                                    self.current_op,
                                    args,
                                    duration
                                );

                                // Log additional details at debug level
                                log::debug!("Simulating Delay operation with args: {:?}", args);
                            }
                            "Transport" => {
                                // Use trace level for verification - zero cost in production
                                log::trace!(
                                    "VERIFICATION: mop_transport:{} args:{:?}",
                                    self.current_op,
                                    args
                                );

                                // Log additional details at debug level
                                log::debug!("Simulating Transport operation with args: {:?}", args);
                            }
                            "Timing" => {
                                // Use trace level for verification - zero cost in production
                                log::trace!(
                                    "VERIFICATION: mop_timing:{} args:{:?} metadata:{:?}",
                                    self.current_op,
                                    args,
                                    metadata
                                );

                                // Log additional details at debug level
                                log::debug!("Simulating Timing operation with args: {:?}", args);
                            }
                            "Skip" => {
                                // Use trace level for verification - zero cost in production
                                log::trace!("VERIFICATION: mop_skip:{}", self.current_op);

                                // Log additional details at debug level
                                log::debug!("Simulating Skip operation");
                            }
                            _ => log::warn!("Unknown machine operation: {}", mop),
                        }
                    }
                    Operation::MetaInstruction {
                        meta,
                        args,
                        metadata: _,
                    } => {
                        log::info!("Processing meta instruction {}: {}", i, meta);

                        // For meta instructions, log that we're simulating them
                        if meta == "barrier" {
                            // Log barrier operation with the operation index for verification in tests
                            // Use trace level for verification - zero cost in production
                            log::trace!(
                                "VERIFICATION: meta_barrier:{} args:{:?}",
                                self.current_op,
                                args
                            );

                            // Log additional details at debug level
                            log::debug!(
                                "Simulating barrier meta instruction with args: {:?}",
                                args
                            );
                        } else {
                            log::warn!("Unknown meta instruction: {}", meta);
                        }
                    }
                    Operation::Comment { .. } => {
                        log::info!("Skipping comment at index {}", i);
                    }
                }
            }

            log::info!(
                "After processing all operations, environment: {:?}",
                self.processor.environment.get_all_variables()
            );

            // Extra pass to specifically handle all Result commands again just to be sure
            log::info!("Extra pass to handle Result commands");

            // First, explicitly look for Result commands
            let mut result_ops = Vec::new();
            for (i, op) in program.ops.iter().enumerate() {
                if let Operation::ClassicalOp {
                    cop, args, returns, ..
                } = op
                {
                    if cop == "Result" {
                        result_ops.push((i, args.clone(), returns.clone()));
                    }
                }
            }

            // Process all Result commands
            log::info!("Found {} Result commands to process", result_ops.len());
            for (i, args, returns) in result_ops {
                log::info!("Re-processing Result operation at index {}", i);
                self.processor
                    .handle_classical_op("Result", &args, &returns, &program.ops, i)?;
            }

            // We no longer need special fallback mapping
            // All variables are now handled generally through the Environment API
            log::info!("Ensuring all variables are available to export mappings");
        }

        // TEMPORARY DEBUGGING: Create a ShotResult directly from our current state
        log::info!("TEMPORARY: Creating result directly from processor state");
        let mut result = ShotResult::default();

        // Process all export mappings to ensure we have values for exports
        log::info!("Processing export mappings into results");
        let exported_values = self.processor.process_export_mappings();

        log::info!("Exported values from mappings: {:?}", exported_values);

        // Add all exported values from process_export_mappings to the results
        for (key, value) in &exported_values {
            result.registers.insert(key.clone(), *value);
            result.registers_u64.insert(key.clone(), u64::from(*value));
            // Also add to i64 registers
            result.registers_i64.insert(key.clone(), i64::from(*value));
            log::info!("Adding exported register {} = {}", key, value);
        }

        // All exports come from environment and export_mappings now

        // If there are no registers in the results or registers are missing, add all variables
        // from the environment to ensure we have a comprehensive result
        if result.registers.is_empty() {
            log::info!("No registers in results, adding all available variables");

            // Add all variables from the environment
            for info in self.processor.environment.get_all_variables() {
                if let Some(value) = self.processor.environment.get(&info.name) {
                    log::info!("Adding variable {} = {} to results", info.name, value);
                    result.registers.insert(info.name.clone(), value.as_u32());
                    result
                        .registers_u64
                        .insert(info.name.clone(), value.as_u64());
                    result
                        .registers_i64
                        .insert(info.name.clone(), value.as_i64());
                }
            }
        }

        log::info!("Returning ShotResult: {:?}", result);
        Ok(result)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Call our internal reset method
        self.reset_state();
        Ok(())
    }
}
