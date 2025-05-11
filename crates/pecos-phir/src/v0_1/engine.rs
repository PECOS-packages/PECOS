use crate::v0_1::ast::{Operation, PHIRProgram};
use crate::v0_1::operations::OperationProcessor;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::{ByteMessage, builder::ByteMessageBuilder};
use pecos_engines::core::shot_results::ShotResult;
use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage};
use std::any::Any;
use std::path::Path;

/// `PHIREngine` processes PHIR programs and generates quantum operations
#[derive(Debug)]
pub struct PHIREngine {
    /// The loaded PHIR program
    program: Option<PHIRProgram>,
    /// Current operation index being processed
    current_op: usize,
    /// Operation processor for handling different operation types
    processor: OperationProcessor,
    /// Builder for constructing `ByteMessages`
    message_builder: ByteMessageBuilder,
}

impl PHIREngine {
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
                processor.handle_variable_definition(data, data_type, variable, *size);
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
    #[must_use]
    pub fn from_program(program: PHIRProgram) -> Self {
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
                processor.handle_variable_definition(data, data_type, variable, *size);
            }
        }

        Self {
            program: Some(program),
            current_op: 0,
            processor,
            message_builder: ByteMessageBuilder::new(),
        }
    }

    /// Resets the engine state
    fn reset_state(&mut self) {
        debug!(
            "INTERNAL RESET: PHIREngine reset before current_op={}",
            self.current_op
        );
        self.current_op = 0;
        debug!(
            "INTERNAL RESET: PHIREngine reset after current_op={}",
            self.current_op
        );
        self.processor.reset();
        // Reset the message builder to reuse allocated memory
        self.message_builder.reset();
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
            debug!("End of program reached, sending flush");
            return Ok(ByteMessage::create_flush());
        }

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
                    self.processor
                        .handle_variable_definition(data, data_type, variable, *size);
                }
                Operation::QuantumOp {
                    qop,
                    angles,
                    args,
                    returns: _,
                } => {
                    debug!("Processing quantum operation: {}", qop);

                    // Clone the operation parameters to avoid borrow issues
                    let qop_str = qop.clone();
                    let args_clone = args.clone();
                    let angles_clone = angles.clone();

                    // Process the quantum operation
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
                Operation::ClassicalOp { cop, args, returns } => {
                    debug!("Processing classical operation: {}", cop);
                    if self.processor.handle_classical_op(cop, args, returns)? {
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

        let commands = self.generate_commands()?;
        if commands.is_empty().unwrap_or(false) {
            debug!("PHIR: start() - No commands to process, returning results immediately");
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            debug!("PHIR: start() - Returning commands for processing");
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ShotResult>, PecosError> {
        // Handle received measurements
        let measurement_results = measurements.parse_measurements()?;
        let ops = match &self.program {
            Some(program) => program.ops.clone(),
            None => vec![],
        };
        self.processor
            .handle_measurements(&measurement_results, &ops)?;

        // Get next batch of commands if any
        let commands = self.generate_commands()?;
        if commands.is_empty().unwrap_or(false) {
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
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
        // First check if quantum_variables is already populated
        let sum: usize = self.processor.quantum_variables.values().sum();
        if sum > 0 {
            return sum;
        }

        // If quantum_variables is empty, directly scan the program ops
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

    fn get_results(&self) -> Result<ShotResult, PecosError> {
        let mut results = ShotResult::default();

        // Process export mappings to get exported values
        let exported_values = self.processor.process_export_mappings();

        // Add all exported values to the results
        log::debug!(
            "PHIR: Adding {} exported values to results",
            exported_values.len()
        );

        for (key, value) in &exported_values {
            results.registers.insert(key.clone(), *value);
            results.registers_u64.insert(key.clone(), u64::from(*value));
            log::debug!("PHIR: Adding exported register {} = {}", key, value);
        }

        // Sanity check - this should only happen if measurements failed or weren't taken
        if results.registers.is_empty() && !self.processor.export_mappings.is_empty() {
            log::warn!(
                "PHIR: No exported values found despite Result commands being present. Check program execution."
            );
        }

        log::debug!("PHIR: Exported {} registers", results.registers.len());
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
            Some(program) => Self {
                program: Some(program.clone()),
                current_op: 0,                        // Reset state in the clone
                processor: OperationProcessor::new(), // Create a fresh processor
                message_builder: ByteMessageBuilder::new(),
            },
            None => Self::empty(),
        }
    }
}

impl Engine for PHIREngine {
    type Input = ();
    type Output = ShotResult;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        // Process operations until we need more input or we're done
        let mut stage = self.start(())?;

        // If we're already done, return the result
        if let EngineStage::Complete(result) = stage {
            return Ok(result);
        }

        // Otherwise, we need to process more (just return an empty measurement result)
        if let EngineStage::NeedsProcessing(_) = stage {
            // Create an empty message to simulate processing
            let empty_message = ByteMessage::builder().build();
            stage = self.continue_processing(empty_message)?;

            if let EngineStage::Complete(result) = stage {
                return Ok(result);
            }
        }

        // If we get here, something went wrong
        Err(PecosError::Processing(
            "Failed to complete processing".to_string(),
        ))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Call our internal reset method
        self.reset_state();
        Ok(())
    }
}
