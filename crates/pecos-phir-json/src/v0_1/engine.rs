use crate::v0_1::ast::{Operation, PHIRProgram, infer_size};
use crate::v0_1::environment::DataType;
use crate::v0_1::foreign_objects::ForeignObject;
use crate::v0_1::operations::OperationProcessor;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::{ByteMessage, builder::ByteMessageBuilder};
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage};
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A frame of the execution stack: a slice of operations and the index of the
/// next one to process. Blocks (`if`/`sequence`/`qparallel`) push a new frame
/// so their child ops flow through the same measurement-boundary deferral as
/// top-level ops, with arbitrary nesting.
#[derive(Debug, Clone)]
struct ExecFrame {
    ops: Vec<Operation>,
    idx: usize,
}

/// `PhirJsonEngine` processes PHIR programs and generates quantum operations
#[derive(Debug)]
pub struct PhirJsonEngine {
    /// The loaded PHIR program
    program: Option<PHIRProgram>,
    /// Execution cursor: a stack of operation frames. The bottom frame is the
    /// top-level program; blocks push nested frames. Persists across batch
    /// yields so a batch can end mid-block and resume after measurements land.
    exec_stack: Vec<ExecFrame>,
    /// Operation processor for handling different operation types
    pub processor: OperationProcessor,
    /// Builder for constructing `ByteMessages`
    message_builder: ByteMessageBuilder,
}

impl PhirJsonEngine {
    /// Build the initial execution stack from the loaded program (a single
    /// bottom frame over the top-level ops), or empty if no program.
    fn initial_stack(program: Option<&PHIRProgram>) -> Vec<ExecFrame> {
        program.map_or_else(Vec::new, |p| {
            vec![ExecFrame {
                ops: p.ops.clone(),
                idx: 0,
            }]
        })
    }

    /// Advance the top execution frame past the current op.
    fn advance_cursor(&mut self) {
        if let Some(frame) = self.exec_stack.last_mut() {
            frame.idx += 1;
        }
    }
}

impl PhirJsonEngine {
    /// Sets a foreign object for executing foreign function calls
    pub fn set_foreign_object(&mut self, foreign_object: Box<dyn ForeignObject>) {
        self.processor.set_foreign_object(foreign_object);
    }

    /// Creates a new instance of `PhirJsonEngine` by loading a PHIR program JSON file.
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
    /// use pecos_phir_json::v0_1::engine::PhirJsonEngine;
    ///
    /// let engine = PhirJsonEngine::new("path_to_program.json");
    /// match engine {
    ///     Ok(engine) => println!("PhirJsonEngine loaded successfully!"),
    ///     Err(e) => eprintln!("Error loading PhirJsonEngine: {}", e),
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        let content = std::fs::read_to_string(path).map_err(PecosError::IO)?;
        Self::from_json(&content)
    }

    /// Creates a new instance of `PhirJsonEngine` from a JSON string.
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
    /// use pecos_phir_json::v0_1::engine::PhirJsonEngine;
    ///
    /// let json = r#"{"format":"PHIR/JSON","version":"0.1.0","metadata":{},"ops":[]}"#;
    /// let engine = PhirJsonEngine::from_json(json);
    /// match engine {
    ///     Ok(engine) => println!("PhirJsonEngine loaded successfully!"),
    ///     Err(e) => eprintln!("Error loading PhirJsonEngine: {}", e),
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
                let _ = processor.handle_variable_definition(
                    data,
                    data_type,
                    variable,
                    infer_size(data_type, *size),
                );
            }
        }

        Ok(Self {
            exec_stack: Self::initial_stack(Some(&program)),
            program: Some(program),
            processor,
            message_builder: ByteMessageBuilder::new(),
        })
    }

    /// Creates a new instance of `PhirJsonEngine` from a parsed `PHIRProgram`.
    ///
    /// # Parameters
    /// - `program`: A `PHIRProgram` instance.
    ///
    /// # Returns
    /// - Returns a new `PhirJsonEngine` initialized with the provided program.
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
                processor.handle_variable_definition(
                    data,
                    data_type,
                    variable,
                    infer_size(data_type, *size),
                )?;
            }
        }

        Ok(Self {
            exec_stack: Self::initial_stack(Some(&program)),
            program: Some(program),
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
        debug!("INTERNAL RESET: PhirJsonEngine reset");

        // Reset the execution cursor to the start of the program.
        self.exec_stack = Self::initial_stack(self.program.as_ref());

        // Log operations for debugging if needed
        if log::log_enabled!(log::Level::Debug)
            && let Some(program) = self.program.as_ref()
        {
            debug!("Operations to process after reset: {}", program.ops.len());
        }

        // Reset the processor state (maintains variable definitions but clears values)
        // This is now a clean reset without preserving values, since the environment
        // is the single source of truth and values should be recomputed as needed
        self.processor.reset();

        // Reset the message builder to reuse allocated memory
        self.message_builder.reset();

        debug!("PhirJsonEngine reset complete, ready for next execution");
    }

    // Create an empty engine without any program
    fn empty() -> Self {
        Self {
            program: None,
            exec_stack: Vec::new(),
            processor: OperationProcessor::new(),
            message_builder: ByteMessageBuilder::new(),
        }
    }

    fn generate_commands_impl(&mut self) -> Result<Option<ByteMessage>, PecosError> {
        // Maximum quantum ops per batch, to bound message size.
        const MAX_BATCH_SIZE: usize = 100;

        // Reset per-batch state.
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();
        self.processor.clear_pending_measurements();
        let mut operation_count = 0;

        loop {
            // Drop exhausted frames (finished blocks / end of program).
            while self.exec_stack.last().is_some_and(|f| f.idx >= f.ops.len()) {
                self.exec_stack.pop();
            }
            let Some(frame) = self.exec_stack.last() else {
                // Nothing left to process. Flush any queued quantum ops.
                if operation_count > 0 {
                    return Ok(Some(self.message_builder.build()));
                }
                debug!("Execution stack empty; shot complete");
                return Ok(None);
            };
            let op = frame.ops[frame.idx].clone();

            match &op {
                Operation::VariableDefinition {
                    data,
                    data_type,
                    variable,
                    size,
                } => {
                    let _ = self.processor.handle_variable_definition(
                        data,
                        data_type,
                        variable,
                        infer_size(data_type, *size),
                    );
                    self.advance_cursor();
                }
                Operation::QuantumOp {
                    qop,
                    angles,
                    args,
                    returns,
                    ..
                } => {
                    let (gate_type, qubit_args, angle_args) =
                        self.processor.process_quantum_op(qop, angles.as_ref(), args)?;
                    self.processor.add_quantum_operation_to_builder(
                        &mut self.message_builder,
                        &gate_type,
                        &qubit_args,
                        &angle_args,
                    )?;
                    if gate_type == "Measure" {
                        // Record the return registers so positional outcomes map
                        // back to them -- works for measurements queued from
                        // inside blocks too, which a top-level scan would miss.
                        // One slot per measured qubit keeps outcomes aligned.
                        self.processor
                            .record_measurement_returns(qubit_args.len(), returns)?;
                    }
                    operation_count += 1;
                    self.advance_cursor();
                    if operation_count >= MAX_BATCH_SIZE {
                        debug!("Reached maximum batch size ({MAX_BATCH_SIZE}), returning batch");
                        return Ok(Some(self.message_builder.build()));
                    }
                }
                Operation::ClassicalOp {
                    cop, args, returns, ..
                } => {
                    // A classical op may read a register measured earlier in this
                    // batch; its result is not available until the batch runs.
                    // Yield the queued quantum ops first (leaving the cursor in
                    // place) so measurements land before this op executes.
                    if operation_count > 0 {
                        debug!("Deferring classical op '{cop}' until batch measurements are applied");
                        return Ok(Some(self.message_builder.build()));
                    }
                    // Pass the op itself so an ffcall can find its function name.
                    let ended = self.processor.handle_classical_op(
                        cop,
                        args,
                        returns,
                        std::slice::from_ref(&op),
                        0,
                    )?;
                    self.advance_cursor();
                    if ended {
                        return Ok(Some(self.message_builder.build()));
                    }
                }
                Operation::Block {
                    block,
                    ops: block_ops,
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => {
                    match block.as_str() {
                        "if" => {
                            // A conditional's condition may read a register
                            // measured earlier in this batch; defer until the
                            // batch's measurements are applied.
                            if operation_count > 0 {
                                debug!("Deferring conditional block until batch measurements are applied");
                                return Ok(Some(self.message_builder.build()));
                            }
                            let branch = match (condition, true_branch) {
                                (Some(cond), Some(tb)) => self.processor.process_conditional_block(
                                    cond,
                                    tb,
                                    false_branch.as_deref(),
                                )?,
                                _ => Vec::new(),
                            };
                            self.advance_cursor();
                            self.exec_stack.push(ExecFrame {
                                ops: branch,
                                idx: 0,
                            });
                        }
                        "sequence" | "qparallel" => {
                            // Flatten the block: its child ops run through the
                            // same cursor (and the same measurement deferral),
                            // with arbitrary nesting.
                            let expanded = self.processor.process_block(block, block_ops)?;
                            self.advance_cursor();
                            self.exec_stack.push(ExecFrame {
                                ops: expanded,
                                idx: 0,
                            });
                        }
                        other => {
                            return Err(PecosError::Input(format!(
                                "Unknown block type: {other}"
                            )));
                        }
                    }
                }
                Operation::MachineOp {
                    mop,
                    args,
                    duration,
                    metadata,
                } => {
                    let mop_result = self.processor.process_machine_op(
                        mop,
                        args.as_ref(),
                        duration.as_ref(),
                        metadata.as_ref(),
                    )?;
                    self.processor
                        .add_machine_operation_to_builder(&mut self.message_builder, &mop_result)?;
                    operation_count += 1;
                    self.advance_cursor();
                }
                Operation::MetaInstruction { meta, args, .. } => {
                    let meta_result = self.processor.process_meta_instruction(meta, args)?;
                    self.processor
                        .add_meta_instruction_to_builder(&mut self.message_builder, &meta_result)?;
                    operation_count += 1;
                    self.advance_cursor();
                }
                Operation::Comment { .. } | Operation::DataExport { .. } => {
                    self.advance_cursor();
                }
            }
        }
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

impl Default for PhirJsonEngine {
    fn default() -> Self {
        Self::empty()
    }
}

impl ControlEngine for PhirJsonEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!("PHIR: start() called, beginning new shot");
        self.exec_stack = Self::initial_stack(self.program.as_ref());
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
        debug!("continue_processing called");

        // Handle received measurements
        let measurement_results = measurements.outcomes()?;
        log::debug!("PhirJsonEngine: Measurement results received: {measurement_results:?}");

        // For Bell state debugging - check if we have 2 qubits and get result patterns
        if let Some(prog) = &self.program
            && prog.ops.iter().any(|op| {
                if let Operation::VariableDefinition {
                    variable,
                    size,
                    data_type,
                    ..
                } = op
                {
                    variable == "q" && infer_size(data_type, *size) == 2
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
            // The cursor re-processes any deferred classical ops after
            // measurements are applied, so no end-of-program fallback is needed.
            let results = self.get_results()?;
            debug!("Completed processing, returning results");
            Ok(EngineStage::Complete(results))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("PhirJsonEngine::reset() implementation for ControlEngine being called!");
        self.reset_state();
        Ok(())
    }
}

impl ClassicalEngine for PhirJsonEngine {
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
                    total += infer_size(data_type, *size);
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
                // Skip quantum variables and internal measurement variables
                if info.data_type == DataType::Qubits {
                    continue;
                }
                if info.name.starts_with("measurement_") {
                    continue;
                }
                if let Some(value) = self.processor.environment.get(&info.name) {
                    exported_values
                        .entry(info.name.clone())
                        .or_insert(value.as_u64());
                }
            }
        } else {
            log::debug!("PHIR: Using environment mappings to determine which registers to include");

            // Keep only the registers that are explicitly mapped as destinations
            // This provides a general approach that works for all tests including Bell state tests
            let destination_registers: BTreeSet<String> =
                mappings.iter().map(|(_, dest)| dest.clone()).collect();

            // Keep only the explicitly mapped destination registers if we have any
            if !destination_registers.is_empty() {
                let mut filtered_values = BTreeMap::new();

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
            // Use add_register with proper width from variable metadata. A
            // signed size-S register is an i(S+1) integer, so it renders S+1
            // bits (the extra sign bit); unsigned registers render S bits.
            let width = self
                .processor
                .environment
                .get_variable_info_opt(key)
                .map_or(32, |info| {
                    if info.data_type.is_signed() {
                        info.size + 1
                    } else {
                        info.size
                    }
                });
            results.add_register_u64(key, *value, width);
            log::debug!("PHIR: Adding mapped register {key} = {value} (width={width})");
        }

        // If nothing has been exported so far, use all available variables
        // This general approach works for all types of programs
        if results.data.is_empty() {
            log::debug!("PHIR: No exported values found - using all available variables");

            // Add all variables from environment with proper widths
            for info in self.processor.environment.get_all_variables() {
                if let Some(value) = self.processor.environment.get(&info.name) {
                    log::debug!("PHIR: Adding variable {} = {} to results", info.name, value);
                    let width = if info.data_type.is_signed() {
                        info.size + 1
                    } else {
                        info.size
                    };
                    results.add_register_u64(&info.name, value.as_u64(), width);
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
                    results.data.insert(dest.clone(), Data::U64(value.as_u64()));
                } else {
                    // If not found in environment, try the exported_values directly
                    // Try to get the value directly from environment if not already found
                    if let Some(value) = self.processor.environment.get(source) {
                        log::debug!(
                            "PHIR: Exporting from environment {source} -> {dest} = {value}"
                        );
                        results.data.insert(dest.clone(), Data::U64(value.as_u64()));
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
                            .insert(info.name.clone(), Data::U64(value.as_u64()));
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
        debug!("PhirJsonEngine::reset() implementation for ClassicalEngine being called!");
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

impl Clone for PhirJsonEngine {
    fn clone(&self) -> Self {
        // Create a new instance with the same program
        match &self.program {
            Some(program) => {
                // Clone the processor with all its state
                // This includes the foreign object, variable definitions, and any results
                let processor = self.processor.clone();

                Self {
                    program: Some(program.clone()),
                    exec_stack: self.exec_stack.clone(), // Preserve execution position
                    processor, // Use the fully cloned processor with preserved state
                    message_builder: ByteMessageBuilder::new(),
                }
            }
            None => Self::empty(),
        }
    }
}

impl Engine for PhirJsonEngine {
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
                log::debug!("PhirJsonEngine cannot process quantum operations directly");
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
                                let _ = self.processor.handle_variable_definition(
                                    data,
                                    data_type,
                                    variable,
                                    infer_size(data_type, *size),
                                );
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
