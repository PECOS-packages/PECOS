use crate::byte_message::{ByteMessage, builder::ByteMessageBuilder};
use crate::core::shot_results::ShotResult;
use crate::engines::{ControlEngine, Engine, EngineStage, classical::ClassicalEngine};
use log::debug;
use pecos_core::errors::PecosError;
use serde::Deserialize;
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
struct PHIRProgram {
    format: String,
    version: String,
    metadata: HashMap<String, String>,
    ops: Vec<Operation>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum Operation {
    VariableDefinition {
        data: String,
        data_type: String,
        variable: String,
        size: usize,
    },
    QuantumOp {
        qop: String,
        #[serde(default)]
        angles: Option<(Vec<f64>, String)>,
        args: Vec<(String, usize)>,
        #[serde(default)]
        returns: Vec<(String, usize)>,
    },
    ClassicalOp {
        cop: String,
        #[serde(default)]
        args: Vec<ArgItem>,
        #[serde(default)]
        returns: Vec<ArgItem>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum ArgItem {
    Indexed((String, usize)),
    Simple(String),
}

// Constants for internal register naming
const MEASUREMENT_PREFIX: &str = "measurement_";

#[derive(Debug)]
pub struct PHIREngine {
    /// The loaded PHIR program
    program: Option<PHIRProgram>,
    /// Current operation index being processed
    current_op: usize,
    /// All measurement results and internal variable values
    /// This includes both raw measurements and internal register values
    measurement_results: HashMap<String, u32>,
    /// Values explicitly exported via the Result operator
    /// These are the values that will be presented to the user in the final output
    exported_values: HashMap<String, u32>,
    /// Mappings from source registers to export names for Result operations
    /// This allows us to apply the mappings when measurements are available
    export_mappings: Vec<(String, String)>,
    /// Mapping of quantum variable names to their sizes
    quantum_variables: HashMap<String, usize>,
    /// Mapping of classical variable names to their types and sizes
    classical_variables: HashMap<String, (String, usize)>,
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
    /// use pecos_engines::engines::phir::PHIREngine;
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
    /// use pecos_engines::engines::phir::PHIREngine;
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

        Ok(Self {
            program: Some(program),
            current_op: 0,
            measurement_results: HashMap::new(),
            exported_values: HashMap::new(),
            export_mappings: Vec::new(),
            quantum_variables: HashMap::new(),
            classical_variables: HashMap::new(),
            message_builder: ByteMessageBuilder::new(),
        })
    }

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
        self.measurement_results.clear();
        self.exported_values.clear();
        self.export_mappings.clear();
        // Reset the message builder to reuse allocated memory
        self.message_builder.reset();
    }

    // Create an empty engine without any program
    fn empty() -> Self {
        Self {
            program: None,
            current_op: 0,
            measurement_results: HashMap::new(),
            exported_values: HashMap::new(),
            export_mappings: Vec::new(),
            quantum_variables: HashMap::new(),
            classical_variables: HashMap::new(),
            message_builder: ByteMessageBuilder::new(),
        }
    }

    fn handle_variable_definition(
        &mut self,
        data: &str,
        data_type: &str,
        variable: &str,
        size: usize,
    ) {
        match data {
            "qvar_define" if data_type == "qubits" => {
                self.quantum_variables.insert(variable.to_string(), size);
                log::debug!("Defined quantum variable {} of size {}", variable, size);
            }
            "cvar_define" => {
                self.classical_variables
                    .insert(variable.to_string(), (data_type.to_string(), size));
                log::debug!(
                    "Defined classical variable {} of type {} and size {}",
                    variable,
                    data_type,
                    size
                );
            }
            _ => log::warn!(
                "Unknown variable definition: {} {} {}",
                data,
                data_type,
                variable
            ),
        }
    }

    fn validate_variable_access(&self, var: &str, idx: usize) -> Result<(), PecosError> {
        // Check quantum variables
        if let Some(&size) = self.quantum_variables.get(var) {
            if idx >= size {
                return Err(PecosError::Input(format!(
                    "Variable access validation failed: Index {idx} out of bounds for quantum variable '{var}' of size {size}"
                )));
            }
            return Ok(());
        }

        // Check classical variables
        if let Some((_, size)) = self.classical_variables.get(var) {
            if idx >= *size {
                return Err(PecosError::Input(format!(
                    "Variable access validation failed: Index {idx} out of bounds for classical variable '{var}' of size {size}"
                )));
            }
            return Ok(());
        }

        Err(PecosError::Input(format!(
            "Variable access validation failed: Variable '{var}' is not defined in the program"
        )))
    }

    fn handle_classical_op(
        &mut self,
        cop: &str,
        args: &[ArgItem],
        returns: &[ArgItem],
    ) -> Result<bool, PecosError> {
        // Extract variable name and index from each ArgItem
        let extract_var_idx = |arg: &ArgItem| -> (String, usize) {
            match arg {
                ArgItem::Indexed((name, idx)) => (name.clone(), *idx),
                ArgItem::Simple(name) => (name.clone(), 0),
            }
        };

        // For most operations, validate all variable accesses
        if cop == "Result" {
            // For Result operation, only validate the source variables (args)
            // The return variables are outputs and don't need to be defined
            for arg in args {
                let (var, idx) = extract_var_idx(arg);
                self.validate_variable_access(&var, idx)?;
            }
        } else {
            for arg in args.iter().chain(returns) {
                let (var, idx) = extract_var_idx(arg);
                self.validate_variable_access(&var, idx)?;
            }
        }

        if cop == "Result" {
            if args.len() == 1 && returns.len() == 1 {
                // Extract source and export info
                let (source_register, _) = extract_var_idx(&args[0]);
                let (export_name, _) = extract_var_idx(&returns[0]);

                log::debug!(
                    "Storing export mapping: {} -> {}",
                    source_register,
                    export_name
                );

                // Instead of immediately exporting, store the mapping for later
                // This allows us to apply the export after all measurements are collected
                self.export_mappings
                    .push((source_register.clone(), export_name.clone()));

                return Ok(true);
            }
            log::warn!("Result operation requires exactly one source and one export target");
            return Ok(true);
        }

        Ok(false)
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::items_after_statements)]
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
                    self.handle_variable_definition(data, data_type, variable, *size);
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
                    // This avoids borrowing self and self.message_builder at the same time
                    match self.process_quantum_op(&qop_str, angles_clone.as_ref(), &args_clone) {
                        Ok((gate_type, qubit_args, angle_args)) => {
                            // Now add the gate to the builder based on the processed parameters
                            match gate_type.as_str() {
                                "RZ" => {
                                    self.message_builder.add_rz(angle_args[0], &[qubit_args[0]]);
                                }
                                "R1XY" => {
                                    self.message_builder.add_r1xy(
                                        angle_args[0],
                                        angle_args[1],
                                        &[qubit_args[0]],
                                    );
                                }
                                "SZZ" => {
                                    self.message_builder
                                        .add_szz(&[qubit_args[0]], &[qubit_args[1]]);
                                }
                                "CX" => {
                                    self.message_builder
                                        .add_cx(&[qubit_args[0]], &[qubit_args[1]]);
                                }
                                "H" => {
                                    self.message_builder.add_h(&[qubit_args[0]]);
                                }
                                "X" => {
                                    self.message_builder.add_x(&[qubit_args[0]]);
                                }
                                "Y" => {
                                    self.message_builder.add_y(&[qubit_args[0]]);
                                }
                                "Z" => {
                                    self.message_builder.add_z(&[qubit_args[0]]);
                                }
                                "Measure" => {
                                    self.message_builder
                                        .add_measurements(&[qubit_args[0]], &[qubit_args[0]]);
                                }
                                _ => {
                                    return Err(PecosError::Gate(format!(
                                        "Unsupported quantum gate operation: Gate type '{gate_type}' is not implemented"
                                    )));
                                }
                            }
                            operation_count += 1;
                            debug!("Added quantum operation to builder");
                        }
                        Err(e) => return Err(e),
                    }
                }
                Operation::ClassicalOp { cop, args, returns } => {
                    debug!("Processing classical operation: {}", cop);
                    if self.handle_classical_op(cop, args, returns)? {
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

    /// Process a quantum operation and return the gate type, qubit arguments, and angle arguments
    fn process_quantum_op(
        &self,
        qop: &str,
        angles: Option<&(Vec<f64>, String)>,
        args: &[(String, usize)],
    ) -> Result<(String, Vec<usize>, Vec<f64>), PecosError> {
        // First validate all variables
        for (var, idx) in args {
            self.validate_variable_access(var, *idx)?;
        }

        // Validate that we have at least one qubit argument
        if args.is_empty() {
            return Err(PecosError::Input(format!(
                "Invalid quantum operation: Operation '{qop}' requires at least one qubit argument"
            )));
        }

        // Extract qubit arguments
        let mut qubit_args = Vec::new();
        for (_, idx) in args {
            qubit_args.push(*idx);
        }

        // Process based on gate type
        match qop {
            // Single-qubit rotation gates
            "RZ" => {
                let theta = angles
                    .as_ref()
                    .map(|(angles, _)| angles[0])
                    .ok_or_else(|| {
                        PecosError::Gate(format!(
                            "Invalid gate parameters: Missing rotation angle for '{qop}' gate"
                        ))
                    })?;
                Ok((qop.to_string(), qubit_args, vec![theta]))
            }
            "R1XY" => {
                if angles.as_ref().map_or(0, |(angles, _)| angles.len()) < 2 {
                    return Err(PecosError::Gate(format!(
                        "Invalid gate parameters: '{qop}' gate requires two angles (phi, theta)"
                    )));
                }
                let (phi, theta) = angles
                    .as_ref()
                    .map(|(angles, _)| (angles[0], angles[1]))
                    .ok_or_else(|| {
                        PecosError::Gate(format!(
                            "Invalid gate parameters: Missing rotation angles for '{qop}' gate"
                        ))
                    })?;
                Ok((qop.to_string(), qubit_args, vec![phi, theta]))
            }

            // Two-qubit gates
            "SZZ" | "ZZ" => {
                if args.len() < 2 {
                    return Err(PecosError::Gate(format!(
                        "Invalid gate parameters: '{qop}' gate requires exactly two qubits"
                    )));
                }
                Ok(("SZZ".to_string(), qubit_args, vec![]))
            }
            "CX" | "CNOT" => {
                if args.len() < 2 {
                    return Err(PecosError::Gate(format!(
                        "Invalid gate parameters: '{qop}' gate requires control and target qubits (2 qubits total)"
                    )));
                }
                Ok(("CX".to_string(), qubit_args, vec![]))
            }

            // Single-qubit Clifford gates
            // Single-qubit Clifford gates and Measurement
            "H" | "X" | "Y" | "Z" | "Measure" => Ok((qop.to_string(), qubit_args, vec![])),

            _ => Err(PecosError::Gate(format!(
                "Unsupported quantum gate operation: Gate type '{qop}' is not implemented"
            ))),
        }
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
        self.measurement_results.clear();
        self.exported_values.clear();
        self.export_mappings.clear();

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
        self.handle_measurements(measurements)?;

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
                    self.handle_variable_definition(data, data_type, variable, *size);
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
                    // This avoids borrowing self and self.message_builder at the same time
                    match self.process_quantum_op(&qop_str, angles_clone.as_ref(), &args_clone) {
                        Ok((gate_type, qubit_args, angle_args)) => {
                            // Now add the gate to the builder based on the processed parameters
                            match gate_type.as_str() {
                                "RZ" => {
                                    self.message_builder.add_rz(angle_args[0], &[qubit_args[0]]);
                                }
                                "R1XY" => {
                                    self.message_builder.add_r1xy(
                                        angle_args[0],
                                        angle_args[1],
                                        &[qubit_args[0]],
                                    );
                                }
                                "SZZ" => {
                                    self.message_builder
                                        .add_szz(&[qubit_args[0]], &[qubit_args[1]]);
                                }
                                "CX" => {
                                    self.message_builder
                                        .add_cx(&[qubit_args[0]], &[qubit_args[1]]);
                                }
                                "H" => {
                                    self.message_builder.add_h(&[qubit_args[0]]);
                                }
                                "X" => {
                                    self.message_builder.add_x(&[qubit_args[0]]);
                                }
                                "Y" => {
                                    self.message_builder.add_y(&[qubit_args[0]]);
                                }
                                "Z" => {
                                    self.message_builder.add_z(&[qubit_args[0]]);
                                }
                                "Measure" => {
                                    self.message_builder
                                        .add_measurements(&[qubit_args[0]], &[qubit_args[0]]);
                                }
                                _ => {
                                    return Err(PecosError::Gate(format!(
                                        "Unsupported quantum gate operation: Gate type '{gate_type}' is not implemented"
                                    )));
                                }
                            }
                            operation_count += 1;
                            debug!("Added quantum operation to builder");
                        }
                        Err(e) => return Err(e),
                    }
                }
                Operation::ClassicalOp { cop, args, returns } => {
                    debug!("Processing classical operation: {}", cop);
                    if self.handle_classical_op(cop, args, returns)? {
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

    fn num_qubits(&self) -> usize {
        // First check if quantum_variables is already populated
        let sum: usize = self.quantum_variables.values().sum();
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
        // Parse measurements using ByteMessage helper
        let measurements = message.parse_measurements()?;

        for (result_id, outcome) in measurements {
            debug!(
                "PHIR: Received measurement result_id={}, outcome={}",
                result_id, outcome
            );

            // Store the measurement with the standard prefix and result_id
            self.measurement_results
                .insert(format!("{MEASUREMENT_PREFIX}{result_id}"), outcome);

            // Also directly map this to the classical variable bits
            // For example, if Measure returns [["m", 0]], we should set m_0 = outcome
            // This lookup would need access to the program, which we have in self.program
            if let Some(program) = &self.program {
                for op in &program.ops {
                    if let Operation::QuantumOp {
                        qop,
                        args: _,
                        returns,
                        ..
                    } = op
                    {
                        if qop == "Measure" && !returns.is_empty() {
                            // Get the variable name and index from the returns field
                            let (var_name, var_idx) = &returns[0];

                            // Check if this is the right measurement result
                            if *var_idx == result_id as usize {
                                // Store with the format "variable_index"
                                let var_key = format!("{var_name}_{var_idx}");
                                self.measurement_results.insert(var_key.clone(), outcome);
                                log::debug!(
                                    "Mapped measurement result_id={} to {}",
                                    result_id,
                                    var_key
                                );

                                // Also update the register value by setting the appropriate bit
                                let entry = self
                                    .measurement_results
                                    .entry(var_name.clone())
                                    .or_insert(0);
                                *entry |= outcome << var_idx;
                                log::debug!("Updated register {} value to {}", var_name, *entry);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn get_results(&self) -> Result<ShotResult, PecosError> {
        let mut results = ShotResult::default();
        let mut exported_values = HashMap::new();

        // Process all stored export mappings
        for (source_register, export_name) in &self.export_mappings {
            log::debug!(
                "Processing export mapping: {} -> {}",
                source_register,
                export_name
            );

            // Check for direct register value first
            if let Some(&value) = self.measurement_results.get(source_register) {
                log::debug!(
                    "Found direct register value for {}: {}",
                    source_register,
                    value
                );
                exported_values.insert(export_name.clone(), value);
                continue;
            }

            // Check for indexed values (e.g., m_0, m_1, etc.)
            let mut register_value = 0u32;
            let mut found_values = false;

            for i in 0..32 {
                // Assuming max 32 bits for registers
                let index_key = format!("{source_register}_{i}");
                if let Some(&value) = self.measurement_results.get(&index_key) {
                    register_value |= value << i;
                    found_values = true;
                    log::debug!("Found indexed value {}_{} = {}", source_register, i, value);
                }
            }

            if found_values {
                log::debug!(
                    "Exporting {} = {} (assembled from bits)",
                    export_name,
                    register_value
                );
                exported_values.insert(export_name.clone(), register_value);
                continue;
            }

            // Check raw measurement results as last resort
            // This handles the case where we didn't capture the measurements in indexed form
            let mut measurement_values = Vec::new();

            for (key, &value) in &self.measurement_results {
                if key.starts_with(MEASUREMENT_PREFIX) {
                    if let Some(idx_str) = key.strip_prefix(MEASUREMENT_PREFIX) {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            measurement_values.push((idx, value));
                            log::debug!("Found measurement value {} at index {}", value, idx);
                        }
                    }
                }
            }

            if !measurement_values.is_empty() {
                // Sort by index to maintain correct order
                measurement_values.sort_by_key(|(idx, _)| *idx);
                let combined_value_str: String = measurement_values
                    .iter()
                    .map(|(_, value)| value.to_string())
                    .collect();

                // Convert combined value to a number
                if let Ok(combined_value) = combined_value_str.parse::<u32>() {
                    log::debug!(
                        "Exporting {} = {} (from raw measurements)",
                        export_name,
                        combined_value
                    );
                    exported_values.insert(export_name.clone(), combined_value);
                    continue;
                }
            }

            log::debug!("No values found to export for {}", source_register);
        }

        // Add all exported values to the results
        log::debug!(
            "PHIR: Adding {} exported values to results",
            exported_values.len()
        );

        for (key, &value) in &exported_values {
            results.registers.insert(key.clone(), value);
            results.registers_u64.insert(key.clone(), u64::from(value));
            log::debug!("PHIR: Adding exported register {} = {}", key, value);
        }

        // Sanity check - this should only happen if measurements failed or weren't taken
        if results.registers.is_empty() && !self.export_mappings.is_empty() {
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
                current_op: 0, // Reset state in the clone
                measurement_results: HashMap::new(),
                exported_values: HashMap::new(),
                export_mappings: Vec::new(), // Reset export mappings in clone
                quantum_variables: self.quantum_variables.clone(),
                classical_variables: self.classical_variables.clone(),
                message_builder: ByteMessageBuilder::new(),
            },
            None => Self::empty(),
        }
    }
}

impl PHIREngine {
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
        format: crate::core::shot_results::OutputFormat,
    ) -> Result<String, PecosError> {
        let shot_result = self.get_results()?;

        // Convert single ShotResult to ShotResults for better formatting
        let mut shot_results = crate::core::shot_results::ShotResults::new();

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_phir_engine_basic() -> Result<(), PecosError> {
        let dir = tempdir().map_err(PecosError::IO)?;
        let program_path = dir.path().join("test.json");

        // Create a test program
        let program = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"test": "true"},
    "ops": [
        {
            "data": "qvar_define",
            "data_type": "qubits",
            "variable": "q",
            "size": 2
        },
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "m",
            "size": 2
        },
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "result",
            "size": 2
        },
        {
            "qop": "R1XY",
            "angles": [[0.1, 0.2], "rad"],
            "args": [["q", 0]]
        },
        {
            "qop": "Measure",
            "args": [["q", 0]],
            "returns": [["m", 0]]
        },
        {"cop": "Result", "args": [["m", 0]], "returns": [["result", 0]]}
    ]
}"#;

        let mut file = File::create(&program_path).map_err(PecosError::IO)?;
        file.write_all(program.as_bytes()).map_err(PecosError::IO)?;

        let mut engine = PHIREngine::new(&program_path)?;

        // Generate commands and verify they're correctly generated
        let command_message = engine.generate_commands()?;

        // Parse the message back to confirm it has the correct operations
        let parsed_commands = command_message.parse_quantum_operations().map_err(|e| {
            PecosError::Input(format!(
                "PHIR test failed: Unable to validate generated quantum operations: {e}"
            ))
        })?;
        assert_eq!(parsed_commands.len(), 2);

        // Create a measurement message and test handling
        // result_id=0, outcome=1
        let message = ByteMessage::builder()
            .add_measurement_results(&[1], &[0])
            .build();

        engine.handle_measurements(message)?;

        // Execute the "Result" classical operation to copy measurement to result
        // Set current_op to position of the Result op
        engine.current_op = 5;

        // Convert to ArgItem format for handle_classical_op
        let args = vec![ArgItem::Indexed(("m".to_string(), 0))];
        let returns = vec![ArgItem::Indexed(("result".to_string(), 0))];

        engine.handle_classical_op("Result", &args, &returns)?;

        // Verify results
        let results = engine.get_results()?;

        // With our implementation, the Result operation should make only the exported register
        // visible in the results. "measurement_0" should no longer be included.
        assert!(
            !results.registers.contains_key("measurement_0"),
            "Internal measurement register should not be in results when using Result instruction"
        );

        // The Result operation maps "m" to "result", so only "result" should be in the output
        assert!(
            results.registers.contains_key("result"),
            "result register should be in results"
        );
        assert_eq!(
            results.registers["result"], 1,
            "result register should have value 1"
        );
        assert_eq!(
            results.registers.len(),
            1,
            "There should be exactly one register in the results"
        );

        Ok(())
    }
}
