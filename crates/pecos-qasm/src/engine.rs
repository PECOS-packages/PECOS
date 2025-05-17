#![allow(clippy::similar_names)]

use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::{ByteMessage, ClassicalEngine, ControlEngine, Engine, EngineStage, ShotResult};
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use crate::ast::{EvaluationContext, Expression, Operation};
use crate::parser::{Program, QASMParser};

/// Gate handler function type
type GateHandler = fn(&mut QASMEngine, &[usize], &[f64]) -> Result<(), PecosError>;

/// Gate information for table-driven processing
struct GateInfo {
    name: &'static str,
    required_qubits: usize,
    required_params: usize,
    handler: GateHandler,
}

/// A QASM Engine that can generate native commands from a QASM program
#[derive(Debug)]
pub struct QASMEngine {
    /// The QASM Program being executed
    program: Option<Program>,

    /// Mapping from result IDs to register names and bit indices
    register_result_mappings: Vec<(u32, String, usize)>,

    /// Classical register values
    classical_registers: HashMap<String, Vec<u32>>,

    /// Raw measurement results (may include bits not in classical registers)
    raw_measurements: HashMap<u32, u32>,

    /// Next available result ID to use for measurements
    next_result_id: u32,

    /// Current operation index in the program
    current_op: usize,

    /// Reusable message builder for generating commands
    message_builder: ByteMessageBuilder,

    /// When true, allows general expressions in if statements
    allow_complex_conditionals: bool,
}

impl QASMEngine {
    // Maximum batch size for quantum operations
    const MAX_BATCH_SIZE: usize = 100;

    /// Create a builder for more complex configurations
    #[must_use]
    pub fn builder() -> crate::engine_builder::QASMEngineBuilder {
        crate::engine_builder::QASMEngineBuilder::new()
    }

    /// Create a new `QASMEngine` and load a QASM program from a file
    pub fn from_file(qasm_path: impl AsRef<Path>) -> Result<Self, PecosError> {
        let mut engine = Self::default();
        let program = QASMParser::parse_file(qasm_path)?;
        engine.load_program(program);

        if let Some(program) = &engine.program {
            let total_qubits = program.total_qubits;
            debug!(
                "Loaded QASM with {} qubits across {} registers",
                total_qubits,
                program.quantum_registers.len()
            );
        }

        Ok(engine)
    }

    /// Load a QASM program into the engine
    pub(crate) fn load_program(&mut self, program: Program) {
        debug!(
            "Loading QASM program with {} quantum registers and {} operations",
            program.quantum_registers.len(),
            program.operations.len()
        );

        debug!(
            "Total qubits from quantum registers: {}",
            program.total_qubits
        );

        // Initialize simulation components
        self.classical_registers.clear();
        self.raw_measurements.clear();
        self.register_result_mappings.clear();
        self.next_result_id = 0;

        self.program = Some(program);
        self.reset_state();
    }

    /// Enable or disable complex conditionals (general expressions in if statements)
    pub fn allow_complex_conditionals(&mut self, allow: bool) -> &mut Self {
        self.allow_complex_conditionals = allow;
        self
    }

    /// Check if complex conditionals are enabled
    #[must_use]
    pub fn complex_conditionals_enabled(&self) -> bool {
        self.allow_complex_conditionals
    }

    /// Get access to the gate definitions from the loaded program
    #[must_use]
    pub fn gate_definitions(
        &self,
    ) -> Option<&std::collections::BTreeMap<String, crate::ast::GateDefinition>> {
        self.program.as_ref().map(|p| &p.gate_definitions)
    }

    /// Get the physical qubit ID for a given quantum register and index
    #[must_use]
    pub fn qubit_id(&self, register_name: &str, index: usize) -> Option<usize> {
        if let Some(program) = &self.program {
            if let Some(qubit_ids) = program.quantum_registers.get(register_name) {
                if index < qubit_ids.len() {
                    return Some(qubit_ids[index]);
                }
            }
        }
        None
    }

    /// Reset the engine's internal state
    fn reset_state(&mut self) {
        debug!("QASMEngine::reset_state()");

        // Reset counters and operational state
        self.current_op = 0;
        self.next_result_id = 0;

        // Clear all collections
        self.raw_measurements.clear();
        self.register_result_mappings.clear();
        self.classical_registers.clear();
        self.message_builder.reset();

        // Re-initialize from program if available
        if let Some(program) = &self.program {
            debug!(
                "Initializing {} classical registers from program",
                program.classical_registers.len()
            );

            // Initialize classical registers to zero
            for (reg_name, size) in &program.classical_registers {
                self.classical_registers
                    .insert(reg_name.clone(), vec![0; *size]);
            }

            debug!(
                "Reset complete. Engine ready with {} classical registers",
                self.classical_registers.len()
            );
        } else {
            debug!("Reset complete. No program loaded.");
        }
    }

    fn update_register_bit(
        &mut self,
        register_name: &str,
        bit_index: usize,
        value: u8,
    ) -> Result<(), PecosError> {
        // Validate bounds if we have a program loaded
        if let Some(program) = &self.program {
            if let Some(size) = program.classical_registers.get(register_name) {
                if bit_index >= *size {
                    return Err(PecosError::Input(format!(
                        "Classical register bit index {bit_index} out of bounds for register '{register_name}' of size {size}"
                    )));
                }
            } else {
                return Err(PecosError::Input(format!(
                    "Classical register '{register_name}' not found"
                )));
            }
        }

        // Get or create the register
        let register = self
            .classical_registers
            .entry(register_name.to_string())
            .or_default();

        // Ensure the register has enough space
        if register.len() <= bit_index {
            register.resize(bit_index + 1, 0);
        }

        // Set the value
        register[bit_index] = u32::from(value);
        Ok(())
    }

    /// Gate handler functions
    #[allow(clippy::unnecessary_wraps)]
    fn handle_h(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_h(&[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_x(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_x(&[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_y(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_y(&[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_z(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_z(&[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_s(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine
            .message_builder
            .add_rz(std::f64::consts::PI / 2.0, &[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_sdg(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine
            .message_builder
            .add_rz(-std::f64::consts::PI / 2.0, &[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_t(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine
            .message_builder
            .add_rz(std::f64::consts::PI / 4.0, &[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_tdg(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine
            .message_builder
            .add_rz(-std::f64::consts::PI / 4.0, &[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_rz(
        engine: &mut QASMEngine,
        qubits: &[usize],
        params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_rz(params[0], &[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_r1xy(
        engine: &mut QASMEngine,
        qubits: &[usize],
        params: &[f64],
    ) -> Result<(), PecosError> {
        engine
            .message_builder
            .add_r1xy(params[0], params[1], &[qubits[0]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_cx(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_cx(&[qubits[0]], &[qubits[1]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_cy(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        // CY = S† · CX · S
        engine
            .message_builder
            .add_rz(-std::f64::consts::PI / 2.0, &[qubits[1]]); // S†
        engine.message_builder.add_cx(&[qubits[0]], &[qubits[1]]);
        engine
            .message_builder
            .add_rz(std::f64::consts::PI / 2.0, &[qubits[1]]); // S
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_cz(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        // CZ = H · CX · H
        engine.message_builder.add_h(&[qubits[1]]);
        engine.message_builder.add_cx(&[qubits[0]], &[qubits[1]]);
        engine.message_builder.add_h(&[qubits[1]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_rzz(
        engine: &mut QASMEngine,
        qubits: &[usize],
        params: &[f64],
    ) -> Result<(), PecosError> {
        engine
            .message_builder
            .add_rzz(params[0], &[qubits[0]], &[qubits[1]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_szz(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        engine.message_builder.add_szz(&[qubits[0]], &[qubits[1]]);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_swap(
        engine: &mut QASMEngine,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), PecosError> {
        // SWAP = CX · CX · CX
        engine.message_builder.add_cx(&[qubits[0]], &[qubits[1]]);
        engine.message_builder.add_cx(&[qubits[1]], &[qubits[0]]);
        engine.message_builder.add_cx(&[qubits[0]], &[qubits[1]]);
        Ok(())
    }

    /// Get the gate table for table-driven processing
    fn get_gate_table() -> Vec<GateInfo> {
        vec![
            // Single-qubit gates
            GateInfo {
                name: "h",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_h,
            },
            GateInfo {
                name: "x",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_x,
            },
            GateInfo {
                name: "y",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_y,
            },
            GateInfo {
                name: "z",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_z,
            },
            GateInfo {
                name: "s",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_s,
            },
            GateInfo {
                name: "sdg",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_sdg,
            },
            GateInfo {
                name: "t",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_t,
            },
            GateInfo {
                name: "tdg",
                required_qubits: 1,
                required_params: 0,
                handler: Self::handle_tdg,
            },
            GateInfo {
                name: "rz",
                required_qubits: 1,
                required_params: 1,
                handler: Self::handle_rz,
            },
            GateInfo {
                name: "r1xy",
                required_qubits: 1,
                required_params: 2,
                handler: Self::handle_r1xy,
            },
            // Two-qubit gates
            GateInfo {
                name: "cx",
                required_qubits: 2,
                required_params: 0,
                handler: Self::handle_cx,
            },
            GateInfo {
                name: "cy",
                required_qubits: 2,
                required_params: 0,
                handler: Self::handle_cy,
            },
            GateInfo {
                name: "cz",
                required_qubits: 2,
                required_params: 0,
                handler: Self::handle_cz,
            },
            GateInfo {
                name: "rzz",
                required_qubits: 2,
                required_params: 1,
                handler: Self::handle_rzz,
            },
            GateInfo {
                name: "szz",
                required_qubits: 2,
                required_params: 0,
                handler: Self::handle_szz,
            },
            GateInfo {
                name: "swap",
                required_qubits: 2,
                required_params: 0,
                handler: Self::handle_swap,
            },
        ]
    }

    /// Process a single gate operation using table-driven approach
    fn process_gate_operation(
        &mut self,
        name: &str,
        qubits: &[usize],
        parameters: &[f64],
    ) -> Result<bool, PecosError> {
        let gate_table = Self::get_gate_table();
        let name_lower = name.to_lowercase();

        // Find the gate in the table
        for gate_info in &gate_table {
            if gate_info.name == name_lower {
                // Validate qubit count
                if qubits.len() != gate_info.required_qubits {
                    return Err(PecosError::Input(format!(
                        "{} gate requires {} qubit{}, got {}",
                        gate_info.name,
                        gate_info.required_qubits,
                        if gate_info.required_qubits == 1 {
                            ""
                        } else {
                            "s"
                        },
                        qubits.len()
                    )));
                }

                // Validate parameter count
                if parameters.len() < gate_info.required_params {
                    return Err(PecosError::Input(format!(
                        "{} gate requires {} parameter{}, got {}",
                        gate_info.name,
                        gate_info.required_params,
                        if gate_info.required_params == 1 {
                            ""
                        } else {
                            "s"
                        },
                        parameters.len()
                    )));
                }

                // Apply the gate
                debug!("Applying {} gate", gate_info.name);
                (gate_info.handler)(self, qubits, parameters)?;
                return Ok(true);
            }
        }

        // Gate not supported
        Err(PecosError::Processing(format!("Unsupported gate: {name}")))
    }

    /// Process a measurement operation
    fn process_measurement(
        &mut self,
        qubit: usize,
        c_reg: &str,
        c_index: usize,
    ) -> Result<(), PecosError> {
        let physical_qubit = qubit;
        let c_register_name = if c_reg.is_empty() { "c" } else { c_reg };

        // Validate classical register bounds
        if let Some(program) = &self.program {
            if let Some(size) = program.classical_registers.get(c_register_name) {
                if c_index >= *size {
                    return Err(PecosError::Input(format!(
                        "Classical register bit index {c_index} out of bounds for register '{c_register_name}' of size {size}"
                    )));
                }
            } else {
                return Err(PecosError::Input(format!(
                    "Classical register '{c_register_name}' not found"
                )));
            }
        }

        // Create a unique result ID
        let result_id = self.next_result_id;
        self.next_result_id += 1;

        // Store the mapping for result handling
        self.register_result_mappings
            .push((result_id, c_register_name.to_string(), c_index));

        debug!(
            "Adding measurement on qubit {} with result_id {}",
            physical_qubit, result_id
        );

        // Add measurement to the command batch
        self.message_builder.add_measurements(
            &[physical_qubit],
            &[usize::try_from(result_id).unwrap_or_default()],
        );

        Ok(())
    }

    /// Process a register measurement operation
    fn process_register_measurement(
        &mut self,
        q_reg: &str,
        c_reg: &str,
        program: &Program,
        current_operation_count: usize,
    ) -> Result<Option<usize>, PecosError> {
        let Some(qubit_ids) = program.quantum_registers.get(q_reg) else {
            return Err(PecosError::Input(format!(
                "Quantum register {q_reg} not found"
            )));
        };

        let Some(&c_size) = program.classical_registers.get(c_reg) else {
            return Err(PecosError::Input(format!(
                "Classical register {c_reg} not found"
            )));
        };

        let measure_count = std::cmp::min(qubit_ids.len(), c_size);

        debug!(
            "Will measure {} qubits from {} to {}",
            measure_count, q_reg, c_reg
        );

        let mut measurements_added = 0;
        for (i, &qubit_id) in qubit_ids.iter().enumerate().take(measure_count) {
            if current_operation_count + measurements_added >= Self::MAX_BATCH_SIZE {
                debug!(
                    "Reached maximum batch size during register measurement, will continue in next batch"
                );
                break;
            }

            self.process_measurement(qubit_id, c_reg, i)?;
            measurements_added += 1;
        }

        if measurements_added < measure_count {
            debug!(
                "Only processed {} of {} measurements in RegMeasure, will continue in next batch",
                measurements_added, measure_count
            );
            return Ok(None);
        }

        Ok(Some(measurements_added))
    }

    /// Process the QASM program and generate `ByteMessage`
    #[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
    fn process_program(&mut self) -> Result<ByteMessage, PecosError> {
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();

        let program = self
            .program
            .as_ref()
            .ok_or_else(|| PecosError::Input("No QASM program loaded".to_string()))?
            .clone();

        let total_ops = program.operations.len();

        debug!(
            "Processing program: current_op: {}/{}",
            self.current_op, total_ops
        );

        if self.current_op >= total_ops {
            debug!("End of program reached, sending flush");
            return Ok(ByteMessage::create_flush());
        }

        let mut operation_count = 0;

        while self.current_op < total_ops && operation_count < Self::MAX_BATCH_SIZE {
            let op = &program.operations[self.current_op];

            match op {
                Operation::Gate {
                    name,
                    parameters,
                    qubits,
                } => {
                    if self.process_gate_operation(name, qubits, parameters)? {
                        operation_count += 1;
                    }
                }
                Operation::Measure {
                    qubit,
                    c_reg,
                    c_index,
                } => {
                    self.process_measurement(*qubit, c_reg, *c_index)?;
                    self.current_op += 1;
                    debug!("Breaking batch after measurement to wait for results");
                    return Ok(self.message_builder.build());
                }
                Operation::RegMeasure { q_reg, c_reg } => {
                    let added_count =
                        self.process_register_measurement(q_reg, c_reg, &program, operation_count)?;

                    if let Some(count) = added_count {
                        operation_count += count;
                    } else {
                        return Ok(self.message_builder.build());
                    }
                }
                Operation::If {
                    condition,
                    operation,
                } => {
                    if !self.allow_complex_conditionals {
                        if let Expression::BinaryOp { op: _, left, right } = condition {
                            let is_valid = matches!(
                                (left.as_ref(), right.as_ref()),
                                (
                                    Expression::Variable(_) | Expression::BitId(_, _),
                                    Expression::Integer(_)
                                )
                            );

                            if !is_valid {
                                return Err(PecosError::Processing(
                                    "Complex conditionals are not allowed. Only register/bit compared to integer is supported in standard OpenQASM 2.0. Enable allow_complex_conditionals to use general expressions.".to_string()
                                ));
                            }
                        } else {
                            return Err(PecosError::Processing(
                                "Invalid conditional format. Expected comparison expression."
                                    .to_string(),
                            ));
                        }
                    }

                    debug!("Evaluating if condition: {:?}", condition);
                    let condition_value = self.evaluate_expression_with_context(condition)?;
                    debug!("Condition value: {}", condition_value);

                    if condition_value != 0 {
                        debug!(
                            "If condition evaluated to true, executing operation: {:?}",
                            operation
                        );

                        match operation.as_ref() {
                            Operation::Gate {
                                name,
                                parameters,
                                qubits,
                            } => {
                                debug!(
                                    "Executing conditional gate {} on qubits {:?}",
                                    name, qubits
                                );
                                if self.process_gate_operation(name, qubits, parameters)? {
                                    operation_count += 1;
                                }
                            }
                            Operation::ClassicalAssignment {
                                target,
                                is_indexed,
                                index,
                                expression,
                            } => {
                                let value = self.evaluate_expression_with_context(expression)?;

                                if *is_indexed {
                                    if let Some(idx) = *index {
                                        self.update_register_bit(
                                            target,
                                            idx,
                                            u8::from(value != 0),
                                        )?;
                                    }
                                } else if let Some(register_size) =
                                    program.classical_registers.get(target.as_str())
                                {
                                    let mut bits = vec![0u32; *register_size];

                                    for (i, bit) in bits.iter_mut().enumerate().take(*register_size)
                                    {
                                        if i < 32 {
                                            *bit = ((value >> i) & 1) as u32;
                                        }
                                    }

                                    debug!(
                                        "Setting register {} to value {} (bits: {:?})",
                                        target, value, bits
                                    );

                                    self.classical_registers.insert(target.clone(), bits);
                                }
                                operation_count += 1;
                            }
                            _ => {
                                debug!("Unsupported operation in if statement");
                            }
                        }
                    } else {
                        debug!("If condition evaluated to false, skipping operation");
                    }
                }
                Operation::ClassicalAssignment {
                    target,
                    is_indexed,
                    index,
                    expression,
                } => {
                    debug!(
                        "Processing classical assignment: {} = {:?}",
                        target, expression
                    );

                    let value = self.evaluate_expression_with_context(expression)?;

                    if *is_indexed {
                        if let Some(idx) = *index {
                            self.update_register_bit(target, idx, u8::from(value != 0))?;
                        }
                    } else if let Some(register_size) =
                        program.classical_registers.get(target.as_str())
                    {
                        let mut bits = vec![0u32; *register_size];

                        for (i, bit) in bits.iter_mut().enumerate().take(*register_size) {
                            if i < 32 {
                                *bit = ((value >> i) & 1) as u32;
                            }
                        }

                        debug!(
                            "Setting register {} to value {} (bits: {:?})",
                            target, value, bits
                        );

                        self.classical_registers.insert(target.clone(), bits);
                    }

                    operation_count += 1;
                }
                _ => {
                    debug!("Skipping unsupported operation type");
                }
            }
            self.current_op += 1;
        }

        Ok(self.message_builder.build())
    }

    /// Evaluate an expression with access to register values
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn evaluate_expression_with_context(&self, expr: &Expression) -> Result<i64, PecosError> {
        match expr {
            Expression::Integer(i) => Ok(*i),
            Expression::Float(f) =>
            {
                #[allow(clippy::cast_possible_truncation)]
                Ok(*f as i64)
            }
            Expression::Variable(name) => {
                if let Some(bits) = self.classical_registers.get(name) {
                    let mut value = 0i64;
                    for (i, &bit) in bits.iter().enumerate() {
                        if i < 32 {
                            value |= i64::from(bit & 1) << i;
                        }
                    }
                    Ok(value)
                } else {
                    debug!("Register {} not found", name);
                    Ok(0)
                }
            }
            Expression::BitId(reg_name, idx) => {
                let bit_value = self
                    .classical_registers
                    .get(reg_name)
                    .and_then(|reg| {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        reg.get(*idx as usize)
                    })
                    .copied()
                    .unwrap_or(0);
                debug!("Evaluating bit {}.{} = {}", reg_name, idx, bit_value);
                Ok(i64::from(bit_value))
            }
            Expression::BinaryOp { op, left, right } => {
                let left_val = self.evaluate_expression_with_context(left)?;
                let right_val = self.evaluate_expression_with_context(right)?;
                debug!("Binary op: {} {} {} = ?", left_val, op, right_val);

                match op.as_str() {
                    "+" => Ok(left_val + right_val),
                    "-" => Ok(left_val - right_val),
                    "*" => Ok(left_val * right_val),
                    "/" => {
                        if right_val != 0 {
                            Ok(left_val / right_val)
                        } else {
                            debug!("Division by zero");
                            Ok(0)
                        }
                    }
                    "&" => Ok(left_val & right_val),
                    "|" => Ok(left_val | right_val),
                    "^" => Ok(left_val ^ right_val),
                    "==" => Ok(i64::from(left_val == right_val)),
                    "!=" => Ok(i64::from(left_val != right_val)),
                    "<" => Ok(i64::from(left_val < right_val)),
                    ">" => Ok(i64::from(left_val > right_val)),
                    "<=" => Ok(i64::from(left_val <= right_val)),
                    ">=" => Ok(i64::from(left_val >= right_val)),
                    "<<" => Ok(left_val << right_val),
                    ">>" => Ok(left_val >> right_val),
                    _ => {
                        debug!("Unsupported binary operation: {}", op);
                        Err(PecosError::Processing(format!(
                            "Unsupported operation: {op}"
                        )))
                    }
                }
            }
            Expression::UnaryOp { op, expr } => {
                let val = self.evaluate_expression_with_context(expr)?;
                match op.as_str() {
                    "-" => Ok(-val),
                    "~" => Ok(!val),
                    _ => {
                        debug!("Unsupported unary operation: {}", op);
                        Err(PecosError::Processing(format!(
                            "Unsupported operation: {op}"
                        )))
                    }
                }
            }
            _ => {
                debug!("Unsupported expression type: {:?}", expr);
                Err(PecosError::Processing(format!(
                    "Unsupported expression: {expr:?}"
                )))
            }
        }
    }
}

impl ClassicalEngine for QASMEngine {
    fn num_qubits(&self) -> usize {
        if let Some(program) = &self.program {
            program.total_qubits
        } else {
            0
        }
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        debug!("QASMEngine::generate_commands() called");

        if self.program.is_none() {
            debug!("No program loaded, returning empty message");
            self.message_builder.reset();
            let _ = self.message_builder.for_quantum_operations();
            return Ok(self.message_builder.build());
        }

        if let Some(program) = &self.program {
            debug!(
                "Current operation: {}/{}",
                self.current_op,
                program.operations.len()
            );

            if self.current_op >= program.operations.len() {
                debug!("End of program detected, returning flush message");
                return Ok(ByteMessage::create_flush());
            }
        }

        if self.current_op == 0 {
            debug!("Starting a new shot (current_op=0)");
            self.message_builder.reset();
            let _ = self.message_builder.for_quantum_operations();
        }

        debug!("Processing program from operation {}", self.current_op);
        let result = self.process_program();
        debug!("Program processing complete");
        result.map_err(|e| {
            PecosError::Processing(format!("QASM engine failed to process program: {e}"))
        })
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        debug!("Handling measurements from ByteMessage");

        match message.measurement_results_as_vec() {
            Ok(results) => {
                let mappings = self.register_result_mappings.clone();

                debug!("Processing {} measurement results", results.len());

                for (result_id, value) in results {
                    debug!("Found measurement result_id={} value={}", result_id, value);

                    if let Some((_, register, bit)) = mappings
                        .iter()
                        .find(|(id, _, _)| *id == u32::try_from(result_id).unwrap_or_default())
                    {
                        debug!(
                            "Updating register {}[{}] with value {}",
                            register, bit, value
                        );

                        let safe_value = u8::try_from(value).unwrap_or(1);
                        self.update_register_bit(register, *bit, safe_value)?;
                    } else {
                        debug!("No register mapping found for result_id={}", result_id);
                    }

                    if let Ok(u32_id) = u32::try_from(result_id) {
                        self.raw_measurements.insert(u32_id, value);
                    }
                }

                Ok(())
            }
            Err(e) => {
                debug!("Error parsing measurement results: {:?}", e);
                Err(PecosError::Input(format!(
                    "Error parsing measurement results: {e}"
                )))
            }
        }
    }

    fn get_results(&self) -> Result<ShotResult, PecosError> {
        let mut result = ShotResult::default();

        let mut reg_names: Vec<_> = self.classical_registers.keys().collect();
        reg_names.sort();

        for reg_name in &reg_names {
            if let Some(values) = self.classical_registers.get(*reg_name) {
                let reg_value = values.iter().enumerate().fold(0, |acc, (i, &v)| {
                    if i >= 32 || v == 0 {
                        acc
                    } else {
                        acc | (v << i)
                    }
                });

                let reg_name_str = (*reg_name).to_string();
                result.registers.insert(reg_name_str.clone(), reg_value);
                result.registers_u64.insert(reg_name_str, reg_value.into());
            }
        }

        Ok(result)
    }

    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
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

impl Clone for QASMEngine {
    fn clone(&self) -> Self {
        let mut engine = Self {
            program: self.program.clone(),
            allow_complex_conditionals: self.allow_complex_conditionals,
            ..Self::default()
        };

        // Re-initialize classical registers from program
        if let Some(program) = &engine.program {
            for (reg_name, size) in &program.classical_registers {
                engine
                    .classical_registers
                    .insert(reg_name.clone(), vec![0; *size]);
            }
        }

        engine
    }
}

impl ControlEngine for QASMEngine {
    type Input = ();
    type Output = ShotResult;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, ShotResult>, PecosError> {
        debug!("QASMEngine::start() called");

        debug!("Preparing engine for new shot");
        self.reset_state();
        self.current_op = 0;

        debug!("Generating initial commands for simulation");
        let commands = self.generate_commands()?;

        if commands.is_empty()? {
            debug!("No commands to process, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            debug!("Commands generated, returning NeedsProcessing");
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ShotResult>, PecosError> {
        debug!("QASMEngine::continue_processing() called");

        let measurement_count = measurements
            .measurement_results_as_vec()
            .map(|results| results.len())
            .unwrap_or(0);
        debug!("Received {} measurements", measurement_count);

        debug!("Processing measurement results");
        self.handle_measurements(measurements)?;

        debug!("Generating next batch of commands");
        let commands = self.generate_commands()?;

        if commands.is_empty()? {
            debug!("No more commands, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            debug!("Unexpected additional commands generated");
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        <Self as ClassicalEngine>::reset(self)
    }
}

impl Engine for QASMEngine {
    type Input = ();
    type Output = ShotResult;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        debug!("QASMEngine::process() called");

        <Self as ClassicalEngine>::reset(self)?;

        debug!("Starting engine to produce commands");
        let stage = self
            .start(input)
            .map_err(|e| PecosError::Processing(format!("Failed to start QASMEngine: {e}")))?;

        match stage {
            EngineStage::Complete(result) => {
                debug!("Shot completed directly in start()");
                Ok(result)
            }
            EngineStage::NeedsProcessing(cmds) => {
                debug!("Processing commands from start()");

                if cmds.is_empty().map_err(|e| {
                    PecosError::Processing(format!("Failed to check if commands are empty: {e}"))
                })? {
                    debug!("Received empty commands, treating as completion");
                    Ok(self.get_results()?)
                } else {
                    debug!("QASMEngine cannot process quantum operations directly");
                    Ok(self.get_results()?)
                }
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        <Self as ControlEngine>::reset(self)
    }
}

impl Default for QASMEngine {
    fn default() -> Self {
        debug!("Creating new QASMEngine");
        Self {
            program: None,
            register_result_mappings: Vec::new(),
            classical_registers: HashMap::new(),
            raw_measurements: HashMap::new(),
            next_result_id: 0,
            current_op: 0,
            message_builder: ByteMessageBuilder::new(),
            allow_complex_conditionals: false,
        }
    }
}

impl EvaluationContext for QASMEngine {
    #[allow(clippy::cast_precision_loss)]
    fn evaluate_float(&self, expr: &Expression) -> Result<f64, PecosError> {
        self.evaluate_expression_with_context(expr)
            .map(|i| i as f64)
    }

    fn evaluate_int(&self, expr: &Expression) -> Result<i64, PecosError> {
        self.evaluate_expression_with_context(expr)
    }
}

impl FromStr for QASMEngine {
    type Err = PecosError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut engine = Self::default();
        let program = QASMParser::parse_str(s)?;
        engine.load_program(program);
        Ok(engine)
    }
}
