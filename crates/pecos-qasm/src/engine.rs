#![allow(clippy::similar_names)]

use bitvec::prelude::*;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::prelude::*;
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use crate::ast::{Expression, Operation};
use crate::bitvec_expression::{
    BitVecExpressionContext, ExpressionValue, evaluate_expression_bitvec,
};
use crate::program::QASMProgram;

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
pub struct QASMEngine {
    /// The QASM Program being executed
    program: Option<QASMProgram>,

    /// Mapping from measurement order to register names and bit indices
    /// Each entry is (`register_name`, `bit_index`) mapped by the order of measurements
    register_result_mappings: Vec<(String, usize)>,

    /// Classical register values stored as `BitVecs`
    classical_registers: BTreeMap<String, BitVec<u8, Lsb0>>,

    /// Raw measurement results (may include bits not in classical registers)
    raw_measurements: BTreeMap<u32, u32>,

    /// Next available result ID to use for measurements

    /// Current operation index in the program
    current_op: usize,

    /// Number of measurements processed so far
    measurements_processed: usize,

    /// Reusable message builder for generating commands
    message_builder: ByteMessageBuilder,

    /// When true, allows general expressions in if statements
    allow_complex_conditionals: bool,

    /// Foreign object for WASM function calls
    #[cfg(feature = "wasm")]
    foreign_object: Option<Box<dyn crate::foreign_objects::ForeignObject>>,
}

impl QASMEngine {
    // Maximum batch size for quantum operations
    const MAX_BATCH_SIZE: usize = 100;

    /// Create a builder for more complex configurations
    #[must_use]
    pub fn builder() -> crate::engine_builder::QASMEngineBuilder {
        crate::engine_builder::QASMEngineBuilder::new()
    }

    /// Create a new `QASMEngine` from a `QASMProgram`
    ///
    /// This is generally used internally. Users should prefer `from_str` or `from_file`.
    #[must_use]
    pub fn new(program: QASMProgram) -> Self {
        let mut engine = Self::default();
        engine.load_program(program);
        engine
    }

    /// Create a new `QASMEngine` and load a QASM program from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(qasm_path: impl AsRef<Path>) -> Result<Self, PecosError> {
        // Import here to avoid circular dependency
        use crate::program::QASMProgram;

        // Parse the program
        let program = QASMProgram::from_file(qasm_path)?;

        // Convert to engine
        Ok(program.into_engine())
    }

    /// Load a QASM program into the engine
    /// Set the foreign object for WASM function calls
    #[cfg(feature = "wasm")]
    pub fn set_foreign_object(
        &mut self,
        foreign_object: Box<dyn crate::foreign_objects::ForeignObject>,
    ) {
        self.foreign_object = Some(foreign_object);
    }

    pub(crate) fn load_program(&mut self, program: QASMProgram) {
        let ast = program.program();
        debug!(
            "Loading QASM program with {} quantum registers and {} operations",
            ast.quantum_registers.len(),
            ast.operations.len()
        );

        debug!("Total qubits from quantum registers: {}", ast.total_qubits);

        // Initialize simulation components
        self.classical_registers.clear();
        self.raw_measurements.clear();
        self.register_result_mappings.clear();

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
        self.program.as_ref().map(|p| &p.program().gate_definitions)
    }

    /// Get the physical qubit ID for a given quantum register and index
    #[must_use]
    pub fn qubit_id(&self, register_name: &str, index: usize) -> Option<usize> {
        if let Some(qasm_program) = &self.program {
            let program = qasm_program.program();
            if let Some(qubit_ids) = program.quantum_registers.get(register_name)
                && index < qubit_ids.len()
            {
                return Some(qubit_ids[index]);
            }
        }
        None
    }

    /// Get the classical register sizes (bit widths)
    #[must_use]
    pub fn classical_register_sizes(&self) -> Option<&std::collections::BTreeMap<String, usize>> {
        self.program
            .as_ref()
            .map(|p| &p.program().classical_registers)
    }

    /// Reset the engine's internal state
    fn reset_state(&mut self) {
        debug!("QASMEngine::reset_state()");

        // Reset counters and operational state
        self.current_op = 0;
        self.measurements_processed = 0;

        // Clear all collections
        self.raw_measurements.clear();
        self.register_result_mappings.clear();
        self.classical_registers.clear();
        self.message_builder.reset();

        // Reset WASM state for new shot
        #[cfg(feature = "wasm")]
        if let Some(ref mut foreign_obj) = self.foreign_object
            && let Err(e) = foreign_obj.new_instance()
        {
            log::error!("Failed to reset WASM instance: {e}");
        }

        // Re-initialize from program if available
        if let Some(qasm_program) = &self.program {
            let program = qasm_program.program();
            debug!(
                "Initializing {} classical registers from program",
                program.classical_registers.len()
            );

            // Initialize classical registers as BitVecs
            for (reg_name, size) in &program.classical_registers {
                let bitvec = BitVec::<u8, Lsb0>::repeat(false, *size);
                self.classical_registers.insert(reg_name.clone(), bitvec);
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
        if let Some(qasm_program) = &self.program {
            let program = qasm_program.program();
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

        // Get the register
        let register = self
            .classical_registers
            .get_mut(register_name)
            .ok_or_else(|| {
                PecosError::Input(format!("Classical register '{register_name}' not found"))
            })?;

        // Set the bit value
        register.set(bit_index, value != 0);
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

    /// Process single-qubit gates
    fn process_single_qubit_gate(
        &mut self,
        gate_type: pecos_core::prelude::GateType,
        qubits: &[usize],
    ) -> Result<(), PecosError> {
        use pecos_core::prelude::GateType;

        for &qubit in qubits {
            match gate_type {
                GateType::X => self.message_builder.add_x(&[qubit]),
                GateType::Y => self.message_builder.add_y(&[qubit]),
                GateType::Z => self.message_builder.add_z(&[qubit]),
                GateType::H => self.message_builder.add_h(&[qubit]),
                GateType::Prep => self.message_builder.add_prep(&[qubit]),
                _ => {
                    return Err(PecosError::Processing(format!(
                        "Gate type {gate_type:?} is not a single-qubit gate"
                    )));
                }
            };
        }
        Ok(())
    }

    /// Process two-qubit gates
    fn process_two_qubit_gate(
        &mut self,
        gate_type: pecos_core::prelude::GateType,
        qubits: &[usize],
    ) -> Result<(), PecosError> {
        use pecos_core::prelude::GateType;

        for chunk in qubits.chunks(2) {
            if chunk.len() == 2 {
                match gate_type {
                    GateType::CX => self.message_builder.add_cx(&[chunk[0]], &[chunk[1]]),
                    GateType::SZZ => self.message_builder.add_szz(&[chunk[0]], &[chunk[1]]),
                    GateType::SZZdg => self.message_builder.add_szzdg(&[chunk[0]], &[chunk[1]]),
                    _ => {
                        return Err(PecosError::Processing(format!(
                            "Gate type {gate_type:?} is not a two-qubit gate"
                        )));
                    }
                };
            }
        }
        Ok(())
    }

    /// Process parameterized gates
    fn process_parameterized_gate(
        &mut self,
        gate_type: pecos_core::prelude::GateType,
        qubits: &[usize],
        params: &[f64],
    ) -> Result<(), PecosError> {
        use pecos_core::prelude::GateType;

        match gate_type {
            GateType::RZ => {
                if let Some(&angle) = params.first() {
                    for &qubit in qubits {
                        self.message_builder.add_rz(angle, &[qubit]);
                    }
                }
            }
            GateType::RZZ => {
                if let Some(&angle) = params.first() {
                    for chunk in qubits.chunks(2) {
                        if chunk.len() == 2 {
                            self.message_builder
                                .add_rzz(angle, &[chunk[0]], &[chunk[1]]);
                        }
                    }
                }
            }
            GateType::R1XY => {
                if params.len() >= 2 {
                    let theta = params[0];
                    let phi = params[1];
                    for &qubit in qubits {
                        self.message_builder.add_r1xy(theta, phi, &[qubit]);
                    }
                }
            }
            GateType::U => {
                if params.len() >= 3 {
                    let theta = params[0];
                    let phi = params[1];
                    let lambda = params[2];
                    for &qubit in qubits {
                        self.message_builder.add_u(theta, phi, lambda, &[qubit]);
                    }
                }
            }
            _ => {
                return Err(PecosError::Processing(format!(
                    "Gate type {gate_type:?} is not a parameterized gate"
                )));
            }
        }
        Ok(())
    }

    /// Process a native gate directly
    fn process_native_gate(&mut self, gate: &pecos_core::prelude::Gate) -> Result<(), PecosError> {
        use pecos_core::prelude::GateType;

        // Convert QubitIds to usize array
        let qubits: Vec<usize> = gate.qubits.iter().map(|q| q.0).collect();

        match gate.gate_type {
            GateType::I | GateType::Idle => Ok(()), // No-op gates
            GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::H
            | GateType::SZ
            | GateType::SZdg
            | GateType::T
            | GateType::Tdg
            | GateType::Prep => self.process_single_qubit_gate(gate.gate_type, &qubits),
            GateType::CX | GateType::SZZ | GateType::SZZdg => {
                self.process_two_qubit_gate(gate.gate_type, &qubits)
            }
            GateType::RZ | GateType::RZZ | GateType::R1XY | GateType::U => {
                self.process_parameterized_gate(gate.gate_type, &qubits, &gate.params)
            }
            GateType::Measure | GateType::MeasureLeaked => Err(PecosError::Processing(
                "Measure and MeasureLeaked gates should be handled by MeasureWithMapping operation"
                    .to_string(),
            )),
        }
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
        if let Some(qasm_program) = &self.program {
            let program = qasm_program.program();
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

        // Store the mapping for result handling by order
        self.register_result_mappings
            .push((c_register_name.to_string(), c_index));

        debug!(
            "Adding measurement on qubit {} (measurement #{})",
            physical_qubit,
            self.register_result_mappings.len() - 1
        );

        // Add measurement to the command batch
        self.message_builder.add_measurements(&[physical_qubit]);

        Ok(())
    }

    /// Process a register measurement operation
    fn process_register_measurement(
        &mut self,
        q_reg: &str,
        c_reg: &str,
        qasm_program: &QASMProgram,
        current_operation_count: usize,
    ) -> Result<Option<usize>, PecosError> {
        let program = qasm_program.program();
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

        debug!("Will measure {measure_count} qubits from {q_reg} to {c_reg}");

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
                "Only processed {measurements_added} of {measure_count} measurements in RegMeasure, will continue in next batch"
            );
            return Ok(None);
        }

        Ok(Some(measurements_added))
    }

    /// Process the QASM program and generate `ByteMessage`
    #[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
    fn process_program_impl(&mut self) -> Result<Option<ByteMessage>, PecosError> {
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();

        // Clone to avoid borrow checking issues
        let qasm_program = self
            .program
            .as_ref()
            .ok_or_else(|| PecosError::Input("No QASM program loaded".to_string()))?
            .clone();

        let program = qasm_program.program();

        let total_ops = program.operations.len();

        debug!(
            "Processing program: current_op: {}/{}",
            self.current_op, total_ops
        );

        if self.current_op >= total_ops {
            debug!("End of program reached, no more commands to generate");

            // With our updated HybridEngine and ControlEngine implementations,
            // we can now consistently return None when there are no more commands,
            // even for the first batch.
            return Ok(None);
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
                Operation::NativeGate(gate) => {
                    // Process native gate directly
                    self.process_native_gate(gate)?;
                    operation_count += 1;
                }
                Operation::MeasureWithMapping {
                    gate,
                    c_reg,
                    c_index,
                } => {
                    // Extract qubit from gate
                    if let Some(qubit_id) = gate.qubits.first() {
                        self.process_measurement(qubit_id.0, c_reg, *c_index)?;
                        self.current_op += 1;
                        debug!("Breaking batch after measurement to wait for results");
                        return Ok(Some(self.message_builder.build()));
                    }
                }
                Operation::RegMeasure { q_reg, c_reg } => {
                    let added_count = self.process_register_measurement(
                        q_reg,
                        c_reg,
                        &qasm_program,
                        operation_count,
                    )?;

                    if let Some(count) = added_count {
                        operation_count += count;
                    } else {
                        return Ok(Some(self.message_builder.build()));
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

                    debug!("Evaluating if condition: {condition:?}");
                    let condition_value = self.evaluate_expression_bitvec(condition)?.as_i64();
                    debug!("Condition value: {condition_value}");

                    if condition_value != 0 {
                        debug!(
                            "If condition evaluated to true, executing operation: {operation:?}"
                        );

                        match operation.as_ref() {
                            Operation::Gate {
                                name,
                                parameters,
                                qubits,
                            } => {
                                debug!("Executing conditional gate {name} on qubits {qubits:?}");
                                if self.process_gate_operation(name, qubits, parameters)? {
                                    operation_count += 1;
                                }
                            }
                            Operation::NativeGate(gate) => {
                                debug!(
                                    "Executing conditional native gate {:?} on qubits {:?}",
                                    gate.gate_type, gate.qubits
                                );
                                self.process_native_gate(gate)?;
                                operation_count += 1;
                            }
                            Operation::ClassicalAssignment {
                                target,
                                is_indexed,
                                index,
                                expression,
                            } => {
                                // Get target register size for width hint
                                let target_width = if *is_indexed {
                                    1 // Single bit assignment
                                } else {
                                    program
                                        .classical_registers
                                        .get(target.as_str())
                                        .copied()
                                        .unwrap_or(64)
                                };

                                let value_expr = self.evaluate_expression_bitvec_with_width(
                                    expression,
                                    target_width,
                                )?;

                                if *is_indexed {
                                    if let Some(idx) = *index {
                                        let bit_value = value_expr.into_bool();
                                        self.update_register_bit(target, idx, u8::from(bit_value))?;
                                    }
                                } else if let Some(register_size) =
                                    program.classical_registers.get(target.as_str())
                                {
                                    let mut result_bitvec = value_expr.into_bitvec();

                                    // Sign extend when resizing (use the MSB as the sign bit)
                                    let sign_bit = if result_bitvec.is_empty() {
                                        false
                                    } else {
                                        result_bitvec[result_bitvec.len() - 1]
                                    };

                                    // Resize to the exact register size with sign extension
                                    result_bitvec.resize(*register_size, sign_bit);

                                    debug!(
                                        "Setting register {} with BitVec of length {}",
                                        target,
                                        result_bitvec.len()
                                    );

                                    self.classical_registers
                                        .insert(target.clone(), result_bitvec);
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
                    debug!("Processing classical assignment: {target} = {expression:?}");

                    // Get target register size for width hint
                    let target_width = if *is_indexed {
                        1 // Single bit assignment
                    } else {
                        program
                            .classical_registers
                            .get(target.as_str())
                            .copied()
                            .unwrap_or(64)
                    };

                    let value_expr =
                        self.evaluate_expression_bitvec_with_width(expression, target_width)?;

                    if *is_indexed {
                        if let Some(idx) = *index {
                            let bit_value = value_expr.into_bool();
                            self.update_register_bit(target, idx, u8::from(bit_value))?;
                        }
                    } else if let Some(register_size) =
                        program.classical_registers.get(target.as_str())
                    {
                        let mut result_bitvec = value_expr.into_bitvec();

                        // Sign extend when resizing (use the MSB as the sign bit)
                        let sign_bit = if result_bitvec.is_empty() {
                            false
                        } else {
                            result_bitvec[result_bitvec.len() - 1]
                        };

                        // Resize to the exact register size with sign extension
                        result_bitvec.resize(*register_size, sign_bit);

                        debug!(
                            "Setting register {} with BitVec of length {}",
                            target,
                            result_bitvec.len()
                        );

                        self.classical_registers
                            .insert(target.clone(), result_bitvec);
                    }

                    operation_count += 1;
                }
                Operation::VoidFunctionCall { expression } => {
                    debug!("Processing void function call: {expression:?}");

                    // Evaluate the expression (which should be a function call)
                    // We use a dummy width of 1 since we'll discard the result anyway
                    let _ = self.evaluate_expression_bitvec_with_width(expression, 1)?;

                    operation_count += 1;
                }
                _ => {
                    debug!("Skipping unsupported operation type");
                }
            }
            self.current_op += 1;
        }

        Ok(Some(self.message_builder.build()))
    }

    /// Evaluate an expression with `BitVec` support
    fn evaluate_expression_bitvec(&self, expr: &Expression) -> Result<ExpressionValue, PecosError> {
        // For non-assignment contexts (like conditionals), let operands determine width
        // by using 0 as the minimum width hint
        evaluate_expression_bitvec(expr, self, 0)
    }

    fn evaluate_expression_bitvec_with_width(
        &mut self,
        expr: &Expression,
        target_width: usize,
    ) -> Result<ExpressionValue, PecosError> {
        // Check if this is a WASM function call
        #[cfg(feature = "wasm")]
        if let Expression::FunctionCall { name, args } = expr
            && let Some(ref _foreign_obj) = self.foreign_object
        {
            // Check if it's not a built-in function
            if !crate::BUILTIN_FUNCTIONS.contains(&name.as_str()) {
                // Evaluate arguments first (while we still have access to self)
                let mut arg_values = Vec::new();
                for arg in args {
                    let val = evaluate_expression_bitvec(arg, self, target_width)?;
                    arg_values.push(val.as_i64());
                }

                // Now call the WASM function with evaluated arguments
                if let Some(ref mut foreign_obj) = self.foreign_object {
                    let results = foreign_obj.exec(name, &arg_values)?;

                    // Convert result back to BitVec
                    if results.is_empty() {
                        // Void function - return 0
                        return Ok(ExpressionValue::BitVec(BitVec::repeat(false, target_width)));
                    } else if results.len() == 1 {
                        // Single return value - convert to BitVec
                        let value = results[0];
                        let mut bitvec = BitVec::<u8, Lsb0>::with_capacity(target_width);
                        for i in 0..target_width {
                            bitvec.push((value >> i) & 1 != 0);
                        }
                        return Ok(ExpressionValue::BitVec(bitvec));
                    }
                    return Err(PecosError::ParseInvalidExpression(format!(
                        "WASM function '{name}' returned {} values, but only single return values are supported in QASM expressions",
                        results.len()
                    )));
                }
            }
        }

        // Use target width as hint for expression evaluation
        evaluate_expression_bitvec(expr, self, target_width)
    }
}

impl ClassicalEngine for QASMEngine {
    fn num_qubits(&self) -> usize {
        if let Some(qasm_program) = &self.program {
            qasm_program.num_qubits()
        } else {
            0
        }
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        debug!("QASMEngine::generate_commands() called");

        // Check if we have a program and if we've reached the end
        let has_more_ops = match &self.program {
            None => {
                debug!("No program loaded, returning empty message");
                return Ok(ByteMessage::create_empty());
            }
            Some(qasm_program) => {
                let program = qasm_program.program();
                debug!(
                    "Current operation: {}/{}",
                    self.current_op,
                    program.operations.len()
                );

                if self.current_op >= program.operations.len() {
                    debug!("End of program detected, returning empty message");
                    return Ok(ByteMessage::create_empty());
                }
                true
            }
        };

        // Initialize if at the beginning of a shot
        if has_more_ops && self.current_op == 0 {
            debug!("Starting a new shot (current_op=0)");
            self.message_builder.reset();
            let _ = self.message_builder.for_quantum_operations();
        }

        debug!("Processing program from operation {}", self.current_op);

        // Process program and map the Option<ByteMessage> to ByteMessage
        let result = self
            .process_program_impl()
            .map(|maybe_message| maybe_message.unwrap_or_else(ByteMessage::create_empty))
            .map_err(|e| {
                PecosError::Processing(format!("QASM engine failed to process program: {e}"))
            });

        debug!("Program processing complete");
        result
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        debug!("Handling measurements from ByteMessage");

        match message.outcomes() {
            Ok(outcomes) => {
                let mappings = self.register_result_mappings.clone();

                debug!("Processing {} measurement results", outcomes.len());
                debug!(
                    "Starting from global measurement index {}",
                    self.measurements_processed
                );

                let num_results = outcomes.len();
                for (local_index, value) in outcomes.into_iter().enumerate() {
                    // Calculate the global index for this measurement
                    let global_index = self.measurements_processed + local_index;
                    debug!(
                        "Found measurement local_index={local_index} global_index={global_index} value={value}"
                    );

                    if let Some((register, bit)) = mappings.get(global_index) {
                        debug!("Updating register {register}[{bit}] with value {value}");

                        let safe_value = u8::try_from(value).unwrap_or(1);
                        self.update_register_bit(register, *bit, safe_value)?;
                    } else {
                        debug!(
                            "No register mapping found for measurement global_index={global_index}"
                        );
                    }

                    self.raw_measurements
                        .insert(u32::try_from(global_index).unwrap_or_default(), value);
                }

                // Update the count of measurements processed
                self.measurements_processed += num_results;

                Ok(())
            }
            Err(e) => {
                debug!("Error parsing measurement results: {e:?}");
                Err(PecosError::Input(format!(
                    "Error parsing measurement results: {e}"
                )))
            }
        }
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        let mut result = Shot::default();

        let mut reg_names: Vec<_> = self.classical_registers.keys().collect();
        reg_names.sort();

        for reg_name in &reg_names {
            if let Some(bitvec) = self.classical_registers.get(*reg_name) {
                // Clone the BitVec directly - it already has the correct width
                let reg_name_str = (*reg_name).to_string();
                result
                    .data
                    .insert(reg_name_str, Data::BitVec(bitvec.clone()));
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

        // Clone foreign object if present
        #[cfg(feature = "wasm")]
        if let Some(ref foreign_obj) = self.foreign_object {
            engine.foreign_object = Some(foreign_obj.clone_box());
        }

        // Re-initialize classical registers from program
        if let Some(qasm_program) = &engine.program {
            let program = qasm_program.program();
            for (reg_name, size) in &program.classical_registers {
                let bitvec = BitVec::<u8, Lsb0>::repeat(false, *size);
                engine.classical_registers.insert(reg_name.clone(), bitvec);
            }
        }

        engine
    }
}

impl ControlEngine for QASMEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!("QASMEngine::start() called");

        debug!("Preparing engine for new shot");
        self.reset_state();
        self.current_op = 0;

        debug!("Generating initial commands for simulation");
        if let Some(commands) = self.process_program_impl()? {
            debug!("Commands generated, returning NeedsProcessing");
            Ok(EngineStage::NeedsProcessing(commands))
        } else {
            debug!("No commands to process, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!("QASMEngine::continue_processing() called");

        let measurement_count = measurements
            .outcomes()
            .map(|outcomes| outcomes.len())
            .unwrap_or(0);
        debug!("Received {measurement_count} measurements");

        debug!("Processing measurement results");
        self.handle_measurements(measurements)?;

        debug!("Generating next batch of commands");
        if let Some(commands) = self.process_program_impl()? {
            debug!("Additional commands generated, returning NeedsProcessing");
            Ok(EngineStage::NeedsProcessing(commands))
        } else {
            debug!("No more commands, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        <Self as ClassicalEngine>::reset(self)
    }
}

impl Engine for QASMEngine {
    type Input = ();
    type Output = Shot;

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
            EngineStage::NeedsProcessing(_cmds) => {
                debug!("QASMEngine cannot process quantum operations directly");
                debug!("Returning best-effort results");
                Ok(self.get_results()?)
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
            classical_registers: BTreeMap::new(),
            raw_measurements: BTreeMap::new(),
            current_op: 0,
            measurements_processed: 0,
            message_builder: ByteMessageBuilder::new(),
            allow_complex_conditionals: false,
            #[cfg(feature = "wasm")]
            foreign_object: None,
        }
    }
}

impl FromStr for QASMEngine {
    type Err = PecosError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Import here to avoid circular dependency
        use crate::program::QASMProgram;

        // Parse the program
        let program = QASMProgram::from_str(s)?;

        // Convert to engine
        Ok(program.into_engine())
    }
}

impl BitVecExpressionContext for QASMEngine {
    fn get_register(&self, name: &str) -> Option<&BitVec<u8, Lsb0>> {
        self.classical_registers.get(name)
    }

    fn get_register_size(&self, name: &str) -> Option<usize> {
        self.classical_register_sizes()
            .and_then(|sizes| sizes.get(name))
            .copied()
    }
}

impl fmt::Debug for QASMEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("QASMEngine");
        s.field("program", &self.program)
            .field("register_result_mappings", &self.register_result_mappings)
            .field("classical_registers", &self.classical_registers)
            .field("raw_measurements", &self.raw_measurements)
            .field("current_op", &self.current_op)
            .field("measurements_processed", &self.measurements_processed)
            .field(
                "allow_complex_conditionals",
                &self.allow_complex_conditionals,
            );

        #[cfg(feature = "wasm")]
        s.field("foreign_object", &self.foreign_object.is_some());

        s.finish_non_exhaustive()
    }
}
