use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::{ByteMessage, ClassicalEngine, ControlEngine, Engine, EngineStage, ShotResult};
use std::any::Any;
use std::collections::HashMap;

use crate::parser::{Expression, Operation, Program, QASMParser};

/// Configuration flags for the QASMEngine
#[derive(Debug, Clone)]
pub struct QASMEngineConfig {
    /// When true, allows general expressions in if statements (not just register/bit compared to integer)
    pub allow_complex_conditionals: bool,
}

impl Default for QASMEngineConfig {
    fn default() -> Self {
        Self {
            // Default to OpenQASM 2.0 spec behavior
            allow_complex_conditionals: false,
        }
    }
}

/// A QASM Engine that can generate native commands from a QASM program
#[derive(Debug)]
pub struct QASMEngine {
    /// The QASM Program being executed
    program: Option<Program>,

    /// Mapping of logical qubits to physical qubit IDs
    qubit_mapping: HashMap<(String, usize), usize>,

    /// Mapping from result IDs to register names and bit indices
    register_result_mappings: Vec<(u32, String, usize)>,

    /// Classical register values
    classical_registers: HashMap<String, Vec<u32>>,

    /// Raw measurement results (may include bits not in classical registers)
    raw_measurements: HashMap<u32, u32>,

    /// Next available result ID to use for measurements
    next_result_id: u32,

    /// Next available physical qubit ID
    next_qubit_id: usize,

    /// Current operation index in the program
    current_op: usize,

    /// Reusable message builder for generating commands
    message_builder: ByteMessageBuilder,

    /// Configuration flags for the engine
    config: QASMEngineConfig,
}

impl QASMEngine {
    /// Create a new QASM Engine
    pub fn new() -> Result<Self, PecosError> {
        debug!("Creating new QASMEngine");

        Ok(Self {
            program: None,
            qubit_mapping: HashMap::new(),
            classical_registers: HashMap::new(),
            register_result_mappings: Vec::new(),
            next_result_id: 0,
            next_qubit_id: 0,
            raw_measurements: HashMap::new(),
            current_op: 0,
            message_builder: ByteMessageBuilder::new(),
            config: QASMEngineConfig::default(),
        })
    }

    /// Create a new `QASMEngine` and load a QASM program from a file
    pub fn with_file(qasm_path: impl AsRef<std::path::Path>) -> Result<Self, PecosError> {
        // Create a new engine
        let mut engine = Self::new()?;

        // Parse the QASM file
        let qasm = std::fs::read_to_string(qasm_path)
            .map_err(|e| PecosError::Resource(format!("Failed to read QASM file: {e}")))?;

        // Parse and load the program
        engine.from_str(&qasm)?;

        // Log information about the loaded program
        if let Some(program) = &engine.program {
            let total_qubits: usize = program.quantum_registers.values().sum();
            debug!(
                "Loaded QASM with {} qubits across {} registers",
                total_qubits,
                program.quantum_registers.len()
            );
        }

        Ok(engine)
    }

    /// Load a QASM program into the engine
    pub fn load_program(&mut self, program: Program) -> Result<(), PecosError> {
        debug!(
            "Loading QASM program with {} quantum registers and {} operations",
            program.quantum_registers.len(),
            program.operations.len()
        );

        // Count total number of qubits from all quantum registers
        let total_qubits: usize = program.quantum_registers.values().sum();
        debug!("Total qubits from quantum registers: {}", total_qubits);

        // Initialize simulation components
        self.qubit_mapping.clear();
        self.classical_registers.clear();
        self.raw_measurements.clear();
        self.register_result_mappings.clear();
        self.next_result_id = 0;
        self.next_qubit_id = 0;

        self.program = Some(program);

        Ok(())
    }

    /// Parse a QASM program from a string and load it
    pub fn from_str(&mut self, qasm: &str) -> Result<(), PecosError> {
        let program = QASMParser::parse_str(qasm)
            .map_err(|e| PecosError::Input(format!("Failed to parse QASM: {e:?}")))?;

        self.load_program(program)
    }

    /// Enable or disable complex conditionals (general expressions in if statements)
    pub fn set_allow_complex_conditionals(&mut self, allow: bool) {
        self.config.allow_complex_conditionals = allow;
    }

    /// Get the current setting for complex conditionals
    pub fn allow_complex_conditionals(&self) -> bool {
        self.config.allow_complex_conditionals
    }

    /// Reset the engine's internal state - ensure full reset for each shot
    /// This is the single source of truth for all reset operations
    fn reset_state(&mut self) {
        debug!("QASMEngine::reset_state()");

        // PHASE 1: Reset counters and operational state
        debug!("Resetting operational state (current_op, result_id, qubit_id)");
        self.current_op = 0;
        self.next_result_id = 0;
        self.next_qubit_id = 0;

        // PHASE 2: Clear all collections
        debug!("Clearing all collections (measurements, mappings, registers)");
        self.raw_measurements.clear();
        self.register_result_mappings.clear();
        self.qubit_mapping.clear();
        self.classical_registers.clear();
        self.message_builder.reset();

        // PHASE 3: Re-initialize from program if available
        if let Some(program) = &self.program {
            debug!(
                "Initializing {} quantum and {} classical registers from program",
                program.quantum_registers.len(),
                program.classical_registers.len()
            );

            // Pre-allocate quantum registers
            for (reg_name, size) in &program.quantum_registers {
                for i in 0..*size {
                    let physical_id = self.next_qubit_id;
                    self.qubit_mapping
                        .insert((reg_name.clone(), i), physical_id);
                    self.next_qubit_id += 1;
                }
            }

            // Initialize classical registers to zero
            for (reg_name, size) in &program.classical_registers {
                self.classical_registers
                    .insert(reg_name.clone(), vec![0; *size]);
            }

            debug!(
                "Reset complete. Engine ready with {} qubits and {} classical registers",
                self.next_qubit_id,
                self.classical_registers.len()
            );
        } else {
            debug!("Reset complete. No program loaded.");
        }
    }

    /// Create a clone of this engine with the same program but fresh state
    #[must_use]
    pub fn clone_with_fresh_state(&self) -> Self {
        let program = self.program.clone();

        Self {
            program,
            qubit_mapping: HashMap::new(),
            classical_registers: HashMap::new(),
            register_result_mappings: Vec::new(),
            next_result_id: 0,
            next_qubit_id: 0,
            raw_measurements: HashMap::new(),
            current_op: 0,
            message_builder: ByteMessageBuilder::new(),
            config: self.config.clone(),
        }
    }

    /// Helper to get a physical qubit ID for a logical qubit
    ///
    /// If the qubit mapping doesn't exist, it will be created
    fn get_physical_qubit(&mut self, index: usize, register_name: &str) -> Result<usize, PecosError> {
        // Validate bounds if we have a program loaded
        if let Some(program) = &self.program {
            if let Some(size) = program.quantum_registers.get(register_name) {
                if index >= *size {
                    return Err(PecosError::Input(format!(
                        "Qubit index {} out of bounds for register '{}' of size {}",
                        index, register_name, size
                    )));
                }
            } else {
                return Err(PecosError::Input(format!(
                    "Quantum register '{}' not found",
                    register_name
                )));
            }
        }

        let key = (register_name.to_string(), index);

        // If we already have a mapping, return it
        if let Some(&physical_id) = self.qubit_mapping.get(&key) {
            return Ok(physical_id);
        }

        // Create a new mapping
        let physical_id = self.next_qubit_id;
        self.qubit_mapping.insert(key, physical_id);
        self.next_qubit_id += 1;

        Ok(physical_id)
    }

    fn update_register_bit(&mut self, register_name: &str, bit_index: usize, value: u8) -> Result<(), PecosError> {
        // Validate bounds if we have a program loaded
        if let Some(program) = &self.program {
            if let Some(size) = program.classical_registers.get(register_name) {
                if bit_index >= *size {
                    return Err(PecosError::Input(format!(
                        "Classical register bit index {} out of bounds for register '{}' of size {}",
                        bit_index, register_name, size
                    )));
                }
            } else {
                return Err(PecosError::Input(format!(
                    "Classical register '{}' not found",
                    register_name
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

    /// Helper function to apply S gate
    fn apply_s(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
        engine.message_builder.add_rz(std::f64::consts::PI / 2.0, &[qubit]);
        Ok(())
    }

    /// Helper function to apply S-dagger gate
    fn apply_sdg(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
        engine.message_builder.add_rz(-std::f64::consts::PI / 2.0, &[qubit]);
        Ok(())
    }

    /// Helper function to apply T gate
    fn apply_t(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
        engine.message_builder.add_rz(std::f64::consts::PI / 4.0, &[qubit]);
        Ok(())
    }

    /// Helper function to apply T-dagger gate
    fn apply_tdg(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
        engine.message_builder.add_rz(-std::f64::consts::PI / 4.0, &[qubit]);
        Ok(())
    }

    /// Helper function to apply CZ gate
    fn apply_cz(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let control = engine.get_physical_qubit(args[0], &regs[0])?;
        let target = engine.get_physical_qubit(args[1], &regs[1])?;

        // CZ = H · CX · H
        engine.message_builder.add_h(&[target]);
        engine.message_builder.add_cx(&[control], &[target]);
        engine.message_builder.add_h(&[target]);
        Ok(())
    }

    /// Helper function to apply CY gate
    fn apply_cy(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let control = engine.get_physical_qubit(args[0], &regs[0])?;
        let target = engine.get_physical_qubit(args[1], &regs[1])?;

        // CY = S† · CX · S
        engine.message_builder.add_rz(-std::f64::consts::PI / 2.0, &[target]); // S†
        engine.message_builder.add_cx(&[control], &[target]);
        engine.message_builder.add_rz(std::f64::consts::PI / 2.0, &[target]);  // S
        Ok(())
    }

    /// Helper function to apply SWAP gate
    fn apply_swap(engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]) -> Result<(), PecosError> {
        let qubit1 = engine.get_physical_qubit(args[0], &regs[0])?;
        let qubit2 = engine.get_physical_qubit(args[1], &regs[1])?;

        // SWAP = CX · CX · CX
        engine.message_builder.add_cx(&[qubit1], &[qubit2]);
        engine.message_builder.add_cx(&[qubit2], &[qubit1]);
        engine.message_builder.add_cx(&[qubit1], &[qubit2]);
        Ok(())
    }

    /// Process a single gate operation using a table-driven approach
    #[allow(clippy::similar_names)]
    fn process_gate_operation(
        &mut self,
        name: &str,
        arguments: &[usize],
        registers: &[String],
        parameters: &[f64],
    ) -> Result<bool, PecosError> {
        // Define gate requirements and handlers using a more structured approach
        // Each entry contains: (required_args, handler_fn)
        struct GateHandler {
            required_args: usize,
            name: &'static str, // For error messages
            apply: fn(&mut QASMEngine, &[usize], &[String], &[f64]) -> Result<(), PecosError>,
        }

        // Single-qubit gate handlers - now return Result
        let apply_h = |engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]| -> Result<(), PecosError> {
            let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
            debug!("Adding H gate on qubit {} (register {})", qubit, regs[0]);
            engine.message_builder.add_h(&[qubit]);
            Ok(())
        };

        let apply_x = |engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]| -> Result<(), PecosError> {
            let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
            debug!("Adding X gate on qubit {} (register {})", qubit, regs[0]);
            engine.message_builder.add_x(&[qubit]);
            Ok(())
        };

        let apply_y = |engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]| -> Result<(), PecosError> {
            let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
            debug!("Adding Y gate on qubit {} (register {})", qubit, regs[0]);
            engine.message_builder.add_y(&[qubit]);
            Ok(())
        };

        let apply_z = |engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]| -> Result<(), PecosError> {
            let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
            debug!("Adding Z gate on qubit {} (register {})", qubit, regs[0]);
            engine.message_builder.add_z(&[qubit]);
            Ok(())
        };

        // RZ rotation gate handler
        let apply_rz = |engine: &mut QASMEngine, args: &[usize], regs: &[String], params: &[f64]| -> Result<(), PecosError> {
            if params.is_empty() {
                return Err(PecosError::Input("RZ gate requires theta parameter".to_string()));
            }
            let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
            debug!("Adding RZ({}) gate on qubit {} (register {})", params[0], qubit, regs[0]);
            engine.message_builder.add_rz(params[0], &[qubit]);
            Ok(())
        };

        // R1XY rotation gate handler
        let apply_r1xy = |engine: &mut QASMEngine, args: &[usize], regs: &[String], params: &[f64]| -> Result<(), PecosError> {
            if params.len() < 2 {
                return Err(PecosError::Input("R1XY gate requires theta and phi parameters".to_string()));
            }
            let qubit = engine.get_physical_qubit(args[0], &regs[0])?;
            debug!("Adding R1XY({}, {}) gate on qubit {} (register {})", params[0], params[1], qubit, regs[0]);
            engine.message_builder.add_r1xy(params[0], params[1], &[qubit]);
            Ok(())
        };

        // Two-qubit gate handlers
        let apply_cx = |engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]| -> Result<(), PecosError> {
            let control = engine.get_physical_qubit(args[0], &regs[0])?;
            let target = engine.get_physical_qubit(args[1], &regs[1])?;
            debug!(
                "Adding CX gate from control {} (register {}) to target {} (register {})",
                control, regs[0], target, regs[1]
            );
            engine.message_builder.add_cx(&[control], &[target]);
            Ok(())
        };

        // ZZ rotation gate handler
        let apply_rzz = |engine: &mut QASMEngine, args: &[usize], regs: &[String], params: &[f64]| -> Result<(), PecosError> {
            if params.is_empty() {
                return Err(PecosError::Input("RZZ gate requires theta parameter".to_string()));
            }
            let qubit1 = engine.get_physical_qubit(args[0], &regs[0])?;
            let qubit2 = engine.get_physical_qubit(args[1], &regs[1])?;
            debug!("Adding RZZ({}) gate on qubits {} and {}", params[0], qubit1, qubit2);
            engine.message_builder.add_rzz(params[0], &[qubit1], &[qubit2]);
            Ok(())
        };

        // Strong ZZ gate handler
        let apply_szz = |engine: &mut QASMEngine, args: &[usize], regs: &[String], _params: &[f64]| -> Result<(), PecosError> {
            let qubit1 = engine.get_physical_qubit(args[0], &regs[0])?;
            let qubit2 = engine.get_physical_qubit(args[1], &regs[1])?;
            debug!("Adding SZZ gate on qubits {} and {}", qubit1, qubit2);
            engine.message_builder.add_szz(&[qubit1], &[qubit2]);
            Ok(())
        };

        // Gate definition table - maps gate names to their handlers
        let gates: &[(&str, GateHandler)] = &[
            (
                "h",
                GateHandler {
                    required_args: 1,
                    name: "H",
                    apply: apply_h,
                },
            ),
            (
                "x",
                GateHandler {
                    required_args: 1,
                    name: "X",
                    apply: apply_x,
                },
            ),
            (
                "y",
                GateHandler {
                    required_args: 1,
                    name: "Y",
                    apply: apply_y,
                },
            ),
            (
                "z",
                GateHandler {
                    required_args: 1,
                    name: "Z",
                    apply: apply_z,
                },
            ),
            (
                "rz",
                GateHandler {
                    required_args: 1,
                    name: "RZ",
                    apply: apply_rz,
                },
            ),
            (
                "r1xy",
                GateHandler {
                    required_args: 1,
                    name: "R1XY",
                    apply: apply_r1xy,
                },
            ),
            (
                "cx",
                GateHandler {
                    required_args: 2,
                    name: "CX",
                    apply: apply_cx,
                },
            ),
            (
                "rzz",
                GateHandler {
                    required_args: 2,
                    name: "RZZ",
                    apply: apply_rzz,
                },
            ),
            (
                "szz",
                GateHandler {
                    required_args: 2,
                    name: "SZZ",
                    apply: apply_szz,
                },
            ),
            (
                "s",
                GateHandler {
                    required_args: 1,
                    name: "S",
                    apply: Self::apply_s,
                },
            ),
            (
                "sdg",
                GateHandler {
                    required_args: 1,
                    name: "SDG",
                    apply: Self::apply_sdg,
                },
            ),
            (
                "t",
                GateHandler {
                    required_args: 1,
                    name: "T",
                    apply: Self::apply_t,
                },
            ),
            (
                "tdg",
                GateHandler {
                    required_args: 1,
                    name: "TDG",
                    apply: Self::apply_tdg,
                },
            ),
            (
                "cz",
                GateHandler {
                    required_args: 2,
                    name: "CZ",
                    apply: Self::apply_cz,
                },
            ),
            (
                "cy",
                GateHandler {
                    required_args: 2,
                    name: "CY",
                    apply: Self::apply_cy,
                },
            ),
            (
                "swap",
                GateHandler {
                    required_args: 2,
                    name: "SWAP",
                    apply: Self::apply_swap,
                },
            ),
        ];

        // Find the gate handler (case-insensitive)
        let name_lower = name.to_lowercase();
        if let Some((_, handler)) = gates.iter().find(|(gate_name, _)| *gate_name == name_lower) {
            // Validate argument count
            if arguments.len() != handler.required_args {
                return Err(PecosError::Input(format!(
                    "{} gate requires {} qubit{}, got {}",
                    handler.name,
                    handler.required_args,
                    if handler.required_args == 1 { "" } else { "s" },
                    arguments.len()
                )));
            }

            // Apply the gate
            (handler.apply)(self, arguments, registers, parameters)?;
            Ok(true)
        } else {
            // Gate not supported
            Err(PecosError::Gate(format!("Unsupported gate: {name}")))
        }
    }

    /// Process a measurement operation
    fn process_measurement(&mut self, qubit: usize, q_reg: &str, bit: usize, c_reg: &str) -> Result<(), PecosError> {
        // Map the qubit
        let register_name = if q_reg.is_empty() { "q" } else { q_reg };
        let physical_qubit = self.get_physical_qubit(qubit, register_name)?;

        // Get the classical register name
        let c_register_name = if c_reg.is_empty() { "c" } else { c_reg };

        // Validate classical register bounds
        if let Some(program) = &self.program {
            if let Some(size) = program.classical_registers.get(c_register_name) {
                if bit >= *size {
                    return Err(PecosError::Input(format!(
                        "Classical register bit index {} out of bounds for register '{}' of size {}",
                        bit, c_register_name, size
                    )));
                }
            } else {
                return Err(PecosError::Input(format!(
                    "Classical register '{}' not found",
                    c_register_name
                )));
            }
        }

        // Create a unique result ID
        let result_id = self.next_result_id;
        self.next_result_id += 1;

        // Store the mapping for result handling
        self.register_result_mappings
            .push((result_id, c_register_name.to_string(), bit));

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

    /// Initialize registers if needed
    fn ensure_registers_initialized(&mut self, program: &Program) {
        if self.qubit_mapping.is_empty() {
            // Initialize quantum registers
            for (reg_name, size) in &program.quantum_registers {
                debug!(
                    "Setting up qubit mapping for register {} with size {}",
                    reg_name, size
                );
                for i in 0..*size {
                    let physical_id = self.next_qubit_id;
                    self.qubit_mapping
                        .insert((reg_name.clone(), i), physical_id);
                    self.next_qubit_id += 1;
                }
            }

            // Initialize classical registers
            for (reg_name, size) in &program.classical_registers {
                debug!(
                    "Setting up classical register {} with size {}",
                    reg_name, size
                );
                self.classical_registers
                    .insert(reg_name.clone(), vec![0; *size]);
            }
        }
    }

    /// Process a register measurement operation (measure `q_reg` -> `c_reg`)
    ///
    /// Returns:
    /// - Some(count) if measurements were added and processing should continue
    /// - None if we hit the batch size limit and need to return the current batch
    fn process_register_measurement(
        &mut self,
        q_reg: &str,
        c_reg: &str,
        program: &Program,
        current_operation_count: usize,
    ) -> Result<Option<usize>, PecosError> {
        // Get the sizes of both registers
        let Some(&q_size) = program.quantum_registers.get(q_reg) else {
            return Err(PecosError::Input(format!(
                "Quantum register {q_reg} not found"
            )));
        };

        let Some(&c_size) = program.classical_registers.get(c_reg) else {
            return Err(PecosError::Input(format!(
                "Classical register {c_reg} not found"
            )));
        };

        // We should measure min(q_size, c_size) qubits
        let measure_count = std::cmp::min(q_size, c_size);

        debug!(
            "Will measure {} qubits from {} to {}",
            measure_count, q_reg, c_reg
        );

        // Create individual measurements for each qubit
        let mut measurements_added = 0;
        for i in 0..measure_count {
            // Check if adding this measurement would exceed batch size
            if current_operation_count + measurements_added >= Self::MAX_BATCH_SIZE {
                debug!(
                    "Reached maximum batch size during register measurement, will continue in next batch"
                );
                break;
            }

            // Use the helper function for individual measurements
            self.process_measurement(i, q_reg, i, c_reg)?;
            measurements_added += 1;
        }

        // If we couldn't add all measurements, don't increment current_op yet
        if measurements_added < measure_count {
            // We'll continue from where we left off on the next batch
            debug!(
                "Only processed {} of {} measurements in RegMeasure, will continue in next batch",
                measurements_added, measure_count
            );
            // Return None to signal that we need to return the current batch
            return Ok(None);
        }

        // Return the number of measurements added
        Ok(Some(measurements_added))
    }

    /// Process the QASM program and generate `ByteMessage` with operations up to `MAX_BATCH_SIZE`
    // Maximum batch size for quantum operations
    // This helps avoid creating excessively large messages
    const MAX_BATCH_SIZE: usize = 100;

    fn process_program(&mut self) -> Result<ByteMessage, PecosError> {
        // CRITICAL: Reset and configure the reusable message builder for quantum operations
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();

        // Ensure we have a program loaded
        let program = self
            .program
            .as_ref()
            .ok_or_else(|| PecosError::Input("No QASM program loaded".to_string()))?
            .clone();

        // Get total operations count for the loaded program
        let total_ops = program.operations.len();

        debug!(
            "Processing program: current_op: {}/{}, qubits: {}",
            self.current_op, total_ops, self.next_qubit_id
        );

        // Check for program completion
        if self.current_op >= total_ops {
            debug!("End of program reached, sending flush");
            return Ok(ByteMessage::create_flush());
        }

        // Ensure registers are properly initialized
        self.ensure_registers_initialized(&program);

        // Process operations up to MAX_BATCH_SIZE or until we reach the end
        let mut operation_count = 0;

        while self.current_op < total_ops && operation_count < Self::MAX_BATCH_SIZE {
            let op = &program.operations[self.current_op];

            match op {
                Operation::Gate {
                    name,
                    parameters,
                    arguments,
                    registers,
                } => {
                    // Use the helper function to process gate operations
                    if self.process_gate_operation(name, arguments, registers, parameters)? {
                        operation_count += 1;
                    }
                }
                Operation::Measure {
                    qubit,
                    q_reg,
                    bit,
                    c_reg,
                } => {
                    // Use the helper function to process measurement operations
                    self.process_measurement(*qubit, q_reg, *bit, c_reg)?;

                    // After a measurement, we need to break the batch to wait for results
                    // before processing any subsequent operations that might depend on them
                    self.current_op += 1;
                    debug!("Breaking batch after measurement to wait for results");
                    return Ok(self.message_builder.build());
                }
                Operation::RegMeasure { q_reg, c_reg } => {
                    let added_count =
                        self.process_register_measurement(q_reg, c_reg, &program, operation_count)?;

                    // If we returned a value, it means we added some measurements
                    if let Some(count) = added_count {
                        operation_count += count;
                    } else {
                        // Need to stop processing and return the current batch
                        return Ok(self.message_builder.build());
                    }
                }
                Operation::If {
                    condition,
                    operation,
                } => {
                    // Check if the condition is allowed based on config
                    if !self.config.allow_complex_conditionals {
                        // Validate that the condition is a simple comparison
                        if let Expression::BinaryOp(left, _op, right) = condition {
                            // Check that left is a register/bit and right is a constant
                            let is_valid = match (left.as_ref(), right.as_ref()) {
                                (Expression::Variable(_), Expression::Integer(_)) => true,
                                (Expression::BitId(_, _), Expression::Integer(_)) => true,
                                _ => false,
                            };

                            if !is_valid {
                                return Err(PecosError::Processing(
                                    "Complex conditionals are not allowed. Only register/bit compared to integer is supported in standard OpenQASM 2.0. Enable allow_complex_conditionals to use general expressions.".to_string()
                                ));
                            }
                        } else {
                            return Err(PecosError::Processing(
                                "Invalid conditional format. Expected comparison expression.".to_string()
                            ));
                        }
                    }

                    // Evaluate the condition - this should return 1 for true, 0 for false
                    debug!("Evaluating if condition: {:?}", condition);
                    let condition_value = self.evaluate_expression_with_context(&condition)?;
                    debug!("Condition value: {}", condition_value);

                    if condition_value != 0 {
                        debug!("If condition evaluated to true, executing operation: {:?}", operation);

                        // Execute the conditional operation
                        match operation.as_ref() {
                            Operation::Gate { name, parameters, arguments, registers } => {
                                // Process the gate operation
                                debug!("Executing conditional gate {} on arguments {:?} with registers {:?}", name, arguments, registers);
                                // Delegate to the standard gate processing
                                if self.process_gate_operation(name, arguments, registers, parameters)? {
                                    operation_count += 1;
                                }
                            }
                            Operation::ClassicalAssignment {
                                target,
                                is_indexed,
                                index,
                                expression,
                            } => {
                                // Evaluate the expression and set the register value
                                let value = self.evaluate_expression_with_context(&expression)?;

                                if *is_indexed {
                                    // Set a specific bit
                                    if let Some(idx) = *index {
                                        self.update_register_bit(&target, idx, if value != 0 { 1 } else { 0 })?;
                                    }
                                } else {
                                    // Set the entire register
                                    if let Some(register_size) = program.classical_registers.get(target.as_str()) {
                                        // Create a zero-filled register of the appropriate size
                                        let mut bits = vec![0u32; *register_size];

                                        // Set bits according to value - treat 'value' as the integer value of the register
                                        // For a register of size n, we store the value using an n-bit representation
                                        for i in 0..*register_size {
                                            if i < 32 { // Only handle up to 32 bits
                                                bits[i] = ((value >> i) & 1) as u32;
                                            }
                                        }

                                        debug!("Setting register {} to value {} (bits: {:?})", target, value, bits);

                                        // Update the register
                                        self.classical_registers.insert(target.clone(), bits);
                                    }
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
                    // Handle classical assignment
                    debug!("Processing classical assignment: {} = {:?}", target, expression);

                    // Evaluate the expression using the full evaluator with register context
                    let value = self.evaluate_expression_with_context(&expression)?;

                    if *is_indexed {
                        // Set a specific bit
                        if let Some(idx) = *index {
                            self.update_register_bit(&target, idx, if value != 0 { 1 } else { 0 })?;
                        }
                    } else {
                        // Set the entire register
                        if let Some(register_size) = program.classical_registers.get(target.as_str()) {
                            // Create a zero-filled register of the appropriate size
                            let mut bits = vec![0u32; *register_size];

                            // Set bits according to value - treat 'value' as the integer value of the register
                            // For a register of size n, we store the value using an n-bit representation
                            for i in 0..*register_size {
                                if i < 32 { // Only handle up to 32 bits
                                    bits[i] = ((value >> i) & 1) as u32;
                                }
                            }

                            debug!("Setting register {} to value {} (bits: {:?})", target, value, bits);

                            // Update the register
                            self.classical_registers.insert(target.clone(), bits);
                        }
                    }

                    operation_count += 1;
                }
                _ => {
                    debug!("Skipping unsupported operation type");
                }
            }
            self.current_op += 1;
        }

        // Build and return the message
        Ok(self.message_builder.build())
    }

    /// Create a new `QASMEngine` with a specific random seed and load a QASM file
    ///
    /// Note: `QASMEngine` itself does not use randomness. The seed is passed through
    /// to the underlying quantum simulation layer when the commands are executed.
    pub fn with_seed(
        qasm_path: impl AsRef<std::path::Path>,
        seed: u64,
    ) -> Result<Self, PecosError> {
        debug!(
            "Creating QASMEngine with seed {} (for passthrough to quantum simulator)",
            seed
        );

        // Create a new engine and load the QASM file
        let engine = Self::with_file(qasm_path)?;

        // QASMEngine does not use randomness directly.
        // The seed will be used by the quantum simulation layer that processes the commands.
        debug!("Seed {} will be used by the quantum simulation layer", seed);

        Ok(engine)
    }

    /// Evaluate an expression with access to register values
    fn evaluate_expression_with_context(&self, expr: &Expression) -> Result<i64, PecosError> {
        match expr {
            Expression::Integer(i) => Ok(*i as i64),
            Expression::Float(f) => Ok(*f as i64),
            Expression::Variable(name) => {
                // Get the register value
                if let Some(bits) = self.classical_registers.get(name) {
                    // Convert bits to integer value
                    let mut value = 0i64;
                    for (i, &bit) in bits.iter().enumerate() {
                        if i < 32 { // Only handle up to 32 bits
                            value |= ((bit & 1) as i64) << i;
                        }
                    }
                    Ok(value)
                } else {
                    debug!("Register {} not found", name);
                    Ok(0)
                }
            }
            Expression::BitId(reg_name, idx) => {
                // Get a bit value from a classical register
                let bit_value = self.classical_registers
                    .get(reg_name)
                    .and_then(|reg| reg.get(*idx as usize))
                    .map(|&v| v as u32)
                    .unwrap_or(0);
                debug!("Evaluating bit {}.{} = {}", reg_name, idx, bit_value);
                Ok(bit_value as i64)
            }
            Expression::BinaryOp(left, op, right) => {
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
                    "==" => Ok(if left_val == right_val { 1 } else { 0 }),
                    "!=" => Ok(if left_val != right_val { 1 } else { 0 }),
                    "<" => Ok(if left_val < right_val { 1 } else { 0 }),
                    ">" => Ok(if left_val > right_val { 1 } else { 0 }),
                    "<=" => Ok(if left_val <= right_val { 1 } else { 0 }),
                    ">=" => Ok(if left_val >= right_val { 1 } else { 0 }),
                    "<<" => Ok(left_val << right_val),
                    ">>" => Ok(left_val >> right_val),
                    _ => {
                        debug!("Unsupported binary operation: {}", op);
                        Err(PecosError::Processing(format!("Unsupported operation: {}", op)))
                    }
                }
            }
            Expression::UnaryOp(op, inner) => {
                let val = self.evaluate_expression_with_context(inner)?;
                match op.as_str() {
                    "-" => Ok(-val), // Simple negation for i64
                    "~" => Ok(!val),
                    _ => {
                        debug!("Unsupported unary operation: {}", op);
                        Err(PecosError::Processing(format!("Unsupported operation: {}", op)))
                    }
                }
            }
            _ => {
                debug!("Unsupported expression type: {:?}", expr);
                Err(PecosError::Processing(format!("Unsupported expression: {:?}", expr)))
            }
        }
    }
}

impl ClassicalEngine for QASMEngine {
    fn num_qubits(&self) -> usize {
        // Return the correct number of qubits based on quantum registers
        // instead of just returning next_qubit_id which might be incorrect
        if let Some(program) = &self.program {
            // Sum up the size of all quantum registers
            program.quantum_registers.values().sum()
        } else {
            self.next_qubit_id
        }
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        debug!("QASMEngine::generate_commands() called");

        if self.program.is_none() {
            // Create an empty message - return a properly structured empty message
            debug!("No program loaded, returning empty message");
            self.message_builder.reset();
            let _ = self.message_builder.for_quantum_operations();
            return Ok(self.message_builder.build());
        }

        // CRITICAL: reset_state may not have been called between shots
        // HybridEngine calls this method directly without always going through start()
        // So we need to manually check if we need to reset state here
        if let Some(program) = &self.program {
            debug!(
                "Current operation: {}/{}",
                self.current_op,
                program.operations.len()
            );

            if self.current_op >= program.operations.len() {
                // If we're at the end of the program, signal completion by returning a flush
                debug!("End of program detected, returning flush message");
                // Instead of resetting state here, return a flush message to signal completion
                return Ok(ByteMessage::create_flush());
            }
        }

        // If it's a new shot (current_op=0), ensure we have a clean slate
        if self.current_op == 0 {
            debug!("Starting a new shot (current_op=0)");
            // Ensure builder is reset for new shot
            self.message_builder.reset();
            let _ = self.message_builder.for_quantum_operations();
        }

        // Process the program to generate commands
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
                // Get a local copy of the mappings to avoid borrowing issues
                let mappings = self.register_result_mappings.clone();

                debug!("Processing {} measurement results", results.len());

                // Process each measurement and update classical registers
                for (result_id, value) in results {
                    debug!("Found measurement result_id={} value={}", result_id, value);

                    // Find the corresponding register and bit index
                    if let Some((_, register, bit)) = mappings
                        .iter()
                        .find(|(id, _, _)| *id == u32::try_from(result_id).unwrap_or_default())
                    {
                        debug!(
                            "Updating register {}[{}] with value {}",
                            register, bit, value
                        );

                        // Update the classical register at the specified bit - safely convert to u8
                        let safe_value = u8::try_from(value).unwrap_or(1); // Default to 1 if truncation would happen
                        self.update_register_bit(register, *bit, safe_value)?;
                    } else {
                        debug!("No register mapping found for result_id={}", result_id);
                    }

                    // Store in raw_measurements for debugging and legacy compatibility - safely convert result_id
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

        // Sort register names for consistent ordering
        let mut reg_names: Vec<_> = self.classical_registers.keys().collect();
        reg_names.sort();

        // Process each register
        for reg_name in &reg_names {
            if let Some(values) = self.classical_registers.get(*reg_name) {
                // Calculate the register's decimal value for bits within u32 range
                let reg_value = values.iter().enumerate().fold(0, |acc, (i, &v)| {
                    if i >= 32 || v == 0 {
                        acc
                    } else {
                        acc | (v << i)
                    }
                });

                // Add the whole register value
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    // CRITICAL: Explicitly override ClassicalEngine::reset method
    fn reset(&mut self) -> Result<(), PecosError> {
        // All reset operations are consolidated in reset_state()
        self.reset_state();
        Ok(())
    }
}

impl Clone for QASMEngine {
    fn clone(&self) -> Self {
        // Create a new engine instance with completely fresh state
        let mut engine = Self {
            program: self.program.clone(),
            qubit_mapping: HashMap::new(),
            classical_registers: HashMap::new(),
            register_result_mappings: Vec::new(),
            next_result_id: 0,
            next_qubit_id: 0,
            raw_measurements: HashMap::new(),
            current_op: 0,
            message_builder: ByteMessageBuilder::new(),
            config: self.config.clone(),
        };

        // Pre-initialize registers if a program is loaded
        if let Some(program) = &engine.program {
            // Initialize quantum registers first
            for (reg_name, size) in &program.quantum_registers {
                for i in 0..*size {
                    let physical_id = engine.next_qubit_id;
                    engine
                        .qubit_mapping
                        .insert((reg_name.clone(), i), physical_id);
                    engine.next_qubit_id += 1;
                }
            }

            // Initialize classical registers to zero
            for (reg_name, size) in &program.classical_registers {
                engine
                    .classical_registers
                    .insert(reg_name.clone(), vec![0; *size]);
            }
        }

        engine
    }
}

// Implement ControlEngine for QASMEngine
impl ControlEngine for QASMEngine {
    type Input = ();
    type Output = ShotResult;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, ShotResult>, PecosError> {
        debug!("QASMEngine::start() called");

        // Reset internal state - this will handle all necessary state reset
        debug!("Preparing engine for new shot");
        self.reset_state();

        // CRITICAL: Explicitly reset current_op to 0
        self.current_op = 0;

        // Generate commands for the simulation
        debug!("Generating initial commands for simulation");
        let commands = self.generate_commands()?;

        // If there are no commands, return results immediately
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

        // Handle the measurement results
        debug!("Processing measurement results");
        self.handle_measurements(measurements)?;

        // Try to get the next batch of commands
        debug!("Generating next batch of commands");
        let commands = self.generate_commands()?;

        // Since QASM processing is a single batch, we should be done
        if commands.is_empty()? {
            debug!("No more commands, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            // This shouldn't happen with our implementation
            debug!("Unexpected additional commands generated");
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Delegate to ClassicalEngine implementation to maintain single source of truth
        <Self as ClassicalEngine>::reset(self)
    }
}

// Update Engine implementation to use ControlEngine methods
impl Engine for QASMEngine {
    type Input = ();
    type Output = ShotResult;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        debug!("QASMEngine::process() called");

        // Reset state via the trait-specific reset method
        <Self as ClassicalEngine>::reset(self)?;

        // Start the engine to produce commands
        debug!("Starting engine to produce commands");
        let stage = self
            .start(input)
            .map_err(|e| PecosError::Processing(format!("Failed to start QASMEngine: {e}")))?;

        // Process based on stage
        match stage {
            EngineStage::Complete(result) => {
                debug!("Shot completed directly in start()");
                // We've completed this shot
                Ok(result)
            }
            EngineStage::NeedsProcessing(cmds) => {
                debug!("Processing commands from start()");

                // Check if the commands are a flush message
                if cmds.is_empty().map_err(|e| {
                    PecosError::Processing(format!("Failed to check if commands are empty: {e}"))
                })? {
                    debug!("Received empty commands, treating as completion");
                    // If we got empty commands, we're done
                    Ok(self.get_results()?)
                } else {
                    // In this standalone implementation, we can't process quantum operations
                    // directly. In normal operation with MonteCarloEngine, these commands
                    // would be sent to the quantum simulation layer.
                    debug!("QASMEngine cannot process quantum operations directly");

                    // Return results with empty measurements
                    Ok(self.get_results()?)
                }
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Delegate to ControlEngine implementation to maintain single source of truth
        <Self as ControlEngine>::reset(self)
    }
}
