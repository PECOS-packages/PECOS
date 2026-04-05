/*!
PHIR Execution Engine

Main execution engine for PHIR programs. This implements the `ClassicalEngine` trait
and can execute PHIR modules directly, integrating with the PECOS quantum simulation
infrastructure.
*/

use super::environment::TypedValue;
use super::processor::PhirProcessor;
use crate::error::Result;
use crate::phir::Module;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::{ByteMessage, builder::ByteMessageBuilder};
use pecos_engines::engine_system::EngineStage;
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{ClassicalEngine, ControlEngine, Engine};
use std::any::Any;

/// PHIR execution engine - executes PHIR modules directly
#[derive(Debug, Clone)]
pub struct PhirEngine {
    /// The PHIR module to execute
    module: Option<Module>,
    /// Operation processor for handling PHIR operations
    pub processor: PhirProcessor,
    /// Builder for constructing `ByteMessages`
    message_builder: ByteMessageBuilder,
    /// Whether we've finished processing all operations
    pub finished: bool,
    /// Current operation index
    pub current_op: usize,
    /// Function operations to process (extracted from module)
    pub function_ops: Vec<crate::phir::Instruction>,
}

impl PhirEngine {
    /// Create a new `PhirEngine` from a PHIR module
    ///
    /// # Errors
    ///
    /// Returns an error if variable definitions cannot be extracted from the module
    pub fn new(module: Module) -> Result<Self> {
        let mut processor = PhirProcessor::new();

        // Extract variable definitions from the PHIR module during initialization
        // This follows the PhirJsonEngine pattern of processing variables upfront
        processor.extract_variable_definitions(&module)?;

        // Extract function operations
        let mut function_ops = Vec::new();

        for block in &module.body.blocks {
            for instruction in &block.operations {
                if let crate::ops::Operation::Builtin(crate::builtin_ops::BuiltinOp::Func(
                    func_op,
                )) = &instruction.operation
                {
                    if func_op.name == "main" {
                        // Extract operations from the main function
                        for region in &func_op.body {
                            for func_block in &region.blocks {
                                function_ops.extend(func_block.operations.clone());
                            }
                        }
                    }
                } else {
                    // For non-function operations, add them directly
                    function_ops.push(instruction.clone());
                }
            }
        }

        Ok(Self {
            module: Some(module),
            processor,
            message_builder: ByteMessageBuilder::new(),
            finished: false,
            current_op: 0,
            function_ops,
        })
    }

    /// Create an empty `PhirEngine` (for testing)
    #[must_use]
    pub fn empty() -> Self {
        Self {
            module: None,
            processor: PhirProcessor::new(),
            message_builder: ByteMessageBuilder::new(),
            finished: true,
            current_op: 0,
            function_ops: Vec::new(),
        }
    }

    /// Set a foreign object for WebAssembly calls
    pub fn set_foreign_object(&mut self, _foreign_object: Box<dyn Send + Sync>) {
        // TODO: Implement WebAssembly foreign object support
        // This would integrate with the existing foreign object system
    }

    /// Get the underlying PHIR module
    #[must_use]
    pub fn module(&self) -> Option<&Module> {
        self.module.as_ref()
    }
}

impl Engine for PhirEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, _input: Self::Input) -> std::result::Result<Self::Output, PecosError> {
        // For ClassicalEngine, the main processing happens in generate_commands and handle_measurements
        // This method is called at the end to get the final result
        self.get_results()
    }

    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        self.processor.reset();
        self.finished = false;
        self.current_op = 0;
        self.message_builder.reset();
        Ok(())
    }
}

impl ClassicalEngine for PhirEngine {
    fn num_qubits(&self) -> usize {
        self.processor.get_qubit_count()
    }

    fn generate_commands(&mut self) -> std::result::Result<ByteMessage, PecosError> {
        const MAX_BATCH_SIZE: usize = 100;

        if self.finished {
            // No more commands to generate - return empty message
            return Ok(ByteMessage::create_empty());
        }

        // Check if we've processed all operations
        if self.current_op >= self.function_ops.len() {
            self.finished = true;
            return Ok(ByteMessage::create_empty());
        }

        // Reset and configure the message builder for quantum operations
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();

        let mut has_quantum_ops = false;
        let mut batch_count = 0;

        // Process operations in batches
        while self.current_op < self.function_ops.len() && batch_count < MAX_BATCH_SIZE {
            let instruction = &self.function_ops[self.current_op];

            // Process based on operation type
            match &instruction.operation {
                crate::ops::Operation::Quantum(quantum_op) => {
                    // Process quantum operation
                    match self.processor.process_quantum_operation(
                        quantum_op,
                        instruction,
                        &mut self.message_builder,
                    ) {
                        Ok(true) => {
                            has_quantum_ops = true;
                            batch_count += 1;
                        }
                        Ok(false) => {}
                        Err(e) => {
                            self.finished = true;
                            return Err(PecosError::Input(format!(
                                "Error processing quantum operation: {e}"
                            )));
                        }
                    }
                }
                crate::ops::Operation::Classical(classical_op) => {
                    // Check if this classical operation depends on measurement results
                    // We need to check if any operand SSA ID will be produced by a future measurement
                    let depends_on_measurements = instruction.operands.iter().any(|operand| {
                        // Check if this operand SSA ID is produced by a Measure operation
                        // Look ahead in the operations to see if this SSA ID is a measurement result
                        self.function_ops[..self.current_op].iter().any(|prev_op| {
                            matches!(
                                prev_op.operation,
                                crate::ops::Operation::Quantum(crate::ops::QuantumOp::Measure)
                            ) && prev_op.results.iter().any(|r| r.id == operand.id)
                        })
                    });

                    if depends_on_measurements {
                        // This operation depends on measurements
                        // If we have quantum ops to send, stop here and wait for measurements
                        if has_quantum_ops {
                            break;
                        }
                        // Otherwise, try to process it - measurements should be available
                    }

                    // Process classical operation
                    if let Err(e) = self
                        .processor
                        .process_classical_operation(classical_op, instruction)
                    {
                        self.finished = true;
                        return Err(PecosError::Input(format!(
                            "Error processing classical operation: {e}"
                        )));
                    }
                }
                crate::ops::Operation::Builtin(builtin_op) => {
                    // Process builtin operation
                    if let Err(e) = self.processor.process_builtin_operation(
                        builtin_op,
                        instruction,
                        &mut self.message_builder,
                    ) {
                        self.finished = true;
                        return Err(PecosError::Input(format!(
                            "Error processing builtin operation: {e}"
                        )));
                    }
                }
                _ => {
                    // Skip other operations for now
                }
            }

            self.current_op += 1;

            // If we have quantum operations and reached batch size, stop here
            if has_quantum_ops && batch_count >= MAX_BATCH_SIZE {
                break;
            }
        }

        // If we've processed all operations, mark as finished
        if self.current_op >= self.function_ops.len() {
            self.finished = true;
        }

        // Build and return the message
        let msg = self.message_builder.build();
        Ok(msg)
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> std::result::Result<(), PecosError> {
        // Extract measurement outcomes from the ByteMessage
        let outcomes = message.outcomes().map_err(|e| {
            PecosError::Input(format!("Failed to extract measurement outcomes: {e}"))
        })?;

        // Convert u32 outcomes to u8 for the processor
        let outcomes_u8: Vec<u8> = outcomes
            .iter()
            .map(|&x| {
                u8::try_from(x).map_err(|_| {
                    PecosError::Input(format!(
                        "Measurement outcome {x} does not fit in u8 (must be 0-255)"
                    ))
                })
            })
            .collect::<std::result::Result<Vec<u8>, PecosError>>()?;

        // Process the measurement results
        self.processor
            .handle_measurement_results(&outcomes_u8)
            .map_err(|e| PecosError::Input(format!("Failed to handle measurement results: {e}")))?;

        // Check if all operations have been processed
        if self.current_op >= self.function_ops.len() {
            self.processor.finalize_exports();
        }

        Ok(())
    }

    fn get_results(&self) -> std::result::Result<Shot, PecosError> {
        let mut shot = Shot::default();

        // Use processor's export results which are set up by Result operations
        let export_results = self.processor.get_export_results();

        for (export_name, value) in export_results {
            let data = match value {
                TypedValue::I8(v) => Data::I32(i32::from(v)),
                TypedValue::I16(v) => Data::I32(i32::from(v)),
                TypedValue::I32(v) => Data::I32(v),
                TypedValue::I64(v) => Data::I64(v),
                TypedValue::U8(v) => Data::U32(u32::from(v)),
                TypedValue::U16(v) => Data::U32(u32::from(v)),
                TypedValue::U32(v) => Data::U32(v),
                TypedValue::U64(v) => Data::U64(v),
                TypedValue::F64(v) => {
                    // Convert f64 to i64 bits for transport
                    #[allow(clippy::cast_possible_wrap)] // intentional bit-reinterpretation
                    Data::I64(v.to_bits() as i64)
                }
                TypedValue::Bool(v) => Data::U32(u32::from(v)),
                TypedValue::BitVec(v) => {
                    // Convert bit vector to integer (bits packed as a number)
                    if v.len() > 32 {
                        let mut result = 0u64;
                        for (i, bit) in v.iter().enumerate().take(64) {
                            if *bit {
                                result |= 1u64 << i;
                            }
                        }
                        Data::U64(result)
                    } else {
                        let mut result = 0u32;
                        for (i, bit) in v.iter().enumerate() {
                            if *bit {
                                result |= 1u32 << i;
                            }
                        }
                        Data::U32(result)
                    }
                }
            };

            shot.data.insert(export_name, data);
        }

        Ok(shot)
    }

    fn compile(&self) -> std::result::Result<(), PecosError> {
        // Validate the PHIR module structure
        if let Some(module) = &self.module {
            // TODO: Add more comprehensive validation
            if module.name.is_empty() {
                return Err(PecosError::Input("Module name cannot be empty".to_string()));
            }
        }

        Ok(())
    }

    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        // IMPORTANT: Override the default no-op implementation
        // to ensure proper reset when called through ClassicalEngine trait
        Engine::reset(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for PhirEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        _input: (),
    ) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        // Reset state for a fresh start
        self.finished = false;
        self.current_op = 0;
        self.processor.reset();
        self.message_builder.reset();

        // Generate first batch of commands
        match self.generate_commands() {
            Ok(commands) => {
                if commands.as_bytes().is_empty() {
                    // No quantum operations, finalize and return results immediately
                    self.processor.finalize_exports();
                    Ok(EngineStage::Complete(self.get_results()?))
                } else {
                    Ok(EngineStage::NeedsProcessing(commands))
                }
            }
            Err(e) => Err(e),
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        // Handle the measurements
        self.handle_measurements(measurements)?;

        // Generate next batch of commands (if any)
        match self.generate_commands() {
            Ok(commands) => {
                if commands.as_bytes().is_empty() || self.finished {
                    // No more commands, finalize exports now that all operations are done
                    self.processor.finalize_exports();
                    Ok(EngineStage::Complete(self.get_results()?))
                } else {
                    Ok(EngineStage::NeedsProcessing(commands))
                }
            }
            Err(e) => Err(e),
        }
    }

    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        // Delegate to Engine trait implementation
        Engine::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phir::{Block, Module, Region};
    use crate::region_kinds::RegionKind;

    #[test]
    fn test_empty_engine() {
        let engine = PhirEngine::empty();
        assert_eq!(engine.num_qubits(), 0); // No qubits in empty engine
        assert!(engine.module().is_none());
    }

    #[test]
    fn test_engine_with_module() {
        let module = Module {
            name: "test_module".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let engine = PhirEngine::new(module).unwrap();
        assert!(engine.module().is_some());
        assert_eq!(engine.module().unwrap().name, "test_module");
    }

    #[test]
    fn test_engine_compile() {
        let module = Module {
            name: "test_module".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let engine = PhirEngine::new(module).unwrap();
        assert!(engine.compile().is_ok());
    }

    // ──────────────────────────────────────────────────────────────────
    // ControlEngine start() / continue_processing() tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_control_engine_start_empty() {
        let module = Module {
            name: "empty_control".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        let result = ControlEngine::start(&mut engine, ());
        // start() should succeed on empty module
        assert!(result.is_ok());
    }

    #[test]
    fn test_control_engine_start_with_quantum_ops() {
        use crate::ops::{Operation, QuantumOp};
        use crate::phir::{Instruction, SSAValue};
        use crate::types::Type;

        let h_instr = Instruction {
            operation: Operation::Quantum(QuantumOp::H),
            operands: vec![SSAValue { id: 0, version: 0 }],
            results: vec![SSAValue { id: 1, version: 0 }],
            result_types: vec![Type::Qubit],
            regions: vec![],
            attributes: std::collections::BTreeMap::new(),
            location: None,
        };

        let module = Module {
            name: "quantum_control".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![h_instr],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        let result = ControlEngine::start(&mut engine, ()).unwrap();
        // Should need processing (quantum ops need to be sent to quantum engine)
        assert!(matches!(result, EngineStage::NeedsProcessing(_)));
    }

    #[test]
    fn test_control_engine_continue_processing() {
        use crate::ops::{Operation, QuantumOp};
        use crate::phir::{Instruction, SSAValue};
        use crate::types::Type;
        use pecos_engines::byte_message::builder::ByteMessageBuilder;

        let measure_instr = Instruction {
            operation: Operation::Quantum(QuantumOp::Measure),
            operands: vec![SSAValue { id: 0, version: 0 }],
            results: vec![SSAValue { id: 1, version: 0 }],
            result_types: vec![Type::Bit],
            regions: vec![],
            attributes: std::collections::BTreeMap::new(),
            location: None,
        };

        let module = Module {
            name: "measure_control".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![measure_instr],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        let stage = ControlEngine::start(&mut engine, ()).unwrap();

        if let EngineStage::NeedsProcessing(_commands) = stage {
            // Provide measurement results
            let mut builder = ByteMessageBuilder::new();
            let _ = builder.for_outcomes();
            builder.add_outcomes(&[0]);
            let measurements = builder.build();

            let result = engine.continue_processing(measurements).unwrap();
            // After processing measurements, should be complete
            assert!(matches!(result, EngineStage::Complete(_)));
        }
    }

    #[test]
    fn test_control_engine_reset() {
        let module = Module {
            name: "reset_control".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        // Run start
        let _ = ControlEngine::start(&mut engine, ()).unwrap();
        // Reset
        ControlEngine::reset(&mut engine).unwrap();
        assert!(!engine.finished);
        assert_eq!(engine.current_op, 0);
        // Should be able to start again
        let result = ControlEngine::start(&mut engine, ());
        assert!(result.is_ok());
    }

    // ──────────────────────────────────────────────────────────────────
    // get_results() data type conversion tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_get_results_bool_export() {
        let module = Module {
            name: "bool_export".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        engine
            .processor
            .final_exports
            .insert("bool_val".to_string(), TypedValue::Bool(true));

        let shot = engine.get_results().unwrap();
        assert_eq!(shot.data.get("bool_val"), Some(&Data::U32(1)));
    }

    #[test]
    fn test_get_results_f64_export() {
        let module = Module {
            name: "f64_export".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        let f64_val = 3.5_f64;
        engine
            .processor
            .final_exports
            .insert("float_val".to_string(), TypedValue::F64(f64_val));

        let shot = engine.get_results().unwrap();
        // F64 is transported as I64 via to_bits
        #[allow(clippy::cast_possible_wrap)] // intentional bit-reinterpretation
        let expected = Data::I64(f64_val.to_bits() as i64);
        assert_eq!(shot.data.get("float_val"), Some(&expected));
    }

    #[test]
    fn test_get_results_bitvec_export() {
        let module = Module {
            name: "bitvec_export".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        // BitVec [true, false, true] = 0b101 = 5
        engine.processor.final_exports.insert(
            "bits".to_string(),
            TypedValue::BitVec(vec![true, false, true]),
        );

        let shot = engine.get_results().unwrap();
        assert_eq!(shot.data.get("bits"), Some(&Data::U32(5)));
    }

    #[test]
    fn test_get_results_bitvec_32_bits() {
        let module = Module {
            name: "bitvec_32".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        // 32 bits: bit 0 and bit 31 set = 0x8000_0001
        let mut bits = vec![false; 32];
        bits[0] = true;
        bits[31] = true;
        engine
            .processor
            .final_exports
            .insert("bits32".to_string(), TypedValue::BitVec(bits));

        let shot = engine.get_results().unwrap();
        assert_eq!(shot.data.get("bits32"), Some(&Data::U32(0x8000_0001)));
    }

    #[test]
    fn test_get_results_bitvec_more_than_32_bits() {
        let module = Module {
            name: "bitvec_large".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();
        // 40 bits: bit 0 and bit 35 set
        let mut bits = vec![false; 40];
        bits[0] = true;
        bits[35] = true;
        engine
            .processor
            .final_exports
            .insert("bits40".to_string(), TypedValue::BitVec(bits));

        let shot = engine.get_results().unwrap();
        // Should use U64 for >32 bits
        assert_eq!(shot.data.get("bits40"), Some(&Data::U64(1 | (1u64 << 35))));
    }

    #[test]
    fn test_get_results_narrow_int_exports() {
        let module = Module {
            name: "narrow_int_export".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();

        engine
            .processor
            .final_exports
            .insert("i8_val".to_string(), TypedValue::I8(42));
        engine
            .processor
            .final_exports
            .insert("i16_val".to_string(), TypedValue::I16(-100));
        engine
            .processor
            .final_exports
            .insert("u8_val".to_string(), TypedValue::U8(255));
        engine
            .processor
            .final_exports
            .insert("u16_val".to_string(), TypedValue::U16(1000));

        let shot = engine.get_results().unwrap();
        assert_eq!(shot.data.get("i8_val"), Some(&Data::I32(42)));
        assert_eq!(shot.data.get("i16_val"), Some(&Data::I32(-100)));
        assert_eq!(shot.data.get("u8_val"), Some(&Data::U32(255)));
        assert_eq!(shot.data.get("u16_val"), Some(&Data::U32(1000)));
    }

    #[test]
    fn test_get_results_i32_i64_u32_u64_exports() {
        let module = Module {
            name: "int_export".to_string(),
            attributes: std::collections::BTreeMap::new(),
            body: Region {
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    operations: vec![],
                    terminator: None,
                    attributes: std::collections::BTreeMap::new(),
                }],
                kind: RegionKind::SSACFG,
                attributes: std::collections::BTreeMap::new(),
            },
        };

        let mut engine = PhirEngine::new(module).unwrap();

        engine
            .processor
            .final_exports
            .insert("i32_val".to_string(), TypedValue::I32(-7));
        engine
            .processor
            .final_exports
            .insert("i64_val".to_string(), TypedValue::I64(-99));
        engine
            .processor
            .final_exports
            .insert("u32_val".to_string(), TypedValue::U32(12345));
        engine
            .processor
            .final_exports
            .insert("u64_val".to_string(), TypedValue::U64(999_999));

        let shot = engine.get_results().unwrap();
        assert_eq!(shot.data.get("i32_val"), Some(&Data::I32(-7)));
        assert_eq!(shot.data.get("i64_val"), Some(&Data::I64(-99)));
        assert_eq!(shot.data.get("u32_val"), Some(&Data::U32(12345)));
        assert_eq!(shot.data.get("u64_val"), Some(&Data::U64(999_999)));
    }
}
