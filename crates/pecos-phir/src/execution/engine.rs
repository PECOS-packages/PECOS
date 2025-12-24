/*!
PHIR Execution Engine

Main execution engine for PHIR programs. This implements the `ClassicalEngine` trait
and can execute PHIR modules directly, integrating with the PECOS quantum simulation
infrastructure.
*/

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
            .map(|&x| u8::try_from(x).expect("Measurement outcome should fit in u8"))
            .collect();

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
                super::environment::TypedValue::I8(v) => Data::I32(i32::from(v)),
                super::environment::TypedValue::I16(v) => Data::I32(i32::from(v)),
                super::environment::TypedValue::I32(v) => Data::I32(v),
                super::environment::TypedValue::I64(v) => Data::I64(v),
                super::environment::TypedValue::U8(v) => Data::U32(u32::from(v)),
                super::environment::TypedValue::U16(v) => Data::U32(u32::from(v)),
                super::environment::TypedValue::U32(v) => Data::U32(v),
                super::environment::TypedValue::U64(v) => Data::U64(v),
                super::environment::TypedValue::Bool(v) => Data::U32(u32::from(v)),
                super::environment::TypedValue::BitVec(v) => {
                    // Convert bit vector to u32 (sum of bits as a number)
                    let mut result = 0u32;
                    for (i, bit) in v.iter().enumerate() {
                        if *bit {
                            result |= 1 << i;
                        }
                    }
                    Data::U32(result)
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
}
