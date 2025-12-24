//! QIS Control Engine - with trait-based interfaces
//!
//! This module implements a `QisEngine` that works with both
//! trait-based interfaces and runtimes, mediating between them.

use crate::qis_interface::{BoxedInterface, ProgramFormat};
use crate::runtime::QisRuntime;
use log::debug;
use pecos_core::prelude::PecosError;
use pecos_engines::noise::utils::NoiseUtils;
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{
    ByteMessage, ByteMessageBuilder, ClassicalEngine, ControlEngine, Engine, EngineStage,
};
use pecos_qis_ffi_types::{OperationCollector as OperationList, QuantumOp};
use pecos_rng::PecosRng;
use std::collections::BTreeMap;

/// QIS Control Engine that mediates between interface and runtime
///
/// This engine contains:
/// - A `QisInterface` implementation (JIT, Helios, etc.) for executing programs
/// - A `QisRuntime` implementation (Native, Selene, etc.) for managing control flow
pub struct QisEngine {
    /// The QIS interface (program executor)
    interface: Option<BoxedInterface>,

    /// The QIS runtime (classical interpreter)
    runtime: Box<dyn QisRuntime>,

    /// Current operations collected from the interface
    current_operations: Option<OperationList>,

    /// Number of qubits in the program
    num_qubits: usize,

    /// Whether we've started processing
    started: bool,

    /// Tracking measurement result IDs for the current batch
    measurement_mapping: Vec<usize>,

    /// Stored measurement results for `get_results()`
    measurement_results: BTreeMap<usize, bool>,

    /// RNG for generating per-shot seeds
    rng: PecosRng,

    /// Current shot seed (stored for quantum engine seeding)
    current_shot_seed: Option<u64>,
}

impl QisEngine {
    /// Create a new engine with the given interface and runtime
    #[must_use]
    pub fn new(interface: BoxedInterface, runtime: Box<dyn QisRuntime>) -> Self {
        Self {
            interface: Some(interface),
            runtime,
            current_operations: None,
            num_qubits: 0,
            started: false,
            measurement_mapping: Vec::new(),
            measurement_results: BTreeMap::new(),
            rng: PecosRng::seed_from_u64(0), // Will be properly seeded via set_seed()
            current_shot_seed: None,
        }
    }

    /// Get the current shot seed for quantum engine seeding
    /// This should be called after `start()` to get the seed generated for the current shot
    #[must_use]
    pub fn current_shot_seed(&self) -> Option<u64> {
        self.current_shot_seed
    }

    /// Initialize the engine by collecting operations from the interface
    ///
    /// This should be called for pre-built interfaces to load operations into the runtime
    ///
    /// # Errors
    /// Returns an error if no interface is available, or if operation collection or runtime loading fails.
    pub fn initialize_from_interface(&mut self) -> Result<(), PecosError> {
        if let Some(ref mut interface) = self.interface {
            debug!("Collecting operations from interface");
            let operations = interface
                .collect_operations()
                .map_err(crate::interface_impl::interface_error_to_pecos)?;
            debug!(
                "Collected {} operations, {} allocated qubits",
                operations.operations.len(),
                operations.allocated_qubits.len()
            );

            // Load operations into runtime
            self.runtime.load_interface(operations).map_err(|e| {
                PecosError::Generic(format!("Failed to load operations into runtime: {e}"))
            })?;
            debug!(
                "Runtime loaded, reporting {} qubits",
                self.runtime.num_qubits()
            );
            Ok(())
        } else {
            Err(PecosError::Generic("No interface available".to_string()))
        }
    }

    /// Create with just a runtime (interface will be set later)
    #[must_use]
    pub fn with_runtime(runtime: Box<dyn QisRuntime>) -> Self {
        Self {
            interface: None,
            runtime,
            current_operations: None,
            num_qubits: 0,
            started: false,
            measurement_mapping: Vec::new(),
            measurement_results: BTreeMap::new(),
            rng: PecosRng::seed_from_u64(0), // Will be properly seeded via set_seed()
            current_shot_seed: None,
        }
    }

    /// Set the interface
    pub fn set_interface(&mut self, interface: BoxedInterface) {
        self.interface = Some(interface);
    }

    /// Load a program into both interface and runtime
    ///
    /// # Errors
    /// Returns an error if no interface is set, or if program loading, operation collection, or runtime loading fails.
    pub fn load_program(
        &mut self,
        program_bytes: &[u8],
        format: ProgramFormat,
    ) -> Result<(), PecosError> {
        debug!("Loading program into QisEngine");

        // Load into the interface
        if let Some(ref mut interface) = self.interface {
            // Note: Thread-local state management (for JIT interface) has been removed.
            // The JIT and Native interfaces have been removed from PECOS - use Selene instead.

            interface
                .load_program(program_bytes, format)
                .map_err(crate::interface_impl::interface_error_to_pecos)?;

            // Collect initial operations to set up the runtime
            let operations = interface
                .collect_operations()
                .map_err(crate::interface_impl::interface_error_to_pecos)?;

            // Load the operations into the runtime first
            self.runtime
                .load_interface(operations.clone())
                .map_err(|e| PecosError::Generic(format!("Failed to load into runtime: {e}")))?;

            // Get qubit count from runtime (it should analyze the operations)
            self.num_qubits = self.runtime.num_qubits();
            debug!("Runtime reports {} qubits", self.num_qubits);
            debug!(
                "Interface had {} allocated qubits: {:?}",
                operations.allocated_qubits.len(),
                operations.allocated_qubits
            );

            self.current_operations = Some(operations);
        } else {
            return Err(PecosError::Generic("No interface set".to_string()));
        }

        Ok(())
    }

    /// Convert quantum operations to `ByteMessage` for the quantum engine
    fn operations_to_bytemessage(
        &mut self,
        ops: Vec<QuantumOp>,
    ) -> Result<ByteMessage, PecosError> {
        let mut builder = ByteMessageBuilder::new();
        self.measurement_mapping.clear();

        for op in ops {
            match op {
                QuantumOp::H(qubit) => {
                    builder.add_h(&[qubit]);
                }
                QuantumOp::X(qubit) => {
                    builder.add_x(&[qubit]);
                }
                QuantumOp::Y(qubit) => {
                    builder.add_y(&[qubit]);
                }
                QuantumOp::Z(qubit) => {
                    builder.add_z(&[qubit]);
                }
                QuantumOp::S(qubit) => {
                    builder.add_sz(&[qubit]);
                }
                QuantumOp::Sdg(qubit) => {
                    builder.add_szdg(&[qubit]);
                }
                QuantumOp::T(qubit) => {
                    builder.add_t(&[qubit]);
                }
                QuantumOp::Tdg(qubit) => {
                    builder.add_tdg(&[qubit]);
                }
                QuantumOp::RX(angle, qubit) => {
                    builder.add_rx(angle, &[qubit]);
                }
                QuantumOp::RY(angle, qubit) => {
                    builder.add_ry(angle, &[qubit]);
                }
                QuantumOp::RZ(angle, qubit) => {
                    builder.add_rz(angle, &[qubit]);
                }
                QuantumOp::RXY(theta, phi, qubit) => {
                    builder.add_r1xy(theta, phi, &[qubit]);
                }
                QuantumOp::CX(control, target) => {
                    builder.add_cx(&[control], &[target]);
                }
                QuantumOp::Measure(qubit, result_id) => {
                    self.measurement_mapping.push(result_id);
                    builder.add_measurements(&[qubit]);
                }
                QuantumOp::ZZ(qubit1, qubit2) => {
                    // ZZ gate is the same as SZZ in PECOS
                    builder.add_szz(&[qubit1], &[qubit2]);
                }
                QuantumOp::RZZ(angle, qubit1, qubit2) => {
                    builder.add_rzz(angle, &[qubit1], &[qubit2]);
                }
                QuantumOp::Reset(qubit) => {
                    builder.add_prep(&[qubit]);
                }
                _ => {
                    // For other operations, we'd need to add more builder methods
                    // or convert to a generic gate representation
                    return Err(PecosError::Generic(format!(
                        "Unsupported operation: {op:?}"
                    )));
                }
            }
        }

        Ok(builder.build())
    }
}

impl Clone for QisEngine {
    fn clone(&self) -> Self {
        Self {
            interface: None, // Can't easily clone boxed trait objects
            runtime: dyn_clone::clone_box(&*self.runtime),
            current_operations: self.current_operations.clone(),
            num_qubits: self.num_qubits,
            started: self.started,
            measurement_mapping: self.measurement_mapping.clone(),
            measurement_results: self.measurement_results.clone(),
            rng: self.rng.clone(),
            current_shot_seed: self.current_shot_seed,
        }
    }
}

impl Engine for QisEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        debug!("QisEngine::process called");

        // Use the ControlEngine implementation for processing
        let mut stage = self.start(())?;

        loop {
            match stage {
                EngineStage::NeedsProcessing(_) => {
                    // In standalone mode, we can't actually execute quantum ops
                    // Just return empty measurements
                    let empty_msg = ByteMessage::builder().build();
                    stage = self.continue_processing(empty_msg)?;
                }
                EngineStage::Complete(shot) => {
                    return Ok(shot);
                }
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        debug!("QisEngine: reset() called");
        self.runtime
            .reset()
            .map_err(|e| PecosError::Generic(format!("Failed to reset runtime: {e}")))?;
        if let Some(ref mut interface) = self.interface {
            interface
                .reset()
                .map_err(crate::interface_impl::interface_error_to_pecos)?;
        }
        self.current_operations = None;
        self.started = false;
        self.measurement_mapping.clear();
        self.measurement_results.clear();
        self.current_shot_seed = None;
        debug!("QisEngine: reset() completed, cleared measurement_results");
        Ok(())
    }
}

impl ClassicalEngine for QisEngine {
    fn num_qubits(&self) -> usize {
        let num_qubits = self.runtime.num_qubits();
        debug!("QisEngine: num_qubits() returning {num_qubits}");
        num_qubits
    }

    fn set_seed(&mut self, seed: u64) {
        // Seed the RNG for generating per-shot seeds
        self.rng = PecosRng::seed_from_u64(seed);
        debug!("QisEngine: Set master seed to {seed}");
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        debug!("QisEngine::generate_commands called");

        // Get next batch of quantum operations from runtime
        match self.runtime.execute_until_quantum() {
            Ok(Some(ops)) => {
                debug!("QisEngine: Runtime returned {} operations", ops.len());
                for op in &ops {
                    debug!("QisEngine: Operation: {op:?}");
                }
                let quantum_ops: Vec<QuantumOp> = ops;
                let msg = self.operations_to_bytemessage(quantum_ops)?;
                debug!(
                    "QisEngine: Generated ByteMessage with {} measurement mappings",
                    self.measurement_mapping.len()
                );

                // Debug: Print the actual ByteMessage content
                debug!("QisEngine: Generated ByteMessage:");
                if let Ok(quantum_ops) = msg.quantum_ops() {
                    debug!("  Quantum ops: {} total", quantum_ops.len());
                    for (i, gate) in quantum_ops.iter().enumerate() {
                        debug!("    Gate {i}: {gate:?}");
                    }
                }
                if let Ok(empty) = msg.is_empty() {
                    debug!("  Is empty: {empty}");
                }

                Ok(msg)
            }
            Ok(None) => {
                debug!("QisEngine: Runtime complete, no more operations");
                Ok(ByteMessage::builder().build())
            }
            Err(e) => {
                debug!("QisEngine: Runtime error: {e}");
                Err(PecosError::Generic(format!("Runtime error: {e}")))
            }
        }
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        debug!("QisEngine::get_results called");
        debug!(
            "QisEngine: get_results() called, stored results: {:?}",
            self.measurement_results
        );

        // Convert stored measurement results to PECOS shot format
        let mut shot = Shot::default();

        // Add measurements from stored results
        for (result_id, value) in &self.measurement_results {
            shot.data.insert(
                format!("measurement_{result_id}"),
                Data::U32(u32::from(*value)),
            );
            debug!(
                "QisEngine: Added to shot: measurement_{} = {}",
                result_id,
                i32::from(*value)
            );
        }

        debug!("QisEngine: Final shot data: {:?}", shot.data);
        debug!(
            "Returning shot with {} measurement results",
            self.measurement_results.len()
        );
        Ok(shot)
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        debug!("QisEngine::handle_measurements called");

        // Extract measurements from ByteMessage
        let measurements = message
            .outcomes()
            .map_err(|e| PecosError::Generic(format!("Failed to parse measurements: {e}")))?;

        debug!(
            "QisEngine: Received {} measurements: {:?}",
            measurements.len(),
            measurements
        );
        debug!(
            "QisEngine: Mapping size: {}, mapping: {:?}",
            self.measurement_mapping.len(),
            self.measurement_mapping
        );

        // Convert to BTreeMap for the runtime and store for get_results()
        let mut measurement_map = BTreeMap::new();
        for (idx, &value) in measurements.iter().enumerate() {
            if idx < self.measurement_mapping.len() {
                let result_id = self.measurement_mapping[idx];
                let bool_value = value != 0;
                measurement_map.insert(result_id, bool_value);

                // Store for get_results()
                self.measurement_results.insert(result_id, bool_value);
                debug!("QisEngine: Stored measurement result_id={result_id}, value={bool_value}");
            }
        }

        debug!(
            "QisEngine: Final measurement_results: {:?}",
            self.measurement_results
        );

        self.runtime
            .provide_measurements(measurement_map)
            .map_err(|e| PecosError::Generic(format!("Failed to provide measurements: {e}")))
    }

    fn compile(&self) -> Result<(), PecosError> {
        // The QIS program is compiled/loaded when the interface is created
        // This method just confirms the engine is ready for execution
        log::info!("QIS program compilation verified - engine ready for execution");
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl ControlEngine for QisEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        _input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        debug!("QisEngine::start called");

        // Clear previous shot's measurement state
        self.measurement_results.clear();
        self.measurement_mapping.clear();
        debug!("QisEngine: Cleared previous measurement results for new shot");

        // Generate a per-shot seed from our RNG
        let shot_seed = self.rng.next_u64();
        debug!("QisEngine: Generated shot seed {shot_seed}");

        // Store the shot seed for quantum engine access
        self.current_shot_seed = Some(shot_seed);

        // Reset the runtime to ensure clean state for new shot
        self.runtime
            .reset()
            .map_err(|e| PecosError::Generic(format!("Failed to reset runtime: {e}")))?;

        // Start a new shot with the generated seed
        self.runtime
            .shot_start(0, Some(shot_seed))
            .map_err(|e| PecosError::Generic(format!("Failed to start shot: {e}")))?;

        self.started = true;

        // Generate initial commands
        let commands = self.generate_commands()?;

        if commands.is_empty()? && self.runtime.is_complete() {
            // Already complete
            let shot = self.get_results()?;
            Ok(EngineStage::Complete(shot))
        } else {
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn continue_processing(
        &mut self,
        input: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        debug!("QisEngine::continue_processing called");

        // Process the response from quantum engine
        if NoiseUtils::has_measurements(&input) {
            self.handle_measurements(input)?;
        }

        // Check if complete
        if self.runtime.is_complete() {
            let shot = self.get_results()?;
            Ok(EngineStage::Complete(shot))
        } else {
            // Generate next batch of commands
            let commands = self.generate_commands()?;
            Ok(EngineStage::NeedsProcessing(commands))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Reset everything
        <Self as Engine>::reset(self)
    }
}

// Tests for QisEngine are in the implementation crates (pecos-qis-jit, pecos-qis-native, etc.)
// since they require actual interface and runtime implementations.
