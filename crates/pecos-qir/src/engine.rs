use crate::command_generation;
use crate::common::get_thread_id;
use crate::compiler::QirCompiler;
use crate::library::QirLibrary;
use crate::measurement;
use log::{debug, trace, warn};
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::{ByteMessage, QuantumCmd, QuantumCommand};
use pecos_engines::core::shot_results::ShotResult;
use pecos_engines::engines::ClassicalEngine;
use pecos_engines::engines::Engine;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Configuration options for the QIR engine
///
/// This struct encapsulates all configuration options for the QIR engine,
/// making it easier to understand what options are available and how they
/// affect the engine's behavior.
///
/// # Examples
///
/// ```
/// use pecos_qir::engine::{QirEngineConfig, QirEngine};
/// use std::path::PathBuf;
///
/// let config = QirEngineConfig::new()
///     .with_assigned_shots(1000)
///     .with_verbose(true);
///
/// let engine = QirEngine::with_config(PathBuf::from("path/to/qir_file.ll"), config);
/// ```
#[derive(Debug, Clone)]
pub struct QirEngineConfig {
    /// Number of shots assigned to this engine
    pub assigned_shots: usize,

    /// Whether to show verbose command logs
    pub verbose: bool,

    /// Maximum number of retries for library loading
    pub max_load_retries: usize,

    /// Timeout in milliseconds between retries
    pub retry_timeout_ms: u64,
}

impl Default for QirEngineConfig {
    fn default() -> Self {
        Self {
            assigned_shots: 0,
            verbose: false,
            max_load_retries: 3,
            retry_timeout_ms: 500,
        }
    }
}

impl QirEngineConfig {
    /// Create a new `QirEngineConfig` with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of shots assigned to this engine
    #[must_use]
    pub fn with_assigned_shots(mut self, shots: usize) -> Self {
        self.assigned_shots = shots;
        self
    }

    /// Set whether to show verbose command logs
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the maximum number of retries for library loading
    #[must_use]
    pub fn with_max_load_retries(mut self, retries: usize) -> Self {
        self.max_load_retries = retries;
        self
    }

    /// Set the timeout in milliseconds between retries
    #[must_use]
    pub fn with_retry_timeout_ms(mut self, timeout: u64) -> Self {
        self.retry_timeout_ms = timeout;
        self
    }
}

/// QIR Engine for executing quantum programs compiled to QIR
///
/// The QIR Engine loads and executes quantum programs that have been compiled to the
/// Quantum Intermediate Representation (QIR). It handles the interaction between the
/// QIR runtime and the quantum system, processing measurement results, and generating
/// quantum operations.
///
/// # Architecture
///
/// ```text
///   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
///   │  QIR File   │────▶│  QIR Library│────▶│ Quantum Cmds│
///   └─────────────┘     └─────────────┘     └─────────────┘
///                                                  │
///                                                  ▼
///   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
///   │ Shot Results│◀────│ Measurements│◀────│Quantum System│
///   └─────────────┘     └─────────────┘     └─────────────┘
/// ```
///
/// # Thread Safety
///
/// The QIR Engine is designed to be thread-safe and can be cloned for use in multiple
/// threads. Each thread gets its own copy of the QIR library to avoid conflicts.
///
/// # Error Handling
///
/// Errors are propagated through the Result type and logged at their source with
/// appropriate context, including the thread ID.
///
/// # Examples
///
/// ```
/// use pecos_qir::engine::{QirEngine, QirEngineConfig};
/// use std::path::PathBuf;
///
/// // Create a QIR engine with default configuration
/// let engine = QirEngine::new(PathBuf::from("path/to/qir_file.ll"));
///
/// // Create a QIR engine with custom configuration
/// let config = QirEngineConfig::new()
///     .with_assigned_shots(1000)
///     .with_verbose(true);
///
/// let engine = QirEngine::with_config(PathBuf::from("path/to/qir_file.ll"), config);
/// ```
pub struct QirEngine {
    /// The loaded QIR library for executing quantum programs
    library: Option<Box<QirLibrary>>,

    /// Map of measurement results by `result_id`
    measurement_results: HashMap<usize, u32>,

    /// Map of result IDs to custom names (like "c")
    result_name_map: measurement::ResultNameMap,

    /// Path to the QIR file to execute
    qir_file: PathBuf,

    /// Path to the compiled library file
    library_path: Option<PathBuf>,

    /// Flag indicating whether commands have been generated for the current shot
    commands_generated: bool,

    /// Number of shots processed so far
    shot_count: usize,

    /// Configuration options for the engine
    config: QirEngineConfig,
}

impl QirEngine {
    /// Helper function to log errors with thread ID context
    fn log_error<E: std::fmt::Display>(context: &str, error: E) -> PecosError {
        let thread_id = get_thread_id();
        warn!("QIR Engine: [Thread {}] {}: {}", thread_id, context, error);
        PecosError::Processing(format!("QIR operation failed - {context}: {error}"))
    }

    /// Create a new QIR engine with default configuration
    ///
    /// # Arguments
    ///
    /// * `qir_file` - Path to the QIR file to execute
    ///
    /// # Returns
    ///
    /// A new QIR engine instance with default configuration
    #[must_use]
    pub fn new(qir_file: PathBuf) -> Self {
        debug!("QIR: Creating new engine with program path: {:?}", qir_file);
        Self {
            library: None,
            measurement_results: HashMap::new(),
            result_name_map: measurement::ResultNameMap::new(),
            qir_file,
            library_path: None,
            commands_generated: false,
            shot_count: 0,
            config: QirEngineConfig::default(),
        }
    }

    /// Create a new QIR engine with custom configuration
    ///
    /// # Arguments
    ///
    /// * `qir_file` - Path to the QIR file to execute
    /// * `config` - Configuration options for the engine
    ///
    /// # Returns
    ///
    /// A new QIR engine instance with the specified configuration
    #[must_use]
    pub fn with_config(qir_file: PathBuf, config: QirEngineConfig) -> Self {
        debug!(
            "QIR: Creating new engine with program path: {:?} and custom config",
            qir_file
        );
        Self {
            library: None,
            measurement_results: HashMap::new(),
            result_name_map: measurement::ResultNameMap::new(),
            qir_file,
            library_path: None,
            commands_generated: false,
            shot_count: 0,
            config,
        }
    }

    /// Set the number of shots assigned to this engine
    ///
    /// # Arguments
    ///
    /// * `shots` - Number of shots to assign
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if the operation fails
    pub fn set_assigned_shots(&mut self, shots: usize) -> Result<(), PecosError> {
        debug!(
            "QIR: Setting assigned shots to {} (but limiting to 1 shot per run_shot call)",
            shots
        );

        // Set the assigned_shots to the number of shots this worker should run
        self.config.assigned_shots = shots;

        Ok(())
    }

    /// Set whether to show verbose command logs
    ///
    /// # Arguments
    ///
    /// * `verbose` - Whether to show verbose command logs
    pub fn set_verbose(&mut self, verbose: bool) {
        self.config.verbose = verbose;
    }

    /// Reset the internal state of the engine
    fn reset_internal_state(&mut self) {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        debug!("QIR: [Thread {}] Resetting internal state", thread_id);

        // Reset shot count to 0
        let old_shot_count = self.shot_count;
        self.shot_count = 0;

        debug!(
            "QIR: [Thread {}] Reset shot count from {} to 0 (assigned_shots={})",
            thread_id, old_shot_count, self.config.assigned_shots
        );

        // Clear measurement results
        self.measurement_results.clear();

        // Reset result name mapping
        self.result_name_map = measurement::ResultNameMap::new();

        // Reset commands_generated flag
        self.commands_generated = false;

        debug!(
            "QIR: [Thread {}] Cleared measurement results and reset flags",
            thread_id
        );

        // Reset the QIR runtime if we have a library
        if let Some(ref library) = self.library {
            debug!("QIR: [Thread {}] Resetting QIR runtime", thread_id);
            if let Err(e) = library.reset() {
                debug!(
                    "QIR: [Thread {}] Failed to reset QIR runtime: {}",
                    thread_id, e
                );
                // Continue despite error
            }
        }
    }

    /// Set up the QIR library
    fn setup_library(&mut self) -> Result<(), PecosError> {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        // If the library is already set up, don't recompile
        if self.library.is_some() {
            trace!(
                "QIR: [Thread {}] Library already set up, skipping compilation",
                thread_id
            );
            return Ok(());
        }

        debug!("QIR: [Thread {}] Setting up library", thread_id);

        // Clean up any existing library
        self.reset_internal_state();

        // Create a unique temporary directory for this thread
        let temp_dir =
            std::env::temp_dir().join(format!("qir_{}_{}", std::process::id(), thread_id));
        if !temp_dir.exists() {
            std::fs::create_dir_all(&temp_dir)
                .map_err(|e| Self::log_error("Failed to create temp directory", e))?;
        }

        // Check if we already have a library path from a previous compilation
        let library_path = if let Some(ref library_path) = self.library_path {
            debug!(
                "QIR: [Thread {}] Using existing library at {:?} as template",
                thread_id, library_path
            );

            // Create a thread-specific copy of the library
            let thread_specific_path = temp_dir.join(format!("lib_thread_{thread_id}.so"));

            // Copy the library to the thread-specific path
            if library_path.exists() {
                debug!(
                    "QIR: [Thread {}] Copying library to thread-specific path: {:?}",
                    thread_id, thread_specific_path
                );

                std::fs::copy(library_path, &thread_specific_path).map_err(|e| {
                    Self::log_error("Failed to copy library to thread-specific path", e)
                })?;

                thread_specific_path
            } else {
                // If the library doesn't exist, compile it
                debug!(
                    "QIR: [Thread {}] Library template doesn't exist, compiling from source",
                    thread_id
                );
                self.compile_library(&temp_dir)?
            }
        } else {
            // If we don't have a library path, compile the QIR file
            debug!(
                "QIR: [Thread {}] No existing library, compiling from source",
                thread_id
            );
            self.compile_library(&temp_dir)?
        };

        // Load the library
        debug!(
            "QIR: [Thread {}] Loading library from {:?}",
            thread_id, library_path
        );

        let library = QirLibrary::load(&library_path)
            .map_err(|e| Self::log_error("Failed to load QIR library", e))?;

        // Store the library and path
        self.library = Some(Box::new(library));
        self.library_path = Some(library_path);

        debug!(
            "QIR: [Thread {}] Successfully set up QIR library",
            thread_id
        );

        Ok(())
    }

    /// Process measurements from the quantum system
    fn process_measurements(&mut self, message: &ByteMessage) -> Result<(), PecosError> {
        // Use the measurement module to process measurements
        measurement::process_measurements(message, &mut self.measurement_results, self.shot_count)?;

        // Reset the commands_generated flag after processing measurements
        self.commands_generated = false;

        // Increment the shot count
        self.shot_count += 1;

        debug!(
            "QIR: [Thread {}] Completed shot {}",
            get_thread_id(),
            self.shot_count
        );

        Ok(())
    }

    /// Get the results of the quantum computation
    ///
    /// # Returns
    ///
    /// * `ShotResult` - The results of the quantum computation
    fn get_results(&self) -> ShotResult {
        // Use the measurement module to get results with custom result names
        measurement::get_results_with_names(&self.measurement_results, &self.result_name_map)
    }

    /// Compile the QIR program
    pub fn compile(&self) -> Result<(), PecosError> {
        debug!("QIR: Compiling program");
        match QirCompiler::compile(&self.qir_file, None) {
            Ok(_path) => {
                debug!("QIR: Compilation successful");
                Ok(())
            }
            Err(e) => {
                let err_str = format!(
                    "QIR compilation failed for '{}': {}",
                    self.qir_file.display(),
                    e
                );
                Err(PecosError::Processing(err_str))
            }
        }
    }

    /// Pre-compile the QIR library to prepare for cloning
    pub fn pre_compile(&mut self) -> Result<(), PecosError> {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        debug!(
            "QIR: [Thread {}] Pre-compiling library for efficient cloning",
            thread_id
        );

        // If the library is already set up, don't recompile
        if self.library.is_some() && self.library_path.is_some() {
            debug!(
                "QIR: [Thread {}] Library already pre-compiled, skipping",
                thread_id
            );
            return Ok(());
        }

        // Compile the QIR program to a library
        let library_path = QirCompiler::compile(&self.qir_file, None)
            .map_err(|e| PecosError::Processing(format!("Failed to compile QIR program: {e}")))?;

        // Store the library path
        self.library_path = Some(library_path.clone());

        // We don't need to load the library here, as each thread will get its own copy
        debug!(
            "QIR: [Thread {}] Library pre-compiled successfully (path: {:?})",
            thread_id, library_path
        );

        Ok(())
    }

    /// Convert a list of `QuantumCommands` to a `ByteMessage`
    fn commands_to_byte_message(commands: &[QuantumCommand]) -> Result<ByteMessage, PecosError> {
        command_generation::commands_to_byte_message(commands)
    }

    /// Run the QIR program and get the commands
    ///
    /// This method runs the QIR program by calling the main function in the library
    /// and retrieves the generated quantum commands.
    ///
    /// # Arguments
    ///
    /// * `library` - The QIR library to run
    ///
    /// # Returns
    ///
    /// * `Result<Vec<QuantumCmd>, BoxError>` - The quantum commands generated by the QIR program
    ///
    /// # Error Handling
    ///
    /// Errors are propagated through the Result type and logged at their source with
    /// appropriate context, including the thread ID.
    fn run_qir_program(&self, library: &QirLibrary) -> Result<Vec<QuantumCmd>, PecosError> {
        // Configure verbosity through environment variable
        if self.config.verbose {
            unsafe {
                std::env::remove_var("QIR_RUNTIME_QUIET");
            }
        } else {
            unsafe {
                std::env::set_var("QIR_RUNTIME_QUIET", "1");
            }
        }

        // Call the main function in the library
        library.call_function(b"main").map_err(|e| {
            // Special case for removed library files
            if e.to_string().contains("No such file or directory") {
                debug!("QIR: Library file was already removed, continuing");
                PecosError::Processing("Library file was already removed".to_string())
            } else {
                Self::log_error("Failed to call main function", e)
            }
        })?;

        // Get the commands generated by the QIR runtime
        let runtime_commands = library
            .get_binary_commands()
            .map_err(|e| Self::log_error("Failed to get binary commands from QIR runtime", e))?;

        // Log all commands for debugging
        debug!("QIR: Binary commands from runtime:");
        for (i, cmd) in runtime_commands.iter().enumerate() {
            debug!("QIR:   [{}] {:?}", i, cmd);
        }

        Ok(runtime_commands)
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        // Only log at trace level to reduce verbosity
        trace!("QIR: Generating commands (shot {})", self.shot_count + 1);

        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        // If we've already generated commands for this shot, return an empty message
        if self.commands_generated {
            trace!("QIR: Commands already generated for this shot, returning empty message");
            return Ok(ByteMessage::create_flush());
        }

        // If we've already processed a shot in this run_shot call, return an empty message
        if self.shot_count > 0 {
            debug!(
                "QIR: [Thread {}] Already processed one shot in this run_shot call, returning empty message",
                thread_id
            );
            return Ok(ByteMessage::create_flush());
        }

        // Set up library if not already done
        if self.library.is_none() {
            debug!(
                "QIR: [Thread {}] Setting up library before generating commands for shot {}",
                thread_id,
                self.shot_count + 1
            );

            // Try to set up the library, handling "Text file busy" error with a retry
            if let Err(e) = self.setup_library() {
                if e.to_string().contains("Text file busy") {
                    debug!("QIR: Got 'Text file busy' error, trying to recover");
                    // Sleep a bit longer to allow the file to be released
                    thread::sleep(Duration::from_millis(500));
                    // Try to set up the library again
                    self.setup_library().map_err(|e| {
                        warn!(
                            "QIR: [Thread {}] Failed to set up library after retry: {}",
                            thread_id, e
                        );
                        e
                    })?;
                } else {
                    warn!(
                        "QIR: [Thread {}] Failed to set up library: {}",
                        thread_id, e
                    );
                    return Err(e);
                }
            }
        }

        // Run the QIR program
        if let Some(library) = &self.library {
            // Run the QIR program and get the commands
            let runtime_commands = self.run_qir_program(library)?;

            // Process the QIR commands to extract result name information
            for cmd in &runtime_commands {
                self.result_name_map.process_command(cmd);
            }

            // Convert binary commands directly to QuantumCommand objects
            // This avoids the string conversion step
            let commands = command_generation::parse_binary_commands(&runtime_commands);

            // Filter out unsupported command types
            let filtered_commands: Vec<QuantumCommand> = commands
                .into_iter()
                .filter(QuantumCommand::is_supported)
                .collect();

            // Identify circuit boundaries by looking for measurement patterns
            let circuit_commands = Self::identify_circuit_boundaries(&filtered_commands);

            debug!(
                "QIR: [Thread {}] Final circuit commands for shot {}:",
                thread_id,
                self.shot_count + 1
            );
            for (i, cmd) in circuit_commands.iter().enumerate() {
                debug!("QIR:   [{}] {:?}", i, cmd);
            }

            // Convert the commands to a ByteMessage
            let message = Self::commands_to_byte_message(&circuit_commands).map_err(|e| {
                warn!(
                    "QIR: [Thread {}] Failed to convert commands to ByteMessage: {}",
                    thread_id, e
                );
                e
            })?;

            // Mark that we've generated commands for this shot
            self.commands_generated = true;

            Ok(message)
        } else {
            warn!("QIR: [Thread {}] No QIR library loaded", thread_id);
            Err(PecosError::Processing(
                "Cannot generate quantum commands: No QIR library loaded. Call compile() or setup_library() first.".to_string(),
            ))
        }
    }

    /// Identify circuit boundaries by analyzing command patterns
    fn identify_circuit_boundaries(commands: &[QuantumCommand]) -> Vec<QuantumCommand> {
        command_generation::identify_circuit_boundaries(commands)
    }

    /// Reset the engine's state and resources
    ///
    /// This is a private implementation that both trait implementations can call.
    /// It handles resetting the internal state and the quantum system if present.
    fn reset_engine(&mut self) {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        debug!(
            "QIR: [Thread {}] Resetting engine (ClassicalEngine trait)",
            thread_id
        );

        // Clean up internal state
        self.reset_internal_state();
    }

    /// Helper method to find qubit allocations in QIR content using regex patterns
    fn find_qubit_allocations(content: &str) -> (usize, bool) {
        let mut max_qubit_index = 0;
        let mut found_allocation = false;

        // Pattern 1: Direct qubit references like "inttoptr (i64 N to %Qubit*)"
        // These patterns are static and validated at development time, so we use expect()
        // instead of unwrap() to provide more context in case of a programming error
        let direct_pattern = Regex::new(r"inttoptr\s*\(\s*i64\s+(\d+)\s+to\s+%Qubit\*\)")
            .expect("Invalid regex pattern for direct qubit references");
        for cap in direct_pattern.captures_iter(content) {
            if let Some(index_match) = cap.get(1) {
                if let Ok(index) = index_match.as_str().parse::<usize>() {
                    max_qubit_index = max_qubit_index.max(index);
                    found_allocation = true;
                }
            }
        }

        // Pattern 2: Qubit allocations like "__quantum__rt__qubit_allocate()"
        let alloc_pattern = Regex::new(r"__quantum__rt__qubit_allocate\(\)")
            .expect("Invalid regex pattern for qubit allocations");
        let alloc_count = alloc_pattern.find_iter(content).count();
        if alloc_count > 0 {
            max_qubit_index = max_qubit_index.max(alloc_count - 1);
            found_allocation = true;
        }

        // Pattern 3: Array allocations like "__quantum__rt__array_create_1d(i64 8, i64 N)"
        let array_pattern =
            Regex::new(r"__quantum__rt__array_create_1d\s*\(\s*i64\s+\d+\s*,\s*i64\s+(\d+)\s*\)")
                .expect("Invalid regex pattern for array allocations");
        for cap in array_pattern.captures_iter(content) {
            if let Some(size_match) = cap.get(1) {
                if let Ok(size) = size_match.as_str().parse::<usize>() {
                    max_qubit_index = max_qubit_index.max(size - 1);
                    found_allocation = true;
                }
            }
        }

        (max_qubit_index, found_allocation)
    }

    fn analyze_qir_file(&self) -> Result<usize, PecosError> {
        let thread_id = get_thread_id();
        debug!(
            "QIR Engine: [Thread {}] Analyzing QIR file: {:?}",
            thread_id, self.qir_file
        );

        // Check if the file exists
        if !self.qir_file.exists() {
            return Err(PecosError::Resource(format!(
                "Unable to analyze QIR file: File not found at path '{}'",
                self.qir_file.display()
            )));
        }

        // Read the file content - using IO error directly
        let content = fs::read_to_string(&self.qir_file)?;

        // Check if the file is empty
        if content.is_empty() {
            return Err(PecosError::Resource(format!(
                "Unable to analyze QIR file: File is empty at path '{}'",
                self.qir_file.display()
            )));
        }

        // Find qubit allocations in the QIR file
        let (max_qubit_index, found_allocation) = Self::find_qubit_allocations(&content);

        if found_allocation {
            // The number of qubits is the maximum index + 1
            let num_qubits = max_qubit_index + 1;
            debug!(
                "QIR Engine: [Thread {}] Found {} qubits in QIR file",
                thread_id, num_qubits
            );
            Ok(num_qubits)
        } else {
            Err(PecosError::Input(format!(
                "Invalid QIR program: No qubit allocations found in file '{}'. The program must contain at least one qubit allocation.",
                self.qir_file.display()
            )))
        }
    }

    /// Helper method to compile the QIR file to a library
    fn compile_library(&self, output_dir: &Path) -> Result<PathBuf, PecosError> {
        let thread_id = get_thread_id();

        debug!(
            "QIR: [Thread {}] Compiling QIR program to library in {:?}",
            thread_id, output_dir
        );

        let output_dir_path = output_dir.to_path_buf();
        QirCompiler::compile(&self.qir_file, Some(&output_dir_path))
            .map_err(|e| PecosError::Processing(format!("Failed to compile QIR program: {e}")))
    }
}

impl ClassicalEngine for QirEngine {
    /// Returns the number of qubits used in the quantum program
    ///
    /// This method determines the number of qubits by:
    /// 1. First checking if we have measurement results, and using the highest `result_id` + 1
    /// 2. If no measurements are available, analyzing the QIR file to find qubit allocations
    /// 3. If analysis fails, returning 0 to indicate an unknown qubit count
    ///
    /// # Return Value
    ///
    /// * If measurements are available: Returns the highest qubit index + 1
    /// * If QIR file analysis succeeds: Returns the number of qubits found in the QIR file
    /// * If both methods fail: Returns 0, indicating an unknown qubit count
    ///
    /// # Note
    ///
    /// A return value of 0 should be interpreted as "unknown qubit count" rather than
    /// "zero qubits". Methods that depend on knowing the qubit count should handle
    /// this special case appropriately, typically by skipping qubit-specific operations
    /// or using alternative approaches.
    fn num_qubits(&self) -> usize {
        let thread_id = get_thread_id();

        // First, check if we have measurement results
        // If we do, we can determine the number of qubits from the highest result ID
        if !self.measurement_results.is_empty() {
            let max_result_id = self.measurement_results.keys().max().unwrap_or(&0);
            let num_qubits = max_result_id + 1;
            debug!(
                "QIR Engine: [Thread {}] Determined {} qubits from measurement results",
                thread_id, num_qubits
            );
            return num_qubits;
        }

        // If we don't have measurement results, analyze the QIR file
        match self.analyze_qir_file() {
            Ok(num_qubits) => {
                debug!(
                    "QIR Engine: [Thread {}] Determined {} qubits from QIR file analysis",
                    thread_id, num_qubits
                );
                num_qubits
            }
            Err(e) => {
                // Log appropriate warning based on error type
                let message = format!("{e}");

                warn!(
                    "QIR Engine: [Thread {}] Could not determine qubit count: {}",
                    thread_id, message
                );

                // Return 0 to indicate unknown qubit count
                warn!(
                    "QIR Engine: [Thread {}] Returning 0 to indicate unknown qubit count",
                    thread_id
                );
                0
            }
        }
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        // Use the implementation from QirEngine to avoid code duplication
        self.generate_commands()
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        // Use the process_measurements implementation
        self.process_measurements(&message)
    }

    fn get_results(&self) -> Result<ShotResult, PecosError> {
        // Use the implementation from QirEngine
        Ok(self.get_results())
    }

    fn compile(&self) -> Result<(), PecosError> {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        debug!("QIR: [Thread {}] Compiling program", thread_id);
        match QirCompiler::compile(&self.qir_file, None) {
            Ok(library_path) => {
                debug!(
                    "QIR: [Thread {}] Compilation successful, library at {:?}",
                    thread_id, library_path
                );
                Ok(())
            }
            Err(e) => {
                let err_str = format!(
                    "QIR compilation failed for '{}': {}",
                    self.qir_file.display(),
                    e
                );
                Err(PecosError::Processing(err_str))
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Call the common reset implementation
        self.reset_engine();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Clone for QirEngine {
    fn clone(&self) -> Self {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();
        debug!("QIR: [Thread {}] Cloning engine", thread_id);

        // Create a new engine with a fresh state
        let cloned = Self {
            library: None,                       // Start with no library, will be loaded on demand
            measurement_results: HashMap::new(), // Start with empty measurements
            result_name_map: measurement::ResultNameMap::new(), // Start with empty result name mapping
            qir_file: self.qir_file.clone(),
            library_path: self.library_path.clone(),
            commands_generated: false,   // Reset commands_generated flag
            shot_count: 0,               // Reset shot count
            config: self.config.clone(), // Keep the configuration
        };

        debug!(
            "QIR: [Thread {}] Created clone with fresh state (library will be loaded on demand)",
            thread_id
        );

        cloned
    }
}

impl Drop for QirEngine {
    fn drop(&mut self) {
        self.reset_internal_state();
    }
}

impl Engine for QirEngine {
    type Input = ();
    type Output = ShotResult;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, PecosError> {
        // Generate commands, process them, and return results
        let commands = self.generate_commands()?;
        // ByteMessage::is_empty() should include context if it fails
        if !commands.is_empty()? {
            // In a real processing scenario, these commands would be sent to a quantum engine
            // Here we're just handling an empty processing case
            self.handle_measurements(ByteMessage::builder().build())?;
        }
        Ok(self.get_results())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_engine();
        Ok(())
    }
}
