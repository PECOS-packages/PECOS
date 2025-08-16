//! QIR Engine Module
//!
//! This module provides the QIR Engine for executing quantum programs compiled to QIR.
use crate::library::QirLibrary;
use crate::linker::QirLinker;
use log::{debug, trace, warn};
use pecos_core::errors::PecosError;
use pecos_engines::Engine;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::engine_system::{ClassicalEngine, ControlEngine, EngineStage};
use pecos_engines::shot_results::{Data, Shot};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Helper function to get the current thread ID as a string
///
/// This function returns the current thread ID formatted as a string.
/// It's used for logging and debugging purposes.
///
/// # Returns
///
/// A string representation of the current thread ID
#[must_use]
pub fn get_thread_id() -> String {
    format!("{:?}", thread::current().id())
}

/// Configuration options for the QIR engine
#[derive(Debug, Clone, Default)]
pub struct QirEngineConfig {
    /// Number of shots assigned to this engine
    pub assigned_shots: usize,
    /// Whether to show verbose command logs
    pub verbose: bool,
}

/// QIR Engine for executing quantum programs compiled to QIR
///
/// The engine loads and executes QIR programs, handling the interaction between
/// the QIR runtime and the quantum system.
pub struct QirEngine {
    /// The loaded QIR library for executing quantum programs
    library: Option<Box<QirLibrary>>,

    /// Map of measurement results by `result_id`
    measurement_results: HashMap<usize, i64>,

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
    /// Helper function to log errors
    fn log_error<E: std::fmt::Display>(context: &str, error: E) -> PecosError {
        warn!("QIR Engine: {context}: {error}");
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
        debug!(
            "QIR: Creating new engine with program path: {}",
            qir_file.display()
        );
        Self {
            library: None,
            measurement_results: HashMap::new(),
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
            "QIR: Creating new engine with program path: {} and custom config",
            qir_file.display()
        );
        Self {
            library: None,
            measurement_results: HashMap::new(),
            qir_file,
            library_path: None,
            commands_generated: false,
            shot_count: 0,
            config,
        }
    }

    /// Set the number of shots assigned to this engine
    pub fn set_assigned_shots(&mut self, shots: usize) {
        debug!("QIR: Setting assigned shots to {shots}");
        self.config.assigned_shots = shots;
    }

    /// Set whether to show verbose command logs
    pub fn set_verbose(&mut self, verbose: bool) {
        self.config.verbose = verbose;
    }

    /// Reset the internal state of the engine
    fn reset_internal_state(&mut self) {
        debug!("QIR: Resetting internal state");
        self.shot_count = 0;
        self.measurement_results.clear();
        self.commands_generated = false;

        if let Some(ref library) = self.library
            && let Err(e) = library.reset()
        {
            debug!("QIR: Failed to reset QIR runtime: {e}");
        }
    }

    /// Set up the QIR library
    fn setup_library(&mut self) -> Result<(), PecosError> {
        // If the library is already set up, don't recompile
        if self.library.is_some() {
            trace!("QIR: Library already set up, skipping compilation");
            return Ok(());
        }

        debug!("QIR: Setting up library");

        // Clean up any existing library
        self.reset_internal_state();

        // Create a unique temporary directory for this thread with more randomness
        let thread_id = get_thread_id();
        // Add timestamp for additional uniqueness across multiple test runs
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        // Use timestamp as a unique identifier - no external dependencies needed
        let temp_dir = std::env::temp_dir().join(format!(
            "qir_{}_{}_{}",
            std::process::id(),
            thread_id,
            timestamp
        ));

        debug!(
            "QIR: Creating unique temporary directory at {}",
            temp_dir.display()
        );

        // Ensure the directory is clean by removing it if it exists
        if temp_dir.exists() {
            debug!("QIR: Temporary directory already exists, removing it first");
            std::fs::remove_dir_all(&temp_dir)
                .map_err(|e| Self::log_error("Failed to clean existing temp directory", e))?;
        }

        // Create the directory
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| Self::log_error("Failed to create temp directory", e))?;

        // Check if we already have a library path from a previous compilation
        let library_path = if let Some(ref library_path) = self.library_path {
            debug!(
                "QIR: Using existing library at {} as template",
                library_path.display()
            );

            // Create a thread-specific copy of the library with platform-specific extension
            let extension = if cfg!(target_os = "windows") {
                "dll"
            } else if cfg!(target_os = "macos") {
                "dylib"
            } else {
                "so"
            };

            let thread_specific_path = temp_dir.join(format!("lib_thread_{thread_id}.{extension}"));

            debug!(
                "QIR: Thread-specific library path: {}",
                thread_specific_path.display()
            );

            // Copy the library to the thread-specific path with verification
            if library_path.exists() {
                // Verify source file is valid before copying
                let metadata = std::fs::metadata(library_path)
                    .map_err(|e| Self::log_error("Failed to get metadata for source library", e))?;

                if !metadata.is_file() {
                    return Err(Self::log_error(
                        "Source library is not a regular file",
                        format!("Path: {}", library_path.display()),
                    ));
                }

                let file_size = metadata.len();
                if file_size < 1024 {
                    return Err(Self::log_error(
                        "Source library file is too small to be valid",
                        format!(
                            "Path: {} (size: {} bytes)",
                            library_path.display(),
                            file_size
                        ),
                    ));
                }

                // Copy the file
                debug!(
                    "QIR: Copying library from {} to {}",
                    library_path.display(),
                    thread_specific_path.display()
                );
                std::fs::copy(library_path, &thread_specific_path).map_err(|e| {
                    Self::log_error("Failed to copy library to thread-specific path", e)
                })?;

                // Verify the copied file
                let copied_metadata = std::fs::metadata(&thread_specific_path)
                    .map_err(|e| Self::log_error("Failed to get metadata for copied library", e))?;

                let copied_size = copied_metadata.len();
                if copied_size != file_size {
                    return Err(Self::log_error(
                        "Copied library file size mismatch",
                        format!("Expected: {file_size} bytes, Got: {copied_size} bytes"),
                    ));
                }

                debug!("QIR: Successfully copied library ({copied_size} bytes)");
                thread_specific_path
            } else {
                // If the library doesn't exist, compile it
                debug!("QIR: Library template doesn't exist, compiling from source");
                self.compile_library(&temp_dir)?
            }
        } else {
            // If we don't have a library path, compile the QIR file
            debug!("QIR: No existing library, compiling from source");
            self.compile_library(&temp_dir)?
        };

        // Load the library
        debug!("QIR: Loading library from {}", library_path.display());

        let library = QirLibrary::load(&library_path)
            .map_err(|e| Self::log_error("Failed to load QIR library", e))?;

        // Store the library and path
        self.library = Some(Box::new(library));
        self.library_path = Some(library_path);

        debug!("QIR: Successfully set up QIR library");

        Ok(())
    }

    /// Process measurements from the quantum system
    fn process_measurements(&mut self, message: &ByteMessage) -> Result<(), PecosError> {
        // Extract raw measurement outcomes
        let outcomes = message.outcomes().map_err(|e| {
            PecosError::Input(format!(
                "Failed to extract measurements from ByteMessage: {e}"
            ))
        })?;

        // Convert to indexed format for compatibility with existing code
        let measurements: Vec<(usize, u32)> = outcomes.into_iter().enumerate().collect();

        self.measurement_results.clear();
        // Convert u32 measurements to i64 for QIR standard
        self.measurement_results.extend(
            measurements
                .iter()
                .map(|(id, value)| (*id, i64::from(*value))),
        );

        // Update the runtime with measurement results
        if let Some(library) = &self.library {
            debug!(
                "QIR: Updating runtime with {} measurement results",
                measurements.len()
            );

            // Convert measurements to the format expected by the runtime
            // The runtime expects pairs of (result_id, value)
            let mut results_data = Vec::with_capacity(measurements.len() * 2);
            for (result_id, value) in measurements {
                debug!("QIR: Measurement result_id={result_id} value={value}");
                results_data.push(u32::try_from(result_id).map_err(|_| {
                    PecosError::Resource(format!(
                        "Result ID {result_id} is too large to fit in u32"
                    ))
                })?);
                results_data.push(value);
            }

            // Call the runtime update function
            library.update_measurement_results(&results_data)?;

            // Now finalize the shot with the measurement results
            library.finalize_shot()?;
        }

        self.commands_generated = false;
        self.shot_count += 1;

        debug!("QIR: Completed shot {}", self.shot_count);
        Ok(())
    }

    /// Get the results of the quantum computation
    ///
    /// # Returns
    ///
    /// * `Shot` - The results of the quantum computation
    fn get_results_impl(&self) -> Shot {
        // Try to get shot results from the runtime
        if let Some(library) = &self.library
            && let Ok(Some(shot)) = library.get_shot_results()
        {
            debug!(
                "QIR: Retrieved shot from runtime with {} registers",
                shot.data.len()
            );
            return shot;
        }

        // Fallback: create shot result from raw measurements
        // This should only happen if the runtime doesn't support shot export
        debug!("QIR: Falling back to raw measurement results");
        let mut shot_result = Shot::default();

        for (&result_id, &value) in &self.measurement_results {
            let name = format!("result_{result_id}");
            // Store all values as I64 for consistency with QIR standard
            shot_result.data.insert(name, Data::I64(value));
        }

        shot_result
    }

    /// Pre-compile the QIR library to prepare for cloning
    ///
    /// # Errors
    ///
    /// Returns an error if the QIR library cannot be pre-compiled.
    pub fn pre_compile(&mut self) -> Result<(), PecosError> {
        // Get the current thread ID for logging
        let thread_id = get_thread_id();

        debug!("QIR: [Thread {thread_id}] Pre-compiling library for efficient cloning");

        // If the library is already set up, don't recompile
        if self.library.is_some() && self.library_path.is_some() {
            debug!("QIR: [Thread {thread_id}] Library already pre-compiled, skipping");
            return Ok(());
        }

        // Compile the QIR program to a library
        let library_path = QirLinker::compile(&self.qir_file, None)
            .map_err(|e| PecosError::Processing(format!("Failed to compile QIR program: {e}")))?;

        // Store the library path
        self.library_path = Some(library_path.clone());

        // We don't need to load the library here, as each thread will get its own copy
        debug!(
            "QIR: [Thread {thread_id}] Library pre-compiled successfully (path: {})",
            library_path.display()
        );

        Ok(())
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
    /// * `Result<ByteMessage, PecosError>` - The binary message generated by the QIR program
    ///
    /// # Error Handling
    ///
    /// Errors are propagated through the Result type and logged at their source with
    /// appropriate context, including the thread ID.
    fn run_qir_program(&self, library: &QirLibrary) -> Result<ByteMessage, PecosError> {
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

        // Get the binary message generated by the QIR runtime
        let runtime_message = library
            .get_binary_commands()
            .map_err(|e| Self::log_error("Failed to get binary commands from QIR runtime", e))?;

        // Log message details for debugging
        debug!(
            "QIR: Binary message from runtime: {} bytes",
            runtime_message.as_bytes().len()
        );

        // Try to parse and log quantum operations for debugging
        if let Ok(operations) = runtime_message.quantum_ops() {
            debug!("QIR: Parsed {} quantum operations:", operations.len());
            for (i, op) in operations.iter().enumerate().take(10) {
                debug!("QIR:   [{i}] {op:?}");
            }
            if operations.len() > 10 {
                debug!("QIR:   ... and {} more operations", operations.len() - 10);
            }
        }

        Ok(runtime_message)
    }

    fn generate_commands_impl(&mut self) -> Result<Option<ByteMessage>, PecosError> {
        // Only log at trace level to reduce verbosity
        trace!("QIR: Generating commands (shot {})", self.shot_count + 1);

        // If we've already generated commands for this shot, return None
        if self.commands_generated {
            trace!("QIR: Commands already generated for this shot, returning None");
            return Ok(None);
        }

        // If we've already processed a shot in this run_shot call, return None
        if self.shot_count > 0 {
            debug!("QIR: Already processed one shot in this run_shot call, returning None");
            return Ok(None);
        }

        // Set up library if not already done
        if self.library.is_none() {
            debug!(
                "QIR: Setting up library before generating commands for shot {}",
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
                        warn!("QIR: Failed to set up library after retry: {e}");
                        e
                    })?;
                } else {
                    warn!("QIR: Failed to set up library: {e}");
                    return Err(e);
                }
            }
        }

        // Run the QIR program
        if let Some(library) = &self.library {
            // Run the QIR program and get the ByteMessage directly
            let runtime_message = self.run_qir_program(library)?;

            debug!(
                "QIR: Got ByteMessage for shot {} with {} bytes",
                self.shot_count + 1,
                runtime_message.as_bytes().len()
            );

            // Mark that we've generated commands for this shot
            self.commands_generated = true;

            // Return the ByteMessage
            Ok(Some(runtime_message))
        } else {
            warn!("QIR: No QIR library loaded");
            Err(PecosError::Processing(
                "Cannot generate quantum commands: No QIR library loaded. Call compile() or setup_library() first.".to_string(),
            ))
        }
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
            if let Some(index_match) = cap.get(1)
                && let Ok(index) = index_match.as_str().parse::<usize>()
            {
                max_qubit_index = max_qubit_index.max(index);
                found_allocation = true;
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
            if let Some(size_match) = cap.get(1)
                && let Ok(size) = size_match.as_str().parse::<usize>()
            {
                max_qubit_index = max_qubit_index.max(size - 1);
                found_allocation = true;
            }
        }

        (max_qubit_index, found_allocation)
    }

    fn analyze_qir_file(&self) -> Result<usize, PecosError> {
        debug!(
            "QIR Engine: Analyzing QIR file: {}",
            self.qir_file.display()
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
            debug!("QIR Engine: Found {num_qubits} qubits in QIR file");
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
        debug!(
            "QIR: Compiling QIR program to library in {}",
            output_dir.display()
        );

        let output_dir_path = output_dir.to_path_buf();
        QirLinker::compile(&self.qir_file, Some(&output_dir_path))
            .map_err(|e| PecosError::Processing(format!("Failed to compile QIR program: {e}")))
    }
}

impl ClassicalEngine for QirEngine {
    /// Returns the number of qubits used in the quantum program
    ///
    /// Returns 0 if the qubit count cannot be determined.
    fn num_qubits(&self) -> usize {
        // First, check if we have measurement results
        // If we do, we can determine the number of qubits from the highest result ID
        if !self.measurement_results.is_empty() {
            let max_result_id = self.measurement_results.keys().max().unwrap_or(&0);
            let num_qubits = max_result_id + 1;
            debug!("QIR Engine: Determined {num_qubits} qubits from measurement results");
            return num_qubits;
        }

        // If we don't have measurement results, analyze the QIR file
        match self.analyze_qir_file() {
            Ok(num_qubits) => {
                debug!("QIR Engine: Determined {num_qubits} qubits from QIR file analysis");
                num_qubits
            }
            Err(e) => {
                warn!("QIR Engine: Could not determine qubit count: {e}");
                // Return 0 to indicate unknown qubit count
                warn!("QIR Engine: Returning 0 to indicate unknown qubit count");
                0
            }
        }
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        // When no commands are left to generate, create an empty message
        // instead of returning an error, to be consistent with other engines
        Ok(self
            .generate_commands_impl()?
            .unwrap_or_else(ByteMessage::create_empty))
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        self.process_measurements(&message)
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        Ok(self.get_results_impl())
    }

    fn compile(&self) -> Result<(), PecosError> {
        debug!("QIR: Compiling program");
        QirLinker::compile(&self.qir_file, None)
            .map(|_| debug!("QIR: Compilation successful"))
            .map_err(|e| {
                PecosError::Processing(format!(
                    "QIR compilation failed for '{}': {}",
                    self.qir_file.display(),
                    e
                ))
            })
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_internal_state();
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
        debug!("QIR: Cloning engine");

        // Create a new engine with a fresh state
        Self {
            library: None,                       // Start with no library, will be loaded on demand
            measurement_results: HashMap::new(), // Start with empty measurements
            qir_file: self.qir_file.clone(),
            library_path: self.library_path.clone(),
            commands_generated: false,   // Reset commands_generated flag
            shot_count: 0,               // Reset shot count
            config: self.config.clone(), // Keep the configuration
        }
    }
}

impl Drop for QirEngine {
    fn drop(&mut self) {
        self.reset_internal_state();
    }
}

impl ControlEngine for QirEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        match self.generate_commands_impl()? {
            Some(commands) => Ok(EngineStage::NeedsProcessing(commands)),
            None => Ok(EngineStage::Complete(self.get_results()?)),
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        // Handle measurements from quantum engine
        self.handle_measurements(measurements)?;

        // Check if we have more commands to process
        match self.generate_commands_impl()? {
            Some(commands) => Ok(EngineStage::NeedsProcessing(commands)),
            None => Ok(EngineStage::Complete(self.get_results()?)),
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_internal_state();
        Ok(())
    }
}

impl Engine for QirEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        // Use the EngineStage pattern for processing
        let mut stage = self.start(input)?;

        while let EngineStage::NeedsProcessing(_commands) = stage {
            // In a real processing scenario, these commands would be sent to a quantum engine
            // Here we're just handling an empty processing case
            let measurements = ByteMessage::builder().build();
            stage = self.continue_processing(measurements)?;
        }

        // Extract the final result
        match stage {
            EngineStage::Complete(output) => Ok(output),
            EngineStage::NeedsProcessing(_) => unreachable!(),
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_internal_state();
        Ok(())
    }
}
