//! Program abstraction for QIS Classical Control Engine
//!
//! This module provides a unified program interface that allows different
//! program types (`Qis`, HUGR, raw `QisInterface`) to be used with
//! the `QisEngine` through a consistent `.program()` API.
//!
//! Default implementations use Selene-based interfaces with explicit
//! error handling - no silent fallbacks are provided.

use pecos_core::errors::PecosError;
use pecos_programs::{Hugr, Qis};
use pecos_qis_ffi_types::OperationCollector;
use std::process::Command;
use tempfile::NamedTempFile;

/// A trait for types that can be converted into a `QisInterface`
///
/// This allows the `QisEngine` builder to accept different program types
/// through a unified `.program()` method, similar to how `QASMEngine` works.
///
/// Default implementations use Selene-based interfaces (Helios for QIS/HUGR programs).
/// If the default is not available, explicit error messages guide users to alternatives.
pub trait IntoQisInterface {
    /// Convert this program into a `QisInterface`
    ///
    /// # Errors
    /// Returns an error directing users to use explicit implementation crates.
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError>;
}

/// Program type classification for interface provider selection
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramType {
    /// LLVM IR text format
    LlvmIr,
    /// QIS bitcode format
    QisBitcode,
    /// HUGR bytes format
    HugrBytes,
}

/// Trait for different `QisInterface` implementation strategies
///
/// This allows pluggable compilation strategies - Selene Helios compilation
/// or other future approaches.
pub trait QisInterfaceProvider: Send + Sync {
    /// Get the interface (may involve compilation/linking)
    ///
    /// # Errors
    /// Returns an error if the interface cannot be obtained (e.g., compilation/linking failures).
    fn get_interface(&mut self) -> Result<OperationCollector, PecosError>;

    /// Get provider type for debugging and logging
    fn provider_type(&self) -> &'static str;

    /// Check if this provider can handle the given program type
    fn can_handle(&self, program_type: &ProgramType) -> bool;

    /// Get any metadata about the compilation process
    fn get_metadata(&self) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }
}

/// Selene Helios-based `QisInterface` provider
///
/// This provider uses Selene's Helios compiler to compile QIS bitcode
/// into optimized quantum programs, then converts the result into a `QisInterface`.
#[derive(Debug)]
pub struct QisSeleneHeliosInterface {
    program_data: Vec<u8>,
    program_type: ProgramType,
    metadata: std::collections::BTreeMap<String, String>,
    helios_config: HeliosConfig,
}

/// Configuration for Selene Helios compilation
#[derive(Debug, Clone)]
pub struct HeliosConfig {
    /// Optimization level (0-3)
    pub opt_level: u8,
    /// Target triple for compilation
    pub target_triple: String,
    /// Additional compilation flags
    pub extra_flags: Vec<String>,
    /// Path to Selene installation
    pub selene_path: Option<std::path::PathBuf>,
}

impl Default for HeliosConfig {
    fn default() -> Self {
        Self {
            opt_level: 2,
            target_triple: "native".to_string(),
            extra_flags: Vec::new(),
            selene_path: None,
        }
    }
}

impl QisSeleneHeliosInterface {
    /// Create a new Selene Helios interface provider from QIS bitcode
    #[must_use]
    pub fn from_bitcode(bitcode: Vec<u8>) -> Self {
        Self::from_bitcode_with_config(bitcode, HeliosConfig::default())
    }

    /// Create a new Selene Helios interface provider with custom configuration
    #[must_use]
    pub fn from_bitcode_with_config(bitcode: Vec<u8>, config: HeliosConfig) -> Self {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("bitcode_size".to_string(), bitcode.len().to_string());
        metadata.insert(
            "compilation_strategy".to_string(),
            "selene_helios".to_string(),
        );
        metadata.insert("opt_level".to_string(), config.opt_level.to_string());

        Self {
            program_data: bitcode,
            program_type: ProgramType::QisBitcode,
            metadata,
            helios_config: config,
        }
    }

    /// Create a new Selene Helios interface provider from HUGR bytes
    #[must_use]
    pub fn from_hugr_bytes(hugr_bytes: Vec<u8>) -> Self {
        Self::from_hugr_bytes_with_config(hugr_bytes, HeliosConfig::default())
    }

    /// Create a new Selene Helios interface provider from HUGR bytes with custom configuration
    #[must_use]
    pub fn from_hugr_bytes_with_config(hugr_bytes: Vec<u8>, config: HeliosConfig) -> Self {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("hugr_size".to_string(), hugr_bytes.len().to_string());
        metadata.insert(
            "compilation_strategy".to_string(),
            "selene_helios".to_string(),
        );
        metadata.insert("opt_level".to_string(), config.opt_level.to_string());

        Self {
            program_data: hugr_bytes,
            program_type: ProgramType::HugrBytes,
            metadata,
            helios_config: config,
        }
    }

    /// Create from LLVM IR text by converting to bitcode
    #[must_use]
    pub fn from_llvm_ir(llvm_ir: &str) -> Self {
        #[cfg(feature = "llvm")]
        {
            // Convert LLVM IR text to bitcode using inkwell
            use inkwell::context::Context;
            use inkwell::targets::{InitializationConfig, Target};

            // Initialize LLVM targets
            Target::initialize_native(&InitializationConfig::default()).ok();

            let context = Context::create();
            let bitcode = match context.create_module_from_ir(
                inkwell::memory_buffer::MemoryBuffer::create_from_memory_range(
                    llvm_ir.as_bytes(),
                    "llvm_ir",
                ),
            ) {
                Ok(module) => {
                    // Write module to bitcode
                    module.write_bitcode_to_memory().as_slice().to_vec()
                }
                Err(e) => {
                    log::error!("Failed to convert LLVM IR to bitcode: {e}");
                    // Store the IR text as-is and let Helios handle it
                    llvm_ir.as_bytes().to_vec()
                }
            };

            let mut interface = Self::from_bitcode(bitcode);
            interface
                .metadata
                .insert("original_format".to_string(), "llvm_ir".to_string());
            interface
                .metadata
                .insert("ir_size".to_string(), llvm_ir.len().to_string());
            interface
        }

        #[cfg(not(feature = "llvm"))]
        {
            // Without LLVM support, store the IR text as-is and let Helios handle it
            let mut interface = Self::from_bitcode(llvm_ir.as_bytes().to_vec());
            interface
                .metadata
                .insert("original_format".to_string(), "llvm_ir".to_string());
            interface
                .metadata
                .insert("ir_size".to_string(), llvm_ir.len().to_string());
            interface
                .metadata
                .insert("llvm_conversion".to_string(), "skipped".to_string());
            interface
        }
    }

    /// Compile the program using Selene Helios and convert to `QisInterface`
    fn compile_with_helios(&mut self) -> Result<OperationCollector, PecosError> {
        log::info!(
            "Using Selene Helios compilation strategy for {:?}",
            self.program_type
        );

        match self.program_type {
            ProgramType::QisBitcode => {
                self.compile_bitcode_with_helios()
            }
            ProgramType::HugrBytes => {
                self.compile_hugr_with_helios()
            }
            ProgramType::LlvmIr => {
                Err(PecosError::Generic(
                    "Selene Helios interface cannot compile LLVM IR text directly.\n\
                     \n\
                     The Helios interface is designed for HUGR bytes and QIS bitcode formats.\n\
                     For LLVM IR text, please convert to bitcode first or use a different interface.\n\
                     \n\
                     This is a deprecated code path - modern PECOS uses Selene for all QIS programs.".to_string()
                ))
            }
        }
    }

    /// Compile QIS bitcode using Selene Helios
    fn compile_bitcode_with_helios(&mut self) -> Result<OperationCollector, PecosError> {
        // Compile bitcode to LLVM IR using Selene Helios
        let _llvm_ir = self.compile_bitcode_to_llvm_ir()?;

        // This old implementation is deprecated - use pecos-qis-selene instead
        Err(PecosError::Processing(
            "QisSeleneHeliosInterface is deprecated. Use pecos_qis_selene::QisHeliosInterface instead.".to_string()
        ))
    }

    /// Compile HUGR bytes using Selene Helios
    fn compile_hugr_with_helios(&mut self) -> Result<OperationCollector, PecosError> {
        // Use Selene HUGR compiler (no fallback)
        let _llvm_ir = compile_hugr_with_selene(&self.program_data)?;

        // This old implementation is deprecated - use pecos-qis-selene instead
        Err(PecosError::Processing(
            "QisSeleneHeliosInterface is deprecated. Use pecos_qis_selene::QisHeliosInterface instead.".to_string()
        ))
    }

    /// Compile QIS bitcode to LLVM IR using Selene Helios compiler
    fn compile_bitcode_to_llvm_ir(&mut self) -> Result<String, PecosError> {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Write bitcode to a temporary file
        let mut bitcode_file = NamedTempFile::new()
            .map_err(|e| PecosError::Generic(format!("Failed to create temp file: {e}")))?;
        bitcode_file
            .write_all(&self.program_data)
            .map_err(|e| PecosError::Generic(format!("Failed to write bitcode: {e}")))?;

        // Try multiple strategies to find and use Selene Helios
        self.try_selene_helios_compilation(&bitcode_file)
    }

    /// Try different strategies for Selene Helios compilation
    fn try_selene_helios_compilation(
        &mut self,
        bitcode_file: &NamedTempFile,
    ) -> Result<String, PecosError> {
        let strategy_names = [
            "Custom Path",
            "Environment Variable",
            "Standard Locations",
            "Conda Environment",
            "System Installation",
        ];

        let strategies = [
            self.try_custom_selene_path(bitcode_file),
            self.try_env_selene_path(bitcode_file),
            self.try_standard_selene_locations(bitcode_file),
            self.try_conda_selene(bitcode_file),
            self.try_system_selene(bitcode_file),
        ];

        for (strategy_name, result) in strategy_names.iter().zip(strategies.iter()) {
            match result {
                Ok(llvm_ir) => {
                    log::info!("Selene Helios compilation succeeded using: {strategy_name}");
                    self.metadata
                        .insert("helios_strategy".to_string(), (*strategy_name).to_string());
                    self.metadata
                        .insert("helios_compilation".to_string(), "success".to_string());
                    self.metadata
                        .insert("llvm_ir_size".to_string(), llvm_ir.len().to_string());
                    return Ok(llvm_ir.clone());
                }
                Err(e) => {
                    log::debug!("Selene Helios strategy '{strategy_name}' failed: {e}");
                    self.metadata.insert(
                        format!(
                            "helios_strategy_{}_error",
                            strategy_name.to_lowercase().replace(' ', "_")
                        ),
                        e.to_string(),
                    );
                }
            }
        }

        // If all strategies fail, provide helpful error message
        Err(PecosError::Generic(format!(
            "Selene Helios compilation failed. Unable to find Selene installation after trying: {}. \n\
             \n\
             To use Selene Helios interface, you need to:\n\
             1. Install Selene (https://github.com/CQCL/selene)\n\
             2. Set SELENE_PATH environment variable to the Selene directory\n\
             \n\
             Selene is the only supported interface for QIS programs in modern PECOS.",
            strategy_names.join(", ")
        )))
    }

    /// Try compilation using user-provided Selene path
    fn try_custom_selene_path(&self, bitcode_file: &NamedTempFile) -> Result<String, PecosError> {
        let selene_path = self
            .helios_config
            .selene_path
            .as_ref()
            .ok_or_else(|| PecosError::Generic("No custom Selene path provided".to_string()))?;

        self.run_selene_helios_compiler(selene_path, bitcode_file)
    }

    /// Try compilation using `SELENE_PATH` environment variable
    fn try_env_selene_path(&self, bitcode_file: &NamedTempFile) -> Result<String, PecosError> {
        let selene_path = std::env::var("SELENE_PATH")
            .map_err(|_| PecosError::Generic("SELENE_PATH not set".to_string()))?;

        let path = std::path::PathBuf::from(selene_path);
        self.run_selene_helios_compiler(&path, bitcode_file)
    }

    /// Try compilation using standard Selene installation locations
    fn try_standard_selene_locations(
        &self,
        bitcode_file: &NamedTempFile,
    ) -> Result<String, PecosError> {
        let standard_paths = [
            "/home/ciaranra/Repos/cl_projects/gup/selene",
            "/opt/selene",
            "/usr/local/selene",
            "~/selene",
            "./selene",
            "../selene",
        ];

        for path_str in &standard_paths {
            let path = std::path::PathBuf::from(path_str);
            if path.exists() && path.join("selene-compilers/helios/python").exists() {
                log::debug!("Found Selene at standard location: {}", path.display());
                return self.run_selene_helios_compiler(&path, bitcode_file);
            }
        }

        Err(PecosError::Generic(
            "No Selene found in standard locations".to_string(),
        ))
    }

    /// Try compilation using conda environment
    fn try_conda_selene(&self, bitcode_file: &NamedTempFile) -> Result<String, PecosError> {
        // Check if we're in a conda environment with Selene
        let python_script = r"
import sys
try:
    import selene_helios_compiler
    print(selene_helios_compiler.__file__)
except ImportError:
    sys.exit(1)
"
        .to_string();

        let output = Command::new("python3")
            .arg("-c")
            .arg(&python_script)
            .output()
            .map_err(|e| PecosError::Generic(format!("Failed to check conda Selene: {e}")))?;

        if !output.status.success() {
            return Err(PecosError::Generic(
                "Selene not available in conda environment".to_string(),
            ));
        }

        // Run compilation directly using available Python module
        self.run_conda_selene_compilation(bitcode_file)
    }

    /// Try compilation using system-installed Selene
    fn try_system_selene(&self, bitcode_file: &NamedTempFile) -> Result<String, PecosError> {
        // Check if selene-helios command is available in PATH
        let output = Command::new("which")
            .arg("selene-helios")
            .output()
            .map_err(|_| PecosError::Generic("selene-helios not in PATH".to_string()))?;

        if !output.status.success() {
            return Err(PecosError::Generic(
                "selene-helios command not found".to_string(),
            ));
        }

        // Use command-line tool
        self.run_system_selene_compilation(bitcode_file)
    }

    /// Run Selene Helios compiler from a specific path
    fn run_selene_helios_compiler(
        &self,
        selene_path: &std::path::Path,
        bitcode_file: &NamedTempFile,
    ) -> Result<String, PecosError> {
        let helios_python_path = selene_path.join("selene-compilers/helios/python");

        if !helios_python_path.exists() {
            return Err(PecosError::Generic(format!(
                "Selene Helios Python path not found: {}",
                helios_python_path.display()
            )));
        }

        let python_script = format!(
            r#"
import sys
sys.path.insert(0, '{helios_python_path}')

try:
    from selene_helios_compiler import compile_bitcode_to_llvm_ir
except ImportError as e:
    print(f"Failed to import Selene Helios compiler: {{e}}", file=sys.stderr)
    sys.exit(1)

try:
    with open('{bitcode_path}', 'rb') as f:
        bitcode = f.read()

    llvm_ir = compile_bitcode_to_llvm_ir(
        bitcode,
        opt_level={opt_level},
        target_triple='{target_triple}'
    )
    print(llvm_ir)
except Exception as e:
    print(f"Compilation failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"#,
            helios_python_path = helios_python_path.display(),
            bitcode_path = bitcode_file.path().display(),
            opt_level = self.helios_config.opt_level,
            target_triple = self.helios_config.target_triple
        );

        let output = Command::new("python3")
            .arg("-c")
            .arg(&python_script)
            .output()
            .map_err(|e| PecosError::Generic(format!("Failed to run Selene Helios: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PecosError::Generic(format!(
                "Selene Helios compilation failed: {stderr}"
            )));
        }

        let llvm_ir = String::from_utf8(output.stdout)
            .map_err(|e| PecosError::Generic(format!("Invalid UTF-8 output: {e}")))?;

        log::debug!(
            "Successfully compiled bitcode using Selene Helios from: {}",
            selene_path.display()
        );
        Ok(llvm_ir.trim().to_string())
    }

    /// Run Selene Helios compilation using conda environment
    fn run_conda_selene_compilation(
        &self,
        bitcode_file: &NamedTempFile,
    ) -> Result<String, PecosError> {
        let python_script = format!(
            r#"
import selene_helios_compiler

try:
    with open('{bitcode_path}', 'rb') as f:
        bitcode = f.read()

    llvm_ir = selene_helios_compiler.compile_bitcode_to_llvm_ir(
        bitcode,
        opt_level={opt_level},
        target_triple='{target_triple}'
    )
    print(llvm_ir)
except Exception as e:
    import sys
    print(f"Conda Selene compilation failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"#,
            bitcode_path = bitcode_file.path().display(),
            opt_level = self.helios_config.opt_level,
            target_triple = self.helios_config.target_triple
        );

        let output = Command::new("python3")
            .arg("-c")
            .arg(&python_script)
            .output()
            .map_err(|e| PecosError::Generic(format!("Failed to run conda Selene: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PecosError::Generic(format!(
                "Conda Selene compilation failed: {stderr}"
            )));
        }

        let llvm_ir = String::from_utf8(output.stdout)
            .map_err(|e| PecosError::Generic(format!("Invalid UTF-8 output: {e}")))?;

        Ok(llvm_ir.trim().to_string())
    }

    /// Run Selene Helios compilation using system command
    fn run_system_selene_compilation(
        &self,
        bitcode_file: &NamedTempFile,
    ) -> Result<String, PecosError> {
        let output = Command::new("selene-helios")
            .arg("compile")
            .arg("--input")
            .arg(bitcode_file.path())
            .arg("--output-format")
            .arg("llvm-ir")
            .arg("--opt-level")
            .arg(self.helios_config.opt_level.to_string())
            .arg("--target-triple")
            .arg(&self.helios_config.target_triple)
            .output()
            .map_err(|e| PecosError::Generic(format!("Failed to run system Selene: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PecosError::Generic(format!(
                "System Selene compilation failed: {stderr}"
            )));
        }

        let llvm_ir = String::from_utf8(output.stdout)
            .map_err(|e| PecosError::Generic(format!("Invalid UTF-8 output: {e}")))?;

        Ok(llvm_ir.trim().to_string())
    }
}

impl QisInterfaceProvider for QisSeleneHeliosInterface {
    fn get_interface(&mut self) -> Result<OperationCollector, PecosError> {
        self.compile_with_helios()
    }

    fn provider_type(&self) -> &'static str {
        "Selene Helios"
    }

    fn can_handle(&self, program_type: &ProgramType) -> bool {
        matches!(
            program_type,
            ProgramType::QisBitcode | ProgramType::HugrBytes
        )
    }

    fn get_metadata(&self) -> std::collections::BTreeMap<String, String> {
        self.metadata.clone()
    }
}

/// Implement `IntoQisInterface` for `OperationCollector` itself (identity conversion)
impl IntoQisInterface for OperationCollector {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Ok(self)
    }
}

/// Trait for building `QisInterface` instances from programs
///
/// This trait allows different compilation strategies (e.g., Helios)
/// to be plugged into the `QisEngineBuilder` through the .`interface()` method.
pub trait QisInterfaceBuilder: Send + Sync + dyn_clone::DynClone {
    /// Build a `QisInterface` from the given program using this builder's strategy
    ///
    /// Since we can't call sized methods on trait objects, each implementation
    /// needs to handle the program type directly
    ///
    /// # Errors
    /// Returns an error if the program cannot be built into an interface.
    fn build_from_qis_program(&self, program: Qis) -> Result<OperationCollector, PecosError>;

    /// Build from HUGR program
    ///
    /// # Errors
    /// Returns an error if the program cannot be built into an interface.
    fn build_from_hugr_program(&self, program: Hugr) -> Result<OperationCollector, PecosError>;

    /// Build from pre-built interface
    ///
    /// # Errors
    /// Returns an error if the interface cannot be processed.
    fn build_from_interface(
        &self,
        interface: OperationCollector,
    ) -> Result<OperationCollector, PecosError>;

    /// Get a descriptive name for this builder
    fn name(&self) -> &'static str;
}

// Implement dyn_clone for the trait
dyn_clone::clone_trait_object!(QisInterfaceBuilder);

/// Enum to specify which interface builder to use (for backwards compatibility)
#[derive(Debug, Clone)]
pub enum InterfaceChoice {
    /// Auto-select (returns error, user must choose explicit implementation)
    Auto,
}

/// Implement `IntoQisInterface` for `Qis`
///
/// Users must explicitly specify runtime and interface using the builder API.
impl IntoQisInterface for Qis {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default QIS interface implementation available.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(qis_program)?;\n\n\
            The Selene Helios interface is the reference implementation for QIS programs."
                .to_string(),
        ))
    }
}

/// Implement `IntoQisInterface` for HUGR bytes
///
/// Users must explicitly specify a runtime and interface.
impl IntoQisInterface for &[u8] {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default interface implementation for HUGR bytes.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(hugr_program)?;"
                .to_string(),
        ))
    }
}

/// Implement `IntoQisInterface` for HUGR bytes (owned)
impl IntoQisInterface for Vec<u8> {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default interface implementation for HUGR bytes.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(hugr_program)?;"
                .to_string(),
        ))
    }
}

/// Implement `IntoQisInterface` for `Hugr`
///
/// Users must explicitly specify a runtime and interface.
impl IntoQisInterface for Hugr {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default interface implementation for HUGR programs.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(hugr_program)?;"
                .to_string(),
        ))
    }
}

/// Wrapper type to represent a QIS Control Engine Program
///
/// This is conceptually equivalent to `QisInterface`, but provides a
/// more semantically clear type name for the builder API.
#[derive(Debug, Clone)]
pub struct QisEngineProgram {
    interface: OperationCollector,
}

impl QisEngineProgram {
    /// Create a new program from a `QisInterface`
    #[must_use]
    pub fn new(interface: OperationCollector) -> Self {
        Self { interface }
    }

    /// Create a program from anything that can be converted to `QisInterface`
    ///
    /// # Errors
    /// Returns an error if the conversion fails
    pub fn from_program<P: IntoQisInterface>(program: P) -> Result<Self, PecosError> {
        let interface = program.into_qis_interface()?;
        Ok(Self::new(interface))
    }

    /// Get the underlying `QisInterface`
    #[must_use]
    pub fn into_interface(self) -> OperationCollector {
        self.interface
    }

    /// Get a reference to the underlying `QisInterface`
    #[must_use]
    pub fn interface(&self) -> &OperationCollector {
        &self.interface
    }
}

impl IntoQisInterface for QisEngineProgram {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Ok(self.interface)
    }
}

impl From<OperationCollector> for QisEngineProgram {
    fn from(interface: OperationCollector) -> Self {
        Self::new(interface)
    }
}

// Tests for program conversion are in the implementation crates (pecos-qis-selene, etc.)
// since they require actual interface implementations.

/// Compile HUGR bytes using Selene's compiler
///
/// This uses Selene's proven HUGR→LLVM compiler, ensuring proper qubit ID
/// management and QIS function generation. Returns explicit error if Selene is not available.
fn compile_hugr_with_selene(hugr_bytes: &[u8]) -> Result<String, PecosError> {
    log::info!("Compiling HUGR with Selene compiler (required)");

    // Use Selene's Python compiler - no fallbacks
    compile_hugr_with_selene_python(hugr_bytes).map_err(|e| {
        PecosError::Generic(format!(
            "Selene Helios compilation failed: {e}\n\n\
                To use Helios interface, ensure Selene is installed and available:\n\
                1. Ensure Selene repository is at ../selene or ../../../selene\n\
                2. Build Selene compilers: 'cargo build --release' in Selene directory\n\
                \n\
                Selene is the only supported interface for QIS programs."
        ))
    })
}

/// Compile HUGR using Selene's Python compiler
fn compile_hugr_with_selene_python(hugr_bytes: &[u8]) -> Result<String, PecosError> {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Write HUGR bytes to a temporary file
    let mut hugr_file = NamedTempFile::new()
        .map_err(|e| PecosError::Generic(format!("Failed to create temp file: {e}")))?;
    hugr_file
        .write_all(hugr_bytes)
        .map_err(|e| PecosError::Generic(format!("Failed to write HUGR bytes: {e}")))?;

    // Call Selene's compiler using Python
    let output = Command::new("python3")
        .arg("-c")
        .arg(format!(
            r"
import sys
sys.path.insert(0, '{}/selene-compilers/hugr_qis/python')
from selene_hugr_qis_compiler import compile_to_llvm_ir

with open('{}', 'rb') as f:
    hugr_bytes = f.read()

llvm_ir = compile_to_llvm_ir(hugr_bytes, opt_level=2, target_triple='native')
print(llvm_ir)
",
            "/home/ciaranra/Repos/cl_projects/gup/selene",
            hugr_file.path().display()
        ))
        .output()
        .map_err(|e| PecosError::Generic(format!("Failed to run Selene compiler: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PecosError::Generic(format!(
            "Selene compiler failed: {stderr}"
        )));
    }

    let llvm_ir = String::from_utf8(output.stdout)
        .map_err(|e| PecosError::Generic(format!("Invalid UTF-8 output: {e}")))?;

    log::debug!("Successfully compiled HUGR using Selene compiler");
    Ok(llvm_ir)
}
