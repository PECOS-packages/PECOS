/*!
PECOS PHIR - MLIR-inspired quantum program representation

This crate provides:
1. PHIR (PECOS High-level IR) - MLIR-inspired SSA representation for parsing, optimization and execution
2. Hierarchical structure: Operations contain Regions contain Blocks contain Operations
3. Progressive lowering: parsing ops → high-level ops → low-level ops → execution
4. Multiple execution strategies: interpreter, Rust codegen, MLIR lowering

Key insight: PHIR follows MLIR's design where everything is an Operation, providing a
unified representation from parsing through execution.

Design Philosophy:
- One representation throughout the compilation pipeline
- Flexibility and extensibility through the dialect system
- QEC can be expressed naturally through operations without special types
- Custom types and operations can be added through dialects as needed
- Progressive complexity - start simple, add sophistication as needed
*/

pub mod analysis; // Dominance, use-def chains, and other analyses
pub mod attributes; // Attribute system for metadata and interface implementation
pub mod builtin_ops; // Builtin operations (Module, Function, etc.)
pub mod dialect; // Dialect registration and management
pub mod error; // Error handling
pub mod execution; // PHIR execution engine
pub mod hugr_dialect; // HUGR dialect operations
#[cfg(feature = "hugr")]
pub mod hugr_parser; // HUGR parsing support
pub mod hugr_to_qis; // HUGR to QIS conversion pass
pub mod mlir_lowering; // PHIR to MLIR lowering
pub mod mlir_toolchain;
pub mod ops; // Core operations
pub mod parsing_ops; // Operations for parsing directly to PHIR
pub mod phir; // Core PHIR structures (Region, Block, Instruction)
pub mod qis_dialect; // QIS dialect operations
pub mod region_kinds; // Region execution semantics
pub mod ron_support; // RON serialization/deserialization for debugging
pub mod slr_helpers; // Helper functions for translating from SLR/qeclib patterns
pub mod traits; // Operation traits and interfaces
pub mod types; // Type system // MLIR to LLVM-IR compilation

// Re-export key types
pub use error::{PhirError, Result};
pub use execution::PhirEngine;
pub use ops::Operation;
pub use phir::Module;
pub use ron_support::{ModuleRonExt, from_ron, from_ron_file, to_ron, to_ron_file};
pub use types::Type;

/// Configuration for PHIR compilation and execution
#[derive(Debug, Clone)]
pub struct PhirConfig {
    /// Enable debug output
    pub debug: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Target triple for LLVM (when using MLIR backend)
    pub target_triple: Option<String>,
    /// Generate LLVM IR instead of MLIR text
    pub generate_llvm_ir: bool,
}

// Additional config for Python compatibility
impl PhirConfig {
    /// Create config with debug output setting
    #[must_use]
    pub fn with_debug_output(debug_output: bool) -> Self {
        Self {
            debug: debug_output,
            optimization_level: 2,
            target_triple: None,
            generate_llvm_ir: true,
        }
    }

    /// Set debug output
    #[must_use]
    pub fn debug_output(&self) -> bool {
        self.debug
    }
}

impl Default for PhirConfig {
    fn default() -> Self {
        Self {
            debug: false,
            optimization_level: 2,
            target_triple: None,
            generate_llvm_ir: true, // Default to generating LLVM IR for compatibility
        }
    }
}

/// Main compilation pipeline: Input format → PHIR → Execution
pub struct Pipeline {
    _config: PhirConfig,
}

impl Pipeline {
    #[must_use]
    pub fn new(config: PhirConfig) -> Self {
        Self { _config: config }
    }

    /// Compile and execute from any supported input format
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or execution fails
    pub fn compile_and_execute<T>(&self, _input: &str, _format: InputFormat) -> Result<T> {
        // TODO: Implement the full pipeline:
        // 1. Parse input to PHIR
        // 2. Lower high-level ops to low-level ops
        // 3. Execute using selected strategy
        Err(PhirError::internal(
            "Pipeline execution not yet implemented",
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputFormat {
    HUGR,
    Guppy,
}

/// Convenience functions for common workflows
pub mod prelude {
    pub use crate::{InputFormat, Module, Operation, PhirConfig, Pipeline, Type};

    /// Quick execution from HUGR
    ///
    /// # Errors
    ///
    /// Returns an error if HUGR parsing or execution fails
    pub fn execute_hugr(hugr_json: &str) -> crate::Result<()> {
        let pipeline = Pipeline::new(PhirConfig::default());
        pipeline.compile_and_execute(hugr_json, InputFormat::HUGR)
    }

    /// Quick execution from Guppy
    ///
    /// # Errors
    ///
    /// Returns an error if Guppy parsing or execution fails
    pub fn execute_guppy(guppy_hugr: &str) -> crate::Result<()> {
        let pipeline = Pipeline::new(PhirConfig::default());
        pipeline.compile_and_execute(guppy_hugr, InputFormat::Guppy)
    }

    // TODO: Quick circuit building - implement when builders module is ready
    // pub fn circuit() -> builders::CircuitBuilder {
    //     builders::CircuitBuilder::new()
    // }
}

/// Helper function to compile a PHIR module to LLVM IR or MLIR text
#[cfg(feature = "hugr")]
fn compile_module_to_output(module: &Module, config: &PhirConfig) -> Result<String> {
    use log::debug;

    // Debug: print PHIR structure if debug mode is enabled
    if config.debug {
        debug!("PHIR Module: {}", module.name);
        if let Some(block) = module.body.blocks.first() {
            for instr in &block.operations {
                if let crate::ops::Operation::Builtin(crate::builtin_ops::BuiltinOp::Func(func)) =
                    &instr.operation
                {
                    debug!("  Function: {}", func.name);
                    if let Some(region) = func.body.first()
                        && let Some(block) = region.blocks.first()
                    {
                        for (j, op) in block.operations.iter().enumerate() {
                            debug!("    Instruction {}: {:?}", j, op.operation);
                            debug!("      Operands: {:?}", op.operands);
                            debug!("      Results: {:?}", op.results);
                        }
                        if let Some(term) = &block.terminator {
                            debug!("    Terminator: {term:?}");
                        }
                    }
                }
            }
        }
    }

    // Convert PHIR to MLIR text
    let mlir_text = mlir_lowering::phir_to_mlir(module, config)?;

    // Debug: print MLIR if debug mode is enabled
    if config.debug {
        debug!("\nGenerated MLIR:\n{mlir_text}");
    }

    // If we're generating MLIR for quantum operations, convert to LLVM IR
    if config.generate_llvm_ir {
        // Convert MLIR to LLVM IR using the toolchain
        let mlir_config = mlir_toolchain::MlirToolchainConfig {
            keep_intermediate_files: config.debug,
            ..Default::default()
        };

        let llvm_ir = mlir_toolchain::mlir_to_llvm_ir(&mlir_text, &mlir_config)
            .map_err(|e| PhirError::internal(format!("Failed to convert MLIR to LLVM IR: {e}")))?;

        // Debug: print LLVM IR if debug mode is enabled
        if config.debug {
            debug!("\nGenerated LLVM IR:\n{llvm_ir}");
        }

        Ok(llvm_ir)
    } else {
        Ok(mlir_text)
    }
}

// HUGR support via tket2 (when enabled)
#[cfg(feature = "hugr")]
/// Compile HUGR JSON directly to LLVM IR via PHIR pipeline
///
/// This function provides a direct path from HUGR JSON to LLVM IR for Python bindings
///
/// # Errors
///
/// Returns an error if HUGR parsing or LLVM IR generation fails
pub fn compile_hugr_via_phir(hugr_json: &str, config: &PhirConfig) -> Result<String> {
    // Parse HUGR to PHIR (handles both actual HUGR and simplified test format)
    let module = hugr_parser::parse_hugr_to_phir(hugr_json)?;
    compile_module_to_output(&module, config)
}

#[cfg(feature = "hugr")]
/// Compile HUGR bytes (JSON or binary) to LLVM IR via PHIR pipeline
///
/// This function handles both JSON and binary HUGR formats
///
/// # Errors
///
/// Returns an error if HUGR parsing or LLVM IR generation fails
pub fn compile_hugr_bytes_via_phir(hugr_bytes: &[u8], config: &PhirConfig) -> Result<String> {
    // Parse HUGR to PHIR
    let module = hugr_parser::parse_hugr_bytes_to_phir(hugr_bytes)?;
    compile_module_to_output(&module, config)
}

#[cfg(feature = "hugr")]
/// Convert HUGR to PHIR and then to MLIR text representation
///
/// This function provides a path from HUGR to MLIR text format for debugging and analysis
///
/// # Errors
///
/// Returns an error if HUGR parsing or MLIR conversion fails
pub fn hugr_to_phir_mlir(hugr_json: &str, config: &PhirConfig) -> Result<String> {
    // Parse HUGR to PHIR
    let module = hugr_parser::parse_hugr_to_phir(hugr_json)?;

    // Convert PHIR to MLIR text
    mlir_lowering::phir_to_mlir(&module, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PhirConfig::default();
        assert_eq!(config.optimization_level, 2);
        assert!(!config.debug);
    }

    #[test]
    fn test_pipeline_creation() {
        let config = PhirConfig::default();
        let _pipeline = Pipeline::new(config);
    }
}
