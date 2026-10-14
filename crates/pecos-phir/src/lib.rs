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
pub mod mlir_lowering; // PHIR to MLIR lowering
pub mod mlir_toolchain;
pub mod ops; // Core operations
pub mod parsing_ops; // Operations for parsing directly to PHIR
pub mod phir; // Core PHIR structures (Region, Block, Instruction)
pub mod qis_dialect; // QIS dialect operations
pub mod qis_emit; // QIS instruction emission helpers
pub mod qis_parser; // QIS LLVM IR parser
pub mod qis_to_quantum; // QIS dialect -> standard QuantumOps lowering
pub mod region_kinds; // Region execution semantics
pub mod ron_support; // RON serialization/deserialization for debugging
pub mod slr_helpers; // Helper functions for translating from SLR/qeclib patterns
pub mod traits; // Operation traits and interfaces
pub mod types; // Type system // MLIR to LLVM-IR compilation

// Re-export key types
pub use error::{PhirError, Result};
pub use execution::{PhirEngine, PhirEngineBuilder, phir_engine};
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

/// Convenience functions for common workflows
pub mod prelude {
    pub use crate::{Module, Operation, PhirConfig, Type};

    // TODO: Quick circuit building - implement when builders module is ready
    // pub fn circuit() -> builders::CircuitBuilder {
    //     builders::CircuitBuilder::new()
    // }
}

/// Parse QIS LLVM IR and lower QIS dialect ops to standard `QuantumOps` in one step.
///
/// # Errors
///
/// Returns an error if parsing or lowering fails.
pub fn parse_qis_to_quantum(llvm_ir: &str) -> Result<Module> {
    let mut module = qis_parser::parse_qis_llvm_ir(llvm_ir)?;
    qis_to_quantum::convert_qis_to_quantum(&mut module)?;
    Ok(module)
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
}
