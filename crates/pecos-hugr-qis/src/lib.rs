/*!
HUGR to QIS (Quantum Instruction Set) Compiler for PECOS

This crate compiles HUGR (Hierarchical Unified Graph Representation) to
QIS-compatible LLVM IR for execution on PECOS quantum simulators.

# Features

This crate provides the full functionality of tket2's qis-compiler but without
Python bindings, making it suitable for pure Rust applications.

## HUGR to QIS LLVM Compilation

```text
HUGR JSON/Bytes → [pecos-hugr-qis] → QIS LLVM IR String/Bitcode
```

The generated LLVM IR can then be executed by any compatible execution engine (e.g., pecos-qis-runtime).

# Example

```rust,no_run
use pecos_hugr_qis::{HugrCompiler, HugrCompilerConfig, CompileArgs};
use tket::hugr::llvm::inkwell::OptimizationLevel;

// Simple compilation with defaults
let hugr_bytes = b"example HUGR data";
let compiler = HugrCompiler::new();
let llvm_ir = compiler.compile_hugr_bytes_to_string(hugr_bytes)?;

// Advanced compilation with custom options
let mut args = CompileArgs::default();
args.opt_level = OptimizationLevel::Aggressive;
args.target_triple = Some("aarch64-apple-darwin".to_string());
args.save_hugr = Some("debug.hugr".into());

let llvm_ir = compiler.compile_hugr_bytes_to_string_with_options(hugr_bytes, &args)?;

// Compile to bitcode instead of text
let bitcode = compiler.compile_hugr_bytes_to_bitcode(hugr_bytes)?;
# Ok::<(), pecos_core::errors::PecosError>(())
```

# Target Triple Support

The compiler supports cross-compilation to different architectures:
- `"native"` or `None` - Use the host machine's architecture
- `"x86_64-unknown-linux-gnu"` - Linux on `x86_64`
- `"aarch64-apple-darwin"` - macOS on Apple Silicon
- `"x86_64-windows-msvc"` - Windows on `x86_64`
- And any other LLVM-supported target triple

# Optimization Levels

The compiler supports standard LLVM optimization levels:
- `OptimizationLevel::None` - No optimization (O0)
- `OptimizationLevel::Less` - Basic optimization (O1)
- `OptimizationLevel::Default` - Standard optimization (O2)
- `OptimizationLevel::Aggressive` - Maximum optimization (O3)
*/

pub mod array;
pub mod compiler;
pub mod prelude;
mod utils;

// Re-export main types and functions
pub use compiler::{
    CompileArgs, compile_hugr_bytes_to_bitcode, compile_hugr_bytes_to_bitcode_with_options,
    compile_hugr_bytes_to_string, compile_hugr_bytes_to_string_with_options,
    get_native_target_machine, get_opt_level, get_target_machine_from_triple,
};

// Re-export read_hugr_envelope from utils
pub use utils::read_hugr_envelope;

// Re-export inkwell's OptimizationLevel for convenience
pub use tket::hugr::llvm::inkwell::OptimizationLevel;

// Extension registry used throughout the crate

// Convenience functions
use pecos_core::errors::PecosError;
use std::path::{Path, PathBuf};

/// Configuration for HUGR compilation
#[derive(Debug, Clone, Default)]
pub struct HugrCompilerConfig {
    /// Output path for the compiled LLVM IR
    pub output_path: Option<PathBuf>,
    /// Entry point symbol (defaults to "qmain")
    pub entry: Option<String>,
    /// LLVM module name (defaults to "hugr")
    pub name: Option<String>,
    /// Save HUGR to file for debugging
    pub save_hugr: Option<PathBuf>,
    /// Target triple (defaults to native)
    pub target_triple: Option<String>,
    /// Optimization level (defaults to O2)
    pub opt_level: Option<OptimizationLevel>,
}

impl HugrCompilerConfig {
    /// Convert to `CompileArgs`
    fn to_compile_args(&self) -> CompileArgs {
        CompileArgs {
            entry: self.entry.clone(),
            name: self.name.clone().unwrap_or_else(|| "hugr".to_string()),
            save_hugr: self.save_hugr.clone(),
            target_triple: self.target_triple.clone(),
            opt_level: self.opt_level.unwrap_or(OptimizationLevel::Default),
        }
    }
}

/// HUGR compiler
#[derive(Debug, Clone, Default)]
pub struct HugrCompiler {
    config: HugrCompilerConfig,
}

impl HugrCompiler {
    /// Create a new compiler with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new compiler with specified configuration
    #[must_use]
    pub fn with_config(config: HugrCompilerConfig) -> Self {
        Self { config }
    }

    /// Compile HUGR bytes to LLVM IR string
    ///
    /// # Errors
    /// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
    pub fn compile_hugr_bytes_to_string(&self, hugr_bytes: &[u8]) -> Result<String, PecosError> {
        compile_hugr_bytes_to_string_with_options(hugr_bytes, &self.config.to_compile_args())
    }

    /// Compile HUGR bytes to LLVM IR string with custom options
    ///
    /// # Errors
    /// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
    pub fn compile_hugr_bytes_to_string_with_options(
        &self,
        hugr_bytes: &[u8],
        args: &CompileArgs,
    ) -> Result<String, PecosError> {
        compile_hugr_bytes_to_string_with_options(hugr_bytes, args)
    }

    /// Compile HUGR bytes to LLVM bitcode
    ///
    /// # Errors
    /// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
    pub fn compile_hugr_bytes_to_bitcode(&self, hugr_bytes: &[u8]) -> Result<Vec<u8>, PecosError> {
        compile_hugr_bytes_to_bitcode_with_options(hugr_bytes, &self.config.to_compile_args())
    }

    /// Compile HUGR bytes to LLVM bitcode with custom options
    ///
    /// # Errors
    /// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
    pub fn compile_hugr_bytes_to_bitcode_with_options(
        &self,
        hugr_bytes: &[u8],
        args: &CompileArgs,
    ) -> Result<Vec<u8>, PecosError> {
        compile_hugr_bytes_to_bitcode_with_options(hugr_bytes, args)
    }

    /// Compile HUGR bytes to file
    ///
    /// # Errors
    /// Returns an error if HUGR compilation or file writing fails.
    pub fn compile_hugr_bytes(
        &self,
        hugr_bytes: &[u8],
        output_path: &Path,
    ) -> Result<(), PecosError> {
        let llvm_ir = self.compile_hugr_bytes_to_string(hugr_bytes)?;
        std::fs::write(output_path, llvm_ir)
            .map_err(|e| PecosError::Generic(format!("Failed to write LLVM IR: {e}")))?;
        Ok(())
    }

    /// Compile HUGR file to LLVM IR file
    ///
    /// # Errors
    /// Returns an error if HUGR file reading, compilation, or output writing fails.
    pub fn compile_hugr<P: AsRef<Path>>(&self, hugr_path: P) -> Result<PathBuf, PecosError> {
        compile_hugr_to_llvm(hugr_path, self.config.output_path.clone())
    }
}

/// Compile a HUGR file to LLVM IR file
///
/// # Arguments
/// * `hugr_path` - Path to the HUGR file
/// * `output_path` - Optional output path (defaults to input with .ll extension)
///
/// # Returns
/// Path to the generated LLVM IR file
///
/// # Errors
/// Returns `PecosError` if compilation fails
pub fn compile_hugr_to_llvm<P: AsRef<Path>>(
    hugr_path: P,
    output_path: Option<PathBuf>,
) -> Result<PathBuf, PecosError> {
    // Read the HUGR file
    let hugr_bytes = std::fs::read(&hugr_path)
        .map_err(|e| PecosError::Generic(format!("Failed to read HUGR file: {e}")))?;

    // Compile to LLVM IR
    let llvm_ir = compile_hugr_bytes_to_string(&hugr_bytes)?;

    // Determine output path
    let output = output_path.unwrap_or_else(|| {
        let mut path = hugr_path.as_ref().to_path_buf();
        path.set_extension("ll");
        path
    });

    // Write to file
    std::fs::write(&output, llvm_ir)
        .map_err(|e| PecosError::Generic(format!("Failed to write LLVM IR: {e}")))?;

    Ok(output)
}

/// Compile a HUGR file to LLVM bitcode file
///
/// # Arguments
/// * `hugr_path` - Path to the HUGR file
/// * `output_path` - Optional output path (defaults to input with .bc extension)
///
/// # Returns
/// Path to the generated LLVM bitcode file
///
/// # Errors
/// Returns `PecosError` if compilation fails
pub fn compile_hugr_to_bitcode<P: AsRef<Path>>(
    hugr_path: P,
    output_path: Option<PathBuf>,
) -> Result<PathBuf, PecosError> {
    // Read the HUGR file
    let hugr_bytes = std::fs::read(&hugr_path)
        .map_err(|e| PecosError::Generic(format!("Failed to read HUGR file: {e}")))?;

    // Compile to LLVM bitcode
    let bitcode = compile_hugr_bytes_to_bitcode(&hugr_bytes)?;

    // Determine output path
    let output = output_path.unwrap_or_else(|| {
        let mut path = hugr_path.as_ref().to_path_buf();
        path.set_extension("bc");
        path
    });

    // Write to file
    std::fs::write(&output, bitcode)
        .map_err(|e| PecosError::Generic(format!("Failed to write LLVM bitcode: {e}")))?;

    Ok(output)
}

/// Check if HUGR bytes are valid
///
/// # Errors
/// Returns an error if the HUGR is invalid
pub fn check_hugr(hugr_bytes: &[u8]) -> Result<(), PecosError> {
    read_hugr_envelope(hugr_bytes)
        .map_err(|e| PecosError::Generic(format!("Invalid HUGR: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HugrCompilerConfig::default();
        assert!(config.output_path.is_none());
        assert!(config.entry.is_none());
        assert!(config.name.is_none());
        assert!(config.save_hugr.is_none());
        assert!(config.target_triple.is_none());
        assert!(config.opt_level.is_none());
    }

    #[test]
    fn test_compiler_creation() {
        let compiler = HugrCompiler::new();
        assert!(matches!(compiler.config, HugrCompilerConfig { .. }));

        let config = HugrCompilerConfig {
            name: Some("test".to_string()),
            ..Default::default()
        };
        let compiler = HugrCompiler::with_config(config);
        assert_eq!(compiler.config.name, Some("test".to_string()));
    }
}
