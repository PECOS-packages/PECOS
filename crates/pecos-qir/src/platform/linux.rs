//! Linux-specific implementations for QIR compilation

use std::path::PathBuf;

/// Handle Linux-specific QIR compilation
pub struct LinuxCompiler;

impl LinuxCompiler {
    /// Get standard LLVM installation paths for Linux
    #[must_use]
    pub fn standard_llvm_paths() -> Vec<PathBuf> {
        vec![
            PathBuf::from("/usr/bin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/usr/lib/llvm/bin"),
        ]
    }

    /// Get executable name for Linux
    #[must_use]
    pub fn executable_name(tool_name: &str) -> String {
        tool_name.to_string()
    }
}
