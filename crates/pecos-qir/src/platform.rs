//! Platform-specific implementations for QIR compilation
//!
//! This module contains platform-specific code for compiling QIR programs,
//! separated to improve maintainability and organization.

use std::path::PathBuf;

// Import platform-specific modules
#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "macos")]
pub mod macos;

// Re-export platform-specific implementations (for backwards compatibility)
#[cfg(target_os = "windows")]
pub use windows::*;

#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(target_os = "macos")]
pub use macos::*;

/// Get standard LLVM installation paths for the current platform
#[must_use]
pub fn standard_llvm_paths() -> Vec<PathBuf> {
    #[cfg(target_os = "windows")]
    return windows::WindowsCompiler::standard_llvm_paths();

    #[cfg(target_os = "linux")]
    return linux::LinuxCompiler::standard_llvm_paths();

    #[cfg(target_os = "macos")]
    return macos::MacOSCompiler::standard_llvm_paths();
}

/// Get platform-specific executable name
#[must_use]
pub fn executable_name(tool_name: &str) -> String {
    #[cfg(target_os = "windows")]
    return windows::WindowsCompiler::executable_name(tool_name);

    #[cfg(target_os = "linux")]
    return linux::LinuxCompiler::executable_name(tool_name);

    #[cfg(target_os = "macos")]
    return macos::MacOSCompiler::executable_name(tool_name);
}
