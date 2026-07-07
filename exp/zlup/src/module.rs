//! Module system for Zlup.
//!
//! This module provides support for importing and organizing code across multiple files.
//!
//! ## Usage
//!
//! ```zlup
//! // Import a local file
//! utils := @import("utils.zlp");
//!
//! // Import from a subdirectory
//! qec := @import("lib/qec.zlp");
//!
//! // Access exported symbols
//! x := utils.helper_function();
//! ```
//!
//! ## Module Resolution
//!
//! Import paths are resolved relative to the importing file's directory.
//! - `"foo.zlp"` -> same directory
//! - `"lib/foo.zlp"` -> lib subdirectory
//! - `"../foo.zlp"` -> parent directory

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::ast::{Program, TopLevelDecl, TypeExpr};
use crate::parser::ParseError;

// =============================================================================
// Errors
// =============================================================================

/// Module loading errors.
#[derive(Debug, Error)]
pub enum ModuleError {
    #[error("module not found: {path}")]
    NotFound { path: String },

    #[error("failed to read module '{path}': {source}")]
    ReadError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse module '{path}': {message}")]
    ParseError { path: String, message: String },

    #[error("circular import detected: {path}")]
    CircularImport { path: String },

    #[error("invalid import path: {path}")]
    InvalidPath { path: String },
}

impl From<ParseError> for ModuleError {
    fn from(err: ParseError) -> Self {
        ModuleError::ParseError {
            path: err.location.file.unwrap_or_default(),
            message: err.message,
        }
    }
}

/// Result type for module operations.
pub type ModuleResult<T> = Result<T, ModuleError>;

// =============================================================================
// Module
// =============================================================================

/// A loaded module.
#[derive(Debug, Clone)]
pub struct Module {
    /// Absolute path to the module file.
    pub path: PathBuf,
    /// The module's AST.
    pub program: Program,
    /// Exported symbols (pub declarations).
    pub exports: BTreeMap<String, ExportedSymbol>,
}

/// An exported symbol from a module.
#[derive(Debug, Clone)]
pub enum ExportedSymbol {
    /// A function declaration with signature.
    Function {
        name: String,
        params: Vec<(String, TypeExpr)>,
        return_type: Option<TypeExpr>,
    },
    /// A constant declaration.
    Const { name: String },
    /// A type declaration (struct, enum, union).
    Type { name: String },
    /// An error set declaration (classical errors).
    ErrorSet {
        name: String,
        /// The error variant names in this set.
        variants: Vec<String>,
    },
    /// A fault set declaration (quantum faults).
    FaultSet {
        name: String,
        /// The fault variant names in this set.
        variants: Vec<String>,
    },
}

impl Module {
    /// Create a new module from an AST.
    pub fn new(path: PathBuf, program: Program) -> Self {
        let exports = Self::collect_exports(&program);
        Self {
            path,
            program,
            exports,
        }
    }

    /// Collect exported symbols from the program.
    fn collect_exports(program: &Program) -> BTreeMap<String, ExportedSymbol> {
        let mut exports = BTreeMap::new();

        for decl in &program.declarations {
            match decl {
                TopLevelDecl::Fn(fn_decl) if fn_decl.is_pub => {
                    let params: Vec<(String, TypeExpr)> = fn_decl
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.ty.clone()))
                        .collect();
                    exports.insert(
                        fn_decl.name.clone(),
                        ExportedSymbol::Function {
                            name: fn_decl.name.clone(),
                            params,
                            return_type: fn_decl.return_type.clone(),
                        },
                    );
                }
                TopLevelDecl::Binding(binding) if binding.is_pub => {
                    exports.insert(
                        binding.name.clone(),
                        ExportedSymbol::Const {
                            name: binding.name.clone(),
                        },
                    );
                }
                TopLevelDecl::Struct(struct_decl) if struct_decl.is_pub => {
                    exports.insert(
                        struct_decl.name.clone(),
                        ExportedSymbol::Type {
                            name: struct_decl.name.clone(),
                        },
                    );
                }
                TopLevelDecl::Enum(enum_decl) if enum_decl.is_pub => {
                    exports.insert(
                        enum_decl.name.clone(),
                        ExportedSymbol::Type {
                            name: enum_decl.name.clone(),
                        },
                    );
                }
                TopLevelDecl::Union(union_decl) if union_decl.is_pub => {
                    exports.insert(
                        union_decl.name.clone(),
                        ExportedSymbol::Type {
                            name: union_decl.name.clone(),
                        },
                    );
                }
                TopLevelDecl::ErrorSet(error_set) if error_set.is_pub => {
                    exports.insert(
                        error_set.name.clone(),
                        ExportedSymbol::ErrorSet {
                            name: error_set.name.clone(),
                            variants: error_set.variants.iter().map(|v| v.name.clone()).collect(),
                        },
                    );
                }
                TopLevelDecl::FaultSet(fault_set) if fault_set.is_pub => {
                    exports.insert(
                        fault_set.name.clone(),
                        ExportedSymbol::FaultSet {
                            name: fault_set.name.clone(),
                            variants: fault_set.variants.iter().map(|v| v.name.clone()).collect(),
                        },
                    );
                }
                _ => {}
            }
        }

        exports
    }

    /// Check if a symbol is exported.
    pub fn has_export(&self, name: &str) -> bool {
        self.exports.contains_key(name)
    }

    /// Get an exported symbol.
    pub fn get_export(&self, name: &str) -> Option<&ExportedSymbol> {
        self.exports.get(name)
    }
}

// =============================================================================
// Module Loader
// =============================================================================

/// Loads and caches modules.
#[derive(Debug, Default)]
pub struct ModuleLoader {
    /// Cached modules by absolute path.
    cache: BTreeMap<PathBuf, Module>,
    /// Currently loading modules (for circular import detection).
    loading: Vec<PathBuf>,
    /// Search paths for modules.
    search_paths: Vec<PathBuf>,
}

impl ModuleLoader {
    /// Create a new module loader.
    pub fn new() -> Self {
        Self {
            cache: BTreeMap::new(),
            loading: Vec::new(),
            search_paths: Vec::new(),
        }
    }

    /// Add a search path for modules.
    pub fn add_search_path(&mut self, path: impl AsRef<Path>) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Load a module from an import path.
    ///
    /// # Arguments
    /// * `import_path` - The path from the @import directive
    /// * `from_file` - The file containing the import (for relative resolution)
    pub fn load(&mut self, import_path: &str, from_file: Option<&Path>) -> ModuleResult<&Module> {
        // Resolve the import path
        let resolved_path = self.resolve_path(import_path, from_file)?;

        // Check cache
        if self.cache.contains_key(&resolved_path) {
            return Ok(self.cache.get(&resolved_path).unwrap());
        }

        // Check for circular imports
        if self.loading.contains(&resolved_path) {
            return Err(ModuleError::CircularImport {
                path: resolved_path.display().to_string(),
            });
        }

        // Mark as loading
        self.loading.push(resolved_path.clone());

        // Load and parse the file
        let source = fs::read_to_string(&resolved_path).map_err(|e| ModuleError::ReadError {
            path: resolved_path.display().to_string(),
            source: e,
        })?;

        let filename = resolved_path.display().to_string();
        let program = crate::parse_file(&source, &filename)?;

        // Create module
        let module = Module::new(resolved_path.clone(), program);

        // Remove from loading
        self.loading.retain(|p| p != &resolved_path);

        // Cache and return
        self.cache.insert(resolved_path.clone(), module);
        Ok(self.cache.get(&resolved_path).unwrap())
    }

    /// Resolve an import path to an absolute path.
    ///
    /// Resolution follows Zig-style semantics:
    /// - `@import("foo.zlp")` - looks for `foo.zlp` directly
    /// - `@import("foo")` - looks for `foo.zlp` OR `foo/foo.zlp` (directory with entry file)
    /// - `@import("std")` - special case for standard library
    fn resolve_path(&self, import_path: &str, from_file: Option<&Path>) -> ModuleResult<PathBuf> {
        // Validate import path
        if import_path.is_empty() {
            return Err(ModuleError::InvalidPath {
                path: import_path.to_string(),
            });
        }

        // Handle special imports
        if import_path == "std" {
            return self.resolve_std();
        }

        // Get the directory of the importing file
        let base_dir = if let Some(from) = from_file {
            from.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };

        // Try to find the module
        self.find_module(import_path, &base_dir)
    }

    /// Resolve the standard library path.
    fn resolve_std(&self) -> ModuleResult<PathBuf> {
        // Standard library location resolution:
        // 1. Check ZLUP_STDLIB_PATH environment variable
        // 2. Check for lib/std relative to executable
        // 3. Check search paths

        // Try environment variable first
        if let Ok(stdlib_path) = std::env::var("ZLUP_STDLIB_PATH") {
            let stdlib = PathBuf::from(&stdlib_path);

            // Try std/std.zlp (directory with entry file)
            let std_dir_entry = stdlib.join("std").join("std.zlp");
            if std_dir_entry.exists() {
                return std_dir_entry
                    .canonicalize()
                    .map_err(|e| ModuleError::ReadError {
                        path: std_dir_entry.display().to_string(),
                        source: e,
                    });
            }

            // Try std.zlp directly
            let std_file = stdlib.join("std.zlp");
            if std_file.exists() {
                return std_file.canonicalize().map_err(|e| ModuleError::ReadError {
                    path: std_file.display().to_string(),
                    source: e,
                });
            }
        }

        // Try relative to executable
        if let Ok(exe_path) = std::env::current_exe()
            && let Some(exe_dir) = exe_path.parent()
        {
            let lib_dir = exe_dir.join("lib");

            // Try lib/std/std.zlp
            let std_dir_entry = lib_dir.join("std").join("std.zlp");
            if std_dir_entry.exists() {
                return std_dir_entry
                    .canonicalize()
                    .map_err(|e| ModuleError::ReadError {
                        path: std_dir_entry.display().to_string(),
                        source: e,
                    });
            }

            // Try lib/std.zlp
            let std_file = lib_dir.join("std.zlp");
            if std_file.exists() {
                return std_file.canonicalize().map_err(|e| ModuleError::ReadError {
                    path: std_file.display().to_string(),
                    source: e,
                });
            }
        }

        // Try search paths
        for search_path in &self.search_paths {
            // Try std/std.zlp
            let std_dir_entry = search_path.join("std").join("std.zlp");
            if std_dir_entry.exists() {
                return std_dir_entry
                    .canonicalize()
                    .map_err(|e| ModuleError::ReadError {
                        path: std_dir_entry.display().to_string(),
                        source: e,
                    });
            }

            // Try std.zlp
            let std_file = search_path.join("std.zlp");
            if std_file.exists() {
                return std_file.canonicalize().map_err(|e| ModuleError::ReadError {
                    path: std_file.display().to_string(),
                    source: e,
                });
            }
        }

        Err(ModuleError::NotFound {
            path: "std (set ZLUP_STDLIB_PATH to point to stdlib directory)".to_string(),
        })
    }

    /// Find a module given an import path and base directory.
    ///
    /// Tries multiple resolution strategies:
    /// 1. Direct path (if it has .zlp extension)
    /// 2. Path with .zlp appended
    /// 3. Directory with entry file (name/name.zlp)
    fn find_module(&self, import_path: &str, base_dir: &Path) -> ModuleResult<PathBuf> {
        let has_extension = import_path.ends_with(".zlp");

        // Build list of candidates to try
        let mut candidates = Vec::new();

        // If it already has .zlp extension, try as-is first
        if has_extension {
            candidates.push(base_dir.join(import_path));
        } else {
            // Try with .zlp extension
            candidates.push(base_dir.join(format!("{}.zlp", import_path)));

            // Try as directory with entry file (Zig-style: foo -> foo/foo.zlp)
            let module_name = Path::new(import_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(import_path);
            candidates.push(
                base_dir
                    .join(import_path)
                    .join(format!("{}.zlp", module_name)),
            );
        }

        // Try candidates relative to base_dir
        for candidate in &candidates {
            if candidate.exists() {
                return candidate
                    .canonicalize()
                    .map_err(|e| ModuleError::ReadError {
                        path: candidate.display().to_string(),
                        source: e,
                    });
            }
        }

        // Try search paths with same candidate patterns
        for search_path in &self.search_paths {
            let search_candidates: Vec<PathBuf> = if has_extension {
                vec![search_path.join(import_path)]
            } else {
                let module_name = Path::new(import_path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(import_path);
                vec![
                    search_path.join(format!("{}.zlp", import_path)),
                    search_path
                        .join(import_path)
                        .join(format!("{}.zlp", module_name)),
                ]
            };

            for candidate in search_candidates {
                if candidate.exists() {
                    return candidate
                        .canonicalize()
                        .map_err(|e| ModuleError::ReadError {
                            path: candidate.display().to_string(),
                            source: e,
                        });
                }
            }
        }

        Err(ModuleError::NotFound {
            path: import_path.to_string(),
        })
    }

    /// Get a cached module.
    pub fn get(&self, path: &Path) -> Option<&Module> {
        self.cache.get(path)
    }

    /// Get all loaded modules.
    pub fn modules(&self) -> impl Iterator<Item = &Module> {
        self.cache.values()
    }

    /// Clear the module cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_module_exports() {
        let source = r#"
            pub fn helper() -> unit {}
            fn private_fn() -> unit {}
            pub VALUE: u32 = 42;
            PRIVATE: u32 = 0;
        "#;

        let program = crate::parse_file(source, "test.zlp").unwrap();
        let module = Module::new(PathBuf::from("test.zlp"), program);

        assert!(module.has_export("helper"));
        assert!(module.has_export("VALUE"));
        assert!(!module.has_export("private_fn"));
        assert!(!module.has_export("PRIVATE"));
    }

    #[test]
    fn test_module_loader_basic() {
        let temp_dir = TempDir::new().unwrap();

        // Create a module file
        let module_path = temp_dir.path().join("utils.zlp");
        let mut file = fs::File::create(&module_path).unwrap();
        writeln!(file, "pub fn helper() -> unit {{}}").unwrap();

        // Create a main file that imports it
        let main_path = temp_dir.path().join("main.zlp");
        let mut file = fs::File::create(&main_path).unwrap();
        writeln!(file, "utils := @import(\"utils.zlp\");").unwrap();
        writeln!(file, "fn main() -> unit {{}}").unwrap();

        // Load the module
        let mut loader = ModuleLoader::new();
        let module = loader.load("utils.zlp", Some(&main_path)).unwrap();

        assert!(module.has_export("helper"));
    }

    #[test]
    fn test_module_loader_not_found() {
        let mut loader = ModuleLoader::new();
        let result = loader.load("nonexistent.zlp", None);
        assert!(matches!(result, Err(ModuleError::NotFound { .. })));
    }

    #[test]
    fn test_circular_import_detection() {
        let temp_dir = TempDir::new().unwrap();

        // Create two files that import each other
        let a_path = temp_dir.path().join("a.zlp");
        let b_path = temp_dir.path().join("b.zlp");

        let mut file = fs::File::create(&a_path).unwrap();
        writeln!(file, "b := @import(\"b.zlp\");").unwrap();
        writeln!(file, "pub fn from_a() -> unit {{}}").unwrap();

        let mut file = fs::File::create(&b_path).unwrap();
        writeln!(file, "a := @import(\"a.zlp\");").unwrap();
        writeln!(file, "pub fn from_b() -> unit {{}}").unwrap();

        // The loader itself doesn't detect cycles during parsing
        // (that would require semantic analysis to process @import)
        // But we can test the loading mechanism
        let mut loader = ModuleLoader::new();

        // Loading a.zlp should work (doesn't process imports during parse)
        let result = loader.load("a.zlp", Some(&temp_dir.path().join("main.zlp")));
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_paths() {
        let temp_dir = TempDir::new().unwrap();
        let lib_dir = temp_dir.path().join("lib");
        fs::create_dir(&lib_dir).unwrap();

        // Create a module in lib/
        let module_path = lib_dir.join("mymod.zlp");
        let mut file = fs::File::create(&module_path).unwrap();
        writeln!(file, "pub fn mymod_fn() -> unit {{}}").unwrap();

        // Load with search path
        let mut loader = ModuleLoader::new();
        loader.add_search_path(&lib_dir);

        let module = loader.load("mymod.zlp", None).unwrap();
        assert!(module.has_export("mymod_fn"));
    }

    #[test]
    fn test_no_extension_import() {
        // @import("utils") should find utils.zlp
        let temp_dir = TempDir::new().unwrap();

        let module_path = temp_dir.path().join("utils.zlp");
        let mut file = fs::File::create(&module_path).unwrap();
        writeln!(file, "pub fn util_fn() -> unit {{}}").unwrap();

        let main_path = temp_dir.path().join("main.zlp");

        let mut loader = ModuleLoader::new();
        let module = loader.load("utils", Some(&main_path)).unwrap();

        assert!(module.has_export("util_fn"));
    }

    #[test]
    fn test_directory_style_import() {
        // @import("mylib") should find mylib/mylib.zlp (Zig-style)
        let temp_dir = TempDir::new().unwrap();

        let lib_dir = temp_dir.path().join("mylib");
        fs::create_dir(&lib_dir).unwrap();

        let module_path = lib_dir.join("mylib.zlp");
        let mut file = fs::File::create(&module_path).unwrap();
        writeln!(file, "pub fn lib_fn() -> unit {{}}").unwrap();

        let main_path = temp_dir.path().join("main.zlp");

        let mut loader = ModuleLoader::new();
        let module = loader.load("mylib", Some(&main_path)).unwrap();

        assert!(module.has_export("lib_fn"));
    }

    #[test]
    fn test_nested_directory_import() {
        // @import("qec/decoder") should find qec/decoder.zlp
        let temp_dir = TempDir::new().unwrap();

        let qec_dir = temp_dir.path().join("qec");
        fs::create_dir(&qec_dir).unwrap();

        let module_path = qec_dir.join("decoder.zlp");
        let mut file = fs::File::create(&module_path).unwrap();
        writeln!(file, "pub fn decode() -> unit {{}}").unwrap();

        let main_path = temp_dir.path().join("main.zlp");

        let mut loader = ModuleLoader::new();
        let module = loader.load("qec/decoder", Some(&main_path)).unwrap();

        assert!(module.has_export("decode"));
    }

    #[test]
    fn test_explicit_extension_preferred() {
        // @import("utils.zlp") should find utils.zlp even if utils/utils.zlp exists
        let temp_dir = TempDir::new().unwrap();

        // Create utils.zlp
        let direct_path = temp_dir.path().join("utils.zlp");
        let mut file = fs::File::create(&direct_path).unwrap();
        writeln!(file, "pub fn direct() -> unit {{}}").unwrap();

        // Create utils/utils.zlp
        let utils_dir = temp_dir.path().join("utils");
        fs::create_dir(&utils_dir).unwrap();
        let dir_path = utils_dir.join("utils.zlp");
        let mut file = fs::File::create(&dir_path).unwrap();
        writeln!(file, "pub fn from_dir() -> unit {{}}").unwrap();

        let main_path = temp_dir.path().join("main.zlp");

        let mut loader = ModuleLoader::new();

        // Explicit extension should find the direct file
        let module = loader.load("utils.zlp", Some(&main_path)).unwrap();
        assert!(module.has_export("direct"));

        // Without extension, should find direct file first (before dir)
        loader.clear_cache();
        let module = loader.load("utils", Some(&main_path)).unwrap();
        assert!(module.has_export("direct"));
    }
}
