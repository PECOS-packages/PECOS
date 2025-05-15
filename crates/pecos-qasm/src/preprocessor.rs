use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use pecos_core::errors::PecosError;

/// Preprocessor for QASM files that handles include statements
/// before parsing. This simplifies the parser by removing the need
/// to handle file I/O during parsing.
///
/// The preprocessor supports:
/// - Standard file system includes
/// - Custom include search paths
/// - Virtual includes (in-memory content)
/// - Circular dependency detection
///
/// Include files are searched in the following order:
/// 1. Custom include paths (if specified)
/// 2. Directory relative to the including file
/// 3. Current working directory
/// 4. Standard locations (./includes, etc.)
pub struct Preprocessor {
    /// Track included files to detect circular dependencies
    included_files: HashSet<PathBuf>,
    /// Virtual includes - map of filename to content
    virtual_includes: HashMap<String, String>,
    /// Custom include paths to search for include files
    custom_include_paths: Vec<PathBuf>,
}

impl Preprocessor {
    pub fn new() -> Self {
        Self {
            included_files: HashSet::new(),
            virtual_includes: HashMap::new(),
            custom_include_paths: Vec::new(),
        }
    }

    /// Add a virtual include file (name + content)
    pub fn add_virtual_include(&mut self, name: &str, content: &str) {
        self.virtual_includes
            .insert(name.to_string(), content.to_string());
    }

    /// Add multiple virtual includes at once
    pub fn add_virtual_includes(&mut self, includes: impl IntoIterator<Item = (String, String)>) {
        for (name, content) in includes {
            self.virtual_includes.insert(name, content);
        }
    }

    /// Add a custom include path to search for include files
    pub fn add_include_path<P: Into<PathBuf>>(&mut self, path: P) {
        self.custom_include_paths.push(path.into());
    }

    /// Add multiple custom include paths at once
    pub fn add_include_paths<I, P>(&mut self, paths: I)
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        for path in paths {
            self.custom_include_paths.push(path.into());
        }
    }

    /// Preprocess a QASM string, resolving all include statements
    pub fn preprocess_str(&mut self, source: &str) -> Result<String, PecosError> {
        self.preprocess_with_base(source, None)
    }

    /// Preprocess a QASM file, resolving all include statements
    pub fn preprocess_file<P: AsRef<Path>>(&mut self, path: P) -> Result<String, PecosError> {
        let path = path.as_ref();
        let canonical_path = path.canonicalize().map_err(|e| {
            PecosError::IO(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to canonicalize path {}: {}", path.display(), e),
            ))
        })?;

        // Check for circular dependencies
        if !self.included_files.insert(canonical_path.clone()) {
            return Err(PecosError::ParseSyntax {
                language: "QASM".to_string(),
                message: format!(
                    "Circular dependency detected: {} was already included",
                    path.display()
                ),
            });
        }

        let source = fs::read_to_string(path).map_err(|e| PecosError::IO(e))?;
        let base_dir = path.parent();

        self.preprocess_with_base(&source, base_dir)
    }

    /// Preprocess QASM source with an optional base directory for resolving includes
    fn preprocess_with_base(
        &mut self,
        source: &str,
        base_dir: Option<&Path>,
    ) -> Result<String, PecosError> {
        // Use a simple regex-based approach to find include statements
        let include_pattern = regex::Regex::new(r#"include\s+"([^"]+)"\s*;"#).unwrap();

        let mut result = source.to_string();

        // Keep replacing includes until there are none left
        while let Some(captures) = include_pattern.captures(&result) {
            let full_match = captures.get(0).unwrap();
            let filename = captures.get(1).unwrap().as_str();

            // Resolve the include and get its content
            let included_content = self.resolve_include(filename, base_dir)?;

            // Replace the include statement with the content
            result = result.replace(full_match.as_str(), &included_content);
        }

        Ok(result)
    }

    /// Resolve an include file, trying virtual includes first, then standard locations
    fn resolve_include(
        &mut self,
        filename: &str,
        base_dir: Option<&Path>,
    ) -> Result<String, PecosError> {
        // First check virtual includes
        if let Some(content) = self.virtual_includes.get(filename) {
            // Clone the content to avoid borrowing issues
            let content = content.clone();

            // For virtual includes, we need to check for circular dependencies differently
            let virtual_path = PathBuf::from(format!("virtual://{}", filename));
            if !self.included_files.insert(virtual_path.clone()) {
                return Err(PecosError::ParseSyntax {
                    language: "QASM".to_string(),
                    message: format!(
                        "Circular dependency detected: virtual include '{}' was already included",
                        filename
                    ),
                });
            }

            // Recursively preprocess the virtual include content
            return self.preprocess_with_base(&content, None);
        }

        // Then try file system paths
        let paths_to_try = self.get_include_paths(filename, base_dir);

        for path in paths_to_try {
            if path.exists() {
                return self.preprocess_file(path);
            }
        }

        Err(PecosError::IO(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Include file '{}' not found", filename),
        )))
    }

    /// Get the list of paths to try for an include file
    fn get_include_paths(&self, filename: &str, base_dir: Option<&Path>) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // First, try custom include paths
        for custom_path in &self.custom_include_paths {
            paths.push(custom_path.join(filename));
        }

        // Then, try relative to the base directory (if provided)
        if let Some(base) = base_dir {
            paths.push(base.join(filename));
            paths.push(base.join("includes").join(filename));
        }

        // Then try relative to current directory
        paths.push(PathBuf::from(filename));
        paths.push(PathBuf::from("includes").join(filename));

        // Finally, try some standard locations
        if let Ok(cwd) = std::env::current_dir() {
            paths.push(cwd.join("includes").join(filename));

            // If we're in a crate subdirectory, try the crate root
            if cwd.ends_with("src") || cwd.ends_with("tests") {
                if let Some(parent) = cwd.parent() {
                    paths.push(parent.join("includes").join(filename));
                }
            }
        }

        paths
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_preprocess_simple() {
        let mut preprocessor = Preprocessor::new();
        let source = r#"
            OPENQASM 2.0;
            qreg q[2];
            h q[0];
        "#;

        let result = preprocessor.preprocess_str(source).unwrap();
        assert_eq!(result.trim(), source.trim());
    }

    #[test]
    fn test_preprocess_with_include() {
        let temp_dir = TempDir::new().unwrap();
        let include_path = temp_dir.path().join("test.inc");

        fs::write(&include_path, "gate h a { u2(0,pi) a; }").unwrap();

        let source = format!(
            r#"
            OPENQASM 2.0;
            include "{}";
            qreg q[2];
            h q[0];
            "#,
            include_path.display()
        );

        let mut preprocessor = Preprocessor::new();
        let result = preprocessor.preprocess_str(&source).unwrap();

        assert!(result.contains("gate h a { u2(0,pi) a; }"));
        assert!(result.contains("qreg q[2];"));
        assert!(!result.contains("include"));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let temp_dir = TempDir::new().unwrap();
        let file1_path = temp_dir.path().join("file1.qasm");
        let file2_path = temp_dir.path().join("file2.qasm");

        // Create circular dependency
        fs::write(
            &file1_path,
            format!(r#"include "{}";"#, file2_path.display()),
        )
        .unwrap();
        fs::write(
            &file2_path,
            format!(r#"include "{}";"#, file1_path.display()),
        )
        .unwrap();

        let mut preprocessor = Preprocessor::new();
        let result = preprocessor.preprocess_file(&file1_path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Circular dependency"));
        }
    }
}
