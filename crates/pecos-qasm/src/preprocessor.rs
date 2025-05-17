use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use pecos_core::errors::PecosError;

/// Simple preprocessor with unified includes
pub struct Preprocessor {
    /// All includes - just name to content
    content: HashMap<String, String>,

    /// Paths to search for missing includes
    search_paths: Vec<PathBuf>,

    /// Track included files (circular dependency detection)
    included: HashSet<String>,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor {
    /// Create a new preprocessor with system includes
    #[must_use]
    pub fn new() -> Self {
        let mut preprocessor = Self {
            content: HashMap::new(),
            search_paths: vec![],
            included: HashSet::new(),
        };

        // Add system includes
        for (name, content) in crate::includes::get_standard_includes() {
            preprocessor
                .content
                .insert(name.to_string(), content.to_string());
        }

        preprocessor
    }

    /// Add or override an include
    pub fn add_include(&mut self, name: &str, content: &str) {
        self.content.insert(name.to_string(), content.to_string());
    }

    /// Add a search path
    pub fn add_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Process QASM source
    pub fn preprocess(&mut self, source: &str) -> Result<String, PecosError> {
        self.included.clear();
        self.preprocess_internal(source, None)
    }

    /// Get include content (from memory or filesystem)
    fn get_include(&mut self, name: &str, base_dir: Option<&Path>) -> Result<String, PecosError> {
        // Check circular dependency
        if !self.included.insert(name.to_string()) {
            return Err(PecosError::ParseSyntax {
                language: "QASM".to_string(),
                message: format!("Circular dependency: '{name}' already included"),
            });
        }

        // Already have it?
        if let Some(content) = self.content.get(name) {
            return Ok(content.clone());
        }

        // Try filesystem
        let content = self.load_from_file(name, base_dir)?;
        self.content.insert(name.to_string(), content.clone());
        Ok(content)
    }

    /// Load from filesystem
    fn load_from_file(&self, name: &str, base_dir: Option<&Path>) -> Result<String, PecosError> {
        // Try relative to current file first
        if let Some(base) = base_dir {
            let path = base.join(name);
            if path.exists() {
                return fs::read_to_string(&path).map_err(|e| PecosError::ParseSyntax {
                    language: "QASM".to_string(),
                    message: format!("Cannot read '{}': {}", path.display(), e),
                });
            }
        }

        // Try search paths
        for search_path in &self.search_paths {
            let path = search_path.join(name);
            if path.exists() {
                return fs::read_to_string(&path).map_err(|e| PecosError::ParseSyntax {
                    language: "QASM".to_string(),
                    message: format!("Cannot read '{}': {}", path.display(), e),
                });
            }
        }

        Err(PecosError::ParseSyntax {
            language: "QASM".to_string(),
            message: format!("Include file '{name}' not found"),
        })
    }

    /// Internal processing
    fn preprocess_internal(
        &mut self,
        source: &str,
        base_dir: Option<&Path>,
    ) -> Result<String, PecosError> {
        let include_pattern = regex::Regex::new(r#"include\s+"([^"]+)"\s*;"#).unwrap();
        let mut result = source.to_string();

        while let Some(captures) = include_pattern.captures(&result) {
            let full_match = captures.get(0).unwrap();
            let filename = captures.get(1).unwrap().as_str();

            let content = self.get_include(filename, base_dir)?;

            // Process recursively
            let processed = if Path::new(filename)
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                == Some("inc")
            {
                let new_base = if let Some(base) = base_dir {
                    base.join(filename)
                        .parent()
                        .map(std::path::Path::to_path_buf)
                } else {
                    Path::new(filename)
                        .parent()
                        .map(std::path::Path::to_path_buf)
                };
                self.preprocess_internal(&content, new_base.as_deref())?
            } else {
                content
            };

            result = result.replace(full_match.as_str(), &processed);
        }

        Ok(result)
    }

    // For compatibility while transitioning
    pub fn preprocess_str(&mut self, source: &str) -> Result<String, PecosError> {
        self.preprocess(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_simple() {
        let mut preprocessor = Preprocessor::new();
        let source = r"
            OPENQASM 2.0;
            qreg q[2];
            h q[0];
        ";

        let result = preprocessor.preprocess(source).unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_preprocess_with_include() {
        let mut preprocessor = Preprocessor::new();
        preprocessor.add_include(
            "test.inc",
            r"
            gate bell a,b {
                h a;
                cx a,b;
            }
        ",
        );

        let source = r#"
            OPENQASM 2.0;
            include "test.inc";
            qreg q[2];
            bell q[0],q[1];
        "#;

        let result = preprocessor.preprocess(source).unwrap();
        assert!(result.contains("gate bell a,b"));
        assert!(!result.contains("include"));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut preprocessor = Preprocessor::new();

        // Create circular includes
        preprocessor.add_include("a.inc", r#"include "b.inc";"#);
        preprocessor.add_include("b.inc", r#"include "a.inc";"#);

        let source = r#"include "a.inc";"#;

        let result = preprocessor.preprocess(source);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Circular dependency")
        );
    }
}
