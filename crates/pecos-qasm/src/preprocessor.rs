use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use std::fs;

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

impl Preprocessor {
    /// Create a new preprocessor with system includes
    pub fn new() -> Self {
        let mut preprocessor = Self {
            content: HashMap::new(),
            search_paths: vec![],
            included: HashSet::new(),
        };
        
        // Add system includes
        for (name, content) in crate::includes::get_standard_includes() {
            preprocessor.content.insert(name, content);
        }
        
        preprocessor
    }

    /// Add or override an include
    pub fn add_include(&mut self, name: &str, content: &str) {
        self.content.insert(name.to_string(), content.to_string());
    }

    /// Add multiple includes at once
    pub fn add_includes<I>(&mut self, includes: I)
    where
        I: IntoIterator<Item = (String, String)>,
    {
        for (name, content) in includes {
            self.add_include(&name, &content);
        }
    }

    /// Add a search path
    pub fn add_path<P: Into<PathBuf>>(&mut self, path: P) {
        self.search_paths.push(path.into());
    }

    /// Add multiple search paths
    pub fn add_paths<I, P>(&mut self, paths: I)
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        for path in paths {
            self.add_path(path);
        }
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
                message: format!("Circular dependency: '{}' already included", name),
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
                return fs::read_to_string(&path)
                    .map_err(|e| PecosError::ParseSyntax {
                        language: "QASM".to_string(),
                        message: format!("Cannot read '{}': {}", path.display(), e),
                    });
            }
        }
        
        // Try search paths
        for search_path in &self.search_paths {
            let path = search_path.join(name);
            if path.exists() {
                return fs::read_to_string(&path)
                    .map_err(|e| PecosError::ParseSyntax {
                        language: "QASM".to_string(),
                        message: format!("Cannot read '{}': {}", path.display(), e),
                    });
            }
        }
        
        Err(PecosError::ParseSyntax {
            language: "QASM".to_string(),
            message: format!("Include file '{}' not found", name),
        })
    }

    /// Internal processing
    fn preprocess_internal(&mut self, source: &str, base_dir: Option<&Path>) -> Result<String, PecosError> {
        let include_pattern = regex::Regex::new(r#"include\s+"([^"]+)"\s*;"#).unwrap();
        let mut result = source.to_string();

        while let Some(captures) = include_pattern.captures(&result) {
            let full_match = captures.get(0).unwrap();
            let filename = captures.get(1).unwrap().as_str();

            let content = self.get_include(filename, base_dir)?;

            // Process recursively
            let processed = if filename.ends_with(".inc") {
                let new_base = if let Some(base) = base_dir {
                    base.join(filename).parent().map(|p| p.to_path_buf())
                } else {
                    Path::new(filename).parent().map(|p| p.to_path_buf())
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

    pub fn add_include_path<P: Into<PathBuf>>(&mut self, path: P) {
        self.add_path(path);
    }

    pub fn add_include_paths<I, P>(&mut self, paths: I)
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.add_paths(paths);
    }

    pub fn add_virtual_include(&mut self, filename: &str, content: &str) {
        self.add_include(filename, content);
    }

    pub fn add_virtual_includes<I>(&mut self, includes: I)
    where
        I: IntoIterator<Item = (String, String)>,
    {
        self.add_includes(includes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_simple() {
        let mut preprocessor = Preprocessor::new();
        let source = r#"
            OPENQASM 2.0;
            qreg q[2];
            h q[0];
        "#;

        let result = preprocessor.preprocess(source).unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_preprocess_with_include() {
        let mut preprocessor = Preprocessor::new();
        preprocessor.add_include("test.inc", r#"
            gate bell a,b {
                h a;
                cx a,b;
            }
        "#);

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
        assert!(result.unwrap_err().to_string().contains("Circular dependency"));
    }
}