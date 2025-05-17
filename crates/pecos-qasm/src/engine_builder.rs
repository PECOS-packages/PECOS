//! Builder pattern for `QASMEngine`

use std::path::{Path, PathBuf};

use crate::engine::QASMEngine;
use crate::parser::{ParseConfig, QASMParser};
use pecos_core::errors::PecosError;

/// Builder for creating and configuring a `QASMEngine`
#[derive(Default)]
pub struct QASMEngineBuilder {
    /// Virtual includes to use (filename -> content)
    virtual_includes: Vec<(String, String)>,
    /// Additional search paths for include files
    include_paths: Vec<String>,
    /// When true, allows general expressions in if statements
    allow_complex_conditionals: bool,
}

impl QASMEngineBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a virtual include (filename -> content)
    #[must_use]
    pub fn with_virtual_include(mut self, filename: &str, content: &str) -> Self {
        self.virtual_includes
            .push((filename.to_string(), content.to_string()));
        self
    }

    /// Add multiple virtual includes
    #[must_use]
    pub fn with_virtual_includes(mut self, includes: &[(&str, &str)]) -> Self {
        for (filename, content) in includes {
            self.virtual_includes
                .push(((*filename).to_string(), (*content).to_string()));
        }
        self
    }

    /// Add an include search path
    #[must_use]
    pub fn with_include_path(mut self, path: &str) -> Self {
        self.include_paths.push(path.to_string());
        self
    }

    /// Add multiple include search paths
    #[must_use]
    pub fn with_include_paths(mut self, paths: &[&str]) -> Self {
        for path in paths {
            self.include_paths.push((*path).to_string());
        }
        self
    }

    /// Enable or disable complex conditionals
    #[must_use]
    pub fn allow_complex_conditionals(mut self, allow: bool) -> Self {
        self.allow_complex_conditionals = allow;
        self
    }

    /// Build a `QASMEngine` from a QASM string
    pub fn build_from_str(self, qasm: &str) -> Result<QASMEngine, PecosError> {
        // Parse with configuration
        let parse_config = ParseConfig {
            includes: self
                .virtual_includes
                .iter()
                .map(|(f, c)| (f.clone(), c.clone()))
                .collect(),
            search_paths: self.include_paths.iter().map(PathBuf::from).collect(),
            ..Default::default()
        };

        let program = QASMParser::parse_with_config(qasm, &parse_config)?;

        let mut engine = QASMEngine::default();
        engine.load_program(program);

        // Apply configuration
        if self.allow_complex_conditionals {
            engine.allow_complex_conditionals(true);
        }

        Ok(engine)
    }

    /// Build a `QASMEngine` from a file
    pub fn build_from_file(self, path: impl AsRef<Path>) -> Result<QASMEngine, PecosError> {
        let content = std::fs::read_to_string(path)?;
        self.build_from_str(&content)
    }
}
