//! Build configuration for Zlup projects.
//!
//! Zlup uses a `zlup.toml` file to configure project settings:
//!
//! ```toml
//! [package]
//! name = "my-quantum-program"
//! version = "0.1.0"
//! entry = "src/main.zlp"
//!
//! [build]
//! strict = false
//! target = "slr"
//! ```
//!
//! ## Package Section
//!
//! - `name`: Project name (required)
//! - `version`: Semantic version (required)
//! - `entry`: Entry point file, relative to config file (default: "main.zlp")
//! - `description`: Optional project description
//! - `authors`: Optional list of authors
//!
//! ## Build Section
//!
//! - `strict`: Enable strict mode / NASA Power of 10 checks (default: false)
//! - `target`: Default compilation target: "slr" or "hugr" (default: "slr")
//! - `output_dir`: Output directory for compiled files (default: "build")

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Configuration file name.
pub const CONFIG_FILE_NAME: &str = "zlup.toml";

/// Errors that can occur when loading configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file '{path}': {source}")]
    ReadError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse config file '{path}': {source}")]
    ParseError {
        path: String,
        #[source]
        source: toml::de::Error,
    },

    #[error("config file not found: searched from '{start_dir}' to filesystem root")]
    NotFound { start_dir: String },

    #[error("invalid target '{target}' in config - expected 'slr' or 'hugr'")]
    InvalidTarget { target: String },
}

/// Complete project configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Package metadata.
    pub package: PackageConfig,

    /// Build settings.
    #[serde(default)]
    pub build: BuildConfig,
}

/// Package metadata section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageConfig {
    /// Project name.
    pub name: String,

    /// Project version (semver).
    pub version: String,

    /// Entry point file, relative to config file.
    #[serde(default = "default_entry")]
    pub entry: PathBuf,

    /// Optional project description.
    pub description: Option<String>,

    /// Optional list of authors.
    #[serde(default)]
    pub authors: Vec<String>,
}

fn default_entry() -> PathBuf {
    PathBuf::from("main.zlp")
}

/// Build settings section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Enable strict mode (NASA Power of 10 checks).
    #[serde(default)]
    pub strict: bool,

    /// Default compilation target.
    #[serde(default)]
    pub target: TargetConfig,

    /// Output directory for compiled files.
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("build")
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            strict: false,
            target: TargetConfig::default(),
            output_dir: default_output_dir(),
        }
    }
}

/// Compilation target configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetConfig {
    /// SLR-AST JSON (Python/PECOS bridge).
    #[default]
    Slr,
    /// HUGR (hardware/experiments).
    Hugr,
}

impl std::fmt::Display for TargetConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetConfig::Slr => write!(f, "slr"),
            TargetConfig::Hugr => write!(f, "hugr"),
        }
    }
}

impl Config {
    /// Load configuration from a specific file path.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path).map_err(|e| ConfigError::ReadError {
            path: path.display().to_string(),
            source: e,
        })?;

        toml::from_str(&content).map_err(|e| ConfigError::ParseError {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Find and load configuration by searching upward from the given directory.
    ///
    /// Searches for `zlup.toml` starting from `start_dir` and moving up to
    /// parent directories until found or reaching the filesystem root.
    pub fn find_and_load(start_dir: &Path) -> Result<(Self, PathBuf), ConfigError> {
        let config_path = Self::find_config_file(start_dir)?;
        let config = Self::from_file(&config_path)?;
        Ok((config, config_path))
    }

    /// Find the configuration file by searching upward from the given directory.
    pub fn find_config_file(start_dir: &Path) -> Result<PathBuf, ConfigError> {
        let mut current = start_dir.to_path_buf();

        loop {
            let config_path = current.join(CONFIG_FILE_NAME);
            if config_path.exists() {
                return Ok(config_path);
            }

            if !current.pop() {
                return Err(ConfigError::NotFound {
                    start_dir: start_dir.display().to_string(),
                });
            }
        }
    }

    /// Get the project root directory (directory containing zlup.toml).
    pub fn project_root(config_path: &Path) -> PathBuf {
        config_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Get the absolute path to the entry file.
    pub fn entry_path(&self, config_path: &Path) -> PathBuf {
        let root = Self::project_root(config_path);
        root.join(&self.package.entry)
    }

    /// Get the absolute path to the output directory.
    pub fn output_path(&self, config_path: &Path) -> PathBuf {
        let root = Self::project_root(config_path);
        root.join(&self.build.output_dir)
    }

    /// Create a minimal configuration for a new project.
    pub fn new(name: &str) -> Self {
        Self {
            package: PackageConfig {
                name: name.to_string(),
                version: "0.1.0".to_string(),
                entry: default_entry(),
                description: None,
                authors: Vec::new(),
            },
            build: BuildConfig::default(),
        }
    }

    /// Serialize the configuration to TOML string.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
            [package]
            name = "test-project"
            version = "0.1.0"
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.package.name, "test-project");
        assert_eq!(config.package.version, "0.1.0");
        assert_eq!(config.package.entry, PathBuf::from("main.zlp"));
        assert!(!config.build.strict);
        assert_eq!(config.build.target, TargetConfig::Slr);
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
            [package]
            name = "quantum-app"
            version = "1.2.3"
            entry = "src/main.zlp"
            description = "A quantum application"
            authors = ["Alice", "Bob"]

            [build]
            strict = true
            target = "hugr"
            output_dir = "out"
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.package.name, "quantum-app");
        assert_eq!(config.package.version, "1.2.3");
        assert_eq!(config.package.entry, PathBuf::from("src/main.zlp"));
        assert_eq!(
            config.package.description,
            Some("A quantum application".to_string())
        );
        assert_eq!(config.package.authors, vec!["Alice", "Bob"]);
        assert!(config.build.strict);
        assert_eq!(config.build.target, TargetConfig::Hugr);
        assert_eq!(config.build.output_dir, PathBuf::from("out"));
    }

    #[test]
    fn test_target_config_serialization() {
        // Test roundtrip through full config
        let toml = r#"
            [package]
            name = "test"
            version = "0.1.0"
            [build]
            target = "slr"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.build.target, TargetConfig::Slr);

        let toml = r#"
            [package]
            name = "test"
            version = "0.1.0"
            [build]
            target = "hugr"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.build.target, TargetConfig::Hugr);
    }

    #[test]
    fn test_new_config() {
        let config = Config::new("my-project");
        assert_eq!(config.package.name, "my-project");
        assert_eq!(config.package.version, "0.1.0");
        assert_eq!(config.package.entry, PathBuf::from("main.zlp"));
        assert!(!config.build.strict);
    }

    #[test]
    fn test_config_to_toml() {
        let config = Config::new("test");
        let toml_str = config.to_toml().unwrap();
        assert!(toml_str.contains("name = \"test\""));
        assert!(toml_str.contains("version = \"0.1.0\""));
    }
}
