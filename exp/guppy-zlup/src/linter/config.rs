//! Configuration for guppy-zlup.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Linter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Enabled lint rules (all rules enabled by default)
    pub enabled_rules: Vec<String>,

    /// Disabled lint rules (takes precedence over enabled_rules)
    pub disabled_rules: Vec<String>,

    /// Maximum cyclomatic complexity for ZLUP007
    pub max_complexity: u32,

    /// Treat warnings as errors
    pub treat_warnings_as_errors: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled_rules: vec![
                "ZLUP001".to_string(),
                "ZLUP002".to_string(),
                "ZLUP003".to_string(),
                "ZLUP004".to_string(),
                "ZLUP005".to_string(),
                "ZLUP006".to_string(),
                "ZLUP007".to_string(),
                "ZLUP008".to_string(),
                "ZLUP009".to_string(),
                "ZLUP010".to_string(),
            ],
            disabled_rules: vec![],
            max_complexity: 10,
            treat_warnings_as_errors: false,
        }
    }
}

impl Config {
    /// Load configuration from a pyproject.toml file.
    pub fn from_pyproject(path: &Path) -> Result<Self, ConfigError> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        let toml: toml::Value = toml::from_str(&content).map_err(ConfigError::Toml)?;

        let tool_config = toml
            .get("tool")
            .and_then(|t| t.get("guppy-zlup"))
            .cloned()
            .unwrap_or(toml::Value::Table(toml::map::Map::new()));

        // Extract values with defaults
        let enabled_rules = tool_config
            .get("enabled_rules")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| Self::default().enabled_rules);

        let disabled_rules = tool_config
            .get("disabled_rules")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let max_complexity = tool_config
            .get("max_complexity")
            .and_then(|v| v.as_integer())
            .map(|v| v as u32)
            .unwrap_or(10);

        let treat_warnings_as_errors = tool_config
            .get("warnings_as_errors")
            .or_else(|| tool_config.get("treat_warnings_as_errors"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            enabled_rules,
            disabled_rules,
            max_complexity,
            treat_warnings_as_errors,
        })
    }

    /// Check if a rule is enabled.
    ///
    /// A rule is enabled if it's in enabled_rules AND not in disabled_rules.
    /// disabled_rules takes precedence.
    pub fn is_rule_enabled(&self, rule_id: &str) -> bool {
        if self.disabled_rules.iter().any(|r| r == rule_id) {
            return false;
        }
        self.enabled_rules.iter().any(|r| r == rule_id)
    }

    /// Add a rule to the disabled list.
    pub fn disable_rule(&mut self, rule_id: &str) {
        if !self.disabled_rules.contains(&rule_id.to_string()) {
            self.disabled_rules.push(rule_id.to_string());
        }
    }

    /// Try to find a pyproject.toml by walking up from the given file path.
    pub fn find_pyproject(start_path: &Path) -> Option<std::path::PathBuf> {
        let mut current = if start_path.is_file() {
            start_path.parent()?
        } else {
            start_path
        };

        loop {
            let candidate = current.join("pyproject.toml");
            if candidate.exists() {
                return Some(candidate);
            }
            current = current.parent()?;
        }
    }
}

/// Configuration error.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.is_rule_enabled("ZLUP001"));
        assert!(config.is_rule_enabled("ZLUP007"));
        assert_eq!(config.max_complexity, 10);
    }

    #[test]
    fn test_rule_enabled() {
        let config = Config {
            enabled_rules: vec!["ZLUP001".to_string(), "ZLUP002".to_string()],
            ..Default::default()
        };

        assert!(config.is_rule_enabled("ZLUP001"));
        assert!(config.is_rule_enabled("ZLUP002"));
        assert!(!config.is_rule_enabled("ZLUP003"));
    }

    #[test]
    fn test_disabled_rules_take_precedence() {
        let mut config = Config::default();
        // ZLUP001 is enabled by default
        assert!(config.is_rule_enabled("ZLUP001"));

        // Disable it
        config.disable_rule("ZLUP001");
        assert!(!config.is_rule_enabled("ZLUP001"));

        // Other rules still enabled
        assert!(config.is_rule_enabled("ZLUP002"));
    }

    #[test]
    fn test_disable_rule_idempotent() {
        let mut config = Config::default();
        config.disable_rule("ZLUP001");
        config.disable_rule("ZLUP001"); // Should not add duplicate
        assert_eq!(config.disabled_rules.len(), 1);
    }
}
