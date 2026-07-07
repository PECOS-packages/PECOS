//! Output formatters for lint results.

use serde::Serialize;

use super::diagnostic::{Diagnostic, Severity};
use super::engine::LintResult;

/// Output format for lint results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable text output (default).
    #[default]
    Text,
    /// JSON output for machine parsing.
    Json,
    /// SARIF format for GitHub Actions integration.
    Sarif,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "sarif" => Ok(OutputFormat::Sarif),
            _ => Err(format!(
                "Unknown output format: '{}'. Valid options: text, json, sarif",
                s
            )),
        }
    }
}

impl LintResult {
    /// Format the lint result as JSON.
    pub fn to_json(&self) -> String {
        let output = JsonOutput {
            diagnostics: &self.diagnostics,
            summary: JsonSummary {
                total: self.diagnostics.len(),
                errors: self
                    .diagnostics
                    .iter()
                    .filter(|d| matches!(d.severity, Severity::Error))
                    .count(),
                warnings: self
                    .diagnostics
                    .iter()
                    .filter(|d| matches!(d.severity, Severity::Warning))
                    .count(),
                has_errors: self.has_errors,
                has_warnings: self.has_warnings,
            },
        };
        serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format the lint result as SARIF (Static Analysis Results Interchange Format).
    pub fn to_sarif(&self, tool_name: &str, tool_version: &str) -> String {
        let sarif = SarifReport::from_lint_result(self, tool_name, tool_version);
        serde_json::to_string_pretty(&sarif).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format the lint result according to the specified format.
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Text => self.to_string(),
            OutputFormat::Json => self.to_json(),
            OutputFormat::Sarif => self.to_sarif("guppy-zlup", env!("CARGO_PKG_VERSION")),
        }
    }
}

// JSON output structures

#[derive(Serialize)]
struct JsonOutput<'a> {
    diagnostics: &'a [Diagnostic],
    summary: JsonSummary,
}

#[derive(Serialize)]
struct JsonSummary {
    total: usize,
    errors: usize,
    warnings: usize,
    has_errors: bool,
    has_warnings: bool,
}

// SARIF output structures (version 2.1.0)

#[derive(Serialize)]
struct SarifReport {
    #[serde(rename = "$schema")]
    schema: &'static str,
    version: &'static str,
    runs: Vec<SarifRun>,
}

#[derive(Serialize)]
struct SarifRun {
    tool: SarifTool,
    results: Vec<SarifResult>,
}

#[derive(Serialize)]
struct SarifTool {
    driver: SarifDriver,
}

#[derive(Serialize)]
struct SarifDriver {
    name: String,
    version: String,
    #[serde(rename = "informationUri")]
    information_uri: &'static str,
    rules: Vec<SarifRule>,
}

#[derive(Serialize)]
struct SarifRule {
    id: String,
    name: String,
    #[serde(rename = "shortDescription")]
    short_description: SarifMessage,
    #[serde(rename = "defaultConfiguration")]
    default_configuration: SarifConfiguration,
}

#[derive(Serialize)]
struct SarifConfiguration {
    level: &'static str,
}

#[derive(Serialize)]
struct SarifResult {
    #[serde(rename = "ruleId")]
    rule_id: String,
    level: &'static str,
    message: SarifMessage,
    locations: Vec<SarifLocation>,
}

#[derive(Serialize)]
struct SarifMessage {
    text: String,
}

#[derive(Serialize)]
struct SarifLocation {
    #[serde(rename = "physicalLocation")]
    physical_location: SarifPhysicalLocation,
}

#[derive(Serialize)]
struct SarifPhysicalLocation {
    #[serde(rename = "artifactLocation")]
    artifact_location: SarifArtifactLocation,
    region: SarifRegion,
}

#[derive(Serialize)]
struct SarifArtifactLocation {
    uri: String,
}

#[derive(Serialize)]
struct SarifRegion {
    #[serde(rename = "startLine")]
    start_line: u32,
    #[serde(rename = "startColumn")]
    start_column: u32,
    #[serde(rename = "endLine", skip_serializing_if = "Option::is_none")]
    end_line: Option<u32>,
    #[serde(rename = "endColumn", skip_serializing_if = "Option::is_none")]
    end_column: Option<u32>,
}

impl SarifReport {
    fn from_lint_result(result: &LintResult, tool_name: &str, tool_version: &str) -> Self {
        // Collect unique rules
        let mut rules: Vec<SarifRule> = Vec::new();
        let mut seen_rules: std::collections::HashSet<String> = std::collections::HashSet::new();

        for diag in &result.diagnostics {
            if !seen_rules.contains(&diag.rule_id) {
                seen_rules.insert(diag.rule_id.clone());
                rules.push(SarifRule {
                    id: diag.rule_id.clone(),
                    name: diag.rule_id.clone(),
                    short_description: SarifMessage {
                        text: get_rule_description(&diag.rule_id),
                    },
                    default_configuration: SarifConfiguration {
                        level: severity_to_sarif_level(&diag.severity),
                    },
                });
            }
        }

        // Convert diagnostics to SARIF results
        let results: Vec<SarifResult> = result
            .diagnostics
            .iter()
            .map(|diag| SarifResult {
                rule_id: diag.rule_id.clone(),
                level: severity_to_sarif_level(&diag.severity),
                message: SarifMessage {
                    text: diag.message.clone(),
                },
                locations: vec![SarifLocation {
                    physical_location: SarifPhysicalLocation {
                        artifact_location: SarifArtifactLocation {
                            uri: diag.location.file.clone().unwrap_or_default(),
                        },
                        region: SarifRegion {
                            start_line: diag.location.line,
                            start_column: diag.location.column,
                            end_line: diag.location.end_line,
                            end_column: diag.location.end_column,
                        },
                    },
                }],
            })
            .collect();

        SarifReport {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            version: "2.1.0",
            runs: vec![SarifRun {
                tool: SarifTool {
                    driver: SarifDriver {
                        name: tool_name.to_string(),
                        version: tool_version.to_string(),
                        information_uri: "https://github.com/PECOS-packages/PECOS",
                        rules,
                    },
                },
                results,
            }],
        }
    }
}

fn severity_to_sarif_level(severity: &Severity) -> &'static str {
    match severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "note",
        Severity::Hint => "note",
    }
}

fn get_rule_description(rule_id: &str) -> String {
    match rule_id {
        "ZLUP001" => "Unbounded loops are prohibited".to_string(),
        "ZLUP002" => "Recursive function calls are prohibited".to_string(),
        "ZLUP003" => "Dynamic allocation inside loops is prohibited".to_string(),
        "ZLUP004" => "Dynamic dispatch is prohibited".to_string(),
        "ZLUP005" => "Unchecked error conditions".to_string(),
        "ZLUP006" => "Missing type annotations".to_string(),
        "ZLUP007" => "Overly complex control flow".to_string(),
        "ZLUP008" => "Excessive call depth".to_string(),
        "ZLUP009" => "Missing assertions in non-trivial functions".to_string(),
        "ZLUP010" => "Mutable global state".to_string(),
        "PARSE" => "Syntax error".to_string(),
        _ => format!("Rule {}", rule_id),
    }
}

#[cfg(test)]
mod tests {
    use super::super::diagnostic::SourceLocation;
    use super::*;

    #[test]
    fn test_output_format_parse() {
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!(
            "sarif".parse::<OutputFormat>().unwrap(),
            OutputFormat::Sarif
        );
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_json_output() {
        let mut result = LintResult::new();
        result.add(Diagnostic::error(
            "ZLUP001",
            "test error",
            SourceLocation::new(1, 1).with_file("test.py"),
        ));

        let json = result.to_json();
        assert!(json.contains("ZLUP001"));
        assert!(json.contains("test error"));
        assert!(json.contains("\"errors\": 1"));
    }

    #[test]
    fn test_sarif_output() {
        let mut result = LintResult::new();
        result.add(Diagnostic::error(
            "ZLUP001",
            "test error",
            SourceLocation::new(1, 1).with_file("test.py"),
        ));

        let sarif = result.to_sarif("guppy-zlup", "0.1.0");
        assert!(sarif.contains("ZLUP001"));
        assert!(sarif.contains("test.py"));
        assert!(sarif.contains("\"version\": \"2.1.0\""));
    }
}
