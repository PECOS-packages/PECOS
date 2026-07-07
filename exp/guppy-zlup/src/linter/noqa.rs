//! Support for inline disable comments (noqa).
//!
//! Supports:
//! - `# noqa` - suppress all warnings on this line
//! - `# noqa: ZLUP001` - suppress specific rule
//! - `# noqa: ZLUP001, ZLUP002` - suppress multiple rules
//! - `# type: ignore` - suppress type-related warnings (ZLUP006)

use std::collections::{BTreeMap, BTreeSet};

use regex::Regex;

/// Parsed noqa directives from source code.
#[derive(Debug, Default)]
pub struct NoqaDirectives {
    /// Lines where all rules are suppressed (# noqa without specific rules).
    pub suppress_all: BTreeSet<u32>,

    /// Map from line number to set of suppressed rule IDs (uppercase).
    pub suppress_rules: BTreeMap<u32, BTreeSet<String>>,

    /// File-level suppressions (rules suppressed for entire file, uppercase).
    pub file_level: BTreeSet<String>,
}

impl NoqaDirectives {
    /// Check if a diagnostic at the given line with the given rule should be suppressed.
    pub fn is_suppressed(&self, line: u32, rule_id: &str) -> bool {
        let rule_upper = rule_id.to_uppercase();

        // Check file-level suppression
        if self.file_level.contains(&rule_upper) || self.file_level.contains("*") {
            return true;
        }

        // Check line-level suppress all
        if self.suppress_all.contains(&line) {
            return true;
        }

        // Check line-level rule suppression
        if let Some(rules) = self.suppress_rules.get(&line)
            && rules.contains(&rule_upper)
        {
            return true;
        }

        false
    }
}

/// Parse noqa directives from source code.
pub fn parse_noqa(source: &str) -> NoqaDirectives {
    let mut directives = NoqaDirectives::default();

    // Regex patterns (case insensitive for noqa rules)
    let noqa_pattern = Regex::new(r"(?i)#\s*noqa(?:\s*:\s*([A-Z0-9,\s]+))?").unwrap();
    let type_ignore_pattern = Regex::new(r"#\s*type:\s*ignore").unwrap();

    for (line_idx, line) in source.lines().enumerate() {
        let line_num = (line_idx + 1) as u32;

        // Check for # noqa comments
        if let Some(captures) = noqa_pattern.captures(line) {
            if let Some(rules_match) = captures.get(1) {
                // Specific rules: # noqa: ZLUP001, ZLUP002
                let rules: BTreeSet<String> = rules_match
                    .as_str()
                    .split(',')
                    .map(|s| s.trim().to_uppercase())
                    .filter(|s| !s.is_empty())
                    .collect();

                // Check if this is a file-level directive (first non-empty, non-comment line area)
                if line_idx < 5 && line.trim().starts_with('#') {
                    directives.file_level.extend(rules.clone());
                }

                directives
                    .suppress_rules
                    .entry(line_num)
                    .or_default()
                    .extend(rules);
            } else {
                // Suppress all: # noqa
                directives.suppress_all.insert(line_num);
            }
        }

        // Check for # type: ignore (suppresses ZLUP006)
        if type_ignore_pattern.is_match(line) {
            directives
                .suppress_rules
                .entry(line_num)
                .or_default()
                .insert("ZLUP006".to_string());
        }
    }

    directives
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noqa_all() {
        let source = "x = 1 / y  # noqa";
        let directives = parse_noqa(source);
        assert!(directives.suppress_all.contains(&1));
        assert!(directives.is_suppressed(1, "ZLUP005"));
    }

    #[test]
    fn test_noqa_specific() {
        let source = "x = 1 / y  # noqa: ZLUP005";
        let directives = parse_noqa(source);
        assert!(directives.is_suppressed(1, "ZLUP005"));
        assert!(!directives.is_suppressed(1, "ZLUP001"));
    }

    #[test]
    fn test_noqa_multiple() {
        let source = "x = 1 / y  # noqa: ZLUP005, ZLUP006";
        let directives = parse_noqa(source);
        assert!(directives.is_suppressed(1, "ZLUP005"));
        assert!(directives.is_suppressed(1, "ZLUP006"));
        assert!(!directives.is_suppressed(1, "ZLUP001"));
    }

    #[test]
    fn test_type_ignore() {
        let source = "def foo(x):  # type: ignore";
        let directives = parse_noqa(source);
        assert!(directives.is_suppressed(1, "ZLUP006"));
        assert!(!directives.is_suppressed(1, "ZLUP001"));
    }

    #[test]
    fn test_not_suppressed() {
        let source = "x = 1 / y";
        let directives = parse_noqa(source);
        assert!(!directives.is_suppressed(1, "ZLUP005"));
    }

    #[test]
    fn test_case_insensitive() {
        let source = "x = 1 / y  # noqa: zlup005";
        let directives = parse_noqa(source);
        assert!(directives.is_suppressed(1, "ZLUP005"));
    }
}
