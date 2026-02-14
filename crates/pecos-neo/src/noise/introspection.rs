// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Introspection utilities for noise models.
//!
//! This module provides tools for understanding and debugging noise configurations:
//!
//! - **Tree visualization**: See the structure of composed noise primitives
//! - **Channel listing**: View all channels in a noise model
//! - **Statistics**: Get information about noise model complexity
//!
//! # Example
//!
//! ```no_run
//! use pecos_neo::noise::prelude::*;
//!
//! let noise = seq![
//!     skip_if_leaked(),
//!     prob(0.01, when_leaked(seep(), pauli())),
//! ];
//!
//! // Print the decision tree
//! println!("{}", noise.describe_tree());
//! // Output:
//! // Seq
//! // ├─ SkipIf(Leaked)
//! // └─ Prob(0.01)
//! //    └─ When(Leaked)
//! //       ├─ then: seep
//! //       └─ else: pauli
//! ```

use std::fmt::Write;

/// Trait for types that can describe themselves as a tree structure.
pub trait DescribeTree {
    /// Get a single-line description (for leaf nodes or brief display).
    fn describe(&self) -> String;

    /// Get a multi-line tree representation.
    fn describe_tree(&self) -> String {
        self.describe_tree_indent(0)
    }

    /// Get tree representation with given indentation level.
    fn describe_tree_indent(&self, indent: usize) -> String {
        // Default: just return describe() with indentation
        format!("{}{}", "  ".repeat(indent), self.describe())
    }

    /// Get children for tree rendering (if any).
    fn children(&self) -> Vec<(&str, &dyn DescribeTree)> {
        Vec::new()
    }
}

/// Helper to format a tree node with children.
pub fn format_tree_node(
    name: &str,
    children: &[(&str, &dyn DescribeTree)],
    indent: usize,
) -> String {
    let mut result = String::new();
    let prefix = "  ".repeat(indent);

    writeln!(result, "{prefix}{name}").unwrap();

    for (i, (label, child)) in children.iter().enumerate() {
        let is_last = i == children.len() - 1;
        let connector = if is_last { "└─" } else { "├─" };
        let child_prefix = if is_last { "   " } else { "│  " };

        // Write the connector and label
        if label.is_empty() {
            write!(result, "{prefix}{connector} ").unwrap();
        } else {
            write!(result, "{prefix}{connector} {label}: ").unwrap();
        }

        // Get child's tree representation
        let child_tree = child.describe_tree_indent(0);
        let child_lines: Vec<&str> = child_tree.lines().collect();

        if child_lines.len() == 1 {
            // Single line child - put on same line
            writeln!(result, "{}", child_lines[0].trim()).unwrap();
        } else {
            // Multi-line child - put on next line with proper indentation
            writeln!(result).unwrap();
            for line in child_lines {
                writeln!(result, "{prefix}{child_prefix}{line}").unwrap();
            }
        }
    }

    result
}

/// Summary information about a noise model.
#[derive(Debug, Clone)]
pub struct NoiseModelSummary {
    /// Total number of channels.
    pub channel_count: usize,
    /// Number of channels responding to each event type.
    pub channels_by_event: ChannelsByEvent,
    /// List of channel names.
    pub channel_names: Vec<String>,
}

/// Count of channels by event type.
#[derive(Debug, Clone, Default)]
pub struct ChannelsByEvent {
    pub before_gate: usize,
    pub after_gate: usize,
    pub before_measurement: usize,
    pub after_measurement: usize,
    pub after_preparation: usize,
    pub idle: usize,
}

impl std::fmt::Display for NoiseModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Noise Model Summary")?;
        writeln!(f, "===================")?;
        writeln!(f, "Total channels: {}", self.channel_count)?;
        writeln!(f)?;
        writeln!(f, "Channels by event:")?;
        writeln!(
            f,
            "  BeforeGate:       {}",
            self.channels_by_event.before_gate
        )?;
        writeln!(
            f,
            "  AfterGate:        {}",
            self.channels_by_event.after_gate
        )?;
        writeln!(
            f,
            "  BeforeMeasurement: {}",
            self.channels_by_event.before_measurement
        )?;
        writeln!(
            f,
            "  AfterMeasurement: {}",
            self.channels_by_event.after_measurement
        )?;
        writeln!(
            f,
            "  AfterPreparation: {}",
            self.channels_by_event.after_preparation
        )?;
        writeln!(f, "  IdleTime:         {}", self.channels_by_event.idle)?;
        writeln!(f)?;
        writeln!(f, "Channel names:")?;
        for name in &self.channel_names {
            writeln!(f, "  - {name}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestNode {
        name: String,
        children: Vec<TestNode>,
    }

    impl DescribeTree for TestNode {
        fn describe(&self) -> String {
            self.name.clone()
        }

        fn describe_tree_indent(&self, indent: usize) -> String {
            if self.children.is_empty() {
                format!("{}{}", "  ".repeat(indent), self.name)
            } else {
                let children: Vec<(&str, &dyn DescribeTree)> = self
                    .children
                    .iter()
                    .map(|c| ("", c as &dyn DescribeTree))
                    .collect();
                format_tree_node(&self.name, &children, indent)
            }
        }
    }

    #[test]
    fn test_tree_formatting() {
        let tree = TestNode {
            name: "Root".to_string(),
            children: vec![
                TestNode {
                    name: "Child1".to_string(),
                    children: vec![],
                },
                TestNode {
                    name: "Child2".to_string(),
                    children: vec![TestNode {
                        name: "Grandchild".to_string(),
                        children: vec![],
                    }],
                },
            ],
        };

        let output = tree.describe_tree();
        assert!(output.contains("Root"));
        assert!(output.contains("Child1"));
        assert!(output.contains("Child2"));
        assert!(output.contains("Grandchild"));
    }
}
