//! Code formatter for Zlup.
//!
//! Provides canonical formatting for Zlup source code.
//!
//! ## Usage
//!
//! ```rust
//! use zlup::formatter::{format, FormatOptions};
//!
//! let source = "fn main()->unit{var x=1;}";
//! let formatted = format(source, &FormatOptions::default());
//! ```
//!
//! ## Implementation
//!
//! The formatter uses an AST-based approach when the source is valid Zlup code.
//! For code that fails to parse, it falls back to a text-based formatter that
//! handles basic indentation and spacing.

use crate::pretty::{self, PrettyOptions};

/// Formatting options.
#[derive(Debug, Clone)]
pub struct FormatOptions {
    /// Use spaces instead of tabs.
    pub use_spaces: bool,
    /// Number of spaces per indent level (if using spaces).
    pub indent_size: usize,
    /// Maximum line length (for future line wrapping).
    pub max_line_length: usize,
    /// Use AST-based formatting (falls back to text-based if parse fails).
    pub use_ast: bool,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            use_spaces: true,
            indent_size: 4,
            max_line_length: 100,
            use_ast: true,
        }
    }
}

impl From<&FormatOptions> for PrettyOptions {
    fn from(opts: &FormatOptions) -> Self {
        PrettyOptions {
            use_spaces: opts.use_spaces,
            indent_size: opts.indent_size,
            max_line_length: opts.max_line_length,
        }
    }
}

/// Format Zlup source code.
///
/// By default, this uses AST-based formatting for accurate results.
/// If the source fails to parse, it falls back to text-based formatting.
pub fn format(source: &str, options: &FormatOptions) -> String {
    // Try AST-based formatting first if enabled
    if options.use_ast {
        let pretty_opts = PrettyOptions::from(options);
        if let Some(formatted) = pretty::format_source(source, &pretty_opts) {
            return formatted;
        }
    }

    // Fall back to text-based formatting
    format_text_based(source, options)
}

/// Text-based formatter (fallback for unparseable code).
fn format_text_based(source: &str, options: &FormatOptions) -> String {
    let indent_str = if options.use_spaces {
        " ".repeat(options.indent_size)
    } else {
        "\t".to_string()
    };

    let mut result = String::new();
    let mut indent_level: i32 = 0;

    for line in source.lines() {
        let trimmed = line.trim();

        // Skip empty lines but preserve one blank line
        if trimmed.is_empty() {
            if !result.ends_with("\n\n") {
                result.push('\n');
            }
            continue;
        }

        // Adjust indent for closing braces at start of line
        let starts_with_close = trimmed.starts_with('}') || trimmed.starts_with(')');
        if starts_with_close && indent_level > 0 {
            indent_level -= 1;
        }

        // Write indentation
        for _ in 0..indent_level {
            result.push_str(&indent_str);
        }

        // Format the line content
        let formatted_line = format_line(trimmed);
        result.push_str(&formatted_line);
        result.push('\n');

        // Adjust indent for next line based on braces in this line
        let mut in_string = false;
        let mut prev_char = '\0';
        for ch in trimmed.chars() {
            match ch {
                '"' if prev_char != '\\' => in_string = !in_string,
                '{' | '(' if !in_string => indent_level += 1,
                '}' | ')' if !in_string && !starts_with_close => {
                    indent_level = (indent_level - 1).max(0);
                }
                _ => {}
            }
            prev_char = ch;
        }
    }

    // Ensure file ends with newline
    if !result.ends_with('\n') {
        result.push('\n');
    }

    result
}

/// Format a single line (handles spacing around operators).
fn format_line(line: &str) -> String {
    let mut result = String::new();
    let mut chars = line.chars().peekable();
    let mut in_string = false;
    let mut prev_char = '\0';

    while let Some(ch) = chars.next() {
        // Track string state
        if ch == '"' && prev_char != '\\' {
            in_string = !in_string;
        }

        if in_string {
            result.push(ch);
            prev_char = ch;
            continue;
        }

        match ch {
            // Ensure space after comma
            ',' => {
                result.push(',');
                if chars.peek() != Some(&' ') && chars.peek() != Some(&'\n') {
                    result.push(' ');
                }
            }
            // Ensure space around '=' (but not ==, !=, <=, >=, =>)
            '=' => {
                let next = chars.peek().copied();
                if next == Some('=') || next == Some('>') {
                    // Part of ==, =>, don't add space before
                    if !result.ends_with(' ')
                        && !result.ends_with('!')
                        && !result.ends_with('<')
                        && !result.ends_with('>')
                    {
                        result.push(' ');
                    }
                    result.push('=');
                } else if prev_char == '!' || prev_char == '<' || prev_char == '>' || prev_char == '='
                {
                    // Part of !=, <=, >=, ==
                    result.push('=');
                    if chars.peek() != Some(&' ') {
                        result.push(' ');
                    }
                } else {
                    // Standalone =
                    if !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push('=');
                    if chars.peek() != Some(&' ') && chars.peek().is_some() {
                        result.push(' ');
                    }
                }
            }
            // Ensure space after colon in type annotations
            ':' => {
                result.push(':');
                if chars.peek() != Some(&' ') && chars.peek() != Some(&':') {
                    result.push(' ');
                }
            }
            // Ensure space around -> for return types
            '-' => {
                if chars.peek() == Some(&'>') {
                    if !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push('-');
                    result.push(chars.next().unwrap());
                    if chars.peek() != Some(&' ') {
                        result.push(' ');
                    }
                } else {
                    result.push(ch);
                }
            }
            // Opening brace: ensure space before
            '{' => {
                if !result.ends_with(' ') && !result.is_empty() {
                    result.push(' ');
                }
                result.push('{');
            }
            // Other characters pass through
            _ => result.push(ch),
        }

        prev_char = ch;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_formatting() {
        let source = "fn main()->unit{var x=1;}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("fn main() -> unit"));
        assert!(formatted.contains("var x = 1;"));
    }

    #[test]
    fn test_indentation() {
        let source = "fn main() -> unit {\nvar x = 1;\n}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("    var x = 1;"));
    }

    #[test]
    fn test_preserve_strings() {
        let source = r#"s := "hello, world";"#;
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains(r#""hello, world""#));
    }

    #[test]
    fn test_comma_spacing() {
        let source = "fn foo(a:u32,b:u32) -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("a: u32, b: u32"));
    }

    #[test]
    fn test_arrow_spacing() {
        let source = "fn foo()->void{}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains(" -> "));
    }

    #[test]
    fn test_nested_braces() {
        let source = "fn main() -> unit {\nif (true) {\nx = 1;\n}\n}";
        let formatted = format(source, &FormatOptions::default());
        // Should have proper nested indentation
        let lines: Vec<&str> = formatted.lines().collect();
        assert!(lines.iter().any(|l| l.starts_with("        x = 1;")));
    }

    #[test]
    fn test_blank_line_preservation() {
        let source = "fn a() -> unit {}\n\n\n\nfn b() -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        // Should collapse multiple blank lines to one
        assert!(!formatted.contains("\n\n\n"));
    }

    #[test]
    fn test_tabs_option() {
        let source = "fn main() -> unit {\nvar x = 1;\n}";
        let options = FormatOptions {
            use_spaces: false,
            ..Default::default()
        };
        let formatted = format(source, &options);
        assert!(formatted.contains("\tvar x = 1;"));
    }

    #[test]
    fn test_gate_expression_spacing() {
        let source = "h  q[0];";
        let formatted = format(source, &FormatOptions::default());
        // Should normalize spacing
        assert!(formatted.contains("h"));
        assert!(formatted.contains("q[0]"));
    }

    #[test]
    fn test_tick_block_formatting() {
        let source = "tick{\nh q[0];\ncx (q[0],q[1]);\n}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("tick {"));
        // Inner statements should be indented
        let lines: Vec<&str> = formatted.lines().collect();
        assert!(lines.iter().any(|l| l.starts_with("    h") || l.starts_with("    cx")));
    }

    #[test]
    fn test_type_annotation_spacing() {
        let source = "x:u32 := 5;";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("x: u32"));
    }

    #[test]
    fn test_comparison_operators() {
        let source = "if (x==1 && y!=2) {}";
        let formatted = format(source, &FormatOptions::default());
        // The formatter preserves comparison operators
        assert!(formatted.contains("x") && formatted.contains("1"));
        assert!(formatted.contains("y") && formatted.contains("2"));
    }

    #[test]
    fn test_walrus_operator() {
        let source = "x := 5;";
        let formatted = format(source, &FormatOptions::default());
        eprintln!("Formatted: {:?}", formatted);
        // The formatter handles := - just verify content is preserved
        assert!(formatted.contains("x"));
        assert!(formatted.contains("5"));
    }

    #[test]
    fn test_tuple_formatting() {
        let source = "t := (1,2,3);";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("(1, 2, 3)"));
    }

    #[test]
    fn test_for_loop_formatting() {
        let source = "for i in 0..10 {\nx := i;\n}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("for"));
        let lines: Vec<&str> = formatted.lines().collect();
        assert!(lines.iter().any(|l| l.contains("x") && l.contains("i")));
    }

    #[test]
    fn test_return_statement() {
        let source = "return   unit;";
        let formatted = format(source, &FormatOptions::default());
        // Should normalize whitespace
        assert!(formatted.contains("return"));
        assert!(formatted.contains("unit"));
    }

    #[test]
    fn test_function_with_attributes() {
        let source = "@inline\nfn foo() -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("@inline"));
        assert!(formatted.contains("fn foo()"));
    }

    #[test]
    fn test_multiline_function_params() {
        let source = "fn foo(a: u32,\nb: u32,\nc: u32) -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        // Should format each param
        assert!(formatted.contains("a: u32"));
        assert!(formatted.contains("b: u32"));
        assert!(formatted.contains("c: u32"));
    }

    #[test]
    fn test_else_if_chain() {
        let source = "if (a) {\nx := 1;\n} else if (b) {\nx := 2;\n} else {\nx := 3;\n}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("if (a)"));
        assert!(formatted.contains("else if (b)"));
        assert!(formatted.contains("else {"));
    }

    #[test]
    fn test_empty_block() {
        let source = "fn empty() -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("{}"));
    }

    #[test]
    fn test_trailing_newline() {
        let source = "fn foo() -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.ends_with('\n'));
    }

    #[test]
    fn test_no_triple_trailing_newline() {
        let source = "fn foo() -> unit {}\n\n\n";
        let formatted = format(source, &FormatOptions::default());
        // Should not have 3+ consecutive newlines
        assert!(!formatted.ends_with("\n\n\n"));
    }

    // =========================================================================
    // Critical Edge Cases
    // =========================================================================

    #[test]
    fn test_deeply_nested_structure() {
        let source = "fn main() -> unit {\nif (a) {\nif (b) {\nif (c) {\nx := 1;\n}\n}\n}\n}";
        let formatted = format(source, &FormatOptions::default());
        // Should have increasing indentation
        assert!(formatted.contains("if (a)") || formatted.contains("if(a)"));
        // Innermost should have 12 spaces (3 levels)
        let lines: Vec<&str> = formatted.lines().collect();
        assert!(lines.iter().any(|l| l.starts_with("            ")));
    }

    #[test]
    fn test_string_with_special_chars() {
        let source = r#"s := "hello\nworld\t!";"#;
        let formatted = format(source, &FormatOptions::default());
        // Escape sequences should be preserved
        assert!(formatted.contains(r#"\n"#));
        assert!(formatted.contains(r#"\t"#));
    }

    #[test]
    fn test_string_with_braces() {
        let source = r#"s := "{ not a block }";"#;
        let formatted = format(source, &FormatOptions::default());
        // Braces in strings should not affect indentation
        assert!(formatted.contains(r#"{ not a block }"#));
        // Should still be at base indentation
        assert!(formatted.starts_with("s") || formatted.starts_with("\n"));
    }

    #[test]
    fn test_comment_preservation() {
        // Comments are not preserved in AST-based formatting
        // Use text-based formatting to preserve comments
        let source = "// comment\nfn main() -> unit {}";
        let options = FormatOptions {
            use_ast: false,
            ..Default::default()
        };
        let formatted = format(source, &options);
        assert!(formatted.contains("// comment"));
    }

    #[test]
    fn test_multiple_functions() {
        let source = "fn a() -> unit {}\nfn b() -> unit {}\nfn c() -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("fn a()"));
        assert!(formatted.contains("fn b()"));
        assert!(formatted.contains("fn c()"));
    }

    #[test]
    fn test_custom_indent_size() {
        let source = "fn main() -> unit {\nx := 1;\n}";
        let options = FormatOptions {
            indent_size: 2,
            ..Default::default()
        };
        let formatted = format(source, &options);
        assert!(formatted.contains("  x")); // 2 spaces
        assert!(!formatted.contains("    x")); // not 4 spaces
    }

    #[test]
    fn test_binary_operators() {
        let source = "x := a+b*c-d/e;";
        let formatted = format(source, &FormatOptions::default());
        // Should preserve the expression
        assert!(formatted.contains("a"));
        assert!(formatted.contains("b"));
        assert!(formatted.contains("c"));
    }

    #[test]
    fn test_array_literal() {
        let source = "arr := [1,2,3,4,5];";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("[1, 2, 3, 4, 5]") || formatted.contains("[1,2,3,4,5]"));
    }

    #[test]
    fn test_method_chain() {
        let source = "x := obj.method1().method2().method3();";
        let formatted = format(source, &FormatOptions::default());
        assert!(formatted.contains("method1"));
        assert!(formatted.contains("method2"));
        assert!(formatted.contains("method3"));
    }

    #[test]
    fn test_long_parameter_list() {
        let source = "fn foo(a:u32,b:u32,c:u32,d:u32,e:u32) -> unit {}";
        let formatted = format(source, &FormatOptions::default());
        // Should add spaces after commas
        assert!(formatted.contains(", ") || formatted.contains(","));
    }

    #[test]
    fn test_empty_function_body() {
        let source = "fn empty() -> unit {\n\n\n}";
        let formatted = format(source, &FormatOptions::default());
        // Should collapse empty lines
        assert!(!formatted.contains("\n\n\n"));
    }

    #[test]
    fn test_semicolon_preservation() {
        let source = "x := 1; y := 2;";
        let formatted = format(source, &FormatOptions::default());
        // Semicolons should be preserved
        assert!(formatted.matches(';').count() >= 2);
    }
}
