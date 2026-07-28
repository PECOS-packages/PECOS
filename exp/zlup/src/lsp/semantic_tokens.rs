//! Semantic token support for syntax highlighting

use once_cell::sync::Lazy;
use tower_lsp::lsp_types::*;

/// Token types used for semantic highlighting
pub static LEGEND: Lazy<SemanticTokensLegend> = Lazy::new(|| SemanticTokensLegend {
    token_types: vec![
        SemanticTokenType::KEYWORD,
        SemanticTokenType::TYPE,
        SemanticTokenType::FUNCTION,
        SemanticTokenType::VARIABLE,
        SemanticTokenType::NUMBER,
        SemanticTokenType::STRING,
        SemanticTokenType::OPERATOR,
        SemanticTokenType::COMMENT,
        SemanticTokenType::PARAMETER,
        SemanticTokenType::PROPERTY,
    ],
    token_modifiers: vec![
        SemanticTokenModifier::DECLARATION,
        SemanticTokenModifier::DEFINITION,
        SemanticTokenModifier::READONLY,
    ],
});

// Token type indices (must match LEGEND order)
const TT_KEYWORD: u32 = 0;
const TT_TYPE: u32 = 1;
const TT_FUNCTION: u32 = 2;
const TT_VARIABLE: u32 = 3;
const TT_NUMBER: u32 = 4;
const TT_STRING: u32 = 5;
const TT_OPERATOR: u32 = 6;
const TT_COMMENT: u32 = 7;
#[allow(dead_code)]
const TT_PARAMETER: u32 = 8;
#[allow(dead_code)]
const TT_PROPERTY: u32 = 9;

/// Keywords in Zlup
const KEYWORDS: &[&str] = &[
    "fn",
    "mut",
    "if",
    "else",
    "for",
    "return",
    "defer",
    "errdefer",
    "struct",
    "enum",
    "union",
    "error",
    "try",
    "catch",
    "orelse",
    "break",
    "continue",
    "comptime",
    "inline",
    "pub",
    "and",
    "or",
    "not",
    "true",
    "false",
    "null",
    "undefined",
];

/// Built-in types
const TYPES: &[&str] = &[
    "unit",
    "bool",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f32",
    "f64",
    "usize",
    "isize",
    "QubitArray",
    "Qubit",
];

/// Quantum gate names (functions)
const GATES: &[&str] = &[
    "h", "H", "x", "X", "y", "Y", "z", "Z", "s", "S", "t", "T", "cx", "CX", "cnot", "CNOT", "cz",
    "CZ", "cy", "CY", "rx", "RX", "ry", "RY", "rz", "RZ", "swap", "SWAP", "ccx", "CCX", "toffoli",
    "mz", "mx", "my", "measure", "pz", "qalloc",
];

/// Built-in functions
const BUILTINS: &[&str] = &["print", "println", "assert", "unreachable"];

/// Generate semantic tokens for source code
pub fn token_types_for_source(source: &str) -> Vec<SemanticToken> {
    let mut tokens = Vec::new();
    let mut prev_line = 0u32;
    let mut prev_char = 0u32;

    for (line_num, line) in source.lines().enumerate() {
        let line_num = line_num as u32;
        let mut char_idx = 0u32;
        let chars: Vec<char> = line.chars().collect();

        while (char_idx as usize) < chars.len() {
            let c = chars[char_idx as usize];

            // Skip whitespace
            if c.is_whitespace() {
                char_idx += 1;
                continue;
            }

            // Line comments
            if c == '/' && chars.get(char_idx as usize + 1) == Some(&'/') {
                let len = (chars.len() - char_idx as usize) as u32;
                add_token(
                    &mut tokens,
                    line_num,
                    char_idx,
                    len,
                    TT_COMMENT,
                    &mut prev_line,
                    &mut prev_char,
                );
                break; // Rest of line is comment
            }

            // String literals
            if c == '"' {
                let start = char_idx;
                char_idx += 1;
                while (char_idx as usize) < chars.len() {
                    let ch = chars[char_idx as usize];
                    if ch == '"' {
                        char_idx += 1;
                        break;
                    }
                    if ch == '\\' {
                        char_idx += 1; // Skip escape
                    }
                    char_idx += 1;
                }
                let len = char_idx - start;
                add_token(
                    &mut tokens,
                    line_num,
                    start,
                    len,
                    TT_STRING,
                    &mut prev_line,
                    &mut prev_char,
                );
                continue;
            }

            // Character literals
            if c == '\'' {
                let start = char_idx;
                char_idx += 1;
                while (char_idx as usize) < chars.len() {
                    let ch = chars[char_idx as usize];
                    if ch == '\'' {
                        char_idx += 1;
                        break;
                    }
                    if ch == '\\' {
                        char_idx += 1;
                    }
                    char_idx += 1;
                }
                let len = char_idx - start;
                add_token(
                    &mut tokens,
                    line_num,
                    start,
                    len,
                    TT_STRING,
                    &mut prev_line,
                    &mut prev_char,
                );
                continue;
            }

            // Numbers
            if c.is_ascii_digit() {
                let start = char_idx;
                while (char_idx as usize) < chars.len() {
                    let ch = chars[char_idx as usize];
                    if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' {
                        char_idx += 1;
                    } else {
                        break;
                    }
                }
                let len = char_idx - start;
                add_token(
                    &mut tokens,
                    line_num,
                    start,
                    len,
                    TT_NUMBER,
                    &mut prev_line,
                    &mut prev_char,
                );
                continue;
            }

            // Identifiers and keywords
            if c.is_alphabetic() || c == '_' {
                let start = char_idx;
                while (char_idx as usize) < chars.len() {
                    let ch = chars[char_idx as usize];
                    if ch.is_alphanumeric() || ch == '_' {
                        char_idx += 1;
                    } else {
                        break;
                    }
                }
                let len = char_idx - start;
                let word: String = chars[start as usize..char_idx as usize].iter().collect();

                let token_type = if KEYWORDS.contains(&word.as_str()) {
                    TT_KEYWORD
                } else if TYPES.contains(&word.as_str()) {
                    TT_TYPE
                } else if GATES.contains(&word.as_str()) || BUILTINS.contains(&word.as_str()) {
                    TT_FUNCTION
                } else {
                    TT_VARIABLE
                };

                add_token(
                    &mut tokens,
                    line_num,
                    start,
                    len,
                    token_type,
                    &mut prev_line,
                    &mut prev_char,
                );
                continue;
            }

            // Operators (multi-char)
            if is_operator_char(c) {
                let start = char_idx;
                while (char_idx as usize) < chars.len()
                    && is_operator_char(chars[char_idx as usize])
                {
                    char_idx += 1;
                }
                let len = char_idx - start;
                add_token(
                    &mut tokens,
                    line_num,
                    start,
                    len,
                    TT_OPERATOR,
                    &mut prev_line,
                    &mut prev_char,
                );
                continue;
            }

            // Single punctuation (skip)
            char_idx += 1;
        }
    }

    tokens
}

fn is_operator_char(c: char) -> bool {
    matches!(
        c,
        '+' | '-' | '*' | '/' | '%' | '=' | '!' | '<' | '>' | '&' | '|' | '^' | '~' | '@'
    )
}

fn add_token(
    tokens: &mut Vec<SemanticToken>,
    line: u32,
    character: u32,
    length: u32,
    token_type: u32,
    prev_line: &mut u32,
    prev_char: &mut u32,
) {
    // LSP semantic tokens use delta encoding
    let delta_line = line - *prev_line;
    let delta_start = if delta_line == 0 {
        character - *prev_char
    } else {
        character
    };

    tokens.push(SemanticToken {
        delta_line,
        delta_start,
        length,
        token_type,
        token_modifiers_bitset: 0,
    });

    *prev_line = line;
    *prev_char = character;
}
