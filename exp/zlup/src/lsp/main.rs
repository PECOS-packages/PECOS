//! Zlups - Language Server Protocol implementation for Zlup
//!
//! Provides IDE features like diagnostics, syntax highlighting, hover, and completions.

use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use zlup::ast::SourceLocation;
use zlup::semantic::SemanticAnalyzer;

mod semantic_tokens;

use semantic_tokens::{LEGEND, token_types_for_source};

/// Document state tracked by the server
#[derive(Debug, Clone)]
struct Document {
    content: String,
    /// Document version for incremental updates (reserved for future use)
    #[allow(dead_code)]
    version: i32,
}

/// The Zlups language server
struct ZlupsServer {
    client: Client,
    documents: Arc<RwLock<BTreeMap<Url, Document>>>,
}

impl ZlupsServer {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Analyze a document and publish diagnostics
    async fn analyze_and_publish(&self, uri: Url, content: &str, version: i32) {
        let diagnostics = self.get_diagnostics(content);
        self.client
            .publish_diagnostics(uri, diagnostics, Some(version))
            .await;
    }

    /// Get diagnostics for the given source code
    fn get_diagnostics(&self, source: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        // Try parsing
        let program = match zlup::parse(source) {
            Ok(p) => p,
            Err(e) => {
                // Extract location from pest error
                let error_str = e.to_string();
                let (line, col) = extract_pest_location(&error_str);
                let message = friendly_parse_error(&error_str);

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position {
                            line: line.saturating_sub(1),
                            character: col.saturating_sub(1),
                        },
                        end: Position {
                            line: line.saturating_sub(1),
                            character: col,
                        },
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: Some(NumberOrString::String("parse-error".to_string())),
                    source: Some("zlups".to_string()),
                    message,
                    ..Default::default()
                });
                return diagnostics;
            }
        };

        // Try semantic analysis
        let mut analyzer = SemanticAnalyzer::new();
        if let Err(e) = analyzer.analyze(&program) {
            let (range, message) = if let Some(loc) = e.location() {
                (location_to_range(loc), e.to_string())
            } else {
                (
                    Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 1,
                        },
                    },
                    e.to_string(),
                )
            };

            diagnostics.push(Diagnostic {
                range,
                severity: Some(DiagnosticSeverity::ERROR),
                code: Some(NumberOrString::String("semantic-error".to_string())),
                source: Some("zlups".to_string()),
                message,
                ..Default::default()
            });
        }

        diagnostics
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for ZlupsServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            legend: LEGEND.clone(),
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                            range: Some(false),
                            ..Default::default()
                        },
                    ),
                ),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                document_formatting_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "zlups".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Zlups server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let content = params.text_document.text;
        let version = params.text_document.version;

        {
            let mut docs = self.documents.write().await;
            docs.insert(
                uri.clone(),
                Document {
                    content: content.clone(),
                    version,
                },
            );
        }

        self.analyze_and_publish(uri, &content, version).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        // We use full sync, so there's exactly one change with the full content
        if let Some(change) = params.content_changes.into_iter().next() {
            let content = change.text;

            {
                let mut docs = self.documents.write().await;
                docs.insert(
                    uri.clone(),
                    Document {
                        content: content.clone(),
                        version,
                    },
                );
            }

            self.analyze_and_publish(uri, &content, version).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let mut docs = self.documents.write().await;
        docs.remove(&params.text_document.uri);
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let docs = self.documents.read().await;
        let Some(doc) = docs.get(&params.text_document.uri) else {
            return Ok(None);
        };

        let tokens = token_types_for_source(&doc.content);
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: tokens,
        })))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let docs = self.documents.read().await;
        let Some(doc) = docs.get(&params.text_document_position_params.text_document.uri) else {
            return Ok(None);
        };

        let position = params.text_document_position_params.position;
        let hover_info = get_hover_info(&doc.content, position);

        Ok(hover_info.map(|content| Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: None,
        }))
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let docs = self.documents.read().await;
        let Some(doc) = docs.get(&params.text_document_position.text_document.uri) else {
            return Ok(None);
        };

        let position = params.text_document_position.position;
        let completions = get_context_aware_completions(&doc.content, position);
        Ok(Some(CompletionResponse::Array(completions)))
    }

    async fn formatting(
        &self,
        params: DocumentFormattingParams,
    ) -> Result<Option<Vec<TextEdit>>> {
        let docs = self.documents.read().await;
        let Some(doc) = docs.get(&params.text_document.uri) else {
            return Ok(None);
        };

        let formatted = format_source(&doc.content, &params.options);

        // If formatting produced changes, return a single edit replacing the whole document
        if formatted != doc.content {
            let line_count = doc.content.lines().count() as u32;
            let last_line_len = doc.content.lines().last().map(|l| l.len()).unwrap_or(0) as u32;

            Ok(Some(vec![TextEdit {
                range: Range {
                    start: Position {
                        line: 0,
                        character: 0,
                    },
                    end: Position {
                        line: line_count,
                        character: last_line_len,
                    },
                },
                new_text: formatted,
            }]))
        } else {
            Ok(None)
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri.clone();
        let position = params.text_document_position_params.position;

        let docs = self.documents.read().await;
        let Some(doc) = docs.get(&uri) else {
            return Ok(None);
        };

        // Find the word at the cursor position
        let Some(word) = get_word_at_position(&doc.content, position) else {
            return Ok(None);
        };

        // Parse and analyze to get symbol table
        let Ok(program) = zlup::parse(&doc.content) else {
            return Ok(None);
        };

        let mut analyzer = SemanticAnalyzer::new();
        // Analyze even if there are errors - we still want partial symbol info
        let _ = analyzer.analyze(&program);

        // Look up the symbol
        if let Some(symbol) = analyzer.symbols.lookup(&word)
            && let Some(loc) = &symbol.location {
                let range = location_to_range(loc);
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri,
                    range,
                })));
            }

        Ok(None)
    }
}

/// Convert a SourceLocation to an LSP Range
fn location_to_range(loc: &SourceLocation) -> Range {
    Range {
        start: Position {
            line: loc.line.saturating_sub(1),
            character: loc.column.saturating_sub(1),
        },
        end: Position {
            line: loc.end_line.saturating_sub(1),
            character: loc.end_column.saturating_sub(1),
        },
    }
}

/// Extract line and column from a pest error message
fn extract_pest_location(error_msg: &str) -> (u32, u32) {
    // Pest errors typically contain " --> line:column"
    if let Some(pos) = error_msg.find(" --> ") {
        let rest = &error_msg[pos + 5..];
        if let Some(colon) = rest.find(':') {
            let line_str = &rest[..colon];
            let col_end = rest[colon + 1..]
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len() - colon - 1);
            let col_str = &rest[colon + 1..colon + 1 + col_end];

            if let (Ok(line), Ok(col)) = (line_str.parse::<u32>(), col_str.parse::<u32>()) {
                return (line, col);
            }
        }
    }
    (1, 1)
}

/// Convert pest parse errors to user-friendly messages
fn friendly_parse_error(pest_message: &str) -> String {
    if pest_message.contains("expected identifier") {
        "expected an identifier".to_string()
    } else if pest_message.contains("expected type_expr") {
        "expected a type".to_string()
    } else if pest_message.contains("expected expr") {
        "expected an expression".to_string()
    } else if pest_message.contains("expected statement") {
        "expected a statement".to_string()
    } else if pest_message.contains("expected \"(\"") {
        "expected '('".to_string()
    } else if pest_message.contains("expected \")\"") {
        "expected ')'".to_string()
    } else if pest_message.contains("expected \"{\"") {
        "expected '{'".to_string()
    } else if pest_message.contains("expected \"}\"") {
        "expected '}'".to_string()
    } else if pest_message.contains("expected \";\"") {
        "expected ';'".to_string()
    } else if pest_message.contains("expected assign_op") {
        "unexpected token - expected assignment or operator".to_string()
    } else if pest_message.contains("expected top_level_decl") {
        "expected a function, constant, or type declaration".to_string()
    } else if pest_message.contains("expected EOI") {
        "unexpected content after end of file".to_string()
    } else {
        // Take only the first line of the error
        pest_message
            .lines()
            .next()
            .unwrap_or(pest_message)
            .to_string()
    }
}

/// Get the word at a given position in the source
fn get_word_at_position(source: &str, position: Position) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let line = lines.get(position.line as usize)?;
    let char_pos = position.character as usize;

    if char_pos > line.len() {
        return None;
    }

    // Find the word boundaries
    let start = line[..char_pos]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = line[char_pos..]
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + char_pos)
        .unwrap_or(line.len());

    if start >= end {
        return None;
    }

    Some(line[start..end].to_string())
}

/// Get hover information for a position in the source
fn get_hover_info(source: &str, position: Position) -> Option<String> {
    let word = get_word_at_position(source, position)?;

    // Return documentation for known items
    match word.as_str() {
        // Keywords
        "fn" => Some("**fn** - Function declaration\n\n```zlup\nfn name(params) -> ReturnType { ... }\n```".to_string()),
        "mut" => Some("**mut** - Mutable binding modifier\n\n```zlup\nmut x := 42;       // type inferred\nmut x: i32 = 42;   // explicit type\n```".to_string()),
        "if" => Some("**if** - Conditional expression\n\n```zlup\nif condition { ... } else { ... }\n```".to_string()),
        "for" => Some("**for** - Bounded iteration loop (NASA Power of 10)\n\n```zlup\nfor i in 0..10 { ... }\n```".to_string()),
        "return" => Some("**return** - Return from function\n\n```zlup\nreturn value;\n```".to_string()),
        "defer" => Some("**defer** - Execute at scope exit\n\n```zlup\ndefer resource.close();\n```".to_string()),
        "errdefer" => Some("**errdefer** - Execute at scope exit on error\n\n```zlup\nerrdefer cleanup();\n```".to_string()),

        // Types
        "unit" => Some("**unit** - Unit type (single value, used for functions with no meaningful return)".to_string()),
        "bool" => Some("**bool** - Boolean type (true/false)".to_string()),
        "i32" => Some("**i32** - 32-bit signed integer".to_string()),
        "i64" => Some("**i64** - 64-bit signed integer".to_string()),
        "u32" => Some("**u32** - 32-bit unsigned integer".to_string()),
        "u64" => Some("**u64** - 64-bit unsigned integer".to_string()),
        "f32" => Some("**f32** - 32-bit floating point".to_string()),
        "f64" => Some("**f64** - 64-bit floating point".to_string()),
        "usize" => Some("**usize** - Platform-dependent unsigned integer (array indices)".to_string()),
        "QubitArray" => Some("**QubitArray** - Array of qubits for quantum operations".to_string()),

        // Built-in functions
        "qalloc" => Some("**qalloc(n)** - Allocate n qubits\n\n```zlup\nvar q = qalloc(4);  // Allocate 4 qubits\n```".to_string()),
        "print" => Some("**print(...)** - Print to stdout".to_string()),

        // Quantum gates
        "h" | "H" => Some("**H** (Hadamard) - Creates superposition\n\n```zlup\nq.h(0);  // Apply H to qubit 0\n```".to_string()),
        "x" | "X" => Some("**X** (Pauli-X) - Bit flip gate\n\n```zlup\nq.x(0);  // Apply X to qubit 0\n```".to_string()),
        "y" | "Y" => Some("**Y** (Pauli-Y) - Y rotation gate\n\n```zlup\nq.y(0);  // Apply Y to qubit 0\n```".to_string()),
        "z" | "Z" => Some("**Z** (Pauli-Z) - Phase flip gate\n\n```zlup\nq.z(0);  // Apply Z to qubit 0\n```".to_string()),
        "cx" | "CX" | "cnot" | "CNOT" => Some("**CX/CNOT** - Controlled-X (CNOT) gate\n\n```zlup\nq.cx(0, 1);  // Control: 0, Target: 1\n```".to_string()),
        "cz" | "CZ" => Some("**CZ** - Controlled-Z gate\n\n```zlup\nq.cz(0, 1);  // Apply CZ between qubits 0 and 1\n```".to_string()),
        "rx" | "RX" => Some("**RX(theta)** - X-axis rotation\n\n```zlup\nq.rx(0, 3.14159);  // Rotate qubit 0 by pi\n```".to_string()),
        "ry" | "RY" => Some("**RY(theta)** - Y-axis rotation\n\n```zlup\nq.ry(0, 3.14159);  // Rotate qubit 0 by pi\n```".to_string()),
        "rz" | "RZ" => Some("**RZ(theta)** - Z-axis rotation\n\n```zlup\nq.rz(0, 3.14159);  // Rotate qubit 0 by pi\n```".to_string()),
        "t" | "T" => Some("**T** - T gate (pi/4 phase)\n\n```zlup\nt q[0];  // Apply T to qubit 0\n```".to_string()),
        "sz" | "SZ" => Some("**SZ** - S gate (pi/2 phase, sqrt of Z)\n\n```zlup\nsz q[0];  // Apply S to qubit 0\n```".to_string()),
        "mz" | "measure" => Some("**mz** - Measure qubit in Z basis\n\n```zlup\nconst result = mz(u1) q[0];  // Measure qubit 0\n```".to_string()),
        "pz" => Some("**pz** - Prepare qubits in |0⟩ state\n\n```zlup\npz q;           // Prepare all qubits\npz {q[0], q[1]}; // Prepare specific qubits\n```".to_string()),

        _ => None,
    }
}

/// Get context-aware completions based on cursor position
fn get_context_aware_completions(source: &str, position: Position) -> Vec<CompletionItem> {
    let lines: Vec<&str> = source.lines().collect();
    let Some(line) = lines.get(position.line as usize) else {
        return get_default_completions();
    };

    let char_pos = position.character as usize;
    let prefix = if char_pos <= line.len() {
        &line[..char_pos]
    } else {
        line
    };

    // Check if we're completing after a '.'
    if let Some(dot_pos) = prefix.rfind('.') {
        // Get the identifier before the dot
        let before_dot = &prefix[..dot_pos];
        let ident_start = before_dot
            .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '[' && c != ']')
            .map(|i| i + 1)
            .unwrap_or(0);
        let identifier = before_dot[ident_start..].trim();

        // Check if this looks like an allocator (contains qalloc or common allocator names)
        if is_likely_allocator(source, identifier) {
            return get_allocator_completions();
        }

        // Generic field/method completions
        return get_method_completions();
    }

    // Check if we're in a type context (after ':' or '->')
    let trimmed = prefix.trim_end();
    if trimmed.ends_with(':') || trimmed.ends_with("->") {
        return get_type_completions();
    }

    // Default: keywords, types, and functions
    get_default_completions()
}

/// Check if an identifier is likely an allocator variable
fn is_likely_allocator(source: &str, identifier: &str) -> bool {
    // Strip array indexing like q[0] -> q
    let base_ident = identifier.split('[').next().unwrap_or(identifier);

    // Check if there's a qalloc assignment for this identifier
    let pattern = format!("{} = qalloc", base_ident);
    if source.contains(&pattern) {
        return true;
    }

    // Check for .child() assignment
    let child_pattern = format!("{} = ", base_ident);
    for line in source.lines() {
        if line.contains(&child_pattern) && line.contains(".child(") {
            return true;
        }
    }

    // Common allocator variable names
    matches!(
        base_ident,
        "q" | "qubits" | "base" | "data" | "ancilla" | "alloc" | "allocator"
    )
}

/// Get completions for allocator methods
fn get_allocator_completions() -> Vec<CompletionItem> {
    let methods = [
        ("child", "Create child allocator", "child(${1:size})"),
        ("release", "Release allocated qubits", "release()"),
    ];

    methods
        .iter()
        .map(|(label, detail, snippet)| CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::METHOD),
            detail: Some(detail.to_string()),
            insert_text: Some(snippet.to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        })
        .collect()
}

/// Get generic method completions
fn get_method_completions() -> Vec<CompletionItem> {
    // Combine allocator methods with common methods
    let mut items = get_allocator_completions();

    // Add common struct methods/fields
    let common = [
        ("len", "Get length", "len()"),
        ("is_empty", "Check if empty", "is_empty()"),
    ];

    for (label, detail, snippet) in common {
        items.push(CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::METHOD),
            detail: Some(detail.to_string()),
            insert_text: Some(snippet.to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        });
    }

    items
}

/// Get type completions
fn get_type_completions() -> Vec<CompletionItem> {
    let types = [
        ("unit", "Unit type (no meaningful return value)"),
        ("bool", "Boolean type"),
        ("u1", "1-bit unsigned (measurement result)"),
        ("u8", "8-bit unsigned integer"),
        ("u16", "16-bit unsigned integer"),
        ("u32", "32-bit unsigned integer"),
        ("u64", "64-bit unsigned integer"),
        ("u128", "128-bit unsigned integer"),
        ("usize", "Platform-sized unsigned integer"),
        ("i8", "8-bit signed integer"),
        ("i16", "16-bit signed integer"),
        ("i32", "32-bit signed integer"),
        ("i64", "64-bit signed integer"),
        ("i128", "128-bit signed integer"),
        ("f32", "32-bit floating point"),
        ("f64", "64-bit floating point"),
        ("a64", "64-bit angle type"),
        ("qubit", "Qubit type"),
        ("bit", "Classical bit"),
    ];

    types
        .iter()
        .map(|(label, detail)| CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::TYPE_PARAMETER),
            detail: Some(detail.to_string()),
            ..Default::default()
        })
        .collect()
}

/// Format source code according to formatting options
fn format_source(source: &str, options: &FormattingOptions) -> String {
    let indent_str = if options.insert_spaces {
        " ".repeat(options.tab_size as usize)
    } else {
        "\t".to_string()
    };

    let mut result = String::new();
    let mut indent_level: i32 = 0;
    let mut in_string = false;
    let mut prev_char = '\0';

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

    // Remove trailing whitespace from each line and ensure single newline at end
    let result: String = result
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");

    if result.is_empty() {
        result
    } else {
        result + "\n"
    }
}

/// Format a single line (spacing around operators, etc.)
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
                    if !result.ends_with(' ') && !result.ends_with('!') && !result.ends_with('<') && !result.ends_with('>') {
                        result.push(' ');
                    }
                    result.push('=');
                } else if prev_char == '!' || prev_char == '<' || prev_char == '>' || prev_char == '=' {
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
            // Ensure space after colon in type annotations (but not ::)
            ':' => {
                result.push(':');
                if chars.peek() != Some(&':') && chars.peek() != Some(&' ') && chars.peek().is_some() {
                    result.push(' ');
                }
            }
            // Collapse multiple spaces
            ' ' => {
                if !result.ends_with(' ') {
                    result.push(' ');
                }
            }
            _ => result.push(ch),
        }

        prev_char = ch;
    }

    result
}

/// Get default completions (keywords, types, functions)
fn get_default_completions() -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Keywords
    let keywords = [
        ("fn", "Function declaration", "fn ${1:name}(${2:params}) -> ${3:unit} {\n\treturn unit;\n}"),
        ("mut", "Mutable binding", "mut ${1:name} := ${0:value};"),
        ("if", "Conditional", "if ${1:condition} {\n\t$0\n}"),
        ("else", "Else branch", "else {\n\t$0\n}"),
        ("for", "Bounded for loop", "for ${1:i} in ${2:0}..${3:n} {\n\t$0\n}"),
        ("return", "Return statement", "return ${0:value};"),
        ("defer", "Defer statement", "defer ${0:expr};"),
        ("errdefer", "Error defer", "errdefer ${0:expr};"),
        ("struct", "Struct type", "struct {\n\t${1:field}: ${2:type},\n}"),
        ("enum", "Enum type", "enum {\n\t${1:Variant},\n}"),
        ("union", "Tagged union", "union(enum) {\n\t${1:Variant}: ${2:type},\n}"),
        ("error", "Error set", "error {\n\t${1:ErrorName},\n}"),
        ("set", "Set literal", "set { ${0:elements} }"),
    ];

    for (label, detail, snippet) in keywords {
        items.push(CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some(detail.to_string()),
            insert_text: Some(snippet.to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        });
    }

    // Types
    let types = [
        "unit", "bool", "i32", "i64", "u32", "u64", "f32", "f64", "usize", "QubitArray",
    ];

    for ty in types {
        items.push(CompletionItem {
            label: ty.to_string(),
            kind: Some(CompletionItemKind::TYPE_PARAMETER),
            detail: Some("Built-in type".to_string()),
            ..Default::default()
        });
    }

    // Quantum functions
    let quantum_funcs = [
        ("qalloc", "Allocate qubits", "qalloc(${1:n})"),
        ("h", "Hadamard gate", "h(${1:qubit})"),
        ("x", "Pauli-X gate", "x(${1:qubit})"),
        ("y", "Pauli-Y gate", "y(${1:qubit})"),
        ("z", "Pauli-Z gate", "z(${1:qubit})"),
        ("cx", "CNOT gate", "cx(${1:control}, ${2:target})"),
        ("cz", "CZ gate", "cz(${1:q1}, ${2:q2})"),
        ("rx", "RX rotation", "rx(${1:qubit}, ${2:angle})"),
        ("ry", "RY rotation", "ry(${1:qubit}, ${2:angle})"),
        ("rz", "RZ rotation", "rz(${1:qubit}, ${2:angle})"),
        ("mz", "Measure Z", "mz(${1:qubit})"),
    ];

    for (label, detail, snippet) in quantum_funcs {
        items.push(CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: Some(detail.to_string()),
            insert_text: Some(snippet.to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        });
    }

    items
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(ZlupsServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
