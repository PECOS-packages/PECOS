//! Recursive descent parser for Zluppy.
//!
//! This module implements a hand-written recursive descent parser rather than
//! using the visitor pattern. This choice aligns with NASA Power of 10 principles:
//!
//! - **Explicit control flow**: No hidden dispatch mechanisms
//! - **Predictable**: Direct function calls, easy to debug
//! - **Simple**: Each grammar rule maps to a function
//!
//! The parser consumes tokens from the pest lexer and builds the AST.

use crate::ast::*;
use pest::Parser;
use pest::iterators::{Pair, Pairs};
use pest_derive::Parser;

/// Pest parser generated from grammar.
#[derive(Parser)]
#[grammar = "zluppy.pest"]
struct ZluppyParser;

/// Parser state.
pub struct ParserState<'a> {
    source: &'a str,
    file: Option<String>,
}

use thiserror::Error;

/// Parse result type.
pub type ParseResult<T> = Result<T, ParseError>;

/// Parse error.
#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct ParseError {
    pub message: String,
    pub location: SourceLocation,
}

// =============================================================================
// Input Size Limits
// =============================================================================

/// Maximum allowed source file size in bytes (10 MB).
/// This prevents DoS attacks via extremely large input files.
pub const MAX_SOURCE_SIZE: usize = 10 * 1024 * 1024;

/// Maximum allowed number of AST nodes per file.
/// This prevents memory exhaustion from deeply nested or repetitive code.
pub const MAX_AST_NODES: usize = 1_000_000;

impl<'a> ParserState<'a> {
    /// Create a new parser state.
    pub fn new(source: &'a str) -> Self {
        Self { source, file: None }
    }

    /// Set the file name for error reporting.
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }

    /// Parse the source into a program AST.
    pub fn parse(&self) -> ParseResult<Program> {
        // Check source size limit
        if self.source.len() > MAX_SOURCE_SIZE {
            return Err(ParseError {
                message: format!(
                    "source file too large: {} bytes exceeds maximum of {} bytes",
                    self.source.len(),
                    MAX_SOURCE_SIZE
                ),
                location: SourceLocation::default(),
            });
        }

        let pairs =
            ZluppyParser::parse(Rule::program, self.source).map_err(|e| self.pest_error(e))?;

        self.parse_program(pairs)
    }

    /// Convert a pest error to our error type.
    fn pest_error(&self, e: pest::error::Error<Rule>) -> ParseError {
        let (line, column, end_line, end_column) = match e.line_col {
            pest::error::LineColLocation::Pos((l, c)) => {
                (l as u32, c as u32, l as u32, c as u32 + 1)
            }
            pest::error::LineColLocation::Span((l, c), (el, ec)) => {
                (l as u32, c as u32, el as u32, ec as u32)
            }
        };
        ParseError {
            message: e.to_string(),
            location: SourceLocation {
                line,
                column,
                end_line,
                end_column,
                file: self.file.clone(),
            },
        }
    }

    /// Get source location from a pest pair.
    fn location(&self, pair: &Pair<Rule>) -> SourceLocation {
        let span = pair.as_span();
        let (line, column) = pair.line_col();
        let (end_line, end_column) = span.end_pos().line_col();
        SourceLocation {
            line: line as u32,
            column: column as u32,
            end_line: end_line as u32,
            end_column: end_column as u32,
            file: self.file.clone(),
        }
    }

    /// Create an error at the given pair's location.
    fn error(&self, pair: &Pair<Rule>, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            location: self.location(pair),
        }
    }

    /// Create an error at an approximate location (for when we don't have a pair).
    fn error_at(&self, location: SourceLocation, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            location,
        }
    }

    /// Expect exactly one inner element from a pair.
    /// Returns an error if the pair has no inner elements.
    fn expect_inner(&self, pair: Pair<'a, Rule>, context: &str) -> ParseResult<Pair<'a, Rule>> {
        let location = self.location(&pair);
        pair.into_inner().next().ok_or_else(|| {
            self.error_at(location, format!("expected inner element in {}", context))
        })
    }

    /// Expect the next element from an iterator.
    /// Returns an error if the iterator is exhausted.
    fn expect_next(
        &self,
        iter: &mut impl Iterator<Item = Pair<'a, Rule>>,
        location: &SourceLocation,
        context: &str,
    ) -> ParseResult<Pair<'a, Rule>> {
        iter.next().ok_or_else(|| {
            self.error_at(
                location.clone(),
                format!("expected {} but found end of input", context),
            )
        })
    }

    // =========================================================================
    // Program Structure
    // =========================================================================

    /// Parse: program = { SOI ~ top_level_decl* ~ EOI }
    fn parse_program(&self, pairs: Pairs<Rule>) -> ParseResult<Program> {
        let mut declarations = Vec::new();

        // The pairs iterator contains a single Rule::program pair
        // We need to descend into its children to find top_level_decl items
        for pair in pairs {
            if pair.as_rule() == Rule::program {
                // Iterate over program's children
                for inner_pair in pair.into_inner() {
                    match inner_pair.as_rule() {
                        Rule::top_level_decl => {
                            declarations.push(self.parse_top_level_decl(inner_pair)?);
                        }
                        Rule::EOI => break,
                        _ => {} // Skip SOI, whitespace, etc.
                    }
                }
            }
        }

        Ok(Program {
            name: self.file.clone().unwrap_or_else(|| "main".to_string()),
            declarations,
            location: None,
        })
    }

    /// Parse: top_level_decl = { binding_decl | fn_decl | ... }
    fn parse_top_level_decl(&self, pair: Pair<'a, Rule>) -> ParseResult<TopLevelDecl> {
        let inner = self.expect_inner(pair, "top_level_decl")?;

        match inner.as_rule() {
            Rule::binding_decl => Ok(TopLevelDecl::Binding(self.parse_binding_decl(inner)?)),
            Rule::fn_decl => Ok(TopLevelDecl::Fn(self.parse_fn_decl(inner)?)),
            Rule::extern_fn_decl => Ok(TopLevelDecl::ExternFn(self.parse_extern_fn_decl(inner)?)),
            Rule::struct_decl => Ok(TopLevelDecl::Struct(self.parse_struct_decl(inner)?)),
            Rule::enum_decl => Ok(TopLevelDecl::Enum(self.parse_enum_decl(inner)?)),
            Rule::union_decl => Ok(TopLevelDecl::Union(self.parse_union_decl(inner)?)),
            Rule::error_set_decl => Ok(TopLevelDecl::ErrorSet(self.parse_error_set_decl(inner)?)),
            Rule::fault_set_decl => Ok(TopLevelDecl::FaultSet(self.parse_fault_set_decl(inner)?)),
            Rule::test_decl => Ok(TopLevelDecl::Test(self.parse_test_decl(inner)?)),
            Rule::declare_gate_decl => Ok(TopLevelDecl::DeclareGate(
                self.parse_declare_gate_decl(inner)?,
            )),
            Rule::gate_decl => Ok(TopLevelDecl::Gate(self.parse_gate_decl(inner)?)),
            _ => Err(self.error(&inner, format!("unexpected {:?}", inner.as_rule()))),
        }
    }

    // =========================================================================
    // Declarations
    // =========================================================================

    /// Parse binding declaration.
    /// New syntax:
    ///   x := 42;           -- immutable, type inferred
    ///   x: i32 = 42;       -- immutable, type explicit
    ///   mut x := 42;       -- mutable, type inferred
    ///   mut x: i32 = 42;   -- mutable, type explicit
    /// Legacy syntax (backward compatible):
    ///   const x = 42;      -- immutable
    ///   var x = 42;        -- mutable
    fn parse_binding_decl(&self, pair: Pair<Rule>) -> ParseResult<Binding> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut is_mutable = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut ty = None;
        let mut value = None;

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::pub_keyword => {
                    is_pub = true;
                }
                Rule::mut_keyword => {
                    is_mutable = true;
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::type_expr => {
                    ty = Some(self.parse_type_expr(item)?);
                }
                Rule::expr => {
                    value = Some(self.parse_expr(item)?);
                }
                Rule::undefined_literal => {
                    value = None; // Explicit undefined
                }
                _ => {}
            }
        }

        Ok(Binding {
            name,
            ty,
            value,
            is_mutable,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse: fn_decl = { "pub"? ~ "inline"? ~ "fn" ~ identifier ~ "(" ~ param_list? ~ ")" ~ fn_error_mode? ~ return_type? ~ block }
    fn parse_fn_decl(&self, pair: Pair<Rule>) -> ParseResult<FnDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut is_inline = false;
        let mut error_mode = None;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut params = Vec::new();
        let mut return_type = None;
        let mut body = None;

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::fn_error_mode => {
                    error_mode = Some(self.parse_fn_error_mode(item)?);
                }
                Rule::identifier | Rule::member_name => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::param_list => {
                    params = self.parse_param_list(item)?;
                }
                Rule::return_type => {
                    return_type = Some(self.parse_return_type(item)?);
                }
                Rule::block => {
                    body = Some(self.parse_block(item)?);
                }
                _ => {
                    if item.as_rule() == Rule::pub_keyword {
                        is_pub = true;
                    } else if item.as_rule() == Rule::inline_keyword {
                        is_inline = true;
                    }
                }
            }
        }

        Ok(FnDecl {
            name,
            params,
            return_type,
            body: body.expect("function must have body"),
            is_pub,
            is_inline,
            error_mode,
            doc_comment,
            location,
        })
    }

    /// Parse external function declaration (FFI).
    /// Grammar: extern_fn_decl = { doc_comment* ~ link_attr? ~ pub_keyword? ~ "extern" ~ string_literal ~ "fn" ~ identifier ~ "(" ~ param_list? ~ ")" ~ return_type? ~ ";" }
    fn parse_extern_fn_decl(&self, pair: Pair<Rule>) -> ParseResult<ExternFnDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut library = None;
        let mut calling_convention = String::new();
        let mut name = String::new();
        let mut params = Vec::new();
        let mut return_type = None;

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::link_attr => {
                    // Extract library name from @link("libname")
                    for link_item in item.into_inner() {
                        if link_item.as_rule() == Rule::string_literal {
                            let s = link_item.as_str();
                            library = Some(s[1..s.len() - 1].to_string());
                        }
                    }
                }
                Rule::pub_keyword => {
                    is_pub = true;
                }
                Rule::string_literal => {
                    // Extract the calling convention from the string literal
                    let s = item.as_str();
                    // Remove quotes
                    calling_convention = s[1..s.len() - 1].to_string();
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::param_list => {
                    params = self.parse_param_list(item)?;
                }
                Rule::return_type => {
                    return_type = Some(self.parse_return_type(item)?);
                }
                _ => {}
            }
        }

        Ok(ExternFnDecl {
            name,
            library,
            calling_convention,
            params,
            return_type,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse function error mode: try or try!
    fn parse_fn_error_mode(&self, pair: Pair<'a, Rule>) -> ParseResult<TryMode> {
        let inner = self.expect_inner(pair, "fn_error_mode")?;
        match inner.as_rule() {
            Rule::try_keyword => Ok(TryMode::Collect),
            Rule::try_bang_keyword => Ok(TryMode::Propagate),
            _ => Err(self.error(&inner, "expected try or try!")),
        }
    }

    /// Parse parameter list.
    fn parse_param_list(&self, pair: Pair<Rule>) -> ParseResult<Vec<Param>> {
        let mut params = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::param {
                params.push(self.parse_param(item)?);
            }
        }
        Ok(params)
    }

    /// Parse a single parameter.
    fn parse_param(&self, pair: Pair<'a, Rule>) -> ParseResult<Param> {
        let location = Some(self.location(&pair));
        let inner_pair = self.expect_inner(pair, "param")?;

        match inner_pair.as_rule() {
            Rule::self_param => {
                // Rust-style self receiver: &self or &mut self
                let text = inner_pair.as_str();
                let is_mutable = text.contains("mut");

                // Self type is *Self (pointer to self) or *const Self
                let self_type = if is_mutable {
                    TypeExpr::Pointer(Box::new(PointerType {
                        pointee: TypeExpr::Named(TypePath {
                            segments: vec!["Self".to_string()],
                            location: location.clone(),
                        }),
                        is_const: false,
                        is_many: false,
                        sentinel: None,
                    }))
                } else {
                    TypeExpr::Pointer(Box::new(PointerType {
                        pointee: TypeExpr::Named(TypePath {
                            segments: vec!["Self".to_string()],
                            location: location.clone(),
                        }),
                        is_const: true,
                        is_many: false,
                        sentinel: None,
                    }))
                };

                Ok(Param {
                    name: "self".to_string(),
                    ty: self_type,
                    is_comptime: false,
                    location,
                })
            }
            Rule::regular_param => {
                // Regular parameter: comptime? name: type
                self.parse_regular_param(inner_pair, location)
            }
            other => {
                return Err(ParseError {
                    message: format!(
                        "unexpected rule {:?}, expected param (self_param or regular_param)",
                        other
                    ),
                    location: location.unwrap_or_default(),
                });
            }
        }
    }

    /// Parse a regular (non-self) parameter.
    fn parse_regular_param(
        &self,
        pair: Pair<Rule>,
        location: Option<SourceLocation>,
    ) -> ParseResult<Param> {
        let inner = pair.into_inner();

        let mut is_comptime = false;
        let mut name = String::new();
        let mut ty = None;

        for item in inner {
            match item.as_rule() {
                Rule::comptime_modifier => {
                    is_comptime = true;
                }
                Rule::identifier => {
                    name = item.as_str().to_string();
                }
                Rule::type_expr => {
                    ty = Some(self.parse_type_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(Param {
            name,
            ty: ty.expect("parameter must have type"),
            is_comptime,
            location,
        })
    }

    /// Parse return type.
    fn parse_return_type(&self, pair: Pair<'a, Rule>) -> ParseResult<TypeExpr> {
        let inner = self.expect_inner(pair, "return_type")?;
        self.parse_type_expr(inner)
    }

    /// Parse struct declaration.
    fn parse_struct_decl(&self, pair: Pair<Rule>) -> ParseResult<StructDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut is_packed = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        let mut associated_consts = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::struct_body => {
                    let (f, m, c) = self.parse_struct_body(item)?;
                    fields = f;
                    methods = m;
                    associated_consts = c;
                }
                _ => {
                    if item.as_rule() == Rule::pub_keyword {
                        is_pub = true;
                    } else if item.as_rule() == Rule::packed_keyword {
                        is_packed = true;
                    }
                }
            }
        }

        Ok(StructDecl {
            name,
            fields,
            methods,
            associated_consts,
            is_pub,
            is_packed,
            doc_comment,
            location,
        })
    }

    /// Parse struct body (fields, methods, and associated bindings).
    fn parse_struct_body(
        &self,
        pair: Pair<Rule>,
    ) -> ParseResult<(Vec<StructField>, Vec<FnDecl>, Vec<Binding>)> {
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        let mut associated_consts = Vec::new();

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::struct_field => {
                    fields.push(self.parse_struct_field(item)?);
                }
                Rule::fn_decl => {
                    methods.push(self.parse_fn_decl(item)?);
                }
                Rule::binding_decl => {
                    // Associated constants defined within the struct
                    associated_consts.push(self.parse_binding_decl(item)?);
                }
                _ => {}
            }
        }

        Ok((fields, methods, associated_consts))
    }

    /// Parse struct field.
    fn parse_struct_field(&self, pair: Pair<Rule>) -> ParseResult<StructField> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut doc_comment = None;
        let mut name = String::new();
        let mut ty = None;
        let mut default = None;

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::identifier | Rule::member_name => {
                    name = item.as_str().to_string();
                }
                Rule::type_expr => {
                    ty = Some(self.parse_type_expr(item)?);
                }
                Rule::expr => {
                    default = Some(self.parse_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(StructField {
            name,
            ty: ty.expect("field must have type"),
            default,
            doc_comment,
            location,
        })
    }

    /// Parse enum declaration.
    fn parse_enum_decl(&self, pair: Pair<Rule>) -> ParseResult<EnumDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut tag_type = None;
        let mut variants = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::type_expr => {
                    tag_type = Some(self.parse_type_expr(item)?);
                }
                Rule::enum_body => {
                    variants = self.parse_enum_body(item)?;
                }
                _ => {
                    if item.as_rule() == Rule::pub_keyword {
                        is_pub = true;
                    }
                }
            }
        }

        Ok(EnumDecl {
            name,
            tag_type,
            variants,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse enum body.
    fn parse_enum_body(&self, pair: Pair<Rule>) -> ParseResult<Vec<EnumVariant>> {
        let mut variants = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::enum_variant {
                variants.push(self.parse_enum_variant(item)?);
            }
        }
        Ok(variants)
    }

    /// Parse enum variant.
    fn parse_enum_variant(&self, pair: Pair<Rule>) -> ParseResult<EnumVariant> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut name = String::new();
        let mut value = None;

        for item in inner {
            match item.as_rule() {
                Rule::identifier => {
                    name = item.as_str().to_string();
                }
                Rule::expr => {
                    value = Some(self.parse_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(EnumVariant {
            name,
            value,
            location,
        })
    }

    /// Parse union declaration.
    /// Grammar: union_decl = { doc_comment* ~ "pub"? ~ identifier ~ "=" ~ "union" ~ union_tag? ~ "{" ~ union_body ~ "}" ~ ";" }
    fn parse_union_decl(&self, pair: Pair<Rule>) -> ParseResult<UnionDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut tag: Option<Option<TypeExpr>> = None;
        let mut fields = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::union_tag => {
                    tag = Some(self.parse_union_tag(item)?);
                }
                Rule::union_body => {
                    fields = self.parse_union_body(item)?;
                }
                _ => {
                    if item.as_rule() == Rule::pub_keyword {
                        is_pub = true;
                    }
                }
            }
        }

        Ok(UnionDecl {
            name,
            tag,
            fields,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse union tag.
    /// Grammar: union_tag = { "(" ~ ("enum" | type_expr) ~ ")" }
    fn parse_union_tag(&self, pair: Pair<Rule>) -> ParseResult<Option<TypeExpr>> {
        let inner = pair.into_inner().next();
        match inner {
            Some(item) if item.as_rule() == Rule::type_expr => {
                Ok(Some(self.parse_type_expr(item)?))
            }
            _ => {
                // "enum" keyword means auto-tagged
                Ok(None)
            }
        }
    }

    /// Parse union body.
    fn parse_union_body(&self, pair: Pair<Rule>) -> ParseResult<Vec<UnionField>> {
        let mut fields = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::union_field {
                fields.push(self.parse_union_field(item)?);
            }
        }
        Ok(fields)
    }

    /// Parse union field.
    /// Grammar: union_field = { identifier ~ (":" ~ type_expr)? }
    fn parse_union_field(&self, pair: Pair<Rule>) -> ParseResult<UnionField> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut name = String::new();
        let mut ty = None;

        for item in inner {
            match item.as_rule() {
                Rule::identifier => {
                    name = item.as_str().to_string();
                }
                Rule::type_expr => {
                    ty = Some(self.parse_type_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(UnionField { name, ty, location })
    }

    /// Parse error set declaration - classical/logical errors.
    /// `DecodeError := error { SyndromeAmbiguous, WeightTooHigh };`
    fn parse_error_set_decl(&self, pair: Pair<Rule>) -> ParseResult<ErrorSetDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut variants = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::error_set_body => {
                    variants = self.parse_error_set_body(item)?;
                }
                Rule::pub_keyword => {
                    is_pub = true;
                }
                _ => {}
            }
        }

        Ok(ErrorSetDecl {
            name,
            variants,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse fault set declaration - quantum/physical faults.
    /// `QuantumFault := fault { Leakage, QubitLoss, GateFailure };`
    fn parse_fault_set_decl(&self, pair: Pair<Rule>) -> ParseResult<FaultSetDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut variants = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::error_set_body => {
                    // Reuse error_set_body parsing for fault variants
                    variants = self.parse_error_set_body(item)?;
                }
                Rule::pub_keyword => {
                    is_pub = true;
                }
                _ => {}
            }
        }

        Ok(FaultSetDecl {
            name,
            variants,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse error set body.
    fn parse_error_set_body(&self, pair: Pair<Rule>) -> ParseResult<Vec<ErrorVariant>> {
        let mut variants = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::error_variant {
                variants.push(self.parse_error_variant(item)?);
            }
        }
        Ok(variants)
    }

    /// Parse a single error variant.
    /// Can be just a name or name with data type: `Leakage: struct { ... }`
    fn parse_error_variant(&self, pair: Pair<Rule>) -> ParseResult<ErrorVariant> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut name = String::new();
        let mut data_type = None;

        for item in inner {
            match item.as_rule() {
                Rule::identifier => {
                    name = item.as_str().to_string();
                }
                Rule::type_expr => {
                    data_type = Some(self.parse_type_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(ErrorVariant {
            name,
            data_type,
            location,
        })
    }

    /// Parse test declaration.
    fn parse_test_decl(&self, pair: Pair<Rule>) -> ParseResult<TestDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut name = String::new();
        let mut body = None;

        for item in inner {
            match item.as_rule() {
                Rule::string_literal => {
                    name = self.parse_string_content(item.as_str());
                }
                Rule::block => {
                    body = Some(self.parse_block(item)?);
                }
                _ => {}
            }
        }

        Ok(TestDecl {
            name,
            body: body.expect("test must have body"),
            location,
        })
    }

    /// Parse target gate declaration: `declare gate name(params)(qubits);`
    fn parse_declare_gate_decl(&self, pair: Pair<Rule>) -> ParseResult<TargetGateDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut params = Vec::new();
        let mut qubits = Vec::new();
        let mut seen_first_param_list = false;

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::pub_keyword => {
                    is_pub = true;
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::gate_param_list => {
                    params = self.parse_gate_param_list(item)?;
                    seen_first_param_list = true;
                }
                Rule::qubit_param_list => {
                    qubits = self.parse_qubit_param_list(item)?;
                }
                _ => {}
            }
        }
        // If we never saw a gate_param_list, params stays empty (no params)
        let _ = seen_first_param_list;

        Ok(TargetGateDecl {
            name,
            params,
            qubits,
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse composite gate declaration: `gate name(params)(qubits) { body }`
    fn parse_gate_decl(&self, pair: Pair<Rule>) -> ParseResult<CompositeGateDecl> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut is_pub = false;
        let mut doc_comment = None;
        let mut name = String::new();
        let mut params = Vec::new();
        let mut qubits = Vec::new();
        let mut body = None;

        for item in inner {
            match item.as_rule() {
                Rule::doc_comment => {
                    doc_comment = Some(item.as_str().trim_start_matches("///").trim().to_string());
                }
                Rule::pub_keyword => {
                    is_pub = true;
                }
                Rule::identifier => {
                    if name.is_empty() {
                        name = item.as_str().to_string();
                    }
                }
                Rule::gate_param_list => {
                    params = self.parse_gate_param_list(item)?;
                }
                Rule::qubit_param_list => {
                    qubits = self.parse_qubit_param_list(item)?;
                }
                Rule::block => {
                    body = Some(self.parse_block(item)?);
                }
                _ => {}
            }
        }

        Ok(CompositeGateDecl {
            name,
            params,
            qubits,
            body: body.expect("gate must have body"),
            is_pub,
            doc_comment,
            location,
        })
    }

    /// Parse a list of gate parameters.
    fn parse_gate_param_list(&self, pair: Pair<Rule>) -> ParseResult<Vec<GateParam>> {
        let mut params = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::gate_param {
                let location = Some(self.location(&item));
                let mut name = String::new();
                let mut ty = None;
                for inner in item.into_inner() {
                    match inner.as_rule() {
                        Rule::identifier => {
                            if name.is_empty() {
                                name = inner.as_str().to_string();
                            }
                        }
                        Rule::type_expr => {
                            ty = Some(self.parse_type_expr(inner)?);
                        }
                        _ => {}
                    }
                }
                params.push(GateParam { name, ty, location });
            }
        }
        Ok(params)
    }

    /// Parse a list of qubit parameters.
    fn parse_qubit_param_list(&self, pair: Pair<Rule>) -> ParseResult<Vec<QubitParam>> {
        let mut qubits = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::qubit_param {
                let location = Some(self.location(&item));
                let name = item.as_str().to_string();
                qubits.push(QubitParam { name, location });
            }
        }
        Ok(qubits)
    }

    // =========================================================================
    // Statements
    // =========================================================================

    /// Parse a statement.
    /// Handles both the case where pair is a `statement` wrapper rule and
    /// where pair is a specific statement type directly (e.g., from `(block | statement)` patterns).
    fn parse_statement(&self, pair: Pair<'a, Rule>) -> ParseResult<Stmt> {
        // If the pair is a `statement` rule, unwrap it to get the inner statement type
        // Otherwise, use the pair directly (for cases like `(block | statement)` in grammar)
        let inner = if pair.as_rule() == Rule::statement {
            self.expect_inner(pair, "statement")?
        } else {
            pair
        };

        match inner.as_rule() {
            Rule::binding_decl => Ok(Stmt::Binding(self.parse_binding_decl(inner)?)),
            Rule::alias_stmt => Ok(Stmt::Alias(self.parse_alias_stmt(inner)?)),
            Rule::assign_stmt => Ok(Stmt::Assign(self.parse_assign_stmt(inner)?)),
            Rule::if_stmt => Ok(Stmt::If(self.parse_if_stmt(inner)?)),
            Rule::for_stmt => Ok(Stmt::For(self.parse_for_stmt(inner)?)),
            Rule::switch_stmt => Ok(Stmt::Switch(self.parse_switch_stmt(inner)?)),
            Rule::tick_stmt => Ok(Stmt::Tick(self.parse_tick_stmt(inner)?)),
            Rule::try_block_stmt => Ok(Stmt::TryBlock(self.parse_try_block_stmt(inner)?)),
            Rule::return_stmt => Ok(Stmt::Return(self.parse_return_stmt(inner)?)),
            Rule::break_stmt => Ok(Stmt::Break(self.parse_break_stmt(inner)?)),
            Rule::continue_stmt => Ok(Stmt::Continue(self.parse_continue_stmt(inner)?)),
            Rule::defer_stmt => Ok(Stmt::Defer(self.parse_defer_stmt(inner)?)),
            Rule::errdefer_stmt => Ok(Stmt::Errdefer(self.parse_errdefer_stmt(inner)?)),
            Rule::block => Ok(Stmt::Block(self.parse_block(inner)?)),
            Rule::expr_stmt => Ok(Stmt::Expr(self.parse_expr_stmt(inner)?)),
            _ => Err(self.error(
                &inner,
                format!("unexpected statement {:?}", inner.as_rule()),
            )),
        }
    }

    /// Parse alias statement.
    /// Grammar: alias_stmt = { "alias" ~ ws ~ identifier ~ ws ~ ":" ~ ws ~ "=" ~ ws ~ expr ~ ws ~ ";" }
    fn parse_alias_stmt(&self, pair: Pair<'a, Rule>) -> ParseResult<AliasBinding> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut name = None;
        let mut source = None;

        for item in inner {
            match item.as_rule() {
                Rule::identifier => {
                    name = Some(item.as_str().to_string());
                }
                Rule::expr => {
                    source = Some(self.parse_expr(item)?);
                }
                _ => {}
            }
        }

        let name = name.ok_or_else(|| {
            self.error_at(
                location.clone().unwrap_or_default(),
                "alias requires a name",
            )
        })?;
        let source = source.ok_or_else(|| {
            self.error_at(
                location.clone().unwrap_or_default(),
                "alias requires a source expression",
            )
        })?;

        Ok(AliasBinding {
            name,
            source,
            location,
        })
    }

    /// Parse tick statement.
    /// Grammar: tick_stmt = { attribute_list? ~ ws ~ "tick" ~ ws ~ attribute_list? ~ ws ~ tick_label? ~ ws ~ tick_body }
    fn parse_tick_stmt(&self, pair: Pair<Rule>) -> ParseResult<TickStmt> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut label = None;
        let mut attrs = Vec::new();
        let mut body = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::attribute_list => {
                    // Parse all attributes and add to attrs
                    attrs.extend(self.parse_attribute_list(item)?);
                }
                Rule::tick_label => {
                    // tick_label = { string_literal | identifier }
                    let label_inner = self.expect_inner(item, "tick_label")?;
                    label = Some(match label_inner.as_rule() {
                        Rule::string_literal => self.parse_string_content(label_inner.as_str()),
                        Rule::identifier => label_inner.as_str().to_string(),
                        _ => label_inner.as_str().to_string(),
                    });
                }
                Rule::tick_body => {
                    // tick_body = { "{" ~ ws ~ (statement ~ ws)* ~ "}" }
                    for stmt_pair in item.into_inner() {
                        if stmt_pair.as_rule() == Rule::statement {
                            body.push(self.parse_statement(stmt_pair)?);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(TickStmt {
            label,
            attrs,
            body,
            location,
        })
    }

    /// Parse an attribute list.
    /// Grammar: attribute_list = { attribute ~ (ws ~ attribute)* }
    fn parse_attribute_list(&self, pair: Pair<Rule>) -> ParseResult<Vec<Attribute>> {
        let mut attrs = Vec::new();
        for attr_pair in pair.into_inner() {
            match attr_pair.as_rule() {
                Rule::attribute => {
                    attrs.push(self.parse_attribute(attr_pair)?);
                }
                Rule::attrs_block => {
                    attrs.extend(self.parse_attrs_block(attr_pair)?);
                }
                _ => {}
            }
        }
        Ok(attrs)
    }

    /// Parse a single attribute.
    /// Grammar: attribute = { "@attr" ~ ws ~ "(" ~ ws ~ identifier ~ ws ~ "," ~ ws ~ attr_value ~ ws ~ ")" }
    fn parse_attribute(&self, pair: Pair<Rule>) -> ParseResult<Attribute> {
        let location = Some(self.location(&pair));
        let mut inner = pair.into_inner();

        // First inner element is the identifier (attribute key)
        let name_pair = inner.next().expect("attribute must have a key");
        let name = name_pair.as_str().to_string();

        // Second element is the value
        let value_pair = inner.next().expect("attribute must have a value");
        let value = Some(self.parse_attr_value(value_pair)?);

        Ok(Attribute {
            name,
            value,
            location,
        })
    }

    /// Parse an attrs block.
    /// Grammar: attrs_block = { "@attrs" ~ ws ~ "(" ~ ws ~ "{" ~ ws ~ attrs_entries? ~ ws ~ "}" ~ ws ~ ")" }
    fn parse_attrs_block(&self, pair: Pair<Rule>) -> ParseResult<Vec<Attribute>> {
        let location = Some(self.location(&pair));
        let mut attrs = Vec::new();

        for item in pair.into_inner() {
            if item.as_rule() == Rule::attrs_entries {
                for entry in item.into_inner() {
                    if entry.as_rule() == Rule::attrs_entry {
                        let mut entry_inner = entry.into_inner();
                        let name = entry_inner
                            .next()
                            .expect("attrs_entry must have key")
                            .as_str()
                            .to_string();
                        let value_pair = entry_inner.next().expect("attrs_entry must have value");
                        let value = Some(self.parse_attr_value(value_pair)?);
                        attrs.push(Attribute {
                            name,
                            value,
                            location: location.clone(),
                        });
                    }
                }
            }
        }

        Ok(attrs)
    }

    /// Parse an attribute value.
    /// Grammar: attr_value = { string_literal | number_literal | bool_literal | identifier }
    fn parse_attr_value(&self, pair: Pair<'a, Rule>) -> ParseResult<AttributeValue> {
        let inner = self.expect_inner(pair, "attr_value")?;
        match inner.as_rule() {
            Rule::string_literal => {
                let s = self.parse_string_content(inner.as_str());
                Ok(AttributeValue::String(s))
            }
            Rule::number_literal => {
                // Parse the number text - handle floats vs integers
                let s = inner.as_str().replace('_', "");
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    let value: f64 = s
                        .parse()
                        .map_err(|_| self.error(&inner, "invalid float literal"))?;
                    Ok(AttributeValue::Float(value))
                } else if s.starts_with("0x") || s.starts_with("0X") {
                    let value = i64::from_str_radix(&s[2..], 16)
                        .map_err(|_| self.error(&inner, "invalid hex literal"))?;
                    Ok(AttributeValue::Int(value))
                } else if s.starts_with("0b") || s.starts_with("0B") {
                    let value = i64::from_str_radix(&s[2..], 2)
                        .map_err(|_| self.error(&inner, "invalid binary literal"))?;
                    Ok(AttributeValue::Int(value))
                } else if s.starts_with("0o") || s.starts_with("0O") {
                    let value = i64::from_str_radix(&s[2..], 8)
                        .map_err(|_| self.error(&inner, "invalid octal literal"))?;
                    Ok(AttributeValue::Int(value))
                } else {
                    let value: i64 = s
                        .parse()
                        .map_err(|_| self.error(&inner, "invalid integer literal"))?;
                    Ok(AttributeValue::Int(value))
                }
            }
            Rule::bool_literal => {
                let value = inner.as_str() == "true";
                Ok(AttributeValue::Bool(value))
            }
            Rule::identifier => {
                // Identifier as value (e.g., for enum-like values)
                Ok(AttributeValue::Ident(inner.as_str().to_string()))
            }
            _ => Err(self.error(&inner, "invalid attribute value")),
        }
    }

    /// Parse a block.
    fn parse_block(&self, pair: Pair<Rule>) -> ParseResult<Block> {
        let location = Some(self.location(&pair));
        let mut statements = Vec::new();
        let mut label = None;
        let mut trailing_expr = None;
        let mut attrs = Vec::new();

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::attribute_list => {
                    attrs.extend(self.parse_attribute_list(item)?);
                }
                Rule::label => {
                    label = Some(self.parse_label(item)?);
                }
                Rule::statement => {
                    statements.push(self.parse_statement(item)?);
                }
                Rule::trailing_expr => {
                    // Trailing expression (block's return value)
                    let inner = self.expect_inner(item, "trailing_expr")?;
                    trailing_expr = Some(Box::new(self.parse_expr(inner)?));
                }
                _ => {}
            }
        }

        Ok(Block {
            label,
            attrs,
            statements,
            trailing_expr,
            location,
        })
    }

    /// Parse a label.
    fn parse_label(&self, pair: Pair<'a, Rule>) -> ParseResult<String> {
        let ident = self.expect_inner(pair, "label")?;
        Ok(ident.as_str().to_string())
    }

    /// Parse try block statement.
    /// Grammar: try_block_stmt = { try_collect_block | try_bang_block }
    fn parse_try_block_stmt(&self, pair: Pair<'a, Rule>) -> ParseResult<TryBlockStmt> {
        let inner = self.expect_inner(pair, "try_block_stmt")?;
        match inner.as_rule() {
            Rule::try_collect_block => self.parse_try_collect_block(inner),
            Rule::try_bang_block => self.parse_try_bang_block(inner),
            _ => Err(self.error(&inner, "expected try or try! block")),
        }
    }

    /// Parse try { } block (collect all errors).
    fn parse_try_collect_block(&self, pair: Pair<Rule>) -> ParseResult<TryBlockStmt> {
        let location = Some(self.location(&pair));
        let mut body = None;
        let mut catch_clause = None;

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::block => {
                    body = Some(self.parse_block(item)?);
                }
                Rule::catch_clause => {
                    catch_clause = Some(self.parse_catch_clause(item)?);
                }
                _ => {}
            }
        }

        Ok(TryBlockStmt {
            mode: TryMode::Collect,
            body: body.expect("try block must have body"),
            catch_clause,
            location,
        })
    }

    /// Parse try! { } block (stop on first error).
    fn parse_try_bang_block(&self, pair: Pair<Rule>) -> ParseResult<TryBlockStmt> {
        let location = Some(self.location(&pair));
        let mut body = None;
        let mut catch_clause = None;

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::block => {
                    body = Some(self.parse_block(item)?);
                }
                Rule::catch_clause => {
                    catch_clause = Some(self.parse_catch_clause(item)?);
                }
                _ => {}
            }
        }

        Ok(TryBlockStmt {
            mode: TryMode::Propagate,
            body: body.expect("try! block must have body"),
            catch_clause,
            location,
        })
    }

    /// Parse catch clause: catch |err| { ... } or catch |err| expr
    fn parse_catch_clause(&self, pair: Pair<Rule>) -> ParseResult<CatchClause> {
        let location = Some(self.location(&pair));
        let mut capture = String::new();
        let mut body = None;

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::identifier => {
                    capture = item.as_str().to_string();
                }
                Rule::block => {
                    let parsed_block = self.parse_block(item)?;
                    body = Some(Expr::Block(Box::new(BlockExpr {
                        label: String::new(),
                        attrs: parsed_block.attrs,
                        statements: parsed_block.statements,
                        trailing_expr: parsed_block.trailing_expr,
                        location: None,
                    })));
                }
                Rule::expr => {
                    body = Some(self.parse_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(CatchClause {
            capture,
            body: body.expect("catch clause must have body"),
            location,
        })
    }

    /// Parse assignment statement.
    fn parse_assign_stmt(&self, pair: Pair<'a, Rule>) -> ParseResult<AssignStmt> {
        let location = Some(self.location(&pair));
        let loc = location.clone().unwrap_or_default();
        let mut inner = pair.into_inner();

        let target = self.parse_expr(self.expect_next(&mut inner, &loc, "target")?)?;
        let op = self.parse_assign_op(self.expect_next(&mut inner, &loc, "operator")?)?;
        let value = self.parse_expr(self.expect_next(&mut inner, &loc, "value")?)?;

        Ok(AssignStmt {
            target,
            op,
            value,
            location,
        })
    }

    /// Parse assignment operator.
    fn parse_assign_op(&self, pair: Pair<Rule>) -> ParseResult<AssignOp> {
        Ok(match pair.as_str() {
            "=" => AssignOp::Assign,
            "+=" => AssignOp::AddAssign,
            "-=" => AssignOp::SubAssign,
            "*=" => AssignOp::MulAssign,
            "/=" => AssignOp::DivAssign,
            "&=" => AssignOp::AndAssign,
            "|=" => AssignOp::OrAssign,
            "^=" => AssignOp::XorAssign,
            _ => return Err(self.error(&pair, "unknown assignment operator")),
        })
    }

    /// Parse if statement.
    fn parse_if_stmt(&self, pair: Pair<'a, Rule>) -> ParseResult<IfStmt> {
        let location = Some(self.location(&pair));
        let loc = location.clone().unwrap_or_default();
        let mut inner = pair.into_inner();

        // First is either if_unwrap_clause or if_condition
        let first = self.expect_next(&mut inner, &loc, "if condition")?;
        let (condition, capture) = match first.as_rule() {
            Rule::if_unwrap_clause => {
                // if value := expr { ... } (Go-style unwrapping)
                let first_loc = self.location(&first);
                let mut clause_inner = first.into_inner();
                let capture_name = self
                    .expect_next(&mut clause_inner, &first_loc, "capture name")?
                    .as_str()
                    .to_string();
                let expr = self.parse_expr(self.expect_next(
                    &mut clause_inner,
                    &first_loc,
                    "unwrap expression",
                )?)?;
                (expr, Some(capture_name))
            }
            Rule::if_condition => {
                // if condition { ... }
                let cond_inner = self.expect_inner(first, "if_condition")?;
                (self.parse_expr(cond_inner)?, None)
            }
            other => {
                return Err(ParseError {
                    message: format!(
                        "unexpected rule {:?}, expected if_unwrap_clause or if_condition",
                        other
                    ),
                    location: location.unwrap_or_default(),
                });
            }
        };

        let mut then_body = None;
        let mut else_body = None;

        for item in inner {
            match item.as_rule() {
                Rule::block if then_body.is_none() => {
                    then_body = Some(self.parse_block(item)?);
                }
                Rule::statement if then_body.is_none() => {
                    // Single statement as then body
                    let stmt = self.parse_statement(item)?;
                    then_body = Some(Block {
                        statements: vec![stmt],
                        label: None,
                        attrs: Vec::new(),
                        trailing_expr: None,
                        location: None,
                    });
                }
                Rule::if_stmt => {
                    else_body = Some(ElseBranch::ElseIf(Box::new(self.parse_if_stmt(item)?)));
                }
                Rule::block => {
                    else_body = Some(ElseBranch::Else(self.parse_block(item)?));
                }
                Rule::statement => {
                    // Single statement as else body
                    let stmt = self.parse_statement(item)?;
                    else_body = Some(ElseBranch::Else(Block {
                        statements: vec![stmt],
                        label: None,
                        attrs: Vec::new(),
                        trailing_expr: None,
                        location: None,
                    }));
                }
                _ => {}
            }
        }

        Ok(IfStmt {
            condition,
            capture,
            then_body: then_body.unwrap_or_else(|| Block {
                statements: vec![],
                label: None,
                attrs: Vec::new(),
                trailing_expr: None,
                location: None,
            }),
            else_body,
            location,
        })
    }

    /// Parse for statement (bounded iteration - NASA Power of 10 compliant).
    fn parse_for_stmt(&self, pair: Pair<Rule>) -> ParseResult<ForStmt> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut label = None;
        let mut is_inline = false;
        let mut range = None;
        let mut captures = Vec::new();
        let mut body = None;

        for item in inner {
            match item.as_rule() {
                Rule::label => label = Some(self.parse_label(item)?),
                Rule::for_range => range = Some(self.parse_for_range(item)?),
                Rule::capture_list => captures = self.parse_capture_list(item)?,
                Rule::block => body = Some(self.parse_block(item)?),
                _ => {
                    if item.as_rule() == Rule::inline_keyword {
                        is_inline = true;
                    }
                }
            }
        }

        Ok(ForStmt {
            label,
            is_inline,
            range: range.expect("for needs range"),
            captures,
            body: body.expect("for needs body"),
            location,
        })
    }

    /// Parse for range.
    fn parse_for_range(&self, pair: Pair<'a, Rule>) -> ParseResult<ForRange> {
        let inner = self.expect_inner(pair, "for_range")?;

        match inner.as_rule() {
            Rule::range_lit => {
                // range_lit = { range_start ~ ".." ~ range_end }
                let loc = self.location(&inner);
                let mut range_inner = inner.into_inner();
                let start_pair = self.expect_next(&mut range_inner, &loc, "range start")?;
                let end_pair = self.expect_next(&mut range_inner, &loc, "range end")?;

                // range_start/range_end = { number_literal | identifier }
                let start = self.parse_range_bound(start_pair)?;
                let end = self.parse_range_bound(end_pair)?;

                Ok(ForRange::Range { start, end })
            }
            Rule::expr => {
                // Collection iteration or single expression
                let first = self.parse_expr(inner)?;
                Ok(ForRange::Collection(first))
            }
            _ => {
                // Try parsing as expression
                let first = self.parse_expr(inner)?;
                Ok(ForRange::Collection(first))
            }
        }
    }

    /// Parse range bound expression.
    /// range_bound = { range_bound_term ~ (ws ~ range_bound_op ~ ws ~ range_bound_term)* }
    fn parse_range_bound(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner();

        // Parse the first term
        let first_term = self.expect_next(&mut inner, &location, "range bound term")?;
        let mut result = self.parse_range_bound_term(first_term)?;

        // Parse remaining (op, term) pairs
        while let Some(op_pair) = inner.next() {
            if op_pair.as_rule() == Rule::range_bound_op {
                let op = match op_pair.as_str() {
                    "+" => BinaryOp::Add,
                    "-" => BinaryOp::Sub,
                    "*" => BinaryOp::Mul,
                    "/" => BinaryOp::Div,
                    _ => unreachable!(),
                };

                let term_pair = inner.next().expect("operator needs right operand");
                let right = self.parse_range_bound_term(term_pair)?;

                result = Expr::Binary(Box::new(BinaryExpr {
                    left: result,
                    op,
                    right,
                    location: Some(location.clone()),
                }));
            }
        }

        Ok(result)
    }

    /// Parse range bound term.
    /// range_bound_term = { number_literal | range_bound_field_access | identifier | "(" ~ ws ~ range_bound ~ ws ~ ")" }
    fn parse_range_bound_term(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let inner = pair
            .into_inner()
            .next()
            .expect("range_bound_term needs content");
        match inner.as_rule() {
            Rule::number_literal | Rule::identifier => self.parse_primary_expr(inner),
            Rule::range_bound_field_access => self.parse_range_bound_field_access(inner),
            Rule::range_bound => self.parse_range_bound(inner),
            _ => unreachable!("unexpected rule in range_bound_term: {:?}", inner.as_rule()),
        }
    }

    /// Parse range bound field access (e.g., self.len, arr.count).
    /// range_bound_field_access = { identifier ~ ("." ~ identifier)+ }
    fn parse_range_bound_field_access(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner();

        // First identifier is the base
        let first = inner.next().expect("field access needs base");
        let mut result = Expr::Ident(Ident {
            name: first.as_str().to_string(),
            location: Some(self.location(&first)),
        });

        // Remaining member names are field accesses
        for field_pair in inner {
            if field_pair.as_rule() == Rule::identifier || field_pair.as_rule() == Rule::member_name
            {
                result = Expr::Field(Box::new(FieldExpr {
                    object: result,
                    field: field_pair.as_str().to_string(),
                    location: Some(location.clone()),
                }));
            }
        }

        Ok(result)
    }

    /// Parse range expression for slicing: expr? ~ ".." ~ expr?
    /// Examples: 0..2, 0.., ..2, ..
    fn parse_range_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let mut start = None;
        let mut end = None;

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::expr => {
                    // First expr is start, second is end
                    if start.is_none() {
                        start = Some(self.parse_expr(item)?);
                    } else {
                        end = Some(self.parse_expr(item)?);
                    }
                }
                _ => {}
            }
        }

        Ok(Expr::Range(Box::new(RangeExpr {
            start,
            end,
            location,
        })))
    }

    /// Parse capture list.
    fn parse_capture_list(&self, pair: Pair<Rule>) -> ParseResult<Vec<String>> {
        let mut captures = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::identifier {
                captures.push(item.as_str().to_string());
            }
        }
        Ok(captures)
    }

    /// Parse switch statement.
    fn parse_switch_stmt(&self, pair: Pair<'a, Rule>) -> ParseResult<SwitchStmt> {
        let location = Some(self.location(&pair));
        let loc = location.clone().unwrap_or_default();
        let mut inner = pair.into_inner();

        let value = self.parse_expr(self.expect_next(&mut inner, &loc, "switch value")?)?;
        let mut prongs = Vec::new();

        for item in inner {
            if item.as_rule() == Rule::switch_prong {
                prongs.push(self.parse_switch_prong(item)?);
            }
        }

        Ok(SwitchStmt {
            value,
            prongs,
            location,
        })
    }

    /// Parse switch prong.
    fn parse_switch_prong(&self, pair: Pair<Rule>) -> ParseResult<SwitchProng> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut cases = Vec::new();
        let mut is_else = false;
        let mut body = None;

        for item in inner {
            match item.as_rule() {
                Rule::switch_case => cases.push(self.parse_switch_case(item)?),
                Rule::expr | Rule::block => body = Some(self.parse_expr(item)?),
                _ => {
                    if item.as_str() == "else" {
                        is_else = true;
                    }
                }
            }
        }

        Ok(SwitchProng {
            cases,
            is_else,
            body: body.expect("prong needs body"),
            location,
        })
    }

    /// Parse switch case.
    fn parse_switch_case(&self, pair: Pair<'a, Rule>) -> ParseResult<SwitchCase> {
        let location = Some(self.location(&pair));
        let loc = location.clone().unwrap_or_default();
        let mut inner = pair.into_inner();

        let value = self.parse_expr(self.expect_next(&mut inner, &loc, "case value")?)?;
        let end = inner.next().map(|p| self.parse_expr(p)).transpose()?;

        Ok(SwitchCase {
            value,
            end,
            location,
        })
    }

    /// Parse return statement.
    fn parse_return_stmt(&self, pair: Pair<Rule>) -> ParseResult<ReturnStmt> {
        let location = Some(self.location(&pair));
        let value = pair
            .into_inner()
            .next()
            .map(|p| self.parse_expr(p))
            .transpose()?;

        Ok(ReturnStmt { value, location })
    }

    /// Parse break statement.
    fn parse_break_stmt(&self, pair: Pair<Rule>) -> ParseResult<BreakStmt> {
        let location = Some(self.location(&pair));
        let mut label = None;
        let mut value = None;

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::identifier => label = Some(item.as_str().to_string()),
                Rule::expr => value = Some(self.parse_expr(item)?),
                _ => {}
            }
        }

        Ok(BreakStmt {
            label,
            value,
            location,
        })
    }

    /// Parse continue statement.
    fn parse_continue_stmt(&self, pair: Pair<Rule>) -> ParseResult<ContinueStmt> {
        let location = Some(self.location(&pair));
        let label = pair
            .into_inner()
            .find(|p| p.as_rule() == Rule::identifier)
            .map(|p| p.as_str().to_string());

        Ok(ContinueStmt { label, location })
    }

    /// Parse defer statement.
    fn parse_defer_stmt(&self, pair: Pair<'a, Rule>) -> ParseResult<DeferStmt> {
        let location = Some(self.location(&pair));
        let inner = self.expect_inner(pair, "defer_stmt")?;
        let body = Box::new(self.parse_statement(inner)?);

        Ok(DeferStmt { body, location })
    }

    /// Parse errdefer statement.
    /// Grammar: errdefer_stmt = { "errdefer" ~ ws ~ ("|" ~ ws ~ identifier ~ ws ~ "|" ~ ws)? ~ (block | statement) }
    fn parse_errdefer_stmt(&self, pair: Pair<Rule>) -> ParseResult<ErrDeferStmt> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut capture = None;
        let mut body = None;

        for item in inner {
            match item.as_rule() {
                Rule::identifier => {
                    // Capture variable name
                    capture = Some(item.as_str().to_string());
                }
                Rule::block | Rule::statement => {
                    // Body of errdefer
                    body = Some(Box::new(self.parse_statement(item)?));
                }
                _ => {
                    // Must be block or statement (fallback)
                    body = Some(Box::new(self.parse_statement(item)?));
                }
            }
        }

        let body = body.ok_or_else(|| {
            self.error_at(
                location.clone().unwrap_or_default(),
                "errdefer requires a body",
            )
        })?;

        Ok(ErrDeferStmt {
            body,
            capture,
            location,
        })
    }

    /// Parse expression statement.
    /// Grammar: expr_stmt = { attribute_list? ~ ws ~ expr ~ ws ~ ";" }
    fn parse_expr_stmt(&self, pair: Pair<Rule>) -> ParseResult<ExprStmt> {
        let location = Some(self.location(&pair));
        let inner = pair.into_inner();

        let mut attrs = Vec::new();
        let mut expr = None;

        for item in inner {
            match item.as_rule() {
                Rule::attribute_list => {
                    attrs = self.parse_attribute_list(item)?;
                }
                Rule::expr => {
                    expr = Some(self.parse_expr(item)?);
                }
                _ => {}
            }
        }

        Ok(ExprStmt {
            expr: expr.expect("expression statement must have an expression"),
            attrs,
            location,
        })
    }

    // =========================================================================
    // Expressions
    // =========================================================================

    /// Parse an expression (entry point).
    fn parse_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let inner = self.expect_inner(pair, "expr")?;

        match inner.as_rule() {
            Rule::comptime_expr => self.parse_comptime_expr(inner),
            Rule::runtime_expr => self.parse_runtime_expr(inner),
            Rule::or_expr => self.parse_or_expr(inner),
            Rule::primary_expr => self.parse_primary_expr(inner),
            _ => self.parse_primary_expr(inner),
        }
    }

    /// Parse comptime expression.
    /// Grammar: comptime_expr = { "comptime" ~ ws ~ (block | runtime_expr) }
    fn parse_comptime_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let inner = self.expect_inner(pair, "comptime_expr")?;

        // The inner can be either a block or a runtime_expr
        let expr = match inner.as_rule() {
            Rule::block => {
                // Convert the block to a BlockExpr
                let block = self.parse_block(inner)?;
                Expr::Block(Box::new(BlockExpr {
                    label: block.label.unwrap_or_default(),
                    attrs: block.attrs,
                    statements: block.statements,
                    trailing_expr: block.trailing_expr,
                    location: block.location,
                }))
            }
            _ => self.parse_expr(inner)?,
        };

        Ok(Expr::Comptime(Box::new(ComptimeExpr {
            inner: expr,
            location,
        })))
    }

    /// Parse runtime expression (binary operators).
    fn parse_runtime_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let inner = self.expect_inner(pair, "runtime_expr")?;
        self.parse_or_expr(inner)
    }

    /// Parse or expression.
    /// Grammar: or_expr = { catch_expr ~ (or_kw ~ catch_expr)* }
    fn parse_or_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_catch_expr(self.expect_next(&mut inner, &loc, "or operand")?)?;

        while let Some(next_pair) = inner.next() {
            // Skip the or_kw operator rule
            let right_pair = if next_pair.as_rule() == Rule::or_kw {
                self.expect_next(&mut inner, &loc, "or right operand")?
            } else {
                next_pair
            };
            let right = self.parse_catch_expr(right_pair)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op: BinaryOp::Or,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse catch expression.
    /// Grammar: catch_expr = { orelse_expr ~ (catch_kw ~ ("|" ~ ws ~ identifier ~ ws ~ "|" ~ ws)? ~ orelse_expr)* }
    fn parse_catch_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let loc = location.clone().unwrap_or_default();
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_orelse_expr(self.expect_next(&mut inner, &loc, "catch operand")?)?;

        while let Some(next_pair) = inner.next() {
            // Skip the catch_kw operator rule if present
            let next_pair = if next_pair.as_rule() == Rule::catch_kw {
                self.expect_next(&mut inner, &loc, "catch handler")?
            } else {
                next_pair
            };

            // Check if this is an identifier (error capture) or an orelse_expr (handler)
            if next_pair.as_rule() == Rule::identifier {
                // This is the capture variable: catch |err| handler
                let capture = Some(next_pair.as_str().to_string());
                let handler = self.parse_orelse_expr(self.expect_next(
                    &mut inner,
                    &loc,
                    "catch handler body",
                )?)?;
                left = Expr::Catch(Box::new(CatchExpr {
                    operand: left,
                    capture,
                    handler,
                    location: location.clone(),
                }));
            } else {
                // No capture: catch handler
                let handler = self.parse_orelse_expr(next_pair)?;
                left = Expr::Catch(Box::new(CatchExpr {
                    operand: left,
                    capture: None,
                    handler,
                    location: location.clone(),
                }));
            }
        }

        Ok(left)
    }

    /// Parse orelse expression.
    /// Grammar: orelse_expr = { and_expr ~ (orelse_kw ~ and_expr)* }
    fn parse_orelse_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_and_expr(self.expect_next(&mut inner, &loc, "orelse operand")?)?;

        while let Some(next_pair) = inner.next() {
            // Skip the orelse_kw operator rule
            let right_pair = if next_pair.as_rule() == Rule::orelse_kw {
                self.expect_next(&mut inner, &loc, "orelse right operand")?
            } else {
                next_pair
            };
            let right = self.parse_and_expr(right_pair)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op: BinaryOp::Orelse,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse and expression.
    /// Grammar: and_expr = { cmp_expr ~ (and_kw ~ cmp_expr)* }
    fn parse_and_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left = self.parse_cmp_expr(self.expect_next(&mut inner, &loc, "and operand")?)?;

        while let Some(next_pair) = inner.next() {
            // Skip the and_kw operator rule
            let right_pair = if next_pair.as_rule() == Rule::and_kw {
                self.expect_next(&mut inner, &loc, "and right operand")?
            } else {
                next_pair
            };
            let right = self.parse_cmp_expr(right_pair)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op: BinaryOp::And,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse binary chain with given operator.
    fn parse_binary_chain(&self, pair: Pair<'a, Rule>, op: BinaryOp) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_next_precedence(self.expect_next(&mut inner, &loc, "binary operand")?)?;

        while let Some(next_pair) = inner.next() {
            // Skip operator rules (symbol operators)
            let right_pair = if matches!(
                next_pair.as_rule(),
                Rule::bitor_op | Rule::bitxor_op | Rule::bitand_op
            ) {
                self.expect_next(&mut inner, &loc, "binary right operand")?
            } else {
                next_pair
            };
            let right = self.parse_next_precedence(right_pair)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse next precedence level.
    fn parse_next_precedence(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        match pair.as_rule() {
            Rule::catch_expr => self.parse_catch_expr(pair),
            Rule::orelse_expr => self.parse_orelse_expr(pair),
            Rule::and_expr => self.parse_and_expr(pair),
            Rule::cmp_expr => self.parse_cmp_expr(pair),
            Rule::bitwise_or_expr => self.parse_binary_chain(pair, BinaryOp::BitOr),
            Rule::bitwise_xor_expr => self.parse_binary_chain(pair, BinaryOp::BitXor),
            Rule::bitwise_and_expr => self.parse_binary_chain(pair, BinaryOp::BitAnd),
            Rule::shift_expr => self.parse_shift_expr(pair),
            Rule::add_expr => self.parse_add_expr(pair),
            Rule::suffixed_expr => self.parse_suffixed_expr(pair),
            Rule::mul_expr => self.parse_mul_expr(pair),
            Rule::unary_expr => self.parse_unary_expr(pair),
            Rule::postfix_expr => self.parse_postfix_expr(pair),
            Rule::primary_expr => self.parse_primary_expr(pair),
            _ => self.parse_primary_expr(pair),
        }
    }

    /// Parse comparison expression.
    fn parse_cmp_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let left =
            self.parse_next_precedence(self.expect_next(&mut inner, &loc, "cmp operand")?)?;

        if let Some(op_pair) = inner.next() {
            let op = self.parse_cmp_op(op_pair)?;
            let right = self.parse_next_precedence(self.expect_next(
                &mut inner,
                &loc,
                "cmp right operand",
            )?)?;
            Ok(Expr::Binary(Box::new(BinaryExpr {
                op,
                left,
                right,
                location: None,
            })))
        } else {
            Ok(left)
        }
    }

    /// Parse comparison operator.
    fn parse_cmp_op(&self, pair: Pair<Rule>) -> ParseResult<BinaryOp> {
        // Check for sub-rules first (in_op, not_in_op)
        if let Some(inner) = pair.clone().into_inner().next() {
            return match inner.as_rule() {
                Rule::in_op => Ok(BinaryOp::In),
                Rule::not_in_op => Ok(BinaryOp::NotIn),
                _ => Err(self.error(&pair, "unknown comparison operator")),
            };
        }
        // Direct string match for simple operators
        Ok(match pair.as_str() {
            "==" => BinaryOp::Eq,
            "!=" => BinaryOp::Ne,
            "<" => BinaryOp::Lt,
            "<=" => BinaryOp::Le,
            ">" => BinaryOp::Gt,
            ">=" => BinaryOp::Ge,
            "in" => BinaryOp::In,
            _ => return Err(self.error(&pair, "unknown comparison operator")),
        })
    }

    /// Parse shift expression.
    fn parse_shift_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_next_precedence(self.expect_next(&mut inner, &loc, "shift operand")?)?;

        while let Some(op_pair) = inner.next() {
            if op_pair.as_rule() != Rule::shift_op {
                continue;
            }
            let op = match op_pair.as_str() {
                "<<" => BinaryOp::Shl,
                ">>" => BinaryOp::Shr,
                _ => continue,
            };
            let right = self.parse_next_precedence(self.expect_next(
                &mut inner,
                &loc,
                "shift right operand",
            )?)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse additive expression.
    fn parse_add_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_next_precedence(self.expect_next(&mut inner, &loc, "add operand")?)?;

        while let Some(op_pair) = inner.next() {
            if op_pair.as_rule() != Rule::add_op {
                continue;
            }
            let op = match op_pair.as_str() {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                _ => continue,
            };
            let right = self.parse_next_precedence(self.expect_next(
                &mut inner,
                &loc,
                "add right operand",
            )?)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse multiplicative expression.
    fn parse_mul_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_next_precedence(self.expect_next(&mut inner, &loc, "mul operand")?)?;

        while let Some(op_pair) = inner.next() {
            if op_pair.as_rule() != Rule::mul_op {
                continue;
            }
            let op = match op_pair.as_str() {
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                "%" => BinaryOp::Mod,
                _ => continue,
            };
            let right = self.parse_next_precedence(self.expect_next(
                &mut inner,
                &loc,
                "mul right operand",
            )?)?;
            left = Expr::Binary(Box::new(BinaryExpr {
                op,
                left,
                right,
                location: None,
            }));
        }

        Ok(left)
    }

    /// Parse suffixed expression: `mul_expr ~ (expr_suffix)?`
    /// Handles both angle units (`0.25 turns`, `pi/4 rad`) and type suffixes (`42 u32`, `1/4 f64`).
    fn parse_suffixed_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner();

        // Parse the expression part
        let value = self.parse_next_precedence(self.expect_next(
            &mut inner,
            &location,
            "expression in suffixed expr",
        )?)?;

        // Check for optional suffix (angle unit or type)
        if let Some(suffix_pair) = inner.next()
            && suffix_pair.as_rule() == Rule::expr_suffix
        {
            // Get the inner rule (angle_unit or type_suffix)
            let inner_suffix = self.expect_inner(suffix_pair, "expr_suffix")?;
            match inner_suffix.as_rule() {
                Rule::angle_unit => {
                    let unit = match inner_suffix.as_str() {
                        "turns" => AngleUnit::Turns,
                        "rad" => AngleUnit::Rad,
                        _ => return Err(self.error(&inner_suffix, "unknown angle unit")),
                    };
                    return Ok(Expr::AngleLit(Box::new(AngleLit {
                        value,
                        unit,
                        location: Some(location),
                    })));
                }
                Rule::type_ascription_suffix => {
                    // Get the actual type keyword
                    let type_inner = self.expect_inner(inner_suffix, "type_ascription_suffix")?;
                    let type_name = type_inner.as_str().to_string();
                    return Ok(Expr::TypeAscription(Box::new(TypeAscription {
                        value,
                        type_name,
                        location: Some(location),
                    })));
                }
                _ => {}
            }
        }

        // No suffix - return the value as-is
        Ok(value)
    }

    /// Parse unary expression.
    fn parse_unary_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner().peekable();
        let mut ops = Vec::new();

        // Collect unary operators
        while let Some(item) = inner.peek() {
            if item.as_rule() == Rule::unary_op {
                // Safe: we just peeked and confirmed it exists
                if let Some(op) = inner.next() {
                    ops.push(op);
                }
            } else {
                break;
            }
        }

        // Parse the operand
        let operand = self.expect_next(&mut inner, &location, "operand in unary expression")?;
        let mut expr = self.parse_postfix_expr(operand)?;

        // Apply operators in reverse order
        for op_pair in ops.into_iter().rev() {
            let op = match op_pair.as_str() {
                "try" => UnaryOp::Try,
                "-" => UnaryOp::Neg,
                "!" => UnaryOp::Not,
                "~" => UnaryOp::BitNot,
                "&" => UnaryOp::AddrOf,
                "*" => UnaryOp::Deref,
                _ => continue,
            };
            expr = Expr::Unary(Box::new(UnaryExpr {
                op,
                operand: expr,
                location: Some(self.location(&op_pair)),
            }));
        }

        Ok(expr)
    }

    /// Parse postfix expression.
    fn parse_postfix_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner();
        let primary = self.expect_next(&mut inner, &location, "primary expression")?;
        let mut expr = self.parse_primary_expr(primary)?;

        // Apply postfix operators
        for op in inner {
            // Capture end location from the operator
            let op_span = op.as_span();
            let (op_end_line, op_end_col) = op_span.end_pos().line_col();

            // Get start location from current expression, or use operator's start
            let expr_start = expr.get_location();
            let (start_line, start_col) = expr_start
                .as_ref()
                .map(|loc| (loc.line, loc.column))
                .unwrap_or_else(|| {
                    let (l, c) = op.line_col();
                    (l as u32, c as u32)
                });

            // Combined location spans from expression start to operator end
            let combined_location = SourceLocation {
                line: start_line,
                column: start_col,
                end_line: op_end_line as u32,
                end_column: op_end_col as u32,
                file: self.file.clone(),
            };

            // postfix_op is a wrapper around call | field_access | index_access | optional_unwrap | error_unwrap
            // We need to get the inner rule
            let actual_op = if op.as_rule() == Rule::postfix_op {
                self.expect_inner(op, "postfix_op")?
            } else {
                op
            };
            match actual_op.as_rule() {
                Rule::call => {
                    let args = self.parse_arg_list(actual_op)?;
                    expr = Expr::Call(Box::new(CallExpr {
                        callee: expr,
                        args,
                        location: Some(combined_location),
                    }));
                }
                Rule::batch_apply => {
                    // batch_apply = { ws ~ "{" ~ ws ~ batch_elements? ~ ws ~ "}" }
                    let mut targets = Vec::new();
                    for inner in actual_op.into_inner() {
                        if inner.as_rule() == Rule::batch_elements {
                            for elem in inner.into_inner() {
                                targets.push(self.parse_expr(elem)?);
                            }
                        }
                    }
                    expr = Expr::BatchApply(Box::new(BatchApplyExpr {
                        operation: expr,
                        targets,
                        location: Some(combined_location),
                    }));
                }
                Rule::field_access => {
                    let field = self
                        .expect_inner(actual_op, "field_access")?
                        .as_str()
                        .to_string();
                    expr = Expr::Field(Box::new(FieldExpr {
                        object: expr,
                        field,
                        location: Some(combined_location),
                    }));
                }
                Rule::index_access => {
                    let index_inner = self.expect_inner(actual_op, "index_access")?;
                    // Check if the index is a range expression (for slicing)
                    let index = if index_inner.as_rule() == Rule::range_expr {
                        self.parse_range_expr(index_inner)?
                    } else {
                        self.parse_expr(index_inner)?
                    };
                    expr = Expr::Index(Box::new(IndexExpr {
                        object: expr,
                        index,
                        location: Some(combined_location),
                    }));
                }
                Rule::optional_unwrap => {
                    expr = Expr::Unary(Box::new(UnaryExpr {
                        op: UnaryOp::OptionalUnwrap,
                        operand: expr,
                        location: Some(combined_location),
                    }));
                }
                Rule::error_unwrap => {
                    expr = Expr::Unary(Box::new(UnaryExpr {
                        op: UnaryOp::ErrorUnwrap,
                        operand: expr,
                        location: Some(combined_location),
                    }));
                }
                _ => {}
            }
        }

        Ok(expr)
    }

    /// Parse argument list.
    fn parse_arg_list(&self, pair: Pair<Rule>) -> ParseResult<Vec<Expr>> {
        let mut args = Vec::new();
        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::arg_list => {
                    for arg in item.into_inner() {
                        if arg.as_rule() == Rule::expr {
                            args.push(self.parse_expr(arg)?);
                        }
                    }
                }
                Rule::expr => {
                    args.push(self.parse_expr(item)?);
                }
                _ => {}
            }
        }
        Ok(args)
    }

    /// Parse primary expression.
    fn parse_primary_expr(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));

        match pair.as_rule() {
            Rule::number_literal => self.parse_number(pair),
            Rule::string_literal => Ok(Expr::StringLit(StringLit {
                value: self.parse_string_content(pair.as_str()),
                location,
            })),
            Rule::raw_string => {
                // Raw string: r"..." - no escape processing
                let s = pair.as_str();
                // Remove r" prefix and " suffix
                let value = s[2..s.len() - 1].to_string();
                Ok(Expr::StringLit(StringLit { value, location }))
            }
            Rule::multiline_string => {
                // Multi-line string: """...""" - preserves newlines
                let s = pair.as_str();
                // Remove """ prefix and """ suffix (3 chars each)
                let content = &s[3..s.len() - 3];
                // Process escape sequences within the content
                let value = self.unescape(content);
                Ok(Expr::StringLit(StringLit { value, location }))
            }
            Rule::f_string => self.parse_f_string(pair),
            Rule::char_literal => {
                let s = pair.as_str();
                let c = self.parse_char_content(s);
                Ok(Expr::CharLit(CharLit { value: c, location }))
            }
            Rule::bool_literal => Ok(Expr::BoolLit(BoolLit {
                value: pair.as_str() == "true",
                location,
            })),
            Rule::none_literal => Ok(Expr::Null(NullLit { location })),
            Rule::undefined_literal => Ok(Expr::Undefined(UndefinedLit { location })),
            Rule::unit_literal => Ok(Expr::Unit(UnitLit { location })),
            Rule::identifier => Ok(Expr::Ident(Ident {
                name: pair.as_str().to_string(),
                location,
            })),
            Rule::builtin_call => self.parse_builtin_call(pair),
            Rule::struct_init => self.parse_struct_init(pair),
            Rule::array_init => self.parse_array_init(pair),
            Rule::if_expr => self.parse_if_expr(pair),
            Rule::try_block_expr => self.parse_try_block_expr(pair),
            Rule::block_expr => self.parse_block_expr(pair),
            Rule::paren_or_tuple => {
                // paren_or_tuple = { "(" ~ ws ~ expr ~ (ws ~ "," ~ ws ~ expr)* ~ (ws ~ ",")? ~ ws ~ ")" }
                // Single element without trailing comma -> parenthesized expr (unwrap)
                // Multiple elements or trailing comma -> tuple
                let pair_str = pair.as_str();
                let has_trailing_comma = pair_str.trim_end_matches(')').trim_end().ends_with(',');

                let mut elements = Vec::new();
                for inner in pair.into_inner() {
                    elements.push(self.parse_expr(inner)?);
                }

                if elements.len() == 1 && !has_trailing_comma {
                    // Single element, no trailing comma -> parenthesized expression
                    Ok(elements.pop().unwrap())
                } else {
                    // Multiple elements or trailing comma -> tuple
                    Ok(Expr::Tuple(Box::new(TupleExpr { elements, location })))
                }
            }
            Rule::bracket_array => {
                // bracket_array = { "[" ~ ws ~ bracket_array_elements? ~ ws ~ "]" }
                let mut elements = Vec::new();
                for inner in pair.into_inner() {
                    if inner.as_rule() == Rule::bracket_array_elements {
                        for elem in inner.into_inner() {
                            elements.push(self.parse_expr(elem)?);
                        }
                    }
                }
                Ok(Expr::BracketArray(Box::new(BracketArrayExpr {
                    elements,
                    location,
                })))
            }
            Rule::set_literal => {
                // set_literal = { "set" ~ ws ~ "{" ~ ws ~ set_elements? ~ ws ~ "}" }
                let mut elements = Vec::new();
                for inner in pair.into_inner() {
                    if inner.as_rule() == Rule::set_elements {
                        for elem in inner.into_inner() {
                            elements.push(self.parse_expr(elem)?);
                        }
                    }
                }
                Ok(Expr::Set(Box::new(SetExpr {
                    elements,
                    element_type: None,
                    location,
                })))
            }
            Rule::measure_expr => {
                // measure_expr = { "mz" ~ ws ~ "(" ~ ws ~ pack_modifier? ~ type_expr ~ ws ~ ")" ~ ws ~ measure_target }
                let mut result_type = None;
                let mut targets = None;
                let mut pack = false;
                for inner in pair.into_inner() {
                    match inner.as_rule() {
                        Rule::pack_modifier => {
                            pack = true;
                        }
                        Rule::type_expr => {
                            result_type = Some(self.parse_type_expr(inner)?);
                        }
                        Rule::measure_target => {
                            // measure_target = { bracket_array | postfix_expr }
                            let target_inner = self.expect_inner(inner, "measure_target")?;
                            // Handle each rule type directly
                            targets = Some(match target_inner.as_rule() {
                                Rule::bracket_array => self.parse_primary_expr(target_inner)?,
                                Rule::postfix_expr => self.parse_postfix_expr(target_inner)?,
                                _ => self.parse_primary_expr(target_inner)?,
                            });
                        }
                        _ => {}
                    }
                }
                Ok(Expr::Measure(Box::new(MeasureExpr {
                    result_type: result_type.expect("measure requires type"),
                    pack,
                    targets: targets.expect("measure requires targets"),
                    location,
                })))
            }
            Rule::channel_expr => {
                // channel_expr = { emit_prefix ~ ws ~ channel_name ~ ws ~ "." ~ ws ~ channel_command }
                // emit_prefix = { "@emit" ~ ws ~ "." }
                // channel_command = { channel_command_name ~ ws ~ "(" ~ ws ~ channel_args? ~ ws ~ ")" }
                let mut channel = None;
                let mut command = None;
                let mut args = Vec::new();

                for item in pair.into_inner() {
                    match item.as_rule() {
                        Rule::emit_prefix => {
                            // Just consume the @emit. prefix
                        }
                        Rule::channel_name => {
                            channel = Some(item.as_str().to_string());
                        }
                        Rule::channel_command => {
                            for cmd_item in item.into_inner() {
                                match cmd_item.as_rule() {
                                    Rule::channel_command_name => {
                                        command = Some(cmd_item.as_str().to_string());
                                    }
                                    Rule::channel_args => {
                                        for arg_item in cmd_item.into_inner() {
                                            if arg_item.as_rule() == Rule::channel_arg {
                                                args.push(self.parse_channel_arg(arg_item)?);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }

                Ok(Expr::Channel(Box::new(ChannelExpr {
                    channel: channel.expect("channel_expr requires channel name"),
                    command: command.expect("channel_expr requires command"),
                    args,
                    location,
                })))
            }
            Rule::result_expr => {
                // result_expr = { "result" ~ ws ~ "(" ~ ws ~ string_literal ~ ws ~ "," ~ ws ~ expr ~ ws ~ ")" }
                let mut tag = None;
                let mut value = None;

                for item in pair.into_inner() {
                    match item.as_rule() {
                        Rule::string_literal => {
                            tag = Some(self.parse_string_content(item.as_str()));
                        }
                        Rule::expr => {
                            value = Some(self.parse_expr(item)?);
                        }
                        _ => {}
                    }
                }

                Ok(Expr::Result(Box::new(ResultExpr {
                    tag: tag.expect("result requires tag"),
                    value: value.expect("result requires value"),
                    location,
                })))
            }
            Rule::gate_expr => {
                // gate_expr = { param_gate_expr | simple_gate_expr }
                let inner = self.expect_inner(pair, "gate_expr")?;
                self.parse_gate_expr_inner(inner, location)
            }
            Rule::param_gate_expr | Rule::simple_gate_expr => {
                self.parse_gate_expr_inner(pair, location)
            }
            Rule::struct_literal => {
                // struct_literal = { ("packed" ~ ws)? ~ "struct" ~ ws ~ "{" ~ ws ~ struct_body ~ ws ~ "}" }
                // Anonymous struct type definition: struct { x: i32, y: i32 }
                let mut is_packed = false;
                let mut fields = Vec::new();

                for item in pair.into_inner() {
                    match item.as_rule() {
                        Rule::packed_keyword => {
                            is_packed = true;
                        }
                        Rule::struct_body => {
                            // Parse struct fields (ignoring methods and bindings for anonymous structs)
                            for body_item in item.into_inner() {
                                if body_item.as_rule() == Rule::struct_field {
                                    fields.push(self.parse_struct_field(body_item)?);
                                }
                            }
                        }
                        _ => {}
                    }
                }

                Ok(Expr::AnonStruct(Box::new(AnonStructExpr {
                    fields,
                    is_packed,
                    location,
                })))
            }
            Rule::enum_literal => {
                // enum_literal = similar to struct_literal
                Ok(Expr::StructInit(Box::new(StructInitExpr {
                    ty: None,
                    fields: Vec::new(),
                    location,
                })))
            }
            Rule::fn_literal => {
                // Anonymous function: fn(params) -> return_type { body }
                let inner = pair.into_inner();
                let mut params = Vec::new();
                let mut return_type = None;
                let mut body = None;

                for item in inner {
                    match item.as_rule() {
                        Rule::param_list => {
                            params = self.parse_param_list(item)?;
                        }
                        Rule::return_type => {
                            return_type = Some(self.parse_return_type(item)?);
                        }
                        Rule::block => {
                            body = Some(self.parse_block(item)?);
                        }
                        _ => {}
                    }
                }

                Ok(Expr::FnLit(Box::new(FnDecl {
                    name: "<anon>".to_string(),
                    params,
                    return_type,
                    body: body.expect("function literal must have body"),
                    is_pub: false,
                    is_inline: false,
                    error_mode: None,
                    doc_comment: None,
                    location,
                })))
            }
            Rule::self_expr => Ok(Expr::Ident(Ident {
                name: "Self".to_string(),
                location,
            })),
            Rule::error_value => {
                // error_value = { "error" ~ "." ~ identifier }
                let inner = self.expect_inner(pair, "error_value")?;
                let name = inner.as_str().to_string();
                Ok(Expr::ErrorValue(Box::new(ErrorValueExpr {
                    name,
                    location,
                })))
            }
            Rule::fault_value => {
                // fault_value = { "fault" ~ "." ~ identifier }
                let inner = self.expect_inner(pair, "fault_value")?;
                let name = inner.as_str().to_string();
                Ok(Expr::FaultValue(Box::new(FaultValueExpr {
                    name,
                    location,
                })))
            }
            Rule::array_type_expr => {
                // array_type_expr = { "[" ~ ws ~ (array_size | "_")? ~ ... ~ "]" ~ ws ~ type_identifier }
                // This represents a type as a value (like [N]T)
                Ok(Expr::Ident(Ident {
                    name: pair.as_str().to_string(),
                    location,
                }))
            }
            Rule::primary_expr => {
                let inner = self.expect_inner(pair, "primary_expr")?;
                self.parse_primary_expr(inner)
            }
            Rule::atom => {
                // atom is a compound-atomic wrapper around leaf expressions
                let inner = self.expect_inner(pair, "atom")?;
                self.parse_primary_expr(inner)
            }
            _ => Err(self.error(&pair, format!("unexpected primary {:?}", pair.as_rule()))),
        }
    }

    /// Parse number literal with optional type suffix.
    fn parse_number(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let raw = pair.as_str();

        // Extract type suffix if present
        let (num_str, suffix) = self.extract_number_suffix(raw);
        let s = num_str.replace('_', "");

        // Check for float (must check before extracting suffix changes things)
        let is_float = s.contains('.')
            || (s.contains('e') || s.contains('E')) && !s.starts_with("0x") && !s.starts_with("0X");

        if is_float {
            let value: f64 = s
                .parse()
                .map_err(|_| self.error_at(loc.clone(), "invalid float literal"))?;
            Ok(Expr::FloatLit(FloatLit {
                value,
                suffix,
                location: Some(loc),
            }))
        } else if s.starts_with("0x") || s.starts_with("0X") {
            let value = i128::from_str_radix(&s[2..], 16)
                .map_err(|_| self.error_at(loc.clone(), "invalid hex literal"))?;
            Ok(Expr::IntLit(IntLit {
                value,
                suffix,
                location: Some(loc),
            }))
        } else if s.starts_with("0b") || s.starts_with("0B") {
            let value = i128::from_str_radix(&s[2..], 2)
                .map_err(|_| self.error_at(loc.clone(), "invalid binary literal"))?;
            Ok(Expr::IntLit(IntLit {
                value,
                suffix,
                location: Some(loc),
            }))
        } else if s.starts_with("0o") || s.starts_with("0O") {
            let value = i128::from_str_radix(&s[2..], 8)
                .map_err(|_| self.error_at(loc.clone(), "invalid octal literal"))?;
            Ok(Expr::IntLit(IntLit {
                value,
                suffix,
                location: Some(loc),
            }))
        } else {
            let value: i128 = s
                .parse()
                .map_err(|_| self.error_at(loc.clone(), "invalid integer literal"))?;
            Ok(Expr::IntLit(IntLit {
                value,
                suffix,
                location: Some(loc),
            }))
        }
    }

    /// Extract type suffix from a number literal string.
    /// Returns (number_part, optional_suffix).
    fn extract_number_suffix<'b>(&self, s: &'b str) -> (&'b str, Option<String>) {
        // Integer suffixes (check longer ones first)
        const INT_SUFFIXES: &[&str] = &[
            "u128", "i128", "usize", "isize", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
            "u1", "i1",
        ];
        // Float suffixes
        const FLOAT_SUFFIXES: &[&str] = &["f128", "f64", "f32", "f16", "a64"];

        // Check for suffix with optional underscore separator
        for &suffix in INT_SUFFIXES.iter().chain(FLOAT_SUFFIXES.iter()) {
            // Check for _suffix pattern
            let with_underscore = format!("_{}", suffix);
            if s.ends_with(&with_underscore) {
                return (
                    &s[..s.len() - with_underscore.len()],
                    Some(suffix.to_string()),
                );
            }
            // Check for direct suffix (no underscore)
            if let Some(prefix) = s.strip_suffix(suffix) {
                // Make sure we're not matching part of a hex digit
                if !prefix.is_empty()
                    && (prefix.ends_with(|c: char| c.is_ascii_digit()) || prefix.ends_with('_'))
                {
                    return (prefix, Some(suffix.to_string()));
                }
            }
        }

        (s, None)
    }

    /// Parse string content (remove quotes and handle escapes).
    fn parse_string_content(&self, s: &str) -> String {
        let s = &s[1..s.len() - 1]; // Remove quotes
        self.unescape(s)
    }

    /// Parse char content.
    fn parse_char_content(&self, s: &str) -> char {
        let s = &s[1..s.len() - 1]; // Remove quotes
        let unescaped = self.unescape(s);
        unescaped.chars().next().unwrap_or('\0')
    }

    /// Parse f-string (Python-style interpolated string): f"Hello {name}!"
    /// Supports format specifiers: f"{x:.2f}", f"{name:>10}"
    fn parse_f_string(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let mut parts = Vec::new();

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::f_string_part => {
                    let inner = self.expect_inner(item, "f_string_part")?;
                    match inner.as_rule() {
                        Rule::f_string_text => {
                            let text = self.unescape(inner.as_str());
                            parts.push(FStringPart::Text(text));
                        }
                        Rule::f_string_interp => {
                            let (expr, format) = self.parse_f_string_interp(inner)?;
                            parts.push(FStringPart::Expr { expr, format });
                        }
                        _ => {}
                    }
                }
                Rule::f_string_text => {
                    let text = self.unescape(item.as_str());
                    parts.push(FStringPart::Text(text));
                }
                Rule::f_string_interp => {
                    let (expr, format) = self.parse_f_string_interp(item)?;
                    parts.push(FStringPart::Expr { expr, format });
                }
                _ => {}
            }
        }

        Ok(Expr::FString(Box::new(FStringExpr { parts, location })))
    }

    /// Parse f-string interpolation: {expr} or {expr:format}
    fn parse_f_string_interp(&self, pair: Pair<'a, Rule>) -> ParseResult<(Expr, Option<String>)> {
        let location = self.location(&pair);
        let mut expr = None;
        let mut format = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expr => {
                    expr = Some(self.parse_expr(inner)?);
                }
                Rule::f_string_format => {
                    // f_string_format = { ":" ~ f_string_format_spec }
                    for fmt_inner in inner.into_inner() {
                        if fmt_inner.as_rule() == Rule::f_string_format_spec {
                            let spec = fmt_inner.as_str().to_string();
                            if !spec.is_empty() {
                                format = Some(spec);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let expr = expr.ok_or_else(|| {
            self.error_at(location, "expected expression in f-string interpolation")
        })?;
        Ok((expr, format))
    }

    /// Unescape string content.
    fn unescape(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('\'') => result.push('\''),
                    Some('0') => result.push('\0'),
                    Some('{') => result.push('{'),
                    Some('}') => result.push('}'),
                    Some('x') => {
                        let hex: String = chars.by_ref().take(2).collect();
                        if let Ok(n) = u8::from_str_radix(&hex, 16) {
                            result.push(n as char);
                        }
                    }
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Parse builtin call (@import, @This, etc.).
    fn parse_builtin_call(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();

        let name = self
            .expect_next(&mut inner, &loc, "builtin name")?
            .as_str()
            .to_string();
        let location = Some(loc);
        let args = if let Some(arg_list) = inner.next() {
            self.parse_arg_list(arg_list)?
        } else {
            Vec::new()
        };

        Ok(Expr::Builtin(Box::new(BuiltinExpr {
            name,
            args,
            location,
        })))
    }

    /// Parse struct initialization.
    fn parse_struct_init(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let inner_pair = self.expect_inner(pair, "struct_init")?;

        // struct_init contains either typed_struct_init or anon_struct_init
        match inner_pair.as_rule() {
            Rule::typed_struct_init => self.parse_typed_struct_init(inner_pair, location),
            Rule::anon_struct_init => self.parse_anon_struct_init(inner_pair, location),
            other => {
                return Err(ParseError {
                    message: format!(
                        "unexpected rule {:?}, expected typed_struct_init or anon_struct_init",
                        other
                    ),
                    location: location.unwrap_or_default(),
                });
            }
        }
    }

    fn parse_typed_struct_init(
        &self,
        pair: Pair<Rule>,
        location: Option<SourceLocation>,
    ) -> ParseResult<Expr> {
        let mut inner = pair.into_inner();

        // First element is type_identifier
        let ty = if let Some(first) = inner.next() {
            if first.as_rule() == Rule::type_identifier {
                Some(self.parse_type_identifier(first)?)
            } else {
                None
            }
        } else {
            None
        };

        let fields = self.collect_field_inits(inner)?;

        Ok(Expr::StructInit(Box::new(StructInitExpr {
            ty,
            fields,
            location,
        })))
    }

    fn parse_anon_struct_init(
        &self,
        pair: Pair<Rule>,
        location: Option<SourceLocation>,
    ) -> ParseResult<Expr> {
        let inner = pair.into_inner();
        let fields = self.collect_field_inits(inner)?;

        Ok(Expr::StructInit(Box::new(StructInitExpr {
            ty: None,
            fields,
            location,
        })))
    }

    fn collect_field_inits(&self, inner: Pairs<Rule>) -> ParseResult<Vec<FieldInit>> {
        let mut fields = Vec::new();
        for item in inner {
            match item.as_rule() {
                Rule::field_init => {
                    fields.push(self.parse_field_init(item)?);
                }
                Rule::field_init_list => {
                    for field in item.into_inner() {
                        if field.as_rule() == Rule::field_init {
                            fields.push(self.parse_field_init(field)?);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(fields)
    }

    /// Parse field initializer.
    fn parse_field_init(&self, pair: Pair<'a, Rule>) -> ParseResult<FieldInit> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();

        let name = self
            .expect_next(&mut inner, &loc, "field name")?
            .as_str()
            .to_string();
        let location = Some(loc);

        // Rust-style: `field: value` or shorthand `field` (when var name matches)
        let value = if let Some(expr_pair) = inner.next() {
            self.parse_expr(expr_pair)?
        } else {
            // Shorthand: `field` expands to `field: field`
            Expr::Ident(Ident {
                name: name.clone(),
                location: location.clone(),
            })
        };

        Ok(FieldInit {
            name,
            value,
            location,
        })
    }

    /// Parse array initialization.
    fn parse_array_init(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let mut inner = pair.into_inner().peekable();

        let ty = if let Some(first) = inner.peek() {
            if first.as_rule() == Rule::type_expr {
                // Safe: we just peeked and confirmed it exists
                if let Some(type_pair) = inner.next() {
                    Some(self.parse_type_expr(type_pair)?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let mut elements = Vec::new();
        for item in inner {
            if item.as_rule() == Rule::expr {
                elements.push(self.parse_expr(item)?);
            }
        }

        Ok(Expr::ArrayInit(Box::new(ArrayInitExpr {
            ty,
            elements,
            location,
        })))
    }

    /// Parse try block expression.
    /// Grammar: try_block_expr = { try_collect_expr | try_bang_expr }
    fn parse_try_block_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let inner = self.expect_inner(pair, "try_block_expr")?;
        let location = Some(self.location(&inner));

        match inner.as_rule() {
            Rule::try_collect_expr => {
                let mut body = None;
                let mut catch_clause = None;
                for item in inner.into_inner() {
                    match item.as_rule() {
                        Rule::block => body = Some(self.parse_block(item)?),
                        Rule::catch_clause => catch_clause = Some(self.parse_catch_clause(item)?),
                        _ => {}
                    }
                }
                Ok(Expr::TryBlock(Box::new(TryBlockExpr {
                    mode: TryMode::Collect,
                    body: body.expect("try block must have body"),
                    catch_clause,
                    location,
                })))
            }
            Rule::try_bang_expr => {
                let mut body = None;
                let mut catch_clause = None;
                for item in inner.into_inner() {
                    match item.as_rule() {
                        Rule::block => body = Some(self.parse_block(item)?),
                        Rule::catch_clause => catch_clause = Some(self.parse_catch_clause(item)?),
                        _ => {}
                    }
                }
                Ok(Expr::TryBlock(Box::new(TryBlockExpr {
                    mode: TryMode::Propagate,
                    body: body.expect("try! block must have body"),
                    catch_clause,
                    location,
                })))
            }
            _ => Err(self.error(&inner, "expected try or try! expression")),
        }
    }

    /// Parse if expression.
    fn parse_if_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<Expr> {
        let loc = self.location(&pair);
        let mut inner = pair.into_inner();

        // First is the condition (in parentheses)
        let cond_pair = self.expect_next(&mut inner, &loc, "if condition")?;
        let condition = self.parse_expr(cond_pair)?;

        // Then block for the "then" branch
        let then_block = self.expect_next(&mut inner, &loc, "then block")?;
        let then_parsed = self.parse_block(then_block)?;
        let then_expr = Expr::Block(Box::new(BlockExpr {
            label: then_parsed.label.clone().unwrap_or_default(),
            attrs: then_parsed.attrs,
            statements: then_parsed.statements,
            trailing_expr: then_parsed.trailing_expr,
            location: then_parsed.location,
        }));

        // Else branch: either if_expr or block
        let else_item = self.expect_next(&mut inner, &loc, "else branch")?;
        let location = Some(loc);
        let else_expr = match else_item.as_rule() {
            Rule::if_expr => self.parse_if_expr(else_item)?,
            Rule::block => {
                let else_parsed = self.parse_block(else_item)?;
                Expr::Block(Box::new(BlockExpr {
                    label: else_parsed.label.clone().unwrap_or_default(),
                    attrs: else_parsed.attrs,
                    statements: else_parsed.statements,
                    trailing_expr: else_parsed.trailing_expr,
                    location: else_parsed.location,
                }))
            }
            _ => {
                return Err(ParseError {
                    message: format!(
                        "Expected block or if_expr in else branch, got {:?}",
                        else_item.as_rule()
                    ),
                    location: self.location(&else_item),
                });
            }
        };

        Ok(Expr::If(Box::new(IfExpr {
            condition,
            then_expr,
            else_expr,
            location,
        })))
    }

    /// Parse block expression.
    fn parse_block_expr(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let location = Some(self.location(&pair));
        let mut statements = Vec::new();
        let mut trailing_expr = None;
        let mut attrs = Vec::new();
        let mut label = String::new();

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::attribute_list => {
                    attrs.extend(self.parse_attribute_list(item)?);
                }
                Rule::label => {
                    label = self.parse_label(item)?;
                }
                Rule::statement => {
                    statements.push(self.parse_statement(item)?);
                }
                Rule::trailing_expr => {
                    let expr_inner = self.expect_inner(item, "trailing_expr")?;
                    trailing_expr = Some(Box::new(self.parse_expr(expr_inner)?));
                }
                _ => {}
            }
        }

        Ok(Expr::Block(Box::new(BlockExpr {
            label,
            attrs,
            statements,
            trailing_expr,
            location,
        })))
    }

    // =========================================================================
    // Types
    // =========================================================================

    /// Parse a type expression.
    fn parse_type_expr(&self, pair: Pair<'a, Rule>) -> ParseResult<TypeExpr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner();

        // type_expr = { type_prefix ~ type_suffix? }
        // First get the type_prefix
        let prefix_pair = self.expect_next(&mut inner, &location, "type_prefix")?;
        let base_type = self.parse_type_prefix(prefix_pair)?;

        // Check for type_suffix (error union: E!T where E is error, T is payload)
        if let Some(suffix_pair) = inner.next()
            && suffix_pair.as_rule() == Rule::type_suffix
        {
            // type_suffix = { "!" ~ type_prefix }
            // In E!T syntax: base_type is E (error), suffix is T (payload)
            let payload_type_pair = self.expect_inner(suffix_pair, "type_suffix")?;
            let payload_type = self.parse_type_prefix(payload_type_pair)?;
            return Ok(TypeExpr::ErrorUnion(Box::new(ErrorUnionType {
                error_type: base_type,
                payload_type,
            })));
        }

        Ok(base_type)
    }

    /// Parse type_prefix.
    fn parse_type_prefix(&self, pair: Pair<'a, Rule>) -> ParseResult<TypeExpr> {
        // type_prefix contains one of the type alternatives
        let inner = self.expect_inner(pair, "type_prefix")?;

        match inner.as_rule() {
            Rule::optional_type => {
                let inner_type =
                    self.parse_type_prefix(self.expect_inner(inner, "optional_type")?)?;
                Ok(TypeExpr::Optional(Box::new(inner_type)))
            }
            Rule::pointer_type => self.parse_pointer_type(inner),
            Rule::array_type => self.parse_array_type(inner),
            Rule::tuple_type => self.parse_tuple_type(inner),
            Rule::fn_type => self.parse_fn_type(inner),
            Rule::set_type => self.parse_set_type(inner),
            Rule::struct_literal => self.parse_struct_type(inner),
            Rule::enum_literal => self.parse_enum_type(inner),
            Rule::builtin_type => self.parse_builtin_type(inner),
            Rule::type_identifier => self.parse_type_identifier(inner),
            _ => Err(ParseError {
                message: format!("unexpected type {:?}", inner.as_rule()),
                location: self.location(&inner),
            }),
        }
    }

    /// Parse inline struct type: struct { x: i32, y: i32 }
    fn parse_struct_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let mut is_packed = false;
        let mut fields = Vec::new();

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::packed_keyword => {
                    is_packed = true;
                }
                Rule::struct_body => {
                    for body_item in item.into_inner() {
                        if body_item.as_rule() == Rule::struct_field {
                            fields.push(self.parse_struct_field(body_item)?);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(TypeExpr::Struct(Box::new(InlineStructType {
            fields,
            is_packed,
        })))
    }

    /// Parse inline enum type: enum { a, b, c } or enum(u8) { a, b, c }
    fn parse_enum_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let mut tag_type = None;
        let mut variants = Vec::new();

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::type_expr => {
                    tag_type = Some(self.parse_type_expr(item)?);
                }
                Rule::enum_body => {
                    for body_item in item.into_inner() {
                        if body_item.as_rule() == Rule::enum_variant {
                            variants.push(self.parse_enum_variant(body_item)?);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(TypeExpr::Enum(Box::new(InlineEnumType {
            variants,
            tag_type,
        })))
    }

    /// Parse pointer type.
    /// Grammar: pointer_type = { pointer_prefix ~ ws ~ ("const" ~ ws)? ~ type_prefix }
    /// Grammar: pointer_prefix = { "[*:" ~ ws ~ expr ~ ws ~ "]" | "[*]" | "*" }
    fn parse_pointer_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let inner = pair.into_inner();
        let mut is_many = false;
        let mut is_const = false;
        let mut sentinel = None;
        let mut pointee = None;

        for item in inner {
            match item.as_rule() {
                Rule::pointer_prefix => {
                    // Parse the pointer prefix to determine is_many and sentinel
                    let prefix_str = item.as_str();
                    if prefix_str == "*" {
                        // Single pointer: *T
                        is_many = false;
                    } else if prefix_str == "[*]" {
                        // Many pointer without sentinel: [*]T
                        is_many = true;
                    } else {
                        // Sentinel-terminated many pointer: [*:expr]T
                        is_many = true;
                        // The prefix contains an expr for the sentinel value
                        for prefix_item in item.into_inner() {
                            if prefix_item.as_rule() == Rule::expr {
                                sentinel = Some(self.parse_expr(prefix_item)?);
                            }
                        }
                    }
                }
                Rule::type_prefix => {
                    pointee = Some(self.parse_type_prefix(item)?);
                }
                _ => {
                    // Check for "const" keyword
                    if item.as_str() == "const" {
                        is_const = true;
                    }
                }
            }
        }

        Ok(TypeExpr::Pointer(Box::new(PointerType {
            pointee: pointee.expect("pointer needs pointee"),
            is_const,
            is_many,
            sentinel,
        })))
    }

    /// Parse array type.
    fn parse_array_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let inner = pair.into_inner();
        let mut size = None;
        let mut sentinel = None;
        let mut element = None;

        for item in inner {
            match item.as_rule() {
                Rule::array_size => {
                    // array_size = { array_size_term ~ (ws ~ array_size_op ~ ws ~ array_size_term)* }
                    size = Some(self.parse_array_size_expr(item)?);
                }
                Rule::expr => {
                    // This is for sentinel: [N:sentinel]T
                    sentinel = Some(self.parse_expr(item)?);
                }
                Rule::type_prefix => {
                    // Grammar uses type_prefix for array element type
                    element = Some(self.parse_type_prefix(item)?);
                }
                _ => {}
            }
        }

        Ok(TypeExpr::Array(Box::new(ArrayType {
            element: element.expect("array needs element type"),
            size,
            sentinel,
        })))
    }

    /// Parse array size expression.
    /// array_size = { array_size_term ~ (ws ~ array_size_op ~ ws ~ array_size_term)* }
    fn parse_array_size_expr(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let location = self.location(&pair);
        let mut inner = pair.into_inner();

        // Parse the first term
        let first_term = inner.next().expect("array_size needs at least one term");
        let mut result = self.parse_array_size_term(first_term)?;

        // Parse remaining (op, term) pairs
        while let Some(op_pair) = inner.next() {
            if op_pair.as_rule() == Rule::array_size_op {
                let op = match op_pair.as_str() {
                    "+" => BinaryOp::Add,
                    "-" => BinaryOp::Sub,
                    "*" => BinaryOp::Mul,
                    "/" => BinaryOp::Div,
                    _ => unreachable!(),
                };

                let term_pair = inner.next().expect("operator needs right operand");
                let right = self.parse_array_size_term(term_pair)?;

                result = Expr::Binary(Box::new(BinaryExpr {
                    left: result,
                    op,
                    right,
                    location: Some(location.clone()),
                }));
            }
        }

        Ok(result)
    }

    /// Parse array size term.
    /// array_size_term = { number_literal | identifier | "(" ~ ws ~ array_size ~ ws ~ ")" }
    fn parse_array_size_term(&self, pair: Pair<Rule>) -> ParseResult<Expr> {
        let inner = pair
            .into_inner()
            .next()
            .expect("array_size_term needs content");
        match inner.as_rule() {
            Rule::number_literal | Rule::identifier => self.parse_primary_expr(inner),
            Rule::array_size => self.parse_array_size_expr(inner),
            _ => unreachable!("unexpected rule in array_size_term: {:?}", inner.as_rule()),
        }
    }

    /// Parse function type.
    fn parse_fn_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let inner = pair.into_inner();
        let mut params = Vec::new();
        let mut return_type = None;

        for item in inner {
            match item.as_rule() {
                Rule::type_list => {
                    // type_list = { type_prefix ~ (ws ~ "," ~ ws ~ type_prefix)* }
                    for ty in item.into_inner() {
                        if ty.as_rule() == Rule::type_prefix {
                            params.push(self.parse_type_prefix(ty)?);
                        }
                    }
                }
                Rule::type_prefix => {
                    // Return type is type_prefix
                    return_type = Some(self.parse_type_prefix(item)?);
                }
                _ => {}
            }
        }

        Ok(TypeExpr::Fn(Box::new(FnType {
            params,
            return_type,
        })))
    }

    /// Parse tuple type: (T1, T2) or (T1, T2, T3, ...)
    fn parse_tuple_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let mut elements = Vec::new();

        for item in pair.into_inner() {
            if item.as_rule() == Rule::type_prefix {
                elements.push(self.parse_type_prefix(item)?);
            }
        }

        Ok(TypeExpr::Tuple(elements))
    }

    /// Parse set type: Set(T)
    fn parse_set_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        // set_type = { "Set" ~ ws ~ "(" ~ ws ~ type_expr ~ ws ~ ")" }
        for item in pair.into_inner() {
            if item.as_rule() == Rule::type_expr {
                let element_type = self.parse_type_expr(item)?;
                return Ok(TypeExpr::Set(Box::new(element_type)));
            }
        }
        Err(ParseError {
            message: "Set type requires element type".to_string(),
            location: SourceLocation::default(),
        })
    }

    /// Parse builtin type.
    fn parse_builtin_type(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        // Check for Self type first (has inner rule)
        let inner = pair.clone().into_inner().next();
        if let Some(inner_pair) = inner
            && inner_pair.as_rule() == Rule::self_type
        {
            return Ok(TypeExpr::Named(TypePath {
                segments: vec!["Self".to_string()],
                location: Some(self.location(&pair)),
            }));
        }

        let s = pair.as_str();

        // Check for quantum types first
        match s {
            "qubit" => return Ok(TypeExpr::Qubit),
            "bit" => return Ok(TypeExpr::Bit),
            "Alloc" => return Ok(TypeExpr::QAlloc(None)),
            "unit" => return Ok(TypeExpr::Unit),
            "type" => return Ok(TypeExpr::Type),
            "anytype" => return Ok(TypeExpr::AnyType),
            "bool" => return Ok(TypeExpr::Primitive(PrimitiveType::Bool)),
            _ => {}
        }

        // Check for integer/float types
        if let Some(prim) = self.parse_primitive_type(s) {
            return Ok(TypeExpr::Primitive(prim));
        }

        Err(ParseError {
            message: format!("unknown builtin type: {}", s),
            location: self.location(&pair),
        })
    }

    /// Parse primitive type from string.
    /// Supports arbitrary bit-width integers like Zig: u1, u4, u7, u128, etc.
    fn parse_primitive_type(&self, s: &str) -> Option<PrimitiveType> {
        // Special cases first
        match s {
            "usize" => return Some(PrimitiveType::Usize),
            "isize" => return Some(PrimitiveType::Isize),
            "f16" => return Some(PrimitiveType::F16),
            "f32" => return Some(PrimitiveType::F32),
            "f64" => return Some(PrimitiveType::F64),
            "f128" => return Some(PrimitiveType::F128),
            "a64" => return Some(PrimitiveType::A64),
            "bool" => return Some(PrimitiveType::Bool),
            _ => {}
        }

        // Arbitrary-width integers: u<bits> or i<bits>
        // Valid bit widths are 1-128
        if let Some(bits_str) = s.strip_prefix('u') {
            if let Ok(bits) = bits_str.parse::<u16>()
                && (1..=128).contains(&bits)
            {
                return Some(PrimitiveType::UInt { bits });
            }
        } else if let Some(bits_str) = s.strip_prefix('i')
            && let Ok(bits) = bits_str.parse::<u16>()
            && (1..=128).contains(&bits)
        {
            return Some(PrimitiveType::IInt { bits });
        }

        None
    }

    /// Parse type identifier (named type).
    fn parse_type_identifier(&self, pair: Pair<Rule>) -> ParseResult<TypeExpr> {
        let mut segments = Vec::new();
        for item in pair.into_inner() {
            if item.as_rule() == Rule::identifier {
                segments.push(item.as_str().to_string());
            }
        }

        Ok(TypeExpr::Named(TypePath {
            segments,
            location: None,
        }))
    }

    /// Parse a channel argument (positional or named).
    fn parse_channel_arg(&self, pair: Pair<Rule>) -> ParseResult<ChannelArg> {
        // channel_arg = { (identifier ~ ws ~ ":" ~ ws ~ expr) | expr }
        let mut name = None;
        let mut value = None;

        for item in pair.into_inner() {
            match item.as_rule() {
                Rule::identifier => {
                    name = Some(item.as_str().to_string());
                }
                Rule::expr => {
                    value = Some(self.parse_expr(item)?);
                }
                _ => {}
            }
        }

        let expr = value.expect("channel_arg requires expression");

        if let Some(n) = name {
            Ok(ChannelArg::Named {
                name: n,
                value: expr,
            })
        } else {
            Ok(ChannelArg::Positional(expr))
        }
    }

    /// Parse a gate expression (either param_gate_expr or simple_gate_expr).
    fn parse_gate_expr_inner(
        &self,
        pair: Pair<Rule>,
        location: Option<SourceLocation>,
    ) -> ParseResult<Expr> {
        let mut gate_kind = None;
        let mut params = Vec::new();
        let mut target = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::param_gate_keyword | Rule::simple_gate_keyword => {
                    gate_kind = Some(self.parse_gate_keyword(inner.as_str())?);
                }
                Rule::gate_params => {
                    // gate_params = { "(" ~ ws ~ arg_list ~ ws ~ ")" }
                    for param_inner in inner.into_inner() {
                        if param_inner.as_rule() == Rule::arg_list {
                            for arg in param_inner.into_inner() {
                                if arg.as_rule() == Rule::expr {
                                    params.push(self.parse_expr(arg)?);
                                }
                            }
                        }
                    }
                }
                Rule::gate_target => {
                    // gate_target = { gate_set_target | tuple_expr | bracket_array | gate_qubit_target }
                    let target_inner = self.expect_inner(inner, "gate_target")?;
                    target = Some(match target_inner.as_rule() {
                        Rule::gate_set_target => {
                            // Parse as a set expression
                            let mut elements = Vec::new();
                            for elem in target_inner.into_inner() {
                                if elem.as_rule() == Rule::batch_elements {
                                    for e in elem.into_inner() {
                                        if e.as_rule() == Rule::expr {
                                            elements.push(self.parse_expr(e)?);
                                        }
                                    }
                                }
                            }
                            Expr::Set(Box::new(SetExpr {
                                elements,
                                element_type: None,
                                location: location.clone(),
                            }))
                        }
                        Rule::paren_or_tuple => self.parse_primary_expr(target_inner)?,
                        Rule::bracket_array => self.parse_primary_expr(target_inner)?,
                        Rule::postfix_expr => self.parse_postfix_expr(target_inner)?,
                        Rule::gate_qubit_target => {
                            // gate_qubit_target = { !operator_keyword ~ postfix_expr }
                            // Just parse the inner postfix_expr
                            let inner_expr =
                                self.expect_inner(target_inner, "gate_qubit_target")?;
                            self.parse_postfix_expr(inner_expr)?
                        }
                        other => {
                            return Err(ParseError {
                                message: format!(
                                    "unexpected rule {:?}, expected gate_target",
                                    other
                                ),
                                location: location.clone().unwrap_or_default(),
                            });
                        }
                    });
                }
                _ => {}
            }
        }

        Ok(Expr::Gate(Box::new(GateExpr {
            kind: gate_kind.expect("gate requires keyword"),
            params,
            target: target.expect("gate requires target"),
            location,
        })))
    }

    /// Parse gate keyword to GateKind.
    fn parse_gate_keyword(&self, s: &str) -> ParseResult<GateKind> {
        use GateKind::*;
        match s {
            // Single-qubit Pauli gates
            "x" => Ok(X),
            "y" => Ok(Y),
            "z" => Ok(Z),
            // Hadamard
            "h" => Ok(H),
            // T gates (fourth root of Z)
            "t" => Ok(T),
            "tdg" => Ok(Tdg),
            // Square root gates
            "sx" => Ok(SX),
            "sy" => Ok(SY),
            "sz" => Ok(SZ),
            "sxdg" => Ok(SXdg),
            "sydg" => Ok(SYdg),
            "szdg" => Ok(SZdg),
            // Rotation gates
            "rx" => Ok(RX),
            "ry" => Ok(RY),
            "rz" => Ok(RZ),
            // Two-qubit gates
            "cx" => Ok(CX),
            "cy" => Ok(CY),
            "cz" => Ok(CZ),
            "ch" => Ok(CH),
            // Two-qubit rotation gates
            "sxx" => Ok(SXX),
            "syy" => Ok(SYY),
            "szz" => Ok(SZZ),
            "sxxdg" => Ok(SXXdg),
            "syydg" => Ok(SYYdg),
            "szzdg" => Ok(SZZdg),
            "rzz" => Ok(RZZ),
            "crz" => Ok(RZZ), // CRZ is effectively RZZ
            // Swap gates
            "swap" => Ok(SWAP),
            "iswap" => Ok(ISWAP),
            // Three-qubit gates
            "ccx" => Ok(CCX), // Toffoli gate
            // Face rotations
            "f" => Ok(F),
            "fdg" => Ok(Fdg),
            "f4" => Ok(F4),
            "f4dg" => Ok(F4dg),
            // Prepare operation
            "pz" => Ok(PZ),
            other => {
                let suggestion = suggest_gate_name(other);
                let message = match suggestion {
                    Some(name) => format!("unknown gate '{}', did you mean '{}'?", other, name),
                    None => format!("unknown gate '{}'", other),
                };
                Err(ParseError {
                    message,
                    location: SourceLocation::default(),
                })
            }
        }
    }
}

/// Known gate names for suggestions.
const KNOWN_GATE_NAMES: &[&str] = &[
    "x", "y", "z", "h", "t", "tdg", "sx", "sy", "sz", "sxdg", "sydg", "szdg", "rx", "ry", "rz",
    "cx", "cy", "cz", "ch", "sxx", "syy", "szz", "sxxdg", "syydg", "szzdg", "rzz", "crz", "swap",
    "iswap", "ccx", "f", "fdg", "f4", "f4dg", "pz",
];

/// Deprecated gate name mappings.
const DEPRECATED_GATES: &[(&str, &str)] = &[("s", "sz"), ("sdg", "szdg")];

/// Suggest a gate name for a misspelled or deprecated gate keyword.
pub fn suggest_gate_name(unknown: &str) -> Option<&'static str> {
    // Check deprecated names first
    for &(old, new) in DEPRECATED_GATES {
        if unknown == old {
            return Some(new);
        }
    }

    // Find closest match by edit distance
    let mut best: Option<(&str, usize)> = None;
    for &name in KNOWN_GATE_NAMES {
        let dist = edit_distance(unknown, name);
        if dist <= 2 {
            match best {
                Some((_, best_dist)) if dist < best_dist => best = Some((name, dist)),
                None => best = Some((name, dist)),
                _ => {}
            }
        }
    }
    best.map(|(name, _)| name)
}

/// Compute the Levenshtein edit distance between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    // Use single-row optimization
    let mut prev = vec![0usize; n + 1];
    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        let mut curr = vec![0usize; n + 1];
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = curr;
    }

    prev[n]
}

/// Parse a Zluppy source string.
pub fn parse(source: &str) -> ParseResult<Program> {
    log::debug!("Parsing {} bytes of source", source.len());
    let result = ParserState::new(source).parse();
    match &result {
        Ok(program) => {
            log::debug!(
                "Parsed {} top-level declarations",
                program.declarations.len()
            );
            log::trace!("AST: {:?}", program);
        }
        Err(e) => {
            log::debug!("Parse error: {}", e);
        }
    }
    result
}

/// Parse a Zluppy source file.
pub fn parse_file(source: &str, filename: impl Into<String>) -> ParseResult<Program> {
    let filename = filename.into();
    log::debug!("Parsing file '{}' ({} bytes)", filename, source.len());
    let result = ParserState::new(source).with_file(&filename).parse();
    match &result {
        Ok(program) => {
            log::debug!(
                "Parsed {} top-level declarations from '{}'",
                program.declarations.len(),
                filename
            );
        }
        Err(e) => {
            log::debug!("Parse error in '{}': {}", filename, e);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty() {
        let result = parse("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_const() {
        // Immutable binding with explicit type
        let result = parse("x: u32 = 42;");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_function() {
        let result = parse("fn main() -> unit { return unit; }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_binding_function_call() {
        // Mutable binding with inferred type
        let source = "mut q := qalloc(2);";
        let result = parse(source);
        assert!(result.is_ok(), "Parse failed: {:?}", result);
        let program = result.unwrap();

        // Check that there's exactly one declaration
        assert_eq!(program.declarations.len(), 1, "Expected 1 declaration");

        // Check that it's a binding declaration
        if let TopLevelDecl::Binding(binding) = &program.declarations[0] {
            assert_eq!(binding.name, "q");
            assert!(binding.is_mutable);
            // Check that the value is a Call expression
            if let Some(Expr::Call(call)) = &binding.value {
                if let Expr::Ident(ident) = &call.callee {
                    assert_eq!(ident.name, "qalloc");
                } else {
                    panic!("Callee should be Ident, got: {:?}", call.callee);
                }
                assert_eq!(call.args.len(), 1, "Expected 1 argument");
            } else {
                panic!("Value should be Call, got: {:?}", binding.value);
            }
        } else {
            panic!(
                "Should be a Binding declaration, got: {:?}",
                program.declarations[0]
            );
        }
    }

    #[test]
    fn test_parse_true_and_expr() {
        // Test that `true and y` parses as an expression
        let source = "x := true and y;\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse failed: {:?}", result);
    }

    #[test]
    fn test_parse_just_true() {
        // Test just `true` as expression - this works
        let source = "x := true;\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse failed: {:?}", result);
    }

    #[test]
    fn test_parse_ident_and_true() {
        // Test `y and true` - boolean on right works
        let source = "x := y and true;\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse failed: {:?}", result);
    }

    // =========================================================================
    // Input Size Limit Tests
    // =========================================================================

    #[test]
    fn test_max_source_size_constant() {
        // Verify the constant is reasonable (10MB)
        assert_eq!(MAX_SOURCE_SIZE, 10 * 1024 * 1024);
    }

    #[test]
    fn test_source_size_limit_enforced() {
        // Create a source that exceeds the limit
        let large_source = "x".repeat(MAX_SOURCE_SIZE + 1);
        let result = parse(&large_source);
        assert!(result.is_err(), "Expected error for oversized source");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("too large"),
            "Expected 'too large' in error message: {}",
            err.message
        );
    }

    #[test]
    fn test_normal_source_within_limit() {
        // Normal source should parse fine
        let source = "fn main() -> unit { return unit; }";
        assert!(source.len() < MAX_SOURCE_SIZE);
        let result = parse(source);
        assert!(
            result.is_ok(),
            "Expected normal source to parse: {:?}",
            result
        );
    }
}
