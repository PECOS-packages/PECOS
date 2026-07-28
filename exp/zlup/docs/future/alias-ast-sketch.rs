// =============================================================================
// AST Sketch for Alias Feature
// =============================================================================
// This is a design sketch, not actual implementation code.

/// Alias binding - creates a named view into existing data.
///
/// Syntax:
/// - `alias name := source;`           - immutable alias
/// - `mut alias name := source;`       - mutable alias (can be reassigned)
/// - `alias { a := x, b := y, ... }`   - grouped aliases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasBinding {
    /// Name of the alias
    pub name: String,
    /// The source expression (typically a slice expression)
    pub source: Expr,
    /// Whether this alias can be reassigned to point elsewhere
    pub is_mutable: bool,
    /// Optional type annotation
    pub ty: Option<TypeExpr>,
    /// Documentation comment
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Grouped alias declaration for partitioning.
///
/// Syntax:
/// ```zlup
/// alias {
///     data := q[0..4],
///     ancilla := q[4..8],
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasGroup {
    /// The aliases in this group (checked for non-overlap)
    pub aliases: Vec<AliasBinding>,
    pub location: Option<SourceLocation>,
}

// In Stmt enum, add:
pub enum Stmt {
    // ... existing variants ...

    /// Single alias binding
    Alias(AliasBinding),

    /// Grouped aliases (enables overlap checking within group)
    AliasGroup(AliasGroup),
}

// =============================================================================
// Semantic Analysis Additions
// =============================================================================

/// Information about an alias tracked during semantic analysis.
#[derive(Debug, Clone)]
pub struct AliasInfo {
    /// Name of the alias
    pub name: String,
    /// Name of the source variable
    pub source_var: String,
    /// Static range if known at compile time
    pub static_range: Option<StaticRange>,
    /// Whether this alias is mutable
    pub is_mutable: bool,
    /// The type of elements in the alias
    pub element_type: Type,
    /// Location for error reporting
    pub location: Option<SourceLocation>,
}

/// A compile-time known range.
#[derive(Debug, Clone)]
pub struct StaticRange {
    pub start: i128,
    pub end: i128,  // exclusive
}

impl StaticRange {
    pub fn overlaps(&self, other: &StaticRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    pub fn len(&self) -> i128 {
        self.end - self.start
    }
}

// Add to SymbolKind enum:
pub enum SymbolKind {
    // ... existing variants ...

    /// An alias (named view into another variable)
    Alias {
        /// Type of the alias (slice type)
        ty: Type,
        /// The source variable this aliases
        source: String,
        /// Static range if known
        range: Option<StaticRange>,
        /// Whether the alias binding is mutable
        is_mutable: bool,
    },
}

// =============================================================================
// Overlap Checking
// =============================================================================

impl SemanticAnalyzer {
    /// Check if a new alias overlaps with existing aliases to the same source.
    fn check_alias_overlap(&self, new_alias: &AliasInfo) -> SemanticResult<()> {
        // Find all existing aliases to the same source
        for (name, symbol) in self.iter_symbols() {
            if let SymbolKind::Alias { source, range, is_mutable, .. } = &symbol.kind {
                if source == &new_alias.source_var {
                    // Same source - check for overlap
                    if let (Some(existing_range), Some(new_range)) = (range, &new_alias.static_range) {
                        if existing_range.overlaps(new_range) {
                            // Overlap detected
                            if *is_mutable || new_alias.is_mutable {
                                // At least one is mutable - error
                                return Err(SemanticError::OverlappingMutableAlias {
                                    new_alias: new_alias.name.clone(),
                                    existing_alias: name.clone(),
                                    source: new_alias.source_var.clone(),
                                    location: new_alias.location.clone().unwrap_or_default(),
                                });
                            }
                            // Both immutable - could warn but allow
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Analyze an alias binding.
    fn analyze_alias(&mut self, alias: &AliasBinding) -> SemanticResult<()> {
        // 1. Analyze the source expression
        let source_ty = self.analyze_expr(&alias.source)?;

        // 2. Verify source is slice-able
        let element_ty = match &source_ty {
            Type::Slice { element, .. } => *element.clone(),
            Type::Array { element, .. } => *element.clone(),
            _ => {
                return Err(SemanticError::TypeMismatch {
                    expected: "slice or array".to_string(),
                    found: source_ty.display_name(),
                    location: alias.location.clone().unwrap_or_default(),
                });
            }
        };

        // 3. Extract source variable name and range (if static)
        let (source_var, static_range) = self.extract_alias_source_info(&alias.source)?;

        // 4. Check for overlaps with existing aliases
        let alias_info = AliasInfo {
            name: alias.name.clone(),
            source_var: source_var.clone(),
            static_range: static_range.clone(),
            is_mutable: alias.is_mutable,
            element_type: element_ty.clone(),
            location: alias.location.clone(),
        };
        self.check_alias_overlap(&alias_info)?;

        // 5. Define the alias in the symbol table
        let alias_ty = Type::Slice {
            element: Box::new(element_ty),
            is_mut: alias.is_mutable,
        };

        self.symbols.define(Symbol {
            name: alias.name.clone(),
            kind: SymbolKind::Alias {
                ty: alias_ty,
                source: source_var,
                range: static_range,
                is_mutable: alias.is_mutable,
            },
            location: alias.location.clone(),
        })?;

        Ok(())
    }

    /// Extract source variable and static range from a slice expression.
    fn extract_alias_source_info(&self, expr: &Expr) -> SemanticResult<(String, Option<StaticRange>)> {
        match expr {
            Expr::Slice(slice) => {
                // Get the base variable name
                let source_var = match &*slice.base {
                    Expr::Ident(ident) => ident.name.clone(),
                    _ => return Ok(("<complex>".to_string(), None)),
                };

                // Try to evaluate range bounds at compile time
                let start = self.try_eval_comptime_int(&slice.start);
                let end = self.try_eval_comptime_int(&slice.end);

                let static_range = match (start, end) {
                    (Some(s), Some(e)) => Some(StaticRange { start: s, end: e }),
                    _ => None,
                };

                Ok((source_var, static_range))
            }
            Expr::Ident(ident) => {
                // Aliasing entire variable - no range restriction
                Ok((ident.name.clone(), None))
            }
            _ => Ok(("<complex>".to_string(), None)),
        }
    }
}

// =============================================================================
// New Semantic Error Variant
// =============================================================================

pub enum SemanticError {
    // ... existing variants ...

    #[error("mutable alias '{new_alias}' overlaps with '{existing_alias}' (both alias '{source}')")]
    OverlappingMutableAlias {
        new_alias: String,
        existing_alias: String,
        source: String,
        location: SourceLocation,
    },

    #[error("alias '{name}' would outlive its source")]
    AliasOutlivesSource {
        name: String,
        location: SourceLocation,
    },
}

// =============================================================================
// Parser Grammar Sketch (PEG)
// =============================================================================

/*
// Add to statement rule:
statement = {
    // ... existing ...
    | alias_stmt
    | alias_group
}

alias_stmt = {
    mut_modifier? ~ "alias" ~ ws ~ identifier ~ ws ~
    (":" ~ ws ~ type_expr ~ ws)? ~
    ":=" ~ ws ~ expr ~ ";"
}

alias_group = {
    "alias" ~ ws ~ "{" ~ ws ~
    (alias_item ~ ("," ~ ws ~ alias_item)* ~ ","?)? ~
    ws ~ "}"
}

alias_item = {
    identifier ~ ws ~ (":" ~ ws ~ type_expr ~ ws)? ~ ":=" ~ ws ~ expr
}

mut_modifier = { "mut" ~ ws }
*/

// =============================================================================
// Code Generation (compiles to slice)
// =============================================================================

// Aliases compile directly to slice references - no runtime overhead.
// The alias keyword is purely for semantic analysis and safety checking.

impl CodeGen {
    fn gen_alias(&mut self, alias: &AliasBinding) -> Result<()> {
        // Generate code for the source expression (produces a slice)
        let source_value = self.gen_expr(&alias.source)?;

        // Bind the slice to the alias name
        self.define_local(&alias.name, source_value);

        Ok(())
    }
}
