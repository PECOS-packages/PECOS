//! Documentation generator for Zluppy programs.
//!
//! Extracts doc comments from AST nodes and produces Markdown documentation.

use crate::ast::*;

/// Configuration for documentation generation.
#[derive(Debug, Clone)]
pub struct DocConfig {
    /// Include private (non-pub) items
    pub include_private: bool,
    /// Show source locations in output
    pub show_locations: bool,
}

impl Default for DocConfig {
    fn default() -> Self {
        Self {
            include_private: false,
            show_locations: false,
        }
    }
}

/// Kind of documented item.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocItemKind {
    Function,
    ExternFunction,
    Struct,
    Enum,
    Union,
    ErrorSet,
    FaultSet,
    Constant,
    Test,
}

impl DocItemKind {
    /// Section heading for this kind.
    pub fn section_heading(&self) -> &'static str {
        match self {
            DocItemKind::Function => "Functions",
            DocItemKind::ExternFunction => "Extern Functions",
            DocItemKind::Struct => "Structs",
            DocItemKind::Enum => "Enums",
            DocItemKind::Union => "Unions",
            DocItemKind::ErrorSet => "Error Sets",
            DocItemKind::FaultSet => "Fault Sets",
            DocItemKind::Constant => "Constants",
            DocItemKind::Test => "Tests",
        }
    }
}

/// A documented item extracted from the AST.
#[derive(Debug, Clone)]
pub struct DocItem {
    /// Kind of item
    pub kind: DocItemKind,
    /// Item name
    pub name: String,
    /// Signature or declaration line
    pub signature: String,
    /// Doc comment text (may be multi-line)
    pub doc: Option<String>,
    /// Child items (struct fields, enum variants, etc.)
    pub children: Vec<DocChild>,
    /// Whether this item is public
    pub is_pub: bool,
    /// Source location
    pub location: Option<SourceLocation>,
}

/// A child of a documented item (field, variant, etc.).
#[derive(Debug, Clone)]
pub struct DocChild {
    /// Child name
    pub name: String,
    /// Type or value description
    pub description: String,
    /// Doc comment
    pub doc: Option<String>,
}

/// Extract documented items from a program AST.
pub fn extract_doc_items(program: &Program, config: &DocConfig) -> Vec<DocItem> {
    let mut items = Vec::new();

    for decl in &program.declarations {
        match decl {
            TopLevelDecl::Fn(f) => {
                if !config.include_private && !f.is_pub {
                    continue;
                }
                items.push(extract_fn_doc(f));
            }
            TopLevelDecl::ExternFn(f) => {
                if !config.include_private && !f.is_pub {
                    continue;
                }
                items.push(extract_extern_fn_doc(f));
            }
            TopLevelDecl::Struct(s) => {
                if !config.include_private && !s.is_pub {
                    continue;
                }
                items.push(extract_struct_doc(s));
            }
            TopLevelDecl::Enum(e) => {
                if !config.include_private && !e.is_pub {
                    continue;
                }
                items.push(extract_enum_doc(e));
            }
            TopLevelDecl::Union(u) => {
                if !config.include_private && !u.is_pub {
                    continue;
                }
                items.push(extract_union_doc(u));
            }
            TopLevelDecl::ErrorSet(e) => {
                if !config.include_private && !e.is_pub {
                    continue;
                }
                items.push(extract_error_set_doc(e));
            }
            TopLevelDecl::FaultSet(f) => {
                if !config.include_private && !f.is_pub {
                    continue;
                }
                items.push(extract_fault_set_doc(f));
            }
            TopLevelDecl::Binding(b) => {
                if !config.include_private && !b.is_pub {
                    continue;
                }
                items.push(extract_binding_doc(b));
            }
            TopLevelDecl::Test(t) => {
                if config.include_private {
                    items.push(extract_test_doc(t));
                }
            }
            TopLevelDecl::DeclareGate(_) | TopLevelDecl::Gate(_) => {
                // Custom gate declarations are not yet included in documentation
            }
        }
    }

    items
}

fn format_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|p| format!("{}: {}", p.name, format_type_expr(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_type_expr(ty: &TypeExpr) -> String {
    // Simplified type expression formatting
    format!("{:?}", ty).chars().take(80).collect()
}

fn extract_fn_doc(f: &FnDecl) -> DocItem {
    let ret = f
        .return_type
        .as_ref()
        .map(|t| format!(" -> {}", format_type_expr(t)))
        .unwrap_or_default();
    let sig = format!("fn {}({}){}", f.name, format_params(&f.params), ret);

    DocItem {
        kind: DocItemKind::Function,
        name: f.name.clone(),
        signature: sig,
        doc: f.doc_comment.clone(),
        children: Vec::new(),
        is_pub: f.is_pub,
        location: f.location.clone(),
    }
}

fn extract_extern_fn_doc(f: &ExternFnDecl) -> DocItem {
    let ret = f
        .return_type
        .as_ref()
        .map(|t| format!(" -> {}", format_type_expr(t)))
        .unwrap_or_default();
    let sig = format!(
        "extern \"{}\" fn {}({}){}",
        f.calling_convention,
        f.name,
        format_params(&f.params),
        ret
    );

    DocItem {
        kind: DocItemKind::ExternFunction,
        name: f.name.clone(),
        signature: sig,
        doc: f.doc_comment.clone(),
        children: Vec::new(),
        is_pub: f.is_pub,
        location: f.location.clone(),
    }
}

fn extract_struct_doc(s: &StructDecl) -> DocItem {
    let children = s
        .fields
        .iter()
        .map(|f| DocChild {
            name: f.name.clone(),
            description: format_type_expr(&f.ty),
            doc: f.doc_comment.clone(),
        })
        .collect();

    DocItem {
        kind: DocItemKind::Struct,
        name: s.name.clone(),
        signature: format!("struct {}", s.name),
        doc: s.doc_comment.clone(),
        children,
        is_pub: s.is_pub,
        location: s.location.clone(),
    }
}

fn extract_enum_doc(e: &EnumDecl) -> DocItem {
    let children = e
        .variants
        .iter()
        .map(|v| DocChild {
            name: v.name.clone(),
            description: String::new(),
            doc: None,
        })
        .collect();

    DocItem {
        kind: DocItemKind::Enum,
        name: e.name.clone(),
        signature: format!("enum {}", e.name),
        doc: e.doc_comment.clone(),
        children,
        is_pub: e.is_pub,
        location: e.location.clone(),
    }
}

fn extract_union_doc(u: &UnionDecl) -> DocItem {
    let children = u
        .fields
        .iter()
        .map(|f| {
            let desc =
                f.ty.as_ref()
                    .map(|t| format_type_expr(t))
                    .unwrap_or_default();
            DocChild {
                name: f.name.clone(),
                description: desc,
                doc: None,
            }
        })
        .collect();

    DocItem {
        kind: DocItemKind::Union,
        name: u.name.clone(),
        signature: format!("union {}", u.name),
        doc: u.doc_comment.clone(),
        children,
        is_pub: u.is_pub,
        location: u.location.clone(),
    }
}

fn extract_error_set_doc(e: &ErrorSetDecl) -> DocItem {
    let children = e
        .variants
        .iter()
        .map(|v| DocChild {
            name: v.name.clone(),
            description: String::new(),
            doc: None,
        })
        .collect();

    DocItem {
        kind: DocItemKind::ErrorSet,
        name: e.name.clone(),
        signature: format!("{} := error {{ ... }}", e.name),
        doc: e.doc_comment.clone(),
        children,
        is_pub: e.is_pub,
        location: e.location.clone(),
    }
}

fn extract_fault_set_doc(f: &FaultSetDecl) -> DocItem {
    let children = f
        .variants
        .iter()
        .map(|v| DocChild {
            name: v.name.clone(),
            description: String::new(),
            doc: None,
        })
        .collect();

    DocItem {
        kind: DocItemKind::FaultSet,
        name: f.name.clone(),
        signature: format!("{} := fault {{ ... }}", f.name),
        doc: f.doc_comment.clone(),
        children,
        is_pub: f.is_pub,
        location: f.location.clone(),
    }
}

fn extract_binding_doc(b: &Binding) -> DocItem {
    let sig = if b.is_mutable {
        format!("mut {}", b.name)
    } else {
        b.name.clone()
    };

    DocItem {
        kind: DocItemKind::Constant,
        name: b.name.clone(),
        signature: sig,
        doc: b.doc_comment.clone(),
        children: Vec::new(),
        is_pub: b.is_pub,
        location: b.location.clone(),
    }
}

fn extract_test_doc(t: &TestDecl) -> DocItem {
    DocItem {
        kind: DocItemKind::Test,
        name: t.name.clone(),
        signature: format!("test \"{}\"", t.name),
        doc: None,
        children: Vec::new(),
        is_pub: false,
        location: t.location.clone(),
    }
}

/// Generate Markdown documentation from extracted doc items.
pub fn generate_markdown(items: &[DocItem], module_name: &str) -> String {
    let mut out = String::new();

    out.push_str(&format!("# {}\n\n", module_name));

    // Group by kind, in order
    let mut kinds: Vec<DocItemKind> = items.iter().map(|i| i.kind.clone()).collect();
    kinds.sort();
    kinds.dedup();

    for kind in kinds {
        let kind_items: Vec<_> = items.iter().filter(|i| i.kind == kind).collect();
        if kind_items.is_empty() {
            continue;
        }

        out.push_str(&format!("## {}\n\n", kind.section_heading()));

        for item in kind_items {
            out.push_str(&format!("### {}\n\n", item.name));
            out.push_str(&format!("```zluppy\n{}\n```\n\n", item.signature));

            if let Some(ref doc) = item.doc {
                out.push_str(doc.trim());
                out.push_str("\n\n");
            }

            if !item.children.is_empty() {
                out.push_str("| Name | Type | Description |\n");
                out.push_str("|------|------|-------------|\n");
                for child in &item.children {
                    let doc_str = child.doc.as_deref().unwrap_or("");
                    out.push_str(&format!(
                        "| `{}` | `{}` | {} |\n",
                        child.name, child.description, doc_str
                    ));
                }
                out.push('\n');
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    // NOTE: The pest grammar currently treats `///` as a regular line comment
    // (consumed by the implicit COMMENT rule), so doc_comment fields in AST
    // nodes are always None. The docgen infrastructure is ready to use them
    // once the grammar is fixed to distinguish doc comments from regular comments.

    #[test]
    fn test_extract_pub_function() {
        let source = r#"
            pub fn add(a: u32, b: u32) -> u32 {
                return a + b;
            }
        "#;
        let program = parse(source).unwrap();
        let config = DocConfig::default();
        let items = extract_doc_items(&program, &config);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, DocItemKind::Function);
        assert_eq!(items[0].name, "add");
    }

    #[test]
    fn test_extract_constant_binding() {
        let source = r#"
            pub Point := struct {
                x: f64,
                y: f64,
            };
        "#;
        let program = parse(source).unwrap();
        let config = DocConfig::default();
        let items = extract_doc_items(&program, &config);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, DocItemKind::Constant);
        assert_eq!(items[0].name, "Point");
    }

    #[test]
    fn test_private_items_excluded() {
        let source = r#"
            pub fn visible() -> unit { return; }
            fn hidden() -> unit { return; }
        "#;
        let program = parse(source).unwrap();

        let config = DocConfig::default();
        let items = extract_doc_items(&program, &config);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "visible");
    }

    #[test]
    fn test_private_items_included_with_flag() {
        let source = r#"
            pub fn visible() -> unit { return; }
            fn hidden() -> unit { return; }
        "#;
        let program = parse(source).unwrap();

        let config = DocConfig {
            include_private: true,
            ..Default::default()
        };
        let items = extract_doc_items(&program, &config);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_no_doc_comment() {
        let source = r#"
            pub fn no_doc() -> unit { return; }
        "#;
        let program = parse(source).unwrap();
        let config = DocConfig::default();
        let items = extract_doc_items(&program, &config);
        assert_eq!(items.len(), 1);
        // Doc comment is None since the grammar eats /// as regular comments
        assert!(items[0].doc.is_none());
    }

    #[test]
    fn test_extract_error_set() {
        let source = r#"
            pub MyError := error { Timeout, InvalidInput };
        "#;
        let program = parse(source).unwrap();
        let config = DocConfig::default();
        let items = extract_doc_items(&program, &config);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, DocItemKind::ErrorSet);
        assert_eq!(items[0].children.len(), 2);
    }

    #[test]
    fn test_generate_markdown_basic() {
        let items = vec![DocItem {
            kind: DocItemKind::Function,
            name: "main".to_string(),
            signature: "fn main() -> unit".to_string(),
            doc: Some("Entry point.".to_string()),
            children: Vec::new(),
            is_pub: true,
            location: None,
        }];
        let md = generate_markdown(&items, "my_module");
        assert!(md.contains("# my_module"));
        assert!(md.contains("## Functions"));
        assert!(md.contains("### main"));
        assert!(md.contains("Entry point."));
    }

    #[test]
    fn test_generate_markdown_with_children() {
        let items = vec![DocItem {
            kind: DocItemKind::Struct,
            name: "Point".to_string(),
            signature: "struct Point".to_string(),
            doc: None,
            children: vec![
                DocChild {
                    name: "x".to_string(),
                    description: "f64".to_string(),
                    doc: None,
                },
                DocChild {
                    name: "y".to_string(),
                    description: "f64".to_string(),
                    doc: None,
                },
            ],
            is_pub: true,
            location: None,
        }];
        let md = generate_markdown(&items, "geometry");
        assert!(md.contains("## Structs"));
        assert!(md.contains("| `x` |"));
        assert!(md.contains("| `y` |"));
    }
}
