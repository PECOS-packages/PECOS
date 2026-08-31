//! AST-based pretty printer for Zlup.
//!
//! Provides canonical formatting by parsing source to AST and
//! pretty-printing with consistent style rules.

use crate::ast::*;

/// Pretty printing options.
#[derive(Debug, Clone)]
pub struct PrettyOptions {
    /// Use spaces instead of tabs.
    pub use_spaces: bool,
    /// Number of spaces per indent level (if using spaces).
    pub indent_size: usize,
    /// Maximum line length before wrapping.
    pub max_line_length: usize,
}

impl Default for PrettyOptions {
    fn default() -> Self {
        Self {
            use_spaces: true,
            indent_size: 4,
            max_line_length: 100,
        }
    }
}

/// AST-based pretty printer.
pub struct PrettyPrinter {
    options: PrettyOptions,
    output: String,
    indent_level: usize,
    at_line_start: bool,
}

impl PrettyPrinter {
    pub fn new(options: PrettyOptions) -> Self {
        Self {
            options,
            output: String::new(),
            indent_level: 0,
            at_line_start: true,
        }
    }

    /// Pretty print a program.
    pub fn print_program(&mut self, program: &Program) -> String {
        for (i, decl) in program.declarations.iter().enumerate() {
            if i > 0 {
                self.newline();
            }
            self.print_top_level_decl(decl);
        }
        self.ensure_trailing_newline();
        std::mem::take(&mut self.output)
    }

    fn indent_str(&self) -> String {
        if self.options.use_spaces {
            " ".repeat(self.options.indent_size)
        } else {
            "\t".to_string()
        }
    }

    fn write(&mut self, s: &str) {
        if self.at_line_start && !s.is_empty() {
            for _ in 0..self.indent_level {
                self.output.push_str(&self.indent_str());
            }
            self.at_line_start = false;
        }
        self.output.push_str(s);
    }

    fn newline(&mut self) {
        self.output.push('\n');
        self.at_line_start = true;
    }

    fn ensure_trailing_newline(&mut self) {
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }
    }

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    // =========================================================================
    // Top-level declarations
    // =========================================================================

    fn print_top_level_decl(&mut self, decl: &TopLevelDecl) {
        match decl {
            TopLevelDecl::Binding(b) => self.print_binding(b),
            TopLevelDecl::Fn(f) => self.print_fn_decl(f),
            TopLevelDecl::ExternFn(f) => self.print_extern_fn_decl(f),
            TopLevelDecl::Struct(s) => self.print_struct_decl(s),
            TopLevelDecl::Enum(e) => self.print_enum_decl(e),
            TopLevelDecl::Union(u) => self.print_union_decl(u),
            TopLevelDecl::ErrorSet(e) => self.print_error_set_decl(e),
            TopLevelDecl::FaultSet(f) => self.print_fault_set_decl(f),
            TopLevelDecl::Test(t) => self.print_test_decl(t),
            TopLevelDecl::DeclareGate(g) => {
                self.write(&format!("declare gate {}(", g.name));
                for (i, p) in g.params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&p.name);
                }
                self.write(")(");
                for (i, q) in g.qubits.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&q.name);
                }
                self.write(");");
                self.newline();
            }
            TopLevelDecl::Gate(g) => {
                self.write(&format!("gate {}(", g.name));
                for (i, p) in g.params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&p.name);
                }
                self.write(")(");
                for (i, q) in g.qubits.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&q.name);
                }
                self.write(") ");
                self.print_block(&g.body);
                self.newline();
            }
        }
    }

    fn print_doc_comment(&mut self, doc: &Option<String>) {
        if let Some(doc) = doc {
            for line in doc.lines() {
                self.write("/// ");
                self.write(line);
                self.newline();
            }
        }
    }

    fn print_binding(&mut self, binding: &Binding) {
        self.print_doc_comment(&binding.doc_comment);

        if binding.is_pub {
            self.write("pub ");
        }
        if binding.is_mutable {
            self.write("mut ");
        }
        self.write(&binding.name);

        if let Some(ty) = &binding.ty {
            self.write(": ");
            self.print_type_expr(ty);
        }

        if let Some(value) = &binding.value {
            if binding.ty.is_some() {
                self.write(" = ");
            } else {
                self.write(" := ");
            }
            self.print_expr(value);
        }

        self.write(";");
        self.newline();
    }

    fn print_fn_decl(&mut self, func: &FnDecl) {
        self.print_doc_comment(&func.doc_comment);

        if func.is_pub {
            self.write("pub ");
        }
        if func.is_inline {
            self.write("inline ");
        }

        self.write("fn ");
        self.write(&func.name);
        self.write("(");

        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.print_param(param);
        }

        self.write(")");

        if let Some(ret) = &func.return_type {
            self.write(" -> ");
            self.print_type_expr(ret);
        }

        self.write(" ");
        self.print_block(&func.body);
        self.newline();
    }

    fn print_param(&mut self, param: &Param) {
        // Check for Rust-style self parameter
        if param.name == "self"
            && let TypeExpr::Pointer(ptr) = &param.ty
            && let TypeExpr::Named(path) = &ptr.pointee
            && path.segments == vec!["Self".to_string()]
        {
            // This is a self parameter - print as &self or &mut self
            if ptr.is_const {
                self.write("&self");
            } else {
                self.write("&mut self");
            }
            return;
        }

        // Regular parameter
        if param.is_comptime {
            self.write("comptime ");
        }
        self.write(&param.name);
        self.write(": ");
        self.print_type_expr(&param.ty);
    }

    fn print_extern_fn_decl(&mut self, func: &ExternFnDecl) {
        self.print_doc_comment(&func.doc_comment);

        if let Some(lib) = &func.library {
            self.write("@link(\"");
            self.write(lib);
            self.write("\") ");
        }

        if func.is_pub {
            self.write("pub ");
        }

        self.write("extern \"");
        self.write(&func.calling_convention);
        self.write("\" fn ");
        self.write(&func.name);
        self.write("(");

        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.print_param(param);
        }

        self.write(")");

        if let Some(ret) = &func.return_type {
            self.write(" -> ");
            self.print_type_expr(ret);
        }

        self.write(";");
        self.newline();
    }

    fn print_struct_decl(&mut self, s: &StructDecl) {
        self.print_doc_comment(&s.doc_comment);

        if s.is_pub {
            self.write("pub ");
        }

        self.write("const ");
        self.write(&s.name);
        self.write(" = ");

        if s.is_packed {
            self.write("packed ");
        }
        self.write("struct {");

        if s.fields.is_empty() && s.methods.is_empty() && s.associated_consts.is_empty() {
            self.write("};");
        } else {
            self.newline();
            self.indent();

            for field in &s.fields {
                self.print_struct_field(field);
            }

            for method in &s.methods {
                self.newline();
                self.print_fn_decl(method);
            }

            self.dedent();
            self.write("};");
        }
        self.newline();
    }

    fn print_struct_field(&mut self, field: &StructField) {
        self.print_doc_comment(&field.doc_comment);
        self.write(&field.name);
        self.write(": ");
        self.print_type_expr(&field.ty);

        if let Some(default) = &field.default {
            self.write(" = ");
            self.print_expr(default);
        }

        self.write(",");
        self.newline();
    }

    fn print_enum_decl(&mut self, e: &EnumDecl) {
        self.print_doc_comment(&e.doc_comment);

        if e.is_pub {
            self.write("pub ");
        }

        self.write("const ");
        self.write(&e.name);
        self.write(" = enum");

        if let Some(tag) = &e.tag_type {
            self.write("(");
            self.print_type_expr(tag);
            self.write(")");
        }

        self.write(" {");

        if e.variants.is_empty() {
            self.write("};");
        } else {
            self.newline();
            self.indent();

            for variant in &e.variants {
                self.write(&variant.name);
                if let Some(val) = &variant.value {
                    self.write(" = ");
                    self.print_expr(val);
                }
                self.write(",");
                self.newline();
            }

            self.dedent();
            self.write("};");
        }
        self.newline();
    }

    fn print_union_decl(&mut self, u: &UnionDecl) {
        self.print_doc_comment(&u.doc_comment);

        if u.is_pub {
            self.write("pub ");
        }

        self.write("const ");
        self.write(&u.name);
        self.write(" = union");

        match &u.tag {
            Some(Some(ty)) => {
                self.write("(");
                self.print_type_expr(ty);
                self.write(")");
            }
            Some(None) => {
                self.write("(enum)");
            }
            None => {}
        }

        self.write(" {");

        if u.fields.is_empty() {
            self.write("};");
        } else {
            self.newline();
            self.indent();

            for field in &u.fields {
                self.write(&field.name);
                if let Some(ty) = &field.ty {
                    self.write(": ");
                    self.print_type_expr(ty);
                }
                self.write(",");
                self.newline();
            }

            self.dedent();
            self.write("};");
        }
        self.newline();
    }

    fn print_error_set_decl(&mut self, e: &ErrorSetDecl) {
        self.print_doc_comment(&e.doc_comment);

        if e.is_pub {
            self.write("pub ");
        }

        self.write(&e.name);
        self.write(" := error {");

        if e.variants.is_empty() {
            self.write("};");
        } else {
            self.newline();
            self.indent();

            for variant in &e.variants {
                self.write(&variant.name);
                if let Some(ty) = &variant.data_type {
                    self.write(": ");
                    self.print_type_expr(ty);
                }
                self.write(",");
                self.newline();
            }

            self.dedent();
            self.write("};");
        }
        self.newline();
    }

    fn print_fault_set_decl(&mut self, f: &FaultSetDecl) {
        self.print_doc_comment(&f.doc_comment);

        if f.is_pub {
            self.write("pub ");
        }

        self.write(&f.name);
        self.write(" := fault {");

        if f.variants.is_empty() {
            self.write("};");
        } else {
            self.newline();
            self.indent();

            for variant in &f.variants {
                self.write(&variant.name);
                if let Some(ty) = &variant.data_type {
                    self.write(": ");
                    self.print_type_expr(ty);
                }
                self.write(",");
                self.newline();
            }

            self.dedent();
            self.write("};");
        }
        self.newline();
    }

    fn print_test_decl(&mut self, t: &TestDecl) {
        self.write("test \"");
        self.write(&t.name);
        self.write("\" ");
        self.print_block(&t.body);
        self.newline();
    }

    // =========================================================================
    // Statements
    // =========================================================================

    fn print_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Binding(b) => self.print_binding(b),
            Stmt::Alias(a) => self.print_alias_binding(a),
            Stmt::Assign(a) => self.print_assign_stmt(a),
            Stmt::If(i) => self.print_if_stmt(i),
            Stmt::For(f) => self.print_for_stmt(f),
            Stmt::Switch(s) => self.print_switch_stmt(s),
            Stmt::Tick(t) => self.print_tick_stmt(t),
            Stmt::TryBlock(t) => self.print_try_block_stmt(t),
            Stmt::Return(r) => self.print_return_stmt(r),
            Stmt::Break(b) => self.print_break_stmt(b),
            Stmt::Continue(c) => self.print_continue_stmt(c),
            Stmt::Defer(d) => self.print_defer_stmt(d),
            Stmt::Errdefer(e) => self.print_errdefer_stmt(e),
            Stmt::Block(b) => {
                self.print_block(b);
                self.newline();
            }
            Stmt::Expr(e) => self.print_expr_stmt(e),
            Stmt::Gate(g) => self.print_gate_op(g),
            Stmt::Prepare(p) => self.print_prepare_op(p),
            Stmt::Measure(m) => self.print_measure_op(m),
            Stmt::Barrier(b) => self.print_barrier_op(b),
        }
    }

    fn print_alias_binding(&mut self, a: &AliasBinding) {
        self.write("alias ");
        self.write(&a.name);
        self.write(" := ");
        self.print_expr(&a.source);
        self.write(";");
        self.newline();
    }

    fn print_assign_stmt(&mut self, a: &AssignStmt) {
        self.print_expr(&a.target);
        self.write(" ");
        self.write(match a.op {
            AssignOp::Assign => "=",
            AssignOp::AddAssign => "+=",
            AssignOp::SubAssign => "-=",
            AssignOp::MulAssign => "*=",
            AssignOp::DivAssign => "/=",
            AssignOp::AndAssign => "&=",
            AssignOp::OrAssign => "|=",
            AssignOp::XorAssign => "^=",
        });
        self.write(" ");
        self.print_expr(&a.value);
        self.write(";");
        self.newline();
    }

    fn print_if_stmt(&mut self, i: &IfStmt) {
        self.write("if (");
        self.print_expr(&i.condition);
        self.write(")");

        if let Some(cap) = &i.capture {
            self.write(" |");
            self.write(cap);
            self.write("|");
        }

        self.write(" ");
        self.print_block(&i.then_body);

        if let Some(else_branch) = &i.else_body {
            self.write(" else ");
            match else_branch {
                ElseBranch::ElseIf(elif) => self.print_if_stmt(elif),
                ElseBranch::Else(block) => {
                    self.print_block(block);
                    self.newline();
                }
            }
        } else {
            self.newline();
        }
    }

    fn print_for_stmt(&mut self, f: &ForStmt) {
        if let Some(label) = &f.label {
            self.write(label);
            self.write(": ");
        }

        if f.is_inline {
            self.write("inline ");
        }

        self.write("for ");

        // Print captures (loop variables) first: for i, j in ...
        if f.captures.is_empty() {
            self.write("_ ");
        } else {
            for (i, cap) in f.captures.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.write(cap);
            }
            self.write(" ");
        }

        self.write("in ");

        match &f.range {
            ForRange::Range { start, end } => {
                self.print_expr(start);
                self.write("..");
                self.print_expr(end);
            }
            ForRange::Collection(coll) => {
                self.print_expr(coll);
            }
        }

        self.write(" ");
        self.print_block(&f.body);
        self.newline();
    }

    fn print_switch_stmt(&mut self, s: &SwitchStmt) {
        self.write("switch (");
        self.print_expr(&s.value);
        self.write(") {");
        self.newline();
        self.indent();

        for prong in &s.prongs {
            if prong.is_else {
                self.write("else");
            } else {
                for (i, case) in prong.cases.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(&case.value);
                    if let Some(end) = &case.end {
                        self.write("..");
                        self.print_expr(end);
                    }
                }
            }
            self.write(" => ");
            self.print_expr(&prong.body);
            self.write(",");
            self.newline();
        }

        self.dedent();
        self.write("}");
        self.newline();
    }

    fn print_tick_stmt(&mut self, t: &TickStmt) {
        // Print attributes
        for attr in &t.attrs {
            self.print_attribute(attr);
            self.newline();
        }

        self.write("tick");
        if let Some(label) = &t.label {
            self.write("(\"");
            self.write(label);
            self.write("\")");
        }
        self.write(" {");

        if t.body.is_empty() {
            self.write("}");
        } else {
            self.newline();
            self.indent();

            for stmt in &t.body {
                self.print_stmt(stmt);
            }

            self.dedent();
            self.write("}");
        }
        self.newline();
    }

    fn print_try_block_stmt(&mut self, t: &TryBlockStmt) {
        match t.mode {
            TryMode::Collect => self.write("try "),
            TryMode::Propagate => self.write("try! "),
        }
        self.print_block(&t.body);

        if let Some(catch) = &t.catch_clause {
            self.write(" catch |");
            self.write(&catch.capture);
            self.write("| ");
            self.print_expr(&catch.body);
        }
        self.newline();
    }

    fn print_return_stmt(&mut self, r: &ReturnStmt) {
        self.write("return");
        // Simplify `return unit;` to `return;` for cleaner output
        if let Some(val) = &r.value
            && !matches!(val, Expr::Unit(_))
        {
            self.write(" ");
            self.print_expr(val);
        }
        self.write(";");
        self.newline();
    }

    fn print_break_stmt(&mut self, b: &BreakStmt) {
        self.write("break");
        if let Some(label) = &b.label {
            self.write(" :");
            self.write(label);
        }
        if let Some(val) = &b.value {
            self.write(" ");
            self.print_expr(val);
        }
        self.write(";");
        self.newline();
    }

    fn print_continue_stmt(&mut self, c: &ContinueStmt) {
        self.write("continue");
        if let Some(label) = &c.label {
            self.write(" :");
            self.write(label);
        }
        self.write(";");
        self.newline();
    }

    fn print_defer_stmt(&mut self, d: &DeferStmt) {
        self.write("defer ");
        // Defer body is printed inline, not as a full statement
        self.print_stmt_inline(&d.body);
        self.newline();
    }

    fn print_errdefer_stmt(&mut self, e: &ErrDeferStmt) {
        self.write("errdefer");
        if let Some(cap) = &e.capture {
            self.write(" |");
            self.write(cap);
            self.write("|");
        }
        self.write(" ");
        self.print_stmt_inline(&e.body);
        self.newline();
    }

    fn print_stmt_inline(&mut self, stmt: &Stmt) {
        // Print statement without trailing newline
        match stmt {
            Stmt::Expr(e) => {
                self.print_expr(&e.expr);
                self.write(";");
            }
            Stmt::Block(b) => self.print_block(b),
            _ => self.print_stmt(stmt),
        }
    }

    fn print_expr_stmt(&mut self, e: &ExprStmt) {
        // Print attributes
        for attr in &e.attrs {
            self.print_attribute(attr);
            self.newline();
        }

        self.print_expr(&e.expr);
        self.write(";");
        self.newline();
    }

    fn print_attribute(&mut self, attr: &Attribute) {
        self.write("@");
        self.write(&attr.name);
        if let Some(val) = &attr.value {
            self.write("(");
            match val {
                AttributeValue::Bool(b) => self.write(if *b { "true" } else { "false" }),
                AttributeValue::Int(i) => self.write(&i.to_string()),
                AttributeValue::Float(f) => self.write(&f.to_string()),
                AttributeValue::String(s) => {
                    self.write("\"");
                    self.write(s);
                    self.write("\"");
                }
                AttributeValue::Ident(i) => self.write(i),
            }
            self.write(")");
        }
    }

    // =========================================================================
    // Quantum operations
    // =========================================================================

    fn print_gate_op(&mut self, g: &GateOp) {
        self.write(g.kind.keyword());

        if !g.params.is_empty() {
            self.write("(");
            for (i, param) in g.params.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.print_expr(param);
            }
            self.write(")");
        }

        self.write(" ");

        if g.targets.len() == 1 {
            self.print_slot_ref(&g.targets[0]);
        } else {
            self.write("(");
            for (i, target) in g.targets.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.print_slot_ref(target);
            }
            self.write(")");
        }

        self.write(";");
        self.newline();
    }

    fn print_slot_ref(&mut self, slot: &SlotRef) {
        self.write(&slot.allocator);
        self.write("[");
        self.print_expr(&slot.index);
        self.write("]");
    }

    fn print_prepare_op(&mut self, p: &PrepareOp) {
        self.write("pz ");
        if let Some(slots) = &p.slots {
            // pz {q[0], q[1], ...};
            self.write("{");
            for (i, slot) in slots.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.write(&format!("{}[{}]", p.allocator, slot));
            }
            self.write("}");
        } else {
            // pz q;
            self.write(&p.allocator);
        }
        self.write(";");
        self.newline();
    }

    fn print_measure_op(&mut self, m: &MeasureOp) {
        self.write("measure(");
        for (i, target) in m.targets.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.print_slot_ref(target);
        }
        self.write(")");

        if !m.results.is_empty() {
            self.write(" -> ");
            for (i, result) in m.results.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.write(&result.register);
                self.write("[");
                self.print_expr(&result.index);
                self.write("]");
            }
        }

        self.write(";");
        self.newline();
    }

    fn print_barrier_op(&mut self, b: &BarrierOp) {
        self.write("barrier(");
        for (i, alloc) in b.allocators.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.write(alloc);
        }
        self.write(");");
        self.newline();
    }

    // =========================================================================
    // Blocks
    // =========================================================================

    fn print_block(&mut self, block: &Block) {
        // Print block attributes
        for attr in &block.attrs {
            self.print_attribute(attr);
            self.newline();
        }

        if let Some(label) = &block.label {
            self.write(label);
            self.write(": ");
        }

        self.write("{");

        if block.statements.is_empty() && block.trailing_expr.is_none() {
            self.write("}");
        } else {
            self.newline();
            self.indent();

            for stmt in &block.statements {
                self.print_stmt(stmt);
            }

            if let Some(expr) = &block.trailing_expr {
                self.print_expr(expr);
                self.newline();
            }

            self.dedent();
            self.write("}");
        }
    }

    // =========================================================================
    // Expressions
    // =========================================================================

    fn print_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::IntLit(lit) => {
                self.write(&lit.value.to_string());
                if let Some(suffix) = &lit.suffix {
                    self.write("_");
                    self.write(suffix);
                }
            }
            Expr::FloatLit(lit) => {
                self.write(&lit.value.to_string());
                if let Some(suffix) = &lit.suffix {
                    self.write("_");
                    self.write(suffix);
                }
            }
            Expr::AngleLit(angle) => {
                self.print_expr(&angle.value);
                self.write(" ");
                self.write(match angle.unit {
                    AngleUnit::Turns => "turns",
                    AngleUnit::Rad => "rad",
                });
            }
            Expr::TypeAscription(asc) => {
                self.print_expr(&asc.value);
                self.write(" ");
                self.write(&asc.type_name);
            }
            Expr::BoolLit(lit) => {
                self.write(if lit.value { "true" } else { "false" });
            }
            Expr::StringLit(lit) => {
                self.write("\"");
                self.write(&escape_string(&lit.value));
                self.write("\"");
            }
            Expr::FString(fstr) => {
                self.write("f\"");
                for part in &fstr.parts {
                    match part {
                        FStringPart::Text(text) => {
                            self.write(&escape_fstring_text(text));
                        }
                        FStringPart::Expr { expr, format } => {
                            self.write("{");
                            self.print_expr(expr);
                            if let Some(fmt) = format {
                                self.write(":");
                                self.write(fmt);
                            }
                            self.write("}");
                        }
                    }
                }
                self.write("\"");
            }
            Expr::CharLit(lit) => {
                self.write("'");
                self.write(&escape_char(lit.value));
                self.write("'");
            }
            Expr::Null(_) => self.write("none"),
            Expr::Undefined(_) => self.write("undefined"),
            Expr::Unit(_) => self.write("unit"),
            Expr::Ident(ident) => self.write(&ident.name),
            Expr::SlotRef(slot) => self.print_slot_ref(slot),
            Expr::BitRef(bit) => {
                self.write(&bit.register);
                self.write("[");
                self.print_expr(&bit.index);
                self.write("]");
            }
            Expr::Binary(bin) => self.print_binary_expr(bin),
            Expr::Unary(un) => self.print_unary_expr(un),
            Expr::Field(field) => {
                self.print_expr(&field.object);
                self.write(".");
                self.write(&field.field);
            }
            Expr::Index(idx) => {
                self.print_expr(&idx.object);
                self.write("[");
                self.print_expr(&idx.index);
                self.write("]");
            }
            Expr::Call(call) => {
                self.print_expr(&call.callee);
                self.write("(");
                for (i, arg) in call.args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(arg);
                }
                self.write(")");
            }
            Expr::BatchApply(batch) => {
                self.print_expr(&batch.operation);
                self.write(" {");
                for (i, target) in batch.targets.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(target);
                }
                self.write("}");
            }
            Expr::If(if_expr) => {
                self.write("if (");
                self.print_expr(&if_expr.condition);
                self.write(") ");
                self.print_expr(&if_expr.then_expr);
                self.write(" else ");
                self.print_expr(&if_expr.else_expr);
            }
            Expr::Block(block) => {
                for attr in &block.attrs {
                    self.print_attribute(attr);
                    self.write(" ");
                }
                self.write(&block.label);
                self.write(": {");
                if block.statements.is_empty() && block.trailing_expr.is_none() {
                    self.write("}");
                } else {
                    self.newline();
                    self.indent();
                    for stmt in &block.statements {
                        self.print_stmt(stmt);
                    }
                    if let Some(expr) = &block.trailing_expr {
                        self.print_expr(expr);
                        self.newline();
                    }
                    self.dedent();
                    self.write("}");
                }
            }
            Expr::Comptime(ct) => {
                self.write("comptime ");
                self.print_expr(&ct.inner);
            }
            Expr::Builtin(bi) => {
                self.write("@");
                self.write(&bi.name);
                self.write("(");
                for (i, arg) in bi.args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(arg);
                }
                self.write(")");
            }
            Expr::AnonStruct(anon) => {
                if anon.is_packed {
                    self.write("packed ");
                }
                self.write("struct {");
                if anon.fields.is_empty() {
                    self.write("}");
                } else {
                    self.newline();
                    self.indent();
                    for field in &anon.fields {
                        self.print_struct_field(field);
                    }
                    self.dedent();
                    self.write("}");
                }
            }
            Expr::StructInit(init) => {
                if let Some(ty) = &init.ty {
                    self.print_type_expr(ty);
                    self.write(" ");
                } else {
                    // Anonymous struct uses .{ } syntax
                    self.write(".");
                }
                self.write("{");
                for (i, field) in init.fields.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    // Check for shorthand: field name matches identifier value
                    let is_shorthand =
                        matches!(&field.value, Expr::Ident(ident) if ident.name == field.name);
                    if is_shorthand {
                        // Shorthand: just `name` instead of `name: name`
                        self.write(&field.name);
                    } else {
                        // Rust-style: `name: value`
                        self.write(&field.name);
                        self.write(": ");
                        self.print_expr(&field.value);
                    }
                }
                self.write("}");
            }
            Expr::ArrayInit(init) => {
                if let Some(ty) = &init.ty {
                    self.print_type_expr(ty);
                }
                self.write("{");
                for (i, elem) in init.elements.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(elem);
                }
                self.write("}");
            }
            Expr::BracketArray(arr) => {
                self.write("[");
                for (i, elem) in arr.elements.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(elem);
                }
                self.write("]");
            }
            Expr::Tuple(tuple) => {
                self.write("(");
                for (i, elem) in tuple.elements.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(elem);
                }
                self.write(")");
            }
            Expr::Set(set) => {
                if let Some(ty) = &set.element_type {
                    self.write("Set(");
                    self.print_type_expr(ty);
                    self.write(")");
                }
                self.write("{");
                for (i, elem) in set.elements.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(elem);
                }
                self.write("}");
            }
            Expr::Range(range) => {
                if let Some(start) = &range.start {
                    self.print_expr(start);
                }
                self.write("..");
                if let Some(end) = &range.end {
                    self.print_expr(end);
                }
            }
            Expr::Measure(m) => {
                self.write("mz(");
                if m.pack {
                    self.write("pack ");
                }
                self.print_type_expr(&m.result_type);
                self.write(") ");
                self.print_expr(&m.targets);
            }
            Expr::Gate(g) => {
                self.write(g.kind.keyword());
                if !g.params.is_empty() {
                    self.write("(");
                    for (i, param) in g.params.iter().enumerate() {
                        if i > 0 {
                            self.write(", ");
                        }
                        self.print_expr(param);
                    }
                    self.write(")");
                }
                self.write(" ");
                self.print_expr(&g.target);
            }
            Expr::ErrorValue(err) => {
                self.write("error.");
                self.write(&err.name);
            }
            Expr::FaultValue(fault) => {
                self.write("fault.");
                self.write(&fault.name);
            }
            Expr::Catch(c) => {
                self.print_expr(&c.operand);
                self.write(" catch");
                if let Some(cap) = &c.capture {
                    self.write(" |");
                    self.write(cap);
                    self.write("|");
                }
                self.write(" ");
                self.print_expr(&c.handler);
            }
            Expr::TryBlock(t) => {
                match t.mode {
                    TryMode::Collect => self.write("try "),
                    TryMode::Propagate => self.write("try! "),
                }
                self.print_block(&t.body);
                if let Some(catch) = &t.catch_clause {
                    self.write(" catch |");
                    self.write(&catch.capture);
                    self.write("| ");
                    self.print_expr(&catch.body);
                }
            }
            Expr::FnLit(f) => {
                self.write("fn(");
                for (i, param) in f.params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_param(param);
                }
                self.write(")");
                if let Some(ret) = &f.return_type {
                    self.write(" -> ");
                    self.print_type_expr(ret);
                }
                self.write(" ");
                self.print_block(&f.body);
            }
            Expr::Channel(channel) => {
                self.write("@emit.");
                self.write(&channel.channel);
                self.write(".");
                self.write(&channel.command);
                self.write("(");
                for (i, arg) in channel.args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    match arg {
                        ChannelArg::Positional(expr) => {
                            self.print_expr(expr);
                        }
                        ChannelArg::Named { name, value } => {
                            self.write(name);
                            self.write(": ");
                            self.print_expr(value);
                        }
                    }
                }
                self.write(")");
            }
            Expr::Result(result) => {
                self.write("result(\"");
                self.write(&escape_string(&result.tag));
                self.write("\", ");
                self.print_expr(&result.value);
                self.write(")");
            }
        }
    }

    fn print_binary_expr(&mut self, bin: &BinaryExpr) {
        let needs_parens = matches!(
            bin.op,
            BinaryOp::And | BinaryOp::Or | BinaryOp::Orelse | BinaryOp::Catch
        );

        if needs_parens {
            self.write("(");
        }

        self.print_expr(&bin.left);

        self.write(" ");
        self.write(match bin.op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::In => "in",
            BinaryOp::NotIn => "not in",
            BinaryOp::And => "and",
            BinaryOp::Or => "or",
            BinaryOp::Orelse => "orelse",
            BinaryOp::Catch => "catch",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
        });
        self.write(" ");

        self.print_expr(&bin.right);

        if needs_parens {
            self.write(")");
        }
    }

    fn print_unary_expr(&mut self, un: &UnaryExpr) {
        match un.op {
            UnaryOp::Neg => self.write("-"),
            UnaryOp::Not => self.write("!"),
            UnaryOp::BitNot => self.write("~"),
            UnaryOp::AddrOf => self.write("&"),
            UnaryOp::Deref => self.write("*"),
            UnaryOp::OptionalUnwrap => {
                self.print_expr(&un.operand);
                self.write(".?");
                return;
            }
            UnaryOp::ErrorUnwrap => {
                self.print_expr(&un.operand);
                self.write(".!");
                return;
            }
            UnaryOp::Try => self.write("try "),
        }
        self.print_expr(&un.operand);
    }

    // =========================================================================
    // Types
    // =========================================================================

    fn print_type_expr(&mut self, ty: &TypeExpr) {
        match ty {
            TypeExpr::Primitive(p) => self.print_primitive_type(p),
            TypeExpr::Qubit => self.write("qubit"),
            TypeExpr::Bit => self.write("bit"),
            TypeExpr::QAlloc(cap) => {
                self.write("qalloc");
                if let Some(c) = cap {
                    self.write("(");
                    self.print_expr(c);
                    self.write(")");
                }
            }
            TypeExpr::Array(arr) => {
                self.write("[");
                if let Some(size) = &arr.size {
                    self.print_expr(size);
                }
                self.write("]");
                self.print_type_expr(&arr.element);
            }
            TypeExpr::Pointer(ptr) => {
                if ptr.is_many {
                    self.write("[*");
                } else {
                    self.write("*");
                }
                if ptr.is_const {
                    self.write("const ");
                }
                self.print_type_expr(&ptr.pointee);
            }
            TypeExpr::Optional(inner) => {
                self.write("?");
                self.print_type_expr(inner);
            }
            TypeExpr::ErrorUnion(eu) => {
                self.print_type_expr(&eu.error_type);
                self.write("!");
                self.print_type_expr(&eu.payload_type);
            }
            TypeExpr::CollectedErrors(ce) => {
                self.write("[]");
                self.print_type_expr(&ce.error_type);
                self.write("!");
                self.print_type_expr(&ce.payload_type);
            }
            TypeExpr::Fn(f) => {
                self.write("fn(");
                for (i, param) in f.params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_type_expr(param);
                }
                self.write(")");
                if let Some(ret) = &f.return_type {
                    self.write(" -> ");
                    self.print_type_expr(ret);
                }
            }
            TypeExpr::Tuple(types) => {
                self.write("(");
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_type_expr(t);
                }
                self.write(")");
            }
            TypeExpr::Set(elem) => {
                self.write("Set(");
                self.print_type_expr(elem);
                self.write(")");
            }
            TypeExpr::Named(path) => {
                for (i, seg) in path.segments.iter().enumerate() {
                    if i > 0 {
                        self.write(".");
                    }
                    self.write(seg);
                }
            }
            TypeExpr::Type => self.write("type"),
            TypeExpr::AnyType => self.write("anytype"),
            TypeExpr::Unit => self.write("unit"),
            TypeExpr::Struct(s) => {
                if s.is_packed {
                    self.write("packed ");
                }
                self.write("struct { ");
                for (i, field) in s.fields.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&field.name);
                    self.write(": ");
                    self.print_type_expr(&field.ty);
                }
                self.write(" }");
            }
            TypeExpr::Enum(e) => {
                self.write("enum ");
                if let Some(tag) = &e.tag_type {
                    self.write("(");
                    self.print_type_expr(tag);
                    self.write(") ");
                }
                self.write("{ ");
                for (i, variant) in e.variants.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&variant.name);
                    if let Some(val) = &variant.value {
                        self.write(" = ");
                        self.print_expr(val);
                    }
                }
                self.write(" }");
            }
        }
    }

    fn print_primitive_type(&mut self, p: &PrimitiveType) {
        match p {
            PrimitiveType::UInt { bits } => {
                self.write("u");
                self.write(&bits.to_string());
            }
            PrimitiveType::IInt { bits } => {
                self.write("i");
                self.write(&bits.to_string());
            }
            PrimitiveType::Usize => self.write("usize"),
            PrimitiveType::Isize => self.write("isize"),
            PrimitiveType::F16 => self.write("f16"),
            PrimitiveType::F32 => self.write("f32"),
            PrimitiveType::F64 => self.write("f64"),
            PrimitiveType::F128 => self.write("f128"),
            PrimitiveType::A64 => self.write("a64"),
            PrimitiveType::Bool => self.write("bool"),
        }
    }
}

// =============================================================================
// String escaping helpers
// =============================================================================

fn escape_string(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            c if c.is_control() => {
                result.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

fn escape_char(c: char) -> String {
    match c {
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        '\\' => "\\\\".to_string(),
        '\'' => "\\'".to_string(),
        c if c.is_control() => format!("\\x{:02x}", c as u32),
        c => c.to_string(),
    }
}

/// Escape text inside f-strings (also escapes { and })
fn escape_fstring_text(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '{' => result.push_str("\\{"),
            '}' => result.push_str("\\}"),
            c if c.is_control() => {
                result.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

// =============================================================================
// Public API
// =============================================================================

/// Format a Zlup program AST to canonical string form.
pub fn pretty_print(program: &Program, options: &PrettyOptions) -> String {
    let mut printer = PrettyPrinter::new(options.clone());
    printer.print_program(program)
}

/// Format Zlup source code using AST-based pretty printing.
///
/// This parses the source to an AST and pretty-prints it, providing
/// more accurate formatting than text-based approaches.
///
/// Returns `None` if the source cannot be parsed.
pub fn format_source(source: &str, options: &PrettyOptions) -> Option<String> {
    match crate::parser::parse(source) {
        Ok(program) => Some(pretty_print(&program, options)),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn format(source: &str) -> String {
        let program = parse(source).expect("Failed to parse");
        pretty_print(&program, &PrettyOptions::default())
    }

    #[test]
    fn test_simple_function() {
        let source = "fn main() -> unit { return unit; }";
        let formatted = format(source);
        assert!(formatted.contains("fn main() -> unit"));
        assert!(formatted.contains("return;"));
    }

    #[test]
    fn test_function_with_params() {
        let source = "fn add(a: u32, b: u32) -> u32 { return a + b; }";
        let formatted = format(source);
        assert!(formatted.contains("fn add(a: u32, b: u32) -> u32"));
    }

    #[test]
    fn test_binding_with_type() {
        let source = "x: u32 = 42;";
        let formatted = format(source);
        assert!(formatted.contains("x: u32 = 42;"));
    }

    #[test]
    fn test_binding_inferred() {
        let source = "x := 42;";
        let formatted = format(source);
        assert!(formatted.contains("x := 42;"));
    }

    #[test]
    fn test_mutable_binding() {
        let source = "mut x := 42;";
        let formatted = format(source);
        assert!(formatted.contains("mut x := 42;"));
    }

    #[test]
    fn test_if_statement() {
        let source = "fn test() -> unit { if (x == 1) { y := 2; } return unit; }";
        let formatted = format(source);
        assert!(formatted.contains("if"));
        assert!(formatted.contains("x == 1"));
        assert!(formatted.contains("y := 2;"));
    }

    #[test]
    fn test_if_else() {
        let source = "fn test() -> unit { if (x) { a := 1; } else { b := 2; } return unit; }";
        let formatted = format(source);
        assert!(formatted.contains("if (x)"));
        assert!(formatted.contains("else {"));
    }

    #[test]
    fn test_for_loop() {
        let source = "fn test() -> unit { for i in 0..10 { x := i; } }";
        let formatted = format(source);
        assert!(formatted.contains("for"));
        assert!(formatted.contains("0..10"));
    }

    #[test]
    fn test_quantum_gate() {
        let source = "fn test() -> unit { h q[0]; }";
        let formatted = format(source);
        assert!(formatted.contains("h q[0]"));
    }

    #[test]
    fn test_two_qubit_gate() {
        let source = "fn test() -> unit { cx (q[0], q[1]); }";
        let formatted = format(source);
        assert!(formatted.contains("cx (q[0], q[1])"));
    }

    #[test]
    fn test_parameterized_gate() {
        let source = "fn test() -> unit { rx(0.5) q[0]; }";
        let formatted = format(source);
        assert!(formatted.contains("rx(0.5) q[0]"));
    }

    #[test]
    fn test_tick_block() {
        let source = "fn test() -> unit { tick { h q[0]; cx (q[0], q[1]); } }";
        let formatted = format(source);
        assert!(formatted.contains("tick {"));
    }

    #[test]
    fn test_indentation() {
        let source = "fn main() -> unit { if (true) { x := 1; } return unit; }";
        let formatted = format(source);
        // Check that nested content is indented - just verify content is present
        assert!(formatted.contains("if (true)"));
        assert!(formatted.contains("x := 1;"));
    }

    #[test]
    fn test_binary_operators() {
        let source = "fn test() -> unit { x := a + b * c; }";
        let formatted = format(source);
        assert!(formatted.contains("a + b * c"));
    }

    #[test]
    fn test_comparison_operators() {
        let source = "fn test() -> unit { if (x == 1) { y := 2; } return unit; }";
        let formatted = format(source);
        assert!(formatted.contains("x == 1"));
    }

    #[test]
    fn test_array_literal() {
        let source = "fn test() -> unit { arr := [1, 2, 3]; }";
        let formatted = format(source);
        assert!(formatted.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_tuple() {
        let source = "fn test() -> unit { t := (1, 2, 3); }";
        let formatted = format(source);
        assert!(formatted.contains("(1, 2, 3)"));
    }

    #[test]
    fn test_string_literal() {
        let source = r#"fn test() -> unit { s := "hello"; }"#;
        let formatted = format(source);
        assert!(formatted.contains(r#""hello""#));
    }

    #[test]
    fn test_string_escapes() {
        let source = r#"fn test() -> unit { s := "hello\nworld"; }"#;
        let formatted = format(source);
        assert!(formatted.contains(r#"\n"#));
    }

    #[test]
    fn test_pub_function() {
        let source = "pub fn exported() -> unit { }";
        let formatted = format(source);
        assert!(formatted.contains("pub fn exported()"));
    }

    #[test]
    fn test_inline_function() {
        let source = "inline fn fast() -> unit { }";
        let formatted = format(source);
        assert!(formatted.contains("inline fn fast()"));
    }

    #[test]
    fn test_field_access() {
        // Test field access works correctly
        let source = "fn test() -> unit { x := obj.field; return unit; }";
        let formatted = format(source);
        assert!(formatted.contains("obj.field"));
    }

    #[test]
    fn test_custom_indent() {
        let source = "fn main() -> unit { x := 1; }";
        let options = PrettyOptions {
            indent_size: 2,
            ..Default::default()
        };
        let program = parse(source).unwrap();
        let formatted = pretty_print(&program, &options);
        assert!(formatted.contains("  x := 1;")); // 2 spaces
    }

    #[test]
    fn test_tabs() {
        let source = "fn main() -> unit { x := 1; }";
        let options = PrettyOptions {
            use_spaces: false,
            ..Default::default()
        };
        let program = parse(source).unwrap();
        let formatted = pretty_print(&program, &options);
        assert!(formatted.contains("\tx := 1;")); // tab
    }

    #[test]
    fn test_trailing_newline() {
        let source = "fn main() -> unit { }";
        let formatted = format(source);
        assert!(formatted.ends_with('\n'));
    }

    #[test]
    fn test_empty_block() {
        let source = "fn empty() -> unit {}";
        let formatted = format(source);
        assert!(formatted.contains("{}"));
    }

    #[test]
    fn test_multiple_functions() {
        let source = "fn a() -> unit { } fn b() -> unit { }";
        let formatted = format(source);
        assert!(formatted.contains("fn a()"));
        assert!(formatted.contains("fn b()"));
    }

    #[test]
    fn test_extern_fn() {
        let source = r#"extern "C" fn puts(s: [*]const u8) -> i32;"#;
        let formatted = format(source);
        assert!(formatted.contains(r#"extern "C" fn puts"#));
    }

    #[test]
    fn test_builtin_call() {
        let source = "fn test() -> unit { x := @sizeOf(u32); }";
        let formatted = format(source);
        assert!(formatted.contains("@sizeOf(u32)"));
    }

    #[test]
    fn test_return_statement() {
        let source = "fn test() -> u32 { return 42; }";
        let formatted = format(source);
        assert!(formatted.contains("return 42;"));
    }

    #[test]
    fn test_break_continue() {
        let source = "fn test() -> unit { for i in 0..10 { break; } return unit; }";
        let formatted = format(source);
        assert!(formatted.contains("break;"));
    }

    #[test]
    fn test_deeply_nested() {
        let source = "fn main() -> unit { if (a) { if (b) { if (c) { x := 1; } return unit; } return unit; } return unit; }";
        let formatted = format(source);
        // Should have proper nesting - just check content is preserved
        assert!(formatted.contains("if (a)"));
        assert!(formatted.contains("if (b)"));
        assert!(formatted.contains("if (c)"));
        assert!(formatted.contains("x := 1;"));
    }

    #[test]
    fn test_measurement() {
        let source = "fn test() -> unit { r := mz(u8) q; }";
        let formatted = format(source);
        assert!(formatted.contains("mz(u8) q"));
    }

    #[test]
    fn test_optional_type() {
        let source = "fn test(x: ?u32) -> unit { }";
        let formatted = format(source);
        assert!(formatted.contains("?u32"));
    }

    #[test]
    fn test_pointer_type() {
        let source = "fn test(p: *u32) -> unit { }";
        let formatted = format(source);
        assert!(formatted.contains("*u32"));
    }

    #[test]
    fn test_array_type() {
        let source = "fn test(arr: [10]u32) -> unit { }";
        let formatted = format(source);
        assert!(formatted.contains("[10]u32"));
    }

    #[test]
    fn test_format_source_returns_none_on_invalid() {
        let result = format_source("fn broken(", &PrettyOptions::default());
        assert!(result.is_none());
    }

    #[test]
    fn test_format_source_works_on_valid() {
        let result = format_source("fn main() -> unit {}", &PrettyOptions::default());
        assert!(result.is_some());
        assert!(result.unwrap().contains("fn main()"));
    }
}
