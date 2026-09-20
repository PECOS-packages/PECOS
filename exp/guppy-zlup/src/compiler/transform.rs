//! Transform Guppy IR to Zlup AST.

use std::collections::BTreeMap;

use zlup::ast::{self as zlup_ast, GateKind as ZlupGateKind};

use crate::ir::{Expr, ExprKind, Function, GateKind, GuppyIR, Param, Stmt, StmtKind, TypeExpr};

/// Transform context for tracking variable information with invariant checking.
#[derive(Default)]
struct TransformContext {
    /// Maps qalloc variable names to their sizes (if known).
    qalloc_sizes: BTreeMap<String, i128>,
    /// Set of variable names that have been declared (for distinguishing new vars from reassignments).
    declared_vars: std::collections::BTreeSet<String>,
    /// Guppy values that need a converted Zlup angle binding.
    angle_vars: std::collections::BTreeSet<String>,
    /// Maps Guppy values to fresh `a64` bindings containing turns.
    angle_bindings: BTreeMap<String, String>,
    /// Guppy values whose source and converted bindings may be reassigned.
    mutable_vars: std::collections::BTreeSet<String>,
    /// Track which allocators have been used in gates (for invariant checking).
    #[cfg(debug_assertions)]
    used_allocators: std::collections::BTreeSet<String>,
}

impl TransformContext {
    /// Register a new qalloc.
    fn register_qalloc(&mut self, name: &str, size: i128) {
        debug_assert!(
            !self.qalloc_sizes.contains_key(name),
            "Invariant violation: duplicate qalloc for '{}'",
            name
        );
        self.qalloc_sizes.insert(name.to_string(), size);
        self.declared_vars.insert(name.to_string());
    }

    /// Register a variable declaration.
    fn declare_var(&mut self, name: &str) {
        self.declared_vars.insert(name.to_string());
    }

    /// Check if a variable is declared.
    fn is_declared(&self, name: &str) -> bool {
        self.declared_vars.contains(name)
    }

    /// Get qalloc size, with invariant check.
    fn get_qalloc_size(&self, name: &str) -> Option<i128> {
        let size = self.qalloc_sizes.get(name).copied();
        #[cfg(debug_assertions)]
        if size.is_some() {
            debug_assert!(
                self.declared_vars.contains(name),
                "Invariant violation: qalloc '{}' in qalloc_sizes but not in declared_vars",
                name
            );
        }
        size
    }

    /// Mark an allocator as used (for gates).
    #[cfg(debug_assertions)]
    fn mark_allocator_used(&mut self, name: &str) {
        debug_assert!(
            self.qalloc_sizes.contains_key(name),
            "Invariant violation: gate uses allocator '{}' before it was allocated",
            name
        );
        self.used_allocators.insert(name.to_string());
    }

    #[cfg(not(debug_assertions))]
    fn mark_allocator_used(&mut self, _name: &str) {
        // No-op in release builds
    }
}

/// Transform Guppy IR to Zlup AST.
pub fn transform(ir: &GuppyIR) -> Result<zlup_ast::Program, TransformError> {
    let mut declarations = Vec::new();

    for func in &ir.functions {
        let decl = transform_function(func)?;
        declarations.push(zlup_ast::TopLevelDecl::Fn(decl));
    }

    Ok(zlup_ast::Program {
        name: ir.source_file.clone().unwrap_or_else(|| "main".to_string()),
        declarations,
        location: None,
    })
}

fn transform_function(func: &Function) -> Result<zlup_ast::FnDecl, TransformError> {
    let angle_vars = infer_angle_variables(&func.body);
    let (angle_bindings, mutable_vars) = make_angle_binding_names(func, &angle_vars);
    let params = func
        .params
        .iter()
        .map(transform_param)
        .collect::<Result<Vec<_>, _>>()?;

    let return_type = func.return_type.as_ref().map(transform_type);

    let mut ctx = TransformContext {
        angle_vars,
        angle_bindings,
        mutable_vars,
        ..TransformContext::default()
    };
    // Add function parameters to declared_vars
    for param in &func.params {
        ctx.declare_var(&param.name);
    }
    let mut body = transform_block_with_ctx(&func.body, &mut ctx)?;
    let parameter_angles = func
        .params
        .iter()
        .filter_map(|param| converted_angle_binding(&param.name, &ctx))
        .collect::<Vec<_>>();
    body.statements.splice(0..0, parameter_angles);

    // Check if the function returns unit and needs an explicit return statement.
    // Zlup requires explicit `return unit;` for functions that return unit.
    let is_unit_return = match &return_type {
        None => true,
        Some(zlup_ast::TypeExpr::Unit) => true,
        Some(zlup_ast::TypeExpr::Named(path)) if path.segments == ["None"] => true,
        _ => false,
    };

    // Check if the last statement is already a return
    let has_trailing_return = body
        .statements
        .last()
        .is_some_and(|stmt| matches!(stmt, zlup_ast::Stmt::Return(_)));

    // Add implicit return if needed (return; is equivalent to return unit; in Zlup)
    if is_unit_return && !has_trailing_return {
        body.statements
            .push(zlup_ast::Stmt::Return(zlup_ast::ReturnStmt {
                value: None,
                location: None,
            }));
    }

    Ok(zlup_ast::FnDecl {
        name: func.name.clone(),
        params,
        return_type,
        body,
        is_pub: func.is_pub.unwrap_or(false),
        is_inline: false,
        error_mode: None,
        doc_comment: None,
        location: None,
    })
}

fn transform_param(param: &Param) -> Result<zlup_ast::Param, TransformError> {
    Ok(zlup_ast::Param {
        name: param.name.clone(),
        ty: transform_type(&param.ty),
        is_comptime: false,
        location: None,
    })
}

fn transform_type(ty: &TypeExpr) -> zlup_ast::TypeExpr {
    match ty.kind.as_str() {
        "primitive" => {
            let name = ty.name.as_deref().unwrap_or("unknown");
            match name {
                "int" => zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::IInt { bits: 64 }),
                "float" => zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::F64),
                "bool" => zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::Bool),
                "None" => zlup_ast::TypeExpr::Unit,
                _ => zlup_ast::TypeExpr::Named(zlup_ast::TypePath {
                    segments: vec![name.to_string()],
                    location: None,
                }),
            }
        }
        "qalloc" => {
            zlup_ast::TypeExpr::QAlloc(ty.size.as_ref().map(|s| Box::new(transform_expr(s))))
        }
        "array" => {
            let element = ty
                .element
                .as_ref()
                .map(|e| transform_type(e))
                .unwrap_or(zlup_ast::TypeExpr::Unit);
            let size = ty.size.as_ref().map(|s| transform_expr(s));
            zlup_ast::TypeExpr::Array(Box::new(zlup_ast::ArrayType {
                element,
                size,
                sentinel: None,
            }))
        }
        "optional" => {
            let element = ty
                .element
                .as_ref()
                .map(|e| transform_type(e))
                .unwrap_or(zlup_ast::TypeExpr::Unit);
            zlup_ast::TypeExpr::Optional(Box::new(element))
        }
        "tuple" => {
            // In quantum code, tuple[bool, ...] typically contains measurement results
            // which are u1 in Zlup, so map bool->u1 within tuples
            let elements: Vec<zlup_ast::TypeExpr> = ty
                .elements
                .iter()
                .map(|elem| {
                    if elem.kind == "primitive" && elem.name.as_deref() == Some("bool") {
                        zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::UInt { bits: 1 })
                    } else {
                        transform_type(elem)
                    }
                })
                .collect();
            zlup_ast::TypeExpr::Tuple(elements)
        }
        "named" => {
            let name = ty.name.as_deref().unwrap_or("unknown");
            zlup_ast::TypeExpr::Named(zlup_ast::TypePath {
                segments: vec![name.to_string()],
                location: None,
            })
        }
        _ => zlup_ast::TypeExpr::Unit,
    }
}

fn transform_block(stmts: &[Stmt]) -> Result<zlup_ast::Block, TransformError> {
    let mut ctx = TransformContext::default();
    transform_block_with_ctx(stmts, &mut ctx)
}

fn transform_block_with_ctx(
    stmts: &[Stmt],
    ctx: &mut TransformContext,
) -> Result<zlup_ast::Block, TransformError> {
    let mut statements = Vec::new();

    for stmt in stmts {
        let transformed = transform_stmt_with_ctx(stmt, ctx)?;
        statements.extend(transformed);
    }

    Ok(zlup_ast::Block {
        label: None,
        attrs: Vec::new(),
        statements,
        trailing_expr: None,
        location: None,
    })
}

fn transform_stmt(stmt: &Stmt) -> Result<Vec<zlup_ast::Stmt>, TransformError> {
    let mut ctx = TransformContext::default();
    transform_stmt_with_ctx(stmt, &mut ctx)
}

fn transform_stmt_with_ctx(
    stmt: &Stmt,
    ctx: &mut TransformContext,
) -> Result<Vec<zlup_ast::Stmt>, TransformError> {
    match stmt.kind {
        StmtKind::Qalloc => {
            let name = stmt
                .name
                .as_ref()
                .ok_or(TransformError::MissingField("name"))?;

            // Extract the size value for tracking
            let size_value = stmt.size.as_ref().and_then(|s| {
                if s.kind == ExprKind::Literal {
                    s.value.as_ref().and_then(|v| v.as_i64()).map(|i| i as i128)
                } else {
                    None
                }
            });

            // Track the qalloc size and register variable using context methods
            ctx.register_qalloc(name, size_value.unwrap_or(1));

            let size = stmt.size.as_ref().map(transform_expr).unwrap_or_else(|| {
                zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                    value: 1,
                    suffix: None,
                    location: None,
                })
            });

            // Create a binding with qalloc(N) call as the value
            // In Zlup: `mut q := qalloc(N);`
            let qalloc_call = zlup_ast::Expr::Call(Box::new(zlup_ast::CallExpr {
                callee: zlup_ast::Expr::Ident(zlup_ast::Ident {
                    name: "qalloc".to_string(),
                    location: None,
                }),
                args: vec![size],
                location: None,
            }));

            Ok(vec![zlup_ast::Stmt::Binding(zlup_ast::Binding {
                name: name.clone(),
                ty: None,
                value: Some(qalloc_call),
                is_mutable: true,
                is_pub: false,
                doc_comment: None,
                location: None,
            })])
        }

        StmtKind::Gate => {
            let gate = stmt.gate.ok_or(TransformError::MissingField("gate"))?;
            let zlup_gate = transform_gate_kind(gate);

            // Invariant check: ensure allocators are defined before use
            for target in &stmt.targets {
                if let Some(allocator) = &target.array {
                    ctx.mark_allocator_used(allocator);
                }
            }

            let targets: Vec<zlup_ast::SlotRef> = stmt
                .targets
                .iter()
                .map(transform_slot_ref)
                .collect::<Result<Vec<_>, _>>()?;

            let params: Vec<zlup_ast::Expr> = stmt
                .params
                .iter()
                .map(|expr| transform_angle_expr(expr, ctx))
                .collect();

            Ok(vec![zlup_ast::Stmt::Gate(zlup_ast::GateOp {
                kind: zlup_gate,
                targets,
                params,
                attrs: Vec::new(),
                location: None,
            })])
        }

        StmtKind::Measure => {
            // Determine if we're measuring a single qubit or multiple qubits
            // and construct the appropriate result type.
            //
            // Single qubit (Index expr like q[0]): mz(u1) q[0]
            // Multiple qubits (Ident like q, or multiple targets): mz([N]u1) q

            let (targets_expr, result_type) = if stmt.targets.len() == 1 {
                let target = &stmt.targets[0];
                if target.kind == ExprKind::Index {
                    // Single qubit measurement: mz(u1) q[i]
                    (
                        transform_expr(target),
                        zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::UInt { bits: 1 }),
                    )
                } else if target.kind == ExprKind::Ident {
                    // Measuring entire register or single qubit
                    // Look up the size from the context
                    let var_name = target.name.as_deref().unwrap_or("");
                    let size = ctx.get_qalloc_size(var_name);

                    let result_type = if size == Some(1) {
                        // Single qubit from qubit() call - returns u1
                        zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::UInt { bits: 1 })
                    } else {
                        // Multi-qubit register: mz([N]u1) q
                        zlup_ast::TypeExpr::Array(Box::new(zlup_ast::ArrayType {
                            element: zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::UInt {
                                bits: 1,
                            }),
                            size: size.map(|n| {
                                zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                                    value: n,
                                    suffix: None,
                                    location: None,
                                })
                            }),
                            sentinel: None,
                        }))
                    };

                    // For single qubits, use index [0] instead of the whole register
                    let target_expr = if size == Some(1) {
                        zlup_ast::Expr::Index(Box::new(zlup_ast::IndexExpr {
                            object: transform_expr(target),
                            index: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                                value: 0,
                                suffix: None,
                                location: None,
                            }),
                            location: None,
                        }))
                    } else {
                        transform_expr(target)
                    };

                    (target_expr, result_type)
                } else {
                    // Other single target, default to u1
                    (
                        transform_expr(target),
                        zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::UInt { bits: 1 }),
                    )
                }
            } else {
                // Multiple explicit targets: mz([N]u1) [q[0], q[1], ...]
                let n = stmt.targets.len() as i128;
                (
                    zlup_ast::Expr::BracketArray(Box::new(zlup_ast::BracketArrayExpr {
                        elements: stmt.targets.iter().map(transform_expr).collect(),
                        location: None,
                    })),
                    zlup_ast::TypeExpr::Array(Box::new(zlup_ast::ArrayType {
                        element: zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::UInt {
                            bits: 1,
                        }),
                        size: Some(zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                            value: n,
                            suffix: None,
                            location: None,
                        })),
                        sentinel: None,
                    })),
                )
            };

            // Create MeasureExpr: mz(T) targets
            let measure_expr = zlup_ast::Expr::Measure(Box::new(zlup_ast::MeasureExpr {
                result_type,
                pack: false,
                targets: targets_expr,
                location: None,
            }));

            // If there's a result variable, create a binding
            // Otherwise just emit the measure as an expression statement
            if let Some(result_name) = stmt.results.first() {
                Ok(vec![zlup_ast::Stmt::Binding(zlup_ast::Binding {
                    name: result_name.clone(),
                    ty: None,
                    value: Some(measure_expr),
                    is_mutable: false,
                    is_pub: false,
                    doc_comment: None,
                    location: None,
                })])
            } else {
                Ok(vec![zlup_ast::Stmt::Expr(zlup_ast::ExprStmt {
                    expr: measure_expr,
                    attrs: Vec::new(),
                    location: None,
                })])
            }
        }

        StmtKind::For => {
            let var = stmt
                .var
                .as_ref()
                .ok_or(TransformError::MissingField("var"))?;
            let range = stmt
                .range
                .as_ref()
                .ok_or(TransformError::MissingField("range"))?;

            // Add loop variable to declared_vars
            ctx.declare_var(var);
            let body = transform_block_with_ctx(&stmt.body, ctx)?;

            Ok(vec![zlup_ast::Stmt::For(zlup_ast::ForStmt {
                label: None,
                is_inline: false,
                range: zlup_ast::ForRange::Range {
                    start: transform_expr(&range.start),
                    end: transform_expr(&range.end),
                },
                captures: vec![var.clone()],
                body,
                location: None,
            })])
        }

        StmtKind::While => {
            // Zlup doesn't support while loops (NASA Power of 10: bounded iteration only).
            // The linter should have already flagged unbounded while loops.
            // Transform to a bounded for loop with max iterations and break condition.
            let condition = stmt
                .condition
                .as_ref()
                .ok_or(TransformError::MissingField("condition"))?;

            let mut body = transform_block_with_ctx(&stmt.body, ctx)?;

            // Handle condition similar to if statements
            let cond_expr = transform_expr(condition);
            let bool_condition = match &condition.kind {
                ExprKind::Ident => zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
                    left: cond_expr,
                    op: zlup_ast::BinaryOp::Ne,
                    right: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                        value: 0,
                        suffix: None,
                        location: None,
                    }),
                    location: None,
                })),
                _ => cond_expr,
            };

            // Insert: if (negated_condition) { break; } at the start of body
            // We invert comparison operators directly to avoid precedence issues
            // where `!i < limit` parses as `(!i) < limit` instead of `!(i < limit)`.
            let negated_condition = negate_condition(&bool_condition);
            let break_if_done = zlup_ast::Stmt::If(zlup_ast::IfStmt {
                condition: negated_condition,
                capture: None,
                then_body: zlup_ast::Block {
                    label: None,
                    attrs: Vec::new(),
                    statements: vec![zlup_ast::Stmt::Break(zlup_ast::BreakStmt {
                        label: None,
                        value: None,
                        location: None,
                    })],
                    trailing_expr: None,
                    location: None,
                },
                else_body: None,
                location: None,
            });
            body.statements.insert(0, break_if_done);

            // Use a bounded for loop (max 1000000 iterations as safety bound)
            Ok(vec![zlup_ast::Stmt::For(zlup_ast::ForStmt {
                label: None,
                is_inline: false,
                range: zlup_ast::ForRange::Range {
                    start: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                        value: 0,
                        suffix: None,
                        location: None,
                    }),
                    end: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                        value: 1000000,
                        suffix: None,
                        location: None,
                    }),
                },
                captures: vec!["_while_iter".to_string()],
                body,
                location: None,
            })])
        }

        StmtKind::Break => Ok(vec![zlup_ast::Stmt::Break(zlup_ast::BreakStmt {
            label: None,
            value: None,
            location: None,
        })]),

        StmtKind::Continue => Ok(vec![zlup_ast::Stmt::Continue(zlup_ast::ContinueStmt {
            label: None,
            location: None,
        })]),

        StmtKind::If => {
            let condition = stmt
                .condition
                .as_ref()
                .ok_or(TransformError::MissingField("condition"))?;

            let then_body = transform_block_with_ctx(&stmt.then_body, ctx)?;
            let else_body = if stmt.else_body.is_empty() {
                None
            } else {
                Some(zlup_ast::ElseBranch::Else(transform_block_with_ctx(
                    &stmt.else_body,
                    ctx,
                )?))
            };

            // In Guppy, measurement results (bool) are used directly as conditions.
            // In Zlup, measurements return u1, and if conditions require bool.
            // We wrap simple identifiers in `!= 0` to convert u1 to bool.
            // But we don't wrap comparisons or boolean ops since they already return bool.
            let cond_expr = transform_expr(condition);
            let bool_condition = match &condition.kind {
                // Simple identifiers (like measurement results) need != 0 conversion
                ExprKind::Ident => zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
                    left: cond_expr,
                    op: zlup_ast::BinaryOp::Ne,
                    right: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                        value: 0,
                        suffix: None,
                        location: None,
                    }),
                    location: None,
                })),
                // Comparisons and binary ops already return bool
                _ => cond_expr,
            };

            Ok(vec![zlup_ast::Stmt::If(zlup_ast::IfStmt {
                condition: bool_condition,
                capture: None,
                then_body,
                else_body,
                location: None,
            })])
        }

        StmtKind::Assign => {
            let target = stmt
                .target
                .as_ref()
                .ok_or(TransformError::MissingField("target"))?;
            let value = stmt
                .value
                .as_ref()
                .ok_or(TransformError::MissingField("value"))?;

            // For simple identifier targets, track if this is a new variable or reassignment
            // by checking if ctx has seen this name before. If not seen, create a binding.
            // If seen, create an assignment.
            if target.kind == ExprKind::Ident {
                let name = target
                    .name
                    .as_ref()
                    .ok_or(TransformError::MissingField("name"))?;
                if ctx.is_declared(name) {
                    // Reassignment to existing variable
                    let mut statements = vec![zlup_ast::Stmt::Assign(zlup_ast::AssignStmt {
                        target: transform_expr(target),
                        op: zlup_ast::AssignOp::Assign,
                        value: transform_expr(value),
                        location: None,
                    })];
                    if let Some(refresh) = converted_angle_assignment(name, ctx) {
                        statements.push(refresh);
                    }
                    Ok(statements)
                } else {
                    // New variable declaration
                    ctx.declare_var(name);
                    let mut statements = vec![zlup_ast::Stmt::Binding(zlup_ast::Binding {
                        name: name.clone(),
                        ty: None,
                        value: Some(transform_expr(value)),
                        is_mutable: ctx.mutable_vars.contains(name),
                        is_pub: false,
                        doc_comment: None,
                        location: None,
                    })];
                    if let Some(binding) = converted_angle_binding(name, ctx) {
                        statements.push(binding);
                    }
                    Ok(statements)
                }
            } else {
                Ok(vec![zlup_ast::Stmt::Assign(zlup_ast::AssignStmt {
                    target: transform_expr(target),
                    op: zlup_ast::AssignOp::Assign,
                    value: transform_expr(value),
                    location: None,
                })])
            }
        }

        StmtKind::Return => {
            let value = stmt.return_value.as_ref().map(transform_expr);

            Ok(vec![zlup_ast::Stmt::Return(zlup_ast::ReturnStmt {
                value,
                location: None,
            })])
        }

        StmtKind::Expr => {
            let expr = stmt
                .expr
                .as_ref()
                .ok_or(TransformError::MissingField("expr"))?;

            Ok(vec![zlup_ast::Stmt::Expr(zlup_ast::ExprStmt {
                expr: transform_expr(expr),
                attrs: Vec::new(),
                location: None,
            })])
        }

        StmtKind::Binding => {
            let name = stmt
                .name
                .as_ref()
                .ok_or(TransformError::MissingField("name"))?;
            let ty = stmt.ty.as_ref().map(transform_type);
            let value = stmt.value.as_ref().map(transform_expr);

            // Register the variable as declared
            ctx.declare_var(name);

            let mut statements = vec![zlup_ast::Stmt::Binding(zlup_ast::Binding {
                name: name.clone(),
                ty,
                value,
                is_mutable: stmt.is_mutable.unwrap_or(false),
                is_pub: false,
                doc_comment: None,
                location: None,
            })];
            if let Some(binding) = converted_angle_binding(name, ctx) {
                statements.push(binding);
            }
            Ok(statements)
        }

        StmtKind::Barrier => Ok(vec![zlup_ast::Stmt::Barrier(zlup_ast::BarrierOp {
            allocators: Vec::new(),
            location: None,
        })]),

        StmtKind::Result => {
            let tag = stmt
                .tag
                .as_ref()
                .ok_or(TransformError::MissingField("tag"))?;
            let value = stmt
                .value
                .as_ref()
                .ok_or(TransformError::MissingField("value"))?;

            // result(tag, value) becomes an expression statement with ResultExpr
            Ok(vec![zlup_ast::Stmt::Expr(zlup_ast::ExprStmt {
                expr: zlup_ast::Expr::Result(Box::new(zlup_ast::ResultExpr {
                    tag: tag.clone(),
                    value: transform_expr(value),
                    location: None,
                })),
                attrs: Vec::new(),
                location: None,
            })])
        }
    }
}

fn transform_angle_expr(expr: &Expr, ctx: &TransformContext) -> zlup_ast::Expr {
    if expr.kind == ExprKind::Ident
        && let Some(name) = expr.name.as_deref()
        && let Some(angle_name) = ctx.angle_bindings.get(name)
    {
        return ident_expr(angle_name);
    }

    if expr.kind == ExprKind::Binary {
        let left = expr.left.as_deref();
        let right = expr.right.as_deref();
        if let (Some(left), Some(right), Some(op)) = (left, right, expr.op.as_deref()) {
            let left_is_angle = expr_references_angle(left, &ctx.angle_vars);
            let right_is_angle = expr_references_angle(right, &ctx.angle_vars);
            let (left, right) = match op {
                "add" | "sub" => (
                    transform_angle_expr(left, ctx),
                    transform_angle_expr(right, ctx),
                ),
                "mul" if left_is_angle && !right_is_angle => {
                    (transform_angle_expr(left, ctx), transform_expr(right))
                }
                "mul" if right_is_angle && !left_is_angle => {
                    (transform_expr(left), transform_angle_expr(right, ctx))
                }
                "div" | "floordiv" if left_is_angle && !right_is_angle => {
                    (transform_angle_expr(left, ctx), transform_expr(right))
                }
                _ => return transform_angle_origin(expr),
            };
            return zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
                op: transform_binary_op(op)
                    .unwrap_or_else(|_| panic!("IR contains unknown binary operator: {op}")),
                left,
                right,
                location: None,
            }));
        }
    }

    if expr.kind == ExprKind::Unary
        && let (Some(operand), Some(op)) = (expr.operand.as_deref(), expr.op.as_deref())
        && expr_references_angle(operand, &ctx.angle_vars)
    {
        return zlup_ast::Expr::Unary(Box::new(zlup_ast::UnaryExpr {
            op: transform_unary_op(op)
                .unwrap_or_else(|_| panic!("IR contains unknown unary operator: {op}")),
            operand: transform_angle_expr(operand, ctx),
            location: None,
        }));
    }

    transform_angle_origin(expr)
}

fn transform_angle_origin(expr: &Expr) -> zlup_ast::Expr {
    angle_from_half_turn_expr(transform_expr(expr))
}

fn angle_from_half_turn_expr(expr: zlup_ast::Expr) -> zlup_ast::Expr {
    // Guppy's `angle` uses half-turns (`crates/pecos-core/src/angle.rs:173`),
    // while Zlup's native angle unit is turns.
    zlup_ast::Expr::AngleLit(Box::new(zlup_ast::AngleLit {
        value: divide_by_two(expr),
        unit: zlup_ast::AngleUnit::Turns,
        location: None,
    }))
}

fn converted_angle_binding(name: &str, ctx: &TransformContext) -> Option<zlup_ast::Stmt> {
    let angle_name = ctx.angle_bindings.get(name)?;
    Some(zlup_ast::Stmt::Binding(zlup_ast::Binding {
        name: angle_name.clone(),
        ty: Some(zlup_ast::TypeExpr::Primitive(zlup_ast::PrimitiveType::A64)),
        value: Some(angle_from_half_turn_expr(ident_expr(name))),
        is_mutable: ctx.mutable_vars.contains(name),
        is_pub: false,
        doc_comment: None,
        location: None,
    }))
}

fn converted_angle_assignment(name: &str, ctx: &TransformContext) -> Option<zlup_ast::Stmt> {
    let angle_name = ctx.angle_bindings.get(name)?;
    Some(zlup_ast::Stmt::Assign(zlup_ast::AssignStmt {
        target: ident_expr(angle_name),
        op: zlup_ast::AssignOp::Assign,
        value: angle_from_half_turn_expr(ident_expr(name)),
        location: None,
    }))
}

fn ident_expr(name: &str) -> zlup_ast::Expr {
    zlup_ast::Expr::Ident(zlup_ast::Ident {
        name: name.to_string(),
        location: None,
    })
}

fn divide_by_two(expr: zlup_ast::Expr) -> zlup_ast::Expr {
    zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
        op: zlup_ast::BinaryOp::Div,
        left: expr,
        right: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
            value: 2,
            suffix: None,
            location: None,
        }),
        location: None,
    }))
}

fn make_angle_binding_names(
    func: &Function,
    angle_vars: &std::collections::BTreeSet<String>,
) -> (BTreeMap<String, String>, std::collections::BTreeSet<String>) {
    let mut used_names = func
        .params
        .iter()
        .map(|param| param.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    collect_bound_names(&func.body, &mut used_names);

    let mut declared_names = func
        .params
        .iter()
        .map(|param| param.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut binding_origins = declared_names.clone();
    let mut mutable_vars = std::collections::BTreeSet::new();
    collect_binding_origins(
        &func.body,
        &mut declared_names,
        &mut binding_origins,
        &mut mutable_vars,
    );

    let bindings = angle_vars
        .iter()
        .filter(|name| binding_origins.contains(*name))
        .map(|name| {
            let base = format!("__zlup_angle_{name}");
            let mut fresh = base.clone();
            let mut suffix = 2;
            while !used_names.insert(fresh.clone()) {
                fresh = format!("{base}_{suffix}");
                suffix += 1;
            }
            (name.clone(), fresh)
        })
        .collect();

    (bindings, mutable_vars)
}

fn collect_binding_origins(
    stmts: &[Stmt],
    declared_names: &mut std::collections::BTreeSet<String>,
    binding_origins: &mut std::collections::BTreeSet<String>,
    mutable_vars: &mut std::collections::BTreeSet<String>,
) {
    for stmt in stmts {
        match stmt.kind {
            StmtKind::Qalloc => {
                if let Some(name) = &stmt.name {
                    declared_names.insert(name.clone());
                }
            }
            StmtKind::Measure => {
                declared_names.extend(stmt.results.iter().cloned());
            }
            StmtKind::For => {
                if let Some(var) = &stmt.var {
                    declared_names.insert(var.clone());
                }
            }
            StmtKind::Binding => {
                if let Some(name) = &stmt.name {
                    declared_names.insert(name.clone());
                    binding_origins.insert(name.clone());
                    if stmt.is_mutable.unwrap_or(false) {
                        mutable_vars.insert(name.clone());
                    }
                }
            }
            StmtKind::Assign => {
                if let Some(target) = &stmt.target
                    && target.kind == ExprKind::Ident
                    && let Some(name) = &target.name
                {
                    if declared_names.insert(name.clone()) {
                        binding_origins.insert(name.clone());
                    } else {
                        mutable_vars.insert(name.clone());
                    }
                }
            }
            _ => {}
        }
        collect_binding_origins(&stmt.body, declared_names, binding_origins, mutable_vars);
        collect_binding_origins(
            &stmt.then_body,
            declared_names,
            binding_origins,
            mutable_vars,
        );
        collect_binding_origins(
            &stmt.else_body,
            declared_names,
            binding_origins,
            mutable_vars,
        );
    }
}

fn collect_bound_names(stmts: &[Stmt], names: &mut std::collections::BTreeSet<String>) {
    for stmt in stmts {
        if let Some(name) = &stmt.name {
            names.insert(name.clone());
        }
        if let Some(var) = &stmt.var {
            names.insert(var.clone());
        }
        names.extend(stmt.results.iter().cloned());
        if let Some(target) = &stmt.target
            && target.kind == ExprKind::Ident
            && let Some(name) = &target.name
        {
            names.insert(name.clone());
        }
        collect_bound_names(&stmt.body, names);
        collect_bound_names(&stmt.then_body, names);
        collect_bound_names(&stmt.else_body, names);
    }
}

fn infer_angle_variables(stmts: &[Stmt]) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    collect_gate_angle_origins(stmts, &mut names);

    loop {
        let previous_len = names.len();
        propagate_angle_bindings(stmts, &mut names);
        if names.len() == previous_len {
            break;
        }
    }
    names
}

fn collect_gate_angle_origins(stmts: &[Stmt], names: &mut std::collections::BTreeSet<String>) {
    for stmt in stmts {
        if stmt.kind == StmtKind::Gate {
            for param in &stmt.params {
                collect_angle_origin_names(param, names);
            }
        }
        collect_gate_angle_origins(&stmt.body, names);
        collect_gate_angle_origins(&stmt.then_body, names);
        collect_gate_angle_origins(&stmt.else_body, names);
    }
}

fn collect_angle_origin_names(expr: &Expr, names: &mut std::collections::BTreeSet<String>) {
    match expr.kind {
        ExprKind::Ident => {
            if let Some(name) = &expr.name {
                names.insert(name.clone());
            }
        }
        ExprKind::Binary => {
            let left = expr.left.as_deref();
            let right = expr.right.as_deref();
            match expr.op.as_deref() {
                Some("add") | Some("sub") => {
                    if let Some(left) = left {
                        collect_angle_origin_names(left, names);
                    }
                    if let Some(right) = right {
                        collect_angle_origin_names(right, names);
                    }
                }
                Some("mul") => {
                    if left.is_some_and(|side| side.kind == ExprKind::Literal) {
                        if let Some(right) = right {
                            collect_angle_origin_names(right, names);
                        }
                    } else if right.is_some_and(|side| side.kind == ExprKind::Literal)
                        && let Some(left) = left
                    {
                        collect_angle_origin_names(left, names);
                    }
                }
                Some("div") | Some("floordiv") => {
                    if let Some(left) = left {
                        collect_angle_origin_names(left, names);
                    }
                }
                _ => {}
            }
        }
        ExprKind::Unary => {
            if let Some(operand) = expr.operand.as_deref() {
                collect_angle_origin_names(operand, names);
            }
        }
        _ => {}
    }
}

fn propagate_angle_bindings(stmts: &[Stmt], names: &mut std::collections::BTreeSet<String>) {
    for stmt in stmts {
        let target_name = match stmt.kind {
            StmtKind::Binding => stmt.name.as_deref(),
            StmtKind::Assign => stmt
                .target
                .as_ref()
                .filter(|target| target.kind == ExprKind::Ident)
                .and_then(|target| target.name.as_deref()),
            _ => None,
        };
        if let (Some(name), Some(value)) = (target_name, stmt.value.as_ref())
            && expr_references_angle(value, names)
        {
            names.insert(name.to_string());
        }
        propagate_angle_bindings(&stmt.body, names);
        propagate_angle_bindings(&stmt.then_body, names);
        propagate_angle_bindings(&stmt.else_body, names);
    }
}

fn expr_references_angle(expr: &Expr, angle_vars: &std::collections::BTreeSet<String>) -> bool {
    match expr.kind {
        ExprKind::Ident => expr
            .name
            .as_ref()
            .is_some_and(|name| angle_vars.contains(name)),
        ExprKind::Binary => {
            expr.left
                .as_deref()
                .is_some_and(|left| expr_references_angle(left, angle_vars))
                || expr
                    .right
                    .as_deref()
                    .is_some_and(|right| expr_references_angle(right, angle_vars))
        }
        ExprKind::Unary => expr
            .operand
            .as_deref()
            .is_some_and(|operand| expr_references_angle(operand, angle_vars)),
        _ => false,
    }
}

fn transform_expr(expr: &Expr) -> zlup_ast::Expr {
    match expr.kind {
        ExprKind::Literal => {
            if let Some(value) = &expr.value {
                match value {
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                                value: i as i128,
                                suffix: None,
                                location: None,
                            })
                        } else if let Some(f) = n.as_f64() {
                            zlup_ast::Expr::FloatLit(zlup_ast::FloatLit {
                                value: f,
                                suffix: None,
                                location: None,
                            })
                        } else {
                            zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                                value: 0,
                                suffix: None,
                                location: None,
                            })
                        }
                    }
                    serde_json::Value::Bool(b) => zlup_ast::Expr::BoolLit(zlup_ast::BoolLit {
                        value: *b,
                        location: None,
                    }),
                    serde_json::Value::String(s) => {
                        zlup_ast::Expr::StringLit(zlup_ast::StringLit {
                            value: s.clone(),
                            location: None,
                        })
                    }
                    serde_json::Value::Null => {
                        zlup_ast::Expr::Null(zlup_ast::NullLit { location: None })
                    }
                    _ => zlup_ast::Expr::Unit(zlup_ast::UnitLit { location: None }),
                }
            } else {
                zlup_ast::Expr::Unit(zlup_ast::UnitLit { location: None })
            }
        }

        ExprKind::Ident => {
            let name = expr.name.as_deref().unwrap_or("_");
            zlup_ast::Expr::Ident(zlup_ast::Ident {
                name: name.to_string(),
                location: None,
            })
        }

        ExprKind::Index => {
            let array = expr.array.as_deref().unwrap_or("_");
            let index = expr
                .index
                .as_ref()
                .map(|i| transform_expr(i))
                .unwrap_or_else(|| {
                    zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                        value: 0,
                        suffix: None,
                        location: None,
                    })
                });

            zlup_ast::Expr::Index(Box::new(zlup_ast::IndexExpr {
                object: zlup_ast::Expr::Ident(zlup_ast::Ident {
                    name: array.to_string(),
                    location: None,
                }),
                index,
                location: None,
            }))
        }

        ExprKind::Binary => {
            let left = expr
                .left
                .as_ref()
                .map(|l| transform_expr(l))
                .unwrap_or_else(|| zlup_ast::Expr::Unit(zlup_ast::UnitLit { location: None }));
            let right = expr
                .right
                .as_ref()
                .map(|r| transform_expr(r))
                .unwrap_or_else(|| zlup_ast::Expr::Unit(zlup_ast::UnitLit { location: None }));
            let op_str = expr.op.as_deref().unwrap_or("unknown");
            let op = transform_binary_op(op_str)
                .unwrap_or_else(|_| panic!("IR contains unknown binary operator: {}", op_str));

            zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
                op,
                left,
                right,
                location: None,
            }))
        }

        ExprKind::Unary => {
            let operand = expr
                .operand
                .as_ref()
                .map(|o| transform_expr(o))
                .unwrap_or_else(|| zlup_ast::Expr::Unit(zlup_ast::UnitLit { location: None }));
            let op_str = expr.op.as_deref().unwrap_or("unknown");
            let op = transform_unary_op(op_str)
                .unwrap_or_else(|_| panic!("IR contains unknown unary operator: {}", op_str));

            zlup_ast::Expr::Unary(Box::new(zlup_ast::UnaryExpr {
                op,
                operand,
                location: None,
            }))
        }

        ExprKind::Call => {
            let callee = expr.callee.as_deref().unwrap_or("_");
            let args: Vec<zlup_ast::Expr> = expr.args.iter().map(transform_expr).collect();

            zlup_ast::Expr::Call(Box::new(zlup_ast::CallExpr {
                callee: zlup_ast::Expr::Ident(zlup_ast::Ident {
                    name: callee.to_string(),
                    location: None,
                }),
                args,
                location: None,
            }))
        }

        ExprKind::Field => {
            let object = expr
                .object
                .as_ref()
                .map(|o| transform_expr(o))
                .unwrap_or_else(|| zlup_ast::Expr::Unit(zlup_ast::UnitLit { location: None }));
            let field = expr.field.as_deref().unwrap_or("_");

            zlup_ast::Expr::Field(Box::new(zlup_ast::FieldExpr {
                object,
                field: field.to_string(),
                location: None,
            }))
        }

        ExprKind::Tuple => {
            let elements: Vec<zlup_ast::Expr> = expr.args.iter().map(transform_expr).collect();
            zlup_ast::Expr::Tuple(Box::new(zlup_ast::TupleExpr {
                elements,
                location: None,
            }))
        }
    }
}

fn transform_slot_ref(expr: &Expr) -> Result<zlup_ast::SlotRef, TransformError> {
    match expr.kind {
        ExprKind::Index => {
            let allocator = expr.array.as_ref().ok_or(TransformError::InvalidSlotRef)?;
            let index = expr
                .index
                .as_ref()
                .map(|i| transform_expr(i))
                .ok_or(TransformError::InvalidSlotRef)?;

            Ok(zlup_ast::SlotRef {
                allocator: allocator.clone(),
                index: Box::new(index),
                location: None,
            })
        }
        ExprKind::Ident => {
            // Single qubit variable (e.g., q0 from q0 = qubit())
            // Treat as allocator[0] since it's a single-qubit allocation
            let name = expr.name.as_ref().ok_or(TransformError::InvalidSlotRef)?;
            Ok(zlup_ast::SlotRef {
                allocator: name.clone(),
                index: Box::new(zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                    value: 0,
                    suffix: None,
                    location: None,
                })),
                location: None,
            })
        }
        _ => Err(TransformError::InvalidSlotRef),
    }
}

fn transform_gate_kind(gate: GateKind) -> ZlupGateKind {
    match gate {
        GateKind::H => ZlupGateKind::H,
        GateKind::X => ZlupGateKind::X,
        GateKind::Y => ZlupGateKind::Y,
        GateKind::Z => ZlupGateKind::Z,
        GateKind::T => ZlupGateKind::T,
        GateKind::Tdg => ZlupGateKind::Tdg,
        GateKind::S => ZlupGateKind::SZ,
        GateKind::Sdg => ZlupGateKind::SZdg,
        GateKind::Sx => ZlupGateKind::SX,
        GateKind::Sy => ZlupGateKind::SY,
        GateKind::Sz => ZlupGateKind::SZ,
        GateKind::Rx => ZlupGateKind::RX,
        GateKind::Ry => ZlupGateKind::RY,
        GateKind::Rz => ZlupGateKind::RZ,
        GateKind::Cx => ZlupGateKind::CX,
        GateKind::Cy => ZlupGateKind::CY,
        GateKind::Cz => ZlupGateKind::CZ,
        GateKind::Swap => ZlupGateKind::SWAP,
        GateKind::Iswap => ZlupGateKind::ISWAP,
        GateKind::Ccx => ZlupGateKind::CCX,
        GateKind::Pz => ZlupGateKind::PZ,
    }
}

/// Negate a condition expression by inverting comparison operators.
/// This avoids operator precedence issues with the unary `!` operator.
fn negate_condition(expr: &zlup_ast::Expr) -> zlup_ast::Expr {
    match expr {
        zlup_ast::Expr::Binary(bin) => {
            // Invert comparison operators
            let inverted_op = match bin.op {
                zlup_ast::BinaryOp::Lt => Some(zlup_ast::BinaryOp::Ge),
                zlup_ast::BinaryOp::Le => Some(zlup_ast::BinaryOp::Gt),
                zlup_ast::BinaryOp::Gt => Some(zlup_ast::BinaryOp::Le),
                zlup_ast::BinaryOp::Ge => Some(zlup_ast::BinaryOp::Lt),
                zlup_ast::BinaryOp::Eq => Some(zlup_ast::BinaryOp::Ne),
                zlup_ast::BinaryOp::Ne => Some(zlup_ast::BinaryOp::Eq),
                _ => None,
            };
            if let Some(new_op) = inverted_op {
                zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
                    left: bin.left.clone(),
                    op: new_op,
                    right: bin.right.clone(),
                    location: bin.location.clone(),
                }))
            } else {
                // For non-comparison binary ops (and, or, etc.), use == false
                zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
                    left: expr.clone(),
                    op: zlup_ast::BinaryOp::Eq,
                    right: zlup_ast::Expr::BoolLit(zlup_ast::BoolLit {
                        value: false,
                        location: None,
                    }),
                    location: None,
                }))
            }
        }
        // For other expressions (idents, etc.), use == 0 to negate
        _ => zlup_ast::Expr::Binary(Box::new(zlup_ast::BinaryExpr {
            left: expr.clone(),
            op: zlup_ast::BinaryOp::Eq,
            right: zlup_ast::Expr::IntLit(zlup_ast::IntLit {
                value: 0,
                suffix: None,
                location: None,
            }),
            location: None,
        })),
    }
}

fn transform_binary_op(op: &str) -> Result<zlup_ast::BinaryOp, TransformError> {
    match op {
        "add" => Ok(zlup_ast::BinaryOp::Add),
        "sub" => Ok(zlup_ast::BinaryOp::Sub),
        "mul" => Ok(zlup_ast::BinaryOp::Mul),
        "div" => Ok(zlup_ast::BinaryOp::Div),
        "floordiv" => Ok(zlup_ast::BinaryOp::Div), // Integer division in Zlup
        "mod" => Ok(zlup_ast::BinaryOp::Mod),
        "eq" => Ok(zlup_ast::BinaryOp::Eq),
        "ne" => Ok(zlup_ast::BinaryOp::Ne),
        "lt" => Ok(zlup_ast::BinaryOp::Lt),
        "le" => Ok(zlup_ast::BinaryOp::Le),
        "gt" => Ok(zlup_ast::BinaryOp::Gt),
        "ge" => Ok(zlup_ast::BinaryOp::Ge),
        "and" => Ok(zlup_ast::BinaryOp::And),
        "or" => Ok(zlup_ast::BinaryOp::Or),
        "bitand" => Ok(zlup_ast::BinaryOp::BitAnd),
        "bitor" => Ok(zlup_ast::BinaryOp::BitOr),
        "bitxor" => Ok(zlup_ast::BinaryOp::BitXor),
        "shl" => Ok(zlup_ast::BinaryOp::Shl),
        "shr" => Ok(zlup_ast::BinaryOp::Shr),
        _ => Err(TransformError::UnsupportedOp(format!(
            "unknown binary operator: {}",
            op
        ))),
    }
}

fn transform_unary_op(op: &str) -> Result<zlup_ast::UnaryOp, TransformError> {
    match op {
        "neg" => Ok(zlup_ast::UnaryOp::Neg),
        "not" => Ok(zlup_ast::UnaryOp::Not),
        "bitnot" => Ok(zlup_ast::UnaryOp::BitNot),
        _ => Err(TransformError::UnsupportedOp(format!(
            "unknown unary operator: {}",
            op
        ))),
    }
}

/// Transform error.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("Missing required field: {0}")]
    MissingField(&'static str),

    #[error("Invalid slot reference")]
    InvalidSlotRef,

    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    #[error("Unsupported operator: {0}")]
    UnsupportedOp(String),

    #[error("Generated code validation failed: {0}")]
    ValidationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_empty_program() {
        let ir = GuppyIR {
            version: "0.1.0".to_string(),
            functions: vec![],
            source_file: None,
        };

        let result = transform(&ir).unwrap();
        assert!(result.declarations.is_empty());
    }

    #[test]
    fn test_transform_simple_function() {
        let ir = GuppyIR {
            version: "0.1.0".to_string(),
            functions: vec![Function {
                name: "main".to_string(),
                params: vec![],
                return_type: None,
                body: vec![],
                is_pub: None,
                location: None,
            }],
            source_file: None,
        };

        let result = transform(&ir).unwrap();
        assert_eq!(result.declarations.len(), 1);
    }
}
