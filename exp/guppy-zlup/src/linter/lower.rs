//! Lower Python AST to Guppy AST.
//!
//! This module converts the rustpython-parser AST into our clean Guppy AST.
//! All Python-specific quirks are handled here, isolating the rest of the
//! codebase from parser API changes.

use rustpython_parser::ast::{self, Constant, Expr as PyExpr, Mod, Ranged, Stmt as PyStmt};

use super::ast::{
    AssignTarget, BinOp, BoolOpKind, CmpOp, Comprehension, ExceptHandler, Expr, ForIter, Function,
    GateKind, Keyword, Module, Param, PrimitiveType, Span, Stmt, Type, UnaryOp, WithItem,
};

/// Errors that can occur during lowering.
#[derive(Debug, Clone)]
pub enum LowerError {
    /// The input was not a module.
    NotAModule,
    /// Unsupported Python construct.
    Unsupported(String, Span),
    /// Parse error from rustpython.
    Parse(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::NotAModule => write!(f, "expected a module"),
            LowerError::Unsupported(msg, _) => write!(f, "unsupported: {}", msg),
            LowerError::Parse(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl std::error::Error for LowerError {}

/// Lower Python source code to a Guppy AST.
pub fn lower_source(source: &str, filename: &str) -> Result<Module, LowerError> {
    let parsed = rustpython_parser::parse(source, rustpython_parser::Mode::Module, filename)
        .map_err(|e| LowerError::Parse(e.to_string()))?;

    lower_mod(&parsed)
}

/// Lower a Python module AST to a Guppy module.
pub fn lower_mod(parsed: &Mod) -> Result<Module, LowerError> {
    let Mod::Module(module) = parsed else {
        return Err(LowerError::NotAModule);
    };

    let mut functions = Vec::new();

    for stmt in &module.body {
        if let PyStmt::FunctionDef(func) = stmt {
            functions.push(lower_function(func)?);
        }
        // Skip other top-level statements for now (imports, classes, etc.)
    }

    Ok(Module {
        functions,
        span: Span::default(),
    })
}

/// Lower a Python function definition to a Guppy function.
fn lower_function(func: &ast::StmtFunctionDef) -> Result<Function, LowerError> {
    let span = make_span(func.range);

    let params = func
        .args
        .args
        .iter()
        .map(|arg| {
            let ty = arg.def.annotation.as_ref().map(|ann| lower_type(ann));
            Param {
                name: arg.def.arg.to_string(),
                ty,
                span: make_span(arg.def.range),
            }
        })
        .collect();

    let return_type = func.returns.as_ref().map(|ann| lower_type(ann));

    let body = func
        .body
        .iter()
        .filter_map(|stmt| lower_stmt(stmt).ok())
        .collect();

    let decorators = func
        .decorator_list
        .iter()
        .filter_map(|d| {
            if let PyExpr::Name(name) = d {
                Some(name.id.to_string())
            } else {
                None
            }
        })
        .collect();

    Ok(Function {
        name: func.name.to_string(),
        params,
        return_type,
        body,
        decorators,
        span,
    })
}

/// Lower a Python type annotation to a Guppy type.
fn lower_type(ann: &PyExpr) -> Type {
    match ann {
        PyExpr::Name(name) => {
            let name_str = name.id.as_str();
            match name_str {
                "int" => Type::Primitive(PrimitiveType::Int),
                "float" => Type::Primitive(PrimitiveType::Float),
                "bool" => Type::Primitive(PrimitiveType::Bool),
                "str" => Type::Primitive(PrimitiveType::Str),
                "None" => Type::Primitive(PrimitiveType::None),
                "qubit" => Type::Qubit { size: None },
                _ => Type::Named {
                    name: name_str.to_string(),
                },
            }
        }
        PyExpr::Subscript(sub) => {
            if let PyExpr::Name(name) = sub.value.as_ref() {
                let name_str = name.id.as_str();
                match name_str {
                    "list" | "List" => Type::Array {
                        element: Box::new(lower_type(&sub.slice)),
                    },
                    "Optional" => Type::Optional {
                        inner: Box::new(lower_type(&sub.slice)),
                    },
                    "qubit" => Type::Qubit {
                        size: lower_expr(&sub.slice).ok().map(Box::new),
                    },
                    "Tuple" | "tuple" => {
                        if let PyExpr::Tuple(tuple) = sub.slice.as_ref() {
                            Type::Tuple {
                                elements: tuple.elts.iter().map(lower_type).collect(),
                            }
                        } else {
                            Type::Tuple {
                                elements: vec![lower_type(&sub.slice)],
                            }
                        }
                    }
                    _ => Type::Named {
                        name: name_str.to_string(),
                    },
                }
            } else {
                Type::Unknown
            }
        }
        PyExpr::Constant(c) if matches!(c.value, Constant::None) => {
            Type::Primitive(PrimitiveType::None)
        }
        _ => Type::Unknown,
    }
}

/// Lower a Python statement to a Guppy statement.
fn lower_stmt(stmt: &PyStmt) -> Result<Stmt, LowerError> {
    let span = stmt_span(stmt);

    match stmt {
        PyStmt::Assign(assign) => lower_assign(assign),

        PyStmt::AnnAssign(ann) => {
            let target = if let PyExpr::Name(name) = ann.target.as_ref() {
                name.id.to_string()
            } else {
                return Err(LowerError::Unsupported(
                    "complex annotated assignment target".into(),
                    span,
                ));
            };

            Ok(Stmt::AnnAssign {
                target,
                annotation: lower_type(&ann.annotation),
                value: ann.value.as_ref().and_then(|v| lower_expr(v).ok()),
                span,
            })
        }

        PyStmt::AugAssign(aug) => {
            let target = lower_assign_target(&aug.target)?;
            let op = lower_operator(aug.op);
            let value = lower_expr(&aug.value)?;

            Ok(Stmt::AugAssign {
                target,
                op,
                value,
                span,
            })
        }

        PyStmt::Expr(expr_stmt) => {
            let expr = &expr_stmt.value;

            // Check if this is a gate call
            if let PyExpr::Call(call) = expr.as_ref()
                && let Some(stmt) = try_lower_gate_call(call, span)?
            {
                return Ok(stmt);
            }

            Ok(Stmt::Expr {
                value: lower_expr(expr)?,
                span,
            })
        }

        PyStmt::For(for_stmt) => {
            let var = if let PyExpr::Name(name) = for_stmt.target.as_ref() {
                name.id.to_string()
            } else {
                return Err(LowerError::Unsupported("complex for target".into(), span));
            };

            let iter = lower_for_iter(&for_stmt.iter)?;

            let body = for_stmt
                .body
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            let orelse = for_stmt
                .orelse
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            Ok(Stmt::For {
                var,
                iter,
                body,
                orelse,
                span,
            })
        }

        PyStmt::While(while_stmt) => {
            let test = lower_expr(&while_stmt.test)?;

            let body = while_stmt
                .body
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            let orelse = while_stmt
                .orelse
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            Ok(Stmt::While {
                test,
                body,
                orelse,
                span,
            })
        }

        PyStmt::If(if_stmt) => {
            let test = lower_expr(&if_stmt.test)?;

            let body = if_stmt
                .body
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            let orelse = if_stmt
                .orelse
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            Ok(Stmt::If {
                test,
                body,
                orelse,
                span,
            })
        }

        PyStmt::Return(ret) => {
            let value = ret.value.as_ref().and_then(|v| lower_expr(v).ok());
            Ok(Stmt::Return { value, span })
        }

        PyStmt::Pass(_) => Ok(Stmt::Pass { span }),

        PyStmt::Break(_) => Ok(Stmt::Break { span }),

        PyStmt::Continue(_) => Ok(Stmt::Continue { span }),

        PyStmt::Try(try_stmt) => {
            let body = try_stmt
                .body
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            let handlers = try_stmt
                .handlers
                .iter()
                .map(|h| {
                    let ast::ExceptHandler::ExceptHandler(handler) = h;
                    ExceptHandler {
                        ty: handler.type_.as_ref().and_then(|t| lower_expr(t).ok()),
                        name: handler.name.as_ref().map(|n| n.to_string()),
                        body: handler
                            .body
                            .iter()
                            .filter_map(|s| lower_stmt(s).ok())
                            .collect(),
                        span: make_span(handler.range),
                    }
                })
                .collect();

            let orelse = try_stmt
                .orelse
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            let finalbody = try_stmt
                .finalbody
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            Ok(Stmt::Try {
                body,
                handlers,
                orelse,
                finalbody,
                span,
            })
        }

        PyStmt::Assert(assert_stmt) => {
            let test = lower_expr(&assert_stmt.test)?;
            let msg = assert_stmt.msg.as_ref().and_then(|m| lower_expr(m).ok());

            Ok(Stmt::Assert { test, msg, span })
        }

        PyStmt::With(with_stmt) => {
            let items = with_stmt
                .items
                .iter()
                .filter_map(|item| {
                    let context = lower_expr(&item.context_expr).ok()?;
                    let target = item
                        .optional_vars
                        .as_ref()
                        .and_then(|v| lower_assign_target(v).ok());
                    // Use the context expression's range for the span
                    Some(WithItem {
                        context,
                        target,
                        span: make_span(item.context_expr.range()),
                    })
                })
                .collect();

            let body = with_stmt
                .body
                .iter()
                .filter_map(|s| lower_stmt(s).ok())
                .collect();

            Ok(Stmt::With { items, body, span })
        }

        _ => Err(LowerError::Unsupported(
            format!("statement: {:?}", std::mem::discriminant(stmt)),
            span,
        )),
    }
}

/// Try to lower a call expression as a gate or measurement.
fn try_lower_gate_call(call: &ast::ExprCall, span: Span) -> Result<Option<Stmt>, LowerError> {
    // Get the function name
    let func_name = match call.func.as_ref() {
        PyExpr::Name(name) => name.id.as_str(),
        _ => return Ok(None),
    };

    // Check if it's a known gate
    if let Some(gate) = GateKind::from_name(func_name) {
        let targets = call
            .args
            .iter()
            .map(lower_expr)
            .collect::<Result<Vec<_>, _>>()?;

        return Ok(Some(Stmt::Gate {
            gate,
            targets,
            params: Vec::new(),
            span,
        }));
    }

    // Check if it's a measurement
    if func_name == "measure" {
        let target = call
            .args
            .first()
            .map(lower_expr)
            .transpose()?
            .ok_or_else(|| LowerError::Unsupported("measure without target".into(), span))?;

        return Ok(Some(Stmt::Measure {
            target,
            result: None,
            span,
        }));
    }

    // Check if it's a barrier
    if func_name == "barrier" {
        let qubits = call
            .args
            .iter()
            .map(lower_expr)
            .collect::<Result<Vec<_>, _>>()?;

        return Ok(Some(Stmt::Barrier { qubits, span }));
    }

    Ok(None)
}

/// Lower a Python assignment statement.
fn lower_assign(assign: &ast::StmtAssign) -> Result<Stmt, LowerError> {
    let span = make_span(assign.range);

    if assign.targets.len() != 1 {
        return Err(LowerError::Unsupported(
            "multiple assignment targets".into(),
            span,
        ));
    }

    let target_expr = &assign.targets[0];

    // Check for qubit allocation: q = qubit[n]
    if let PyExpr::Subscript(sub) = assign.value.as_ref()
        && let PyExpr::Name(name) = sub.value.as_ref()
        && name.id.as_str() == "qubit"
    {
        let var_name = if let PyExpr::Name(n) = target_expr {
            n.id.to_string()
        } else {
            return Err(LowerError::Unsupported(
                "complex qalloc target".into(),
                span,
            ));
        };

        let size = lower_expr(&sub.slice)?;

        return Ok(Stmt::Qalloc {
            name: var_name,
            size,
            span,
        });
    }

    // Check for measurement: m = measure(q)
    if let PyExpr::Call(call) = assign.value.as_ref()
        && let PyExpr::Name(func_name) = call.func.as_ref()
        && func_name.id.as_str() == "measure"
    {
        let result_name = if let PyExpr::Name(n) = target_expr {
            n.id.to_string()
        } else {
            return Err(LowerError::Unsupported(
                "complex measure target".into(),
                span,
            ));
        };

        let target = call
            .args
            .first()
            .map(lower_expr)
            .transpose()?
            .ok_or_else(|| LowerError::Unsupported("measure without target".into(), span))?;

        return Ok(Stmt::Measure {
            target,
            result: Some(result_name),
            span,
        });
    }

    // Regular assignment
    let target = lower_assign_target(target_expr)?;
    let value = lower_expr(&assign.value)?;

    Ok(Stmt::Assign {
        target,
        value,
        span,
    })
}

/// Lower an assignment target.
fn lower_assign_target(expr: &PyExpr) -> Result<AssignTarget, LowerError> {
    let span = make_span(expr.range());

    match expr {
        PyExpr::Name(name) => Ok(AssignTarget::Name {
            name: name.id.to_string(),
            span,
        }),
        PyExpr::Subscript(sub) => Ok(AssignTarget::Subscript {
            value: Box::new(lower_expr(&sub.value)?),
            slice: Box::new(lower_expr(&sub.slice)?),
            span,
        }),
        PyExpr::Attribute(attr) => Ok(AssignTarget::Attribute {
            value: Box::new(lower_expr(&attr.value)?),
            attr: attr.attr.to_string(),
            span,
        }),
        PyExpr::Tuple(tuple) => Ok(AssignTarget::Tuple {
            elts: tuple
                .elts
                .iter()
                .map(lower_assign_target)
                .collect::<Result<_, _>>()?,
            span,
        }),
        _ => Err(LowerError::Unsupported(
            "complex assignment target".into(),
            span,
        )),
    }
}

/// Lower a for loop iterator.
fn lower_for_iter(expr: &PyExpr) -> Result<ForIter, LowerError> {
    // Check for range(...)
    if let PyExpr::Call(call) = expr
        && let PyExpr::Name(name) = call.func.as_ref()
        && name.id.as_str() == "range"
    {
        match call.args.len() {
            1 => {
                return Ok(ForIter::Range {
                    start: None,
                    end: Box::new(lower_expr(&call.args[0])?),
                    step: None,
                });
            }
            2 => {
                return Ok(ForIter::Range {
                    start: Some(Box::new(lower_expr(&call.args[0])?)),
                    end: Box::new(lower_expr(&call.args[1])?),
                    step: None,
                });
            }
            3 => {
                return Ok(ForIter::Range {
                    start: Some(Box::new(lower_expr(&call.args[0])?)),
                    end: Box::new(lower_expr(&call.args[1])?),
                    step: Some(Box::new(lower_expr(&call.args[2])?)),
                });
            }
            _ => {}
        }
    }

    Ok(ForIter::Iter(Box::new(lower_expr(expr)?)))
}

/// Lower a Python expression to a Guppy expression.
fn lower_expr(expr: &PyExpr) -> Result<Expr, LowerError> {
    let span = make_span(expr.range());

    match expr {
        PyExpr::Constant(c) => match &c.value {
            Constant::Int(i) => {
                let value = i.to_string().parse::<i64>().unwrap_or(0);
                Ok(Expr::IntLit { value, span })
            }
            Constant::Float(f) => Ok(Expr::FloatLit { value: *f, span }),
            Constant::Str(s) => Ok(Expr::StrLit {
                value: s.clone(),
                span,
            }),
            Constant::Bool(b) => Ok(Expr::BoolLit { value: *b, span }),
            Constant::None => Ok(Expr::NoneLit { span }),
            _ => Err(LowerError::Unsupported("constant type".into(), span)),
        },

        PyExpr::Name(name) => Ok(Expr::Name {
            name: name.id.to_string(),
            span,
        }),

        PyExpr::Subscript(sub) => Ok(Expr::Subscript {
            value: Box::new(lower_expr(&sub.value)?),
            slice: Box::new(lower_expr(&sub.slice)?),
            span,
        }),

        PyExpr::Attribute(attr) => Ok(Expr::Attribute {
            value: Box::new(lower_expr(&attr.value)?),
            attr: attr.attr.to_string(),
            span,
        }),

        PyExpr::BinOp(binop) => Ok(Expr::BinOp {
            left: Box::new(lower_expr(&binop.left)?),
            op: lower_operator(binop.op),
            right: Box::new(lower_expr(&binop.right)?),
            span,
        }),

        PyExpr::UnaryOp(unary) => Ok(Expr::UnaryOp {
            op: lower_unary_op(unary.op),
            operand: Box::new(lower_expr(&unary.operand)?),
            span,
        }),

        PyExpr::Compare(cmp) => {
            let ops = cmp.ops.iter().map(|op| lower_cmp_op(*op)).collect();
            let comparators = cmp
                .comparators
                .iter()
                .map(lower_expr)
                .collect::<Result<_, _>>()?;

            Ok(Expr::Compare {
                left: Box::new(lower_expr(&cmp.left)?),
                ops,
                comparators,
                span,
            })
        }

        PyExpr::BoolOp(boolop) => {
            let op = match boolop.op {
                ast::BoolOp::And => BoolOpKind::And,
                ast::BoolOp::Or => BoolOpKind::Or,
            };

            let values = boolop
                .values
                .iter()
                .map(lower_expr)
                .collect::<Result<_, _>>()?;

            Ok(Expr::BoolOp { op, values, span })
        }

        PyExpr::Call(call) => {
            let func = lower_expr(&call.func)?;
            let args = call.args.iter().map(lower_expr).collect::<Result<_, _>>()?;
            let keywords = call
                .keywords
                .iter()
                .map(|kw| {
                    Ok(Keyword {
                        name: kw.arg.as_ref().map(|a| a.to_string()),
                        value: lower_expr(&kw.value)?,
                    })
                })
                .collect::<Result<_, LowerError>>()?;

            Ok(Expr::Call {
                func: Box::new(func),
                args,
                keywords,
                span,
            })
        }

        PyExpr::IfExp(ifexp) => Ok(Expr::IfExp {
            test: Box::new(lower_expr(&ifexp.test)?),
            body: Box::new(lower_expr(&ifexp.body)?),
            orelse: Box::new(lower_expr(&ifexp.orelse)?),
            span,
        }),

        PyExpr::List(list) => {
            let elts = list.elts.iter().map(lower_expr).collect::<Result<_, _>>()?;
            Ok(Expr::List { elts, span })
        }

        PyExpr::Tuple(tuple) => {
            let elts = tuple
                .elts
                .iter()
                .map(lower_expr)
                .collect::<Result<_, _>>()?;
            Ok(Expr::Tuple { elts, span })
        }

        PyExpr::Dict(dict) => {
            let keys = dict
                .keys
                .iter()
                .map(|k| k.as_ref().and_then(|k| lower_expr(k).ok()))
                .collect();
            let values = dict
                .values
                .iter()
                .map(lower_expr)
                .collect::<Result<_, _>>()?;
            Ok(Expr::Dict { keys, values, span })
        }

        PyExpr::Set(set) => {
            let elts = set.elts.iter().map(lower_expr).collect::<Result<_, _>>()?;
            Ok(Expr::Set { elts, span })
        }

        PyExpr::ListComp(comp) => Ok(Expr::ListComp {
            elt: Box::new(lower_expr(&comp.elt)?),
            generators: comp
                .generators
                .iter()
                .map(lower_comprehension)
                .collect::<Result<_, _>>()?,
            span,
        }),

        PyExpr::DictComp(comp) => Ok(Expr::DictComp {
            key: Box::new(lower_expr(&comp.key)?),
            value: Box::new(lower_expr(&comp.value)?),
            generators: comp
                .generators
                .iter()
                .map(lower_comprehension)
                .collect::<Result<_, _>>()?,
            span,
        }),

        PyExpr::SetComp(comp) => Ok(Expr::SetComp {
            elt: Box::new(lower_expr(&comp.elt)?),
            generators: comp
                .generators
                .iter()
                .map(lower_comprehension)
                .collect::<Result<_, _>>()?,
            span,
        }),

        PyExpr::GeneratorExp(gen_expr) => Ok(Expr::GeneratorExp {
            elt: Box::new(lower_expr(&gen_expr.elt)?),
            generators: gen_expr
                .generators
                .iter()
                .map(lower_comprehension)
                .collect::<Result<_, _>>()?,
            span,
        }),

        PyExpr::Lambda(lambda) => {
            let params = lambda
                .args
                .args
                .iter()
                .map(|a| a.def.arg.to_string())
                .collect();
            Ok(Expr::Lambda {
                params,
                body: Box::new(lower_expr(&lambda.body)?),
                span,
            })
        }

        _ => Err(LowerError::Unsupported(
            format!("expression: {:?}", std::mem::discriminant(expr)),
            span,
        )),
    }
}

/// Lower a comprehension clause.
fn lower_comprehension(comp: &ast::Comprehension) -> Result<Comprehension, LowerError> {
    Ok(Comprehension {
        target: lower_assign_target(&comp.target)?,
        iter: lower_expr(&comp.iter)?,
        ifs: comp.ifs.iter().filter_map(|e| lower_expr(e).ok()).collect(),
        is_async: comp.is_async,
    })
}

/// Lower a binary operator.
fn lower_operator(op: ast::Operator) -> BinOp {
    match op {
        ast::Operator::Add => BinOp::Add,
        ast::Operator::Sub => BinOp::Sub,
        ast::Operator::Mult => BinOp::Mult,
        ast::Operator::Div => BinOp::Div,
        ast::Operator::FloorDiv => BinOp::FloorDiv,
        ast::Operator::Mod => BinOp::Mod,
        ast::Operator::Pow => BinOp::Pow,
        ast::Operator::LShift => BinOp::LShift,
        ast::Operator::RShift => BinOp::RShift,
        ast::Operator::BitOr => BinOp::BitOr,
        ast::Operator::BitXor => BinOp::BitXor,
        ast::Operator::BitAnd => BinOp::BitAnd,
        ast::Operator::MatMult => BinOp::MatMult,
    }
}

/// Lower a unary operator.
fn lower_unary_op(op: ast::UnaryOp) -> UnaryOp {
    match op {
        ast::UnaryOp::Invert => UnaryOp::Invert,
        ast::UnaryOp::Not => UnaryOp::Not,
        ast::UnaryOp::UAdd => UnaryOp::UAdd,
        ast::UnaryOp::USub => UnaryOp::USub,
    }
}

/// Lower a comparison operator.
fn lower_cmp_op(op: ast::CmpOp) -> CmpOp {
    match op {
        ast::CmpOp::Eq => CmpOp::Eq,
        ast::CmpOp::NotEq => CmpOp::NotEq,
        ast::CmpOp::Lt => CmpOp::Lt,
        ast::CmpOp::LtE => CmpOp::LtE,
        ast::CmpOp::Gt => CmpOp::Gt,
        ast::CmpOp::GtE => CmpOp::GtE,
        ast::CmpOp::Is => CmpOp::Is,
        ast::CmpOp::IsNot => CmpOp::IsNot,
        ast::CmpOp::In => CmpOp::In,
        ast::CmpOp::NotIn => CmpOp::NotIn,
    }
}

/// Create a Span from a TextRange.
fn make_span(range: rustpython_parser::text_size::TextRange) -> Span {
    Span {
        start: range.start().into(),
        end: range.end().into(),
    }
}

/// Get the span of a statement.
fn stmt_span(stmt: &PyStmt) -> Span {
    match stmt {
        PyStmt::FunctionDef(s) => make_span(s.range),
        PyStmt::AsyncFunctionDef(s) => make_span(s.range),
        PyStmt::ClassDef(s) => make_span(s.range),
        PyStmt::Return(s) => make_span(s.range),
        PyStmt::Delete(s) => make_span(s.range),
        PyStmt::Assign(s) => make_span(s.range),
        PyStmt::TypeAlias(s) => make_span(s.range),
        PyStmt::AugAssign(s) => make_span(s.range),
        PyStmt::AnnAssign(s) => make_span(s.range),
        PyStmt::For(s) => make_span(s.range),
        PyStmt::AsyncFor(s) => make_span(s.range),
        PyStmt::While(s) => make_span(s.range),
        PyStmt::If(s) => make_span(s.range),
        PyStmt::With(s) => make_span(s.range),
        PyStmt::AsyncWith(s) => make_span(s.range),
        PyStmt::Match(s) => make_span(s.range),
        PyStmt::Raise(s) => make_span(s.range),
        PyStmt::Try(s) => make_span(s.range),
        PyStmt::TryStar(s) => make_span(s.range),
        PyStmt::Assert(s) => make_span(s.range),
        PyStmt::Import(s) => make_span(s.range),
        PyStmt::ImportFrom(s) => make_span(s.range),
        PyStmt::Global(s) => make_span(s.range),
        PyStmt::Nonlocal(s) => make_span(s.range),
        PyStmt::Expr(s) => make_span(s.range),
        PyStmt::Pass(s) => make_span(s.range),
        PyStmt::Break(s) => make_span(s.range),
        PyStmt::Continue(s) => make_span(s.range),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_simple_function() {
        let source = r#"
def foo(x: int) -> int:
    return x + 1
"#;
        let module = lower_source(source, "<test>").unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "foo");
        assert_eq!(module.functions[0].params.len(), 1);
        assert_eq!(module.functions[0].params[0].name, "x");
    }

    #[test]
    fn test_lower_gate_call() {
        let source = r#"
def bell():
    q = qubit[2]
    h(q[0])
    cx(q[0], q[1])
"#;
        let module = lower_source(source, "<test>").unwrap();
        assert_eq!(module.functions.len(), 1);

        let body = &module.functions[0].body;
        assert!(matches!(body[0], Stmt::Qalloc { .. }));
        assert!(matches!(
            body[1],
            Stmt::Gate {
                gate: GateKind::H,
                ..
            }
        ));
        assert!(matches!(
            body[2],
            Stmt::Gate {
                gate: GateKind::Cx,
                ..
            }
        ));
    }

    #[test]
    fn test_lower_for_range() {
        let source = r#"
def loop():
    for i in range(10):
        pass
"#;
        let module = lower_source(source, "<test>").unwrap();
        let body = &module.functions[0].body;

        if let Stmt::For { iter, .. } = &body[0] {
            assert!(matches!(iter, ForIter::Range { start: None, .. }));
        } else {
            panic!("expected for loop");
        }
    }
}
