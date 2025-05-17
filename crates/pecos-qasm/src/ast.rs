use pecos_core::errors::PecosError;
use std::collections::HashMap;
use std::fmt;

// Helper functions for formatting QASM output
fn format_list<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    items: &[T],
    separator: &str,
    prefix: &str,
    suffix: &str,
) -> fmt::Result {
    if !items.is_empty() {
        write!(f, "{prefix}")?;
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                write!(f, "{separator}")?;
            }
            write!(f, "{item}")?;
        }
        write!(f, "{suffix}")?;
    }
    Ok(())
}

fn format_params<T: fmt::Display>(f: &mut fmt::Formatter<'_>, params: &[T]) -> fmt::Result {
    format_list(f, params, ", ", "(", ")")
}

fn format_qubits(
    f: &mut fmt::Formatter<'_>,
    qubits: &[String],
    first_separator: &str,
) -> fmt::Result {
    for (i, qubit) in qubits.iter().enumerate() {
        if i == 0 {
            write!(f, "{first_separator}{qubit}")?;
        } else {
            write!(f, ", {qubit}")?;
        }
    }
    Ok(())
}

/// Represents a gate definition
#[derive(Debug, Clone)]
pub struct GateDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub qargs: Vec<String>,
    pub body: Vec<GateOperation>,
}

/// Represents an opaque gate declaration
#[derive(Debug, Clone)]
pub struct OpaqueGateDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub qargs: Vec<String>,
}

/// Represents an operation within a gate definition
#[derive(Debug, Clone)]
pub struct GateOperation {
    pub name: String,
    pub params: Vec<Expression>,
    pub qargs: Vec<String>,
}

impl fmt::Display for GateOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        format_params(f, &self.params)?;
        format_qubits(f, &self.qargs, " ")?;
        Ok(())
    }
}

/// Represents different types of operations in a QASM program
#[derive(Debug, Clone)]
pub enum Operation {
    Gate {
        name: String,
        parameters: Vec<f64>,
        qubits: Vec<usize>,
    },
    Measure {
        qubit: usize,
        c_reg: String,
        c_index: usize,
    },
    RegMeasure {
        q_reg: String,
        c_reg: String,
    },
    If {
        condition: Expression,
        operation: Box<Operation>,
    },
    Reset {
        qubit: usize,
    },
    Barrier {
        qubits: Vec<usize>,
    },
    ClassicalAssignment {
        target: String,
        is_indexed: bool,
        index: Option<usize>,
        expression: Expression,
    },
    OpaqueGate {
        name: String,
        params: Vec<String>,
        qargs: Vec<String>,
    },
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Gate {
                name,
                parameters,
                qubits,
            } => {
                write!(f, "{name}")?;
                format_params(f, parameters)?;

                for (i, qubit) in qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " gid[{qubit}]")?;
                    } else {
                        write!(f, ", gid[{qubit}]")?;
                    }
                }
                Ok(())
            }
            Operation::Measure {
                qubit,
                c_reg,
                c_index,
            } => {
                write!(f, "measure gid[{qubit}] -> {c_reg}[{c_index}]")
            }
            Operation::If {
                condition,
                operation,
            } => {
                write!(f, "if ({condition}) {operation}")
            }
            Operation::Reset { qubit } => {
                write!(f, "reset gid[{qubit}]")
            }
            Operation::Barrier { qubits } => {
                write!(f, "barrier")?;
                for (i, qubit) in qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " gid[{qubit}]")?;
                    } else {
                        write!(f, ", gid[{qubit}]")?;
                    }
                }
                Ok(())
            }
            Operation::RegMeasure { q_reg, c_reg } => {
                write!(f, "measure {q_reg} -> {c_reg}")
            }
            Operation::ClassicalAssignment {
                target,
                is_indexed,
                index,
                expression,
            } => {
                if *is_indexed {
                    if let Some(idx) = index {
                        write!(f, "{target}[{idx}] = {expression}")
                    } else {
                        write!(f, "{target} = {expression}")
                    }
                } else {
                    write!(f, "{target} = {expression}")
                }
            }
            Operation::OpaqueGate {
                name,
                params,
                qargs,
            } => {
                write!(f, "opaque {name}")?;
                if !params.is_empty() {
                    write!(f, "(")?;
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{param}")?;
                    }
                    write!(f, ")")?;
                }
                write!(f, " ")?;
                for (i, qarg) in qargs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{qarg}")?;
                }
                Ok(())
            }
        }
    }
}

/// Display wrapper for Operation that includes qubit mapping context
pub struct OperationDisplay<'a> {
    pub operation: &'a Operation,
    pub qubit_map: &'a HashMap<usize, (String, usize)>,
}

impl fmt::Display for OperationDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.operation {
            Operation::Gate {
                name,
                parameters,
                qubits,
            } => {
                write!(f, "{name}")?;
                format_params(f, parameters)?;

                for (i, &qubit_id) in qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " ")?;
                    } else {
                        write!(f, ", ")?;
                    }

                    let (reg_name, index) = self
                        .qubit_map
                        .get(&qubit_id)
                        .expect("Global qubit ID must exist in qubit_map");
                    write!(f, "{reg_name}[{index}]")?;
                }
                Ok(())
            }
            Operation::Measure {
                qubit,
                c_reg,
                c_index,
            } => {
                let (q_reg, q_index) = self
                    .qubit_map
                    .get(qubit)
                    .expect("Global qubit ID must exist in qubit_map");
                write!(f, "measure {q_reg}[{q_index}] -> {c_reg}[{c_index}]")
            }
            Operation::Reset { qubit } => {
                let (q_reg, q_index) = self
                    .qubit_map
                    .get(qubit)
                    .expect("Global qubit ID must exist in qubit_map");
                write!(f, "reset {q_reg}[{q_index}]")
            }
            Operation::Barrier { qubits } => {
                write!(f, "barrier")?;
                for (i, &qubit_id) in qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " ")?;
                    } else {
                        write!(f, ", ")?;
                    }
                    let (reg_name, index) = self
                        .qubit_map
                        .get(&qubit_id)
                        .expect("Global qubit ID must exist in qubit_map");
                    write!(f, "{reg_name}[{index}]")?;
                }
                Ok(())
            }
            _ => self.operation.fmt(f),
        }
    }
}

/// Represents expressions in classical operations
#[derive(Debug, Clone)]
pub enum Expression {
    Integer(i64),
    Float(f64),
    Pi,
    Variable(String),
    BitId(String, i64),
    BinaryOp {
        op: String,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    UnaryOp {
        op: String,
        expr: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Integer(val) => write!(f, "{val}"),
            Expression::Float(val) => write!(f, "{val}"),
            Expression::Pi => write!(f, "pi"),
            Expression::Variable(name) => write!(f, "{name}"),
            Expression::BitId(reg_name, idx) => write!(f, "{reg_name}[{idx}]"),
            Expression::BinaryOp { op, left, right } => write!(f, "({left} {op} {right})"),
            Expression::UnaryOp { op, expr } => write!(f, "{op}({expr})"),
            Expression::FunctionCall { name, args } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Simplified evaluation context - merged trait and implementation
pub struct EvaluationCtx<'a> {
    pub params: Option<&'a HashMap<String, f64>>,
}

impl Expression {
    /// Evaluate expression with an optional parameter context
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(&self, context: Option<&EvaluationCtx>) -> Result<f64, PecosError> {
        match self {
            Expression::Integer(i) =>
            {
                #[allow(clippy::cast_precision_loss)]
                Ok(*i as f64)
            }
            Expression::Float(f) => Ok(*f),
            Expression::Pi => Ok(std::f64::consts::PI),
            Expression::Variable(name) => {
                if let Some(ctx) = context {
                    if let Some(params) = ctx.params {
                        params
                            .get(name)
                            .copied()
                            .ok_or_else(|| PecosError::ParseInvalidIdentifier(name.clone()))
                    } else {
                        Err(PecosError::ParseInvalidExpression(format!(
                            "Cannot evaluate variable '{name}' without parameters"
                        )))
                    }
                } else {
                    Err(PecosError::ParseInvalidExpression(format!(
                        "Cannot evaluate variable '{name}' without context"
                    )))
                }
            }
            Expression::BinaryOp { op, left, right } => {
                let left_val = left.evaluate(context)?;
                let right_val = right.evaluate(context)?;
                match op.as_str() {
                    "+" => Ok(left_val + right_val),
                    "-" => Ok(left_val - right_val),
                    "*" => Ok(left_val * right_val),
                    "/" => Ok(left_val / right_val),
                    "**" => Ok(left_val.powf(right_val)),
                    "&" =>
                    {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        Ok((left_val as i64 & right_val as i64) as f64)
                    }
                    "|" =>
                    {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        Ok((left_val as i64 | right_val as i64) as f64)
                    }
                    "^" =>
                    {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        Ok((left_val as i64 ^ right_val as i64) as f64)
                    }
                    "==" =>
                    {
                        #[allow(clippy::cast_precision_loss)]
                        Ok(i64::from((left_val - right_val).abs() < f64::EPSILON) as f64)
                    }
                    "!=" =>
                    {
                        #[allow(clippy::cast_precision_loss)]
                        Ok(i64::from((left_val - right_val).abs() >= f64::EPSILON) as f64)
                    }
                    "<" =>
                    {
                        #[allow(clippy::cast_precision_loss)]
                        Ok(i64::from(left_val < right_val) as f64)
                    }
                    ">" =>
                    {
                        #[allow(clippy::cast_precision_loss)]
                        Ok(i64::from(left_val > right_val) as f64)
                    }
                    "<=" =>
                    {
                        #[allow(clippy::cast_precision_loss)]
                        Ok(i64::from(left_val <= right_val) as f64)
                    }
                    ">=" =>
                    {
                        #[allow(clippy::cast_precision_loss)]
                        Ok(i64::from(left_val >= right_val) as f64)
                    }
                    "<<" =>
                    {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        Ok(((left_val as i64) << (right_val as i64)) as f64)
                    }
                    ">>" =>
                    {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        Ok(((left_val as i64) >> (right_val as i64)) as f64)
                    }
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unsupported binary operation: {op}"
                    ))),
                }
            }
            Expression::UnaryOp { op, expr } => {
                let val = expr.evaluate(context)?;
                match op.as_str() {
                    "-" => Ok(-val),
                    "~" =>
                    {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        Ok((!(val as i64)) as f64)
                    }
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unsupported unary operation: {op}"
                    ))),
                }
            }
            Expression::BitId(reg_name, idx) => {
                // BitId requires special handling - for now just return an error
                Err(PecosError::ParseInvalidExpression(format!(
                    "Cannot evaluate BitId({reg_name}, {idx}) without register context"
                )))
            }
            Expression::FunctionCall { name, args } => {
                if args.len() != 1 {
                    return Err(PecosError::ParseInvalidExpression(format!(
                        "Function {} expects exactly 1 argument, got {}",
                        name,
                        args.len()
                    )));
                }

                let arg_val = args[0].evaluate(context)?;

                match name.as_str() {
                    "sin" => Ok(arg_val.sin()),
                    "cos" => Ok(arg_val.cos()),
                    "tan" => Ok(arg_val.tan()),
                    "exp" => Ok(arg_val.exp()),
                    "ln" => {
                        if arg_val <= 0.0 {
                            Err(PecosError::ParseInvalidExpression(format!(
                                "ln({arg_val}) is undefined for non-positive values"
                            )))
                        } else {
                            Ok(arg_val.ln())
                        }
                    }
                    "sqrt" => {
                        if arg_val < 0.0 {
                            Err(PecosError::ParseInvalidExpression(format!(
                                "sqrt({arg_val}) is undefined for negative values"
                            )))
                        } else {
                            Ok(arg_val.sqrt())
                        }
                    }
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unknown function: {name}"
                    ))),
                }
            }
        }
    }

    /// Compatibility method for existing code
    pub fn evaluate_with_context(
        &self,
        context: Option<&dyn crate::ast::EvaluationContext>,
    ) -> Result<f64, PecosError> {
        if let Some(ctx) = context {
            // Use the trait's evaluate_float method
            ctx.evaluate_float(self)
        } else {
            // Evaluate without context
            self.evaluate(None)
        }
    }
}

// For compatibility with existing code, we keep the trait
pub trait EvaluationContext {
    fn evaluate_float(&self, expr: &Expression) -> Result<f64, PecosError>;
    fn evaluate_int(&self, expr: &Expression) -> Result<i64, PecosError> {
        #[allow(clippy::cast_possible_truncation)]
        self.evaluate_float(expr).map(|f| f as i64)
    }
}

// Simple implementation for compatibility
pub struct EvaluationContextImpl<'a> {
    pub params: Option<&'a HashMap<String, f64>>,
}

impl EvaluationContext for EvaluationContextImpl<'_> {
    fn evaluate_float(&self, expr: &Expression) -> Result<f64, PecosError> {
        let ctx = EvaluationCtx {
            params: self.params,
        };
        expr.evaluate(Some(&ctx))
    }
}

pub type ParameterContext<'a> = EvaluationContextImpl<'a>;
