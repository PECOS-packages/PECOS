use ::bitvec::prelude::*;
use pecos_core::prelude::{Gate, GateType};
use pecos_core::{bitvec, errors::PecosError};
use std::collections::BTreeMap;
use std::fmt;

// Helper function for formatting parameters
fn format_params<T: fmt::Display>(f: &mut fmt::Formatter<'_>, params: &[T]) -> fmt::Result {
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
        for (i, qarg) in self.qargs.iter().enumerate() {
            write!(f, "{}{qarg}", if i == 0 { " " } else { ", " })?;
        }
        Ok(())
    }
}

/// Represents different types of operations in a QASM program
#[derive(Debug, Clone)]
pub enum Operation {
    /// Gate operation (before expansion - string-based)
    Gate {
        name: String,
        parameters: Vec<f64>,
        qubits: Vec<usize>,
    },

    /// Native gate operations (after expansion - typed)
    NativeGate(Gate),

    /// Measurement with classical register mapping
    MeasureWithMapping {
        gate: Gate, // Gate with GateType::Measure
        c_reg: String,
        c_index: usize,
    },

    /// Register-level measurement (needs expansion)
    RegMeasure { q_reg: String, c_reg: String },

    /// Barrier operation
    Barrier { qubits: Vec<usize> },

    /// Classical assignment operation
    ClassicalAssignment {
        target: String,
        is_indexed: bool,
        index: Option<usize>,
        expression: Expression,
    },

    /// Void function call (standalone function call with no assignment)
    VoidFunctionCall { expression: Expression },

    /// Conditional operation
    If {
        condition: Expression,
        operation: Box<Operation>,
    },

    /// Opaque gate declaration (not yet implemented)
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
                    write!(f, "{} gid[{qubit}]", if i == 0 { " " } else { ", " })?;
                }
                Ok(())
            }
            Operation::NativeGate(gate) => {
                write!(f, "{}", gate.gate_type)?;
                format_params(f, &gate.params)?;
                for (i, qubit) in gate.qubits.iter().enumerate() {
                    write!(f, "{} gid[{}]", if i == 0 { " " } else { ", " }, qubit.0)?;
                }
                Ok(())
            }
            Operation::MeasureWithMapping {
                gate,
                c_reg,
                c_index,
            } => {
                if let Some(qubit) = gate.qubits.first() {
                    write!(f, "measure gid[{}] -> {c_reg}[{c_index}]", qubit.0)
                } else {
                    write!(f, "measure <invalid> -> {c_reg}[{c_index}]")
                }
            }
            Operation::If {
                condition,
                operation,
            } => write!(f, "if ({condition}) {operation}"),
            Operation::Barrier { qubits } => {
                write!(f, "barrier")?;
                for (i, qubit) in qubits.iter().enumerate() {
                    write!(f, "{} gid[{qubit}]", if i == 0 { " " } else { ", " })?;
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
            } => match (*is_indexed, index) {
                (true, Some(idx)) => write!(f, "{target}[{idx}] = {expression}"),
                _ => write!(f, "{target} = {expression}"),
            },
            Operation::OpaqueGate {
                name,
                params,
                qargs,
            } => {
                write!(f, "opaque {name}")?;
                format_params(f, params)?;
                for (i, qarg) in qargs.iter().enumerate() {
                    write!(f, "{} {qarg}", if i == 0 { " " } else { ", " })?;
                }
                Ok(())
            }
            Operation::VoidFunctionCall { expression } => {
                write!(f, "{expression}")
            }
        }
    }
}

/// Display wrapper for Operation that includes qubit mapping context
pub struct OperationDisplay<'a> {
    pub operation: &'a Operation,
    pub qubit_map: &'a BTreeMap<usize, (String, usize)>,
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
                        .unwrap_or_else(|| panic!("BUG: Qubit ID {qubit_id} not found in qubit_map. This indicates a bug in the QASM parser."));
                    write!(f, "{reg_name}[{index}]")?;
                }
                Ok(())
            }
            Operation::NativeGate(gate) => {
                // Display gate type in QASM format
                let gate_name = if gate.gate_type == GateType::Prep {
                    "reset".to_string() // PECOS Prep -> QASM reset
                } else {
                    // Use lowercase for QASM display
                    let name = format!("{}", gate.gate_type);
                    name.to_lowercase()
                };
                write!(f, "{gate_name}")?;
                format_params(f, &gate.params)?;

                for (i, qubit) in gate.qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " ")?;
                    } else {
                        write!(f, ", ")?;
                    }

                    let qubit_id = qubit.0;
                    let (reg_name, index) = self
                        .qubit_map
                        .get(&qubit_id)
                        .unwrap_or_else(|| panic!("BUG: Qubit ID {qubit_id} not found in qubit_map. This indicates a bug in the QASM parser."));
                    write!(f, "{reg_name}[{index}]")?;
                }
                Ok(())
            }
            Operation::MeasureWithMapping {
                gate,
                c_reg,
                c_index,
            } => {
                if let Some(qubit) = gate.qubits.first() {
                    let qubit_id = qubit.0;
                    let (q_reg, q_index) = self
                        .qubit_map
                        .get(&qubit_id)
                        .unwrap_or_else(|| panic!("BUG: Qubit ID {qubit_id} not found in qubit_map. This indicates a bug in the QASM parser."));
                    write!(f, "measure {q_reg}[{q_index}] -> {c_reg}[{c_index}]")
                } else {
                    write!(f, "measure <invalid> -> {c_reg}[{c_index}]")
                }
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
                        .unwrap_or_else(|| panic!("BUG: Qubit ID {qubit_id} not found in qubit_map. This indicates a bug in the QASM parser."));
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
    Integer(BitVec<u8, Lsb0>),
    Float(f64),
    Pi,
    Variable(String),
    BitId(String, usize),
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
            Expression::Integer(bitvec) => {
                write!(f, "{}", bitvec::to_decimal_string(bitvec))
            }
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
    pub params: Option<&'a BTreeMap<String, f64>>,
}

impl Expression {
    /// Evaluate expression as a floating-point value for gate parameters
    ///
    /// This method is used to evaluate expressions that appear as gate parameters,
    /// such as `rx(pi/2, q[0])`. It supports:
    /// - Basic arithmetic: +, -, *, /, ** (power)
    /// - Mathematical functions: sin, cos, tan, exp, ln, sqrt
    /// - Constants: pi
    /// - Variables (from parameter context)
    ///
    /// It does NOT support:
    /// - Bitwise operations (&, |, ^, ~)
    /// - Comparisons (==, !=, <, >, <=, >=)
    /// - Shift operations (<<, >>)
    /// - Bit references (reg[idx])
    ///
    /// # Errors
    ///
    /// Returns an error if the expression cannot be evaluated (e.g., undefined variables,
    /// unsupported operations, or operations that don't make sense for floating-point values).
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(&self, context: Option<&EvaluationCtx>) -> Result<f64, PecosError> {
        match self {
            Expression::Integer(bitvec) => {
                // Convert BitVec to f64 (limited to 53 bits of precision)
                let mut value = 0.0;
                for (i, bit) in bitvec.iter().enumerate() {
                    if i < 53 && *bit {
                        // Use f64::from for smaller values to avoid precision loss warning
                        if i < 32 {
                            value += f64::from(1u32 << i);
                        } else {
                            // For larger values, we accept the precision limitation of f64
                            // Use i32::try_from to handle potential truncation on 64-bit systems
                            if let Ok(i_i32) = i32::try_from(i) {
                                value += 2.0_f64.powi(i_i32);
                            } else {
                                // If i is too large for i32, the bit position is beyond f64's precision anyway
                                break;
                            }
                        }
                    }
                }
                Ok(value)
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
                    "&" | "|" | "^" | "<<" | ">>" => {
                        Err(PecosError::ParseInvalidExpression(format!(
                            "Bitwise operation '{op}' is not supported in gate parameter expressions. Gate parameters must be floating-point values."
                        )))
                    }
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Operation '{op}' is not supported in gate parameter expressions. Only +, -, *, /, ** are allowed."
                    ))),
                }
            }
            Expression::UnaryOp { op, expr } => {
                let val = expr.evaluate(context)?;
                match op.as_str() {
                    "-" => Ok(-val),
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Operation '{op}' is not supported in gate parameter expressions. Only unary minus (-) is allowed."
                    ))),
                }
            }
            Expression::BitId(reg_name, idx) => Err(PecosError::ParseInvalidExpression(format!(
                "Bit reference {reg_name}[{idx}] is not allowed in gate parameter expressions."
            ))),
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
                                "ln({arg_val}) is undefined"
                            )))
                        } else {
                            Ok(arg_val.ln())
                        }
                    }
                    "sqrt" => {
                        if arg_val < 0.0 {
                            Err(PecosError::ParseInvalidExpression(format!(
                                "sqrt({arg_val}) is undefined"
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
}
