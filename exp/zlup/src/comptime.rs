//! Compile-time evaluation for Zlup.
//!
//! This module provides compile-time evaluation of expressions, enabling:
//! - `comptime` blocks that execute at compile time
//! - `comptime` function parameters (generics)
//! - Compile-time type manipulation
//! - Constant folding and propagation
//!
//! ## Comptime Values
//!
//! Values that can exist at compile time:
//! - Integers (i64 for signed, u64 for unsigned)
//! - Floats (f64)
//! - Booleans
//! - Types (the `type` type)
//! - Arrays of comptime values
//! - Structs with comptime fields
//!
//! ## Example
//!
//! ```zlup
//! // Comptime block
//! N := comptime {
//!     mut sum: u32 = 0;
//!     for i in 0..10 {
//!         sum += i;
//!     }
//!     sum
//! };
//!
//! // Comptime function parameter (generic)
//! fn makeArray(comptime T: type, comptime N: usize) -> [N]T {
//!     mut arr: [N]T = undefined;
//!     return arr;
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt;

use crate::ast::{
    BinaryOp, Expr, FStringPart, FnDecl, ForRange, PrimitiveType, Stmt, TypeExpr, UnaryOp,
};
use crate::rational::Rational;
use crate::semantic::{BitWidth, SemanticError, Type};

// =============================================================================
// Type Information (for @typeInfo, @fieldNames, @enumFields builtins)
// =============================================================================

/// The kind of a type, returned by @typeInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeInfoKind {
    /// Primitive types (bool, integers, floats)
    Primitive,
    /// Array type with fixed size
    Array,
    /// Slice type (dynamic size)
    Slice,
    /// Pointer type
    Pointer,
    /// Optional type (?T)
    Optional,
    /// Error union type (E!T)
    ErrorUnion,
    /// Struct type
    Struct,
    /// Enum type
    Enum,
    /// Union type
    Union,
    /// Error set type
    ErrorSet,
    /// Fault set type
    FaultSet,
    /// Function type
    Function,
    /// Tuple type
    Tuple,
    /// The type type (metatype)
    Type,
    /// Unit type
    Unit,
    /// Never type (bottom)
    Never,
    /// Quantum types (Qubit, Bit, Allocator)
    Quantum,
    /// Unknown/unresolved type
    Unknown,
}

impl TypeInfoKind {
    /// Convert to a string representation for use in comptime values.
    pub fn as_str(&self) -> &'static str {
        match self {
            TypeInfoKind::Primitive => "primitive",
            TypeInfoKind::Array => "array",
            TypeInfoKind::Slice => "slice",
            TypeInfoKind::Pointer => "pointer",
            TypeInfoKind::Optional => "optional",
            TypeInfoKind::ErrorUnion => "error_union",
            TypeInfoKind::Struct => "struct",
            TypeInfoKind::Enum => "enum",
            TypeInfoKind::Union => "union",
            TypeInfoKind::ErrorSet => "error_set",
            TypeInfoKind::FaultSet => "fault_set",
            TypeInfoKind::Function => "function",
            TypeInfoKind::Tuple => "tuple",
            TypeInfoKind::Type => "type",
            TypeInfoKind::Unit => "unit",
            TypeInfoKind::Never => "never",
            TypeInfoKind::Quantum => "quantum",
            TypeInfoKind::Unknown => "unknown",
        }
    }
}

/// A value known at compile time.
#[derive(Debug, Clone)]
pub enum ComptimeValue {
    /// Signed integer value
    Int(i64),
    /// Unsigned integer value
    Uint(u64),
    /// Floating point value
    Float(f64),
    /// Rational number (exact fraction like 1/4)
    Rational(Rational),
    /// Boolean value
    Bool(bool),
    /// A type value (for `type` type)
    Type(Type),
    /// Null value (for optionals)
    Null,
    /// Undefined value
    Undefined,
    /// Unit value (the single value of the unit type)
    Unit,
    /// Array of comptime values
    Array(Vec<ComptimeValue>),
    /// Struct with named fields
    Struct {
        name: String,
        fields: BTreeMap<String, ComptimeValue>,
    },
    /// Slice reference (ptr + len)
    Slice { data: Vec<ComptimeValue> },
    /// String literal
    String(String),
    /// A comptime function (for generic type constructors)
    Function(Box<FnDecl>),
}

impl PartialEq for ComptimeValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => a == b,
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => a == b,
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => a == b,
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => a == b,
            (ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => a == b,
            (ComptimeValue::Type(a), ComptimeValue::Type(b)) => a == b,
            (ComptimeValue::Null, ComptimeValue::Null) => true,
            (ComptimeValue::Undefined, ComptimeValue::Undefined) => true,
            (ComptimeValue::Unit, ComptimeValue::Unit) => true,
            (ComptimeValue::Array(a), ComptimeValue::Array(b)) => a == b,
            (
                ComptimeValue::Struct {
                    name: n1,
                    fields: f1,
                },
                ComptimeValue::Struct {
                    name: n2,
                    fields: f2,
                },
            ) => n1 == n2 && f1 == f2,
            (ComptimeValue::Slice { data: d1 }, ComptimeValue::Slice { data: d2 }) => d1 == d2,
            (ComptimeValue::String(a), ComptimeValue::String(b)) => a == b,
            (ComptimeValue::Function(_), ComptimeValue::Function(_)) => false, // Functions not comparable
            _ => false,
        }
    }
}

impl fmt::Display for ComptimeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComptimeValue::Int(v) => write!(f, "{}", v),
            ComptimeValue::Uint(v) => write!(f, "{}", v),
            ComptimeValue::Float(v) => write!(f, "{}", v),
            ComptimeValue::Rational(v) => write!(f, "{}", v),
            ComptimeValue::Bool(v) => write!(f, "{}", v),
            ComptimeValue::Type(t) => write!(f, "{}", t.display_name()),
            ComptimeValue::Null => write!(f, "null"),
            ComptimeValue::Undefined => write!(f, "undefined"),
            ComptimeValue::Unit => write!(f, "unit"),
            ComptimeValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ComptimeValue::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, ".{} = {}", k, v)?;
                }
                write!(f, " }}")
            }
            ComptimeValue::Slice { data } => {
                write!(f, "&[")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ComptimeValue::String(s) => write!(f, "\"{}\"", s),
            ComptimeValue::Function(func) => write!(f, "<fn {}>", func.name),
        }
    }
}

impl ComptimeValue {
    /// Get the type of this comptime value.
    pub fn get_type(&self) -> Type {
        match self {
            ComptimeValue::Int(_) => Type::IInt {
                bits: BitWidth::BITS_64,
            },
            ComptimeValue::Uint(_) => Type::UInt {
                bits: BitWidth::BITS_64,
            },
            ComptimeValue::Float(_) => Type::F64,
            ComptimeValue::Rational(_) => Type::F64, // Rationals coerce to f64 when needed
            ComptimeValue::Bool(_) => Type::Bool,
            ComptimeValue::Type(_) => Type::Type,
            ComptimeValue::Null => Type::Optional {
                inner: Box::new(Type::Unknown),
            },
            ComptimeValue::Undefined => Type::Unknown,
            ComptimeValue::Unit => Type::Unit,
            ComptimeValue::Array(arr) => {
                let elem_ty = arr.first().map(|v| v.get_type()).unwrap_or(Type::Unknown);
                Type::Array {
                    element: Box::new(elem_ty),
                    size: Some(arr.len() as u64),
                }
            }
            ComptimeValue::Struct { name, fields } => {
                let field_types: Vec<(String, Type)> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), v.get_type()))
                    .collect();
                Type::Struct {
                    name: name.clone(),
                    fields: field_types,
                }
            }
            ComptimeValue::Slice { data } => {
                let elem_ty = data.first().map(|v| v.get_type()).unwrap_or(Type::Unknown);
                Type::Slice {
                    element: Box::new(elem_ty),
                }
            }
            ComptimeValue::String(_) => Type::Slice {
                element: Box::new(Type::UInt {
                    bits: BitWidth::BITS_8,
                }),
            },
            ComptimeValue::Function(_) => Type::Type, // Comptime functions return types
        }
    }

    /// Try to convert to i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ComptimeValue::Int(v) => Some(*v),
            ComptimeValue::Uint(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to convert to u64.
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            ComptimeValue::Int(v) => Some(*v as u64),
            ComptimeValue::Uint(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to convert to f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ComptimeValue::Float(v) => Some(*v),
            ComptimeValue::Int(v) => Some(*v as f64),
            ComptimeValue::Uint(v) => Some(*v as f64),
            ComptimeValue::Rational(r) => Some(r.to_f64()),
            _ => None,
        }
    }

    /// Try to get as Rational.
    pub fn as_rational(&self) -> Option<Rational> {
        match self {
            ComptimeValue::Rational(r) => Some(*r),
            ComptimeValue::Int(v) => Some(Rational::from_int(*v)),
            ComptimeValue::Uint(v) => Some(Rational::from_int(*v as i64)),
            ComptimeValue::Float(v) => Rational::from_f64_common(*v),
            _ => None,
        }
    }

    /// Try to convert to bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ComptimeValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to convert to usize.
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            ComptimeValue::Int(v) => (*v).try_into().ok(),
            ComptimeValue::Uint(v) => (*v).try_into().ok(),
            _ => None,
        }
    }

    /// Try to get as type.
    pub fn as_type(&self) -> Option<&Type> {
        match self {
            ComptimeValue::Type(t) => Some(t),
            _ => None,
        }
    }

    /// Check if this is a truthy value.
    pub fn is_truthy(&self) -> bool {
        match self {
            ComptimeValue::Bool(v) => *v,
            ComptimeValue::Int(v) => *v != 0,
            ComptimeValue::Uint(v) => *v != 0,
            ComptimeValue::Rational(r) => !r.is_zero(),
            ComptimeValue::Null => false,
            ComptimeValue::Undefined => false,
            _ => true,
        }
    }
}

/// Result of comptime evaluation.
pub type ComptimeResult<T> = Result<T, ComptimeError>;

/// Errors during comptime evaluation.
#[derive(Debug, Clone)]
pub struct ComptimeError {
    pub message: String,
}

impl fmt::Display for ComptimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ComptimeError {}

impl From<ComptimeError> for SemanticError {
    fn from(err: ComptimeError) -> Self {
        SemanticError::ComptimeError {
            message: err.message,
            location: crate::ast::SourceLocation::default(),
        }
    }
}

/// Comptime evaluation context.
///
/// Tracks variables and their values during comptime evaluation.
#[derive(Debug, Clone, Default)]
pub struct ComptimeContext {
    /// Variable bindings in the current scope.
    scopes: Vec<BTreeMap<String, ComptimeValue>>,
}

impl ComptimeContext {
    /// Create a new comptime context.
    pub fn new() -> Self {
        Self {
            scopes: vec![BTreeMap::new()],
        }
    }

    /// Push a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a variable in the current scope.
    pub fn define(&mut self, name: &str, value: ComptimeValue) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    /// Look up a variable.
    pub fn lookup(&self, name: &str) -> Option<&ComptimeValue> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        None
    }

    /// Update a variable's value.
    pub fn update(&mut self, name: &str, value: ComptimeValue) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return true;
            }
        }
        false
    }
}

/// Comptime evaluator.
///
/// Maximum recursion depth for comptime evaluation to prevent stack overflow.
const MAX_COMPTIME_DEPTH: usize = 256;

/// Evaluates expressions at compile time, producing `ComptimeValue` results.
#[derive(Debug)]
pub struct ComptimeEvaluator {
    /// Evaluation context.
    pub context: ComptimeContext,
    /// Current recursion depth for detecting infinite recursion.
    depth: usize,
    /// Memoization cache for comptime function calls.
    /// Key: (function_name, serialized_args), Value: cached result
    memo_cache: BTreeMap<(String, String), ComptimeValue>,
}

impl Default for ComptimeEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ComptimeEvaluator {
    /// Create a new comptime evaluator.
    pub fn new() -> Self {
        Self {
            context: ComptimeContext::new(),
            depth: 0,
            memo_cache: BTreeMap::new(),
        }
    }

    /// Serialize comptime values to a string key for memoization cache.
    fn serialize_args_for_cache(args: &[ComptimeValue]) -> String {
        args.iter()
            .map(|v| match v {
                ComptimeValue::Int(n) => format!("i{}", n),
                ComptimeValue::Uint(n) => format!("u{}", n),
                ComptimeValue::Float(f) => format!("f{}", f),
                ComptimeValue::Rational(r) => format!("r{}/{}", r.numerator(), r.denominator()),
                ComptimeValue::Bool(b) => format!("b{}", b),
                ComptimeValue::String(s) => format!("s{}", s),
                ComptimeValue::Type(t) => format!("t{}", Self::serialize_type_for_cache(t)),
                ComptimeValue::Array(arr) => {
                    format!("a[{}]", Self::serialize_args_for_cache(arr))
                }
                ComptimeValue::Slice { data } => {
                    format!("sl[{}]", Self::serialize_args_for_cache(data))
                }
                ComptimeValue::Struct { name, fields } => {
                    let field_strs: Vec<_> = fields
                        .iter()
                        .map(|(k, v)| {
                            format!(
                                "{}:{}",
                                k,
                                Self::serialize_args_for_cache(std::slice::from_ref(v))
                            )
                        })
                        .collect();
                    format!("st{}[{}]", name, field_strs.join(";"))
                }
                ComptimeValue::Function(f) => format!("fn{}", f.name),
                ComptimeValue::Null => "null".to_string(),
                ComptimeValue::Undefined => "undef".to_string(),
                ComptimeValue::Unit => "unit".to_string(),
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Serialize a Type to a unique string for cache keys.
    /// Unlike display_name(), this includes full structural information for anonymous types.
    fn serialize_type_for_cache(ty: &Type) -> String {
        match ty {
            Type::Struct { name, fields } => {
                // For structs (especially anonymous ones), include full field information
                let field_strs: Vec<_> = fields
                    .iter()
                    .map(|(field_name, field_ty)| {
                        format!(
                            "{}:{}",
                            field_name,
                            Self::serialize_type_for_cache(field_ty)
                        )
                    })
                    .collect();
                format!("struct{}[{}]", name, field_strs.join(";"))
            }
            Type::Array { element, size } => {
                format!(
                    "[{}]{}",
                    size.unwrap_or(0),
                    Self::serialize_type_for_cache(element)
                )
            }
            Type::Slice { element } => {
                format!("[]{}", Self::serialize_type_for_cache(element))
            }
            Type::Pointer {
                pointee, is_const, ..
            } => {
                let prefix = if *is_const { "*const" } else { "*" };
                format!("{}{}", prefix, Self::serialize_type_for_cache(pointee))
            }
            Type::Optional { inner } => {
                format!("?{}", Self::serialize_type_for_cache(inner))
            }
            Type::Tuple { elements } => {
                let elem_strs: Vec<_> = elements
                    .iter()
                    .map(Self::serialize_type_for_cache)
                    .collect();
                format!("({})", elem_strs.join(","))
            }
            // For other types, display_name() is sufficient
            _ => ty.display_name(),
        }
    }

    /// Check and increment recursion depth, returning error if too deep.
    fn enter_eval(&mut self) -> ComptimeResult<()> {
        self.depth += 1;
        if self.depth > MAX_COMPTIME_DEPTH {
            Err(ComptimeError {
                message: format!(
                    "comptime evaluation exceeded maximum recursion depth of {}",
                    MAX_COMPTIME_DEPTH
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Decrement recursion depth when exiting evaluation.
    fn exit_eval(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Validate that a float result is finite (not Inf or NaN).
    /// Returns an error if the value is not a valid finite number.
    fn validate_float(value: f64) -> ComptimeResult<ComptimeValue> {
        if value.is_finite() {
            Ok(ComptimeValue::Float(value))
        } else if value.is_nan() {
            Err(ComptimeError {
                message: "floating-point operation resulted in NaN".to_string(),
            })
        } else {
            Err(ComptimeError {
                message: "floating-point operation resulted in infinity".to_string(),
            })
        }
    }

    /// Resolve a built-in type name to a Type.
    /// Returns None if the name is not a built-in type.
    fn resolve_builtin_type(&self, name: &str) -> Option<Type> {
        // Special cases first
        match name {
            "bool" => return Some(Type::Bool),
            "usize" => return Some(Type::Usize),
            "isize" => return Some(Type::Isize),
            "f16" => return Some(Type::F16),
            "f32" => return Some(Type::F32),
            "f64" => return Some(Type::F64),
            "f128" => return Some(Type::F128),
            "a64" => return Some(Type::A64),
            "type" => return Some(Type::Type),
            "unit" => return Some(Type::Unit),
            _ => {}
        }

        // Arbitrary-width integers: u<bits> or i<bits>
        // Valid bit widths are 1-128
        if let Some(bits_str) = name.strip_prefix('u') {
            if let Ok(bits) = bits_str.parse::<u16>()
                && let Some(bw) = BitWidth::new(bits)
            {
                return Some(Type::UInt { bits: bw });
            }
        } else if let Some(bits_str) = name.strip_prefix('i')
            && let Ok(bits) = bits_str.parse::<u16>()
            && let Some(bw) = BitWidth::new(bits)
        {
            return Some(Type::IInt { bits: bw });
        }

        None
    }

    /// Resolve a TypeExpr to a Type at comptime.
    fn resolve_type_expr(&mut self, type_expr: &TypeExpr) -> ComptimeResult<Type> {
        match type_expr {
            TypeExpr::Primitive(prim) => Ok(match prim {
                PrimitiveType::Bool => Type::Bool,
                PrimitiveType::UInt { bits } => Type::UInt {
                    bits: BitWidth::new(*bits).unwrap_or(BitWidth::BITS_64),
                },
                PrimitiveType::IInt { bits } => Type::IInt {
                    bits: BitWidth::new(*bits).unwrap_or(BitWidth::BITS_64),
                },
                PrimitiveType::Usize => Type::Usize,
                PrimitiveType::Isize => Type::Isize,
                PrimitiveType::F16 => Type::F16,
                PrimitiveType::F32 => Type::F32,
                PrimitiveType::F64 => Type::F64,
                PrimitiveType::F128 => Type::F128,
                PrimitiveType::A64 => Type::A64,
            }),
            TypeExpr::Qubit => Ok(Type::Qubit),
            TypeExpr::Bit => Ok(Type::Bit),
            TypeExpr::QAlloc(_) => Ok(Type::Allocator { capacity: None }),
            TypeExpr::Array(array) => {
                let element = self.resolve_type_expr(&array.element)?;
                let size = if let Some(size_expr) = &array.size {
                    self.eval_expr(size_expr)?.to_usize().map(|n| n as u64)
                } else {
                    None
                };
                Ok(Type::Array {
                    element: Box::new(element),
                    size,
                })
            }
            TypeExpr::Pointer(ptr) => {
                let pointee = self.resolve_type_expr(&ptr.pointee)?;
                Ok(Type::Pointer {
                    pointee: Box::new(pointee),
                    is_const: ptr.is_const,
                    is_many: ptr.is_many,
                })
            }
            TypeExpr::Optional(inner) => {
                let inner_ty = self.resolve_type_expr(inner)?;
                Ok(Type::Optional {
                    inner: Box::new(inner_ty),
                })
            }
            TypeExpr::Named(path) => {
                // Try to resolve as built-in type first
                let full_name = path.segments.join("::");
                if let Some(ty) = self.resolve_builtin_type(&full_name) {
                    return Ok(ty);
                }
                // Try to look up in context (for comptime-defined types)
                if let Some(ComptimeValue::Type(ty)) = self.context.lookup(&full_name) {
                    return Ok(ty.clone());
                }
                // For single-segment names, try direct lookup
                if path.segments.len() == 1
                    && let Some(ComptimeValue::Type(ty)) = self.context.lookup(&path.segments[0])
                {
                    return Ok(ty.clone());
                }
                // Unresolved named type - return Unknown (will be resolved by semantic analyzer)
                Ok(Type::Unknown)
            }
            TypeExpr::Type => Ok(Type::Type),
            TypeExpr::Unit => Ok(Type::Unit),
            TypeExpr::AnyType => Ok(Type::Unknown), // anytype resolves to unknown at comptime
            TypeExpr::Tuple(types) => {
                let mut resolved = Vec::new();
                for ty in types {
                    resolved.push(self.resolve_type_expr(ty)?);
                }
                Ok(Type::Tuple { elements: resolved })
            }
            TypeExpr::Set(inner) => {
                let inner_ty = self.resolve_type_expr(inner)?;
                Ok(Type::Set {
                    element: Box::new(inner_ty),
                })
            }
            _ => Err(ComptimeError {
                message: format!(
                    "type expression not yet supported at comptime: {:?}",
                    type_expr
                ),
            }),
        }
    }

    /// Evaluate a binary operation on comptime values.
    pub fn eval_binary_op(
        &self,
        op: BinaryOp,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match op {
            // Arithmetic operations
            BinaryOp::Add => self.eval_add(left, right),
            BinaryOp::Sub => self.eval_sub(left, right),
            BinaryOp::Mul => self.eval_mul(left, right),
            BinaryOp::Div => self.eval_div(left, right),
            BinaryOp::Mod => self.eval_mod(left, right),

            // Comparison operations
            BinaryOp::Eq => self.eval_eq(left, right),
            BinaryOp::Ne => self.eval_ne(left, right),
            BinaryOp::Lt => self.eval_lt(left, right),
            BinaryOp::Le => self.eval_le(left, right),
            BinaryOp::Gt => self.eval_gt(left, right),
            BinaryOp::Ge => self.eval_ge(left, right),

            // Logical operations
            BinaryOp::And => self.eval_and(left, right),
            BinaryOp::Or => self.eval_or(left, right),

            // Bitwise operations
            BinaryOp::BitAnd => self.eval_bit_and(left, right),
            BinaryOp::BitOr => self.eval_bit_or(left, right),
            BinaryOp::BitXor => self.eval_bit_xor(left, right),
            BinaryOp::Shl => self.eval_shl(left, right),
            BinaryOp::Shr => self.eval_shr(left, right),

            // Optional operations
            BinaryOp::Orelse => self.eval_orelse(left, right),

            // Set membership (not evaluable at comptime)
            BinaryOp::In | BinaryOp::NotIn => Err(ComptimeError {
                message: "set membership not supported at comptime".to_string(),
            }),

            // Error handling (not evaluable at comptime)
            BinaryOp::Catch => Err(ComptimeError {
                message: "catch expression not supported at comptime".to_string(),
            }),
        }
    }

    /// Evaluate a unary operation.
    pub fn eval_unary_op(
        &self,
        op: UnaryOp,
        operand: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match op {
            UnaryOp::Neg => match operand {
                ComptimeValue::Int(v) => Ok(ComptimeValue::Int(-v)),
                ComptimeValue::Float(v) => Ok(ComptimeValue::Float(-v)),
                ComptimeValue::Rational(r) => Ok(ComptimeValue::Rational(-*r)),
                _ => Err(ComptimeError {
                    message: format!("cannot negate {}", operand),
                }),
            },
            UnaryOp::Not => match operand {
                ComptimeValue::Bool(v) => Ok(ComptimeValue::Bool(!v)),
                _ => Err(ComptimeError {
                    message: format!("cannot apply ! to {}", operand),
                }),
            },
            UnaryOp::BitNot => match operand {
                ComptimeValue::Int(v) => Ok(ComptimeValue::Int(!v)),
                ComptimeValue::Uint(v) => Ok(ComptimeValue::Uint(!v)),
                _ => Err(ComptimeError {
                    message: format!("cannot apply ~ to {}", operand),
                }),
            },
            UnaryOp::Deref => Err(ComptimeError {
                message: "cannot dereference at comptime".to_string(),
            }),
            UnaryOp::AddrOf => Err(ComptimeError {
                message: "cannot take address at comptime".to_string(),
            }),
            UnaryOp::OptionalUnwrap => match operand {
                ComptimeValue::Null => Err(ComptimeError {
                    message: "unwrapped null value with .?".to_string(),
                }),
                other => Ok(other.clone()),
            },
            UnaryOp::ErrorUnwrap => match operand {
                ComptimeValue::Null => Err(ComptimeError {
                    message: "unwrapped error value with .!".to_string(),
                }),
                other => Ok(other.clone()),
            },
            UnaryOp::Try => Err(ComptimeError {
                message: "try expression not supported at comptime".to_string(),
            }),
        }
    }

    // Arithmetic operations

    fn eval_add(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => a
                .checked_add(*b)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in addition".to_string(),
                }),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => a
                .checked_add(*b)
                .map(ComptimeValue::Uint)
                .ok_or_else(|| ComptimeError {
                    message: "unsigned integer overflow in addition".to_string(),
                }),
            (ComptimeValue::Int(a), ComptimeValue::Uint(b)) => a
                .checked_add(*b as i64)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in addition".to_string(),
                }),
            (ComptimeValue::Uint(a), ComptimeValue::Int(b)) => (*a as i64)
                .checked_add(*b)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in addition".to_string(),
                }),
            // Rational arithmetic
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(*a + *b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Rational(*a + Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(Rational::from_int(*a) + *b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Uint(b)) => {
                Ok(ComptimeValue::Rational(*a + Rational::from_int(*b as i64)))
            }
            (ComptimeValue::Uint(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(Rational::from_int(*a as i64) + *b))
            }
            // Float arithmetic - validate results are finite
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Self::validate_float(a + b),
            (ComptimeValue::Float(a), ComptimeValue::Int(b)) => Self::validate_float(a + *b as f64),
            (ComptimeValue::Int(a), ComptimeValue::Float(b)) => Self::validate_float(*a as f64 + b),
            // Rational with float promotes to float
            (ComptimeValue::Rational(a), ComptimeValue::Float(b)) => {
                Self::validate_float(a.to_f64() + b)
            }
            (ComptimeValue::Float(a), ComptimeValue::Rational(b)) => {
                Self::validate_float(a + b.to_f64())
            }
            _ => Err(ComptimeError {
                message: format!("cannot add {} and {}", left, right),
            }),
        }
    }

    fn eval_sub(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => a
                .checked_sub(*b)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in subtraction".to_string(),
                }),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => a
                .checked_sub(*b)
                .map(ComptimeValue::Uint)
                .ok_or_else(|| ComptimeError {
                    message: "unsigned integer underflow in subtraction".to_string(),
                }),
            (ComptimeValue::Int(a), ComptimeValue::Uint(b)) => a
                .checked_sub(*b as i64)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in subtraction".to_string(),
                }),
            (ComptimeValue::Uint(a), ComptimeValue::Int(b)) => (*a as i64)
                .checked_sub(*b)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in subtraction".to_string(),
                }),
            // Rational arithmetic
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(*a - *b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Rational(*a - Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(Rational::from_int(*a) - *b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Uint(b)) => {
                Ok(ComptimeValue::Rational(*a - Rational::from_int(*b as i64)))
            }
            (ComptimeValue::Uint(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(Rational::from_int(*a as i64) - *b))
            }
            // Float arithmetic - validate results are finite
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Self::validate_float(a - b),
            (ComptimeValue::Float(a), ComptimeValue::Int(b)) => Self::validate_float(a - *b as f64),
            (ComptimeValue::Int(a), ComptimeValue::Float(b)) => Self::validate_float(*a as f64 - b),
            // Rational with float promotes to float
            (ComptimeValue::Rational(a), ComptimeValue::Float(b)) => {
                Self::validate_float(a.to_f64() - b)
            }
            (ComptimeValue::Float(a), ComptimeValue::Rational(b)) => {
                Self::validate_float(a - b.to_f64())
            }
            _ => Err(ComptimeError {
                message: format!("cannot subtract {} and {}", left, right),
            }),
        }
    }

    fn eval_mul(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => a
                .checked_mul(*b)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in multiplication".to_string(),
                }),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => a
                .checked_mul(*b)
                .map(ComptimeValue::Uint)
                .ok_or_else(|| ComptimeError {
                    message: "unsigned integer overflow in multiplication".to_string(),
                }),
            (ComptimeValue::Int(a), ComptimeValue::Uint(b)) => a
                .checked_mul(*b as i64)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in multiplication".to_string(),
                }),
            (ComptimeValue::Uint(a), ComptimeValue::Int(b)) => (*a as i64)
                .checked_mul(*b)
                .map(ComptimeValue::Int)
                .ok_or_else(|| ComptimeError {
                    message: "integer overflow in multiplication".to_string(),
                }),
            // Rational arithmetic
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(*a * *b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Rational(*a * Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(Rational::from_int(*a) * *b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Uint(b)) => {
                Ok(ComptimeValue::Rational(*a * Rational::from_int(*b as i64)))
            }
            (ComptimeValue::Uint(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Rational(Rational::from_int(*a as i64) * *b))
            }
            // Float arithmetic - validate results are finite
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Self::validate_float(a * b),
            (ComptimeValue::Float(a), ComptimeValue::Int(b)) => Self::validate_float(a * *b as f64),
            (ComptimeValue::Int(a), ComptimeValue::Float(b)) => Self::validate_float(*a as f64 * b),
            // Rational with float promotes to float
            (ComptimeValue::Rational(a), ComptimeValue::Float(b)) => {
                Self::validate_float(a.to_f64() * b)
            }
            (ComptimeValue::Float(a), ComptimeValue::Rational(b)) => {
                Self::validate_float(a * b.to_f64())
            }
            _ => Err(ComptimeError {
                message: format!("cannot multiply {} and {}", left, right),
            }),
        }
    }

    fn eval_div(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else if a % b == 0 {
                    // Exact division - return integer
                    Ok(ComptimeValue::Int(a / b))
                } else {
                    // Non-exact division - return Rational for exact fraction representation
                    // This prevents subtle bugs like `1/4 turns` being 0
                    Ok(ComptimeValue::Rational(Rational::new(*a, *b)))
                }
            }
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else if a % b == 0 {
                    // Exact division - return unsigned integer
                    Ok(ComptimeValue::Uint(a / b))
                } else {
                    // Non-exact division - return Rational
                    Ok(ComptimeValue::Rational(Rational::new(*a as i64, *b as i64)))
                }
            }
            // Rational arithmetic
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                if b.is_zero() {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Ok(ComptimeValue::Rational(*a / *b))
                }
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Ok(ComptimeValue::Rational(*a / Rational::from_int(*b)))
                }
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                if b.is_zero() {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Ok(ComptimeValue::Rational(Rational::from_int(*a) / *b))
                }
            }
            // Float arithmetic - check for division by zero
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => {
                if *b == 0.0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(a / b)
                }
            }
            (ComptimeValue::Int(a), ComptimeValue::Float(b)) => {
                if *b == 0.0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(*a as f64 / b)
                }
            }
            (ComptimeValue::Float(a), ComptimeValue::Int(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(a / *b as f64)
                }
            }
            (ComptimeValue::Uint(a), ComptimeValue::Float(b)) => {
                if *b == 0.0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(*a as f64 / b)
                }
            }
            (ComptimeValue::Float(a), ComptimeValue::Uint(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(a / *b as f64)
                }
            }
            // Rational with float promotes to float
            (ComptimeValue::Rational(a), ComptimeValue::Float(b)) => {
                if *b == 0.0 {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(a.to_f64() / b)
                }
            }
            (ComptimeValue::Float(a), ComptimeValue::Rational(b)) => {
                if b.is_zero() {
                    Err(ComptimeError {
                        message: "division by zero".to_string(),
                    })
                } else {
                    Self::validate_float(a / b.to_f64())
                }
            }
            _ => Err(ComptimeError {
                message: format!("cannot divide {} and {}", left, right),
            }),
        }
    }

    fn eval_mod(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "modulo by zero".to_string(),
                    })
                } else {
                    Ok(ComptimeValue::Int(a % b))
                }
            }
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => {
                if *b == 0 {
                    Err(ComptimeError {
                        message: "modulo by zero".to_string(),
                    })
                } else {
                    Ok(ComptimeValue::Uint(a % b))
                }
            }
            _ => Err(ComptimeError {
                message: format!("cannot modulo {} and {}", left, right),
            }),
        }
    }

    // Comparison operations

    fn eval_eq(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        Ok(ComptimeValue::Bool(left == right))
    }

    fn eval_ne(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        Ok(ComptimeValue::Bool(left != right))
    }

    fn eval_lt(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a < b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Bool(a < b)),
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a < b)),
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(a < b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a < Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(Rational::from_int(*a) < *b))
            }
            _ => Err(ComptimeError {
                message: format!("cannot compare {} < {}", left, right),
            }),
        }
    }

    fn eval_le(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a <= b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Bool(a <= b)),
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a <= b)),
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(a <= b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a <= Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(Rational::from_int(*a) <= *b))
            }
            _ => Err(ComptimeError {
                message: format!("cannot compare {} <= {}", left, right),
            }),
        }
    }

    fn eval_gt(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a > b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Bool(a > b)),
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a > b)),
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(a > b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a > Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(Rational::from_int(*a) > *b))
            }
            _ => Err(ComptimeError {
                message: format!("cannot compare {} > {}", left, right),
            }),
        }
    }

    fn eval_ge(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a >= b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Bool(a >= b)),
            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a >= b)),
            (ComptimeValue::Rational(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(a >= b))
            }
            (ComptimeValue::Rational(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a >= Rational::from_int(*b)))
            }
            (ComptimeValue::Int(a), ComptimeValue::Rational(b)) => {
                Ok(ComptimeValue::Bool(Rational::from_int(*a) >= *b))
            }
            _ => Err(ComptimeError {
                message: format!("cannot compare {} >= {}", left, right),
            }),
        }
    }

    // Logical operations

    fn eval_and(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => Ok(ComptimeValue::Bool(*a && *b)),
            _ => Err(ComptimeError {
                message: format!("cannot apply 'and' to {} and {}", left, right),
            }),
        }
    }

    fn eval_or(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => Ok(ComptimeValue::Bool(*a || *b)),
            _ => Err(ComptimeError {
                message: format!("cannot apply 'or' to {} and {}", left, right),
            }),
        }
    }

    // Bitwise operations

    fn eval_bit_and(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a & b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Uint(a & b)),
            _ => Err(ComptimeError {
                message: format!("cannot apply & to {} and {}", left, right),
            }),
        }
    }

    fn eval_bit_or(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a | b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Uint(a | b)),
            _ => Err(ComptimeError {
                message: format!("cannot apply | to {} and {}", left, right),
            }),
        }
    }

    fn eval_bit_xor(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match (left, right) {
            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a ^ b)),
            (ComptimeValue::Uint(a), ComptimeValue::Uint(b)) => Ok(ComptimeValue::Uint(a ^ b)),
            _ => Err(ComptimeError {
                message: format!("cannot apply ^ to {} and {}", left, right),
            }),
        }
    }

    fn eval_shl(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        let shift = right.as_uint().ok_or_else(|| ComptimeError {
            message: "shift amount must be unsigned integer".to_string(),
        })?;

        // Validate shift amount is within bounds (max 63 for 64-bit integers)
        if shift >= 64 {
            return Err(ComptimeError {
                message: format!(
                    "shift amount {} is too large (max 63 for 64-bit integers)",
                    shift
                ),
            });
        }
        let shift = shift as u32;

        match left {
            ComptimeValue::Int(a) => Ok(ComptimeValue::Int(a << shift)),
            ComptimeValue::Uint(a) => Ok(ComptimeValue::Uint(a << shift)),
            _ => Err(ComptimeError {
                message: format!("cannot shift {}", left),
            }),
        }
    }

    fn eval_shr(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        let shift = right.as_uint().ok_or_else(|| ComptimeError {
            message: "shift amount must be unsigned integer".to_string(),
        })?;

        // Validate shift amount is within bounds (max 63 for 64-bit integers)
        if shift >= 64 {
            return Err(ComptimeError {
                message: format!(
                    "shift amount {} is too large (max 63 for 64-bit integers)",
                    shift
                ),
            });
        }
        let shift = shift as u32;

        match left {
            ComptimeValue::Int(a) => Ok(ComptimeValue::Int(a >> shift)),
            ComptimeValue::Uint(a) => Ok(ComptimeValue::Uint(a >> shift)),
            _ => Err(ComptimeError {
                message: format!("cannot shift {}", left),
            }),
        }
    }

    // Optional operations

    fn eval_orelse(
        &self,
        left: &ComptimeValue,
        right: &ComptimeValue,
    ) -> ComptimeResult<ComptimeValue> {
        match left {
            ComptimeValue::Null => Ok(right.clone()),
            other => Ok(other.clone()),
        }
    }

    // =========================================================================
    // Expression Evaluation
    // =========================================================================

    /// Evaluate an expression at compile time.
    pub fn eval_expr(&mut self, expr: &Expr) -> ComptimeResult<ComptimeValue> {
        // Check recursion depth
        self.enter_eval()?;
        let result = self.eval_expr_inner(expr);
        self.exit_eval();
        result
    }

    /// Inner expression evaluation (after depth check).
    fn eval_expr_inner(&mut self, expr: &Expr) -> ComptimeResult<ComptimeValue> {
        match expr {
            // Literals
            Expr::IntLit(lit) => {
                // Convert i128 to i64, checking for overflow
                let value = lit.value.try_into().map_err(|_| ComptimeError {
                    message: format!("integer literal {} too large for comptime i64", lit.value),
                })?;
                Ok(ComptimeValue::Int(value))
            }
            Expr::FloatLit(lit) => Ok(ComptimeValue::Float(lit.value)),
            Expr::AngleLit(angle) => {
                // Evaluate the inner expression and convert to turns (native unit)
                let val = self.eval_expr(&angle.value)?;
                use crate::ast::AngleUnit;

                // For Rational values, preserve exact fraction when possible
                if let ComptimeValue::Rational(r) = &val {
                    match angle.unit {
                        AngleUnit::Turns => {
                            // Already in turns - preserve exact rational
                            return Ok(val);
                        }
                        AngleUnit::Rad => {
                            // For rational radians, we can't preserve precision
                            // since radians involve pi (irrational)
                            // But we can try to detect if it represents n*pi/d
                            let radians = r.to_f64();
                            if let Some(turns_rational) = Rational::radians_to_turns(radians) {
                                return Ok(ComptimeValue::Rational(turns_rational));
                            }
                            let turns = radians / (2.0 * std::f64::consts::PI);
                            return Ok(ComptimeValue::Float(turns));
                        }
                    }
                }

                let numeric = match &val {
                    ComptimeValue::Float(f) => *f,
                    ComptimeValue::Int(i) => *i as f64,
                    ComptimeValue::Uint(u) => *u as f64,
                    _ => {
                        return Err(ComptimeError {
                            message: format!("angle value must be numeric, found {:?}", val),
                        });
                    }
                };

                // For radian values, try to detect pi-multiples and preserve precision
                if let AngleUnit::Rad = angle.unit
                    && let Some(turns_rational) = Rational::radians_to_turns(numeric)
                {
                    return Ok(ComptimeValue::Rational(turns_rational));
                }

                // Fall back to float conversion
                let turns = angle.unit.to_turns(numeric);
                Ok(ComptimeValue::Float(turns))
            }
            Expr::TypeAscription(asc) => {
                // Evaluate the inner expression and convert to the specified type
                let val = self.eval_expr(&asc.value)?;
                // Convert based on target type
                match asc.type_name.as_str() {
                    "f64" | "f32" | "f16" | "f128" => {
                        let f = match val {
                            ComptimeValue::Float(f) => f,
                            ComptimeValue::Int(i) => i as f64,
                            ComptimeValue::Uint(u) => u as f64,
                            ComptimeValue::Rational(r) => r.to_f64(),
                            _ => {
                                return Err(ComptimeError {
                                    message: format!(
                                        "cannot convert {:?} to {}",
                                        val, asc.type_name
                                    ),
                                });
                            }
                        };
                        Ok(ComptimeValue::Float(f))
                    }
                    "a64" => {
                        // a64 expects a value in turns - preserve Rational if possible
                        match val {
                            ComptimeValue::Rational(r) => Ok(ComptimeValue::Rational(r)),
                            ComptimeValue::Float(f) => Ok(ComptimeValue::Float(f)),
                            ComptimeValue::Int(i) => {
                                Ok(ComptimeValue::Rational(Rational::from_int(i)))
                            }
                            ComptimeValue::Uint(u) => {
                                Ok(ComptimeValue::Rational(Rational::from_int(u as i64)))
                            }
                            _ => Err(ComptimeError {
                                message: format!("cannot convert {:?} to a64", val),
                            }),
                        }
                    }
                    t if t.starts_with('u') => {
                        let u = match val {
                            ComptimeValue::Uint(u) => u,
                            ComptimeValue::Int(i) if i >= 0 => i as u64,
                            ComptimeValue::Float(f) if f >= 0.0 => f as u64,
                            _ => {
                                return Err(ComptimeError {
                                    message: format!(
                                        "cannot convert {:?} to {}",
                                        val, asc.type_name
                                    ),
                                });
                            }
                        };
                        Ok(ComptimeValue::Uint(u))
                    }
                    t if t.starts_with('i') => {
                        let i = match val {
                            ComptimeValue::Int(i) => i,
                            ComptimeValue::Uint(u) => u as i64,
                            ComptimeValue::Float(f) => f as i64,
                            _ => {
                                return Err(ComptimeError {
                                    message: format!(
                                        "cannot convert {:?} to {}",
                                        val, asc.type_name
                                    ),
                                });
                            }
                        };
                        Ok(ComptimeValue::Int(i))
                    }
                    _ => Err(ComptimeError {
                        message: format!("unknown type suffix: {}", asc.type_name),
                    }),
                }
            }
            Expr::BoolLit(lit) => Ok(ComptimeValue::Bool(lit.value)),
            Expr::StringLit(lit) => Ok(ComptimeValue::String(lit.value.clone())),
            Expr::FString(fstr) => {
                // Evaluate f-string by concatenating all parts
                let mut result = String::new();
                for part in &fstr.parts {
                    match part {
                        FStringPart::Text(text) => result.push_str(text),
                        FStringPart::Expr { expr, format } => {
                            let val = self.eval_expr(expr)?;
                            // TODO: Apply format specifier at comptime if needed
                            let _ = format; // Acknowledge format spec (unused at comptime for now)
                            result.push_str(&val.to_string());
                        }
                    }
                }
                Ok(ComptimeValue::String(result))
            }
            Expr::CharLit(lit) => Ok(ComptimeValue::Uint(lit.value as u64)),
            Expr::Null(_) => Ok(ComptimeValue::Null),
            Expr::Undefined(_) => Ok(ComptimeValue::Undefined),
            Expr::Unit(_) => Ok(ComptimeValue::Unit),

            Expr::Ident(ident) => {
                // First check local context
                if let Some(val) = self.context.lookup(&ident.name) {
                    return Ok(val.clone());
                }

                // Check for built-in types (u8, u32, i64, bool, etc.)
                if let Some(ty) = self.resolve_builtin_type(&ident.name) {
                    return Ok(ComptimeValue::Type(ty));
                }

                Err(ComptimeError {
                    message: format!("undefined variable '{}' at comptime", ident.name),
                })
            }

            Expr::Binary(binary) => {
                let left = self.eval_expr(&binary.left)?;
                let right = self.eval_expr(&binary.right)?;
                self.eval_binary_op(binary.op, &left, &right)
            }

            Expr::Unary(unary) => {
                let operand = self.eval_expr(&unary.operand)?;
                self.eval_unary_op(unary.op, &operand)
            }

            Expr::If(if_expr) => {
                let cond = self.eval_expr(&if_expr.condition)?;
                if cond.is_truthy() {
                    self.eval_expr(&if_expr.then_expr)
                } else {
                    self.eval_expr(&if_expr.else_expr)
                }
            }

            Expr::Block(block) => {
                self.context.push_scope();

                for stmt in &block.statements {
                    self.eval_stmt(stmt)?;
                }

                // Evaluate trailing expression if present (block's return value)
                let result = if let Some(trailing) = &block.trailing_expr {
                    self.eval_expr(trailing)?
                } else {
                    ComptimeValue::Unit
                };

                self.context.pop_scope();
                Ok(result)
            }

            Expr::Comptime(comptime) => {
                // Already in comptime context, just evaluate inner
                self.eval_expr(&comptime.inner)
            }

            Expr::Index(index) => {
                let object = self.eval_expr(&index.object)?;
                let idx = self.eval_expr(&index.index)?;

                match (&object, &idx) {
                    (ComptimeValue::Array(arr), ComptimeValue::Int(i)) => {
                        let i = *i as usize;
                        arr.get(i).cloned().ok_or_else(|| ComptimeError {
                            message: format!(
                                "index {} out of bounds for array of length {}",
                                i,
                                arr.len()
                            ),
                        })
                    }
                    (ComptimeValue::Array(arr), ComptimeValue::Uint(i)) => {
                        let i = *i as usize;
                        arr.get(i).cloned().ok_or_else(|| ComptimeError {
                            message: format!(
                                "index {} out of bounds for array of length {}",
                                i,
                                arr.len()
                            ),
                        })
                    }
                    _ => Err(ComptimeError {
                        message: format!("cannot index {} with {}", object, idx),
                    }),
                }
            }

            Expr::Field(field) => {
                let object = self.eval_expr(&field.object)?;

                match object {
                    ComptimeValue::Struct { fields, .. } => fields
                        .get(&field.field)
                        .cloned()
                        .ok_or_else(|| ComptimeError {
                            message: format!("no field '{}' on struct", field.field),
                        }),
                    _ => Err(ComptimeError {
                        message: format!("cannot access field on {}", object),
                    }),
                }
            }

            Expr::Range(range) => {
                let start = range
                    .start
                    .as_ref()
                    .map(|e| self.eval_expr(e))
                    .transpose()?
                    .unwrap_or(ComptimeValue::Int(0));
                let end = range
                    .end
                    .as_ref()
                    .map(|e| self.eval_expr(e))
                    .transpose()?
                    .ok_or_else(|| ComptimeError {
                        message: "range must have an end".to_string(),
                    })?;

                match (&start, &end) {
                    (ComptimeValue::Int(s), ComptimeValue::Int(e)) => {
                        let arr: Vec<ComptimeValue> = (*s..*e).map(ComptimeValue::Int).collect();
                        Ok(ComptimeValue::Array(arr))
                    }
                    (ComptimeValue::Uint(s), ComptimeValue::Uint(e)) => {
                        let arr: Vec<ComptimeValue> = (*s..*e).map(ComptimeValue::Uint).collect();
                        Ok(ComptimeValue::Array(arr))
                    }
                    _ => Err(ComptimeError {
                        message: "range bounds must be integers".to_string(),
                    }),
                }
            }

            Expr::ArrayInit(arr) => {
                let values: ComptimeResult<Vec<ComptimeValue>> =
                    arr.elements.iter().map(|e| self.eval_expr(e)).collect();
                Ok(ComptimeValue::Array(values?))
            }

            Expr::BracketArray(arr) => {
                let values: ComptimeResult<Vec<ComptimeValue>> =
                    arr.elements.iter().map(|e| self.eval_expr(e)).collect();
                Ok(ComptimeValue::Array(values?))
            }

            // Comptime function calls
            Expr::Call(call) => {
                // First, resolve the callee
                let callee = self.eval_expr(&call.callee)?;

                match callee {
                    ComptimeValue::Function(func) => {
                        // Evaluate arguments
                        let mut arg_values = Vec::new();
                        for arg in &call.args {
                            arg_values.push(self.eval_expr(arg)?);
                        }

                        // Check argument count
                        if arg_values.len() != func.params.len() {
                            return Err(ComptimeError {
                                message: format!(
                                    "function '{}' expects {} arguments, got {}",
                                    func.name,
                                    func.params.len(),
                                    arg_values.len()
                                ),
                            });
                        }

                        // Check memoization cache
                        let cache_key = (
                            func.name.clone(),
                            Self::serialize_args_for_cache(&arg_values),
                        );
                        if let Some(cached_result) = self.memo_cache.get(&cache_key) {
                            return Ok(cached_result.clone());
                        }

                        // Create new scope for function execution
                        self.context.push_scope();

                        // Clone arg_values for binding (we need them for caching too)
                        // Bind parameters to argument values
                        for (param, value) in func.params.iter().zip(arg_values.clone()) {
                            self.context.define(&param.name, value);
                        }

                        // Execute function body
                        let mut result = ComptimeValue::Unit;
                        for stmt in &func.body.statements {
                            // Check for early return
                            if let Stmt::Return(ret) = stmt {
                                result = if let Some(value) = &ret.value {
                                    self.eval_expr(value)?
                                } else {
                                    ComptimeValue::Unit
                                };
                                self.context.pop_scope();
                                // Cache the result before returning
                                self.memo_cache.insert(cache_key, result.clone());
                                return Ok(result);
                            }
                            self.eval_stmt(stmt)?;
                        }

                        // Evaluate trailing expression if present
                        if let Some(trailing) = &func.body.trailing_expr {
                            result = self.eval_expr(trailing)?;
                        }

                        self.context.pop_scope();

                        // Cache the result
                        self.memo_cache.insert(cache_key, result.clone());
                        Ok(result)
                    }
                    _ => Err(ComptimeError {
                        message: format!("cannot call non-function value: {}", callee),
                    }),
                }
            }

            Expr::SlotRef(_) => Err(ComptimeError {
                message: "qubit references not supported at comptime".to_string(),
            }),

            Expr::BitRef(_) => Err(ComptimeError {
                message: "bit references not supported at comptime".to_string(),
            }),

            Expr::Builtin(builtin) => self.eval_builtin(builtin),

            Expr::AnonStruct(anon) => {
                // Convert anonymous struct expression to a Type value
                let mut fields = Vec::new();
                for field in &anon.fields {
                    let field_ty = self.resolve_type_expr(&field.ty)?;
                    fields.push((field.name.clone(), field_ty));
                }
                Ok(ComptimeValue::Type(Type::Struct {
                    name: "<anon>".to_string(),
                    fields,
                }))
            }

            Expr::StructInit(_) => Err(ComptimeError {
                message: "struct init not yet supported at comptime".to_string(),
            }),

            Expr::Tuple(tuple) => {
                let values: ComptimeResult<Vec<ComptimeValue>> =
                    tuple.elements.iter().map(|e| self.eval_expr(e)).collect();
                Ok(ComptimeValue::Array(values?)) // Represent tuples as arrays for now
            }

            Expr::Set(_) => Err(ComptimeError {
                message: "set literals not supported at comptime".to_string(),
            }),

            // Error/fault handling expressions
            Expr::ErrorValue(_) => Err(ComptimeError {
                message: "error values not supported at comptime".to_string(),
            }),

            Expr::FaultValue(_) => Err(ComptimeError {
                message: "fault values not supported at comptime".to_string(),
            }),

            Expr::Catch(_) => Err(ComptimeError {
                message: "catch expressions not supported at comptime".to_string(),
            }),

            Expr::TryBlock(_) => Err(ComptimeError {
                message: "try blocks not supported at comptime".to_string(),
            }),

            // Batch apply is a runtime operation
            Expr::BatchApply(_) => Err(ComptimeError {
                message: "batch apply not supported at comptime".to_string(),
            }),

            // Measurement is a runtime operation
            Expr::Measure(_) => Err(ComptimeError {
                message: "measurement not supported at comptime".to_string(),
            }),

            // Gate operations are runtime-only
            Expr::Gate(_) => Err(ComptimeError {
                message: "gate operations not supported at comptime".to_string(),
            }),

            // Function literal - creates a comptime function value
            Expr::FnLit(func) => Ok(ComptimeValue::Function(func.clone())),

            // Channel expressions (@emit.log.*, @emit.sim.*, @emit.hw.*, custom channels)
            // At comptime, these evaluate to unit (actual behavior happens at runtime)
            Expr::Channel(channel) => {
                // Evaluate all argument expressions at comptime
                for arg in &channel.args {
                    self.eval_expr(arg.value())?;
                }
                Ok(ComptimeValue::Unit)
            }

            // Result expressions - emit tagged values to caller
            // At comptime, these evaluate to unit (emission happens at runtime)
            Expr::Result(_result) => {
                // result() always evaluates to unit
                // The actual emission happens at runtime
                Ok(ComptimeValue::Unit)
            }
        }
    }

    /// Evaluate a for range to get iterable values.
    fn eval_for_range(&mut self, range: &ForRange) -> ComptimeResult<Vec<ComptimeValue>> {
        match range {
            ForRange::Range { start, end } => {
                let start_val = self.eval_expr(start)?;
                let end_val = self.eval_expr(end)?;

                match (&start_val, &end_val) {
                    (ComptimeValue::Int(s), ComptimeValue::Int(e)) => {
                        Ok((*s..*e).map(ComptimeValue::Int).collect())
                    }
                    (ComptimeValue::Uint(s), ComptimeValue::Uint(e)) => {
                        Ok((*s..*e).map(ComptimeValue::Uint).collect())
                    }
                    _ => Err(ComptimeError {
                        message: "range bounds must be integers".to_string(),
                    }),
                }
            }
            ForRange::Collection(expr) => {
                let val = self.eval_expr(expr)?;
                match val {
                    ComptimeValue::Array(arr) => Ok(arr),
                    _ => Err(ComptimeError {
                        message: "cannot iterate over non-array at comptime".to_string(),
                    }),
                }
            }
        }
    }

    /// Evaluate a builtin function at compile time.
    fn eval_builtin(&mut self, builtin: &crate::ast::BuiltinExpr) -> ComptimeResult<ComptimeValue> {
        match builtin.name.as_str() {
            // Support both snake_case (preferred) and camelCase (legacy) for builtins
            "size_of" | "sizeOf" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@size_of requires a type argument".to_string(),
                    });
                }
                // Get the type name from the argument
                let size = self.get_type_size(&builtin.args[0])?;
                Ok(ComptimeValue::Uint(size))
            }
            "type_name" | "typeName" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@type_name requires a type argument".to_string(),
                    });
                }
                let name = self.get_type_name(&builtin.args[0])?;
                Ok(ComptimeValue::String(name))
            }
            "align_of" | "alignOf" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@align_of requires a type argument".to_string(),
                    });
                }
                // For simplicity, alignment equals size for primitive types
                let align = self.get_type_size(&builtin.args[0])?;
                Ok(ComptimeValue::Uint(align))
            }
            "type_info" | "typeInfo" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@type_info requires a type argument".to_string(),
                    });
                }
                self.eval_type_info(&builtin.args[0])
            }
            "field_names" | "fieldNames" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@field_names requires a type argument".to_string(),
                    });
                }
                self.eval_field_names(&builtin.args[0])
            }
            "enum_fields" | "enumFields" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@enum_fields requires a type argument".to_string(),
                    });
                }
                self.eval_enum_fields(&builtin.args[0])
            }
            "type_from_info" | "Type" => {
                if builtin.args.is_empty() {
                    return Err(ComptimeError {
                        message: "@type_from_info requires a TypeInfo argument".to_string(),
                    });
                }
                self.eval_type_from_info(&builtin.args[0])
            }
            _ => Err(ComptimeError {
                message: format!("builtin @{} not supported at comptime", builtin.name),
            }),
        }
    }

    /// Get the size of a type expression in bytes.
    fn get_type_size(&self, expr: &Expr) -> ComptimeResult<u64> {
        match expr {
            Expr::Ident(ident) => {
                match ident.name.as_str() {
                    "u8" | "i8" | "bool" => Ok(1),
                    "u16" | "i16" => Ok(2),
                    "u32" | "i32" | "f32" => Ok(4),
                    "u64" | "i64" | "f64" | "usize" | "isize" => Ok(8),
                    "u128" | "i128" => Ok(16),
                    "a64" => Ok(8), // Angle type
                    "unit" => Ok(0),
                    // Qubit and allocator are abstract, but we can assign sizes
                    "Qubit" | "qubit" => Ok(8), // Pointer-sized
                    _ => Err(ComptimeError {
                        message: format!("unknown type '{}' for @size_of", ident.name),
                    }),
                }
            }
            // Pointer types
            Expr::Unary(unary) if matches!(unary.op, UnaryOp::AddrOf) => {
                Ok(8) // Pointers are 8 bytes on 64-bit
            }
            _ => Err(ComptimeError {
                message: "cannot determine size of complex type expression".to_string(),
            }),
        }
    }

    /// Get the name of a type expression.
    fn get_type_name(&self, expr: &Expr) -> ComptimeResult<String> {
        match expr {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            _ => Err(ComptimeError {
                message: "cannot determine name of complex type expression".to_string(),
            }),
        }
    }

    /// Get the TypeInfoKind for a Type.
    fn get_type_info_kind(ty: &Type) -> TypeInfoKind {
        match ty {
            Type::Bool
            | Type::UInt { .. }
            | Type::IInt { .. }
            | Type::Usize
            | Type::Isize
            | Type::F16
            | Type::F32
            | Type::F64
            | Type::F128
            | Type::A64 => TypeInfoKind::Primitive,
            Type::Array { .. } => TypeInfoKind::Array,
            Type::Slice { .. } => TypeInfoKind::Slice,
            Type::Set { .. } => TypeInfoKind::Struct, // Set is like a collection
            Type::Pointer { .. } => TypeInfoKind::Pointer,
            Type::Optional { .. } => TypeInfoKind::Optional,
            Type::ErrorUnion { .. } | Type::CollectedErrors { .. } => TypeInfoKind::ErrorUnion,
            Type::Struct { .. } => TypeInfoKind::Struct,
            Type::Enum { .. } => TypeInfoKind::Enum,
            Type::Union { .. } => TypeInfoKind::Union,
            Type::ErrorSet { .. } => TypeInfoKind::ErrorSet,
            Type::FaultSet { .. } => TypeInfoKind::FaultSet,
            Type::Function { .. } => TypeInfoKind::Function,
            Type::Tuple { .. } => TypeInfoKind::Tuple,
            Type::Type => TypeInfoKind::Type,
            Type::Unit => TypeInfoKind::Unit,
            Type::Never => TypeInfoKind::Never,
            Type::Qubit | Type::Bit | Type::Allocator { .. } => TypeInfoKind::Quantum,
            Type::Comptime(inner) => Self::get_type_info_kind(inner),
            Type::Module { .. } => TypeInfoKind::Struct, // Module is like a namespace
            Type::AnyError | Type::AnyFault | Type::Unknown => TypeInfoKind::Unknown,
        }
    }

    /// Resolve a type from an expression (for @type_info and related builtins).
    fn resolve_type_from_expr(&self, expr: &Expr) -> ComptimeResult<Type> {
        match expr {
            Expr::Ident(ident) => {
                // Check if it's a type in the context
                if let Some(val) = self.context.lookup(&ident.name)
                    && let ComptimeValue::Type(ty) = val
                {
                    return Ok(ty.clone());
                }
                // Try to resolve primitive types
                match ident.name.as_str() {
                    "bool" => Ok(Type::Bool),
                    "u8" => Ok(Type::UInt {
                        bits: BitWidth::must(8),
                    }),
                    "u16" => Ok(Type::UInt {
                        bits: BitWidth::must(16),
                    }),
                    "u32" => Ok(Type::UInt {
                        bits: BitWidth::must(32),
                    }),
                    "u64" => Ok(Type::UInt {
                        bits: BitWidth::must(64),
                    }),
                    "u128" => Ok(Type::UInt {
                        bits: BitWidth::must(128),
                    }),
                    "i8" => Ok(Type::IInt {
                        bits: BitWidth::must(8),
                    }),
                    "i16" => Ok(Type::IInt {
                        bits: BitWidth::must(16),
                    }),
                    "i32" => Ok(Type::IInt {
                        bits: BitWidth::must(32),
                    }),
                    "i64" => Ok(Type::IInt {
                        bits: BitWidth::must(64),
                    }),
                    "i128" => Ok(Type::IInt {
                        bits: BitWidth::must(128),
                    }),
                    "usize" => Ok(Type::Usize),
                    "isize" => Ok(Type::Isize),
                    "f16" => Ok(Type::F16),
                    "f32" => Ok(Type::F32),
                    "f64" => Ok(Type::F64),
                    "f128" => Ok(Type::F128),
                    "a64" => Ok(Type::A64),
                    "unit" => Ok(Type::Unit),
                    "type" => Ok(Type::Type),
                    "never" => Ok(Type::Never),
                    "Qubit" | "qubit" => Ok(Type::Qubit),
                    "Bit" | "bit" => Ok(Type::Bit),
                    _ => Err(ComptimeError {
                        message: format!("unknown type '{}'", ident.name),
                    }),
                }
            }
            _ => Err(ComptimeError {
                message: "complex type expressions not yet supported in @type_info".to_string(),
            }),
        }
    }

    /// Evaluate @type_info(T) - returns a struct with type information.
    fn eval_type_info(&self, expr: &Expr) -> ComptimeResult<ComptimeValue> {
        let ty = self.resolve_type_from_expr(expr)?;
        let kind = Self::get_type_info_kind(&ty);

        let mut fields = BTreeMap::new();
        fields.insert(
            "kind".to_string(),
            ComptimeValue::String(kind.as_str().to_string()),
        );
        fields.insert("name".to_string(), ComptimeValue::String(ty.display_name()));

        // Add type-specific information
        match &ty {
            Type::Struct {
                name,
                fields: struct_fields,
            } => {
                let field_names: Vec<ComptimeValue> = struct_fields
                    .iter()
                    .map(|(n, _)| ComptimeValue::String(n.clone()))
                    .collect();
                fields.insert("fields".to_string(), ComptimeValue::Array(field_names));
                fields.insert(
                    "struct_name".to_string(),
                    ComptimeValue::String(name.clone()),
                );
            }
            Type::Enum { name, variants } => {
                let variant_names: Vec<ComptimeValue> = variants
                    .iter()
                    .map(|v| ComptimeValue::String(v.clone()))
                    .collect();
                fields.insert("variants".to_string(), ComptimeValue::Array(variant_names));
                fields.insert("enum_name".to_string(), ComptimeValue::String(name.clone()));
            }
            Type::Union {
                name,
                fields: union_fields,
                is_tagged,
            } => {
                let field_names: Vec<ComptimeValue> = union_fields
                    .iter()
                    .map(|(n, _)| ComptimeValue::String(n.clone()))
                    .collect();
                fields.insert("fields".to_string(), ComptimeValue::Array(field_names));
                fields.insert(
                    "union_name".to_string(),
                    ComptimeValue::String(name.clone()),
                );
                fields.insert("is_tagged".to_string(), ComptimeValue::Bool(*is_tagged));
            }
            Type::ErrorSet { name, errors } => {
                let error_names: Vec<ComptimeValue> = errors
                    .iter()
                    .map(|(n, _)| ComptimeValue::String(n.clone()))
                    .collect();
                fields.insert("errors".to_string(), ComptimeValue::Array(error_names));
                fields.insert(
                    "error_set_name".to_string(),
                    ComptimeValue::String(name.clone()),
                );
            }
            Type::FaultSet { name, faults } => {
                let fault_names: Vec<ComptimeValue> = faults
                    .iter()
                    .map(|(n, _)| ComptimeValue::String(n.clone()))
                    .collect();
                fields.insert("faults".to_string(), ComptimeValue::Array(fault_names));
                fields.insert(
                    "fault_set_name".to_string(),
                    ComptimeValue::String(name.clone()),
                );
            }
            Type::Array { element, size } => {
                fields.insert("element".to_string(), ComptimeValue::Type(*element.clone()));
                if let Some(sz) = size {
                    fields.insert("size".to_string(), ComptimeValue::Uint(*sz));
                }
            }
            Type::Slice { element } => {
                fields.insert("element".to_string(), ComptimeValue::Type(*element.clone()));
            }
            Type::Pointer {
                pointee,
                is_const,
                is_many,
            } => {
                fields.insert("pointee".to_string(), ComptimeValue::Type(*pointee.clone()));
                fields.insert("is_const".to_string(), ComptimeValue::Bool(*is_const));
                fields.insert("is_many".to_string(), ComptimeValue::Bool(*is_many));
            }
            Type::Optional { inner } => {
                fields.insert("child".to_string(), ComptimeValue::Type(*inner.clone()));
            }
            Type::ErrorUnion { error, payload } => {
                fields.insert("error".to_string(), ComptimeValue::Type(*error.clone()));
                fields.insert("payload".to_string(), ComptimeValue::Type(*payload.clone()));
            }
            Type::Function {
                params,
                return_type,
            } => {
                let param_types: Vec<ComptimeValue> = params
                    .iter()
                    .map(|p| ComptimeValue::Type(p.clone()))
                    .collect();
                fields.insert("params".to_string(), ComptimeValue::Array(param_types));
                fields.insert(
                    "return_type".to_string(),
                    ComptimeValue::Type(*return_type.clone()),
                );
            }
            Type::Tuple { elements } => {
                let element_types: Vec<ComptimeValue> = elements
                    .iter()
                    .map(|e| ComptimeValue::Type(e.clone()))
                    .collect();
                fields.insert("elements".to_string(), ComptimeValue::Array(element_types));
            }
            _ => {
                // Primitive types don't have additional info
            }
        }

        Ok(ComptimeValue::Struct {
            name: "TypeInfo".to_string(),
            fields,
        })
    }

    /// Evaluate @field_names(T) - returns an array of field name strings for structs.
    fn eval_field_names(&self, expr: &Expr) -> ComptimeResult<ComptimeValue> {
        let ty = self.resolve_type_from_expr(expr)?;

        match &ty {
            Type::Struct { fields, .. } => {
                let names: Vec<ComptimeValue> = fields
                    .iter()
                    .map(|(name, _)| ComptimeValue::String(name.clone()))
                    .collect();
                Ok(ComptimeValue::Array(names))
            }
            Type::Union { fields, .. } => {
                let names: Vec<ComptimeValue> = fields
                    .iter()
                    .map(|(name, _)| ComptimeValue::String(name.clone()))
                    .collect();
                Ok(ComptimeValue::Array(names))
            }
            _ => Err(ComptimeError {
                message: format!(
                    "@field_names requires a struct or union type, got {}",
                    ty.display_name()
                ),
            }),
        }
    }

    /// Evaluate @enum_fields(T) - returns an array of enum variant names.
    fn eval_enum_fields(&self, expr: &Expr) -> ComptimeResult<ComptimeValue> {
        let ty = self.resolve_type_from_expr(expr)?;

        match &ty {
            Type::Enum { variants, .. } => {
                let names: Vec<ComptimeValue> = variants
                    .iter()
                    .map(|v| ComptimeValue::String(v.clone()))
                    .collect();
                Ok(ComptimeValue::Array(names))
            }
            Type::ErrorSet { errors, .. } => {
                let names: Vec<ComptimeValue> = errors
                    .iter()
                    .map(|(name, _)| ComptimeValue::String(name.clone()))
                    .collect();
                Ok(ComptimeValue::Array(names))
            }
            Type::FaultSet { faults, .. } => {
                let names: Vec<ComptimeValue> = faults
                    .iter()
                    .map(|(name, _)| ComptimeValue::String(name.clone()))
                    .collect();
                Ok(ComptimeValue::Array(names))
            }
            _ => Err(ComptimeError {
                message: format!(
                    "@enum_fields requires an enum, error set, or fault set type, got {}",
                    ty.display_name()
                ),
            }),
        }
    }

    /// Evaluate @Type(info) - construct a type from a TypeInfo struct.
    /// This is the reverse of @type_info.
    fn eval_type_from_info(&mut self, expr: &Expr) -> ComptimeResult<ComptimeValue> {
        let info = self.eval_expr(expr)?;

        match info {
            ComptimeValue::Struct { name, fields } if name == "TypeInfo" => {
                // Get the kind field
                let kind = fields.get("kind").ok_or_else(|| ComptimeError {
                    message: "@Type requires TypeInfo with 'kind' field".to_string(),
                })?;

                let kind_str = match kind {
                    ComptimeValue::String(s) => s.as_str(),
                    _ => {
                        return Err(ComptimeError {
                            message: "TypeInfo.kind must be a string".to_string(),
                        });
                    }
                };

                // Construct the type based on kind
                let ty = match kind_str {
                    "primitive" => {
                        // Get the name to determine which primitive
                        let name = fields
                            .get("name")
                            .and_then(|v| {
                                if let ComptimeValue::String(s) = v {
                                    Some(s.as_str())
                                } else {
                                    None
                                }
                            })
                            .ok_or_else(|| ComptimeError {
                                message: "primitive TypeInfo requires 'name' field".to_string(),
                            })?;

                        match name {
                            "bool" => Type::Bool,
                            "u8" => Type::UInt {
                                bits: BitWidth::must(8),
                            },
                            "u16" => Type::UInt {
                                bits: BitWidth::must(16),
                            },
                            "u32" => Type::UInt {
                                bits: BitWidth::must(32),
                            },
                            "u64" => Type::UInt {
                                bits: BitWidth::must(64),
                            },
                            "i8" => Type::IInt {
                                bits: BitWidth::must(8),
                            },
                            "i16" => Type::IInt {
                                bits: BitWidth::must(16),
                            },
                            "i32" => Type::IInt {
                                bits: BitWidth::must(32),
                            },
                            "i64" => Type::IInt {
                                bits: BitWidth::must(64),
                            },
                            "f32" => Type::F32,
                            "f64" => Type::F64,
                            _ => {
                                return Err(ComptimeError {
                                    message: format!("unknown primitive type '{}'", name),
                                });
                            }
                        }
                    }
                    "array" => {
                        let element = fields.get("element").ok_or_else(|| ComptimeError {
                            message: "array TypeInfo requires 'element' field".to_string(),
                        })?;
                        let element_ty = match element {
                            ComptimeValue::Type(t) => t.clone(),
                            _ => {
                                return Err(ComptimeError {
                                    message: "TypeInfo.element must be a type".to_string(),
                                });
                            }
                        };
                        let size = fields.get("size").and_then(|v| match v {
                            ComptimeValue::Uint(n) => Some(*n),
                            ComptimeValue::Int(n) if *n >= 0 => Some(*n as u64),
                            _ => None,
                        });
                        Type::Array {
                            element: Box::new(element_ty),
                            size,
                        }
                    }
                    "slice" => {
                        let element = fields.get("element").ok_or_else(|| ComptimeError {
                            message: "slice TypeInfo requires 'element' field".to_string(),
                        })?;
                        let element_ty = match element {
                            ComptimeValue::Type(t) => t.clone(),
                            _ => {
                                return Err(ComptimeError {
                                    message: "TypeInfo.element must be a type".to_string(),
                                });
                            }
                        };
                        Type::Slice {
                            element: Box::new(element_ty),
                        }
                    }
                    "optional" => {
                        let child = fields.get("child").ok_or_else(|| ComptimeError {
                            message: "optional TypeInfo requires 'child' field".to_string(),
                        })?;
                        let child_ty = match child {
                            ComptimeValue::Type(t) => t.clone(),
                            _ => {
                                return Err(ComptimeError {
                                    message: "TypeInfo.child must be a type".to_string(),
                                });
                            }
                        };
                        Type::Optional {
                            inner: Box::new(child_ty),
                        }
                    }
                    "unit" => Type::Unit,
                    "never" => Type::Never,
                    "type" => Type::Type,
                    _ => {
                        return Err(ComptimeError {
                            message: format!("cannot construct type from kind '{}'", kind_str),
                        });
                    }
                };

                Ok(ComptimeValue::Type(ty))
            }
            _ => Err(ComptimeError {
                message: "@Type requires a TypeInfo struct argument".to_string(),
            }),
        }
    }

    /// Evaluate a statement at compile time.
    pub fn eval_stmt(&mut self, stmt: &Stmt) -> ComptimeResult<ComptimeValue> {
        match stmt {
            Stmt::Binding(binding) => {
                let value = if let Some(init) = &binding.value {
                    self.eval_expr(init)?
                } else {
                    ComptimeValue::Undefined
                };
                self.context.define(&binding.name, value);
                Ok(ComptimeValue::Undefined)
            }

            Stmt::Alias(alias) => {
                // Aliases are evaluated as their source expression at comptime
                let value = self.eval_expr(&alias.source)?;
                self.context.define(&alias.name, value);
                Ok(ComptimeValue::Undefined)
            }

            Stmt::Assign(assign) => {
                let value = self.eval_expr(&assign.value)?;

                // Handle simple identifier assignment
                if let Expr::Ident(ident) = &assign.target {
                    if !self.context.update(&ident.name, value) {
                        return Err(ComptimeError {
                            message: format!("undefined variable '{}'", ident.name),
                        });
                    }
                } else {
                    return Err(ComptimeError {
                        message: "complex assignment targets not supported at comptime".to_string(),
                    });
                }
                Ok(ComptimeValue::Undefined)
            }

            Stmt::Expr(expr_stmt) => {
                self.eval_expr(&expr_stmt.expr)?;
                Ok(ComptimeValue::Undefined)
            }

            Stmt::Return(ret) => {
                if let Some(value) = &ret.value {
                    self.eval_expr(value)
                } else {
                    Ok(ComptimeValue::Undefined)
                }
            }

            Stmt::If(if_stmt) => {
                let cond = self.eval_expr(&if_stmt.condition)?;

                if cond.is_truthy() {
                    self.context.push_scope();
                    for stmt in &if_stmt.then_body.statements {
                        self.eval_stmt(stmt)?;
                    }
                    self.context.pop_scope();
                } else if let Some(else_branch) = &if_stmt.else_body {
                    match else_branch {
                        crate::ast::ElseBranch::Else(block) => {
                            self.context.push_scope();
                            for stmt in &block.statements {
                                self.eval_stmt(stmt)?;
                            }
                            self.context.pop_scope();
                        }
                        crate::ast::ElseBranch::ElseIf(nested_if) => {
                            self.eval_stmt(&Stmt::If(*nested_if.clone()))?;
                        }
                    }
                }
                Ok(ComptimeValue::Undefined)
            }

            Stmt::For(for_stmt) => {
                let values = self.eval_for_range(&for_stmt.range)?;

                self.context.push_scope();

                // Get binding name from captures
                let binding = for_stmt.captures.first();

                for value in values {
                    if let Some(name) = binding {
                        self.context.define(name, value);
                    }
                    for stmt in &for_stmt.body.statements {
                        self.eval_stmt(stmt)?;
                    }
                }

                self.context.pop_scope();
                Ok(ComptimeValue::Undefined)
            }

            Stmt::Block(block) => {
                self.context.push_scope();
                for stmt in &block.statements {
                    self.eval_stmt(stmt)?;
                }
                // Evaluate trailing expression if present
                let result = if let Some(trailing) = &block.trailing_expr {
                    self.eval_expr(trailing)?
                } else {
                    ComptimeValue::Unit
                };
                self.context.pop_scope();
                Ok(result)
            }

            Stmt::Defer(_) => Err(ComptimeError {
                message: "defer not supported at comptime".to_string(),
            }),

            Stmt::Errdefer(_) => Err(ComptimeError {
                message: "errdefer not supported at comptime".to_string(),
            }),

            Stmt::Break(_) => Err(ComptimeError {
                message: "break not yet supported at comptime".to_string(),
            }),

            Stmt::Continue(_) => Err(ComptimeError {
                message: "continue not yet supported at comptime".to_string(),
            }),

            Stmt::Switch(_) => Err(ComptimeError {
                message: "switch not yet supported at comptime".to_string(),
            }),

            Stmt::Tick(_) => Err(ComptimeError {
                message: "tick blocks not supported at comptime".to_string(),
            }),

            Stmt::TryBlock(_) => Err(ComptimeError {
                message: "try blocks not supported at comptime".to_string(),
            }),

            Stmt::Gate(_) => Err(ComptimeError {
                message: "quantum operations not supported at comptime".to_string(),
            }),

            Stmt::Prepare(_) => Err(ComptimeError {
                message: "prepare operations not supported at comptime".to_string(),
            }),

            Stmt::Measure(_) => Err(ComptimeError {
                message: "measurement operations not supported at comptime".to_string(),
            }),

            Stmt::Barrier(_) => Err(ComptimeError {
                message: "barrier operations not supported at comptime".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comptime_value_display() {
        assert_eq!(ComptimeValue::Int(42).to_string(), "42");
        assert_eq!(ComptimeValue::Bool(true).to_string(), "true");
        assert_eq!(ComptimeValue::Null.to_string(), "null");
    }

    #[test]
    fn test_comptime_arithmetic() {
        let eval = ComptimeEvaluator::new();

        let a = ComptimeValue::Int(10);
        let b = ComptimeValue::Int(3);

        assert_eq!(
            eval.eval_binary_op(BinaryOp::Add, &a, &b).unwrap(),
            ComptimeValue::Int(13)
        );
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Sub, &a, &b).unwrap(),
            ComptimeValue::Int(7)
        );
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Mul, &a, &b).unwrap(),
            ComptimeValue::Int(30)
        );
        // 10/3 is not exact, so returns Rational (prevents subtle bugs like 1/4 turns = 0)
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Div, &a, &b).unwrap(),
            ComptimeValue::Rational(Rational::new(10, 3))
        );
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Mod, &a, &b).unwrap(),
            ComptimeValue::Int(1)
        );
    }

    #[test]
    fn test_comptime_comparison() {
        let eval = ComptimeEvaluator::new();

        let a = ComptimeValue::Int(10);
        let b = ComptimeValue::Int(3);

        assert_eq!(
            eval.eval_binary_op(BinaryOp::Lt, &a, &b).unwrap(),
            ComptimeValue::Bool(false)
        );
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Gt, &a, &b).unwrap(),
            ComptimeValue::Bool(true)
        );
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Eq, &a, &b).unwrap(),
            ComptimeValue::Bool(false)
        );
    }

    #[test]
    fn test_comptime_context() {
        let mut ctx = ComptimeContext::new();

        ctx.define("x", ComptimeValue::Int(42));
        assert_eq!(ctx.lookup("x"), Some(&ComptimeValue::Int(42)));

        ctx.push_scope();
        ctx.define("y", ComptimeValue::Int(10));
        assert_eq!(ctx.lookup("x"), Some(&ComptimeValue::Int(42)));
        assert_eq!(ctx.lookup("y"), Some(&ComptimeValue::Int(10)));

        ctx.pop_scope();
        assert_eq!(ctx.lookup("x"), Some(&ComptimeValue::Int(42)));
        assert_eq!(ctx.lookup("y"), None);
    }

    #[test]
    fn test_comptime_unary() {
        let eval = ComptimeEvaluator::new();

        let a = ComptimeValue::Int(42);
        assert_eq!(
            eval.eval_unary_op(UnaryOp::Neg, &a).unwrap(),
            ComptimeValue::Int(-42)
        );

        let b = ComptimeValue::Bool(true);
        assert_eq!(
            eval.eval_unary_op(UnaryOp::Not, &b).unwrap(),
            ComptimeValue::Bool(false)
        );
    }

    #[test]
    fn test_comptime_division_by_zero() {
        let eval = ComptimeEvaluator::new();

        let a = ComptimeValue::Int(10);
        let b = ComptimeValue::Int(0);

        let result = eval.eval_binary_op(BinaryOp::Div, &a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_comptime_orelse() {
        let eval = ComptimeEvaluator::new();

        let null = ComptimeValue::Null;
        let fallback = ComptimeValue::Int(42);

        assert_eq!(
            eval.eval_binary_op(BinaryOp::Orelse, &null, &fallback)
                .unwrap(),
            ComptimeValue::Int(42)
        );

        let some = ComptimeValue::Int(10);
        assert_eq!(
            eval.eval_binary_op(BinaryOp::Orelse, &some, &fallback)
                .unwrap(),
            ComptimeValue::Int(10)
        );
    }

    #[test]
    fn test_comptime_sizeof() {
        use crate::ast::{BuiltinExpr, Ident};

        let mut eval = ComptimeEvaluator::new();

        // Test @sizeOf(u8)
        let builtin = BuiltinExpr {
            name: "sizeOf".to_string(),
            args: vec![Expr::Ident(Ident {
                name: "u8".to_string(),
                location: None,
            })],
            location: None,
        };
        let result = eval.eval_builtin(&builtin).unwrap();
        assert_eq!(result, ComptimeValue::Uint(1));

        // Test @sizeOf(u32)
        let builtin = BuiltinExpr {
            name: "sizeOf".to_string(),
            args: vec![Expr::Ident(Ident {
                name: "u32".to_string(),
                location: None,
            })],
            location: None,
        };
        let result = eval.eval_builtin(&builtin).unwrap();
        assert_eq!(result, ComptimeValue::Uint(4));

        // Test @sizeOf(u64)
        let builtin = BuiltinExpr {
            name: "sizeOf".to_string(),
            args: vec![Expr::Ident(Ident {
                name: "u64".to_string(),
                location: None,
            })],
            location: None,
        };
        let result = eval.eval_builtin(&builtin).unwrap();
        assert_eq!(result, ComptimeValue::Uint(8));
    }

    #[test]
    fn test_comptime_typename() {
        use crate::ast::{BuiltinExpr, Ident};

        let mut eval = ComptimeEvaluator::new();

        let builtin = BuiltinExpr {
            name: "typeName".to_string(),
            args: vec![Expr::Ident(Ident {
                name: "i32".to_string(),
                location: None,
            })],
            location: None,
        };
        let result = eval.eval_builtin(&builtin).unwrap();
        assert_eq!(result, ComptimeValue::String("i32".to_string()));
    }

    #[test]
    fn test_comptime_to_usize() {
        assert_eq!(ComptimeValue::Int(42).to_usize(), Some(42));
        assert_eq!(ComptimeValue::Uint(100).to_usize(), Some(100));
        assert_eq!(ComptimeValue::Bool(true).to_usize(), None);
        assert_eq!(ComptimeValue::Null.to_usize(), None);
    }

    #[test]
    fn test_comptime_type_values() {
        let eval = ComptimeEvaluator::new();

        // Test that built-in type names resolve to Type values
        assert!(eval.resolve_builtin_type("u8").is_some());
        assert!(eval.resolve_builtin_type("u32").is_some());
        assert!(eval.resolve_builtin_type("bool").is_some());
        assert!(eval.resolve_builtin_type("unknown_type").is_none());
    }

    #[test]
    fn test_comptime_function_value() {
        use crate::ast::{Block, Param, TypeExpr};

        // Create a simple comptime function that returns a type
        let func = FnDecl {
            name: "makeArray".to_string(),
            params: vec![Param {
                name: "T".to_string(),
                ty: TypeExpr::Type,
                is_comptime: true,
                location: None,
            }],
            return_type: Some(TypeExpr::Type),
            body: Block {
                label: None,
                attrs: vec![],
                statements: vec![],
                trailing_expr: Some(Box::new(Expr::Ident(crate::ast::Ident {
                    name: "T".to_string(),
                    location: None,
                }))),
                location: None,
            },
            is_pub: false,
            is_inline: false,
            error_mode: None,
            doc_comment: None,
            location: None,
        };

        let func_value = ComptimeValue::Function(Box::new(func));
        assert!(matches!(func_value.get_type(), Type::Type));
    }

    #[test]
    fn test_comptime_anon_struct() {
        use crate::ast::{AnonStructExpr, PrimitiveType, StructField, TypeExpr};

        let mut eval = ComptimeEvaluator::new();

        // Create anonymous struct expression: struct { x: u32, y: u32 }
        let anon = AnonStructExpr {
            fields: vec![
                StructField {
                    name: "x".to_string(),
                    ty: TypeExpr::Primitive(PrimitiveType::UInt { bits: 32 }),
                    default: None,
                    doc_comment: None,
                    location: None,
                },
                StructField {
                    name: "y".to_string(),
                    ty: TypeExpr::Primitive(PrimitiveType::UInt { bits: 32 }),
                    default: None,
                    doc_comment: None,
                    location: None,
                },
            ],
            is_packed: false,
            location: None,
        };

        let result = eval.eval_expr(&Expr::AnonStruct(Box::new(anon))).unwrap();

        // Verify it's a Type value with a Struct type
        if let ComptimeValue::Type(ty) = result {
            if let Type::Struct { fields, .. } = ty {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "x");
                assert_eq!(fields[1].0, "y");
            } else {
                panic!("Expected Struct type");
            }
        } else {
            panic!("Expected Type value");
        }
    }

    #[test]
    fn test_comptime_function_call() {
        use crate::ast::{Block, CallExpr, Ident, Param, TypeExpr};

        let mut eval = ComptimeEvaluator::new();

        // Create a function: fn(comptime T: type) -> type { T }
        // This is an identity function for types
        let func = FnDecl {
            name: "identity".to_string(),
            params: vec![Param {
                name: "T".to_string(),
                ty: TypeExpr::Type,
                is_comptime: true,
                location: None,
            }],
            return_type: Some(TypeExpr::Type),
            body: Block {
                label: None,
                attrs: vec![],
                statements: vec![],
                trailing_expr: Some(Box::new(Expr::Ident(Ident {
                    name: "T".to_string(),
                    location: None,
                }))),
                location: None,
            },
            is_pub: false,
            is_inline: false,
            error_mode: None,
            doc_comment: None,
            location: None,
        };

        // Store function in context
        eval.context
            .define("identity", ComptimeValue::Function(Box::new(func)));

        // Call identity(u32)
        let call = CallExpr {
            callee: Expr::Ident(Ident {
                name: "identity".to_string(),
                location: None,
            }),
            args: vec![Expr::Ident(Ident {
                name: "u32".to_string(),
                location: None,
            })],
            location: None,
        };

        let result = eval.eval_expr(&Expr::Call(Box::new(call))).unwrap();

        // The result should be the u32 type
        if let ComptimeValue::Type(ty) = result {
            assert!(matches!(ty, Type::UInt { bits } if bits == BitWidth::BITS_32));
        } else {
            panic!("Expected Type value, got {:?}", result);
        }
    }

    /// Test the full pattern: inline for + comptime function calls for type construction.
    /// This is the key pattern that replaces recursion for building nested types.
    ///
    /// Simulates:
    /// ```zlup
    /// WrapArray := fn(comptime Inner: type) -> type {
    ///     struct { data: [7]Inner }
    /// };
    ///
    /// Code := comptime {
    ///     mut T := u8;
    ///     inline for _ in 0..3 {
    ///         T = WrapArray(T);
    ///     }
    ///     T
    /// };
    /// ```
    #[test]
    fn test_inline_for_with_comptime_function_nested_types() {
        use crate::ast::{
            AnonStructExpr, ArrayType, Block, CallExpr, Ident, IntLit, Param, StructField, TypeExpr,
        };

        let mut eval = ComptimeEvaluator::new();

        // Create WrapArray function: fn(comptime Inner: type) -> type { struct { data: [7]Inner } }
        // The function body returns an anonymous struct with a field `data: [7]Inner`
        let wrap_array_func = FnDecl {
            name: "WrapArray".to_string(),
            params: vec![Param {
                name: "Inner".to_string(),
                ty: TypeExpr::Type,
                is_comptime: true,
                location: None,
            }],
            return_type: Some(TypeExpr::Type),
            body: Block {
                label: None,
                attrs: vec![],
                statements: vec![],
                // Return: struct { data: [7]Inner }
                trailing_expr: Some(Box::new(Expr::AnonStruct(Box::new(AnonStructExpr {
                    fields: vec![StructField {
                        name: "data".to_string(),
                        ty: TypeExpr::Array(Box::new(ArrayType {
                            element: TypeExpr::Named(crate::ast::TypePath {
                                segments: vec!["Inner".to_string()],
                                location: None,
                            }),
                            size: Some(Expr::IntLit(IntLit {
                                value: 7,
                                suffix: None,
                                location: None,
                            })),
                            sentinel: None,
                        })),
                        default: None,
                        doc_comment: None,
                        location: None,
                    }],
                    is_packed: false,
                    location: None,
                })))),
                location: None,
            },
            is_pub: false,
            is_inline: false,
            error_mode: None,
            doc_comment: None,
            location: None,
        };

        // Store function in context
        eval.context.define(
            "WrapArray",
            ComptimeValue::Function(Box::new(wrap_array_func)),
        );

        // Now simulate the comptime block:
        // comptime {
        //     mut T := u8;
        //     inline for _ in 0..3 { T = WrapArray(T); }
        //     T
        // }

        // Step 1: mut T := u8;
        eval.context.define(
            "T",
            ComptimeValue::Type(Type::UInt {
                bits: BitWidth::BITS_8,
            }),
        );

        // Step 2: Simulate inline for with 3 iterations
        for _ in 0..3 {
            // T = WrapArray(T)
            let call = CallExpr {
                callee: Expr::Ident(Ident {
                    name: "WrapArray".to_string(),
                    location: None,
                }),
                args: vec![Expr::Ident(Ident {
                    name: "T".to_string(),
                    location: None,
                })],
                location: None,
            };

            let new_type = eval.eval_expr(&Expr::Call(Box::new(call))).unwrap();
            eval.context.update("T", new_type);
        }

        // Step 3: Get final T
        let result = eval.context.lookup("T").unwrap().clone();

        // Verify the structure: struct { data: [7]struct { data: [7]struct { data: [7]u8 } } }
        if let ComptimeValue::Type(ty) = result {
            // Level 1: struct { data: [7]... }
            if let Type::Struct { fields, .. } = &ty {
                assert_eq!(fields.len(), 1, "Expected 1 field at level 1");
                assert_eq!(fields[0].0, "data", "Field name should be 'data'");

                // Level 1 field type: [7]struct { ... }
                if let Type::Array {
                    element: level2,
                    size,
                } = &fields[0].1
                {
                    assert_eq!(*size, Some(7), "Array size should be 7 at level 1");

                    // Level 2: struct { data: [7]... }
                    if let Type::Struct {
                        fields: fields2, ..
                    } = level2.as_ref()
                    {
                        assert_eq!(fields2.len(), 1, "Expected 1 field at level 2");

                        // Level 2 field type: [7]struct { ... }
                        if let Type::Array {
                            element: level3,
                            size: size2,
                        } = &fields2[0].1
                        {
                            assert_eq!(*size2, Some(7), "Array size should be 7 at level 2");

                            // Level 3: struct { data: [7]u8 }
                            if let Type::Struct {
                                fields: fields3, ..
                            } = level3.as_ref()
                            {
                                assert_eq!(fields3.len(), 1, "Expected 1 field at level 3");

                                // Level 3 field type: [7]u8
                                if let Type::Array {
                                    element: inner,
                                    size: size3,
                                } = &fields3[0].1
                                {
                                    assert_eq!(
                                        *size3,
                                        Some(7),
                                        "Array size should be 7 at level 3"
                                    );
                                    assert!(
                                        matches!(inner.as_ref(), Type::UInt { bits } if *bits == BitWidth::BITS_8),
                                        "Innermost type should be u8, got {:?}",
                                        inner
                                    );
                                } else {
                                    panic!("Level 3 field should be array, got {:?}", fields3[0].1);
                                }
                            } else {
                                panic!("Level 3 should be struct, got {:?}", level3);
                            }
                        } else {
                            panic!("Level 2 field should be array, got {:?}", fields2[0].1);
                        }
                    } else {
                        panic!("Level 2 should be struct, got {:?}", level2);
                    }
                } else {
                    panic!("Level 1 field should be array, got {:?}", fields[0].1);
                }
            } else {
                panic!("Result should be struct, got {:?}", ty);
            }
        } else {
            panic!("Expected Type value, got {:?}", result);
        }
    }

    /// Test that inline for loop in comptime block works end-to-end.
    /// Uses the actual for loop evaluation, not manual simulation.
    #[test]
    fn test_comptime_block_with_inline_for() {
        use crate::ast::{AssignStmt, Block, ForRange, ForStmt, Ident, IntLit, Stmt};

        let mut eval = ComptimeEvaluator::new();

        // Simulate:
        // comptime {
        //     mut sum := 0;
        //     for i in 0..5 { sum = sum + i; }
        //     sum
        // }

        // Create the for loop
        let for_stmt = ForStmt {
            captures: vec!["i".to_string()],
            range: ForRange::Range {
                start: Expr::IntLit(IntLit {
                    value: 0,
                    suffix: None,
                    location: None,
                }),
                end: Expr::IntLit(IntLit {
                    value: 5,
                    suffix: None,
                    location: None,
                }),
            },
            body: Block {
                label: None,
                attrs: vec![],
                statements: vec![
                    // sum = sum + i
                    Stmt::Assign(AssignStmt {
                        target: Expr::Ident(Ident {
                            name: "sum".to_string(),
                            location: None,
                        }),
                        op: crate::ast::AssignOp::Assign,
                        value: Expr::Binary(Box::new(crate::ast::BinaryExpr {
                            left: Expr::Ident(Ident {
                                name: "sum".to_string(),
                                location: None,
                            }),
                            op: BinaryOp::Add,
                            right: Expr::Ident(Ident {
                                name: "i".to_string(),
                                location: None,
                            }),
                            location: None,
                        })),
                        location: None,
                    }),
                ],
                trailing_expr: None,
                location: None,
            },
            is_inline: true,
            label: None,
            location: None,
        };

        // Initialize sum
        eval.context.define("sum", ComptimeValue::Int(0));

        // Execute the for loop
        eval.eval_stmt(&Stmt::For(for_stmt)).unwrap();

        // Check result: 0 + 1 + 2 + 3 + 4 = 10
        let result = eval.context.lookup("sum").unwrap();
        assert_eq!(*result, ComptimeValue::Int(10), "Sum should be 10");
    }

    #[test]
    fn test_fraction_division_returns_rational() {
        // Integer division that's not exact should return Rational
        // This is critical for angle expressions like `1/4 turns`
        let eval = ComptimeEvaluator::new();

        // 1/4 should be Rational(1/4), not 0 or Float
        let one = ComptimeValue::Int(1);
        let four = ComptimeValue::Int(4);
        let result = eval.eval_div(&one, &four).unwrap();
        assert_eq!(
            result,
            ComptimeValue::Rational(Rational::new(1, 4)),
            "1/4 should be Rational(1/4)"
        );
        // Verify it converts to correct float
        assert_eq!(result.as_float(), Some(0.25), "1/4 as float should be 0.25");

        // 1/8 should be Rational(1/8)
        let eight = ComptimeValue::Int(8);
        let result = eval.eval_div(&one, &eight).unwrap();
        assert_eq!(
            result,
            ComptimeValue::Rational(Rational::new(1, 8)),
            "1/8 should be Rational(1/8)"
        );
        assert_eq!(
            result.as_float(),
            Some(0.125),
            "1/8 as float should be 0.125"
        );

        // 4/2 is exact, should return Int
        let two = ComptimeValue::Int(2);
        let result = eval.eval_div(&four, &two).unwrap();
        assert_eq!(result, ComptimeValue::Int(2), "4/2 should be Int(2)");

        // 10/5 is exact, should return Int
        let ten = ComptimeValue::Int(10);
        let five = ComptimeValue::Int(5);
        let result = eval.eval_div(&ten, &five).unwrap();
        assert_eq!(result, ComptimeValue::Int(2), "10/5 should be Int(2)");
    }

    #[test]
    fn test_rational_arithmetic() {
        let eval = ComptimeEvaluator::new();

        // 1/4 + 1/4 = 1/2
        let quarter = ComptimeValue::Rational(Rational::new(1, 4));
        let result = eval.eval_add(&quarter, &quarter).unwrap();
        assert_eq!(result, ComptimeValue::Rational(Rational::new(1, 2)));

        // 1/2 - 1/4 = 1/4
        let half = ComptimeValue::Rational(Rational::new(1, 2));
        let result = eval.eval_sub(&half, &quarter).unwrap();
        assert_eq!(result, ComptimeValue::Rational(Rational::new(1, 4)));

        // 1/4 * 2 = 1/2
        let two = ComptimeValue::Int(2);
        let result = eval.eval_mul(&quarter, &two).unwrap();
        assert_eq!(result, ComptimeValue::Rational(Rational::new(1, 2)));

        // 1/2 / 2 = 1/4
        let result = eval.eval_div(&half, &two).unwrap();
        assert_eq!(result, ComptimeValue::Rational(Rational::new(1, 4)));

        // 1/3 + 1/3 + 1/3 = 1
        let third = ComptimeValue::Rational(Rational::new(1, 3));
        let two_thirds = eval.eval_add(&third, &third).unwrap();
        let result = eval.eval_add(&two_thirds, &third).unwrap();
        assert_eq!(result, ComptimeValue::Rational(Rational::new(1, 1)));
    }

    #[test]
    fn test_rational_comparison() {
        let eval = ComptimeEvaluator::new();

        let quarter = ComptimeValue::Rational(Rational::new(1, 4));
        let half = ComptimeValue::Rational(Rational::new(1, 2));
        let one = ComptimeValue::Int(1);

        // 1/4 < 1/2
        assert_eq!(
            eval.eval_lt(&quarter, &half).unwrap(),
            ComptimeValue::Bool(true)
        );

        // 1/4 < 1
        assert_eq!(
            eval.eval_lt(&quarter, &one).unwrap(),
            ComptimeValue::Bool(true)
        );

        // 1/2 > 1/4
        assert_eq!(
            eval.eval_gt(&half, &quarter).unwrap(),
            ComptimeValue::Bool(true)
        );
    }

    #[test]
    fn test_rational_negation() {
        let eval = ComptimeEvaluator::new();

        let quarter = ComptimeValue::Rational(Rational::new(1, 4));
        let result = eval.eval_unary_op(UnaryOp::Neg, &quarter).unwrap();
        assert_eq!(result, ComptimeValue::Rational(Rational::new(-1, 4)));
    }

    // =========================================================================
    // Advanced Builtin Tests (@type_info, @field_names, @enum_fields)
    // =========================================================================

    #[test]
    fn test_type_info_kind() {
        use crate::semantic::Type;

        // Test primitive types
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Bool),
            TypeInfoKind::Primitive
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::UInt {
                bits: BitWidth::must(32)
            }),
            TypeInfoKind::Primitive
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::F64),
            TypeInfoKind::Primitive
        );

        // Test compound types
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Array {
                element: Box::new(Type::Bool),
                size: Some(4)
            }),
            TypeInfoKind::Array
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Slice {
                element: Box::new(Type::Bool)
            }),
            TypeInfoKind::Slice
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Optional {
                inner: Box::new(Type::Bool)
            }),
            TypeInfoKind::Optional
        );

        // Test user-defined types
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Struct {
                name: "Point".to_string(),
                fields: vec![("x".to_string(), Type::F64), ("y".to_string(), Type::F64),],
            }),
            TypeInfoKind::Struct
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Enum {
                name: "Color".to_string(),
                variants: vec!["Red".to_string(), "Green".to_string(), "Blue".to_string()],
            }),
            TypeInfoKind::Enum
        );

        // Test special types
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Unit),
            TypeInfoKind::Unit
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Never),
            TypeInfoKind::Never
        );
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Type),
            TypeInfoKind::Type
        );

        // Test quantum types
        assert_eq!(
            ComptimeEvaluator::get_type_info_kind(&Type::Qubit),
            TypeInfoKind::Quantum
        );
    }

    #[test]
    fn test_type_info_kind_as_str() {
        assert_eq!(TypeInfoKind::Primitive.as_str(), "primitive");
        assert_eq!(TypeInfoKind::Array.as_str(), "array");
        assert_eq!(TypeInfoKind::Struct.as_str(), "struct");
        assert_eq!(TypeInfoKind::Enum.as_str(), "enum");
        assert_eq!(TypeInfoKind::Optional.as_str(), "optional");
        assert_eq!(TypeInfoKind::Quantum.as_str(), "quantum");
    }

    #[test]
    fn test_resolve_type_from_expr_primitives() {
        use crate::ast::{Ident, SourceLocation};

        let eval = ComptimeEvaluator::new();

        // Test primitive type resolution
        let bool_expr = Expr::Ident(Ident {
            name: "bool".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.resolve_type_from_expr(&bool_expr).unwrap();
        assert_eq!(result, Type::Bool);

        let u32_expr = Expr::Ident(Ident {
            name: "u32".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.resolve_type_from_expr(&u32_expr).unwrap();
        assert_eq!(
            result,
            Type::UInt {
                bits: BitWidth::must(32)
            }
        );

        let f64_expr = Expr::Ident(Ident {
            name: "f64".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.resolve_type_from_expr(&f64_expr).unwrap();
        assert_eq!(result, Type::F64);
    }

    #[test]
    fn test_eval_type_info_primitive() {
        use crate::ast::{Ident, SourceLocation};

        let eval = ComptimeEvaluator::new();

        let u32_expr = Expr::Ident(Ident {
            name: "u32".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.eval_type_info(&u32_expr).unwrap();

        // Should be a struct with kind and name
        if let ComptimeValue::Struct { name, fields } = result {
            assert_eq!(name, "TypeInfo");
            assert_eq!(
                fields.get("kind"),
                Some(&ComptimeValue::String("primitive".to_string()))
            );
            // Name should be "u32"
            if let Some(ComptimeValue::String(type_name)) = fields.get("name") {
                assert!(type_name.contains("u32") || type_name.contains("UInt"));
            }
        } else {
            panic!("Expected TypeInfo struct");
        }
    }

    #[test]
    fn test_eval_field_names_struct() {
        use crate::ast::{Ident, SourceLocation};

        let mut eval = ComptimeEvaluator::new();

        // Register a struct type in context
        eval.context.define(
            "Point",
            ComptimeValue::Type(Type::Struct {
                name: "Point".to_string(),
                fields: vec![
                    ("x".to_string(), Type::F64),
                    ("y".to_string(), Type::F64),
                    ("z".to_string(), Type::F64),
                ],
            }),
        );

        let point_expr = Expr::Ident(Ident {
            name: "Point".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.eval_field_names(&point_expr).unwrap();

        // Should be an array of strings
        if let ComptimeValue::Array(names) = result {
            assert_eq!(names.len(), 3);
            assert_eq!(names[0], ComptimeValue::String("x".to_string()));
            assert_eq!(names[1], ComptimeValue::String("y".to_string()));
            assert_eq!(names[2], ComptimeValue::String("z".to_string()));
        } else {
            panic!("Expected array of field names");
        }
    }

    #[test]
    fn test_eval_enum_fields() {
        use crate::ast::{Ident, SourceLocation};

        let mut eval = ComptimeEvaluator::new();

        // Register an enum type in context
        eval.context.define(
            "Color",
            ComptimeValue::Type(Type::Enum {
                name: "Color".to_string(),
                variants: vec!["Red".to_string(), "Green".to_string(), "Blue".to_string()],
            }),
        );

        let color_expr = Expr::Ident(Ident {
            name: "Color".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.eval_enum_fields(&color_expr).unwrap();

        // Should be an array of strings
        if let ComptimeValue::Array(variants) = result {
            assert_eq!(variants.len(), 3);
            assert_eq!(variants[0], ComptimeValue::String("Red".to_string()));
            assert_eq!(variants[1], ComptimeValue::String("Green".to_string()));
            assert_eq!(variants[2], ComptimeValue::String("Blue".to_string()));
        } else {
            panic!("Expected array of enum variants");
        }
    }

    #[test]
    fn test_eval_type_from_info() {
        use crate::ast::{Ident, SourceLocation};

        let mut eval = ComptimeEvaluator::new();

        // Create a TypeInfo struct for an array type
        let mut fields = BTreeMap::new();
        fields.insert(
            "kind".to_string(),
            ComptimeValue::String("array".to_string()),
        );
        fields.insert(
            "name".to_string(),
            ComptimeValue::String("[4]u32".to_string()),
        );
        fields.insert(
            "element".to_string(),
            ComptimeValue::Type(Type::UInt {
                bits: BitWidth::must(32),
            }),
        );
        fields.insert("size".to_string(), ComptimeValue::Uint(4));

        // Store the TypeInfo in context
        eval.context.define(
            "my_info",
            ComptimeValue::Struct {
                name: "TypeInfo".to_string(),
                fields,
            },
        );

        let info_expr = Expr::Ident(Ident {
            name: "my_info".to_string(),
            location: Some(SourceLocation::default()),
        });
        let result = eval.eval_type_from_info(&info_expr).unwrap();

        // Should be an array type
        if let ComptimeValue::Type(Type::Array { element, size }) = result {
            assert_eq!(
                *element,
                Type::UInt {
                    bits: BitWidth::must(32)
                }
            );
            assert_eq!(size, Some(4));
        } else {
            panic!("Expected array type, got {:?}", result);
        }
    }

    /// Test that comptime function memoization works correctly.
    /// Calling the same function with the same args should return cached result.
    #[test]
    fn test_comptime_memoization() {
        use crate::ast::{Block, FnDecl, Ident, IntLit, Param};

        let mut eval = ComptimeEvaluator::new();

        // Define a simple function: fn add_10(n: i32) -> i32 { n + 10 }
        let add_10_func = FnDecl {
            name: "add_10".to_string(),
            params: vec![Param {
                name: "n".to_string(),
                ty: TypeExpr::Named(crate::ast::TypePath {
                    segments: vec!["i32".to_string()],
                    location: None,
                }),
                is_comptime: true,
                location: None,
            }],
            return_type: Some(TypeExpr::Named(crate::ast::TypePath {
                segments: vec!["i32".to_string()],
                location: None,
            })),
            body: Block {
                label: None,
                attrs: vec![],
                statements: vec![],
                trailing_expr: Some(Box::new(Expr::Binary(Box::new(crate::ast::BinaryExpr {
                    left: Expr::Ident(Ident {
                        name: "n".to_string(),
                        location: None,
                    }),
                    op: BinaryOp::Add,
                    right: Expr::IntLit(IntLit {
                        value: 10,
                        suffix: None,
                        location: None,
                    }),
                    location: None,
                })))),
                location: None,
            },
            is_pub: false,
            is_inline: false,
            error_mode: None,
            doc_comment: None,
            location: None,
        };

        eval.context
            .define("add_10", ComptimeValue::Function(Box::new(add_10_func)));

        // Call the function with argument 5
        let call1 = crate::ast::CallExpr {
            callee: Expr::Ident(Ident {
                name: "add_10".to_string(),
                location: None,
            }),
            args: vec![Expr::IntLit(IntLit {
                value: 5,
                suffix: None,
                location: None,
            })],
            location: None,
        };

        // First call - should compute and cache
        let result1 = eval
            .eval_expr(&Expr::Call(Box::new(call1.clone())))
            .unwrap();
        assert_eq!(result1, ComptimeValue::Int(15));

        // Verify it's in the cache
        let cache_key = (
            "add_10".to_string(),
            ComptimeEvaluator::serialize_args_for_cache(&[ComptimeValue::Int(5)]),
        );
        assert!(
            eval.memo_cache.contains_key(&cache_key),
            "Result should be cached after first call"
        );

        // Second call with same args - should return cached value
        let result2 = eval.eval_expr(&Expr::Call(Box::new(call1))).unwrap();
        assert_eq!(result2, ComptimeValue::Int(15));

        // Call with different args - should compute new result
        let call2 = crate::ast::CallExpr {
            callee: Expr::Ident(Ident {
                name: "add_10".to_string(),
                location: None,
            }),
            args: vec![Expr::IntLit(IntLit {
                value: 20,
                suffix: None,
                location: None,
            })],
            location: None,
        };
        let result3 = eval.eval_expr(&Expr::Call(Box::new(call2))).unwrap();
        assert_eq!(result3, ComptimeValue::Int(30));

        // Both entries should be in cache
        assert_eq!(eval.memo_cache.len(), 2);
    }
}
