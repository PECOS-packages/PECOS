/*!
Type system for PHIR

This module defines the complete type system used throughout PECOS PHIR, including:
- Quantum types (qubits, quantum registers)
- Classical types (integers, floats, booleans)
- Composite types (arrays, tuples, functions)
- QEC-aware types (logical qubits, syndrome data)
- Extension types for custom dialects
*/

use std::collections::BTreeMap;

/// Core type system used by PHIR
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Type {
    // ===== Quantum Types =====
    /// Single qubit
    Qubit,
    /// Quantum register of specified size
    QuantumReg(usize),

    // ===== Classical Types =====
    /// Single bit
    Bit,
    /// Boolean value
    Bool,
    /// Signed integer with specified width
    Int(IntWidth),
    /// Unsigned integer with specified width
    UInt(IntWidth),
    /// Floating point with specified precision
    Float(FloatPrecision),
    /// UTF-8 string
    String,

    // ===== Composite Types =====
    /// Array of elements of the same type
    Array(Box<Type>, ArraySize),
    /// Tuple of heterogeneous types
    Tuple(Vec<Type>),
    /// Function signature
    Function(FunctionType),
    /// Optional/nullable type
    Optional(Box<Type>),

    // ===== Memory Types =====
    /// Reference/pointer to a type
    Ref(Box<Type>),
    /// Mutable reference
    MutRef(Box<Type>),

    // ===== Extension Types =====
    /// Custom types from dialects
    Custom(CustomType),

    // ===== Special Types =====
    /// Unit type (no value)
    Unit,
    /// Bottom type (never returns)
    Never,
    /// Unknown/inferred type
    Unknown,
    /// Future type (for lazy measurements)
    Future,
}

/// Integer bit widths
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum IntWidth {
    /// 8-bit integer
    I8,
    /// 16-bit integer
    I16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 128-bit integer
    I128,
    /// Pointer-sized integer
    ISize,
    /// Custom width
    Custom(u32),
}

/// Floating point precisions
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FloatPrecision {
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// 128-bit float
    F128,
    /// Custom precision
    Custom(u32),
}

/// Array size specification
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ArraySize {
    /// Statically known size
    Fixed(usize),
    /// Dynamically determined size
    Dynamic,
    /// Size determined by type parameter
    Parametric(String),
}

/// Function type signature
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default)]
pub struct FunctionType {
    /// Input parameter types
    pub inputs: Vec<Type>,
    /// Output/return types
    pub outputs: Vec<Type>,
    /// Whether function is variadic
    pub variadic: bool,
}

/// Custom type from dialect extension
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// Dialect namespace
    pub dialect: String,
    /// Type name within dialect
    pub name: String,
    /// Type parameters
    pub parameters: Vec<TypeParameter>,
}

/// Type parameters for generic/parametric types
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TypeParameter {
    /// Type parameter
    Type(Type),
    /// Integer parameter
    Int(i64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Bool(bool),
}

/// Ordered float wrapper for hashing/equality
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OrderedFloat(pub u64); // Bit representation of f64

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat(f.to_bits())
    }
}

impl From<OrderedFloat> for f64 {
    fn from(of: OrderedFloat) -> Self {
        f64::from_bits(of.0)
    }
}

/// Type registry for managing custom types from dialects
pub struct TypeRegistry {
    /// Registered custom types
    custom_types: BTreeMap<String, CustomTypeDefinition>,
    /// Type aliases
    aliases: BTreeMap<String, Type>,
}

/// Definition of a custom type
pub struct CustomTypeDefinition {
    /// Full type name (dialect.name)
    pub full_name: String,
    /// Type parameters
    pub parameters: Vec<TypeParameterDef>,
    /// Size in bytes (if known)
    pub size: Option<usize>,
    /// Alignment requirements
    pub alignment: Option<usize>,
    /// Whether type is copyable
    pub copyable: bool,
}

/// Type parameter definition
pub struct TypeParameterDef {
    pub name: String,
    pub kind: TypeParameterKind,
    pub default: Option<TypeParameter>,
}

/// Kind of type parameter
#[derive(Clone, Debug, PartialEq)]
pub enum TypeParameterKind {
    Type,
    IntValue,
    StringValue,
    BoolValue,
}

impl Type {
    /// Get the size of this type in bytes (if statically known)
    #[must_use]
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            Type::Bit | Type::Bool => Some(1),
            Type::Int(width) | Type::UInt(width) => Some(width.bytes()),
            Type::Float(precision) => Some(precision.bytes()),
            Type::Qubit | Type::Ref(_) | Type::MutRef(_) => Some(8), // 64-bit pointers/quantum state
            Type::QuantumReg(n) => Some(8 * (1 << n)),               // Exponential state space
            Type::Array(elem_type, ArraySize::Fixed(n)) => {
                elem_type.size_bytes().map(|elem_size| elem_size * n)
            }
            Type::Tuple(types) => types
                .iter()
                .map(Type::size_bytes)
                .collect::<Option<Vec<_>>>()
                .map(|sizes| sizes.iter().sum()),
            Type::Unit => Some(0),
            _ => None, // Unknown or dynamic size
        }
    }

    /// Check if this type is quantum (contains qubits)
    #[must_use]
    pub fn is_quantum(&self) -> bool {
        match self {
            Type::Qubit | Type::QuantumReg(_) => true,
            Type::Array(elem_type, _) => elem_type.is_quantum(),
            Type::Tuple(types) => types.iter().any(Type::is_quantum),
            Type::Optional(inner) | Type::Ref(inner) | Type::MutRef(inner) => inner.is_quantum(),
            _ => false,
        }
    }

    /// Check if this type is classical (no quantum components)
    #[must_use]
    pub fn is_classical(&self) -> bool {
        !self.is_quantum()
    }

    /// Check if this type is copyable (can be duplicated)
    #[must_use]
    pub fn is_copyable(&self) -> bool {
        match self {
            // Classical primitive types, references, and function pointers are copyable
            Type::Bit
            | Type::Bool
            | Type::Int(_)
            | Type::UInt(_)
            | Type::Float(_)
            | Type::String
            | Type::Ref(_)
            | Type::MutRef(_)
            | Type::Unit
            | Type::Function(_) => true,
            // Composite types are copyable if all elements are
            Type::Array(elem_type, _) => elem_type.is_copyable(),
            Type::Tuple(types) => types.iter().all(Type::is_copyable),
            Type::Optional(inner) => inner.is_copyable(),
            // Quantum types, futures, and unknown types are not copyable
            Type::Qubit
            | Type::QuantumReg(_)
            | Type::Future
            | Type::Never
            | Type::Unknown
            | Type::Custom(_) => false,
        }
    }

    /// Check if this type is linear (must be consumed exactly once)
    #[must_use]
    pub fn is_linear(&self) -> bool {
        !self.is_copyable()
    }

    /// Get the default value for this type (if any)
    #[must_use]
    pub fn default_value(&self) -> Option<DefaultValue> {
        match self {
            Type::Bit | Type::Bool => Some(DefaultValue::Bool(false)),
            Type::Int(_) | Type::UInt(_) => Some(DefaultValue::Int(0)),
            Type::Float(_) => Some(DefaultValue::Float(0.0)),
            Type::String => Some(DefaultValue::String(String::new())),
            Type::Unit => Some(DefaultValue::Unit),
            Type::Array(elem_type, ArraySize::Fixed(n)) => elem_type
                .default_value()
                .map(|default| DefaultValue::Array(vec![default; *n])),
            _ => None, // No default value
        }
    }

    /// Check type compatibility for operations
    #[must_use]
    pub fn is_compatible_with(&self, other: &Type) -> bool {
        match (self, other) {
            // Exact match
            (a, b) if a == b => true,

            // Integer promotions (same signedness only)
            (Type::Int(w1), Type::Int(w2)) | (Type::UInt(w1), Type::UInt(w2)) => {
                w1.can_promote_to(w2)
            }
            // Note: Mixed signed/unsigned (Int and UInt) are incompatible - handled by default case

            // Float promotions
            (Type::Float(p1), Type::Float(p2)) => p1.can_promote_to(p2),

            // Array compatibility
            (Type::Array(t1, s1), Type::Array(t2, s2)) => t1.is_compatible_with(t2) && s1 == s2,

            // Reference and optional compatibility
            (Type::Ref(t1), Type::Ref(t2) | Type::MutRef(t2))
            | (Type::MutRef(t1), Type::MutRef(t2))
            | (Type::Optional(t1), Type::Optional(t2)) => t1.is_compatible_with(t2),
            (t1, Type::Optional(t2)) => t1.is_compatible_with(t2),

            _ => false,
        }
    }
}

impl IntWidth {
    #[must_use]
    pub fn bytes(&self) -> usize {
        match self {
            IntWidth::I8 => 1,
            IntWidth::I16 => 2,
            IntWidth::I32 => 4,
            IntWidth::I64 | IntWidth::ISize => 8, // Assume 64-bit platform
            IntWidth::I128 => 16,
            IntWidth::Custom(bits) => (*bits as usize).div_ceil(8), // Round up to bytes
        }
    }

    #[must_use]
    pub fn bits(&self) -> u32 {
        match self {
            IntWidth::I8 => 8,
            IntWidth::I16 => 16,
            IntWidth::I32 => 32,
            IntWidth::I64 | IntWidth::ISize => 64, // Assume 64-bit platform
            IntWidth::I128 => 128,
            IntWidth::Custom(bits) => *bits,
        }
    }

    #[must_use]
    pub fn can_promote_to(&self, other: &IntWidth) -> bool {
        self.bits() <= other.bits()
    }
}

impl FloatPrecision {
    #[must_use]
    pub fn bytes(&self) -> usize {
        match self {
            FloatPrecision::F32 => 4,
            FloatPrecision::F64 => 8,
            FloatPrecision::F128 => 16,
            FloatPrecision::Custom(bits) => (*bits as usize).div_ceil(8),
        }
    }

    #[must_use]
    pub fn bits(&self) -> u32 {
        match self {
            FloatPrecision::F32 => 32,
            FloatPrecision::F64 => 64,
            FloatPrecision::F128 => 128,
            FloatPrecision::Custom(bits) => *bits,
        }
    }

    #[must_use]
    pub fn can_promote_to(&self, other: &FloatPrecision) -> bool {
        self.bits() <= other.bits()
    }
}

/// Default values for types
#[derive(Clone, Debug, PartialEq)]
pub enum DefaultValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<DefaultValue>),
    Tuple(Vec<DefaultValue>),
    Unit,
}

impl TypeRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            custom_types: BTreeMap::new(),
            aliases: BTreeMap::new(),
        }
    }

    /// Register a custom type from a dialect
    pub fn register_type(&mut self, def: CustomTypeDefinition) {
        self.custom_types.insert(def.full_name.clone(), def);
    }

    /// Create a type alias
    pub fn register_alias(&mut self, alias: String, target: Type) {
        self.aliases.insert(alias, target);
    }

    /// Resolve a type name to a Type
    #[must_use]
    pub fn resolve_type(&self, name: &str) -> Option<Type> {
        // Check aliases first
        if let Some(aliased_type) = self.aliases.get(name) {
            return Some(aliased_type.clone());
        }

        // Check custom types
        if let Some(_def) = self.custom_types.get(name) {
            // Parse the custom type name
            if let Some((dialect, type_name)) = name.split_once('.') {
                return Some(Type::Custom(CustomType {
                    dialect: dialect.to_string(),
                    name: type_name.to_string(),
                    parameters: vec![], // TODO: Parse parameters
                }));
            }
        }

        None
    }
}

// Convenience constructors for common types
#[must_use]
pub fn qubit_type() -> Type {
    Type::Qubit
}
#[must_use]
pub fn bit_type() -> Type {
    Type::Bit
}
#[must_use]
pub fn bool_type() -> Type {
    Type::Bool
}
#[must_use]
pub fn int_type() -> Type {
    Type::Int(IntWidth::I32)
}
#[must_use]
pub fn int64_type() -> Type {
    Type::Int(IntWidth::I64)
}
#[must_use]
pub fn float_type() -> Type {
    Type::Float(FloatPrecision::F64)
}
#[must_use]
pub fn string_type() -> Type {
    Type::String
}
#[must_use]
pub fn unit_type() -> Type {
    Type::Unit
}

#[must_use]
pub fn array_type(elem_type: Type, size: usize) -> Type {
    Type::Array(Box::new(elem_type), ArraySize::Fixed(size))
}

#[must_use]
pub fn dynamic_array_type(elem_type: Type) -> Type {
    Type::Array(Box::new(elem_type), ArraySize::Dynamic)
}

#[must_use]
pub fn tuple_type(types: Vec<Type>) -> Type {
    Type::Tuple(types)
}

#[must_use]
pub fn function_type(inputs: Vec<Type>, outputs: Vec<Type>) -> Type {
    Type::Function(FunctionType {
        inputs,
        outputs,
        variadic: false,
    })
}

#[must_use]
pub fn optional_type(inner: Type) -> Type {
    Type::Optional(Box::new(inner))
}

#[must_use]
pub fn ref_type(inner: Type) -> Type {
    Type::Ref(Box::new(inner))
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Qubit => write!(f, "!quantum.qubit"),
            Type::QuantumReg(n) => write!(f, "!quantum.reg<{n}>"),
            Type::Bit => write!(f, "!classical.bit"),
            Type::Bool => write!(f, "!classical.bool"),
            Type::Int(width) => write!(f, "!classical.int<{}>", width.bits()),
            Type::UInt(width) => write!(f, "!classical.uint<{}>", width.bits()),
            Type::Float(precision) => write!(f, "!classical.float<{}>", precision.bits()),
            Type::String => write!(f, "!classical.string"),
            Type::Array(elem, ArraySize::Fixed(n)) => write!(f, "!array<{elem}, {n}>"),
            Type::Array(elem, ArraySize::Dynamic) => write!(f, "!array<{elem}, ?>"),
            Type::Array(elem, ArraySize::Parametric(param)) => {
                write!(f, "!array<{elem}, {param}>")
            }
            Type::Tuple(types) => {
                write!(f, "!tuple<")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ">")
            }
            Type::Function(func) => {
                write!(f, "!function<(")?;
                for (i, input) in func.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{input}")?;
                }
                write!(f, ") -> (")?;
                for (i, output) in func.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{output}")?;
                }
                write!(f, ")>")
            }
            Type::Optional(inner) => write!(f, "!optional<{inner}>"),
            Type::Ref(inner) => write!(f, "!ref<{inner}>"),
            Type::MutRef(inner) => write!(f, "!mut_ref<{inner}>"),
            Type::Custom(custom) => write!(f, "!{}.{}", custom.dialect, custom.name),
            Type::Unit => write!(f, "!unit"),
            Type::Never => write!(f, "!never"),
            Type::Unknown => write!(f, "!unknown"),
            Type::Future => write!(f, "!future"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_properties() {
        assert!(qubit_type().is_quantum());
        assert!(!qubit_type().is_classical());
        assert!(!qubit_type().is_copyable());
        assert!(qubit_type().is_linear());

        assert!(!int_type().is_quantum());
        assert!(int_type().is_classical());
        assert!(int_type().is_copyable());
        assert!(!int_type().is_linear());
    }

    #[test]
    fn test_type_sizes() {
        assert_eq!(bit_type().size_bytes(), Some(1));
        assert_eq!(int64_type().size_bytes(), Some(8));
        assert_eq!(array_type(int_type(), 10).size_bytes(), Some(40));
        assert_eq!(
            tuple_type(vec![int_type(), float_type()]).size_bytes(),
            Some(12)
        );
    }

    #[test]
    fn test_type_compatibility() {
        assert!(int_type().is_compatible_with(&int_type()));
        assert!(Type::Int(IntWidth::I32).is_compatible_with(&Type::Int(IntWidth::I64)));
        assert!(!Type::Int(IntWidth::I64).is_compatible_with(&Type::Int(IntWidth::I32)));
    }

    #[test]
    fn test_type_display() {
        assert_eq!(qubit_type().to_string(), "!quantum.qubit");
        assert_eq!(int_type().to_string(), "!classical.int<32>");
        assert_eq!(
            array_type(qubit_type(), 5).to_string(),
            "!array<!quantum.qubit, 5>"
        );
    }
}
