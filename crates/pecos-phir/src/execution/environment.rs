/*!
PHIR Execution Environment

Environment for managing variables and classical state during PHIR execution.
This is adapted from the pecos-phir-json environment but works with PHIR types.
*/

use crate::error::{PhirError, Result};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

/// Represents the data type of a variable in PHIR execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Signed 32-bit integer
    I32,
    /// Signed 64-bit integer
    I64,
    /// Unsigned 8-bit integer
    U8,
    /// Unsigned 16-bit integer
    U16,
    /// Unsigned 32-bit integer
    U32,
    /// Unsigned 64-bit integer
    U64,
    /// 64-bit floating point
    F64,
    /// Boolean value
    Bool,
    /// Quantum bits (qubits)
    Qubits,
}

impl FromStr for DataType {
    type Err = PhirError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "i8" => Ok(DataType::I8),
            "i16" => Ok(DataType::I16),
            "i32" => Ok(DataType::I32),
            "i64" => Ok(DataType::I64),
            "u8" => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),
            "f64" => Ok(DataType::F64),
            "bool" => Ok(DataType::Bool),
            "qubits" => Ok(DataType::Qubits),
            _ => Err(PhirError::internal(format!("Unsupported data type: {s}"))),
        }
    }
}

impl DataType {
    /// Returns the bit width of the data type
    #[must_use]
    pub fn bit_width(&self) -> usize {
        match self {
            DataType::I8 | DataType::U8 => 8,
            DataType::I16 | DataType::U16 => 16,
            DataType::I32 | DataType::U32 => 32,
            DataType::I64 | DataType::U64 | DataType::F64 => 64,
            DataType::Bool => 1,
            DataType::Qubits => 0, // Qubits don't have a fixed bit width
        }
    }

    /// Checks if the data type is signed
    #[must_use]
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 | DataType::F64
        )
    }
}

/// Represents a typed value in the execution environment
#[derive(Debug, Clone, PartialEq)]
pub enum TypedValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F64(f64),
    Bool(bool),
    BitVec(Vec<bool>),
    /// Struct or array fields in declaration order.
    Aggregate(Vec<TypedValue>),
}

impl TypedValue {
    /// Convert to u64 for measurement results
    ///
    /// # Errors
    ///
    /// Returns an error if the value is a `BitVec` or aggregate.
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            TypedValue::I8(v) => Ok(u64::try_from(*v).unwrap_or(0)),
            TypedValue::I16(v) => Ok(u64::try_from(*v).unwrap_or(0)),
            TypedValue::I32(v) => Ok(u64::try_from(*v).unwrap_or(0)),
            TypedValue::I64(v) => Ok(u64::try_from(*v).unwrap_or(0)),
            TypedValue::U8(v) => Ok(u64::from(*v)),
            TypedValue::U16(v) => Ok(u64::from(*v)),
            TypedValue::U32(v) => Ok(u64::from(*v)),
            TypedValue::U64(v) => Ok(*v),
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            TypedValue::F64(v) => Ok(*v as u64),
            TypedValue::Bool(v) => Ok(u64::from(*v)),
            TypedValue::Aggregate(_) => Err(PhirError::internal("Cannot convert Aggregate to u64")),
            TypedValue::BitVec(_) => Err(PhirError::internal("Cannot convert BitVec to u64")),
        }
    }

    /// Build the recursively zeroed value of a PHIR type.
    ///
    /// # Errors
    /// Returns an error for an unknown type, including unknown aggregate fields.
    pub fn zero(ty: &crate::types::Type) -> Result<Self> {
        use crate::types::{ArraySize, IntWidth, Type};
        Ok(match ty {
            Type::Tuple(fields) => {
                Self::Aggregate(fields.iter().map(Self::zero).collect::<Result<_>>()?)
            }
            Type::Array(element, ArraySize::Fixed(n)) => Self::Aggregate(
                (0..*n)
                    .map(|_| Self::zero(element))
                    .collect::<Result<_>>()?,
            ),
            Type::Int(IntWidth::I8 | IntWidth::I16 | IntWidth::I32) => Self::I32(0),
            Type::UInt(IntWidth::I8 | IntWidth::I16 | IntWidth::I32) => Self::U32(0),
            Type::UInt(_) => Self::U64(0),
            Type::Bool => Self::Bool(false),
            Type::Float(_) => Self::F64(0.0),
            Type::Unknown => {
                return Err(PhirError::internal(format!(
                    "No zero value for type {ty:?}"
                )));
            }
            Type::Int(_)
            | Type::Qubit
            | Type::QuantumReg(_)
            | Type::Bit
            | Type::String
            | Type::Array(_, ArraySize::Dynamic | ArraySize::Parametric(_))
            | Type::Function(_)
            | Type::Optional(_)
            | Type::Ref(_)
            | Type::MutRef(_)
            | Type::Custom(_)
            | Type::Unit
            | Type::Never
            | Type::Future => Self::I64(0),
        })
    }

    /// Reset a value while preserving its runtime shape and scalar tags.
    pub(crate) fn zero_like(&self) -> Self {
        match self {
            Self::I8(_) => Self::I8(0),
            Self::I16(_) => Self::I16(0),
            Self::I32(_) => Self::I32(0),
            Self::I64(_) => Self::I64(0),
            Self::U8(_) => Self::U8(0),
            Self::U16(_) => Self::U16(0),
            Self::U32(_) => Self::U32(0),
            Self::U64(_) => Self::U64(0),
            Self::F64(_) => Self::F64(0.0),
            Self::Bool(_) => Self::Bool(false),
            Self::BitVec(bits) => Self::BitVec(vec![false; bits.len()]),
            Self::Aggregate(fields) => {
                Self::Aggregate(fields.iter().map(Self::zero_like).collect())
            }
        }
    }
}

impl fmt::Display for TypedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedValue::I8(v) => write!(f, "{v}"),
            TypedValue::I16(v) => write!(f, "{v}"),
            TypedValue::I32(v) => write!(f, "{v}"),
            TypedValue::I64(v) => write!(f, "{v}"),
            TypedValue::U8(v) => write!(f, "{v}"),
            TypedValue::U16(v) => write!(f, "{v}"),
            TypedValue::U32(v) => write!(f, "{v}"),
            TypedValue::U64(v) => write!(f, "{v}"),
            TypedValue::F64(v) => write!(f, "{v}"),
            TypedValue::Bool(v) => write!(f, "{v}"),
            TypedValue::Aggregate(fields) => {
                write!(f, "{{")?;
                for (index, field) in fields.iter().enumerate() {
                    if index != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{field}")?;
                }
                write!(f, "}}")
            }
            TypedValue::BitVec(v) => {
                let bits: String = v.iter().map(|b| if *b { '1' } else { '0' }).collect();
                write!(f, "{bits}")
            }
        }
    }
}

/// Variable definition in the environment
#[derive(Debug, Clone)]
pub struct VariableDefinition {
    pub data_type: DataType,
    pub size: usize,
    pub value: Option<TypedValue>,
}

/// Execution environment for PHIR programs
#[derive(Debug, Clone)]
pub struct Environment {
    /// Variable definitions and their current values
    variables: BTreeMap<String, VariableDefinition>,
    /// Mapping from variable names to their bit positions (for result extraction)
    bit_mappings: BTreeMap<String, Vec<usize>>,
}

impl Environment {
    /// Create a new empty environment
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: BTreeMap::new(),
            bit_mappings: BTreeMap::new(),
        }
    }

    /// Add a variable definition to the environment
    ///
    /// # Errors
    ///
    /// Currently always returns Ok, but may return errors in future for duplicate variables
    pub fn add_variable(&mut self, name: &str, data_type: DataType, size: usize) -> Result<()> {
        let var_def = VariableDefinition {
            data_type,
            size,
            value: None,
        };

        self.variables.insert(name.to_string(), var_def);
        Ok(())
    }

    /// Set the value of a variable
    ///
    /// # Errors
    ///
    /// Currently always returns Ok, but may return errors in future for type mismatches
    pub fn set_variable(&mut self, name: &str, value: TypedValue) -> Result<()> {
        if let Some(var_def) = self.variables.get_mut(name) {
            // TODO: Add type checking here
            var_def.value = Some(value);
            Ok(())
        } else {
            Err(PhirError::internal(format!("Variable not found: {name}")))
        }
    }

    /// Get the value of a variable
    ///
    /// # Errors
    ///
    /// Currently always returns Ok with None if variable not found
    pub fn get_variable(&self, name: &str) -> Result<Option<&TypedValue>> {
        if let Some(var_def) = self.variables.get(name) {
            Ok(var_def.value.as_ref())
        } else {
            Err(PhirError::internal(format!("Variable not found: {name}")))
        }
    }

    /// Check if a variable exists
    #[must_use]
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get all variable names
    #[must_use]
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Get all variables with their values (for result extraction)
    #[must_use]
    pub fn get_all_variables(&self) -> BTreeMap<String, TypedValue> {
        let mut result = BTreeMap::new();
        for (name, var_def) in &self.variables {
            if let Some(value) = &var_def.value {
                result.insert(name.clone(), value.clone());
            }
        }
        result
    }

    /// Reset all variable values but keep definitions
    pub fn reset(&mut self) {
        for var_def in self.variables.values_mut() {
            // Reset to default value based on data type (always 0)
            let default_value = match var_def.data_type {
                DataType::I8 => TypedValue::I8(0),
                DataType::I16 => TypedValue::I16(0),
                DataType::I32 => TypedValue::I32(0),
                DataType::I64 => TypedValue::I64(0),
                DataType::U8 => TypedValue::U8(0),
                DataType::U16 => TypedValue::U16(0),
                DataType::U32 => TypedValue::U32(0),
                DataType::U64 => TypedValue::U64(0),
                DataType::F64 => TypedValue::F64(0.0),
                DataType::Bool => TypedValue::Bool(false),
                DataType::Qubits => {
                    // For qubit arrays, create a bit vector of all false
                    TypedValue::BitVec(vec![false; var_def.size])
                }
            };
            var_def.value = Some(default_value);
        }
    }

    /// Clear all variables and definitions
    pub fn clear(&mut self) {
        self.variables.clear();
        self.bit_mappings.clear();
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}
