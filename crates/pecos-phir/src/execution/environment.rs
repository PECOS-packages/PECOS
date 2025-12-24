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
            DataType::I64 | DataType::U64 => 64,
            DataType::Bool => 1,
            DataType::Qubits => 0, // Qubits don't have a fixed bit width
        }
    }

    /// Checks if the data type is signed
    #[must_use]
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64
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
    Bool(bool),
    BitVec(Vec<bool>),
}

impl TypedValue {
    /// Convert to u64 for measurement results
    ///
    /// # Errors
    ///
    /// Returns an error if the value is a `BitVec` which cannot be converted to u64
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
            TypedValue::Bool(v) => Ok(u64::from(*v)),
            TypedValue::BitVec(_) => Err(PhirError::internal("Cannot convert BitVec to u64")),
        }
    }

    /// Get the data type of this value
    #[must_use]
    pub fn data_type(&self) -> DataType {
        match self {
            TypedValue::I8(_) => DataType::I8,
            TypedValue::I16(_) => DataType::I16,
            TypedValue::I32(_) => DataType::I32,
            TypedValue::I64(_) => DataType::I64,
            TypedValue::U8(_) => DataType::U8,
            TypedValue::U16(_) => DataType::U16,
            TypedValue::U32(_) => DataType::U32,
            TypedValue::U64(_) => DataType::U64,
            TypedValue::Bool(_) => DataType::Bool,
            TypedValue::BitVec(_) => DataType::Qubits, // BitVec represents qubit measurements
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
            TypedValue::Bool(v) => write!(f, "{v}"),
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
