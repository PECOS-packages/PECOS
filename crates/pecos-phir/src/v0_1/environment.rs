use pecos_core::errors::PecosError;
use std::collections::HashMap;
use std::fmt;

/// Represents the data type of a variable
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl DataType {
    /// Creates a `DataType` from a string representation
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, PecosError> {
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
            _ => Err(PecosError::Input(format!("Unsupported data type: {s}"))),
        }
    }

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

    /// Returns the maximum value for this data type
    #[must_use]
    pub fn max_value(&self) -> u64 {
        match self {
            DataType::I8 => i8::MAX as u64,
            DataType::I16 => i16::MAX as u64,
            DataType::I32 => i32::MAX as u64,
            DataType::I64 => i64::MAX as u64,
            DataType::U8 => u64::from(u8::MAX),
            DataType::U16 => u64::from(u16::MAX),
            DataType::U32 => u64::from(u32::MAX),
            DataType::U64 => u64::MAX,
            DataType::Bool => 1,
            DataType::Qubits => 0,
        }
    }

    /// Returns the minimum value for this data type
    #[must_use]
    pub fn min_value(&self) -> i64 {
        match self {
            DataType::I8 => i64::from(i8::MIN),
            DataType::I16 => i64::from(i16::MIN),
            DataType::I32 => i64::from(i32::MIN),
            DataType::I64 => i64::MIN,
            DataType::U8
            | DataType::U16
            | DataType::U32
            | DataType::U64
            | DataType::Bool
            | DataType::Qubits => 0,
        }
    }

    /// Applies type constraints to a value based on the bit width and signedness
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    pub fn constrain_value(&self, value: u64) -> u64 {
        match self {
            DataType::I8 => (value as i8) as u64,
            DataType::I16 => (value as i16) as u64,
            DataType::I32 => (value as i32) as u64,
            DataType::I64 => (value as i64) as u64,
            DataType::U8 => value & 0xFF,
            DataType::U16 => value & 0xFFFF,
            DataType::U32 => value & 0xFFFF_FFFF,
            DataType::U64 | DataType::Qubits => value, // Full 64-bit range for these types
            DataType::Bool => value & 1,
        }
    }
}

// Implement Display for DataType
impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::I8 => write!(f, "i8"),
            DataType::I16 => write!(f, "i16"),
            DataType::I32 => write!(f, "i32"),
            DataType::I64 => write!(f, "i64"),
            DataType::U8 => write!(f, "u8"),
            DataType::U16 => write!(f, "u16"),
            DataType::U32 => write!(f, "u32"),
            DataType::U64 => write!(f, "u64"),
            DataType::Bool => write!(f, "bool"),
            DataType::Qubits => write!(f, "qubits"),
        }
    }
}

/// Represents a variable value that can be typed
#[derive(Debug, Clone, Copy)]
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
}

impl TypedValue {
    /// Creates a new `TypedValue` with the specified data type and value
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    pub fn new(data_type: &DataType, value: u64) -> Self {
        match data_type {
            DataType::I8 => TypedValue::I8(value as i8),
            DataType::I16 => TypedValue::I16(value as i16),
            DataType::I32 => TypedValue::I32(value as i32),
            DataType::I64 => TypedValue::I64(value as i64),
            DataType::U8 => TypedValue::U8(value as u8),
            DataType::U16 => TypedValue::U16(value as u16),
            DataType::U32 => TypedValue::U32(value as u32),
            DataType::U64 | DataType::Qubits => TypedValue::U64(value), // U64 and Qubits both use U64
            DataType::Bool => TypedValue::Bool(value != 0),
        }
    }

    /// Creates a typed value from a raw u64, inferring the type as i32
    /// This is for backward compatibility with code that uses raw values
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn from_raw(value: u64) -> Self {
        TypedValue::I32(value as i32)
    }

    /// Gets the value as a u64 (for uniform storage)
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn as_u64(&self) -> u64 {
        match self {
            TypedValue::I8(val) => *val as u64,
            TypedValue::I16(val) => *val as u64,
            TypedValue::I32(val) => *val as u64,
            TypedValue::I64(val) => *val as u64,
            TypedValue::U8(val) => u64::from(*val),
            TypedValue::U16(val) => u64::from(*val),
            TypedValue::U32(val) => u64::from(*val),
            TypedValue::U64(val) => *val,
            TypedValue::Bool(val) => u64::from(*val),
        }
    }

    /// Gets the value as an i64 (for expressions)
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn as_i64(&self) -> i64 {
        match self {
            TypedValue::I8(val) => i64::from(*val),
            TypedValue::I16(val) => i64::from(*val),
            TypedValue::I32(val) => i64::from(*val),
            TypedValue::I64(val) => *val,
            TypedValue::U8(val) => i64::from(*val),
            TypedValue::U16(val) => i64::from(*val),
            TypedValue::U32(val) => i64::from(*val),
            TypedValue::U64(val) => *val as i64,
            TypedValue::Bool(val) => i64::from(*val),
        }
    }

    /// Gets the value as a boolean
    #[must_use]
    pub fn as_bool(&self) -> bool {
        match self {
            TypedValue::I8(val) => *val != 0,
            TypedValue::I16(val) => *val != 0,
            TypedValue::I32(val) => *val != 0,
            TypedValue::I64(val) => *val != 0,
            TypedValue::U8(val) => *val != 0,
            TypedValue::U16(val) => *val != 0,
            TypedValue::U32(val) => *val != 0,
            TypedValue::U64(val) => *val != 0,
            TypedValue::Bool(val) => *val,
        }
    }

    /// Gets the value as a u32
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_u32(&self) -> u32 {
        self.as_u64() as u32
    }
}

// Implement Display for TypedValue for better logging
impl fmt::Display for TypedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedValue::I8(val) => write!(f, "{val}"),
            TypedValue::I16(val) => write!(f, "{val}"),
            TypedValue::I32(val) => write!(f, "{val}"),
            TypedValue::I64(val) => write!(f, "{val}"),
            TypedValue::U8(val) => write!(f, "{val}"),
            TypedValue::U16(val) => write!(f, "{val}"),
            TypedValue::U32(val) => write!(f, "{val}"),
            TypedValue::U64(val) => write!(f, "{val}"),
            TypedValue::Bool(val) => write!(f, "{val}"),
        }
    }
}

// From implementation for TypedValue to u64
impl From<TypedValue> for u64 {
    fn from(value: TypedValue) -> Self {
        value.as_u64()
    }
}

// From implementation for u64 to TypedValue
impl From<u64> for TypedValue {
    fn from(value: u64) -> Self {
        TypedValue::from_raw(value)
    }
}

// From implementation for i64 to TypedValue
impl From<i64> for TypedValue {
    fn from(value: i64) -> Self {
        TypedValue::I64(value)
    }
}

// From implementation for i32 to TypedValue
impl From<i32> for TypedValue {
    fn from(value: i32) -> Self {
        TypedValue::I32(value)
    }
}

// From implementation for bool to TypedValue
impl From<bool> for TypedValue {
    fn from(value: bool) -> Self {
        TypedValue::Bool(value)
    }
}

// From implementation for TypedValue to u32
impl From<TypedValue> for u32 {
    #[allow(clippy::cast_possible_truncation)]
    fn from(value: TypedValue) -> Self {
        value.as_u64() as u32
    }
}

// From implementation for u32 to TypedValue
impl From<u32> for TypedValue {
    fn from(value: u32) -> Self {
        TypedValue::U32(value)
    }
}

// From implementation for TypedValue to i64
impl From<TypedValue> for i64 {
    fn from(value: TypedValue) -> Self {
        value.as_i64()
    }
}

// To handle option comparisons safely, we implement PartialEq on our own types
impl PartialEq<u64> for TypedValue {
    fn eq(&self, other: &u64) -> bool {
        self.as_u64() == *other
    }
}

impl PartialEq<i64> for TypedValue {
    fn eq(&self, other: &i64) -> bool {
        self.as_i64() == *other
    }
}

impl PartialEq<u32> for TypedValue {
    fn eq(&self, other: &u32) -> bool {
        self.as_u32() == *other
    }
}

impl PartialEq<i32> for TypedValue {
    fn eq(&self, other: &i32) -> bool {
        self.as_i64() == i64::from(*other)
    }
}

impl PartialEq<TypedValue> for u64 {
    fn eq(&self, other: &TypedValue) -> bool {
        *self == other.as_u64()
    }
}

impl PartialEq<TypedValue> for i64 {
    fn eq(&self, other: &TypedValue) -> bool {
        *self == other.as_i64()
    }
}

impl PartialEq<TypedValue> for u32 {
    fn eq(&self, other: &TypedValue) -> bool {
        *self == other.as_u32()
    }
}

impl PartialEq<TypedValue> for i32 {
    fn eq(&self, other: &TypedValue) -> bool {
        i64::from(*self) == other.as_i64()
    }
}

// Add integer support for BoolBit (already defined above)
impl PartialEq<i32> for BoolBit {
    fn eq(&self, other: &i32) -> bool {
        (self.0 && *other != 0) || (!self.0 && *other == 0)
    }
}

impl PartialEq<u32> for BoolBit {
    fn eq(&self, other: &u32) -> bool {
        (self.0 && *other != 0) || (!self.0 && *other == 0)
    }
}

// Implement bit shifting for TypedValue
impl std::ops::Shr<usize> for TypedValue {
    type Output = u64;

    fn shr(self, rhs: usize) -> Self::Output {
        self.as_u64() >> rhs
    }
}

impl std::ops::Shr<usize> for &TypedValue {
    type Output = u64;

    fn shr(self, rhs: usize) -> Self::Output {
        self.as_u64() >> rhs
    }
}

/// Wrapper for boolean bit values to solve trait implementation issues
/// with `convert_into`
#[derive(Debug, Clone, Copy)]
pub struct BoolBit(pub bool);

impl fmt::Display for BoolBit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<bool> for BoolBit {
    fn eq(&self, other: &bool) -> bool {
        self.0 == *other
    }
}

impl PartialEq<BoolBit> for bool {
    fn eq(&self, other: &BoolBit) -> bool {
        *self == other.0
    }
}

impl From<BoolBit> for bool {
    fn from(bit: BoolBit) -> Self {
        bit.0
    }
}

impl From<bool> for BoolBit {
    fn from(value: bool) -> Self {
        BoolBit(value)
    }
}

impl From<u64> for BoolBit {
    fn from(value: u64) -> Self {
        BoolBit(value != 0)
    }
}

impl From<u32> for BoolBit {
    fn from(value: u32) -> Self {
        BoolBit(value != 0)
    }
}

impl From<i64> for BoolBit {
    fn from(value: i64) -> Self {
        BoolBit(value != 0)
    }
}

impl From<i32> for BoolBit {
    fn from(value: i32) -> Self {
        BoolBit(value != 0)
    }
}

impl From<BoolBit> for u32 {
    fn from(bit: BoolBit) -> Self {
        u32::from(bit.0)
    }
}

impl From<BoolBit> for i32 {
    fn from(bit: BoolBit) -> Self {
        i32::from(bit.0)
    }
}

impl TypedValue {
    /// Gets the data type of this value
    #[must_use]
    pub fn get_type(&self) -> DataType {
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
        }
    }

    /// Gets a specific bit from the value
    pub fn get_bit(&self, idx: usize) -> Result<bool, PecosError> {
        // Check that idx is within the bit width of the type
        let bit_width = self.get_type().bit_width();
        if idx >= bit_width {
            return Err(PecosError::Input(format!(
                "Bit index {idx} out of range for type with bit width {bit_width}"
            )));
        }

        // Extract the bit
        let val = self.as_u64();
        Ok(((val >> idx) & 1) != 0)
    }

    /// Sets a specific bit in the value
    pub fn with_bit_set(&self, idx: usize, bit_value: bool) -> Result<TypedValue, PecosError> {
        // Check that idx is within the bit width of the type
        let bit_width = self.get_type().bit_width();
        if idx >= bit_width {
            return Err(PecosError::Input(format!(
                "Bit index {idx} out of range for type with bit width {bit_width}"
            )));
        }

        // Update the bit
        let val = self.as_u64();
        let new_val = if bit_value {
            val | (1 << idx)
        } else {
            val & !(1 << idx)
        };

        // Return new typed value
        Ok(TypedValue::new(&self.get_type(), new_val))
    }
}

/// Metadata for a variable
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Name of the variable
    pub name: String,
    /// Data type of the variable
    pub data_type: DataType,
    /// Size of the variable (number of elements)
    pub size: usize,
    /// Additional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Environment for storing variables with efficient access
#[derive(Debug, Clone)]
pub struct Environment {
    /// Values of all variables (stored with their type information)
    values: Vec<TypedValue>,
    /// Maps variable names to indices in the values vector
    name_to_index: HashMap<String, usize>,
    /// Metadata for each variable
    metadata: Vec<VariableInfo>,
    /// Maps source variable names to destination names for output
    mappings: Vec<(String, String)>,
}

impl Environment {
    /// Creates a new empty environment
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            name_to_index: HashMap::new(),
            metadata: Vec::new(),
            mappings: Vec::new(),
        }
    }

    /// Resets all variable values to zero while keeping their definitions
    pub fn reset_values(&mut self) {
        for (i, info) in self.metadata.iter().enumerate() {
            // Reset according to type
            self.values[i] = TypedValue::new(&info.data_type, 0);
        }
        self.mappings.clear();
    }

    /// Adds a new variable to the environment
    pub fn add_variable(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
    ) -> Result<(), PecosError> {
        self.add_variable_with_metadata(name, data_type, size, None)
    }

    /// Adds a new variable to the environment with metadata
    pub fn add_variable_with_metadata(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<(), PecosError> {
        if self.name_to_index.contains_key(name) {
            return Err(PecosError::Input(format!(
                "Variable '{name}' already exists"
            )));
        }

        let index = self.values.len();
        self.name_to_index.insert(name.to_string(), index);

        // Initialize with zero value of appropriate type
        self.values.push(TypedValue::new(&data_type, 0));

        self.metadata.push(VariableInfo {
            name: name.to_string(),
            data_type,
            size,
            metadata,
        });

        Ok(())
    }

    /// Checks if a variable exists in the environment
    #[must_use]
    pub fn has_variable(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Gets the typed value of a variable
    #[must_use]
    pub fn get(&self, name: &str) -> Option<TypedValue> {
        self.name_to_index.get(name).map(|&idx| self.values[idx])
    }

    /// Gets the raw u64 value of a variable (for backward compatibility)
    #[must_use]
    pub fn get_raw(&self, name: &str) -> Option<u64> {
        self.get(name).map(|v| v.as_u64())
    }

    /// Sets the value of a variable with type checking
    ///
    /// Accepts any type that can be converted to `TypedValue`
    pub fn set<T: Into<TypedValue>>(&mut self, name: &str, value: T) -> Result<(), PecosError> {
        let typed_value = value.into();
        if let Some(&idx) = self.name_to_index.get(name) {
            // Get the data type of the variable
            let expected_type = &self.metadata[idx].data_type;

            // For now, we'll be lenient with type checking for backward compatibility
            // Just apply constraints to ensure the value fits within the data type
            let raw_value = typed_value.as_u64();
            let constrained_value = expected_type.constrain_value(raw_value);

            // Create a new typed value with the correct type and set it
            self.values[idx] = TypedValue::new(expected_type, constrained_value);
            Ok(())
        } else {
            Err(PecosError::Input(format!("Variable '{name}' not found")))
        }
    }

    /// Sets the value of a variable using a raw u64 (for backward compatibility)
    pub fn set_raw(&mut self, name: &str, value: u64) -> Result<(), PecosError> {
        if let Some(&idx) = self.name_to_index.get(name) {
            // Apply constraints based on data type
            let data_type = &self.metadata[idx].data_type;
            let constrained_value = data_type.constrain_value(value);

            // Create a typed value and set it
            self.values[idx] = TypedValue::new(data_type, constrained_value);
            Ok(())
        } else {
            Err(PecosError::Input(format!("Variable '{name}' not found")))
        }
    }

    /// Gets metadata for a variable
    pub fn get_variable_info(&self, name: &str) -> Result<&VariableInfo, PecosError> {
        if let Some(&idx) = self.name_to_index.get(name) {
            Ok(&self.metadata[idx])
        } else {
            Err(PecosError::Input(format!("Variable '{name}' not found")))
        }
    }

    /// Gets metadata for a variable as Option
    #[must_use]
    pub fn get_variable_info_opt(&self, name: &str) -> Option<&VariableInfo> {
        self.name_to_index.get(name).map(|&idx| &self.metadata[idx])
    }

    /// Gets a specific bit from a variable
    pub fn get_bit(&self, var_name: &str, bit_index: usize) -> Result<BoolBit, PecosError> {
        if let Some(&idx) = self.name_to_index.get(var_name) {
            // Check bit index is in range
            if bit_index >= self.metadata[idx].size {
                return Err(PecosError::Input(format!(
                    "Bit index {} out of range for variable '{}' with size {}",
                    bit_index, var_name, self.metadata[idx].size
                )));
            }

            // Extract the bit using the TypedValue method
            self.values[idx].get_bit(bit_index).map(BoolBit)
        } else {
            Err(PecosError::Input(format!(
                "Variable '{var_name}' not found"
            )))
        }
    }

    /// Sets a specific bit in a variable
    pub fn set_bit<T: Into<BoolBit>>(
        &mut self,
        var_name: &str,
        bit_index: usize,
        bit_value: T,
    ) -> Result<(), PecosError> {
        let bool_bit = bit_value.into();
        let bool_value = bool_bit.0;

        if let Some(&idx) = self.name_to_index.get(var_name) {
            // Check bit index is in range
            if bit_index >= self.metadata[idx].size {
                return Err(PecosError::Input(format!(
                    "Bit index {} out of range for variable '{}' with size {}",
                    bit_index, var_name, self.metadata[idx].size
                )));
            }

            // Create a new value with the bit set
            let new_value = self.values[idx].with_bit_set(bit_index, bool_value)?;

            // Set the new value
            self.values[idx] = new_value;
            Ok(())
        } else {
            Err(PecosError::Input(format!(
                "Variable '{var_name}' not found"
            )))
        }
    }

    /// Gets all variable names in the environment
    #[must_use]
    pub fn get_variable_names(&self) -> Vec<String> {
        self.metadata.iter().map(|info| info.name.clone()).collect()
    }

    /// Gets all variables of a specific type
    #[must_use]
    pub fn get_variables_of_type(&self, data_type: &DataType) -> Vec<&VariableInfo> {
        self.metadata
            .iter()
            .filter(|info| &info.data_type == data_type)
            .collect()
    }

    /// Gets all variables in the environment
    #[must_use]
    pub fn get_all_variables(&self) -> &[VariableInfo] {
        &self.metadata
    }

    /// Gets all measurement result variables and their values
    #[must_use]
    pub fn get_measurement_results(&self) -> HashMap<String, TypedValue> {
        let mut results = HashMap::new();
        for (i, info) in self.metadata.iter().enumerate() {
            // Include all variables that start with "m" or "measurement"
            if info.name.starts_with('m') || info.name.starts_with("measurement") {
                results.insert(info.name.clone(), self.values[i]);
            }
        }

        // If no measurement variables were found, add all mapped variables
        if results.is_empty() && !self.mappings.is_empty() {
            for (source, dest) in &self.mappings {
                if let Some(&idx) = self.name_to_index.get(source) {
                    results.insert(dest.clone(), self.values[idx]);
                }
            }
        }

        results
    }

    /// Gets the total number of qubits in the environment
    #[must_use]
    pub fn count_qubits(&self) -> usize {
        self.get_variables_of_type(&DataType::Qubits)
            .iter()
            .map(|info| info.size)
            .sum()
    }

    /// Returns the total number of variables in the environment
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Checks if the environment is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Adds a mapping from source variable to destination name
    /// This is used for tracking variable mappings for program outputs
    pub fn add_mapping(&mut self, source: &str, destination: &str) -> Result<(), PecosError> {
        // Check if source variable exists
        if !self.has_variable(source) {
            return Err(PecosError::Input(format!(
                "Cannot map nonexistent variable '{source}' to '{destination}'"
            )));
        }

        // Add the mapping
        self.mappings
            .push((source.to_string(), destination.to_string()));
        Ok(())
    }

    /// Gets all variable mappings
    #[must_use]
    pub fn get_mappings(&self) -> &[(String, String)] {
        &self.mappings
    }

    /// Clears all mappings
    pub fn clear_mappings(&mut self) {
        self.mappings.clear();
    }

    /// Gets mapped results from the environment
    ///
    /// This method returns mapped results from defined mappings or falls back to all variables
    /// if no mappings are defined or no mapped variables have values.
    #[must_use]
    pub fn get_mapped_results(&self) -> HashMap<String, u32> {
        let mut results = HashMap::new();

        // Apply all mappings from source to destination
        for (source, dest) in &self.mappings {
            if let Some(value) = self.get(source) {
                results.insert(dest.clone(), value.as_u32());
            }
        }

        // If no mappings exist or no values were found, return all variables that have values
        if results.is_empty() {
            for (i, info) in self.metadata.iter().enumerate() {
                let value = self.values[i];
                results.insert(info.name.clone(), value.as_u32());
            }
        }

        results
    }

    /// Copy a variable value to another variable
    /// Used for Result operation in Python implementation
    pub fn copy_variable(&mut self, src_name: &str, dst_name: &str) -> Result<(), PecosError> {
        // Check if source exists
        if let Some(src_idx) = self.name_to_index.get(src_name) {
            let src_value = self.values[*src_idx];
            let src_info = &self.metadata[*src_idx];

            // If destination doesn't exist, create it
            if !self.has_variable(dst_name) {
                self.add_variable(dst_name, src_info.data_type.clone(), src_info.size)?;
            }

            // Set the destination value
            if let Some(dst_idx) = self.name_to_index.get(dst_name) {
                self.values[*dst_idx] = src_value;
                Ok(())
            } else {
                // This should never happen as we just created the variable if it didn't exist
                Err(PecosError::Input(format!(
                    "Failed to copy '{src_name}' to '{dst_name}': destination not found after creation"
                )))
            }
        } else {
            Err(PecosError::Input(format!(
                "Failed to copy '{src_name}' to '{dst_name}': source not found"
            )))
        }
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_basic_operations() {
        let mut env = Environment::new();

        // Add variables
        env.add_variable("x", DataType::I32, 32).unwrap();
        env.add_variable("y", DataType::U8, 8).unwrap();

        // Set values
        env.set_raw("x", 42).unwrap();
        env.set_raw("y", 255).unwrap();

        // Get values
        assert_eq!(env.get_raw("x"), Some(42));
        assert_eq!(env.get_raw("y"), Some(255));

        // Check variable existence
        assert!(env.has_variable("x"));
        assert!(!env.has_variable("z"));
    }

    #[test]
    fn test_environment_type_constraints() {
        let mut env = Environment::new();

        // Add variables with different types
        env.add_variable("i8_var", DataType::I8, 8).unwrap();
        env.add_variable("u8_var", DataType::U8, 8).unwrap();

        // Test i8 constraints (-128 to 127)
        env.set_raw("i8_var", 127).unwrap();
        assert_eq!(env.get_raw("i8_var"), Some(127));

        env.set_raw("i8_var", 128).unwrap(); // Should wrap to -128
        assert_eq!(env.get_raw("i8_var"), Some(0xFFFF_FFFF_FFFF_FF80)); // -128 as u64

        // Test u8 constraints (0 to 255)
        env.set_raw("u8_var", 255).unwrap();
        assert_eq!(env.get_raw("u8_var"), Some(255));

        env.set_raw("u8_var", 256).unwrap(); // Should be masked to 0
        assert_eq!(env.get_raw("u8_var"), Some(0));
    }

    #[test]
    fn test_environment_bit_operations() {
        let mut env = Environment::new();

        // Add variable
        env.add_variable("bits", DataType::U8, 8).unwrap();
        env.set_raw("bits", 0).unwrap();

        // Set bits
        env.set_bit("bits", 0, true).unwrap(); // Set bit 0
        env.set_bit("bits", 2, true).unwrap(); // Set bit 2

        // Should have value 0b101 = 5
        assert_eq!(env.get_raw("bits"), Some(5));

        // Get bits
        assert_eq!(env.get_bit("bits", 0).unwrap(), true);
        assert_eq!(env.get_bit("bits", 1).unwrap(), false);
        assert_eq!(env.get_bit("bits", 2).unwrap(), true);

        // Clear a bit
        env.set_bit("bits", 0, false).unwrap();

        // Should have value 0b100 = 4
        assert_eq!(env.get_raw("bits"), Some(4));
    }

    #[test]
    fn test_environment_variable_copying() {
        let mut env = Environment::new();

        // Add source variable
        env.add_variable("source", DataType::I32, 32).unwrap();
        env.set_raw("source", 42).unwrap();

        // Copy to destination (creates new variable)
        env.copy_variable("source", "dest").unwrap();

        // Check that destination exists and has same value
        assert!(env.has_variable("dest"));
        assert_eq!(env.get_raw("dest"), Some(42));

        // Modify source and verify destination is unchanged
        env.set_raw("source", 99).unwrap();
        assert_eq!(env.get_raw("source"), Some(99));
        assert_eq!(env.get_raw("dest"), Some(42));
    }
}
