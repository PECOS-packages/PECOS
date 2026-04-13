use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use pecos_core::BitUInt;
use pecos_core::errors::PecosError;

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

impl FromStr for DataType {
    type Err = PecosError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
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

// Add integer support for BoolBit
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
    pub metadata: Option<BTreeMap<String, serde_json::Value>>,
}

/// A variable value stored as a fixed-width integer using `BitUInt`.
///
/// All values are stored as `BitUInt` (unsigned) with the declared bit width.
/// Signedness is tracked via the `DataType` in `VariableInfo` and applied
/// during `as_i64()` using two's complement interpretation.
///
/// This matches how Python PECOS handles classical variables: raw bits are
/// stored, and sign interpretation happens at the API boundary.
#[derive(Debug, Clone)]
pub struct BitValue {
    /// The raw value as an N-bit unsigned integer (N = register size)
    inner: BitUInt,
    /// Whether this value should be interpreted as signed
    signed: bool,
    /// The data type's full bit width (e.g. 64 for i64, 32 for u32)
    /// Used for sign interpretation: values are negative only when the
    /// sign bit at this width is set, matching Python PECOS dtype behavior.
    type_width: u16,
}

impl BitValue {
    /// Create a new zero value for the given data type and size.
    #[must_use]
    pub fn zero(data_type: &DataType, size: usize) -> Self {
        let raw_tw = data_type.bit_width();
        // For types with 0 bit width (qubits), use the declared size
        let tw = u16::try_from(if raw_tw > 0 { raw_tw } else { size })
            .unwrap_or(64)
            .max(1);
        Self {
            inner: BitUInt::zero(tw),
            signed: data_type.is_signed(),
            type_width: tw,
        }
    }

    /// Create a new value from a raw u64 for the given data type and size.
    ///
    /// Storage is at the full type width. The value is masked to `size` bits
    /// for user-assigned values (matching PHIR semantics where `size` limits
    /// the data bits the user can write, but the underlying register is N bits).
    #[must_use]
    pub fn from_u64(data_type: &DataType, size: usize, value: u64) -> Self {
        let raw_tw = data_type.bit_width();
        let tw = u16::try_from(if raw_tw > 0 { raw_tw } else { size })
            .unwrap_or(64)
            .max(1);
        let s = u16::try_from(size).unwrap_or(tw);
        // Mask to `size` data bits before storing at type width.
        // This ensures user-assigned values respect the declared register size,
        // while the full type-width storage allows bitwise ops on all N bits.
        let masked = if s < tw {
            value & ((1u64 << s) - 1)
        } else {
            value
        };
        Self {
            inner: BitUInt::new(tw, masked),
            signed: data_type.is_signed(),
            type_width: tw,
        }
    }

    /// Get value as u64 (raw bits, truncates for >64 bit values).
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        self.inner.to_u64().unwrap_or(0)
    }

    /// Get the inner `BitUInt` (for expression evaluation without truncation).
    #[must_use]
    pub fn to_bituint(&self) -> BitUInt {
        self.inner.clone()
    }

    /// Get value as i64, interpreting sign via two's complement using the
    /// TYPE's bit width (not the register size).
    ///
    /// This matches Python behavior where `i64(3)` is positive even if stored
    /// in a 2-bit register, because the sign bit is at position 63 (i64 type
    /// width), not at position 1 (register size).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn as_i64(&self) -> i64 {
        let raw = self.as_u64();
        if self.signed {
            let tw = self.type_width;
            if tw >= 64 {
                return raw as i64;
            }
            // Two's complement using TYPE width
            let sign_bit = 1u64 << (tw - 1);
            if raw & sign_bit != 0 {
                let mask = !((1u64 << tw) - 1);
                (raw | mask) as i64
            } else {
                raw as i64
            }
        } else {
            raw as i64
        }
    }

    /// Get value as u32.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_u32(&self) -> u32 {
        self.as_u64() as u32
    }

    /// Get value as bool.
    #[must_use]
    pub fn as_bool(&self) -> bool {
        !self.inner.is_zero()
    }

    /// Get the declared bit width.
    #[must_use]
    pub fn size(&self) -> u16 {
        self.inner.size()
    }

    /// Whether this value is signed.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.signed
    }

    /// Get a specific bit.
    ///
    /// # Errors
    ///
    /// Returns `PecosError::Input` if `idx` is out of range for this value's type width.
    pub fn get_bit(&self, idx: usize) -> Result<bool, PecosError> {
        let idx16 = u16::try_from(idx)
            .map_err(|_| PecosError::Input(format!("Bit index {idx} too large")))?;
        if idx16 >= self.inner.size() {
            return Err(PecosError::Input(format!(
                "Bit index {idx} out of range for type with bit width {}",
                self.inner.size()
            )));
        }
        Ok(self.inner.get_bit(idx16))
    }

    /// Set a specific bit, returning new value.
    ///
    /// # Errors
    ///
    /// Returns `PecosError::Input` if `idx` is out of range for this value's type width.
    pub fn with_bit_set(&self, idx: usize, bit: bool) -> Result<BitValue, PecosError> {
        let idx16 = u16::try_from(idx)
            .map_err(|_| PecosError::Input(format!("Bit index {idx} too large")))?;
        if idx16 >= self.inner.size() {
            return Err(PecosError::Input(format!(
                "Bit index {idx} out of range for type with bit width {}",
                self.inner.size()
            )));
        }
        let mut new_inner = self.inner.clone();
        new_inner.set_bit(idx16, bit);
        Ok(BitValue {
            inner: new_inner,
            signed: self.signed,
            type_width: self.type_width,
        })
    }

    /// Get the data type of this value.
    #[must_use]
    pub fn get_type(&self) -> DataType {
        let size = self.inner.size();
        if self.signed {
            match size {
                1..=8 => DataType::I8,
                9..=16 => DataType::I16,
                17..=32 => DataType::I32,
                _ => DataType::I64,
            }
        } else {
            match size {
                1..=8 => DataType::U8,
                9..=16 => DataType::U16,
                17..=32 => DataType::U32,
                _ => DataType::U64,
            }
        }
    }
}

impl fmt::Display for BitValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.signed {
            write!(f, "{}", self.as_i64())
        } else {
            write!(f, "{}", self.as_u64())
        }
    }
}

/// Environment for storing variables with efficient access
#[derive(Debug, Clone)]
pub struct Environment {
    /// Values of all variables using proper fixed-width integer types
    values: Vec<BitValue>,
    /// Maps variable names to indices in the values vector
    name_to_index: BTreeMap<String, usize>,
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
            name_to_index: BTreeMap::new(),
            metadata: Vec::new(),
            mappings: Vec::new(),
        }
    }

    /// Resets all variable values to zero while keeping their definitions
    pub fn reset_values(&mut self) {
        for (i, info) in self.metadata.iter().enumerate() {
            self.values[i] = BitValue::zero(&info.data_type, info.size);
        }
        self.mappings.clear();
    }

    /// Adds a new variable to the environment
    ///
    /// # Errors
    /// Returns an error if a variable with the same name already exists.
    pub fn add_variable(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
    ) -> Result<(), PecosError> {
        self.add_variable_with_metadata(name, data_type, size, None)
    }

    /// Adds a new variable to the environment with metadata
    ///
    /// # Errors
    /// Returns an error if a variable with the same name already exists.
    pub fn add_variable_with_metadata(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    ) -> Result<(), PecosError> {
        if self.name_to_index.contains_key(name) {
            return Err(PecosError::Input(format!(
                "Variable '{name}' already exists"
            )));
        }

        let index = self.values.len();
        self.name_to_index.insert(name.to_string(), index);

        // Initialize with zero value of appropriate type and declared size
        self.values.push(BitValue::zero(&data_type, size));

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

    /// Gets the value of a variable as a `BitValue`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&BitValue> {
        self.name_to_index.get(name).map(|&idx| &self.values[idx])
    }

    /// Gets the raw u64 value of a variable.
    #[must_use]
    pub fn get_raw(&self, name: &str) -> Option<u64> {
        self.get(name).map(BitValue::as_u64)
    }

    /// Sets the value of a variable using a u64.
    ///
    /// The value is automatically masked to the variable's declared bit width
    /// by the underlying `BitInt`/`BitUInt` storage. This is the same as `set_raw`.
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist.
    pub fn set(&mut self, name: &str, value: u64) -> Result<(), PecosError> {
        self.set_raw(name, value)
    }

    /// Sets the value of a variable using a raw u64.
    ///
    /// The value is automatically masked to the variable's declared bit width.
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist.
    pub fn set_raw(&mut self, name: &str, value: u64) -> Result<(), PecosError> {
        if let Some(&idx) = self.name_to_index.get(name) {
            let info = &self.metadata[idx];
            // BitValue::from_u64 automatically masks to the declared size
            self.values[idx] = BitValue::from_u64(&info.data_type, info.size, value);
            Ok(())
        } else {
            Err(PecosError::Input(format!("Variable '{name}' not found")))
        }
    }

    /// Gets metadata for a variable
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist.
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
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist or the bit index is out of range.
    pub fn get_bit(&self, var_name: &str, bit_index: usize) -> Result<BoolBit, PecosError> {
        if let Some(&idx) = self.name_to_index.get(var_name) {
            self.values[idx].get_bit(bit_index).map(BoolBit)
        } else {
            Err(PecosError::Input(format!(
                "Variable '{var_name}' not found"
            )))
        }
    }

    /// Sets a specific bit in a variable
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist or the bit index is out of range.
    pub fn set_bit<T: Into<BoolBit>>(
        &mut self,
        var_name: &str,
        bit_index: usize,
        bit_value: T,
    ) -> Result<(), PecosError> {
        let bool_value = bit_value.into().0;

        if let Some(&idx) = self.name_to_index.get(var_name) {
            self.values[idx] = self.values[idx].with_bit_set(bit_index, bool_value)?;
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
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_measurement_results(&self) -> BTreeMap<String, u32> {
        let mut results = BTreeMap::new();
        for (i, info) in self.metadata.iter().enumerate() {
            if info.name.starts_with('m') || info.name.starts_with("measurement") {
                results.insert(info.name.clone(), self.values[i].as_u64() as u32);
            }
        }

        if results.is_empty() && !self.mappings.is_empty() {
            for (source, dest) in &self.mappings {
                if let Some(&idx) = self.name_to_index.get(source) {
                    results.insert(dest.clone(), self.values[idx].as_u64() as u32);
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
    ///
    /// # Errors
    /// Returns an error if the source variable doesn't exist.
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
    pub fn get_mapped_results(&self) -> BTreeMap<String, u32> {
        let mut results = BTreeMap::new();

        // Apply all mappings from source to destination
        for (source, dest) in &self.mappings {
            if let Some(value) = self.get(source) {
                results.insert(dest.clone(), value.as_u32());
            }
        }

        // If no mappings exist or no values were found, return all variables that have values
        if results.is_empty() {
            for (i, info) in self.metadata.iter().enumerate() {
                results.insert(info.name.clone(), self.values[i].as_u32());
            }
        }

        results
    }

    /// Copy a variable value to another variable
    /// Used for Result operation in Python implementation
    ///
    /// # Errors
    /// Returns an error if the source variable doesn't exist.
    pub fn copy_variable(&mut self, src_name: &str, dst_name: &str) -> Result<(), PecosError> {
        // Check if source exists
        if let Some(src_idx) = self.name_to_index.get(src_name) {
            let src_value = self.values[*src_idx].clone();
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

        // Test i8 constraints: 8-bit signed, raw bits stored as BitUInt(8)
        env.set_raw("i8_var", 127).unwrap();
        assert_eq!(env.get_raw("i8_var"), Some(127));
        assert_eq!(env.get("i8_var").map(BitValue::as_i64), Some(127));

        env.set_raw("i8_var", 128).unwrap(); // 128 masked to 8 bits = 0x80
        assert_eq!(env.get_raw("i8_var"), Some(128)); // Raw bits: 0x80
        assert_eq!(env.get("i8_var").map(BitValue::as_i64), Some(-128)); // Signed: -128

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
