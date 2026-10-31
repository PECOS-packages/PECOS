use super::declarations::{DeclarationKind, Declarations};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
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

    /// Checks if the data type is one of the fixed-width integer types.
    #[must_use]
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64
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

/// A whole-register or indexed reference in a Result mapping.
#[derive(Debug, Clone)]
pub(crate) struct ResultReference {
    pub name: String,
    pub index: Option<u32>,
}

impl ResultReference {
    pub(crate) fn from_arg(arg: &super::ast::ArgItem, field: &str) -> Result<Self, PecosError> {
        let value = serde_json::to_value(arg)
            .expect("PHIR arguments contain only JSON-serializable values");
        match super::classical_validation::result_reference(&value, field)? {
            super::ast::ArgItem::Simple(name) => Ok(Self { name, index: None }),
            super::ast::ArgItem::Indexed((name, index)) => Ok(Self {
                name,
                index: Some(u32::try_from(index).expect("validated Result indices fit u32")),
            }),
            _ => unreachable!("Result reference validation returns only whole or indexed names"),
        }
    }
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
    /// Storage is at the full type width, holding the register's value as a
    /// two's-complement pattern. In PHIR a signed size-S register is an
    /// `i(S+1)` integer -- S data bits plus a sign bit -- so the value wraps to
    /// `S + 1` bits and is sign-extended into the high bits (so `as_i64` and
    /// full-width bitwise ops see the correct sign). An unsigned size-S register
    /// is a `u(S)`: it wraps to S bits with no sign bit. Register widths are
    /// validated to fit the backing type when the variable is defined.
    #[must_use]
    pub fn from_u64(data_type: &DataType, size: usize, value: u64) -> Self {
        let raw_tw = data_type.bit_width();
        let tw = u16::try_from(if raw_tw > 0 { raw_tw } else { size })
            .unwrap_or(64)
            .max(1);
        let s = u16::try_from(size).unwrap_or(tw);
        let signed = data_type.is_signed();
        // Effective width: S+1 for signed (sign bit), S for unsigned. Capped at
        // the type width defensively (validation already guarantees it fits).
        let width = if signed {
            s.saturating_add(1).min(tw)
        } else {
            s.min(tw)
        };
        let stored = if width >= 64 {
            // Full-width: the value already carries its own sign bit at bit 63.
            value
        } else {
            let mask = (1u64 << width) - 1;
            let masked = value & mask;
            if signed && (masked >> (width - 1)) & 1 == 1 {
                // Sign-extend into the high bits; `BitUInt::new` truncates back
                // to the type width, giving a full-width two's-complement value.
                masked | !mask
            } else {
                masked
            }
        };
        Self {
            inner: BitUInt::new(tw, stored),
            signed,
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

    /// Get value as i64, interpreting sign via two's complement at the type
    /// width.
    ///
    /// `from_u64` already sign-extends a signed register's `i(S+1)` value into
    /// the high bits, so reinterpreting at the type width recovers the correct
    /// signed value (e.g. an `i64` size-4 register assigned -1 reads back as
    /// -1, and assigned 15 reads back as 15).
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
    pub(crate) exports: Option<super::exports::ExportMap>,
    /// Declaration-order global qubit ranges, separate from classical storage.
    quantum_registers: BTreeMap<String, Range<usize>>,
    num_qubits: usize,
    declarations: Declarations,
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
            exports: None,
            quantum_registers: BTreeMap::new(),
            num_qubits: 0,
            declarations: Declarations::default(),
        }
    }

    /// Explicit output names and their storage sources, including executed Results.
    pub(crate) fn selected_outputs(&self) -> Option<super::exports::ExportMap> {
        self.exports.as_ref().map(|exports| {
            let mut outputs = exports.clone();
            for (_, destination) in &self.mappings {
                outputs.insert(destination.clone(), destination.clone());
            }
            outputs
        })
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
        let kind = if data_type == DataType::Qubits {
            DeclarationKind::Quantum
        } else {
            DeclarationKind::Classical
        };
        self.register_variable(name, data_type, size, metadata, kind)
    }

    pub(crate) fn add_classical_variable(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
    ) -> Result<(), PecosError> {
        self.register_variable(name, data_type, size, None, DeclarationKind::Classical)
    }

    fn register_variable(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
        kind: DeclarationKind,
    ) -> Result<(), PecosError> {
        self.declarations.ensure_available(name, kind)?;
        Self::validate_variable_size(name, &data_type, size)?;

        if data_type == DataType::Qubits {
            self.add_quantum_register_with_kind(name, size, kind)?;
        } else {
            self.declarations.register(name, kind)?;
        }
        self.insert_variable_storage(name, data_type, size, metadata);
        Ok(())
    }

    /// Ensure classical runtime storage without making a PHIR-JSON declaration.
    pub(crate) fn ensure_classical_variable(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
    ) -> Result<(), PecosError> {
        if !self.has_variable(name) {
            Self::validate_variable_size(name, &data_type, size)?;
            self.insert_variable_storage(name, data_type, size, None);
        }
        Ok(())
    }

    pub(crate) fn validate_variable_size(
        name: &str,
        data_type: &DataType,
        size: usize,
    ) -> Result<(), PecosError> {
        if *data_type == DataType::Bool && size > 1 {
            return Err(PecosError::Input(format!(
                "Register {} declares bool with size {size}; use size 1 or an unsigned integer type",
                serde_json::json!(name)
            )));
        }
        // A signed size-S register is an `i(S+1)` integer (S data bits + a sign
        // bit), so `S + 1` must fit the backing width N. An unsigned size-S
        // register is a `u(S)` and needs `S <= N`. Fail fast otherwise.
        if data_type.is_integer() {
            let tw = data_type.bit_width();
            let needed = if data_type.is_signed() {
                size + 1
            } else {
                size
            };
            if needed > tw {
                let kind = if data_type.is_signed() {
                    "signed"
                } else {
                    "unsigned"
                };
                let limit = if data_type.is_signed() { tw - 1 } else { tw };
                return Err(PecosError::Input(format!(
                    "Register '{name}' declares {kind} type {data_type} with size {size}, \
                     which does not fit its {tw}-bit backing type. A size-{size} {kind} \
                     register needs {needed} bits. Use size <= {limit} or a wider integer type."
                )));
            }
        }

        Ok(())
    }

    fn insert_variable_storage(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    ) {
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
    }

    /// Define a quantum register without adding classical value storage.
    /// Quantum and classical declarations share one program-wide namespace.
    pub(crate) fn add_quantum_register(
        &mut self,
        name: &str,
        size: usize,
    ) -> Result<(), PecosError> {
        self.add_quantum_register_with_kind(name, size, DeclarationKind::Quantum)
    }

    fn add_quantum_register_with_kind(
        &mut self,
        name: &str,
        size: usize,
        kind: DeclarationKind,
    ) -> Result<(), PecosError> {
        self.declarations.ensure_available(name, kind)?;
        let end = self.num_qubits.checked_add(size).ok_or_else(|| {
            PecosError::Input(format!("Qubit count overflow defining register '{name}'"))
        })?;
        self.declarations.register(name, kind)?;
        self.quantum_registers
            .insert(name.to_string(), self.num_qubits..end);
        self.num_qubits = end;
        Ok(())
    }

    /// Resolve a register-local qubit index to its declaration-order global ID.
    ///
    /// # Errors
    /// Returns an error for an unknown quantum register or an out-of-bounds index.
    pub fn resolve_qubit(&self, name: &str, index: usize) -> Result<usize, PecosError> {
        let range = self.quantum_registers.get(name).ok_or_else(|| {
            PecosError::Input(format!(
                "Unknown quantum register '{name}'; declare it with qvar_define before use"
            ))
        })?;
        let size = range.end - range.start;
        if index >= size {
            return Err(PecosError::Input(format!(
                "Index {index} out of bounds for quantum register '{name}' of size {size}; use an index below {size}"
            )));
        }
        Ok(range.start + index)
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
            Err(PecosError::RuntimeUndefinedVariable {
                name: name.to_string(),
            })
        }
    }

    /// Resolve a declared classical register for Result or measurement operations.
    pub(crate) fn classical_info(&self, name: &str) -> Result<&VariableInfo, PecosError> {
        self.classical_info_for(name, "Result or measurement")
    }

    pub(crate) fn classical_info_for(
        &self,
        name: &str,
        operation: &str,
    ) -> Result<&VariableInfo, PecosError> {
        self.get_variable_info_opt(name)
            .filter(|info| info.data_type != DataType::Qubits)
            .ok_or_else(|| PecosError::Input(format!(
                "{operation}: unknown classical variable {}; declare it with cvar_define before use", serde_json::json!(name)
            )))
    }

    /// Validate a measurement against the declared size, not the backing integer width.
    pub(crate) fn validate_measurement_write(
        &self,
        name: &str,
        index: usize,
    ) -> Result<(), PecosError> {
        self.validate_classical_write(name, index, "Measurement")
    }

    fn validate_classical_write(
        &self,
        name: &str,
        index: usize,
        operation: &str,
    ) -> Result<(), PecosError> {
        let size = self.classical_info_for(name, operation)?.size;
        if index >= size {
            return Err(PecosError::Input(format!(
                "{operation} write {name}[{index}] is out of range for a register of declared size {size}; use an index below {size}"
            )));
        }
        Ok(())
    }

    pub(crate) fn validate_assignment_targets(
        &self,
        returns: &[super::ast::ArgItem],
    ) -> Result<(), PecosError> {
        for target in returns {
            match target {
                super::ast::ArgItem::Simple(name) => {
                    self.classical_info_for(name, "Assignment")?;
                }
                super::ast::ArgItem::Indexed((name, index)) => {
                    self.validate_classical_write(name, *index, "Assignment")?;
                }
                _ => {
                    return Err(PecosError::Input(format!(
                        "Assignment: invalid target {}; supply a declared variable name or [name, bit index]",
                        serde_json::to_value(target)
                            .expect("PHIR arguments contain only JSON-serializable values")
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_result_index(&self, reference: &ResultReference) -> Result<(), PecosError> {
        if let Some(index) = reference.index {
            let size = self
                .get_variable_info_opt(&reference.name)
                .map_or(64, |info| info.size);
            if index as usize >= size {
                return Err(PecosError::Input(format!(
                    "Result: bit index {index} is invalid for register {} of declared size {size}; use an index below {size}",
                    serde_json::json!(reference.name)
                )));
            }
        }
        Ok(())
    }

    /// Validate all sources before any copies, then validate destination names and indices.
    pub(crate) fn validate_result(
        &self,
        args: &[super::ast::ArgItem],
        returns: &[super::ast::ArgItem],
    ) -> Result<Vec<(ResultReference, ResultReference)>, PecosError> {
        super::operations::validate_result_counts(args.len(), returns.len())?;
        let sources = args
            .iter()
            .map(|arg| ResultReference::from_arg(arg, "args"))
            .collect::<Result<Vec<_>, _>>()?;
        let destinations = returns
            .iter()
            .map(|arg| ResultReference::from_arg(arg, "returns"))
            .collect::<Result<Vec<_>, _>>()?;
        for source in &sources {
            self.classical_info(&source.name)?;
            self.validate_result_index(source)?;
        }
        for destination in &destinations {
            if self.quantum_registers.contains_key(&destination.name) {
                return Err(PecosError::Input(format!(
                    "Result: destination {} is quantum; use a classical variable or a new export name",
                    serde_json::json!(destination.name)
                )));
            }
            self.validate_result_index(destination)?;
        }
        Ok(sources.into_iter().zip(destinations).collect())
    }

    /// Create Result destinations with the native whole-register or indexed storage rules.
    pub(crate) fn prepare_result_destination(
        &mut self,
        source: &ResultReference,
        destination: &ResultReference,
    ) -> Result<(), PecosError> {
        if !self.has_variable(&destination.name) {
            let (data_type, size) = if let Some(index) = destination.index {
                (DataType::U64, (index as usize + 1).max(32))
            } else {
                let info = self.classical_info(&source.name)?;
                (info.data_type.clone(), info.size)
            };
            self.ensure_classical_variable(&destination.name, data_type, size)?;
        }
        // Earlier pairs may have created this destination with a smaller size.
        self.validate_result_index(destination)
    }

    /// Copy Result values in source order, preserving bit indices and destination widths.
    pub(crate) fn apply_result(
        &mut self,
        args: &[super::ast::ArgItem],
        returns: &[super::ast::ArgItem],
    ) -> Result<(), PecosError> {
        for (source, destination) in self.validate_result(args, returns)? {
            self.copy_result_pair(&source, &destination)?;
            self.add_mapping(&destination.name, &destination.name)?;
        }
        Ok(())
    }

    fn copy_result_pair(
        &mut self,
        source: &ResultReference,
        destination: &ResultReference,
    ) -> Result<(), PecosError> {
        let value = if let Some(index) = source.index {
            u64::from(self.get_bit(&source.name, index as usize)?.0)
        } else {
            let value = self.get(&source.name).ok_or_else(|| {
                PecosError::Input(format!(
                    "Result: source {} has no value; declare it with cvar_define before use",
                    serde_json::json!(source.name)
                ))
            })?;
            if self.classical_info(&source.name)?.data_type.is_signed() {
                u64::from_ne_bytes(value.as_i64().to_ne_bytes())
            } else {
                value.as_u64()
            }
        };
        self.prepare_result_destination(source, destination)?;
        if let Some(index) = destination.index {
            self.set_bit(&destination.name, index as usize, value & 1)?;
        } else {
            self.set(&destination.name, value)?;
        }
        Ok(())
    }

    /// Gets metadata for a variable
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist.
    pub fn get_variable_info(&self, name: &str) -> Result<&VariableInfo, PecosError> {
        if let Some(&idx) = self.name_to_index.get(name) {
            Ok(&self.metadata[idx])
        } else {
            Err(PecosError::RuntimeUndefinedVariable {
                name: name.to_string(),
            })
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
            Err(PecosError::RuntimeUndefinedVariable {
                name: var_name.to_string(),
            })
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
            let updated = self.values[idx].with_bit_set(bit_index, bool_value)?;
            // Re-normalize to the register's declared i(S+1)/u(S) width so that
            // setting the sign bit -- or a bit above the declared width -- wraps
            // and sign-extends the same way a whole-value assignment does.
            let info = &self.metadata[idx];
            self.values[idx] = BitValue::from_u64(&info.data_type, info.size, updated.as_u64());
            Ok(())
        } else {
            Err(PecosError::RuntimeUndefinedVariable {
                name: var_name.to_string(),
            })
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
        self.num_qubits
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
        if let Some(outputs) = self.selected_outputs() {
            return outputs
                .into_iter()
                .filter_map(|(output, source)| {
                    self.get(&source).map(|value| (output, value.as_u32()))
                })
                .collect();
        }
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

    /// Copy a whole classical register, retaining the destination's declared width.
    ///
    /// # Errors
    /// Returns an error for an unknown classical source or a quantum destination.
    pub fn copy_variable(&mut self, src_name: &str, dst_name: &str) -> Result<(), PecosError> {
        let args = [super::ast::ArgItem::Simple(src_name.to_string())];
        let returns = [super::ast::ArgItem::Simple(dst_name.to_string())];
        for (source, destination) in self.validate_result(&args, &returns)? {
            self.copy_result_pair(&source, &destination)?;
        }
        Ok(())
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
    fn quantum_redeclaration_precedes_count_overflow() {
        let mut env = Environment::new();
        env.add_quantum_register("a", 1).unwrap();
        let error = env.add_quantum_register("a", usize::MAX).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Input error: Variable 'a' is already declared as quantum; cannot redeclare as quantum"
        );
    }

    #[test]
    fn test_environment_basic_operations() {
        let mut env = Environment::new();

        // Add variables
        env.add_variable("x", DataType::I32, 31).unwrap();
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
        env.add_variable("i8_var", DataType::I8, 7).unwrap();
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
        env.add_variable("source", DataType::I32, 31).unwrap();
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
