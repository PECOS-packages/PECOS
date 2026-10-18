//! Shared JSON validation for classical declarations and Result references.
use super::{
    ast::{ArgItem, infer_size, register_size},
    environment::{DataType, Environment},
};
use pecos_core::errors::PecosError;
use serde_json::Value;
use std::fmt;

/// A malformed Result reference, retaining its Python type/value distinction.
#[derive(Debug)]
pub struct ResultReferenceError {
    /// True for a wrong JSON type; false for an invalid integer index value.
    pub wrong_type: bool,
    message: String,
}
impl fmt::Display for ResultReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ResultReferenceError {}

/// Decode a Result reference without losing the offending JSON value.
///
/// # Errors
/// Rejects wrong types and invalid indices with a named, actionable error.
pub fn result_reference(value: &Value, field: &str) -> Result<ArgItem, PecosError> {
    decode_result_reference(value, field).map_err(|error| PecosError::Input(error.to_string()))
}

fn decode_result_reference(value: &Value, field: &str) -> Result<ArgItem, ResultReferenceError> {
    if let Some(name) = value.as_str() {
        return Ok(ArgItem::Simple(name.to_string()));
    }
    let mut wrong_type = true;
    if let Some(parts) = value.as_array()
        && parts.len() == 2
        && let Some(name) = parts[0].as_str()
    {
        if let Some(index) = parts[1].as_u64() {
            if let Ok(index) = u32::try_from(index) {
                return Ok(ArgItem::Indexed((name.to_string(), index as usize)));
            }
            wrong_type = false;
        } else if parts[1].as_i64().is_some() {
            wrong_type = false;
        }
    }
    Err(ResultReferenceError {
        wrong_type,
        message: format!(
            "Result: invalid {field} entry {value}; supply a variable name or [name, integer bit index from 0 to 63]"
        ),
    })
}

impl ResultReferenceError {
    /// Recover the type/value distinction from a canonical Input diagnostic.
    ///
    /// `PecosError::Input` owns a string. Decode only this validator's exact
    /// diagnostic, reusing its rules instead of classifying JSON types again.
    #[must_use]
    pub fn from_input_message(message: &str) -> Option<Self> {
        let body = message.strip_prefix("Result: invalid ")?;
        let (field, body) = body.split_once(" entry ")?;
        let encoded = body
            .strip_suffix("; supply a variable name or [name, integer bit index from 0 to 63]")?;
        let value: Value = serde_json::from_str(encoded).ok()?;
        let error = decode_result_reference(&value, field).err()?;
        (error.to_string() == message).then_some(error)
    }
}

pub(crate) fn result_field(op: &Value, field: &str) -> Result<Vec<ArgItem>, PecosError> {
    let value = op.get(field).ok_or_else(|| {
        PecosError::Input(format!(
            "Result: missing {field}; supply an array of variable references"
        ))
    })?;
    let values = value.as_array().ok_or_else(|| {
        PecosError::Input(format!(
            "Result: invalid {field} {value}; supply an array of variable references"
        ))
    })?;
    values
        .iter()
        .map(|value| result_reference(value, field))
        .collect()
}

/// Validate Result syntax, declaration size fields and bool widths before AST/schema decoding.
///
/// # Errors
/// Returns the same errors as direct Result/declaration validation.
pub fn validate_classical_inputs(program: &Value) -> Result<(), PecosError> {
    fn visit(ops: &[Value]) -> Result<(), PecosError> {
        for op in ops {
            if op.get("cop").and_then(Value::as_str) == Some("Result") {
                result_field(op, "args")?;
                result_field(op, "returns")?;
            }
            let data = op.get("data").and_then(Value::as_str);
            if matches!(data, Some("cvar_define" | "qvar_define"))
                && let Some(name) = op.get("variable").and_then(Value::as_str)
            {
                let size = register_size(name, op.get("size"))?;
                if data == Some("cvar_define")
                    && op.get("data_type").and_then(Value::as_str) == Some("bool")
                {
                    Environment::validate_variable_size(
                        name,
                        &DataType::Bool,
                        infer_size("bool", size),
                    )?;
                }
            }
            if op.get("block").is_some() {
                for field in ["ops", "true_branch", "false_branch"] {
                    if let Some(nested) = op.get(field).and_then(Value::as_array) {
                        visit(nested)?;
                    }
                }
            }
        }
        Ok(())
    }
    if let Some(ops) = program.get("ops").and_then(Value::as_array) {
        visit(ops)?;
    }
    Ok(())
}
