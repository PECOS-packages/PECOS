//! Typed named outputs shared by the FFI writer and QIS reader.

/// Values accumulated under one result tag, with an explicit JSON element type.
/// Calls retain their declared type. Bool/integer mixing widens the entire tag
/// to I64/U64; pure-bool tags produce U32 shot data, widened tags I64/U64 data.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "values")]
pub enum NamedResult {
    #[serde(rename = "bool")]
    Bool(Vec<bool>),
    #[serde(rename = "i64")]
    I64(Vec<i64>),
    #[serde(rename = "u64")]
    U64(Vec<u64>),
    /// IEEE 754 bits on the wire: JSON numbers cannot represent NaN or infinity.
    #[serde(rename = "f64")]
    F64(#[serde(with = "float_bits")] Vec<f64>),
}

impl NamedResult {
    /// The element type, including for an empty array.
    #[must_use]
    pub fn element_type(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::I64(_) => "i64",
            Self::U64(_) => "u64",
            Self::F64(_) => "f64",
        }
    }

    /// Number of accumulated elements.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Whether this tag has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append compatible declared types, widening bools to the integer type.
    ///
    /// # Errors
    /// Returns the existing and incoming types on a mismatch, without mutation.
    pub fn append(&mut self, values: Self) -> Result<(), (&'static str, &'static str)> {
        let replacement = match (&mut *self, values) {
            (Self::Bool(target), Self::Bool(values)) => {
                target.extend(values);
                None
            }
            (Self::I64(target), Self::I64(values)) => {
                target.extend(values);
                None
            }
            (Self::U64(target), Self::U64(values)) => {
                target.extend(values);
                None
            }
            (Self::F64(target), Self::F64(values)) => {
                target.extend(values);
                None
            }
            (Self::I64(target), Self::Bool(values)) => {
                target.extend(values.into_iter().map(i64::from));
                None
            }
            (Self::U64(target), Self::Bool(values)) => {
                target.extend(values.into_iter().map(u64::from));
                None
            }
            (Self::Bool(target), Self::I64(values)) => Some(Self::I64(
                target
                    .iter()
                    .copied()
                    .map(i64::from)
                    .chain(values)
                    .collect(),
            )),
            (Self::Bool(target), Self::U64(values)) => Some(Self::U64(
                target
                    .iter()
                    .copied()
                    .map(u64::from)
                    .chain(values)
                    .collect(),
            )),
            (target, values) => return Err((target.element_type(), values.element_type())),
        };
        if let Some(replacement) = replacement {
            *self = replacement;
        }
        Ok(())
    }
}

/// A program termination or output error recorded on its execution context.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProgramError {
    Exit { code: i32, message: String },
    Panic { code: i32, message: String },
    NamedResult(String),
    InvalidInput { entry: String, detail: String },
}

impl std::fmt::Display for ProgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exit { code, message } => {
                write!(f, "QIS program exit: code={code}, message={message}")
            }
            Self::Panic { code, message } => {
                write!(f, "QIS program panic: code={code}, message={message}")
            }
            Self::NamedResult(message) => f.write_str(message),
            Self::InvalidInput { entry, detail } => {
                write!(f, "QIS invalid FFI input in {entry}: {detail}")
            }
        }
    }
}

mod float_bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        Vec::<u64>::deserialize(deserializer)
            .map(|bits| bits.into_iter().map(f64::from_bits).collect())
    }
}
