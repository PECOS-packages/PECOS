// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use ::bitvec::prelude::*;
use num_bigint::BigInt;
use pecos_core::bitvec;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Represents a data value that can be stored in a shot result.
///
/// This enum supports common numeric types and provides a flexible way to store
/// measurement outcomes and other data from quantum program execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Data {
    /// 8-bit unsigned integer
    U8(u8),
    /// 16-bit unsigned integer
    U16(u16),
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit unsigned integer
    U64(u64),
    /// 8-bit signed integer
    I8(i8),
    /// 16-bit signed integer
    I16(i16),
    /// 32-bit signed integer
    I32(i32),
    /// 64-bit signed integer
    I64(i64),
    /// 32-bit floating point
    F32(f32),
    /// 64-bit floating point
    F64(f64),
    /// String data
    String(String),
    /// Boolean value
    Bool(bool),
    /// Arbitrary precision integer
    BigInt(BigInt),
    /// Byte array for efficient binary data storage
    Bytes(Vec<u8>),
    /// Bit vector with indexed bit access
    BitVec(BitVec<u8, Lsb0>),
    /// JSON value for complex or dynamic data
    Json(JsonValue),
    /// Vector of data values (for tuples, arrays, multiple measurements, etc.)
    Vec(Vec<Data>),
}

impl Data {
    /// Create a Vec variant from a vector of Data values
    #[must_use]
    pub fn from_vec(values: Vec<Data>) -> Self {
        Self::Vec(values)
    }

    /// Create a Vec variant from a vector of i32 values
    #[must_use]
    pub fn from_i32_vec(values: Vec<i32>) -> Self {
        Self::Vec(values.into_iter().map(Data::I32).collect())
    }

    /// Create a Vec variant from a vector of u32 values
    #[must_use]
    pub fn from_u32_vec(values: Vec<u32>) -> Self {
        Self::Vec(values.into_iter().map(Data::U32).collect())
    }

    /// Create a Bytes variant from a Vec<u8>
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }

    /// Create a `BitVec` variant from a bitstring (string of '0' and '1' chars)
    /// Returns None if the string contains non-binary characters
    #[must_use]
    pub fn from_bitstring(bitstring: &str) -> Option<Self> {
        bitvec::from_bitstring(bitstring).map(Self::BitVec)
    }

    /// Create a Bytes variant from a bitstring (for backward compatibility)
    #[must_use]
    pub fn from_bitstring_as_bytes(bitstring: &str) -> Option<Self> {
        // Convert bitstring to bytes (8 bits per byte)
        let mut bytes = Vec::new();
        let chars: Vec<char> = bitstring.chars().collect();

        for chunk in chars.chunks(8) {
            let mut byte = 0u8;
            for (i, &ch) in chunk.iter().enumerate() {
                match ch {
                    '0' => {} // bit is already 0
                    '1' => byte |= 1 << (7 - i),
                    _ => return None, // Invalid character
                }
            }
            bytes.push(byte);
        }

        Some(Self::Bytes(bytes))
    }

    /// Convert to a bitstring representation
    #[must_use]
    pub fn to_bitstring(&self) -> Option<String> {
        match self {
            Self::Bytes(bytes) => {
                let mut result = String::with_capacity(bytes.len() * 8);
                for byte in bytes {
                    for i in (0..8).rev() {
                        result.push(if (byte >> i) & 1 == 1 { '1' } else { '0' });
                    }
                }
                Some(result)
            }
            Self::BitVec(bv) => Some(bitvec::to_bitstring(bv)),
            _ => None,
        }
    }

    /// Convert data to a value string suitable for JSON output
    ///
    /// For integer-like types (including `BitVec`), returns decimal representation.
    /// For other types, returns a sensible string representation.
    #[must_use]
    pub fn to_value_string(&self) -> String {
        match self {
            // Integer types -> decimal
            Self::U8(v) => v.to_string(),
            Self::U16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::U64(v) => v.to_string(),
            Self::I8(v) => v.to_string(),
            Self::I16(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::Bool(v) => if *v { "1" } else { "0" }.to_string(),
            Self::BitVec(bv) => {
                // Convert BitVec to decimal string
                bitvec::to_decimal_string(bv)
            }
            Self::BigInt(v) => v.to_string(),
            // Other types -> sensible string representation
            Self::F32(v) => v.to_string(),
            Self::F64(v) => v.to_string(),
            Self::String(v) => v.clone(),
            Self::Bytes(v) => format!("{v:?}"), // Could use hex or base64
            Self::Json(v) => v.to_string(),
            Self::Vec(v) => {
                let strings: Vec<String> = v.iter().map(Data::to_value_string).collect();
                format!("[{}]", strings.join(", "))
            }
        }
    }

    /// Try to convert the data to a u32 value if possible
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v) => Some(u32::from(*v)),
            Self::U16(v) => Some(u32::from(*v)),
            Self::U32(v) => Some(*v),
            Self::U64(v) => u32::try_from(*v).ok(),
            Self::I8(v) => u32::try_from(*v).ok(),
            Self::I16(v) => u32::try_from(*v).ok(),
            Self::I32(v) => u32::try_from(*v).ok(),
            Self::I64(v) => u32::try_from(*v).ok(),
            Self::Bool(v) => Some(u32::from(*v)),
            Self::Json(v) => v.as_u64().and_then(|n| u32::try_from(n).ok()),
            Self::BigInt(v) => u32::try_from(v).ok(),
            Self::Bytes(v) => {
                // Try to interpret first 4 bytes as little-endian u32
                if v.len() >= 4 {
                    Some(u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                } else {
                    None
                }
            }
            Self::BitVec(v) => {
                // Convert up to 32 bits to u32
                bitvec::to_u32(v)
            }
            _ => None,
        }
    }

    /// Get the inner vector if this is a Vec variant
    #[must_use]
    pub fn as_vec(&self) -> Option<&Vec<Data>> {
        match self {
            Self::Vec(v) => Some(v),
            _ => None,
        }
    }

    /// Convert Vec variant to vector of u32 values if possible
    #[must_use]
    pub fn as_u32_vec(&self) -> Option<Vec<u32>> {
        match self {
            Self::Vec(v) => {
                let mut result = Vec::with_capacity(v.len());
                for item in v {
                    match item.as_u32() {
                        Some(val) => result.push(val),
                        None => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Convert Vec variant to vector of i32 values if possible
    #[must_use]
    pub fn as_i32_vec(&self) -> Option<Vec<i32>> {
        match self {
            Self::Vec(v) => {
                let mut result = Vec::with_capacity(v.len());
                for item in v {
                    match item {
                        Data::I32(val) => result.push(*val),
                        Data::I16(val) => result.push(i32::from(*val)),
                        Data::I8(val) => result.push(i32::from(*val)),
                        Data::U8(val) => result.push(i32::from(*val)),
                        Data::U16(val) => result.push(i32::from(*val)),
                        Data::U32(val) => {
                            if let Ok(i) = i32::try_from(*val) {
                                result.push(i);
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Convert the Data to a JSON value
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        match self {
            Self::U8(v) => JsonValue::from(*v),
            Self::U16(v) => JsonValue::from(*v),
            Self::U32(v) => JsonValue::from(*v),
            Self::U64(v) => JsonValue::from(*v),
            Self::I8(v) => JsonValue::from(*v),
            Self::I16(v) => JsonValue::from(*v),
            Self::I32(v) => JsonValue::from(*v),
            Self::I64(v) => JsonValue::from(*v),
            Self::F32(v) => serde_json::Number::from_f64(f64::from(*v))
                .map_or(JsonValue::Null, JsonValue::Number),
            Self::F64(v) => {
                serde_json::Number::from_f64(*v).map_or(JsonValue::Null, JsonValue::Number)
            }
            Self::String(v) => JsonValue::from(v.clone()),
            Self::Bool(v) => JsonValue::from(*v),
            Self::BigInt(v) => JsonValue::from(v.to_string()),
            Self::Bytes(v) => JsonValue::Array(v.iter().map(|&b| JsonValue::from(b)).collect()),
            Self::BitVec(bv) => {
                // Convert BitVec to decimal integer
                let mut value = 0u64;
                for (i, bit) in bv.iter().enumerate() {
                    if *bit && i < 64 {
                        value |= 1u64 << i;
                    }
                }
                JsonValue::from(value)
            }
            Self::Json(v) => v.clone(),
            Self::Vec(v) => JsonValue::Array(v.iter().map(Data::to_json_value).collect()),
        }
    }
}

// Implement Display trait for Data instead of inherent to_string method
impl std::fmt::Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::BigInt(v) => write!(f, "{v}"),
            Self::Bytes(v) => write!(f, "{v:?}"), // Could also use hex or base64
            Self::BitVec(bv) => write!(f, "{}", bitvec::to_bitstring(bv)),
            Self::Json(v) => write!(f, "{v}"),
            Self::Vec(v) => {
                write!(f, "[")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
        }
    }
}
