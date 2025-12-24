// Copyright 2025 The PECOS Developers
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

//! Vectorized representation of data for columnar operations.

use super::data::Data;
use bitvec::prelude::*;
use num_bigint::BigInt;
use pecos_core::errors::PecosError;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Represents a vector of data values of the same type.
///
/// This enum mirrors the `Data` enum but contains vectors for each variant,
/// enabling efficient columnar operations on homogeneous data sets.
///
/// # Example
/// ```
/// use pecos_engines::{DataVec, Data};
///
/// // Create a DataVec from a vector of Data values
/// let data_values = vec![Data::U32(1), Data::U32(2), Data::U32(3)];
/// let data_vec = DataVec::from_data_vec(data_values).unwrap();
///
/// // Access and manipulate the data
/// if let DataVec::U32(ref values) = data_vec {
///     assert_eq!(values.len(), 3);
///     assert_eq!(values[0], 1);
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DataVec {
    /// Vector of 8-bit unsigned integers
    U8(Vec<u8>),
    /// Vector of 16-bit unsigned integers
    U16(Vec<u16>),
    /// Vector of 32-bit unsigned integers
    U32(Vec<u32>),
    /// Vector of 64-bit unsigned integers
    U64(Vec<u64>),
    /// Vector of 8-bit signed integers
    I8(Vec<i8>),
    /// Vector of 16-bit signed integers
    I16(Vec<i16>),
    /// Vector of 32-bit signed integers
    I32(Vec<i32>),
    /// Vector of 64-bit signed integers
    I64(Vec<i64>),
    /// Vector of 32-bit floating point values
    F32(Vec<f32>),
    /// Vector of 64-bit floating point values
    F64(Vec<f64>),
    /// Vector of strings
    String(Vec<String>),
    /// Vector of boolean values
    Bool(Vec<bool>),
    /// Vector of arbitrary precision integers
    BigInt(Vec<BigInt>),
    /// Vector of byte arrays
    Bytes(Vec<Vec<u8>>),
    /// Vector of bit vectors
    BitVec(Vec<BitVec<u8, Lsb0>>),
    /// Vector of JSON values
    Json(Vec<JsonValue>),
}

impl DataVec {
    /// Get the length of the vector
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::BigInt(v) => v.len(),
            Self::Bytes(v) => v.len(),
            Self::BitVec(v) => v.len(),
            Self::Json(v) => v.len(),
        }
    }

    /// Check if the vector is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a `Data` value to the vector
    ///
    /// # Errors
    /// Returns an error if the data type doesn't match the vector variant
    pub fn push(&mut self, data: Data) -> Result<(), PecosError> {
        match (self, data) {
            (Self::U8(v), Data::U8(val)) => v.push(val),
            (Self::U16(v), Data::U16(val)) => v.push(val),
            (Self::U32(v), Data::U32(val)) => v.push(val),
            (Self::U64(v), Data::U64(val)) => v.push(val),
            (Self::I8(v), Data::I8(val)) => v.push(val),
            (Self::I16(v), Data::I16(val)) => v.push(val),
            (Self::I32(v), Data::I32(val)) => v.push(val),
            (Self::I64(v), Data::I64(val)) => v.push(val),
            (Self::F32(v), Data::F32(val)) => v.push(val),
            (Self::F64(v), Data::F64(val)) => v.push(val),
            (Self::String(v), Data::String(val)) => v.push(val),
            (Self::Bool(v), Data::Bool(val)) => v.push(val),
            (Self::BigInt(v), Data::BigInt(val)) => v.push(val),
            (Self::Bytes(v), Data::Bytes(val)) => v.push(val),
            (Self::BitVec(v), Data::BitVec(val)) => v.push(val),
            (Self::Json(v), Data::Json(val)) => v.push(val),
            _ => {
                return Err(PecosError::Processing(
                    "Data type mismatch when pushing to DataVec".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Get the element at the specified index as a `Data` value
    ///
    /// Returns `None` if the index is out of bounds
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Data> {
        match self {
            Self::U8(v) => v.get(index).map(|&val| Data::U8(val)),
            Self::U16(v) => v.get(index).map(|&val| Data::U16(val)),
            Self::U32(v) => v.get(index).map(|&val| Data::U32(val)),
            Self::U64(v) => v.get(index).map(|&val| Data::U64(val)),
            Self::I8(v) => v.get(index).map(|&val| Data::I8(val)),
            Self::I16(v) => v.get(index).map(|&val| Data::I16(val)),
            Self::I32(v) => v.get(index).map(|&val| Data::I32(val)),
            Self::I64(v) => v.get(index).map(|&val| Data::I64(val)),
            Self::F32(v) => v.get(index).map(|&val| Data::F32(val)),
            Self::F64(v) => v.get(index).map(|&val| Data::F64(val)),
            Self::String(v) => v.get(index).map(|val| Data::String(val.clone())),
            Self::Bool(v) => v.get(index).map(|&val| Data::Bool(val)),
            Self::BigInt(v) => v.get(index).map(|val| Data::BigInt(val.clone())),
            Self::Bytes(v) => v.get(index).map(|val| Data::Bytes(val.clone())),
            Self::BitVec(v) => v.get(index).map(|val| Data::BitVec(val.clone())),
            Self::Json(v) => v.get(index).map(|val| Data::Json(val.clone())),
        }
    }

    /// Create a `DataVec` from a vector of `Data` values
    ///
    /// # Errors
    /// Returns an error if the vector is empty or contains mixed data types
    pub fn from_data_vec(data: Vec<Data>) -> Result<Self, PecosError> {
        if data.is_empty() {
            return Err(PecosError::Processing(
                "Cannot create DataVec from empty vector".to_string(),
            ));
        }

        // Check the type of the first element and create the appropriate vector
        let mut result = match &data[0] {
            Data::U8(_) => Self::U8(Vec::with_capacity(data.len())),
            Data::U16(_) => Self::U16(Vec::with_capacity(data.len())),
            Data::U32(_) => Self::U32(Vec::with_capacity(data.len())),
            Data::U64(_) => Self::U64(Vec::with_capacity(data.len())),
            Data::I8(_) => Self::I8(Vec::with_capacity(data.len())),
            Data::I16(_) => Self::I16(Vec::with_capacity(data.len())),
            Data::I32(_) => Self::I32(Vec::with_capacity(data.len())),
            Data::I64(_) => Self::I64(Vec::with_capacity(data.len())),
            Data::F32(_) => Self::F32(Vec::with_capacity(data.len())),
            Data::F64(_) => Self::F64(Vec::with_capacity(data.len())),
            Data::String(_) => Self::String(Vec::with_capacity(data.len())),
            Data::Bool(_) => Self::Bool(Vec::with_capacity(data.len())),
            Data::BigInt(_) => Self::BigInt(Vec::with_capacity(data.len())),
            Data::Bytes(_) => Self::Bytes(Vec::with_capacity(data.len())),
            Data::BitVec(_) => Self::BitVec(Vec::with_capacity(data.len())),
            Data::Json(_) => Self::Json(Vec::with_capacity(data.len())),
            Data::Vec(_) => {
                // For nested vectors, we need to create a nested DataVec
                // For now, return an error as this is complex to handle
                return Err(PecosError::Processing(
                    "Cannot create DataVec from nested vectors".to_string(),
                ));
            }
        };

        // Push all elements, checking for type consistency
        for item in data {
            result.push(item)?;
        }

        Ok(result)
    }

    /// Convert the `DataVec` to a vector of `Data` values
    #[must_use]
    pub fn to_data_vec(&self) -> Vec<Data> {
        let mut result = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            if let Some(data) = self.get(i) {
                result.push(data);
            }
        }
        result
    }

    /// Convert the `DataVec` to a JSON array
    ///
    /// Each element is converted to its appropriate JSON representation
    #[must_use]
    pub fn to_json_array(&self) -> JsonValue {
        match self {
            Self::U8(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::U16(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::U32(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::U64(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::I8(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::I16(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::I32(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::I64(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::F32(v) => JsonValue::Array(
                v.iter()
                    .map(|&x| {
                        serde_json::Number::from_f64(f64::from(x))
                            .map_or(JsonValue::Null, JsonValue::Number)
                    })
                    .collect(),
            ),
            Self::F64(v) => JsonValue::Array(
                v.iter()
                    .map(|&x| {
                        serde_json::Number::from_f64(x).map_or(JsonValue::Null, JsonValue::Number)
                    })
                    .collect(),
            ),
            Self::String(v) => {
                JsonValue::Array(v.iter().map(|x| JsonValue::from(x.clone())).collect())
            }
            Self::Bool(v) => JsonValue::Array(v.iter().map(|&x| JsonValue::from(x)).collect()),
            Self::BigInt(v) => {
                JsonValue::Array(v.iter().map(|x| JsonValue::from(x.to_string())).collect())
            }
            Self::Bytes(v) => JsonValue::Array(
                v.iter()
                    .map(|bytes| {
                        JsonValue::Array(bytes.iter().map(|&b| JsonValue::from(b)).collect())
                    })
                    .collect(),
            ),
            Self::BitVec(v) => JsonValue::Array(
                v.iter()
                    .map(|bv| {
                        // Convert BitVec to decimal integer
                        let mut value = 0u64;
                        for (i, bit) in bv.iter().enumerate() {
                            if *bit && i < 64 {
                                value |= 1u64 << i;
                            }
                        }
                        JsonValue::from(value)
                    })
                    .collect(),
            ),
            Self::Json(v) => JsonValue::Array(v.clone()),
        }
    }

    /// Create a new empty `DataVec` of the specified type
    ///
    /// # Example
    /// ```
    /// use pecos_engines::{DataVec, DataVecType};
    ///
    /// let vec = DataVec::new_empty(DataVecType::U32);
    /// assert!(vec.is_empty());
    /// ```
    #[must_use]
    pub fn new_empty(data_type: DataVecType) -> Self {
        match data_type {
            DataVecType::U8 => Self::U8(Vec::new()),
            DataVecType::U16 => Self::U16(Vec::new()),
            DataVecType::U32 => Self::U32(Vec::new()),
            DataVecType::U64 => Self::U64(Vec::new()),
            DataVecType::I8 => Self::I8(Vec::new()),
            DataVecType::I16 => Self::I16(Vec::new()),
            DataVecType::I32 => Self::I32(Vec::new()),
            DataVecType::I64 => Self::I64(Vec::new()),
            DataVecType::F32 => Self::F32(Vec::new()),
            DataVecType::F64 => Self::F64(Vec::new()),
            DataVecType::String => Self::String(Vec::new()),
            DataVecType::Bool => Self::Bool(Vec::new()),
            DataVecType::BigInt => Self::BigInt(Vec::new()),
            DataVecType::Bytes => Self::Bytes(Vec::new()),
            DataVecType::BitVec => Self::BitVec(Vec::new()),
            DataVecType::Json => Self::Json(Vec::new()),
        }
    }

    /// Get the type of this `DataVec`
    #[must_use]
    pub fn data_type(&self) -> DataVecType {
        match self {
            Self::U8(_) => DataVecType::U8,
            Self::U16(_) => DataVecType::U16,
            Self::U32(_) => DataVecType::U32,
            Self::U64(_) => DataVecType::U64,
            Self::I8(_) => DataVecType::I8,
            Self::I16(_) => DataVecType::I16,
            Self::I32(_) => DataVecType::I32,
            Self::I64(_) => DataVecType::I64,
            Self::F32(_) => DataVecType::F32,
            Self::F64(_) => DataVecType::F64,
            Self::String(_) => DataVecType::String,
            Self::Bool(_) => DataVecType::Bool,
            Self::BigInt(_) => DataVecType::BigInt,
            Self::Bytes(_) => DataVecType::Bytes,
            Self::BitVec(_) => DataVecType::BitVec,
            Self::Json(_) => DataVecType::Json,
        }
    }
}

/// Type identifier for `DataVec` variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataVecType {
    /// 8-bit unsigned integer type
    U8,
    /// 16-bit unsigned integer type
    U16,
    /// 32-bit unsigned integer type
    U32,
    /// 64-bit unsigned integer type
    U64,
    /// 8-bit signed integer type
    I8,
    /// 16-bit signed integer type
    I16,
    /// 32-bit signed integer type
    I32,
    /// 64-bit signed integer type
    I64,
    /// 32-bit floating point type
    F32,
    /// 64-bit floating point type
    F64,
    /// String type
    String,
    /// Boolean type
    Bool,
    /// Arbitrary precision integer type
    BigInt,
    /// Byte array type
    Bytes,
    /// Bit vector type
    BitVec,
    /// JSON value type
    Json,
}

impl DataVecType {
    /// Get the type from a `Data` value
    #[must_use]
    pub fn from_data(data: &Data) -> Self {
        match data {
            Data::U8(_) => Self::U8,
            Data::U16(_) => Self::U16,
            Data::U32(_) => Self::U32,
            Data::U64(_) => Self::U64,
            Data::I8(_) => Self::I8,
            Data::I16(_) => Self::I16,
            Data::I32(_) => Self::I32,
            Data::I64(_) => Self::I64,
            Data::F32(_) => Self::F32,
            Data::F64(_) => Self::F64,
            Data::String(_) => Self::String,
            Data::Bool(_) => Self::Bool,
            Data::BigInt(_) => Self::BigInt,
            Data::Bytes(_) => Self::Bytes,
            Data::BitVec(_) => Self::BitVec,
            Data::Json(_) => Self::Json,
            Data::Vec(_) => {
                // For nested vectors, we can't determine a single type
                // This is a limitation of the current type system
                Self::Json // Use Json as a fallback for complex types
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_vec_creation() {
        // Test creating from homogeneous data
        let data = vec![Data::U32(1), Data::U32(2), Data::U32(3)];
        let data_vec = DataVec::from_data_vec(data).unwrap();

        assert_eq!(data_vec.len(), 3);
        assert!(!data_vec.is_empty());
        assert_eq!(data_vec.get(0), Some(Data::U32(1)));
        assert_eq!(data_vec.get(1), Some(Data::U32(2)));
        assert_eq!(data_vec.get(2), Some(Data::U32(3)));
        assert_eq!(data_vec.get(3), None);
    }

    #[test]
    fn test_data_vec_push() {
        let mut data_vec = DataVec::new_empty(DataVecType::F64);

        assert!(data_vec.push(Data::F64(std::f64::consts::PI)).is_ok());
        assert!(data_vec.push(Data::F64(2.71)).is_ok());
        assert_eq!(data_vec.len(), 2);

        // Test type mismatch
        assert!(data_vec.push(Data::U32(42)).is_err());
    }

    #[test]
    fn test_mixed_types_error() {
        let data = vec![Data::U32(1), Data::F64(2.0), Data::U32(3)];
        let result = DataVec::from_data_vec(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_vec_error() {
        let data: Vec<Data> = vec![];
        let result = DataVec::from_data_vec(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_json_array() {
        // Test U32
        let data_vec = DataVec::U32(vec![1, 2, 3]);
        let json = data_vec.to_json_array();
        assert_eq!(json, serde_json::json!([1, 2, 3]));

        // Test String
        let data_vec = DataVec::String(vec!["a".to_string(), "b".to_string()]);
        let json = data_vec.to_json_array();
        assert_eq!(json, serde_json::json!(["a", "b"]));

        // Test BitVec
        let mut bv1 = BitVec::<u8, Lsb0>::new();
        bv1.push(true);
        bv1.push(false);
        bv1.push(true);
        let mut bv2 = BitVec::<u8, Lsb0>::new();
        bv2.push(false);
        bv2.push(true);
        let data_vec = DataVec::BitVec(vec![bv1, bv2]);
        let json = data_vec.to_json_array();
        assert_eq!(json, serde_json::json!([5, 2])); // 101 = 5, 010 = 2
    }

    #[test]
    fn test_bitvec_support() {
        let mut bv1 = BitVec::<u8, Lsb0>::new();
        bv1.push(true);
        bv1.push(false);

        let mut bv2 = BitVec::<u8, Lsb0>::new();
        bv2.push(false);
        bv2.push(true);

        let data = vec![Data::BitVec(bv1.clone()), Data::BitVec(bv2.clone())];
        let data_vec = DataVec::from_data_vec(data).unwrap();

        if let DataVec::BitVec(ref vecs) = data_vec {
            assert_eq!(vecs.len(), 2);
            assert_eq!(vecs[0], bv1);
            assert_eq!(vecs[1], bv2);
        } else {
            panic!("Expected BitVec variant");
        }
    }

    #[test]
    fn test_conversion_roundtrip() {
        let original_data = vec![
            Data::String("hello".to_string()),
            Data::String("world".to_string()),
        ];

        let data_vec = DataVec::from_data_vec(original_data.clone()).unwrap();
        let converted_back = data_vec.to_data_vec();

        assert_eq!(original_data, converted_back);
    }

    #[test]
    fn test_data_type() {
        let data_vec = DataVec::Bool(vec![true, false]);
        assert_eq!(data_vec.data_type(), DataVecType::Bool);

        let data = Data::Bool(true);
        assert_eq!(DataVecType::from_data(&data), DataVecType::Bool);
    }

    #[test]
    fn test_bigint_support() {
        let big1 = BigInt::from(u128::MAX);
        let big2 = BigInt::from(42);

        let data = vec![Data::BigInt(big1.clone()), Data::BigInt(big2.clone())];
        let data_vec = DataVec::from_data_vec(data).unwrap();

        if let DataVec::BigInt(ref values) = data_vec {
            assert_eq!(values[0], big1);
            assert_eq!(values[1], big2);
        } else {
            panic!("Expected BigInt variant");
        }

        let json = data_vec.to_json_array();
        assert!(json.is_array());
    }

    #[test]
    fn test_bytes_support() {
        let bytes1 = vec![0xFF, 0x00, 0xAB];
        let bytes2 = vec![0x12, 0x34];

        let data = vec![Data::Bytes(bytes1.clone()), Data::Bytes(bytes2.clone())];
        let data_vec = DataVec::from_data_vec(data).unwrap();

        if let DataVec::Bytes(ref vecs) = data_vec {
            assert_eq!(vecs[0], bytes1);
            assert_eq!(vecs[1], bytes2);
        } else {
            panic!("Expected Bytes variant");
        }
    }

    #[test]
    fn test_json_support() {
        let json1 = serde_json::json!({"a": 1, "b": "hello"});
        let json2 = serde_json::json!([1, 2, 3]);

        let data = vec![Data::Json(json1.clone()), Data::Json(json2.clone())];
        let data_vec = DataVec::from_data_vec(data).unwrap();

        if let DataVec::Json(ref values) = data_vec {
            assert_eq!(values[0], json1);
            assert_eq!(values[1], json2);
        } else {
            panic!("Expected Json variant");
        }
    }
}
