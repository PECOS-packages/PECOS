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

//! Columnar representation of shot data for efficient analysis.

use super::data::Data;
use super::data_vec::DataVec;
use bitvec::prelude::*;
use pecos_core::errors::PecosError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt;

/// A columnar representation of shot data.
///
/// `ShotMap` transforms the row-based shot data (where each shot contains multiple registers)
/// into column-based data (where each register has a vector of values across all shots).
/// This format is more efficient for analyzing specific registers across many shots.
///
/// # Example
/// ```
/// # use pecos_results::{ShotVec, Shot, Data};
/// # use pecos_results::{BitVecDisplayFormat, ShotMapDisplayExt};
/// # use pecos_core::errors::PecosError;
/// # fn main() -> Result<(), PecosError> {
/// let mut shot_vec = ShotVec::new();
/// let mut shot = Shot::default();
/// shot.add_register("q", 5, 3);
/// shot.data.insert("phase".to_string(), Data::F64(0.5));
/// shot_vec.shots.push(shot);
///
/// let shot_map = shot_vec.try_as_shot_map()?;
///
/// // Access all values for a specific register
/// if let Some(values) = shot_map.get("q") {
///     println!("Found {} values for register", values.len());
/// }
///
/// // Display with different formats
/// println!("{}", shot_map.display()); // Default decimal for BitVecs
/// println!("{}", shot_map.display().bitvec_binary()); // Binary format for BitVecs
/// println!("{}", shot_map.display().bitvec_hex()); // Hexadecimal format for BitVecs
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotMap {
    /// Map from register names to typed data vectors
    data: BTreeMap<String, DataVec>,
    /// Number of shots (all columns have the same length)
    num_shots: usize,
}

impl ShotMap {
    /// Create a new `ShotMap` from a `BTreeMap` of columnar data
    ///
    /// # Errors
    /// Returns an error if:
    /// - The columns have different lengths
    /// - A column contains mixed data types
    pub(crate) fn new(data: BTreeMap<String, Vec<Data>>) -> Result<Self, PecosError> {
        let num_shots = if data.is_empty() {
            0
        } else {
            // SAFETY: we just checked `data.is_empty()` above
            let first_len = data.values().next().expect("data is non-empty").len();

            // Verify all columns have the same length
            for (name, values) in &data {
                if values.len() != first_len {
                    return Err(PecosError::Processing(format!(
                        "Column '{}' has {} values but expected {}",
                        name,
                        values.len(),
                        first_len
                    )));
                }
            }

            first_len
        };

        // Convert Vec<Data> to DataVec for each column
        let mut typed_data = BTreeMap::new();
        for (name, values) in data {
            let data_vec = DataVec::from_data_vec(values)?;
            typed_data.insert(name, data_vec);
        }

        Ok(Self {
            data: typed_data,
            num_shots,
        })
    }

    /// Get the number of shots
    #[must_use]
    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    /// Get the number of registers
    #[must_use]
    pub fn num_registers(&self) -> usize {
        self.data.len()
    }

    /// Get all register names
    #[must_use]
    pub fn register_names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.data.keys().map(std::string::String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Get values for a specific register
    #[must_use]
    pub fn get(&self, register: &str) -> Option<&DataVec> {
        self.data.get(register)
    }

    /// Check if a register exists
    #[must_use]
    pub fn contains_register(&self, register: &str) -> bool {
        self.data.contains_key(register)
    }

    /// Iterate over all registers and their values
    pub fn iter(&self) -> impl Iterator<Item = (&String, &DataVec)> {
        self.data.iter()
    }

    /// Get a reference to the internal data
    #[must_use]
    pub fn as_map(&self) -> &BTreeMap<String, DataVec> {
        &self.data
    }

    /// Consume the `ShotMap` and return the internal `BTreeMap`
    #[must_use]
    pub fn into_map(self) -> BTreeMap<String, DataVec> {
        self.data
    }

    /// Try to get F64 values from a register
    ///
    /// # Returns
    /// - `Ok(Vec<f64>)` if the register exists and contains F64 data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{Data, Shot, ShotMap, ShotVec};
    /// # use pecos_core::errors::PecosError;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.data.insert("phase".to_string(), Data::F64(0.5));
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// // Extract all F64 values from the "phase" register
    /// match shot_map.try_f64s("phase") {
    ///     Ok(phases) => println!("Found {} phase values", phases.len()),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_f64s(&self, register: &str) -> Result<Vec<f64>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::F64(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain F64 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get boolean values from a register
    ///
    /// # Returns
    /// - `Ok(Vec<bool>)` if the register exists and contains Bool data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bools(&self, register: &str) -> Result<Vec<bool>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::Bool(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain Bool data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get U32 values from a register
    ///
    /// # Returns
    /// - `Ok(Vec<u32>)` if the register exists and contains U32 data
    /// - `Err` if the register doesn't exist or contains a different data type
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_u32s(&self, register: &str) -> Result<Vec<u32>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::U32(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain U32 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BitVec` values as u64 integers
    ///
    /// Converts each `BitVec` to a u64 value. `BitVecs` with more than 64 bits
    /// will have their higher bits truncated.
    ///
    /// # Returns
    /// - `Ok(Vec<u64>)` if the register exists and contains `BitVec` data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// # use pecos_core::errors::PecosError;
    /// # use bitvec::prelude::*;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.add_register("qubits", 5, 3); // value 5 with 3 bits
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// // Extract quantum measurements as integers
    /// match shot_map.try_bits_as_u64("qubits") {
    ///     Ok(measurements) => {
    ///         for (shot, value) in measurements.iter().enumerate() {
    ///             println!("Shot {}: {}", shot, value);
    ///         }
    ///     }
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bits_as_u64(&self, register: &str) -> Result<Vec<u64>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::BitVec(vecs)) => Ok(vecs
                .iter()
                .map(|bv| {
                    let mut value = 0u64;
                    for (i, bit) in bv.iter().enumerate().take(64) {
                        if *bit {
                            value |= 1u64 << i;
                        }
                    }
                    value
                })
                .collect()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BitVec data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BitVec` values as binary strings
    ///
    /// Returns strings like "1101" where the leftmost character represents
    /// the most significant bit.
    ///
    /// # Returns
    /// - `Ok(Vec<String>)` if the register exists and contains `BitVec` data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// # use pecos_core::errors::PecosError;
    /// # use bitvec::prelude::*;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.add_register("qubits", 13, 4); // value 13 (1101) with 4 bits
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// // Extract quantum states as binary strings
    /// if let Ok(states) = shot_map.try_bits_as_binary("qubits") {
    ///     for (shot, state) in states.iter().enumerate() {
    ///         println!("Shot {}: |{}⟩", shot, state);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bits_as_binary(&self, register: &str) -> Result<Vec<String>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::BitVec(vecs)) => Ok(vecs
                .iter()
                .map(|bv| {
                    let mut result = String::with_capacity(bv.len());
                    // Iterate in reverse to put MSB first
                    for i in (0..bv.len()).rev() {
                        result.push(if bv[i] { '1' } else { '0' });
                    }
                    result
                })
                .collect()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BitVec data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BitVec` values as `BigUint` integers
    ///
    /// Converts each `BitVec` to a `BigUint` value, supporting arbitrary precision
    /// for `BitVecs` of any size.
    ///
    /// # Returns
    /// - `Ok(Vec<BigUint>)` if the register exists and contains `BitVec` data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// # use pecos_core::errors::PecosError;
    /// # use num_bigint::BigUint;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.add_register("large_reg", 0, 100); // 100-bit register
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// // Extract values as BigUint for arbitrary precision
    /// if let Ok(values) = shot_map.try_bits_as_biguint("large_reg") {
    ///     for (i, val) in values.iter().enumerate() {
    ///         println!("Shot {}: {}", i, val);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bits_as_biguint(
        &self,
        register: &str,
    ) -> Result<Vec<num_bigint::BigUint>, PecosError> {
        use num_bigint::BigUint;

        match self.data.get(register) {
            Some(DataVec::BitVec(vecs)) => Ok(vecs
                .iter()
                .map(|bv| {
                    if bv.is_empty() {
                        BigUint::from(0u32)
                    } else {
                        let bytes = bv.as_raw_slice();
                        BigUint::from_bytes_le(bytes)
                    }
                })
                .collect()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BitVec data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BitVec` values as decimal strings
    ///
    /// Converts each `BitVec` to its decimal representation as a string.
    /// This supports arbitrary precision for `BitVecs` of any size.
    ///
    /// # Returns
    /// - `Ok(Vec<String>)` if the register exists and contains `BitVec` data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// # use pecos_core::errors::PecosError;
    /// # use bitvec::prelude::*;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.add_register("measurement", 255, 8); // value 255 with 8 bits
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// // Extract measurement results as decimal strings
    /// if let Ok(results) = shot_map.try_bits_as_decimal("measurement") {
    ///     for (shot, value) in results.iter().enumerate() {
    ///         println!("Shot {}: {}", shot, value);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bits_as_decimal(&self, register: &str) -> Result<Vec<String>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::BitVec(vecs)) => Ok(vecs
                .iter()
                .map(|bv| {
                    if bv.is_empty() {
                        "0".to_string()
                    } else if bv.len() <= 64 {
                        // For small BitVecs, use u64
                        let mut value = 0u64;
                        for (i, bit) in bv.iter().enumerate() {
                            if *bit {
                                value |= 1u64 << i;
                            }
                        }
                        value.to_string()
                    } else {
                        // For large BitVecs, use BigInt
                        use num_bigint::BigUint;
                        let bytes = bv.as_raw_slice();
                        BigUint::from_bytes_le(bytes).to_string()
                    }
                })
                .collect()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BitVec data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get U8 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_u8s(&self, register: &str) -> Result<Vec<u8>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::U8(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain U8 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get U16 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_u16s(&self, register: &str) -> Result<Vec<u16>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::U16(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain U16 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get U64 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_u64s(&self, register: &str) -> Result<Vec<u64>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::U64(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain U64 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get I8 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_i8s(&self, register: &str) -> Result<Vec<i8>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::I8(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain I8 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get I16 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_i16s(&self, register: &str) -> Result<Vec<i16>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::I16(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain I16 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get I32 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_i32s(&self, register: &str) -> Result<Vec<i32>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::I32(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain I32 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get I64 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_i64s(&self, register: &str) -> Result<Vec<i64>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::I64(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain I64 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get F32 values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_f32s(&self, register: &str) -> Result<Vec<f32>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::F32(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain F32 data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get String values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_strings(&self, register: &str) -> Result<Vec<String>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::String(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain String data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BigInt` values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bigints(&self, register: &str) -> Result<Vec<num_bigint::BigInt>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::BigInt(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BigInt data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get Bytes values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bytes(&self, register: &str) -> Result<Vec<Vec<u8>>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::Bytes(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain Bytes data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BitVec` values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bitvecs(&self, register: &str) -> Result<Vec<BitVec<u8, Lsb0>>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::BitVec(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BitVec data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get Json values from a register
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_jsons(&self, register: &str) -> Result<Vec<serde_json::Value>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::Json(v)) => Ok(v.clone()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain Json data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Try to get `BitVec` values as hexadecimal strings
    ///
    /// Returns strings like "1A2F" in uppercase hexadecimal.
    ///
    /// # Returns
    /// - `Ok(Vec<String>)` if the register exists and contains `BitVec` data
    /// - `Err` if the register doesn't exist or contains a different data type
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// # use pecos_core::errors::PecosError;
    /// # use bitvec::prelude::*;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.add_register("register", 0xAB, 8); // value 171 (0xAB) with 8 bits
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// // Extract register values as hex strings
    /// if let Ok(values) = shot_map.try_bits_as_hex("register") {
    ///     for (shot, hex) in values.iter().enumerate() {
    ///         println!("Shot {}: 0x{}", shot, hex);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns `PecosError::Processing` if:
    /// - The register doesn't exist
    /// - The register exists but contains a different data type
    pub fn try_bits_as_hex(&self, register: &str) -> Result<Vec<String>, PecosError> {
        match self.data.get(register) {
            Some(DataVec::BitVec(vecs)) => Ok(vecs
                .iter()
                .map(|bv| {
                    if bv.is_empty() {
                        "0".to_string()
                    } else if bv.len() <= 64 {
                        // For small BitVecs, use u64
                        let mut value = 0u64;
                        for (i, bit) in bv.iter().enumerate() {
                            if *bit {
                                value |= 1u64 << i;
                            }
                        }
                        format!("{value:X}")
                    } else {
                        // For large BitVecs, convert to hex via bytes
                        let bytes = bv.as_raw_slice();
                        // Convert bytes to hex string (reverse for big-endian display)
                        let hex_string = bytes.iter().rev().fold(String::new(), |mut acc, b| {
                            use std::fmt::Write;
                            write!(&mut acc, "{b:02X}").unwrap();
                            acc
                        });
                        hex_string.trim_start_matches('0').to_string()
                    }
                })
                .collect()),
            Some(_) => Err(PecosError::Processing(format!(
                "Register '{register}' exists but does not contain BitVec data"
            ))),
            None => Err(PecosError::Processing(format!(
                "Register '{register}' not found"
            ))),
        }
    }

    /// Convert the `ShotMap` to JSON
    ///
    /// Returns a `serde_json::Value` with the columnar data.
    /// Each register is mapped to an array of values.
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot, Data};
    /// # use pecos_core::errors::PecosError;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// let mut shot = Shot::default();
    /// shot.data.insert("phase".to_string(), Data::F64(0.5));
    /// shot.data.insert("count".to_string(), Data::U32(42));
    /// shot_vec.shots.push(shot);
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// let json_value = shot_map.to_json();
    /// println!("{}", json_value);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();

        for (name, data_vec) in &self.data {
            map.insert(name.clone(), data_vec.to_json_array());
        }

        Value::Object(map)
    }

    /// Convert the `ShotMap` to a simplified JSON representation
    ///
    /// This method converts `BitVec` data to simple integer values and
    /// preserves other data types as-is. This is useful for cleaner
    /// JSON output when working with quantum measurement data.
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// # use pecos_core::errors::PecosError;
    /// # fn main() -> Result<(), PecosError> {
    /// let mut shot_vec = ShotVec::new();
    /// // Add shots with BitVec data for q0
    /// for value in [0, 1, 0, 1].iter() {
    ///     let mut shot = Shot::default();
    ///     shot.add_register("q0", *value, 1);
    ///     shot_vec.shots.push(shot);
    /// }
    /// let shot_map = shot_vec.try_as_shot_map()?;
    ///
    /// let simple_json = shot_map.to_simple_json();
    /// // BitVecs are converted to integers: {"q0": [0, 1, 0, 1]}
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn to_simple_json(&self) -> Value {
        let mut map = Map::new();

        for (name, data_vec) in &self.data {
            let json_array = match data_vec {
                DataVec::BitVec(bit_vecs) => {
                    // Convert BitVecs to integers
                    let values: Vec<Value> = bit_vecs
                        .iter()
                        .map(|bv| {
                            let mut value = 0u64;
                            for (i, bit) in bv.iter().enumerate() {
                                if *bit && i < 64 {
                                    value |= 1u64 << i;
                                }
                            }
                            Value::Number(value.into())
                        })
                        .collect();
                    Value::Array(values)
                }
                // For other types, use the standard JSON array conversion
                _ => data_vec.to_json_array(),
            };

            map.insert(name.clone(), json_array);
        }

        Value::Object(map)
    }
}

// Implement Display for ShotMap that delegates to the display() method
impl fmt::Display for ShotMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Import the extension trait to get the display() method
        use crate::shot_map_formatter::ShotMapDisplayExt;
        // Delegate to the display formatter
        write!(f, "{}", self.display())
    }
}

// Implement IntoIterator for owned ShotMap
impl IntoIterator for ShotMap {
    type Item = (String, DataVec);
    type IntoIter = std::collections::btree_map::IntoIter<String, DataVec>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

// Implement IntoIterator for &ShotMap
impl<'a> IntoIterator for &'a ShotMap {
    type Item = (&'a String, &'a DataVec);
    type IntoIter = std::collections::btree_map::Iter<'a, String, DataVec>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shot, ShotVec};

    #[test]
    fn test_shot_map_creation() {
        let mut shot_vec = ShotVec::new();

        for i in 0..3 {
            let mut shot = Shot::default();
            shot.add_register("a", i, 2);
            shot.data.insert("b".to_string(), Data::Bool(i % 2 == 0));
            shot_vec.shots.push(shot);
        }

        let shot_map = shot_vec.try_as_shot_map().unwrap();

        assert_eq!(shot_map.num_shots(), 3);
        assert_eq!(shot_map.num_registers(), 2);
        assert!(shot_map.contains_register("a"));
        assert!(shot_map.contains_register("b"));
    }

    #[test]
    fn test_display_impl() {
        use crate::shot_map_formatter::ShotMapDisplayExt;

        let mut shot_vec = ShotVec::new();

        let mut shot = Shot::default();
        shot.add_register("q", 5, 3); // 5 = "101" with 3 bits
        shot.data.insert("count".to_string(), Data::U32(42));
        shot_vec.shots.push(shot);

        let shot_map = shot_vec.try_as_shot_map().unwrap();

        // Test that Display is implemented and works
        let display_str = format!("{shot_map}");
        assert!(display_str.contains("\"q\""));
        assert!(display_str.contains("\"count\""));

        // Verify it matches the explicit display() call
        let explicit_display = format!("{}", shot_map.display());
        assert_eq!(display_str, explicit_display);
    }

    #[test]
    fn test_extract_methods() {
        let mut data = BTreeMap::new();
        data.insert(
            "values".to_string(),
            vec![Data::U32(1), Data::U32(2), Data::U32(3)],
        );
        data.insert(
            "flags".to_string(),
            vec![Data::Bool(true), Data::Bool(false), Data::Bool(true)],
        );

        let shot_map = ShotMap::new(data).unwrap();

        let values = shot_map.try_u32s("values").unwrap();
        assert_eq!(values, vec![1, 2, 3]);

        let flags = shot_map.try_bools("flags").unwrap();
        assert_eq!(flags, vec![true, false, true]);

        // Test error cases
        assert!(shot_map.try_u32s("flags").is_err());
        assert!(shot_map.try_bools("values").is_err());
        assert!(shot_map.try_u32s("nonexistent").is_err());
    }

    #[test]
    fn test_mismatched_lengths() {
        let mut data = BTreeMap::new();
        data.insert(
            "a".to_string(),
            vec![Data::U32(1), Data::U32(2), Data::U32(3)],
        );
        data.insert("b".to_string(), vec![Data::U32(4), Data::U32(5)]);

        let result = ShotMap::new(data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // The actual error might be about column 'a' or 'b' depending on HashMap iteration order
        assert!(
            err_msg.contains("has 2 values but expected 3")
                || err_msg.contains("has 3 values but expected 2")
        );
    }

    #[test]
    fn test_bitvec_extract_methods() {
        let mut shot_vec = ShotVec::new();

        // Add some shots with BitVec data
        for i in 0..4 {
            let mut shot = Shot::default();

            // Add small BitVec (3 bits)
            shot.add_register("small", i, 3);

            // Add medium BitVec (8 bits)
            shot.add_register("medium", i * 17, 8);

            // Add a register that's not a BitVec for testing
            shot.data.insert("notbitvec".to_string(), Data::U32(i));

            shot_vec.shots.push(shot);
        }

        let shot_map = shot_vec.try_as_shot_map().unwrap();

        // Test try_bits_as_u64
        let u64_values = shot_map.try_bits_as_u64("small").unwrap();
        assert_eq!(u64_values, vec![0, 1, 2, 3]);

        let medium_u64 = shot_map.try_bits_as_u64("medium").unwrap();
        assert_eq!(medium_u64, vec![0, 17, 34, 51]);

        // Test with non-existent register
        let result = shot_map.try_bits_as_u64("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));

        // Test with non-BitVec register
        let result = shot_map.try_bits_as_u64("notbitvec");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not contain BitVec data")
        );

        // Test try_bits_as_binary
        let binary_strings = shot_map.try_bits_as_binary("small").unwrap();
        assert_eq!(binary_strings, vec!["000", "001", "010", "011"]);

        let medium_binary = shot_map.try_bits_as_binary("medium").unwrap();
        assert_eq!(
            medium_binary,
            vec!["00000000", "00010001", "00100010", "00110011"]
        );

        // Test try_bits_as_decimal
        let decimal_strings = shot_map.try_bits_as_decimal("small").unwrap();
        assert_eq!(decimal_strings, vec!["0", "1", "2", "3"]);

        let medium_decimal = shot_map.try_bits_as_decimal("medium").unwrap();
        assert_eq!(medium_decimal, vec!["0", "17", "34", "51"]);

        // Test try_bits_as_hex
        let hex_strings = shot_map.try_bits_as_hex("small").unwrap();
        assert_eq!(hex_strings, vec!["0", "1", "2", "3"]);

        let medium_hex = shot_map.try_bits_as_hex("medium").unwrap();
        assert_eq!(medium_hex, vec!["0", "11", "22", "33"]);
    }
}
