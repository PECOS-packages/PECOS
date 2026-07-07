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

use super::{data::Data, shot::Shot};
use pecos_core::errors::PecosError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Represents the results of multiple shots (executions) of a quantum program.
///
/// This struct contains a vector of individual shot results, providing a clean
/// and flexible way to store and access measurement data from multiple executions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotVec {
    /// Vector of individual shot results
    pub shots: Vec<Shot>,
}

impl Default for ShotVec {
    fn default() -> Self {
        Self::new()
    }
}

impl ShotVec {
    /// Creates a new empty `ShotVec` instance.
    #[must_use]
    pub fn new() -> Self {
        Self { shots: Vec::new() }
    }

    /// Get the total number of shots
    #[must_use]
    pub fn len(&self) -> usize {
        self.shots.len()
    }

    /// Check if there are no shots
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shots.is_empty()
    }

    /// Get all register names (excluding metadata entries)
    #[must_use]
    pub fn get_register_names(&self) -> Vec<String> {
        if self.shots.is_empty() {
            return Vec::new();
        }

        // Get keys from first shot and filter out metadata entries
        let mut names: Vec<String> = self.shots[0]
            .data
            .keys()
            .filter(|k| !k.starts_with("_width_"))
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// Try to convert the shot vector to a `ShotMap` (columnar format)
    ///
    /// This method transforms the row-based shot data into column-based data where:
    /// - Keys are register names
    /// - Values are vectors containing the `Data` value for each shot
    ///
    /// # Returns
    /// - `Ok(ShotMap)` containing the columnar representation
    /// - `Err(PecosError)` if not all shots have the same register keys
    ///
    /// # Errors
    /// Returns a `PecosError` if:
    /// - Not all shots have the same register keys
    /// - A register is missing from any shot after the first
    ///
    /// # Example
    /// ```
    /// # use pecos_results::{ShotVec, Shot};
    /// let mut shot_vec = ShotVec::new();
    ///
    /// // Add shots with consistent structure
    /// for i in 0..3 {
    ///     let mut shot = Shot::default();
    ///     shot.add_register("a", i, 2);
    ///     shot.add_register("b", i * 2, 3);
    ///     shot_vec.shots.push(shot);
    /// }
    ///
    /// // Convert to ShotMap
    /// match shot_vec.try_as_shot_map() {
    ///     Ok(shot_map) => {
    ///         // Access all values for register "a"
    ///         let a_values = shot_map.get("a").unwrap();
    ///         assert_eq!(a_values.len(), 3);
    ///     }
    ///     Err(e) => {
    ///         // Handle inconsistent shot structure
    ///     }
    /// }
    /// ```
    ///
    /// # Panics
    /// This function should not panic under normal usage. The `unwrap()` call is protected
    /// by prior validation that ensures the key exists in the `BTreeMap`.
    pub fn try_as_shot_map(&self) -> Result<super::shot_map::ShotMap, PecosError> {
        if self.is_empty() {
            return super::shot_map::ShotMap::new(BTreeMap::new());
        }

        // Get register names from the first shot
        let register_names = self.get_register_names();

        // Initialize the columnar map with empty vectors
        let mut columnar_map: BTreeMap<String, Vec<Data>> = BTreeMap::new();
        for name in &register_names {
            columnar_map.insert(name.clone(), Vec::with_capacity(self.len()));
        }

        // Iterate through all shots and populate the columnar data
        for (shot_idx, shot) in self.shots.iter().enumerate() {
            // Check that this shot has the same keys
            let shot_keys: Vec<String> = shot
                .data
                .keys()
                .filter(|k| !k.starts_with("_width_"))
                .cloned()
                .collect();

            if shot_keys.len() != register_names.len() {
                return Err(PecosError::Processing(format!(
                    "Shot {} has {} registers, but expected {} based on first shot",
                    shot_idx,
                    shot_keys.len(),
                    register_names.len()
                )));
            }

            // Add each register's value to the appropriate column
            for name in &register_names {
                match shot.data.get(name) {
                    Some(data) => {
                        columnar_map
                            .get_mut(name)
                            .expect("key was inserted during initialization")
                            .push(data.clone());
                    }
                    None => {
                        return Err(PecosError::Processing(format!(
                            "Shot {shot_idx} is missing register '{name}' which was present in the first shot"
                        )));
                    }
                }
            }
        }

        super::shot_map::ShotMap::new(columnar_map)
    }

    /// Format results as binary strings for all registers
    ///
    /// Returns a map where each register name maps to a vector of binary strings,
    /// one per shot. Each binary string is zero-padded to the register's width.
    #[must_use]
    pub fn format_as_binary_strings(&self) -> BTreeMap<String, Vec<String>> {
        let register_names = self.get_register_names();
        let mut result = BTreeMap::new();

        for name in register_names {
            let binary_strings: Vec<String> = self
                .shots
                .iter()
                .map(|shot| shot.register_to_binary_string(&name).unwrap_or_default())
                .collect();
            result.insert(name, binary_strings);
        }

        result
    }

    /// Creates a serializable representation for JSON output
    ///
    /// Returns an array of shot objects, where each shot contains its data
    /// with numerical values (U32, `BitVec`, etc.) converted to decimal numbers.
    ///
    /// # Example Output
    /// ```json
    /// [{"c": 3}, {"c": 0}, {"c": 3}, ...]
    /// ```
    #[must_use]
    pub fn create_json_value(&self) -> serde_json::Value {
        use serde_json::{Map, Value};

        let shots: Vec<Value> = self
            .shots
            .iter()
            .map(|shot| {
                let mut obj = Map::new();

                for (key, data) in &shot.data {
                    // Skip metadata entries
                    if key.starts_with('_') {
                        continue;
                    }

                    // Use to_value_string for simplicity, then parse as number if possible
                    let value_str = data.to_value_string();

                    // Try to parse as number first, fallback to string
                    let value = if let Ok(n) = value_str.parse::<u64>() {
                        Value::Number(n.into())
                    } else if let Ok(n) = value_str.parse::<i64>() {
                        Value::Number(n.into())
                    } else if let Ok(n) = value_str.parse::<f64>() {
                        serde_json::Number::from_f64(n)
                            .map_or_else(|| Value::String(value_str), Value::Number)
                    } else {
                        Value::String(value_str)
                    };

                    obj.insert(key.clone(), value);
                }

                Value::Object(obj)
            })
            .collect();

        Value::Array(shots)
    }

    /// Converts the `ShotVec` to a compact JSON string
    ///
    /// # Returns
    ///
    /// A compact JSON string without whitespace or formatting
    #[must_use]
    pub fn to_compact_json(&self) -> String {
        if self.shots.iter().all(|shot| shot.data.is_empty()) {
            return "[]".to_string();
        }

        // If we have complex JSON data, serialize the full shots
        if self
            .shots
            .iter()
            .any(|shot| shot.data.values().any(|v| matches!(v, Data::Json(_))))
        {
            serde_json::to_string(&self.shots).unwrap_or_else(|_| "[]".to_string())
        } else {
            // Otherwise use the aggregated format
            let json_value = self.create_json_value();
            serde_json::to_string(&json_value).unwrap_or_else(|_| "{}".to_string())
        }
    }

    /// Creates a `ShotVec` instance from a slice of `Shot` instances.
    ///
    /// This method simply collects the individual shot results into a vector.
    ///
    /// # Parameters
    ///
    /// * `results` - A slice of `Shot` instances to process
    ///
    /// # Returns
    ///
    /// A new `ShotVec` instance containing the processed measurement results
    #[must_use]
    pub fn from_measurements(results: &[Shot]) -> Self {
        Self {
            shots: results.to_vec(),
        }
    }

    /// Prints the `ShotVec` to stdout.
    pub fn print(&self) {
        println!("{self}");
    }
}

impl fmt::Display for ShotVec {
    /// Formats the shot results for display using compact JSON.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_compact_json())
    }
}
