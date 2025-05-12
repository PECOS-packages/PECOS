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

#![allow(clippy::similar_names)]
// For percentage calculations below with large usize values converted to f64,
// we accept the potential precision loss since the values are used only for display
// with a single decimal place, and the precision loss would only be observable
// with extremely large shot counts (> 2^53).
#![allow(clippy::cast_precision_loss)]

use crate::byte_message::ByteMessage;
use pecos_core::errors::PecosError;
use std::collections::HashMap;
use std::fmt;

/// Represents the results of a single shot (execution) of a quantum program.
///
/// This struct contains mappings of register names to measurement outcomes in various formats.
/// Measurement outcomes can be represented in multiple ways:
/// - 32-bit unsigned integers (standard format)
/// - 64-bit unsigned integers (for values larger than `u32::MAX`)
/// - 64-bit signed integers (when sign interpretation is needed)
///
/// ## Field Usage Guidelines
///
/// - `registers`: Standard 32-bit values for most measurement outcomes
/// - `registers_u64`: Extended 64-bit unsigned values for large results
/// - `registers_i64`: Extended 64-bit signed values when sign interpretation is needed
///
/// Values that don't fit in 32 bits are stored in both formats (truncated in 32-bit fields)
/// with the complete value in the 64-bit fields.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ShotResult {
    /// Direct mapping of register names to 32-bit integer values
    /// Standard representation for classical registers in QASM and similar models
    pub registers: HashMap<String, u32>,

    /// Extended mapping supporting 64-bit unsigned values for large results
    /// Used when measurement outcomes exceed what a u32 can represent (> 4,294,967,295)
    pub registers_u64: HashMap<String, u64>,

    /// Extended mapping supporting 64-bit signed values when needed
    /// Useful for applications requiring sign interpretation
    pub registers_i64: HashMap<String, i64>,
}

impl ShotResult {
    /// Create a `ShotResult` directly from a `ByteMessage` containing measurement results.
    ///
    /// This method extracts measurement results from a `ByteMessage` and creates a `ShotResult`
    /// with properly mapped result IDs to names.
    ///
    /// # Parameters
    ///
    /// * `message` - A `ByteMessage` containing measurement results
    /// * `result_id_to_name` - A mapping from `result_id` to a human-readable name
    ///
    /// # Returns
    ///
    /// A new `ShotResult` instance containing the processed measurement results
    ///
    /// # Errors
    ///
    /// Returns an error if the `ByteMessage` cannot be parsed or doesn't contain valid measurement results
    pub fn from_byte_message(
        message: &ByteMessage,
        result_id_to_name: &HashMap<usize, String>,
    ) -> Result<Self, PecosError> {
        // Extract the measurement results from the ByteMessage
        let measurements = message.measurement_results_as_vec()?;

        let mut result = Self::default();

        // Process each measurement
        for (result_id, value) in measurements {
            // Get the name for this result_id, or use a default if not found
            let name = result_id_to_name
                .get(&result_id)
                .cloned()
                .unwrap_or_else(|| format!("result_{result_id}"));

            // Add to registers fields
            result.registers.insert(name.clone(), value);
            result.registers_u64.insert(name, u64::from(value));
        }

        Ok(result)
    }

    /// Creates a binary string representation of results.
    ///
    /// This is a convenience method that creates a binary string from register values.
    ///
    /// # Parameters
    ///
    /// * `registers` - Optional list of register names to include. If None, all registers are used.
    /// * `sort_by_name` - Whether to sort registers by name (true) or use provided order (false)
    ///
    /// # Returns
    ///
    /// A binary string representation of the specified registers
    #[must_use]
    pub fn create_binary_string(&self, registers: Option<&[&str]>, sort_by_name: bool) -> String {
        let mut register_entries: Vec<(&String, &u32)> = match registers {
            Some(names) => names
                .iter()
                .filter_map(|&name| self.registers.get_key_value(name))
                .collect(),
            None => self.registers.iter().collect(),
        };

        if sort_by_name {
            register_entries.sort_by(|(name1, _), (name2, _)| name1.cmp(name2));
        }

        register_entries
            .iter()
            .map(|&(_, value)| if *value > 0 { '1' } else { '0' })
            .collect()
    }
}

/// Represents the results of multiple shots (executions) of a quantum program.
///
/// This struct contains the aggregated results from multiple program executions ("shots").
/// Results are stored in multiple formats for flexibility:
///
/// - String-based representation: `shots` field for text display
/// - Integer vectors: `register_shots` fields for numerical analysis
///
/// ## Display Order
///
/// When formatted for display, registers are shown in this priority order:
/// 1. 32-bit registers first (for compatibility)
/// 2. 64-bit unsigned registers (if not already shown in 32-bit)
/// 3. 64-bit signed registers (if not already shown in other formats)
///
/// This ensures each register appears exactly once in the output, even if it's
/// stored in multiple formats internally.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShotResults {
    /// Each element is a mapping of register names to string values for a single shot
    pub shots: Vec<HashMap<String, String>>,

    /// Direct mapping of register names to 32-bit integer values across shots
    /// The outer `HashMap` maps register names to a vector of values, one per shot
    pub register_shots: HashMap<String, Vec<u32>>,

    /// Extended mapping supporting 64-bit unsigned values for large results
    /// Used when measurement outcomes exceed what a u32 can represent
    pub register_shots_u64: HashMap<String, Vec<u64>>,

    /// Extended mapping supporting 64-bit signed values when sign interpretation is needed
    /// Used for applications requiring sign interpretation
    pub register_shots_i64: HashMap<String, Vec<i64>>,
}

impl Default for ShotResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Defines the output format for `ShotResults`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Pretty-printed JSON with indentation for readability
    PrettyJson,
    /// Compact JSON without extra whitespace
    CompactJson,
    /// Compact JSON with each register on a new line for better readability
    PrettyCompactJson,
    /// Format showing frequencies of each outcome
    Frequency,
}

impl ShotResults {
    /// Creates a new empty `ShotResults` instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shots: Vec::new(),
            register_shots: HashMap::new(),
            register_shots_u64: HashMap::new(),
            register_shots_i64: HashMap::new(),
        }
    }

    /// Converts the `ShotResults` to a JSON string representation
    ///
    /// This creates a proper JSON structure that represents the shot results
    /// and is used by the Display implementation for consistent output.
    ///
    /// # Returns
    ///
    /// A pretty-printed JSON string representation of the shot results
    #[must_use]
    pub fn to_json(&self) -> String {
        // Default to pretty-printed JSON
        self.to_string_with_format(OutputFormat::PrettyJson)
    }

    /// Creates a serializable representation for JSON output
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` containing the cleaned-up shot results
    #[must_use]
    fn create_json_value(&self) -> serde_json::Value {
        use serde_json::{Map, Value};

        // Start with an empty JSON object
        let mut result = Map::new();

        // Track registers we've already processed
        let mut displayed_registers = std::collections::HashSet::new();

        // Process in priority order: u32, u64, i64

        // First add u32 registers
        for reg_name in self.register_shots.keys() {
            let values = &self.register_shots[reg_name];
            result.insert(
                reg_name.clone(),
                Value::Array(values.iter().map(|&v| Value::Number(v.into())).collect()),
            );
            displayed_registers.insert(reg_name);
        }

        // Then add u64 registers not already included
        for reg_name in self.register_shots_u64.keys() {
            if !displayed_registers.contains(reg_name) {
                let values = &self.register_shots_u64[reg_name];
                // For u64 values that fit into a JSON number, use Number, otherwise String
                let json_values: Vec<Value> = values
                    .iter()
                    .map(|&v| {
                        // Convert without unsafe cast to avoid potential wrapping
                        if let Ok(v_i64) = i64::try_from(v) {
                            Value::Number(serde_json::Number::from(v_i64))
                        } else {
                            // For very large values, use strings
                            Value::String(v.to_string())
                        }
                    })
                    .collect();
                result.insert(reg_name.clone(), Value::Array(json_values));
                displayed_registers.insert(reg_name);
            }
        }

        // Finally add i64 registers not already included
        for reg_name in self.register_shots_i64.keys() {
            if !displayed_registers.contains(reg_name) {
                let values = &self.register_shots_i64[reg_name];
                result.insert(
                    reg_name.clone(),
                    Value::Array(values.iter().map(|&v| Value::Number(v.into())).collect()),
                );
            }
        }

        Value::Object(result)
    }

    /// Converts the `ShotResults` to a string representation with the specified format
    ///
    /// # Parameters
    ///
    /// * `format` - The output format to use (`PrettyJson`, `CompactJson`, Tabular, or Concise)
    ///
    /// # Returns
    ///
    /// A string representation of the shot results in the specified format
    #[must_use]
    pub fn to_string_with_format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::PrettyJson => self.to_pretty_json(),
            OutputFormat::CompactJson => self.to_compact_json(),
            OutputFormat::PrettyCompactJson => self.to_pretty_compact_json(),
            OutputFormat::Frequency => self.to_frequency_format(),
        }
    }

    /// Formats the shot results showing frequencies of each outcome
    ///
    /// Instead of showing all shots individually, this counts occurrences of each value
    /// and presents them in a histogram-like format for better readability.
    ///
    /// # Returns
    ///
    /// A string representation showing frequencies of outcomes
    #[must_use]
    fn to_frequency_format(&self) -> String {
        use std::collections::BTreeMap;

        // If no data, return early
        if self.register_shots.is_empty()
            && self.register_shots_u64.is_empty()
            && self.register_shots_i64.is_empty()
        {
            if self.shots.is_empty() {
                return "No results available.".to_string();
            }

            // For shot-based format, convert to a more readable form
            let mut output = String::new();
            // We'll collect stats for each register
            let mut register_stats: BTreeMap<&String, BTreeMap<&String, usize>> = BTreeMap::new();

            // Count occurrences of each value for each register
            for shot in &self.shots {
                for (key, value) in shot {
                    let reg_stats = register_stats.entry(key).or_default();
                    *reg_stats.entry(value).or_default() += 1;
                }
            }

            // Build the output
            output.push_str("Results (from ");
            output.push_str(&self.shots.len().to_string());
            output.push_str(" shots):\n");

            for (reg_name, stats) in &register_stats {
                // A formatting error here should never happen with a simple string, but handle it safely
                use std::fmt::Write;
                // Ignoring the error as this write to a String cannot fail in practice
                let _ = write!(output, "  {reg_name}: ");

                let mut stat_entries: Vec<_> = stats.iter().collect();
                // Sort stats by value for consistent ordering
                stat_entries.sort_by(|a, b| {
                    a.0.parse::<i64>()
                        .unwrap_or(0)
                        .cmp(&b.0.parse::<i64>().unwrap_or(0))
                });

                let total_shots = self.shots.len();
                let entries: Vec<_> = stat_entries
                    .iter()
                    .map(|(val, count)| {
                        let count_val = **count; // Dereference properly
                        // For very large count values and total_shots (≥2^53), this calculation
                        // could lose precision. However, for our use case this is fine because:
                        // 1. We only display with 1 decimal place precision
                        // 2. It's extremely unlikely to encounter shots counts > 2^53 (~9 quadrillion)
                        // We're effectively calculating: (count / total) * 100
                        let percentage = 100.0 * (count_val as f64 / total_shots as f64);
                        format!("{val}={percentage:.1}%")
                    })
                    .collect();

                output.push_str(&entries.join(", "));
                output.push('\n');
            }

            return output;
        }

        // Convert to JSON value for consistent handling
        let json_value = self.create_json_value();

        // Extract the registers and values
        let mut view = HashMap::new();
        if let serde_json::Value::Object(obj) = &json_value {
            for (key, value) in obj {
                if let serde_json::Value::Array(arr) = value {
                    let values: Vec<serde_json::Value> = arr.clone();
                    view.insert(key.clone(), values);
                }
            }
        }

        let num_shots = view.values().next().map_or(0, std::vec::Vec::len);

        if num_shots == 0 {
            return "No results available.".to_string();
        }

        // Create a BTreeMap for register names to ensure consistent ordering
        let mut register_results: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

        // Count occurrences for each register value
        for (reg_name, values) in &view {
            let mut value_counts: BTreeMap<String, usize> = BTreeMap::new();

            for value in values {
                let val_str = value.to_string().trim_matches('"').to_string();
                *value_counts.entry(val_str).or_default() += 1;
            }

            register_results.insert(reg_name.clone(), value_counts);
        }

        // Build the output string
        let mut output = String::new();
        output.push_str("Results (from ");
        output.push_str(&num_shots.to_string());
        output.push_str(" shots):\n");

        for (reg_name, counts) in &register_results {
            // A formatting error here should never happen with a simple string, but handle it safely
            use std::fmt::Write;
            // Ignoring the error as this write to a String cannot fail in practice
            let _ = write!(output, "  {reg_name}: ");

            let entries: Vec<_> = counts
                .iter()
                .map(|(val, count)| {
                    let count_val = *count; // Dereference properly
                    // For very large count values and num_shots (≥2^53), this calculation
                    // could lose precision. However, for our use case this is fine because:
                    // 1. We only display with 1 decimal place precision
                    // 2. It's extremely unlikely to encounter shots counts > 2^53 (~9 quadrillion)
                    // We're effectively calculating: (count / total) * 100
                    let percentage = 100.0 * (count_val as f64 / num_shots as f64);
                    format!("{val}={percentage:.1}%")
                })
                .collect();

            output.push_str(&entries.join(", "));
            output.push('\n');
        }

        output
    }

    /// Converts the `ShotResults` to a pretty-printed JSON string
    ///
    /// # Returns
    ///
    /// A pretty-printed JSON string
    #[must_use]
    fn to_pretty_json(&self) -> String {
        if !self.register_shots.is_empty()
            || !self.register_shots_u64.is_empty()
            || !self.register_shots_i64.is_empty()
        {
            // Use the JSON Value representation
            let json_value = self.create_json_value();
            serde_json::to_string_pretty(&json_value).unwrap_or_else(|_| "{}".to_string())
        } else {
            // Use the shot-based format
            serde_json::to_string_pretty(&self.shots).unwrap_or_else(|_| "[]".to_string())
        }
    }

    /// Converts the `ShotResults` to a compact JSON string
    ///
    /// # Returns
    ///
    /// A compact JSON string without whitespace or formatting
    #[must_use]
    fn to_compact_json(&self) -> String {
        if !self.register_shots.is_empty()
            || !self.register_shots_u64.is_empty()
            || !self.register_shots_i64.is_empty()
        {
            // Use the JSON Value representation
            let json_value = self.create_json_value();
            serde_json::to_string(&json_value).unwrap_or_else(|_| "{}".to_string())
        } else {
            // Use the shot-based format
            serde_json::to_string(&self.shots).unwrap_or_else(|_| "[]".to_string())
        }
    }

    /// Converts the `ShotResults` to a pretty compact JSON string
    ///
    /// This format is compact but with each register on its own line for better readability.
    /// It strikes a balance between the fully pretty-printed JSON and the fully compact version.
    ///
    /// # Returns
    ///
    /// A JSON string with minimal indentation but each register on a new line
    #[must_use]
    fn to_pretty_compact_json(&self) -> String {
        use std::fmt::Write;

        if self.register_shots.is_empty()
            && self.register_shots_u64.is_empty()
            && self.register_shots_i64.is_empty()
        {
            if self.shots.is_empty() {
                return "[]".to_string();
            }

            // For shot-based format in pretty compact form
            let json_string =
                serde_json::to_string(&self.shots).unwrap_or_else(|_| "[]".to_string());
            return json_string;
        }

        // Use the JSON Value representation
        let json_value = self.create_json_value();

        // For register-based format, build a custom format with each register on a new line
        if let serde_json::Value::Object(obj) = json_value {
            let mut result = String::from("{");

            // Sort keys for consistent output
            let mut keys: Vec<_> = obj.keys().collect();
            keys.sort();

            // Process each register
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    result.push(',');
                }
                result.push_str("\n  ");

                // Add the key with quotes
                // Ignoring the error as this write to a String cannot fail in practice
                let _ = write!(result, "\"{key}\":");

                // Add the value (compact format)
                if let Some(value) = obj.get(*key) {
                    let value_str = serde_json::to_string(value).unwrap_or_default();
                    result.push_str(&value_str);
                }
            }

            // Close the object
            if !keys.is_empty() {
                result.push('\n');
            }
            result.push('}');

            result
        } else {
            // Fallback to compact JSON
            serde_json::to_string(&json_value).unwrap_or_else(|_| "{}".to_string())
        }
    }

    /// Creates a `ShotResults` instance from a slice of `ShotResult` instances.
    ///
    /// This method processes each `ShotResult`, extracting measurements and formatting
    /// them appropriately for the `ShotResults` structure.
    ///
    /// # Parameters
    ///
    /// * `results` - A slice of `ShotResult` instances to process
    ///
    /// # Returns
    ///
    /// A new `ShotResults` instance containing the processed measurement results
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn from_measurements(results: &[ShotResult]) -> Self {
        let mut shots = Vec::with_capacity(results.len());
        let mut register_shots: HashMap<String, Vec<u32>> = HashMap::new();
        let mut register_shots_u64: HashMap<String, Vec<u64>> = HashMap::new();
        let mut register_shots_i64: HashMap<String, Vec<i64>> = HashMap::new();

        for shot in results {
            let mut processed_results = HashMap::new();

            // Process all register types with priority order for string representation

            // First collect all register names across all types
            let mut all_register_names = shot
                .registers
                .keys()
                .collect::<std::collections::HashSet<_>>();
            all_register_names.extend(shot.registers_u64.keys());
            all_register_names.extend(shot.registers_i64.keys());

            // Process each register in priority order (u32, u64, i64)
            for reg_name in all_register_names {
                // Add 32-bit value to vector and string representation
                if let Some(&value) = shot.registers.get(reg_name) {
                    register_shots
                        .entry(reg_name.clone())
                        .or_default()
                        .push(value);
                    processed_results.insert(reg_name.clone(), value.to_string());
                }

                // Add 64-bit unsigned value to vector
                if let Some(&value) = shot.registers_u64.get(reg_name) {
                    register_shots_u64
                        .entry(reg_name.clone())
                        .or_default()
                        .push(value);
                    // Add to string representation only if not already added
                    if !processed_results.contains_key(reg_name) {
                        processed_results.insert(reg_name.clone(), value.to_string());
                    }
                }

                // Add 64-bit signed value to vector
                if let Some(&value) = shot.registers_i64.get(reg_name) {
                    register_shots_i64
                        .entry(reg_name.clone())
                        .or_default()
                        .push(value);
                    // Add to string representation only if not already added
                    if !processed_results.contains_key(reg_name) {
                        processed_results.insert(reg_name.clone(), value.to_string());
                    }
                }
            }

            shots.push(processed_results);
        }

        Self {
            shots,
            register_shots,
            register_shots_u64,
            register_shots_i64,
        }
    }

    /// Create a `ShotResults` instance directly from a `ByteMessage` containing measurement results.
    ///
    /// This method extracts measurement results from a `ByteMessage` and creates a `ShotResults`
    /// instance with properly formatted results. It's more efficient than going through
    /// `ShotResult` instances and provides better context about the measurements.
    ///
    /// # Parameters
    ///
    /// * `message` - A `ByteMessage` containing measurement results
    ///
    /// # Errors
    ///
    /// Returns a `PecosError` if the measurements cannot be extracted from the `ByteMessage`
    /// or if there are issues with creating the `ShotResults` instance.
    pub fn from_byte_message(message: &ByteMessage) -> Result<Self, PecosError> {
        // Extract the measurement results from the ByteMessage
        let measurements = message.measurement_results_as_vec()?;

        let mut result = Self::new();

        // Process each measurement
        for (result_id, value) in measurements {
            // Get the name for this result_id, or use a default if not found
            let name = format!("result_{result_id}");

            // Add the measurement to the results
            result.shots[0].insert(name, value.to_string());
        }

        Ok(result)
    }

    /// Prints the `ShotResults` to stdout.
    pub fn print(&self) {
        println!("{self}");
    }
}

impl fmt::Display for ShotResults {
    /// Formats the shot results for display using JSON.
    ///
    /// This implementation uses the `to_string_with_format` method to generate a consistent,
    /// properly formatted JSON representation of the shot results.
    /// By default, it uses the pretty compact format which displays each register on its own line
    /// for better readability while keeping the values compact.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the pretty compact JSON serialization for display
        write!(
            f,
            "{}",
            self.to_string_with_format(OutputFormat::PrettyCompactJson)
        )
    }
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;

    #[test]
    fn test_shot_results_display_64bit() {
        // Create shot results with various register types
        let mut shot_results = ShotResults::new();

        // Add a standard 32-bit register
        shot_results
            .register_shots
            .insert("reg_32".to_string(), vec![42]);

        // Add a large 64-bit register (larger than u32::MAX)
        let large_value = 1u64 << 34; // 2^34 = 17,179,869,184 (>4B)
        shot_results
            .register_shots_u64
            .insert("reg_64".to_string(), vec![large_value]);

        // Add a signed 64-bit register with negative value
        shot_results
            .register_shots_i64
            .insert("reg_signed".to_string(), vec![-42]);

        // Add a register that exists in multiple formats (should only display once)
        shot_results
            .register_shots
            .insert("multi_format".to_string(), vec![100]);
        shot_results
            .register_shots_u64
            .insert("multi_format".to_string(), vec![100]);

        // Convert to string
        let json_string = shot_results.to_json(); // Default to pretty format
        let display_string = format!("{shot_results}");

        // Print the actual JSON for debugging
        println!("PRETTY JSON STRING: {json_string}");

        // The display string should match the pretty JSON string in content
        // but not necessarily order (HashMap order isn't guaranteed)
        // Instead, verify that both are valid JSON and contain the same data
        let json_value1: serde_json::Value = serde_json::from_str(&display_string).unwrap();
        let json_value2: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        // Verify that both contain the same registers with the same values
        assert_eq!(
            json_value1.as_object().unwrap().len(),
            json_value2.as_object().unwrap().len(),
            "JSON objects should have the same number of keys"
        );

        // Verify that all registers appear in the JSON (with more flexible checks)
        assert!(json_string.contains("\"reg_32\""));
        assert!(json_string.contains("42")); // Number could be formatted differently
        assert!(json_string.contains("\"reg_64\""));
        assert!(json_string.contains("17179869184"));
        assert!(json_string.contains("\"reg_signed\""));
        assert!(json_string.contains("-42"));

        // Verify that multi_format register appears only once
        let count = json_string.matches("multi_format").count();
        assert_eq!(count, 1, "multi_format should appear exactly once");

        // Now test the shot-based format by clearing register data
        // and setting up the shots vector directly
        shot_results = ShotResults::new(); // Create a fresh instance

        // Create a single shot with various registers
        let mut shot_map = HashMap::new();
        shot_map.insert("reg_32".to_string(), "42".to_string());
        shot_map.insert("reg_64".to_string(), "17179869184".to_string());
        shot_map.insert("reg_signed".to_string(), "-42".to_string());

        // Add the shot to results
        shot_results.shots.push(shot_map);

        // Format and check output
        let shot_json = shot_results.to_json();
        let shot_display = format!("{shot_results}");

        // Verify that both are valid JSON and contain the same data
        let json_value1: serde_json::Value = serde_json::from_str(&shot_display).unwrap();
        let json_value2: serde_json::Value = serde_json::from_str(&shot_json).unwrap();

        // Verify both have the same number of keys
        assert_eq!(
            json_value1.as_array().unwrap().len(),
            json_value2.as_array().unwrap().len(),
            "JSON arrays should have the same number of elements"
        );

        // Check the content
        assert!(shot_json.contains("\"reg_32\""));
        assert!(shot_json.contains("\"42\""));
        assert!(shot_json.contains("\"reg_64\""));
        assert!(shot_json.contains("\"17179869184\""));
        assert!(shot_json.contains("\"reg_signed\""));
        assert!(shot_json.contains("\"-42\""));
    }

    #[test]
    fn test_shot_results_format_options() {
        // Create shot results with multiple shots
        let mut shot_results = ShotResults::new();

        // Add register data for 3 shots
        shot_results
            .register_shots
            .insert("c".to_string(), vec![0, 3, 2]);
        shot_results
            .register_shots
            .insert("q".to_string(), vec![1, 0, 1]);

        // Test compact format
        let compact_json = shot_results.to_string_with_format(OutputFormat::CompactJson);
        println!("COMPACT FORMAT: {compact_json}");

        // Compact format should not have newlines
        assert!(!compact_json.contains('\n'));
        // Still should contain the data
        assert!(compact_json.contains("\"c\":[0,3,2]"));

        // Test pretty compact format
        let pretty_compact_json =
            shot_results.to_string_with_format(OutputFormat::PrettyCompactJson);
        println!("PRETTY COMPACT FORMAT: \n{pretty_compact_json}");

        // Pretty compact format should have newlines but minimal indentation
        assert!(pretty_compact_json.contains('\n'));
        // Each register should be on its own line
        assert!(pretty_compact_json.matches('\n').count() >= 3); // At least 3 newlines (opening, 2 registers, closing)
        // Should contain the data
        assert!(pretty_compact_json.contains("\"c\":[0,3,2]"));
        assert!(pretty_compact_json.contains("\"q\":[1,0,1]"));

        // Test pretty format
        let pretty_json = shot_results.to_string_with_format(OutputFormat::PrettyJson);
        println!("PRETTY FORMAT: {pretty_json}");

        // Pretty format should have newlines and spacing
        assert!(pretty_json.contains('\n'));
        assert!(pretty_json.contains("  "));
    }
}
