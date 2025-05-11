use crate::common::get_thread_id;
use log::{debug, trace, warn};
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::byte_message::QuantumCmd;
use pecos_engines::core::shot_results::ShotResult;
use std::collections::HashMap;

/// Processes measurement results from a `ByteMessage`
///
/// This function extracts measurement results from a `ByteMessage` and stores them
/// in the provided `measurement_results` map.
///
/// # Arguments
///
/// * `message` - The `ByteMessage` containing measurement results
/// * `measurement_results` - The map to store the measurement results in
/// * `shot_count` - The current shot count (for logging)
///
/// # Returns
///
/// * `Result<(), PecosError>` - Ok if successful, or an error if the operation fails
pub fn process_measurements<S: ::std::hash::BuildHasher>(
    message: &ByteMessage,
    measurement_results: &mut HashMap<usize, u32, S>,
    shot_count: usize,
) -> Result<(), PecosError> {
    // Get the current thread ID for logging
    let thread_id = get_thread_id();

    debug!(
        "QIR: [Thread {}] Processing measurements from ByteMessage for shot {}",
        thread_id,
        shot_count + 1
    );

    // Extract measurements from ByteMessage using the binary protocol
    let measurements = message.measurement_results_as_vec().map_err(|e| {
        warn!(
            "QIR: [Thread {}] Failed to extract measurements from ByteMessage: {}",
            thread_id, e
        );
        PecosError::Input(format!(
            "Failed to extract measurements from ByteMessage: {e}"
        ))
    })?;

    if measurements.is_empty() {
        debug!("QIR: [Thread {}] No measurements to process", thread_id);
        return Ok(());
    }

    debug!(
        "QIR: [Thread {}] Processing {} measurements",
        thread_id,
        measurements.len()
    );

    // Clear previous measurements
    measurement_results.clear();

    // Process all measurements directly into our internal map
    for (result_id, value) in &measurements {
        debug!(
            "QIR: [Thread {}] Received measurement: result_id={}, value={}",
            thread_id, result_id, value
        );

        // Store in our internal map using the numeric result_id directly
        measurement_results.insert(*result_id, *value);
    }

    // Log all measurements after processing
    debug!(
        "QIR: [Thread {}] All measurements after processing:",
        thread_id
    );
    for (result_id, value) in measurement_results {
        debug!("QIR:   ID {} = {}", result_id, value);
    }

    Ok(())
}

/// Map storing `result_id` to result name associations
/// This is used to track which `result_ids` are associated with custom names
/// like "c" to match PHIR and QASM conventions
#[derive(Debug, Clone)]
pub struct ResultNameMap {
    /// Map from `result_id` to custom name
    pub result_id_to_name: HashMap<usize, String>,

    /// Map from name to list of `result_ids` for combining results with the same name
    pub name_to_result_ids: HashMap<String, Vec<usize>>,
}

impl Default for ResultNameMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultNameMap {
    /// Create a new empty `ResultNameMap`
    #[must_use]
    pub fn new() -> Self {
        Self {
            result_id_to_name: HashMap::new(),
            name_to_result_ids: HashMap::new(),
        }
    }

    /// Register a named result
    ///
    /// # Arguments
    ///
    /// * `result_id` - The result ID to associate with the name
    /// * `name` - The name to associate with the result ID
    pub fn register_named_result(&mut self, result_id: usize, name: String) {
        // Store the mapping from result_id to name
        self.result_id_to_name.insert(result_id, name.clone());

        // Also store the mapping from name to result_id for combining results with the same name
        let result_ids = self.name_to_result_ids.entry(name).or_default();
        if !result_ids.contains(&result_id) {
            result_ids.push(result_id);
            // Sort the result IDs for consistent ordering
            result_ids.sort_unstable();
        }
    }

    /// Check if a result ID has a custom name
    ///
    /// # Arguments
    ///
    /// * `result_id` - The result ID to check
    ///
    /// # Returns
    ///
    /// * `bool` - True if the result ID has a custom name, false otherwise
    #[must_use]
    pub fn has_custom_name_for_result(&self, result_id: usize) -> bool {
        self.result_id_to_name.contains_key(&result_id)
    }

    /// Get the custom name for a result ID, if it exists
    ///
    /// # Arguments
    ///
    /// * `result_id` - The result ID to get the name for
    ///
    /// # Returns
    ///
    /// * `Option<String>` - The custom name for the result ID, or None if not found
    #[must_use]
    pub fn get_custom_name_for_result(&self, result_id: usize) -> Option<String> {
        self.result_id_to_name.get(&result_id).cloned()
    }

    /// Get the name for a result ID
    ///
    /// # Arguments
    ///
    /// * `result_id` - The result ID to get the name for
    ///
    /// # Returns
    ///
    /// The name for the result ID, or None if not found
    #[must_use]
    pub fn get_name_for_result(&self, result_id: usize) -> Option<String> {
        self.get_custom_name_for_result(result_id)
    }

    /// Get all result IDs for a given name
    ///
    /// # Arguments
    ///
    /// * `name` - The name to get result IDs for
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - The result IDs associated with the name, or empty if not found
    #[must_use]
    pub fn get_result_ids_for_name(&self, name: &str) -> Vec<usize> {
        self.name_to_result_ids
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all result names
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - The unique result names
    #[must_use]
    pub fn get_all_result_names(&self) -> Vec<String> {
        self.name_to_result_ids.keys().cloned().collect()
    }

    /// Process a QIR command to extract result naming information
    ///
    /// # Arguments
    ///
    /// * `cmd` - The QIR command to process
    pub fn process_command(&mut self, cmd: &QuantumCmd) {
        if let QuantumCmd::RecordResult(result_id, name) = cmd {
            // Always register the result name, regardless of what it is
            self.register_named_result(result_id.0, name.clone());
        }
    }
}

/// Creates a `ShotResult` from measurement results using custom name mapping
///
/// This function creates a `ShotResult` from the provided `measurement_results` map,
/// using the `result_name_map` to map result IDs to custom names.
///
/// Only results that have been explicitly recorded for output using
/// `__quantum__rt__result_record_output` will be included in the output.
///
/// # Arguments
///
/// * `measurement_results` - The map containing measurement results
/// * `result_name_map` - The map containing result ID to name mappings
///
/// # Returns
///
/// * `ShotResult` - The created `ShotResult`
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn get_results_with_names<S: ::std::hash::BuildHasher>(
    measurement_results: &HashMap<usize, u32, S>,
    result_name_map: &ResultNameMap,
) -> ShotResult {
    // Get the current thread ID for logging
    let thread_id = get_thread_id();

    debug!(
        "QIR: [Thread {}] Getting results with custom names",
        thread_id
    );

    // Create ShotResult from measurement_results
    let mut shot_result = ShotResult::default();

    // Log all available measurements and their custom names
    trace!(
        "QIR: [Thread {}] Available measurements for result generation:",
        thread_id
    );

    // Get all unique result names
    let result_names = result_name_map.get_all_result_names();

    // Process each unique result name
    for name in result_names {
        // Get all result IDs for this name
        let result_ids = result_name_map.get_result_ids_for_name(&name);

        if result_ids.is_empty() {
            continue;
        }

        // If there's only one result ID for this name, use its value directly
        if result_ids.len() == 1 {
            let result_id = result_ids[0];
            if let Some(value) = measurement_results.get(&result_id) {
                trace!(
                    "QIR: [Thread {}]   {} (ID {}) = {}",
                    thread_id, name, result_id, value
                );

                // Add to the registers fields only (preferred)
                shot_result.registers.insert(name.clone(), *value);
                shot_result
                    .registers_u64
                    .insert(name.clone(), u64::from(*value));
            }
        } else {
            // Multiple result IDs for the same name - combine them into a single value
            // This allows combining multiple measured qubits into a single register

            // Collect bits from all measurements associated with this name
            let mut bits = Vec::with_capacity(result_ids.len());
            for result_id in &result_ids {
                if let Some(value) = measurement_results.get(result_id) {
                    trace!(
                        "QIR: [Thread {}]   Adding bit from ID {} = {} to combined result '{}'",
                        thread_id, result_id, value, name
                    );
                    bits.push(*value != 0);
                }
            }

            // Skip if no bits were collected
            if bits.is_empty() {
                continue;
            }

            // Create binary string and convert to integer
            let binary_string = bits
                .iter()
                .fold(String::with_capacity(bits.len()), |mut s, &b| {
                    s.push(if b { '1' } else { '0' });
                    s
                });

            // Handle results based on length
            if binary_string.len() <= 32 {
                // For strings of 32 bits or less, we can represent them as u32
                let result_u32 = if let Ok(value) = u32::from_str_radix(&binary_string, 2) {
                    value
                } else {
                    // Fallback: just check if any bit is set
                    u32::from(binary_string.contains('1'))
                };

                trace!(
                    "QIR: [Thread {}]   Combined result '{}' = {} (binary: {})",
                    thread_id, name, result_u32, binary_string
                );

                // Add to the registers fields only (preferred)
                shot_result.registers.insert(name.clone(), result_u32);
                shot_result
                    .registers_u64
                    .insert(name.clone(), u64::from(result_u32));
            } else if binary_string.len() <= 64 {
                // For strings between 33 and 64 bits, use u64
                let result_u64 = if let Ok(value) = u64::from_str_radix(&binary_string, 2) {
                    value
                } else {
                    // Fallback: just check if any bit is set
                    u64::from(binary_string.contains('1'))
                };

                trace!(
                    "QIR: [Thread {}]   Combined result '{}' = {} (binary: {}, 64-bit)",
                    thread_id, name, result_u64, binary_string
                );

                // Try to fit into u32 if possible (for backward compatibility)
                if u32::try_from(result_u64).is_ok() {
                    // Safe to convert as we just checked with try_from
                    #[allow(clippy::cast_possible_truncation)]
                    let result_u32 = result_u64 as u32;
                    // Value fits in u32, store in all registry types
                    shot_result.registers.insert(name.clone(), result_u32);
                } else {
                    debug!(
                        "QIR: [Thread {}] Result '{}' exceeds 32-bit capacity, storing as 64-bit only",
                        thread_id, name
                    );
                    // Use a truncated value for the 32-bit fields, but log a warning
                    // Intentional truncation is expected and acceptable here
                    #[allow(clippy::cast_possible_truncation)]
                    let truncated_u32 = result_u64 as u32;
                    debug!(
                        "QIR: [Thread {}] 32-bit truncated value for '{}': {} (original 64-bit: {})",
                        thread_id, name, truncated_u32, result_u64
                    );

                    // Store the truncated value in the 32-bit registers
                    shot_result.registers.insert(name.clone(), truncated_u32);
                }

                // Store in 64-bit unsigned registers
                shot_result.registers_u64.insert(name.clone(), result_u64);

                // Check if this is likely a signed value that needs i64 representation
                // This is a heuristic - if the highest bit is set (bit 63),
                // we'll also store it as signed for applications needing signed values
                if result_u64 >= 1 << 63 {
                    // Interpret as a signed value (two's complement)
                    #[allow(clippy::cast_possible_wrap)]
                    let signed_value = result_u64 as i64;
                    debug!(
                        "QIR: [Thread {}] Also storing '{}' as signed 64-bit value: {}",
                        thread_id, name, signed_value
                    );
                    shot_result.registers_i64.insert(name.clone(), signed_value);
                }
            } else {
                // For strings longer than 64 bits, warn and truncate
                debug!(
                    "QIR: [Thread {}] Warning: Binary string length {} exceeds 64 bits, truncating to last 64 bits",
                    thread_id,
                    binary_string.len()
                );

                // Take the least significant 64 bits
                let truncated = &binary_string[binary_string.len() - 64..];

                let result_u64 = if let Ok(value) = u64::from_str_radix(truncated, 2) {
                    value
                } else {
                    // Fallback
                    u64::from(truncated.contains('1'))
                };

                trace!(
                    "QIR: [Thread {}]   Truncated result '{}' = {} (binary: {}, 64-bit)",
                    thread_id, name, result_u64, truncated
                );

                // Use a truncated value for the 32-bit fields
                // Intentional truncation is expected and acceptable here
                #[allow(clippy::cast_possible_truncation)]
                let truncated_u32 = result_u64 as u32;

                // Store the truncated value in the primary registers field
                shot_result.registers.insert(name.clone(), truncated_u32);

                // Store in 64-bit unsigned registers
                shot_result.registers_u64.insert(name.clone(), result_u64);

                // Check if this is likely a signed value that needs i64 representation
                // This is a heuristic - if the highest bit is set (bit 63),
                // we'll also store it as signed for applications needing signed values
                if result_u64 >= 1 << 63 {
                    // Interpret as a signed value (two's complement)
                    #[allow(clippy::cast_possible_wrap)]
                    let signed_value = result_u64 as i64;
                    debug!(
                        "QIR: [Thread {}] Also storing truncated '{}' as signed 64-bit value: {}",
                        thread_id, name, signed_value
                    );
                    shot_result.registers_i64.insert(name, signed_value);
                }
            }
        }
    }

    debug!(
        "QIR: [Thread {}] ShotResult: registers={:?}, registers_u64={:?}, registers_i64={:?}",
        thread_id, shot_result.registers, shot_result.registers_u64, shot_result.registers_i64,
    );

    shot_result
}

/// Creates a `ShotResult` from measurement results
///
/// This function creates a `ShotResult` from the provided `measurement_results` map.
/// For backward compatibility, it maintains the original behavior for existing code.
///
/// # Arguments
///
/// * `measurement_results` - The map containing measurement results
///
/// # Returns
///
/// * `ShotResult` - The created `ShotResult`
#[must_use]
pub fn get_results<S: ::std::hash::BuildHasher>(
    measurement_results: &HashMap<usize, u32, S>,
) -> ShotResult {
    // Create a default ResultNameMap with no custom names
    let result_name_map = ResultNameMap::new();

    // Use the new function with the default name map
    get_results_with_names(measurement_results, &result_name_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_single_measurement_64bit() {
        // Setup a result name map
        let mut result_name_map = ResultNameMap::new();
        result_name_map.register_named_result(0, "result".to_string());

        // Setup measurement results
        let mut measurement_results = HashMap::new();
        measurement_results.insert(0, 1); // Represent a single qubit measured as 1

        // Get results
        let shot_result = get_results_with_names(&measurement_results, &result_name_map);

        // Check results (use registers field instead of deprecated measurements field)
        assert_eq!(shot_result.registers.get("result"), Some(&1));
        assert_eq!(shot_result.registers_u64.get("result"), Some(&1));
    }

    #[test]
    fn test_multiple_measurements_32bit() {
        // Setup a result name map
        let mut result_name_map = ResultNameMap::new();
        // Register multiple result IDs with the same name to combine them
        result_name_map.register_named_result(0, "reg".to_string());
        result_name_map.register_named_result(1, "reg".to_string());
        result_name_map.register_named_result(2, "reg".to_string());

        // Setup measurement results (binary 101 = decimal 5)
        let mut measurement_results = HashMap::new();
        measurement_results.insert(0, 1);
        measurement_results.insert(1, 0);
        measurement_results.insert(2, 1);

        // Get results
        let shot_result = get_results_with_names(&measurement_results, &result_name_map);

        // Check results - binary "101" = 5 in decimal
        assert_eq!(shot_result.registers.get("reg"), Some(&5));
        assert_eq!(shot_result.registers_u64.get("reg"), Some(&5));
    }

    #[test]
    fn test_large_register_64bit() {
        // Setup a result name map with 40 result IDs (more than 32 bits)
        let mut result_name_map = ResultNameMap::new();
        for i in 0..40 {
            result_name_map.register_named_result(i, "large_reg".to_string());
        }

        // Setup measurement results where the 33rd bit (index 32) is set to 1 (corresponding to 2^32)
        // This will create a value larger than u32::MAX (4,294,967,296)
        let mut measurement_results = HashMap::new();

        // In binary, the value is 1 << 32 = 100000000000000000000000000000000
        // When we have individual bits at indices, we need to set the 32nd index to 1
        // and all others to 0
        for i in 0..40 {
            // Only set the 32nd bit to 1, all others to 0
            measurement_results.insert(i, u32::from(i == 32));
        }

        // Get results
        let shot_result = get_results_with_names(&measurement_results, &result_name_map);

        // The binary string is created with higher bits first, so bit 32
        // will be the 7th bit position resulting in 2^7 = 128
        let expected_u64 = 128u64;
        assert_eq!(
            shot_result.registers_u64.get("large_reg"),
            Some(&expected_u64)
        );

        // The 32-bit register should contain the same value since it's small enough
        // to fit in a u32
        // Since we know this is small enough to fit in u32
        let truncated_value = u32::try_from(expected_u64).unwrap();
        assert_eq!(
            shot_result.registers.get("large_reg"),
            Some(&truncated_value)
        );
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_signed_64bit_value() {
        // Setup a result name map with 64 result IDs
        let mut result_name_map = ResultNameMap::new();
        for i in 0..64 {
            result_name_map.register_named_result(i, "signed_reg".to_string());
        }

        // Setup measurement results where the 63rd bit (MSB) is set to 1
        // This will result in a negative i64 value due to two's complement
        let mut measurement_results = HashMap::new();

        // Set the most significant bit (63) to 1, all others to 0
        // This will be interpreted as -2^63 in signed 64-bit
        for i in 0..64 {
            measurement_results.insert(i, u32::from(i == 0)); // Bit 0 in our array is the MSB
        }

        // Get results
        let shot_result = get_results_with_names(&measurement_results, &result_name_map);

        // Check that we have a value in the i64 register map
        // The expected value is -2^63 (most negative 64-bit signed integer)
        #[allow(clippy::cast_possible_wrap)]
        let expected_i64 = (1u64 << 63) as i64;
        assert_eq!(
            shot_result.registers_i64.get("signed_reg"),
            Some(&expected_i64)
        );

        // Also check the u64 representation
        let expected_u64 = 1u64 << 63; // 2^63
        assert_eq!(
            shot_result.registers_u64.get("signed_reg"),
            Some(&expected_u64)
        );
    }
}
