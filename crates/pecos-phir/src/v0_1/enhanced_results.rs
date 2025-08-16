use crate::v0_1::environment::{BoolBit, Environment};
use pecos_core::errors::PecosError;
use std::collections::HashMap;

/// Enhanced result handling functions for PHIR
/// These provide similar functionality to the Python PHIR interpreter's result handling
pub trait EnhancedResultHandling {
    /// Get a specific bit from a variable
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist or the bit index is out of range.
    fn get_result_bit(&self, var_name: &str, bit_index: usize) -> Result<BoolBit, PecosError>;

    /// Get multiple bits from a variable
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist or any bit index is out of range.
    fn get_result_bits(
        &self,
        var_name: &str,
        bit_indices: &[usize],
    ) -> Result<Vec<BoolBit>, PecosError>;

    /// Convert a variable to a bit string
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist.
    fn get_result_as_bit_string(
        &self,
        var_name: &str,
        width: Option<usize>,
    ) -> Result<String, PecosError>;

    /// Convert a variable to a binary string (like '0b101')
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist.
    fn get_result_as_binary_string(&self, var_name: &str) -> Result<String, PecosError>;

    /// Get results with various formats
    fn get_formatted_results(&self, format: ResultFormat) -> HashMap<String, String>;
}

/// Format options for result values
pub enum ResultFormat {
    /// Integer format (decimal)
    Integer,
    /// Hexadecimal format (0x...)
    Hex,
    /// Binary format (0b...)
    Binary,
    /// Bit string format (0101...)
    BitString(usize), // Width of the bit string
}

impl EnhancedResultHandling for Environment {
    fn get_result_bit(&self, var_name: &str, bit_index: usize) -> Result<BoolBit, PecosError> {
        self.get_bit(var_name, bit_index)
    }

    fn get_result_bits(
        &self,
        var_name: &str,
        bit_indices: &[usize],
    ) -> Result<Vec<BoolBit>, PecosError> {
        let mut result = Vec::with_capacity(bit_indices.len());

        for &idx in bit_indices {
            let bit = self.get_bit(var_name, idx)?;
            result.push(bit);
        }

        Ok(result)
    }

    fn get_result_as_bit_string(
        &self,
        var_name: &str,
        width: Option<usize>,
    ) -> Result<String, PecosError> {
        if let Some(value) = self.get(var_name) {
            let bits = format!("{:b}", value.as_u64());

            if let Some(width) = width {
                // Pad with zeros to the specified width
                Ok(format!("{bits:0>width$}"))
            } else {
                // Return as is
                Ok(bits)
            }
        } else {
            Err(PecosError::Input(format!(
                "Variable '{var_name}' not found"
            )))
        }
    }

    fn get_result_as_binary_string(&self, var_name: &str) -> Result<String, PecosError> {
        if let Some(value) = self.get(var_name) {
            let bits = format!("{:b}", value.as_u64());
            Ok(format!("0b{bits}"))
        } else {
            Err(PecosError::Input(format!(
                "Variable '{var_name}' not found"
            )))
        }
    }

    fn get_formatted_results(&self, format: ResultFormat) -> HashMap<String, String> {
        let mut results = HashMap::new();

        // Process all mappings first
        for (source, dest) in self.get_mappings() {
            if let Some(value) = self.get(source) {
                let formatted = match format {
                    ResultFormat::Integer => value.to_string(),
                    ResultFormat::Hex => format!("0x{:x}", value.as_u64()),
                    ResultFormat::Binary => format!("0b{:b}", value.as_u64()),
                    ResultFormat::BitString(width) => {
                        format!("{:0>width$b}", value.as_u64(), width = width)
                    }
                };
                results.insert(dest.clone(), formatted);
            }
        }

        // If no mappings exist, include all measurement variables (those starting with 'm')
        if results.is_empty() {
            for info in self.get_all_variables() {
                if (info.name.starts_with('m') || info.name.starts_with("measurement"))
                    && let Some(value) = self.get(&info.name)
                {
                    let formatted = match format {
                        ResultFormat::Integer => value.to_string(),
                        ResultFormat::Hex => format!("0x{:x}", value.as_u64()),
                        ResultFormat::Binary => format!("0b{:b}", value.as_u64()),
                        ResultFormat::BitString(width) => {
                            format!("{:0>width$b}", value.as_u64(), width = width)
                        }
                    };
                    results.insert(info.name.clone(), formatted);
                }
            }
        }

        // If still empty, include all variables
        if results.is_empty() {
            for info in self.get_all_variables() {
                if let Some(value) = self.get(&info.name) {
                    let formatted = match format {
                        ResultFormat::Integer => value.to_string(),
                        ResultFormat::Hex => format!("0x{:x}", value.as_u64()),
                        ResultFormat::Binary => format!("0b{:b}", value.as_u64()),
                        ResultFormat::BitString(width) => {
                            format!("{:0>width$b}", value.as_u64(), width = width)
                        }
                    };
                    results.insert(info.name.clone(), formatted);
                }
            }
        }

        results
    }
}

/// Utility functions to help with result processing
pub struct ResultUtils;

impl ResultUtils {
    /// Combines bits into a single integer
    #[must_use]
    pub fn bits_to_int(bits: &[BoolBit]) -> u64 {
        let mut result = 0u64;

        for (i, bit) in bits.iter().enumerate() {
            if bit.0 {
                result |= 1 << i;
            }
        }

        result
    }

    /// Combines bits into a single integer using the specified indices
    #[must_use]
    pub fn bits_to_int_with_indices(bits: &[BoolBit], indices: &[usize]) -> u64 {
        let mut result = 0u64;

        for (&bit, &idx) in bits.iter().zip(indices.iter()) {
            if bit.0 {
                result |= 1 << idx;
            }
        }

        result
    }

    /// Combines named result bits into a map of variable values
    #[must_use]
    pub fn named_bits_to_map(bit_map: &HashMap<String, Vec<BoolBit>>) -> HashMap<String, u64> {
        let mut result = HashMap::new();

        for (name, bits) in bit_map {
            result.insert(name.clone(), Self::bits_to_int(bits));
        }

        result
    }

    /// Combines a set of bit values at the specified indices
    ///
    /// # Errors
    /// Returns an error if the variable doesn't exist or any bit index is out of range.
    pub fn combine_bits(
        env: &Environment,
        var_name: &str,
        bit_indices: &[usize],
    ) -> Result<u64, PecosError> {
        let bits = env.get_result_bits(var_name, bit_indices)?;
        Ok(Self::bits_to_int_with_indices(&bits, bit_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::environment::DataType;

    #[test]
    fn test_get_result_bits() {
        let mut env = Environment::new();

        // Add a variable
        env.add_variable("register", DataType::U32, 32).unwrap();

        // Set the value to 0b10101 (21 in decimal)
        env.set_raw("register", 0b10101).unwrap();

        // Get individual bits
        let bit0 = env.get_result_bit("register", 0).unwrap();
        let bit1 = env.get_result_bit("register", 1).unwrap();
        let bit2 = env.get_result_bit("register", 2).unwrap();
        let bit3 = env.get_result_bit("register", 3).unwrap();
        let bit4 = env.get_result_bit("register", 4).unwrap();

        assert!(bit0.0); // LSB
        assert!(!bit1.0);
        assert!(bit2.0);
        assert!(!bit3.0);
        assert!(bit4.0);

        // Get multiple bits at once
        let indices = [0, 2, 4];
        let multi_bits = env.get_result_bits("register", &indices).unwrap();
        assert_eq!(multi_bits.len(), 3);
        assert!(multi_bits[0].0);
        assert!(multi_bits[1].0);
        assert!(multi_bits[2].0);

        // Combine bits into an integer using standard method (positions only)
        let value = ResultUtils::bits_to_int(&multi_bits);
        assert_eq!(value, 0b111);

        // Combine bits into an integer with indices preserved
        let value = ResultUtils::bits_to_int_with_indices(&multi_bits, &indices);
        assert_eq!(value, 0b10101);
    }

    #[test]
    fn test_formatted_results() {
        let mut env = Environment::new();

        // Add variables
        env.add_variable("m0", DataType::U8, 8).unwrap();
        env.add_variable("result", DataType::U8, 8).unwrap();

        // Set values
        env.set_raw("m0", 5).unwrap(); // 0b101
        env.set_raw("result", 10).unwrap(); // 0b1010

        // Add a mapping
        env.add_mapping("m0", "output").unwrap();

        // Get formatted results - should use mappings
        let int_results = env.get_formatted_results(ResultFormat::Integer);
        assert_eq!(int_results.get("output"), Some(&"5".to_string()));

        // Get binary results
        let bin_results = env.get_formatted_results(ResultFormat::Binary);
        assert_eq!(bin_results.get("output"), Some(&"0b101".to_string()));

        // Get hex results
        let hex_results = env.get_formatted_results(ResultFormat::Hex);
        assert_eq!(hex_results.get("output"), Some(&"0x5".to_string()));

        // Get bit string results with padding
        let bit_string_results = env.get_formatted_results(ResultFormat::BitString(8));
        assert_eq!(
            bit_string_results.get("output"),
            Some(&"00000101".to_string())
        );
    }

    #[test]
    fn test_result_utils_combine_bits() {
        let mut env = Environment::new();

        // Add a variable
        env.add_variable("bits", DataType::U16, 16).unwrap();

        // Set bits individually
        env.set_bit("bits", 0, true).unwrap();
        env.set_bit("bits", 2, true).unwrap();
        env.set_bit("bits", 4, true).unwrap();

        // Value should be 0b10101 = 21
        assert_eq!(env.get_raw("bits"), Some(21));

        // Combine specific bits
        let combined = ResultUtils::combine_bits(&env, "bits", &[0, 2, 4]).unwrap();
        assert_eq!(combined, 21);

        // Try a different combination
        let combined = ResultUtils::combine_bits(&env, "bits", &[1, 3]).unwrap();
        assert_eq!(combined, 0); // Both bits are 0

        // Try bits in different order
        let combined = ResultUtils::combine_bits(&env, "bits", &[4, 2, 0]).unwrap();
        assert_eq!(combined, 21); // Still 0b10101

        // Test indices are preserved correctly - should give 0b10001 = 17
        let combined = ResultUtils::combine_bits(&env, "bits", &[0, 4]).unwrap();
        assert_eq!(combined, 17);
    }
}
