//! Formatter for displaying `ShotMap` data in various formats

use super::{DataVec, ShotMap};
use ::bitvec::prelude::*;
use pecos_core::bitvec;
use std::fmt;

/// Display options for formatting `ShotMap` data
#[derive(Debug, Clone)]
pub struct ShotMapDisplayOptions {
    /// How to display `BitVec` data
    pub bitvec_format: BitVecDisplayFormat,
    /// Maximum number of shots to display (None = all)
    pub max_shots: Option<usize>,
    /// Whether to sort registers alphabetically
    pub sort_registers: bool,
    /// Indentation for nested structures
    pub indent: String,
}

impl Default for ShotMapDisplayOptions {
    fn default() -> Self {
        Self {
            bitvec_format: BitVecDisplayFormat::BinaryPrefixed,
            max_shots: None,
            sort_registers: true,
            indent: "  ".to_string(),
        }
    }
}

/// Format options for displaying `BitVec` data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitVecDisplayFormat {
    /// Display as binary strings with prefix (e.g., "0b101")
    BinaryPrefixed,
    /// Display as binary strings without prefix (e.g., "101")
    Binary,
    /// Display as decimal values (e.g., 5)
    Decimal,
    /// Display as hexadecimal values (e.g., "0x5")
    Hexadecimal,
    /// Display as array of booleans (e.g., [true, false, true])
    BoolArray,
}

/// A wrapper type for formatting `ShotMap` in a human-readable way
pub struct ShotMapDisplay<'a> {
    map: &'a ShotMap,
    options: ShotMapDisplayOptions,
}

impl<'a> ShotMapDisplay<'a> {
    /// Create a new display wrapper with default options
    #[must_use]
    pub fn new(map: &'a ShotMap) -> Self {
        Self {
            map,
            options: ShotMapDisplayOptions::default(),
        }
    }

    /// Set the `BitVec` display format
    #[must_use]
    pub fn bitvec_format(mut self, format: BitVecDisplayFormat) -> Self {
        self.options.bitvec_format = format;
        self
    }

    /// Display `BitVecs` as binary strings with prefix (e.g., "0b101")
    #[must_use]
    pub fn bitvec_binary_prefixed(mut self) -> Self {
        self.options.bitvec_format = BitVecDisplayFormat::BinaryPrefixed;
        self
    }

    /// Display `BitVecs` as binary strings without prefix (e.g., "101")
    #[must_use]
    pub fn bitvec_binary(mut self) -> Self {
        self.options.bitvec_format = BitVecDisplayFormat::Binary;
        self
    }

    /// Display `BitVecs` as decimal values (e.g., 5)
    #[must_use]
    pub fn bitvec_decimal(mut self) -> Self {
        self.options.bitvec_format = BitVecDisplayFormat::Decimal;
        self
    }

    /// Display `BitVecs` as hexadecimal values (e.g., "0x5")
    #[must_use]
    pub fn bitvec_hex(mut self) -> Self {
        self.options.bitvec_format = BitVecDisplayFormat::Hexadecimal;
        self
    }

    /// Display `BitVecs` as boolean arrays (e.g., [true, false, true])
    #[must_use]
    pub fn bitvec_bool_array(mut self) -> Self {
        self.options.bitvec_format = BitVecDisplayFormat::BoolArray;
        self
    }

    /// Set maximum number of shots to display
    #[must_use]
    pub fn max_shots(mut self, max: usize) -> Self {
        self.options.max_shots = Some(max);
        self
    }

    /// Enable/disable register sorting
    #[must_use]
    pub fn sort_registers(mut self, sort: bool) -> Self {
        self.options.sort_registers = sort;
        self
    }

    /// Set custom indentation
    #[must_use]
    pub fn indent(mut self, indent: impl Into<String>) -> Self {
        self.options.indent = indent.into();
        self
    }

    /// Apply custom display options
    #[must_use]
    pub fn with_options(mut self, options: ShotMapDisplayOptions) -> Self {
        self.options = options;
        self
    }

    /// Format a `BitVec` according to the current options
    fn format_bitvec(&self, bitvec: &BitVec<u8>) -> String {
        match self.options.bitvec_format {
            BitVecDisplayFormat::BinaryPrefixed => {
                format!("0b{}", bitvec::to_bitstring(bitvec))
            }
            BitVecDisplayFormat::Binary => {
                format!("\"{}\"", bitvec::to_bitstring(bitvec))
            }
            BitVecDisplayFormat::Decimal => bitvec::to_decimal_string(bitvec),
            BitVecDisplayFormat::Hexadecimal => {
                bitvec::to_hex_string(bitvec) // Already includes "0x" prefix
            }
            BitVecDisplayFormat::BoolArray => bitvec::to_bool_array(bitvec),
        }
    }

    /// Format a single value from a `DataVec`
    fn format_value(&self, data_vec: &DataVec, index: usize) -> Option<String> {
        match data_vec {
            DataVec::U8(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::U16(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::U32(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::U64(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::I8(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::I16(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::I32(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::I64(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::F32(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::F64(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::String(v) => v.get(index).map(|x| format!("\"{x}\"")),
            DataVec::Bool(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::BigInt(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::Bytes(v) => v.get(index).map(|x| format!("{x:?}")),
            DataVec::BitVec(v) => v.get(index).map(|x| self.format_bitvec(x)),
            DataVec::Json(v) => v.get(index).map(std::string::ToString::to_string),
            DataVec::Vec(v) => v.get(index).map(|inner| {
                let items: Vec<String> =
                    inner.iter().map(std::string::ToString::to_string).collect();
                format!("[{}]", items.join(", "))
            }),
        }
    }
}

impl fmt::Display for ShotMapDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_shots = self.map.num_shots();
        let max_shots = self.options.max_shots.unwrap_or(num_shots).min(num_shots);

        // Get and optionally sort register names
        let mut registers: Vec<_> = self.map.register_names();
        if self.options.sort_registers {
            registers.sort_unstable();
        }

        write!(f, "{{")?;

        // Display each register
        let mut first_register = true;
        for register in registers {
            if let Some(data_vec) = self.map.get(register) {
                if !first_register {
                    write!(f, ", ")?;
                }
                first_register = false;

                write!(f, "\"{register}\": [")?;

                // Display values
                let mut first_value = true;
                for i in 0..max_shots {
                    if let Some(value) = self.format_value(data_vec, i) {
                        if !first_value {
                            write!(f, ", ")?;
                        }
                        write!(f, "{value}")?;
                        first_value = false;
                    }
                }

                if max_shots < data_vec.len() {
                    write!(f, ", ...")?;
                }

                write!(f, "]")?;
            }
        }

        write!(f, "}}")
    }
}

/// Extension trait to add display methods to `ShotMap`
pub trait ShotMapDisplayExt {
    /// Create a displayable wrapper for this `ShotMap`
    fn display(&self) -> ShotMapDisplay<'_>;

    /// Create a display with custom options
    fn display_with(&self, options: ShotMapDisplayOptions) -> ShotMapDisplay<'_>;
}

impl ShotMapDisplayExt for ShotMap {
    fn display(&self) -> ShotMapDisplay<'_> {
        ShotMapDisplay::new(self)
    }

    fn display_with(&self, options: ShotMapDisplayOptions) -> ShotMapDisplay<'_> {
        ShotMapDisplay::new(self).with_options(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shot_results::{Data, Shot, ShotVec};

    #[test]
    fn test_display_formatting() {
        let mut shot_vec = ShotVec::new();

        for i in 0..3 {
            let mut shot = Shot::default();
            shot.add_register("q", i, 3);
            shot.data.insert("count".to_string(), Data::U32(i));
            shot.data
                .insert("phase".to_string(), Data::F64(f64::from(i) * 0.5));
            shot_vec.shots.push(shot);
        }

        let shot_map = shot_vec.try_as_shot_map().unwrap();

        // Test default display (binary with prefix, no quotes)
        let display = format!("{}", shot_map.display());
        assert!(display.starts_with('{'));
        assert!(display.ends_with('}'));
        assert!(display.contains("0b000")); // Value 0 with prefix, no quotes
        assert!(display.contains("0b001")); // Value 1 with prefix, no quotes
        assert!(display.contains("0b010")); // Value 2 with prefix, no quotes

        // Test with binary format (no prefix, with quotes)
        let display_binary = format!("{}", shot_map.display().bitvec_binary());
        assert!(display_binary.contains("\"000\"")); // Value 0 with quotes
        assert!(display_binary.contains("\"001\"")); // Value 1 with quotes
        assert!(display_binary.contains("\"010\"")); // Value 2 with quotes

        // Test with decimal format
        let display_decimal = format!(
            "{}",
            shot_map
                .display()
                .bitvec_format(BitVecDisplayFormat::Decimal)
        );
        assert!(display_decimal.contains(": [0, 1, 2]"));

        // Test with hex format (no quotes)
        let display_hex = format!("{}", shot_map.display().bitvec_hex());
        assert!(display_hex.contains("0x0")); // Value 0 as hex, no quotes
        assert!(display_hex.contains("0x1")); // Value 1 as hex, no quotes
        assert!(display_hex.contains("0x2")); // Value 2 as hex, no quotes

        // Test JSON-like format
        let display_json = format!("{}", shot_map.display());
        assert!(display_json.contains("\"count\""));
        assert!(display_json.contains("\"phase\""));
        assert!(display_json.contains("\"q\""));
    }
}
