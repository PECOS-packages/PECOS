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

//! Shot results and data structures for quantum program execution.
//!
//! This module provides comprehensive data structures for storing and manipulating
//! the results of quantum program executions. It includes:
//!
//! - **Data Types**: The `Data` enum for flexible value storage
//! - **Single Results**: The `Shot` struct for individual execution results
//! - **Collections**: The `ShotVec` struct for multiple executions
//! - **Columnar Analysis**: The `ShotMap` struct for efficient analysis
//! - **Formatting**: Display and export utilities
//!
//! # Design Philosophy
//!
//! The module is designed around the following principles:
//! - **Flexibility**: Support for diverse data types and quantum backends
//! - **Efficiency**: Optimized for common operations like analysis and export
//! - **Compatibility**: Easy conversion between row-based and columnar formats
//! - **Extensibility**: JSON support for custom and complex data
//!
//! # Main Types
//!
//! ## `Data` - Flexible Value Storage
//! ```
//! use pecos_results::Data;
//! use bitvec::prelude::*;
//!
//! // Support for various numeric types
//! let measurement = Data::U32(42);
//! let phase = Data::F64(3.14159);
//!
//! // BitVec for quantum register results
//! let mut bits = BitVec::<u8, bitvec::order::Lsb0>::new();
//! bits.push(true);
//! bits.push(false);
//! let register = Data::BitVec(bits);
//! ```
//!
//! ## `Shot` - Single Execution Results
//! ```
//! use pecos_results::{Shot, Data};
//!
//! let mut shot = Shot::default();
//! shot.add_register("qubits", 5, 3);  // 3-bit register with value 5
//! shot.data.insert("error_rate".to_string(), Data::F64(0.001));
//! ```
//!
//! ## `ShotVec` - Multiple Executions
//! ```
//! use pecos_results::{ShotVec, Shot};
//!
//! let mut results = ShotVec::new();
//! for i in 0..100 {
//!     let mut shot = Shot::default();
//!     shot.add_register("measurement", i % 8, 3);
//!     results.shots.push(shot);
//! }
//!
//! // Convert to JSON for export
//! let json = results.to_compact_json();
//! ```
//!
//! ## `ShotMap` - Columnar Analysis
//! ```
//! # use pecos_results::{ShotVec, Shot};
//! # let mut results = ShotVec::new();
//! # for i in 0..100 {
//! #     let mut shot = Shot::default();
//! #     shot.add_register("measurement", i % 8, 3);
//! #     results.shots.push(shot);
//! # }
//! // Convert to columnar format for analysis
//! let shot_map = results.try_as_shot_map().unwrap();
//!
//! // Efficient analysis of specific registers
//! let measurements = shot_map.try_bits_as_u64("measurement").unwrap();
//! let average: f64 = measurements.iter().sum::<u64>() as f64 / measurements.len() as f64;
//! ```

#![allow(clippy::similar_names)]
// For percentage calculations below with large usize values converted to f64,
// we accept the potential precision loss since the values are used only for display
// with a single decimal place, and the precision loss would only be observable
// with extremely large shot counts (> 2^53).
#![allow(clippy::cast_precision_loss)]

// Sub-modules
pub mod conversions;
pub mod data;
pub mod data_vec;
pub mod shot;
pub mod shot_map;
pub mod shot_map_formatter;
#[cfg(test)]
mod shot_tests;
pub mod shot_vec;

// Re-export all public types for backward compatibility
pub use data::Data;
pub use data_vec::{DataVec, DataVecType};
pub use shot::Shot;
pub use shot_map::ShotMap;
pub use shot_map_formatter::{
    BitVecDisplayFormat, ShotMapDisplay, ShotMapDisplayExt, ShotMapDisplayOptions,
};
pub use shot_vec::ShotVec;

// Re-export for tests and benchmarks that may reference the full module path
#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;

    #[test]
    fn test_shot_results_display_64bit() {
        // Create a shot with various data types
        let mut shot1 = Shot::default();
        shot1.data.insert("reg_32".to_string(), Data::U32(42));

        // Add a large 64-bit register (larger than u32::MAX)
        let large_value = 1u64 << 34; // 2^34 = 17,179,869,184 (>4B)
        shot1
            .data
            .insert("reg_64".to_string(), Data::U64(large_value));

        // Add a signed 64-bit register with negative value
        shot1.data.insert("reg_signed".to_string(), Data::I64(-42));

        // Add some floating point data
        shot1
            .data
            .insert("float_val".to_string(), Data::F64(std::f64::consts::PI));

        // Create ShotVec with one shot
        let shot_results = ShotVec { shots: vec![shot1] };

        // Convert to string
        let json_string = shot_results.to_compact_json();
        let display_string = format!("{shot_results}");

        // The display string should match the compact JSON string
        assert_eq!(display_string, json_string);

        // Verify that both are valid JSON and contain the same data
        let json_value1: serde_json::Value = serde_json::from_str(&display_string).unwrap();
        let json_value2: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        // Verify that both are arrays with the same length
        assert_eq!(
            json_value1.as_array().unwrap().len(),
            json_value2.as_array().unwrap().len(),
            "JSON arrays should have the same number of shots"
        );

        // Verify that all registers appear in the JSON
        assert!(json_string.contains("\"reg_32\""));
        assert!(json_string.contains("42"));
        assert!(json_string.contains("\"reg_64\""));
        assert!(json_string.contains("17179869184"));
        assert!(json_string.contains("\"reg_signed\""));
        assert!(json_string.contains("-42"));
        assert!(json_string.contains("\"float_val\""));
        assert!(json_string.contains("3.14159"));
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together correctly
        let mut shot_vec = ShotVec::new();

        for i in 0..5 {
            let mut shot = Shot::default();
            shot.add_register("qubits", i, 3);
            shot.data
                .insert("phase".to_string(), Data::F64(f64::from(i) * 0.1));
            shot_vec.shots.push(shot);
        }

        // Convert to ShotMap
        let shot_map = shot_vec.try_as_shot_map().unwrap();

        // Test data access
        assert_eq!(shot_map.num_shots(), 5);
        assert_eq!(shot_map.num_registers(), 2); // qubits + phase (width metadata filtered out)

        // Test formatting
        let display_output = format!("{}", shot_map.display());
        assert!(display_output.contains("\"qubits\""));
        assert!(display_output.contains("\"phase\""));
    }
}
