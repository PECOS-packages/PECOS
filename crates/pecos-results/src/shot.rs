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

use super::data::Data;
use bitvec::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Represents the results of a single shot (execution) of a quantum program.
///
/// This struct contains a flexible mapping of data values for storing measurement
/// outcomes and other execution results. Complex or engine-specific data can be
/// stored using the `Data::Json` variant.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    /// Mapping of names to data values (measurements, calculations, complex data, etc.)
    pub data: BTreeMap<String, Data>,
}

impl Shot {
    /// Add a register with a specific bit width to the shot
    ///
    /// This stores the register value as a `BitVec` and also stores metadata about its width.
    /// The width is important for proper formatting (e.g., zero-padding in binary representation).
    ///
    /// # Parameters
    ///
    /// * `name` - The register name
    /// * `value` - The register value as u32
    /// * `width` - The bit width of the register
    pub fn add_register(&mut self, name: &str, value: u32, width: usize) {
        // Create a BitVec with the specified width
        let mut bv = BitVec::<u8, Lsb0>::with_capacity(width);

        // Set bits from the value
        for i in 0..width {
            if i < 32 {
                // Only shift if within u32 bounds
                bv.push((value >> i) & 1 == 1);
            } else {
                // For bits beyond u32, push zeros
                bv.push(false);
            }
        }

        // Store the BitVec
        self.data.insert(name.to_string(), Data::BitVec(bv));

        // Store the width metadata with a special key
        self.data.insert(
            format!("_width_{name}"),
            Data::U32(u32::try_from(width).unwrap_or(u32::MAX)),
        );
    }

    /// Get a register's bit width if it was stored with `add_register`
    #[must_use]
    pub fn get_register_width(&self, name: &str) -> Option<usize> {
        self.data
            .get(&format!("_width_{name}"))
            .and_then(Data::as_u32)
            .map(|w| w as usize)
    }

    /// Create a binary string for a register, respecting its stored width
    #[must_use]
    pub fn register_to_binary_string(&self, name: &str) -> Option<String> {
        match self.data.get(name)? {
            Data::BitVec(bv) => {
                // For BitVec, the length IS the width
                let width = bv.len();
                let mut result = String::with_capacity(width);
                for i in (0..width).rev() {
                    result.push(if bv[i] { '1' } else { '0' });
                }
                Some(result)
            }
            Data::U32(v) => {
                // For U32, check if we have stored width metadata
                let width = self.get_register_width(name).unwrap_or(32);
                Some(format!("{v:0width$b}"))
            }
            _ => None,
        }
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
        let mut register_entries: Vec<(String, u32)> = match registers {
            Some(names) => names
                .iter()
                .filter_map(|&name| {
                    self.data
                        .get(name)
                        .and_then(Data::as_u32)
                        .map(|v| (name.to_string(), v))
                })
                .collect(),
            None => self
                .data
                .iter()
                .filter_map(|(name, data)| data.as_u32().map(|v| (name.clone(), v)))
                .collect(),
        };

        if sort_by_name {
            register_entries.sort_by(|(name1, _), (name2, _)| name1.cmp(name2));
        }

        register_entries
            .iter()
            .fold(String::new(), |mut acc, (_, value)| {
                use std::fmt::Write;
                write!(&mut acc, "{value:b}").unwrap();
                acc
            })
    }
}
