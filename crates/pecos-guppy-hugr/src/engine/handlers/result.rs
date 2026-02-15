// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Result capture operations (`tket.result`).
//!
//! This module handles operations that capture output values for the caller:
//! - `result_bool`, `result_int`, `result_uint`, `result_f64`: Scalar results
//! - `result_array_bool`, `result_array_int`, `result_array_uint`, `result_array_f64`: Array results
//!
//! Captured results are stored in [`GuppyHugrEngine::captured_results`](super::super::GuppyHugrEngine)
//! and can be retrieved via [`GuppyHugrEngine::get_captured_results`](super::super::GuppyHugrEngine::get_captured_results).

use log::debug;
use tket::hugr::{Hugr, HugrView, Node, NodeIndex};

use crate::engine::GuppyHugrEngine;
use crate::engine::types::{CapturedResult, ClassicalValue, ResultValue};

impl GuppyHugrEngine {
    /// Handle tket.result operations for capturing output values.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn handle_result_op(&mut self, hugr: &Hugr, node: Node, op_name: &str) -> bool {
        debug!("Processing tket.result operation: {op_name} at {node:?}");

        // Get the label from the first input port (typically the operation has a label parameter)
        // For now, use the operation name as the label; proper label extraction requires parsing HUGR params
        let label = self.extract_result_label(hugr, node, op_name);

        match op_name {
            "result_bool" => {
                let input_value = self.get_input_value(hugr, node, 0);
                if let Some(ref value) = input_value
                    && let Some(b) = value.as_bool()
                {
                    debug!("Captured result_bool: label={label}, value={b}");
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::Bool(b),
                    });
                    true
                } else {
                    // Input not ready - defer processing
                    debug!("result_bool at {node:?}: deferring - input not ready");
                    false
                }
            }
            "result_int" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(i) = value.as_int()
                {
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::Int(i),
                    });
                    debug!("Captured result_int: {i}");
                }
                true
            }
            "result_uint" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(u) = value.as_uint()
                {
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::UInt(u),
                    });
                    debug!("Captured result_uint: {u}");
                }
                true
            }
            "result_f64" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(f) = value.as_float()
                {
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::Float(f),
                    });
                    debug!("Captured result_f64: {f}");
                }
                true
            }
            "result_array_bool" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(arr) = value.as_array()
                {
                    let bools: Vec<bool> = arr.iter().filter_map(ClassicalValue::as_bool).collect();
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::ArrayBool(bools),
                    });
                }
                true
            }
            "result_array_int" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(arr) = value.as_array()
                {
                    let ints: Vec<i64> = arr.iter().filter_map(ClassicalValue::as_int).collect();
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::ArrayInt(ints),
                    });
                }
                true
            }
            "result_array_uint" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(arr) = value.as_array()
                {
                    let uints: Vec<u64> = arr.iter().filter_map(ClassicalValue::as_uint).collect();
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::ArrayUInt(uints),
                    });
                }
                true
            }
            "result_array_f64" => {
                if let Some(value) = self.get_input_value(hugr, node, 0)
                    && let Some(arr) = value.as_array()
                {
                    let floats: Vec<f64> =
                        arr.iter().filter_map(ClassicalValue::as_float).collect();
                    self.captured_results.push(CapturedResult {
                        label,
                        value: ResultValue::ArrayFloat(floats),
                    });
                }
                true
            }
            _ => {
                debug!("Unknown tket.result operation: {op_name}");
                false
            }
        }
    }

    /// Extract result label from operation parameters.
    #[allow(clippy::unused_self)] // Consistent with other handler methods; may use self in future
    pub(crate) fn extract_result_label(&self, hugr: &Hugr, node: Node, op_name: &str) -> String {
        // Try to extract label from the ExtensionOp's debug representation
        // The debug format typically includes the label as a string parameter
        let op = hugr.get_optype(node);
        if let Some(ext_op) = op.as_extension_op() {
            let debug_str = format!("{ext_op:?}");
            // Look for quoted string patterns that might be labels
            // Common patterns: "label", label="value", or ("label", ...)
            if let Some(label) = Self::extract_string_from_debug(&debug_str)
                && !label.is_empty()
                && label != op_name
            {
                return label;
            }
        }
        // Fallback: use node ID as label
        format!("{op_name}_{}", node.index())
    }

    /// Try to extract a string label from a debug representation.
    pub(crate) fn extract_string_from_debug(debug_str: &str) -> Option<String> {
        // Look for pattern: args: [String("label")]
        // This is the format used by ExtensionOp's debug output
        if let Some(args_idx) = debug_str.find("args: [String(\"") {
            let start = args_idx + "args: [String(\"".len();
            if let Some(end) = debug_str[start..].find("\")]") {
                let label = &debug_str[start..start + end];
                if !label.is_empty() {
                    return Some(label.to_string());
                }
            }
        }

        // Fallback: look for quoted strings, but skip common non-label values
        let mut in_quotes = false;
        let mut quote_char = '"';
        let mut label = String::new();

        for c in debug_str.chars() {
            if !in_quotes && (c == '"' || c == '\'') {
                in_quotes = true;
                quote_char = c;
                label.clear();
            } else if in_quotes && c == quote_char {
                // Found end of quoted string
                if !label.is_empty()
                    && !label.contains("::")
                    && !label.starts_with("tket")
                    && !label.contains("Op")
                    && !label.contains("result")
                    && !label.contains("Report")
                {
                    return Some(label);
                }
                in_quotes = false;
                label.clear();
            } else if in_quotes {
                label.push(c);
            }
        }

        None
    }
}
