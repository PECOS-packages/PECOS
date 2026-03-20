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

//! ASCII rendering of quantum circuit wire diagrams.

use pecos_core::gate_type::GateType;

use super::circuit_layout::CircuitLayout;

/// Options for ASCII circuit rendering.
#[derive(Debug, Clone)]
pub struct CircuitAsciiOptions {
    /// Whether to show qubit labels (q0, q1, ...).
    pub show_qubit_labels: bool,
}

impl Default for CircuitAsciiOptions {
    fn default() -> Self {
        Self {
            show_qubit_labels: true,
        }
    }
}

/// Render a circuit layout as an ASCII wire diagram.
///
/// Produces output like:
/// ```text
/// q0: -[H]-*--[M]-----
///           |       ||
/// q1: ------X--[X]--
///               ^
/// c0: ==========+
/// ```
#[must_use]
pub fn render_circuit_ascii(layout: &CircuitLayout, options: &CircuitAsciiOptions) -> String {
    if layout.num_qubits == 0 {
        return String::new();
    }

    // Compute the width of each step column. Each gate label gets padded to a
    // uniform cell width per column so wires align.
    let col_widths: Vec<usize> = (0..layout.num_steps)
        .map(|step| {
            let max_label = (0..layout.num_qubits)
                .filter_map(|q| layout.get(q, step))
                .map(|slot| cell_label(slot.gate_type, &slot.label).len())
                .max()
                .unwrap_or(1);
            // Minimum 3 chars for wire segments, plus brackets
            max_label.max(3)
        })
        .collect();

    // Qubit label width for alignment
    let label_width = if options.show_qubit_labels {
        let max_q = layout.num_qubits.saturating_sub(1);
        format!("q{max_q}").len() + 2 // "q0: "
    } else {
        0
    };

    let mut lines: Vec<String> = Vec::new();

    // Render each qubit wire (with a spacer line between for multi-qubit connections)
    for q in 0..layout.num_qubits {
        let mut wire_line = String::new();
        let mut conn_line = String::new(); // line below for vertical connections

        // Qubit label
        if options.show_qubit_labels {
            wire_line.push_str(&format!(
                "{:>width$}: ",
                format!("q{q}"),
                width = label_width - 2
            ));
            conn_line.push_str(&" ".repeat(label_width));
        }

        let mut has_connections = false;

        for (step, &col_w) in col_widths.iter().enumerate() {
            if let Some(slot) = layout.get(q, step) {
                let cell = cell_label(slot.gate_type, &slot.label);
                let padded = format!("{cell:^width$}", width = col_w);
                wire_line.push_str(&padded);

                // Check if this gate connects down to another qubit
                let connects_down = slot.qubits.len() > 1
                    && slot.qubits.first() == Some(&q)
                    && slot.qubits.iter().any(|&qb| qb > q);

                if connects_down {
                    let mid = col_w / 2;
                    let mut conn_cell: Vec<char> = " ".repeat(col_w).chars().collect();
                    if mid < conn_cell.len() {
                        conn_cell[mid] = '|';
                    }
                    conn_line.push_str(&conn_cell.into_iter().collect::<String>());
                    has_connections = true;
                } else if slot.has_condition || slot.meas_cbit.is_some() {
                    // Vertical line for condition/measurement
                    let mid = col_w / 2;
                    let mut conn_cell: Vec<char> = " ".repeat(col_w).chars().collect();
                    if mid < conn_cell.len() {
                        conn_cell[mid] = '|';
                    }
                    conn_line.push_str(&conn_cell.into_iter().collect::<String>());
                    has_connections = true;
                } else {
                    conn_line.push_str(&" ".repeat(col_w));
                }
            } else {
                // Check if a multi-qubit gate from above passes through this qubit
                let pass_through = is_pass_through(layout, q, step);
                if pass_through {
                    let mid = col_w / 2;
                    let mut cell: Vec<char> = "-".repeat(col_w).chars().collect();
                    if mid < cell.len() {
                        cell[mid] = '|';
                    }
                    wire_line.push_str(&cell.into_iter().collect::<String>());
                    // Continue the vertical line below
                    let mut conn_cell: Vec<char> = " ".repeat(col_w).chars().collect();
                    if mid < conn_cell.len() {
                        conn_cell[mid] = '|';
                    }
                    conn_line.push_str(&conn_cell.into_iter().collect::<String>());
                    has_connections = true;
                } else {
                    // Check if this is a CX/CZ target position
                    let is_target = is_multi_qubit_target(layout, q, step);
                    if let Some(target_label) = is_target {
                        let padded = format!("{target_label:^width$}", width = col_w);
                        wire_line.push_str(&padded);
                    } else {
                        wire_line.push_str(&"-".repeat(col_w));
                    }
                    conn_line.push_str(&" ".repeat(col_w));
                }
            }

            // Wire between columns
            if step < layout.num_steps - 1 {
                wire_line.push('-');
                conn_line.push(' ');
            }
        }

        lines.push(wire_line);
        if has_connections && q < layout.num_qubits - 1 {
            lines.push(conn_line);
        }
    }

    // Render classical wires
    if layout.num_cbits > 0 {
        for c in 0..layout.num_cbits {
            let mut cbit_line = String::new();

            if options.show_qubit_labels {
                cbit_line.push_str(&format!(
                    "{:>width$}: ",
                    format!("c{c}"),
                    width = label_width - 2
                ));
            }

            for (step, &col_w) in col_widths.iter().enumerate() {
                cbit_line.push_str(&"=".repeat(col_w));
                if step < layout.num_steps - 1 {
                    cbit_line.push('=');
                }
            }

            lines.push(cbit_line);
        }
    }

    lines.join("\n")
}

/// Get the display label for a gate in ASCII.
fn cell_label(gate_type: GateType, label: &str) -> String {
    match gate_type {
        GateType::CX => "*".to_string(),   // control dot
        GateType::CZ => "*".to_string(),   // control dot (symmetric)
        GateType::SWAP => "X".to_string(), // swap cross
        GateType::MZ | GateType::MeasureFree => "[M]".to_string(),
        GateType::PZ => "[0]".to_string(),
        _ => format!("[{label}]"),
    }
}

/// Check if a multi-qubit gate passes through this qubit position.
fn is_pass_through(layout: &CircuitLayout, qubit: usize, step: usize) -> bool {
    // Look at all qubits above to see if any gate spans through this qubit
    for q in 0..qubit {
        if let Some(slot) = layout.get(q, step)
            && slot.qubits.len() > 1
        {
            let min_q = *slot.qubits.first().unwrap_or(&0);
            let max_q = *slot.qubits.last().unwrap_or(&0);
            if qubit > min_q && qubit < max_q && !slot.qubits.contains(&qubit) {
                return true;
            }
        }
    }
    false
}

/// Check if this qubit is the target of a multi-qubit gate placed on another qubit.
fn is_multi_qubit_target(layout: &CircuitLayout, qubit: usize, step: usize) -> Option<String> {
    for q in 0..layout.num_qubits {
        if q == qubit {
            continue;
        }
        if let Some(slot) = layout.get(q, step)
            && slot.qubits.contains(&qubit)
        {
            return match slot.gate_type {
                GateType::CX => Some("X".to_string()),   // target of CNOT
                GateType::CZ => Some("*".to_string()),   // CZ is symmetric
                GateType::SWAP => Some("X".to_string()), // swap cross
                _ => Some(format!("[{}]", slot.label)),  // other multi-qubit
            };
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_ascii_single_gate() {
        let mut dag = DagCircuit::new();
        dag.h(0);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let ascii = render_circuit_ascii(&layout, &CircuitAsciiOptions::default());

        assert!(ascii.contains("[H]"));
        assert!(ascii.contains("q0"));
    }

    #[test]
    fn test_ascii_two_qubit_gate() {
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let ascii = render_circuit_ascii(&layout, &CircuitAsciiOptions::default());

        assert!(ascii.contains("[H]"));
        assert!(ascii.contains("*")); // CX control
        assert!(ascii.contains("X")); // CX target
        assert!(ascii.contains("q0"));
        assert!(ascii.contains("q1"));
    }

    #[test]
    fn test_ascii_parallel_gates() {
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.x(1);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let ascii = render_circuit_ascii(&layout, &CircuitAsciiOptions::default());

        assert!(ascii.contains("[H]"));
        assert!(ascii.contains("[X]"));
    }

    #[test]
    fn test_ascii_empty_circuit() {
        let dag = DagCircuit::new();
        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let ascii = render_circuit_ascii(&layout, &CircuitAsciiOptions::default());
        assert!(ascii.is_empty());
    }

    #[test]
    fn test_ascii_with_measurement() {
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.mz(0);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let ascii = render_circuit_ascii(&layout, &CircuitAsciiOptions::default());

        assert!(ascii.contains("[H]"));
        assert!(ascii.contains("[M]"));
    }

    #[test]
    fn test_ascii_no_labels() {
        let mut dag = DagCircuit::new();
        dag.h(0);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let options = CircuitAsciiOptions {
            show_qubit_labels: false,
        };
        let ascii = render_circuit_ascii(&layout, &options);

        assert!(ascii.contains("[H]"));
        assert!(!ascii.contains("q0"));
    }
}
