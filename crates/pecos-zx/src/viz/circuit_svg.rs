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

//! SVG rendering of quantum circuit wire diagrams.

use pecos_core::gate_type::GateType;

use super::circuit_layout::CircuitLayout;

/// Options for SVG circuit rendering.
#[derive(Debug, Clone)]
pub struct CircuitSvgOptions {
    /// Vertical spacing between qubit wires (pixels).
    pub wire_spacing: f64,
    /// Horizontal width per time step (pixels).
    pub step_width: f64,
    /// Width of gate boxes (pixels).
    pub gate_width: f64,
    /// Height of gate boxes (pixels).
    pub gate_height: f64,
    /// Font size for gate labels (pixels).
    pub font_size: f64,
    /// Whether to show qubit labels (q0, q1, ...).
    pub show_qubit_labels: bool,
    /// Whether to show classical wires.
    pub show_classical_wires: bool,
    /// Left margin for qubit labels.
    pub left_margin: f64,
    /// Top margin.
    pub top_margin: f64,
}

impl Default for CircuitSvgOptions {
    fn default() -> Self {
        Self {
            wire_spacing: 50.0,
            step_width: 60.0,
            gate_width: 36.0,
            gate_height: 30.0,
            font_size: 14.0,
            show_qubit_labels: true,
            show_classical_wires: true,
            left_margin: 50.0,
            top_margin: 20.0,
        }
    }
}

/// Render a circuit layout as an SVG string.
#[must_use]
pub fn render_circuit_svg(layout: &CircuitLayout, options: &CircuitSvgOptions) -> String {
    let mut svg = String::new();

    let total_width = options.left_margin + (layout.num_steps as f64) * options.step_width + 40.0;
    let quantum_height = (layout.num_qubits as f64) * options.wire_spacing;
    let classical_height = if options.show_classical_wires && layout.num_cbits > 0 {
        (layout.num_cbits as f64) * options.wire_spacing * 0.6
    } else {
        0.0
    };
    let total_height = options.top_margin + quantum_height + classical_height + 40.0;

    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">"##
    ));
    svg.push('\n');

    // Background
    svg.push_str(&format!(
        r##"  <rect width="{total_width}" height="{total_height}" fill="white"/>"##
    ));
    svg.push('\n');

    // Draw qubit wires
    for q in 0..layout.num_qubits {
        let y = wire_y(q, options);
        let x1 = options.left_margin;
        let x2 = options.left_margin + (layout.num_steps as f64) * options.step_width;

        svg.push_str(&format!(
            r##"  <line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#888" stroke-width="1"/>"##
        ));
        svg.push('\n');

        // Qubit labels
        if options.show_qubit_labels {
            svg.push_str(&format!(
                r##"  <text x="{}" y="{}" font-size="{}" fill="#444" text-anchor="end" dominant-baseline="middle">q{q}</text>"##,
                options.left_margin - 8.0,
                y,
                options.font_size
            ));
            svg.push('\n');
        }
    }

    // Draw classical wires
    if options.show_classical_wires && layout.num_cbits > 0 {
        let cbit_y_start = wire_y(layout.num_qubits, options);
        for c in 0..layout.num_cbits {
            let y = cbit_y_start + (c as f64) * options.wire_spacing * 0.6;
            let x1 = options.left_margin;
            let x2 = options.left_margin + (layout.num_steps as f64) * options.step_width;

            // Double line for classical wire
            svg.push_str(&format!(
                r##"  <line x1="{x1}" y1="{}" x2="{x2}" y2="{}" stroke="#666" stroke-width="1"/>"##,
                y - 1.5,
                y - 1.5
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r##"  <line x1="{x1}" y1="{}" x2="{x2}" y2="{}" stroke="#666" stroke-width="1"/>"##,
                y + 1.5,
                y + 1.5
            ));
            svg.push('\n');

            if options.show_qubit_labels {
                svg.push_str(&format!(
                    r##"  <text x="{}" y="{y}" font-size="{}" fill="#444" text-anchor="end" dominant-baseline="middle">c{c}</text>"##,
                    options.left_margin - 8.0,
                    options.font_size - 2.0
                ));
                svg.push('\n');
            }
        }
    }

    // Draw gates
    for q in 0..layout.num_qubits {
        for step in 0..layout.num_steps {
            if let Some(slot) = layout.get(q, step) {
                let cx = step_x(step, options);
                let cy = wire_y(q, options);

                match slot.gate_type {
                    GateType::MZ | GateType::MeasureFree => {
                        render_measurement(&mut svg, cx, cy, options);
                    }
                    GateType::PZ => {
                        render_gate_box(&mut svg, cx, cy, "|0>", options);
                    }
                    GateType::CX => {
                        render_cx(&mut svg, cx, &slot.qubits, options);
                    }
                    GateType::CZ => {
                        render_cz(&mut svg, cx, &slot.qubits, options);
                    }
                    GateType::SWAP => {
                        render_swap(&mut svg, cx, &slot.qubits, options);
                    }
                    _ => {
                        render_gate_box(&mut svg, cx, cy, &slot.label, options);
                    }
                }

                // Draw multi-qubit connecting line
                if slot.qubits.len() > 1
                    && !matches!(slot.gate_type, GateType::CX | GateType::CZ | GateType::SWAP)
                {
                    let y_min = wire_y(*slot.qubits.first().unwrap(), options);
                    let y_max = wire_y(*slot.qubits.last().unwrap(), options);
                    svg.push_str(&format!(
                        r##"  <line x1="{cx}" y1="{y_min}" x2="{cx}" y2="{y_max}" stroke="#333" stroke-width="2"/>"##
                    ));
                    svg.push('\n');
                }

                // Draw condition line to classical wire
                if slot.has_condition
                    && let Some(cbit_idx) = slot.cbit
                {
                    let cbit_y = wire_y(layout.num_qubits, options)
                        + (cbit_idx as f64) * options.wire_spacing * 0.6;
                    svg.push_str(&format!(
                            r##"  <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cbit_y}" stroke="#666" stroke-width="1" stroke-dasharray="4,2"/>"##
                        ));
                    svg.push('\n');
                    // Condition dot on classical wire
                    svg.push_str(&format!(
                        r##"  <circle cx="{cx}" cy="{cbit_y}" r="4" fill="#666"/>"##
                    ));
                    svg.push('\n');
                }

                // Draw measurement arrow to classical wire
                if let Some(cbit_idx) = slot.meas_cbit {
                    let meas_bottom = cy + options.gate_height / 2.0;
                    let cbit_y = wire_y(layout.num_qubits, options)
                        + (cbit_idx as f64) * options.wire_spacing * 0.6;
                    svg.push_str(&format!(
                        r##"  <line x1="{cx}" y1="{meas_bottom}" x2="{cx}" y2="{cbit_y}" stroke="#666" stroke-width="1" stroke-dasharray="4,2"/>"##
                    ));
                    svg.push('\n');
                }
            }
        }
    }

    svg.push_str("</svg>\n");
    svg
}

fn wire_y(qubit: usize, options: &CircuitSvgOptions) -> f64 {
    options.top_margin + (qubit as f64) * options.wire_spacing + options.wire_spacing / 2.0
}

fn step_x(step: usize, options: &CircuitSvgOptions) -> f64 {
    options.left_margin + (step as f64) * options.step_width + options.step_width / 2.0
}

fn render_gate_box(svg: &mut String, cx: f64, cy: f64, label: &str, options: &CircuitSvgOptions) {
    let x = cx - options.gate_width / 2.0;
    let y = cy - options.gate_height / 2.0;

    svg.push_str(&format!(
        r##"  <rect x="{x}" y="{y}" width="{}" height="{}" fill="white" stroke="#333" stroke-width="1.5" rx="3"/>"##,
        options.gate_width, options.gate_height
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r##"  <text x="{cx}" y="{cy}" font-size="{}" fill="#333" text-anchor="middle" dominant-baseline="middle">{label}</text>"##,
        options.font_size
    ));
    svg.push('\n');
}

fn render_measurement(svg: &mut String, cx: f64, cy: f64, options: &CircuitSvgOptions) {
    let x = cx - options.gate_width / 2.0;
    let y = cy - options.gate_height / 2.0;
    let w = options.gate_width;
    let h = options.gate_height;

    // Box
    svg.push_str(&format!(
        r##"  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="#333" stroke-width="1.5" rx="3"/>"##
    ));
    svg.push('\n');

    // Meter arc
    let arc_y = cy + 2.0;
    let arc_r = w * 0.3;
    svg.push_str(&format!(
        r##"  <path d="M {},{arc_y} A {arc_r},{arc_r} 0 0 1 {},{arc_y}" fill="none" stroke="#333" stroke-width="1.5"/>"##,
        cx - arc_r,
        cx + arc_r,
    ));
    svg.push('\n');

    // Arrow
    svg.push_str(&format!(
        r##"  <line x1="{cx}" y1="{arc_y}" x2="{}" y2="{}" stroke="#333" stroke-width="1.5"/>"##,
        cx + arc_r * 0.7,
        cy - h * 0.25,
    ));
    svg.push('\n');
}

fn render_cx(svg: &mut String, cx: f64, qubits: &[usize], options: &CircuitSvgOptions) {
    if qubits.len() < 2 {
        return;
    }
    let control_y = wire_y(qubits[0], options);
    let target_y = wire_y(qubits[1], options);

    // Vertical line
    svg.push_str(&format!(
        r##"  <line x1="{cx}" y1="{control_y}" x2="{cx}" y2="{target_y}" stroke="#333" stroke-width="1.5"/>"##
    ));
    svg.push('\n');

    // Control dot
    svg.push_str(&format!(
        r##"  <circle cx="{cx}" cy="{control_y}" r="4" fill="#333"/>"##
    ));
    svg.push('\n');

    // Target circle (XOR symbol)
    let r = 10.0;
    svg.push_str(&format!(
        r##"  <circle cx="{cx}" cy="{target_y}" r="{r}" fill="white" stroke="#333" stroke-width="1.5"/>"##
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r##"  <line x1="{cx}" y1="{}" x2="{cx}" y2="{}" stroke="#333" stroke-width="1.5"/>"##,
        target_y - r,
        target_y + r,
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r##"  <line x1="{}" y1="{target_y}" x2="{}" y2="{target_y}" stroke="#333" stroke-width="1.5"/>"##,
        cx - r,
        cx + r,
    ));
    svg.push('\n');
}

fn render_cz(svg: &mut String, cx: f64, qubits: &[usize], options: &CircuitSvgOptions) {
    if qubits.len() < 2 {
        return;
    }
    let y1 = wire_y(qubits[0], options);
    let y2 = wire_y(qubits[1], options);

    // Vertical line
    svg.push_str(&format!(
        r##"  <line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#333" stroke-width="1.5"/>"##
    ));
    svg.push('\n');

    // Both dots (CZ is symmetric)
    svg.push_str(&format!(
        r##"  <circle cx="{cx}" cy="{y1}" r="4" fill="#333"/>"##
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r##"  <circle cx="{cx}" cy="{y2}" r="4" fill="#333"/>"##
    ));
    svg.push('\n');
}

fn render_swap(svg: &mut String, cx: f64, qubits: &[usize], options: &CircuitSvgOptions) {
    if qubits.len() < 2 {
        return;
    }
    let y1 = wire_y(qubits[0], options);
    let y2 = wire_y(qubits[1], options);
    let s = 6.0; // cross size

    // Vertical line
    svg.push_str(&format!(
        r##"  <line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#333" stroke-width="1.5"/>"##
    ));
    svg.push('\n');

    // X crosses at each qubit
    for &y in &[y1, y2] {
        svg.push_str(&format!(
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" stroke-width="1.5"/>"##,
            cx - s,
            y - s,
            cx + s,
            y + s,
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" stroke-width="1.5"/>"##,
            cx - s,
            y + s,
            cx + s,
            y - s,
        ));
        svg.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_svg_output_valid() {
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let svg = render_circuit_svg(&layout, &CircuitSvgOptions::default());

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("q0"));
        assert!(svg.contains("q1"));
    }

    #[test]
    fn test_svg_contains_gates() {
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.x(1);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let svg = render_circuit_svg(&layout, &CircuitSvgOptions::default());

        assert!(svg.contains(">H<"));
        assert!(svg.contains(">X<"));
    }

    #[test]
    fn test_svg_measurement() {
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.mz(0);

        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let options = CircuitSvgOptions::default();
        let svg = render_circuit_svg(&layout, &options);

        // Measurement renders a meter symbol with an arc path, not text
        assert!(svg.contains("<path d=\"M"));
    }

    #[test]
    fn test_svg_empty_circuit() {
        let dag = DagCircuit::new();
        let layout = super::super::circuit_layout::layout_from_dag(&dag);
        let svg = render_circuit_svg(&layout, &CircuitSvgOptions::default());

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
