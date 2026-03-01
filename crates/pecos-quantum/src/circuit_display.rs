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

//! Circuit diagram rendering for [`TickCircuit`] and [`DagCircuit`].
//!
//! Produces horizontal qubit-wire diagrams with gate symbols placed at
//! tick/layer columns, suitable for terminal display. Delegates the actual
//! grid layout and character rendering to
//! [`pecos_core::circuit_diagram::CircuitDiagram`].

use pecos_core::circuit_diagram::{CellColor, CircuitDiagram, DiagramCell, DiagramOptions, GateFamily};
use pecos_core::gate_type::GateType;
use pecos_core::{Gate, QubitId};
use std::collections::BTreeSet;

// ==================== Gate symbols ====================

/// Short symbol for a gate type.
fn gate_symbol(gate_type: GateType) -> &'static str {
    match gate_type {
        GateType::H => "H",
        GateType::X => "X",
        GateType::Y => "Y",
        GateType::Z => "Z",
        GateType::SX => "SX",
        GateType::SXdg => "SXdg",
        GateType::SY => "SY",
        GateType::SYdg => "SYdg",
        GateType::SZ => "SZ",
        GateType::SZdg => "SZdg",
        GateType::T => "T",
        GateType::Tdg => "Tdg",
        GateType::RX => "Rx",
        GateType::RY => "Ry",
        GateType::RZ => "Rz",
        GateType::U => "U",
        GateType::R1XY => "R1XY",
        GateType::CX => "CX",
        GateType::CY => "CY",
        GateType::CZ => "CZ",
        GateType::CH => "CH",
        GateType::SZZ => "SZZ",
        GateType::SZZdg => "SZZdg",
        GateType::SWAP => "SWAP",
        GateType::CRZ => "CRZ",
        GateType::RXX => "RXX",
        GateType::RYY => "RYY",
        GateType::RZZ => "RZZ",
        GateType::CCX => "CCX",
        GateType::Measure => "MZ",
        GateType::MeasureLeaked => "ML",
        GateType::MeasureFree => "MF",
        GateType::Prep => "PZ",
        GateType::QAlloc => "QA",
        GateType::QFree => "QF",
        GateType::I | GateType::Idle => "I",
        GateType::MeasCrosstalkGlobalPayload | GateType::MeasCrosstalkLocalPayload => "XT",
        GateType::Custom => "?",
    }
}

/// Format an angle as a compact string in turns, e.g. `.25` or `.333333`.
fn format_angle_turns(angle: pecos_core::Angle64) -> String {
    let radians = angle.to_radians();
    let turns = radians / std::f64::consts::TAU;
    let turns = turns.rem_euclid(1.0);
    if (turns - 0.0).abs() < 1e-9 {
        return "0".to_string();
    }
    format!(".{}", format!("{turns:.6}").trim_start_matches("0.").trim_end_matches('0'))
}

/// Build the full symbol string for a gate, including angles if parameterized.
fn full_gate_symbol(gate: &Gate) -> String {
    let base = gate_symbol(gate.gate_type);
    if gate.angles.is_empty() {
        return base.to_string();
    }
    let angle_strs: Vec<String> = gate.angles.iter().copied().map(format_angle_turns).collect();
    format!("{base}({})", angle_strs.join(","))
}

// ==================== Color mapping ====================

/// Map a `GateType` to its diagram color category.
fn gate_color(gate_type: GateType) -> CellColor {
    match gate_type {
        GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
            CellColor::Measurement
        }
        GateType::Prep | GateType::QAlloc | GateType::QFree => CellColor::Preparation,
        _ if gate_type.quantum_arity() >= 2 => CellColor::MultiQubit,
        GateType::Idle | GateType::I => CellColor::None,
        _ => CellColor::SingleQubit,
    }
}

// ==================== Family mapping ====================

/// Map a `GateType` to its diagram family bracket/stroke style.
fn gate_family(gate_type: GateType) -> GateFamily {
    match gate_type {
        GateType::I | GateType::X | GateType::Y | GateType::Z => GateFamily::Pauli,
        GateType::SX
        | GateType::SXdg
        | GateType::SY
        | GateType::SYdg
        | GateType::SZ
        | GateType::SZdg => GateFamily::SLike,
        GateType::H => GateFamily::HLike,
        GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
            GateFamily::Measurement
        }
        GateType::Prep | GateType::QAlloc | GateType::QFree => GateFamily::Preparation,
        _ => GateFamily::Default,
    }
}

// ==================== Grid building ====================

/// Decompose a single `Gate` into per-row cell assignments.
fn decompose_gate(
    gate: &Gate,
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
    num_rows: usize,
) -> Vec<(usize, DiagramCell, CellColor)> {
    let arity = gate.gate_type.quantum_arity();
    let qubits = &gate.qubits;
    let mut cells = Vec::new();
    let color = gate_color(gate.gate_type);

    if arity == 1 {
        let sym = full_gate_symbol(gate);
        let family = gate_family(gate.gate_type);
        for &q in qubits {
            if let Some(&row) = qubit_to_row.get(&q) {
                cells.push((row, DiagramCell::Gate(sym.clone(), family), color));
            }
        }
    } else if arity == 2 {
        let sym = full_gate_symbol(gate);
        for pair in qubits.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            let (q_a, q_b) = (pair[0], pair[1]);
            let Some(&row_a) = qubit_to_row.get(&q_a) else {
                continue;
            };
            let Some(&row_b) = qubit_to_row.get(&q_b) else {
                continue;
            };

            let (top, bottom) = if row_a < row_b {
                (row_a, row_b)
            } else {
                (row_b, row_a)
            };

            match gate.gate_type {
                GateType::CX => {
                    cells.push((row_a, DiagramCell::Control, CellColor::ControlDot));
                    cells.push((
                        row_b,
                        DiagramCell::Gate("X".to_string(), GateFamily::Default),
                        CellColor::MultiQubit,
                    ));
                }
                GateType::CY => {
                    cells.push((row_a, DiagramCell::Control, CellColor::ControlDot));
                    cells.push((
                        row_b,
                        DiagramCell::Gate("Y".to_string(), GateFamily::Default),
                        CellColor::MultiQubit,
                    ));
                }
                GateType::CZ => {
                    cells.push((row_a, DiagramCell::Control, CellColor::ControlDot));
                    cells.push((row_b, DiagramCell::Control, CellColor::ControlDot));
                }
                GateType::CH => {
                    cells.push((row_a, DiagramCell::Control, CellColor::ControlDot));
                    cells.push((
                        row_b,
                        DiagramCell::Gate("H".to_string(), GateFamily::Default),
                        CellColor::MultiQubit,
                    ));
                }
                GateType::SWAP => {
                    cells.push((
                        row_a,
                        DiagramCell::Gate("x".to_string(), GateFamily::Default),
                        CellColor::MultiQubit,
                    ));
                    cells.push((
                        row_b,
                        DiagramCell::Gate("x".to_string(), GateFamily::Default),
                        CellColor::MultiQubit,
                    ));
                }
                _ => {
                    let family = gate_family(gate.gate_type);
                    cells.push((row_a, DiagramCell::Gate(sym.clone(), family), color));
                    cells.push((row_b, DiagramCell::Gate(sym.clone(), family), color));
                }
            }

            // Intermediate rows: crossings on qubit wires.
            for row in (top + 1)..bottom {
                if row < num_rows {
                    cells.push((row, DiagramCell::Crossing, CellColor::MultiQubit));
                }
            }
        }
    } else if arity == 3 {
        for triple in qubits.chunks(3) {
            if triple.len() < 3 {
                continue;
            }
            let (c0, c1, t) = (triple[0], triple[1], triple[2]);
            let rows: Vec<Option<usize>> = [c0, c1, t]
                .iter()
                .map(|q| qubit_to_row.get(q).copied())
                .collect();
            if rows.iter().any(Option::is_none) {
                continue;
            }
            let rows: Vec<usize> = rows.into_iter().map(|r| r.unwrap()).collect();
            let top = *rows.iter().min().unwrap();
            let bottom = *rows.iter().max().unwrap();

            cells.push((rows[0], DiagramCell::Control, CellColor::ControlDot));
            cells.push((rows[1], DiagramCell::Control, CellColor::ControlDot));
            cells.push((
                rows[2],
                DiagramCell::Gate("X".to_string(), GateFamily::Default),
                CellColor::MultiQubit,
            ));

            let gate_rows: BTreeSet<usize> = rows.iter().copied().collect();
            for row in (top + 1)..bottom {
                if !gate_rows.contains(&row) && row < num_rows {
                    cells.push((row, DiagramCell::Crossing, CellColor::MultiQubit));
                }
            }
        }
    }

    cells
}

// ==================== Diagram building ====================

/// Build a `CircuitDiagram` from gate layers.
///
/// Returns `None` when `layers` contain no qubits.
fn build_diagram(layers: &[Vec<&Gate>]) -> Option<CircuitDiagram> {
    let mut qubit_set = BTreeSet::new();
    for layer in layers {
        for gate in layer {
            for &q in &gate.qubits {
                qubit_set.insert(q);
            }
        }
    }
    let qubits: Vec<QubitId> = qubit_set.into_iter().collect();
    if qubits.is_empty() {
        return None;
    }

    let qubit_to_row: std::collections::BTreeMap<QubitId, usize> = qubits
        .iter()
        .enumerate()
        .map(|(i, &q)| (q, i))
        .collect();
    let num_rows = qubits.len();

    let labels: Vec<String> = qubits.iter().map(|q| format!("q{}", q.0)).collect();
    let mut diagram = CircuitDiagram::with_labels(labels);

    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer_idx > 0 {
            diagram.advance();
        }
        for gate in layer {
            let entries = decompose_gate(gate, &qubit_to_row, num_rows);
            for (row, cell, color) in entries {
                if row < num_rows {
                    diagram.set_cell(row, cell, color);
                }
            }
        }
    }

    Some(diagram)
}

// ==================== Public rendering entry points ====================

/// Format a circuit as a text wire diagram.
///
/// `header` - text for the first line (e.g. "`TickCircuit`: 3 qubits, 4 ticks").
/// `layers` - each element is a slice of gates that execute in parallel.
/// `options` - rendering options (symbol set, color).
pub(crate) fn format_circuit(
    header: &str,
    layers: &[Vec<&Gate>],
    options: &DiagramOptions,
) -> String {
    match build_diagram(layers) {
        Some(diagram) => diagram.render(header, options),
        None => format!("{header}\n"),
    }
}

/// Format a circuit as an SVG wire diagram.
pub(crate) fn format_circuit_svg(header: &str, layers: &[Vec<&Gate>]) -> String {
    match build_diagram(layers) {
        Some(diagram) => diagram.render_svg(header),
        None => format!("<svg xmlns=\"http://www.w3.org/2000/svg\"><text x=\"10\" y=\"20\" font-family=\"monospace\" font-size=\"14\">{header}</text></svg>"),
    }
}

/// Format a circuit as a `TikZ` `tikzpicture`.
pub(crate) fn format_circuit_tikz(header: &str, layers: &[Vec<&Gate>]) -> String {
    if let Some(diagram) = build_diagram(layers) {
        diagram.render_tikz(header)
    } else {
        let mut out = String::new();
        if !header.is_empty() {
            use std::fmt::Write;
            writeln!(out, "% {header}").unwrap();
        }
        out.push_str("\\begin{tikzpicture}\n\\end{tikzpicture}\n");
        out
    }
}

/// Format a circuit as a Graphviz DOT digraph.
pub(crate) fn format_circuit_dot(header: &str, layers: &[Vec<&Gate>]) -> String {
    if let Some(diagram) = build_diagram(layers) {
        diagram.render_dot(header)
    } else {
        let mut out = String::from("digraph circuit {\n  rankdir=LR;\n");
        if !header.is_empty() {
            use std::fmt::Write;
            writeln!(out, "  label=\"{header}\";").unwrap();
        }
        out.push_str("}\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::Angle64;

    fn render_tick(build: impl FnOnce(&mut crate::TickCircuit)) -> String {
        let mut tc = crate::TickCircuit::new();
        build(&mut tc);
        tc.to_ascii()
    }

    fn render_tick_color(build: impl FnOnce(&mut crate::TickCircuit)) -> String {
        let mut tc = crate::TickCircuit::new();
        build(&mut tc);
        tc.to_color_ascii()
    }

    #[test]
    fn single_qubit_gates_on_correct_wires() {
        let out = render_tick(|tc| {
            tc.tick().h(&[0]);
            tc.tick().x(&[1]);
        });
        assert!(out.contains("q0:"));
        assert!(out.contains("q1:"));
        let q0_line = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q0_line.contains("<H>"));
        assert!(!q0_line.contains("(X)"));
        assert!(q1_line.contains("(X)"));
        assert!(!q1_line.contains("<H>"));
    }

    #[test]
    fn cx_shows_control_target_connector() {
        let out = render_tick(|tc| {
            tc.tick().h(&[0, 1, 2]);
            tc.tick().cx(&[(0, 2)]);
        });
        assert!(out.contains('.'));
        assert!(out.contains("[X]")); // CX target uses Default brackets
        assert!(out.contains('|'));
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q1_line.contains('+'));
    }

    #[test]
    fn multi_tick_alignment() {
        let out = render_tick(|tc| {
            tc.tick().h(&[0]);
            tc.tick().cx(&[(0, 1)]);
            tc.tick().h(&[1]);
        });
        let qubit_lines: Vec<&str> = out.lines().filter(|l| l.starts_with('q')).collect();
        assert!(qubit_lines.len() >= 2);
        let len0 = qubit_lines[0].len();
        for line in &qubit_lines {
            assert_eq!(line.len(), len0, "Lines should have equal length");
        }
    }

    #[test]
    fn parameterized_gate_includes_angle() {
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        });
        assert!(out.contains("Rz("));
        assert!(out.contains(".25"));
    }

    #[test]
    fn empty_circuit_shows_header_only() {
        let tc = crate::TickCircuit::new();
        let out = tc.to_ascii();
        assert!(out.contains("TickCircuit:"));
        assert!(!out.contains("q0:"));
    }

    #[test]
    fn color_version_contains_ansi_plain_does_not() {
        let plain = render_tick(|tc| {
            tc.tick().h(&[0]);
        });
        let colored = render_tick_color(|tc| {
            tc.tick().h(&[0]);
        });
        assert!(!plain.contains("\x1b["));
        assert!(colored.contains("\x1b["));
    }

    #[test]
    fn non_contiguous_qubit_ids() {
        let out = render_tick(|tc| {
            tc.tick().h(&[5]);
            tc.tick().h(&[10]);
        });
        assert!(out.contains("q5:"));
        assert!(out.contains("q10:"));
    }

    #[test]
    fn dag_and_tick_produce_identical_output() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().h(&[1]);

        let mut dag = crate::DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);
        dag.h(1);

        let tick_out = tc.to_ascii();
        let dag_out = dag.to_ascii();

        let tick_lines: Vec<&str> = tick_out.lines().filter(|l| l.starts_with('q')).collect();
        let dag_lines: Vec<&str> = dag_out.lines().filter(|l| l.starts_with('q')).collect();
        assert_eq!(tick_lines, dag_lines);
    }

    #[test]
    fn cz_shows_two_controls() {
        let out = render_tick(|tc| {
            tc.tick().cz(&[(0, 1)]);
        });
        let q0_line = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q0_line.contains('.'));
        assert!(q1_line.contains('.'));
    }

    #[test]
    fn swap_shows_x_on_both() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]).h(&[1]);
        tc.tick();
        let swap_gate = Gate::simple(
            GateType::SWAP,
            smallvec::smallvec![QubitId::from(0usize), QubitId::from(1usize)],
        );
        tc.get_tick_mut(1).unwrap().add_gate(swap_gate);
        let out = tc.to_ascii();
        let q0_line = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q0_line.contains("[x]"));
        assert!(q1_line.contains("[x]"));
    }

    #[test]
    fn measurement_and_prep() {
        let out = render_tick(|tc| {
            tc.tick().pz(&[0]);
            tc.tick().h(&[0]);
            tc.tick().mz(&[0]);
        });
        assert!(out.contains("(PZ|"));
        assert!(out.contains("<H>"));
        assert!(out.contains("|MZ)"));
    }

    #[test]
    fn batched_single_qubit_gates() {
        let out = render_tick(|tc| {
            tc.tick().h(&[0, 1, 2]);
        });
        let q0_line = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        let q2_line = out.lines().find(|l| l.starts_with("q2:")).unwrap();
        assert!(q0_line.contains("<H>"));
        assert!(q1_line.contains("<H>"));
        assert!(q2_line.contains("<H>"));
    }

    #[test]
    fn unicode_uses_box_drawing() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        let out = tc.to_unicode();
        assert!(out.contains('\u{2500}')); // ─
        assert!(!out.contains("---")); // no plain dashes as wire
    }

    #[test]
    fn unicode_control_dot() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().cx(&[(0, 1)]);
        let out = tc.to_unicode();
        assert!(out.contains('\u{25CF}')); // ●
    }

    #[test]
    fn to_ascii_color_deprecated_alias() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        #[allow(deprecated)]
        let a = tc.to_ascii_color();
        let b = tc.to_color_ascii();
        assert_eq!(a, b);
    }

    // ====================== SVG integration ======================

    #[test]
    fn tick_svg_contains_gate_elements() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        let svg = tc.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains(">H</text>"));
        assert!(svg.contains("<circle")); // control dot
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn dag_svg_matches_tick_structure() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);

        let mut dag = crate::DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);

        let tick_svg = tc.to_svg();
        let dag_svg = dag.to_svg();
        // Both should contain the same gate elements.
        assert!(tick_svg.contains(">H</text>"));
        assert!(dag_svg.contains(">H</text>"));
        assert!(tick_svg.contains(">X</text>"));
        assert!(dag_svg.contains(">X</text>"));
    }

    // ====================== TikZ integration ======================

    #[test]
    fn tick_tikz_contains_commands() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        let tikz = tc.to_tikz();
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("\\end{tikzpicture}"));
        assert!(tikz.contains("{H}"));
        assert!(tikz.contains("\\node[ctrl"));
    }

    #[test]
    fn dag_tikz_contains_commands() {
        let mut dag = crate::DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);
        let tikz = dag.to_tikz();
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("{H}"));
    }

    // ====================== DOT integration ======================

    #[test]
    fn tick_dot_contains_graph() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        let dot = tc.to_dot();
        assert!(dot.contains("digraph circuit"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("label=\"H\""));
        assert!(dot.contains("shape=point, width=0.12")); // control
    }

    #[test]
    fn dag_dot_contains_graph() {
        let mut dag = crate::DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);
        let dot = dag.to_dot();
        assert!(dot.contains("digraph circuit"));
        assert!(dot.contains("label=\"H\""));
    }

    // ====================== Gate family integration ======================

    #[test]
    fn family_brackets_in_tick_output() {
        let out = render_tick(|tc| {
            tc.tick().pz(&[0]);
            tc.tick().h(&[0]);
            tc.tick().sx(&[0]);
            tc.tick().x(&[0]);
            tc.tick().mz(&[0]);
        });
        assert!(out.contains("(PZ|")); // Preparation
        assert!(out.contains("<H>")); // HLike
        assert!(out.contains("[SX]")); // SLike
        assert!(out.contains("(X)")); // Pauli
        assert!(out.contains("|MZ)")); // Measurement
    }

    #[test]
    fn family_brackets_in_dag_output() {
        let mut dag = crate::DagCircuit::new();
        dag.pz(0);
        dag.h(0);
        dag.sx(0);
        dag.x(0);
        dag.mz(0);
        let out = dag.to_ascii();
        assert!(out.contains("(PZ|"));
        assert!(out.contains("<H>"));
        assert!(out.contains("[SX]"));
        assert!(out.contains("(X)"));
        assert!(out.contains("|MZ)"));
    }

    #[test]
    fn svg_hlike_has_dotted_stroke() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        let svg = tc.to_svg();
        assert!(svg.contains("stroke-dasharray=\"2,2\""));
    }

    #[test]
    fn svg_slike_has_dashed_stroke() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().sz(&[0]);
        let svg = tc.to_svg();
        assert!(svg.contains("stroke-dasharray=\"4,3\""));
    }
}
