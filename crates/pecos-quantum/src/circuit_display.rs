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

use pecos_core::circuit_diagram::{AngleUnit, CellColor, CircuitDiagram, DiagramCell, GateFamily};
use pecos_core::gate_type::GateType;
use pecos_core::{Gate, QubitId};
use std::collections::BTreeSet;

// ==================== Gate symbols ====================

/// Short symbol for a gate type.
fn gate_symbol(gate_type: GateType) -> &'static str {
    match gate_type {
        GateType::H => "H",
        GateType::F => "F",
        GateType::Fdg => "Fdg",
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
        GateType::RX => "RX",
        GateType::RY => "RY",
        GateType::RZ => "RZ",
        GateType::U => "U",
        GateType::R1XY => "R1XY",
        GateType::CX => "CX",
        GateType::CY => "CY",
        GateType::CZ => "CZ",
        GateType::CH => "CH",
        GateType::SXX => "SXX",
        GateType::SXXdg => "SXXdg",
        GateType::SYY => "SYY",
        GateType::SYYdg => "SYYdg",
        GateType::SZZ => "SZZ",
        GateType::SZZdg => "SZZdg",
        GateType::SWAP => "SWAP",
        GateType::CRZ => "CRZ",
        GateType::RXX => "RXX",
        GateType::RYY => "RYY",
        GateType::RZZ => "RZZ",
        GateType::RXXRYYRZZ => "RXXRYYRZZ",
        GateType::U2q => "U2q",
        GateType::CCX => "CCX",
        GateType::MZ => "MZ",
        GateType::MeasureLeaked => "ML",
        GateType::MeasureFree => "MF",
        GateType::PZ => "PZ",
        GateType::QAlloc => "QA",
        GateType::QFree => "QF",
        GateType::I | GateType::Idle => "I",
        GateType::MeasCrosstalkGlobalPayload | GateType::MeasCrosstalkLocalPayload => "XT",
        GateType::Custom => "?",
    }
}

/// Format an angle according to the given unit.
fn format_angle(angle: pecos_core::Angle64, unit: AngleUnit) -> String {
    match unit {
        AngleUnit::Radians => format_angle_radians(angle),
        AngleUnit::Turns => format_angle_turns(angle),
    }
}

/// Format an angle as a compact string using pi notation with fractions where
/// possible, e.g. `\u{03C0}/4`, `3\u{03C0}/2`, `\u{03C0}`.
///
/// The internal fixed-point representation stores angles as `fraction / 2^64`
/// turns, so the coefficient of pi is `fraction / 2^63`. Since the denominator
/// is a power of two, we reduce by extracting trailing zeros to get an exact
/// `p/q` ratio. When the fraction is not "nice" we fall back to decimal radians.
fn format_angle_radians(angle: pecos_core::Angle64) -> String {
    let fraction = angle.fraction();
    if fraction == 0 {
        return "0".to_string();
    }

    // coefficient of pi = fraction / 2^63
    let k = fraction.trailing_zeros(); // 0..=63 for non-zero u64
    let p = fraction >> k; // numerator (odd, >= 1)
    let q_exp = 63_u32.saturating_sub(k); // denominator = 2^q_exp
    let q: u64 = 1_u64.checked_shl(q_exp).unwrap_or(0);

    // Only use pi notation if the fraction is "nice".
    if q == 0 || q > 128 || p > 512 {
        let radians = angle.to_radians();
        return format!("{radians:.4}");
    }

    let pi = '\u{03C0}';
    match (p, q) {
        (1, 1) => format!("{pi}"),
        (2, 1) => format!("2{pi}"),
        (p, 1) => format!("{p}{pi}"),
        (1, q) => format!("{pi}/{q}"),
        (p, q) => format!("{p}{pi}/{q}"),
    }
}

/// Format an angle as a compact string in turns using fractions where possible,
/// e.g. `1/4`, `3/8`, `1`. Falls back to decimal for non-nice fractions.
///
/// The internal representation stores angles as `fraction / 2^64` turns.
/// Since the denominator is a power of two we reduce by extracting trailing
/// zeros to get an exact `p/q` ratio.
fn format_angle_turns(angle: pecos_core::Angle64) -> String {
    let fraction = angle.fraction();
    if fraction == 0 {
        return "0".to_string();
    }

    // turns = fraction / 2^64
    let k = fraction.trailing_zeros(); // 0..=63 for non-zero u64
    let p = fraction >> k; // numerator (odd, >= 1)
    let q_exp = 64_u32.saturating_sub(k); // denominator = 2^q_exp
    let q: u128 = 1_u128.checked_shl(q_exp).unwrap_or(0);

    if q == 0 || q > 128 || p > 512 {
        // Fall back to decimal
        let turns = angle.to_radians() / std::f64::consts::TAU;
        let turns = turns.rem_euclid(1.0);
        return format!("{turns:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
    }

    match (p, q) {
        (1, 1) => "1".to_string(),
        (p, 1) => format!("{p}"),
        (1, q) => format!("1/{q}"),
        (p, q) => format!("{p}/{q}"),
    }
}

/// Build the full symbol string for a gate, including angles if parameterized.
fn full_gate_symbol(gate: &Gate, unit: AngleUnit) -> String {
    let base = gate_symbol(gate.gate_type);
    if gate.angles.is_empty() {
        return base.to_string();
    }
    let angle_strs: Vec<String> = gate
        .angles
        .iter()
        .copied()
        .map(|a| format_angle(a, unit))
        .collect();
    format!("{base}({})", angle_strs.join(","))
}

// ==================== Color mapping ====================

/// Map a `GateType` to its diagram color using the PECOS axis color algebra.
fn gate_color(gate_type: GateType) -> CellColor {
    match gate_type {
        GateType::X | GateType::RX | GateType::RXX => CellColor::XAxis,
        GateType::Y | GateType::RY | GateType::RYY => CellColor::YAxis,
        GateType::Z
        | GateType::RZ
        | GateType::T
        | GateType::Tdg
        | GateType::RZZ
        | GateType::MZ
        | GateType::PZ
        | GateType::SZZ
        | GateType::SZZdg
        | GateType::CRZ => CellColor::ZAxis,
        GateType::SX | GateType::SXdg | GateType::SXX | GateType::SXXdg => CellColor::YZMix,
        GateType::SY
        | GateType::SYdg
        | GateType::SYY
        | GateType::SYYdg
        | GateType::H
        | GateType::F
        | GateType::Fdg
        | GateType::CH => CellColor::XZMix,
        GateType::SZ | GateType::SZdg => CellColor::XYMix,
        // No clear single-axis color: idle, alloc/free, multi-qubit, custom
        GateType::Idle
        | GateType::I
        | GateType::MeasureLeaked
        | GateType::MeasureFree
        | GateType::QAlloc
        | GateType::QFree
        | GateType::Custom
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::CX
        | GateType::CY
        | GateType::CZ
        | GateType::CCX
        | GateType::SWAP
        | GateType::U
        | GateType::R1XY
        | GateType::RXXRYYRZZ
        | GateType::U2q => CellColor::None,
    }
}

// ==================== Family mapping ====================

/// Map a `GateType` to its diagram family bracket/stroke style.
///
/// Most gates use `Default` brackets (`[G]`). Only measurement and preparation
/// gates keep their asymmetric brackets (`|MZ)` and `(PZ|`).
fn gate_family(gate_type: GateType) -> GateFamily {
    match gate_type {
        GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => GateFamily::Measurement,
        GateType::PZ | GateType::QAlloc | GateType::QFree => GateFamily::Preparation,
        _ => GateFamily::Default,
    }
}

// ==================== Visual range and sublayer splitting ====================

/// Compute the set of rows a gate visually occupies.
///
/// Single-qubit gates occupy only their target rows. Multi-qubit gates occupy
/// `min_row..=max_row` (all intermediate rows included) because the vertical
/// connector line passes through them.
fn compute_visual_range(
    gate: &Gate,
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
) -> BTreeSet<usize> {
    let rows: Vec<usize> = gate
        .qubits
        .iter()
        .filter_map(|q| qubit_to_row.get(q).copied())
        .collect();
    if rows.is_empty() {
        return BTreeSet::new();
    }
    let arity = gate.gate_type.quantum_arity();
    if arity <= 1 {
        rows.into_iter().collect()
    } else {
        let min = *rows.iter().min().unwrap();
        let max = *rows.iter().max().unwrap();
        (min..=max).collect()
    }
}

/// Split a layer of gates into sublayers such that no two gates in the same
/// sublayer have overlapping visual row ranges.
///
/// Uses a first-fit algorithm with gates sorted by their minimum visual row.
/// For interval graphs (contiguous visual ranges, which all multi-qubit gates
/// produce), sorting by left endpoint guarantees an optimal split that uses the
/// minimum number of sublayers (equal to the maximum clique size). Without the
/// sort, insertion order can produce unnecessary extra sublayers.
fn split_layer_into_sublayers<'a>(
    layer: &[&'a Gate],
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
) -> Vec<Vec<&'a Gate>> {
    // Compute visual ranges and sort by minimum row for optimal coloring.
    let mut gates_with_range: Vec<(&'a Gate, BTreeSet<usize>)> = layer
        .iter()
        .map(|&gate| (gate, compute_visual_range(gate, qubit_to_row)))
        .collect();
    gates_with_range.sort_by_key(|(_, range)| range.iter().next().copied().unwrap_or(0));

    let mut sublayers: Vec<(BTreeSet<usize>, Vec<&'a Gate>)> = Vec::new();

    for (gate, visual_range) in gates_with_range {
        let mut placed = false;
        for (occupied, gates) in &mut sublayers {
            if occupied.is_disjoint(&visual_range) {
                occupied.extend(&visual_range);
                gates.push(gate);
                placed = true;
                break;
            }
        }
        if !placed {
            sublayers.push((visual_range, vec![gate]));
        }
    }

    sublayers.into_iter().map(|(_, gates)| gates).collect()
}

// ==================== Grid building ====================

/// Place a single gate's decomposed cells and connectors into a diagram.
fn place_gate(
    gate: &Gate,
    diagram: &mut CircuitDiagram,
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
    num_rows: usize,
    angle_unit: AngleUnit,
) {
    let decomposed = decompose_gate(gate, qubit_to_row, num_rows, angle_unit);
    for (row, cell, color) in decomposed.cells {
        if row < num_rows {
            diagram.set_cell(row, cell, color);
        }
    }
    if let Some((top, bottom)) = decomposed.connector {
        if let Some(label) = decomposed.connector_label {
            diagram.add_labeled_connector(top, bottom, label);
        } else {
            diagram.add_connector(top, bottom);
        }
    }
}

/// Result of decomposing a gate: cells and optional connector span.
struct DecomposedGate {
    cells: Vec<(usize, DiagramCell, CellColor)>,
    /// Vertical connector span `(top_row, bottom_row)` if this is a multi-qubit gate.
    connector: Option<(usize, usize)>,
    /// Optional label to display on the connector line (for symmetric two-qubit gates).
    connector_label: Option<String>,
}

/// Decompose a single `Gate` into per-row cell assignments.
fn decompose_gate(
    gate: &Gate,
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
    num_rows: usize,
    angle_unit: AngleUnit,
) -> DecomposedGate {
    let arity = gate.gate_type.quantum_arity();
    let qubits = &gate.qubits;
    let mut cells = Vec::new();

    let color = gate_color(gate.gate_type);
    let mut connector = None;
    let mut connector_label = None;

    if arity == 1 {
        let sym = full_gate_symbol(gate, angle_unit);
        let family = gate_family(gate.gate_type);
        for &q in qubits {
            if let Some(&row) = qubit_to_row.get(&q) {
                cells.push((row, DiagramCell::Gate(sym.clone(), family), color));
            }
        }
    } else if arity == 2 {
        let sym = full_gate_symbol(gate, angle_unit);
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
                        CellColor::XAxis,
                    ));
                }
                GateType::CY => {
                    cells.push((row_a, DiagramCell::Control, CellColor::ControlDot));
                    cells.push((
                        row_b,
                        DiagramCell::Gate("Y".to_string(), GateFamily::Default),
                        CellColor::YAxis,
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
                        CellColor::XZMix,
                    ));
                }
                GateType::SWAP => {
                    cells.push((
                        row_a,
                        DiagramCell::Gate("x".to_string(), GateFamily::Default),
                        CellColor::None,
                    ));
                    cells.push((
                        row_b,
                        DiagramCell::Gate("x".to_string(), GateFamily::Default),
                        CellColor::None,
                    ));
                }
                // Symmetric two-qubit interactions: dots on both wires,
                // label on the connector line between them.
                GateType::RXX
                | GateType::RYY
                | GateType::RZZ
                | GateType::SXX
                | GateType::SXXdg
                | GateType::SYY
                | GateType::SYYdg
                | GateType::SZZ
                | GateType::SZZdg => {
                    cells.push((row_a, DiagramCell::Control, color));
                    cells.push((row_b, DiagramCell::Control, color));
                    connector_label = Some(sym.clone());
                }
                _ => {
                    let family = gate_family(gate.gate_type);
                    cells.push((row_a, DiagramCell::Gate(sym.clone(), family), color));
                    cells.push((row_b, DiagramCell::Gate(sym.clone(), family), color));
                }
            }

            connector = Some((top, bottom));

            // Intermediate rows: crossings on qubit wires.
            for row in (top + 1)..bottom {
                if row < num_rows {
                    cells.push((row, DiagramCell::Crossing, CellColor::None));
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
                CellColor::XAxis,
            ));

            connector = Some((top, bottom));

            let gate_rows: BTreeSet<usize> = rows.iter().copied().collect();
            for row in (top + 1)..bottom {
                if !gate_rows.contains(&row) && row < num_rows {
                    cells.push((row, DiagramCell::Crossing, CellColor::None));
                }
            }
        }
    }

    DecomposedGate {
        cells,
        connector,
        connector_label,
    }
}

// ==================== Diagram building ====================

/// Build a `CircuitDiagram` from gate layers.
///
/// Returns `None` when `layers` contain no qubits.
fn build_diagram(layers: &[Vec<&Gate>], angle_unit: AngleUnit) -> Option<CircuitDiagram> {
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

    let qubit_to_row: std::collections::BTreeMap<QubitId, usize> =
        qubits.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let num_rows = qubits.len();

    let labels: Vec<String> = qubits.iter().map(|q| format!("q{}", q.0)).collect();
    let mut diagram = CircuitDiagram::with_labels(labels);

    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer.is_empty() {
            if layer_idx > 0 {
                diagram.advance();
            }
            continue;
        }

        let sublayers = split_layer_into_sublayers(layer, &qubit_to_row);
        let is_split = sublayers.len() > 1;
        let mut start_col = 0;

        for (sub_idx, sublayer) in sublayers.iter().enumerate() {
            if layer_idx > 0 || sub_idx > 0 {
                diagram.advance();
            }
            if sub_idx == 0 && is_split {
                start_col = diagram.current_col();
            }
            for gate in sublayer {
                place_gate(gate, &mut diagram, &qubit_to_row, num_rows, angle_unit);
            }
        }

        if is_split {
            let end_col = diagram.current_col();
            diagram.add_column_group(format!("t{layer_idx}"), start_col, end_col);
        }
    }

    Some(diagram)
}

/// Build a `CircuitDiagram` from layers, returning an empty 0-qubit diagram if
/// there are no qubits. Used by `render_with` on `TickCircuit`/`DagCircuit`.
pub(crate) fn build_diagram_or_empty(
    layers: &[Vec<&Gate>],
    angle_unit: AngleUnit,
) -> CircuitDiagram {
    build_diagram(layers, angle_unit).unwrap_or_else(|| CircuitDiagram::new(0))
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
        assert!(q0_line.contains("[H]"));
        assert!(!q0_line.contains("[X]"));
        assert!(q1_line.contains("[X]"));
        assert!(!q1_line.contains("[H]"));
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
        // Use a non-special angle (pi/8) that won't be simplified to a named gate.
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::from_turn_ratio(1, 16), &[0]);
        });
        assert!(out.contains("RZ("));
        assert!(out.contains("\u{03C0}/8"));
    }

    #[test]
    fn angle_format_common_fractions() {
        // pi (half turn)
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::HALF_TURN, &[0]);
        });
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        assert!(
            q0.contains("\u{03C0})"),
            "half turn should show as pi: {q0}"
        );
        assert!(
            !q0.contains('/'),
            "half turn should not have a denominator: {q0}"
        );

        // pi/4 (eighth turn)
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::from_turn_ratio(1, 8), &[0]);
        });
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        assert!(
            q0.contains("\u{03C0}/4"),
            "eighth turn should show as pi/4: {q0}"
        );

        // 3pi/4
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::from_turn_ratio(3, 8), &[0]);
        });
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        assert!(
            q0.contains("3\u{03C0}/4"),
            "3/8 turn should show as 3pi/4: {q0}"
        );

        // zero
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::ZERO, &[0]);
        });
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        assert!(q0.contains("(0)"), "zero should show as 0: {q0}");
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
        assert!(out.contains("[H]"));
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
        assert!(q0_line.contains("[H]"));
        assert!(q1_line.contains("[H]"));
        assert!(q2_line.contains("[H]"));
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
        assert!(out.contains("[H]")); // Default
        assert!(out.contains("[SX]")); // Default
        assert!(out.contains("[X]")); // Default
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
        assert!(out.contains("[H]"));
        assert!(out.contains("[SX]"));
        assert!(out.contains("[X]"));
        assert!(out.contains("|MZ)"));
    }

    #[test]
    fn svg_gates_have_solid_strokes() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().sz(&[0]);
        let svg = tc.to_svg();
        // All gates now use Default family with solid strokes (no dasharray).
        assert!(!svg.contains("stroke-dasharray"));
    }

    // ==================== render_with tests ====================

    #[test]
    fn tick_render_with_default_matches_to_ascii() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        let style = pecos_core::circuit_diagram::DiagramStyle::default();
        let via_render_with = tc.render_with(&style).text();
        let via_to_ascii = tc.to_ascii();
        assert_eq!(via_render_with, via_to_ascii);
    }

    #[test]
    fn tick_render_with_custom_palette_svg() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        let style = pecos_core::circuit_diagram::DiagramStyle::builder()
            .xz_mix("#AABBCC", "#112233", "#445566")
            .build();
        let svg = tc.render_with(&style).svg();
        // H is XZMix, so the custom colors should appear.
        assert!(svg.contains("#AABBCC"));
        assert!(svg.contains("#112233"));
    }

    #[test]
    fn tick_render_with_monochrome() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().x(&[0]);
        let style = pecos_core::circuit_diagram::DiagramStyle::builder()
            .color(false)
            .build();
        let svg = tc.render_with(&style).svg();
        // XAxis color should NOT appear.
        assert!(!svg.contains("#FFB0B0"));
    }

    #[test]
    fn dag_render_with_default_matches_to_ascii() {
        let mut dag = crate::DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);
        let style = pecos_core::circuit_diagram::DiagramStyle::default();
        let via_render_with = dag.render_with(&style).text();
        let via_to_ascii = dag.to_ascii();
        assert_eq!(via_render_with, via_to_ascii);
    }

    #[test]
    fn tick_render_with_ascii_and_unicode() {
        let mut tc = crate::TickCircuit::new();
        tc.tick().h(&[0]);
        let style = pecos_core::circuit_diagram::DiagramStyle::default();
        let r = tc.render_with(&style);
        let ascii = r.ascii();
        let unicode = r.unicode();
        assert!(ascii.contains('-'));
        assert!(unicode.contains('\u{2500}'));
    }

    // ==================== Rotation display tests ====================

    #[test]
    fn rotation_displays_angle_faithfully() {
        // Visualizer should display the rotation gate as-is (no simplification).
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        });
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        assert!(q0.contains("RZ("), "should show RZ label: {q0}");
        assert!(q0.contains("\u{03C0}/2"), "should show angle: {q0}");
    }

    #[test]
    fn non_special_angle_displays_rotation() {
        let out = render_tick(|tc| {
            tc.tick().rz(Angle64::from_turn_ratio(1, 6), &[0]);
        });
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        assert!(
            q0.contains("RZ("),
            "non-special angle should keep RZ label: {q0}"
        );
    }

    #[test]
    fn rzz_displays_as_symmetric_gate() {
        let out = render_tick(|tc| {
            let eighth = Angle64::QUARTER_TURN / 2u64;
            tc.tick().rzz(eighth, &[(0, 1)]);
        });
        assert!(
            out.contains("[RZZ("),
            "RZZ should show bracketed label: {out}"
        );
    }

    // ==================== Sub-column splitting tests ====================

    #[test]
    fn overlapping_cx_cz_splits_into_two_columns() {
        let out = render_tick(|tc| {
            tc.tick().h(&[0, 1, 2, 3]);
            let mut t = tc.tick();
            t.cx(&[(0, 2)]);
            t.cz(&[(1, 3)]);
            tc.tick().mz(&[0, 1, 2, 3]);
        });
        // Both gates should be visible (no overwriting).
        assert!(out.contains("[X]"), "CX target should be visible: {out}");
        let dot_count = out.matches('.').count();
        assert!(
            dot_count >= 3,
            "should have control dots for CX and CZ: {out}"
        );
        // Bracket annotation should be present.
        assert!(
            out.contains("t1"),
            "bracket label for tick 1 should appear: {out}"
        );
    }

    #[test]
    fn non_overlapping_gates_stay_in_one_column() {
        let out = render_tick(|tc| {
            let mut t = tc.tick();
            t.cx(&[(0, 1)]);
            t.cz(&[(2, 3)]);
        });
        // No bracket annotation since no splitting needed.
        assert!(
            !out.contains("|--"),
            "should not have bracket dashes: {out}"
        );
    }

    #[test]
    fn single_qubit_gates_never_split() {
        let out = render_tick(|tc| {
            tc.tick().h(&[0]).x(&[1]).z(&[2]);
        });
        // No bracket annotation for single-qubit gates in the same tick.
        assert!(
            !out.contains("|--"),
            "should not have bracket dashes: {out}"
        );
    }

    #[test]
    fn chain_overlap_uses_optimal_two_sublayers() {
        // Chain pattern: CZ(0,2)-CX(1,4)-CX(3,6)-CZ(5,7)
        // Max clique = 2, so optimal split is 2 sub-columns.
        // Without sorting by min row, naive first-fit with this insertion
        // order would produce 3: CZ(0,2)+CZ(5,7) in bin 1, then CX(1,4)
        // conflicts bin 1 -> bin 2, then CX(3,6) conflicts both -> bin 3.
        let out = render_tick(|tc| {
            tc.tick().h(&[0, 1, 2, 3, 4, 5, 6, 7]);
            let mut t = tc.tick();
            // Deliberately add in worst-case order for naive greedy.
            t.cz(&[(0, 2)]);
            t.cz(&[(5, 7)]);
            t.cx(&[(1, 4)]);
            t.cx(&[(3, 6)]);
        });
        // Count the bracket dashes in the annotation line to determine the
        // number of sub-columns. With 2 sub-columns we get one bracket group
        // spanning 2 columns. With 3 we would get a wider span.
        // Verify only one bracket group (one "t1" label).
        let bracket_lines: Vec<&str> = out.lines().filter(|l| l.contains("t1")).collect();
        assert_eq!(
            bracket_lines.len(),
            1,
            "should have exactly one bracket: {out}"
        );

        // Count diagram columns: with optimal splitting, the tick should use
        // exactly 2 sub-columns. Count distinct column positions by looking
        // at how many gate symbols appear on the first qubit wire.
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        let q1 = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        // q0 has H in tick 0, then control dot (.) in the overlap tick.
        // q1 has H in tick 0, then either crossing or control in the overlap tick.
        // All 4 gates should be visible.
        assert!(out.contains("[X]"), "CX target should be visible: {out}");
        // Count control dots: CZ(0,2) has 2 dots, CZ(5,7) has 2 dots,
        // CX(1,4) has 1 dot, CX(3,6) has 1 dot = 6 total.
        let dot_count = out.matches('.').count();
        assert!(
            dot_count >= 6,
            "all 6 control dots should be visible (got {dot_count}): {out}"
        );

        // Verify 2 sub-columns, not 3: count the column separators in the
        // bracket line. A 2-sub-column bracket has the pattern |---t1---|
        // spanning 2 column widths. A 3-sub-column bracket would be wider.
        // More direct check: count how many [H] appear on q0.
        let h_on_q0 = q0.matches("[H]").count();
        let dots_on_q0 = q0.matches('.').count();
        assert_eq!(h_on_q0, 1, "q0 should have one H: {q0}");
        assert_eq!(dots_on_q0, 1, "q0 should have one control dot: {q0}");
        // q1 should have H and either a crossing or a control
        let h_on_q1 = q1.matches("[H]").count();
        assert_eq!(h_on_q1, 1, "q1 should have one H: {q1}");
    }

    #[test]
    fn overlapping_rzz_szz_splits_correctly() {
        let out = render_tick(|tc| {
            let quarter = Angle64::QUARTER_TURN;
            let mut t = tc.tick();
            t.rzz(quarter, &[(0, 2)]);
            t.szz(&[(1, 3)]);
        });
        // Both gates should be visible after splitting.
        assert!(out.contains("t0"), "bracket should appear: {out}");
        // RZZ label should appear in a connector row.
        assert!(
            out.contains("[RZZ(") || out.contains("RZZ("),
            "RZZ label should be visible: {out}"
        );
        // SZZ label should appear.
        assert!(
            out.contains("[SZZ]") || out.contains("SZZ"),
            "SZZ label should be visible: {out}"
        );
    }

    #[test]
    fn non_adjacent_rzz_renders_label() {
        // RZZ spanning 3 rows (1 intermediate qubit).
        let out = render_tick(|tc| {
            let quarter = Angle64::QUARTER_TURN;
            tc.tick().h(&[0, 1, 2]);
            tc.tick().rzz(quarter, &[(0, 2)]);
        });
        assert!(out.contains("RZZ("), "RZZ label should be visible: {out}");

        // RZZ spanning 5 rows (3 intermediate qubits).
        let out = render_tick(|tc| {
            let quarter = Angle64::QUARTER_TURN;
            tc.tick().h(&[0, 1, 2, 3, 4]);
            tc.tick().rzz(quarter, &[(0, 4)]);
        });
        assert!(
            out.contains("RZZ("),
            "RZZ label should be visible on wide span: {out}"
        );
    }

    #[test]
    fn three_mutually_overlapping_gates_use_three_sublayers() {
        // Three gates that all pairwise overlap: need 3 sub-columns.
        let out = render_tick(|tc| {
            let mut t = tc.tick();
            t.cx(&[(0, 3)]);
            t.cz(&[(1, 4)]);
            t.cx(&[(2, 5)]);
        });
        // All three gates pairwise overlap (ranges {0..3}, {1..4}, {2..5}),
        // so max clique = 3, requiring 3 sub-columns.
        assert!(out.contains("t0"), "bracket should appear: {out}");
        // All gates should be visible.
        let dot_count = out.matches('.').count();
        assert!(
            dot_count >= 4,
            "should have control dots for all gates (got {dot_count}): {out}"
        );
    }
}
