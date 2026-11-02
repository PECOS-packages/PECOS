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

use pecos_core::circuit_diagram::{
    AngleUnit, CellColor, CircuitDiagram, DiagramCell, DiagramStyle, GateFamily,
};
use pecos_core::gate_type::GateType;
use pecos_core::{Gate, QubitId};
use std::borrow::Cow;
use std::collections::BTreeSet;

/// An invalid half-open tick or dependency-layer range for diagram rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagramRangeError {
    /// Requested inclusive start index.
    pub start: usize,
    /// Requested exclusive end index.
    pub end: usize,
    /// Number of ticks or dependency layers in the source circuit.
    pub length: usize,
}

impl std::fmt::Display for DiagramRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid diagram range {}..{}: expected start <= end <= {}",
            self.start, self.end, self.length
        )
    }
}

impl std::error::Error for DiagramRangeError {}

pub(crate) fn validate_diagram_range(
    range: &std::ops::Range<usize>,
    length: usize,
) -> Result<(), DiagramRangeError> {
    if range.start > range.end || range.end > length {
        return Err(DiagramRangeError {
            start: range.start,
            end: range.end,
            length,
        });
    }
    Ok(())
}

/// Borrowed operation and display metadata, independent of circuit storage.
#[derive(Clone, Copy)]
pub(crate) struct DiagramGate<'a> {
    gate: &'a Gate,
    symbol: Option<&'a str>,
    operation_id: Option<&'a str>,
}

impl<'a> DiagramGate<'a> {
    pub(crate) fn new(gate: &'a Gate, symbol: Option<&'a crate::Attribute>) -> Self {
        Self {
            gate,
            symbol: match symbol {
                Some(crate::Attribute::String(name)) if gate.gate_type == GateType::Custom => {
                    Some(name)
                }
                _ => None,
            },
            operation_id: None,
        }
    }

    pub(crate) fn with_operation_id(mut self, value: Option<&'a crate::Attribute>) -> Self {
        if let Some(crate::Attribute::String(id)) = value {
            self.operation_id = Some(id);
        }
        self
    }
}

/// These commands describe one operation over their entire support, rather
/// than a batch of one-qubit operations despite their type-level arity of one.
fn has_dynamic_support(gate: &Gate) -> bool {
    matches!(gate.gate_type, GateType::Custom | GateType::Channel)
}

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
        GateType::RXY1Q => "RXY1Q",
        GateType::RXYXY2Q => "RXYXY2Q",
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
        GateType::RXX => "RXX",
        GateType::RYY => "RYY",
        GateType::RZZ => "RZZ",
        GateType::RXXRYYRZZ => "RXXRYYRZZ",
        GateType::U2q => "U2q",
        GateType::CCX => "CCX",
        GateType::CCZ => "CCZ",
        GateType::CS => "CS",
        GateType::CSdg => "CSdg",
        GateType::MX => "MX",
        GateType::MZ => "MZ",
        GateType::MeasureLeaked => "MZ_L",
        GateType::MeasureFree => "MF",
        GateType::MPZ => "MPZ",
        GateType::PX => "PX",
        GateType::PZ => "PZ",
        GateType::QAlloc => "QA",
        GateType::QFree => "QF",
        GateType::I | GateType::Idle => "I",
        GateType::MeasCrosstalkGlobalPayload => "XT-excl",
        GateType::MeasCrosstalkLocalPayload => "XT-local",
        GateType::Channel => "Ch",
        GateType::Custom => "?",
        GateType::TrackedPauliMeta => "TP",
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
        if radians < 0.00005 {
            return format!("{radians:.4e}");
        }
        if std::f64::consts::TAU - radians < 0.00005 {
            // Negate in fixed point first: subtracting two f64 values close
            // to TAU can erase the smallest representable nonzero angles.
            let residual = -(-angle).to_radians();
            return format!("{residual:.4e}");
        }
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
        let turns = angle.to_turns();
        if turns < 5e-7 {
            return format!("{turns:.6e}");
        }
        if 1.0 - turns < 5e-7 {
            let residual = -(-angle).to_turns();
            return format!("{residual:.6e}");
        }
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
fn full_gate_symbol(gate: &Gate, unit: AngleUnit, symbol: Option<&str>) -> String {
    if gate.gate_type == GateType::Idle {
        return format!("Idle({})", gate.idle_duration());
    }
    let base = symbol.unwrap_or_else(|| gate_symbol(gate.gate_type));
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
        GateType::X | GateType::RX | GateType::RXX | GateType::MX | GateType::PX => {
            CellColor::XAxis
        }
        GateType::Y | GateType::RY | GateType::RYY => CellColor::YAxis,
        GateType::Z
        | GateType::RZ
        | GateType::T
        | GateType::Tdg
        | GateType::CCZ
        | GateType::CS
        | GateType::CSdg
        | GateType::RZZ
        | GateType::MZ
        | GateType::PZ
        | GateType::SZZ
        | GateType::SZZdg => CellColor::ZAxis,
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
        | GateType::MPZ
        | GateType::QAlloc
        | GateType::QFree
        | GateType::Custom
        | GateType::Channel
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::CX
        | GateType::CY
        | GateType::CZ
        | GateType::CCX
        | GateType::SWAP
        | GateType::U
        | GateType::RXY1Q
        | GateType::RXYXY2Q
        | GateType::RXXRYYRZZ
        | GateType::U2q
        | GateType::TrackedPauliMeta => CellColor::None,
    }
}

// ==================== Family mapping ====================

/// Map a `GateType` to its diagram family bracket/stroke style.
///
/// Most gates use `Default` brackets (`[G]`). Only measurement and preparation
/// gates keep their asymmetric brackets (`|MZ)` and `(PZ|`).
fn gate_family(gate_type: GateType) -> GateFamily {
    match gate_type {
        GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree | GateType::MPZ => {
            GateFamily::Measurement
        }
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
    if arity <= 1 && !has_dynamic_support(gate) {
        rows.into_iter().collect()
    } else {
        let min = *rows.iter().min().expect("rows is non-empty");
        let max = *rows.iter().max().expect("rows is non-empty");
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
    layer: &[DiagramGate<'a>],
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
) -> Vec<Vec<DiagramGate<'a>>> {
    // Compute visual ranges and sort by minimum row for optimal coloring.
    let mut gates_with_range: Vec<(DiagramGate<'a>, BTreeSet<usize>)> = layer
        .iter()
        .map(|&gate| (gate, compute_visual_range(gate.gate, qubit_to_row)))
        .collect();
    gates_with_range.sort_by_key(|(_, range)| range.iter().next().copied().unwrap_or(0));

    let mut sublayers: Vec<(BTreeSet<usize>, Vec<DiagramGate<'a>>)> = Vec::new();

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
    gate: DiagramGate<'_>,
    diagram: &mut CircuitDiagram,
    qubit_to_row: &std::collections::BTreeMap<QubitId, usize>,
    num_rows: usize,
    style: &DiagramStyle,
) {
    diagram.set_operation_id(gate.operation_id);
    let decomposed = decompose_gate(gate.gate, qubit_to_row, num_rows, style, gate.symbol);
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
    diagram.set_operation_id(None);
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
    style: &DiagramStyle,
    symbol: Option<&str>,
) -> DecomposedGate {
    let angle_unit = style.angle_unit;
    let arity = gate.gate_type.quantum_arity();
    let qubits = &gate.qubits;
    let mut cells = Vec::new();

    let color = gate_color(gate.gate_type);
    let mut connector = None;
    let mut connector_label = None;

    if (has_dynamic_support(gate) && qubits.len() > 1) || gate.gate_type == GateType::U2q {
        let sym = full_gate_symbol(gate, angle_unit, symbol);
        let rows: Vec<_> = qubits
            .iter()
            .enumerate()
            .filter_map(|(operand, q)| qubit_to_row.get(q).map(|&row| (operand, row)))
            .collect();
        for &(operand, row) in &rows {
            cells.push((
                row,
                DiagramCell::Gate(format!("{sym}[{operand}]"), gate_family(gate.gate_type)),
                color,
            ));
        }
        if let (Some(top), Some(bottom)) = (
            rows.iter().map(|&(_, row)| row).min(),
            rows.iter().map(|&(_, row)| row).max(),
        ) {
            connector = Some((top, bottom));
            let gate_rows: BTreeSet<_> = rows.iter().map(|&(_, row)| row).collect();
            for row in (top + 1)..bottom {
                if !gate_rows.contains(&row) && row < num_rows {
                    cells.push((row, DiagramCell::Crossing, CellColor::None));
                }
            }
        }
    } else if arity == 1 {
        let sym = full_gate_symbol(gate, angle_unit, symbol);
        let family = gate_family(gate.gate_type);
        // MZ_L returns 0/1/2 through runtime engines even though the current
        // circuit record allocator does not include it in its predicate.
        let show_id = style.show_measurement_ids
            && (gate.gate_type.consumes_measurement_record()
                || gate.gate_type == GateType::MeasureLeaked);
        for (position, &q) in qubits.iter().enumerate() {
            if let Some(&row) = qubit_to_row.get(&q) {
                let label = if show_id {
                    // Match operand order before sorting rows. A malformed batch
                    // cannot establish a reliable association even for its prefix.
                    let id = gate
                        .meas_ids
                        .get(position)
                        .filter(|_| gate.meas_ids.len() == qubits.len());
                    id.map_or_else(|| format!("{sym}[m?]"), |id| format!("{sym}[{id}]"))
                } else {
                    sym.clone()
                };
                cells.push((row, DiagramCell::Gate(label, family), color));
            }
        }
    } else if arity == 2 {
        let sym = full_gate_symbol(gate, angle_unit, symbol);
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
            let rows: Vec<usize> = rows
                .into_iter()
                .map(|r| r.expect("checked for None above"))
                .collect();
            let top = *rows.iter().min().expect("rows has exactly 3 elements");
            let bottom = *rows.iter().max().expect("rows has exactly 3 elements");

            cells.push((rows[0], DiagramCell::Control, CellColor::ControlDot));
            cells.push((rows[1], DiagramCell::Control, CellColor::ControlDot));
            cells.push((
                rows[2],
                DiagramCell::Gate(
                    if gate.gate_type == GateType::CCZ {
                        "Z"
                    } else {
                        "X"
                    }
                    .to_string(),
                    GateFamily::Default,
                ),
                if gate.gate_type == GateType::CCZ {
                    CellColor::ZAxis
                } else {
                    CellColor::XAxis
                },
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
fn build_diagram(
    layers: &[Vec<DiagramGate<'_>>],
    style: &DiagramStyle,
    prefix: &str,
) -> Option<CircuitDiagram> {
    let mut qubit_set = BTreeSet::new();
    for layer in layers {
        for gate in layer {
            for &q in &gate.gate.qubits {
                qubit_set.insert(q);
            }
        }
    }
    let qubits: Vec<QubitId> = qubit_set.into_iter().collect();
    if qubits.is_empty() {
        return None;
    }

    Some(build_diagram_with_context(
        layers, &qubits, style, prefix, 0, false,
    ))
}

/// Build selected layers using the source circuit's qubit rows and axis indices.
pub(crate) fn build_diagram_with_context(
    layers: &[Vec<DiagramGate<'_>>],
    qubits: &[QubitId],
    style: &DiagramStyle,
    prefix: &str,
    offset: usize,
    label_all: bool,
) -> CircuitDiagram {
    if layers.is_empty() || qubits.is_empty() {
        return CircuitDiagram::new(0);
    }

    let qubit_to_row: std::collections::BTreeMap<QubitId, usize> =
        qubits.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let num_rows = qubits.len();

    let labels: Vec<String> = qubits.iter().map(|q| format!("q{}", q.0)).collect();
    let mut diagram = CircuitDiagram::with_labels(labels);

    for (layer_idx, layer) in layers.iter().enumerate() {
        let original_idx = offset + layer_idx;
        if layer.is_empty() {
            if layer_idx > 0 {
                diagram.advance();
            }
            let col = diagram.current_col();
            diagram.add_column_group(format!("{prefix}{original_idx}"), col, col);
            continue;
        }

        // Each multi-qubit instance needs its own connector and visual range.
        // Keep batching as a storage detail, not a diagram primitive.
        let gates: Vec<(Cow<'_, Gate>, Option<&str>, Option<&str>)> = layer
            .iter()
            .flat_map(|display_gate| {
                let gate = display_gate.gate;
                let arity = gate.gate_type.quantum_arity();
                if has_dynamic_support(gate) || arity <= 1 || gate.qubits.len() <= arity {
                    vec![(
                        Cow::Borrowed(gate),
                        display_gate.symbol,
                        display_gate.operation_id,
                    )]
                } else {
                    gate.qubits
                        .chunks(arity)
                        .map(|qubits| {
                            (
                                Cow::Owned(Gate {
                                    gate_type: gate.gate_type,
                                    angles: gate.angles.clone(),
                                    params: gate.params.clone(),
                                    qubits: qubits.iter().copied().collect(),
                                    meas_ids: gate.meas_ids.clone(),
                                    channel: gate.channel.clone(),
                                }),
                                display_gate.symbol,
                                display_gate.operation_id,
                            )
                        })
                        .collect()
                }
            })
            .collect();
        let gate_refs: Vec<_> = gates
            .iter()
            .map(|(gate, symbol, operation_id)| DiagramGate {
                gate: gate.as_ref(),
                symbol: *symbol,
                operation_id: *operation_id,
            })
            .collect();
        let sublayers = split_layer_into_sublayers(&gate_refs, &qubit_to_row);
        let is_split = sublayers.len() > 1;
        let mut start_col = 0;

        for (sub_idx, sublayer) in sublayers.iter().enumerate() {
            if layer_idx > 0 || sub_idx > 0 {
                diagram.advance();
            }
            if sub_idx == 0 {
                start_col = diagram.current_col();
            }
            for gate in sublayer {
                place_gate(*gate, &mut diagram, &qubit_to_row, num_rows, style);
            }
        }

        if is_split || label_all {
            let end_col = diagram.current_col();
            diagram.add_column_group(format!("{prefix}{original_idx}"), start_col, end_col);
        }
    }

    diagram
}

/// Build a `CircuitDiagram` from layers, returning an empty 0-qubit diagram if
/// there are no qubits. Used by `render_with` on `TickCircuit`/`DagCircuit`.
pub(crate) fn build_diagram_or_empty(
    layers: &[Vec<DiagramGate<'_>>],
    style: &DiagramStyle,
    prefix: &str,
) -> CircuitDiagram {
    build_diagram(layers, style, prefix).unwrap_or_else(|| CircuitDiagram::new(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::Angle64;

    #[test]
    fn measurement_labels_follow_operand_ids_across_ranges_and_dag_conversion() {
        let style = DiagramStyle::builder().show_measurement_ids(true).build();
        let mut circuit = crate::TickCircuit::new();
        circuit.tick().h(&[2, 8]);
        let mut gate = Gate::mz(&[8, 2]);
        gate.meas_ids = vec![
            pecos_core::MeasId::from_raw(42),
            pecos_core::MeasId::from_raw(7),
        ]
        .into();
        circuit.tick().try_add_gate(gate).unwrap();
        let dag = crate::DagCircuit::try_from(&circuit).unwrap();
        let before = format!("{circuit:?}");
        for renderer in [
            circuit.render_with(&style),
            dag.render_with(&style),
            circuit.render_range(1..2, &style).unwrap(),
            dag.render_range(1..2, &style).unwrap(),
        ] {
            let ascii = renderer.ascii();
            for (qubit, id) in [(2, 7), (8, 42)] {
                let row = ascii
                    .lines()
                    .find(|line| line.starts_with(&format!("q{qubit}:")))
                    .unwrap();
                assert!(row.contains(&format!("MZ[m{id}]")), "{ascii}");
            }
            for output in [
                renderer.svg(),
                renderer.tikz(),
                renderer.dot(),
                renderer.unicode(),
            ] {
                assert!(output.contains("MZ[m42]"), "{output}");
                assert!(output.contains("MZ[m7]"), "{output}");
            }
        }
        assert_eq!(format!("{circuit:?}"), before);
        let hidden = DiagramStyle::builder().show_measurement_ids(false).build();
        assert_eq!(circuit.render_with(&hidden).svg(), circuit.to_svg());
        assert!(!circuit.to_ascii().contains("[m"));
    }

    #[test]
    fn measurement_labels_cover_ordinary_and_leakage_aware_measurements() {
        let style = DiagramStyle::builder().show_measurement_ids(true).build();
        for (gt, name) in [
            (GateType::MX, "MX"),
            (GateType::MZ, "MZ"),
            (GateType::MPZ, "MPZ"),
            (GateType::MeasureFree, "MF"),
            (GateType::MeasureLeaked, "MZ_L"),
        ] {
            let mut circuit = crate::TickCircuit::new();
            let mut gate = Gate::simple(gt, vec![QubitId(8), QubitId(2)]);
            gate.meas_ids = vec![
                pecos_core::MeasId::from_raw(0),
                pecos_core::MeasId::from_raw(9000),
            ]
            .into();
            circuit.tick().try_add_gate(gate).unwrap();
            let dag = crate::DagCircuit::try_from(&circuit).unwrap();
            for output in [
                circuit.render_with(&style).ascii(),
                dag.render_with(&style).ascii(),
            ] {
                assert!(output.contains(&format!("{name}[m0]")), "{output}");
                assert!(output.contains(&format!("{name}[m9000]")), "{output}");
            }
        }
    }

    #[test]
    fn measurement_labels_do_not_invent_ids_for_missing_or_malformed_batches() {
        let style = DiagramStyle::builder().show_measurement_ids(true).build();
        for gt in [
            GateType::MX,
            GateType::MZ,
            GateType::MPZ,
            GateType::MeasureFree,
            GateType::MeasureLeaked,
        ] {
            for ids in [vec![], vec![7], vec![7, 9, 11]] {
                let mut gate = Gate::simple(gt, vec![QubitId(8), QubitId(2)]);
                gate.meas_ids = ids
                    .iter()
                    .copied()
                    .map(pecos_core::MeasId::from_raw)
                    .collect();
                let before = gate.clone();
                // Malformed payloads cannot pass public insertion validation;
                // exercise the renderer directly to prevent plausible false IDs.
                let layers = vec![vec![DiagramGate::new(&gate, None)]];
                let output = build_diagram_or_empty(&layers, &style, "t").render_text("", &style);
                let expected = 2;
                assert_eq!(output.matches("[m?]").count(), expected, "{output}");
                assert!(!output.contains("[m7]"));
                assert_eq!(gate, before);
            }
        }
        let mut circuit = crate::TickCircuit::new();
        circuit.tick().try_add_gate(Gate::mz(&[2, 8])).unwrap();
        let before = format!("{circuit:?}");
        assert_eq!(
            circuit.render_with(&style).ascii().matches("[m?]").count(),
            2
        );
        assert_eq!(format!("{circuit:?}"), before);
        circuit.tick().custom_gate("MY", &[2], &[]).unwrap();
        assert!(circuit.render_with(&style).ascii().contains("[MY]"));
        assert!(!circuit.render_with(&style).ascii().contains("MY[m"));
    }

    #[test]
    fn measurement_labels_preserve_duplicate_id_evidence() {
        let style = DiagramStyle::builder().show_measurement_ids(true).build();
        let mut circuit = crate::TickCircuit::new();
        for qubit in [0, 1] {
            let mut gate = Gate::mz(&[qubit]);
            gate.meas_ids = [pecos_core::MeasId::from_raw(42)].into();
            circuit.tick().try_add_gate(gate).unwrap();
        }
        assert_eq!(
            circuit
                .render_with(&style)
                .ascii()
                .matches("MZ[m42]")
                .count(),
            2
        );
        assert!(crate::DagCircuit::try_from(&circuit).is_err());
    }

    #[test]
    fn diagram_ranges_check_bounds_without_clamping() {
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().h(&[0]);
        tick.tick().x(&[0]);
        let dag = crate::DagCircuit::try_from(&tick).unwrap();
        for (start, end) in [(2, 1), (0, 3), (3, 3), (usize::MAX, usize::MAX)] {
            let range = start..end;
            for result in [
                tick.render_range(range.clone(), &style),
                dag.render_range(range.clone(), &style),
            ] {
                let error = result.err().expect("invalid range must fail");
                assert_eq!(error.start, range.start);
                assert_eq!(error.end, range.end);
                assert_eq!(error.length, 2);
                assert!(error.to_string().contains("start <= end <= 2"));
            }
        }
        for range in [0..0, 0..2, 1..2, 2..2] {
            assert!(tick.render_range(range.clone(), &style).is_ok());
            assert!(dag.render_range(range, &style).is_ok());
        }
    }

    #[test]
    fn empty_diagram_ranges_render_only_the_selected_header() {
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().h(&[8]);
        let dag = crate::DagCircuit::try_from(&tick).unwrap();
        assert_eq!(
            tick.render_range(1..1, &style).unwrap().ascii(),
            "TickCircuit: 1 qubit, ticks 1..1 of 1\n"
        );
        assert_eq!(
            dag.render_range(0..0, &style).unwrap().ascii(),
            "DagCircuit: 1 qubit, layers 0..0 of 1\n"
        );
        for renderer in [
            crate::TickCircuit::new()
                .render_range(0..0, &style)
                .unwrap(),
            crate::DagCircuit::new().render_range(0..0, &style).unwrap(),
        ] {
            assert!(renderer.ascii().contains("0 qubits"));
            assert!(renderer.svg().contains("<svg"));
        }
    }

    #[test]
    fn range_rendering_retains_idle_sparse_qubits_and_original_indices() {
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().h(&[3, 50]);
        tick.tick().x(&[3]);
        tick.tick().z(&[3]);
        let dag = crate::DagCircuit::try_from(&tick).unwrap();
        for (renderer, prefix) in [
            (tick.render_range(1..3, &style).unwrap(), "t"),
            (dag.render_range(1..3, &style).unwrap(), "l"),
        ] {
            for output in [
                renderer.ascii(),
                renderer.svg(),
                renderer.tikz(),
                renderer.dot(),
            ] {
                assert!(output.contains("q3"), "{output}");
                assert!(output.contains("q50"), "{output}");
                assert!(output.contains(&format!("{prefix}1")), "{output}");
                assert!(output.contains(&format!("{prefix}2")), "{output}");
                assert!(!output.contains(&format!("{prefix}0")), "{output}");
            }
            assert!(!renderer.ascii().contains("[H]"));
            assert!(renderer.ascii().contains("[X]"));
            assert!(renderer.ascii().contains("[Z]"));
        }
    }

    #[test]
    fn range_rendering_keeps_selected_empty_ticks() {
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().h(&[4]);
        tick.tick();
        tick.tick();
        let renderer = tick.render_range(1..3, &style).unwrap();
        for output in [
            renderer.ascii(),
            renderer.svg(),
            renderer.tikz(),
            renderer.dot(),
        ] {
            assert!(output.contains("q4"), "{output}");
            assert!(output.contains("t1"), "{output}");
            assert!(output.contains("t2"), "{output}");
        }
        let mut no_qubits = crate::TickCircuit::new();
        no_qubits.tick();
        assert_eq!(
            no_qubits.render_range(0..1, &style).unwrap().ascii(),
            "TickCircuit: 0 qubits, ticks 0..1 of 1\n"
        );
    }

    #[test]
    fn range_rendering_keeps_crossing_subcolumns_without_a_leading_gap() {
        let gates = [Gate::cx(&[(0, 2)]), Gate::cx(&[(1, 3)])];
        let layers = vec![
            gates
                .iter()
                .map(|gate| DiagramGate::new(gate, None))
                .collect(),
        ];
        let diagram = build_diagram_with_context(
            &layers,
            &[QubitId(0), QubitId(1), QubitId(2), QubitId(3)],
            &DiagramStyle::default(),
            "t",
            7,
            true,
        );
        assert_eq!(diagram.current_col(), 1, "exactly two visual columns");
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().h(&[0, 1, 2, 3]);
        tick.tick().cx(&[(0, 2), (1, 3)]);
        let dag = crate::DagCircuit::try_from(&tick).unwrap();
        for (renderer, label) in [
            (tick.render_range(1..2, &style).unwrap(), "t1"),
            (dag.render_range(1..2, &style).unwrap(), "l1"),
        ] {
            assert_eq!(renderer.ascii().matches("[X]").count(), 2);
            assert_eq!(renderer.svg().matches("<circle").count(), 2);
            assert!(renderer.svg().contains(&format!(">{label}</text>")));
        }
    }

    #[test]
    fn range_rendering_does_not_mutate_circuits_or_metadata() {
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().meta("round", crate::Attribute::Int(4)).h(&[0]);
        tick.tick().mz(&[0]);
        let dag = crate::DagCircuit::try_from(&tick).unwrap();
        let tick_before = format!("{tick:?}");
        let dag_before = format!("{dag:?}");
        let _ = tick.render_range(1..2, &style).unwrap().svg();
        let _ = dag.render_range(1..2, &style).unwrap().svg();
        assert_eq!(format!("{tick:?}"), tick_before);
        assert_eq!(format!("{dag:?}"), dag_before);
    }

    #[test]
    fn full_diagram_range_keeps_the_same_gate_layout() {
        let style = crate::DiagramStyle::default();
        let mut tick = crate::TickCircuit::new();
        tick.tick().h(&[0, 1, 2, 3]);
        tick.tick().cx(&[(0, 2), (1, 3)]);
        tick.tick().mz(&[0]);
        let dag = crate::DagCircuit::try_from(&tick).unwrap();
        for (full, selected) in [
            (
                tick.to_ascii(),
                tick.render_range(0..3, &style).unwrap().ascii(),
            ),
            (
                dag.to_ascii(),
                dag.render_range(0..3, &style).unwrap().ascii(),
            ),
        ] {
            let rows = |text: &str| {
                text.lines()
                    .filter(|line| line.starts_with('q'))
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            };
            assert_eq!(rows(&full), rows(&selected));
        }
    }

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
    fn custom_names_and_operand_order_survive_tick_and_dag_rendering() {
        let mut circuit = crate::TickCircuit::new();
        circuit
            .tick()
            .custom_gate("Entangle", &[5, 0, 9, 2], &[])
            .unwrap();
        circuit.get_tick_mut(0).unwrap().add_gate(Gate::h(&[4]));
        circuit.tick().custom_gate("Phase", &[0], &[]).unwrap();
        let dag = crate::DagCircuit::try_from(&circuit).unwrap();
        for text in [circuit.to_ascii(), dag.to_ascii()] {
            for (qubit, operand) in [(5, 0), (0, 1), (9, 2), (2, 3)] {
                let row = text
                    .lines()
                    .find(|line| line.starts_with(&format!("q{qubit}:")))
                    .unwrap();
                assert!(row.contains(&format!("Entangle[{operand}]")), "{text}");
            }
            assert!(text.contains("[Phase]"), "{text}");
            assert!(
                text.contains("[H]"),
                "the crossing must not erase H: {text}"
            );
            assert!(!text.contains('?'), "{text}");
        }
        assert!(circuit.to_svg().contains(">t0</text>"));
        assert!(dag.to_svg().contains(">l0</text>"));
        assert_eq!(circuit.to_svg().matches("<line ").count(), 6);
    }

    #[test]
    fn correlated_channels_keep_a_connector_and_singletons_remain_single() {
        let mut circuit = crate::TickCircuit::new();
        circuit
            .tick()
            .channel(pecos_core::channel::Depolarizing2(0.1, 0, 2));
        circuit.get_tick_mut(0).unwrap().add_gate(Gate::h(&[1]));
        circuit
            .tick()
            .channel(pecos_core::channel::Depolarizing(0.1, 0));
        let dag = crate::DagCircuit::try_from(&circuit).unwrap();
        for text in [circuit.to_ascii(), dag.to_ascii()] {
            assert!(text.contains("[Ch[0]]"), "{text}");
            assert!(text.contains("[Ch[1]]"), "{text}");
            assert!(text.contains("[Ch]"), "{text}");
            assert!(text.contains("[H]"), "{text}");
        }
        for svg in [circuit.to_svg(), dag.to_svg()] {
            assert_eq!(
                svg.matches("<line ").count(),
                4,
                "three wires and one joint channel"
            );
        }
        assert!(circuit.to_svg().contains(">t0</text>"));
        assert!(dag.to_svg().contains(">l0</text>"));
    }

    #[test]
    fn u2q_preserves_operand_order_and_resets_indices_for_each_batch_instance() {
        let angles = [
            [Angle64::try_from_radians(0.1).unwrap(); 3],
            [Angle64::try_from_radians(0.2).unwrap(); 3],
        ];
        let gate = |pairs: &[(usize, usize)]| Gate::u2q(angles, angles[0], angles, pairs);
        let render = |gate| {
            let mut circuit = crate::TickCircuit::new();
            circuit.tick();
            circuit.get_tick_mut(0).unwrap().add_gate(gate);
            circuit.to_ascii()
        };
        let forward = render(gate(&[(0, 2)]));
        let reverse = render(gate(&[(2, 0)]));
        assert_ne!(forward, reverse);
        assert!(
            forward
                .lines()
                .find(|line| line.starts_with("q0:"))
                .unwrap()
                .contains(")[0]")
        );
        assert!(
            reverse
                .lines()
                .find(|line| line.starts_with("q0:"))
                .unwrap()
                .contains(")[1]")
        );
        let batched = render(gate(&[(0, 2), (1, 3)]));
        assert_eq!(batched.matches(")[0]").count(), 2);
        assert_eq!(batched.matches(")[1]").count(), 2);
        assert!(!batched.contains(")[2]"));
    }

    #[test]
    fn tiny_nonzero_angles_do_not_round_to_zero_or_a_full_turn() {
        for angle in [Angle64::try_from_radians(1e-7).unwrap(), Angle64::new(1)] {
            for unit in [AngleUnit::Radians, AngleUnit::Turns] {
                let positive = format_angle(angle, unit);
                let negative = format_angle(-angle, unit);
                assert!(positive.contains('e'), "{positive}");
                assert!(
                    negative.starts_with('-') && negative.contains('e'),
                    "{negative}"
                );
                assert!(positive.parse::<f64>().unwrap() > 0.0);
                assert!(negative.parse::<f64>().unwrap() < 0.0);
                assert_eq!(format_angle(Angle64::ZERO, unit), "0");
            }
        }
        assert_eq!(
            format_angle_radians(Angle64::try_from_radians(0.123_456).unwrap()),
            "0.1235"
        );
        assert_eq!(
            format_angle_turns(Angle64::try_from_turns(0.123_456).unwrap()),
            "0.123456"
        );
    }

    #[test]
    fn diagonal_gate_layout_covers_every_operand() {
        for gt in [GateType::CS, GateType::CSdg, GateType::CCZ] {
            let qubits: Vec<_> = (0..gt.quantum_arity()).map(|q| QubitId(q * 2)).collect();
            let rows = qubits
                .iter()
                .enumerate()
                .map(|(row, &q)| (q, row * 2))
                .collect();
            let gate = Gate::simple(gt, qubits.clone());
            let layout = decompose_gate(&gate, &rows, 5, &DiagramStyle::default(), None);
            for row in (0..gt.quantum_arity()).map(|q| q * 2) {
                assert!(
                    layout
                        .cells
                        .iter()
                        .any(|(r, cell, _)| *r == row && !matches!(cell, DiagramCell::Crossing))
                );
            }
            assert_eq!(gate_color(gt), CellColor::ZAxis);
            assert_eq!(gate_symbol(gt), gt.to_string());
        }
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
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.h(&[1]);

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
    fn batched_two_qubit_gates_keep_every_connector_and_target() {
        for pairs in [&[(0, 1), (2, 3)][..], &[(0, 2), (1, 3)][..]] {
            let mut circuit = crate::TickCircuit::new();
            circuit.tick().cx(pairs);
            let svg = circuit.to_svg();
            assert_eq!(
                svg.matches("<line ").count(),
                6,
                "four wires and two connectors"
            );
            assert_eq!(
                svg.matches(">X</text>").count(),
                2,
                "both targets must survive"
            );
            let text = circuit.to_ascii();
            assert_eq!(text.matches("[X]").count(), 2);
        }
    }

    #[test]
    fn batched_three_qubit_gates_keep_every_connector() {
        let mut circuit = crate::TickCircuit::new();
        circuit.tick().ccx(&[(0, 1, 2), (3, 4, 5)]);
        assert_eq!(circuit.to_svg().matches("<line ").count(), 8);
    }

    #[test]
    fn empty_ticks_keep_their_positions_in_every_wire_format() {
        let mut circuit = crate::TickCircuit::new();
        circuit.tick();
        circuit.tick().h(&[0]);
        circuit.tick();
        circuit.tick();
        for output in [circuit.to_ascii(), circuit.to_svg(), circuit.to_tikz()] {
            for label in ["t0", "t2", "t3"] {
                assert!(output.contains(label), "missing {label} in {output}");
            }
        }
    }

    #[test]
    fn dag_split_groups_are_dependency_layers() {
        let mut circuit = crate::TickCircuit::new();
        circuit.tick().cx(&[(0, 2), (1, 3)]);
        let dag = crate::DagCircuit::try_from(&circuit).unwrap();
        assert!(dag.to_svg().contains(">l0</text>"));
    }

    #[test]
    fn idle_duration_is_distinct_from_identity() {
        let mut circuit = crate::TickCircuit::new();
        circuit.tick();
        circuit
            .get_tick_mut(0)
            .unwrap()
            .add_gate(Gate::simple(GateType::I, vec![QubitId(0)]));
        circuit.tick().idle(20u64, &[0]);
        let text = circuit.to_ascii();
        assert!(text.contains("[I]"));
        assert!(text.contains("[Idle(20)]"));
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
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);

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
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
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
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
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
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.sx(&[0]);
        dag.x(&[0]);
        dag.mz(&[0]);
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
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
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
