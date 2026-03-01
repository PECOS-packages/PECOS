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

//! Shared circuit diagram rendering engine.
//!
//! Produces horizontal qubit-wire diagrams with gate columns, used by
//! [`Operator`](crate::Operator), [`TickCircuit`], and [`DagCircuit`].

use std::fmt::Write;

// ============================================================================
// Types
// ============================================================================

/// What occupies a single (row, column) position in the diagram grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiagramCell {
    /// Empty wire segment.
    Wire,
    /// A gate symbol to render on this qubit wire, with its family style.
    Gate(String, GateFamily),
    /// Control dot for a multi-qubit gate.
    Control,
    /// Vertical connector between qubits of a multi-qubit gate.
    Connector,
    /// Wire crossing: a wire passes through a vertical connector.
    Crossing,
}

/// Color category for a diagram cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CellColor {
    /// No special color (default terminal color).
    #[default]
    None,
    /// Single-qubit gate (blue).
    SingleQubit,
    /// Multi-qubit gate (green).
    MultiQubit,
    /// Measurement (yellow).
    Measurement,
    /// Preparation / allocation (cyan).
    Preparation,
    /// Control dot (bold green).
    ControlDot,
}

impl CellColor {
    /// SVG/DOT fill color (light tint for gates, solid for controls).
    #[must_use]
    pub fn hex_fill(self) -> &'static str {
        match self {
            Self::None => "#FFFFFF",
            Self::SingleQubit => "#A8C8F0",
            Self::MultiQubit => "#A8E0A8",
            Self::Measurement => "#F0E0A0",
            Self::Preparation => "#A0E8E8",
            Self::ControlDot => "#2D8A2D",
        }
    }

    /// SVG/TikZ border/stroke color.
    #[must_use]
    pub fn hex_stroke(self) -> &'static str {
        match self {
            Self::None => "#888888",
            Self::SingleQubit => "#2255AA",
            Self::MultiQubit => "#226622",
            Self::Measurement => "#AA8800",
            Self::Preparation => "#008888",
            Self::ControlDot => "#1A5A1A",
        }
    }

    /// Text color inside gates (SVG/TikZ).
    #[must_use]
    pub fn hex_text(self) -> &'static str {
        match self {
            Self::None => "#333333",
            Self::SingleQubit => "#1A3A7A",
            Self::MultiQubit => "#1A4A1A",
            Self::Measurement => "#6A5500",
            Self::Preparation => "#005A5A",
            Self::ControlDot => "#FFFFFF",
        }
    }

    /// Short name for `\definecolor` in `TikZ`.
    #[must_use]
    pub fn tikz_name(self) -> &'static str {
        match self {
            Self::None => "cellNone",
            Self::SingleQubit => "cellSQ",
            Self::MultiQubit => "cellMQ",
            Self::Measurement => "cellMeas",
            Self::Preparation => "cellPrep",
            Self::ControlDot => "cellCtrl",
        }
    }
}

/// Gate family classification for visual bracket/stroke styling.
///
/// This provides a second visual dimension (shape/stroke) orthogonal to the
/// existing color dimension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GateFamily {
    /// Default bracket style `[T]`, solid stroke.
    #[default]
    Default,
    /// Pauli gates `(X)`, solid stroke.
    Pauli,
    /// S-like gates `[SZ]`, dashed stroke.
    SLike,
    /// Hadamard-like gates `<H>`, dotted stroke.
    HLike,
    /// F-like composites `{F}`, dash-dot stroke (reserved).
    FLike,
    /// Measurement gates `|MZ)`, solid stroke.
    Measurement,
    /// Preparation gates `(PZ|`, solid stroke.
    Preparation,
}

impl GateFamily {
    /// Opening bracket for text rendering.
    #[must_use]
    pub fn open_bracket(self) -> &'static str {
        match self {
            Self::Default | Self::SLike => "[",
            Self::Pauli | Self::Preparation => "(",
            Self::HLike => "<",
            Self::FLike => "{",
            Self::Measurement => "|",
        }
    }

    /// Closing bracket for text rendering.
    #[must_use]
    pub fn close_bracket(self) -> &'static str {
        match self {
            Self::Default | Self::SLike => "]",
            Self::Pauli | Self::Measurement => ")",
            Self::HLike => ">",
            Self::FLike => "}",
            Self::Preparation => "|",
        }
    }

    /// SVG `stroke-dasharray` value. Empty string means solid.
    #[must_use]
    pub fn svg_dasharray(self) -> &'static str {
        match self {
            Self::Default | Self::Pauli | Self::Measurement | Self::Preparation => "",
            Self::SLike => "4,3",
            Self::HLike => "2,2",
            Self::FLike => "6,2,2,2",
        }
    }

    /// `TikZ` dash pattern name. Empty string means solid.
    #[must_use]
    pub fn tikz_dash(self) -> &'static str {
        match self {
            Self::Default | Self::Pauli | Self::Measurement | Self::Preparation => "",
            Self::SLike => "dashed",
            Self::HLike => "dotted",
            Self::FLike => "dashdotted",
        }
    }

    /// DOT/Graphviz `style` value. Empty string means default (solid).
    #[must_use]
    pub fn dot_style(self) -> &'static str {
        match self {
            Self::Default | Self::Pauli | Self::Measurement | Self::Preparation => "",
            Self::SLike | Self::FLike => "dashed",
            Self::HLike => "dotted",
        }
    }
}

/// Which character set to use for rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SymbolSet {
    /// Plain ASCII: `-`, `|`, `.`, `+`
    #[default]
    Ascii,
    /// Unicode box-drawing: `─`, `│`, `●`, `+`
    Unicode,
}

/// Options controlling diagram appearance.
#[derive(Clone, Debug)]
pub struct DiagramOptions {
    pub symbols: SymbolSet,
    pub color: bool,
}

impl DiagramOptions {
    /// Plain ASCII, no color.
    #[must_use]
    pub fn ascii() -> Self {
        Self {
            symbols: SymbolSet::Ascii,
            color: false,
        }
    }

    /// ASCII with ANSI color.
    #[must_use]
    pub fn color_ascii() -> Self {
        Self {
            symbols: SymbolSet::Ascii,
            color: true,
        }
    }

    /// Unicode box-drawing, no color.
    #[must_use]
    pub fn unicode() -> Self {
        Self {
            symbols: SymbolSet::Unicode,
            color: false,
        }
    }

    /// Unicode box-drawing with ANSI color.
    #[must_use]
    pub fn color_unicode() -> Self {
        Self {
            symbols: SymbolSet::Unicode,
            color: true,
        }
    }
}

// ============================================================================
// ANSI color codes
// ============================================================================

const ANSI_RESET: &str = "\x1b[0m";

fn ansi_code(color: CellColor) -> &'static str {
    match color {
        CellColor::None => "",
        CellColor::SingleQubit => "\x1b[34m",
        CellColor::MultiQubit => "\x1b[32m",
        CellColor::Measurement => "\x1b[33m",
        CellColor::Preparation => "\x1b[36m",
        CellColor::ControlDot => "\x1b[1;32m",
    }
}

// ============================================================================
// CircuitDiagram builder
// ============================================================================

/// A grid-based circuit diagram builder.
///
/// The diagram is organized as a grid of `columns x rows`, where each row
/// corresponds to a qubit wire and each column to a time step / layer.
pub struct CircuitDiagram {
    labels: Vec<String>,
    columns: Vec<Vec<(DiagramCell, CellColor)>>,
    current_col: usize,
}

impl CircuitDiagram {
    /// Create a new diagram for `n` qubits with default labels `q0`, `q1`, ...
    #[must_use]
    pub fn new(n: usize) -> Self {
        let labels: Vec<String> = (0..n).map(|i| format!("q{i}")).collect();
        Self {
            labels,
            columns: vec![vec![(DiagramCell::Wire, CellColor::None); n]],
            current_col: 0,
        }
    }

    /// Create a new diagram with custom labels.
    #[must_use]
    pub fn with_labels(labels: Vec<String>) -> Self {
        let n = labels.len();
        Self {
            labels,
            columns: vec![vec![(DiagramCell::Wire, CellColor::None); n]],
            current_col: 0,
        }
    }

    /// Number of qubit rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.labels.len()
    }

    fn ensure_column(&mut self) {
        while self.current_col >= self.columns.len() {
            self.columns.push(vec![
                (DiagramCell::Wire, CellColor::None);
                self.num_rows()
            ]);
        }
    }

    /// Set a cell at the given row in the current column.
    pub fn set_cell(&mut self, row: usize, cell: DiagramCell, color: CellColor) {
        self.ensure_column();
        if row < self.num_rows() {
            self.columns[self.current_col][row] = (cell, color);
        }
    }

    /// Place a gate symbol on a row with a family bracket/stroke style.
    pub fn add_gate(&mut self, row: usize, name: &str, color: CellColor, family: GateFamily) {
        self.set_cell(row, DiagramCell::Gate(name.to_string(), family), color);
    }

    /// Place a control dot on a row.
    pub fn add_control(&mut self, row: usize) {
        self.set_cell(row, DiagramCell::Control, CellColor::ControlDot);
    }

    /// Fill vertical connectors/crossings between `top` and `bottom` (exclusive).
    ///
    /// Rows that are qubit wires get `Crossing`; other rows get `Connector`.
    /// Since every row in a `CircuitDiagram` is a qubit wire, this always
    /// places `Crossing` cells. The `color` is applied to all intermediate cells.
    pub fn connect_vertical(&mut self, top: usize, bottom: usize, color: CellColor) {
        self.ensure_column();
        let (lo, hi) = if top < bottom {
            (top, bottom)
        } else {
            (bottom, top)
        };
        for row in (lo + 1)..hi {
            if row < self.num_rows() {
                // All rows in CircuitDiagram are qubit wires -> Crossing.
                self.columns[self.current_col][row] = (DiagramCell::Crossing, color);
            }
        }
    }

    /// Advance to the next column.
    pub fn advance(&mut self) {
        self.current_col += 1;
    }

    /// Render the diagram to a string.
    ///
    /// If `header` is non-empty, it is printed as the first line followed by
    /// a blank line.
    #[must_use]
    pub fn render(&self, header: &str, options: &DiagramOptions) -> String {
        let num_rows = self.num_rows();
        if num_rows == 0 {
            return if header.is_empty() {
                String::new()
            } else {
                format!("{header}\n")
            };
        }

        // Strip trailing all-Wire columns.
        let num_cols = self.effective_columns();
        if num_cols == 0 {
            return if header.is_empty() {
                String::new()
            } else {
                format!("{header}\n")
            };
        }

        // Column widths (based on widest cell content).
        let col_widths: Vec<usize> = (0..num_cols)
            .map(|c| {
                self.columns[c]
                    .iter()
                    .map(|(cell, _)| cell_content_width(cell))
                    .max()
                    .unwrap_or(1)
            })
            .collect();

        let label_width = self.labels.iter().map(String::len).max().unwrap_or(2);

        let wire_char = match options.symbols {
            SymbolSet::Ascii => '-',
            SymbolSet::Unicode => '\u{2500}', // ─
        };

        let mut out = String::new();
        if !header.is_empty() {
            writeln!(out, "{header}").unwrap();
            writeln!(out).unwrap();
        }

        for row in 0..num_rows {
            write!(out, "{:>label_width$}: ", self.labels[row]).unwrap();

            for (col_idx, &width) in col_widths.iter().enumerate() {
                let (ref cell, color) = self.columns[col_idx][row];
                let rendered = render_cell(cell, width, wire_char, options);

                if options.color && !matches!(cell, DiagramCell::Wire) {
                    let code = ansi_code(color);
                    if code.is_empty() {
                        write!(out, "{wire_char}{rendered}{wire_char}").unwrap();
                    } else {
                        write!(out, "{wire_char}{code}{rendered}{ANSI_RESET}{wire_char}").unwrap();
                    }
                } else {
                    write!(out, "{wire_char}{rendered}{wire_char}").unwrap();
                }
            }

            writeln!(out).unwrap();

            // Connector row between qubit wires.
            if row + 1 < num_rows {
                let connector_line =
                    self.render_connector_row(row, num_cols, &col_widths, options);
                if let Some(line) = connector_line {
                    writeln!(out, "{}", line.trim_end()).unwrap();
                }
            }
        }

        out
    }

    // ========================================================================
    // SVG rendering
    // ========================================================================

    /// Render the diagram as a standalone SVG string.
    ///
    /// If `header` is non-empty it is rendered as a `<text>` title at the top.
    #[must_use]
    pub fn render_svg(&self, header: &str) -> String {
        const ROW_SPACING: f64 = 40.0;
        const COL_SPACING: f64 = 60.0;
        const GATE_H: f64 = 24.0;
        const LABEL_MARGIN: f64 = 50.0;
        const CTRL_RADIUS: f64 = 5.0;
        const FONT_SIZE: f64 = 13.0;
        const GATE_RX: f64 = 4.0;

        let num_rows = self.num_rows();
        let num_cols = self.effective_columns();

        if num_rows == 0 || num_cols == 0 {
            return if header.is_empty() {
                "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string()
            } else {
                format!(
                    "<svg xmlns=\"http://www.w3.org/2000/svg\">\
                     <text x=\"10\" y=\"20\" font-family=\"monospace\" font-size=\"14\">{header}</text>\
                     </svg>"
                )
            };
        }

        // Column widths in characters (used to compute gate box widths).
        let col_widths: Vec<usize> = (0..num_cols)
            .map(|c| {
                self.columns[c]
                    .iter()
                    .map(|(cell, _)| cell_content_width(cell))
                    .max()
                    .unwrap_or(1)
            })
            .collect();

        let header_offset: f64 = if header.is_empty() { 0.0 } else { 30.0 };
        let svg_width =
            LABEL_MARGIN + (num_cols as f64) * COL_SPACING + COL_SPACING * 0.5 + 20.0;
        let svg_height = header_offset + (num_rows as f64) * ROW_SPACING + ROW_SPACING * 0.5;

        let mut out = String::new();
        writeln!(
            out,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{svg_width}\" height=\"{svg_height}\">"
        )
        .unwrap();
        writeln!(
            out,
            "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
        )
        .unwrap();

        if !header.is_empty() {
            writeln!(
                out,
                "<text x=\"10\" y=\"20\" font-family=\"monospace\" font-size=\"14\" fill=\"#333\">{header}</text>"
            )
            .unwrap();
        }

        // Qubit labels and wires.
        for row in 0..num_rows {
            let y = header_offset + ROW_SPACING * (row as f64 + 0.5);
            // Label
            writeln!(
                out,
                "<text x=\"{x}\" y=\"{ty}\" font-family=\"monospace\" font-size=\"{FONT_SIZE}\" \
                 text-anchor=\"end\" dominant-baseline=\"middle\" fill=\"#333\">{label}</text>",
                x = LABEL_MARGIN - 6.0,
                ty = y,
                label = self.labels[row],
            )
            .unwrap();
            // Wire
            let x_end = LABEL_MARGIN + (num_cols as f64) * COL_SPACING;
            writeln!(
                out,
                "<line x1=\"{LABEL_MARGIN}\" y1=\"{y}\" x2=\"{x_end}\" y2=\"{y}\" stroke=\"#CCCCCC\" stroke-width=\"1\"/>",
            )
            .unwrap();
        }

        // Gate cells.
        for (col_idx, col_width) in col_widths.iter().enumerate() {
            let cx = LABEL_MARGIN + COL_SPACING * (col_idx as f64 + 0.5);
            let gate_w = (*col_width as f64) * 9.0 + 8.0;

            for row in 0..num_rows {
                let cy = header_offset + ROW_SPACING * (row as f64 + 0.5);
                let (ref cell, color) = self.columns[col_idx][row];

                match cell {
                    DiagramCell::Wire => {}
                    DiagramCell::Gate(s, family) => {
                        let dash = family.svg_dasharray();
                        let dash_attr = if dash.is_empty() {
                            String::new()
                        } else {
                            format!(" stroke-dasharray=\"{dash}\"")
                        };
                        writeln!(
                            out,
                            "<rect x=\"{rx}\" y=\"{ry}\" width=\"{gate_w}\" height=\"{GATE_H}\" \
                             rx=\"{GATE_RX}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"{dash_attr}/>",
                            rx = cx - gate_w / 2.0,
                            ry = cy - GATE_H / 2.0,
                            fill = color.hex_fill(),
                            stroke = color.hex_stroke(),
                        )
                        .unwrap();
                        writeln!(
                            out,
                            "<text x=\"{cx}\" y=\"{cy}\" font-family=\"monospace\" font-size=\"{FONT_SIZE}\" \
                             text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"{fill}\">{s}</text>",
                            fill = color.hex_text(),
                        )
                        .unwrap();
                    }
                    DiagramCell::Control => {
                        writeln!(
                            out,
                            "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{CTRL_RADIUS}\" \
                             fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1\"/>",
                            fill = color.hex_fill(),
                            stroke = color.hex_stroke(),
                        )
                        .unwrap();
                    }
                    DiagramCell::Crossing => {
                        // Vertical line segment through this row (rendered below
                        // as a connector) + horizontal wire already drawn.
                        writeln!(
                            out,
                            "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"2\" fill=\"#888\"/>",
                        )
                        .unwrap();
                    }
                    DiagramCell::Connector => {
                        // Pure vertical connector (no qubit wire) -- small dot.
                        writeln!(
                            out,
                            "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"2\" fill=\"#888\"/>",
                        )
                        .unwrap();
                    }
                }
            }

            // Vertical connectors between multi-qubit cells in this column.
            let mut top: Option<usize> = None;
            let mut bottom: Option<usize> = None;
            for row in 0..num_rows {
                let (ref cell, color) = self.columns[col_idx][row];
                let is_part = !matches!(cell, DiagramCell::Wire) && is_multi_color(color);
                if is_part {
                    if top.is_none() {
                        top = Some(row);
                    }
                    bottom = Some(row);
                }
            }
            if let (Some(t), Some(b)) = (top, bottom)
                && t != b
            {
                let y1 = header_offset + ROW_SPACING * (t as f64 + 0.5);
                let y2 = header_offset + ROW_SPACING * (b as f64 + 0.5);
                writeln!(
                    out,
                    "<line x1=\"{cx}\" y1=\"{y1}\" x2=\"{cx}\" y2=\"{y2}\" \
                     stroke=\"{stroke}\" stroke-width=\"1.5\"/>",
                    stroke = CellColor::ControlDot.hex_stroke(),
                )
                .unwrap();
            }
        }

        writeln!(out, "</svg>").unwrap();
        out
    }

    // ========================================================================
    // TikZ rendering
    // ========================================================================

    /// Render the diagram as a `TikZ` `tikzpicture` environment.
    ///
    /// Requires only `\usepackage{tikz}` -- no quantikz. If `header` is
    /// non-empty it is emitted as a `TikZ` comment.
    #[must_use]
    pub fn render_tikz(&self, header: &str) -> String {
        const ROW_STEP: f64 = 0.8;
        const COL_STEP: f64 = 1.2;
        const GATE_W: f64 = 0.7;
        const GATE_H: f64 = 0.5;
        const CTRL_R: f64 = 0.08;

        let num_rows = self.num_rows();
        let num_cols = self.effective_columns();

        let mut out = String::new();

        if !header.is_empty() {
            writeln!(out, "% {header}").unwrap();
        }

        writeln!(out, "\\begin{{tikzpicture}}").unwrap();

        // Color definitions.
        for &c in &[
            CellColor::None,
            CellColor::SingleQubit,
            CellColor::MultiQubit,
            CellColor::Measurement,
            CellColor::Preparation,
            CellColor::ControlDot,
        ] {
            let name = c.tikz_name();
            writeln!(
                out,
                "  \\definecolor{{{name}Fill}}{{HTML}}{{{fill}}}",
                fill = &c.hex_fill()[1..], // strip #
            )
            .unwrap();
            writeln!(
                out,
                "  \\definecolor{{{name}Stroke}}{{HTML}}{{{stroke}}}",
                stroke = &c.hex_stroke()[1..],
            )
            .unwrap();
            writeln!(
                out,
                "  \\definecolor{{{name}Text}}{{HTML}}{{{text}}}",
                text = &c.hex_text()[1..],
            )
            .unwrap();
        }

        // Styles.
        writeln!(
            out,
            "  \\tikzstyle{{gate}}=[draw, rounded corners=2pt, minimum width={GATE_W}cm, \
             minimum height={GATE_H}cm, inner sep=1pt, font=\\footnotesize\\ttfamily]"
        )
        .unwrap();
        writeln!(
            out,
            "  \\tikzstyle{{ctrl}}=[circle, fill, inner sep=0pt, minimum size={r}cm]",
            r = CTRL_R * 2.0,
        )
        .unwrap();

        if num_rows == 0 || num_cols == 0 {
            writeln!(out, "\\end{{tikzpicture}}").unwrap();
            return out;
        }

        // Wires and labels.
        for row in 0..num_rows {
            let y = -(row as f64) * ROW_STEP;
            let x_start = -0.5;
            let x_end = (num_cols as f64) * COL_STEP + 0.3;
            writeln!(
                out,
                "  \\draw[gray] ({x_start:.2},{y:.2}) -- ({x_end:.2},{y:.2});",
            )
            .unwrap();
            writeln!(
                out,
                "  \\node[anchor=east, font=\\footnotesize\\ttfamily] at ({lx:.2},{y:.2}) {{{label}}};",
                lx = x_start - 0.15,
                label = self.labels[row],
            )
            .unwrap();
        }

        // Gates, controls, connectors.
        for col_idx in 0..num_cols {
            let x = (col_idx as f64 + 0.5) * COL_STEP;

            for row in 0..num_rows {
                let y = -(row as f64) * ROW_STEP;
                let (ref cell, color) = self.columns[col_idx][row];
                let name = color.tikz_name();

                match cell {
                    DiagramCell::Wire => {}
                    DiagramCell::Gate(s, family) => {
                        let dash = family.tikz_dash();
                        let dash_opt = if dash.is_empty() {
                            String::new()
                        } else {
                            format!(", {dash}")
                        };
                        writeln!(
                            out,
                            "  \\node[gate, fill={name}Fill, draw={name}Stroke, text={name}Text{dash_opt}] at ({x:.2},{y:.2}) {{{s}}};",
                        )
                        .unwrap();
                    }
                    DiagramCell::Control => {
                        writeln!(
                            out,
                            "  \\node[ctrl, fill={name}Fill, draw={name}Stroke] at ({x:.2},{y:.2}) {{}};",
                        )
                        .unwrap();
                    }
                    DiagramCell::Crossing | DiagramCell::Connector => {
                        writeln!(
                            out,
                            "  \\node[circle, fill=gray, inner sep=0pt, minimum size=0.06cm] at ({x:.2},{y:.2}) {{}};",
                        )
                        .unwrap();
                    }
                }
            }

            // Vertical connector lines.
            let mut top: Option<usize> = None;
            let mut bottom: Option<usize> = None;
            for row in 0..num_rows {
                let (ref cell, color) = self.columns[col_idx][row];
                if !matches!(cell, DiagramCell::Wire) && is_multi_color(color) {
                    if top.is_none() {
                        top = Some(row);
                    }
                    bottom = Some(row);
                }
            }
            if let (Some(t), Some(b)) = (top, bottom)
                && t != b
            {
                let y1 = -(t as f64) * ROW_STEP;
                let y2 = -(b as f64) * ROW_STEP;
                writeln!(
                    out,
                    "  \\draw[cellCtrlStroke] ({x:.2},{y1:.2}) -- ({x:.2},{y2:.2});",
                )
                .unwrap();
            }
        }

        writeln!(out, "\\end{{tikzpicture}}").unwrap();
        out
    }

    // ========================================================================
    // DOT / Graphviz rendering
    // ========================================================================

    /// Render the diagram as a Graphviz DOT `digraph` with `rankdir=LR`.
    ///
    /// If `header` is non-empty it is set as the graph `label`.
    #[must_use]
    pub fn render_dot(&self, header: &str) -> String {
        let num_rows = self.num_rows();
        let num_cols = self.effective_columns();

        let mut out = String::new();
        writeln!(out, "digraph circuit {{").unwrap();
        writeln!(out, "  rankdir=LR;").unwrap();
        writeln!(out, "  node [fontname=\"Courier\", fontsize=11];").unwrap();
        writeln!(out, "  edge [arrowhead=none];").unwrap();

        if !header.is_empty() {
            writeln!(out, "  label=\"{header}\";").unwrap();
            writeln!(out, "  labelloc=t;").unwrap();
        }

        if num_rows == 0 || num_cols == 0 {
            writeln!(out, "}}").unwrap();
            return out;
        }

        // Node IDs: "r{row}c{col}" for gate cells, "r{row}_in"/"r{row}_out" for endpoints.

        // Input label nodes.
        writeln!(out, "  // Input labels").unwrap();
        writeln!(out, "  {{ rank=same;").unwrap();
        for row in 0..num_rows {
            writeln!(
                out,
                "    r{row}_in [label=\"{label}\", shape=plaintext];",
                label = self.labels[row],
            )
            .unwrap();
        }
        writeln!(out, "  }}").unwrap();

        // Output nodes (invisible).
        writeln!(out, "  // Output nodes").unwrap();
        writeln!(out, "  {{ rank=same;").unwrap();
        for row in 0..num_rows {
            writeln!(
                out,
                "    r{row}_out [label=\"\", shape=none, width=0, height=0];",
            )
            .unwrap();
        }
        writeln!(out, "  }}").unwrap();

        // Gate columns.
        for col_idx in 0..num_cols {
            writeln!(out, "  // Column {col_idx}").unwrap();
            writeln!(out, "  {{ rank=same;").unwrap();
            for row in 0..num_rows {
                let (ref cell, color) = self.columns[col_idx][row];
                let node_id = format!("r{row}c{col_idx}");

                match cell {
                    DiagramCell::Wire => {
                        writeln!(
                            out,
                            "    {node_id} [label=\"\", shape=point, width=0.01];",
                        )
                        .unwrap();
                    }
                    DiagramCell::Gate(s, family) => {
                        let dot_style = family.dot_style();
                        let style_val = if dot_style.is_empty() {
                            "filled".to_string()
                        } else {
                            format!("\"filled,{dot_style}\"")
                        };
                        writeln!(
                            out,
                            "    {node_id} [label=\"{s}\", shape=box, style={style_val}, \
                             fillcolor=\"{fill}\", color=\"{stroke}\", fontcolor=\"{text}\"];",
                            fill = color.hex_fill(),
                            stroke = color.hex_stroke(),
                            text = color.hex_text(),
                        )
                        .unwrap();
                    }
                    DiagramCell::Control => {
                        writeln!(
                            out,
                            "    {node_id} [label=\"\", shape=point, width=0.12, \
                             style=filled, fillcolor=\"{fill}\"];",
                            fill = color.hex_fill(),
                        )
                        .unwrap();
                    }
                    DiagramCell::Crossing | DiagramCell::Connector => {
                        writeln!(
                            out,
                            "    {node_id} [label=\"\", shape=point, width=0.05];",
                        )
                        .unwrap();
                    }
                }
            }
            writeln!(out, "  }}").unwrap();
        }

        // Wire edges.
        writeln!(out, "  // Wires").unwrap();
        for row in 0..num_rows {
            let mut prev = format!("r{row}_in");
            for col_idx in 0..num_cols {
                let cur = format!("r{row}c{col_idx}");
                writeln!(out, "  {prev} -> {cur};").unwrap();
                prev = cur;
            }
            writeln!(out, "  {prev} -> r{row}_out;").unwrap();
        }

        // Vertical connector edges.
        writeln!(out, "  // Vertical connectors").unwrap();
        for col_idx in 0..num_cols {
            let mut top: Option<usize> = None;
            let mut bottom: Option<usize> = None;
            for row in 0..num_rows {
                let (ref cell, color) = self.columns[col_idx][row];
                if !matches!(cell, DiagramCell::Wire) && is_multi_color(color) {
                    if top.is_none() {
                        top = Some(row);
                    }
                    bottom = Some(row);
                }
            }
            if let (Some(t), Some(b)) = (top, bottom)
                && t != b
            {
                // Connect consecutive multi-qubit rows.
                let mut prev_row = t;
                for row in (t + 1)..=b {
                    let (ref cell, color) = self.columns[col_idx][row];
                    if !matches!(cell, DiagramCell::Wire) && is_multi_color(color) {
                        writeln!(
                            out,
                            "  r{prev_row}c{col_idx} -> r{row}c{col_idx} [style=dashed, dir=none, constraint=false];",
                        )
                        .unwrap();
                        prev_row = row;
                    }
                }
            }
        }

        writeln!(out, "}}").unwrap();
        out
    }

    /// Count effective columns (strip trailing all-Wire columns).
    fn effective_columns(&self) -> usize {
        let mut n = self.columns.len();
        while n > 0 {
            let all_wire = self.columns[n - 1]
                .iter()
                .all(|(cell, _)| matches!(cell, DiagramCell::Wire));
            if all_wire {
                n -= 1;
            } else {
                break;
            }
        }
        n
    }

    /// Render the connector row between `row` and `row + 1`.
    /// Returns `None` if no connectors are needed.
    fn render_connector_row(
        &self,
        row: usize,
        num_cols: usize,
        col_widths: &[usize],
        options: &DiagramOptions,
    ) -> Option<String> {
        let label_width = self.labels.iter().map(String::len).max().unwrap_or(2);
        let mut line = String::new();
        write!(line, "{:>width$}  ", "", width = label_width).unwrap();
        let mut has_connector = false;

        for (col_idx, &width) in col_widths.iter().enumerate() {
            if col_idx >= num_cols {
                break;
            }
            let (ref cell, color) = self.columns[col_idx][row];
            let (ref next_cell, next_color) = self.columns[col_idx][row + 1];

            // Show a vertical connector when both this row and the next have
            // non-Wire cells that are part of a multi-qubit gate (same color
            // category, both colored).
            let show = !matches!(cell, DiagramCell::Wire)
                && !matches!(next_cell, DiagramCell::Wire)
                && is_multi_color(color)
                && is_multi_color(next_color);

            if show {
                has_connector = true;
                let pad_total = width.saturating_sub(1);
                let pad_left = pad_total / 2;
                let pad_right = pad_total - pad_left;
                let left: String = std::iter::repeat_n(' ', pad_left).collect();
                let right: String = std::iter::repeat_n(' ', pad_right).collect();
                if options.color {
                    write!(
                        line,
                        " {left}{}{ANSI_RESET}{right} ",
                        format_args!("{}|", ansi_code(CellColor::ControlDot)),
                    )
                    .unwrap();
                } else {
                    write!(line, " {left}|{right} ").unwrap();
                }
            } else {
                let spaces: String = std::iter::repeat_n(' ', width + 2).collect();
                write!(line, "{spaces}").unwrap();
            }
        }

        if has_connector { Some(line) } else { None }
    }
}

// ============================================================================
// Rendering helpers
// ============================================================================

/// Content width of a cell (before padding).
fn cell_content_width(cell: &DiagramCell) -> usize {
    match cell {
        DiagramCell::Gate(s, _) => s.len() + 2, // +2 for brackets
        DiagramCell::Wire | DiagramCell::Control | DiagramCell::Crossing | DiagramCell::Connector => 1,
    }
}

/// Render a single cell into the given column width.
fn render_cell(cell: &DiagramCell, width: usize, wire_char: char, options: &DiagramOptions) -> String {
    match cell {
        DiagramCell::Wire => {
            std::iter::repeat_n(wire_char, width).collect()
        }
        DiagramCell::Gate(s, family) => {
            let bracketed = format!("{}{s}{}", family.open_bracket(), family.close_bracket());
            pad_center(&bracketed, width, wire_char)
        }
        DiagramCell::Control => {
            let dot = match options.symbols {
                SymbolSet::Ascii => ".",
                SymbolSet::Unicode => "\u{25CF}", // ●
            };
            pad_center(dot, width, wire_char)
        }
        DiagramCell::Crossing => {
            pad_center("+", width, wire_char)
        }
        DiagramCell::Connector => {
            // Connector on a qubit wire row -- treat as crossing.
            pad_center("|", width, wire_char)
        }
    }
}

/// Center `s` within `width` characters, padding with `pad_char`.
fn pad_center(s: &str, width: usize, pad_char: char) -> String {
    let content_width = s.chars().count();
    let pad_total = width.saturating_sub(content_width);
    let pad_left = pad_total / 2;
    let pad_right = pad_total - pad_left;
    let left: String = std::iter::repeat_n(pad_char, pad_left).collect();
    let right: String = std::iter::repeat_n(pad_char, pad_right).collect();
    format!("{left}{s}{right}")
}

/// Whether a color indicates a multi-qubit gate context.
fn is_multi_color(color: CellColor) -> bool {
    matches!(
        color,
        CellColor::MultiQubit | CellColor::ControlDot
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_diagram() {
        let d = CircuitDiagram::new(0);
        let out = d.render("test", &DiagramOptions::ascii());
        assert_eq!(out, "test\n");
    }

    #[test]
    fn single_gate_ascii() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("[H]"));
        // q1 should be just wire
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(!q1_line.contains("["));
    }

    #[test]
    fn single_gate_unicode() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);
        let out = d.render("", &DiagramOptions::unicode());
        assert!(out.contains("[H]"));
        assert!(out.contains('\u{2500}')); // ─
        assert!(!out.contains('-'));
    }

    #[test]
    fn control_dot_ascii_vs_unicode() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::MultiQubit, GateFamily::Default);
        d.connect_vertical(0, 1, CellColor::MultiQubit);

        let ascii = d.render("", &DiagramOptions::ascii());
        assert!(ascii.contains('.'));

        let unicode = d.render("", &DiagramOptions::unicode());
        assert!(unicode.contains('\u{25CF}')); // ●
    }

    #[test]
    fn color_output_contains_ansi() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);

        let plain = d.render("", &DiagramOptions::ascii());
        let color = d.render("", &DiagramOptions::color_ascii());

        assert!(!plain.contains("\x1b["));
        assert!(color.contains("\x1b[34m")); // blue
        assert!(color.contains(ANSI_RESET));
    }

    #[test]
    fn crossing_between_qubits() {
        let mut d = CircuitDiagram::new(3);
        d.add_control(0);
        d.add_gate(2, "X", CellColor::MultiQubit, GateFamily::Default);
        d.connect_vertical(0, 2, CellColor::MultiQubit);

        let out = d.render("", &DiagramOptions::ascii());
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q1_line.contains('+'));
    }

    #[test]
    fn multi_column_advance() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);
        d.advance();
        d.add_gate(1, "X", CellColor::SingleQubit, GateFamily::Default);

        let out = d.render("", &DiagramOptions::ascii());
        let q0 = out.lines().find(|l| l.starts_with("q0:")).unwrap();
        let q1 = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q0.contains("[H]"));
        assert!(!q0.contains("[X]"));
        assert!(q1.contains("[X]"));
        assert!(!q1.contains("[H]"));
    }

    #[test]
    fn header_is_printed() {
        let d = CircuitDiagram::new(1);
        // Single wire column is all-Wire, so effective_columns == 0
        let out = d.render("My Header", &DiagramOptions::ascii());
        assert!(out.starts_with("My Header\n"));
    }

    #[test]
    fn connector_row_between_multi_qubit() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::MultiQubit, GateFamily::Default);

        let out = d.render("", &DiagramOptions::ascii());
        // Should have a | connector between q0 and q1
        assert!(out.contains('|'));
    }

    #[test]
    fn lines_have_equal_length() {
        let mut d = CircuitDiagram::new(3);
        d.add_gate(0, "SX", CellColor::SingleQubit, GateFamily::Default);
        d.advance();
        d.add_control(0);
        d.add_gate(2, "X", CellColor::MultiQubit, GateFamily::Default);
        d.connect_vertical(0, 2, CellColor::MultiQubit);

        let out = d.render("", &DiagramOptions::ascii());
        let qubit_lines: Vec<&str> = out.lines().filter(|l| l.starts_with('q')).collect();
        assert!(qubit_lines.len() >= 2);
        let len0 = qubit_lines[0].len();
        for line in &qubit_lines {
            assert_eq!(line.len(), len0, "qubit lines should have equal length");
        }
    }

    // ====================== SVG tests ======================

    #[test]
    fn svg_empty_diagram() {
        let d = CircuitDiagram::new(0);
        let out = d.render_svg("");
        assert!(out.contains("<svg"));
        assert!(out.contains("</svg>"));
    }

    #[test]
    fn svg_single_gate() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);
        let out = d.render_svg("");
        assert!(out.contains("<svg"));
        assert!(out.contains("<rect"));
        assert!(out.contains(">H</text>"));
        assert!(out.contains("q0</text>"));
        assert!(out.contains("q1</text>"));
        assert!(out.contains("#A8C8F0")); // SingleQubit fill
    }

    #[test]
    fn svg_control_and_connector() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::MultiQubit, GateFamily::Default);
        let out = d.render_svg("");
        assert!(out.contains("<circle")); // control dot
        assert!(out.contains("<rect")); // gate box
        assert!(out.contains("<line")); // vertical connector
    }

    #[test]
    fn svg_header() {
        let d = CircuitDiagram::new(0);
        let out = d.render_svg("My Circuit");
        assert!(out.contains("My Circuit"));
    }

    // ====================== TikZ tests ======================

    #[test]
    fn tikz_empty_diagram() {
        let d = CircuitDiagram::new(0);
        let out = d.render_tikz("");
        assert!(out.contains("\\begin{tikzpicture}"));
        assert!(out.contains("\\end{tikzpicture}"));
    }

    #[test]
    fn tikz_single_gate() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);
        let out = d.render_tikz("");
        assert!(out.contains("\\begin{tikzpicture}"));
        assert!(out.contains("\\end{tikzpicture}"));
        assert!(out.contains("\\definecolor"));
        assert!(out.contains("cellSQFill"));
        assert!(out.contains("\\node[gate"));
        assert!(out.contains("{H}"));
        assert!(out.contains("\\draw[gray]")); // wires
    }

    #[test]
    fn tikz_control_and_connector() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::MultiQubit, GateFamily::Default);
        let out = d.render_tikz("");
        assert!(out.contains("\\node[ctrl"));
        assert!(out.contains("\\node[gate"));
        assert!(out.contains("cellCtrlStroke")); // vertical connector
    }

    #[test]
    fn tikz_header_as_comment() {
        let d = CircuitDiagram::new(0);
        let out = d.render_tikz("My Circuit");
        assert!(out.contains("% My Circuit"));
    }

    // ====================== DOT tests ======================

    #[test]
    fn dot_empty_diagram() {
        let d = CircuitDiagram::new(0);
        let out = d.render_dot("");
        assert!(out.contains("digraph circuit"));
        assert!(out.contains("rankdir=LR"));
    }

    #[test]
    fn dot_single_gate() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::Default);
        let out = d.render_dot("");
        assert!(out.contains("digraph circuit"));
        assert!(out.contains("shape=box"));
        assert!(out.contains("label=\"H\""));
        assert!(out.contains("r0_in"));
        assert!(out.contains("r0_out"));
        assert!(out.contains("r1_in"));
        assert!(out.contains("#A8C8F0")); // SingleQubit fill
    }

    #[test]
    fn dot_control_and_connector() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::MultiQubit, GateFamily::Default);
        let out = d.render_dot("");
        assert!(out.contains("shape=point, width=0.12")); // control dot
        assert!(out.contains("shape=box")); // gate
        assert!(out.contains("style=dashed")); // vertical connector
    }

    #[test]
    fn dot_header_as_label() {
        let d = CircuitDiagram::new(0);
        let out = d.render_dot("My Circuit");
        assert!(out.contains("label=\"My Circuit\""));
    }

    // ====================== Gate family bracket tests ======================

    #[test]
    fn pauli_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "X", CellColor::SingleQubit, GateFamily::Pauli);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("(X)"));
    }

    #[test]
    fn hlike_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::HLike);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("<H>"));
    }

    #[test]
    fn slike_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::SingleQubit, GateFamily::SLike);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("[SZ]"));
    }

    #[test]
    fn flike_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "F", CellColor::SingleQubit, GateFamily::FLike);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("{F}"));
    }

    #[test]
    fn measurement_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "MZ", CellColor::Measurement, GateFamily::Measurement);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("|MZ)"));
    }

    #[test]
    fn preparation_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "PZ", CellColor::Preparation, GateFamily::Preparation);
        let out = d.render("", &DiagramOptions::ascii());
        assert!(out.contains("(PZ|"));
    }

    // ====================== Gate family stroke tests ======================

    #[test]
    fn svg_slike_dasharray() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::SingleQubit, GateFamily::SLike);
        let out = d.render_svg("");
        assert!(out.contains("stroke-dasharray=\"4,3\""));
    }

    #[test]
    fn svg_hlike_dasharray() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::HLike);
        let out = d.render_svg("");
        assert!(out.contains("stroke-dasharray=\"2,2\""));
    }

    #[test]
    fn svg_default_no_dasharray() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "T", CellColor::SingleQubit, GateFamily::Default);
        let out = d.render_svg("");
        assert!(!out.contains("stroke-dasharray"));
    }

    #[test]
    fn tikz_slike_dashed() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::SingleQubit, GateFamily::SLike);
        let out = d.render_tikz("");
        assert!(out.contains(", dashed]"));
    }

    #[test]
    fn dot_hlike_dotted() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::SingleQubit, GateFamily::HLike);
        let out = d.render_dot("");
        assert!(out.contains("filled,dotted"));
    }
}
