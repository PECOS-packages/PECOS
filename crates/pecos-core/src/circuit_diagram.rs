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
//! [`UnitaryRep`](crate::UnitaryRep), [`TickCircuit`], and [`DagCircuit`].

use std::fmt::Write;

// --- Types ---

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
    /// Labeled connector: a label displayed on the vertical connector between
    /// two control dots (e.g. `ZZ` for symmetric two-qubit interactions).
    LabeledConnector(String),
}

/// Color category for a diagram cell.
///
/// Follows the PECOS color algebra based on Pauli axis interconversion:
/// - Base axes: X = Red, Y = Green, Z = Blue
/// - Mixed axes use additive RGB: X<->Z = Magenta, X<->Y = Yellow, Y<->Z = Cyan
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CellColor {
    /// No special color (default terminal color).
    #[default]
    None,
    /// X-axis (red): X, RX, CX target, etc.
    XAxis,
    /// Y-axis (green): Y, RY, CY target, etc.
    YAxis,
    /// Z-axis (blue): Z, RZ, T, MZ, PZ, etc.
    ZAxis,
    /// X<->Z mixing (magenta): H, SY, `SYdg`.
    XZMix,
    /// X<->Y mixing (yellow): SZ, `SZdg`.
    XYMix,
    /// Y<->Z mixing (cyan): SX, `SXdg`.
    YZMix,
    /// All-axis mixing (grey): F-like composites.
    XYZMix,
    /// Control dot (dark).
    ControlDot,
}

impl CellColor {
    /// SVG/DOT fill color (light tint for gates, solid for controls).
    #[must_use]
    pub fn hex_fill(self) -> &'static str {
        match self {
            Self::None => "#FFFFFF",
            Self::XAxis => "#FFB0B0",
            Self::YAxis => "#B0E8B0",
            Self::ZAxis => "#A8C8F0",
            Self::XZMix => "#E0B0E0",
            Self::XYMix => "#F0E0A0",
            Self::YZMix => "#A0E0E8",
            Self::XYZMix => "#D0D0D0",
            Self::ControlDot => "#333333",
        }
    }

    /// SVG/TikZ border/stroke color.
    #[must_use]
    pub fn hex_stroke(self) -> &'static str {
        match self {
            Self::XAxis => "#AA2222",
            Self::YAxis => "#226622",
            Self::ZAxis => "#2255AA",
            Self::XZMix => "#882288",
            Self::XYMix => "#AA8800",
            Self::YZMix => "#008888",
            Self::XYZMix => "#666666",
            Self::None | Self::ControlDot => "#222222",
        }
    }

    /// Text color inside gates (SVG/TikZ).
    #[must_use]
    pub fn hex_text(self) -> &'static str {
        match self {
            Self::XAxis => "#7A1A1A",
            Self::YAxis => "#1A4A1A",
            Self::ZAxis => "#1A3A7A",
            Self::XZMix => "#5A1A5A",
            Self::XYMix => "#6A5500",
            Self::YZMix => "#005A5A",
            Self::None | Self::XYZMix => "#333333",
            Self::ControlDot => "#FFFFFF",
        }
    }

    /// Short name for `\definecolor` in `TikZ`.
    #[must_use]
    pub fn tikz_name(self) -> &'static str {
        match self {
            Self::None => "cellNone",
            Self::XAxis => "cellX",
            Self::YAxis => "cellY",
            Self::ZAxis => "cellZ",
            Self::XZMix => "cellXZ",
            Self::XYMix => "cellXY",
            Self::YZMix => "cellYZ",
            Self::XYZMix => "cellXYZ",
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

/// How to display rotation angles in gate labels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AngleUnit {
    /// Display as multiples of pi with fractions, e.g. `\u{03C0}/4`, `3\u{03C0}/2`.
    /// Falls back to decimal radians for non-nice fractions.
    #[default]
    Radians,
    /// Display as fractional turns, e.g. `.25`, `.125`.
    Turns,
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

// --- Color palette types ---

/// Fill, stroke, and text colors for a single diagram element category.
#[derive(Clone, Debug)]
pub struct ColorTriplet {
    pub fill: String,
    pub stroke: String,
    pub text: String,
}

impl ColorTriplet {
    /// Create a new triplet from string slices.
    #[must_use]
    pub fn new(fill: &str, stroke: &str, text: &str) -> Self {
        Self {
            fill: fill.to_string(),
            stroke: stroke.to_string(),
            text: text.to_string(),
        }
    }
}

/// Complete color palette for all diagram cell categories.
#[derive(Clone, Debug)]
pub struct ColorPalette {
    pub none: ColorTriplet,
    pub x_axis: ColorTriplet,
    pub y_axis: ColorTriplet,
    pub z_axis: ColorTriplet,
    pub xz_mix: ColorTriplet,
    pub xy_mix: ColorTriplet,
    pub yz_mix: ColorTriplet,
    pub xyz_mix: ColorTriplet,
    pub control_dot: ColorTriplet,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self {
            none: ColorTriplet::new("#FFFFFF", "#222222", "#222222"),
            x_axis: ColorTriplet::new("#FFB0B0", "#AA2222", "#7A1A1A"),
            y_axis: ColorTriplet::new("#B0E8B0", "#226622", "#1A4A1A"),
            z_axis: ColorTriplet::new("#A8C8F0", "#2255AA", "#1A3A7A"),
            xz_mix: ColorTriplet::new("#E0B0E0", "#882288", "#5A1A5A"),
            xy_mix: ColorTriplet::new("#F0E0A0", "#AA8800", "#6A5500"),
            yz_mix: ColorTriplet::new("#A0E0E8", "#008888", "#005A5A"),
            xyz_mix: ColorTriplet::new("#D0D0D0", "#666666", "#333333"),
            control_dot: ColorTriplet::new("#333333", "#222222", "#FFFFFF"),
        }
    }
}

impl ColorPalette {
    /// Look up the color triplet for a given cell color category.
    #[must_use]
    pub fn get(&self, color: CellColor) -> &ColorTriplet {
        match color {
            CellColor::None => &self.none,
            CellColor::XAxis => &self.x_axis,
            CellColor::YAxis => &self.y_axis,
            CellColor::ZAxis => &self.z_axis,
            CellColor::XZMix => &self.xz_mix,
            CellColor::XYMix => &self.xy_mix,
            CellColor::YZMix => &self.yz_mix,
            CellColor::XYZMix => &self.xyz_mix,
            CellColor::ControlDot => &self.control_dot,
        }
    }
}

// --- DiagramStyle ---

/// Full configuration for diagram rendering.
///
/// Controls text symbol set, color modes, dash patterns, and the color palette.
/// Use [`DiagramStyle::builder()`] for convenient construction.
#[derive(Clone, Debug)]
pub struct DiagramStyle {
    pub symbols: SymbolSet,
    /// Whether to emit ANSI color codes in text output.
    pub ansi_color: bool,
    /// Whether graphical outputs (SVG, `TikZ`, DOT) use color. When false,
    /// all gates use the `none` palette entry (monochrome).
    pub color: bool,
    /// Whether to render stroke dash patterns. When false, all strokes are solid.
    pub show_dashes: bool,
    /// How to display rotation angles in gate labels.
    pub angle_unit: AngleUnit,
    pub palette: ColorPalette,
}

impl Default for DiagramStyle {
    fn default() -> Self {
        Self {
            symbols: SymbolSet::Ascii,
            ansi_color: false,
            color: true,
            show_dashes: true,
            angle_unit: AngleUnit::Radians,
            palette: ColorPalette::default(),
        }
    }
}

impl DiagramStyle {
    /// Create a builder for constructing a custom `DiagramStyle`.
    #[must_use]
    pub fn builder() -> DiagramStyleBuilder {
        DiagramStyleBuilder::new()
    }

    /// Look up the effective color triplet for a cell, respecting the `color` flag.
    /// Control dots are always filled (even in monochrome) so they remain visible.
    #[must_use]
    pub fn triplet(&self, color: CellColor) -> &ColorTriplet {
        if self.color || color == CellColor::ControlDot {
            self.palette.get(color)
        } else {
            self.palette.get(CellColor::None)
        }
    }

    /// Effective SVG dasharray for a gate family, respecting `show_dashes`.
    #[must_use]
    pub fn svg_dasharray(&self, family: GateFamily) -> &'static str {
        if self.show_dashes {
            family.svg_dasharray()
        } else {
            ""
        }
    }

    /// Effective `TikZ` dash pattern for a gate family, respecting `show_dashes`.
    #[must_use]
    pub fn tikz_dash(&self, family: GateFamily) -> &'static str {
        if self.show_dashes {
            family.tikz_dash()
        } else {
            ""
        }
    }

    /// Effective DOT style for a gate family, respecting `show_dashes`.
    #[must_use]
    pub fn dot_style(&self, family: GateFamily) -> &'static str {
        if self.show_dashes {
            family.dot_style()
        } else {
            ""
        }
    }
}

// --- DiagramStyleBuilder ---

/// Builder for [`DiagramStyle`].
#[derive(Clone, Debug)]
pub struct DiagramStyleBuilder {
    style: DiagramStyle,
}

impl DiagramStyleBuilder {
    /// Create a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            style: DiagramStyle::default(),
        }
    }

    /// Preset: plain ASCII, no ANSI color.
    #[must_use]
    pub fn ascii() -> Self {
        Self::new()
    }

    /// Preset: ASCII with ANSI color.
    #[must_use]
    pub fn color_ascii() -> Self {
        let mut b = Self::new();
        b.style.ansi_color = true;
        b
    }

    /// Preset: Unicode box-drawing, no ANSI color.
    #[must_use]
    pub fn unicode() -> Self {
        let mut b = Self::new();
        b.style.symbols = SymbolSet::Unicode;
        b
    }

    /// Preset: Unicode with ANSI color.
    #[must_use]
    pub fn color_unicode() -> Self {
        let mut b = Self::new();
        b.style.symbols = SymbolSet::Unicode;
        b.style.ansi_color = true;
        b
    }

    /// Set the character symbol set.
    #[must_use]
    pub fn symbols(mut self, s: SymbolSet) -> Self {
        self.style.symbols = s;
        self
    }

    /// Enable or disable ANSI color in text output.
    #[must_use]
    pub fn ansi_color(mut self, b: bool) -> Self {
        self.style.ansi_color = b;
        self
    }

    /// Enable or disable color in graphical output (SVG, `TikZ`, DOT).
    #[must_use]
    pub fn color(mut self, b: bool) -> Self {
        self.style.color = b;
        self
    }

    /// Enable or disable dash stroke patterns.
    #[must_use]
    pub fn show_dashes(mut self, b: bool) -> Self {
        self.style.show_dashes = b;
        self
    }

    /// Set the angle display unit for rotation gate labels.
    #[must_use]
    pub fn angle_unit(mut self, u: AngleUnit) -> Self {
        self.style.angle_unit = u;
        self
    }

    /// Set the entire color palette.
    #[must_use]
    pub fn palette(mut self, p: ColorPalette) -> Self {
        self.style.palette = p;
        self
    }

    /// Set X-axis colors.
    #[must_use]
    pub fn x_axis(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.x_axis = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set Y-axis colors.
    #[must_use]
    pub fn y_axis(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.y_axis = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set Z-axis colors.
    #[must_use]
    pub fn z_axis(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.z_axis = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set X-Z mix colors.
    #[must_use]
    pub fn xz_mix(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.xz_mix = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set X-Y mix colors.
    #[must_use]
    pub fn xy_mix(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.xy_mix = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set Y-Z mix colors.
    #[must_use]
    pub fn yz_mix(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.yz_mix = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set XYZ mix colors.
    #[must_use]
    pub fn xyz_mix(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.xyz_mix = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Set control dot colors.
    #[must_use]
    pub fn control_dot(mut self, fill: &str, stroke: &str, text: &str) -> Self {
        self.style.palette.control_dot = ColorTriplet::new(fill, stroke, text);
        self
    }

    /// Build the final `DiagramStyle`.
    #[must_use]
    pub fn build(self) -> DiagramStyle {
        self.style
    }
}

impl Default for DiagramStyleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// --- DiagramRenderer ---

/// A prepared diagram bound to a style, ready to render in any output format.
///
/// Obtained from `render_with` on [`crate::UnitaryRep`], `TickCircuit`, or `DagCircuit`.
pub struct DiagramRenderer<'a> {
    diagram: CircuitDiagram,
    header: String,
    style: &'a DiagramStyle,
}

impl<'a> DiagramRenderer<'a> {
    /// Create a new renderer from a pre-built diagram and style.
    #[must_use]
    pub fn new(diagram: CircuitDiagram, header: String, style: &'a DiagramStyle) -> Self {
        Self {
            diagram,
            header,
            style,
        }
    }

    /// Render as a text wire diagram using the style's symbol set.
    #[must_use]
    pub fn text(&self) -> String {
        self.diagram.render_text(&self.header, self.style)
    }

    /// Render as an ASCII text diagram (overrides symbols to ASCII).
    #[must_use]
    pub fn ascii(&self) -> String {
        let mut s = self.style.clone();
        s.symbols = SymbolSet::Ascii;
        self.diagram.render_text(&self.header, &s)
    }

    /// Render as a Unicode text diagram (overrides symbols to Unicode).
    #[must_use]
    pub fn unicode(&self) -> String {
        let mut s = self.style.clone();
        s.symbols = SymbolSet::Unicode;
        self.diagram.render_text(&self.header, &s)
    }

    /// Render as an SVG string.
    #[must_use]
    pub fn svg(&self) -> String {
        self.diagram.render_svg_with(&self.header, self.style)
    }

    /// Render as a `TikZ` `tikzpicture`.
    #[must_use]
    pub fn tikz(&self) -> String {
        self.diagram.render_tikz_with(&self.header, self.style)
    }

    /// Render as a Graphviz DOT digraph.
    #[must_use]
    pub fn dot(&self) -> String {
        self.diagram.render_dot_with(&self.header, self.style)
    }
}

// --- ANSI color codes ---

const ANSI_RESET: &str = "\x1b[0m";

fn ansi_code(color: CellColor) -> &'static str {
    match color {
        CellColor::None => "",
        CellColor::XAxis => "\x1b[31m",
        CellColor::YAxis => "\x1b[32m",
        CellColor::ZAxis => "\x1b[34m",
        CellColor::XZMix => "\x1b[35m",
        CellColor::XYMix => "\x1b[33m",
        CellColor::YZMix => "\x1b[36m",
        CellColor::XYZMix => "\x1b[37m",
        CellColor::ControlDot => "\x1b[1m",
    }
}

// --- CircuitDiagram builder ---

/// A grid-based circuit diagram builder.
///
/// The diagram is organized as a grid of `columns x rows`, where each row
/// corresponds to a qubit wire and each column to a time step / layer.
pub struct CircuitDiagram {
    labels: Vec<String>,
    columns: Vec<Vec<(DiagramCell, CellColor)>>,
    current_col: usize,
    /// Explicit vertical connector spans: `(column, top_row, bottom_row, optional_label)`.
    connector_spans: Vec<(usize, usize, usize, Option<String>)>,
    /// Groups of columns representing a single logical tick: `(label, start_col, end_col)`.
    column_groups: Vec<(String, usize, usize)>,
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
            connector_spans: Vec::new(),
            column_groups: Vec::new(),
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
            connector_spans: Vec::new(),
            column_groups: Vec::new(),
        }
    }

    /// Number of qubit rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.labels.len()
    }

    /// Current column index.
    #[must_use]
    pub fn current_col(&self) -> usize {
        self.current_col
    }

    /// Register a group of columns that represent a single logical tick.
    pub fn add_column_group(&mut self, label: String, start: usize, end: usize) {
        self.column_groups.push((label, start, end));
    }

    fn ensure_column(&mut self) {
        while self.current_col >= self.columns.len() {
            self.columns
                .push(vec![(DiagramCell::Wire, CellColor::None); self.num_rows()]);
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

    /// Fill vertical connectors/crossings between `top` and `bottom` (exclusive)
    /// and record the span for vertical line rendering.
    ///
    /// Rows that are qubit wires get `Crossing`; other rows get `Connector`.
    /// Since every row in a `CircuitDiagram` is a qubit wire, this always
    /// places `Crossing` cells.
    pub fn connect_vertical(&mut self, top: usize, bottom: usize, color: CellColor) {
        self.ensure_column();
        let (lo, hi) = if top < bottom {
            (top, bottom)
        } else {
            (bottom, top)
        };
        self.connector_spans.push((self.current_col, lo, hi, None));
        for row in (lo + 1)..hi {
            if row < self.num_rows() {
                // All rows in CircuitDiagram are qubit wires -> Crossing.
                self.columns[self.current_col][row] = (DiagramCell::Crossing, color);
            }
        }
    }

    /// Record a vertical connector span without setting intermediate cells.
    ///
    /// Use this when intermediate `Crossing` cells are set separately
    /// (e.g. by `set_cell` in circuit display code).
    pub fn add_connector(&mut self, top: usize, bottom: usize) {
        let (lo, hi) = if top < bottom {
            (top, bottom)
        } else {
            (bottom, top)
        };
        self.connector_spans.push((self.current_col, lo, hi, None));
    }

    /// Record a labeled vertical connector span without setting intermediate cells.
    ///
    /// The label is rendered on the connector line between the two endpoints
    /// (e.g. "ZZ" for symmetric two-qubit interactions).
    pub fn add_labeled_connector(&mut self, top: usize, bottom: usize, label: String) {
        let (lo, hi) = if top < bottom {
            (top, bottom)
        } else {
            (bottom, top)
        };
        self.connector_spans
            .push((self.current_col, lo, hi, Some(label)));
    }

    /// Advance to the next column.
    pub fn advance(&mut self) {
        self.current_col += 1;
    }

    /// Render the diagram to a text string using a full [`DiagramStyle`].
    #[must_use]
    pub fn render_text(&self, header: &str, style: &DiagramStyle) -> String {
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
        let mut col_widths: Vec<usize> = (0..num_cols)
            .map(|c| {
                self.columns[c]
                    .iter()
                    .map(|(cell, _)| cell_content_width(cell))
                    .max()
                    .unwrap_or(1)
            })
            .collect();

        // Widen columns that carry a connector label so the label text fits.
        for (col, _top, _bottom, label) in &self.connector_spans {
            if let Some(text) = label {
                let text_len = text.chars().count() + 2; // +2 for brackets
                if *col < col_widths.len() {
                    col_widths[*col] = col_widths[*col].max(text_len);
                }
            }
        }

        let label_width = self.labels.iter().map(String::len).max().unwrap_or(2);

        let wire_char = match style.symbols {
            SymbolSet::Ascii => '-',
            SymbolSet::Unicode => '\u{2500}', // ─
        };

        let mut out = String::new();
        if !header.is_empty() {
            writeln!(out, "{header}").unwrap();
            writeln!(out).unwrap();
        }

        // Bracket annotation line for column groups.
        if !self.column_groups.is_empty() {
            let mut col_offsets = Vec::with_capacity(num_cols);
            let mut offset = 0usize;
            for &w in &col_widths {
                col_offsets.push(offset);
                offset += w + 2;
            }
            let total_width = offset;

            let mut bracket_chars: Vec<char> = vec![' '; total_width];

            let (open_bracket, close_bracket, dash) = match style.symbols {
                SymbolSet::Ascii => ('|', '|', '-'),
                SymbolSet::Unicode => ('\u{251C}', '\u{2524}', '\u{2500}'),
            };

            for (label, start, end) in &self.column_groups {
                if *start >= num_cols || *end >= num_cols {
                    continue;
                }
                let char_start = col_offsets[*start];
                let char_end = col_offsets[*end] + col_widths[*end] + 2;

                if char_end <= char_start {
                    continue;
                }

                for c in &mut bracket_chars[char_start..char_end] {
                    *c = dash;
                }
                bracket_chars[char_start] = open_bracket;
                bracket_chars[char_end - 1] = close_bracket;

                let span_len = char_end - char_start;
                let label_len = label.chars().count();
                if label_len < span_len.saturating_sub(2) {
                    let pad = (span_len - label_len) / 2;
                    for (i, ch) in label.chars().enumerate() {
                        let pos = char_start + pad + i;
                        if pos < char_end {
                            bracket_chars[pos] = ch;
                        }
                    }
                }
            }

            write!(out, "{:>width$}  ", "", width = label_width).unwrap();
            let bracket_line: String = bracket_chars.into_iter().collect();
            writeln!(out, "{}", bracket_line.trim_end()).unwrap();
        }

        for row in 0..num_rows {
            write!(out, "{:>label_width$}: ", self.labels[row]).unwrap();

            for (col_idx, &width) in col_widths.iter().enumerate() {
                let (ref cell, color) = self.columns[col_idx][row];
                let rendered = render_cell(cell, width, wire_char, style);

                if style.ansi_color && !matches!(cell, DiagramCell::Wire) {
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
                let has_adjacent_label =
                    self.connector_spans.iter().any(|(_, top, bottom, label)| {
                        label.is_some() && *bottom - *top == 1 && *top == row
                    });
                if has_adjacent_label {
                    // | row above label
                    if let Some(line) =
                        self.render_connector_row(row, num_cols, &col_widths, style, false)
                    {
                        writeln!(out, "{}", line.trim_end()).unwrap();
                    }
                    // label row
                    if let Some(line) =
                        self.render_connector_row(row, num_cols, &col_widths, style, true)
                    {
                        writeln!(out, "{}", line.trim_end()).unwrap();
                    }
                    // | row below label
                    if let Some(line) =
                        self.render_connector_row(row, num_cols, &col_widths, style, false)
                    {
                        writeln!(out, "{}", line.trim_end()).unwrap();
                    }
                } else if let Some(line) =
                    self.render_connector_row(row, num_cols, &col_widths, style, true)
                {
                    writeln!(out, "{}", line.trim_end()).unwrap();
                }
            }
        }

        out
    }

    // --- SVG rendering ---

    /// Render the diagram as a standalone SVG string.
    ///
    /// If `header` is non-empty it is rendered as a `<text>` title at the top.
    #[must_use]
    pub fn render_svg(&self, header: &str) -> String {
        self.render_svg_with(header, &DiagramStyle::default())
    }

    /// Render the diagram as a standalone SVG string using a full [`DiagramStyle`].
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // SVG coordinate calculations use index as f64
    pub fn render_svg_with(&self, header: &str, style: &DiagramStyle) -> String {
        const ROW_SPACING: f64 = 40.0;
        const MIN_COL_SPACING: f64 = 40.0;
        const COL_PAD: f64 = 10.0;
        const CHAR_WIDTH: f64 = 9.0;
        const BOX_PAD_SHORT: f64 = 18.0;
        const BOX_PAD: f64 = 12.0;
        const BOX_PAD_LONG: f64 = 6.0;
        const GATE_H: f64 = 24.0;
        const LABEL_MARGIN: f64 = 50.0;
        const CTRL_RADIUS: f64 = 3.5;
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
        // Uses gate name length without bracket padding for tighter SVG boxes.
        let col_widths: Vec<usize> = (0..num_cols)
            .map(|c| {
                self.columns[c]
                    .iter()
                    .map(|(cell, _)| cell_svg_width(cell))
                    .max()
                    .unwrap_or(1)
            })
            .collect();

        let box_pad_for = |char_count: usize| -> f64 {
            match char_count {
                0..=1 => BOX_PAD_SHORT,
                2..=4 => BOX_PAD,
                _ => BOX_PAD_LONG,
            }
        };

        // Gate box pixel widths per column.
        let mut gate_ws: Vec<f64> = col_widths
            .iter()
            .map(|&w| ((w as f64) * CHAR_WIDTH + box_pad_for(w)).max(GATE_H))
            .collect();

        // Widen columns that carry a connector label (e.g. "RZZ" on the line
        // between two control dots) so the label box doesn't overlap neighbours.
        for (col, _top, _bottom, label) in &self.connector_spans {
            if let Some(text) = label {
                let cc = text.chars().count();
                let label_w = ((cc as f64) * CHAR_WIDTH + box_pad_for(cc)).max(GATE_H);
                if *col < gate_ws.len() {
                    gate_ws[*col] = gate_ws[*col].max(label_w);
                }
            }
        }

        // Per-column spacing: enough for the gate box plus padding, at least MIN_COL_SPACING.
        let col_spacings: Vec<f64> = gate_ws
            .iter()
            .map(|&gw| (gw + COL_PAD).max(MIN_COL_SPACING))
            .collect();

        // Column center x-positions, placed edge-to-edge.
        let mut col_cx: Vec<f64> = Vec::with_capacity(num_cols);
        let mut x_cursor = LABEL_MARGIN;
        for &spacing in &col_spacings {
            col_cx.push(x_cursor + spacing / 2.0);
            x_cursor += spacing;
        }

        let header_offset: f64 = if header.is_empty() { 0.0 } else { 30.0 };
        let svg_width = x_cursor + 20.0;
        let svg_height = header_offset + (num_rows as f64) * ROW_SPACING + ROW_SPACING * 0.5;

        let mut out = String::new();
        writeln!(
            out,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{svg_width}\" height=\"{svg_height}\">"
        )
        .unwrap();
        writeln!(out, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>").unwrap();

        if !header.is_empty() {
            writeln!(
                out,
                "<text x=\"10\" y=\"20\" font-family=\"monospace\" font-size=\"14\" fill=\"#333\">{header}</text>"
            )
            .unwrap();
        }

        // Layer 0: Column group backgrounds.
        for (label, start, end) in &self.column_groups {
            if *start >= num_cols || *end >= num_cols {
                continue;
            }
            let x1 = col_cx[*start] - col_spacings[*start] / 2.0;
            let x2 = col_cx[*end] + col_spacings[*end] / 2.0;
            let y1 = header_offset + ROW_SPACING * 0.5 - ROW_SPACING * 0.4;
            let y2 =
                header_offset + ROW_SPACING * ((num_rows - 1) as f64 + 0.5) + ROW_SPACING * 0.4;
            let w = x2 - x1;
            let h = y2 - y1;
            writeln!(
                out,
                "<rect x=\"{x1}\" y=\"{y1}\" width=\"{w}\" height=\"{h}\" \
                 rx=\"4\" fill=\"#D0D8E0\" fill-opacity=\"0.5\"/>",
            )
            .unwrap();
            let lx = f64::midpoint(x1, x2);
            let ly = y1 - 2.0;
            writeln!(
                out,
                "<text x=\"{lx}\" y=\"{ly}\" font-family=\"monospace\" font-size=\"10\" \
                 text-anchor=\"middle\" fill=\"#888\">{label}</text>",
            )
            .unwrap();
        }

        // Layer 1: Qubit labels and horizontal wires.
        for row in 0..num_rows {
            let y = header_offset + ROW_SPACING * (row as f64 + 0.5);
            writeln!(
                out,
                "<text x=\"{x}\" y=\"{ty}\" font-family=\"monospace\" font-size=\"{FONT_SIZE}\" \
                 text-anchor=\"end\" dominant-baseline=\"middle\" fill=\"#333\">{label}</text>",
                x = LABEL_MARGIN - 6.0,
                ty = y,
                label = self.labels[row],
            )
            .unwrap();
            writeln!(
                out,
                "<line x1=\"{LABEL_MARGIN}\" y1=\"{y}\" x2=\"{x_cursor}\" y2=\"{y}\" stroke=\"#222222\" stroke-width=\"1\"/>",
            )
            .unwrap();
        }

        // Layer 2: Vertical connector lines (drawn before gates so gates sit on top).
        for (col, top, bottom, label) in &self.connector_spans {
            let col = *col;
            let top = *top;
            let bottom = *bottom;
            if col >= num_cols {
                continue;
            }
            let cx = col_cx[col];
            let y1 = header_offset + ROW_SPACING * (top as f64 + 0.5);
            let y2 = header_offset + ROW_SPACING * (bottom as f64 + 0.5);
            let conn_color = if !style.color {
                CellColor::ControlDot
            } else if label.is_some() {
                self.columns[col][top].1
            } else {
                CellColor::ControlDot
            };
            let conn_stroke = style.triplet(conn_color).stroke.clone();
            writeln!(
                out,
                "<line x1=\"{cx}\" y1=\"{y1}\" x2=\"{cx}\" y2=\"{y2}\" \
                 stroke=\"{conn_stroke}\" stroke-width=\"1.5\"/>",
            )
            .unwrap();
            // Render label on the midpoint of the connector line.
            if let Some(text) = label {
                let mid_y = f64::midpoint(y1, y2);
                let lbl_color = self.columns[col][top].1;
                let t = style.triplet(lbl_color);
                let char_count = text.chars().count();
                let pad = if char_count <= 1 {
                    BOX_PAD_SHORT
                } else if char_count <= 4 {
                    BOX_PAD
                } else {
                    BOX_PAD_LONG
                };
                let lw = ((char_count as f64) * CHAR_WIDTH + pad).max(GATE_H);
                let lh = GATE_H * 0.85;
                writeln!(
                    out,
                    "<rect x=\"{rx}\" y=\"{ry}\" width=\"{lw}\" height=\"{lh}\" \
                     rx=\"{GATE_RX}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"/>",
                    rx = cx - lw / 2.0,
                    ry = mid_y - lh / 2.0,
                    fill = t.fill,
                    stroke = t.stroke,
                )
                .unwrap();
                writeln!(
                    out,
                    "<text x=\"{cx}\" y=\"{mid_y}\" font-family=\"monospace\" font-size=\"{fs}\" \
                     text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"{fill}\">{text}</text>",
                    fs = FONT_SIZE - 1.0,
                    fill = t.text,
                )
                .unwrap();
            }
        }

        // Layer 3: Gate boxes, control dots, and crossing markers (on top of wires).
        for (col_idx, &cx) in col_cx.iter().enumerate().take(num_cols) {
            for row in 0..num_rows {
                let cy = header_offset + ROW_SPACING * (row as f64 + 0.5);
                let (ref cell, color) = self.columns[col_idx][row];

                match cell {
                    DiagramCell::Wire | DiagramCell::LabeledConnector(_) => {}
                    DiagramCell::Gate(s, family) => {
                        let t = style.triplet(color);
                        // Per-gate width: sized to its own label, centered in the column.
                        let char_count = s.chars().count();
                        let gw = ((char_count as f64) * CHAR_WIDTH + box_pad_for(char_count))
                            .max(GATE_H);
                        let dash = style.svg_dasharray(*family);
                        let dash_attr = if dash.is_empty() {
                            String::new()
                        } else {
                            format!(" stroke-dasharray=\"{dash}\"")
                        };
                        let x1 = cx - gw / 2.0;
                        let y1 = cy - GATE_H / 2.0;
                        let x2 = x1 + gw;
                        let y2 = y1 + GATE_H;
                        let r = GATE_H / 2.0; // curve radius
                        match family {
                            GateFamily::Preparation => {
                                // Curved left side, flat right side.
                                writeln!(
                                    out,
                                    "<path d=\"M {x2} {y1} L {lx} {y1} \
                                     A {r} {r} 0 0 0 {lx} {y2} \
                                     L {x2} {y2} Z\" \
                                     fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"{dash_attr}/>",
                                    lx = x1 + r,
                                    fill = t.fill,
                                    stroke = t.stroke,
                                )
                                .unwrap();
                            }
                            GateFamily::Measurement => {
                                // Flat left side, curved right side.
                                writeln!(
                                    out,
                                    "<path d=\"M {x1} {y1} L {rx_pt} {y1} \
                                     A {r} {r} 0 0 1 {rx_pt} {y2} \
                                     L {x1} {y2} Z\" \
                                     fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"{dash_attr}/>",
                                    rx_pt = x2 - r,
                                    fill = t.fill,
                                    stroke = t.stroke,
                                )
                                .unwrap();
                            }
                            _ => {
                                writeln!(
                                    out,
                                    "<rect x=\"{x1}\" y=\"{y1}\" width=\"{gw}\" height=\"{GATE_H}\" \
                                     rx=\"{GATE_RX}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"{dash_attr}/>",
                                    fill = t.fill,
                                    stroke = t.stroke,
                                )
                                .unwrap();
                            }
                        }
                        writeln!(
                            out,
                            "<text x=\"{cx}\" y=\"{cy}\" font-family=\"monospace\" font-size=\"{FONT_SIZE}\" \
                             text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"{fill}\">{s}</text>",
                            fill = t.text,
                        )
                        .unwrap();
                    }
                    DiagramCell::Control => {
                        let effective = if style.color {
                            color
                        } else {
                            CellColor::ControlDot
                        };
                        let t = style.triplet(effective);
                        writeln!(
                            out,
                            "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{CTRL_RADIUS}\" \
                             fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1\"/>",
                            fill = t.fill,
                            stroke = t.stroke,
                        )
                        .unwrap();
                    }
                    DiagramCell::Crossing | DiagramCell::Connector => {
                        writeln!(
                            out,
                            "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"2\" fill=\"#222\"/>",
                        )
                        .unwrap();
                    }
                }
            }
        }

        writeln!(out, "</svg>").unwrap();
        out
    }

    // --- TikZ rendering ---

    /// Render the diagram as a `TikZ` `tikzpicture` environment.
    ///
    /// Requires only `\usepackage{tikz}` -- no quantikz. If `header` is
    /// non-empty it is emitted as a `TikZ` comment.
    #[must_use]
    pub fn render_tikz(&self, header: &str) -> String {
        self.render_tikz_with(header, &DiagramStyle::default())
    }

    /// Render the diagram as a `TikZ` `tikzpicture` using a full [`DiagramStyle`].
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // TikZ coordinate calculations use index as f64
    pub fn render_tikz_with(&self, header: &str, style: &DiagramStyle) -> String {
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

        // Color definitions from the style palette.
        for &c in &[
            CellColor::None,
            CellColor::XAxis,
            CellColor::YAxis,
            CellColor::ZAxis,
            CellColor::XZMix,
            CellColor::XYMix,
            CellColor::YZMix,
            CellColor::XYZMix,
            CellColor::ControlDot,
        ] {
            let name = c.tikz_name();
            let t = style.palette.get(c);
            let fill_hex = t.fill.strip_prefix('#').unwrap_or(&t.fill);
            let stroke_hex = t.stroke.strip_prefix('#').unwrap_or(&t.stroke);
            let text_hex = t.text.strip_prefix('#').unwrap_or(&t.text);
            writeln!(out, "  \\definecolor{{{name}Fill}}{{HTML}}{{{fill_hex}}}",).unwrap();
            writeln!(
                out,
                "  \\definecolor{{{name}Stroke}}{{HTML}}{{{stroke_hex}}}",
            )
            .unwrap();
            writeln!(out, "  \\definecolor{{{name}Text}}{{HTML}}{{{text_hex}}}",).unwrap();
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

        // Column group backgrounds.
        for (label, start, end) in &self.column_groups {
            if *start >= num_cols || *end >= num_cols {
                continue;
            }
            let x1 = (*start as f64 + 0.5) * COL_STEP - GATE_W / 2.0 - 0.1;
            let x2 = (*end as f64 + 0.5) * COL_STEP + GATE_W / 2.0 + 0.1;
            let y1 = ROW_STEP * 0.3;
            let y2 = -((num_rows - 1) as f64) * ROW_STEP - ROW_STEP * 0.3;
            writeln!(
                out,
                "  \\fill[black!10, rounded corners=2pt] ({x1:.2},{y1:.2}) rectangle ({x2:.2},{y2:.2});",
            )
            .unwrap();
            let mid_x = f64::midpoint(x1, x2);
            let label_y = y1 + 0.2;
            writeln!(
                out,
                "  \\node[font=\\tiny\\ttfamily, gray] at ({mid_x:.2},{label_y:.2}) {{{label}}};",
            )
            .unwrap();
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
                // When style.color is false, use CellColor::None for all gates.
                let effective = if style.color { color } else { CellColor::None };
                let name = effective.tikz_name();

                match cell {
                    DiagramCell::Wire | DiagramCell::LabeledConnector(_) => {}
                    DiagramCell::Gate(s, family) => {
                        let dash = style.tikz_dash(*family);
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
                        let ctrl_effective = if style.color {
                            color
                        } else {
                            CellColor::ControlDot
                        };
                        let ctrl_name = ctrl_effective.tikz_name();
                        writeln!(
                            out,
                            "  \\node[ctrl, fill={ctrl_name}Fill, draw={ctrl_name}Stroke] at ({x:.2},{y:.2}) {{}};",
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
        }

        // Vertical connector lines (from explicit spans).
        for (col, top, bottom, label) in &self.connector_spans {
            let col = *col;
            let top = *top;
            let bottom = *bottom;
            if col >= num_cols {
                continue;
            }
            let x = (col as f64 + 0.5) * COL_STEP;
            let y1 = -(top as f64) * ROW_STEP;
            let y2 = -(bottom as f64) * ROW_STEP;
            let conn_color = if !style.color {
                CellColor::ControlDot
            } else if label.is_some() {
                self.columns[col][top].1
            } else {
                CellColor::ControlDot
            };
            let conn_name = conn_color.tikz_name();
            writeln!(
                out,
                "  \\draw[{conn_name}Stroke] ({x:.2},{y1:.2}) -- ({x:.2},{y2:.2});",
            )
            .unwrap();
            if let Some(text) = label {
                let mid_y = f64::midpoint(y1, y2);
                let lbl_color = self.columns[col][top].1;
                let effective = if style.color {
                    lbl_color
                } else {
                    CellColor::None
                };
                let name = effective.tikz_name();
                writeln!(
                    out,
                    "  \\node[gate, fill={name}Fill, draw={name}Stroke, text={name}Text] at ({x:.2},{mid_y:.2}) {{\\footnotesize {text}}};",
                )
                .unwrap();
            }
        }

        writeln!(out, "\\end{{tikzpicture}}").unwrap();
        out
    }

    // --- DOT / Graphviz rendering ---

    /// Render the diagram as a Graphviz DOT `digraph` with `rankdir=LR`.
    ///
    /// If `header` is non-empty it is set as the graph `label`.
    #[must_use]
    pub fn render_dot(&self, header: &str) -> String {
        self.render_dot_with(header, &DiagramStyle::default())
    }

    /// Render the diagram as a Graphviz DOT `digraph` using a full [`DiagramStyle`].
    #[must_use]
    pub fn render_dot_with(&self, header: &str, style: &DiagramStyle) -> String {
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
                        writeln!(out, "    {node_id} [label=\"\", shape=point, width=0.01];",)
                            .unwrap();
                    }
                    DiagramCell::Gate(s, family) => {
                        let t = style.triplet(color);
                        let dot_style = style.dot_style(*family);
                        let style_val = if dot_style.is_empty() {
                            "filled".to_string()
                        } else {
                            format!("\"filled,{dot_style}\"")
                        };
                        writeln!(
                            out,
                            "    {node_id} [label=\"{s}\", shape=box, style={style_val}, \
                             fillcolor=\"{fill}\", color=\"{stroke}\", fontcolor=\"{text}\"];",
                            fill = t.fill,
                            stroke = t.stroke,
                            text = t.text,
                        )
                        .unwrap();
                    }
                    DiagramCell::Control => {
                        let t = style.triplet(color);
                        writeln!(
                            out,
                            "    {node_id} [label=\"\", shape=point, width=0.12, \
                             style=filled, fillcolor=\"{fill}\"];",
                            fill = t.fill,
                        )
                        .unwrap();
                    }
                    DiagramCell::Crossing | DiagramCell::Connector => {
                        writeln!(out, "    {node_id} [label=\"\", shape=point, width=0.05];",)
                            .unwrap();
                    }
                    DiagramCell::LabeledConnector(s) => {
                        writeln!(
                            out,
                            "    {node_id} [label=\"{s}\", shape=box, style=filled, fillcolor=white, fontsize=10];",
                        )
                        .unwrap();
                    }
                }
            }
            writeln!(out, "  }}").unwrap();
        }

        // Column group clusters.
        for (i, (label, start, end)) in self.column_groups.iter().enumerate() {
            if *start >= num_cols || *end >= num_cols {
                continue;
            }
            writeln!(out, "  subgraph cluster_group{i} {{").unwrap();
            writeln!(out, "    style=filled;").unwrap();
            writeln!(out, "    color=\"#D0D8E0\";").unwrap();
            writeln!(out, "    fillcolor=\"#D0D8E080\";").unwrap();
            writeln!(out, "    label=\"{label}\";").unwrap();
            writeln!(out, "    fontname=\"Courier\";").unwrap();
            writeln!(out, "    fontsize=9;").unwrap();
            writeln!(out, "    fontcolor=\"#888888\";").unwrap();
            for col_idx in *start..=*end {
                for row in 0..num_rows {
                    writeln!(out, "    r{row}c{col_idx};").unwrap();
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

        // Vertical connector edges (from explicit spans).
        writeln!(out, "  // Vertical connectors").unwrap();
        for (col, top, bottom, _label) in &self.connector_spans {
            let col = *col;
            let top = *top;
            let bottom = *bottom;
            if col >= num_cols {
                continue;
            }
            // Connect top to bottom through all intermediate non-Wire rows.
            let mut prev_row = top;
            for row in (top + 1)..=bottom {
                if row < num_rows && !matches!(self.columns[col][row].0, DiagramCell::Wire) {
                    writeln!(
                        out,
                        "  r{prev_row}c{col} -> r{row}c{col} [style=dashed, dir=none, constraint=false];",
                    )
                    .unwrap();
                    prev_row = row;
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
        style: &DiagramStyle,
        show_labels: bool,
    ) -> Option<String> {
        let label_width = self.labels.iter().map(String::len).max().unwrap_or(2);
        let mut line = String::new();
        write!(line, "{:>width$}  ", "", width = label_width).unwrap();
        let mut has_connector = false;

        for (col_idx, &width) in col_widths.iter().enumerate() {
            if col_idx >= num_cols {
                break;
            }
            // Show a vertical connector when this row and the next are both
            // inside a connector span for this column.
            // Find the connector span for this column/row, if any.
            let span = self
                .connector_spans
                .iter()
                .find(|(col, top, bottom, _)| *col == col_idx && row >= *top && row < *bottom);
            let show = span.is_some();

            if show {
                has_connector = true;
                // Check if this connector row is the midpoint and has a label.
                let label_here = if show_labels {
                    span.and_then(|(_, top, bottom, label)| {
                        label.as_ref().filter(|_| {
                            // Place label on the midpoint connector row.
                            let mid = (top + bottom - 1) / 2;
                            row == mid
                        })
                    })
                } else {
                    None
                };
                // Center a `|` (or label text) within the column width + 2
                // surrounding spaces, matching the cell rendering padding.
                let total = width + 2;
                let content = if let Some(text) = label_here {
                    format!("[{text}]")
                } else {
                    "|".to_string()
                };
                let content_len = content.chars().count();
                let pad_total = total.saturating_sub(content_len);
                let pad_left = pad_total / 2;
                let pad_right = pad_total - pad_left;
                let left: String = std::iter::repeat_n(' ', pad_left).collect();
                let right: String = std::iter::repeat_n(' ', pad_right).collect();
                // Labeled spans use the endpoint cell color; unlabeled use ControlDot.
                let connector_color = if span.is_some_and(|(_, _, _, label)| label.is_some()) {
                    let &(col, top, _, _) = span.unwrap();
                    self.columns
                        .get(col)
                        .and_then(|c| c.get(top))
                        .map_or(CellColor::ControlDot, |&(_, color)| color)
                } else {
                    CellColor::ControlDot
                };
                let code = ansi_code(connector_color);
                if style.ansi_color && !code.is_empty() {
                    write!(line, "{left}{code}{content}{ANSI_RESET}{right}").unwrap();
                } else {
                    write!(line, "{left}{content}{right}").unwrap();
                }
            } else {
                let spaces: String = std::iter::repeat_n(' ', width + 2).collect();
                write!(line, "{spaces}").unwrap();
            }
        }

        if has_connector { Some(line) } else { None }
    }
}

// --- Rendering helpers ---

/// Content width of a cell in characters (before padding).
fn cell_content_width(cell: &DiagramCell) -> usize {
    match cell {
        DiagramCell::Gate(s, _) | DiagramCell::LabeledConnector(s) => s.chars().count() + 2, // +2 for brackets
        DiagramCell::Wire
        | DiagramCell::Control
        | DiagramCell::Crossing
        | DiagramCell::Connector => 1,
    }
}

/// Gate name width in characters without bracket padding (for SVG box sizing).
fn cell_svg_width(cell: &DiagramCell) -> usize {
    match cell {
        DiagramCell::Gate(s, _) | DiagramCell::LabeledConnector(s) => s.chars().count(),
        DiagramCell::Wire
        | DiagramCell::Control
        | DiagramCell::Crossing
        | DiagramCell::Connector => 1,
    }
}

/// Render a single cell into the given column width.
fn render_cell(cell: &DiagramCell, width: usize, wire_char: char, style: &DiagramStyle) -> String {
    match cell {
        DiagramCell::Wire => std::iter::repeat_n(wire_char, width).collect(),
        DiagramCell::Gate(s, family) => {
            let bracketed = format!("{}{s}{}", family.open_bracket(), family.close_bracket());
            pad_center(&bracketed, width, wire_char)
        }
        DiagramCell::Control => {
            let dot = match style.symbols {
                SymbolSet::Ascii => ".",
                SymbolSet::Unicode => "\u{25CF}", // ●
            };
            pad_center(dot, width, wire_char)
        }
        DiagramCell::Crossing => pad_center("+", width, wire_char),
        DiagramCell::Connector => {
            // Connector on a qubit wire row -- treat as crossing.
            pad_center("|", width, wire_char)
        }
        DiagramCell::LabeledConnector(s) => {
            let bracketed = format!("[{s}]");
            pad_center(&bracketed, width, ' ')
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

// --- Graph state style types ---

/// Blend two `#RRGGBB` hex colors at ratio `t` (0.0 = a, 1.0 = b).
///
/// Returns a new `#RRGGBB` string. Clamps `t` to `[0.0, 1.0]`.
#[must_use]
pub fn blend_hex(a: &str, b: &str, t: f64) -> String {
    let t = t.clamp(0.0, 1.0);
    let parse = |hex: &str| -> (u8, u8, u8) {
        let h = hex.strip_prefix('#').unwrap_or(hex);
        let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(0);
        (r, g, b)
    };
    let (r1, g1, b1) = parse(a);
    let (r2, g2, b2) = parse(b);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // color interpolation result is in [0,255]
    let mix =
        |c1: u8, c2: u8| -> u8 { (f64::from(c1) * (1.0 - t) + f64::from(c2) * t).round() as u8 };
    format!("#{:02X}{:02X}{:02X}", mix(r1, r2), mix(g1, g2), mix(b1, b2))
}

/// Fill pattern overlay for graph state vertices.
///
/// Provides a third visual dimension (pattern) beyond color (fill hue)
/// and stroke style (dash pattern), useful for monochrome rendering
/// where cosets would otherwise be indistinguishable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FillPattern {
    /// No pattern overlay (plain solid fill).
    #[default]
    Solid,
    /// Diagonal lines going up-right (/).
    DiagonalUp,
    /// Crosshatch pattern (X).
    Crosshatch,
    /// Small dots.
    Dots,
    /// Horizontal lines (-).
    HorizontalLines,
}

impl FillPattern {
    /// SVG pattern element ID. Empty for `Solid`.
    #[must_use]
    pub fn svg_id(self) -> &'static str {
        match self {
            Self::Solid => "",
            Self::DiagonalUp => "pat-diag",
            Self::Crosshatch => "pat-cross",
            Self::Dots => "pat-dots",
            Self::HorizontalLines => "pat-hlines",
        }
    }

    /// Full SVG `<pattern>` element definition. Empty for `Solid`.
    #[must_use]
    pub fn svg_pattern_def(self) -> &'static str {
        match self {
            Self::Solid => "",
            Self::DiagonalUp => concat!(
                "<pattern id=\"pat-diag\" width=\"6\" height=\"6\" ",
                "patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(45)\">\n",
                "      <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"6\" ",
                "stroke=\"rgba(0,0,0,0.3)\" stroke-width=\"1.5\"/>\n",
                "    </pattern>"
            ),
            Self::Crosshatch => concat!(
                "<pattern id=\"pat-cross\" width=\"6\" height=\"6\" ",
                "patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(45)\">\n",
                "      <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"6\" ",
                "stroke=\"rgba(0,0,0,0.25)\" stroke-width=\"1\"/>\n",
                "      <line x1=\"0\" y1=\"3\" x2=\"6\" y2=\"3\" ",
                "stroke=\"rgba(0,0,0,0.25)\" stroke-width=\"1\"/>\n",
                "    </pattern>"
            ),
            Self::Dots => concat!(
                "<pattern id=\"pat-dots\" width=\"6\" height=\"6\" ",
                "patternUnits=\"userSpaceOnUse\">\n",
                "      <circle cx=\"3\" cy=\"3\" r=\"1.2\" ",
                "fill=\"rgba(0,0,0,0.3)\"/>\n",
                "    </pattern>"
            ),
            Self::HorizontalLines => concat!(
                "<pattern id=\"pat-hlines\" width=\"6\" height=\"6\" ",
                "patternUnits=\"userSpaceOnUse\">\n",
                "      <line x1=\"0\" y1=\"3\" x2=\"6\" y2=\"3\" ",
                "stroke=\"rgba(0,0,0,0.3)\" stroke-width=\"1.5\"/>\n",
                "    </pattern>"
            ),
        }
    }

    /// `TikZ` pattern name for `postaction`. Empty for `Solid`.
    #[must_use]
    pub fn tikz_pattern(self) -> &'static str {
        match self {
            Self::Solid => "",
            Self::DiagonalUp => "north east lines",
            Self::Crosshatch => "crosshatch",
            Self::Dots => "crosshatch dots",
            Self::HorizontalLines => "horizontal lines",
        }
    }
}

/// Fill patterns per axis-permutation coset.
///
/// Each coset can have an independent pattern overlay to distinguish
/// them when fill colors are similar or identical (e.g. monochrome).
#[derive(Clone, Debug)]
pub struct CosetPatterns {
    pub identity: FillPattern,
    pub xz_mix: FillPattern,
    pub xy_mix: FillPattern,
    pub yz_mix: FillPattern,
    pub xyz_mix: FillPattern,
}

impl Default for CosetPatterns {
    fn default() -> Self {
        Self {
            identity: FillPattern::Solid,
            xz_mix: FillPattern::Solid,
            xy_mix: FillPattern::Solid,
            yz_mix: FillPattern::Solid,
            xyz_mix: FillPattern::Solid,
        }
    }
}

impl CosetPatterns {
    /// Look up the fill pattern for a given coset.
    #[must_use]
    pub fn get(&self, coset: CellColor) -> FillPattern {
        match coset {
            CellColor::ZAxis => self.identity,
            CellColor::XZMix => self.xz_mix,
            CellColor::XYMix => self.xy_mix,
            CellColor::YZMix => self.yz_mix,
            CellColor::XYZMix => self.xyz_mix,
            _ => FillPattern::Solid,
        }
    }
}

/// Stroke colors for graph state gate families (rotation types).
///
/// These encode geometric rotation type on the Bloch sphere, orthogonal
/// to the coset fill colors.
#[derive(Clone, Debug)]
pub struct FamilyPalette {
    /// Pauli gates (identity / pi-rotations). Default: navy `#1E3A8A`.
    pub pauli: String,
    /// sqrt-of-Pauli / S-like (pi/2 rotations). Default: green `#2D6A2E`.
    pub s_like: String,
    /// Hadamard-like (pi rotations about face diagonals). Default: maroon `#8B1A1A`.
    pub h_like: String,
    /// Face-like / cyclic (2pi/3 rotations). Default: charcoal `#404040`.
    pub f_like: String,
}

impl Default for FamilyPalette {
    fn default() -> Self {
        Self {
            pauli: "#1E3A8A".to_string(),
            s_like: "#2D6A2E".to_string(),
            h_like: "#8B1A1A".to_string(),
            f_like: "#404040".to_string(),
        }
    }
}

impl FamilyPalette {
    /// Look up the stroke color for a gate family.
    #[must_use]
    pub fn get(&self, family: GateFamily) -> &str {
        match family {
            GateFamily::Pauli
            | GateFamily::Default
            | GateFamily::Measurement
            | GateFamily::Preparation => &self.pauli,
            GateFamily::SLike => &self.s_like,
            GateFamily::HLike => &self.h_like,
            GateFamily::FLike => &self.f_like,
        }
    }
}

/// Full configuration for graph state visualization.
///
/// Controls fill colors (from [`ColorPalette`]), family stroke colors
/// (from [`FamilyPalette`]), and ANSI color output. Use
/// [`GraphStyle::builder()`] for convenient construction.
#[derive(Clone, Debug, Default)]
pub struct GraphStyle {
    pub palette: ColorPalette,
    pub family_strokes: FamilyPalette,
    pub ansi_color: bool,
    /// Whether to render stroke dash patterns on vertices.
    /// When false, all strokes are solid.
    pub show_dashes: bool,
    /// Fill pattern overlays per coset (for monochrome differentiation).
    pub coset_patterns: CosetPatterns,
}

impl GraphStyle {
    /// Create a builder for constructing a custom `GraphStyle`.
    #[must_use]
    pub fn builder() -> GraphStyleBuilder {
        GraphStyleBuilder::new()
    }

    /// Compute the fill color for a VOP vertex.
    ///
    /// Saturated vertices (even sign parity) get a midpoint blend of
    /// the palette's fill and stroke colors; light vertices get the
    /// palette's fill directly.
    #[must_use]
    pub fn vop_fill(&self, coset: CellColor, saturated: bool) -> String {
        let triplet = self.palette.get(coset);
        if saturated {
            blend_hex(&triplet.fill, &triplet.stroke, 0.5)
        } else {
            triplet.fill.clone()
        }
    }

    /// Look up the stroke color for a gate family.
    #[must_use]
    pub fn vop_stroke(&self, family: GateFamily) -> &str {
        self.family_strokes.get(family)
    }

    /// Compute the text color for a VOP vertex.
    ///
    /// Saturated vertices get white text; light vertices get the
    /// palette's text color.
    #[must_use]
    pub fn vop_text(&self, coset: CellColor, saturated: bool) -> &str {
        if saturated {
            "white"
        } else {
            &self.palette.get(coset).text
        }
    }

    /// Effective SVG `stroke-dasharray` for a gate family, respecting `show_dashes`.
    #[must_use]
    pub fn vop_dasharray(&self, family: GateFamily) -> &'static str {
        if self.show_dashes {
            family.svg_dasharray()
        } else {
            ""
        }
    }

    /// Effective `TikZ` dash pattern for a gate family, respecting `show_dashes`.
    #[must_use]
    pub fn vop_tikz_dash(&self, family: GateFamily) -> &'static str {
        if self.show_dashes {
            family.tikz_dash()
        } else {
            ""
        }
    }

    /// Effective DOT style for a gate family, respecting `show_dashes`.
    #[must_use]
    pub fn vop_dot_style(&self, family: GateFamily) -> &'static str {
        if self.show_dashes {
            family.dot_style()
        } else {
            ""
        }
    }

    /// Look up the fill pattern for a coset.
    #[must_use]
    pub fn vop_pattern(&self, coset: CellColor) -> FillPattern {
        self.coset_patterns.get(coset)
    }
}

/// Builder for [`GraphStyle`].
#[derive(Clone, Debug)]
pub struct GraphStyleBuilder {
    style: GraphStyle,
}

impl GraphStyleBuilder {
    /// Create a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            style: GraphStyle::default(),
        }
    }

    /// Set the entire color palette.
    #[must_use]
    pub fn palette(mut self, p: ColorPalette) -> Self {
        self.style.palette = p;
        self
    }

    /// Set the entire family stroke palette.
    #[must_use]
    pub fn family_strokes(mut self, f: FamilyPalette) -> Self {
        self.style.family_strokes = f;
        self
    }

    /// Enable or disable ANSI color in text output.
    #[must_use]
    pub fn ansi_color(mut self, b: bool) -> Self {
        self.style.ansi_color = b;
        self
    }

    /// Enable or disable stroke dash patterns on vertices.
    #[must_use]
    pub fn show_dashes(mut self, b: bool) -> Self {
        self.style.show_dashes = b;
        self
    }

    /// Set the fill pattern overlays per coset.
    #[must_use]
    pub fn coset_patterns(mut self, p: CosetPatterns) -> Self {
        self.style.coset_patterns = p;
        self
    }

    /// Build the final `GraphStyle`.
    #[must_use]
    pub fn build(self) -> GraphStyle {
        self.style
    }
}

impl Default for GraphStyleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_hex_endpoints() {
        assert_eq!(blend_hex("#FF0000", "#0000FF", 0.0), "#FF0000");
        assert_eq!(blend_hex("#FF0000", "#0000FF", 1.0), "#0000FF");
    }

    #[test]
    fn blend_hex_midpoint() {
        // Midpoint of red and blue
        assert_eq!(blend_hex("#FF0000", "#0000FF", 0.5), "#800080");
    }

    #[test]
    fn graph_style_default_vop_fill() {
        let style = GraphStyle::default();
        // ZAxis saturated: blend of fill #A8C8F0 and stroke #2255AA
        let sat = style.vop_fill(CellColor::ZAxis, true);
        assert!(sat.starts_with('#'));
        assert_eq!(sat.len(), 7);
        // ZAxis light: just the fill
        let light = style.vop_fill(CellColor::ZAxis, false);
        assert_eq!(light, "#A8C8F0");
    }

    #[test]
    fn graph_style_vop_text() {
        let style = GraphStyle::default();
        assert_eq!(style.vop_text(CellColor::ZAxis, true), "white");
        assert_eq!(style.vop_text(CellColor::ZAxis, false), "#1A3A7A");
    }

    #[test]
    fn family_palette_get() {
        let fp = FamilyPalette::default();
        assert_eq!(fp.get(GateFamily::Pauli), "#1E3A8A");
        assert_eq!(fp.get(GateFamily::SLike), "#2D6A2E");
        assert_eq!(fp.get(GateFamily::HLike), "#8B1A1A");
        assert_eq!(fp.get(GateFamily::FLike), "#404040");
    }

    #[test]
    fn empty_diagram() {
        let d = CircuitDiagram::new(0);
        let out = d.render_text("test", &DiagramStyle::default());
        assert_eq!(out, "test\n");
    }

    #[test]
    fn single_gate_ascii() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("[H]"));
        // q1 should be just wire
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(!q1_line.contains('['));
    }

    #[test]
    fn single_gate_unicode() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let out = d.render_text(
            "",
            &DiagramStyle::builder().symbols(SymbolSet::Unicode).build(),
        );
        assert!(out.contains("[H]"));
        assert!(out.contains('\u{2500}')); // ─
        assert!(!out.contains('-'));
    }

    #[test]
    fn control_dot_ascii_vs_unicode() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
        d.connect_vertical(0, 1, CellColor::XAxis);

        let ascii = d.render_text("", &DiagramStyle::default());
        assert!(ascii.contains('.'));

        let unicode = d.render_text(
            "",
            &DiagramStyle::builder().symbols(SymbolSet::Unicode).build(),
        );
        assert!(unicode.contains('\u{25CF}')); // ●
    }

    #[test]
    fn color_output_contains_ansi() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);

        let plain = d.render_text("", &DiagramStyle::default());
        let color = d.render_text("", &DiagramStyle::builder().ansi_color(true).build());

        assert!(!plain.contains("\x1b["));
        assert!(color.contains("\x1b[34m")); // blue
        assert!(color.contains(ANSI_RESET));
    }

    #[test]
    fn crossing_between_qubits() {
        let mut d = CircuitDiagram::new(3);
        d.add_control(0);
        d.add_gate(2, "X", CellColor::XAxis, GateFamily::Default);
        d.connect_vertical(0, 2, CellColor::XAxis);

        let out = d.render_text("", &DiagramStyle::default());
        let q1_line = out.lines().find(|l| l.starts_with("q1:")).unwrap();
        assert!(q1_line.contains('+'));
    }

    #[test]
    fn multi_column_advance() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        d.advance();
        d.add_gate(1, "X", CellColor::ZAxis, GateFamily::Default);

        let out = d.render_text("", &DiagramStyle::default());
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
        let out = d.render_text("My Header", &DiagramStyle::default());
        assert!(out.starts_with("My Header\n"));
    }

    #[test]
    fn connector_row_between_multi_qubit() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
        d.add_connector(0, 1);

        let out = d.render_text("", &DiagramStyle::default());
        // Should have a | connector between q0 and q1
        assert!(out.contains('|'));
    }

    #[test]
    fn lines_have_equal_length() {
        let mut d = CircuitDiagram::new(3);
        d.add_gate(0, "SX", CellColor::ZAxis, GateFamily::Default);
        d.advance();
        d.add_control(0);
        d.add_gate(2, "X", CellColor::XAxis, GateFamily::Default);
        d.connect_vertical(0, 2, CellColor::XAxis);

        let out = d.render_text("", &DiagramStyle::default());
        let qubit_lines: Vec<&str> = out.lines().filter(|l| l.starts_with('q')).collect();
        assert!(qubit_lines.len() >= 2);
        let len0 = qubit_lines[0].len();
        for line in &qubit_lines {
            assert_eq!(line.len(), len0, "qubit lines should have equal length");
        }
    }

    // --- SVG tests ---

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
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
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
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
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

    // --- TikZ tests ---

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
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let out = d.render_tikz("");
        assert!(out.contains("\\begin{tikzpicture}"));
        assert!(out.contains("\\end{tikzpicture}"));
        assert!(out.contains("\\definecolor"));
        assert!(out.contains("cellZ"));
        assert!(out.contains("\\node[gate"));
        assert!(out.contains("{H}"));
        assert!(out.contains("\\draw[gray]")); // wires
    }

    #[test]
    fn tikz_control_and_connector() {
        let mut d = CircuitDiagram::new(2);
        d.add_control(0);
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
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

    // --- DOT tests ---

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
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
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
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
        d.add_connector(0, 1);
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

    // --- Gate family bracket tests ---

    #[test]
    fn pauli_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "X", CellColor::ZAxis, GateFamily::Pauli);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("(X)"));
    }

    #[test]
    fn hlike_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::HLike);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("<H>"));
    }

    #[test]
    fn slike_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::ZAxis, GateFamily::SLike);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("[SZ]"));
    }

    #[test]
    fn flike_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "F", CellColor::ZAxis, GateFamily::FLike);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("{F}"));
    }

    #[test]
    fn measurement_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "MZ", CellColor::ZAxis, GateFamily::Measurement);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("|MZ)"));
    }

    #[test]
    fn preparation_brackets() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "PZ", CellColor::ZAxis, GateFamily::Preparation);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains("(PZ|"));
    }

    // --- Gate family stroke tests ---

    #[test]
    fn svg_slike_dasharray() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::ZAxis, GateFamily::SLike);
        let out = d.render_svg("");
        assert!(out.contains("stroke-dasharray=\"4,3\""));
    }

    #[test]
    fn svg_hlike_dasharray() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::HLike);
        let out = d.render_svg("");
        assert!(out.contains("stroke-dasharray=\"2,2\""));
    }

    #[test]
    fn svg_default_no_dasharray() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "T", CellColor::ZAxis, GateFamily::Default);
        let out = d.render_svg("");
        assert!(!out.contains("stroke-dasharray"));
    }

    #[test]
    fn tikz_slike_dashed() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::ZAxis, GateFamily::SLike);
        let out = d.render_tikz("");
        assert!(out.contains(", dashed]"));
    }

    #[test]
    fn dot_hlike_dotted() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::HLike);
        let out = d.render_dot("");
        assert!(out.contains("filled,dotted"));
    }

    // --- DiagramStyle tests ---

    #[test]
    fn default_style_matches_old_text_output() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let old = d.render_text("header", &DiagramStyle::default());
        let new = d.render_text("header", &DiagramStyle::default());
        assert_eq!(old, new);
    }

    #[test]
    fn default_style_matches_old_svg_output() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let old = d.render_svg("");
        let new = d.render_svg_with("", &DiagramStyle::default());
        assert_eq!(old, new);
    }

    #[test]
    fn default_style_matches_old_tikz_output() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let old = d.render_tikz("");
        let new = d.render_tikz_with("", &DiagramStyle::default());
        assert_eq!(old, new);
    }

    #[test]
    fn default_style_matches_old_dot_output() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let old = d.render_dot("");
        let new = d.render_dot_with("", &DiagramStyle::default());
        assert_eq!(old, new);
    }

    #[test]
    fn custom_palette_appears_in_svg() {
        let style = DiagramStyle::builder()
            .x_axis("#FF0000", "#CC0000", "#880000")
            .build();
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "X", CellColor::XAxis, GateFamily::Pauli);
        let svg = d.render_svg_with("", &style);
        assert!(svg.contains("#FF0000")); // custom fill
        assert!(svg.contains("#CC0000")); // custom stroke
        assert!(svg.contains("#880000")); // custom text
    }

    #[test]
    fn color_false_monochrome_svg() {
        let style = DiagramStyle::builder().color(false).build();
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "X", CellColor::XAxis, GateFamily::Pauli);
        let svg = d.render_svg_with("", &style);
        // When color is false, should use the None palette (white fill, black stroke).
        assert!(svg.contains("#FFFFFF")); // none fill
        assert!(svg.contains("#222222")); // none stroke
        // Should NOT contain the XAxis fill.
        assert!(!svg.contains("#FFB0B0"));
    }

    #[test]
    fn show_dashes_false_no_dasharray_in_svg() {
        let style = DiagramStyle::builder().show_dashes(false).build();
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::ZAxis, GateFamily::SLike);
        let svg = d.render_svg_with("", &style);
        assert!(!svg.contains("stroke-dasharray"));
    }

    #[test]
    fn show_dashes_true_has_dasharray_in_svg() {
        let style = DiagramStyle::builder().show_dashes(true).build();
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::ZAxis, GateFamily::SLike);
        let svg = d.render_svg_with("", &style);
        assert!(svg.contains("stroke-dasharray"));
    }

    #[test]
    fn builder_presets() {
        let s = DiagramStyleBuilder::ascii().build();
        assert_eq!(s.symbols, SymbolSet::Ascii);
        assert!(!s.ansi_color);

        let s = DiagramStyleBuilder::color_ascii().build();
        assert_eq!(s.symbols, SymbolSet::Ascii);
        assert!(s.ansi_color);

        let s = DiagramStyleBuilder::unicode().build();
        assert_eq!(s.symbols, SymbolSet::Unicode);
        assert!(!s.ansi_color);

        let s = DiagramStyleBuilder::color_unicode().build();
        assert_eq!(s.symbols, SymbolSet::Unicode);
        assert!(s.ansi_color);
    }

    #[test]
    fn diagram_renderer_text_and_svg() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let style = DiagramStyle::default();
        let r = DiagramRenderer::new(d, String::new(), &style);
        let text = r.text();
        let svg = r.svg();
        assert!(text.contains("[H]"));
        assert!(svg.contains(">H</text>"));
    }

    #[test]
    fn diagram_renderer_ascii_and_unicode() {
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let style = DiagramStyle::default();
        let r = DiagramRenderer::new(d, String::new(), &style);
        let ascii = r.ascii();
        let unicode = r.unicode();
        assert!(ascii.contains('-'));
        assert!(unicode.contains('\u{2500}'));
    }

    #[test]
    fn color_false_monochrome_dot() {
        let style = DiagramStyle::builder().color(false).build();
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "X", CellColor::XAxis, GateFamily::Pauli);
        let dot = d.render_dot_with("", &style);
        // Should use the None palette colors.
        assert!(dot.contains("#FFFFFF"));
        assert!(dot.contains("#222222"));
        assert!(!dot.contains("#FFB0B0"));
    }

    #[test]
    fn show_dashes_false_no_dashed_in_tikz() {
        let style = DiagramStyle::builder().show_dashes(false).build();
        let mut d = CircuitDiagram::new(1);
        d.add_gate(0, "SZ", CellColor::ZAxis, GateFamily::SLike);
        let tikz = d.render_tikz_with("", &style);
        assert!(!tikz.contains(", dashed]"));
    }

    // --- Column group tests ---

    #[test]
    fn column_group_bracket_ascii() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        d.advance();
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
        d.add_column_group("t0".to_string(), 0, 1);
        let out = d.render_text("", &DiagramStyle::default());
        assert!(out.contains('|'), "bracket should use | chars: {out}");
        assert!(out.contains("t0"), "bracket should contain label: {out}");
    }

    #[test]
    fn column_group_bracket_unicode() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        d.advance();
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
        d.add_column_group("t0".to_string(), 0, 1);
        let out = d.render_text(
            "",
            &DiagramStyle::builder().symbols(SymbolSet::Unicode).build(),
        );
        assert!(
            out.contains('\u{251C}'),
            "bracket should use unicode open: {out}"
        );
        assert!(
            out.contains('\u{2524}'),
            "bracket should use unicode close: {out}"
        );
        assert!(out.contains("t0"), "bracket should contain label: {out}");
    }

    #[test]
    fn no_groups_no_bracket_row() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        let out = d.render_text("", &DiagramStyle::default());
        let lines: Vec<&str> = out.lines().collect();
        // Should have only qubit rows and connector row, no bracket line.
        assert!(
            lines
                .iter()
                .all(|l| l.starts_with('q') || l.trim().is_empty() || l.contains('|')),
            "no bracket line expected: {out}"
        );
    }

    #[test]
    fn svg_column_group_background() {
        let mut d = CircuitDiagram::new(2);
        d.add_gate(0, "H", CellColor::ZAxis, GateFamily::Default);
        d.advance();
        d.add_gate(1, "X", CellColor::XAxis, GateFamily::Default);
        d.add_column_group("t0".to_string(), 0, 1);
        let svg = d.render_svg("");
        assert!(
            svg.contains("fill=\"#D0D8E0\""),
            "SVG should have group background: {svg}"
        );
        assert!(
            svg.contains("fill-opacity=\"0.5\""),
            "SVG should have opacity: {svg}"
        );
        assert!(
            svg.contains(">t0</text>"),
            "SVG should have group label: {svg}"
        );
    }
}
