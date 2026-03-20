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

//! SVG rendering for ZX diagrams.
//!
//! Generates SVG XML strings directly, with no external rendering dependencies.
//! Follows ZX calculus visual conventions:
//! - Circles for Z and X spiders (colored per the chosen [`ColorScheme`])
//! - Yellow squares for H boxes
//! - Small dark circles for boundary vertices
//! - Solid lines for normal edges
//! - Dashed lines with midpoint square for Hadamard edges
//! - Phase labels as rational multiples of pi

use std::collections::HashMap;
use std::fmt::Write;

use num_traits::Zero;
use quizx::detection_webs::{Pauli, PauliWeb};
use quizx::graph::{EType, GraphLike, V, VType};

use super::colors::{self, ColorScheme, Palette};
use super::layout::{LayoutAlgorithm, LayoutOptions, compute_layout};
use crate::pauli_web::{PauliWebResult, WebClassification, classify_webs};

/// Pauli web overlay data bundling webs with their classifications.
#[derive(Debug, Clone)]
pub struct WebOverlay {
    /// The Pauli webs to render.
    pub webs: Vec<PauliWeb>,
    /// Classification for each web (same length as `webs`).
    pub classifications: Vec<WebClassification>,
    /// Original indices for color and legend numbering.
    /// When `None`, indices are `0..webs.len()`.
    pub indices: Option<Vec<usize>>,
}

impl WebOverlay {
    /// Build from a [`PauliWebResult`], classifying each web automatically.
    #[must_use]
    pub fn from_result(result: &PauliWebResult) -> Self {
        let classifications = classify_webs(result);
        Self {
            webs: result.webs.clone(),
            classifications,
            indices: None,
        }
    }

    /// Extract a single web by index, preserving its original color and label.
    ///
    /// The returned overlay contains one web whose color index matches the
    /// original position, so "Web 2" rendered alone still uses color slot 2
    /// and is labeled "Web 2: ...".
    #[must_use]
    pub fn single(&self, index: usize) -> Self {
        let orig_idx = self.original_index(index);
        Self {
            webs: vec![self.webs[index].clone()],
            classifications: vec![self.classifications[index].clone()],
            indices: Some(vec![orig_idx]),
        }
    }

    /// Number of webs in this overlay.
    #[must_use]
    pub fn len(&self) -> usize {
        self.webs.len()
    }

    /// Whether this overlay has no webs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.webs.is_empty()
    }

    /// The original index for internal position `i` (for color and legend numbering).
    #[must_use]
    pub fn original_index(&self, i: usize) -> usize {
        self.indices.as_ref().map_or(i, |idx| idx[i])
    }
}

/// Options for SVG rendering.
#[derive(Debug, Clone)]
pub struct SvgOptions {
    /// Radius of spider circles (pixels).
    pub spider_radius: f64,
    /// Radius of boundary circles (pixels).
    pub boundary_radius: f64,
    /// Size of H-box squares (half side length, pixels).
    pub h_box_size: f64,
    /// Font size for phase labels (pixels).
    pub font_size: f64,
    /// Whether to show phase labels.
    pub show_phases: bool,
    /// Whether to show vertex IDs (for debugging).
    pub show_ids: bool,
    /// Whether to show boundary labels (in[0], out[0], etc.).
    pub show_boundary_labels: bool,
    /// Layout algorithm to use.
    pub layout: LayoutAlgorithm,
    /// Layout options.
    pub layout_options: LayoutOptions,
    /// Pauli web overlay data (optional).
    pub web_overlay: Option<WebOverlay>,
    /// Whether to show Pauli operator labels (X/Y/Z) on web overlay edges.
    pub show_web_labels: bool,
    /// Edge stroke width.
    pub edge_width: f64,
    /// Size of the Hadamard midpoint square (half side length).
    pub hadamard_square_size: f64,
    /// Color scheme for rendering.
    pub color_scheme: ColorScheme,
    /// Curved edge overrides (quadratic bezier): maps (source_vertex, target_vertex)
    /// to a signed bow height in pixels. Uses a single control point at the edge
    /// midpoint, offset perpendicular to the edge by the bow amount.
    /// Edges not in this map (or `cubic_edges`) are drawn straight.
    pub curved_edges: HashMap<(usize, usize), f64>,
    /// Cubic bezier edge overrides: maps (source_vertex, target_vertex) to a pair
    /// of signed bow heights (bow_at_1_3, bow_at_2_3). The two control points sit
    /// at the 1/3 and 2/3 points along the edge, each offset perpendicular by its
    /// bow value. Use opposite signs for an S-curve.
    pub cubic_edges: HashMap<(usize, usize), (f64, f64)>,
    /// When true, degree-2 Z/X spiders with zero phase (identity spiders) are not
    /// drawn. The edges through them are still rendered, so the wire path is
    /// preserved while the spider circles disappear.
    pub hide_identities: bool,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            spider_radius: 15.0,
            boundary_radius: 5.0,
            h_box_size: 10.0,
            font_size: 12.0,
            show_phases: true,
            show_ids: false,
            show_boundary_labels: true,
            layout: LayoutAlgorithm::default(),
            layout_options: LayoutOptions::default(),
            web_overlay: None,
            show_web_labels: true,
            edge_width: 2.0,
            hadamard_square_size: 5.0,
            color_scheme: ColorScheme::default(),
            curved_edges: HashMap::new(),
            cubic_edges: HashMap::new(),
            hide_identities: false,
        }
    }
}

/// Render a ZX graph as an SVG string.
///
/// The rendering pipeline:
/// 1. Compute layout positions
/// 2. Draw edges (normal and Hadamard)
/// 3. Draw Pauli web overlay (if provided)
/// 4. Draw vertices (spiders, H boxes, boundaries)
/// 5. Draw phase labels
/// 6. Draw boundary labels
#[must_use]
pub fn render_svg(graph: &impl GraphLike, options: &SvgOptions) -> String {
    let positions = compute_layout(graph, options.layout, &options.layout_options);
    let palette = options.color_scheme.palette();

    // Compute SVG dimensions
    let (width, height) = compute_dimensions(&positions, options);

    let mut svg = String::new();

    // SVG header
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#,
    );
    let _ = write!(
        svg,
        r#"<rect width="100%" height="100%" fill="{}"/>"#,
        palette.background
    );

    // Build input/output sets for boundary labeling
    let input_set: HashMap<V, usize> = graph
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let output_set: HashMap<V, usize> = graph
        .outputs()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Draw edges
    for (s, t, ety) in graph.edges() {
        if let (Some(&(x1, y1)), Some(&(x2, y2))) = (positions.get(&s), positions.get(&t)) {
            draw_edge(
                &mut svg,
                (x1, y1),
                (x2, y2),
                ety,
                palette,
                options,
                Some((s, t)),
            );
        }
    }

    // Draw Pauli web overlay
    if let Some(ref overlay) = options.web_overlay {
        draw_web_overlay(&mut svg, overlay, &positions, options);
    }

    // Draw vertices (drawn after edges so they appear on top)
    for v in graph.vertices() {
        if let Some(&(x, y)) = positions.get(&v) {
            let vtype = graph.vertex_type(v);

            // Skip identity spiders when hide_identities is enabled
            if options.hide_identities
                && matches!(vtype, VType::Z | VType::X)
                && graph.degree(v) == 2
                && graph.phase(v).is_zero()
            {
                continue;
            }

            draw_vertex(&mut svg, x, y, vtype, palette, options);

            // Phase label
            if options.show_phases && !matches!(vtype, VType::B) {
                let phase = graph.phase(v);
                if !phase.is_zero() {
                    let label = format_phase(phase);
                    draw_phase_label(&mut svg, x, y, &label, palette, options);
                }
            }

            // Vertex ID label
            if options.show_ids {
                draw_id_label(&mut svg, x, y, v, palette, options);
            }

            // Boundary labels
            if options.show_boundary_labels && vtype == VType::B {
                if let Some(&idx) = input_set.get(&v) {
                    draw_boundary_label(&mut svg, x, y, &format!("in[{idx}]"), palette, options);
                } else if let Some(&idx) = output_set.get(&v) {
                    draw_boundary_label(&mut svg, x, y, &format!("out[{idx}]"), palette, options);
                }
            }
        }
    }

    // Draw web legend (on top of everything)
    if let Some(ref overlay) = options.web_overlay {
        draw_legend(&mut svg, overlay, width, options);
    }

    svg.push_str("</svg>");
    svg
}

fn compute_dimensions(positions: &HashMap<V, (f64, f64)>, options: &SvgOptions) -> (f64, f64) {
    if positions.is_empty() {
        return (100.0, 100.0);
    }

    let max_x = positions.values().map(|p| p.0).fold(0.0_f64, f64::max);
    let max_y = positions.values().map(|p| p.1).fold(0.0_f64, f64::max);

    // Extra padding for curved edges that extend beyond vertex bounding box
    let quad_max = options
        .curved_edges
        .values()
        .map(|b| b.abs())
        .fold(0.0_f64, f64::max);
    let cubic_max = options
        .cubic_edges
        .values()
        .map(|(a, b)| a.abs().max(b.abs()))
        .fold(0.0_f64, f64::max);

    let pad =
        options.layout_options.padding + options.spider_radius * 2.0 + quad_max.max(cubic_max);

    // Reserve extra width for the legend so it doesn't overlap graph content
    let legend_extra = options
        .web_overlay
        .as_ref()
        .map_or(0.0, |o| legend_width(o, options));

    (max_x + pad + legend_extra, max_y + pad)
}

/// The resolved curve geometry for an edge.
enum CurveKind {
    Straight,
    /// Quadratic bezier with one control point.
    Quadratic {
        cx: f64,
        cy: f64,
    },
    /// Cubic bezier with two control points.
    Cubic {
        c1x: f64,
        c1y: f64,
        c2x: f64,
        c2y: f64,
    },
}

/// Resolve the curve type for an edge by checking cubic_edges, then curved_edges.
fn resolve_curve(
    options: &SvgOptions,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    edge_key: Option<(usize, usize)>,
) -> CurveKind {
    let (s, t) = match edge_key {
        Some(k) => k,
        None => return CurveKind::Straight,
    };

    // Check cubic first
    if let Some(&(b1, b2)) = options.cubic_edges.get(&(s, t)) {
        let ((c1x, c1y), (c2x, c2y)) = cubic_controls(x1, y1, x2, y2, b1, b2);
        return CurveKind::Cubic { c1x, c1y, c2x, c2y };
    }
    if let Some(&(b1, b2)) = options.cubic_edges.get(&(t, s)) {
        // Reverse: swap control points and negate bows
        let ((c1x, c1y), (c2x, c2y)) = cubic_controls(x1, y1, x2, y2, -b2, -b1);
        return CurveKind::Cubic { c1x, c1y, c2x, c2y };
    }

    // Check quadratic
    if let Some(&bow) = options.curved_edges.get(&(s, t)) {
        let (cx, cy) = quad_control(x1, y1, x2, y2, bow);
        return CurveKind::Quadratic { cx, cy };
    }
    if let Some(&bow) = options.curved_edges.get(&(t, s)) {
        let (cx, cy) = quad_control(x1, y1, x2, y2, -bow);
        return CurveKind::Quadratic { cx, cy };
    }

    CurveKind::Straight
}

/// Perpendicular unit vector for an edge (rotated 90 degrees clockwise in SVG coords).
fn perp_unit(dx: f64, dy: f64) -> (f64, f64) {
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        return (0.0, 0.0);
    }
    (-dy / len, dx / len)
}

/// Compute the quadratic bezier control point for a curved edge.
fn quad_control(x1: f64, y1: f64, x2: f64, y2: f64, bow: f64) -> (f64, f64) {
    let (px, py) = perp_unit(x2 - x1, y2 - y1);
    let mx = (x1 + x2) / 2.0;
    let my = (y1 + y2) / 2.0;
    (mx + bow * px, my + bow * py)
}

/// Compute the two cubic bezier control points.
/// `bow1` offsets the control point at the 1/3 mark, `bow2` at the 2/3 mark.
fn cubic_controls(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    bow1: f64,
    bow2: f64,
) -> ((f64, f64), (f64, f64)) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let (px, py) = perp_unit(dx, dy);
    let c1 = (x1 + dx / 3.0 + bow1 * px, y1 + dy / 3.0 + bow1 * py);
    let c2 = (
        x1 + 2.0 * dx / 3.0 + bow2 * px,
        y1 + 2.0 * dy / 3.0 + bow2 * py,
    );
    (c1, c2)
}

/// Write the SVG path/line element for an edge and return the visual midpoint.
fn write_edge_path(
    svg: &mut String,
    from: (f64, f64),
    to: (f64, f64),
    curve: &CurveKind,
    stroke: &str,
    width: f64,
    dashed: bool,
) -> (f64, f64) {
    let (x1, y1) = from;
    let (x2, y2) = to;
    let dash = if dashed {
        r#" stroke-dasharray="6,3""#
    } else {
        ""
    };
    match *curve {
        CurveKind::Straight => {
            let _ = write!(
                svg,
                r#"<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash}/>"#,
            );
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        }
        CurveKind::Quadratic { cx, cy } => {
            let _ = write!(
                svg,
                r#"<path d="M {x1} {y1} Q {cx} {cy} {x2} {y2}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}/>"#,
            );
            // Quadratic bezier midpoint at t=0.5: (p0 + 2*control + p2) / 4
            ((x1 + 2.0 * cx + x2) / 4.0, (y1 + 2.0 * cy + y2) / 4.0)
        }
        CurveKind::Cubic { c1x, c1y, c2x, c2y } => {
            let _ = write!(
                svg,
                r#"<path d="M {x1} {y1} C {c1x} {c1y} {c2x} {c2y} {x2} {y2}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}/>"#,
            );
            // Cubic bezier midpoint at t=0.5: (p0 + 3*c1 + 3*c2 + p3) / 8
            (
                (x1 + 3.0 * c1x + 3.0 * c2x + x2) / 8.0,
                (y1 + 3.0 * c1y + 3.0 * c2y + y2) / 8.0,
            )
        }
    }
}

fn draw_edge(
    svg: &mut String,
    from: (f64, f64),
    to: (f64, f64),
    ety: EType,
    palette: &Palette,
    options: &SvgOptions,
    edge_key: Option<(usize, usize)>,
) {
    let (x1, y1) = from;
    let (x2, y2) = to;
    let curve = resolve_curve(options, x1, y1, x2, y2, edge_key);

    match ety {
        EType::H => {
            let (mx, my) = write_edge_path(
                svg,
                from,
                to,
                &curve,
                palette.edge_hadamard,
                options.edge_width,
                true,
            );
            // Hadamard midpoint square
            let s = options.hadamard_square_size;
            let _ = write!(
                svg,
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}"/>"#,
                mx - s,
                my - s,
                s * 2.0,
                s * 2.0,
                palette.hadamard_square,
                palette.edge_hadamard,
            );
        }
        _ => {
            // Normal and W-type edges
            write_edge_path(
                svg,
                from,
                to,
                &curve,
                palette.edge_normal,
                options.edge_width,
                false,
            );
        }
    }
}

fn draw_vertex(
    svg: &mut String,
    x: f64,
    y: f64,
    vtype: VType,
    palette: &Palette,
    options: &SvgOptions,
) {
    match vtype {
        VType::Z => {
            let _ = write!(
                svg,
                r#"<circle cx="{x}" cy="{y}" r="{}" fill="{}" stroke="{}" stroke-width="1.5"/>"#,
                options.spider_radius, palette.z_fill, palette.z_stroke,
            );
        }
        VType::X => {
            let _ = write!(
                svg,
                r#"<circle cx="{x}" cy="{y}" r="{}" fill="{}" stroke="{}" stroke-width="1.5"/>"#,
                options.spider_radius, palette.x_fill, palette.x_stroke,
            );
        }
        VType::H => {
            let s = options.h_box_size;
            let _ = write!(
                svg,
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="1.5"/>"#,
                x - s,
                y - s,
                s * 2.0,
                s * 2.0,
                palette.h_fill,
                palette.h_stroke,
            );
        }
        VType::B => {
            let _ = write!(
                svg,
                r#"<circle cx="{x}" cy="{y}" r="{}" fill="{}" stroke="{}"/>"#,
                options.boundary_radius, palette.boundary_fill, palette.boundary_stroke,
            );
        }
        _ => {
            // WInput, WOutput, ZBox: draw as small squares
            let s = options.boundary_radius;
            let _ = write!(
                svg,
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}"/>"#,
                x - s,
                y - s,
                s * 2.0,
                s * 2.0,
                palette.boundary_fill,
                palette.boundary_stroke,
            );
        }
    }
}

fn draw_phase_label(
    svg: &mut String,
    x: f64,
    y: f64,
    label: &str,
    palette: &Palette,
    options: &SvgOptions,
) {
    let _ = write!(
        svg,
        r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="middle" font-family="serif">{}</text>"#,
        x,
        y - options.spider_radius - 4.0,
        options.font_size,
        palette.phase_text,
        label,
    );
}

fn draw_id_label(svg: &mut String, x: f64, y: f64, v: V, palette: &Palette, options: &SvgOptions) {
    let _ = write!(
        svg,
        r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="middle" font-family="monospace">{v}</text>"#,
        x,
        y + options.spider_radius + options.font_size + 2.0,
        options.font_size * 0.8,
        palette.label_text,
    );
}

fn draw_boundary_label(
    svg: &mut String,
    x: f64,
    y: f64,
    label: &str,
    palette: &Palette,
    options: &SvgOptions,
) {
    let _ = write!(
        svg,
        r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="middle" font-family="monospace">{label}</text>"#,
        x,
        y + options.boundary_radius + options.font_size + 2.0,
        options.font_size * 0.8,
        palette.label_text,
    );
}

fn draw_web_overlay(
    svg: &mut String,
    overlay: &WebOverlay,
    positions: &HashMap<V, (f64, f64)>,
    options: &SvgOptions,
) {
    let n = overlay.webs.len();
    if n == 0 {
        return;
    }

    // Each web gets a decreasing stroke width so on shared edges the thinner
    // (later-drawn, higher z-order) web is visible nested inside the thicker one.
    let max_width = options.edge_width * 4.0;
    let min_width = options.edge_width * 1.5;
    let step = if n > 1 {
        (max_width - min_width) / (n - 1) as f64
    } else {
        0.0
    };

    // Pass 1: draw all web lines (thickest first = lowest z-order)
    for (web_idx, web) in overlay.webs.iter().enumerate() {
        let orig = overlay.original_index(web_idx);
        let color = colors::WEB_COLORS[orig % colors::WEB_COLORS.len()];
        let width = max_width - web_idx as f64 * step;

        for &(from, to) in web.edge_operators.keys() {
            if let (Some(&(x1, y1)), Some(&(x2, y2))) = (positions.get(&from), positions.get(&to)) {
                let _ = write!(
                    svg,
                    r#"<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>"#,
                );
            }
        }
    }

    // Pass 2: draw all Pauli labels on top of all lines
    if options.show_web_labels {
        let label_size = options.font_size * 0.85;
        let pad = 2.0;

        for (web_idx, web) in overlay.webs.iter().enumerate() {
            let orig = overlay.original_index(web_idx);
            let opaque = colors::WEB_COLORS_OPAQUE[orig % colors::WEB_COLORS_OPAQUE.len()];

            for (&(from, to), &pauli) in &web.edge_operators {
                if let (Some(&(x1, y1)), Some(&(x2, y2))) =
                    (positions.get(&from), positions.get(&to))
                {
                    let mx = (x1 + x2) / 2.0;
                    let my = (y1 + y2) / 2.0;
                    let label = match pauli {
                        Pauli::X => "X",
                        Pauli::Y => "Y",
                        Pauli::Z => "Z",
                    };

                    // White background rect for readability
                    let _ = write!(
                        svg,
                        r#"<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="white" opacity="0.85"/>"#,
                        mx - label_size / 2.0 - pad,
                        my - label_size / 2.0 - pad,
                        label_size + pad * 2.0,
                        label_size + pad * 2.0,
                    );
                    let _ = write!(
                        svg,
                        r#"<text x="{mx}" y="{}" font-size="{label_size}" fill="{opaque}" text-anchor="middle" font-family="monospace" font-weight="bold">{label}</text>"#,
                        my + label_size / 3.0,
                    );
                }
            }
        }
    }
}

fn classification_label(c: &WebClassification) -> &'static str {
    match c {
        WebClassification::Detector => "Detector",
        WebClassification::InputStabilizer => "Input Stabilizer",
        WebClassification::OutputStabilizer => "Output Stabilizer",
        WebClassification::Propagated => "Propagated",
    }
}

/// Compute the width of the legend box (including margin), or 0.0 if no legend.
fn legend_width(overlay: &WebOverlay, options: &SvgOptions) -> f64 {
    if overlay.webs.is_empty() {
        return 0.0;
    }
    let font_size = options.font_size;
    let swatch_size = font_size;
    let padding = 8.0;
    let swatch_text_gap = 4.0;
    let char_width = font_size * 0.6;
    let max_label_len = overlay
        .webs
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let orig = overlay.original_index(i);
            let class_label = overlay
                .classifications
                .get(i)
                .map_or("", classification_label);
            format!("Web {orig}: {class_label}").len()
        })
        .max()
        .unwrap_or(0);
    let text_width = max_label_len as f64 * char_width;
    // box_width + left margin + right margin
    padding + swatch_size + swatch_text_gap + text_width + padding + 20.0
}

fn draw_legend(svg: &mut String, overlay: &WebOverlay, svg_width: f64, options: &SvgOptions) {
    if overlay.webs.is_empty() {
        return;
    }

    let font_size = options.font_size;
    let row_height = font_size + 6.0;
    let swatch_size = font_size;
    let padding = 8.0;
    let swatch_text_gap = 4.0;

    let char_width = font_size * 0.6;
    let max_label_len = overlay
        .webs
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let orig = overlay.original_index(i);
            let class_label = overlay
                .classifications
                .get(i)
                .map_or("", classification_label);
            format!("Web {orig}: {class_label}").len()
        })
        .max()
        .unwrap_or(0);
    let text_width = max_label_len as f64 * char_width;
    let box_width = padding + swatch_size + swatch_text_gap + text_width + padding;
    let box_height = padding + overlay.webs.len() as f64 * row_height + padding;

    let box_x = svg_width - box_width - 10.0;
    let box_y = 10.0;

    let stroke_color = "#999";
    let text_color = "#333";
    let font = "sans-serif";

    // Legend background
    let _ = write!(
        svg,
        r#"<rect x="{box_x}" y="{box_y}" width="{box_width}" height="{box_height}" rx="4" fill="white" stroke="{stroke_color}" stroke-width="1" opacity="0.95"/>"#,
    );

    for (i, _web) in overlay.webs.iter().enumerate() {
        let orig = overlay.original_index(i);
        let opaque = colors::WEB_COLORS_OPAQUE[orig % colors::WEB_COLORS_OPAQUE.len()];
        let class_label = overlay
            .classifications
            .get(i)
            .map_or("", classification_label);

        let row_y = box_y + padding + i as f64 * row_height;

        // Color swatch
        let _ = write!(
            svg,
            r#"<rect x="{}" y="{}" width="{swatch_size}" height="{swatch_size}" rx="2" fill="{opaque}"/>"#,
            box_x + padding,
            row_y,
        );

        // Label text
        let text_x = box_x + padding + swatch_size + swatch_text_gap;
        let text_y = row_y + swatch_size - 2.0;
        let _ = write!(
            svg,
            r#"<text x="{text_x}" y="{text_y}" font-size="{font_size}" fill="{text_color}" font-family="{font}">Web {orig}: {class_label}</text>"#,
        );
    }
}

/// Format a QuiZX phase as a human-readable string.
///
/// Displays common phases as rational multiples of pi:
/// 0 -> "" (empty), 1 -> "pi", 1/2 -> "pi/2", -1/2 -> "-pi/2", etc.
pub(crate) fn format_phase(phase: quizx::phase::Phase) -> String {
    let r = phase.to_rational();
    let (numer, denom) = (*r.numer(), *r.denom());

    match (numer, denom) {
        (0, _) => String::new(),
        (1, 1) => "pi".to_string(),
        (-1, 1) => "-pi".to_string(),
        (1, _) => format!("pi/{denom}"),
        (-1, _) => format!("-pi/{denom}"),
        (_, 1) => format!("{numer}pi"),
        _ => format!("{numer}pi/{denom}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::from_adjacency_matrix;

    #[test]
    fn test_render_svg_basic() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,
            true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 2);
        let svg = render_svg(&g, &SvgOptions::default());

        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("circle")); // Should have circles for spiders/boundaries
        assert!(svg.contains("line")); // Should have lines for edges
    }

    #[test]
    fn test_render_empty_graph() {
        use quizx::vec_graph::Graph;
        let g = Graph::new();
        let svg = render_svg(&g, &SvgOptions::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn test_format_phase() {
        use num_traits::{One, Zero};
        use quizx::phase::Phase;
        assert_eq!(format_phase(Phase::zero()), "");
        assert_eq!(format_phase(Phase::one()), "pi");
        assert_eq!(format_phase(Phase::new((1, 2))), "pi/2");
        assert_eq!(format_phase(Phase::new((-1, 2))), "-pi/2");
        assert_eq!(format_phase(Phase::new((1, 4))), "pi/4");
        assert_eq!(format_phase(Phase::new((3, 4))), "3pi/4");
    }

    /// Build a small graph (B -- Z -- X -- B) with a single web for overlay tests.
    fn graph_with_web() -> (quizx::vec_graph::Graph, WebOverlay) {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let x = g.add_vertex(VType::X);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, x);
        g.add_edge(x, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let mut web = PauliWeb::new();
        web.set_edge(b0, z, Pauli::Z);
        web.set_edge(z, x, Pauli::X);
        web.set_edge(x, b1, Pauli::Y);

        let overlay = WebOverlay {
            webs: vec![web],
            classifications: vec![WebClassification::Propagated],
            indices: None,
        };
        (g, overlay)
    }

    #[test]
    fn test_web_overlay_with_labels() {
        let (g, overlay) = graph_with_web();
        let opts = SvgOptions {
            web_overlay: Some(overlay),
            ..SvgOptions::default()
        };
        let svg = render_svg(&g, &opts);

        // Pauli label text elements should be present
        assert!(svg.contains(">X</text>"), "expected X label in SVG");
        assert!(svg.contains(">Y</text>"), "expected Y label in SVG");
        assert!(svg.contains(">Z</text>"), "expected Z label in SVG");
    }

    #[test]
    fn test_web_overlay_labels_disabled() {
        let (g, overlay) = graph_with_web();
        let opts = SvgOptions {
            web_overlay: Some(overlay),
            show_web_labels: false,
            ..SvgOptions::default()
        };
        let svg = render_svg(&g, &opts);

        // Overlay lines should be present (thick stroke-linecap="round")
        assert!(
            svg.contains("stroke-linecap=\"round\""),
            "expected overlay lines in SVG"
        );
        // But no Pauli label text
        assert!(
            !svg.contains(">X</text>"),
            "X label should not appear when labels disabled"
        );
        assert!(
            !svg.contains(">Y</text>"),
            "Y label should not appear when labels disabled"
        );
        assert!(
            !svg.contains(">Z</text>"),
            "Z label should not appear when labels disabled"
        );
    }

    #[test]
    fn test_legend_rendering() {
        let (g, overlay) = graph_with_web();
        let opts = SvgOptions {
            web_overlay: Some(overlay),
            ..SvgOptions::default()
        };
        let svg = render_svg(&g, &opts);

        assert!(
            svg.contains("Web 0: Propagated"),
            "legend should contain web classification"
        );
    }

    #[test]
    fn test_web_overlay_from_result_round_trip() {
        let result = crate::pauli_web::PauliWebResult {
            webs: vec![PauliWeb::new(), PauliWeb::new()],
            input_ids: vec![0],
            output_ids: vec![1],
        };
        let overlay = WebOverlay::from_result(&result);
        assert_eq!(overlay.webs.len(), result.webs.len());
        assert_eq!(overlay.classifications.len(), result.webs.len());
    }
}
