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

//! ASCII rendering for ZX diagrams.
//!
//! Produces plain-text representations suitable for terminal output,
//! log files, and environments where graphical rendering is unavailable.
//!
//! # Conventions
//!
//! - `(Z)` for Z spiders, `(X)` for X spiders, `[H]` for H boxes, `*` for boundaries
//! - `-` for horizontal edges, `|` for vertical, `/` `\` for diagonal
//! - Hadamard edges use `.` with `H` at the midpoint
//! - Phase labels appear above their vertex
//! - Optional ANSI terminal colors via `use_color`

use std::collections::{HashMap, HashSet};
use std::fmt;

use num_traits::Zero;
use quizx::graph::{EType, GraphLike, V, VType};

use super::colors::{AnsiColor, ColorScheme};
use super::layout::{LayoutAlgorithm, LayoutOptions, compute_layout};

/// Options for ASCII rendering.
#[derive(Debug, Clone)]
pub struct AsciiOptions {
    /// Whether to show phase labels.
    pub show_phases: bool,
    /// Whether to show vertex IDs (for debugging).
    pub show_ids: bool,
    /// Whether to show boundary labels (i0, o0, etc.).
    pub show_boundary_labels: bool,
    /// Layout algorithm to use.
    pub layout: LayoutAlgorithm,
    /// Layout options (tuned for character-grid spacing).
    pub layout_options: LayoutOptions,
    /// Whether to emit ANSI color escape codes.
    pub use_color: bool,
    /// Color scheme for ANSI coloring.
    pub color_scheme: ColorScheme,
    /// When true, degree-2 Z/X spiders with zero phase (identity spiders) are not
    /// drawn. The edges through them extend through the vertex position so there
    /// is no gap in the wire.
    pub hide_identities: bool,
}

impl Default for AsciiOptions {
    fn default() -> Self {
        Self {
            show_phases: true,
            show_ids: false,
            show_boundary_labels: true,
            layout: LayoutAlgorithm::default(),
            layout_options: LayoutOptions {
                x_spacing: 8.0,
                y_spacing: 3.0,
                padding: 2.0,
                force_iterations: 100,
            },
            use_color: false,
            color_scheme: ColorScheme::default(),
            hide_identities: false,
        }
    }
}

/// A single cell in the character grid, with optional color.
#[derive(Clone, Copy)]
struct Cell {
    ch: char,
    color: Option<AnsiColor>,
}

impl Cell {
    fn blank() -> Self {
        Self {
            ch: ' ',
            color: None,
        }
    }
}

/// A 2D character buffer for building ASCII art.
struct CharGrid {
    cells: Vec<Vec<Cell>>,
    width: usize,
    height: usize,
    use_color: bool,
}

impl CharGrid {
    fn new(width: usize, height: usize, use_color: bool) -> Self {
        Self {
            cells: vec![vec![Cell::blank(); width]; height],
            width,
            height,
            use_color,
        }
    }

    fn set(&mut self, x: usize, y: usize, ch: char, color: Option<AnsiColor>) {
        if x < self.width && y < self.height {
            self.cells[y][x] = Cell { ch, color };
        }
    }

    fn write_str(&mut self, x: usize, y: usize, s: &str, color: Option<AnsiColor>) {
        for (i, ch) in s.chars().enumerate() {
            self.set(x + i, y, ch, color);
        }
    }
}

impl fmt::Display for CharGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first_row = true;
        for row in &self.cells {
            // Find the last non-space character to trim trailing whitespace
            let last_non_space = row.iter().rposition(|c| c.ch != ' ').map_or(0, |i| i + 1);

            if !first_row {
                writeln!(f)?;
            }
            first_row = false;

            if self.use_color {
                let mut current_color: Option<AnsiColor> = None;
                for cell in &row[..last_non_space] {
                    if cell.color != current_color {
                        if current_color.is_some() {
                            write!(f, "{}", AnsiColor::reset())?;
                        }
                        if let Some(c) = cell.color {
                            write!(f, "{}", c.code())?;
                        }
                        current_color = cell.color;
                    }
                    write!(f, "{}", cell.ch)?;
                }
                if current_color.is_some() {
                    write!(f, "{}", AnsiColor::reset())?;
                }
            } else {
                let line: String = row[..last_non_space].iter().map(|c| c.ch).collect();
                write!(f, "{}", line)?;
            }
        }

        Ok(())
    }
}

/// Render a ZX graph as ASCII art.
///
/// The rendering pipeline:
/// 1. Compute layout positions with ASCII-scale spacing
/// 2. Quantize float positions to integer character-grid coordinates
/// 3. Draw edges (using Bresenham's line algorithm)
/// 4. Draw vertices on top (overwriting edge characters)
/// 5. Draw phase and boundary labels
#[must_use]
pub fn render_ascii(graph: &impl GraphLike, options: &AsciiOptions) -> String {
    let positions = compute_layout(graph, options.layout, &options.layout_options);

    if positions.is_empty() {
        return String::new();
    }

    // Quantize to integer grid coordinates
    let grid_positions = quantize_positions(&positions);

    // Determine grid dimensions with margin for labels and symbols
    let max_x = grid_positions.values().map(|p| p.0).max().unwrap_or(0);
    let max_y = grid_positions.values().map(|p| p.1).max().unwrap_or(0);
    let width = max_x + 8;
    let height = max_y + 3;

    let mut grid = CharGrid::new(width, height, options.use_color);

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

    // Build set of identity spiders to hide
    let hidden: HashSet<V> = if options.hide_identities {
        graph
            .vertices()
            .filter(|&v| {
                matches!(graph.vertex_type(v), VType::Z | VType::X)
                    && graph.degree(v) == 2
                    && graph.phase(v).is_zero()
            })
            .collect()
    } else {
        HashSet::new()
    };

    // Draw edges first
    for (s, t, ety) in graph.edges() {
        if let (Some(&(sx, sy)), Some(&(tx, ty))) = (grid_positions.get(&s), grid_positions.get(&t))
        {
            let edge_color = if options.use_color {
                match ety {
                    EType::H => Some(options.color_scheme.ansi_hadamard()),
                    _ => None,
                }
            } else {
                None
            };
            draw_edge(
                &mut grid,
                (sx as i32, sy as i32),
                (tx as i32, ty as i32),
                ety,
                edge_color,
                hidden.contains(&s),
                hidden.contains(&t),
            );
        }
    }

    // Draw vertices on top (overwrites edge characters at vertex positions)
    for v in graph.vertices() {
        if hidden.contains(&v) {
            continue;
        }
        if let Some(&(x, y)) = grid_positions.get(&v) {
            let vtype = graph.vertex_type(v);
            let vertex_color = if options.use_color {
                Some(match vtype {
                    VType::Z => options.color_scheme.ansi_z(),
                    VType::X => options.color_scheme.ansi_x(),
                    VType::H => options.color_scheme.ansi_h(),
                    _ => AnsiColor::White,
                })
            } else {
                None
            };
            draw_vertex(&mut grid, x, y, vtype, vertex_color);

            // Phase label above
            if options.show_phases && !matches!(vtype, VType::B) {
                let phase = graph.phase(v);
                if !phase.is_zero() {
                    let label = format_phase_ascii(phase);
                    let label_x = x.saturating_sub(label.len() / 2);
                    if y > 0 {
                        grid.write_str(label_x, y - 1, &label, vertex_color);
                    }
                }
            }

            // Vertex ID below
            if options.show_ids {
                let id_label = format!("{v}");
                let label_x = x.saturating_sub(id_label.len() / 2);
                grid.write_str(label_x, y + 1, &id_label, None);
            }

            // Boundary labels below
            if options.show_boundary_labels && vtype == VType::B {
                if let Some(&idx) = input_set.get(&v) {
                    let label = format!("i{idx}");
                    let label_x = x.saturating_sub(label.len() / 2);
                    grid.write_str(label_x, y + 1, &label, None);
                } else if let Some(&idx) = output_set.get(&v) {
                    let label = format!("o{idx}");
                    let label_x = x.saturating_sub(label.len() / 2);
                    grid.write_str(label_x, y + 1, &label, None);
                }
            }
        }
    }

    grid.to_string()
}

/// Quantize float positions to integer character-grid coordinates.
fn quantize_positions(positions: &HashMap<V, (f64, f64)>) -> HashMap<V, (usize, usize)> {
    positions
        .iter()
        .map(|(&v, &(x, y))| {
            (
                v,
                (x.round().max(0.0) as usize, y.round().max(0.0) as usize),
            )
        })
        .collect()
}

/// Draw a vertex symbol onto the grid.
fn draw_vertex(grid: &mut CharGrid, x: usize, y: usize, vtype: VType, color: Option<AnsiColor>) {
    match vtype {
        VType::Z => grid.write_str(x.saturating_sub(1), y, "(Z)", color),
        VType::X => grid.write_str(x.saturating_sub(1), y, "(X)", color),
        VType::H => grid.write_str(x.saturating_sub(1), y, "[H]", color),
        VType::B => grid.set(x, y, '*', color),
        _ => grid.set(x, y, '?', color),
    }
}

/// Draw an edge between two grid positions using Bresenham's line algorithm.
///
/// When `include_start` / `include_end` are true the edge character is drawn
/// at the start / end Bresenham point (the vertex position). This is used for
/// hidden identity spiders so the wire extends through without a gap.
fn draw_edge(
    grid: &mut CharGrid,
    from: (i32, i32),
    to: (i32, i32),
    ety: EType,
    color: Option<AnsiColor>,
    include_start: bool,
    include_end: bool,
) {
    let points = bresenham_line(from.0, from.1, to.0, to.1);

    if points.len() <= 2 {
        return; // Vertices adjacent or overlapping; no room for edge characters
    }

    let is_hadamard = matches!(ety, EType::H);
    let mid = points.len() / 2;

    let first = if include_start { 0 } else { 1 };
    let last = if include_end {
        points.len()
    } else {
        points.len() - 1
    };

    for i in first..last {
        let (px, py) = points[i];
        if px < 0 || py < 0 {
            continue;
        }
        let ux = px as usize;
        let uy = py as usize;

        if is_hadamard && i == mid {
            grid.set(ux, uy, 'H', color);
        } else if is_hadamard {
            grid.set(ux, uy, '.', color);
        } else {
            // Determine character from local step direction
            let (dx_step, dy_step) = if i > 0 {
                let (prev_x, prev_y) = points[i - 1];
                (px - prev_x, py - prev_y)
            } else {
                // i == 0 (include_start): use direction to next point
                let (next_x, next_y) = points[i + 1];
                (next_x - px, next_y - py)
            };
            let ch = match (dx_step, dy_step) {
                (_, 0) => '-',
                (0, _) => '|',
                (1, 1) | (-1, -1) => '\\',
                _ => '/',
            };
            grid.set(ux, uy, ch, color);
        }
    }
}

/// Bresenham's line algorithm. Returns all integer points from (x0,y0) to (x1,y1).
fn bresenham_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        points.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }

    points
}

/// Format a QuiZX phase as a plain ASCII string.
///
/// Uses the same logic as the SVG `format_phase`: rational multiples of pi
/// displayed as `pi/4`, `3pi/4`, `-pi/2`, etc.
fn format_phase_ascii(phase: quizx::phase::Phase) -> String {
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
    fn test_render_ascii_basic() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,
            true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 2);
        let ascii = render_ascii(&g, &AsciiOptions::default());

        assert!(ascii.contains("(Z)"));
        assert!(ascii.contains('*'));
        assert!(ascii.contains('-'));
    }

    #[test]
    fn test_render_ascii_with_hadamard() {
        let mut g = crate::ZxGraph::new();
        let b0 = g.add_vertex(VType::B);
        g.set_coord(b0, (0.0, 0.0));
        let z0 = g.add_vertex(VType::Z);
        g.set_coord(z0, (1.0, 0.0));
        let z1 = g.add_vertex(VType::Z);
        g.set_coord(z1, (2.0, 0.0));
        let b1 = g.add_vertex(VType::B);
        g.set_coord(b1, (3.0, 0.0));

        g.add_edge(b0, z0);
        g.add_edge_with_type(z0, z1, EType::H);
        g.add_edge(z1, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let ascii = render_ascii(&g, &AsciiOptions::default());
        assert!(ascii.contains('H'));
        assert!(ascii.contains('.'));
    }

    #[test]
    fn test_render_ascii_with_phase() {
        let mut g = crate::ZxGraph::new();
        let b0 = g.add_vertex(VType::B);
        g.set_coord(b0, (0.0, 0.0));
        let z = g.add_vertex(VType::Z);
        g.set_coord(z, (1.0, 0.0));
        let b1 = g.add_vertex(VType::B);
        g.set_coord(b1, (2.0, 0.0));

        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_phase(z, (1, 4)); // pi/4
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let ascii = render_ascii(&g, &AsciiOptions::default());
        assert!(ascii.contains("pi/4"));
    }

    #[test]
    fn test_format_phase_ascii() {
        use num_traits::{One, Zero};
        use quizx::phase::Phase;
        assert_eq!(format_phase_ascii(Phase::zero()), "");
        assert_eq!(format_phase_ascii(Phase::one()), "pi");
        assert_eq!(format_phase_ascii(Phase::new((1, 2))), "pi/2");
        assert_eq!(format_phase_ascii(Phase::new((-1, 2))), "-pi/2");
        assert_eq!(format_phase_ascii(Phase::new((1, 4))), "pi/4");
        assert_eq!(format_phase_ascii(Phase::new((3, 4))), "3pi/4");
    }

    #[test]
    fn test_render_empty_graph() {
        use quizx::vec_graph::Graph;
        let g = Graph::new();
        let ascii = render_ascii(&g, &AsciiOptions::default());
        assert!(ascii.is_empty());
    }

    #[test]
    fn test_render_ascii_with_ids() {
        let mut g = crate::ZxGraph::new();
        let z = g.add_vertex(VType::Z);
        g.set_coord(z, (1.0, 0.0));

        let opts = AsciiOptions {
            show_ids: true,
            show_phases: false,
            show_boundary_labels: false,
            ..AsciiOptions::default()
        };
        let ascii = render_ascii(&g, &opts);
        assert!(ascii.contains("(Z)"));
        assert!(ascii.contains(&z.to_string()));
    }

    #[test]
    fn test_render_ascii_boundary_labels() {
        let mut g = crate::ZxGraph::new();
        let b0 = g.add_vertex(VType::B);
        g.set_coord(b0, (0.0, 0.0));
        let b1 = g.add_vertex(VType::B);
        g.set_coord(b1, (1.0, 0.0));
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let ascii = render_ascii(&g, &AsciiOptions::default());
        assert!(ascii.contains("i0"));
        assert!(ascii.contains("o0"));
    }

    #[test]
    fn test_bresenham_horizontal() {
        let points = bresenham_line(0, 0, 5, 0);
        assert_eq!(points.len(), 6);
        for (i, &(x, y)) in points.iter().enumerate() {
            assert_eq!(x, i as i32);
            assert_eq!(y, 0);
        }
    }

    #[test]
    fn test_bresenham_vertical() {
        let points = bresenham_line(0, 0, 0, 4);
        assert_eq!(points.len(), 5);
        for (i, &(x, y)) in points.iter().enumerate() {
            assert_eq!(x, 0);
            assert_eq!(y, i as i32);
        }
    }

    #[test]
    fn test_bresenham_diagonal() {
        let points = bresenham_line(0, 0, 3, 3);
        assert_eq!(points.len(), 4);
        for (i, &(x, y)) in points.iter().enumerate() {
            assert_eq!(x, i as i32);
            assert_eq!(y, i as i32);
        }
    }

    #[test]
    fn test_render_ascii_colored_has_ansi_codes() {
        let mut g = crate::ZxGraph::new();
        let z = g.add_vertex(VType::Z);
        g.set_coord(z, (1.0, 0.0));

        let opts = AsciiOptions {
            use_color: true,
            ..AsciiOptions::default()
        };
        let ascii = render_ascii(&g, &opts);
        // Should contain ANSI escape sequences
        assert!(ascii.contains("\x1b["));
        // Should contain the reset sequence
        assert!(ascii.contains("\x1b[0m"));
    }

    #[test]
    fn test_render_ascii_no_color_no_ansi() {
        let mut g = crate::ZxGraph::new();
        let z = g.add_vertex(VType::Z);
        g.set_coord(z, (1.0, 0.0));

        let ascii = render_ascii(&g, &AsciiOptions::default());
        // Default (no color) should have no escape sequences
        assert!(!ascii.contains("\x1b["));
    }
}
