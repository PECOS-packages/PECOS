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

//! TikZ rendering for ZX diagrams.
//!
//! Generates TikZ/PGF code for inclusion in LaTeX documents.
//! Uses semantic style names (`zxZ`, `zxX`, `zxH`) so the same diagram
//! code works with any color scheme -- only the preamble changes.
//!
//! # Usage
//!
//! The generated TikZ code requires a preamble defining the node styles.
//! Use [`tikz_preamble()`] to get style definitions for a given
//! [`ColorScheme`], or define your own.
//!
//! ```text
//! \documentclass{standalone}
//! % paste tikz_preamble(scheme) output here
//! \begin{document}
//! % paste render_tikz() output here
//! \end{document}
//! ```

use std::collections::HashMap;
use std::fmt::Write;

use num_traits::Zero;
use quizx::graph::{EType, GraphLike, V, VType};

use super::colors::ColorScheme;
use super::layout::{LayoutAlgorithm, LayoutOptions, compute_layout};

/// Options for TikZ rendering.
#[derive(Debug, Clone)]
pub struct TikzOptions {
    /// Scale factor for coordinates (TikZ units, typically cm).
    pub scale: f64,
    /// Whether to show phase labels.
    pub show_phases: bool,
    /// Whether to show vertex IDs as labels (for debugging).
    pub show_ids: bool,
    /// Whether to show boundary labels (in[0], out[0], etc.).
    pub show_boundary_labels: bool,
    /// Layout algorithm to use.
    pub layout: LayoutAlgorithm,
    /// Layout options (positions are converted to TikZ coordinates).
    pub layout_options: LayoutOptions,
    /// Whether to wrap output in a tikzpicture environment.
    pub wrap_in_environment: bool,
    /// Whether to include the preamble style definitions.
    pub include_preamble: bool,
    /// Color scheme for rendering.
    pub color_scheme: ColorScheme,
}

impl Default for TikzOptions {
    fn default() -> Self {
        Self {
            scale: 1.0,
            show_phases: true,
            show_ids: false,
            show_boundary_labels: true,
            layout: LayoutAlgorithm::default(),
            layout_options: LayoutOptions::default(),
            wrap_in_environment: true,
            include_preamble: false,
            color_scheme: ColorScheme::default(),
        }
    }
}

/// Returns a TikZ preamble with node style definitions for the given color scheme.
///
/// Include this in your LaTeX document preamble (before `\begin{document}`).
#[must_use]
pub fn tikz_preamble(scheme: ColorScheme) -> String {
    let (z_draw, z_fill, hadamard_color) = match scheme {
        ColorScheme::Pecos => ("blue!60!black", "blue!30", "magenta"),
        ColorScheme::ZxCanonical => ("green!60!black", "green!30", "blue"),
    };

    format!(
        r"\usepackage{{tikz}}
\usetikzlibrary{{decorations.markings}}

% ZX calculus node styles ({scheme_name})
\tikzstyle{{zxZ}}=[circle, draw={z_draw}, fill={z_fill}, minimum size=6mm, inner sep=1pt]
\tikzstyle{{zxX}}=[circle, draw=red!60!black, fill=red!30, minimum size=6mm, inner sep=1pt]
\tikzstyle{{zxH}}=[rectangle, draw=yellow!60!black, fill=yellow!40, minimum size=4mm, inner sep=1pt]
\tikzstyle{{zxboundary}}=[circle, draw=black, fill=black, minimum size=1.5mm, inner sep=0pt]
\tikzstyle{{zxwire}}=[-]
\tikzstyle{{zxhadamard}}=[-, {hadamard_color}, dashed]
\tikzstyle{{zxhad_midpoint}}=[rectangle, draw={hadamard_color}, fill=yellow!60, minimum size=2.5mm, inner sep=0pt]
\tikzstyle{{zxlabel}}=[font=\footnotesize, text=gray]",
        scheme_name = match scheme {
            ColorScheme::Pecos => "PECOS",
            ColorScheme::ZxCanonical => "ZX canonical",
        },
    )
}

/// Returns a complete standalone LaTeX document wrapping the given TikZ body.
///
/// Useful for quick previewing of a ZX diagram.
#[must_use]
pub fn standalone_document(tikz_body: &str, scheme: ColorScheme) -> String {
    format!(
        r"\documentclass[border=5pt]{{standalone}}
{preamble}
\begin{{document}}
{body}
\end{{document}}",
        preamble = tikz_preamble(scheme),
        body = tikz_body
    )
}

/// Render a ZX graph as TikZ code.
///
/// The output can be included directly in a LaTeX document that has
/// the appropriate style definitions (see [`tikz_preamble()`]).
#[must_use]
pub fn render_tikz(graph: &impl GraphLike, options: &TikzOptions) -> String {
    let positions = compute_layout(graph, options.layout, &options.layout_options);

    // Convert pixel positions to TikZ coordinates (scale down, flip y)
    let tikz_positions = to_tikz_coords(&positions, options);

    let mut out = String::new();

    if options.include_preamble {
        let _ = writeln!(out, "{}", tikz_preamble(options.color_scheme));
        let _ = writeln!(out);
    }

    if options.wrap_in_environment {
        let _ = writeln!(out, "\\begin{{tikzpicture}}");
    }

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

    // Emit nodes
    for v in graph.vertices() {
        if let Some(&(x, y)) = tikz_positions.get(&v) {
            let vtype = graph.vertex_type(v);
            let phase = graph.phase(v);

            let style = match vtype {
                VType::Z => "zxZ",
                VType::X => "zxX",
                VType::H => "zxH",
                VType::B => "zxboundary",
                _ => "zxboundary",
            };

            // Build the label (phase text inside the node)
            let label = if options.show_phases && !matches!(vtype, VType::B) && !phase.is_zero() {
                format_phase_latex(phase)
            } else {
                String::new()
            };

            let _ = writeln!(
                out,
                "  \\node[{style}] (v{v}) at ({x:.2}, {y:.2}) {{{label}}};",
            );

            // Boundary label below the node
            if options.show_boundary_labels && vtype == VType::B {
                if let Some(&idx) = input_set.get(&v) {
                    let _ = writeln!(
                        out,
                        "  \\node[zxlabel, below=1pt] at (v{v}) {{\\tiny in$_{idx}$}};",
                    );
                } else if let Some(&idx) = output_set.get(&v) {
                    let _ = writeln!(
                        out,
                        "  \\node[zxlabel, below=1pt] at (v{v}) {{\\tiny out$_{idx}$}};",
                    );
                }
            }

            // Vertex ID label
            if options.show_ids {
                let _ = writeln!(
                    out,
                    "  \\node[zxlabel, above=1pt] at (v{v}) {{\\tiny {v}}};",
                );
            }
        }
    }

    let _ = writeln!(out);

    // Emit edges
    for (s, t, ety) in graph.edges() {
        if tikz_positions.contains_key(&s) && tikz_positions.contains_key(&t) {
            match ety {
                EType::N => {
                    let _ = writeln!(out, "  \\draw[zxwire] (v{s}) -- (v{t});");
                }
                EType::H => {
                    let _ = writeln!(out, "  \\draw[zxhadamard] (v{s}) -- (v{t});");
                    // Midpoint Hadamard box
                    let _ = writeln!(
                        out,
                        "  \\node[zxhad_midpoint] at ($(v{s})!0.5!(v{t})$) {{}};",
                    );
                }
                _ => {
                    let _ = writeln!(out, "  \\draw[zxwire] (v{s}) -- (v{t});");
                }
            }
        }
    }

    if options.wrap_in_environment {
        let _ = writeln!(out, "\\end{{tikzpicture}}");
    }

    out
}

/// Convert pixel-space positions to TikZ coordinates.
///
/// TikZ coordinates are in cm, with y-axis pointing up (opposite to SVG).
/// We scale down from pixels and flip the y-axis.
fn to_tikz_coords(
    positions: &HashMap<V, (f64, f64)>,
    options: &TikzOptions,
) -> HashMap<V, (f64, f64)> {
    if positions.is_empty() {
        return HashMap::new();
    }

    let max_y = positions.values().map(|p| p.1).fold(0.0_f64, f64::max);

    // Convert: scale from pixels to cm, flip y
    let px_to_cm = options.scale / options.layout_options.x_spacing;

    positions
        .iter()
        .map(|(&v, &(px, py))| {
            let x = px * px_to_cm;
            let y = (max_y - py) * px_to_cm; // flip y so qubit 0 is at top
            (v, (x, y))
        })
        .collect()
}

/// Format a QuiZX phase as a LaTeX string.
///
/// Uses `\pi` for the pi symbol and proper fractions.
fn format_phase_latex(phase: quizx::phase::Phase) -> String {
    let r = phase.to_rational();
    let (numer, denom) = (*r.numer(), *r.denom());

    match (numer, denom) {
        (0, _) => String::new(),
        (1, 1) => "$\\pi$".to_string(),
        (-1, 1) => "$-\\pi$".to_string(),
        (1, _) => format!("$\\frac{{\\pi}}{{{denom}}}$"),
        (-1, _) => format!("$-\\frac{{\\pi}}{{{denom}}}$"),
        (_, 1) => format!("${numer}\\pi$"),
        _ if numer < 0 => format!(
            "$-\\frac{{{numer_abs}\\pi}}{{{denom}}}$",
            numer_abs = -numer
        ),
        _ => format!("$\\frac{{{numer}\\pi}}{{{denom}}}$"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::from_adjacency_matrix;

    #[test]
    fn test_render_tikz_basic() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,
            true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 2);
        let tikz = render_tikz(&g, &TikzOptions::default());

        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("\\end{tikzpicture}"));
        assert!(tikz.contains("\\node[zxZ]"));
        assert!(tikz.contains("\\node[zxboundary]"));
        assert!(tikz.contains("\\draw[zxwire]"));
    }

    #[test]
    fn test_render_tikz_with_hadamard() {
        // Build a graph with a Hadamard edge
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

        let tikz = render_tikz(&g, &TikzOptions::default());
        assert!(tikz.contains("\\draw[zxhadamard]"));
        assert!(tikz.contains("zxhad_midpoint"));
    }

    #[test]
    fn test_render_tikz_with_phase() {
        let mut g = crate::ZxGraph::new();
        let z = g.add_vertex(VType::Z);
        g.set_coord(z, (1.0, 0.0));
        g.set_phase(z, (1, 4)); // pi/4

        let tikz = render_tikz(&g, &TikzOptions::default());
        assert!(tikz.contains("\\pi"));
    }

    #[test]
    fn test_format_phase_latex() {
        use quizx::phase::Phase;
        assert_eq!(format_phase_latex(Phase::new((1, 2))), "$\\frac{\\pi}{2}$");
        assert_eq!(
            format_phase_latex(Phase::new((-1, 4))),
            "$-\\frac{\\pi}{4}$"
        );
        assert_eq!(format_phase_latex(Phase::new((3, 4))), "$\\frac{3\\pi}{4}$");
    }

    #[test]
    fn test_standalone_document() {
        let tikz = "\\begin{tikzpicture}\n\\end{tikzpicture}\n";
        let doc = standalone_document(tikz, ColorScheme::default());
        assert!(doc.contains("\\documentclass"));
        assert!(doc.contains("\\usepackage{tikz}"));
        assert!(doc.contains("\\begin{document}"));
        assert!(doc.contains("\\end{document}"));
    }

    #[test]
    fn test_preamble_has_styles() {
        let preamble = tikz_preamble(ColorScheme::default());
        assert!(preamble.contains("zxZ"));
        assert!(preamble.contains("zxX"));
        assert!(preamble.contains("zxH"));
        assert!(preamble.contains("zxboundary"));
        assert!(preamble.contains("zxhadamard"));
    }

    #[test]
    fn test_preamble_pecos_uses_blue() {
        let preamble = tikz_preamble(ColorScheme::Pecos);
        assert!(preamble.contains("blue!60!black"));
        assert!(preamble.contains("PECOS"));
    }

    #[test]
    fn test_preamble_canonical_uses_green() {
        let preamble = tikz_preamble(ColorScheme::ZxCanonical);
        assert!(preamble.contains("green!60!black"));
        assert!(preamble.contains("ZX canonical"));
    }
}
