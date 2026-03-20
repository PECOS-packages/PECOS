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

//! Interactive HTML viewer for ZX diagrams with toggleable Pauli web overlays.
//!
//! Generates a self-contained HTML file embedding an SVG diagram where each
//! Pauli web is a separate `<g>` group that can be toggled via checkboxes.
//! Includes pan and zoom via mouse/trackpad.

use std::collections::HashMap;
use std::fmt::Write;

use num_traits::Zero;
use quizx::detection_webs::Pauli;
use quizx::graph::{EType, GraphLike, V, VType};

use crate::ZxGraph;
use quizx::simplify;

use super::colors::{self, Palette};
use super::layout::compute_layout;
use super::svg::{SvgOptions, WebOverlay};
use super::tikz::{TikzOptions, render_tikz, standalone_document};

/// Per-vertex data collected during rendering for the JS interactivity layer.
struct VertexJsInfo {
    x: f64,
    y: f64,
    vtype: &'static str,
    phase: String,
    degree: usize,
    boundary: String,
    /// Hit radius for hover/click detection.
    r: f64,
    /// Neighbor vertex IDs.
    neighbors: Vec<V>,
}

/// Render a ZX graph as an interactive HTML file with toggleable web overlays.
///
/// Returns a self-contained HTML string. Each web in the overlay gets its own
/// SVG `<g>` group and a checkbox in the control panel. Pan and zoom are
/// handled via vanilla JavaScript.
#[must_use]
pub fn render_html(graph: &impl GraphLike, options: &SvgOptions) -> String {
    let positions = compute_layout(graph, options.layout, &options.layout_options);
    let palette = options.color_scheme.palette();
    let (width, height) = compute_svg_dimensions(&positions, options);

    // Pre-render TikZ for export
    let tikz_opts = TikzOptions {
        color_scheme: options.color_scheme,
        ..TikzOptions::default()
    };
    let tikz_body = render_tikz(graph, &tikz_opts);
    let tikz_doc = standalone_document(&tikz_body, options.color_scheme);

    let mut html = String::new();

    // HTML header with embedded styles
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n");
    html.push_str("<title>ZX Diagram Viewer</title>\n");
    write_styles(&mut html);
    html.push_str("</head>\n<body>\n");

    // Control panel
    write_control_panel(&mut html, options);

    // SVG container (position: relative for absolute tooltip positioning)
    let _ = write!(
        html,
        "<div id=\"svg-container\" style=\"position:relative\">\n\
         <div id=\"tooltip\"></div>\n\
         <svg id=\"zx-svg\" xmlns=\"http://www.w3.org/2000/svg\" \
         width=\"100%\" height=\"100%\" viewBox=\"0 0 {width} {height}\">\n"
    );

    // Base graph (background + edges + vertices)
    let (svg_body, vertex_infos) =
        render_svg_body(graph, options, palette, &positions, width, height);
    html.push_str(&svg_body);

    // Web overlay groups (one <g> per web)
    if let Some(ref overlay) = options.web_overlay {
        write_web_groups(&mut html, overlay, &positions, options);
    }

    html.push_str("</svg>\n</div>\n");

    // JavaScript for interactivity
    write_script(&mut html, options, &tikz_doc, &vertex_infos);

    html.push_str("</body>\n</html>");
    html
}

fn compute_svg_dimensions(positions: &HashMap<V, (f64, f64)>, options: &SvgOptions) -> (f64, f64) {
    if positions.is_empty() {
        return (100.0, 100.0);
    }
    let max_x = positions.values().map(|p| p.0).fold(0.0_f64, f64::max);
    let max_y = positions.values().map(|p| p.1).fold(0.0_f64, f64::max);
    let pad = options.layout_options.padding + options.spider_radius * 2.0;
    (max_x + pad, max_y + pad)
}

/// Render the SVG body (background, edges, vertices) for a graph.
///
/// Returns the SVG inner HTML (background rect + `<g id="base-graph">` with
/// edges/vertices/labels) and per-vertex info for the JS interactivity layer.
fn render_svg_body(
    graph: &impl GraphLike,
    options: &SvgOptions,
    palette: &Palette,
    positions: &HashMap<V, (f64, f64)>,
    width: f64,
    height: f64,
) -> (String, Vec<(V, VertexJsInfo)>) {
    let mut body = String::new();

    // Background
    let _ = writeln!(
        body,
        "<rect class=\"zx-bg\" width=\"{width}\" height=\"{height}\" fill=\"{}\"/>",
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

    // Base graph group
    body.push_str("<g id=\"base-graph\">\n");

    // Edges
    for (s, t, ety) in graph.edges() {
        if let (Some(&(x1, y1)), Some(&(x2, y2))) = (positions.get(&s), positions.get(&t)) {
            write_edge(&mut body, (x1, y1), (x2, y2), ety, s, t, palette, options);
        }
    }

    // Vertices -- also collect per-vertex info for the JS interactivity layer
    let mut vertex_infos: Vec<(V, VertexJsInfo)> = Vec::new();

    for v in graph.vertices() {
        if let Some(&(x, y)) = positions.get(&v) {
            let vtype = graph.vertex_type(v);
            let degree = graph.degree(v);

            if options.hide_identities
                && matches!(vtype, VType::Z | VType::X)
                && degree == 2
                && graph.phase(v).is_zero()
            {
                continue;
            }

            let phase = graph.phase(v);
            let phase_str = if !phase.is_zero() && !matches!(vtype, VType::B) {
                super::svg::format_phase(phase)
            } else {
                String::new()
            };

            let boundary = if vtype == VType::B {
                if let Some(&idx) = input_set.get(&v) {
                    format!("in:{idx}")
                } else if let Some(&idx) = output_set.get(&v) {
                    format!("out:{idx}")
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            write_vertex(
                &mut body, x, y, vtype, v, degree, &phase_str, &boundary, palette, options,
            );

            if options.show_phases && !phase_str.is_empty() {
                write_phase_label(&mut body, x, y, &phase_str, palette, options);
            }

            if options.show_boundary_labels && vtype == VType::B {
                if let Some(&idx) = input_set.get(&v) {
                    write_boundary_label(&mut body, x, y, &format!("in[{idx}]"), palette, options);
                } else if let Some(&idx) = output_set.get(&v) {
                    write_boundary_label(&mut body, x, y, &format!("out[{idx}]"), palette, options);
                }
            }

            let vtype_str = match vtype {
                VType::Z => "Z",
                VType::X => "X",
                VType::H => "H",
                VType::B => "B",
                _ => "B",
            };
            let r = match vtype {
                VType::Z | VType::X => options.spider_radius,
                VType::H => options.h_box_size,
                _ => options.boundary_radius,
            };
            let neighbors: Vec<V> = graph.neighbors(v).collect();
            vertex_infos.push((
                v,
                VertexJsInfo {
                    x,
                    y,
                    vtype: vtype_str,
                    phase: phase_str,
                    degree,
                    boundary,
                    r,
                    neighbors,
                },
            ));
        }
    }

    body.push_str("</g>\n"); // end base-graph
    (body, vertex_infos)
}

fn write_styles(html: &mut String) {
    html.push_str(
        "<style>\n\
         * { margin: 0; padding: 0; box-sizing: border-box; }\n\
         body { background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; \
                display: flex; height: 100vh; overflow: hidden; }\n\
         #controls { width: 240px; padding: 16px; background: #16213e; \
                     border-right: 1px solid #333; overflow-y: auto; flex-shrink: 0; }\n\
         #controls h2 { font-size: 14px; margin-bottom: 12px; color: #aaa; \
                        text-transform: uppercase; letter-spacing: 1px; }\n\
         .web-toggle { display: flex; align-items: center; gap: 8px; \
                       padding: 6px 0; cursor: pointer; }\n\
         .web-toggle input { cursor: pointer; }\n\
         .swatch { width: 28px; height: 22px; border-radius: 3px; flex-shrink: 0; \
                   border: 1px solid #555; padding: 0; cursor: pointer; \
                   -webkit-appearance: none; appearance: none; }\n\
         .swatch::-webkit-color-swatch-wrapper { padding: 0; }\n\
         .swatch::-webkit-color-swatch { border: none; border-radius: 2px; }\n\
         .swatch::-moz-color-swatch { border: none; border-radius: 2px; }\n\
         .web-label { font-size: 13px; }\n\
         .web-class { font-size: 11px; color: #888; }\n\
         #svg-container { flex: 1; overflow: hidden; cursor: grab; }\n\
         #svg-container.dragging { cursor: grabbing; }\n\
         .palette-section { margin-top: 16px; border-top: 1px solid #333; padding-top: 12px; }\n\
         .palette-section h2 { font-size: 14px; margin-bottom: 8px; color: #aaa; \
                               text-transform: uppercase; letter-spacing: 1px; }\n\
         .palette-select { width: 100%; padding: 5px 8px; background: #1a3a5c; color: #ccc; \
                           border: 1px solid #444; border-radius: 4px; font-size: 13px; \
                           cursor: pointer; }\n\
         .palette-select:hover { background: #245080; }\n\
         .export-section { margin-top: 16px; border-top: 1px solid #333; padding-top: 12px; }\n\
         .export-section h2 { font-size: 14px; margin-bottom: 8px; color: #aaa; \
                              text-transform: uppercase; letter-spacing: 1px; }\n\
         .export-btn { display: block; width: 100%; padding: 6px 10px; margin-bottom: 6px; \
                       background: #1a3a5c; color: #ccc; border: 1px solid #444; \
                       border-radius: 4px; cursor: pointer; font-size: 13px; \
                       text-align: left; }\n\
         .export-btn:hover { background: #245080; color: #fff; }\n\
         .help { font-size: 11px; color: #666; margin-top: 16px; line-height: 1.5; }\n\
         #tooltip { display: none; position: absolute; background: #16213e; color: #e0e0e0;\n\
                    font-size: 12px; padding: 6px 10px; border-radius: 4px;\n\
                    border: 1px solid #444; pointer-events: none; z-index: 100;\n\
                    line-height: 1.4; max-width: 220px; }\n\
         .vertex-hover { filter: brightness(1.3); stroke-width: 3 !important; }\n\
         .vertex-selected { stroke: #ffcc00 !important; stroke-width: 3 !important; }\n\
         .edge-highlight { stroke: #ffcc00 !important; stroke-width: 3 !important; }\n\
         .selection-section { margin-top: 16px; border-top: 1px solid #333; padding-top: 12px; }\n\
         .selection-section h2 { font-size: 14px; margin-bottom: 8px; color: #aaa;\n\
                                 text-transform: uppercase; letter-spacing: 1px; }\n\
         #selection-content { font-size: 13px; line-height: 1.6; }\n\
         .nb-link { color: #6af; cursor: pointer; text-decoration: underline; }\n\
         .nb-link:hover { color: #9cf; }\n\
         .rewrite-section { margin-top: 16px; border-top: 1px solid #333; padding-top: 12px; }\n\
         .rewrite-section h2 { font-size: 14px; margin-bottom: 8px; color: #aaa;\n\
                               text-transform: uppercase; letter-spacing: 1px; }\n\
         .state-indicator { display: block; padding: 6px 10px; background: #1a3a5c;\n\
                            border-radius: 4px; font-size: 13px; margin-bottom: 8px; color: #ccc; }\n\
         .state-indicator.viewing-rewrite { background: #3a1a5c; }\n\
         .rewrite-btn { display: block; width: 100%; padding: 6px 10px; margin-bottom: 6px;\n\
                        background: #1a3a5c; color: #ccc; border: 1px solid #444;\n\
                        border-radius: 4px; cursor: pointer; font-size: 13px;\n\
                        text-align: left; }\n\
         .rewrite-btn:hover { background: #245080; color: #fff; }\n\
         .undo-redo { display:flex; gap:6px; margin:8px 0; }\n\
         .undo-redo button { flex:1; padding:6px 10px; background:#1a3a5c; color:#ccc;\n\
                             border:1px solid #444; border-radius:4px; cursor:pointer;\n\
                             font-size:13px; }\n\
         .undo-redo button:hover:not(:disabled) { background:#245080; color:#fff; }\n\
         .undo-redo button:disabled { opacity:0.4; cursor:default; }\n\
         </style>\n",
    );
}

fn classification_label(c: &crate::pauli_web::WebClassification) -> &'static str {
    use crate::pauli_web::WebClassification;
    match c {
        WebClassification::Detector => "Detector",
        WebClassification::InputStabilizer => "Input Stabilizer",
        WebClassification::OutputStabilizer => "Output Stabilizer",
        WebClassification::Propagated => "Propagated",
    }
}

fn write_control_panel(html: &mut String, options: &SvgOptions) {
    html.push_str("<div id=\"controls\">\n<h2>Pauli Webs</h2>\n");

    if let Some(ref overlay) = options.web_overlay {
        for (i, _web) in overlay.webs.iter().enumerate() {
            let orig = overlay.original_index(i);
            let opaque = colors::WEB_COLORS_OPAQUE[orig % colors::WEB_COLORS_OPAQUE.len()];
            let class_label = overlay
                .classifications
                .get(i)
                .map_or("", classification_label);

            let _ = writeln!(
                html,
                "<label class=\"web-toggle\">\
                 <input type=\"checkbox\" checked onchange=\"toggleWeb({i})\">\
                 <input type=\"color\" class=\"swatch\" id=\"swatch-{i}\" \
                  value=\"{opaque}\" onchange=\"changeWebColor({i}, this.value)\">\
                 <span><span class=\"web-label\">Web {orig}</span><br>\
                 <span class=\"web-class\">{class_label}</span></span>\
                 </label>"
            );
        }
    } else {
        html.push_str("<p style=\"color:#666;font-size:13px\">No web overlay data.</p>\n");
    }

    // Palette switcher
    let pecos_sel = if options.color_scheme == colors::ColorScheme::Pecos {
        " selected"
    } else {
        ""
    };
    let zx_sel = if options.color_scheme == colors::ColorScheme::ZxCanonical {
        " selected"
    } else {
        ""
    };
    let _ = write!(
        html,
        "<div class=\"palette-section\">\n\
         <h2>Color Scheme</h2>\n\
         <select class=\"palette-select\" onchange=\"switchPalette(this.value)\">\n\
         <option value=\"pecos\"{pecos_sel}>PECOS (RGB = XYZ)</option>\n\
         <option value=\"zx\"{zx_sel}>ZX Canonical</option>\n\
         </select>\n\
         </div>\n"
    );

    // Selection info panel (populated by JS on click)
    html.push_str(
        "<div id=\"selection-panel\" class=\"selection-section\" style=\"display:none\">\n\
         <h2>Selection</h2>\n\
         <div id=\"selection-content\"></div>\n\
         </div>\n",
    );

    // Export buttons
    html.push_str(
        "<div class=\"export-section\">\n\
         <h2>Export</h2>\n\
         <button class=\"export-btn\" onclick=\"saveSvg()\">Save SVG</button>\n\
         <button class=\"export-btn\" onclick=\"saveTikz()\">Save TikZ (.tex)</button>\n\
         </div>\n",
    );

    html.push_str(
        "<div class=\"help\">Scroll to zoom<br>Drag to pan<br>Double-click to reset<br>\
         Hover for info<br>Click to select</div>\n",
    );
    html.push_str("</div>\n");
}

fn write_web_groups(
    html: &mut String,
    overlay: &WebOverlay,
    positions: &HashMap<V, (f64, f64)>,
    options: &SvgOptions,
) {
    let label_size = options.font_size * 0.85;
    let pad = 2.0;

    for (i, web) in overlay.webs.iter().enumerate() {
        let orig = overlay.original_index(i);
        let color = colors::WEB_COLORS[orig % colors::WEB_COLORS.len()];
        let opaque = colors::WEB_COLORS_OPAQUE[orig % colors::WEB_COLORS_OPAQUE.len()];
        let width = options.edge_width * 3.0;

        let _ = writeln!(html, "<g id=\"web-{i}\" class=\"web-layer\">");

        for (&(from, to), &pauli) in &web.edge_operators {
            if let (Some(&(x1, y1)), Some(&(x2, y2))) = (positions.get(&from), positions.get(&to)) {
                let _ = writeln!(
                    html,
                    "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" \
                     stroke=\"{color}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>"
                );

                if options.show_web_labels {
                    let mx = (x1 + x2) / 2.0;
                    let my = (y1 + y2) / 2.0;
                    let label = match pauli {
                        Pauli::X => "X",
                        Pauli::Y => "Y",
                        Pauli::Z => "Z",
                    };
                    let _ = writeln!(
                        html,
                        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"2\" \
                         fill=\"white\" opacity=\"0.85\"/>",
                        mx - label_size / 2.0 - pad,
                        my - label_size / 2.0 - pad,
                        label_size + pad * 2.0,
                        label_size + pad * 2.0,
                    );
                    let _ = writeln!(
                        html,
                        "<text x=\"{mx}\" y=\"{}\" font-size=\"{label_size}\" fill=\"{opaque}\" \
                         text-anchor=\"middle\" font-family=\"monospace\" \
                         font-weight=\"bold\">{label}</text>",
                        my + label_size / 3.0,
                    );
                }
            }
        }

        html.push_str("</g>\n");
    }
}

fn write_script(
    html: &mut String,
    options: &SvgOptions,
    tikz: &str,
    vertex_infos: &[(V, VertexJsInfo)],
) {
    let n = options.web_overlay.as_ref().map_or(0, |o| o.webs.len());
    let tikz_escaped = escape_js_string(tikz);

    // Emit palette data for both schemes
    let pecos = colors::ColorScheme::Pecos.palette();
    let zx = colors::ColorScheme::ZxCanonical.palette();

    html.push_str("<script>\n'use strict';\n");

    // Vertex info map for hover/click interactivity
    html.push_str("const vertices = {\n");
    for (v, info) in vertex_infos {
        let neighbors_js: Vec<String> = info.neighbors.iter().map(|n| n.to_string()).collect();
        let _ = writeln!(
            html,
            "  {v}:{{x:{x},y:{y},type:'{t}',phase:'{p}',degree:{d},boundary:'{b}',r:{r},\
             neighbors:[{nb}]}},",
            x = info.x,
            y = info.y,
            t = info.vtype,
            p = escape_js_string(&info.phase),
            d = info.degree,
            b = escape_js_string(&info.boundary),
            r = info.r,
            nb = neighbors_js.join(","),
        );
    }
    html.push_str("};\n\n");

    // Palette definitions
    write_palette_js(html, "pecos", pecos);
    write_palette_js(html, "zx", zx);

    let _ = write!(
        html,
        "const palettes = {{ pecos: palettePecos, zx: paletteZx }};\n\
         \n\
         function switchPalette(name) {{\n\
           const p = palettes[name];\n\
           if (!p) return;\n\
           const s = svg;\n\
           s.querySelectorAll('.zx-bg').forEach(e => e.setAttribute('fill', p.bg));\n\
           s.querySelectorAll('.zx-z').forEach(e => {{\n\
             e.setAttribute('fill', p.zFill); e.setAttribute('stroke', p.zStroke);\n\
           }});\n\
           s.querySelectorAll('.zx-x').forEach(e => {{\n\
             e.setAttribute('fill', p.xFill); e.setAttribute('stroke', p.xStroke);\n\
           }});\n\
           s.querySelectorAll('.zx-h').forEach(e => {{\n\
             e.setAttribute('fill', p.hFill); e.setAttribute('stroke', p.hStroke);\n\
           }});\n\
           s.querySelectorAll('.zx-b').forEach(e => {{\n\
             e.setAttribute('fill', p.bFill); e.setAttribute('stroke', p.bStroke);\n\
           }});\n\
           s.querySelectorAll('.zx-edge').forEach(e => e.setAttribute('stroke', p.edgeNormal));\n\
           s.querySelectorAll('.zx-hedge').forEach(e => e.setAttribute('stroke', p.edgeH));\n\
           s.querySelectorAll('.zx-hsquare').forEach(e => {{\n\
             e.setAttribute('fill', p.hSquare); e.setAttribute('stroke', p.edgeH);\n\
           }});\n\
           s.querySelectorAll('.zx-phase').forEach(e => e.setAttribute('fill', p.phaseText));\n\
           s.querySelectorAll('.zx-blabel').forEach(e => e.setAttribute('fill', p.labelText));\n\
         }}\n\
         \n"
    );

    let _ = write!(
        html,
        "const svg = document.getElementById('zx-svg');\n\
         const container = document.getElementById('svg-container');\n\
         const tooltip = document.getElementById('tooltip');\n\
         const selPanel = document.getElementById('selection-panel');\n\
         const selContent = document.getElementById('selection-content');\n\
         const webGroups = [];\n\
         for (let i = 0; i < {n}; i++) {{\n\
           webGroups.push(document.getElementById('web-' + i));\n\
         }}\n\
         \n\
         function toggleWeb(i) {{\n\
           const g = webGroups[i];\n\
           if (!g) return;\n\
           g.style.display = g.style.display === 'none' ? '' : 'none';\n\
         }}\n\
         \n\
         function changeWebColor(i, hex) {{\n\
           const g = webGroups[i];\n\
           if (!g) return;\n\
           const r = parseInt(hex.slice(1,3),16);\n\
           const gv = parseInt(hex.slice(3,5),16);\n\
           const b = parseInt(hex.slice(5,7),16);\n\
           const rgba = 'rgba('+r+','+gv+','+b+',0.4)';\n\
           g.querySelectorAll('line').forEach(e => e.setAttribute('stroke', rgba));\n\
           g.querySelectorAll('text').forEach(e => e.setAttribute('fill', hex));\n\
         }}\n\
         \n\
         // Export: Save SVG (respects current web visibility)\n\
         function saveSvg() {{\n\
           const clone = svg.cloneNode(true);\n\
           for (let i = 0; i < {n}; i++) {{\n\
             const orig = webGroups[i];\n\
             const g = clone.querySelector('#web-' + i);\n\
             if (orig && g && orig.style.display === 'none') {{\n\
               g.remove();\n\
             }}\n\
           }}\n\
           clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');\n\
           clone.removeAttribute('width');\n\
           clone.removeAttribute('height');\n\
           const blob = new Blob([clone.outerHTML], {{type: 'image/svg+xml'}});\n\
           downloadBlob(blob, 'zx-diagram.svg');\n\
         }}\n\
         \n\
         // Export: Save TikZ\n\
         const tikzData = '{tikz_escaped}';\n\
         function saveTikz() {{\n\
           const blob = new Blob([tikzData], {{type: 'text/plain'}});\n\
           downloadBlob(blob, 'zx-diagram.tex');\n\
         }}\n\
         \n\
         function downloadBlob(blob, filename) {{\n\
           const url = URL.createObjectURL(blob);\n\
           const a = document.createElement('a');\n\
           a.href = url;\n\
           a.download = filename;\n\
           a.click();\n\
           URL.revokeObjectURL(url);\n\
         }}\n\
         \n\
         // Pan and zoom\n\
         let viewBox = svg.viewBox.baseVal;\n\
         let isPanning = false;\n\
         let startX, startY;\n\
         \n\
         container.addEventListener('wheel', function(e) {{\n\
           e.preventDefault();\n\
           const scale = e.deltaY > 0 ? 1.1 : 0.9;\n\
           const pt = svgPoint(e);\n\
           viewBox.x = pt.x - (pt.x - viewBox.x) * scale;\n\
           viewBox.y = pt.y - (pt.y - viewBox.y) * scale;\n\
           viewBox.width *= scale;\n\
           viewBox.height *= scale;\n\
         }}, {{passive: false}});\n\
         \n\
         container.addEventListener('mousedown', function(e) {{\n\
           isPanning = true;\n\
           startX = e.clientX;\n\
           startY = e.clientY;\n\
           container.classList.add('dragging');\n\
         }});\n\
         \n\
         window.addEventListener('mousemove', function(e) {{\n\
           if (!isPanning) return;\n\
           const dx = (e.clientX - startX) * viewBox.width / container.clientWidth;\n\
           const dy = (e.clientY - startY) * viewBox.height / container.clientHeight;\n\
           viewBox.x -= dx;\n\
           viewBox.y -= dy;\n\
           startX = e.clientX;\n\
           startY = e.clientY;\n\
         }});\n\
         \n\
         window.addEventListener('mouseup', function() {{\n\
           isPanning = false;\n\
           container.classList.remove('dragging');\n\
         }});\n\
         \n\
         container.addEventListener('dblclick', function() {{\n\
           const bb = svg.getBBox();\n\
           viewBox.x = 0;\n\
           viewBox.y = 0;\n\
           viewBox.width = bb.x + bb.width + 40;\n\
           viewBox.height = bb.y + bb.height + 40;\n\
         }});\n\
         \n\
         function svgPoint(e) {{\n\
           const ctm = svg.getScreenCTM();\n\
           if (ctm) {{\n\
             return new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());\n\
           }}\n\
           // Fallback if getScreenCTM unavailable\n\
           const rect = container.getBoundingClientRect();\n\
           return {{\n\
             x: viewBox.x + (e.clientX - rect.left) / rect.width * viewBox.width,\n\
             y: viewBox.y + (e.clientY - rect.top) / rect.height * viewBox.height\n\
           }};\n\
         }}\n\
         \n"
    );

    // Hover tooltip and click-to-select interactivity
    html.push_str(
        "// Find nearest vertex to SVG coordinates within hit radius\n\
         function findVertex(sx, sy) {\n\
           let best = null, bestD = Infinity;\n\
           for (const id in vertices) {\n\
             const v = vertices[id];\n\
             const dx = sx - v.x, dy = sy - v.y;\n\
             const d = Math.sqrt(dx*dx + dy*dy);\n\
             if (d <= v.r * 1.5 && d < bestD) { best = id; bestD = d; }\n\
           }\n\
           return best;\n\
         }\n\
         \n\
         const typeNames = {Z:'Z spider',X:'X spider',H:'H box',B:'Boundary'};\n\
         let hoveredVid = null;\n\
         let selectedVid = null;\n\
         \n\
         // Hover tooltip\n\
         container.addEventListener('mousemove', function(e) {\n\
           if (isPanning) { tooltip.style.display = 'none'; return; }\n\
           const pt = svgPoint(e);\n\
           const vid = findVertex(pt.x, pt.y);\n\
           if (vid !== hoveredVid) {\n\
             if (hoveredVid !== null) {\n\
               const el = svg.querySelector('[data-vid=\"'+hoveredVid+'\"]');\n\
               if (el) el.classList.remove('vertex-hover');\n\
             }\n\
             hoveredVid = vid;\n\
             if (vid !== null) {\n\
               const el = svg.querySelector('[data-vid=\"'+vid+'\"]');\n\
               if (el) el.classList.add('vertex-hover');\n\
             }\n\
           }\n\
           if (vid !== null) {\n\
             const v = vertices[vid];\n\
             let html = '<b>' + typeNames[v.type] + '</b>';\n\
             if (v.phase) html += '<br>Phase: ' + v.phase;\n\
             html += '<br>Degree: ' + v.degree;\n\
             html += '<br>ID: ' + vid;\n\
             if (v.boundary) html += '<br>' + v.boundary;\n\
             tooltip.innerHTML = html;\n\
             tooltip.style.display = 'block';\n\
             tooltip.style.left = (e.clientX - container.getBoundingClientRect().left + 12) + 'px';\n\
             tooltip.style.top = (e.clientY - container.getBoundingClientRect().top - 10) + 'px';\n\
           } else {\n\
             tooltip.style.display = 'none';\n\
           }\n\
         });\n\
         \n\
         container.addEventListener('mouseleave', function() {\n\
           tooltip.style.display = 'none';\n\
           if (hoveredVid !== null) {\n\
             const el = svg.querySelector('[data-vid=\"'+hoveredVid+'\"]');\n\
             if (el) el.classList.remove('vertex-hover');\n\
             hoveredVid = null;\n\
           }\n\
         });\n\
         \n\
         // Click-to-select\n\
         function clearSelection() {\n\
           if (selectedVid !== null) {\n\
             const el = svg.querySelector('[data-vid=\"'+selectedVid+'\"]');\n\
             if (el) el.classList.remove('vertex-selected');\n\
             svg.querySelectorAll('.edge-highlight').forEach(e => e.classList.remove('edge-highlight'));\n\
           }\n\
           selectedVid = null;\n\
           selPanel.style.display = 'none';\n\
         }\n\
         \n\
         function selectVertex(vid) {\n\
           clearSelection();\n\
           if (vid === null) return;\n\
           selectedVid = vid;\n\
           const el = svg.querySelector('[data-vid=\"'+vid+'\"]');\n\
           if (el) el.classList.add('vertex-selected');\n\
           // Highlight connected edges\n\
           svg.querySelectorAll('line[data-from=\"'+vid+'\"], line[data-to=\"'+vid+'\"]')\n\
             .forEach(e => e.classList.add('edge-highlight'));\n\
           // Populate selection panel\n\
           const v = vertices[vid];\n\
           let h = '<b>Type:</b> ' + typeNames[v.type];\n\
           if (v.phase) h += '<br><b>Phase:</b> ' + v.phase;\n\
           h += '<br><b>Degree:</b> ' + v.degree;\n\
           h += '<br><b>ID:</b> ' + vid;\n\
           if (v.boundary) h += '<br><b>Boundary:</b> ' + v.boundary;\n\
           if (v.neighbors.length > 0) {\n\
             h += '<br><b>Neighbors:</b> ';\n\
             h += v.neighbors.map(function(nid) {\n\
               return '<a class=\"nb-link\" data-vid=\"'+nid+'\">'+nid+'</a>';\n\
             }).join(', ');\n\
           }\n\
           selContent.innerHTML = h;\n\
           selPanel.style.display = 'block';\n\
         }\n\
         \n\
         function panToVertex(vid) {\n\
           const v = vertices[vid];\n\
           if (!v) return;\n\
           viewBox.x = v.x - viewBox.width / 2;\n\
           viewBox.y = v.y - viewBox.height / 2;\n\
         }\n\
         \n\
         container.addEventListener('click', function(e) {\n\
           const pt = svgPoint(e);\n\
           const vid = findVertex(pt.x, pt.y);\n\
           if (vid !== null) selectVertex(vid);\n\
           else clearSelection();\n\
         });\n\
         \n\
         // Neighbor link clicks in selection panel\n\
         selPanel.addEventListener('click', function(e) {\n\
           const link = e.target.closest('.nb-link');\n\
           if (!link) return;\n\
           const vid = link.getAttribute('data-vid');\n\
           if (vid !== null && vertices[vid]) {\n\
             selectVertex(vid);\n\
             panToVertex(vid);\n\
           }\n\
         });\n\
         </script>\n",
    );
}

/// Emit a JS palette object definition for the given palette.
fn write_palette_js(html: &mut String, name: &str, p: &Palette) {
    // Capitalize first letter for the JS variable name
    let var_name = format!("palette{}{}", &name[..1].to_uppercase(), &name[1..]);
    let _ = writeln!(
        html,
        "const {var_name} = {{\n\
         \x20 zFill:'{zf}', zStroke:'{zs}',\n\
         \x20 xFill:'{xf}', xStroke:'{xs}',\n\
         \x20 hFill:'{hf}', hStroke:'{hs}',\n\
         \x20 bFill:'{bf}', bStroke:'{bs}',\n\
         \x20 edgeNormal:'{en}', edgeH:'{eh}', hSquare:'{hq}',\n\
         \x20 phaseText:'{pt}', labelText:'{lt}', bg:'{bg}'\n\
         }};",
        zf = p.z_fill,
        zs = p.z_stroke,
        xf = p.x_fill,
        xs = p.x_stroke,
        hf = p.h_fill,
        hs = p.h_stroke,
        bf = p.boundary_fill,
        bs = p.boundary_stroke,
        en = p.edge_normal,
        eh = p.edge_hadamard,
        hq = p.hadamard_square,
        pt = p.phase_text,
        lt = p.label_text,
        bg = p.background,
    );
}

/// Escape a string for embedding in a single-quoted JS string literal.
fn escape_js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            _ => out.push(c),
        }
    }
    out
}

// --- SVG element writers (simplified versions of svg.rs helpers) ---

#[allow(clippy::too_many_arguments)]
fn write_edge(
    html: &mut String,
    from: (f64, f64),
    to: (f64, f64),
    ety: EType,
    s: V,
    t: V,
    palette: &Palette,
    options: &SvgOptions,
) {
    let (x1, y1) = from;
    let (x2, y2) = to;
    let data = format!("data-from=\"{s}\" data-to=\"{t}\"");
    match ety {
        EType::H => {
            let _ = writeln!(
                html,
                "<line class=\"zx-hedge\" x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" \
                 stroke=\"{}\" stroke-width=\"{}\" stroke-dasharray=\"6,3\" {data}/>",
                palette.edge_hadamard, options.edge_width,
            );
            let mx = (x1 + x2) / 2.0;
            let my = (y1 + y2) / 2.0;
            let hs = options.hadamard_square_size;
            let _ = writeln!(
                html,
                "<rect class=\"zx-hsquare\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" \
                 fill=\"{}\" stroke=\"{}\"/>",
                mx - hs,
                my - hs,
                hs * 2.0,
                hs * 2.0,
                palette.hadamard_square,
                palette.edge_hadamard,
            );
        }
        _ => {
            let _ = writeln!(
                html,
                "<line class=\"zx-edge\" x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" \
                 stroke=\"{}\" stroke-width=\"{}\" {data}/>",
                palette.edge_normal, options.edge_width,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_vertex(
    html: &mut String,
    x: f64,
    y: f64,
    vtype: VType,
    v: V,
    degree: usize,
    phase_str: &str,
    boundary: &str,
    palette: &Palette,
    options: &SvgOptions,
) {
    let vtype_str = match vtype {
        VType::Z => "Z",
        VType::X => "X",
        VType::H => "H",
        VType::B => "B",
        _ => "B",
    };
    let boundary_attr = if boundary.is_empty() {
        String::new()
    } else {
        format!(" data-boundary=\"{boundary}\"")
    };
    let data = format!(
        "data-vid=\"{v}\" data-vtype=\"{vtype_str}\" data-degree=\"{degree}\" \
         data-phase=\"{phase_str}\"{boundary_attr}"
    );

    match vtype {
        VType::Z => {
            let _ = writeln!(
                html,
                "<circle class=\"zx-z\" cx=\"{x}\" cy=\"{y}\" r=\"{}\" fill=\"{}\" \
                 stroke=\"{}\" stroke-width=\"1.5\" {data}/>",
                options.spider_radius, palette.z_fill, palette.z_stroke,
            );
        }
        VType::X => {
            let _ = writeln!(
                html,
                "<circle class=\"zx-x\" cx=\"{x}\" cy=\"{y}\" r=\"{}\" fill=\"{}\" \
                 stroke=\"{}\" stroke-width=\"1.5\" {data}/>",
                options.spider_radius, palette.x_fill, palette.x_stroke,
            );
        }
        VType::H => {
            let s = options.h_box_size;
            let _ = writeln!(
                html,
                "<rect class=\"zx-h\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" \
                 fill=\"{}\" stroke=\"{}\" stroke-width=\"1.5\" {data}/>",
                x - s,
                y - s,
                s * 2.0,
                s * 2.0,
                palette.h_fill,
                palette.h_stroke,
            );
        }
        VType::B => {
            let _ = writeln!(
                html,
                "<circle class=\"zx-b\" cx=\"{x}\" cy=\"{y}\" r=\"{}\" fill=\"{}\" \
                 stroke=\"{}\" {data}/>",
                options.boundary_radius, palette.boundary_fill, palette.boundary_stroke,
            );
        }
        _ => {
            let s = options.boundary_radius;
            let _ = writeln!(
                html,
                "<rect class=\"zx-b\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" \
                 fill=\"{}\" stroke=\"{}\" {data}/>",
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

fn write_phase_label(
    html: &mut String,
    x: f64,
    y: f64,
    label: &str,
    palette: &Palette,
    options: &SvgOptions,
) {
    let _ = writeln!(
        html,
        "<text class=\"zx-phase\" x=\"{x}\" y=\"{}\" font-size=\"{}\" fill=\"{}\" \
         text-anchor=\"middle\" font-family=\"serif\">{}</text>",
        y - options.spider_radius - 4.0,
        options.font_size,
        palette.phase_text,
        label,
    );
}

fn write_boundary_label(
    html: &mut String,
    x: f64,
    y: f64,
    label: &str,
    palette: &Palette,
    options: &SvgOptions,
) {
    let _ = writeln!(
        html,
        "<text class=\"zx-blabel\" x=\"{x}\" y=\"{}\" font-size=\"{}\" fill=\"{}\" \
         text-anchor=\"middle\" font-family=\"monospace\">{label}</text>",
        y + options.boundary_radius + options.font_size + 2.0,
        options.font_size * 0.8,
        palette.label_text,
    );
}

// --- ZX rewrite exploration (dynamic JS-side engine) ---

/// Serialize a graph to a JS object literal for the browser-side rewrite engine.
///
/// Produces: `{vertices:{id:{type:'Z',phase:[num,den]},...}, edges:{"lo,hi":etype,...},
///            inputs:[...], outputs:[...]}`
fn emit_graph_data_js(graph: &impl GraphLike) -> String {
    let mut js = String::from("{vertices:{");
    for v in graph.vertices() {
        let vtype = match graph.vertex_type(v) {
            VType::Z => "Z",
            VType::X => "X",
            VType::H => "H",
            VType::B => "B",
            _ => "B",
        };
        let phase = graph.phase(v);
        let r = phase.to_rational();
        let (numer, denom) = (*r.numer(), *r.denom());
        let _ = write!(js, "{v}:{{type:'{vtype}',phase:[{numer},{denom}]}},");
    }
    js.push_str("},edges:{");
    for (s, t, ety) in graph.edges() {
        let (lo, hi) = if s < t { (s, t) } else { (t, s) };
        let etype = match ety {
            EType::H => 1,
            _ => 0,
        };
        let _ = write!(js, "'{lo},{hi}':{etype},");
    }
    js.push_str("},inputs:[");
    for (i, &v) in graph.inputs().iter().enumerate() {
        if i > 0 {
            js.push(',');
        }
        let _ = write!(js, "{v}");
    }
    js.push_str("],outputs:[");
    for (i, &v) in graph.outputs().iter().enumerate() {
        if i > 0 {
            js.push(',');
        }
        let _ = write!(js, "{v}");
    }
    js.push_str("]}");
    js
}

/// Serialize a positions map to a JS object literal: `{id:[x,y],...}`
fn emit_positions_js(positions: &HashMap<V, (f64, f64)>) -> String {
    let mut js = String::from("{");
    for (&v, &(x, y)) in positions {
        let _ = write!(js, "{v}:[{x},{y}],");
    }
    js.push('}');
    js
}

/// Serialize rendering options to a JS object literal for the browser-side renderer.
fn emit_render_options_js(options: &SvgOptions) -> String {
    format!(
        "{{spiderRadius:{},boundaryRadius:{},hBoxSize:{},fontSize:{},\
         edgeWidth:{},hSquareSize:{},showPhases:{},showBoundaryLabels:{}}}",
        options.spider_radius,
        options.boundary_radius,
        options.h_box_size,
        options.font_size,
        options.edge_width,
        options.hadamard_square_size,
        options.show_phases,
        options.show_boundary_labels,
    )
}

/// A pre-rendered state for a simplified graph (Rust-computed layout).
struct SimplifyState {
    label: String,
    svg_body: String,
    width: f64,
    height: f64,
    graph_js: String,
    positions_js: String,
}

/// Compute a simplification of the graph and render it.
fn compute_simplify_state(
    graph: &ZxGraph,
    label: &str,
    simplify_fn: fn(&mut ZxGraph) -> bool,
    options: &SvgOptions,
) -> SimplifyState {
    let mut g = graph.clone();
    let _ = simplify_fn(&mut g);

    let palette = options.color_scheme.palette();
    let positions = compute_layout(&g, options.layout, &options.layout_options);
    let (width, height) = compute_svg_dimensions(&positions, options);
    let (svg_body, _vertex_infos) =
        render_svg_body(&g, options, palette, &positions, width, height);

    SimplifyState {
        label: label.to_string(),
        svg_body,
        width,
        height,
        graph_js: emit_graph_data_js(&g),
        positions_js: emit_positions_js(&positions),
    }
}

/// Render a ZX graph as an interactive HTML file with dynamic rewrite chaining.
///
/// Takes `&ZxGraph` (concrete `vec_graph::Graph`) because simplifications need `Clone`.
/// The original graph and simplification results are pre-rendered as SVG with
/// Rust-computed layouts (high quality). A JavaScript rewrite engine embedded in the
/// HTML computes further rewrites dynamically, enabling unlimited rewrite chaining
/// with undo/redo navigation.
#[must_use]
pub fn render_html_with_rewrites(graph: &ZxGraph, options: &SvgOptions) -> String {
    let positions = compute_layout(graph, options.layout, &options.layout_options);
    let palette = options.color_scheme.palette();
    let (width, height) = compute_svg_dimensions(&positions, options);

    // Pre-render TikZ for export
    let tikz_opts = TikzOptions {
        color_scheme: options.color_scheme,
        ..TikzOptions::default()
    };
    let tikz_body = render_tikz(graph, &tikz_opts);
    let tikz_doc = standalone_document(&tikz_body, options.color_scheme);

    // Compute original state SVG body
    let (original_svg_body, _original_vertex_infos) =
        render_svg_body(graph, options, palette, &positions, width, height);

    // Web overlay HTML for original state
    let web_overlay_html = if let Some(ref overlay) = options.web_overlay {
        let mut web_html = String::new();
        write_web_groups(&mut web_html, overlay, &positions, options);
        web_html
    } else {
        String::new()
    };

    // Serialize graph data and positions for JS engine
    let initial_graph_js = emit_graph_data_js(graph);
    let initial_positions_js = emit_positions_js(&positions);
    let render_options_js = emit_render_options_js(options);

    // Pre-compute simplification states
    let clifford_state =
        compute_simplify_state(graph, "Clifford Simplify", simplify::clifford_simp, options);
    let full_state = compute_simplify_state(graph, "Full Simplify", simplify::full_simp, options);

    // Build HTML
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n");
    html.push_str("<title>ZX Diagram Viewer</title>\n");
    write_styles(&mut html);
    html.push_str("</head>\n<body>\n");

    // Control panel with rewrite section
    write_rewrite_control_panel(&mut html, options);

    // SVG container (initially shows original state)
    let _ = write!(
        html,
        "<div id=\"svg-container\" style=\"position:relative\">\n\
         <div id=\"tooltip\"></div>\n\
         <svg id=\"zx-svg\" xmlns=\"http://www.w3.org/2000/svg\" \
         width=\"100%\" height=\"100%\" viewBox=\"0 0 {width} {height}\">\n"
    );
    html.push_str(&original_svg_body);
    html.push_str(&web_overlay_html);
    html.push_str("</svg>\n</div>\n");

    // JavaScript with dynamic rewrite engine
    write_rewrite_script(
        &mut html,
        options,
        &tikz_doc,
        &initial_graph_js,
        &initial_positions_js,
        &render_options_js,
        &original_svg_body,
        &web_overlay_html,
        width,
        height,
        &clifford_state,
        &full_state,
    );

    html.push_str("</body>\n</html>");
    html
}

fn write_rewrite_control_panel(html: &mut String, options: &SvgOptions) {
    html.push_str("<div id=\"controls\">\n<h2>Pauli Webs</h2>\n");

    // Web toggles
    if let Some(ref overlay) = options.web_overlay {
        for (i, _web) in overlay.webs.iter().enumerate() {
            let orig = overlay.original_index(i);
            let opaque = colors::WEB_COLORS_OPAQUE[orig % colors::WEB_COLORS_OPAQUE.len()];
            let class_label = overlay
                .classifications
                .get(i)
                .map_or("", classification_label);

            let _ = writeln!(
                html,
                "<label class=\"web-toggle\">\
                 <input type=\"checkbox\" checked onchange=\"toggleWeb({i})\">\
                 <input type=\"color\" class=\"swatch\" id=\"swatch-{i}\" \
                  value=\"{opaque}\" onchange=\"changeWebColor({i}, this.value)\">\
                 <span><span class=\"web-label\">Web {orig}</span><br>\
                 <span class=\"web-class\">{class_label}</span></span>\
                 </label>"
            );
        }
    } else {
        html.push_str("<p style=\"color:#666;font-size:13px\">No web overlay data.</p>\n");
    }

    // Palette switcher
    let pecos_sel = if options.color_scheme == colors::ColorScheme::Pecos {
        " selected"
    } else {
        ""
    };
    let zx_sel = if options.color_scheme == colors::ColorScheme::ZxCanonical {
        " selected"
    } else {
        ""
    };
    let _ = write!(
        html,
        "<div class=\"palette-section\">\n\
         <h2>Color Scheme</h2>\n\
         <select class=\"palette-select\" onchange=\"switchPalette(this.value)\">\n\
         <option value=\"pecos\"{pecos_sel}>PECOS (RGB = XYZ)</option>\n\
         <option value=\"zx\"{zx_sel}>ZX Canonical</option>\n\
         </select>\n\
         </div>\n"
    );

    // Selection info panel
    html.push_str(
        "<div id=\"selection-panel\" class=\"selection-section\" style=\"display:none\">\n\
         <h2>Selection</h2>\n\
         <div id=\"selection-content\"></div>\n\
         </div>\n",
    );

    // Rewrite section with undo/redo
    html.push_str(
        "<div id=\"rewrite-panel\" class=\"rewrite-section\">\n\
         <h2>Rewrites</h2>\n\
         <div id=\"step-indicator\" class=\"state-indicator\">Step 1 of 1</div>\n\
         <div class=\"undo-redo\">\n\
         <button id=\"undo-btn\" onclick=\"undo()\" disabled>Undo</button>\n\
         <button id=\"redo-btn\" onclick=\"redo()\" disabled>Redo</button>\n\
         </div>\n\
         <div id=\"vertex-rewrites\" style=\"display:none\">\n\
         <p style=\"color:#888;font-size:12px;margin:8px 0 4px\">Available for selection:</p>\n\
         <div id=\"vertex-rewrite-list\"></div>\n\
         </div>\n\
         <div style=\"margin-top:8px\">\n\
         <p style=\"color:#888;font-size:12px;margin-bottom:4px\">Whole-graph:</p>\n\
         <button class=\"rewrite-btn\" onclick=\"applySimpState('clifford')\">\
         Clifford Simplify</button>\n\
         <button class=\"rewrite-btn\" onclick=\"applySimpState('full')\">\
         Full Simplify</button>\n\
         </div>\n\
         </div>\n",
    );

    // Export buttons
    html.push_str(
        "<div class=\"export-section\">\n\
         <h2>Export</h2>\n\
         <button class=\"export-btn\" onclick=\"saveSvg()\">Save SVG</button>\n\
         <button class=\"export-btn\" onclick=\"saveTikz()\">Save TikZ (.tex)</button>\n\
         </div>\n",
    );

    html.push_str(
        "<div class=\"help\">Scroll to zoom<br>Drag to pan<br>Double-click to reset<br>\
         Hover for info<br>Click to select<br>Click vertex for rewrites<br>\
         Ctrl+Z undo, Ctrl+Shift+Z redo</div>\n",
    );
    html.push_str("</div>\n");
}

#[allow(clippy::too_many_arguments)]
fn write_rewrite_script(
    html: &mut String,
    options: &SvgOptions,
    tikz: &str,
    initial_graph_js: &str,
    initial_positions_js: &str,
    render_options_js: &str,
    original_svg_body: &str,
    web_overlay_html: &str,
    width: f64,
    height: f64,
    clifford_state: &SimplifyState,
    full_state: &SimplifyState,
) {
    let n = options.web_overlay.as_ref().map_or(0, |o| o.webs.len());
    let tikz_escaped = escape_js_string(tikz);
    let pecos = colors::ColorScheme::Pecos.palette();
    let zx = colors::ColorScheme::ZxCanonical.palette();

    html.push_str("<script>\n'use strict';\n");

    // Embedded data from Rust
    let _ = writeln!(html, "const initialGraph = {};", initial_graph_js);
    let _ = writeln!(html, "const initialPositions = {};", initial_positions_js);
    let _ = writeln!(html, "const renderOpts = {};", render_options_js);
    let _ = writeln!(
        html,
        "const originalSvgBody = '{}';",
        escape_js_string(original_svg_body)
    );
    let _ = writeln!(
        html,
        "const webOverlayHtml = '{}';",
        escape_js_string(web_overlay_html)
    );
    let _ = writeln!(html, "const originalWidth = {width};");
    let _ = writeln!(html, "const originalHeight = {height};");

    // Simplification pre-computed states
    let _ = writeln!(
        html,
        "const simpStates = {{\n\
         clifford:{{label:'{}',svgBody:'{}',width:{},height:{},graph:{},positions:{}}},\n\
         full:{{label:'{}',svgBody:'{}',width:{},height:{},graph:{},positions:{}}}\n\
         }};",
        escape_js_string(&clifford_state.label),
        escape_js_string(&clifford_state.svg_body),
        clifford_state.width,
        clifford_state.height,
        clifford_state.graph_js,
        clifford_state.positions_js,
        escape_js_string(&full_state.label),
        escape_js_string(&full_state.svg_body),
        full_state.width,
        full_state.height,
        full_state.graph_js,
        full_state.positions_js,
    );

    // Palette definitions
    write_palette_js(html, "pecos", pecos);
    write_palette_js(html, "zx", zx);

    let _ = write!(
        html,
        "const palettes = {{ pecos: palettePecos, zx: paletteZx }};\n\
         let currentPalette = '{}';\n\n",
        if options.color_scheme == colors::ColorScheme::Pecos {
            "pecos"
        } else {
            "zx"
        }
    );

    // Write the JS rewrite engine
    write_js_rewrite_engine(html);

    // DOM setup, pan/zoom, hover/click, web toggles, export
    let _ = write!(
        html,
        "const svg = document.getElementById('zx-svg');\n\
         const container = document.getElementById('svg-container');\n\
         const tooltip = document.getElementById('tooltip');\n\
         const selPanel = document.getElementById('selection-panel');\n\
         const selContent = document.getElementById('selection-content');\n\
         const stepIndicator = document.getElementById('step-indicator');\n\
         const undoBtn = document.getElementById('undo-btn');\n\
         const redoBtn = document.getElementById('redo-btn');\n\
         const vertexRewriteDiv = document.getElementById('vertex-rewrites');\n\
         const vertexRewriteList = document.getElementById('vertex-rewrite-list');\n\
         let webGroups = [];\n\
         for (let i = 0; i < {n}; i++) {{\n\
           const g = document.getElementById('web-' + i);\n\
           if (g) webGroups.push(g);\n\
         }}\n\
         \n\
         function toggleWeb(i) {{\n\
           const g = webGroups[i];\n\
           if (!g) return;\n\
           g.style.display = g.style.display === 'none' ? '' : 'none';\n\
         }}\n\
         \n\
         function changeWebColor(i, hex) {{\n\
           const g = webGroups[i];\n\
           if (!g) return;\n\
           const r = parseInt(hex.slice(1,3),16);\n\
           const gv = parseInt(hex.slice(3,5),16);\n\
           const b = parseInt(hex.slice(5,7),16);\n\
           const rgba = 'rgba('+r+','+gv+','+b+',0.4)';\n\
           g.querySelectorAll('line').forEach(e => e.setAttribute('stroke', rgba));\n\
           g.querySelectorAll('text').forEach(e => e.setAttribute('fill', hex));\n\
         }}\n\
         \n"
    );

    // switchPalette
    html.push_str(
        "function switchPalette(name) {\n\
           const p = palettes[name];\n\
           if (!p) return;\n\
           currentPalette = name;\n\
           const s = svg;\n\
           s.querySelectorAll('.zx-bg').forEach(e => e.setAttribute('fill', p.bg));\n\
           s.querySelectorAll('.zx-z').forEach(e => {\n\
             e.setAttribute('fill', p.zFill); e.setAttribute('stroke', p.zStroke);\n\
           });\n\
           s.querySelectorAll('.zx-x').forEach(e => {\n\
             e.setAttribute('fill', p.xFill); e.setAttribute('stroke', p.xStroke);\n\
           });\n\
           s.querySelectorAll('.zx-h').forEach(e => {\n\
             e.setAttribute('fill', p.hFill); e.setAttribute('stroke', p.hStroke);\n\
           });\n\
           s.querySelectorAll('.zx-b').forEach(e => {\n\
             e.setAttribute('fill', p.bFill); e.setAttribute('stroke', p.bStroke);\n\
           });\n\
           s.querySelectorAll('.zx-edge').forEach(e => e.setAttribute('stroke', p.edgeNormal));\n\
           s.querySelectorAll('.zx-hedge').forEach(e => e.setAttribute('stroke', p.edgeH));\n\
           s.querySelectorAll('.zx-hsquare').forEach(e => {\n\
             e.setAttribute('fill', p.hSquare); e.setAttribute('stroke', p.edgeH);\n\
           });\n\
           s.querySelectorAll('.zx-phase').forEach(e => e.setAttribute('fill', p.phaseText));\n\
           s.querySelectorAll('.zx-blabel').forEach(e => e.setAttribute('fill', p.labelText));\n\
         }\n\n",
    );

    // Export functions
    let _ = write!(
        html,
        "function saveSvg() {{\n\
           const clone = svg.cloneNode(true);\n\
           clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');\n\
           clone.removeAttribute('width');\n\
           clone.removeAttribute('height');\n\
           const blob = new Blob([clone.outerHTML], {{type: 'image/svg+xml'}});\n\
           downloadBlob(blob, 'zx-diagram.svg');\n\
         }}\n\
         \n\
         const tikzData = '{tikz_escaped}';\n\
         function saveTikz() {{\n\
           const blob = new Blob([tikzData], {{type: 'text/plain'}});\n\
           downloadBlob(blob, 'zx-diagram.tex');\n\
         }}\n\
         \n\
         function downloadBlob(blob, filename) {{\n\
           const url = URL.createObjectURL(blob);\n\
           const a = document.createElement('a');\n\
           a.href = url;\n\
           a.download = filename;\n\
           a.click();\n\
           URL.revokeObjectURL(url);\n\
         }}\n\
         \n"
    );

    // Pan and zoom
    html.push_str(
        "let viewBox = svg.viewBox.baseVal;\n\
         let isPanning = false;\n\
         let startX, startY;\n\
         \n\
         container.addEventListener('wheel', function(e) {\n\
           e.preventDefault();\n\
           const scale = e.deltaY > 0 ? 1.1 : 0.9;\n\
           const pt = svgPoint(e);\n\
           viewBox.x = pt.x - (pt.x - viewBox.x) * scale;\n\
           viewBox.y = pt.y - (pt.y - viewBox.y) * scale;\n\
           viewBox.width *= scale;\n\
           viewBox.height *= scale;\n\
         }, {passive: false});\n\
         \n\
         container.addEventListener('mousedown', function(e) {\n\
           isPanning = true;\n\
           startX = e.clientX;\n\
           startY = e.clientY;\n\
           container.classList.add('dragging');\n\
         });\n\
         \n\
         window.addEventListener('mousemove', function(e) {\n\
           if (!isPanning) return;\n\
           const dx = (e.clientX - startX) * viewBox.width / container.clientWidth;\n\
           const dy = (e.clientY - startY) * viewBox.height / container.clientHeight;\n\
           viewBox.x -= dx;\n\
           viewBox.y -= dy;\n\
           startX = e.clientX;\n\
           startY = e.clientY;\n\
         });\n\
         \n\
         window.addEventListener('mouseup', function() {\n\
           isPanning = false;\n\
           container.classList.remove('dragging');\n\
         });\n\
         \n\
         container.addEventListener('dblclick', function() {\n\
           const bb = svg.getBBox();\n\
           viewBox.x = 0;\n\
           viewBox.y = 0;\n\
           viewBox.width = bb.x + bb.width + 40;\n\
           viewBox.height = bb.y + bb.height + 40;\n\
         });\n\
         \n\
         function svgPoint(e) {\n\
           const ctm = svg.getScreenCTM();\n\
           if (ctm) {\n\
             return new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());\n\
           }\n\
           const rect = container.getBoundingClientRect();\n\
           return {\n\
             x: viewBox.x + (e.clientX - rect.left) / rect.width * viewBox.width,\n\
             y: viewBox.y + (e.clientY - rect.top) / rect.height * viewBox.height\n\
           };\n\
         }\n\n",
    );

    // History stack and undo/redo
    write_js_history_engine(html);

    // Hover, click, and selection with dynamic rewrite integration
    write_js_interaction(html);

    html.push_str("</script>\n");
}

/// Emit the JS rewrite engine: graph class, phase arithmetic, 5 rewrite rules, SVG rendering.
fn write_js_rewrite_engine(html: &mut String) {
    html.push_str(
        "// --- Phase arithmetic (rational, exact, in half-turns) ---\n\
         function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) { [a,b] = [b, a%b]; } return a || 1; }\n\
         function phaseNew(n, d) {\n\
           if (d < 0) { n = -n; d = -d; }\n\
           const g = gcd(Math.abs(n), d); n /= g; d /= g;\n\
           // normalize to (-1,1] in half-turns: n/d in (-d, d]\n\
           if (d > 1) { n = ((n % (2*d)) + 2*d) % (2*d); if (n > d) n -= 2*d; }\n\
           else { n = ((n % 2) + 2) % 2; if (n > 1) n -= 2; }\n\
           return [n, d];\n\
         }\n\
         function phaseIsZero(p) { return p[0] === 0; }\n\
         function phaseIsPauli(p) { return p[0] === 0 || (Math.abs(p[0]) === 1 && p[1] === 1); }\n\
         function phaseIsProperClifford(p) {\n\
           return (Math.abs(p[0]) === 1 && p[1] === 2);\n\
         }\n\
         function phaseAdd(a, b) {\n\
           return phaseNew(a[0]*b[1] + b[0]*a[1], a[1]*b[1]);\n\
         }\n\
         function phaseNeg(p) { return phaseNew(-p[0], p[1]); }\n\
         function phaseAddPi(p) { return phaseAdd(p, [1,1]); }\n\
         function formatPhase(p) {\n\
           const [n, d] = p;\n\
           if (n === 0) return '';\n\
           if (n === 1 && d === 1) return 'pi';\n\
           if (n === -1 && d === 1) return '-pi';\n\
           if (n === 1) return 'pi/' + d;\n\
           if (n === -1) return '-pi/' + d;\n\
           if (d === 1) return n + 'pi';\n\
           return n + 'pi/' + d;\n\
         }\n\n",
    );

    html.push_str(
        "// --- Graph class ---\n\
         class ZXGraph {\n\
           constructor() {\n\
             this.verts = new Map();  // id -> {type, phase:[n,d]}\n\
             this.edges = new Map();  // 'lo,hi' -> etype (0=N, 1=H)\n\
             this.adj = new Map();    // id -> Set of neighbor ids\n\
             this.inputs = [];\n\
             this.outputs = [];\n\
             this.nextId = 0;\n\
           }\n\
           static fromData(data) {\n\
             const g = new ZXGraph();\n\
             let maxId = 0;\n\
             for (const id in data.vertices) {\n\
               const vid = Number(id);\n\
               const v = data.vertices[id];\n\
               g.verts.set(vid, {type: v.type, phase: [...v.phase]});\n\
               g.adj.set(vid, new Set());\n\
               if (vid >= maxId) maxId = vid + 1;\n\
             }\n\
             g.nextId = maxId;\n\
             for (const key in data.edges) {\n\
               const [s, t] = key.split(',').map(Number);\n\
               g.edges.set(key, data.edges[key]);\n\
               g.adj.get(s).add(t);\n\
               g.adj.get(t).add(s);\n\
             }\n\
             g.inputs = [...data.inputs];\n\
             g.outputs = [...data.outputs];\n\
             return g;\n\
           }\n\
           clone() {\n\
             const g = new ZXGraph();\n\
             for (const [id, v] of this.verts) g.verts.set(id, {type:v.type, phase:[...v.phase]});\n\
             for (const [k, e] of this.edges) g.edges.set(k, e);\n\
             for (const [id, s] of this.adj) g.adj.set(id, new Set(s));\n\
             g.inputs = [...this.inputs];\n\
             g.outputs = [...this.outputs];\n\
             g.nextId = this.nextId;\n\
             return g;\n\
           }\n\
           edgeKey(s, t) { return s < t ? s+','+t : t+','+s; }\n\
           hasVertex(v) { return this.verts.has(v); }\n\
           vertexType(v) { return this.verts.get(v).type; }\n\
           phase(v) { return this.verts.get(v).phase; }\n\
           setPhase(v, p) { this.verts.get(v).phase = p; }\n\
           addPhase(v, p) { this.setPhase(v, phaseAdd(this.phase(v), p)); }\n\
           degree(v) { return this.adj.has(v) ? this.adj.get(v).size : 0; }\n\
           neighbors(v) { return this.adj.has(v) ? this.adj.get(v) : new Set(); }\n\
           hasEdge(s, t) { return this.edges.has(this.edgeKey(s, t)); }\n\
           edgeType(s, t) { return this.edges.get(this.edgeKey(s, t)); }\n\
           addVertex(type, phase) {\n\
             const id = this.nextId++;\n\
             this.verts.set(id, {type, phase: phase || [0,1]});\n\
             this.adj.set(id, new Set());\n\
             return id;\n\
           }\n\
           removeVertex(v) {\n\
             for (const nb of this.neighbors(v)) {\n\
               this.edges.delete(this.edgeKey(v, nb));\n\
               this.adj.get(nb).delete(v);\n\
             }\n\
             this.adj.delete(v);\n\
             this.verts.delete(v);\n\
           }\n\
           addEdgeSimple(s, t, etype) {\n\
             const k = this.edgeKey(s, t);\n\
             this.edges.set(k, etype);\n\
             this.adj.get(s).add(t);\n\
             this.adj.get(t).add(s);\n\
           }\n\
           removeEdge(s, t) {\n\
             this.edges.delete(this.edgeKey(s, t));\n\
             if (this.adj.has(s)) this.adj.get(s).delete(t);\n\
             if (this.adj.has(t)) this.adj.get(t).delete(s);\n\
           }\n\
           // Smart edge addition: handles parallel edges per ZX calculus rules\n\
           addEdgeSmart(s, t, etype) {\n\
             if (s === t) {\n\
               // Self-loop: H self-loop adds pi\n\
               if (etype === 1) this.addPhase(s, [1,1]);\n\
               return;\n\
             }\n\
             if (!this.hasEdge(s, t)) {\n\
               this.addEdgeSimple(s, t, etype);\n\
               return;\n\
             }\n\
             const existing = this.edgeType(s, t);\n\
             const st = this.vertexType(s), tt = this.vertexType(t);\n\
             const sameColor = (st === tt);\n\
             if (sameColor) {\n\
               // Same color: N+N=keep, H+H=remove, N+H or H+N: set N + add pi\n\
               if (existing === 0 && etype === 0) { /* N+N: do nothing */ }\n\
               else if (existing === 1 && etype === 1) { this.removeEdge(s, t); }\n\
               else { // N+H or H+N\n\
                 const k = this.edgeKey(s, t);\n\
                 this.edges.set(k, 0); // set to N\n\
                 this.addPhase(s, [1,1]);\n\
               }\n\
             } else {\n\
               // Opposite color: N+N=remove, H+H=keep, N+H or H+N: set H + add pi\n\
               if (existing === 0 && etype === 0) { this.removeEdge(s, t); }\n\
               else if (existing === 1 && etype === 1) { /* H+H: do nothing */ }\n\
               else { // N+H or H+N\n\
                 const k = this.edgeKey(s, t);\n\
                 this.edges.set(k, 1); // set to H\n\
                 this.addPhase(s, [1,1]);\n\
               }\n\
             }\n\
           }\n\
         }\n\n",
    );

    // Five rewrite rule check/apply functions
    html.push_str(
        "// --- Rewrite rules ---\n\
         function checkSpiderFusion(g, v0, v1) {\n\
           if (v0 === v1) return false;\n\
           if (!g.hasEdge(v0, v1)) return false;\n\
           if (g.edgeType(v0, v1) !== 0) return false; // must be Normal\n\
           const t0 = g.vertexType(v0), t1 = g.vertexType(v1);\n\
           return (t0 === 'Z' || t0 === 'X') && t0 === t1;\n\
         }\n\
         function applySpiderFusion(g, v0, v1) {\n\
           for (const nb of g.neighbors(v1)) {\n\
             if (nb === v0) continue;\n\
             g.addEdgeSmart(v0, nb, g.edgeType(v1, nb));\n\
           }\n\
           g.addPhase(v0, g.phase(v1));\n\
           g.removeVertex(v1);\n\
         }\n\n\
         function checkRemoveId(g, v) {\n\
           const t = g.vertexType(v);\n\
           if (t !== 'Z' && t !== 'X') return false;\n\
           if (!phaseIsZero(g.phase(v))) return false;\n\
           return g.degree(v) === 2;\n\
         }\n\
         function applyRemoveId(g, v) {\n\
           const nbs = [...g.neighbors(v)];\n\
           const v0 = nbs[0], v1 = nbs[1];\n\
           const et0 = g.edgeType(v, v0), et1 = g.edgeType(v, v1);\n\
           const newEt = et0 ^ et1; // XOR: N+N=N, N+H=H, H+H=N\n\
           g.removeVertex(v);\n\
           g.addEdgeSmart(v0, v1, newEt);\n\
         }\n\n\
         function checkLocalComp(g, v) {\n\
           if (g.vertexType(v) !== 'Z') return false;\n\
           if (!phaseIsProperClifford(g.phase(v))) return false;\n\
           for (const nb of g.neighbors(v)) {\n\
             if (g.vertexType(nb) !== 'Z') return false;\n\
             if (g.edgeType(v, nb) !== 1) return false; // must be H\n\
           }\n\
           return true;\n\
         }\n\
         function applyLocalComp(g, v) {\n\
           const p = g.phase(v);\n\
           const nbs = [...g.neighbors(v)];\n\
           for (const nb of nbs) g.addPhase(nb, phaseNeg(p));\n\
           // Complete graph among neighbors with H edges\n\
           for (let i = 0; i < nbs.length; i++) {\n\
             for (let j = i+1; j < nbs.length; j++) {\n\
               g.addEdgeSmart(nbs[i], nbs[j], 1);\n\
             }\n\
           }\n\
           g.removeVertex(v);\n\
         }\n\n\
         function checkPivot(g, v0, v1) {\n\
           if (g.vertexType(v0) !== 'Z' || g.vertexType(v1) !== 'Z') return false;\n\
           if (!phaseIsPauli(g.phase(v0)) || !phaseIsPauli(g.phase(v1))) return false;\n\
           if (!g.hasEdge(v0, v1) || g.edgeType(v0, v1) !== 1) return false;\n\
           for (const nb of g.neighbors(v0)) {\n\
             if (g.vertexType(nb) !== 'Z') return false;\n\
             if (g.edgeType(v0, nb) !== 1) return false;\n\
           }\n\
           for (const nb of g.neighbors(v1)) {\n\
             if (g.vertexType(nb) !== 'Z') return false;\n\
             if (g.edgeType(v1, nb) !== 1) return false;\n\
           }\n\
           return true;\n\
         }\n\
         function applyPivot(g, v0, v1) {\n\
           const p0 = g.phase(v0), p1 = g.phase(v1);\n\
           const ns0 = [...g.neighbors(v0)], ns1 = [...g.neighbors(v1)];\n\
           // Bipartite complement\n\
           for (const n0 of ns0) {\n\
             if (n0 === v1) continue;\n\
             g.addPhase(n0, p1);\n\
             for (const n1 of ns1) {\n\
               if (n1 === v0 || n1 === n0) continue;\n\
               g.addEdgeSmart(n0, n1, 1);\n\
             }\n\
           }\n\
           for (const n1 of ns1) {\n\
             if (n1 === v0) continue;\n\
             g.addPhase(n1, p0);\n\
           }\n\
           g.removeVertex(v0);\n\
           g.removeVertex(v1);\n\
         }\n\n\
         function checkPiCopy(g, v) {\n\
           const vt = g.vertexType(v);\n\
           if (vt !== 'Z' && vt !== 'X') return false;\n\
           if (g.degree(v) === 0) return false;\n\
           const opp = vt === 'Z' ? 'X' : 'Z';\n\
           for (const nb of g.neighbors(v)) {\n\
             const et = g.edgeType(v, nb);\n\
             const nt = g.vertexType(nb);\n\
             if (et === 0 && nt !== opp) return false;\n\
             if (et === 1 && nt !== vt) return false;\n\
           }\n\
           return true;\n\
         }\n\
         function applyPiCopy(g, v) {\n\
           g.setPhase(v, phaseNeg(g.phase(v)));\n\
           for (const nb of g.neighbors(v)) g.addPhase(nb, [1,1]);\n\
         }\n\n",
    );

    // findRewrites function
    html.push_str(
        "function findRewrites(g) {\n\
           const rws = [];\n\
           for (const [v] of g.verts) {\n\
             if (checkRemoveId(g, v)) rws.push({rule:'removeId', v0:v, v1:null,\n\
               label:'Remove Id (v'+v+')'});\n\
             if (checkLocalComp(g, v)) rws.push({rule:'localComp', v0:v, v1:null,\n\
               label:'Local Comp (v'+v+')'});\n\
             if (checkPiCopy(g, v)) rws.push({rule:'piCopy', v0:v, v1:null,\n\
               label:'Pi Copy (v'+v+')'});\n\
           }\n\
           for (const [key] of g.edges) {\n\
             const [s, t] = key.split(',').map(Number);\n\
             if (checkSpiderFusion(g, s, t)) rws.push({rule:'spiderFusion', v0:s, v1:t,\n\
               label:'Spider Fusion (v'+s+', v'+t+')'});\n\
             if (checkPivot(g, s, t)) rws.push({rule:'pivot', v0:s, v1:t,\n\
               label:'Pivot (v'+s+', v'+t+')'});\n\
           }\n\
           return rws;\n\
         }\n\
         function applyRuleToGraph(g, rw) {\n\
           switch(rw.rule) {\n\
             case 'spiderFusion': applySpiderFusion(g, rw.v0, rw.v1); break;\n\
             case 'removeId': applyRemoveId(g, rw.v0); break;\n\
             case 'localComp': applyLocalComp(g, rw.v0); break;\n\
             case 'pivot': applyPivot(g, rw.v0, rw.v1); break;\n\
             case 'piCopy': applyPiCopy(g, rw.v0); break;\n\
           }\n\
         }\n\n",
    );

    // SVG rendering function
    html.push_str(
        "// --- JS SVG renderer ---\n\
         function renderGraphSvg(graph, positions, opts) {\n\
           const p = palettes[currentPalette];\n\
           let maxX = 0, maxY = 0;\n\
           for (const id in positions) {\n\
             const [x, y] = positions[id];\n\
             if (x > maxX) maxX = x;\n\
             if (y > maxY) maxY = y;\n\
           }\n\
           const pad = 50;\n\
           const w = maxX + pad, h = maxY + pad;\n\
           let svg = '<rect class=\"zx-bg\" width=\"'+w+'\" height=\"'+h+'\" fill=\"'+p.bg+'\"/>';\n\
           svg += '<g id=\"base-graph\">';\n\
           // Build input/output sets\n\
           const inputSet = {}; graph.inputs.forEach((v,i) => inputSet[v] = i);\n\
           const outputSet = {}; graph.outputs.forEach((v,i) => outputSet[v] = i);\n\
           // Edges\n\
           for (const [key, etype] of graph.edges) {\n\
             const [s, t] = key.split(',').map(Number);\n\
             const ps = positions[s], pt = positions[t];\n\
             if (!ps || !pt) continue;\n\
             if (etype === 1) {\n\
               svg += '<line class=\"zx-hedge\" x1=\"'+ps[0]+'\" y1=\"'+ps[1]+\n\
                 '\" x2=\"'+pt[0]+'\" y2=\"'+pt[1]+'\" stroke=\"'+p.edgeH+\n\
                 '\" stroke-width=\"'+opts.edgeWidth+'\" stroke-dasharray=\"6,3\"'+\n\
                 ' data-from=\"'+s+'\" data-to=\"'+t+'\"/>';\n\
               const mx = (ps[0]+pt[0])/2, my = (ps[1]+pt[1])/2;\n\
               const hs = opts.hSquareSize;\n\
               svg += '<rect class=\"zx-hsquare\" x=\"'+(mx-hs)+'\" y=\"'+(my-hs)+\n\
                 '\" width=\"'+(hs*2)+'\" height=\"'+(hs*2)+'\" fill=\"'+p.hSquare+\n\
                 '\" stroke=\"'+p.edgeH+'\"/>';\n\
             } else {\n\
               svg += '<line class=\"zx-edge\" x1=\"'+ps[0]+'\" y1=\"'+ps[1]+\n\
                 '\" x2=\"'+pt[0]+'\" y2=\"'+pt[1]+'\" stroke=\"'+p.edgeNormal+\n\
                 '\" stroke-width=\"'+opts.edgeWidth+'\" data-from=\"'+s+'\" data-to=\"'+t+'\"/>';\n\
             }\n\
           }\n\
           // Vertices\n\
           for (const [v, vd] of graph.verts) {\n\
             const pos = positions[v];\n\
             if (!pos) continue;\n\
             const [x, y] = pos;\n\
             const phStr = formatPhase(vd.phase);\n\
             let boundary = '';\n\
             if (vd.type === 'B') {\n\
               if (v in inputSet) boundary = 'in:'+inputSet[v];\n\
               else if (v in outputSet) boundary = 'out:'+outputSet[v];\n\
             }\n\
             const battr = boundary ? ' data-boundary=\"'+boundary+'\"' : '';\n\
             const data = 'data-vid=\"'+v+'\" data-vtype=\"'+vd.type+\n\
               '\" data-degree=\"'+graph.degree(v)+'\" data-phase=\"'+phStr+'\"'+battr;\n\
             if (vd.type === 'Z') {\n\
               svg += '<circle class=\"zx-z\" cx=\"'+x+'\" cy=\"'+y+'\" r=\"'+opts.spiderRadius+\n\
                 '\" fill=\"'+p.zFill+'\" stroke=\"'+p.zStroke+'\" stroke-width=\"1.5\" '+data+'/>';\n\
             } else if (vd.type === 'X') {\n\
               svg += '<circle class=\"zx-x\" cx=\"'+x+'\" cy=\"'+y+'\" r=\"'+opts.spiderRadius+\n\
                 '\" fill=\"'+p.xFill+'\" stroke=\"'+p.xStroke+'\" stroke-width=\"1.5\" '+data+'/>';\n\
             } else if (vd.type === 'H') {\n\
               const s = opts.hBoxSize;\n\
               svg += '<rect class=\"zx-h\" x=\"'+(x-s)+'\" y=\"'+(y-s)+'\" width=\"'+(s*2)+\n\
                 '\" height=\"'+(s*2)+'\" fill=\"'+p.hFill+'\" stroke=\"'+p.hStroke+\n\
                 '\" stroke-width=\"1.5\" '+data+'/>';\n\
             } else {\n\
               svg += '<circle class=\"zx-b\" cx=\"'+x+'\" cy=\"'+y+'\" r=\"'+opts.boundaryRadius+\n\
                 '\" fill=\"'+p.bFill+'\" stroke=\"'+p.bStroke+'\" '+data+'/>';\n\
             }\n\
             // Phase label\n\
             if (opts.showPhases && phStr && vd.type !== 'B') {\n\
               svg += '<text class=\"zx-phase\" x=\"'+x+'\" y=\"'+(y-opts.spiderRadius-4)+\n\
                 '\" font-size=\"'+opts.fontSize+'\" fill=\"'+p.phaseText+\n\
                 '\" text-anchor=\"middle\" font-family=\"serif\">'+phStr+'</text>';\n\
             }\n\
             // Boundary label\n\
             if (opts.showBoundaryLabels && vd.type === 'B' && boundary) {\n\
               const lbl = (v in inputSet) ? 'in['+inputSet[v]+']' : 'out['+outputSet[v]+']';\n\
               svg += '<text class=\"zx-blabel\" x=\"'+x+'\" y=\"'+(y+opts.boundaryRadius+opts.fontSize+2)+\n\
                 '\" font-size=\"'+(opts.fontSize*0.8)+'\" fill=\"'+p.labelText+\n\
                 '\" text-anchor=\"middle\" font-family=\"monospace\">'+lbl+'</text>';\n\
             }\n\
           }\n\
           svg += '</g>';\n\
           return {svg, width: w, height: h};\n\
         }\n\n",
    );
}

/// Emit JS history stack, undo/redo, and state management.
fn write_js_history_engine(html: &mut String) {
    html.push_str(
        "// --- History stack ---\n\
         // Each entry: {graph, positions, svgBody, width, height, label, isPreRendered, webHtml}\n\
         const history = [];\n\
         let historyIndex = -1;\n\
         \n\
         function buildVertexInfo(graph, positions) {\n\
           const info = {};\n\
           const inputSet = {}; graph.inputs.forEach((v,i) => inputSet[v] = i);\n\
           const outputSet = {}; graph.outputs.forEach((v,i) => outputSet[v] = i);\n\
           for (const [v, vd] of graph.verts) {\n\
             const pos = positions[v];\n\
             if (!pos) continue;\n\
             const phStr = formatPhase(vd.phase);\n\
             let boundary = '';\n\
             if (vd.type === 'B') {\n\
               if (v in inputSet) boundary = 'in:'+inputSet[v];\n\
               else if (v in outputSet) boundary = 'out:'+outputSet[v];\n\
             }\n\
             const r = (vd.type === 'Z' || vd.type === 'X') ? renderOpts.spiderRadius :\n\
                       vd.type === 'H' ? renderOpts.hBoxSize : renderOpts.boundaryRadius;\n\
             info[v] = {x:pos[0], y:pos[1], type:vd.type, phase:phStr,\n\
                        degree:graph.degree(v), boundary, r,\n\
                        neighbors:[...graph.neighbors(v)]};\n\
           }\n\
           return info;\n\
         }\n\
         \n\
         function pushPreRendered(svgBody, graph, positions, w, h, label, webHtml) {\n\
           // Truncate redo history\n\
           history.length = historyIndex + 1;\n\
           history.push({graph, positions, svgBody, width:w, height:h,\n\
                         label, isPreRendered:true, webHtml: webHtml||''});\n\
           historyIndex = history.length - 1;\n\
         }\n\
         \n\
         function pushJsRendered(graph, positions, label) {\n\
           history.length = historyIndex + 1;\n\
           const r = renderGraphSvg(graph, positions, renderOpts);\n\
           history.push({graph, positions, svgBody:r.svg, width:r.width, height:r.height,\n\
                         label, isPreRendered:false, webHtml:''});\n\
           historyIndex = history.length - 1;\n\
         }\n\
         \n\
         let vertices = {};\n\
         let hoveredVid = null;\n\
         let selectedVid = null;\n\
         \n\
         function renderCurrentState() {\n\
           const state = history[historyIndex];\n\
           svg.innerHTML = state.svgBody + state.webHtml;\n\
           svg.setAttribute('viewBox', '0 0 ' + state.width + ' ' + state.height);\n\
           viewBox = svg.viewBox.baseVal;\n\
           vertices = buildVertexInfo(state.graph, state.positions);\n\
           hoveredVid = null;\n\
           selectedVid = null;\n\
           selPanel.style.display = 'none';\n\
           vertexRewriteDiv.style.display = 'none';\n\
           stepIndicator.textContent = 'Step ' + (historyIndex+1) + ' of ' + history.length;\n\
           stepIndicator.className = historyIndex === 0\n\
             ? 'state-indicator' : 'state-indicator viewing-rewrite';\n\
           undoBtn.disabled = historyIndex === 0;\n\
           redoBtn.disabled = historyIndex === history.length - 1;\n\
           // Re-attach web groups for original state\n\
           webGroups = [];\n\
           if (state.webHtml) {\n\
             for (let i = 0; ; i++) {\n\
               const g = document.getElementById('web-' + i);\n\
               if (!g) break;\n\
               webGroups.push(g);\n\
             }\n\
             for (let i = 0; i < webGroups.length; i++) {\n\
               const cb = document.querySelector(\n\
                 'input[onchange=\"toggleWeb(' + i + ')\"]');\n\
               if (cb && !cb.checked && webGroups[i]) {\n\
                 webGroups[i].style.display = 'none';\n\
               }\n\
             }\n\
           }\n\
         }\n\
         \n\
         function undo() {\n\
           if (historyIndex > 0) { historyIndex--; renderCurrentState(); }\n\
         }\n\
         function redo() {\n\
           if (historyIndex < history.length - 1) { historyIndex++; renderCurrentState(); }\n\
         }\n\
         \n\
         function applyRewrite(rwIdx) {\n\
           const state = history[historyIndex];\n\
           const rws = findRewrites(state.graph);\n\
           if (rwIdx < 0 || rwIdx >= rws.length) return;\n\
           const rw = rws[rwIdx];\n\
           const newGraph = state.graph.clone();\n\
           applyRuleToGraph(newGraph, rw);\n\
           // Inherit positions from parent; removed vertices get dropped\n\
           const newPos = {};\n\
           for (const [v] of newGraph.verts) {\n\
             if (state.positions[v]) newPos[v] = [...state.positions[v]];\n\
             else {\n\
               // New vertex (shouldn't happen in these rules, but fallback)\n\
               newPos[v] = [50 + Math.random()*200, 50 + Math.random()*200];\n\
             }\n\
           }\n\
           pushJsRendered(newGraph, newPos, rw.label);\n\
           renderCurrentState();\n\
         }\n\
         \n\
         function applySimpState(key) {\n\
           const ss = simpStates[key];\n\
           if (!ss) return;\n\
           const graph = ZXGraph.fromData(ss.graph);\n\
           pushPreRendered(ss.svgBody, graph, ss.positions, ss.width, ss.height, ss.label, '');\n\
           renderCurrentState();\n\
         }\n\
         \n\
         // Initialize with original state\n\
         pushPreRendered(originalSvgBody, ZXGraph.fromData(initialGraph),\n\
           initialPositions, originalWidth, originalHeight, 'Original', webOverlayHtml);\n\
         renderCurrentState();\n\
         \n\
         // Keyboard shortcuts\n\
         document.addEventListener('keydown', function(e) {\n\
           if (e.ctrlKey || e.metaKey) {\n\
             if (e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }\n\
             else if (e.key === 'z' && e.shiftKey) { e.preventDefault(); redo(); }\n\
             else if (e.key === 'y') { e.preventDefault(); redo(); }\n\
           }\n\
         });\n\n",
    );
}

/// Emit JS hover, click, selection, and rewrite list interaction.
fn write_js_interaction(html: &mut String) {
    html.push_str(
        "// --- Interaction ---\n\
         function findVertex(sx, sy) {\n\
           let best = null, bestD = Infinity;\n\
           for (const id in vertices) {\n\
             const v = vertices[id];\n\
             const dx = sx - v.x, dy = sy - v.y;\n\
             const d = Math.sqrt(dx*dx + dy*dy);\n\
             if (d <= v.r * 1.5 && d < bestD) { best = id; bestD = d; }\n\
           }\n\
           return best;\n\
         }\n\
         \n\
         const typeNames = {Z:'Z spider',X:'X spider',H:'H box',B:'Boundary'};\n\
         \n\
         container.addEventListener('mousemove', function(e) {\n\
           if (isPanning) { tooltip.style.display = 'none'; return; }\n\
           const pt = svgPoint(e);\n\
           const vid = findVertex(pt.x, pt.y);\n\
           if (vid !== hoveredVid) {\n\
             if (hoveredVid !== null) {\n\
               const el = svg.querySelector('[data-vid=\"'+hoveredVid+'\"]');\n\
               if (el) el.classList.remove('vertex-hover');\n\
             }\n\
             hoveredVid = vid;\n\
             if (vid !== null) {\n\
               const el = svg.querySelector('[data-vid=\"'+vid+'\"]');\n\
               if (el) el.classList.add('vertex-hover');\n\
             }\n\
           }\n\
           if (vid !== null) {\n\
             const v = vertices[vid];\n\
             let ht = '<b>' + typeNames[v.type] + '</b>';\n\
             if (v.phase) ht += '<br>Phase: ' + v.phase;\n\
             ht += '<br>Degree: ' + v.degree;\n\
             ht += '<br>ID: ' + vid;\n\
             if (v.boundary) ht += '<br>' + v.boundary;\n\
             tooltip.innerHTML = ht;\n\
             tooltip.style.display = 'block';\n\
             tooltip.style.left = (e.clientX - container.getBoundingClientRect().left + 12) + 'px';\n\
             tooltip.style.top = (e.clientY - container.getBoundingClientRect().top - 10) + 'px';\n\
           } else {\n\
             tooltip.style.display = 'none';\n\
           }\n\
         });\n\
         \n\
         container.addEventListener('mouseleave', function() {\n\
           tooltip.style.display = 'none';\n\
           if (hoveredVid !== null) {\n\
             const el = svg.querySelector('[data-vid=\"'+hoveredVid+'\"]');\n\
             if (el) el.classList.remove('vertex-hover');\n\
             hoveredVid = null;\n\
           }\n\
         });\n\
         \n\
         function clearSelection() {\n\
           if (selectedVid !== null) {\n\
             const el = svg.querySelector('[data-vid=\"'+selectedVid+'\"]');\n\
             if (el) el.classList.remove('vertex-selected');\n\
             svg.querySelectorAll('.edge-highlight').forEach(e => e.classList.remove('edge-highlight'));\n\
           }\n\
           selectedVid = null;\n\
           selPanel.style.display = 'none';\n\
           vertexRewriteDiv.style.display = 'none';\n\
         }\n\
         \n\
         function selectVertex(vid) {\n\
           clearSelection();\n\
           if (vid === null) return;\n\
           selectedVid = vid;\n\
           const el = svg.querySelector('[data-vid=\"'+vid+'\"]');\n\
           if (el) el.classList.add('vertex-selected');\n\
           svg.querySelectorAll('line[data-from=\"'+vid+'\"], line[data-to=\"'+vid+'\"]')\n\
             .forEach(e => e.classList.add('edge-highlight'));\n\
           const v = vertices[vid];\n\
           let h = '<b>Type:</b> ' + typeNames[v.type];\n\
           if (v.phase) h += '<br><b>Phase:</b> ' + v.phase;\n\
           h += '<br><b>Degree:</b> ' + v.degree;\n\
           h += '<br><b>ID:</b> ' + vid;\n\
           if (v.boundary) h += '<br><b>Boundary:</b> ' + v.boundary;\n\
           if (v.neighbors.length > 0) {\n\
             h += '<br><b>Neighbors:</b> ';\n\
             h += v.neighbors.map(function(nid) {\n\
               return '<a class=\"nb-link\" data-vid=\"'+nid+'\">'+nid+'</a>';\n\
             }).join(', ');\n\
           }\n\
           selContent.innerHTML = h;\n\
           selPanel.style.display = 'block';\n\
           // Dynamic rewrite list for current graph state\n\
           const state = history[historyIndex];\n\
           const rws = findRewrites(state.graph);\n\
           const vidNum = Number(vid);\n\
           const matching = rws.map((rw, i) => ({rw, i}))\n\
             .filter(({rw}) => rw.v0 === vidNum || rw.v1 === vidNum);\n\
           if (matching.length > 0) {\n\
             let btns = '';\n\
             for (const {rw, i} of matching) {\n\
               btns += '<button class=\"rewrite-btn\" onclick=\"applyRewrite('+i+')\">' +\n\
                 rw.label + '</button>';\n\
             }\n\
             vertexRewriteList.innerHTML = btns;\n\
             vertexRewriteDiv.style.display = 'block';\n\
           } else {\n\
             vertexRewriteDiv.style.display = 'none';\n\
           }\n\
         }\n\
         \n\
         function panToVertex(vid) {\n\
           const v = vertices[vid];\n\
           if (!v) return;\n\
           viewBox.x = v.x - viewBox.width / 2;\n\
           viewBox.y = v.y - viewBox.height / 2;\n\
         }\n\
         \n\
         container.addEventListener('click', function(e) {\n\
           const pt = svgPoint(e);\n\
           const vid = findVertex(pt.x, pt.y);\n\
           if (vid !== null) selectVertex(vid);\n\
           else clearSelection();\n\
         });\n\
         \n\
         selPanel.addEventListener('click', function(e) {\n\
           const link = e.target.closest('.nb-link');\n\
           if (!link) return;\n\
           const vid = link.getAttribute('data-vid');\n\
           if (vid !== null && vertices[vid]) {\n\
             selectVertex(vid);\n\
             panToVertex(vid);\n\
           }\n\
         });\n",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli_web::WebClassification;
    use quizx::detection_webs::{Pauli as QPauli, PauliWeb};

    #[test]
    fn test_html_contains_structure() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html(&g, &SvgOptions::default());
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("</svg>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_html_web_toggles() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let mut web = PauliWeb::new();
        web.set_edge(b0, z, QPauli::X);

        let opts = SvgOptions {
            web_overlay: Some(WebOverlay {
                webs: vec![web],
                classifications: vec![WebClassification::Detector],
                indices: None,
            }),
            ..SvgOptions::default()
        };

        let html = render_html(&g, &opts);
        assert!(html.contains("id=\"web-0\""), "should have web group");
        assert!(html.contains("toggleWeb(0)"), "should have toggle JS");
        assert!(html.contains("Detector"), "should show classification");
    }

    #[test]
    fn test_html_pan_zoom_script() {
        use quizx::vec_graph::Graph;
        let g = Graph::new();
        let html = render_html(&g, &SvgOptions::default());
        assert!(html.contains("addEventListener('wheel'"));
        assert!(html.contains("isPanning"));
    }

    #[test]
    fn test_html_web_color_picker() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let mut web = PauliWeb::new();
        web.set_edge(b0, z, QPauli::Z);

        let opts = SvgOptions {
            web_overlay: Some(WebOverlay {
                webs: vec![web],
                classifications: vec![WebClassification::Detector],
                indices: None,
            }),
            ..SvgOptions::default()
        };

        let html = render_html(&g, &opts);
        assert!(
            html.contains("type=\"color\""),
            "should have color picker input"
        );
        assert!(
            html.contains("changeWebColor(0"),
            "should have color change handler"
        );
        assert!(
            html.contains("function changeWebColor"),
            "should have changeWebColor function"
        );
    }

    #[test]
    fn test_html_palette_switcher() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html(&g, &SvgOptions::default());
        // Has the dropdown
        assert!(html.contains("switchPalette(this.value)"));
        assert!(html.contains("PECOS (RGB = XYZ)"));
        assert!(html.contains("ZX Canonical"));
        // Has palette JS objects with both schemes' Z-spider colors
        assert!(html.contains("#6495ED"), "PECOS z_fill");
        assert!(html.contains("#98FB98"), "ZX canonical z_fill");
        // Has the switchPalette function
        assert!(html.contains("function switchPalette"));
        // SVG elements have classes
        assert!(html.contains("class=\"zx-z\""));
        assert!(html.contains("class=\"zx-b\""));
        assert!(html.contains("class=\"zx-bg\""));
        assert!(html.contains("class=\"zx-edge\""));
    }

    #[test]
    fn test_html_export_buttons() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html(&g, &SvgOptions::default());
        assert!(html.contains("saveSvg()"), "should have SVG save button");
        assert!(html.contains("saveTikz()"), "should have TikZ save button");
        assert!(html.contains("Save SVG"), "should have SVG button label");
        assert!(html.contains("Save TikZ"), "should have TikZ button label");
        // Embedded TikZ data should contain tikzpicture
        assert!(html.contains("tikzpicture"), "should embed TikZ content");
        assert!(html.contains("downloadBlob"), "should have download helper");
    }

    #[test]
    fn test_html_data_attributes() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html(&g, &SvgOptions::default());

        // Vertex data attributes
        assert!(
            html.contains("data-vid="),
            "should have vertex ID data attributes"
        );
        assert!(
            html.contains("data-vtype=\"Z\""),
            "should have Z vertex type"
        );
        assert!(
            html.contains("data-vtype=\"B\""),
            "should have B vertex type"
        );
        assert!(
            html.contains("data-degree="),
            "should have degree data attribute"
        );
        assert!(
            html.contains("data-boundary=\"in:0\""),
            "should have input boundary"
        );
        assert!(
            html.contains("data-boundary=\"out:0\""),
            "should have output boundary"
        );

        // Edge data attributes
        assert!(
            html.contains("data-from="),
            "should have edge source data attribute"
        );
        assert!(
            html.contains("data-to="),
            "should have edge target data attribute"
        );
    }

    #[test]
    fn test_html_vertex_info_js() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html(&g, &SvgOptions::default());
        assert!(
            html.contains("const vertices = {"),
            "should have vertex info JS object"
        );
        assert!(
            html.contains("type:'Z'"),
            "should have Z type in vertex info"
        );
        assert!(
            html.contains("type:'B'"),
            "should have B type in vertex info"
        );
        assert!(
            html.contains("neighbors:["),
            "should have neighbors in vertex info"
        );
    }

    #[test]
    fn test_html_tooltip_and_selection_panel() {
        use quizx::vec_graph::Graph;

        let g = Graph::new();
        let html = render_html(&g, &SvgOptions::default());
        assert!(html.contains("id=\"tooltip\""), "should have tooltip div");
        assert!(
            html.contains("id=\"selection-panel\""),
            "should have selection panel"
        );
        assert!(
            html.contains("id=\"selection-content\""),
            "should have selection content div"
        );
        assert!(
            html.contains("findVertex"),
            "should have findVertex function"
        );
        assert!(
            html.contains("selectVertex"),
            "should have selectVertex function"
        );
        assert!(
            html.contains("clearSelection"),
            "should have clearSelection function"
        );
        assert!(
            html.contains("Click to select"),
            "should have click help text"
        );
    }

    #[test]
    fn test_rewrite_html_has_graph_data() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html_with_rewrites(&g, &SvgOptions::default());
        assert!(
            html.contains("const initialGraph = {"),
            "should have embedded graph data"
        );
        assert!(
            html.contains("const initialPositions = {"),
            "should have embedded positions"
        );
    }

    #[test]
    fn test_rewrite_html_has_rewrite_engine() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html_with_rewrites(&g, &SvgOptions::default());
        assert!(
            html.contains("function findRewrites"),
            "should have findRewrites"
        );
        assert!(
            html.contains("function applySpiderFusion"),
            "should have applySpiderFusion"
        );
        assert!(
            html.contains("function checkRemoveId"),
            "should have checkRemoveId"
        );
        assert!(
            html.contains("function checkLocalComp"),
            "should have checkLocalComp"
        );
        assert!(
            html.contains("function checkPivot"),
            "should have checkPivot"
        );
        assert!(
            html.contains("function checkPiCopy"),
            "should have checkPiCopy"
        );
    }

    #[test]
    fn test_rewrite_html_has_undo_redo() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html_with_rewrites(&g, &SvgOptions::default());
        assert!(html.contains("function undo"), "should have undo function");
        assert!(html.contains("function redo"), "should have redo function");
        assert!(html.contains("id=\"undo-btn\""), "should have undo button");
        assert!(html.contains("id=\"redo-btn\""), "should have redo button");
    }

    #[test]
    fn test_rewrite_html_has_simplify_buttons() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html_with_rewrites(&g, &SvgOptions::default());
        assert!(
            html.contains("Clifford Simplify"),
            "should have Clifford Simplify button"
        );
        assert!(
            html.contains("Full Simplify"),
            "should have Full Simplify button"
        );
    }

    #[test]
    fn test_render_svg_body_standalone() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let opts = SvgOptions::default();
        let palette = opts.color_scheme.palette();
        let positions = compute_layout(&g, opts.layout, &opts.layout_options);
        let (width, height) = compute_svg_dimensions(&positions, &opts);
        let (body, infos) = render_svg_body(&g, &opts, palette, &positions, width, height);

        assert!(body.contains("base-graph"), "should have base-graph group");
        assert!(body.contains("zx-bg"), "should have background rect");
        assert!(!infos.is_empty(), "should have vertex infos");
    }

    #[test]
    fn test_original_render_html_unchanged() {
        use quizx::graph::GraphLike;
        use quizx::vec_graph::Graph;

        let mut g = Graph::new();
        let b0 = g.add_vertex(VType::B);
        let z = g.add_vertex(VType::Z);
        let b1 = g.add_vertex(VType::B);
        g.add_edge(b0, z);
        g.add_edge(z, b1);
        g.set_inputs(vec![b0]);
        g.set_outputs(vec![b1]);

        let html = render_html(&g, &SvgOptions::default());
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("const vertices = {"));
        assert!(html.contains("</html>"));
    }
}
