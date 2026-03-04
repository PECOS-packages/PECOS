// Standalone binary to generate an HTML demo of all diagram styles.
// Run from the PECOS workspace root:
//   cargo run --example style_demo -p pecos-quantum

use pecos_core::circuit_diagram::{AngleUnit, DiagramStyle, GraphStyle};
use pecos_core::{Angle64, ColorPalette, ColorTriplet, CosetPatterns, FamilyPalette, FillPattern};
use pecos_qsim::GraphState;
use pecos_quantum::TickCircuit;
use pecos_quantum::pass::{
    AbsorbBasisGates, CancelInverses, CircuitPass, CompactTicks, MergeAdjacentRotations,
    PassPipeline, PeepholeOptimize, RemoveIdentity, SimplifyRotations,
};
use std::fmt::Write as _;
use std::fs;

fn build_circuit() -> TickCircuit {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1, 2, 3]);
    tc.tick().x(&[0]).y(&[1]).z(&[2]);
    tc.tick().sx(&[0]).sy(&[1]).sz(&[2]);
    tc.tick().h(&[0, 1, 2, 3]);
    tc.tick().t(&[0]).tdg(&[1]).rz(Angle64::QUARTER_TURN, &[2]);
    tc.tick().cx(&[(0, 1)]).cz(&[(2, 3)]);
    let eighth = Angle64::QUARTER_TURN / 2u64;
    tc.tick()
        .rzz(Angle64::QUARTER_TURN, &[(0, 1)])
        .rzz(eighth, &[(2, 3)]);
    tc.tick().mz(&[0, 1, 2, 3]);
    tc
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn ansi_to_html(s: &str) -> String {
    let mut out = String::new();
    let mut in_span = false;
    let mut i = 0;
    let bytes = s.as_bytes();

    while i < bytes.len() {
        if bytes[i] == b'\x1b' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Parse ANSI escape
            let start = i + 2;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'm' {
                end += 1;
            }
            if end < bytes.len() {
                let code = &s[start..end];
                if in_span {
                    out.push_str("</span>");
                    in_span = false;
                }
                // Handle compound codes like "1;34" (bold + color)
                let style = ansi_code_to_css(code);
                if let Some(css) = style {
                    write!(out, "<span style=\"{css}\">").unwrap();
                    in_span = true;
                }
                i = end + 1;
                continue;
            }
        }

        let ch = s[i..].chars().next().unwrap();
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
        i += ch.len_utf8();
    }
    if in_span {
        out.push_str("</span>");
    }
    out
}

fn ansi_code_to_css(code: &str) -> Option<String> {
    if code == "0" {
        return None;
    }
    let parts: Vec<&str> = code.split(';').collect();
    let mut bold = false;
    let mut color = None;
    for part in &parts {
        match *part {
            "1" => bold = true,
            "31" => color = Some("#AA2222"),
            "32" => color = Some("#226622"),
            "33" => color = Some("#AA8800"),
            "34" => color = Some("#2255AA"),
            "35" => color = Some("#882288"),
            "36" => color = Some("#008888"),
            "37" => color = Some("#666666"),
            "90" => color = Some("#888888"),
            _ => {}
        }
    }
    match (bold, color) {
        (true, Some(c)) => Some(format!("font-weight:bold;color:{c}")),
        (true, None) => Some("font-weight:bold".to_string()),
        (false, Some(c)) => Some(format!("color:{c}")),
        (false, None) => None,
    }
}

fn section(title: &str, body: &str) -> String {
    format!("<h2>{title}</h2>\n{body}\n")
}

fn pre_block(content: &str) -> String {
    format!("<pre class=\"output\">{content}</pre>")
}

fn svg_block(svg: &str) -> String {
    format!("<div class=\"svg-container\">{svg}</div>")
}

fn code_block(lang: &str, content: &str) -> String {
    format!(
        "<details><summary>Show {lang} source</summary><pre class=\"code\">{}</pre></details>",
        escape_html(content)
    )
}

fn main() {
    let tc = build_circuit();

    let default_style = DiagramStyle::default();

    let custom_palette = DiagramStyle::builder()
        .x_axis("#FF6666", "#CC0000", "#660000")
        .z_axis("#6666FF", "#0000CC", "#000066")
        .xz_mix("#CC66CC", "#990099", "#660066")
        .build();

    let monochrome = DiagramStyle::builder().color(false).build();

    let no_dashes = DiagramStyle::builder().show_dashes(false).build();

    let mono_no_dashes = DiagramStyle::builder()
        .color(false)
        .show_dashes(false)
        .build();

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PECOS Visualization Demo</title>
<style>
  body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1200px;
    margin: 2em auto;
    padding: 0 1em;
    background: #fafafa;
    color: #222;
  }
  h1 { border-bottom: 2px solid #ddd; padding-bottom: 0.5em; }
  h2 { margin-top: 2em; color: #444; }
  h3 { color: #666; }
  pre.output {
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 1em;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 14px;
    line-height: 1.4;
  }
  pre.code {
    background: #f0f0f0;
    padding: 1em;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.3;
    border: 1px solid #ddd;
  }
  .svg-container {
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1em;
    margin: 0.5em 0;
    overflow-x: auto;
  }
  details { margin: 0.5em 0; }
  summary { cursor: pointer; color: #555; font-size: 0.9em; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1em;
  }
  @media (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<h1>PECOS Visualization Demo</h1>
<p>Circuit diagrams and graph state visualizations with configurable styles.</p>
<h2 style="margin-top:1em;">Circuit Diagrams</h2>
<p>All gate families: Prep, Pauli, S-like, H-like, Default (T), multi-qubit (CX/CZ), Measure.</p>
"#,
    );

    // -- Text outputs --
    html.push_str(&section(
        "ASCII (plain)",
        &pre_block(&escape_html(&tc.to_ascii())),
    ));

    html.push_str(&section(
        "ASCII (ANSI color)",
        &pre_block(&ansi_to_html(&tc.to_color_ascii())),
    ));

    html.push_str(&section(
        "Unicode (plain)",
        &pre_block(&escape_html(&tc.to_unicode())),
    ));

    html.push_str(&section(
        "Unicode (ANSI color)",
        &pre_block(&ansi_to_html(&tc.to_color_unicode())),
    ));

    // -- SVG outputs --
    html.push_str("<h2>SVG Outputs</h2>\n<div class=\"grid\">\n");

    let r_default = tc.render_with(&default_style);
    write!(
        html,
        "<div><h3>Default</h3>{}</div>",
        svg_block(&r_default.svg())
    )
    .unwrap();

    let r_custom = tc.render_with(&custom_palette);
    write!(
        html,
        "<div><h3>Custom Palette</h3>{}</div>",
        svg_block(&r_custom.svg())
    )
    .unwrap();

    let r_mono = tc.render_with(&monochrome);
    write!(
        html,
        "<div><h3>Monochrome (color: false)</h3>{}</div>",
        svg_block(&r_mono.svg())
    )
    .unwrap();

    let r_nodash = tc.render_with(&no_dashes);
    write!(
        html,
        "<div><h3>No Dashes (show_dashes: false)</h3>{}</div>",
        svg_block(&r_nodash.svg())
    )
    .unwrap();

    let r_mono_nodash = tc.render_with(&mono_no_dashes);
    write!(
        html,
        "<div><h3>Monochrome + No Dashes</h3>{}</div>",
        svg_block(&r_mono_nodash.svg())
    )
    .unwrap();

    html.push_str("</div>\n");

    // -- TikZ --
    html.push_str(&section(
        "TikZ (default)",
        &code_block("TikZ", &r_default.tikz()),
    ));
    html.push_str(&section(
        "TikZ (monochrome)",
        &code_block("TikZ", &r_mono.tikz()),
    ));

    // -- DOT --
    html.push_str(&section(
        "DOT / Graphviz (default)",
        &code_block("DOT", &r_default.dot()),
    ));
    html.push_str(&section(
        "DOT / Graphviz (monochrome)",
        &code_block("DOT", &r_mono.dot()),
    ));

    // -- Angle unit comparison --
    html.push_str("<h2>Angle Units: Radians vs Turns</h2>\n");
    {
        // Build a circuit with several parameterized gates
        let mut angle_tc = TickCircuit::new();
        angle_tc.tick().pz(&[0, 1, 2]);
        let eighth = Angle64::QUARTER_TURN / 2u64;
        angle_tc
            .tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(eighth, &[1])
            .rz(Angle64::HALF_TURN, &[2]);
        angle_tc
            .tick()
            .rx(Angle64::QUARTER_TURN, &[0])
            .ry(eighth, &[1]);
        angle_tc.tick().mz(&[0, 1, 2]);

        let radians_style = DiagramStyle::builder()
            .angle_unit(AngleUnit::Radians)
            .build();
        let turns_style = DiagramStyle::builder().angle_unit(AngleUnit::Turns).build();

        let r_rad = angle_tc.render_with(&radians_style);
        let r_turns = angle_tc.render_with(&turns_style);

        html.push_str("<div class=\"grid\">\n");
        write!(
            html,
            "<div><h3>Radians (default)</h3>{}{}</div>",
            pre_block(&escape_html(&r_rad.ascii())),
            svg_block(&r_rad.svg()),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>Turns</h3>{}{}</div>",
            pre_block(&escape_html(&r_turns.ascii())),
            svg_block(&r_turns.svg()),
        )
        .unwrap();
        html.push_str("</div>\n");
    }

    // -- Rotation simplification comparison (pass-based) --
    html.push_str("<h2>Rotation Simplification (Circuit Pass)</h2>\n");
    {
        let mut rot_tc = TickCircuit::new();
        rot_tc.tick().pz(&[0, 1, 2, 3]);
        rot_tc
            .tick()
            .rz(Angle64::HALF_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[1])
            .rx(Angle64::QUARTER_TURN, &[2])
            .ry(Angle64::QUARTER_TURN, &[3]);
        let eighth = Angle64::QUARTER_TURN / 2u64;
        rot_tc
            .tick()
            .rz(eighth, &[0])
            .rx(Angle64::HALF_TURN, &[1])
            .ry(Angle64::HALF_TURN, &[2])
            .rz(Angle64::THREE_QUARTERS_TURN, &[3]);
        rot_tc.tick().mz(&[0, 1, 2, 3]);

        // Clone and apply the SimplifyRotations pass to one copy.
        let mut simplified_tc = rot_tc.clone();
        SimplifyRotations.apply_tick(&mut simplified_tc);

        let style = DiagramStyle::default();
        let r_before = rot_tc.render_with(&style);
        let r_after = simplified_tc.render_with(&style);

        html.push_str("<div class=\"grid\">\n");
        write!(
            html,
            "<div><h3>Before pass</h3>{}{}</div>",
            pre_block(&escape_html(&r_before.ascii())),
            svg_block(&r_before.svg()),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>After SimplifyRotations</h3>{}{}</div>",
            pre_block(&escape_html(&r_after.ascii())),
            svg_block(&r_after.svg()),
        )
        .unwrap();
        html.push_str("</div>\n");
    }

    // -- Peephole optimization comparison --
    html.push_str("<h2>Peephole Optimization (Circuit Pass)</h2>\n");
    {
        let mut peep_tc = TickCircuit::new();
        peep_tc.tick().pz(&[0, 1, 2, 3]);
        // H-CX-H on target -> CZ
        peep_tc.tick().h(&[1]);
        peep_tc.tick().cx(&[(0, 1)]);
        peep_tc.tick().h(&[1]);
        // H-CZ-H on one qubit -> CX
        peep_tc.tick().h(&[2]);
        peep_tc.tick().cz(&[(2, 3)]);
        peep_tc.tick().h(&[2]);
        peep_tc.tick().mz(&[0, 1, 2, 3]);

        let mut optimized_tc = peep_tc.clone();
        PeepholeOptimize.apply_tick(&mut optimized_tc);

        let style = DiagramStyle::default();
        let r_before = peep_tc.render_with(&style);
        let r_after = optimized_tc.render_with(&style);

        html.push_str("<div class=\"grid\">\n");
        write!(
            html,
            "<div><h3>Before pass</h3>{}{}</div>",
            pre_block(&escape_html(&r_before.ascii())),
            svg_block(&r_before.svg()),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>After PeepholeOptimize</h3>{}{}</div>",
            pre_block(&escape_html(&r_after.ascii())),
            svg_block(&r_after.svg()),
        )
        .unwrap();
        html.push_str("</div>\n");
    }

    // -- Full pass pipeline comparison --
    html.push_str("<h2>Full Pass Pipeline</h2>\n");
    {
        let mut pipe_tc = TickCircuit::new();
        pipe_tc.tick().pz(&[0, 1, 2, 3]);
        // Z-diagonal after prep (absorbed)
        pipe_tc.tick().t(&[0]).sz(&[1]).cz(&[(2, 3)]);
        // Mergeable rotations
        pipe_tc.tick().rz(Angle64::QUARTER_TURN, &[0]).h(&[1]);
        pipe_tc.tick().rz(Angle64::QUARTER_TURN, &[0]).cx(&[(1, 2)]);
        // Cancellable pair
        pipe_tc.tick().h(&[1]);
        // Z-diagonal before measure (absorbed)
        pipe_tc.tick().tdg(&[2]).sz(&[3]);
        pipe_tc.tick().mz(&[0, 1, 2, 3]);

        let pipeline = PassPipeline::new()
            .then(AbsorbBasisGates)
            .then(MergeAdjacentRotations)
            .then(RemoveIdentity)
            .then(SimplifyRotations)
            .then(CancelInverses)
            .then(PeepholeOptimize)
            .then(CompactTicks);

        let mut optimized_tc = pipe_tc.clone();
        pipeline.apply_tick(&mut optimized_tc);

        let style = DiagramStyle::default();
        let r_before = pipe_tc.render_with(&style);
        let r_after = optimized_tc.render_with(&style);

        html.push_str("<div class=\"grid\">\n");
        write!(
            html,
            "<div><h3>Before pipeline</h3>{}{}</div>",
            pre_block(&escape_html(&r_before.ascii())),
            svg_block(&r_before.svg()),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>After pipeline</h3>{}{}</div>",
            pre_block(&escape_html(&r_after.ascii())),
            svg_block(&r_after.svg()),
        )
        .unwrap();
        html.push_str("</div>\n");
    }

    // -- Operator example --
    html.push_str("<h2>Operator Algebra</h2>\n");
    {
        use pecos_core::operator::{CX, H, T};
        let circuit = T(1) * CX(0, 1) * H(0);
        let op_renderer = circuit.render_with(2, &default_style);
        write!(
            html,
            "<h3>T(1) * CX(0,1) * H(0)</h3>\n{}{}",
            pre_block(&escape_html(&op_renderer.ascii())),
            svg_block(&op_renderer.svg()),
        )
        .unwrap();
    }

    // -- Overlapping multi-qubit gates --
    html.push_str("<h2>Overlapping Multi-qubit Gates (Sub-column Splitting)</h2>\n");
    {
        let mut overlap_tc = TickCircuit::new();
        overlap_tc.tick().h(&[0, 1, 2, 3]);
        let mut t = overlap_tc.tick();
        t.cx(&[(0, 2)]);
        t.cz(&[(1, 3)]);
        overlap_tc.tick().mz(&[0, 1, 2, 3]);

        let r = overlap_tc.render_with(&default_style);
        html.push_str("<p>CX(0,2) and CZ(1,3) in the same tick have overlapping visual ranges, \
                        so they are split into separate sub-columns with a bracket annotation.</p>\n");
        write!(
            html,
            "{}{}",
            pre_block(&escape_html(&r.ascii())),
            svg_block(&r.svg()),
        )
        .unwrap();
    }

    // ================================================================
    // Graph State Visualization
    // ================================================================

    html.push_str(
        "<h2 style=\"border-bottom:2px solid #ddd; padding-bottom:0.5em; margin-top:3em;\">\
                   Graph State Visualization</h2>\n",
    );
    html.push_str("<p>Graph states visualized with the PECOS color algebra: \
                   fill hue = axis permutation coset, brightness = sign parity, \
                   stroke = gate family. All formats share the same <code>GraphStyle</code> palette.</p>\n");

    let gs_default = GraphStyle::default();

    // -- Pattern gallery --
    html.push_str("<h2>Graph State Patterns</h2>\n<div class=\"grid\">\n");

    let patterns: &[(&str, GraphState)] = &[
        ("Linear Cluster (5)", GraphState::linear_cluster(5)),
        ("Ring (6)", GraphState::ring(6)),
        ("Star (5)", GraphState::star(5)),
        ("Complete K4", GraphState::complete(4)),
        ("2D Lattice (2x3)", GraphState::lattice_2d(2, 3)),
    ];

    for (label, gs) in patterns {
        write!(
            html,
            "<div><h3>{label}</h3>{}{}</div>",
            pre_block(&escape_html(&gs.to_ascii())),
            svg_block(&gs.render_with(&gs_default).svg()),
        )
        .unwrap();
    }
    html.push_str("</div>\n");

    // -- Graph state with non-identity VOPs --
    html.push_str("<h2>Graph States with VOPs</h2>\n");
    html.push_str(
        "<p>When local Cliffords (VOPs) are applied to vertices, \
                   the fill color encodes the axis permutation coset and \
                   the stroke encodes the gate family.</p>\n",
    );
    {
        use pecos_qsim::clifford_frame::CliffordFrame;

        let mut gs = GraphState::ring(6);
        gs.set_vop(0, CliffordFrame::H); // H-like, X<->Z coset
        gs.set_vop(1, CliffordFrame::SZ); // S-like, X<->Y coset
        gs.set_vop(2, CliffordFrame::SX); // S-like, Y<->Z coset
        gs.set_vop(4, CliffordFrame::from_index(7)); // F-like, cyclic fwd
        gs.set_vop(5, CliffordFrame::from_index(8)); // F-like, cyclic inv

        html.push_str("<div class=\"grid\">\n");
        write!(
            html,
            "<div><h3>ASCII</h3>{}</div>",
            pre_block(&escape_html(&gs.to_ascii())),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>Color ASCII</h3>{}</div>",
            pre_block(&ansi_to_html(&gs.to_color_ascii())),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>Unicode</h3>{}</div>",
            pre_block(&escape_html(&gs.to_unicode())),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>Color Unicode</h3>{}</div>",
            pre_block(&ansi_to_html(&gs.to_color_unicode())),
        )
        .unwrap();
        html.push_str("</div>\n");

        let r = gs.render_with(&gs_default);
        html.push_str("<div class=\"grid\">\n");
        write!(html, "<div><h3>SVG</h3>{}</div>", svg_block(&r.svg())).unwrap();
        write!(
            html,
            "<div><h3>DOT</h3>{}</div>",
            code_block("DOT", &r.dot()),
        )
        .unwrap();
        html.push_str("</div>\n");
        html.push_str(&section("TikZ", &code_block("TikZ", &r.tikz())));
    }

    // -- All 24 Cliffords showcase --
    html.push_str("<h2>All 24 Single-Qubit Cliffords</h2>\n");
    html.push_str(
        "<p>Each vertex has a different Clifford VOP, showing the full \
                   color algebra: 5 coset hues, 2 brightness levels (saturated/light), \
                   4 gate-family strokes.</p>\n",
    );
    {
        use pecos_qsim::clifford_frame::CliffordFrame;

        // Build a 24-vertex graph with each vertex having a unique Clifford VOP.
        // No edges -- just showcasing the VOP colors.
        let mut gs = GraphState::new(24);
        for i in 0..24 {
            gs.set_vop(i, CliffordFrame::from_index(i as u8));
        }

        let r = gs.render_with(&gs_default);
        write!(
            html,
            "{}{}",
            pre_block(&escape_html(&gs.to_ascii())),
            svg_block(&r.svg()),
        )
        .unwrap();
    }

    // -- SVG style variations --
    html.push_str("<h2>Graph Style Variations</h2>\n<div class=\"grid\">\n");
    {
        use pecos_qsim::clifford_frame::CliffordFrame;

        let mut gs = GraphState::star(5);
        gs.set_vop(1, CliffordFrame::H);
        gs.set_vop(2, CliffordFrame::SZ);
        gs.set_vop(3, CliffordFrame::SX);
        gs.set_vop(4, CliffordFrame::from_index(7));

        // Default
        write!(
            html,
            "<div><h3>Default</h3>{}</div>",
            svg_block(&gs.render_with(&gs_default).svg()),
        )
        .unwrap();

        // Custom palette: warm tones
        let warm_palette = ColorPalette {
            z_axis: ColorTriplet::new("#FFB0B0", "#AA2222", "#7A1A1A"),
            xz_mix: ColorTriplet::new("#E0B0E0", "#882288", "#5A1A5A"),
            xy_mix: ColorTriplet::new("#F0E0A0", "#AA8800", "#6A5500"),
            yz_mix: ColorTriplet::new("#FFD0B0", "#CC6600", "#884400"),
            xyz_mix: ColorTriplet::new("#E0D0C0", "#887766", "#554433"),
            ..ColorPalette::default()
        };
        let warm_style = GraphStyle::builder().palette(warm_palette).build();
        write!(
            html,
            "<div><h3>Custom Palette (warm)</h3>{}</div>",
            svg_block(&gs.render_with(&warm_style).svg()),
        )
        .unwrap();

        // Monochrome: varying grey levels, uniform strokes, dashes + patterns
        let mono_palette = ColorPalette {
            z_axis: ColorTriplet::new("#D8D8D8", "#555555", "#333333"),
            xz_mix: ColorTriplet::new("#C4C4C4", "#555555", "#333333"),
            xy_mix: ColorTriplet::new("#B0B0B0", "#555555", "#333333"),
            yz_mix: ColorTriplet::new("#9C9C9C", "#555555", "#222222"),
            xyz_mix: ColorTriplet::new("#888888", "#555555", "#222222"),
            ..ColorPalette::default()
        };
        let mono_families = FamilyPalette {
            pauli: "#555555".to_string(),
            s_like: "#555555".to_string(),
            h_like: "#555555".to_string(),
            f_like: "#555555".to_string(),
        };
        let mono_patterns = CosetPatterns {
            identity: FillPattern::Solid,
            xz_mix: FillPattern::DiagonalUp,
            xy_mix: FillPattern::Crosshatch,
            yz_mix: FillPattern::Dots,
            xyz_mix: FillPattern::HorizontalLines,
        };
        let mono_style = GraphStyle::builder()
            .palette(mono_palette)
            .family_strokes(mono_families)
            .show_dashes(true)
            .coset_patterns(mono_patterns)
            .build();
        write!(
            html,
            "<div><h3>Monochrome (grey + patterns + dashes)</h3>{}</div>",
            svg_block(&gs.render_with(&mono_style).svg()),
        )
        .unwrap();

        // Custom family strokes
        let bold_families = FamilyPalette {
            pauli: "#0000AA".to_string(),
            s_like: "#00AA00".to_string(),
            h_like: "#AA0000".to_string(),
            f_like: "#AA00AA".to_string(),
        };
        let bold_style = GraphStyle::builder().family_strokes(bold_families).build();
        write!(
            html,
            "<div><h3>Bold Family Strokes</h3>{}</div>",
            svg_block(&gs.render_with(&bold_style).svg()),
        )
        .unwrap();
    }
    html.push_str("</div>\n");

    // -- Local complementation --
    html.push_str("<h2>Local Complementation</h2>\n");
    html.push_str(
        "<p>Applying local complementation to vertex 0 of a star graph \
                   complements edges among its neighbors and updates VOPs.</p>\n",
    );
    {
        let gs_before = GraphState::star(5);
        let mut gs_after = gs_before.clone();
        gs_after.local_complement(0);

        html.push_str("<div class=\"grid\">\n");
        write!(
            html,
            "<div><h3>Before LC(0)</h3>{}{}</div>",
            pre_block(&escape_html(&gs_before.to_ascii())),
            svg_block(&gs_before.render_with(&gs_default).svg()),
        )
        .unwrap();
        write!(
            html,
            "<div><h3>After LC(0)</h3>{}{}</div>",
            pre_block(&escape_html(&gs_after.to_ascii())),
            svg_block(&gs_after.render_with(&gs_default).svg()),
        )
        .unwrap();
        html.push_str("</div>\n");
    }

    html.push_str("</body>\n</html>\n");

    let path = "/tmp/pecos_style_demo.html";
    fs::write(path, &html).unwrap();
    println!("Written to {path}");

    // Open in default browser
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", path])
        .spawn();
}
