// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! A repeatable visual inspection gallery, generated entirely in Rust.
//!
//! `cargo run -p pecos-quantum --example circuit_gallery -- /tmp/pecos-gallery`
//! Open index.html locally. Existing files with these names are replaced.

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, MeasId, QubitId};
use pecos_quantum::{AngleUnit, DagCircuit, DiagramRenderer, DiagramStyle, TickCircuit};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Gallery {
    directory: PathBuf,
    html: String,
    tex: String,
}

impl Gallery {
    fn new(directory: PathBuf) -> Result<Self> {
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            html: String::from(
                r#"<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PECOS circuit gallery</title>
<style>
body { margin: 0 auto; padding: 24px; max-width: 1100px; color: #172333;
       background: #f3f5f8; font: 16px/1.5 system-ui, sans-serif; }
h1, h2 { line-height: 1.2; } h2 { font-size: 21px; }
section { background: white; border: 1px solid #ccd4de; border-radius: 8px;
          padding: 20px; margin: 24px 0; }
.diagram { overflow-x: auto; padding: 12px 0; }
img { display: block; max-width: none; }
a { color: #135aa4; } a:focus-visible { outline: 3px solid #135aa4; }
nav a { display: inline-block; margin: 4px 16px 4px 0; }
</style>
<main><h1>PECOS circuit gallery</h1>
<p>Structural ticks and dependency layers, not physical execution time.
Each note describes what to check. Wide diagrams scroll horizontally;
open the SVG to zoom. No notebook, server, or network connection is needed.</p>
<p><a href="gallery.tex">Combined TikZ document</a> (requires LaTeX with TikZ,
geometry, and graphicx to compile). Individual TeX links are fragments.</p>
"#,
            ),
            tex: String::from(
                "\\documentclass{article}\n\\usepackage[a4paper,landscape,margin=1cm]{geometry}\n\\usepackage{tikz}\n\\usepackage{graphicx}\n\\pagestyle{empty}\n\\begin{document}\n",
            ),
        })
    }

    fn add(
        &mut self,
        id: &str,
        title: &str,
        note: &str,
        diagram: DiagramRenderer<'_>,
    ) -> Result<()> {
        // IDs below are fixed filename stems; titles and notes are plain text.
        let diagram = diagram.with_title(title);
        for (suffix, content) in [
            ("svg", diagram.svg()),
            ("tex", diagram.tikz()),
            ("dot", diagram.dot()),
            ("txt", diagram.unicode()),
        ] {
            fs::write(self.directory.join(format!("{id}.{suffix}")), content)?;
        }
        writeln!(
            self.tex,
            "\\sbox0{{\\input{{{id}.tex}}}}\n\\noindent\\ifdim\\wd0>\\linewidth\\resizebox{{\\linewidth}}{{!}}{{\\usebox0}}\\else\\usebox0\\fi\n\\clearpage"
        )?;
        writeln!(
            self.html,
            "<section id=\"{id}\"><h2>{}</h2><p>{}</p>\n\
             <div class=\"diagram\" tabindex=\"0\" role=\"region\" aria-label=\"{} diagram\">\
             <img src=\"{id}.svg\" alt=\"{}\"></div>\n\
             <nav aria-label=\"{} exports\"><a href=\"{id}.svg\">SVG</a>\
             <a href=\"{id}.tex\">TikZ</a><a href=\"{id}.txt\">Text</a>\
             <a href=\"{id}.dot\">DOT</a></nav></section>",
            html_text(title),
            html_text(note),
            html_text(title),
            html_text(note),
            html_text(title),
        )?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.html.push_str("</main></html>\n");
        self.tex.push_str("\\end{document}\n");
        fs::write(self.directory.join("index.html"), self.html)?;
        fs::write(self.directory.join("gallery.tex"), self.tex)?;
        println!("Open {}", self.directory.join("index.html").display());
        Ok(())
    }
}

fn html_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn scheduling(gallery: &mut Gallery, style: &DiagramStyle) -> Result<()> {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    circuit.tick().h(&[0]);
    circuit.tick().x(&[1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[0, 1]);
    gallery.add("ticks", "Explicit ticks", "H(q0) and X(q1) occupy separate ticks. The final readout has identities m0 and m1, not outcome values.", circuit.render_with(style))?;
    let dag = DagCircuit::try_from(&circuit)?;
    gallery.add("layers", "Dependency layers", "The same H(q0) and X(q1) now share layer l1 because they are independent. Compare this with the explicit-tick view.", dag.render_with(style))?;
    gallery.add("readout", "Readout panel", "Selecting ticks 4..5 retains both qubit rows, the original t4 label, and measurement identities m0 and m1.", circuit.render_range(4..5, style)?)?;
    gallery.add("empty-range", "Empty range", "An empty interval 2..2 shows only its descriptive header. It must not invent gates or renumber the selected range.", circuit.render_range(2..2, style)?)?;
    Ok(())
}

fn connectors(gallery: &mut Gallery, style: &DiagramStyle) -> Result<()> {
    let mut circuit = TickCircuit::new();
    circuit.tick().cx(&[(5, 0), (4, 1), (2, 3)]);
    gallery.add("controls", "Nested controls", "Three disjoint CX gates share t0. Dots are controls and X boxes are targets; connector crossings must not turn intermediate wires into operands.", circuit.render_with(style))?;

    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0, 1, 2, 3, 4]);
    circuit
        .tick()
        .rzz(Angle64::from_turn_ratio(1, 8), &[(0, 4)])
        .x(&[1])
        .h(&[2])
        .z(&[3]);
    gallery.add("interaction", "Nonadjacent interaction", "Only q0 and q4 participate in RZZ. Its label sits between wires; the independent X, H, and Z gates remain visible in the same tick.", circuit.render_with(style))?;
    let monochrome = DiagramStyle::builder().color(false).build();
    gallery.add("monochrome", "Monochrome interaction", "The same circuit remains legible without color. Endpoints, labels, and gate shapes carry its meaning.", circuit.render_with(&monochrome))?;

    let mut circuit = TickCircuit::new();
    circuit
        .tick()
        .custom_gate("Block<&>_A", &[9, 2], &[])?
        .h(&[5]);
    gallery.add("custom", "Sparse custom operands", "Only rows q2, q5, and q9 exist. The custom gate retains its literal name Block<&>_A; operand [0] is q9 and [1] is q2. H on q5 must remain separate.", circuit.render_with(style))?;

    let mut circuit = TickCircuit::new();
    circuit
        .tick()
        .channel(pecos_core::channel::Depolarizing2(0.1, 0, 2))
        .h(&[1]);
    gallery.add("channel", "Correlated channel support", "Ch[0] and Ch[1] are connected endpoints of one channel on q0 and q2. H on q1 is independent. The diagram does not encode the channel probability or payload.", circuit.render_with(style))?;
    Ok(())
}

fn measurements_and_angles(gallery: &mut Gallery, style: &DiagramStyle) -> Result<()> {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[2, 8]);
    let mut gate = Gate::simple(GateType::MeasureLeaked, vec![QubitId(8), QubitId(2)]);
    gate.meas_ids = vec![MeasId::from_raw(42), MeasId::from_raw(7)].into();
    circuit.tick().try_add_gate(gate)?;
    gallery.add("leakage", "Leakage-aware readout", "MZ_L is one leakage-aware Z measurement with outcomes 0, 1, or 2=L. q2 retains m7 and q8 retains m42 despite the reversed operand list; these are IDs, not outcomes.", circuit.render_with(style))?;

    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0]);
    circuit.tick();
    circuit.tick().idle(20, &[0]);
    circuit.tick();
    gallery.add("empty-ticks", "Empty ticks and idle duration", "Empty t1 and trailing t3 remain visible. Idle(20) labels an abstract duration without scaling the horizontal axis.", circuit.render_with(style))?;

    let mut circuit = TickCircuit::new();
    for angle in [
        Angle64::ZERO,
        Angle64::try_from_radians(1e-7).unwrap(),
        Angle64::try_from_radians(-1e-7).unwrap(),
        Angle64::from_turn_ratio(1, 8),
    ] {
        circuit.tick().rz(angle, &[0]);
    }
    gallery.add("angles", "Small and fractional angles", "Zero stays distinct from tiny positive and negative rotations. The last angle is pi/4; the small negative label is the residual below a full turn.", circuit.render_with(style))?;
    let turns = DiagramStyle::builder().angle_unit(AngleUnit::Turns).build();
    gallery.add("turns", "Angles in turns", "The same rotations in turns. The last angle is 1/8 of a full turn, and tiny rotations remain nonzero.", circuit.render_with(&turns))?;
    Ok(())
}

fn main() -> Result<()> {
    let directory = std::env::args_os().nth(1).map(PathBuf::from).ok_or(
        "provide an output directory: cargo run -p pecos-quantum --example circuit_gallery -- /tmp/pecos-gallery",
    )?;
    let mut gallery = Gallery::new(directory)?;
    let style = DiagramStyle::builder().show_measurement_ids(true).build();
    scheduling(&mut gallery, &style)?;
    connectors(&mut gallery, &style)?;
    measurements_and_angles(&mut gallery, &style)?;
    gallery.finish()
}
