// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Export standalone circuit diagrams, without Python or a notebook.
//!
//! Run from the workspace root:
//! `cargo run -p pecos-quantum --example circuit_exports -- /tmp/pecos-diagrams`
//! Existing files with the example's names are replaced.

use pecos_quantum::{AngleUnit, DagCircuit, DiagramStyle, TickCircuit};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args_os().nth(1).map(PathBuf::from).ok_or(
        "provide an output directory: cargo run -p pecos-quantum --example circuit_exports -- /tmp/pecos-diagrams",
    )?;
    fs::create_dir_all(&directory)?;

    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    circuit.tick().h(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[0, 1]);

    fs::write(directory.join("circuit.svg"), circuit.to_svg())?;
    fs::write(directory.join("circuit.tex"), circuit.to_tikz())?;
    fs::write(directory.join("circuit.dot"), circuit.to_dot())?;
    fs::write(directory.join("circuit.txt"), circuit.to_unicode())?;

    let style = DiagramStyle::builder()
        .color(false)
        .angle_unit(AngleUnit::Turns)
        .build();
    fs::write(
        directory.join("circuit-monochrome.svg"),
        circuit.render_with(&style).svg(),
    )?;

    // Preserve source tick numbers and qubit rows in a named local view.
    let panel = circuit
        .render_range(1..3, &style)?
        .with_title("Entanglement");
    fs::write(directory.join("entanglement.svg"), panel.svg())?;
    fs::write(directory.join("entanglement.tex"), panel.tikz())?;

    let identified = DiagramStyle::builder().show_measurement_ids(true).build();
    let readout = circuit
        .render_range(3..4, &identified)?
        .with_title("Readout");
    fs::write(directory.join("readout-ids.svg"), readout.svg())?;
    fs::write(directory.join("readout-ids.tex"), readout.tikz())?;

    // A DAG uses dependency layers; the tick circuit retains explicit ticks.
    let dag = DagCircuit::try_from(&circuit)?;
    fs::write(directory.join("dependencies.svg"), dag.to_svg())?;

    // Associate SVG components with owned source batches for inspection tools.
    let inspection = circuit.inspect(&identified);
    fs::write(directory.join("circuit-inspection.svg"), inspection.svg())?;
    fs::write(
        directory.join("circuit-inspector.html"),
        inspection.to_html(),
    )?;
    for operation in inspection.operations() {
        println!("{}: {:?}", operation.id, operation.source);
    }

    println!("{}", circuit.to_unicode());
    println!("Exported circuit diagrams to {}", directory.display());
    Ok(())
}
