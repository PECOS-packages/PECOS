// Standalone binary to generate demo SVGs.
// Run from the PECOS workspace root:
//   cargo run --example svg_demo

use pecos_quantum::{DagCircuit, TickCircuit};
use pecos_core::Angle64;
use std::fs;

fn main() {
    let dir = "/tmp/pecos_svg_demo";

    // --- Circuit 1: All gate families in a TickCircuit ---
    let mut tc = TickCircuit::new();
    // tick 0: prep
    tc.tick().pz(&[0, 1, 2, 3]);
    // tick 1: Pauli family
    tc.tick().x(&[0]).y(&[1]).z(&[2]).i(&[3]);
    // tick 2: S-like family
    tc.tick().sx(&[0]).sy(&[1]).sz(&[2]);
    // tick 3: H-like family
    tc.tick().h(&[0, 1, 2, 3]);
    // tick 4: Default (T gate)
    tc.tick().t(&[0]).tdg(&[1]).rz(Angle64::QUARTER_TURN, &[2]);
    // tick 5: multi-qubit
    tc.tick().cx(&[(0, 1)]).cz(&[(2, 3)]);
    // tick 6: measure
    tc.tick().mz(&[0, 1, 2, 3]);

    let svg1 = tc.to_svg();
    fs::write(format!("{dir}/families.svg"), &svg1).unwrap();

    // --- Circuit 2: Teleportation-style circuit ---
    let mut tc2 = TickCircuit::new();
    tc2.tick().pz(&[0, 1, 2]);
    tc2.tick().h(&[1]);
    tc2.tick().cx(&[(1, 2)]);
    tc2.tick().cx(&[(0, 1)]);
    tc2.tick().h(&[0]);
    tc2.tick().mz(&[0, 1]);

    let svg2 = tc2.to_svg();
    fs::write(format!("{dir}/teleport.svg"), &svg2).unwrap();

    // --- Circuit 3: DagCircuit with mixed gates ---
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.h(&[0]);
    dag.sx(&[1]);
    dag.cx(&[(0, 1)]);
    dag.sz(&[0]);
    dag.h(&[1]);
    dag.mz(&[0, 1]);

    let svg3 = dag.to_svg();
    fs::write(format!("{dir}/dag_mixed.svg"), &svg3).unwrap();

    println!("SVGs written to {dir}/");
}
