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

//! ASCII output for ZX diagrams.
//!
//! Prints plain-text ZX diagram representations to stdout.
//! Useful for quick terminal inspection and debugging.
//! Demonstrates:
//! 1. Bell state circuit rendered in all formats
//! 2. Graph state (linear cluster) rendered in all formats
//! 3. Before/after simplification comparison
//! 4. Colored terminal output with PECOS and ZX canonical color schemes
//! 5. Gate color algebra showcase (forward/inverse brightness)

use pecos_quantum::DagCircuit;
use pecos_zx::convert::dag_to_zx;
use pecos_zx::graph::from_adjacency_matrix;
use pecos_zx::simplify;
use pecos_zx::viz::ColorScheme;
use pecos_zx::viz::Renderer;
use pecos_zx::viz::ascii::{AsciiOptions, render_ascii};
use pecos_zx::viz::colors::AnsiColor;

fn main() {
    println!("=== ASCII Output for ZX Diagrams ===\n");

    bell_state_ascii();
    graph_state_ascii();
    simplification_ascii();
    colored_output();
    gate_color_algebra();
}

/// Bell state circuit -> ZX -> all formats.
fn bell_state_ascii() {
    let mut dag = DagCircuit::new();
    dag.h(0);
    dag.cx(0, 1);

    let graph = dag_to_zx(&dag).expect("conversion failed");

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "bell_state");
}

/// Graph state as rendered output.
fn graph_state_ascii() {
    #[rustfmt::skip]
    let linear = vec![
        false, true,  false, false,
        true,  false, true,  false,
        false, true,  false, true,
        false, false, true,  false,
    ];
    let graph = from_adjacency_matrix(&linear, 4);

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "graph_state_linear");
}

/// Before/after simplification in all formats.
fn simplification_ascii() {
    let mut dag = DagCircuit::new();
    dag.h(0);
    dag.h(0);
    dag.cx(0, 1);
    dag.cx(0, 1);
    dag.h(1);

    let mut graph = dag_to_zx(&dag).expect("conversion failed");

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "simplify_before");

    simplify::clifford_simp(&mut graph);

    println!();
    r.render(&graph, "simplify_after");
}

/// Demonstrate colored ASCII output with both color schemes.
fn colored_output() {
    println!("\n--- Colored Output ---\n");

    let mut dag = DagCircuit::new();
    dag.h(0);
    dag.cx(0, 1);

    let graph = dag_to_zx(&dag).expect("conversion failed");

    println!("PECOS color scheme (Blue=Z, Red=X):\n");
    let pecos_opts = AsciiOptions {
        use_color: true,
        color_scheme: ColorScheme::Pecos,
        ..AsciiOptions::default()
    };
    let ascii = render_ascii(&graph, &pecos_opts);
    println!("{ascii}\n");

    println!("ZX canonical color scheme (Green=Z, Red=X):\n");
    let canonical_opts = AsciiOptions {
        use_color: true,
        color_scheme: ColorScheme::ZxCanonical,
        ..AsciiOptions::default()
    };
    let ascii = render_ascii(&graph, &canonical_opts);
    println!("{ascii}\n");
}

/// Showcase the PECOS gate color algebra.
///
/// Colors are derived from additive RGB mixing of the Pauli axes each gate
/// interconverts. Brightness distinguishes forward (follows X->Y->Z->X cycle)
/// from inverse (dagger).
fn gate_color_algebra() {
    println!("--- Gate Color Algebra (PECOS: RGB = XYZ) ---\n");

    let pecos = ColorScheme::Pecos;
    let reset = AnsiColor::reset();

    println!("  Base Pauli colors:");
    let x_color = pecos.ansi_x();
    let z_color = pecos.ansi_z();
    println!(
        "    X = {}Red{reset}    Y = {}Green{reset}    Z = {}Blue{reset}\n",
        x_color.code(),
        AnsiColor::Green.code(),
        z_color.code(),
    );

    println!("  Gate colors (hue = axis pair, brightness = direction):");
    println!("  Forward follows the cycle X -> Y -> Z -> X");
    println!("  Inverse (dagger) goes against it.\n");

    // X <-> Z: Magenta (H, SY / SY†)
    let xz_fwd = pecos.ansi_gate_xz_fwd();
    let xz_inv = pecos.ansi_gate_xz_inv();
    println!("    X <-> Z  (Red + Blue = Magenta):");
    println!("      {}H       (self-adjoint){reset}", xz_fwd.code(),);
    println!(
        "      {}SY      (forward: Z -> X){reset}    {}SY dg   (inverse: X -> Z){reset}",
        xz_fwd.code(),
        xz_inv.code(),
    );

    // X <-> Y: Yellow (SZ / SZ†)
    let xy_fwd = pecos.ansi_gate_xy_fwd();
    let xy_inv = pecos.ansi_gate_xy_inv();
    println!("    X <-> Y  (Red + Green = Yellow):");
    println!(
        "      {}SZ      (forward: X -> Y){reset}    {}SZ dg   (inverse: Y -> X){reset}",
        xy_fwd.code(),
        xy_inv.code(),
    );

    // Y <-> Z: Cyan (SX / SX†)
    let yz_fwd = pecos.ansi_gate_yz_fwd();
    let yz_inv = pecos.ansi_gate_yz_inv();
    println!("    Y <-> Z  (Green + Blue = Cyan):");
    println!(
        "      {}SX      (forward: Y -> Z){reset}    {}SX dg   (inverse: Z -> Y){reset}",
        yz_fwd.code(),
        yz_inv.code(),
    );

    // All axes: Grey (F / F†)
    let xyz_fwd = pecos.ansi_gate_xyz_fwd();
    let xyz_inv = pecos.ansi_gate_xyz_inv();
    println!("    X, Y, Z  (R + G + B = Grey/White):");
    println!(
        "      {}F       (forward: X->Y->Z->X){reset}    {}F dg    (inverse: X->Z->Y->X){reset}",
        xyz_fwd.code(),
        xyz_inv.code(),
    );

    println!();
}
