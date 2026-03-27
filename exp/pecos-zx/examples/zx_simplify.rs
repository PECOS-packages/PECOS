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

//! ZX simplification: redundant circuit -> simplify -> extract optimized circuit.
//!
//! Demonstrates:
//! - Building a circuit with redundant gates (H*H, CX*CX)
//! - Converting to ZX and simplifying
//! - Extracting the optimized circuit back
//! - Comparing gate counts

use pecos_quantum::DagCircuit;
use pecos_zx::convert::{dag_to_zx, zx_to_dag};
use pecos_zx::simplify;
use pecos_zx::viz::Renderer;

fn main() {
    // Build a redundant circuit:
    // H(0), H(0) -- cancels to identity
    // CX(0,1), CX(0,1) -- cancels to identity
    // H(1), T(1), H(1) -- does not cancel but simplifies
    let mut dag = DagCircuit::new();
    dag.h(&[0]);
    dag.h(&[0]); // H*H = I
    dag.cx(&[(0, 1)]);
    dag.cx(&[(0, 1)]); // CX*CX = I
    dag.h(&[1]);
    dag.sz(&[1]);
    dag.h(&[1]);

    let original_count = dag.gate_count();
    println!("=== ZX Simplification ===\n");
    println!("Original circuit: {original_count} gates");
    println!("  H(0), H(0), CX(0,1), CX(0,1), H(1), S(1), H(1)\n");

    // Convert to ZX
    let mut graph = dag_to_zx(&dag).expect("conversion failed");

    // Render before
    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "simplify_before");

    // Simplify
    simplify::clifford_simp(&mut graph);

    // Render after
    println!();
    r.render(&graph, "simplify_after");

    // Extract back to circuit
    match zx_to_dag(&graph) {
        Ok(optimized) => {
            let optimized_count = optimized.gate_count();
            println!("\nOptimized circuit: {optimized_count} gates");
            println!(
                "Reduction: {} -> {} ({:.0}% fewer)",
                original_count,
                optimized_count,
                (1.0 - optimized_count as f64 / original_count as f64) * 100.0
            );
        }
        Err(e) => {
            println!("\nCould not extract circuit: {e}");
            println!("(This is expected for some simplified graphs that lack gflow)");
        }
    }
}
