//! Simple profiling binary for DAG fault analyzer

use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;

fn build_syndrome_circuit(data_qubits: usize, ancilla_qubits: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    // Compute grid size for 2D connectivity (ceiling of sqrt)
    let s = data_qubits.isqrt();
    let grid_size = if s * s == data_qubits { s } else { s + 1 };

    // Build connectivity map
    let mut ancilla_neighbors: Vec<Vec<usize>> = Vec::with_capacity(ancilla_qubits);
    for a_idx in 0..ancilla_qubits {
        let row = a_idx / (grid_size - 1).max(1);
        let col = a_idx % (grid_size - 1).max(1);
        let mut neighbors = Vec::with_capacity(4);
        let offsets = [(0, 0), (0, 1), (1, 0), (1, 1)];
        for (dr, dc) in offsets {
            let data_row = row + dr;
            let data_col = col + dc;
            if data_row < grid_size && data_col < grid_size {
                let data_idx = data_row * grid_size + data_col;
                if data_idx < data_qubits {
                    neighbors.push(data_idx);
                }
            }
        }
        ancilla_neighbors.push(neighbors);
    }

    // Build circuit
    for a in 0..ancilla_qubits {
        dag.pz(&[data_qubits + a]);
    }
    for (a, neighbors) in ancilla_neighbors.iter().enumerate() {
        for &d in neighbors {
            dag.cx(&[(d, data_qubits + a)]);
        }
    }
    for a in 0..ancilla_qubits {
        dag.mz(&[data_qubits + a]);
    }

    dag
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let distance = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let reuse = args.get(3).is_some_and(|s| s == "reuse");

    let data_qubits = distance * distance;
    let ancilla_qubits = data_qubits - 1;

    println!(
        "Profiling d={distance} ({data_qubits} data + {ancilla_qubits} ancilla) for {iterations} iterations"
    );

    let dag = build_syndrome_circuit(data_qubits, ancilla_qubits);

    if reuse {
        // Reuse mode: create propagator once, build map multiple times
        let propagator = DagFaultAnalyzer::new(&dag);
        for _ in 0..iterations {
            let _map = propagator.build_influence_map();
        }
    } else {
        // Default: create new propagator each iteration
        for _ in 0..iterations {
            let propagator = DagFaultAnalyzer::new(&dag);
            let _map = propagator.build_influence_map();
        }
    }

    // Print some stats at the end
    let propagator = DagFaultAnalyzer::new(&dag);
    let map = propagator.build_influence_map();
    let stats = map.memory_stats();

    println!("\n=== Statistics ===");
    println!("Locations: {}", map.locations.len());
    println!("Detectors: {}", map.detectors.len());
    println!("Total bytes: {}", stats.total_bytes);
    println!("Done");
}
