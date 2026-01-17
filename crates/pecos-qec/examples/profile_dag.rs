//! Simple profiling binary for DAG backward propagator

use pecos_qec::fault_tolerance::backward_propagator::DagBackwardPropagator;
use pecos_quantum::DagCircuit;

fn build_syndrome_circuit(data_qubits: usize, ancilla_qubits: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    // Compute grid size for 2D connectivity
    let grid_size = (data_qubits as f64).sqrt().ceil() as usize;

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
        dag.pz(data_qubits + a);
    }
    for a in 0..ancilla_qubits {
        for &d in &ancilla_neighbors[a] {
            dag.cx(d, data_qubits + a);
        }
    }
    for a in 0..ancilla_qubits {
        dag.mz(data_qubits + a);
    }

    dag
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let distance = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let mode = args.get(3).map(|s| s.as_str()).unwrap_or("vec");

    let data_qubits = distance * distance;
    let ancilla_qubits = data_qubits - 1;

    println!(
        "Profiling d={} ({} data + {} ancilla) for {} iterations (mode={})",
        distance, data_qubits, ancilla_qubits, iterations, mode
    );

    let dag = build_syndrome_circuit(data_qubits, ancilla_qubits);

    match mode {
        "full" => {
            for _ in 0..iterations {
                let propagator = DagBackwardPropagator::new(&dag);
                let _map = propagator.build_influence_map();
            }
        }
        "fast" => {
            for _ in 0..iterations {
                let propagator = DagBackwardPropagator::new(&dag);
                let _map = propagator.build_influence_map_fast();
            }
        }
        "vec" | _ => {
            for _ in 0..iterations {
                let propagator = DagBackwardPropagator::new(&dag);
                let _map = propagator.build_influence_map_vec();
            }
        }
    }

    println!("Done");
}
