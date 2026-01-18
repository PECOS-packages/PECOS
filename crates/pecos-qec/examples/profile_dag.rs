//! Simple profiling binary for DAG fault analyzer

use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
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
    let mode = args.get(3).map_or("soa", std::string::String::as_str);

    let data_qubits = distance * distance;
    let ancilla_qubits = data_qubits - 1;

    println!(
        "Profiling d={distance} ({data_qubits} data + {ancilla_qubits} ancilla) for {iterations} iterations (mode={mode})"
    );

    let dag = build_syndrome_circuit(data_qubits, ancilla_qubits);

    match mode {
        "btree" => {
            // Original BTreeMap-based implementation
            for _ in 0..iterations {
                let propagator = DagFaultAnalyzer::new(&dag);
                let _map = propagator.build_influence_map();
            }
        }
        "soa" => {
            // Optimized SoA (Struct of Arrays) implementation
            for _ in 0..iterations {
                let propagator = DagFaultAnalyzer::new(&dag);
                let _map = propagator.build_influence_map_soa();
            }
        }
        "compare" => {
            // Compare btree vs soa performance
            let propagator = DagFaultAnalyzer::new(&dag);

            // Warm up
            let _ = propagator.build_influence_map();
            let _ = propagator.build_influence_map_soa();

            // Benchmark btree
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _map = propagator.build_influence_map();
            }
            let btree_time = start.elapsed();

            // Benchmark soa
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _map = propagator.build_influence_map_soa();
            }
            let soa_time = start.elapsed();

            // Memory comparison
            let btree_map = propagator.build_influence_map();
            let soa_map = propagator.build_influence_map_soa();
            let soa_stats = soa_map.memory_stats();

            let btree_us = btree_time.as_micros() as f64 / f64::from(iterations);
            let soa_us = soa_time.as_micros() as f64 / f64::from(iterations);

            println!("\n=== Performance Comparison ===");
            println!("BTree: {btree_us:>8.2} us/iter (baseline)");
            println!("SoA:   {:>8.2} us/iter ({:.2}x)", soa_us, soa_us / btree_us);

            println!("\n=== Memory Statistics ===");
            println!("Locations: {}", btree_map.influences.len());
            println!("Detectors: {}", btree_map.detectors.len());
            println!("SoA total bytes: {}", soa_stats.total_bytes);

            return;
        }
        // Reuse mode: create propagator once, build map multiple times
        "reuse" | _ => {
            let propagator = DagFaultAnalyzer::new(&dag);
            for _ in 0..iterations {
                let _map = propagator.build_influence_map_soa();
            }
        }
    }

    println!("Done");
}
