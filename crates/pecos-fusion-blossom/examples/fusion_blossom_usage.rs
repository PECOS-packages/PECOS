//! Example of using the Fusion Blossom decoder

use ndarray::{Array2, array};
use pecos_fusion_blossom::{
    DecodingOptions, FusionBlossomConfig, FusionBlossomDecoder, SolverType, StandardCode,
    SyndromeData,
};

#[allow(clippy::too_many_lines)] // Example demonstrates multiple usage patterns
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fusion Blossom Decoder Example ===\n");

    // Example 1: Simple manual graph construction
    println!("Example 1: Manual graph construction");
    {
        let config = FusionBlossomConfig {
            num_nodes: Some(6),
            num_observables: 2,
            solver_type: SolverType::Legacy,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config)?;

        // Add edges to create a simple surface code patch
        // Edges with observable 0
        decoder.add_edge(0, 1, &[0], Some(1.0))?;
        decoder.add_edge(1, 2, &[0], Some(1.0))?;
        decoder.add_edge(3, 4, &[0], Some(1.0))?;
        decoder.add_edge(4, 5, &[0], Some(1.0))?;

        // Edges with observable 1
        decoder.add_edge(0, 3, &[1], Some(1.0))?;
        decoder.add_edge(1, 4, &[1], Some(1.0))?;
        decoder.add_edge(2, 5, &[1], Some(1.0))?;

        // Boundary edges
        decoder.add_boundary_edge(0, &[], Some(2.0))?;
        decoder.add_boundary_edge(2, &[], Some(2.0))?;
        decoder.add_boundary_edge(3, &[], Some(2.0))?;
        decoder.add_boundary_edge(5, &[], Some(2.0))?;

        println!("{}", decoder.graph_summary());

        // Decode a syndrome
        let syndrome = array![1, 0, 1, 0, 0, 0];
        println!("Syndrome: {syndrome:?}");

        let result = decoder.decode(&syndrome.view())?;
        println!("Decoded observables: {:?}", result.observable);
        println!("Total weight: {:.2}", result.weight);
        println!("Matched edges: {:?}\n", result.matched_edges);
    }

    // Example 2: Create decoder from check matrix
    println!("Example 2: Decoder from check matrix");
    {
        // Simple repetition code check matrix
        let check_matrix: Array2<u8> = array![
            [1, 1, 0, 0, 0], // Check 0: errors 0,1
            [0, 1, 1, 0, 0], // Check 1: errors 1,2
            [0, 0, 1, 1, 0], // Check 2: errors 2,3
            [0, 0, 0, 1, 1], // Check 3: errors 3,4
        ];

        // Different weights for different error types
        let weights = vec![1.0, 1.0, 1.0, 1.0, 2.0];

        let config = FusionBlossomConfig {
            num_nodes: None, // Will be inferred
            num_observables: 5,
            solver_type: SolverType::Serial, // Using improved solver
            max_tree_size: None,
        };

        let mut decoder =
            FusionBlossomDecoder::from_check_matrix(&check_matrix, Some(&weights), config)?;

        println!("{}", decoder.graph_summary());

        // Decode a syndrome indicating errors
        let syndrome = array![1, 1, 0, 0];
        println!("Syndrome: {syndrome:?}");

        let result = decoder.decode(&syndrome.view())?;
        println!("Decoded observables: {:?}", result.observable);
        println!("Total weight: {:.2}", result.weight);
        println!("Observable errors detected: ");
        for (i, &obs) in result.observable.iter().enumerate() {
            if obs != 0 {
                println!("  - Observable {i} flipped");
            }
        }
    }

    // Example 3: Weighted matching
    println!("\nExample 3: Weighted matching with error probabilities");
    {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 3,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config)?;

        // Add edges with different weights (converted from error probabilities)
        // Weight = -log(p) where p is error probability
        let p1: f64 = 0.01;
        let p2: f64 = 0.05;
        let p3: f64 = 0.001;

        decoder.add_edge(0, 1, &[0], Some(-p1.ln()))?;
        decoder.add_edge(1, 2, &[1], Some(-p2.ln()))?;
        decoder.add_edge(2, 3, &[2], Some(-p3.ln()))?;
        decoder.add_edge(0, 2, &[0, 1], Some(-p2.ln()))?;
        decoder.add_edge(1, 3, &[1, 2], Some(-p1.ln()))?;

        // Add boundary edges
        decoder.add_boundary_edge(0, &[], Some(-p2.ln()))?;
        decoder.add_boundary_edge(3, &[], Some(-p2.ln()))?;

        println!("{}", decoder.graph_summary());

        // Decode syndrome
        let syndrome = array![1, 0, 1, 0];
        println!("Syndrome: {syndrome:?}");

        let result = decoder.decode(&syndrome.view())?;
        println!("Decoded observables: {:?}", result.observable);
        println!("Total weight: {:.6}", result.weight);
        println!(
            "Most likely error probability: {:.6}",
            (-result.weight).exp()
        );
    }

    // Example 4: Dynamic weights and erasures
    println!("\nExample 4: Dynamic weights and erasures");
    {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config)?;

        // Create a simple path graph
        decoder.add_edge(0, 1, &[0], Some(10.0))?; // edge 0
        decoder.add_edge(1, 2, &[0], Some(10.0))?; // edge 1
        decoder.add_edge(2, 3, &[0], Some(10.0))?; // edge 2

        println!("{}", decoder.graph_summary());

        // Decode with erasures (known errors on edges 0 and 2)
        let syndrome_data = SyndromeData {
            defects: vec![0, 3],
            erasures: Some(vec![0, 2]), // Mark edges 0 and 2 as erasures
            dynamic_weights: None,
        };

        let result = decoder.decode_advanced(syndrome_data)?;
        println!("With erasures - Matched edges: {:?}", result.matched_edges);
        println!("Observable: {:?}", result.observable);

        // Clear and decode with dynamic weights
        decoder.clear();

        let syndrome_data = SyndromeData {
            defects: vec![0, 3],
            erasures: None,
            dynamic_weights: Some(vec![(1, 1000)]), // Make edge 1 very cheap
        };

        let result = decoder.decode_advanced(syndrome_data)?;
        println!(
            "With dynamic weights - Matched edges: {:?}",
            result.matched_edges
        );
    }

    // Example 5: Standard QEC codes
    println!("\nExample 5: Standard QEC codes");
    {
        // Create a code capacity planar code
        let code = StandardCode::CodeCapacityPlanar {
            d: 5,
            p: 0.01,
            max_half_weight: 1000,
        };

        let config = FusionBlossomConfig {
            num_nodes: None,
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: Some(10), // Use hybrid union-find/MWPM
        };

        let decoder = FusionBlossomDecoder::from_standard_code(code, config)?;
        println!("Planar code d=5: {}", decoder.graph_summary());

        // Create a phenomenological rotated code
        let code = StandardCode::PhenomenologicalRotated {
            d: 3,
            p: 0.01,
            p_measurement: 0.02,
            max_half_weight: 1000,
        };

        let config = FusionBlossomConfig::default();

        let decoder = FusionBlossomDecoder::from_standard_code(code, config)?;
        println!("Rotated code d=3: {}", decoder.graph_summary());
    }

    // Example 6: Solver comparison and reuse
    println!("\nExample 6: Solver comparison and reuse");
    {
        let mut config = FusionBlossomConfig {
            num_nodes: Some(8),
            num_observables: 1,
            solver_type: SolverType::Legacy,
            max_tree_size: None,
        };

        let mut decoder_legacy = FusionBlossomDecoder::new(config)?;
        config.solver_type = SolverType::Serial;
        let mut decoder_serial = FusionBlossomDecoder::new(config)?;

        // Build same graph for both
        for decoder in [&mut decoder_legacy, &mut decoder_serial] {
            for i in 0..7 {
                decoder.add_edge(i, i + 1, &[0], Some(1.0))?;
            }
            decoder.add_edge(0, 7, &[0], Some(2.0))?; // Ring closure
        }

        // Test multiple syndromes
        let syndromes = [
            array![1, 0, 0, 0, 0, 0, 0, 1],
            array![0, 1, 0, 0, 0, 1, 0, 0],
            array![1, 0, 1, 0, 1, 0, 1, 0],
        ];

        println!("Comparing Legacy vs Serial solver:");
        for (i, syndrome) in syndromes.iter().enumerate() {
            let result_legacy = decoder_legacy.decode(&syndrome.view())?;
            let result_serial = decoder_serial.decode(&syndrome.view())?;

            println!(
                "  Syndrome {}: Legacy weight={:.2}, Serial weight={:.2}",
                i, result_legacy.weight, result_serial.weight
            );

            // Clear for next iteration
            decoder_legacy.clear();
            decoder_serial.clear();
        }
    }

    // Example 7: Perfect matching details
    println!("\nExample 7: Perfect matching details");
    {
        let config = FusionBlossomConfig {
            num_nodes: Some(6),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config)?;

        // Create a simple graph
        decoder.add_edge(0, 1, &[0], Some(1.0))?;
        decoder.add_edge(1, 2, &[0], Some(2.0))?;
        decoder.add_edge(3, 4, &[0], Some(1.0))?;
        decoder.add_edge(4, 5, &[0], Some(2.0))?;
        decoder.add_boundary_edge(0, &[], Some(3.0))?;
        decoder.add_boundary_edge(2, &[], Some(3.0))?;
        decoder.add_boundary_edge(3, &[], Some(3.0))?;
        decoder.add_boundary_edge(5, &[], Some(3.0))?;

        let syndrome_data = SyndromeData::from_defects(vec![0, 2, 3, 5]);
        let options = DecodingOptions {
            include_perfect_matching: true,
        };

        let result = decoder.decode_with_options(syndrome_data, options)?;

        println!("Decoding result:");
        println!("  Total weight: {:.2}", result.weight);
        println!("  Matched edges: {:?}", result.matched_edges);

        if let Some(pm) = result.perfect_matching {
            println!("  Perfect matching details:");
            println!("    Number of matches: {}", pm.match_count);
            for (v1, v2, is_virtual) in pm.matched_pairs {
                let virtual_str = if is_virtual {
                    " (includes virtual)"
                } else {
                    ""
                };
                println!("    Matched: {v1} <-> {v2}{virtual_str}");
            }
        } else {
            println!("  Perfect matching details not available for this solver type");
        }
    }

    // Example 8: Solver performance comparison
    println!("\nExample 8: Solver performance comparison");
    {
        use std::time::Instant;

        // Create a larger graph for performance testing
        let size = 20;
        let num_nodes = size * size;

        for solver_type in [SolverType::Legacy, SolverType::Serial] {
            let config = FusionBlossomConfig {
                num_nodes: Some(num_nodes),
                num_observables: 1,
                solver_type,
                max_tree_size: None,
            };

            let mut decoder = FusionBlossomDecoder::new(config)?;

            // Create a grid graph
            for i in 0..size {
                for j in 0..size {
                    let node = i * size + j;
                    // Right edge
                    if j < size - 1 {
                        decoder.add_edge(node, node + 1, &[0], Some(1.0))?;
                    }
                    // Down edge
                    if i < size - 1 {
                        decoder.add_edge(node, node + size, &[0], Some(1.0))?;
                    }
                }
            }

            // Create random syndrome
            let mut syndrome = vec![0u8; num_nodes];
            syndrome[0] = 1;
            syndrome[num_nodes / 2] = 1;
            syndrome[num_nodes - 1] = 1;
            syndrome[num_nodes / 4] = 1;

            let syndrome_array = ndarray::Array1::from_vec(syndrome);

            let start = Instant::now();
            let result = decoder.decode(&syndrome_array.view())?;
            let elapsed = start.elapsed();

            println!(
                "  {:?}: {:.3} ms, weight={:.2}",
                solver_type,
                elapsed.as_secs_f64() * 1000.0,
                result.weight
            );
        }
    }

    Ok(())
}
