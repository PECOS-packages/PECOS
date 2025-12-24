//! Example showing `PyMatching` API usage

use pecos_pymatching::{
    BatchConfig, CheckMatrix, CheckMatrixConfig, MergeStrategy, PyMatchingConfig, PyMatchingDecoder,
};

use std::path::Path;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PyMatching API Example");
    println!("========================\n");

    // Example 1: Create decoder using builder pattern
    println!("Example 1: Creating decoder with builder pattern");
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(6)
        .observables(2)
        .build()?;
    println!(
        "Created decoder with {} nodes and {} observables",
        decoder.num_nodes(),
        decoder.num_observables()
    );

    // Add edges to create a simple matching graph
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None)?;
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None)?;
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None)?;
    decoder.add_edge(3, 4, &[1], Some(1.0), None, None)?;
    decoder.add_edge(4, 5, &[0], Some(1.0), None, None)?;

    // Add boundary edges
    decoder.add_boundary_edge(0, &[], Some(1.0), None, None)?;
    decoder.add_boundary_edge(5, &[], Some(1.0), None, None)?;

    println!("Added {} edges", decoder.num_edges());

    // Example 2: Decode detection events
    println!("\nExample 2: Decoding detection events");
    let mut detection_events = vec![0u8; 6];
    detection_events[1] = 1; // Detection at node 1
    detection_events[4] = 1; // Detection at node 4

    let result = decoder.decode(&detection_events).unwrap();
    println!(
        "Decoding result: observables = {:?}, weight = {}",
        result.observable, result.weight
    );

    // Example 3: Load from DEM string
    println!("\nExample 3: Loading from DEM string");
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 D3 L0
    ";

    match PyMatchingDecoder::from_dem(dem_string) {
        Ok(mut dem_decoder) => {
            println!(
                "Loaded decoder from DEM with {} detectors and {} observables",
                dem_decoder.num_detectors(),
                dem_decoder.num_observables()
            );

            // Decode with DEM decoder
            let mut events = vec![0u8; dem_decoder.num_detectors()];
            if events.len() >= 2 {
                events[0] = 1;
                events[1] = 1;
                let result = dem_decoder.decode(&events).unwrap();
                println!("DEM decoding result: observables = {:?}", result.observable);
            }
        }
        Err(e) => println!("Failed to load from DEM: {e}"),
    }

    // Example 4: Demonstrate merge strategies
    println!("\nExample 4: Edge merge strategies");
    let mut merge_decoder = PyMatchingDecoder::new(PyMatchingConfig {
        num_nodes: Some(3),
        num_observables: 2,
        ..Default::default()
    })?;

    // Add initial edge
    merge_decoder.add_edge(0, 1, &[0], Some(2.0), None, None)?;

    // Try to add parallel edge with SmallestWeight strategy
    merge_decoder.add_edge(
        0,
        1,
        &[1],
        Some(1.0),
        None,
        Some(MergeStrategy::SmallestWeight),
    )?;

    let edge_data = merge_decoder.get_edge_data(0, 1)?;
    println!(
        "After merge with SmallestWeight: weight = {}",
        edge_data.weight
    );

    // Example 5: Batch decoding
    println!("\nExample 5: Batch decoding");
    let batch_config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut batch_decoder = PyMatchingDecoder::new(batch_config)?;

    // Create a simple square
    batch_decoder.add_edge(0, 1, &[0], Some(1.0), None, None)?;
    batch_decoder.add_edge(1, 2, &[1], Some(1.0), None, None)?;
    batch_decoder.add_edge(2, 3, &[0], Some(1.0), None, None)?;
    batch_decoder.add_edge(3, 0, &[1], Some(1.0), None, None)?;

    // Method 1: Low-level batch decode
    let num_shots = 3;
    let num_detectors = 4;
    let mut shots = vec![0u8; num_shots * num_detectors];

    // Shot 0: detections at 0 and 2
    shots[0] = 1;
    shots[2] = 1;

    // Shot 1: detections at 1 and 3
    shots[4 + 1] = 1;
    shots[4 + 3] = 1;

    // Shot 2: no detections

    let batch_result = batch_decoder.decode_batch_with_config(
        &shots,
        num_shots,
        num_detectors,
        BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: true,
        },
    )?;
    println!("Batch decoded {} shots", batch_result.predictions.len());
    for (i, weight) in batch_result.weights.iter().enumerate() {
        println!("  Shot {i}: weight = {weight}");
    }

    // Method 2: Using decode_batch_with_config (modern API)
    println!("\nUsing decode_batch_with_config:");
    let shot_vecs = [
        vec![1, 0, 1, 0], // Shot 0
        vec![0, 1, 0, 1], // Shot 1
        vec![0, 0, 0, 0], // Shot 2
    ];

    // Flatten shots for decode_batch
    let flat_shots: Vec<u8> = shot_vecs.iter().flatten().copied().collect();
    let batch_config = BatchConfig {
        bit_packed_input: false,
        bit_packed_output: false,
        return_weights: true,
    };

    let corrections = batch_decoder.decode_batch_with_config(
        &flat_shots,
        shot_vecs.len(),
        shot_vecs[0].len(),
        batch_config,
    )?;
    println!("Got {} corrections", corrections.predictions.len());
    for i in 0..shot_vecs.len() {
        println!(
            "  Shot {}: weight = {}",
            i,
            corrections.weights.get(i).unwrap_or(&0.0)
        );
    }

    // Example 6: Check matrix support
    println!("\nExample 6: Creating decoder from check matrix");

    // Create a simple repetition code check matrix
    // H = [[1, 1, 0, 0],
    //      [0, 1, 1, 0],
    //      [0, 0, 1, 1]]
    let check_matrix = vec![
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 3, 1),
    ];

    let matrix =
        CheckMatrix::from_triplets(check_matrix, 3, 4).with_weights(vec![1.0, 1.5, 1.5, 1.0])?;
    let matrix_decoder = PyMatchingDecoder::from_check_matrix(&matrix)?;

    println!(
        "Check matrix decoder has {} nodes",
        matrix_decoder.num_nodes()
    );

    // Example 7: Dense check matrix
    println!("\nExample 7: Dense check matrix");
    let dense_matrix = vec![vec![1, 1, 0, 0], vec![0, 1, 1, 0], vec![0, 0, 1, 1]];

    let dense_check_matrix = CheckMatrix::from_dense_vec(&dense_matrix)?;
    let config = CheckMatrixConfig {
        use_virtual_boundary: false,
        ..Default::default()
    };
    let dense_decoder =
        PyMatchingDecoder::from_check_matrix_with_config(&dense_check_matrix, config)?;

    println!(
        "Dense matrix decoder has {} nodes",
        dense_decoder.num_nodes()
    );

    // Example 7b: Advanced check matrix API with configuration
    println!("\nExample 7b: Advanced check matrix API with configuration");
    // Using the advanced API when you need custom configuration
    let advanced_check_matrix = vec![
        (0, 0, 1),
        (0, 1, 1), // Check 0 involves qubits 0 and 1
        (1, 1, 1),
        (1, 2, 1), // Check 1 involves qubits 1 and 2
        (2, 2, 1),
        (2, 3, 1), // Check 2 involves qubits 2 and 3
    ];

    let advanced_matrix = CheckMatrix::from_triplets(advanced_check_matrix, 3, 4)
        .with_weights(vec![1.0, 1.5, 1.5, 1.0])?;

    let advanced_config = CheckMatrixConfig {
        repetitions: 2,
        error_probabilities: Some(vec![0.01, 0.02, 0.02, 0.01]),
        timelike_weights: Some(vec![1.0, 1.0, 1.0]),
        measurement_error_probabilities: Some(vec![0.001, 0.001, 0.001]),
        use_virtual_boundary: true,
        weights: None, // Use weights from matrix
    };

    let advanced_decoder =
        PyMatchingDecoder::from_check_matrix_with_config(&advanced_matrix, advanced_config)?;

    println!(
        "Advanced API decoder has {} nodes and {} observables",
        advanced_decoder.num_nodes(),
        advanced_decoder.num_observables()
    );

    // Example 8: File I/O (if file exists)
    println!("\nExample 8: File I/O");
    let dem_path = Path::new("example.dem");
    if dem_path.exists() {
        match PyMatchingDecoder::from_dem_file(dem_path) {
            Ok(file_decoder) => {
                println!(
                    "Loaded decoder from file with {} detectors",
                    file_decoder.num_detectors()
                );
            }
            Err(e) => println!("Error loading from file: {e}"),
        }
    } else {
        println!("No example.dem file found, skipping file I/O test");
    }

    // Example 9: Advanced decoding outputs
    println!("\nExample 9: Advanced decoding outputs");

    // Decode to matched pairs
    match decoder.decode_to_matched_pairs(&detection_events) {
        Ok(pairs) => {
            println!("Matched detection event pairs:");
            for pair in pairs {
                match pair.detector2 {
                    Some(d2) => println!("  Detection {} matched with {}", pair.detector1, d2),
                    None => println!("  Detection {} matched to boundary", pair.detector1),
                }
            }
        }
        Err(e) => println!("decode_to_matched_pairs error: {e}"),
    }

    // Decode to matched pairs dictionary
    match decoder.decode_to_matched_pairs_dict(&detection_events) {
        Ok(match_dict) => {
            println!("\nMatched pairs as dictionary:");
            for (det, partner) in &match_dict {
                match partner {
                    Some(p) => println!("  {det} -> {p}"),
                    None => println!("  {det} -> boundary"),
                }
            }

            // Check specific match
            if let Some(match_for_1) = match_dict.get(&1) {
                println!("\nDetection event 1 is matched to: {match_for_1:?}");
            }
        }
        Err(e) => println!("decode_to_matched_pairs_dict error: {e}"),
    }

    // Decode to edges
    match decoder.decode_to_edges(&detection_events) {
        Ok(edges) => {
            println!("\nEdges in matching solution:");
            for edge in edges {
                match edge.detector2 {
                    Some(d2) => println!("  Edge: detector {} - detector {}", edge.detector1, d2),
                    None => println!("  Edge: detector {} - boundary", edge.detector1),
                }
            }
        }
        Err(e) => println!("decode_to_edges error: {e}"),
    }

    // Example 10: Noise simulation
    println!("\nExample 10: Noise simulation");
    match decoder.add_noise(5, 42) {
        Ok(noise_result) => {
            println!("Generated {} noise samples", noise_result.errors.len());
            for (i, (errors, syndrome)) in noise_result
                .errors
                .iter()
                .zip(noise_result.syndromes.iter())
                .enumerate()
            {
                let error_count = errors.iter().filter(|&&e| e != 0).count();
                let syndrome_count = syndrome.iter().filter(|&&s| s != 0).count();
                println!("  Sample {i}: {error_count} errors, {syndrome_count} syndrome bits");
            }
        }
        Err(e) => println!("add_noise error: {e}"),
    }

    // Example 11: Path finding
    println!("\nExample 11: Path finding");
    match decoder.get_shortest_path(0, 5) {
        Ok(path) => {
            println!("Shortest path from 0 to 5: {path:?}");
            println!("Path length: {} nodes", path.len());
        }
        Err(e) => println!("get_shortest_path error: {e}"),
    }

    // Test path with boundary
    match decoder.get_shortest_path(0, 3) {
        Ok(path) => {
            println!("Shortest path from 0 to 3: {path:?}");
        }
        Err(e) => println!("get_shortest_path error: {e}"),
    }

    // Example 12: Random Number Generation
    println!("\nExample 12: Random Number Generation");

    // Set seed for reproducibility
    PyMatchingDecoder::set_seed(12345)?;
    println!("Set RNG seed to 12345");

    // Generate some random floats
    for i in 0..5 {
        let r = PyMatchingDecoder::rand_float(0.0, 1.0)?;
        println!("  Random float {i}: {r:.6}");
    }

    // Randomize seed
    PyMatchingDecoder::randomize()?;
    println!("\nRandomized RNG seed");

    // Generate more random floats (will be different)
    for _ in 0..3 {
        let r = PyMatchingDecoder::rand_float(10.0, 20.0)?;
        println!("  Random float in [10, 20): {r:.6}");
    }

    println!("\nPyMatching example complete!");
    Ok(())
}
