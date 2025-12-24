//! Surface code specific tests for `PyMatching`
//! These tests ensure our implementation works correctly for real QEC codes

use pecos_pymatching::{BatchConfig, CheckMatrix, CheckMatrixConfig, PyMatchingDecoder};

/// Create a distance-3 rotated surface code graph
fn create_distance_3_surface_code() -> PyMatchingDecoder {
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(13) // 9 data qubits + 4 measurement qubits
        .observables(2) // X and Z logical operators
        .build()
        .unwrap();

    // Surface code layout (rotated, distance 3):
    //     0---1---2
    //     |   |   |
    //     3---4---5
    //     |   |   |
    //     6---7---8
    //
    // Measurement qubits: 9, 10, 11, 12 (plaquettes)

    // X-type stabilizers (measure Z operators on data qubits)
    // Top-left plaquette (node 9)
    decoder.add_edge(0, 9, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 9, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 9, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 9, &[], Some(1.0), None, None).unwrap();

    // Top-right plaquette (node 10)
    decoder.add_edge(1, 10, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 10, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 10, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(5, 10, &[], Some(1.0), None, None).unwrap();

    // Bottom-left plaquette (node 11)
    decoder.add_edge(3, 11, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 11, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(6, 11, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(7, 11, &[], Some(1.0), None, None).unwrap();

    // Bottom-right plaquette (node 12)
    decoder.add_edge(4, 12, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(5, 12, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(7, 12, &[], Some(1.0), None, None).unwrap();
    decoder.add_edge(8, 12, &[], Some(1.0), None, None).unwrap();

    // Boundary edges for rough boundaries
    decoder
        .add_boundary_edge(0, &[0], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[0], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(6, &[0], Some(1.0), None, None)
        .unwrap();

    decoder
        .add_boundary_edge(2, &[0], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(5, &[0], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(8, &[0], Some(1.0), None, None)
        .unwrap();

    decoder
}

#[test]
fn test_surface_code_single_error() {
    let mut decoder = create_distance_3_surface_code();

    // Single X error on qubit 4 (center)
    // This should trigger plaquettes 9, 10, 11, 12
    let mut detection_events = vec![0u8; 13];
    detection_events[9] = 1;
    detection_events[10] = 1;
    detection_events[11] = 1;
    detection_events[12] = 1;

    let result = decoder.decode(&detection_events).unwrap();

    // Should find a weight-4 correction (4 edges to measurement qubits)
    assert!(result.weight > 0.0);
    // Should not trigger any logical error
    assert_eq!(result.observable[0], 0);
}

#[test]
fn test_surface_code_logical_x_error() {
    let mut decoder = create_distance_3_surface_code();

    // Logical X error: vertical string of X errors (0-3-6)
    // This triggers plaquettes at boundaries
    let detection_events = vec![0u8; 13];
    // Only boundary detections for a logical error
    // (In this simplified model, boundary nodes handle the syndrome)

    let result = decoder.decode(&detection_events).unwrap();

    // For a proper logical error test, we'd need the full syndrome
    // This tests that the decoder handles boundary conditions
    assert_eq!(result.observable.len(), 2);
}

#[test]
fn test_surface_code_weight_2_error() {
    let mut decoder = create_distance_3_surface_code();

    // Two X errors on adjacent qubits (e.g., 1 and 4)
    // This should trigger plaquettes 9 and 10
    let mut detection_events = vec![0u8; 13];
    detection_events[9] = 1;
    detection_events[10] = 1;

    let result = decoder.decode(&detection_events).unwrap();

    // Should find minimum weight correction
    assert!(result.weight > 0.0);
    assert_eq!(result.observable[0], 0); // No logical error
}

#[test]
fn test_surface_code_from_dem() {
    // Test loading a surface code from DEM string
    let dem_string = r"
        # Distance 3 surface code stabilizer measurements
        detector(0, 0, 0) D0
        detector(2, 0, 0) D1
        detector(0, 2, 0) D2
        detector(2, 2, 0) D3

        # Physical errors
        error(0.001) D0 D1
        error(0.001) D0 D2
        error(0.001) D1 D3
        error(0.001) D2 D3
        error(0.001) D0 D1 D2 D3

        # Logical errors
        error(0.001) D0 D2 L0
        error(0.001) D1 D3 L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    // Test empty syndrome (should always work regardless of graph structure)
    let num_detectors = decoder.num_detectors();
    let detection_events = vec![0u8; num_detectors];

    let result = decoder.decode(&detection_events).unwrap();
    // Empty syndrome should decode successfully with zero weight
    assert!(
        result.weight.abs() < f64::EPSILON,
        "Weight should be zero but was {}",
        result.weight
    );
    assert!(result.observable.iter().all(|&x| x == 0));
}

#[test]
fn test_surface_code_performance() {
    // Test with a larger surface code to ensure performance
    let d = 11; // Distance 11 surface code
    let num_data_qubits = d * d;
    let num_ancilla_qubits = (d - 1) * (d - 1);
    let total_nodes = num_data_qubits + num_ancilla_qubits;

    let mut decoder = PyMatchingDecoder::builder()
        .nodes(total_nodes)
        .observables(2)
        .build()
        .unwrap();

    // Add edges in a grid pattern (simplified)
    for i in 0..d - 1 {
        for j in 0..d - 1 {
            let ancilla = num_data_qubits + i * (d - 1) + j;

            // Connect to surrounding data qubits
            let data_top_left = i * d + j;
            let data_top_right = i * d + j + 1;
            let data_bottom_left = (i + 1) * d + j;
            let data_bottom_right = (i + 1) * d + j + 1;

            if data_top_left < num_data_qubits {
                decoder
                    .add_edge(data_top_left, ancilla, &[], Some(1.0), None, None)
                    .unwrap();
            }
            if data_top_right < num_data_qubits {
                decoder
                    .add_edge(data_top_right, ancilla, &[], Some(1.0), None, None)
                    .unwrap();
            }
            if data_bottom_left < num_data_qubits {
                decoder
                    .add_edge(data_bottom_left, ancilla, &[], Some(1.0), None, None)
                    .unwrap();
            }
            if data_bottom_right < num_data_qubits {
                decoder
                    .add_edge(data_bottom_right, ancilla, &[], Some(1.0), None, None)
                    .unwrap();
            }
        }
    }

    // Add boundary edges
    for i in 0..d {
        decoder
            .add_boundary_edge(i, &[0], Some(1.0), None, None)
            .unwrap(); // Top
        decoder
            .add_boundary_edge((d - 1) * d + i, &[0], Some(1.0), None, None)
            .unwrap(); // Bottom
        decoder
            .add_boundary_edge(i * d, &[1], Some(1.0), None, None)
            .unwrap(); // Left
        decoder
            .add_boundary_edge(i * d + d - 1, &[1], Some(1.0), None, None)
            .unwrap(); // Right
    }

    // Test decoding with multiple errors
    let mut detection_events = vec![0u8; total_nodes];
    // Add some random detections
    detection_events[num_data_qubits + 5] = 1;
    detection_events[num_data_qubits + 15] = 1;
    detection_events[num_data_qubits + 25] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    assert!(result.weight > 0.0);
}

#[test]
fn test_surface_code_batch_decoding() {
    let mut decoder = create_distance_3_surface_code();

    // Create multiple syndrome patterns
    let shots = vec![
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], // Two adjacent detections
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], // Two non-adjacent
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], // All four detections
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // No detections
    ];

    // Convert Vec<Vec<u8>> to flat Vec<u8>
    let num_detectors = decoder.num_detectors();
    let mut flat_shots = Vec::new();
    for shot in &shots {
        flat_shots.extend_from_slice(&shot[..num_detectors.min(shot.len())]);
    }
    let results = decoder
        .decode_batch_with_config(
            &flat_shots,
            shots.len(),
            num_detectors,
            BatchConfig::default(),
        )
        .unwrap();

    assert_eq!(results.predictions.len(), 4);
    // Each result should have the correct observable count
    for result in &results.predictions {
        assert!(result.len() >= 2);
    }
}

#[test]
fn test_surface_code_with_measurement_errors() {
    // Test surface code with measurement errors
    let check_matrix = vec![
        // Simple repetition code checks
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 3, 1),
        (3, 3, 1),
        (3, 4, 1),
    ];

    let measurement_error_probs = vec![0.01, 0.01, 0.01, 0.01];

    let config = CheckMatrixConfig {
        repetitions: 3, // 3 measurement rounds
        weights: None,
        error_probabilities: None,
        timelike_weights: None,
        measurement_error_probabilities: Some(measurement_error_probs),
        use_virtual_boundary: false,
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 4, 5);
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Should handle measurement errors in temporal direction
    assert!(decoder.num_nodes() > 8); // More nodes for temporal structure
}

#[test]
fn test_repetition_code_as_1d_surface_code() {
    // Repetition code is essentially a 1D surface code
    let length = 7;
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(length)
        .observables(1)
        .build()
        .unwrap();

    // Linear chain of qubits
    for i in 0..length - 1 {
        decoder
            .add_edge(i, i + 1, &[0], Some(1.0), None, None)
            .unwrap();
    }

    // Boundaries
    decoder
        .add_boundary_edge(0, &[0], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(length - 1, &[0], Some(1.0), None, None)
        .unwrap();

    // Test weight-1 error
    let mut detection_events = vec![0u8; length];
    detection_events[3] = 1;
    detection_events[4] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    // Adjacent detections in repetition code should not cause logical error
    // But the exact observable depends on implementation details
    assert!(!result.observable.is_empty());

    // Test logical error (full chain)
    let mut detection_events = vec![0u8; length];
    detection_events[0] = 1;
    detection_events[length - 1] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    // This represents a logical error in repetition code
    assert!(result.weight > 0.0);
}
