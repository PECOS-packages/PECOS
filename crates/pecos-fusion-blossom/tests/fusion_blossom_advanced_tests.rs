//! Advanced tests for Fusion Blossom decoder

mod tests {
    use ndarray::array;
    use pecos_fusion_blossom::{
        FusionBlossomConfig, FusionBlossomDecoder, SolverType, StandardCode, SyndromeData,
    };

    #[test]
    fn test_solver_types() {
        let config_legacy = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Legacy,
            max_tree_size: None,
        };

        let config_serial = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder_legacy = FusionBlossomDecoder::new(config_legacy).unwrap();
        let mut decoder_serial = FusionBlossomDecoder::new(config_serial).unwrap();

        // Add same edges to both
        decoder_legacy.add_edge(0, 1, &[0], Some(1.0)).unwrap();
        decoder_legacy.add_edge(1, 2, &[0], Some(1.0)).unwrap();
        decoder_legacy.add_edge(2, 3, &[0], Some(1.0)).unwrap();

        decoder_serial.add_edge(0, 1, &[0], Some(1.0)).unwrap();
        decoder_serial.add_edge(1, 2, &[0], Some(1.0)).unwrap();
        decoder_serial.add_edge(2, 3, &[0], Some(1.0)).unwrap();

        // Test both solvers produce same result
        let syndrome = array![1, 0, 0, 1];

        let result_legacy = decoder_legacy.decode(&syndrome.view()).unwrap();
        let result_serial = decoder_serial.decode(&syndrome.view()).unwrap();

        assert_eq!(result_legacy.observable, result_serial.observable);
        assert!(
            (result_legacy.weight - result_serial.weight).abs() < f64::EPSILON,
            "Legacy and serial solvers gave different weights: {} vs {}",
            result_legacy.weight,
            result_serial.weight
        );
    }

    #[test]
    fn test_dynamic_weights() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 3,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Create edges with default weights
        decoder.add_edge(0, 1, &[0], Some(10.0)).unwrap(); // edge 0
        decoder.add_edge(1, 2, &[1], Some(10.0)).unwrap(); // edge 1
        decoder.add_edge(2, 3, &[2], Some(10.0)).unwrap(); // edge 2
        decoder.add_boundary_edge(0, &[], Some(5.0)).unwrap(); // edge 3
        decoder.add_boundary_edge(3, &[], Some(5.0)).unwrap(); // edge 4

        // Decode with dynamic weights - make boundary edges cheaper
        let syndrome_data = SyndromeData {
            defects: vec![0, 3],
            erasures: None,
            dynamic_weights: Some(vec![(3, 1000), (4, 1000)]), // Very low weights for boundary edges
        };

        let result = decoder.decode_advanced(syndrome_data).unwrap();

        // Should use boundary edges due to lower dynamic weights
        assert!(result.matched_edges.contains(&3) || result.matched_edges.contains(&4));
    }

    #[test]
    fn test_erasure_decoding() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 3,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Create a path graph
        decoder.add_edge(0, 1, &[0], Some(10.0)).unwrap(); // edge 0
        decoder.add_edge(1, 2, &[1], Some(10.0)).unwrap(); // edge 1
        decoder.add_edge(2, 3, &[2], Some(10.0)).unwrap(); // edge 2

        // Mark edge 0 and 2 as erasures (known errors)
        let syndrome_data = SyndromeData {
            defects: vec![0, 3],
            erasures: Some(vec![0, 2]),
            dynamic_weights: None,
        };

        let result = decoder.decode_advanced(syndrome_data).unwrap();

        // Should include the erasure edges
        assert!(result.matched_edges.contains(&0));
        assert!(result.matched_edges.contains(&2));
    }

    #[test]
    fn test_clear_and_reuse() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
        decoder.add_edge(1, 2, &[0], Some(1.0)).unwrap();
        decoder.add_edge(2, 3, &[0], Some(1.0)).unwrap();

        // First decode
        let syndrome1 = array![1, 0, 0, 1];
        let result1 = decoder.decode(&syndrome1.view()).unwrap();

        // Clear and decode again
        decoder.clear();

        let syndrome2 = array![0, 1, 1, 0];
        let result2 = decoder.decode(&syndrome2.view()).unwrap();

        // Should get different results
        assert_ne!(result1.matched_edges, result2.matched_edges);
    }

    #[test]
    fn test_max_tree_size() {
        // Test union-find decoder (max_tree_size = 0)
        let config_uf = FusionBlossomConfig {
            num_nodes: Some(6),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: Some(0), // Pure union-find
        };

        // Test MWPM decoder (max_tree_size = None)
        let config_mwpm = FusionBlossomConfig {
            num_nodes: Some(6),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None, // Pure MWPM
        };

        let mut decoder_uf = FusionBlossomDecoder::new(config_uf).unwrap();
        let mut decoder_mwpm = FusionBlossomDecoder::new(config_mwpm).unwrap();

        // Create same graph for both
        for decoder in [&mut decoder_uf, &mut decoder_mwpm] {
            decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
            decoder.add_edge(1, 2, &[0], Some(1.0)).unwrap();
            decoder.add_edge(2, 3, &[0], Some(1.0)).unwrap();
            decoder.add_edge(3, 4, &[0], Some(1.0)).unwrap();
            decoder.add_edge(4, 5, &[0], Some(1.0)).unwrap();
            decoder.add_edge(0, 5, &[0], Some(5.0)).unwrap(); // Expensive shortcut
        }

        let syndrome = array![1, 0, 0, 0, 0, 1];

        let result_uf = decoder_uf.decode(&syndrome.view()).unwrap();
        let result_mwpm = decoder_mwpm.decode(&syndrome.view()).unwrap();

        // Both should find valid matchings, but may differ
        assert!(!result_uf.matched_edges.is_empty());
        assert!(!result_mwpm.matched_edges.is_empty());
    }

    #[test]
    fn test_standard_codes() {
        // Test code capacity planar code
        let code = StandardCode::CodeCapacityPlanar {
            d: 5,
            p: 0.01,
            max_half_weight: 1000,
        };

        let config = FusionBlossomConfig {
            num_nodes: None,
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::from_standard_code(code, config).unwrap();

        // Should have correct number of nodes
        assert!(decoder.graph_summary().contains("nodes"));

        // Extract actual number of nodes from the decoder
        let summary = decoder.graph_summary();
        let num_nodes = summary
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Should parse number of nodes");

        // Test decoding with a simple syndrome
        let mut syndrome = vec![0u8; num_nodes];
        if num_nodes > 10 {
            syndrome[0] = 1;
            syndrome[10] = 1;
        } else if num_nodes > 1 {
            syndrome[0] = 1;
            syndrome[1] = 1;
        }

        let syndrome_array = ndarray::Array1::from_vec(syndrome);
        let result = decoder.decode(&syndrome_array.view());

        // Should decode successfully
        assert!(result.is_ok(), "Decoding failed: {:?}", result.err());
    }

    #[test]
    fn test_phenomenological_code() {
        let code = StandardCode::PhenomenologicalPlanar {
            d: 3,
            p: 0.01,
            p_measurement: 0.02,
            max_half_weight: 1000,
        };

        let config = FusionBlossomConfig::default();

        let decoder = FusionBlossomDecoder::from_standard_code(code, config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_rotated_codes() {
        // Test rotated surface code
        let code = StandardCode::CodeCapacityRotated {
            d: 5,
            p: 0.01,
            max_half_weight: 1000,
        };

        let config = FusionBlossomConfig::default();

        let decoder = FusionBlossomDecoder::from_standard_code(code, config);
        assert!(decoder.is_ok());
    }
}
