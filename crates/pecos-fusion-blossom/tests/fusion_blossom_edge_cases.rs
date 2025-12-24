//! Edge case tests for Fusion Blossom decoder

mod tests {
    use ndarray::array;
    use pecos_fusion_blossom::{
        DecodingOptions, FusionBlossomConfig, FusionBlossomDecoder, SolverType, SyndromeData,
    };

    #[test]
    fn test_empty_graph() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();
        // Don't add any edges

        let syndrome = array![0, 0, 0, 0];
        let result = decoder.decode(&syndrome.view());

        assert!(result.is_ok());
        let decoding = result.unwrap();
        assert_eq!(decoding.observable, vec![0]);
        assert!(
            decoding.weight.abs() < f64::EPSILON,
            "Weight should be zero but was {}",
            decoding.weight
        );
        assert!(decoding.matched_edges.is_empty());
    }

    #[test]
    fn test_single_node_graph() {
        let config = FusionBlossomConfig {
            num_nodes: Some(1),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        let syndrome = array![0];
        let result = decoder.decode(&syndrome.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_virtual_vertices() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Add edges but make all vertices virtual (boundaries)
        decoder.add_boundary_edge(0, &[], Some(1.0)).unwrap();
        decoder.add_boundary_edge(1, &[], Some(1.0)).unwrap();
        decoder.add_boundary_edge(2, &[], Some(1.0)).unwrap();
        decoder.add_boundary_edge(3, &[], Some(1.0)).unwrap();

        let syndrome = array![1, 1, 0, 0];
        let result = decoder.decode(&syndrome.view());

        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_dynamic_weights() {
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

        // Set dynamic weight on existing edges
        let syndrome_data = SyndromeData {
            defects: vec![0, 3],
            erasures: None,
            dynamic_weights: Some(vec![(1, 10)]), // Make middle edge very cheap
        };

        let result = decoder.decode_advanced(syndrome_data);
        assert!(result.is_ok());
        let decoding = result.unwrap();
        // Should use all three edges due to dynamic weight
        assert_eq!(decoding.matched_edges.len(), 3);
    }

    #[test]
    fn test_empty_erasures() {
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

        let syndrome_data = SyndromeData {
            defects: vec![0, 3],
            erasures: Some(vec![]), // Empty erasures
            dynamic_weights: None,
        };

        let result = decoder.decode_advanced(syndrome_data);
        assert!(result.is_ok());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // num_nodes (1000) fits exactly in f64
    fn test_large_graph_stress() {
        let num_nodes = 1000;
        let config = FusionBlossomConfig {
            num_nodes: Some(num_nodes),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Create a chain graph
        for i in 0..(num_nodes - 1) {
            decoder.add_edge(i, i + 1, &[0], Some(1.0)).unwrap();
        }

        // Create syndrome with defects at ends
        let mut syndrome = vec![0u8; num_nodes];
        syndrome[0] = 1;
        syndrome[num_nodes - 1] = 1;

        let syndrome_array = ndarray::Array1::from_vec(syndrome);
        let result = decoder.decode(&syndrome_array.view());

        assert!(result.is_ok());
        let decoding = result.unwrap();
        assert!(
            (decoding.weight - (num_nodes - 1) as f64).abs() < f64::EPSILON,
            "Weight should be {} but was {}",
            (num_nodes - 1) as f64,
            decoding.weight
        );
    }

    #[test]
    fn test_perfect_matching_request() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
        decoder.add_edge(2, 3, &[0], Some(1.0)).unwrap();
        decoder.add_boundary_edge(1, &[], Some(2.0)).unwrap();
        decoder.add_boundary_edge(2, &[], Some(2.0)).unwrap();

        let syndrome_data = SyndromeData::from_defects(vec![0, 1, 2, 3]);
        let options = DecodingOptions {
            include_perfect_matching: true,
        };

        let result = decoder.decode_with_options(syndrome_data, options).unwrap();

        // Currently perfect matching details are not available for Serial solver
        assert!(result.perfect_matching.is_none());

        // But we still get the matched edges
        assert!(!result.matched_edges.is_empty());
    }

    #[test]
    fn test_different_solvers() {
        for solver_type in [SolverType::Legacy, SolverType::Serial] {
            let config = FusionBlossomConfig {
                num_nodes: Some(4),
                num_observables: 1,
                solver_type,
                max_tree_size: None,
            };

            let mut decoder = FusionBlossomDecoder::new(config).unwrap();

            // Create a simple chain
            decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
            decoder.add_edge(1, 2, &[0], Some(1.0)).unwrap();
            decoder.add_edge(2, 3, &[0], Some(1.0)).unwrap();

            let syndrome = array![1, 0, 0, 1];
            let result = decoder.decode(&syndrome.view());

            assert!(result.is_ok(), "Solver {solver_type:?} failed");
            let decoding = result.unwrap();
            assert!(
                (decoding.weight - 3.0).abs() < f64::EPSILON,
                "Weight should be 3.0 but was {}",
                decoding.weight
            ); // Should use all three edges
        }
    }

    #[test]
    fn test_zero_weight_edges() {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Add zero-weight edges
        decoder.add_edge(0, 1, &[0], Some(0.0)).unwrap();
        decoder.add_edge(1, 2, &[0], Some(0.0)).unwrap();
        decoder.add_edge(2, 3, &[0], Some(0.0)).unwrap();

        let syndrome = array![1, 0, 0, 1];
        let result = decoder.decode(&syndrome.view());

        assert!(result.is_ok());
        let decoding = result.unwrap();
        assert!(
            decoding.weight.abs() < f64::EPSILON,
            "Weight should be zero but was {}",
            decoding.weight
        );
    }

    #[test]
    fn test_disconnected_components() {
        let config = FusionBlossomConfig {
            num_nodes: Some(6),
            num_observables: 2,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Create two disconnected components
        decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
        decoder.add_edge(1, 2, &[0], Some(1.0)).unwrap();

        decoder.add_edge(3, 4, &[1], Some(1.0)).unwrap();
        decoder.add_edge(4, 5, &[1], Some(1.0)).unwrap();

        // Add boundary edges to connect components
        decoder.add_boundary_edge(0, &[], Some(10.0)).unwrap();
        decoder.add_boundary_edge(3, &[], Some(10.0)).unwrap();

        let syndrome = array![1, 0, 0, 1, 0, 0];
        let result = decoder.decode(&syndrome.view());

        assert!(result.is_ok());
    }

    #[test]
    fn test_very_large_weights() {
        let config = FusionBlossomConfig {
            num_nodes: Some(3),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Add edges with very large weights
        decoder.add_edge(0, 1, &[0], Some(1e6)).unwrap();
        decoder.add_edge(1, 2, &[0], Some(1e6)).unwrap();
        decoder.add_boundary_edge(0, &[], Some(1.0)).unwrap();
        decoder.add_boundary_edge(2, &[], Some(1.0)).unwrap();

        let syndrome = array![1, 0, 1];
        let result = decoder.decode(&syndrome.view());

        assert!(result.is_ok());
        let decoding = result.unwrap();
        // Should use boundary edges due to lower weight
        assert!(decoding.weight < 1000.0);
    }
}
