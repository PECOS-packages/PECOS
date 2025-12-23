//! Complete FFI bridge to `PyMatching` C++ library
//! Exposes all major `PyMatching` functionality

#[cxx::bridge]
pub mod ffi {
    // Enums
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[repr(u8)]
    pub enum MergeStrategy {
        Disallow = 0,
        Independent = 1,
        SmallestWeight = 2,
        KeepOriginal = 3,
        Replace = 4,
    }

    // Edge data structure
    #[derive(Debug, Clone)]
    pub struct EdgeData {
        pub node1: usize,
        pub node2: usize, // SIZE_MAX for boundary edges
        pub observables: Vec<usize>,
        pub weight: f64,
        pub error_probability: f64,
    }

    // Matched pair structure
    #[derive(Debug, Clone)]
    pub struct MatchedPair {
        pub detector1: i64,
        pub detector2: i64, // -1 for boundary
    }

    // Decoding result for >64 observables
    #[derive(Debug)]
    pub struct ExtendedMatchingResult {
        pub observables: Vec<u8>,
        pub weight: f64,
    }

    // Batch decoding result
    #[derive(Debug)]
    pub struct BatchDecodingResult {
        pub predictions: Vec<u8>, // Bit-packed predictions
        pub weights: Vec<f64>,    // Weight for each shot
    }

    unsafe extern "C++" {
        include!("pymatching_bridge.h");

        // Main PyMatching graph type
        type PyMatchingGraph;

        // ===== Construction =====
        #[must_use]
        fn create_pymatching_graph(num_nodes: usize) -> UniquePtr<PyMatchingGraph>;
        #[must_use]
        fn create_pymatching_graph_with_observables(
            num_nodes: usize,
            num_observables: usize,
        ) -> UniquePtr<PyMatchingGraph>;
        fn create_pymatching_graph_from_dem(dem_string: &str)
        -> Result<UniquePtr<PyMatchingGraph>>;

        // ===== Edge Management =====
        fn add_edge(
            graph: Pin<&mut PyMatchingGraph>,
            node1: usize,
            node2: usize,
            observables: &[usize],
            weight: f64,
            error_probability: f64,
            merge_strategy: MergeStrategy,
        ) -> Result<()>;

        fn add_boundary_edge(
            graph: Pin<&mut PyMatchingGraph>,
            node: usize,
            observables: &[usize],
            weight: f64,
            error_probability: f64,
            merge_strategy: MergeStrategy,
        ) -> Result<()>;

        // ===== Graph Queries =====
        fn pymatching_get_num_nodes(graph: &PyMatchingGraph) -> usize;
        fn pymatching_get_num_detectors(graph: &PyMatchingGraph) -> usize;
        fn pymatching_get_num_edges(graph: &PyMatchingGraph) -> usize;
        fn pymatching_get_num_observables(graph: &PyMatchingGraph) -> usize;
        fn pymatching_set_min_num_observables(
            graph: Pin<&mut PyMatchingGraph>,
            num_observables: usize,
        );

        fn has_edge(graph: &PyMatchingGraph, node1: usize, node2: usize) -> bool;
        fn has_boundary_edge(graph: &PyMatchingGraph, node: usize) -> bool;

        fn pymatching_get_edge_data(
            graph: &PyMatchingGraph,
            node1: usize,
            node2: usize,
        ) -> Result<EdgeData>;

        fn pymatching_get_boundary_edge_data(
            graph: &PyMatchingGraph,
            node: usize,
        ) -> Result<EdgeData>;

        fn pymatching_get_all_edges(graph: &PyMatchingGraph) -> Vec<EdgeData>;

        // ===== Boundary Management =====
        fn pymatching_get_boundary(graph: &PyMatchingGraph) -> Vec<usize>;
        fn pymatching_set_boundary(graph: Pin<&mut PyMatchingGraph>, boundary: &[usize]);
        fn pymatching_is_boundary_node(graph: &PyMatchingGraph, node: usize) -> bool;

        // ===== Decoding Methods =====

        // For ≤64 observables (returns obs mask and weight)
        fn decode_detection_events_64(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<ExtendedMatchingResult>; // Contains obs_mask as first observable byte and weight

        // For any number of observables
        fn decode_detection_events_extended(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<ExtendedMatchingResult>;

        // Decode to matched detection event pairs
        fn decode_to_matched_pairs(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<Vec<MatchedPair>>;

        // Decode to edges in the matching
        fn decode_to_edges(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<Vec<MatchedPair>>;

        // Batch decoding
        fn decode_batch(
            graph: Pin<&mut PyMatchingGraph>,
            shots: &[u8], // Flat array of shots
            num_shots: usize,
            num_detectors: usize,
            bit_packed_shots: bool,
            bit_packed_predictions: bool,
        ) -> Result<BatchDecodingResult>;

        // ===== Path Finding =====
        fn get_shortest_path(
            graph: Pin<&mut PyMatchingGraph>,
            source: usize,
            target: usize,
        ) -> Result<Vec<usize>>;

        // ===== Noise Simulation =====
        fn add_noise(
            graph: &PyMatchingGraph,
            num_samples: usize,
            rng_seed: u64,
        ) -> Result<BatchDecodingResult>; // Uses predictions for errors and weights for syndromes

        // ===== Weight Information =====
        fn get_edge_weight_normalising_constant(
            graph: &PyMatchingGraph,
            num_distinct_weights: usize,
        ) -> f64;

        fn all_edges_have_error_probabilities(graph: &PyMatchingGraph) -> bool;

        // ===== Validation =====
        fn validate_detector_indices(
            graph: &PyMatchingGraph,
            detection_events: &[u8],
        ) -> Result<()>;

        // ===== Random Number Generation =====
        fn pymatching_set_seed(seed: u32) -> Result<()>;
        fn pymatching_randomize() -> Result<()>;
        fn pymatching_rand_float(from: f64, to: f64) -> Result<f64>;
    }
}
