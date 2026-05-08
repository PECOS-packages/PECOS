//! Complete FFI bridge to `PyMatching` C++ library
//!
//! This module provides the low-level FFI bindings to the `PyMatching` C++ library.
//! Users should prefer the high-level [`PyMatchingDecoder`](crate::PyMatchingDecoder) API.

#[cxx::bridge]
pub(crate) mod ffi {
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

        type PyMatchingGraph;

        // ===== Construction =====

        /// Create a new `PyMatching` graph with the given number of nodes.
        #[must_use]
        fn create_pymatching_graph(num_nodes: usize) -> UniquePtr<PyMatchingGraph>;

        /// Create a new `PyMatching` graph with specified nodes and observables.
        #[must_use]
        fn create_pymatching_graph_with_observables(
            num_nodes: usize,
            num_observables: usize,
        ) -> UniquePtr<PyMatchingGraph>;

        /// Create a `PyMatching` graph from a detector error model string.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the DEM string is malformed.
        fn create_pymatching_graph_from_dem(dem_string: &str)
        -> Result<UniquePtr<PyMatchingGraph>>;

        /// Create a `PyMatching` graph from a DEM string with correlation support.
        ///
        /// When `enable_correlations` is true, the decoder tracks edge correlations
        /// during graph construction and uses them during decoding.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the DEM string is malformed.
        fn create_pymatching_graph_from_dem_with_correlations(
            dem_string: &str,
            enable_correlations: bool,
        ) -> Result<UniquePtr<PyMatchingGraph>>;

        // ===== Edge Management =====

        /// Add an edge between two nodes.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if nodes are invalid or edge conflicts
        /// with merge strategy.
        fn add_edge(
            graph: Pin<&mut PyMatchingGraph>,
            node1: usize,
            node2: usize,
            observables: &[usize],
            weight: f64,
            error_probability: f64,
            merge_strategy: MergeStrategy,
        ) -> Result<()>;

        /// Add a boundary edge connecting a node to the boundary.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if node is invalid or edge conflicts
        /// with merge strategy.
        fn add_boundary_edge(
            graph: Pin<&mut PyMatchingGraph>,
            node: usize,
            observables: &[usize],
            weight: f64,
            error_probability: f64,
            merge_strategy: MergeStrategy,
        ) -> Result<()>;

        // ===== Graph Queries =====

        /// Get the number of nodes in the graph.
        fn pymatching_get_num_nodes(graph: &PyMatchingGraph) -> usize;

        /// Get the number of detectors (non-boundary nodes).
        fn pymatching_get_num_detectors(graph: &PyMatchingGraph) -> usize;

        /// Get the number of edges in the graph.
        fn pymatching_get_num_edges(graph: &PyMatchingGraph) -> usize;

        /// Get the number of observables.
        fn pymatching_get_num_observables(graph: &PyMatchingGraph) -> usize;

        /// Set the minimum number of observables.
        fn pymatching_set_min_num_observables(
            graph: Pin<&mut PyMatchingGraph>,
            num_observables: usize,
        );

        /// Check if an edge exists between two nodes.
        fn has_edge(graph: &PyMatchingGraph, node1: usize, node2: usize) -> bool;

        /// Check if a boundary edge exists for a node.
        fn has_boundary_edge(graph: &PyMatchingGraph, node: usize) -> bool;

        /// Get edge data for an edge between two nodes.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the edge does not exist.
        fn pymatching_get_edge_data(
            graph: &PyMatchingGraph,
            node1: usize,
            node2: usize,
        ) -> Result<EdgeData>;

        /// Get edge data for a boundary edge.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the boundary edge does not exist.
        fn pymatching_get_boundary_edge_data(
            graph: &PyMatchingGraph,
            node: usize,
        ) -> Result<EdgeData>;

        /// Get all edges in the graph.
        fn pymatching_get_all_edges(graph: &PyMatchingGraph) -> Vec<EdgeData>;

        // ===== Boundary Management =====

        /// Get all boundary node indices.
        fn pymatching_get_boundary(graph: &PyMatchingGraph) -> Vec<usize>;

        /// Set the boundary nodes.
        fn pymatching_set_boundary(graph: Pin<&mut PyMatchingGraph>, boundary: &[usize]);

        /// Check if a node is a boundary node.
        fn pymatching_is_boundary_node(graph: &PyMatchingGraph, node: usize) -> bool;

        // ===== Decoding Methods =====

        /// Decode detection events (optimized for <=64 observables).
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if detection events are invalid or decoding fails.
        fn decode_detection_events_64(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<ExtendedMatchingResult>;

        /// Decode detection events (for any number of observables).
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if detection events are invalid or decoding fails.
        fn decode_detection_events_extended(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<ExtendedMatchingResult>;

        /// Decode to matched detection event pairs.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if detection events are invalid or matching fails.
        fn decode_to_matched_pairs(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<Vec<MatchedPair>>;

        /// Decode to edges in the matching.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if detection events are invalid or matching fails.
        fn decode_to_edges(
            graph: Pin<&mut PyMatchingGraph>,
            detection_events: &[u8],
        ) -> Result<Vec<MatchedPair>>;

        /// Batch decode multiple shots.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if shots are malformed or decoding fails.
        fn decode_batch(
            graph: Pin<&mut PyMatchingGraph>,
            shots: &[u8],
            num_shots: usize,
            num_detectors: usize,
            bit_packed_shots: bool,
            bit_packed_predictions: bool,
        ) -> Result<BatchDecodingResult>;

        // ===== Path Finding =====

        /// Find the shortest path between two nodes.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if nodes are invalid or no path exists.
        fn get_shortest_path(
            graph: Pin<&mut PyMatchingGraph>,
            source: usize,
            target: usize,
        ) -> Result<Vec<usize>>;

        // ===== Noise Simulation =====

        /// Generate noise samples based on edge error probabilities.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if error probabilities are not set.
        fn add_noise(
            graph: &PyMatchingGraph,
            num_samples: usize,
            rng_seed: u64,
        ) -> Result<BatchDecodingResult>;

        // ===== Weight Information =====

        /// Get the normalizing constant for edge weights.
        fn get_edge_weight_normalising_constant(
            graph: &PyMatchingGraph,
            num_distinct_weights: usize,
        ) -> f64;

        /// Check if all edges have error probabilities set.
        fn all_edges_have_error_probabilities(graph: &PyMatchingGraph) -> bool;

        // ===== Validation =====

        /// Validate that detector indices in detection events are valid.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if any detector index is out of bounds.
        fn validate_detector_indices(
            graph: &PyMatchingGraph,
            detection_events: &[u8],
        ) -> Result<()>;

        // ===== Random Number Generation =====

        /// Set the RNG seed for reproducibility.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if seeding fails.
        fn pymatching_set_seed(seed: u32) -> Result<()>;

        /// Randomize the RNG state.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if randomization fails.
        fn pymatching_randomize() -> Result<()>;

        /// Generate a random float in the given range.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the range is invalid.
        fn pymatching_rand_float(from: f64, to: f64) -> Result<f64>;
    }
}
