// Complete C++ bridge header for PyMatching
#pragma once

#include "rust/cxx.h"
#include <memory>
#include <vector>
#include <cstdint>
#include <limits>

// Forward declarations for Rust types
enum class MergeStrategy : uint8_t;
struct EdgeData;
struct MatchedPair;
struct ExtendedMatchingResult;
struct BatchDecodingResult;

// Main PyMatching graph wrapper
class PyMatchingGraph {
public:
    // Constructors
    PyMatchingGraph(size_t num_nodes);
    PyMatchingGraph(size_t num_nodes, size_t num_observables);
    static std::unique_ptr<PyMatchingGraph> from_dem(
        const std::string& dem_string, bool enable_correlations = false);
    ~PyMatchingGraph();

    // Edge management
    void add_edge(
        size_t node1,
        size_t node2,
        const rust::Slice<const size_t> observables,
        double weight,
        double error_probability,
        MergeStrategy merge_strategy);

    void add_boundary_edge(
        size_t node,
        const rust::Slice<const size_t> observables,
        double weight,
        double error_probability,
        MergeStrategy merge_strategy);

    // Graph queries
    size_t get_num_nodes() const;
    size_t get_num_detectors() const;
    size_t get_num_edges() const;
    size_t get_num_observables() const;
    void set_min_num_observables(size_t num_observables);

    bool has_edge(size_t node1, size_t node2) const;
    bool has_boundary_edge(size_t node) const;

    EdgeData get_edge_data(size_t node1, size_t node2) const;
    EdgeData get_boundary_edge_data(size_t node) const;
    rust::Vec<EdgeData> get_all_edges() const;

    // Boundary management
    rust::Vec<size_t> get_boundary() const;
    void set_boundary(const rust::Slice<const size_t> boundary);
    bool is_boundary_node(size_t node) const;

    // Decoding methods
    ExtendedMatchingResult decode_detection_events_64(
        const rust::Slice<const uint8_t> detection_events);

    ExtendedMatchingResult decode_detection_events_extended(
        const rust::Slice<const uint8_t> detection_events);

    rust::Vec<MatchedPair> decode_to_matched_pairs(
        const rust::Slice<const uint8_t> detection_events);

    rust::Vec<MatchedPair> decode_to_edges(
        const rust::Slice<const uint8_t> detection_events);

    BatchDecodingResult decode_batch(
        const rust::Slice<const uint8_t> shots,
        size_t num_shots,
        size_t num_detectors,
        bool bit_packed_shots,
        bool bit_packed_predictions);

    // Path finding
    rust::Vec<size_t> get_shortest_path(size_t source, size_t target);

    // Noise simulation
    BatchDecodingResult add_noise(
        size_t num_samples,
        uint64_t rng_seed) const;

    // Weight information
    double get_edge_weight_normalising_constant(size_t num_distinct_weights) const;
    bool all_edges_have_error_probabilities() const;

    // Validation
    void validate_detector_indices(const rust::Slice<const uint8_t> detection_events) const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Free functions for FFI
std::unique_ptr<PyMatchingGraph> create_pymatching_graph(size_t num_nodes);
std::unique_ptr<PyMatchingGraph> create_pymatching_graph_with_observables(
    size_t num_nodes, size_t num_observables);
std::unique_ptr<PyMatchingGraph> create_pymatching_graph_from_dem(
    const rust::Str dem_string);
std::unique_ptr<PyMatchingGraph> create_pymatching_graph_from_dem_with_correlations(
    const rust::Str dem_string, bool enable_correlations);

void add_edge(
    PyMatchingGraph& graph,
    size_t node1,
    size_t node2,
    const rust::Slice<const size_t> observables,
    double weight,
    double error_probability,
    MergeStrategy merge_strategy);

void add_boundary_edge(
    PyMatchingGraph& graph,
    size_t node,
    const rust::Slice<const size_t> observables,
    double weight,
    double error_probability,
    MergeStrategy merge_strategy);

size_t pymatching_get_num_nodes(const PyMatchingGraph& graph);
size_t pymatching_get_num_detectors(const PyMatchingGraph& graph);
size_t pymatching_get_num_edges(const PyMatchingGraph& graph);
size_t pymatching_get_num_observables(const PyMatchingGraph& graph);
void pymatching_set_min_num_observables(PyMatchingGraph& graph, size_t num_observables);

bool has_edge(const PyMatchingGraph& graph, size_t node1, size_t node2);
bool has_boundary_edge(const PyMatchingGraph& graph, size_t node);

EdgeData pymatching_get_edge_data(const PyMatchingGraph& graph, size_t node1, size_t node2);
EdgeData pymatching_get_boundary_edge_data(const PyMatchingGraph& graph, size_t node);
rust::Vec<EdgeData> pymatching_get_all_edges(const PyMatchingGraph& graph);

rust::Vec<size_t> pymatching_get_boundary(const PyMatchingGraph& graph);
void pymatching_set_boundary(PyMatchingGraph& graph, const rust::Slice<const size_t> boundary);
bool pymatching_is_boundary_node(const PyMatchingGraph& graph, size_t node);

ExtendedMatchingResult decode_detection_events_64(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events);

ExtendedMatchingResult decode_detection_events_extended(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events);

rust::Vec<MatchedPair> decode_to_matched_pairs(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events);

rust::Vec<MatchedPair> decode_to_edges(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events);

BatchDecodingResult decode_batch(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> shots,
    size_t num_shots,
    size_t num_detectors,
    bool bit_packed_shots,
    bool bit_packed_predictions);

rust::Vec<size_t> get_shortest_path(
    PyMatchingGraph& graph,
    size_t source,
    size_t target);

BatchDecodingResult add_noise(
    const PyMatchingGraph& graph,
    size_t num_samples,
    uint64_t rng_seed);

double get_edge_weight_normalising_constant(
    const PyMatchingGraph& graph,
    size_t num_distinct_weights);

bool all_edges_have_error_probabilities(const PyMatchingGraph& graph);

void validate_detector_indices(
    const PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events);

// Random Number Generation
void pymatching_set_seed(uint32_t seed);
void pymatching_randomize();
double pymatching_rand_float(double from, double to);
