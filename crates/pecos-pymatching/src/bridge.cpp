//! Complete C++ bridge implementation for PyMatching
#include "rust/cxx.h"
#include "pecos-pymatching/src/bridge.rs.h"
#include <memory>
#include <vector>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <queue>
#include <iostream>
#include <mutex>

// PyMatching includes
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/io.h"
#include "pymatching/sparse_blossom/search/search_graph.h"
#include "pymatching/rand/rand_gen.h"

// Stim includes
#include "stim.h"

// Global mutex to protect PyMatching's global RNG state
static std::mutex g_pymatching_rng_mutex;

// Implementation class using PIMPL pattern
class PyMatchingGraph::Impl {
public:
    std::unique_ptr<pm::UserGraph> user_graph_;
    std::unique_ptr<pm::Mwpm> mwpm_;
    pm::SearchFlooder* search_flooder_ = nullptr;
    double normalising_constant_ = 1.0;

    // Constructor
    Impl(size_t num_nodes, size_t num_observables) {
        user_graph_ = std::make_unique<pm::UserGraph>(num_nodes, num_observables);
    }

    // Initialize MWPM decoder when needed
    void ensure_mwpm(bool include_search_graph = false) {
        if (!mwpm_ || (include_search_graph && !search_flooder_)) {
            normalising_constant_ = user_graph_->get_edge_weight_normalising_constant(pm::NUM_DISTINCT_WEIGHTS);
            if (normalising_constant_ == 0) {
                normalising_constant_ = 1.0;
            }

            // Create MWPM instance using UserGraph's to_mwpm method
            auto flooder = user_graph_->to_mwpm(pm::NUM_DISTINCT_WEIGHTS, include_search_graph);
            mwpm_ = std::make_unique<pm::Mwpm>(std::move(flooder));

            // Search flooder is included when requested
            search_flooder_ = include_search_graph ? &mwpm_->search_flooder : nullptr;
        }
    }

    // Reset decoder state after each use
    void reset_mwpm() {
        if (mwpm_) {
            mwpm_->reset();
        }
    }
};

// ===== PyMatchingGraph Implementation =====

PyMatchingGraph::PyMatchingGraph(size_t num_nodes)
    : pimpl_(std::make_unique<Impl>(num_nodes, 64)) {}

PyMatchingGraph::PyMatchingGraph(size_t num_nodes, size_t num_observables)
    : pimpl_(std::make_unique<Impl>(num_nodes, num_observables)) {}

PyMatchingGraph::~PyMatchingGraph() = default;

std::unique_ptr<PyMatchingGraph> PyMatchingGraph::from_dem(const std::string& dem_string) {
    try {
        auto dem = stim::DetectorErrorModel(dem_string.c_str());

        // Create user graph from DEM
        auto user_graph = pm::detector_error_model_to_user_graph(dem);

        // Create PyMatchingGraph and move the user graph
        auto graph = std::make_unique<PyMatchingGraph>(
            user_graph.get_num_nodes(),
            user_graph.get_num_observables()
        );

        // Replace the default user graph with the one from DEM
        graph->pimpl_->user_graph_ = std::make_unique<pm::UserGraph>(std::move(user_graph));

        return graph;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse DEM: ") + e.what());
    }
}

// ===== Edge Management =====

void PyMatchingGraph::add_edge(
    size_t node1,
    size_t node2,
    const rust::Slice<const size_t> observables,
    double weight,
    double error_probability,
    MergeStrategy merge_strategy) {

    // Use data()+size() instead of begin()/end() iterators to avoid
    // Xcode 15.4 libc++ pointer_traits incompatibility with cxx iterators in C++20
    std::vector<size_t> obs_vec(observables.data(), observables.data() + observables.size());

    // Convert merge strategy enum
    pm::MERGE_STRATEGY pm_strategy;
    switch (merge_strategy) {
        case MergeStrategy::Disallow:
            pm_strategy = pm::DISALLOW;
            break;
        case MergeStrategy::Independent:
            pm_strategy = pm::INDEPENDENT;
            break;
        case MergeStrategy::SmallestWeight:
            pm_strategy = pm::SMALLEST_WEIGHT;
            break;
        case MergeStrategy::KeepOriginal:
            pm_strategy = pm::KEEP_ORIGINAL;
            break;
        case MergeStrategy::Replace:
            pm_strategy = pm::REPLACE;
            break;
    }

    try {
        if (std::isfinite(error_probability) && error_probability > 0 && error_probability < 1) {
            pimpl_->user_graph_->add_or_merge_edge(node1, node2, obs_vec, NAN, error_probability, pm_strategy);
        } else {
            pimpl_->user_graph_->add_or_merge_edge(node1, node2, obs_vec, weight, NAN, pm_strategy);
        }
        pimpl_->mwpm_.reset();  // Invalidate cached MWPM
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to add edge: ") + e.what());
    }
}

void PyMatchingGraph::add_boundary_edge(
    size_t node,
    const rust::Slice<const size_t> observables,
    double weight,
    double error_probability,
    MergeStrategy merge_strategy) {

    std::vector<size_t> obs_vec(observables.data(), observables.data() + observables.size());

    // Convert merge strategy
    pm::MERGE_STRATEGY pm_strategy;
    switch (merge_strategy) {
        case MergeStrategy::Disallow:
            pm_strategy = pm::DISALLOW;
            break;
        case MergeStrategy::Independent:
            pm_strategy = pm::INDEPENDENT;
            break;
        case MergeStrategy::SmallestWeight:
            pm_strategy = pm::SMALLEST_WEIGHT;
            break;
        case MergeStrategy::KeepOriginal:
            pm_strategy = pm::KEEP_ORIGINAL;
            break;
        case MergeStrategy::Replace:
            pm_strategy = pm::REPLACE;
            break;
    }

    try {
        if (std::isfinite(error_probability) && error_probability > 0 && error_probability < 1) {
            pimpl_->user_graph_->add_or_merge_boundary_edge(node, obs_vec, NAN, error_probability, pm_strategy);
        } else {
            pimpl_->user_graph_->add_or_merge_boundary_edge(node, obs_vec, weight, NAN, pm_strategy);
        }
        pimpl_->mwpm_.reset();  // Invalidate cached MWPM
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to add boundary edge: ") + e.what());
    }
}

// ===== Graph Queries =====

size_t PyMatchingGraph::get_num_nodes() const {
    return pimpl_->user_graph_->get_num_nodes();
}

size_t PyMatchingGraph::get_num_detectors() const {
    return pimpl_->user_graph_->get_num_detectors();
}

size_t PyMatchingGraph::get_num_edges() const {
    return pimpl_->user_graph_->get_num_edges();
}

size_t PyMatchingGraph::get_num_observables() const {
    return pimpl_->user_graph_->get_num_observables();
}

void PyMatchingGraph::set_min_num_observables(size_t num_observables) {
    pimpl_->user_graph_->set_min_num_observables(num_observables);
    pimpl_->mwpm_.reset();  // Invalidate cached MWPM
}

bool PyMatchingGraph::has_edge(size_t node1, size_t node2) const {
    return pimpl_->user_graph_->has_edge(node1, node2);
}

bool PyMatchingGraph::has_boundary_edge(size_t node) const {
    return pimpl_->user_graph_->has_boundary_edge(node);
}

EdgeData PyMatchingGraph::get_edge_data(size_t node1, size_t node2) const {
    // Find the edge in the list
    for (const auto& edge : pimpl_->user_graph_->edges) {
        if ((edge.node1 == node1 && edge.node2 == node2) ||
            (edge.node1 == node2 && edge.node2 == node1)) {
            EdgeData data;
            data.node1 = node1;
            data.node2 = node2;
            data.observables = rust::Vec<size_t>();
            for (auto obs : edge.observable_indices) {
                data.observables.push_back(obs);
            }
            data.weight = edge.weight;
            data.error_probability = edge.error_probability;
            return data;
        }
    }

    throw std::runtime_error("Edge not found");
}

EdgeData PyMatchingGraph::get_boundary_edge_data(size_t node) const {
    // Check if this is a boundary node
    if (!pimpl_->user_graph_->has_boundary_edge(node)) {
        throw std::runtime_error("Boundary edge not found");
    }

    // Find boundary edge (edge with node2 == SIZE_MAX)
    for (const auto& edge : pimpl_->user_graph_->edges) {
        if (edge.node1 == node && edge.node2 == SIZE_MAX) {
            EdgeData data;
            data.node1 = node;
            data.node2 = std::numeric_limits<size_t>::max();  // Sentinel for boundary
            data.observables = rust::Vec<size_t>();
            for (auto obs : edge.observable_indices) {
                data.observables.push_back(obs);
            }
            data.weight = edge.weight;
            data.error_probability = edge.error_probability;
            return data;
        }
    }

    throw std::runtime_error("Boundary edge not found");
}

rust::Vec<EdgeData> PyMatchingGraph::get_all_edges() const {
    rust::Vec<EdgeData> all_edges;

    // Add all edges (regular and boundary)
    for (const auto& edge : pimpl_->user_graph_->edges) {
        EdgeData data;
        data.node1 = edge.node1;
        data.node2 = edge.node2;
        data.observables = rust::Vec<size_t>();
        for (auto obs : edge.observable_indices) {
            data.observables.push_back(obs);
        }
        data.weight = edge.weight;
        data.error_probability = edge.error_probability;
        all_edges.push_back(data);
    }

    return all_edges;
}

// ===== Boundary Management =====

rust::Vec<size_t> PyMatchingGraph::get_boundary() const {
    rust::Vec<size_t> boundary;
    auto boundary_set = pimpl_->user_graph_->get_boundary();
    for (auto node : boundary_set) {
        boundary.push_back(node);
    }
    return boundary;
}

void PyMatchingGraph::set_boundary(const rust::Slice<const size_t> boundary) {
    std::set<size_t> boundary_set(boundary.data(), boundary.data() + boundary.size());
    pimpl_->user_graph_->set_boundary(boundary_set);
    pimpl_->mwpm_.reset();  // Invalidate cached MWPM
}

bool PyMatchingGraph::is_boundary_node(size_t node) const {
    return pimpl_->user_graph_->is_boundary_node(node);
}

// ===== Decoding Methods =====

ExtendedMatchingResult PyMatchingGraph::decode_detection_events_64(
    const rust::Slice<const uint8_t> detection_events) {

    pimpl_->ensure_mwpm();

    // Convert detection events to vector of indices
    std::vector<uint64_t> detections;
    for (size_t i = 0; i < detection_events.size(); i++) {
        if (detection_events[i]) {
            detections.push_back(i);
        }
    }

    try {
        auto result = pm::decode_detection_events_for_up_to_64_observables(*pimpl_->mwpm_, detections);
        pimpl_->reset_mwpm();

        ExtendedMatchingResult ext_result;
        ext_result.observables = rust::Vec<uint8_t>();

        // Pack obs_mask into bytes
        for (size_t i = 0; i < 8; i++) {
            ext_result.observables.push_back((result.obs_mask >> (i * 8)) & 0xFF);
        }

        ext_result.weight = static_cast<double>(result.weight) / pimpl_->normalising_constant_;
        return ext_result;
    } catch (const std::exception& e) {
        pimpl_->reset_mwpm();
        throw std::runtime_error(std::string("Decoding failed: ") + e.what());
    }
}

ExtendedMatchingResult PyMatchingGraph::decode_detection_events_extended(
    const rust::Slice<const uint8_t> detection_events) {

    pimpl_->ensure_mwpm();

    // Convert detection events
    std::vector<uint64_t> detections;
    for (size_t i = 0; i < detection_events.size(); i++) {
        if (detection_events[i]) {
            detections.push_back(i);
        }
    }

    try {
        size_t num_obs = pimpl_->user_graph_->get_num_observables();
        std::vector<uint8_t> obs_vec(num_obs, 0);
        pm::total_weight_int weight = 0;

        pm::decode_detection_events(*pimpl_->mwpm_, detections, obs_vec.data(), weight);
        pimpl_->reset_mwpm();

        ExtendedMatchingResult result;
        result.observables = rust::Vec<uint8_t>();
        for (auto val : obs_vec) {
            result.observables.push_back(val);
        }
        result.weight = static_cast<double>(weight) / pimpl_->normalising_constant_;

        return result;
    } catch (const std::exception& e) {
        pimpl_->reset_mwpm();
        throw std::runtime_error(std::string("Decoding failed: ") + e.what());
    }
}

rust::Vec<MatchedPair> PyMatchingGraph::decode_to_matched_pairs(
    const rust::Slice<const uint8_t> detection_events) {

    pimpl_->ensure_mwpm();

    // Convert detection events
    std::vector<uint64_t> detections;
    for (size_t i = 0; i < detection_events.size(); i++) {
        if (detection_events[i]) {
            detections.push_back(i);
        }
    }

    try {
        // Call PyMatching's decode to match edges
        pm::decode_detection_events_to_match_edges(*pimpl_->mwpm_, detections);

        // Extract the matched pairs from mwpm
        rust::Vec<MatchedPair> pairs;
        for (const auto& match_edge : pimpl_->mwpm_->flooder.match_edges) {
            MatchedPair pair;
            pair.detector1 = match_edge.loc_from - &pimpl_->mwpm_->flooder.graph.nodes[0];
            pair.detector2 = match_edge.loc_to ?
                match_edge.loc_to - &pimpl_->mwpm_->flooder.graph.nodes[0] : -1;
            pairs.push_back(pair);
        }

        pimpl_->reset_mwpm();
        return pairs;
    } catch (const std::exception& e) {
        pimpl_->reset_mwpm();
        throw std::runtime_error(std::string("Decode to matched pairs failed: ") + e.what());
    }
}

rust::Vec<MatchedPair> PyMatchingGraph::decode_to_edges(
    const rust::Slice<const uint8_t> detection_events) {

    pimpl_->ensure_mwpm();

    // Convert detection events
    std::vector<uint64_t> detections;
    for (size_t i = 0; i < detection_events.size(); i++) {
        if (detection_events[i]) {
            detections.push_back(i);
        }
    }

    try {
        // Ensure we have search flooder for edge extraction
        pimpl_->ensure_mwpm(true); // true = include search graph

        // Call PyMatching's decode to edges
        std::vector<int64_t> edges;
        pm::decode_detection_events_to_edges(*pimpl_->mwpm_, detections, edges);

        // Convert to MatchedPair format
        rust::Vec<MatchedPair> edge_pairs;
        for (size_t i = 0; i < edges.size() / 2; i++) {
            MatchedPair pair;
            pair.detector1 = edges[2 * i];
            pair.detector2 = edges[2 * i + 1];
            edge_pairs.push_back(pair);
        }

        pimpl_->reset_mwpm();
        return edge_pairs;
    } catch (const std::exception& e) {
        pimpl_->reset_mwpm();
        throw std::runtime_error(std::string("Decode to edges failed: ") + e.what());
    }
}

BatchDecodingResult PyMatchingGraph::decode_batch(
    const rust::Slice<const uint8_t> shots,
    size_t num_shots,
    size_t num_detectors,
    bool bit_packed_shots,
    bool bit_packed_predictions) {

    pimpl_->ensure_mwpm();

    BatchDecodingResult result;
    result.predictions = rust::Vec<uint8_t>();
    result.weights = rust::Vec<double>();

    size_t num_obs = pimpl_->user_graph_->get_num_observables();
    size_t obs_bytes_per_shot = bit_packed_predictions ? ((num_obs + 7) / 8) : num_obs;
    size_t det_bytes_per_shot = bit_packed_shots ? ((num_detectors + 7) / 8) : num_detectors;

    // Pre-allocate result space
    result.predictions.reserve(num_shots * obs_bytes_per_shot);
    result.weights.reserve(num_shots);

    try {
        for (size_t shot = 0; shot < num_shots; shot++) {
            // Extract detection events for this shot
            std::vector<uint64_t> detections;
            size_t shot_offset = shot * det_bytes_per_shot;

            if (bit_packed_shots) {
                // Unpack bit-packed detection events
                for (size_t byte = 0; byte < det_bytes_per_shot; byte++) {
                    if (shot_offset + byte < shots.size()) {
                        uint8_t byte_val = shots[shot_offset + byte];
                        for (size_t bit = 0; bit < 8 && (byte * 8 + bit) < num_detectors; bit++) {
                            if (byte_val & (1 << bit)) {
                                detections.push_back(byte * 8 + bit);
                            }
                        }
                    }
                }
            } else {
                // Direct unpacked format
                for (size_t i = 0; i < num_detectors; i++) {
                    if (shot_offset + i < shots.size() && shots[shot_offset + i]) {
                        detections.push_back(i);
                    }
                }
            }

            // Decode
            if (num_obs <= 64) {
                auto res = pm::decode_detection_events_for_up_to_64_observables(*pimpl_->mwpm_, detections);

                if (bit_packed_predictions) {
                    // Pack obs_mask into bytes
                    for (size_t byte = 0; byte < obs_bytes_per_shot; byte++) {
                        uint8_t val = 0;
                        for (size_t bit = 0; bit < 8 && byte * 8 + bit < num_obs; bit++) {
                            if (res.obs_mask & (1ULL << (byte * 8 + bit))) {
                                val |= (1 << bit);
                            }
                        }
                        result.predictions.push_back(val);
                    }
                } else {
                    // Unpacked format - one byte per observable
                    for (size_t i = 0; i < num_obs; i++) {
                        result.predictions.push_back((res.obs_mask >> i) & 1);
                    }
                }

                result.weights.push_back(static_cast<double>(res.weight) / pimpl_->normalising_constant_);
            } else {
                std::vector<uint8_t> obs_vec(num_obs, 0);
                pm::total_weight_int weight = 0;

                pm::decode_detection_events(*pimpl_->mwpm_, detections, obs_vec.data(), weight);

                if (bit_packed_predictions) {
                    // Pack observables into bytes
                    for (size_t byte = 0; byte < obs_bytes_per_shot; byte++) {
                        uint8_t val = 0;
                        for (size_t bit = 0; bit < 8 && byte * 8 + bit < num_obs; bit++) {
                            if (obs_vec[byte * 8 + bit]) {
                                val |= (1 << bit);
                            }
                        }
                        result.predictions.push_back(val);
                    }
                } else {
                    // Unpacked format - copy directly
                    for (size_t i = 0; i < num_obs; i++) {
                        result.predictions.push_back(obs_vec[i]);
                    }
                }

                result.weights.push_back(static_cast<double>(weight) / pimpl_->normalising_constant_);
            }

            pimpl_->reset_mwpm();
        }

        return result;
    } catch (const std::exception& e) {
        pimpl_->reset_mwpm();
        throw std::runtime_error(std::string("Batch decoding failed: ") + e.what());
    }
}

// ===== Path Finding =====

rust::Vec<size_t> PyMatchingGraph::get_shortest_path(size_t source, size_t target) {
    rust::Vec<size_t> path;

    try {
        // Validate nodes
        size_t num_nodes = pimpl_->user_graph_->get_num_nodes();

        if (source >= num_nodes) {
            throw std::invalid_argument("Source node " + std::to_string(source) + " is out of bounds");
        }
        if (target >= num_nodes) {
            throw std::invalid_argument("Target node " + std::to_string(target) + " is out of bounds");
        }

        // PyMatching's shortest path requires the MWPM with search graph
        // We need to ensure it's initialized before calling
        // Note: This modifies internal state, so this method cannot be const

        // Try to get the shortest path
        // PyMatching may segfault on disconnected graphs, so we wrap in a try-catch
        try {
            std::vector<size_t> result_path;
            pimpl_->user_graph_->get_nodes_on_shortest_path_from_source(source, target, result_path);


            // Convert to rust::Vec
            for (size_t node : result_path) {
                path.push_back(node);
            }
        } catch (...) {
            // PyMatching crashed or threw an exception
            // This typically happens with disconnected graphs
            // Return empty path
        }

        return path;
    } catch (const std::exception& e) {
        // PyMatching throws exceptions for various cases:
        // - Disconnected graphs
        // - Both source and target are boundary nodes
        // - Invalid configurations
        // We'll handle these gracefully by returning an empty path
        return path;
    }
}

// ===== Noise Simulation =====

BatchDecodingResult PyMatchingGraph::add_noise(
    size_t num_samples,
    uint64_t rng_seed) const {

    BatchDecodingResult result;
    result.predictions = rust::Vec<uint8_t>();
    result.weights = rust::Vec<double>();

    try {
        // Calculate sizes
        size_t num_observables = pimpl_->user_graph_->get_num_observables();
        size_t num_detectors = pimpl_->user_graph_->get_num_detectors();

        // Lock mutex for entire noise generation to ensure deterministic results
        std::lock_guard<std::mutex> lock(g_pymatching_rng_mutex);

        // Seed the internal RNG
        pm::set_seed((uint32_t)rng_seed);

        // Generate noise samples
        for (size_t sample = 0; sample < num_samples; sample++) {
            std::vector<uint8_t> error_vec(num_observables, 0);
            std::vector<uint8_t> syndrome_vec(num_detectors, 0);

            // Call PyMatching's add_noise
            pimpl_->user_graph_->add_noise(error_vec.data(), syndrome_vec.data());

            // Copy errors to result (as predictions)
            for (auto val : error_vec) {
                result.predictions.push_back(val);
            }

            // Copy syndrome to weights (reinterpret as double for now)
            // In the actual API, syndromes would be returned separately
            for (auto val : syndrome_vec) {
                result.weights.push_back(static_cast<double>(val));
            }
        }

        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Noise simulation failed: ") + e.what());
    }
}

// ===== Weight Information =====

double PyMatchingGraph::get_edge_weight_normalising_constant(size_t num_distinct_weights) const {
    return pimpl_->user_graph_->get_edge_weight_normalising_constant(num_distinct_weights);
}

bool PyMatchingGraph::all_edges_have_error_probabilities() const {
    return pimpl_->user_graph_->all_edges_have_error_probabilities();
}

// ===== Validation =====

void PyMatchingGraph::validate_detector_indices(const rust::Slice<const uint8_t> detection_events) const {
    size_t num_detectors = pimpl_->user_graph_->get_num_detectors();

    if (detection_events.size() > num_detectors) {
        throw std::runtime_error("Detection events array larger than number of detectors");
    }
}

// ===== Free Functions for FFI =====

std::unique_ptr<PyMatchingGraph> create_pymatching_graph(size_t num_nodes) {
    return std::make_unique<PyMatchingGraph>(num_nodes);
}

std::unique_ptr<PyMatchingGraph> create_pymatching_graph_with_observables(
    size_t num_nodes, size_t num_observables) {
    return std::make_unique<PyMatchingGraph>(num_nodes, num_observables);
}

std::unique_ptr<PyMatchingGraph> create_pymatching_graph_from_dem(const rust::Str dem_string) {
    return PyMatchingGraph::from_dem(std::string(dem_string));
}

void add_edge(
    PyMatchingGraph& graph,
    size_t node1,
    size_t node2,
    const rust::Slice<const size_t> observables,
    double weight,
    double error_probability,
    MergeStrategy merge_strategy) {
    graph.add_edge(node1, node2, observables, weight, error_probability, merge_strategy);
}

void add_boundary_edge(
    PyMatchingGraph& graph,
    size_t node,
    const rust::Slice<const size_t> observables,
    double weight,
    double error_probability,
    MergeStrategy merge_strategy) {
    graph.add_boundary_edge(node, observables, weight, error_probability, merge_strategy);
}

size_t pymatching_get_num_nodes(const PyMatchingGraph& graph) {
    return graph.get_num_nodes();
}

size_t pymatching_get_num_detectors(const PyMatchingGraph& graph) {
    return graph.get_num_detectors();
}

size_t pymatching_get_num_edges(const PyMatchingGraph& graph) {
    return graph.get_num_edges();
}

size_t pymatching_get_num_observables(const PyMatchingGraph& graph) {
    return graph.get_num_observables();
}

void pymatching_set_min_num_observables(PyMatchingGraph& graph, size_t num_observables) {
    graph.set_min_num_observables(num_observables);
}

bool has_edge(const PyMatchingGraph& graph, size_t node1, size_t node2) {
    return graph.has_edge(node1, node2);
}

bool has_boundary_edge(const PyMatchingGraph& graph, size_t node) {
    return graph.has_boundary_edge(node);
}

EdgeData pymatching_get_edge_data(const PyMatchingGraph& graph, size_t node1, size_t node2) {
    return graph.get_edge_data(node1, node2);
}

EdgeData pymatching_get_boundary_edge_data(const PyMatchingGraph& graph, size_t node) {
    return graph.get_boundary_edge_data(node);
}

rust::Vec<EdgeData> pymatching_get_all_edges(const PyMatchingGraph& graph) {
    return graph.get_all_edges();
}

rust::Vec<size_t> pymatching_get_boundary(const PyMatchingGraph& graph) {
    return graph.get_boundary();
}

void pymatching_set_boundary(PyMatchingGraph& graph, const rust::Slice<const size_t> boundary) {
    graph.set_boundary(boundary);
}

bool pymatching_is_boundary_node(const PyMatchingGraph& graph, size_t node) {
    return graph.is_boundary_node(node);
}

ExtendedMatchingResult decode_detection_events_64(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events) {
    return graph.decode_detection_events_64(detection_events);
}

ExtendedMatchingResult decode_detection_events_extended(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events) {
    return graph.decode_detection_events_extended(detection_events);
}

rust::Vec<MatchedPair> decode_to_matched_pairs(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events) {
    return graph.decode_to_matched_pairs(detection_events);
}

rust::Vec<MatchedPair> decode_to_edges(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events) {
    return graph.decode_to_edges(detection_events);
}

BatchDecodingResult decode_batch(
    PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> shots,
    size_t num_shots,
    size_t num_detectors,
    bool bit_packed_shots,
    bool bit_packed_predictions) {
    return graph.decode_batch(shots, num_shots, num_detectors, bit_packed_shots, bit_packed_predictions);
}

rust::Vec<size_t> get_shortest_path(
    PyMatchingGraph& graph,
    size_t source,
    size_t target) {
    return graph.get_shortest_path(source, target);
}

BatchDecodingResult add_noise(
    const PyMatchingGraph& graph,
    size_t num_samples,
    uint64_t rng_seed) {
    return graph.add_noise(num_samples, rng_seed);
}

double get_edge_weight_normalising_constant(
    const PyMatchingGraph& graph,
    size_t num_distinct_weights) {
    return graph.get_edge_weight_normalising_constant(num_distinct_weights);
}

bool all_edges_have_error_probabilities(const PyMatchingGraph& graph) {
    return graph.all_edges_have_error_probabilities();
}

void validate_detector_indices(
    const PyMatchingGraph& graph,
    const rust::Slice<const uint8_t> detection_events) {
    graph.validate_detector_indices(detection_events);
}

// ===== Random Number Generation Functions =====

void pymatching_set_seed(uint32_t seed) {
    // Lock mutex to protect global RNG state
    std::lock_guard<std::mutex> lock(g_pymatching_rng_mutex);
    pm::set_seed(seed);
}

void pymatching_randomize() {
    // Lock mutex to protect global RNG state
    std::lock_guard<std::mutex> lock(g_pymatching_rng_mutex);
    pm::randomize();
}

double pymatching_rand_float(double from, double to) {
    // Lock mutex to protect global RNG state
    std::lock_guard<std::mutex> lock(g_pymatching_rng_mutex);
    return pm::rand_float(from, to);
}
