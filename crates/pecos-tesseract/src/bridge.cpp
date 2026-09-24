//! C++ bridge implementation for Tesseract decoder

#include "tesseract_bridge.h"
#include "pecos-tesseract/src/bridge.rs.h"
#include <memory>
#include <stdexcept>
#include <sstream>
#include <numeric>   // Required for std::iota on MSVC

// Include Tesseract headers
#include "tesseract.h"
#include "tesseract_trellis.h"
#include "common.h"
#include "utils.h"

// Include Stim headers
#include "stim/dem/detector_error_model.h"

using namespace tesseract_decoder;

namespace {

stim::DetectorErrorModel parse_dem(const std::string& dem_string) {
    try {
        return stim::DetectorErrorModel(dem_string);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse DEM string: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Failed to parse DEM string: unknown error");
    }
}

}  // namespace

// PIMPL implementation to hide Tesseract details
class TesseractDecoderWrapper::Impl {
private:
    std::unique_ptr<TesseractDecoder> decoder_;
    TesseractConfig config_;

public:
    Impl(const std::string& dem_string, const TesseractConfigRepr& config_repr) {
        // Convert config representation to TesseractConfig
        TesseractConfig config;
        config.dem = parse_dem(dem_string);
        config.det_beam = (config_repr.det_beam == std::numeric_limits<uint16_t>::max()) ?
                          INF_DET_BEAM : static_cast<int>(config_repr.det_beam);
        config.beam_climbing = config_repr.beam_climbing;
        config.no_revisit_dets = config_repr.no_revisit_dets;
        config.verbose = config_repr.verbose;
        config.merge_errors = config_repr.merge_errors;
        config.pqlimit = config_repr.pqlimit;
        config.det_penalty = config_repr.det_penalty;

        // Twenty BFS-based detector orderings from a fixed seed, the shape
        // this wrap has always shipped. Upstream's generator before commit
        // ccb61bc ("Fix detector traversal order semantics") emitted the
        // inverse permutation, so earlier pins searched scrambled orders;
        // true BFS orders cost about 2.5x more per shot on a 936-detector
        // BB144 model (210 ms versus 83 ms, 1000 shots) and fail no more
        // often (0 versus 2 with the fast preset). The decoder resolves the
        // orderings against its flattened, merged DEM at construction.
        config.detector_orders = make_detector_orders(20, DetectorOrder::Method::BFS, 2384753);

        config_ = config;
        decoder_ = std::make_unique<TesseractDecoder>(std::move(config));
    }

    DecodingResultRepr decode_detections(const rust::Slice<const uint64_t> detections) {
        std::vector<uint64_t> det_vec(detections.data(), detections.data() + detections.size());

        decoder_->decode_to_errors(det_vec);

        DecodingResultRepr result;
        result.predicted_errors = rust::Vec<size_t>();
        for (size_t err : decoder_->predicted_errors_buffer) {
            result.predicted_errors.push_back(err);
        }

        result.observables_mask = vector_to_u64_mask(
            decoder_->get_flipped_observables(decoder_->predicted_errors_buffer));
        result.cost = decoder_->cost_from_errors(decoder_->predicted_errors_buffer);
        result.low_confidence = decoder_->low_confidence_flag;

        return result;
    }

    DecodingResultRepr decode_detections_with_order(
        const rust::Slice<const uint64_t> detections,
        size_t det_order
    ) {
        std::vector<uint64_t> det_vec(detections.data(), detections.data() + detections.size());

        decoder_->decode_to_errors(det_vec, det_order, config_.det_beam);

        DecodingResultRepr result;
        result.predicted_errors = rust::Vec<size_t>();
        for (size_t err : decoder_->predicted_errors_buffer) {
            result.predicted_errors.push_back(err);
        }

        result.observables_mask = vector_to_u64_mask(
            decoder_->get_flipped_observables(decoder_->predicted_errors_buffer));
        result.cost = decoder_->cost_from_errors(decoder_->predicted_errors_buffer);
        result.low_confidence = decoder_->low_confidence_flag;

        return result;
    }

    size_t get_num_detectors() const {
        return config_.dem.count_detectors();
    }

    // Predicted errors and every per-error accessor use flattened-DEM
    // indices; the decoder keeps merged, nonzero mechanisms in its own
    // compact index space and exposes the map between them.
    size_t get_num_errors() const {
        return decoder_->dem_error_to_error.size();
    }

    size_t retained_index(size_t dem_error_idx) const {
        size_t retained = decoder_->dem_error_to_error.at(dem_error_idx);
        if (retained == std::numeric_limits<size_t>::max()) {
            throw std::invalid_argument("DEM error index " + std::to_string(dem_error_idx) +
                                        " was merged into another mechanism or removed for zero probability");
        }
        return retained;
    }

    size_t get_num_observables() const {
        return config_.dem.count_observables();
    }

    uint16_t get_det_beam() const {
        return (config_.det_beam == INF_DET_BEAM) ?
               std::numeric_limits<uint16_t>::max() : static_cast<uint16_t>(config_.det_beam);
    }

    bool get_beam_climbing() const {
        return config_.beam_climbing;
    }

    bool get_no_revisit_dets() const {
        return config_.no_revisit_dets;
    }

    bool get_verbose() const {
        return config_.verbose;
    }

    bool get_merge_errors() const {
        return config_.merge_errors;
    }

    size_t get_pqlimit() const {
        return config_.pqlimit;
    }

    double get_det_penalty() const {
        return config_.det_penalty;
    }

    double get_error_probability(size_t dem_error_idx) const {
        return decoder_->errors[retained_index(dem_error_idx)].get_probability();
    }

    double get_error_cost(size_t dem_error_idx) const {
        return decoder_->errors[retained_index(dem_error_idx)].likelihood_cost;
    }

    rust::Vec<int32_t> get_error_detectors(size_t dem_error_idx) const {
        rust::Vec<int32_t> detectors;
        for (int det : decoder_->errors[retained_index(dem_error_idx)].symptom.detectors) {
            detectors.push_back(static_cast<int32_t>(det));
        }
        return detectors;
    }

    uint64_t get_error_observables(size_t dem_error_idx) const {
        return vector_to_u64_mask(decoder_->errors[retained_index(dem_error_idx)].symptom.observables);
    }

    uint64_t mask_from_errors(const rust::Slice<const size_t> error_indices) const {
        std::vector<size_t> indices(error_indices.data(), error_indices.data() + error_indices.size());
        return vector_to_u64_mask(decoder_->get_flipped_observables(indices));
    }

    double cost_from_errors(const rust::Slice<const size_t> error_indices) const {
        std::vector<size_t> indices(error_indices.data(), error_indices.data() + error_indices.size());
        return decoder_->cost_from_errors(indices);
    }
};

// TesseractDecoderWrapper implementation
TesseractDecoderWrapper::TesseractDecoderWrapper(const std::string& dem_string, const TesseractConfigRepr& config_repr)
    : pimpl_(std::make_unique<Impl>(dem_string, config_repr)) {
}

TesseractDecoderWrapper::~TesseractDecoderWrapper() = default;

void TesseractDecoderWrapper::init(const std::string& dem_string, const TesseractConfigRepr& config) {
    pimpl_ = std::make_unique<Impl>(dem_string, config);
}

DecodingResultRepr TesseractDecoderWrapper::decode_detections(const rust::Slice<const uint64_t> detections) {
    return pimpl_->decode_detections(detections);
}

DecodingResultRepr TesseractDecoderWrapper::decode_detections_with_order(
    const rust::Slice<const uint64_t> detections,
    size_t det_order
) {
    return pimpl_->decode_detections_with_order(detections, det_order);
}

size_t TesseractDecoderWrapper::get_num_detectors() const {
    return pimpl_->get_num_detectors();
}

size_t TesseractDecoderWrapper::get_num_errors() const {
    return pimpl_->get_num_errors();
}

size_t TesseractDecoderWrapper::get_num_observables() const {
    return pimpl_->get_num_observables();
}

uint16_t TesseractDecoderWrapper::get_det_beam() const {
    return pimpl_->get_det_beam();
}

bool TesseractDecoderWrapper::get_beam_climbing() const {
    return pimpl_->get_beam_climbing();
}

bool TesseractDecoderWrapper::get_no_revisit_dets() const {
    return pimpl_->get_no_revisit_dets();
}

bool TesseractDecoderWrapper::get_verbose() const {
    return pimpl_->get_verbose();
}

bool TesseractDecoderWrapper::get_merge_errors() const {
    return pimpl_->get_merge_errors();
}

size_t TesseractDecoderWrapper::get_pqlimit() const {
    return pimpl_->get_pqlimit();
}

double TesseractDecoderWrapper::get_det_penalty() const {
    return pimpl_->get_det_penalty();
}

double TesseractDecoderWrapper::get_error_probability(size_t error_idx) const {
    return pimpl_->get_error_probability(error_idx);
}

double TesseractDecoderWrapper::get_error_cost(size_t error_idx) const {
    return pimpl_->get_error_cost(error_idx);
}

rust::Vec<int32_t> TesseractDecoderWrapper::get_error_detectors(size_t error_idx) const {
    return pimpl_->get_error_detectors(error_idx);
}

uint64_t TesseractDecoderWrapper::get_error_observables(size_t error_idx) const {
    return pimpl_->get_error_observables(error_idx);
}

uint64_t TesseractDecoderWrapper::mask_from_errors(const rust::Slice<const size_t> error_indices) const {
    return pimpl_->mask_from_errors(error_indices);
}

double TesseractDecoderWrapper::cost_from_errors(const rust::Slice<const size_t> error_indices) const {
    return pimpl_->cost_from_errors(error_indices);
}

// Trellis-mode decoder wrapper
class TesseractTrellisDecoderWrapper::Impl {
private:
    std::unique_ptr<TesseractTrellisDecoder> decoder_;

    static TesseractTrellisRankingMode ranking_mode_from_repr(TesseractTrellisRankingModeRepr repr) {
        switch (repr) {
            case TesseractTrellisRankingModeRepr::MassOnly:
                return TesseractTrellisRankingMode::MassOnly;
            case TesseractTrellisRankingModeRepr::FutureDetcostRanked:
                return TesseractTrellisRankingMode::FutureDetcostRanked;
            case TesseractTrellisRankingModeRepr::FutureActiveDetcostRanked:
                return TesseractTrellisRankingMode::FutureActiveDetcostRanked;
        }
        throw std::invalid_argument("Unknown trellis ranking mode");
    }

public:
    Impl(const std::string& dem_string, const TesseractTrellisConfigRepr& config_repr) {
        TesseractTrellisConfig config;
        config.dem = parse_dem(dem_string);
        config.beam_width = config_repr.beam_width;
        config.beam_eps = config_repr.beam_eps;
        config.future_detcost_scale = config_repr.future_detcost_scale;
        config.verbose = config_repr.verbose;
        config.merge_errors = config_repr.merge_errors;
        config.ranking_mode = ranking_mode_from_repr(config_repr.ranking_mode);

        decoder_ = std::make_unique<TesseractTrellisDecoder>(std::move(config));
    }

    TesseractTrellisResultRepr decode_detections(const rust::Slice<const uint64_t> detections) {
        std::vector<uint64_t> det_vec(detections.data(), detections.data() + detections.size());

        decoder_->decode_shot(det_vec);

        TesseractTrellisResultRepr result;
        result.observables_mask = decoder_->predicted_obs_mask;
        result.observable_probability = decoder_->observable_probability();
        result.low_confidence = decoder_->low_confidence_flag;
        result.num_states_expanded = decoder_->num_states_expanded;
        result.num_states_merged = decoder_->num_states_merged;
        result.max_beam_size_seen = decoder_->max_beam_size_seen;
        result.max_frontier_width_seen = decoder_->max_frontier_width_seen;
        return result;
    }

    size_t get_num_detectors() const {
        return decoder_->num_detectors;
    }

    size_t get_num_errors() const {
        return decoder_->errors.size();
    }

    size_t get_num_observables() const {
        return decoder_->num_observables;
    }
};

TesseractTrellisDecoderWrapper::TesseractTrellisDecoderWrapper(
    const std::string& dem_string,
    const TesseractTrellisConfigRepr& config_repr
) : pimpl_(std::make_unique<Impl>(dem_string, config_repr)) {
}

TesseractTrellisDecoderWrapper::~TesseractTrellisDecoderWrapper() = default;

TesseractTrellisResultRepr TesseractTrellisDecoderWrapper::decode_detections(
    const rust::Slice<const uint64_t> detections
) {
    return pimpl_->decode_detections(detections);
}

size_t TesseractTrellisDecoderWrapper::get_num_detectors() const {
    return pimpl_->get_num_detectors();
}

size_t TesseractTrellisDecoderWrapper::get_num_errors() const {
    return pimpl_->get_num_errors();
}

size_t TesseractTrellisDecoderWrapper::get_num_observables() const {
    return pimpl_->get_num_observables();
}

// FFI function implementations
std::unique_ptr<TesseractDecoderWrapper> create_tesseract_decoder(
    const rust::Str dem_string,
    const TesseractConfigRepr& config
) {
    try {
        std::string dem_str(dem_string);
        return std::make_unique<TesseractDecoderWrapper>(dem_str, config);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create Tesseract decoder: " + std::string(e.what()));
    }
}

DecodingResultRepr decode_detections(
    TesseractDecoderWrapper& decoder,
    const rust::Slice<const uint64_t> detections
) {
    try {
        return decoder.decode_detections(detections);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

DecodingResultRepr decode_detections_with_order(
    TesseractDecoderWrapper& decoder,
    const rust::Slice<const uint64_t> detections,
    size_t det_order
) {
    try {
        return decoder.decode_detections_with_order(detections, det_order);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding with order failed: " + std::string(e.what()));
    }
}

size_t get_num_detectors(const TesseractDecoderWrapper& decoder) {
    return decoder.get_num_detectors();
}

size_t get_num_errors(const TesseractDecoderWrapper& decoder) {
    return decoder.get_num_errors();
}

size_t get_num_observables(const TesseractDecoderWrapper& decoder) {
    return decoder.get_num_observables();
}

uint16_t get_det_beam(const TesseractDecoderWrapper& decoder) {
    return decoder.get_det_beam();
}

bool get_beam_climbing(const TesseractDecoderWrapper& decoder) {
    return decoder.get_beam_climbing();
}

bool get_no_revisit_dets(const TesseractDecoderWrapper& decoder) {
    return decoder.get_no_revisit_dets();
}

bool get_verbose(const TesseractDecoderWrapper& decoder) {
    return decoder.get_verbose();
}

bool get_merge_errors(const TesseractDecoderWrapper& decoder) {
    return decoder.get_merge_errors();
}

size_t get_pqlimit(const TesseractDecoderWrapper& decoder) {
    return decoder.get_pqlimit();
}

double get_det_penalty(const TesseractDecoderWrapper& decoder) {
    return decoder.get_det_penalty();
}

double get_error_probability(const TesseractDecoderWrapper& decoder, size_t error_idx) {
    return decoder.get_error_probability(error_idx);
}

double get_error_cost(const TesseractDecoderWrapper& decoder, size_t error_idx) {
    return decoder.get_error_cost(error_idx);
}

rust::Vec<int32_t> get_error_detectors(const TesseractDecoderWrapper& decoder, size_t error_idx) {
    return decoder.get_error_detectors(error_idx);
}

uint64_t get_error_observables(const TesseractDecoderWrapper& decoder, size_t error_idx) {
    return decoder.get_error_observables(error_idx);
}

uint64_t mask_from_errors(
    const TesseractDecoderWrapper& decoder,
    const rust::Slice<const size_t> error_indices
) {
    return decoder.mask_from_errors(error_indices);
}

double cost_from_errors(
    const TesseractDecoderWrapper& decoder,
    const rust::Slice<const size_t> error_indices
) {
    return decoder.cost_from_errors(error_indices);
}

std::unique_ptr<TesseractTrellisDecoderWrapper> create_tesseract_trellis_decoder(
    const rust::Str dem_string,
    const TesseractTrellisConfigRepr& config
) {
    try {
        std::string dem_str(dem_string);
        return std::make_unique<TesseractTrellisDecoderWrapper>(dem_str, config);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create Tesseract trellis decoder: " + std::string(e.what()));
    }
}

TesseractTrellisResultRepr trellis_decode_detections(
    TesseractTrellisDecoderWrapper& decoder,
    const rust::Slice<const uint64_t> detections
) {
    try {
        return decoder.decode_detections(detections);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

size_t trellis_num_detectors(const TesseractTrellisDecoderWrapper& decoder) {
    return decoder.get_num_detectors();
}

size_t trellis_num_errors(const TesseractTrellisDecoderWrapper& decoder) {
    return decoder.get_num_errors();
}

size_t trellis_num_observables(const TesseractTrellisDecoderWrapper& decoder) {
    return decoder.get_num_observables();
}
