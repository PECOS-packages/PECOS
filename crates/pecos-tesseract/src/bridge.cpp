//! C++ bridge implementation for Tesseract decoder

#include "tesseract_bridge.h"
#include "pecos-tesseract/src/bridge.rs.h"
#include <memory>
#include <stdexcept>
#include <sstream>
#include <numeric>  // Required for std::iota on MSVC

// Include Tesseract headers
#include "tesseract.h"
#include "common.h"
#include "utils.h"

// Include Stim headers
#include "stim/dem/detector_error_model.h"

// PIMPL implementation to hide Tesseract details
class TesseractDecoderWrapper::Impl {
private:
    std::unique_ptr<TesseractDecoder> decoder_;
    TesseractConfig config_;

public:
    Impl(const std::string& dem_string, const TesseractConfigRepr& config_repr) {
        // Parse the DEM string using the string_view constructor
        stim::DetectorErrorModel dem;
        try {
            dem = stim::DetectorErrorModel(dem_string);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse DEM string: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Failed to parse DEM string: unknown error");
        }

        // Convert config representation to TesseractConfig
        TesseractConfig config;
        config.dem = std::move(dem);
        config.det_beam = (config_repr.det_beam == std::numeric_limits<uint16_t>::max()) ?
                          INF_DET_BEAM : static_cast<int>(config_repr.det_beam);
        config.beam_climbing = config_repr.beam_climbing;
        config.no_revisit_dets = config_repr.no_revisit_dets;
        config.at_most_two_errors_per_detector = config_repr.at_most_two_errors_per_detector;
        config.verbose = config_repr.verbose;
        config.pqlimit = config_repr.pqlimit;
        config.det_penalty = config_repr.det_penalty;

        // Initialize detector orders with a default ordering
        if (config.det_orders.empty()) {
            std::vector<size_t> default_order;
            size_t num_dets = config.dem.count_detectors();
            for (size_t i = 0; i < num_dets; ++i) {
                default_order.push_back(i);
            }
            config.det_orders.push_back(default_order);
        }

        config_ = config;
        decoder_ = std::make_unique<TesseractDecoder>(std::move(config));
    }

    DecodingResultRepr decode_detections(const rust::Slice<const uint64_t> detections) {
        std::vector<uint64_t> det_vec(detections.begin(), detections.end());

        decoder_->decode_to_errors(det_vec);

        DecodingResultRepr result;
        result.predicted_errors = rust::Vec<size_t>();
        for (size_t err : decoder_->predicted_errors_buffer) {
            result.predicted_errors.push_back(err);
        }

        result.observables_mask = decoder_->mask_from_errors(decoder_->predicted_errors_buffer);
        result.cost = decoder_->cost_from_errors(decoder_->predicted_errors_buffer);
        result.low_confidence = decoder_->low_confidence_flag;

        return result;
    }

    DecodingResultRepr decode_detections_with_order(
        const rust::Slice<const uint64_t> detections,
        size_t det_order
    ) {
        std::vector<uint64_t> det_vec(detections.begin(), detections.end());

        decoder_->decode_to_errors(det_vec, det_order);

        DecodingResultRepr result;
        result.predicted_errors = rust::Vec<size_t>();
        for (size_t err : decoder_->predicted_errors_buffer) {
            result.predicted_errors.push_back(err);
        }

        result.observables_mask = decoder_->mask_from_errors(decoder_->predicted_errors_buffer);
        result.cost = decoder_->cost_from_errors(decoder_->predicted_errors_buffer);
        result.low_confidence = decoder_->low_confidence_flag;

        return result;
    }

    size_t get_num_detectors() const {
        return config_.dem.count_detectors();
    }

    size_t get_num_errors() const {
        return decoder_->errors.size();
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

    bool get_at_most_two_errors_per_detector() const {
        return config_.at_most_two_errors_per_detector;
    }

    bool get_verbose() const {
        return config_.verbose;
    }

    size_t get_pqlimit() const {
        return config_.pqlimit;
    }

    double get_det_penalty() const {
        return config_.det_penalty;
    }

    double get_error_probability(size_t error_idx) const {
        if (error_idx >= decoder_->errors.size()) {
            throw std::out_of_range("Error index out of range");
        }
        return decoder_->errors[error_idx].probability;
    }

    double get_error_cost(size_t error_idx) const {
        if (error_idx >= decoder_->errors.size()) {
            throw std::out_of_range("Error index out of range");
        }
        return decoder_->errors[error_idx].likelihood_cost;
    }

    rust::Vec<int32_t> get_error_detectors(size_t error_idx) const {
        if (error_idx >= decoder_->errors.size()) {
            throw std::out_of_range("Error index out of range");
        }

        rust::Vec<int32_t> detectors;
        for (int det : decoder_->errors[error_idx].symptom.detectors) {
            detectors.push_back(static_cast<int32_t>(det));
        }
        return detectors;
    }

    uint64_t get_error_observables(size_t error_idx) const {
        if (error_idx >= decoder_->errors.size()) {
            throw std::out_of_range("Error index out of range");
        }
        return decoder_->errors[error_idx].symptom.observables;
    }

    uint64_t mask_from_errors(const rust::Slice<const size_t> error_indices) const {
        // Work around Tesseract bug: functions ignore parameter and use internal buffer
        // So we calculate the mask ourselves
        uint64_t mask = 0;
        for (size_t ei : error_indices) {
            if (ei < decoder_->errors.size()) {
                mask ^= decoder_->errors[ei].symptom.observables;
            }
        }
        return mask;
    }

    double cost_from_errors(const rust::Slice<const size_t> error_indices) const {
        // Work around Tesseract bug: functions ignore parameter and use internal buffer
        // So we calculate the cost ourselves
        double total_cost = 0;
        for (size_t ei : error_indices) {
            if (ei < decoder_->errors.size()) {
                total_cost += decoder_->errors[ei].likelihood_cost;
            }
        }
        return total_cost;
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

bool TesseractDecoderWrapper::get_at_most_two_errors_per_detector() const {
    return pimpl_->get_at_most_two_errors_per_detector();
}

bool TesseractDecoderWrapper::get_verbose() const {
    return pimpl_->get_verbose();
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

bool get_at_most_two_errors_per_detector(const TesseractDecoderWrapper& decoder) {
    return decoder.get_at_most_two_errors_per_detector();
}

bool get_verbose(const TesseractDecoderWrapper& decoder) {
    return decoder.get_verbose();
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
