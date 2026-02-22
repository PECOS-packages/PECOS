//! C++ bridge implementation for Chromobius decoder

#include "chromobius_bridge.h"
#include "pecos-chromobius/src/bridge.rs.h"
#include <memory>
#include <stdexcept>
#include <array>    // Required for std::array on MSVC

// Include Chromobius headers
#include "chromobius/decode/decoder.h"
#include "chromobius/datatypes/conf.h"

// Include Stim headers
#include "stim/dem/detector_error_model.h"

// PIMPL implementation to hide Chromobius details
class ChromobiusDecoderWrapper::Impl {
private:
    chromobius::Decoder decoder_;
    size_t num_detectors_;
    size_t num_observables_;

public:
    Impl(const std::string& dem_string, bool drop_mobius_errors_involving_remnant_errors) {
        // Parse the DEM string using Stim
        stim::DetectorErrorModel dem;
        try {
            dem = stim::DetectorErrorModel(dem_string);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse DEM string: ") + e.what());
        }

        // Configure Chromobius decoder options
        chromobius::DecoderConfigOptions options;
        options.drop_mobius_errors_involving_remnant_errors = drop_mobius_errors_involving_remnant_errors;
        options.ignore_decomposition_failures = false;
        options.include_coords_in_mobius_dem = false;
        // Use default matcher (PyMatching)

        // Create decoder
        try {
            decoder_ = chromobius::Decoder::from_dem(dem, options);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to create Chromobius decoder: ") + e.what());
        }

        // Store counts
        num_detectors_ = dem.count_detectors();
        num_observables_ = dem.count_observables();
    }

    uint64_t decode_detection_events(const rust::Slice<const uint8_t> bit_packed_detection_events) {
        // Create a mutable copy since Chromobius modifies the input
        // Use data()+size() instead of begin()/end() iterators to avoid
        // Xcode 15.4 libc++ pointer_traits incompatibility with cxx iterators in C++20
        std::vector<uint8_t> mutable_data(bit_packed_detection_events.data(), bit_packed_detection_events.data() + bit_packed_detection_events.size());

        // Decode
        chromobius::obsmask_int result = decoder_.decode_detection_events(mutable_data);

        return static_cast<uint64_t>(result);
    }

    uint64_t decode_detection_events_with_weight(
        const rust::Slice<const uint8_t> bit_packed_detection_events,
        float& weight_out
    ) {
        // Create a mutable copy since Chromobius modifies the input
        std::vector<uint8_t> mutable_data(bit_packed_detection_events.data(), bit_packed_detection_events.data() + bit_packed_detection_events.size());

        // Decode with weight
        chromobius::obsmask_int result = decoder_.decode_detection_events(mutable_data, &weight_out);

        return static_cast<uint64_t>(result);
    }

    size_t get_num_detectors() const {
        return num_detectors_;
    }

    size_t get_num_observables() const {
        return num_observables_;
    }
};

// ChromobiusDecoderWrapper implementation
ChromobiusDecoderWrapper::ChromobiusDecoderWrapper(
    const std::string& dem_string,
    bool drop_mobius_errors_involving_remnant_errors
) : pimpl_(std::make_unique<Impl>(dem_string, drop_mobius_errors_involving_remnant_errors)) {
}

ChromobiusDecoderWrapper::~ChromobiusDecoderWrapper() = default;
ChromobiusDecoderWrapper::ChromobiusDecoderWrapper(ChromobiusDecoderWrapper&&) noexcept = default;
ChromobiusDecoderWrapper& ChromobiusDecoderWrapper::operator=(ChromobiusDecoderWrapper&&) noexcept = default;

void ChromobiusDecoderWrapper::init(
    const std::string& dem_string,
    bool drop_mobius_errors_involving_remnant_errors
) {
    pimpl_ = std::make_unique<Impl>(dem_string, drop_mobius_errors_involving_remnant_errors);
}

uint64_t ChromobiusDecoderWrapper::decode_detection_events(
    const rust::Slice<const uint8_t> bit_packed_detection_events
) {
    return pimpl_->decode_detection_events(bit_packed_detection_events);
}

uint64_t ChromobiusDecoderWrapper::decode_detection_events_with_weight(
    const rust::Slice<const uint8_t> bit_packed_detection_events,
    float& weight_out
) {
    return pimpl_->decode_detection_events_with_weight(bit_packed_detection_events, weight_out);
}

size_t ChromobiusDecoderWrapper::get_num_detectors() const {
    return pimpl_->get_num_detectors();
}

size_t ChromobiusDecoderWrapper::get_num_observables() const {
    return pimpl_->get_num_observables();
}

// FFI function implementations
std::unique_ptr<ChromobiusDecoderWrapper> create_chromobius_decoder(
    const rust::Str dem_string,
    bool drop_mobius_errors_involving_remnant_errors
) {
    try {
        std::string dem_str(dem_string);
        return std::make_unique<ChromobiusDecoderWrapper>(dem_str, drop_mobius_errors_involving_remnant_errors);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create Chromobius decoder: " + std::string(e.what()));
    }
}

uint64_t decode_detection_events(
    ChromobiusDecoderWrapper& decoder,
    const rust::Slice<const uint8_t> bit_packed_detection_events
) {
    try {
        return decoder.decode_detection_events(bit_packed_detection_events);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

uint64_t decode_detection_events_with_weight(
    ChromobiusDecoderWrapper& decoder,
    const rust::Slice<const uint8_t> bit_packed_detection_events,
    float& weight_out
) {
    try {
        return decoder.decode_detection_events_with_weight(bit_packed_detection_events, weight_out);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding with weight failed: " + std::string(e.what()));
    }
}

size_t chromobius_get_num_detectors(const ChromobiusDecoderWrapper& decoder) {
    return decoder.get_num_detectors();
}

size_t chromobius_get_num_observables(const ChromobiusDecoderWrapper& decoder) {
    return decoder.get_num_observables();
}
