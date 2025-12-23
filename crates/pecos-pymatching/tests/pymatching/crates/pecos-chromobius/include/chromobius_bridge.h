//! C++ header for Chromobius decoder bridge

#ifndef CHROMOBIUS_BRIDGE_H
#define CHROMOBIUS_BRIDGE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "rust/cxx.h"

// Define export/import macros for shared library
#ifdef _WIN32
  #ifdef CHROMOBIUS_BRIDGE_EXPORTS
    #define CHROMOBIUS_API __declspec(dllexport)
  #else
    #define CHROMOBIUS_API __declspec(dllimport)
  #endif
#else
  #define CHROMOBIUS_API __attribute__((visibility("default")))
#endif

// Forward declarations
// Note: No namespace needed as ChromobiusDecoderWrapper uses PIMPL pattern

// ChromobiusDecoderWrapper must be outside namespace for CXX
class CHROMOBIUS_API ChromobiusDecoderWrapper {
public:
    ChromobiusDecoderWrapper(const std::string& dem_string, bool drop_mobius_errors_involving_remnant_errors);
    ~ChromobiusDecoderWrapper();

    // Disable copy
    ChromobiusDecoderWrapper(const ChromobiusDecoderWrapper&) = delete;
    ChromobiusDecoderWrapper& operator=(const ChromobiusDecoderWrapper&) = delete;

    // Allow move
    ChromobiusDecoderWrapper(ChromobiusDecoderWrapper&&) = default;
    ChromobiusDecoderWrapper& operator=(ChromobiusDecoderWrapper&&) = default;

    // Initialize decoder (for use after default construction)
    void init(const std::string& dem_string, bool drop_mobius_errors_involving_remnant_errors);

    // Decode detection events to predicted observables
    uint64_t decode_detection_events(const rust::Slice<const uint8_t> bit_packed_detection_events);

    // Decode and get weight
    uint64_t decode_detection_events_with_weight(
        const rust::Slice<const uint8_t> bit_packed_detection_events,
        float& weight_out
    );

    // Get decoder properties
    size_t get_num_detectors() const;
    size_t get_num_observables() const;

private:
    // Use PIMPL to hide Chromobius implementation details
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// FFI function declarations with unique names to avoid collisions
CHROMOBIUS_API std::unique_ptr<ChromobiusDecoderWrapper> create_chromobius_decoder(
    const rust::Str dem_string,
    bool drop_mobius_errors_involving_remnant_errors
);

CHROMOBIUS_API uint64_t decode_detection_events(
    ChromobiusDecoderWrapper& decoder,
    const rust::Slice<const uint8_t> bit_packed_detection_events
);

CHROMOBIUS_API uint64_t decode_detection_events_with_weight(
    ChromobiusDecoderWrapper& decoder,
    const rust::Slice<const uint8_t> bit_packed_detection_events,
    float& weight_out
);

CHROMOBIUS_API size_t chromobius_get_num_detectors(const ChromobiusDecoderWrapper& decoder);

CHROMOBIUS_API size_t chromobius_get_num_observables(const ChromobiusDecoderWrapper& decoder);

#endif // CHROMOBIUS_BRIDGE_H
