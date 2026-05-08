//! C++ header for Tesseract decoder bridge

#pragma once

#include "rust/cxx.h"
#include <memory>
#include <vector>
#include <cstdint>

// Forward declare the Rust types
struct TesseractConfigRepr;
struct DecodingResultRepr;

// Simple wrapper class for Tesseract decoder
// CXX bridge requires the complete type definition
class TesseractDecoderWrapper {
public:
    TesseractDecoderWrapper(const std::string& dem_string, const TesseractConfigRepr& config);
    ~TesseractDecoderWrapper(); // Must be defined in .cpp where Impl is complete

    // We'll implement these methods in the .cpp file
    void init(const std::string& dem_string, const TesseractConfigRepr& config);
    DecodingResultRepr decode_detections(const rust::Slice<const uint64_t> detections);
    DecodingResultRepr decode_detections_with_order(const rust::Slice<const uint64_t> detections, size_t det_order);

    // Getter methods
    size_t get_num_detectors() const;
    size_t get_num_errors() const;
    size_t get_num_observables() const;
    uint16_t get_det_beam() const;
    bool get_beam_climbing() const;
    bool get_no_revisit_dets() const;
    bool get_verbose() const;
    size_t get_pqlimit() const;
    double get_det_penalty() const;
    double get_error_probability(size_t error_idx) const;
    double get_error_cost(size_t error_idx) const;
    rust::Vec<int32_t> get_error_detectors(size_t error_idx) const;
    uint64_t get_error_observables(size_t error_idx) const;
    uint64_t mask_from_errors(const rust::Slice<const size_t> error_indices) const;
    double cost_from_errors(const rust::Slice<const size_t> error_indices) const;

private:
    // We'll use PIMPL pattern to hide the actual Tesseract implementation
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Note: We avoid defining TesseractDecoder alias to prevent conflicts
// The CXX bridge will use TesseractDecoderWrapper directly

// Function declarations that match the CXX bridge
std::unique_ptr<TesseractDecoderWrapper> create_tesseract_decoder(
    const rust::Str dem_string,
    const TesseractConfigRepr& config
);

DecodingResultRepr decode_detections(
    TesseractDecoderWrapper& decoder,
    const rust::Slice<const uint64_t> detections
);

DecodingResultRepr decode_detections_with_order(
    TesseractDecoderWrapper& decoder,
    const rust::Slice<const uint64_t> detections,
    size_t det_order
);

size_t get_num_detectors(const TesseractDecoderWrapper& decoder);
size_t get_num_errors(const TesseractDecoderWrapper& decoder);
size_t get_num_observables(const TesseractDecoderWrapper& decoder);

uint16_t get_det_beam(const TesseractDecoderWrapper& decoder);
bool get_beam_climbing(const TesseractDecoderWrapper& decoder);
bool get_no_revisit_dets(const TesseractDecoderWrapper& decoder);
bool get_verbose(const TesseractDecoderWrapper& decoder);
size_t get_pqlimit(const TesseractDecoderWrapper& decoder);
double get_det_penalty(const TesseractDecoderWrapper& decoder);

double get_error_probability(const TesseractDecoderWrapper& decoder, size_t error_idx);
double get_error_cost(const TesseractDecoderWrapper& decoder, size_t error_idx);
rust::Vec<int32_t> get_error_detectors(const TesseractDecoderWrapper& decoder, size_t error_idx);
uint64_t get_error_observables(const TesseractDecoderWrapper& decoder, size_t error_idx);

uint64_t mask_from_errors(
    const TesseractDecoderWrapper& decoder,
    const rust::Slice<const size_t> error_indices
);

double cost_from_errors(
    const TesseractDecoderWrapper& decoder,
    const rust::Slice<const size_t> error_indices
);
