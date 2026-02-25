#include "ldpc_ffi.h"
#include "pecos-ldpc-decoders/src/bridge.rs.h"  // Include the generated cxx header
#include <memory>
#include <stdexcept>
#include <vector>
#include <set>
#include <map>
#include <robin_map.h>
#include <robin_set.h>
#include "gf2sparse.hpp"  // Include before bp.hpp to get complete type
#include "bp.hpp"
#include "osd.hpp"
#include "lsd.hpp"
#include "flip.hpp"
#include "sparse_matrix_util.hpp"
#include "gf2sparse_linalg.hpp"  // Include before union_find.hpp for dependencies
#include "union_find.hpp"
#include "mbp.hpp"

using namespace ldpc;

// Destructor implementations
BpOsdDecoder::~BpOsdDecoder() {
    // IMPORTANT: Delete in reverse construction order
    // osd_decoder created last, so delete first; then bp_decoder; then pcm last
    if (osd_decoder) delete static_cast<osd::OsdDecoder*>(osd_decoder);
    if (bp_decoder) delete static_cast<bp::BpDecoder*>(bp_decoder);
    if (pcm) delete static_cast<bp::BpSparse*>(pcm);
}

BpLsdDecoder::~BpLsdDecoder() {
    // IMPORTANT: Delete child decoders BEFORE pcm because they contain references to pcm
    // Deleting pcm first causes use-after-free when their destructors try to access it
    if (lsd_decoder) delete static_cast<lsd::LsdDecoder*>(lsd_decoder);
    if (bp_decoder) delete static_cast<bp::BpDecoder*>(bp_decoder);
    if (pcm) delete static_cast<bp::BpSparse*>(pcm);
}

// Helper function to create PCM from sparse representation
static bp::BpSparse* create_pcm_from_sparse(const SparseMatrixRepr& sparse) {
    auto pcm = new bp::BpSparse(sparse.rows, sparse.cols);

    if (sparse.row_indices.size() != sparse.col_indices.size()) {
        throw std::runtime_error("Row and column indices must have the same length");
    }

    for (size_t i = 0; i < sparse.row_indices.size(); i++) {
        pcm->insert_entry(sparse.row_indices[i], sparse.col_indices[i]);
    }

    return pcm;
}

// BP+OSD Decoder implementation
std::unique_ptr<BpOsdDecoder> create_bp_osd_decoder(
    const SparseMatrixRepr& pcm,
    rust::Slice<const double> channel_probs,
    int32_t max_iter,
    int32_t bp_method,
    int32_t bp_schedule,
    double ms_scaling_factor,
    int32_t osd_method,
    int32_t osd_order,
    int32_t input_vector_type,
    int32_t omp_thread_count,
    rust::Slice<const int32_t> serial_schedule_order,
    int32_t random_schedule_seed
) {
    if (channel_probs.size() != pcm.cols) {
        throw std::runtime_error("Channel probabilities length must match number of columns");
    }

    auto decoder = std::make_unique<BpOsdDecoder>();

    // Initialize all pointers to nullptr to avoid deleting garbage in destructor
    decoder->pcm = nullptr;
    decoder->bp_decoder = nullptr;
    decoder->osd_decoder = nullptr;

    // Create PCM
    decoder->pcm = create_pcm_from_sparse(pcm);

    // Copy channel probabilities
    // Use data()+size() instead of begin()/end() iterators to avoid
    // Xcode 15.4 libc++ pointer_traits incompatibility with cxx iterators in C++20
    decoder->channel_probs.assign(channel_probs.data(), channel_probs.data() + channel_probs.size());

    // Convert serial schedule order
    std::vector<int> serial_schedule_vec;
    if (!serial_schedule_order.empty()) {
        serial_schedule_vec.assign(serial_schedule_order.data(), serial_schedule_order.data() + serial_schedule_order.size());
    }

    // Create BP decoder
    decoder->bp_decoder = new bp::BpDecoder(
        *static_cast<bp::BpSparse*>(decoder->pcm),
        decoder->channel_probs,
        max_iter,
        static_cast<bp::BpMethod>(bp_method),
        static_cast<bp::BpSchedule>(bp_schedule),
        ms_scaling_factor,
        omp_thread_count,
        serial_schedule_vec,
        random_schedule_seed,
        true,  // random_schedule_at_every_iteration
        static_cast<bp::BpInputType>(input_vector_type)
    );

    // Create OSD decoder if enabled
    decoder->use_osd = (osd_method != 0);  // 0 = OSD_OFF
    if (decoder->use_osd) {
        decoder->osd_decoder = new osd::OsdDecoder(
            *static_cast<bp::BpSparse*>(decoder->pcm),
            static_cast<osd::OsdMethod>(osd_method),
            osd_order,
            decoder->channel_probs
        );
    }

    // Store parameters
    decoder->max_iter = max_iter;
    decoder->bp_method = bp_method;
    decoder->bp_schedule = bp_schedule;
    decoder->ms_scaling_factor = ms_scaling_factor;
    decoder->osd_method = osd_method;
    decoder->osd_order = osd_order;
    decoder->input_vector_type = input_vector_type;
    decoder->omp_thread_count = omp_thread_count;
    decoder->random_schedule_seed = random_schedule_seed;

    return decoder;
}

DecodingResult decode_bp_osd(
    BpOsdDecoder& decoder,
    rust::Slice<const uint8_t> input_vector
) {
    auto pcm = static_cast<bp::BpSparse*>(decoder.pcm);

    // Validate input size based on input type
    if (decoder.input_vector_type == bp::BpInputType::SYNDROME) {
        if (input_vector.size() != pcm->m) {
            throw std::runtime_error("Syndrome length must match number of checks");
        }
    } else if (decoder.input_vector_type == bp::BpInputType::RECEIVED_VECTOR) {
        if (input_vector.size() != pcm->n) {
            throw std::runtime_error("Received vector length must match number of bits");
        }
    } else { // AUTO
        // BP decoder will handle the auto detection
        if (input_vector.size() != pcm->m && input_vector.size() != pcm->n) {
            throw std::runtime_error("Input vector length must match either number of checks or bits");
        }
    }

    // Convert input to vector
    std::vector<uint8_t> input_vec(input_vector.data(), input_vector.data() + input_vector.size());

    // First try BP decoding
    auto bp_decoder = static_cast<bp::BpDecoder*>(decoder.bp_decoder);
    auto bp_result = bp_decoder->decode(input_vec);

    DecodingResult result;
    result.converged = bp_decoder->converge;
    result.iterations = bp_decoder->iterations;

    // If BP converged or OSD is not enabled, return BP result
    if (bp_decoder->converge || !decoder.use_osd) {
        result.decoding = rust::Vec<uint8_t>();
        result.decoding.reserve(bp_result.size());
        for (auto val : bp_result) {
            result.decoding.push_back(val);
        }
        return result;
    }

    // BP didn't converge, use OSD
    // Note: OSD requires syndrome input. If we received a received_vector, we would need
    // to compute the syndrome first. For now, we require syndrome input when OSD is enabled.
    if (decoder.input_vector_type == bp::BpInputType::RECEIVED_VECTOR) {
        throw std::runtime_error("OSD decoding requires syndrome input. Please use InputVectorType::Syndrome when OSD is enabled.");
    }

    std::vector<double> log_prob_ratios(bp_decoder->log_prob_ratios);
    auto osd_decoder = static_cast<osd::OsdDecoder*>(decoder.osd_decoder);
    auto osd_result = osd_decoder->decode(input_vec, log_prob_ratios);

    result.decoding = rust::Vec<uint8_t>();
    result.decoding.reserve(osd_result.size());
    for (auto val : osd_result) {
        result.decoding.push_back(val);
    }

    return result;
}

rust::Vec<double> get_log_prob_ratios_osd(const BpOsdDecoder& decoder) {
    rust::Vec<double> llrs;
    auto bp_decoder = static_cast<const bp::BpDecoder*>(decoder.bp_decoder);
    llrs.reserve(bp_decoder->log_prob_ratios.size());
    for (auto val : bp_decoder->log_prob_ratios) {
        llrs.push_back(val);
    }
    return llrs;
}

// BP+LSD Decoder implementation
std::unique_ptr<BpLsdDecoder> create_bp_lsd_decoder(
    const SparseMatrixRepr& pcm,
    rust::Slice<const double> channel_probs,
    int32_t max_iter,
    int32_t bp_method,
    int32_t bp_schedule,
    double ms_scaling_factor,
    int32_t lsd_method,
    int32_t lsd_order,
    int32_t bits_per_step,
    int32_t input_vector_type,
    int32_t omp_thread_count,
    rust::Slice<const int32_t> serial_schedule_order,
    int32_t random_schedule_seed
) {
    if (channel_probs.size() != pcm.cols) {
        throw std::runtime_error("Channel probabilities length must match number of columns");
    }

    auto decoder = std::make_unique<BpLsdDecoder>();

    // Initialize all pointers to nullptr to avoid deleting garbage in destructor
    decoder->pcm = nullptr;
    decoder->bp_decoder = nullptr;
    decoder->lsd_decoder = nullptr;

    // Create PCM
    decoder->pcm = create_pcm_from_sparse(pcm);

    // Copy channel probabilities
    decoder->channel_probs.assign(channel_probs.data(), channel_probs.data() + channel_probs.size());

    // Convert serial schedule order
    std::vector<int> serial_schedule_vec;
    if (!serial_schedule_order.empty()) {
        serial_schedule_vec.assign(serial_schedule_order.data(), serial_schedule_order.data() + serial_schedule_order.size());
    }

    // Create BP decoder
    decoder->bp_decoder = new bp::BpDecoder(
        *static_cast<bp::BpSparse*>(decoder->pcm),
        decoder->channel_probs,
        max_iter,
        static_cast<bp::BpMethod>(bp_method),
        static_cast<bp::BpSchedule>(bp_schedule),
        ms_scaling_factor,
        omp_thread_count,
        serial_schedule_vec,
        random_schedule_seed,
        true,  // random_schedule_at_every_iteration
        static_cast<bp::BpInputType>(input_vector_type)
    );

    // Create LSD decoder
    decoder->lsd_decoder = new lsd::LsdDecoder(
        *static_cast<bp::BpSparse*>(decoder->pcm),
        static_cast<osd::OsdMethod>(lsd_method),
        lsd_order
    );

    decoder->bits_per_step = bits_per_step;

    // Store parameters
    decoder->max_iter = max_iter;
    decoder->bp_method = bp_method;
    decoder->bp_schedule = bp_schedule;
    decoder->ms_scaling_factor = ms_scaling_factor;
    decoder->lsd_method = lsd_method;
    decoder->lsd_order = lsd_order;
    decoder->input_vector_type = input_vector_type;
    decoder->omp_thread_count = omp_thread_count;
    decoder->random_schedule_seed = random_schedule_seed;

    return decoder;
}

DecodingResult decode_bp_lsd(
    BpLsdDecoder& decoder,
    rust::Slice<const uint8_t> input_vector
) {
    auto pcm = static_cast<bp::BpSparse*>(decoder.pcm);

    // Validate input size based on input type
    if (decoder.input_vector_type == bp::BpInputType::SYNDROME) {
        if (input_vector.size() != pcm->m) {
            throw std::runtime_error("Syndrome length must match number of checks");
        }
    } else if (decoder.input_vector_type == bp::BpInputType::RECEIVED_VECTOR) {
        if (input_vector.size() != pcm->n) {
            throw std::runtime_error("Received vector length must match number of bits");
        }
    } else { // AUTO
        // BP decoder will handle the auto detection
        if (input_vector.size() != pcm->m && input_vector.size() != pcm->n) {
            throw std::runtime_error("Input vector length must match either number of checks or bits");
        }
    }

    // Convert input to vector
    std::vector<uint8_t> input_vec(input_vector.data(), input_vector.data() + input_vector.size());

    // First try BP decoding
    auto bp_decoder = static_cast<bp::BpDecoder*>(decoder.bp_decoder);
    auto bp_result = bp_decoder->decode(input_vec);

    DecodingResult result;
    result.converged = bp_decoder->converge;
    result.iterations = bp_decoder->iterations;

    // If BP converged, return BP result
    if (bp_decoder->converge) {
        result.decoding = rust::Vec<uint8_t>();
        result.decoding.reserve(bp_result.size());
        for (auto val : bp_result) {
            result.decoding.push_back(val);
        }
        return result;
    }

    // BP didn't converge, use LSD
    // Note: LSD requires syndrome input. If we received a received_vector, we would need
    // to compute the syndrome first. For now, we require syndrome input when LSD is used.
    if (decoder.input_vector_type == bp::BpInputType::RECEIVED_VECTOR) {
        throw std::runtime_error("LSD decoding requires syndrome input. Please use InputVectorType::Syndrome when LSD is enabled.");
    }

    // Convert log probability ratios to bit weights
    std::vector<double> bit_weights(pcm->n);
    for (size_t i = 0; i < bit_weights.size(); i++) {
        double llr = bp_decoder->log_prob_ratios[i];
        // Convert LLR to probability that bit is 1
        bit_weights[i] = 1.0 / (1.0 + std::exp(llr));
    }

    // Run LSD
    auto lsd_decoder = static_cast<lsd::LsdDecoder*>(decoder.lsd_decoder);
    auto lsd_result = lsd_decoder->lsd_decode(
        input_vec,
        bit_weights,
        decoder.bits_per_step,
        true  // is_on_the_fly
    );

    result.decoding = rust::Vec<uint8_t>();
    result.decoding.reserve(lsd_result.size());
    for (auto val : lsd_result) {
        result.decoding.push_back(val);
    }

    return result;
}

rust::Vec<double> get_log_prob_ratios_lsd(const BpLsdDecoder& decoder) {
    rust::Vec<double> llrs;
    auto bp_decoder = static_cast<const bp::BpDecoder*>(decoder.bp_decoder);
    llrs.reserve(bp_decoder->log_prob_ratios.size());
    for (auto val : bp_decoder->log_prob_ratios) {
        llrs.push_back(val);
    }
    return llrs;
}

// Getter implementations for BP+OSD decoder
uint32_t get_check_count_osd(const BpOsdDecoder& decoder) {
    return static_cast<bp::BpSparse*>(decoder.pcm)->m;
}

uint32_t get_bit_count_osd(const BpOsdDecoder& decoder) {
    return static_cast<bp::BpSparse*>(decoder.pcm)->n;
}

rust::Vec<double> get_channel_probs_osd(const BpOsdDecoder& decoder) {
    rust::Vec<double> probs;
    probs.reserve(decoder.channel_probs.size());
    for (auto val : decoder.channel_probs) {
        probs.push_back(val);
    }
    return probs;
}

int32_t get_max_iter_osd(const BpOsdDecoder& decoder) {
    return decoder.max_iter;
}

int32_t get_bp_method_osd(const BpOsdDecoder& decoder) {
    return decoder.bp_method;
}

int32_t get_bp_schedule_osd(const BpOsdDecoder& decoder) {
    return decoder.bp_schedule;
}

double get_ms_scaling_factor_osd(const BpOsdDecoder& decoder) {
    return decoder.ms_scaling_factor;
}

int32_t get_osd_method_osd(const BpOsdDecoder& decoder) {
    return decoder.osd_method;
}

int32_t get_osd_order_osd(const BpOsdDecoder& decoder) {
    return decoder.osd_order;
}

bool get_converged_osd(const BpOsdDecoder& decoder) {
    return static_cast<const bp::BpDecoder*>(decoder.bp_decoder)->converge;
}

int32_t get_iterations_osd(const BpOsdDecoder& decoder) {
    return static_cast<const bp::BpDecoder*>(decoder.bp_decoder)->iterations;
}

rust::Vec<uint8_t> get_bp_decoding_osd(const BpOsdDecoder& decoder) {
    rust::Vec<uint8_t> decoding;
    auto bp_decoder = static_cast<const bp::BpDecoder*>(decoder.bp_decoder);
    decoding.reserve(bp_decoder->decoding.size());
    for (auto val : bp_decoder->decoding) {
        decoding.push_back(val);
    }
    return decoding;
}

int32_t get_input_vector_type_osd(const BpOsdDecoder& decoder) {
    return decoder.input_vector_type;
}

int32_t get_omp_thread_count_osd(const BpOsdDecoder& decoder) {
    return decoder.omp_thread_count;
}

int32_t get_random_schedule_seed_osd(const BpOsdDecoder& decoder) {
    return decoder.random_schedule_seed;
}

// Getter implementations for BP+LSD decoder
uint32_t get_check_count_lsd(const BpLsdDecoder& decoder) {
    return static_cast<bp::BpSparse*>(decoder.pcm)->m;
}

uint32_t get_bit_count_lsd(const BpLsdDecoder& decoder) {
    return static_cast<bp::BpSparse*>(decoder.pcm)->n;
}

rust::Vec<double> get_channel_probs_lsd(const BpLsdDecoder& decoder) {
    rust::Vec<double> probs;
    probs.reserve(decoder.channel_probs.size());
    for (auto val : decoder.channel_probs) {
        probs.push_back(val);
    }
    return probs;
}

int32_t get_max_iter_lsd(const BpLsdDecoder& decoder) {
    return decoder.max_iter;
}

int32_t get_bp_method_lsd(const BpLsdDecoder& decoder) {
    return decoder.bp_method;
}

int32_t get_bp_schedule_lsd(const BpLsdDecoder& decoder) {
    return decoder.bp_schedule;
}

double get_ms_scaling_factor_lsd(const BpLsdDecoder& decoder) {
    return decoder.ms_scaling_factor;
}

int32_t get_lsd_method_lsd(const BpLsdDecoder& decoder) {
    return decoder.lsd_method;
}

int32_t get_lsd_order_lsd(const BpLsdDecoder& decoder) {
    return decoder.lsd_order;
}

int32_t get_bits_per_step_lsd(const BpLsdDecoder& decoder) {
    return decoder.bits_per_step;
}

bool get_converged_lsd(const BpLsdDecoder& decoder) {
    return static_cast<const bp::BpDecoder*>(decoder.bp_decoder)->converge;
}

int32_t get_iterations_lsd(const BpLsdDecoder& decoder) {
    return static_cast<const bp::BpDecoder*>(decoder.bp_decoder)->iterations;
}

int32_t get_input_vector_type_lsd(const BpLsdDecoder& decoder) {
    return decoder.input_vector_type;
}

int32_t get_omp_thread_count_lsd(const BpLsdDecoder& decoder) {
    return decoder.omp_thread_count;
}

int32_t get_random_schedule_seed_lsd(const BpLsdDecoder& decoder) {
    return decoder.random_schedule_seed;
}

// Statistics functions for BP+LSD
void set_do_stats_lsd(BpLsdDecoder& decoder, bool enable) {
    auto lsd_decoder = static_cast<lsd::LsdDecoder*>(decoder.lsd_decoder);
    lsd_decoder->set_do_stats(enable);
}

bool get_do_stats_lsd(const BpLsdDecoder& decoder) {
    auto lsd_decoder = static_cast<const lsd::LsdDecoder*>(decoder.lsd_decoder);
    return lsd_decoder->get_do_stats();
}

rust::String get_statistics_json_lsd(const BpLsdDecoder& decoder) {
    // We need non-const access to call toString(), which is not const
    auto lsd_decoder = const_cast<lsd::LsdDecoder*>(static_cast<const lsd::LsdDecoder*>(decoder.lsd_decoder));
    return rust::String(lsd_decoder->statistics.toString());
}

// Soft Information BP Decoder implementation
SoftInfoBpDecoder::~SoftInfoBpDecoder() {
    // IMPORTANT: Delete bp_decoder BEFORE pcm because bp_decoder contains a reference to pcm
    // Deleting pcm first causes use-after-free when bp_decoder's destructor tries to access it
    if (bp_decoder) delete static_cast<bp::BpDecoder*>(bp_decoder);
    if (pcm) delete static_cast<bp::BpSparse*>(pcm);
}

std::unique_ptr<SoftInfoBpDecoder> create_soft_info_bp_decoder(
    const SparseMatrixRepr& pcm_repr,
    rust::Slice<const double> channel_probs,
    int32_t max_iter,
    int32_t bp_method,
    double ms_scaling_factor,
    int32_t omp_thread_count,
    rust::Slice<const int32_t> serial_schedule_order,
    int32_t random_schedule_seed
) {
    auto decoder = std::make_unique<SoftInfoBpDecoder>();

    // Initialize all pointers to nullptr to avoid deleting garbage in destructor
    decoder->pcm = nullptr;
    decoder->bp_decoder = nullptr;

    // Create sparse matrix
    auto pcm = new bp::BpSparse(pcm_repr.rows, pcm_repr.cols);
    for (size_t i = 0; i < pcm_repr.row_indices.size(); ++i) {
        pcm->insert_entry(pcm_repr.row_indices[i], pcm_repr.col_indices[i]);
    }
    decoder->pcm = pcm;

    // Store channel probabilities
    decoder->channel_probs = std::vector<double>(channel_probs.data(), channel_probs.data() + channel_probs.size());

    // Store parameters
    decoder->max_iter = max_iter;
    decoder->bp_method = bp_method;
    decoder->ms_scaling_factor = ms_scaling_factor;
    decoder->omp_thread_count = omp_thread_count;
    decoder->random_schedule_seed = random_schedule_seed;

    // Convert serial schedule order
    std::vector<int> schedule_order;
    if (serial_schedule_order.size() == pcm->n) {
        schedule_order.reserve(serial_schedule_order.size());
        for (auto idx : serial_schedule_order) {
            schedule_order.push_back(idx);
        }
    }

    // Create BP decoder with serial schedule for soft info decoding
    auto bp_decoder = new bp::BpDecoder(
        *pcm,
        decoder->channel_probs,
        max_iter,
        static_cast<bp::BpMethod>(bp_method),
        bp::BpSchedule::SERIAL,  // Always use serial schedule for soft info decoding
        ms_scaling_factor,
        omp_thread_count,
        schedule_order,
        random_schedule_seed
    );

    decoder->bp_decoder = bp_decoder;

    return decoder;
}

DecodingResult decode_soft_info_bp(
    SoftInfoBpDecoder& decoder,
    rust::Slice<const double> soft_syndrome,
    double cutoff,
    double sigma
) {
    auto bp_decoder = static_cast<bp::BpDecoder*>(decoder.bp_decoder);

    // Convert soft syndrome to std::vector
    std::vector<double> soft_syn(soft_syndrome.data(), soft_syndrome.data() + soft_syndrome.size());

    // Perform soft information decoding
    auto& decoding = bp_decoder->soft_info_decode_serial(soft_syn, cutoff, sigma);

    DecodingResult result;
    result.decoding.reserve(decoding.size());
    for (auto bit : decoding) {
        result.decoding.push_back(bit);
    }
    result.converged = bp_decoder->converge;
    result.iterations = bp_decoder->iterations;

    return result;
}

// Getter functions for Soft Info BP decoder
uint32_t get_check_count_soft(const SoftInfoBpDecoder& decoder) {
    return static_cast<const bp::BpSparse*>(decoder.pcm)->m;
}

uint32_t get_bit_count_soft(const SoftInfoBpDecoder& decoder) {
    return static_cast<const bp::BpSparse*>(decoder.pcm)->n;
}

rust::Vec<double> get_channel_probs_soft(const SoftInfoBpDecoder& decoder) {
    rust::Vec<double> result;
    result.reserve(decoder.channel_probs.size());
    for (auto prob : decoder.channel_probs) {
        result.push_back(prob);
    }
    return result;
}

int32_t get_max_iter_soft(const SoftInfoBpDecoder& decoder) {
    return decoder.max_iter;
}

int32_t get_bp_method_soft(const SoftInfoBpDecoder& decoder) {
    return decoder.bp_method;
}

double get_ms_scaling_factor_soft(const SoftInfoBpDecoder& decoder) {
    return decoder.ms_scaling_factor;
}

bool get_converged_soft(const SoftInfoBpDecoder& decoder) {
    return static_cast<const bp::BpDecoder*>(decoder.bp_decoder)->converge;
}

int32_t get_iterations_soft(const SoftInfoBpDecoder& decoder) {
    return static_cast<const bp::BpDecoder*>(decoder.bp_decoder)->iterations;
}

int32_t get_omp_thread_count_soft(const SoftInfoBpDecoder& decoder) {
    return decoder.omp_thread_count;
}

int32_t get_random_schedule_seed_soft(const SoftInfoBpDecoder& decoder) {
    return decoder.random_schedule_seed;
}

rust::Vec<double> get_log_prob_ratios_soft(const SoftInfoBpDecoder& decoder) {
    auto bp_decoder = static_cast<const bp::BpDecoder*>(decoder.bp_decoder);
    rust::Vec<double> result;
    result.reserve(bp_decoder->log_prob_ratios.size());
    for (auto llr : bp_decoder->log_prob_ratios) {
        result.push_back(llr);
    }
    return result;
}

// Flip Decoder implementation
FlipDecoder::~FlipDecoder() {
    if (pcm) delete static_cast<bp::BpSparse*>(pcm);
    if (flip_decoder) delete static_cast<flip::FlipDecoder*>(flip_decoder);
}

std::unique_ptr<FlipDecoder> create_flip_decoder(
    const SparseMatrixRepr& pcm_repr,
    int32_t max_iter,
    int32_t pfreq,
    int32_t seed
) {
    auto decoder = std::make_unique<FlipDecoder>();

    // Initialize all pointers to nullptr to avoid deleting garbage in destructor
    decoder->pcm = nullptr;
    decoder->flip_decoder = nullptr;

    // Create sparse matrix
    auto pcm = new bp::BpSparse(pcm_repr.rows, pcm_repr.cols);
    for (size_t i = 0; i < pcm_repr.row_indices.size(); ++i) {
        pcm->insert_entry(pcm_repr.row_indices[i], pcm_repr.col_indices[i]);
    }
    decoder->pcm = pcm;

    // Store parameters
    decoder->max_iter = max_iter;
    decoder->pfreq = pfreq;
    decoder->seed = seed;

    // Create flip decoder
    decoder->flip_decoder = new flip::FlipDecoder(*pcm, max_iter, pfreq, seed);

    return decoder;
}

DecodingResult decode_flip(
    FlipDecoder& decoder,
    rust::Slice<const uint8_t> syndrome
) {
    auto flip_decoder = static_cast<flip::FlipDecoder*>(decoder.flip_decoder);

    // Convert syndrome to std::vector
    std::vector<uint8_t> synd(syndrome.data(), syndrome.data() + syndrome.size());

    // Perform decoding
    auto& decoding = flip_decoder->decode(synd);

    DecodingResult result;
    result.decoding.reserve(decoding.size());
    for (auto bit : decoding) {
        result.decoding.push_back(bit);
    }
    result.converged = flip_decoder->converge;
    result.iterations = flip_decoder->iterations;

    return result;
}

// Getter functions for Flip decoder
uint32_t get_check_count_flip(const FlipDecoder& decoder) {
    return static_cast<const bp::BpSparse*>(decoder.pcm)->m;
}

uint32_t get_bit_count_flip(const FlipDecoder& decoder) {
    return static_cast<const bp::BpSparse*>(decoder.pcm)->n;
}

int32_t get_max_iter_flip(const FlipDecoder& decoder) {
    return decoder.max_iter;
}

bool get_converged_flip(const FlipDecoder& decoder) {
    return static_cast<const flip::FlipDecoder*>(decoder.flip_decoder)->converge;
}

int32_t get_iterations_flip(const FlipDecoder& decoder) {
    return static_cast<const flip::FlipDecoder*>(decoder.flip_decoder)->iterations;
}

// Union Find Decoder implementation
UnionFindDecoder::~UnionFindDecoder() {
    if (pcm) delete static_cast<bp::BpSparse*>(pcm);
    if (uf_decoder) delete static_cast<ldpc::uf::UfDecoder*>(uf_decoder);
}

std::unique_ptr<UnionFindDecoder> create_union_find_decoder(
    const SparseMatrixRepr& pcm_repr,
    int32_t uf_method
) {
    auto decoder = std::make_unique<UnionFindDecoder>();

    // Initialize all pointers to nullptr to avoid deleting garbage in destructor
    decoder->pcm = nullptr;
    decoder->uf_decoder = nullptr;

    // Create sparse matrix
    auto pcm = new bp::BpSparse(pcm_repr.rows, pcm_repr.cols);
    for (size_t i = 0; i < pcm_repr.row_indices.size(); ++i) {
        pcm->insert_entry(pcm_repr.row_indices[i], pcm_repr.col_indices[i]);
    }
    decoder->pcm = pcm;

    // Store parameters
    decoder->uf_method = uf_method;

    // Create UF decoder
    decoder->uf_decoder = new uf::UfDecoder(*pcm);

    return decoder;
}

DecodingResult decode_union_find(
    UnionFindDecoder& decoder,
    rust::Slice<const uint8_t> syndrome,
    rust::Slice<const double> llrs,
    int32_t bits_per_step
) {
    auto uf_decoder = static_cast<uf::UfDecoder*>(decoder.uf_decoder);

    // Convert syndrome to std::vector
    std::vector<uint8_t> synd(syndrome.data(), syndrome.data() + syndrome.size());

    // Convert LLRs to std::vector (can be empty)
    std::vector<double> llr_vec(llrs.data(), llrs.data() + llrs.size());

    // Perform decoding based on method
    std::vector<uint8_t>* decoding_ptr;
    if (decoder.uf_method == 0) { // Inversion method
        decoding_ptr = &uf_decoder->matrix_decode(synd, llr_vec, bits_per_step);
    } else { // Peeling method
        decoding_ptr = &uf_decoder->peel_decode(synd, llr_vec, bits_per_step);
    }
    auto& decoding = *decoding_ptr;

    DecodingResult result;
    result.decoding.reserve(decoding.size());
    for (auto bit : decoding) {
        result.decoding.push_back(bit);
    }
    result.converged = true; // UF decoder doesn't have a converge flag
    result.iterations = 1;   // UF decoder doesn't track iterations

    return result;
}

// Getter functions for Union Find decoder
uint32_t get_check_count_uf(const UnionFindDecoder& decoder) {
    return static_cast<const bp::BpSparse*>(decoder.pcm)->m;
}

uint32_t get_bit_count_uf(const UnionFindDecoder& decoder) {
    return static_cast<const bp::BpSparse*>(decoder.pcm)->n;
}

// MBP Decoder implementation
MbpDecoder::~MbpDecoder() {
    // The mbp_decoder owns the pcm, so it will delete it
    if (mbp_decoder) delete static_cast<::mbp_decoder*>(mbp_decoder);
    if (pcmx) delete static_cast<bp::BpSparse*>(pcmx);
    if (pcmz) delete static_cast<bp::BpSparse*>(pcmz);
}

std::unique_ptr<MbpDecoder> create_mbp_decoder(
    const SparseMatrixRepr& hx,
    const SparseMatrixRepr& hz,
    double error_rate,
    rust::Slice<const double> xyz_bias,
    int32_t max_iter,
    int32_t bp_method,
    double ms_scaling_factor,
    int32_t omp_thread_count
) {
    if (hx.cols != hz.cols) {
        throw std::runtime_error("HX and HZ must have the same number of columns (qubits)");
    }

    if (xyz_bias.size() != 3) {
        throw std::runtime_error("xyz_bias must have exactly 3 elements");
    }

    auto decoder = std::make_unique<MbpDecoder>();

    // Initialize all pointers to nullptr to avoid deleting garbage in destructor
    decoder->pcm = nullptr;
    decoder->pcmx = nullptr;
    decoder->pcmz = nullptr;
    decoder->mbp_decoder = nullptr;

    // Store sizes
    decoder->qubit_count = hx.cols;
    decoder->stab_count = hx.rows + hz.rows;

    // Create HX and HZ matrices
    decoder->pcmx = create_pcm_from_sparse(hx);
    decoder->pcmz = create_pcm_from_sparse(hz);

    // Create GF(4) parity check matrix
    // HZ checks come first with value 3 (Z stabilizers)
    // HX checks come after with value 1 (X stabilizers)
    auto pcm_gf4 = new mbp_sparse(decoder->stab_count, decoder->qubit_count);

    // Add Z stabilizers (value 3)
    for (size_t i = 0; i < hz.row_indices.size(); i++) {
        pcm_gf4->insert_entry(hz.row_indices[i], hz.col_indices[i], 3);
    }

    // Add X stabilizers (value 1)
    for (size_t i = 0; i < hx.row_indices.size(); i++) {
        pcm_gf4->insert_entry(hx.row_indices[i] + hz.rows, hx.col_indices[i], 1);
    }

    decoder->pcm = pcm_gf4;

    // Create error channel (3 x n matrix for X, Y, Z errors)
    std::vector<std::vector<double>> error_channel(3);
    for (int i = 0; i < 3; i++) {
        error_channel[i].resize(decoder->qubit_count);
        for (int j = 0; j < decoder->qubit_count; j++) {
            error_channel[i][j] = error_rate * xyz_bias[i];
        }
    }

    // Create alpha parameter (3 x n matrix, default to all 1s)
    std::vector<std::vector<double>> alpha(3);
    for (int i = 0; i < 3; i++) {
        alpha[i].resize(decoder->qubit_count, 1.0);
    }

    // Create MBP decoder
    decoder->mbp_decoder = new mbp_decoder(
        pcm_gf4,
        error_channel,
        max_iter == 0 ? decoder->qubit_count : max_iter,
        alpha,
        0.0,  // beta parameter
        bp_method,
        ms_scaling_factor
    );

    // Store parameters
    decoder->max_iter = max_iter;
    decoder->bp_method = bp_method;
    decoder->ms_scaling_factor = ms_scaling_factor;

    return decoder;
}

DecodingResult decode_mbp(
    MbpDecoder& decoder,
    rust::Slice<const uint8_t> syndrome
) {
    auto mbp = static_cast<mbp_decoder*>(decoder.mbp_decoder);

    if (syndrome.size() != decoder.stab_count) {
        throw std::runtime_error("Syndrome length must match number of stabilizers");
    }

    // Convert syndrome to vector
    std::vector<uint8_t> synd(syndrome.data(), syndrome.data() + syndrome.size());

    // Decode - returns GF(4) decoding
    auto& gf4_decoding = mbp->decode(synd);

    DecodingResult result;
    result.converged = mbp->converge;
    result.iterations = mbp->iterations;

    // Convert GF(4) to binary (just return the GF(4) values for now)
    // In practice, you might want to convert to X and Z components
    result.decoding = rust::Vec<uint8_t>();
    result.decoding.reserve(gf4_decoding.size());
    for (auto val : gf4_decoding) {
        result.decoding.push_back(val);
    }

    return result;
}

// Getter functions for MBP decoder
uint32_t get_check_count_mbp(const MbpDecoder& decoder) {
    return decoder.stab_count;
}

uint32_t get_bit_count_mbp(const MbpDecoder& decoder) {
    return decoder.qubit_count;
}

int32_t get_max_iter_mbp(const MbpDecoder& decoder) {
    return decoder.max_iter;
}

bool get_converged_mbp(const MbpDecoder& decoder) {
    return static_cast<const mbp_decoder*>(decoder.mbp_decoder)->converge;
}

int32_t get_iterations_mbp(const MbpDecoder& decoder) {
    return static_cast<const mbp_decoder*>(decoder.mbp_decoder)->iterations;
}
