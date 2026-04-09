/*
 * PECOS Foreign Plugin Interface
 *
 * C header for implementing PECOS decoders and simulators in any language
 * that can produce C-compatible function pointers.
 *
 * Link against: libpecos_ffi.so (Linux), libpecos_ffi.dylib (macOS), pecos_ffi.dll (Windows)
 * Build with:   cargo build -p pecos-ffi --release
 *
 * Copyright 2026 The PECOS Developers
 * Licensed under Apache-2.0
 */

#ifndef PECOS_FOREIGN_H
#define PECOS_FOREIGN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version constants. Set these as the `version` field of each vtable.
 * PECOS checks the version on construction and rejects mismatches. */
#define PECOS_DECODER_VTABLE_VERSION 1
#define PECOS_SIMULATOR_VTABLE_VERSION 1

/* =========================================================================
 * Decoder Plugin Interface
 * ========================================================================= */

/**
 * Result of a decode operation, filled by the foreign decoder.
 *
 * On success (return code 0):
 *   - observable_ptr/observable_len: the decoded observable vector
 *   - weight: cost of the solution
 *   - converged: 0=false, 1=true, -1=unknown
 *   - error_ptr must be NULL
 *
 * On error (return code != 0):
 *   - error_ptr/error_len: UTF-8 error message (not null-terminated)
 *   - observable_ptr must be NULL
 */
typedef struct {
    uint8_t *observable_ptr;
    size_t   observable_len;
    double   weight;
    int8_t   converged;
    const uint8_t *error_ptr;
    size_t   error_len;
} PecosDecodingResultRaw;

/**
 * VTable for a foreign decoder plugin.
 *
 * Fill this struct with function pointers and pass it to PECOS along with
 * an opaque handle to your decoder instance.
 */
typedef struct {
    /** ABI version. Must equal PECOS_DECODER_VTABLE_VERSION. */
    uint32_t version;

    /**
     * Decode a syndrome.
     *
     * @param handle      Opaque decoder handle
     * @param input_ptr   Pointer to syndrome bytes
     * @param input_len   Number of syndrome bytes
     * @param result_out  Caller-allocated result struct to fill
     * @return 0 on success, non-zero on error
     */
    int32_t (*decode)(void *handle, const uint8_t *input_ptr, size_t input_len,
                      PecosDecodingResultRaw *result_out);

    /** Return the number of checks (parity check matrix rows). */
    size_t (*check_count)(const void *handle);

    /** Return the number of bits (parity check matrix columns). */
    size_t (*bit_count)(const void *handle);

    /** Free the observable array from a result. NULL ptr must be a no-op. */
    void (*free_result)(uint8_t *ptr, size_t len);

    /** Free an error message string. NULL ptr must be a no-op. */
    void (*free_error)(const uint8_t *ptr, size_t len);

    /** Destroy the decoder. Called once on cleanup. NULL handle must be a no-op. */
    void (*destroy)(void *handle);
} PecosDecoderVTable;

/* =========================================================================
 * Simulator Plugin Interface
 * ========================================================================= */

/**
 * Measurement result from a Z-basis measurement.
 */
typedef struct {
    uint8_t outcome;          /* 0 = |0>, 1 = |1> */
    uint8_t is_deterministic; /* 0 = random, 1 = deterministic */
} PecosMeasurementResult;

/**
 * VTable for a foreign simulator plugin.
 *
 * Required gates (Clifford): sz, h, cx, mz
 * Optional gates (rotations): rx, rz, rzz -- set to NULL for Clifford-only simulators
 *
 * Qubit indices are size_t values. Two-qubit gates receive interleaved pairs:
 * [control0, target0, control1, target1, ...] with num_pairs giving the pair count.
 */
typedef struct {
    /** ABI version. Must equal PECOS_SIMULATOR_VTABLE_VERSION. */
    uint32_t version;

    /* -- Clifford gates (required) -- */

    /** Apply S (sqrt-Z) gate to each qubit. */
    void (*sz)(void *handle, const size_t *qubits, size_t num_qubits);

    /** Apply Hadamard gate to each qubit. */
    void (*h)(void *handle, const size_t *qubits, size_t num_qubits);

    /** Apply CNOT to each (control, target) pair. */
    void (*cx)(void *handle, const size_t *pairs, size_t num_pairs);

    /**
     * Measure qubits in the Z basis.
     * Write results into results_out (caller-allocated, length = num_qubits).
     */
    void (*mz)(void *handle, const size_t *qubits, size_t num_qubits,
               PecosMeasurementResult *results_out);

    /* -- Rotation gates (optional, NULL for Clifford-only) -- */

    /** Apply RX(theta) to each qubit. theta is in radians. May be NULL. */
    void (*rx)(void *handle, double theta, const size_t *qubits, size_t num_qubits);

    /** Apply RZ(theta) to each qubit. theta is in radians. May be NULL. */
    void (*rz)(void *handle, double theta, const size_t *qubits, size_t num_qubits);

    /** Apply RZZ(theta) to each pair. theta is in radians. May be NULL. */
    void (*rzz)(void *handle, double theta, const size_t *pairs, size_t num_pairs);

    /* -- Lifecycle -- */

    /** Reset the simulator to initial state (all qubits to |0>). */
    void (*reset)(void *handle);

    /** Set the RNG seed for reproducibility. May be NULL. */
    void (*set_seed)(void *handle, uint64_t seed);

    /** Destroy the simulator and free all resources. NULL handle must be a no-op. */
    void (*destroy)(void *handle);
} PecosSimulatorVTable;

/* =========================================================================
 * Plugin Discovery Protocol
 *
 * To make a discoverable plugin, compile a shared library that exports
 * `pecos_plugin_init`. Place the .so/.dylib/.dll in ~/.pecos/plugins/.
 * PECOS will load it automatically.
 * ========================================================================= */

#define PECOS_PLUGIN_API_VERSION 1

/**
 * Descriptor filled by a plugin's init function.
 * Set unused fields to NULL.
 */
typedef struct {
    const char *name;                       /**< Plugin name (static string) */
    uint32_t plugin_api_version;            /**< Must be PECOS_PLUGIN_API_VERSION */
    void *decoder_handle;                   /**< Opaque decoder state, or NULL */
    const PecosDecoderVTable *decoder_vtable; /**< Decoder vtable, or NULL */
    void *simulator_handle;                 /**< Opaque simulator state, or NULL */
    const PecosSimulatorVTable *simulator_vtable; /**< Simulator vtable, or NULL */
} PecosPluginDescriptor;

/**
 * Plugin entry point. Every discoverable plugin must export this function.
 *
 * Fill `desc` with the plugin's capabilities. Return 0 on success, non-zero on error.
 */
/* int pecos_plugin_init(PecosPluginDescriptor *desc); */

/* =========================================================================
 * Bridge functions (provided by libpecos_ffi)
 * ========================================================================= */

/* -- Version queries -- */
uint32_t pecos_decoder_vtable_version(void);
uint32_t pecos_simulator_vtable_version(void);

/* -- Decoder lifecycle -- */
typedef struct PecosDecoder PecosDecoder; /* opaque */
PecosDecoder *pecos_foreign_decoder_create(void *handle, const PecosDecoderVTable *vtable);
size_t pecos_foreign_decoder_check_count(const PecosDecoder *decoder);
size_t pecos_foreign_decoder_bit_count(const PecosDecoder *decoder);
int32_t pecos_foreign_decoder_decode(PecosDecoder *decoder, const uint8_t *input, size_t len,
                                     PecosDecodingResultRaw *result_out);
void pecos_foreign_decoder_free_observable(uint8_t *ptr, size_t len);
void pecos_foreign_decoder_free_error(const uint8_t *ptr, size_t len);
void pecos_foreign_decoder_free(PecosDecoder *decoder);

/* -- Simulator lifecycle -- */
typedef struct PecosSimulator PecosSimulator; /* opaque */
PecosSimulator *pecos_foreign_simulator_create(void *handle, const PecosSimulatorVTable *vtable);
_Bool pecos_foreign_simulator_supports_rotations(const PecosSimulator *sim);
void pecos_foreign_simulator_free(PecosSimulator *sim);

/* -- Engine lifecycle -- */
typedef struct PecosEngine PecosEngine; /* opaque */
PecosEngine *pecos_engine_create(const char *engine_type, size_t num_qubits, uint64_t seed);
int32_t pecos_engine_process(PecosEngine *engine, const uint8_t *input, size_t input_len,
                             uint8_t **output, size_t *output_len);
int32_t pecos_engine_reset(PecosEngine *engine);
void pecos_engine_free(PecosEngine *engine);
void pecos_free_bytes(uint8_t *ptr, size_t len);

/* -- Circuit builder -- */
typedef struct PecosCircuitBuilder PecosCircuitBuilder; /* opaque */
PecosCircuitBuilder *pecos_circuit_new(void);
void pecos_circuit_h(PecosCircuitBuilder *c, const size_t *qubits, size_t n);
void pecos_circuit_x(PecosCircuitBuilder *c, const size_t *qubits, size_t n);
void pecos_circuit_z(PecosCircuitBuilder *c, const size_t *qubits, size_t n);
void pecos_circuit_sz(PecosCircuitBuilder *c, const size_t *qubits, size_t n);
void pecos_circuit_cx(PecosCircuitBuilder *c, const size_t *pairs, size_t num_pairs);
void pecos_circuit_rx(PecosCircuitBuilder *c, double theta, const size_t *qubits, size_t n);
void pecos_circuit_rz(PecosCircuitBuilder *c, double theta, const size_t *qubits, size_t n);
void pecos_circuit_rzz(PecosCircuitBuilder *c, double theta, const size_t *pairs, size_t num_pairs);
void pecos_circuit_mz(PecosCircuitBuilder *c, const size_t *qubits, size_t n);
void pecos_circuit_build(PecosCircuitBuilder *c, uint8_t **output, size_t *output_len);
void pecos_circuit_reset(PecosCircuitBuilder *c);
void pecos_circuit_free(PecosCircuitBuilder *c);

/* -- Result parsing -- */
int32_t pecos_parse_outcomes(const uint8_t *output, size_t output_len,
                             uint32_t **outcomes, size_t *num_outcomes);
void pecos_free_outcomes(uint32_t *ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* PECOS_FOREIGN_H */
