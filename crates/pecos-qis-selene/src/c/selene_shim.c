/**
 * PECOS Selene Runtime Shim
 *
 * This C library implements the selene_* API and forwards calls to PECOS's
 * thread-local QIS interface for operation collection.
 *
 * Architecture:
 *   program.x → ___qalloc() [from libhelios.a]
 *             → selene_qalloc() [from this shim]
 *             → pecos_collect_operation() [calls Rust FFI]
 *             → pecos_qis_interface::with_interface()
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <inttypes.h>  // For portable format specifiers

// Selene API types (matching selene.h)
typedef struct SeleneInstance {
    int dummy;  // Opaque struct - we don't use it
} SeleneInstance;

typedef struct {
    uint32_t error_code;
} selene_void_result_t;

typedef struct {
    uint32_t error_code;
    uint64_t value;
} selene_u64_result_t;

typedef struct {
    uint32_t error_code;
    uint32_t value;
} selene_u32_result_t;

typedef struct {
    uint32_t error_code;
    double value;
} selene_f64_result_t;

typedef struct {
    uint32_t error_code;
    bool value;
} selene_bool_result_t;

typedef struct {
    uint32_t error_code;
    uint64_t reference;
} selene_future_result_t;

typedef struct {
    const char *data;
    uint64_t length;
    bool owned;
} selene_string_t;

// =============================================================================
// Forward declarations of PECOS FFI functions
// These will be provided by the Rust pecos-qis-interface crate
// =============================================================================

// On Windows, functions imported from DLLs need __declspec(dllimport)
// On Unix, no special declaration is needed
#ifdef _WIN32
#define IMPORT_API __declspec(dllimport)
#else
#define IMPORT_API
#endif

// These functions are implemented in pecos-qis-interface/src/ffi.rs
// and exported with #[unsafe(no_mangle)]
IMPORT_API extern int64_t __quantum__rt__qubit_allocate(void);
IMPORT_API extern void __quantum__rt__qubit_release(int64_t qubit);
IMPORT_API extern void __quantum__qis__rxy__body(double theta, double phi, int64_t qubit);
IMPORT_API extern void __quantum__qis__rz__body(double theta, int64_t qubit);
IMPORT_API extern void __quantum__qis__rzz__body(double theta, int64_t qubit1, int64_t qubit2);
IMPORT_API extern void __quantum__qis__reset__body(int64_t qubit);
IMPORT_API extern int32_t __quantum__qis__m__body(int64_t qubit, int64_t result);
IMPORT_API extern int64_t __quantum__rt__result_allocate(void);

// Note: For lazy measurement and future operations, we need special handling
// since PECOS doesn't have native support yet. For now, we'll use placeholders.

// =============================================================================
// Export macros for cross-platform DLL symbol visibility
// =============================================================================

// On Windows, we need __declspec(dllexport) to make symbols visible in DLLs
// On Unix, we use __attribute__((visibility("default"))) with -fvisibility=hidden
#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// =============================================================================
// Helper macros
// =============================================================================

#define SUCCESS(type) ((type){.error_code = 0})
#define SUCCESS_VAL(type, val) ((type){.error_code = 0, .value = val})
#define SUCCESS_REF(type, ref) ((type){.error_code = 0, .reference = ref})

// =============================================================================
// Qubit allocation and deallocation
// =============================================================================

EXPORT_API selene_u64_result_t selene_qalloc(SeleneInstance *instance) {
    (void)instance;  // Unused - we use thread-local storage
    fprintf(stderr, "[SHIM] selene_qalloc() called\n");
    fflush(stderr);
    int64_t qubit_id = __quantum__rt__qubit_allocate();
    fprintf(stderr, "[SHIM] __quantum__rt__qubit_allocate() returned: %" PRId64 "\n", qubit_id);
    fflush(stderr);

    // Check if allocation failed (negative values indicate errors in some implementations)
    if (qubit_id < 0) {
        fprintf(stderr, "[SHIM] ERROR: Qubit allocation failed with id: %" PRId64 ", returning error 100000\n", qubit_id);
        fflush(stderr);
        return (selene_u64_result_t){.error_code = 100000, .value = 0};
    }

    selene_u64_result_t result = SUCCESS_VAL(selene_u64_result_t, (uint64_t)qubit_id);
    fprintf(stderr, "[SHIM] selene_qalloc() returning success with value: %" PRIu64 ", error_code: %u\n",
            result.value, result.error_code);
    fflush(stderr);
    return result;
}

EXPORT_API selene_void_result_t selene_qfree(SeleneInstance *instance, uint64_t q) {
    (void)instance;
    __quantum__rt__qubit_release((int64_t)q);
    return SUCCESS(selene_void_result_t);
}

// =============================================================================
// Quantum gates
// =============================================================================

EXPORT_API selene_void_result_t selene_rxy(SeleneInstance *instance, uint64_t q, double theta, double phi) {
    (void)instance;
    // Note: pecos-qis-interface uses r1xy which takes (theta, phi, qubit)
    // We need to check the signature - looking at ffi.rs it's:
    // pub unsafe extern "C" fn __quantum__qis__r1xy__body(theta: f64, phi: f64, qubit: i64)
    IMPORT_API extern void __quantum__qis__r1xy__body(double theta, double phi, int64_t qubit);
    __quantum__qis__r1xy__body(theta, phi, (int64_t)q);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_rz(SeleneInstance *instance, uint64_t q, double theta) {
    (void)instance;
    __quantum__qis__rz__body(theta, (int64_t)q);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_rzz(SeleneInstance *instance, uint64_t q1, uint64_t q2, double theta) {
    (void)instance;
    __quantum__qis__rzz__body(theta, (int64_t)q1, (int64_t)q2);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_qubit_reset(SeleneInstance *instance, uint64_t q) {
    (void)instance;
    __quantum__qis__reset__body((int64_t)q);
    return SUCCESS(selene_void_result_t);
}

// =============================================================================
// Measurement
// =============================================================================

EXPORT_API selene_bool_result_t selene_qubit_measure(SeleneInstance *instance, uint64_t q) {
    (void)instance;
    // For immediate measurement, we allocate a result and measure
    int64_t result_id = __quantum__rt__result_allocate();
    int32_t result = __quantum__qis__m__body((int64_t)q, result_id);
    return (selene_bool_result_t){.error_code = 0, .value = (bool)result};
}

EXPORT_API selene_future_result_t selene_qubit_lazy_measure(SeleneInstance *instance, uint64_t q) {
    (void)instance;
    // For lazy measurement, we allocate a result ID and queue the measurement
    // The actual measurement result will be retrieved later
    int64_t result_id = __quantum__rt__result_allocate();
    __quantum__qis__m__body((int64_t)q, result_id);
    // Return the result ID as the future reference
    return SUCCESS_REF(selene_future_result_t, (uint64_t)result_id);
}

EXPORT_API selene_future_result_t selene_qubit_lazy_measure_leaked(SeleneInstance *instance, uint64_t q) {
    // Same as lazy_measure for now
    return selene_qubit_lazy_measure(instance, q);
}

// =============================================================================
// Future operations
// =============================================================================

EXPORT_API selene_bool_result_t selene_future_read_bool(SeleneInstance *instance, uint64_t r) {
    (void)instance;
    // Read the measurement result
    // We need a function to retrieve stored results
    IMPORT_API extern int32_t __quantum__rt__result_get_one(int64_t result);
    int32_t value = __quantum__rt__result_get_one((int64_t)r);
    return (selene_bool_result_t){.error_code = 0, .value = (bool)value};
}

EXPORT_API selene_u64_result_t selene_future_read_u64(SeleneInstance *instance, uint64_t r) {
    (void)instance;
    // For now, treat as bool and convert to u64
    IMPORT_API extern int32_t __quantum__rt__result_get_one(int64_t result);
    int32_t value = __quantum__rt__result_get_one((int64_t)r);
    return SUCCESS_VAL(selene_u64_result_t, (uint64_t)value);
}

EXPORT_API selene_void_result_t selene_refcount_increment(SeleneInstance *instance, uint64_t r) {
    (void)instance;
    (void)r;
    // No-op for PECOS - we don't do refcounting
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_refcount_decrement(SeleneInstance *instance, uint64_t r) {
    (void)instance;
    (void)r;
    // No-op for PECOS - we don't do refcounting
    return SUCCESS(selene_void_result_t);
}

// =============================================================================
// Print operations (for debug/output)
// =============================================================================

EXPORT_API selene_void_result_t selene_print_bool(SeleneInstance *instance, selene_string_t tag, bool value) {
    (void)instance;
    // Use the print_bool FFI function if available
    // Signature: pub unsafe extern "C" fn print_bool(label_ptr: *const u8, label_len: i64, value: bool)
    IMPORT_API extern void print_bool(const uint8_t *label_ptr, int64_t label_len, bool value);
    print_bool((const uint8_t*)tag.data, (int64_t)tag.length, value);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_i64(SeleneInstance *instance, selene_string_t tag, int64_t value) {
    (void)instance;
    printf("%.*s: %" PRId64 "\n", (int)tag.length, tag.data, value);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_u64(SeleneInstance *instance, selene_string_t tag, uint64_t value) {
    (void)instance;
    printf("%.*s: %" PRIu64 "\n", (int)tag.length, tag.data, value);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_f64(SeleneInstance *instance, selene_string_t tag, double value) {
    (void)instance;
    printf("%.*s: %f\n", (int)tag.length, tag.data, value);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_bool_array(SeleneInstance *instance, selene_string_t tag,
                                             const bool *ptr, uint64_t length) {
    (void)instance;
    printf("%.*s: [", (int)tag.length, tag.data);
    for (uint64_t i = 0; i < length; i++) {
        printf("%s%s", ptr[i] ? "true" : "false", (i < length - 1) ? ", " : "");
    }
    printf("]\n");
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_i64_array(SeleneInstance *instance, selene_string_t tag,
                                            const int64_t *ptr, uint64_t length) {
    (void)instance;
    printf("%.*s: [", (int)tag.length, tag.data);
    for (uint64_t i = 0; i < length; i++) {
        printf("%" PRId64 "%s", ptr[i], (i < length - 1) ? ", " : "");
    }
    printf("]\n");
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_u64_array(SeleneInstance *instance, selene_string_t tag,
                                            const uint64_t *ptr, uint64_t length) {
    (void)instance;
    printf("%.*s: [", (int)tag.length, tag.data);
    for (uint64_t i = 0; i < length; i++) {
        printf("%" PRIu64 "%s", ptr[i], (i < length - 1) ? ", " : "");
    }
    printf("]\n");
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_f64_array(SeleneInstance *instance, selene_string_t tag,
                                            const double *ptr, uint64_t length) {
    (void)instance;
    printf("%.*s: [", (int)tag.length, tag.data);
    for (uint64_t i = 0; i < length; i++) {
        printf("%f%s", ptr[i], (i < length - 1) ? ", " : "");
    }
    printf("]\n");
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_panic(SeleneInstance *instance, selene_string_t message,
                                       uint32_t error_code) {
    (void)instance;
    fprintf(stderr, "[SHIM] selene_print_panic() called with error_code=%u\n", error_code);
    fprintf(stderr, "PANIC [%u]: %.*s\n", error_code, (int)message.length, message.data);
    fflush(stderr);
    return SUCCESS(selene_void_result_t);
}

// =============================================================================
// Stub implementations for functions we don't need yet
// =============================================================================

EXPORT_API selene_void_result_t selene_dump_state(SeleneInstance *instance, selene_string_t message,
                                       const uint64_t *qubits, uint64_t qubits_length) {
    (void)instance; (void)message; (void)qubits; (void)qubits_length;
    // No-op - state dumping not supported in operation collection mode
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_set_tc(SeleneInstance *instance, uint64_t time_cursor) {
    fprintf(stderr, "[SHIM] !!!!! selene_set_tc(%" PRIu64 ") called !!!!!\n", time_cursor);
    fflush(stderr);
    (void)instance; (void)time_cursor;
    // No-op - time cursor not used
    fprintf(stderr, "[SHIM] selene_set_tc returning SUCCESS\n");
    fflush(stderr);
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_u64_result_t selene_get_tc(SeleneInstance *instance) {
    fprintf(stderr, "[SHIM] selene_get_tc() called\n");
    fflush(stderr);
    (void)instance;
    return SUCCESS_VAL(selene_u64_result_t, 0);
}

EXPORT_API selene_u64_result_t selene_get_current_shot(SeleneInstance *instance) {
    (void)instance;
    return SUCCESS_VAL(selene_u64_result_t, 0);
}

EXPORT_API selene_void_result_t selene_local_barrier(SeleneInstance *instance, const uint64_t *qubit_ids,
                                         uint64_t qubit_ids_length, uint64_t sleep_time) {
    (void)instance; (void)qubit_ids; (void)qubit_ids_length; (void)sleep_time;
    // No-op - barriers not needed for operation collection
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_global_barrier(SeleneInstance *instance, uint64_t sleep_time) {
    (void)instance; (void)sleep_time;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_u64_result_t selene_shot_count(SeleneInstance *instance) {
    (void)instance;
    // Return 1 shot for operation collection mode
    return SUCCESS_VAL(selene_u64_result_t, 1);
}

EXPORT_API selene_void_result_t selene_on_shot_start(SeleneInstance *instance, uint64_t shot_index) {
    (void)instance; (void)shot_index;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_on_shot_end(SeleneInstance *instance) {
    (void)instance;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_load_config(SeleneInstance **instance, const char *config_file) {
    (void)config_file;
    // Return a dummy instance pointer - we don't actually use it
    static SeleneInstance dummy_instance;
    *instance = &dummy_instance;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_exit(SeleneInstance *instance) {
    (void)instance;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_print_exit(SeleneInstance *instance, selene_string_t message,
                                      uint32_t error_code) {
    (void)instance;
    fprintf(stderr, "EXIT [%u]: %.*s\n", error_code, (int)message.length, message.data);
    return SUCCESS(selene_void_result_t);
}

// Random number generation stubs
EXPORT_API selene_void_result_t selene_random_seed(SeleneInstance *instance, uint64_t seed) {
    (void)instance; (void)seed;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_void_result_t selene_random_advance(SeleneInstance *instance, uint64_t delta) {
    (void)instance; (void)delta;
    return SUCCESS(selene_void_result_t);
}

EXPORT_API selene_u32_result_t selene_random_u32(SeleneInstance *instance) {
    (void)instance;
    return (selene_u32_result_t){.error_code = 0, .value = 0};
}

EXPORT_API selene_u32_result_t selene_random_u32_bounded(SeleneInstance *instance, uint32_t bound) {
    (void)instance; (void)bound;
    return (selene_u32_result_t){.error_code = 0, .value = 0};
}

EXPORT_API selene_f64_result_t selene_random_f64(SeleneInstance *instance) {
    (void)instance;
    return (selene_f64_result_t){.error_code = 0, .value = 0.0};
}

EXPORT_API selene_u64_result_t selene_custom_runtime_call(SeleneInstance *instance, uint64_t tag,
                                               const uint8_t *data, uint64_t data_length) {
    (void)instance; (void)tag; (void)data; (void)data_length;
    return SUCCESS_VAL(selene_u64_result_t, 0);
}

// =============================================================================
// In-process execution support with setjmp/longjmp
// =============================================================================

// This is the jump buffer used by Helios's interface.c
// We DEFINE it here (not extern) so it's available when program.so is loaded.
// The program.so will have an `extern jmp_buf user_program_jmpbuf` declaration
// that will resolve to this definition when loaded with RTLD_GLOBAL.
jmp_buf user_program_jmpbuf;

/**
 * Wrapper function to safely call qmain with setjmp/longjmp support
 *
 * This function sets up the exception handling mechanism that Helios expects:
 * 1. Calls setjmp to save the current stack state
 * 2. Calls qmain(0) to execute the quantum program
 * 3. If an error occurs and longjmp is called, we catch it and return the error code
 *
 * Returns: 0 on success, error code on failure
 */
typedef uint64_t (*qmain_fn_t)(uint64_t);

EXPORT_API uint64_t pecos_call_qmain_with_setjmp(qmain_fn_t qmain) {
    fprintf(stderr, "[SHIM] Setting up setjmp before calling qmain...\n");
    fflush(stderr);

    // Initialize shot context to match what interface.c main() does
    // This might be required for proper execution
    static SeleneInstance dummy_instance;
    fprintf(stderr, "[SHIM] Calling selene_on_shot_start(dummy, 0)...\n");
    fflush(stderr);
    selene_void_result_t start_result = selene_on_shot_start(&dummy_instance, 0);
    if (start_result.error_code != 0) {
        fprintf(stderr, "[SHIM] selene_on_shot_start failed with error: %u\n", start_result.error_code);
        fflush(stderr);
        return start_result.error_code;
    }

    int error_code = setjmp(user_program_jmpbuf);
    if (error_code == 0) {
        // Normal path - call qmain
        fprintf(stderr, "[SHIM] setjmp complete, calling qmain(0)...\n");
        fflush(stderr);
        uint64_t result = qmain(0);
        fprintf(stderr, "[SHIM] qmain returned successfully: %" PRIu64 "\n", result);
        fflush(stderr);

        // Clean up shot context
        fprintf(stderr, "[SHIM] Calling selene_on_shot_end...\n");
        fflush(stderr);
        selene_void_result_t end_result = selene_on_shot_end(&dummy_instance);
        if (end_result.error_code != 0) {
            fprintf(stderr, "[SHIM] selene_on_shot_end failed with error: %u\n", end_result.error_code);
        }

        return result;
    } else {
        // longjmp was called - an error occurred
        fprintf(stderr, "[SHIM] longjmp caught error code: %d (0x%X)\n", error_code, error_code);
        fflush(stderr);

        // Clean up even on error
        selene_on_shot_end(&dummy_instance);

        if (error_code < 1000) {
            // Recoverable error - return 0 but log it
            fprintf(stderr, "[SHIM] Recoverable error, continuing\n");
            fflush(stderr);
            return 0;
        } else {
            // Fatal error - return error code
            fprintf(stderr, "[SHIM] Fatal error: %d\n", error_code);
            fflush(stderr);
            return (uint64_t)error_code;
        }
    }
}
