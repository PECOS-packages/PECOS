// Independent C declarations of the public Selene ABI.
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

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


extern selene_u64_result_t selene_qalloc(SeleneInstance *instance);
extern selene_void_result_t selene_qfree(SeleneInstance *instance, uint64_t q);
extern selene_void_result_t selene_rxy(SeleneInstance *instance, uint64_t q, double theta, double phi);
extern selene_void_result_t selene_rz(SeleneInstance *instance, uint64_t q, double theta);
extern selene_void_result_t selene_rzz(SeleneInstance *instance, uint64_t q1, uint64_t q2, double theta);
extern selene_void_result_t selene_qubit_reset(SeleneInstance *instance, uint64_t q);
extern selene_bool_result_t selene_qubit_measure(SeleneInstance *instance, uint64_t q);
extern selene_future_result_t selene_qubit_lazy_measure(SeleneInstance *instance, uint64_t q);
extern selene_future_result_t selene_qubit_lazy_measure_leaked(SeleneInstance *instance, uint64_t q);
extern selene_bool_result_t selene_future_read_bool(SeleneInstance *instance, uint64_t r);
extern selene_u64_result_t selene_future_read_u64(SeleneInstance *instance, uint64_t r);
extern selene_void_result_t selene_refcount_increment(SeleneInstance *instance, uint64_t r);
extern selene_void_result_t selene_refcount_decrement(SeleneInstance *instance, uint64_t r);
extern selene_void_result_t selene_print_bool(SeleneInstance *instance, selene_string_t tag, bool value);
extern selene_void_result_t selene_print_i64(SeleneInstance *instance, selene_string_t tag, int64_t value);
extern selene_void_result_t selene_print_u64(SeleneInstance *instance, selene_string_t tag, uint64_t value);
extern selene_void_result_t selene_print_f64(SeleneInstance *instance, selene_string_t tag, double value);
extern selene_void_result_t selene_print_bool_array(SeleneInstance *instance, selene_string_t tag,
                                             const bool *ptr, uint64_t length);
extern selene_void_result_t selene_print_i64_array(SeleneInstance *instance, selene_string_t tag,
                                            const int64_t *ptr, uint64_t length);
extern selene_void_result_t selene_print_u64_array(SeleneInstance *instance, selene_string_t tag,
                                            const uint64_t *ptr, uint64_t length);
extern selene_void_result_t selene_print_f64_array(SeleneInstance *instance, selene_string_t tag,
                                            const double *ptr, uint64_t length);
extern selene_void_result_t selene_print_panic(SeleneInstance *instance, selene_string_t message,
                                       uint32_t error_code);
extern selene_void_result_t selene_dump_state(SeleneInstance *instance, selene_string_t message,
                                       const uint64_t *qubits, uint64_t qubits_length);
extern selene_void_result_t selene_set_tc(SeleneInstance *instance, uint64_t time_cursor);
extern selene_u64_result_t selene_get_tc(SeleneInstance *instance);
extern selene_u64_result_t selene_get_current_shot(SeleneInstance *instance);
extern selene_void_result_t selene_local_barrier(SeleneInstance *instance, const uint64_t *qubit_ids,
                                         uint64_t qubit_ids_length, uint64_t sleep_time);
extern selene_void_result_t selene_global_barrier(SeleneInstance *instance, uint64_t sleep_time);
extern selene_u64_result_t selene_shot_count(SeleneInstance *instance);
extern selene_void_result_t selene_on_shot_start(SeleneInstance *instance, uint64_t shot_index);
extern selene_void_result_t selene_on_shot_end(SeleneInstance *instance);
extern selene_void_result_t selene_load_config(SeleneInstance **instance, const char *config_file);
extern selene_void_result_t selene_exit(SeleneInstance *instance);
extern selene_void_result_t selene_print_exit(SeleneInstance *instance, selene_string_t message,
                                      uint32_t error_code);
extern selene_void_result_t selene_random_seed(SeleneInstance *instance, uint64_t seed);
extern selene_void_result_t selene_random_advance(SeleneInstance *instance, uint64_t delta);
extern selene_u32_result_t selene_random_u32(SeleneInstance *instance);
extern selene_u32_result_t selene_random_u32_bounded(SeleneInstance *instance, uint32_t bound);
extern selene_f64_result_t selene_random_f64(SeleneInstance *instance);
extern selene_u64_result_t selene_custom_runtime_call(SeleneInstance *instance, uint64_t tag,
                                               const uint8_t *data, uint64_t data_length);
extern void *pecos_create_execution_context(void);
extern void pecos_register_execution_context(void *);
extern void pecos_destroy_execution_context(void *);
extern void pecos_clear_program_error(void);
extern char *pecos_get_program_error_json(void);
extern char *pecos_get_named_results_json(void);
extern void pecos_free_named_results_json(char *);
extern void *heap_alloc(int64_t);
extern size_t pecos_get_context_allocation_free_count(void);
extern bool pecos_program_exited(void);
typedef void (*handler_t)(void);
extern handler_t pecos_get_program_panic_handler(void);
extern void pecos_set_program_panic_handler(handler_t);

// Report the first failing source line after releasing the oracle's resources.
#define CHECK(expr) do { if (!(expr)) { failure = __LINE__; goto cleanup; } } while (0)
#define OK(expr) CHECK((expr).error_code == 0)
#define LAYOUT(type, size, alignment, field, offset) \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type); \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type)

// The paired Rust layout test uses these same fixed 64-bit C ABI values.
#if UINTPTR_MAX == UINT64_MAX
LAYOUT(SeleneInstance, 4, 4, dummy, 0);
LAYOUT(selene_void_result_t, 4, 4, error_code, 0);
LAYOUT(selene_u64_result_t, 16, 8, value, 8);
LAYOUT(selene_u32_result_t, 8, 4, value, 4);
LAYOUT(selene_f64_result_t, 16, 8, value, 8);
LAYOUT(selene_bool_result_t, 8, 4, value, 4);
LAYOUT(selene_future_result_t, 16, 8, reference, 8);
_Static_assert(offsetof(selene_u64_result_t, error_code) == 0, "error code: selene_u64_result_t");
_Static_assert(offsetof(selene_u32_result_t, error_code) == 0, "error code: selene_u32_result_t");
_Static_assert(offsetof(selene_f64_result_t, error_code) == 0, "error code: selene_f64_result_t");
_Static_assert(offsetof(selene_bool_result_t, error_code) == 0, "error code: selene_bool_result_t");
_Static_assert(offsetof(selene_future_result_t, error_code) == 0, "error code: selene_future_result_t");
LAYOUT(selene_string_t, 24, 8, data, 0);
_Static_assert(offsetof(selene_string_t, length) == 8, "string length");
_Static_assert(offsetof(selene_string_t, owned) == 16, "string owned");
#endif

static bool error_contains(const char *expected) {
    char *error = pecos_get_program_error_json();
    if (error == NULL) return false;
    bool matches = strstr(error, expected) != NULL;
    pecos_free_named_results_json(error);
    return matches;
}

int abi_oracle(void (*set_measurements)(void)) {
    int failure = 0;
    char *json = NULL;
    void *ctx = pecos_create_execution_context();
    pecos_register_execution_context(ctx);
    SeleneInstance *instance = NULL, *again = NULL;
    OK(selene_load_config(&instance, NULL));
    OK(selene_load_config(&again, "ignored"));
    CHECK(instance != NULL && instance == again);
    OK(selene_on_shot_start(instance, 123));
    selene_u64_result_t q0 = selene_qalloc(instance);
    selene_u64_result_t q1 = selene_qalloc(instance);
    CHECK(q0.error_code == 0 && q0.value == 0);
    CHECK(q1.error_code == 0 && q1.value == 1);
    OK(selene_rxy(instance, q1.value, 0.25, 0.75));
    OK(selene_rz(instance, q0.value, 1.25));
    OK(selene_rzz(instance, q1.value, q0.value, 1.75));
    OK(selene_qubit_reset(instance, q1.value));
    selene_bool_result_t measured = selene_qubit_measure(instance, q0.value);
    CHECK(measured.error_code == 0 && !measured.value);
    selene_future_result_t future = selene_qubit_lazy_measure(instance, q1.value);
    CHECK(future.error_code == 0 && future.reference == 1);
    selene_future_result_t leaked = selene_qubit_lazy_measure_leaked(instance, q0.value);
    CHECK(leaked.error_code == 0 && leaked.reference == 2);
    set_measurements();
    selene_bool_result_t b = selene_future_read_bool(instance, future.reference);
    selene_u64_result_t u = selene_future_read_u64(instance, leaked.reference);
    CHECK(b.error_code == 0 && b.value && u.error_code == 0 && u.value == 1);
    OK(selene_refcount_increment(instance, future.reference));
    OK(selene_refcount_decrement(instance, future.reference));
    OK(selene_random_seed(instance, 42));
    selene_u32_result_t r = selene_random_u32(instance);
    CHECK(r.error_code == 0 && r.value == 1085446021);
    r = selene_random_u32_bounded(instance, 10);
    CHECK(r.error_code == 0 && r.value == 0);
    selene_f64_result_t f = selene_random_f64(instance);
    CHECK(f.error_code == 0 && f.value == 0.18373215361498296);
    OK(selene_random_advance(instance, UINT64_MAX));
    // By-value strings carry nontrivial length and owned fields; ownership is ignored.
    const char label[] = {'i', 'n', 't', 'X'};
    selene_string_t tag = {label, 3, true};
    OK(selene_print_i64(instance, tag, INT64_MIN));
    tag = (selene_string_t){"uint", 4, false};
    OK(selene_print_u64(instance, tag, UINT64_MAX));
    tag = (selene_string_t){"float", 5, true};
    OK(selene_print_f64(instance, tag, -2.5));
    tag = (selene_string_t){"bool", 4, false};
    OK(selene_print_bool(instance, tag, true));
    const int64_t ia[] = {-7, 8};
    const uint64_t ua[] = {UINT64_MAX, 9};
    const double fa[] = {-0.5, 2.25};
    const bool ba[] = {true, false};
    OK(selene_print_i64_array(instance, (selene_string_t){"ia", 2, true}, ia, 2));
    OK(selene_print_u64_array(instance, (selene_string_t){"ua", 2, false}, ua, 2));
    OK(selene_print_f64_array(instance, (selene_string_t){"fa", 2, true}, fa, 2));
    OK(selene_print_bool_array(instance, (selene_string_t){"ba", 2, false}, ba, 2));
    json = pecos_get_named_results_json();
    CHECK(json != NULL);
    // Match each tag, declared type, and complete value array independently.
    CHECK(strstr(json, "\"bool\":{\"type\":\"bool\",\"values\":[true]}") != NULL);
    CHECK(strstr(json, "\"int\":{\"type\":\"i64\",\"values\":[-9223372036854775808]}") != NULL);
    CHECK(strstr(json, "\"uint\":{\"type\":\"u64\",\"values\":[18446744073709551615]}") != NULL);
    // Floating outputs serialize their IEEE-754 bit patterns.
    CHECK(strstr(json, "\"float\":{\"type\":\"f64\",\"values\":[13836183955189006336]}") != NULL);
    CHECK(strstr(json, "\"ba\":{\"type\":\"bool\",\"values\":[true,false]}") != NULL);
    CHECK(strstr(json, "\"ia\":{\"type\":\"i64\",\"values\":[-7,8]}") != NULL);
    CHECK(strstr(json, "\"ua\":{\"type\":\"u64\",\"values\":[18446744073709551615,9]}") != NULL);
    CHECK(strstr(json, "\"fa\":{\"type\":\"f64\",\"values\":[13826050856027422720,4612248968380809216]}") != NULL);
    CHECK(strstr(json, "intX") == NULL);
    pecos_free_named_results_json(json);
    json = NULL;
    OK(selene_dump_state(instance, tag, NULL, UINT64_MAX));
    OK(selene_set_tc(instance, 999));
    CHECK(selene_get_tc(instance).value == 0);
    CHECK(selene_get_current_shot(instance).value == 0);
    CHECK(selene_shot_count(instance).value == 1);
    OK(selene_local_barrier(instance, NULL, 88, 99));
    OK(selene_global_barrier(instance, 111));
    u = selene_custom_runtime_call(instance, 99, NULL, UINT64_MAX);
    CHECK(u.error_code == 0 && u.value == 0);
    OK(selene_qfree(instance, q0.value));
    OK(selene_qfree(instance, UINT64_MAX));
    CHECK(error_contains("-1"));
    pecos_clear_program_error();
    size_t freed = pecos_get_context_allocation_free_count();
    CHECK(heap_alloc(32) != NULL);
    OK(selene_on_shot_start(instance, 456));
    CHECK(pecos_get_context_allocation_free_count() == freed + 1);
    selene_random_u32(instance);
    CHECK(error_contains("random_seed"));
    CHECK(heap_alloc(32) != NULL);
    OK(selene_on_shot_end(instance));
    CHECK(pecos_get_context_allocation_free_count() == freed + 2);
    OK(selene_exit(instance));
cleanup:
    if (json != NULL) pecos_free_named_results_json(json);
    pecos_register_execution_context(NULL);
    pecos_destroy_execution_context(ctx);
    return failure;
}

#ifdef BYTE_TEST_MAIN
static int reached;
extern uint64_t pecos_call_void_main_with_setjmp(void (*)(void));
static void panic_entry(void) {
    selene_print_panic(NULL, (selene_string_t){"raw panic", 9, false}, 1001);
    reached = 1;
}
int main(void) {
    int failure = 0;
    void *ctx = pecos_create_execution_context();
    const char bounded[] = {'a','b','c'};
    const char nul[] = {'d',0,'e'};
    const char raw[] = {(char)0xff, (char)0x80, 'z'};
    OK(selene_print_exit(NULL, (selene_string_t){bounded, 3, false}, UINT32_MAX));
    OK(selene_print_exit(NULL, (selene_string_t){nul, 3, true}, 17));
    OK(selene_print_exit(NULL, (selene_string_t){raw, 3, false}, 0));
    OK(selene_print_exit(NULL, (selene_string_t){"long", UINT64_MAX, false}, 8));
    OK(selene_print_exit(NULL, (selene_string_t){"max", INT32_MAX, false}, 9));
    pecos_register_execution_context(ctx);
    panic_entry();
    CHECK(reached == 1);
    CHECK(error_contains("raw panic"));
    reached = 0;
    CHECK(pecos_call_void_main_with_setjmp(panic_entry) == 1);
    CHECK(reached == 0);
    CHECK(error_contains("raw panic"));
cleanup:
    pecos_register_execution_context(NULL);
    pecos_destroy_execution_context(ctx);
    if (failure != 0) fprintf(stderr, "C byte oracle failed at selene_abi.c:%d\n", failure);
    return failure == 0 ? 0 : 1;
}
#endif
