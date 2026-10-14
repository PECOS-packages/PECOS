// Execution recovery stays in C: Rust cannot call a returns-twice setjmp.
#include <stdint.h>
#include <stdbool.h>
#include <setjmp.h>

#ifdef _MSC_VER
#define PECOS_THREAD_LOCAL __declspec(thread)
#else
#define PECOS_THREAD_LOCAL _Thread_local
#endif

typedef struct SeleneInstance { int dummy; } SeleneInstance;
typedef struct { uint32_t error_code; } selene_void_result_t;
typedef void (*program_panic_handler_t)(void);
extern selene_void_result_t selene_on_shot_start(SeleneInstance *, uint64_t);
extern selene_void_result_t selene_on_shot_end(SeleneInstance *);
extern void pecos_clear_program_error(void);
extern program_panic_handler_t pecos_get_program_panic_handler(void);
extern void pecos_set_program_panic_handler(program_panic_handler_t);
extern bool pecos_program_panic_handler_is_installed(void);
extern bool pecos_program_exited(void);

// One recovery target per thread; nested guards are unsupported.
static PECOS_THREAD_LOCAL jmp_buf user_program_jmpbuf;

// Always jump with sentinel 1. The recorded termination variant determines
// success or failure; the original signed code remains in the context.
// Both output ABIs use this guard, including calls made outside a wrapper.
static void pecos_program_panic_transfer(void) {
    if (pecos_program_panic_handler_is_installed()) {
        longjmp(user_program_jmpbuf, 1);
    }
}

/**
 * Wrapper function to safely call qmain with setjmp/longjmp support
 *
 * This function sets up the exception handling mechanism that Helios expects:
 * 1. Calls setjmp to save the current stack state
 * 2. Calls qmain(0) to execute the quantum program
 * 3. If an error occurs and longjmp is called, we catch it and return the error code
 *
 * Safety: nested wrappers and re-entrant FFI calls while context mutexes are
 * held are unsupported (one jump buffer per thread, non-reentrant mutexes).
 *
 * Returns: 0 on success, error code on failure
 */
typedef uint64_t (*qmain_fn_t)(uint64_t);

uint64_t pecos_guard_qmain_with_setjmp(qmain_fn_t qmain) {
    // Initialize shot context to match what interface.c main() does
    // Must be thread-local for concurrent execution by multiple workers
    static PECOS_THREAD_LOCAL SeleneInstance dummy_instance;
    selene_void_result_t start_result = selene_on_shot_start(&dummy_instance, 0);
    if (start_result.error_code != 0) {
        return start_result.error_code;
    }

    pecos_clear_program_error();
    program_panic_handler_t previous_handler = pecos_get_program_panic_handler();
    // C23 7.13.1.1p4-5 permits setjmp in this controlling comparison,
    // but not in an initializer. The recorded termination determines status.
    if (setjmp(user_program_jmpbuf) == 0) {
        pecos_set_program_panic_handler(pecos_program_panic_transfer);
        // Normal path - call qmain
        uint64_t result = qmain(0);

        // Clean up shot context
        pecos_set_program_panic_handler(previous_handler);
        selene_on_shot_end(&dummy_instance);

        return result;
    } else {
        // longjmp was called - an error occurred
        // Clean up even on error
        pecos_set_program_panic_handler(previous_handler);
        selene_on_shot_end(&dummy_instance);

        return pecos_program_exited() ? 0 : 1;
    }
}

/**
 * Wrapper function to safely call a `void main()` entry point.
 *
 * QIR programs can use either `i64 @qmain(i64)` (the Helios profile, with an
 * explicit error-code return value) or `void @main()` (the simpler "base
 * profile" form, with no return value). The two have incompatible C calling
 * conventions: calling `void @main()` through the qmain wrapper (which expects
 * `uint64_t (*)(uint64_t)`) is undefined behaviour and reads whatever happens
 * to be in the return register, producing seemingly-random "error" codes.
 *
 * This wrapper exists so the Rust executor can dispatch on the entry-point
 * symbol it finds and call each kind through the matching ABI.
 *
 * Safety: nested wrappers and re-entrant FFI calls while context mutexes are
 * held are unsupported (one jump buffer per thread, non-reentrant mutexes).
 *
 * Returns: 0 on success, error code on failure (when longjmp is used).
 */
typedef void (*void_main_fn_t)(void);

uint64_t pecos_guard_void_main_with_setjmp(void_main_fn_t main_func) {
    static PECOS_THREAD_LOCAL SeleneInstance dummy_instance;
    selene_void_result_t start_result = selene_on_shot_start(&dummy_instance, 0);
    if (start_result.error_code != 0) {
        return start_result.error_code;
    }

    pecos_clear_program_error();
    program_panic_handler_t previous_handler = pecos_get_program_panic_handler();
    // C23 7.13.1.1p4-5 permits setjmp in this controlling comparison,
    // but not in an initializer. The recorded termination determines status.
    if (setjmp(user_program_jmpbuf) == 0) {
        pecos_set_program_panic_handler(pecos_program_panic_transfer);
        main_func();
        pecos_set_program_panic_handler(previous_handler);
        selene_on_shot_end(&dummy_instance);
        return 0;
    } else {
        pecos_set_program_panic_handler(previous_handler);
        selene_on_shot_end(&dummy_instance);
        return pecos_program_exited() ? 0 : 1;
    }
}
