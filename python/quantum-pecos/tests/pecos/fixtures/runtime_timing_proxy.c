/* Public synthetic timestamp proxy for sim() integration tests. */
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <selene/runtime.h>

static SeleneRuntimePluginDescriptorV1 original;
static SeleneRuntimePluginDescriptorV1 proxy;
/* Deliberately synthetic timing: one second before the second RXY callback.
 * INITIAL_NANOS offsets every batch timestamp equally, adding an initial gap
 * while preserving the gaps between subsequent batches.
 * This is NOT a calibration or timing model for the public runtime. */
struct InstanceTiming {
    RuntimeInstance instance;
    uint64_t clock;
    uint64_t shot_origin;
    unsigned rxy_count;
#ifdef PREP_GAP_NANOS
    bool prep_emitted;
#endif
    struct InstanceTiming *next;
};
static struct InstanceTiming *instances;
static pthread_mutex_t instances_mutex = PTHREAD_MUTEX_INITIALIZER;
/* Instances may move between worker threads, but calls on one instance are
 * serialized. Only synchronous callback forwarding is thread-local. */
static _Thread_local struct InstanceTiming *current;

static struct InstanceTiming *timing_for(RuntimeInstance instance) {
    pthread_mutex_lock(&instances_mutex);
    struct InstanceTiming *timing = instances;
    while (timing && timing->instance != instance) timing = timing->next;
    pthread_mutex_unlock(&instances_mutex);
    return timing;
}
static SeleneErrno init(RuntimeInstance *instance, uint64_t qubits, uint64_t start,
                        uint32_t argc, const char *const *argv) {
    struct InstanceTiming *timing = calloc(1, sizeof(*timing));
    if (!timing) return ENOMEM;
    SeleneErrno rc = original.init_fn(instance, qubits, start, argc, argv);
    if (rc != 0) {
        free(timing);
        return rc;
    }
    timing->instance = *instance;
    timing->clock = start;
    timing->shot_origin = start;
    pthread_mutex_lock(&instances_mutex);
    timing->next = instances;
    instances = timing;
    pthread_mutex_unlock(&instances_mutex);
    return 0;
}
static SeleneErrno exit_runtime(RuntimeInstance instance) {
    SeleneErrno rc = original.exit_fn ? original.exit_fn(instance) : 0;
    pthread_mutex_lock(&instances_mutex);
    struct InstanceTiming **link = &instances;
    while (*link && (*link)->instance != instance) link = &(*link)->next;
    if (*link) {
        struct InstanceTiming *timing = *link;
        *link = timing->next;
        free(timing);
    }
    pthread_mutex_unlock(&instances_mutex);
    return rc;
}
static _Thread_local RuntimeGetOperationHandle forwarding;
static _Thread_local bool has_batch;
#ifdef PREP_GAP_NANOS
static void reset(SeleneRuntimeGetOperationInstance instance, uint64_t q) {
    (void)instance;
    current->prep_emitted = true;
    forwarding.interface.reset_fn(forwarding.instance, q);
}
#endif
static void time_batch(SeleneRuntimeGetOperationInstance instance, uint64_t start, uint64_t duration) {
    (void)instance; (void)start; (void)duration;
    has_batch = true;
}
static SeleneErrno start(RuntimeInstance instance, uint64_t shot, uint64_t seed) {
    struct InstanceTiming *timing = timing_for(instance);
    if (!timing) return EINVAL;
    /* Only shots begun through shot_start advance the origin; this proxy does
     * not model lowering on a kept instance without a new shot_start. */
    timing->shot_origin = timing->clock;
    timing->rxy_count = 0;
#ifdef PREP_GAP_NANOS
    timing->prep_emitted = false;
#endif
    return original.shot_start_fn(instance, shot, seed);
}
static void rxy(SeleneRuntimeGetOperationInstance instance, uint64_t q, double theta, double phi) {
    (void)instance;
    ++current->rxy_count;
    forwarding.interface.rxy_fn(forwarding.instance, q, theta, phi);
#ifdef SYNTHETIC_EVENT
    if (current->rxy_count == 1) {
        const uint8_t payload[] = {1};
        forwarding.interface.custom_fn(forwarding.instance, 4242, payload, sizeof(payload));
    }
#endif
}
static SeleneErrno next(RuntimeInstance instance, RuntimeGetOperationHandle ops) {
    current = timing_for(instance);
    if (!current) return EINVAL;
    forwarding = ops;
    has_batch = false;
    RuntimeGetOperationHandle wrapped = ops;
    wrapped.interface.rxy_fn = rxy;
    wrapped.interface.set_batch_time_fn = time_batch;
#ifdef PREP_GAP_NANOS
    /* The program starts with a prep; isolate it before coalescing readouts. */
    bool prep_batch = !current->prep_emitted;
    wrapped.interface.reset_fn = reset;
#endif
    SeleneErrno rc;
    bool any_batch = false;
    do {
        has_batch = false;
        rc = original.get_next_operations_fn(instance, wrapped);
        any_batch = any_batch || has_batch;
#ifdef PREP_GAP_NANOS
        if (prep_batch && current->prep_emitted) break;
#endif
#ifndef COALESCE_QUEUED
        break;
#endif
        /* Test-only batching: drain only currently queued operations. A program
         * waiting on a measurement remains suspended until PECOS returns it.
         * This deliberately invents one zero-duration batch, not runtime timing. */
    } while (rc == 0 && has_batch);
    if (rc == 0 && any_batch) {
        uint64_t relative = INITIAL_NANOS;
#ifdef PREP_GAP_NANOS
        if (!prep_batch) relative += PREP_GAP_NANOS + (current->rxy_count >= 2 ? GAP_NANOS : 0);
#else
        relative += current->rxy_count >= 2 ? GAP_NANOS : 0;
#endif
        /* Base timestamps are intentionally replaced. The proxy's emitted
         * zero-duration batches, not the base runtime's clock, define this clock. */
        uint64_t start = current->shot_origin + relative;
        ops.interface.set_batch_time_fn(ops.instance, start, 0);
        current->clock = start;
    }
    return rc;
}

const SeleneRuntimePluginDescriptorV1 *selene_runtime_get_plugin_descriptor_v1(void) {
    /* PECOS configures worker clones on the host before running shots. */
    if (proxy.struct_size == 0) {
        void *library = dlopen(BASE_LIBRARY, RTLD_NOW | RTLD_LOCAL);
        if (!library) return NULL;
        const SeleneRuntimePluginDescriptorV1 *descriptor =
            dlsym(library, "selene_runtime_plugin_descriptor_v1");
        if (!descriptor) {
            const SeleneRuntimePluginDescriptorV1 *(*get_descriptor)(void) =
                dlsym(library, "selene_runtime_get_plugin_descriptor_v1");
            if (!get_descriptor) return NULL;
            descriptor = get_descriptor();
        }
        if (!descriptor || descriptor->struct_size != sizeof(original) ||
            descriptor->api_version != SELENE_RUNTIME_CURRENT_API_VERSION) return NULL;
        original = *descriptor;
        proxy = original;
        proxy.init_fn = init;
        proxy.exit_fn = exit_runtime;
        proxy.shot_start_fn = start;
        proxy.get_next_operations_fn = next;
    }
    return &proxy;
}
