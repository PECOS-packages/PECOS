/* Public synthetic timestamp proxy for sim() integration tests. */
#include <dlfcn.h>
#include <selene/runtime.h>

static SeleneRuntimePluginDescriptorV1 original;
static SeleneRuntimePluginDescriptorV1 proxy;
/* Deliberately synthetic timing: one second before the second RXY callback.
 * INITIAL_NANOS offsets every batch timestamp equally, adding an initial gap
 * while preserving the gaps between subsequent batches.
 * This is NOT a calibration or timing model for the public runtime. */
static _Thread_local unsigned rxy_count;
static _Thread_local RuntimeGetOperationHandle forwarding;
static _Thread_local bool has_batch;
static void time_batch(SeleneRuntimeGetOperationInstance instance, uint64_t start, uint64_t duration) {
    (void)instance; (void)start; (void)duration;
    has_batch = true;
}
static SeleneErrno start(RuntimeInstance instance, uint64_t shot, uint64_t seed) {
    rxy_count = 0;
    return original.shot_start_fn(instance, shot, seed);
}
static void rxy(SeleneRuntimeGetOperationInstance instance, uint64_t q, double theta, double phi) {
    (void)instance;
    ++rxy_count;
    forwarding.interface.rxy_fn(forwarding.instance, q, theta, phi);
#ifdef SYNTHETIC_EVENT
    if (rxy_count == 1) {
        const uint8_t payload[] = {1};
        forwarding.interface.custom_fn(forwarding.instance, 4242, payload, sizeof(payload));
    }
#endif
}
static SeleneErrno next(RuntimeInstance instance, RuntimeGetOperationHandle ops) {
    forwarding = ops;
    has_batch = false;
    RuntimeGetOperationHandle wrapped = ops;
    wrapped.interface.rxy_fn = rxy;
    wrapped.interface.set_batch_time_fn = time_batch;
    SeleneErrno rc;
    bool any_batch = false;
    do {
        has_batch = false;
        rc = original.get_next_operations_fn(instance, wrapped);
        any_batch = any_batch || has_batch;
#ifndef COALESCE_QUEUED
        break;
#endif
        /* Test-only batching: drain only currently queued operations. A program
         * waiting on a measurement remains suspended until PECOS returns it.
         * This deliberately invents one zero-duration batch, not runtime timing. */
    } while (rc == 0 && has_batch);
    if (rc == 0 && any_batch) ops.interface.set_batch_time_fn(ops.instance, INITIAL_NANOS + (rxy_count >= 2 ? GAP_NANOS : 0), 0);
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
        proxy.shot_start_fn = start;
        proxy.get_next_operations_fn = next;
    }
    return &proxy;
}
