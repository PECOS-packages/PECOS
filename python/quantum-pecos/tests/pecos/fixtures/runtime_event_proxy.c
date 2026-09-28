/* Public synthetic test proxy: prepend one invented event per shot. */
#include <dlfcn.h>
#include <selene/runtime.h>

static SeleneRuntimePluginDescriptorV1 original;
static SeleneRuntimePluginDescriptorV1 proxy;
/* Tests run one runtime per thread. No physical interpretation is assigned. */
static _Thread_local bool event_pending;

static SeleneErrno start(RuntimeInstance instance, uint64_t shot, uint64_t seed) {
    event_pending = true;
    return original.shot_start_fn(instance, shot, seed);
}

static SeleneErrno next(RuntimeInstance instance, RuntimeGetOperationHandle ops) {
    if (event_pending) {
        static const unsigned char payload[] = {17, 29, 43};
        event_pending = false;
        ops.interface.custom_fn(ops.instance, 424242, payload, sizeof(payload));
        ops.interface.set_batch_time_fn(ops.instance, 0, 0);
        return 0;
    }
    return original.get_next_operations_fn(instance, ops);
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
