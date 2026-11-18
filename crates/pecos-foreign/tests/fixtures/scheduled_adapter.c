#include "pecos_scheduled_adapter.h"
#include <stdatomic.h>
#include <stdlib.h>
#ifndef BAD_DESCRIPTOR
#define BAD_DESCRIPTOR 0
#endif
static _Atomic uint64_t counts[4];
uint64_t probe_count(uint32_t i) { return i < 4 ? atomic_load(&counts[i]) : 0; }
typedef struct { uint8_t mode; uint64_t next_batch, run, worker, shot; } Session;
static int32_t create(const uint8_t *config, size_t size, uint64_t run, uint64_t worker, uint64_t shot, void **out) {
    *out = NULL;
    if (size != 1) return 15;
    if (config[0] == 5) return 0;
    if (config[0] == 6) return 16;
    Session *s = calloc(1, sizeof(*s));
    if (!s) return 17;
    s->mode = config[0]; s->run = run; s->worker = worker; s->shot = shot;
    *out = s; atomic_fetch_add(&counts[0], 1);
    return config[0] == 4 ? 18 : 0;
}
static void destroy(void *instance) { free(instance); atomic_fetch_add(&counts[1], 1); }
static int32_t validate(const void *instance, const PecosScheduledBatchV1 *batch) {
    const Session *s = instance;
    atomic_fetch_add(&counts[2], 1);
    if (s->mode == 1) return 20;
    PecosScheduledOpV1 op;
    if (s->mode == 17) { batch->read_operation(batch->context, batch->operation_count, &op); return 61; }
    if (s->mode == 2) { batch->read_operation(batch->context, batch->operation_count, &op); return 0; }
    if (s->mode == 12) { batch->read_measurement(batch->context, batch->measurement_count, NULL); return 0; }
    for (uint64_t i = 0; i < batch->operation_count; ++i) {
        if (batch->read_operation(batch->context, i, &op)) return 21;
        if (op.reserved || op.kind > 1) return 22;
        if (op.kind == 1 && (op.words[0] != 4242 || op.payload_len != 1 || op.payload[0] > 1)) return 23;
    }
    for (uint64_t i = 0; i < batch->measurement_count; ++i) {
        PecosScheduledResultV1 m;
        if (batch->read_measurement(batch->context, i, &m)) return 24;
        if (batch->read_operation(batch->context, m.operation_index, &op)) return 25;
        if (op.kind != 0 || (op.words[0] != 6 && op.words[0] != 7)) return 26;
        if (s->mode == 14 && (m.runtime_result != 900 || m.program_result != 17 || s->run != 11 || s->worker != 3 || s->shot != 5)) return 27;
    }
    return 0;
}
static int32_t translate(void *instance, const PecosScheduledBatchV1 *batch, const PecosScheduledSinkV1 *sink) {
    Session *s = instance;
    atomic_fetch_add(&counts[3], 1);
    if (s->mode == 7) return 30;
    if (batch->batch_index != s->next_batch++) return 31;
    if (s->mode == 3) { sink->emit_source_gate(sink->context, batch->operation_count); return 0; }
    if (s->mode == 8) { const uint64_t w[5] = {2,0,1,0,0}; sink->emit_gate(sink->context,w); return 0; }
    if (s->mode == 9) { sink->emit_crosstalk(sink->context,0,NULL,SIZE_MAX); return 0; }
    if (s->mode == 11) { const uint64_t ids[2] = {0,0}; sink->emit_crosstalk(sink->context,0,ids,2); return 0; }
    for (uint64_t i = 0; i < batch->operation_count; ++i) {
        PecosScheduledOpV1 op;
        if (batch->read_operation(batch->context,i,&op)) return 32;
        if (op.kind == 0) {
            if (s->mode == 10 && op.words[0] == 6) continue;
            if (s->mode == 13) { if (sink->emit_gate(sink->context,op.words)) return 33; }
            else if (sink->emit_source_gate(sink->context,i)) return 34;
        } else if (s->mode == 15 || s->mode == 16) {
            const uint64_t ids[1] = {0};
            if (sink->emit_crosstalk(sink->context, s->mode == 15 ? 0 : 1, ids, s->mode == 15 ? 1 : 0)) return 36;
        } else if (op.payload[0]) {
            const uint64_t x[5] = {1,0,0,UINT64_C(1)<<63,0};
            if (sink->emit_gate(sink->context,x)) return 35;
        }
    }
    return 0;
}
int32_t pecos_scheduled_adapter_init(uint32_t version, size_t size, PecosScheduledAdapterV1 *out) {
    if (version != 1 || size != sizeof(*out)) return 40;
    *out = (PecosScheduledAdapterV1){1,sizeof(*out),3,0,create,validate,translate,destroy};
    switch (BAD_DESCRIPTOR) {
        default: break;
        case 1: out->abi_version = 2; break;
        case 2: --out->struct_size; break;
        case 3: out->capabilities = 0; break;
        case 4: out->reserved = 1; break;
        case 5: out->create = NULL; break;
        case 6: out->validate = NULL; break;
        case 7: out->translate = NULL; break;
        case 8: out->destroy = NULL; break;
        case 9: return 41;
    }
    return 0;
}
