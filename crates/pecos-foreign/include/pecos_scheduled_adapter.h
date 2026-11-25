#ifndef PECOS_SCHEDULED_ADAPTER_H
#define PECOS_SCHEDULED_ADAPTER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental ABI v1. See design/native-scheduled-adapter.md for the contract.
 * All views, payloads and sinks are borrowed for one synchronous callback.
 * Call host functions only on the invoking thread, without reentry or retention.
 * Emitted buffers are borrowed for the emit call and copied before return.
 * All nonempty pointers must be valid, aligned and non-overlapping with host
 * contexts/output storage. Sessions must bound their retained state.
 * No exceptions or panics may cross this boundary. */
#define PECOS_SCHEDULED_ADAPTER_ABI 1u
#define PECOS_SCHEDULED_ADAPTER_CAPABILITIES 3u /* independent, movable sessions */
typedef struct {
    uint32_t kind; /* 0 gate, 1 custom */
    uint32_t reserved; /* zero */
    /* Gate: code,q0,q1,angle0,angle1; exact unsigned fraction of one turn.
     * codes: 1 RXY,2 RZ,3 RZZ,4 RXYXY,5 PZ,6 MZ,7 MeasureZLeaked.
     * Unused fields are zero. Custom: tag,0,0,0,0. */
    uint64_t words[5];
    const uint8_t *payload; /* custom only, may be NULL when length is zero */
    size_t payload_len;
} PecosScheduledOpV1;
typedef struct {
    uint64_t operation_index, runtime_result, program_result;
} PecosScheduledResultV1;
typedef struct {
    uint64_t runtime_shot_id, batch_index, start_nanos, duration_nanos;
    uint64_t operation_count, measurement_count;
    void *context;
    int32_t (*read_operation)(void *, uint64_t, PecosScheduledOpV1 *);
    int32_t (*read_measurement)(void *, uint64_t, PecosScheduledResultV1 *);
} PecosScheduledBatchV1;
typedef struct {
    void *context;
    int32_t (*emit_gate)(void *, const uint64_t *); /* exactly five words */
    int32_t (*emit_source_gate)(void *, uint64_t);
    /* kind 0 local victims, 1 global exclusions; unique IDs, preserve order. */
    int32_t (*emit_crosstalk)(void *, uint32_t, const uint64_t *, size_t);
} PecosScheduledSinkV1;
typedef struct {
    uint32_t abi_version, struct_size;
    uint64_t capabilities, reserved;
    /* Success requires a non-NULL output. Every non-NULL output is valid for destroy, even on failure. Copy retained
     * config; host context identifies a shot, not an RNG seed. */
    int32_t (*create)(const uint8_t *, size_t, uint64_t, uint64_t, uint64_t, void **);
    /* validate is deterministic and side-effect free. No output sink. */
    int32_t (*validate)(const void *, const PecosScheduledBatchV1 *);
    int32_t (*translate)(void *, const PecosScheduledBatchV1 *, const PecosScheduledSinkV1 *);
    void (*destroy)(void *);
} PecosScheduledAdapterV1;
/* Plugin exports this function. Reject unsupported version or exact size before
 * writing; fill host-owned output without retaining it. Return zero on success.
 * This symbol is distinct from runtime, decoder and simulator entry points. */
int32_t pecos_scheduled_adapter_init(uint32_t version, size_t size, PecosScheduledAdapterV1 *out);
#ifdef __cplusplus
}
#endif
#endif
