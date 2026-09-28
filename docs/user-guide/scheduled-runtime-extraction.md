# Native scheduled-batch extraction (Rust)

`QisRuntime::lower_scheduled_operations` provides an opt-in extraction interface
for consumers that need original native batches. Selene implements it; other
runtimes reject it by default. It returns owned `ScheduledBatch` values before
PECOS inserts idle operations or flattens the runtime schedule.

Each batch carries integer start/duration, runtime-local shot ID, batch ordinal,
and operations in emission order. Custom events retain their opaque tag and bytes.
Measurement records preserve the native future ID, source-program result ID,
operation position and source leakage-aware result kind. Empty timed batches are
retained. Timings are not reconstructed from gates or idles.

This API extracts data. It does not execute gates, interpret custom events, invoke
metadata handlers, or acknowledge physical effects. The existing flat-output
custom-event policy governs flat lowering only; returning an opaque event through
scheduled extraction is never evidence that it is metadata or has been modeled.
The ordinary QIS engine and Python `sim()` do not use this interface yet.

## Calling contract

1. Configure explicit, nonzero physical capacity using `set_num_qubits`, and call
   `shot_start`. Capacity cannot change on an initialized scheduled session without
   reset. Input source identifiers must fit capacity arithmetic.
2. Submit native RXY, RZ, RZZ, RXYXY2Q, reset and measurement operations, allocations,
   releases and barriers through `lower_scheduled_operations`. Arbitrary gates,
   non-finite source angles, source trace metadata and output records are rejected
   before submission. Callers must not drop unsupported source metadata silently.
3. Preserve returned batches as units. No gate/noise execution consumer is supplied
   by this API. Consumers must separately validate events and physical support.
4. At completion, call `drain_pending_scheduled_operations` and handle all returned
   work. This forces the native terminal barrier; unsupported terminal flush fails.
   A successful extraction or runtime `shot_end` does not certify physics execution.

Flat and scheduled lowering cannot be mixed before a new shot or reset. Batch
ordinals continue across extraction calls and clear on shot start/reset. Runtime
shot IDs are local to the native runtime; they are not globally unique host
run/worker IDs. Host-context plumbing remains a separate integration requirement.

## Bounds and failures

Each call admits at most 64 batches, 4096 operations and 256 KiB of opaque payload.
Callback allocation is checked against the remaining operation/payload budget;
oversize payloads fail before reading their bytes. The callback stops appending
following its first error. Batch validation rejects timestamp overflow, non-finite
angles, physical qubits outside configured capacity and unmapped measurement IDs.
No additional batch history is retained after return. These limits cover extraction
buffers, not native plugin internals, all runtime bookkeeping, or caller-owned data.

On an extraction failure after submission, partial returned output is discarded
and the failure is latched until successful reset. Caught Rust panics also leave
failure latched. Native scheduler state is not rolled back. Cloning a scheduled
session requires reset before reuse; it is not a live native snapshot. Existing
flat-runtime cloning behavior is preserved.

## Remaining integration

An execution path needs explicit consumer admission, host/worker/shot identity,
a suitable mandatory envelope, and one owner of timing and idle/noise translation.
The existing sequential v2 runtime frame is not extended by this change. The
consumer must preserve controller lifecycle, measurement semantics, RNG usage and
metadata-insertion invariance. No simulator equivalence or performance claim follows
from extraction tests, and no Python route is newly enabled.
