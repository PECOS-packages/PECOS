# Opaque Selene runtime events

Selene runtimes can emit plugin-defined custom events alongside scheduled gates.
PECOS copies each event's tag and payload bytes before the callback returns and
retains the original batch start time, duration, batch ordinal, and operation
position. These positions refer to runtime output before QIS lowering inserts
idle operations. Event ordering is emission ordering, not a sort by timestamp.

The Rust `SeleneRuntime` API provides two policies:

- `Capture` (explicit opt-in) retains events without changing quantum execution. This
  preserves compatibility with runtimes that emit optional metadata. It does not
  certify that custom physical effects have been simulated.
- `RejectUnhandled` (default) fails unless a configured handler explicitly acknowledges the
  event as `MetadataOnly`. Use this policy when unsupported information must stop
  execution. Unknown events and unsupported physical effects must remain
  `Unsupported`; PECOS cannot infer their meaning from opaque bytes.

```rust
use pecos_qis::{
    RuntimeCustomEventDisposition, RuntimeCustomEventPolicy, SeleneRuntime,
};

let mut runtime = SeleneRuntime::new("example-runtime.so");
runtime.set_custom_event_policy(RuntimeCustomEventPolicy::RejectUnhandled);
runtime.set_custom_event_handler(|event| {
    // Invented metadata protocol: tag 7301, version byte 1, no physical effect.
    Ok(if event.tag == 7301 && event.data == [1] {
        RuntimeCustomEventDisposition::MetadataOnly
    } else {
        RuntimeCustomEventDisposition::Unsupported
    })
});
```

## Python execution policy

Select the policy when configuring the runtime for `sim()`:

```python
from guppylang import guppy
from guppylang.std.quantum import measure, qubit

from pecos import Guppy, selene_engine, sim


@guppy
def prepare_zero() -> bool:
    return measure(qubit()).read()


# The default runtime emits no custom events, so strict execution succeeds.
engine = selene_engine(custom_event_policy="reject_unhandled")
results = sim(Guppy(prepare_zero)).classical(engine).qubits(1).seed(42).run(10)
assert not any(results.to_dict()["measurement_0"])
```

The same keyword is accepted by `pecos.qis_engine().selene_runtime(...)` and by
native `pecos_rslib` runtime configuration methods. It follows the runtime through
plugin-object, library-path, built-runtime and installed-plugin fallback selection.
Unknown policy values raise `ValueError` before runtime discovery or configuration.

The default `"reject_unhandled"` stops execution with the event tag, runtime,
and batch/operation/timing context. Explicit `"capture"` retains the old behavior:
custom events do not change the simulated physics. Without the batch adapter
described below, strict policy rejects **every** custom event, including metadata.
A successful run in capture mode is not a fidelity check;
strict rejection identifies missing integration, rather than implementing it.
Selecting the policy does not enable the separate bounded runtime-frame executor.

## Shared simulation and tracing adapters

A Selene-compatible runtime integration can provide `batch_adapter_factory` on
its plugin object, or callers can pass that keyword to `selene_engine()`,
`qis_engine().selene_runtime(...)`, `capture_qis_operation_trace()`, or
`trace_program_to_tick_circuit()`. PECOS treats all runtime plugins alike and
contains no private event tags or schemas.

The zero-argument factory returns a fresh object with `normalize(batch)` for
each shot. The method receives an owned dictionary containing `shot_id`,
`batch_index`, `num_qubits`, `start_time_nanos`, `duration_nanos`, and
`operations`. Operations use the following representation:

```python
{"Rxy": {"qubit_id": 0, "theta": 1.57, "phi": 0.0}}
{"Rz": {"qubit_id": 0, "theta": 0.5}}
{"Rzz": {"qubit_id_1": 0, "qubit_id_2": 1, "theta": 0.5}}
{"Rpp": {"qubit_id_1": 0, "qubit_id_2": 1, "theta": 0.5, "phi": 0.0}}
{"Reset": {"qubit_id": 0}}
{"Measure": {"qubit_id": 0, "result_id": 7}}
{"MeasureLeaked": {"qubit_id": 0, "result_id": 7}}
{"Custom": {"tag": 7301, "data": [1]}}
```

Return a list of ordinary operations in the same representation. Every custom
event must be understood: metadata may produce no gate, whereas physical effects
must be represented by gates. If a payload duplicates ordinary gates or timing,
the integration must verify that relationship rather than emit them twice.
Unknown schemas, stochastic channels, repumping, or other effects outside the
adapter's supported semantics must raise. PECOS cannot verify an arbitrary
adapter's physical interpretation; installing it is an explicit trust boundary.

PECOS validates finite angles, qubit bounds, output shapes, expansion limits, and
unchanged measurement order, kind, targets, and native result IDs. Output cannot
contain custom events. Native batches and expansions are limited to 4096 operations,
with at most 256 KiB of source payload per batch. Errors poison the shot until
reset. Adapters are fresh across shots and worker clones. Clean native runtime
instances remain reusable across shots, whether or not an adapter is configured.
Callbacks must not call back into built simulations or share mutable shot state.

Program-to-runtime custom calls use `QisEngine`'s existing control loop. A call
returns only after the quantum engine has executed emitted work and measurement
feedback has returned to the runtime. Normalizers describe operations; they do
not execute a second simulator inside the callback. Ordinary normalized gates
pass through the configured PECOS noise and quantum engines.

Effects requiring an execution-time noise/quantum exchange use the scheduled-event
route and its supported typed effects, such as measurement-crosstalk exposures.
Its noise engine can request internal measurements, consume their results and
continue processing without exposing them as program outputs. Unsupported effects
must raise; neither an adapter nor capture mode supplies a missing physical model.
The scheduled route currently rejects operation-trace capture, so it cannot be
used to certify a DEM through the ordinary trace-replay API.

Both ordinary simulation and tracing use this normalization **before** idle
insertion. The runtime's integer nanosecond timestamps remain authoritative, relative to
the current shot's origin when a native instance is reused.
Each normalized qubit receives one gap before its first touching operation in a
batch, except before reset; all operations on it share the original batch's busy
interval. Adapters cannot retime batches or invent idle durations. Empty batches
still reach the adapter, allowing persistent bookkeeping without creating idle
sites. No final idle tail is added. This shared-interval rule applies only to
adapter-normalized output. Without an adapter, ordinary native batches retain
per-operation timing checks: multiple timed operations touching the same qubit,
including a gate after reset, still reject if their busy intervals overlap.

For Rust integrations, use `SeleneRuntime::set_batch_adapter_factory` with
`RuntimeBatchAdapter`. The Python factory is the equivalent bridge. This is an
ordinary QIS runtime normalizer, separate from the scheduled-noise adapter
factory. Combining it with scheduled transport is rejected before extraction;
that route uses its own scheduled adapter and does not support operation tracing.

Capture-only traces now carry `runtime_events_complete=false` once an unhandled
event occurs. Draining the event history cannot clear that flag. Audited replay
rejects such traces, so enabling capture cannot silently certify a DEM.

## Native clock convention

Clock convention and event interpretation are independent settings. Set
`clock_model="continuous"` (the default) for a native clock that runs across
shots, or `clock_model="shot_relative"` for a runtime whose native `shot_start`
hook restarts its clock at zero. Select the convention explicitly on
`selene_engine(...)`, `qis_engine().selene_runtime(...)`, or the tracing helpers;
a runtime plugin may instead expose a `clock_model` property. An explicit
argument overrides that property. Rust integrations use
`SeleneRuntime::with_clock_model(RuntimeClockModel::ShotRelative)`.

Both conventions reuse clean native instances, with or without an event adapter.
Continuous timestamps are rebased by the collected end of the previous shot;
shot-relative timestamps are already in the trace's time base. A clean host
`reset()` does not restart the native clock: the shot-relative convention takes
effect when the next native shot starts. Neither mode guesses an offset from
observed timestamps, changes a batch duration, or changes idle-noise parameters.
A clock that moves behind the previous shot's origin still fails under the
default continuous convention, even when an adapter is installed. Declare the
runtime's actual contract rather than selecting a mode to suppress that error.

## Rust handler and lifecycle contract

Handlers run synchronously during lowering, after the plugin callback has returned
and final batch timing is available. This is callback emission order during
lowering, not execution-time interleaving: preceding gates or measurements may
not yet have been simulated. No measurement outcome or before/after execution
phase is supplied. Handler errors propagate in both policies.
On flat and metadata routes, a rejected event remains available through
`custom_events()` for diagnosis. Scheduled extraction does not retain a history.
A failed callback batch, batch-conversion error, or handler error latches a terminal
failure in the runtime. Further lowering, draining, measurement updates, and shot
completion/start fail until `QisRuntime::reset()` succeeds. Draining captured
events, changing policy/handler, or cloning does not clear the failure. Handler
panics unwind in Rust outside the C callback; if a caller catches one, the runtime
still requires reset. Handler side effects and preceding scheduling changes are
not rolled back.
Handlers are shared across runtime clones and must be thread-safe. Prefer
stateless handlers: PECOS does not reset state captured by a closure.

`custom_events()` exposes events from the current shot; `take_custom_events()`
drains them without changing the batch numbering. Shot start and runtime reset
clear the history and numbering. Shot end retains history until the next start.
Capturing retains payload memory, so direct users processing large streams can
periodically drain it between lowering calls. There is no hard retention limit,
and one lowering call may drain many runtime batches before returning. Clones
copy history independently and share the handler. Events do not carry a shot or
worker ID; a shared handler must not infer such identity from batch ordinals.

Configure a runtime before moving it into `qis_engine().runtime(runtime)`. Once
owned by the engine, the runtime history accessors are not exposed through
`QisRuntime`; use the configured handler for external observation. A handler
does not drain the retained history or provide bounded-memory streaming.

The per-event Rust handler is a runtime transport and metadata-handling interface. It does not
serialize events into operation traces, expose Python handler configuration, or
translate custom events into quantum operations or noise channels. Events carry
no inferred qubit targets. Metadata acknowledgement never advances a qubit clock
or injects idle/noise operations; ordinary scheduling, idles and measurement
result routing remain in the existing lowering path. Modeling custom physical
effects needs a separate downstream integration; this API does not implement the
broader event-driven noise proposal in PR #591.

The callback layout and payload lifetime follow the pinned Selene 0.3
`RuntimeGetOperationInterface` and `BatchExtractor`: the plugin supplies readable
bytes for the callback duration, and PECOS owns its copy thereafter. Null with a
nonzero length and lengths above `isize::MAX` produce an FFI error after the
callback returns. Zero-length payloads accept null. Allocation failures are
recorded by fallible payload/operation-vector reservations, and later callbacks
stop appending after the first batch error. This does not promise recovery from
all process-wide out-of-memory conditions. As with the rest of the plugin ABI, the
plugin must return the supplied live instance pointer and valid non-null payload
allocations; arbitrary dangling foreign pointers cannot be validated by PECOS.

Scheduled extraction enforces the same policy before returning a batch. Its
metadata handler runs during extraction; events remain in the returned batch,
without a second history copy. To pass custom physical events to a configured
downstream scheduled consumer, explicitly select `Capture`. That consumer must
validate and model the events; capture alone does not simulate them.
