# Opaque Selene runtime events

Selene runtimes can emit plugin-defined custom events alongside scheduled gates.
PECOS copies each event's tag and payload bytes before the callback returns and
retains the original batch start time, duration, batch ordinal, and operation
position. These positions refer to runtime output before QIS lowering inserts
idle operations. Event ordering is emission ordering, not a sort by timestamp.

The Rust `SeleneRuntime` API provides two policies:

- `Capture` (default) retains events without changing quantum execution. This
  preserves compatibility with runtimes that emit optional metadata. It does not
  certify that custom physical effects have been simulated.
- `RejectUnhandled` fails unless a configured handler explicitly acknowledges the
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

Handlers run synchronously during lowering, after the plugin callback has returned
and final batch timing is available. This is callback emission order during
lowering, not execution-time interleaving: preceding gates or measurements may
not yet have been simulated. No measurement outcome or before/after execution
phase is supplied. Handler errors propagate in both policies.
A rejected event remains available through `custom_events()` for diagnosis.
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

This is a Rust runtime transport and metadata-handling interface. It does not
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
