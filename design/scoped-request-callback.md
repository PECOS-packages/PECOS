# Scoped native request callback

## Status and boundary

The FFI library exports `pecos_call_qmain_with_request_callback_v1` and
`pecos_call_void_main_with_request_callback_v1`. They run a guarded program and
service synchronous result/runtime requests on the calling thread. This is the
FFI boundary only. `QisEngine`, Monte Carlo and Python `sim()` still use staged
execution; no native-runtime parity or performance result is claimed here.

## Host protocol

The host resolves all symbols from the same admitted FFI library. Before entry
it registers a live context, selects its native-handle mode, enables dynamic
execution, resets collection and assigns program shot identity as usual. The
context must be exclusively owned for this execution, with no staged consumer
or Rust TLS quantum executor. Other threads may request cooperative cancellation.

Each entry takes `(context, program_entry, callback, userdata)` and returns a
`u64` status. The callback is a C function `(userdata, result_id) -> i32`.
Its userdata may borrow the host's stack for this invocation. Registration is
thread-local, context-bound, and removed after the C guard returns. No callback
pointer is stored in a cloneable or cross-thread execution context.

The existing request path exports pending operations before calling the host.
No context lock or collector/TLS borrow is held across the callback. The callback
drains operations using `pecos_get_pending_operations_with_context`, frees the
returned collector using `pecos_free_operations`, processes all runtime-owned
work and feedback, stores the response with the explicit-context result setters,
and calls `pecos_signal_result_ready_with_context`. Only then may it return zero.
Nonzero status, missing readiness/outcome, cancellation and recorded callback-side
program errors fail the request. Readiness flags are retired after each callback.
The existing QIR typed reader still checks Boolean representability and stored
values. Native Selene requests retain their existing host-side validation
contract: the host must validate a native Boolean response as 0 or 1 before
storing it (the Selene Boolean adapter converts nonzero values to true).

The callback must not reset/reconfigure/destroy the context, replace transfer
handlers, or execute program instructions (including cached result reads).
Recursive waits, context rebinding and nested guarded entry are rejected, but
these checks are not a sandbox for arbitrary program-API reentry. A failed scope
permanently cancels its context; recovery requires a fresh context and whole-host
reset, never a switch to a staged consumer of the partial execution.

## Recovery

The C guard tracks active execution independently of the panic-transfer handler.
Nested entry is rejected before shot-start cleanup or overwriting its jump buffer.
While the host callback runs, the transfer handler is suspended; it is restored
only after callback return. The outer scalar read boundary may then transfer to
the C guard, after callback-owned Rust values have been dropped.

A Rust host must catch panics inside its C trampoline, retain the original
payload/error in its own scoped state, and return nonzero. Neither Rust unwinding
nor a nonlocal transfer may cross that callback boundary. The FFI wrapper's
failure latch takes precedence over a program `Exit` recorded by the callback,
even if the underlying C guard reports that exit as successful.
The wrapper also checks cancellation and recorded terminal errors after program
return, including errors recorded after the last callback. An ordinary program
exit remains successful unless accompanied by a callback failure or cancellation.
After successful entry the host must collect terminal operations through
`pecos_qis_get_operations` (and the matching free) and finish the existing runtime
drain/certification sequence. Entry status alone does not certify a complete shot.

Cancellation remains cooperative. Running on the host thread cannot detach an
unresponsive program; the future host integration must expose an explicit mode
choice and preserve staged execution for callers requiring its existing policy.

## Validation and next integration

FFI tests cover both entry signatures, same-thread calls, reply failures, callback
panic conversion, callback-side exit, cancellation, recursive waits, nested entry
with a live outer allocation, context rebinding, cleanup and fresh-context reuse.
A compiled LLVM fixture resolves the new exports from the admitted shared library,
uses repeated native requests and full-width replies, branches on a reply, and
checks early termination and subsequent recovery on the same thread.

The next milestone connects `QisEngine::run_with` to these exports while reusing
request lowering, runtime draining/forcing, measurement feedback, decoder calls
and terminal certification. The callback must submit emitted work through the
borrowed `QuantumSystem` driver, including all inner noise/simulator exchanges.
Require staged/scoped parity for scheduling/idling, futures, noise-generated
measurements, outputs, shot/RNG identity and recovery before enabling Python or
measuring a throughput change.
