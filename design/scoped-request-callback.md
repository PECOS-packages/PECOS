# Scoped native request callback

## Status and boundary

The FFI library exports `pecos_call_qmain_with_request_callback_v1` and
`pecos_call_void_main_with_request_callback_v1`. They run a guarded program and
service synchronous result/runtime requests on the calling thread. This is the
FFI boundary and an opt-in Rust host integration. A `QisEngine` configured with
`set_scoped_execution(true)` uses it through `ControlEngine::run_with`, including
the existing Monte Carlo/HybridEngine route. Python QIS builders expose an
explicit `.scoped_execution()` opt-in; staged
execution remains the default. No private-runtime or performance result is claimed.

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
values. Native Boolean reads now also reject values other than 0 or 1 at the typed FFI
boundary, before Boolean conversion. Integer and leakage futures keep their
full-width values. This validation applies to both staged and scoped hosts.

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
Rust program-entry frames, unlike host callbacks, must use `C-unwind`: Windows
implements longjmp with stack unwinding. Those frames must still own no Rust
destructors across recovery, since Unix longjmp skips them. The outer guard
exports remain non-unwinding C boundaries.
The wrapper also checks cancellation and recorded terminal errors after program
return, including errors recorded after the last callback. An ordinary program
exit remains successful unless accompanied by a callback failure or cancellation.
After successful entry the host must collect terminal operations through
`pecos_qis_get_operations` (and the matching free) and finish the existing runtime
drain/certification sequence. Entry status alone does not certify a complete shot.

Cancellation remains cooperative. Running on the host thread cannot detach an
unresponsive program; the future host integration must expose an explicit mode
choice and preserve staged execution for callers requiring its existing policy.

## Host integration and validation

FFI tests cover both entry signatures, same-thread calls, reply failures, callback
panic conversion, callback-side exit, cancellation, recursive waits, nested entry
with a live outer allocation, context rebinding, cleanup and fresh-context reuse.
A compiled LLVM fixture resolves the new exports from the admitted shared library,
uses repeated native requests and full-width replies, branches on a reply, and
checks early termination and subsequent recovery on the same thread.

`QisEngine::run_with` shares shot setup, runtime request processing, feedback,
and terminal certification with the staged route. It moves the interface out
while the callback borrows the controller and the controlled-engine driver. The
host rejects nested program execution before TLS registration or reset. Failed
execution restores interface ownership, retains the primary typed error or panic,
and poisons the engine (including clones) until whole-host reset. Teardown
failure is retained even when a primary failure takes precedence. Direct staged
`start`/`continue_processing` calls are rejected while scoped mode is selected.

Public tests compare both routes through Monte Carlo with seeded replay,
multiple workers, Simple/SoftRZ runtimes, scheduled transport, embedded decoding,
leakage outcomes, and noise-owned simulator measurement round trips. Additional
comparisons check exact emitted wire batches for legacy, v6 and v7 transport,
measurement-dependent control, both program entry shapes, terminal tail delivery,
and caching multiple nonnative outcomes without splitting a quantum batch.

Staged orchestration may emit empty messages while its worker runs. Scoped
execution does not synthesize those worker polls. All messages actually emitted
by request/feedback/terminal processing still reach the controlled engine,
including empty or timed scheduled frames. The batch comparison excludes only
legacy empty wait messages; it is not an equivalence guarantee for arbitrary
noise models that deliberately change state on worker polls.

Before adopting the Python option downstream or making performance claims,
validate the downstream runtime and matched noise profiles on this rebuilt stack.
Runtime scheduling and idle intervals remain authoritative: the host neither
reconstructs static traces nor reschedules emitted work.
