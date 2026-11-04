# Python context lifetime in QIS simulations

Python callbacks reached from a Rust Monte Carlo worker normally attach to Python
for each call. A temporary Python thread state can be created and destroyed each
time. QIS simulations can opt into retaining that state for one logical worker's
shot loop:

```python
from pecos import Qis, selene_engine, sim

# A minimal QIS program that prepares and measures one qubit.
qis_program = Qis(
    """
declare void @setup(i64)
declare i64 @___qalloc()
declare i64 @___lazy_measure(i64)
declare void @___qfree(i64)
declare i64 @teardown()
define i64 @qmain(i64 %arg) #0 {
  call void @setup(i64 %arg)
  %q = call i64 @___qalloc()
  %r = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)
  %end = call i64 @teardown()
  ret i64 %end
}
attributes #0 = { "EntryPoint" }
"""
)
builder = (
    sim(qis_program).classical(selene_engine("selene_simple_runtime")).qubits(1).workers(1).python_worker_context(True)
)
assert builder.run(4).to_dict()["measurement_0"] == [0] * 4
# The option also propagates through build() and supported trace capture.
assert builder.build().run(4).to_dict()["measurement_0"] == [0] * 4
```

The default is `False`. This option currently accepts only QIS builders; other
program routes reject it. It applies to callbacks on the Monte Carlo worker,
including scheduled-event adapters. Callbacks on threads owned by a runtime or
plugin are outside this scope.

## Observable differences

With the option enabled, `threading.local` and `contextvars.ContextVar` values can
persist across callbacks and shots within the same logical worker job. Callbacks
that depend on that state may give different results when the number of workers
changes. Do not enable this option for callbacks requiring a fresh context per
call. It does not copy the caller's Python context into workers.

A new run creates new worker scopes. A logical worker job is not the same as a
physical pool thread: two jobs scheduled onto one thread still have separate
scopes. Empty jobs do not enter a scope. Thread-local object finalizers run at
scope teardown, rather than after every callback. Python exceptions retain the
existing simulation-error translation; exception-object identity is not promised.

The attachment guard is lexical and never stored in a cloneable engine. Engine
cloning and initial worker seeding still happen on the caller before entry.
Per-shot seeding, resets, controller execution, noise/quantum continuations, and
destruction of the worker engine happen inside the scope. Explicit `reset()` on a
built simulation acts on its template on the caller, outside any worker scope.
The GIL is released during Rust work and waits, and reacquired for Python calls.
Built QIS simulation `run`, `run_with_workers`, and `reset` calls are rejected
from worker callbacks and thread-local finalizers, preventing a wait on the calling
simulation's held mutex. External caller threads can still wait normally.
All logical worker jobs finish or fail before an error is returned; their context
guards unwind on failures and panics. There is no new cancellation interface.

The policy changes interpreter context lifetime only. Runtime scheduling,
measurement feedback, shot identities, noise parameters, and simulation backends
are unchanged. It makes no simulator-speed or model-equivalence guarantee. Use it
only while the main Python interpreter is running; subinterpreters and concurrent
interpreter shutdown are not supported by this policy. Free-threaded Python has
not been validated for this option.

## Rust extension point

`MonteCarloEngine::run_with_workers_in_scope`, the seed-report counterpart, and
`run_with_workers_from_seed_report_in_scope` accept a `WorkerScope`. Engines remain
independent of Python. The scope must invoke its owned work synchronously, exactly
once, on the calling thread and clean up before returning. The host rejects a
successful scope that skips work or moves execution to another thread, preserves
work errors, and translates panics into processing errors. An entry failure drops
the unexecuted engine outside the context. Scope implementations are trusted Rust
code, not a sandbox for untrusted callback implementations.
