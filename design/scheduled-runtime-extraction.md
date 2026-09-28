# Scheduled extraction draft: consumer required before landing

PR #896 remains an extraction experiment. It has no workspace execution consumer;
the next acceptance gate is a first consumer that determines admission, envelope,
backpressure and host-identity requirements. Do not land this as a completed
runtime integration milestone. Python sim() and QuantumSystem still use flat output.

The Rust trait API returns original native batches before idle insertion, retaining
integer timing, operation order, opaque payloads and both measurement namespaces.
Runtime-local shot IDs are not host run/worker identities. Source metadata and
non-native gates reject explicitly. API details live in rustdoc.

## Changes following external review

The former 64-batch per-call limit failed on normal 200-gate programs, including
soft-RZ schedules held until a terminal barrier. Aggregate batch/operation/payload
caps were removed, following the review's alternative to backpressure. No additional
barriers or subdivision of a native batch are introduced. This matches the flat
path's aggregate allocation behavior: output memory grows with the drained schedule.
A bounded streaming interface must be designed with the actual consumer.

Individual callback batches still have explicit limits: 4096 operations and 256 KiB
of opaque bytes. These limit allocation/read exposure within a native callback;
they are not a bound on total extraction memory. Oversize indivisible batches fail
and require reset. Suitability of these limits is also subject to consumer review.

Scheduled shot completion and replacement now require a successful terminal drain
since the last submitted input. Reset is the explicit way to abandon work. This
tracks scheduler draining, not downstream execution of returned batches.

Real native-runtime regressions cover the immediate and deferred 200-gate cases,
completion without a drain, renewed submission after draining, and clone isolation.
The clone regression is checked by temporarily disabling the poison guard: it must
fail rather than obtain its error from a nonexistent native library. Public fixtures
contain no device-specific event interpretation.
