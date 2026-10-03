# Selene runtime execution

These rules apply to the flat, metadata and scheduled lowering routes of
`SeleneRuntime`. Python QIS `sim()` requires `.qubits(N)`; the inference rules below
concern Rust and direct-runtime callers. For timing and noise on the scheduled
route, see [Scheduled idle simulation](scheduled-idle-simulation.md). For operation
capture, see [Runtime QIS tracing](runtime-qis-tracing.md).

## Allocation lifetimes and capacity

Every qubit handle stays alive through ordinary and leakage-aware measurements.
An explicit handle lives from `AllocateQubit`; an implicit legacy handle lives from
its first quantum operation. Both live until `ReleaseQubit` or shot end. Later
operations on a live handle reach the same native allocation, including across
feedback continuations.

Duplicate live allocation and use after release without re-allocation are rejected
on every lowering route, including direct lowering and all Selene routes. This
also rejects explicit allocation of an already-live legacy static handle. Releasing
a handle that is not live is a no-op and does not prevent its legacy first use.

This changes explicit-allocation loops that measured without releasing: with
`.qubits(1)`, allocating and measuring a new handle each iteration rejects at the
second allocation. Release each handle when its lifetime ends, or reserve capacity
for all handles that must remain live. Measurement alone does not free any
allocation. Allocating an already-live handle is invalid; admission detects this
before submitting any operation from that input, including allocations that would
otherwise precede the duplicate.

Compatibility: a legacy program that touches more distinct static qubit handles
than the configured capacity and relied on measurement to recycle slots is now
rejected at admission. Release handles explicitly when their lifetimes end.

Capacity admission counts all handles still live from earlier inputs, including
measured handles not yet released. Set capacity to the maximum simultaneously live
allocation count before running a dynamic program. Scheduled extraction requires
an explicit nonzero capacity. Flat and metadata routes can infer it from a complete
preloaded collector or the first input needing qubits. Result-only inputs and empty
barriers defer initialization when no capacity is configured or inferred.

The plugin ABI has no state-preserving resize. True capacity growth rejects before
native submission; configure sufficient capacity before execution, or reset and
start a new shot. A larger legacy logical ID alone does not require more native
capacity within a shot: peak live use determines whether a continuation fits.
Conservative initial sizing still uses legacy indices and retains that estimate
across reset, so sparse IDs can enlarge the next shot's inferred capacity. An
explicit capacity avoids this overestimate.

## Prep at lifetime start

`QisEngine` starts every used qubit lifetime with exactly one prep (`Reset`, lowered
as `PZ`). The prep establishes |0> and receives the noise model's normal prep noise,
including when a released simulator slot is reused. Allocation defers the prep
until the first quantum operation on that handle. If that operation is a program
reset, it counts as the lifetime's prep; otherwise the engine inserts a prep before
it. A legacy static handle without an allocation receives the same treatment at
its first use in the shot. An allocated handle released without use needs no prep.

This rule applies to direct, runtime-provided and scheduled lowering. Prep state
persists across feedback continuations. Release ends the lifetime; re-allocation starts a
new one. Later program resets remain preps with normal prep noise. `SeleneRuntime`
lowers the resets supplied by `QisEngine` and does not add preps itself.

Source operation traces retain the program's operations. Inserted preps appear
only in lowered output, and source trace metadata stays with its program operation.
Trace metadata is local to each chunk and does not carry across feedback continuations.

Compatibility: noisy results change for QIR and legacy-handle programs, which now
receive a prep at lifetime start. Results also change for programs on the direct
lowering path that previously received a second prep on allocation in addition to
their first program reset, such as Guppy programs. These now receive just one prep
at lifetime start.

## Rejection and recovery

| Failure | Flat and metadata routes | Scheduled route |
| --- | --- | --- |
| Capacity, duplicate-allocation or use-after-release admission | Native state is unchanged; correct the input and retry | Native state is unchanged, but collection latches the rejection; reset is required |
| Input submission, feedback or terminal draining | Reset required | Reset required |
| Native shot-start callback | Reset required, including failure during lazy initialization | Reset required |

Scheduled collection conservatively latches every error after collection begins.
Initial checks of source shape, identifiers and finite angles happen before
collection and do not themselves poison the runtime. Native state remaining intact
after admission rejection does not imply that a scheduled shot remains usable.

The mutation guards also latch validation errors encountered inside submission or
feedback, even if the error precedes the first native callback in that call. For
example, a conflicting trace-metadata label or a non-Boolean ordinary measurement
outcome requires reset. Correcting and retrying that call alone does not recover
the shot. This conservative policy avoids relying on how far a call progressed;
these checks do not currently provide a separate whole-input preflight contract.

Errors and caught panics in guarded calls block continuation and shot completion
until a successful reset. Failure remains latched across clones. A failed lazy
shot-start callback cannot be bypassed merely because the plugin was initialized.

## Clones and shot boundaries

Cloning an initialized native runtime requires reset before use. The plugin ABI
cannot copy live allocations, pending results or scheduler state. Configuration
templates without a native instance remain cloneable for fresh shots. The main
`QisEngine` execution path resets before each shot.

Shot lifecycle cleanup remains the plugin's responsibility. The public runtimes
clear allocations at `shot_end`, including measured handles not explicitly released
by the program. This does not provide automatic mid-shot growth or native snapshots.
