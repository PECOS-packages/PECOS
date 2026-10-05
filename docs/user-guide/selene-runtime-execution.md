# Selene runtime execution

These rules apply to the flat, metadata and scheduled lowering routes of
`SeleneRuntime`. Python QIS `sim()` requires `.qubits(N)`; the inference rules below
concern Rust and direct-runtime callers. For timing and noise on the scheduled
route, see [Scheduled idle simulation](scheduled-idle-simulation.md). For operation
capture, see [Runtime QIS tracing](runtime-qis-tracing.md).

## Native gate lowering

All Selene runtimes receive gates through their native entry points. Flat,
metadata and scheduled lowering share the same decomposition table for H, Pauli,
phase, controlled and rotation gates. PECOS simulates only the operations emitted
by the plugin: RXY1Q, RZ, RZZ, RXYXY2Q, reset and measurement, together with the
existing idle/timing operations. Each runtime declares its accepted ABI gate
entry points. Generic plugins and the simple runtime accept `{rxy, rz, rzz, rpp}`;
the soft-RZ constructor (including selection by name) declares `{rxy, rz, rzz}`.
Reset, measure, and measure_leaked are mandatory lifetime operations.

When rpp is available, RXYXY2Q is submitted natively. Otherwise PECOS lowers it
using four RXY pulses and one RZZ: on each qubit apply RXY(-pi/2, phi+pi/2),
then RZZ(theta) on the pair, then RXY(pi/2, phi+pi/2) on each qubit. Noise applies
separately to these emitted native gates. Every input is checked against the
declared set before any of its operations are submitted. If no registered
lowering fits, the error identifies the source operation, qubits, runtime,
declared set, missing entry points, and how to declare a custom set.

Custom Rust plugins declare their set explicitly:

```rust
use pecos_qis::{RuntimeNativeGate, RuntimeNativeGateSet, SeleneRuntime};
let runtime = SeleneRuntime::new("libcustom_runtime.so").with_native_gate_set(
    RuntimeNativeGateSet::new([RuntimeNativeGate::Rxy, RuntimeNativeGate::Rz, RuntimeNativeGate::Rzz]),
);
```

This declaration survives cloning and reset. The full-set generic default is
the plugin ABI contract; it does not infer capabilities from a library filename.
Python builders expose `native_gates=["rxy", "rz", "rzz"]` on
`selene_engine(...)`, `qis_engine().selene_runtime(...)`, and the Rust binding's
`selene_runtime_plugin(...)`. A generic Python plugin object can also expose a
`native_gates` property; an explicit argument takes precedence. Omission uses the
known runtime's declaration or the full ABI contract for a generic plugin.

The ABI's emitted gate callbacks map to PECOS gates independently of the accepted
ingress set; the complete association is documented on
`SeleneRuntimeGetOperationInterface` in `selene_runtime.rs`. Other operations can
arrive only as custom events. Unhandled events fail with runtime and batch/operation
context plus handler/Capture guidance. Incompatible plugin API versions report the
plugin version, PECOS's supported version range, and a rebuild instruction.

Noise applies per emitted native gate. Programs that previously passed non-native
gates directly to the simulator therefore have different noisy results. Soft-RZ
can fold virtual Z rotations into later pulse axes without emitting an RZ gate.

A decomposed gate's source metadata labels exactly one native: its first non-RZ
operation, or its first RZ if the sequence contains only Z rotations. On runtimes
that fold virtual Z, labels on Z-only gates (Z, S, Sdg, T, Tdg, and RZ-only
sequences) have no emitted gate. Optional labels are dropped; labels with
`source_lowering_required=true` fail the shot. These labels never move to a later
pulse, which may have its own label.

Metadata matching retains only outstanding labels and temporary empty guards on
their qubits. Emission retires earlier unmatched records touching the emitted
operation's qubits, including absorbed virtual Z anchors; required labels still
fail when retired without an emission. Unlabelled flat input creates no matching
records. When tracking begins on a qubit without an outstanding label, PECOS uses
a local barrier and drains its earlier untracked work before registering the new
label. This boundary prevents an older identical pulse from taking the new label,
including across flat-to-metadata transitions; it can release queued work earlier
than an otherwise identical unlabelled program.

Flat and metadata routes release queued work on an Idle's qubit before emitting
the Idle. Scheduled extraction accepts decomposable gates but continues to reject
source Idle and trace metadata. Leakage-aware measurements release queued work on
the measured qubit so their results are available even when a plugin's result
forcing only recognizes ordinary measurements.

At program completion, flat and metadata routes lower a final global barrier
through the normal metadata-aware path. Any queued native gates become one final
batch, retaining their source labels; an empty drain adds no batch. Certification
then checks for unexpected late operations and still fails if any remain. Scheduled
execution retains its own terminal batch drain.

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
For example, static `H(0); AllocateQubit { id: 0 }` rejects because handle 0 is
already live. `AllocateQubit { id: 0 }; H(0)` is valid; mixing static and explicitly
allocated handles is allowed when it respects these lifetime rules. Programs do not
choose allocated ids: `__quantum__rt__qubit_allocate` issues them in order from 0,
in the same index space as static handles, and measurement does not end a static
handle's lifetime. A program such as `x(0); m(0); qubit_allocate()` is therefore
rejected, because the allocator returns 0 while static handle 0 is live. Keep static
handles out of the ids the allocator will issue, or use only allocated qubits.

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

For static integer handles that are never explicitly allocated, prep occurs at
first use, even if the program touches that qubit much later than other qubits.
Its prep noise, crosstalk and idle timing therefore start at that first use.
This differs from Quantinuum's `qir_qis` converter: version 0.1.8 calls
`qir_qis.init_qubit(i)` for every one of `required_num_qubits` at program entry.
Each call emits `___qalloc` and `___reset` before the first gate. Converted QIR
already has explicit allocations and resets and is unaffected by the static-handle
first-use rule; that distinction concerns hand-written integer-handle QIS.

The QASM engine adds no implicit lifetime prep. Its explicit `reset` parses as
`PZ` ([parser](../../crates/pecos-qasm/src/parser/operations.rs)) and is emitted as
such ([engine](../../crates/pecos-qasm/src/engine.rs)). With prep noise, the same
circuit can therefore produce different results through QASM and static-handle QIS.

Inserted preps are submitted to the runtime as resets and take the reset duration
it assigns. PECOS's convenience constructors in
[`selene_runtimes.rs`](../../crates/pecos-qis/src/selene_runtimes.rs) default gate
and reset durations to zero. Runtimes configured through `with_plugin_config`
can assign nonzero durations, including soft-RZ, so inserted preps can shift later
operations' start times.

This rule applies to direct, runtime-provided and scheduled lowering. Prep state
persists across feedback continuations. Release ends the lifetime; re-allocation starts a
new one. Later program resets remain preps with normal prep noise. `SeleneRuntime`
lowers the resets supplied by `QisEngine` and does not add preps itself.

Source operation traces retain the program's operations. Inserted preps appear
only in lowered output, and source trace metadata stays with its program operation.
On the direct route, qubit-scoped trace metadata attaches to the next operation
on that qubit, matching Selene, rather than to the next lowered gate. Dangling
metadata with no following compatible operation remains local to its source chunk:
the direct route drops it and Selene rejects it. Once metadata attaches to a
submitted operation, Selene retains the record across feedback continuations until
that operation emits. Inserted preps carry no source labels; untracked preps need
no matching records.

Native emission order on each qubit determines retirement: when a later native
operation emits, earlier unmatched records touching its qubits retire; records
on disjoint qubits remain live. This includes absorbed RZ records. Labels are
retired without transferring them; `source_lowering_required=true` instead fails
the shot if no physical emission exists. Release drains the ending lifetime
before freeing the native slot and retires its remaining virtual RZ records.
Records are cleared at shot start and reset.

The flat/metadata terminal policy rejects gates emitted after the final lowered
batch ([`QisEngine::verify_runtime_drained`](../../crates/pecos-qis/src/ccengine.rs)).
A labelled gate emitted by terminal drain therefore fails loudly, even without
`source_lowering_required`; terminal verification cannot carry that label to a
simulated gate. After the forced drain, outstanding virtual RZ records are retired
unless they require an emission; any other outstanding source operation fails.

Compatibility: noisy results change for QIR and legacy-handle programs, which now
receive a prep at lifetime start. Results also change for programs on the direct
lowering path that previously received a second prep on allocation in addition to
their first program reset, such as Guppy programs. These now receive just one prep
at lifetime start.

## Rejection and recovery

| Failure | Flat and metadata routes | Scheduled route |
| --- | --- | --- |
| Capacity, duplicate-allocation, use-after-release, or declared-native-set admission | Native state is unchanged; correct the input and retry | Native state is unchanged, but collection latches the rejection; reset is required |
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
