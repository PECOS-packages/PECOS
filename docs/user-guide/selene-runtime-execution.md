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

Each handle introduced by an explicit qubit-allocation record stays alive through
measurements until the program releases it. A later reset or measurement on that
handle reaches the same native allocation, including across feedback continuations.
Implicit legacy handles release on measurement, even in mixed streams.

This changes explicit-allocation loops that measured without releasing: with
`.qubits(1)`, allocating and measuring a new handle each iteration rejects at the
second allocation. Release each handle when its lifetime ends, or reserve capacity
for all handles that must remain live. Measurement alone does not free an explicit
allocation. Allocating an already-live handle is invalid; admission detects this
before submitting any operation from that input, including allocations that would
otherwise precede the duplicate.

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

## Rejection and recovery

| Failure | Flat and metadata routes | Scheduled route |
| --- | --- | --- |
| Capacity, duplicate-allocation, or declared-native-set admission | Native state is unchanged; correct the input and retry | Native state is unchanged, but collection latches the rejection; reset is required |
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
