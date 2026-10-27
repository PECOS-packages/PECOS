# Selene runtime execution

These rules apply to the flat, metadata and scheduled lowering routes of
`SeleneRuntime`. Python QIS `sim()` requires `.qubits(N)`; the inference rules below
concern Rust and direct-runtime callers. For timing and noise on the scheduled
route, see [Scheduled idle simulation](scheduled-idle-simulation.md). For operation
capture, see [Runtime QIS tracing](runtime-qis-tracing.md).

## Shot identity and randomness

`QisEngine` calls runtime `shot_start` with IDs `0, 1, 2, ...` per engine
instance. Reset and reseeding retain the sequence; a template clone starts its
own sequence at zero. A failed `shot_start` consumes its ID because the plugin
may already have observed it. Exhaustion returns an error instead of wrapping.
Scheduled envelopes use that same runtime ID, including terminal batches.
Operation trace `shot_index` remains one-based for compatibility.

These runtime IDs are local to each worker engine. Monte Carlo derives seeds
from the global shot index independently of worker partitioning, and its host
`ShotContext` identifies the run, worker and local shot separately. Changing the
runtime ID origin does not change PECOS's seed derivation or consume extra RNG
draws. A plugin that uses the ID itself to choose a random stream can nevertheless
produce different results; record the PECOS revision when comparing old runs.
Selene's own driver uses component base seeds plus the shot ID, so matching IDs
alone does not establish identical random streams between the two drivers.

### Program-visible shot indices

The QIS `get_current_shot()` import and `selene_get_current_shot` adapter both
return the job-wide program index. Monte Carlo assigns `0..shots-1` independently
of worker partitioning, after reset and before program execution. Every new run
starts at zero; seed-report replay uses each worker's recorded `first_shot`
offset, including nonzero offsets. The value survives measurement feedback and
the compiled wrapper's local shot-start hook. Each invocation still executes
one shot, so `selene_shot_count` returns one.

Direct Rust `QisEngine` callers default to the engine's local runtime ID. Hosts
can assign the next invocation with `ClassicalEngine::set_program_shot` after
reset. The override is consumed when an admitted start is attempted, including
execution failure. A successful reset or template clone clears it. The override
does not change runtime callback IDs, trace indices or seed derivation.

Program-seeded Guppy RNGs use their explicit seed separately from simulator and
noise seeds. `RNG(seed)` restarts the same stream on each invocation;
`RNG(seed + get_current_shot())` selects a shot-dependent stream that is invariant
to worker partitioning. Querying the index does not consume RNG state. This also
supports parameter-array selection by shot index. Program indices alone do not
make simulator/noise streams identical between PECOS and Selene.

The legacy `___set_random_index` import and broader WASM RNG integration remain
unsupported by this change. See the remaining work in
[issue #941](https://github.com/PECOS-packages/PECOS/issues/941).

### Guppy RNG compatibility coverage

The live Selene comparisons in `tests/guppy/test_qis_classical_rng.py` cover both
Guppy RNG implementations:

- `guppylang.std.qsystem.random.RNG`: signed integers, bounded sampling, floats,
  random angles, Clifford angles, forward/backward jumps, shuffle and discrete
  distributions. Tests include signed seed boundaries, large jumps, rejection
  sampling, handle recreation and multiple live handles.
- `guppylang.std.random.seeded_pcg32`: native Guppy integer and bounded draws,
  with interleaved generators and a platform RNG to check state independence.

The platform RNG has shared state within one program execution; multiple handles
must not be assumed to provide independent streams. Native Guppy PCG32 values
keep their state locally. See the [Guppy RNG guide](https://docs.quantinuum.com/guppy/language_guide/random.html).

Comparisons preserve repeated output labels so every draw in a loop is checked.
Additional tests cover shot-dependent seeds, quantum measurement feedback,
invalid distribution weights, oversized bounds and a valid simulation after an
aborted invocation. These are functional comparisons using the installed Guppy,
compiler and Selene versions; they do not establish a speed advantage or equality
of simulator/noise RNG streams.


The expanded RNG test file passed 44 tests in each of these local environments
on October 18, 2026 (live Selene controls plus PECOS validation/recovery tests):

| Stack | Guppy | Selene | HUGR-QIS compiler |
| --- | --- | --- | --- |
| Pinned | 1.0.1 | 0.3.4 | 0.4.3 |
| Isolated newer stack | 1.1.3 | 0.3.5 | 0.5.1 |

Both used selene-core 0.3.3, HUGR 0.18.8 and tket 0.15.11. The newer-stack
experiment used PECOS source imports and the same local native extension; it does
not change package dependency constraints or certify the whole package on those
versions. It emits compiler API deprecation warnings.

Boundary coverage includes signed jump extremes, unsigned native seed extremes,
empty/singleton shuffles, deterministic distributions and reset/reuse after an
RNG error following measurement feedback. The shuffle tests inspect length,
elements and subsequent draws: Selene rejects empty-array output, and PECOS
presents a single-element result as a scalar, so raw output layouts differ.

Known limitations and intentional validation differences:

- PECOS rejects nonpositive platform bounds with a Python exception. In both
  tested Selene versions, zero aborts the reference subprocess and negative
  bounds are reinterpreted as unsigned. Those reference behaviors are outside
  the valid input domain and are not a compatibility target.
- Native Guppy PCG32 handles a zero bound by exiting the shot. PECOS exposes its
  reason/code through `program_exits`, as described below. A missing sample is
  not evidence of a successful RNG draw.
- This suite does not establish exhaustive behavior for all invalid inputs,
  other dependency versions, WASM integration or simulation speed.

## Per-shot program exits

Ordinary Guppy exits (QIS codes 0 through 999) finish the current shot and preserve
its outputs recorded before the exit. Codes 1000 and above fail the job, matching
Selene's launcher and result parser. Panics and other execution failures raise an
error; they do not return an ordinary-exit metadata entry.

**Compatibility change:** code 1000 previously finished a shot in PECOS. It now
fails the job. Programs that intentionally request a recoverable exit must use a
code below 1000. Guppy's prose documents signals through 1000, but live Selene
0.3.4/0.3.5 and the inspected Selene source use the strict boundary. See
[the launcher](https://github.com/Quantinuum/selene/blob/458d640dc4cf84a7a7988ea65ce5a9f7bcb8c511/selene-ext/interfaces/helios_qis/c/src/interface.c#L63)
and [issue #1372](https://github.com/PECOS-packages/PECOS/issues/1372).
Negative QIS codes also fail: the C adapter passes their 32-bit pattern to Selene
as an unsigned value. PECOS retains the signed QIS code in its exception text,
while Selene reports the unsigned value; compare the bit pattern at these ABI
edges rather than requiring identical exception strings.

```python
from guppylang import guppy
from guppylang.std.platform import exit as guppy_exit
from guppylang.std.qsystem.utils import get_current_shot
from pecos import Guppy, selene_engine, sim


@guppy
def program() -> None:
    if get_current_shot() == 1:
        guppy_exit("done", 1)


shots = sim(Guppy(program)).classical(selene_engine()).qubits(1).run(4)
print(shots.program_exits)
assert shots.program_exits == [None, {"code": 1, "message": "done"}, None, None]
```

`program_exits` has one entry per result row, in the same order as the shot data.
`None` means the engine reported no ordinary exit; it is not a capability claim
for other engines. A dictionary contains the original integer code and message.
The dictionaries are copies. Diagnostics remain available even if branch-dependent
outputs make `to_dict()` or `to_shot_map()` reject inconsistent registers.
Successful `to_shot_map()` conversions preserve `program_exits` and the shot count,
including shots that produced no registers.

Diagnostics are separate from named outputs. `to_dict()`, binary/display output,
and register-oriented JSON export retain their existing data-only formats; no
exit markers enter detector or observable columns. Selene presents an ordinary
exit as `("exit: <message>", code)` in its per-shot results. Compare that marker
with PECOS metadata rather than expecting identical result dictionaries.
For example, native `seeded_pcg32(...).next_int_bounded(nat(0))` produces no sample
and reports code 1 with message `PCG32.next_int_bounded: bound must be positive`.

Rust callers use `Shot.program_exit: Option<ProgramExit>` or
`ShotMap::program_exits()`. Serde serialization preserves this metadata; older
serialized results without it remain readable. Rust struct literals that specify
`Shot.data` must also initialize `program_exit` (or use `..Default::default()`).
Reset, template cloning and subsequent shots do not inherit an earlier exit.
Both the direct QIS termination entry and Selene's termination adapter report
metadata. The standalone `selene_print_exit` diagnostic printer does not terminate
a shot, and the `selene_exit` instance-teardown hook does not represent a program
exit; their behavior is unchanged.

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

Metadata matching retains source records, including unlabelled guards that stop
older identical operations from consuming later labels. Emission retires earlier
unmatched records touching the emitted operation's qubits, including absorbed
virtual Z anchors; required labels still fail when retired without an emission.
Recording labels and releasing a qubit add no scheduling barriers. Records remain
queued across physical-slot reuse until emission or native shot completion.

Flat and metadata routes release queued work on an Idle's qubit before emitting
the Idle. Scheduled extraction accepts decomposable gates but continues to reject
source Idle and trace metadata. Measurement requests remain lazy: the runtime
determines when they execute, and PECOS forces a specific result only when the
program reads it. Both ordinary and leakage-aware reads use the native
`force_result` contract. An unsatisfiable read fails promptly; PECOS does not
insert a compatibility barrier that could change the runtime's timing. The pinned
public SoftRZ runtime cannot force pending leakage-aware results; PECOS exposes
that failure just as Selene does. Leakage-aware results already released by the
runtime remain readable.
PECOS polls after each native submission. Each emitted measurement batch receives
feedback before the next extraction or source submission, including between
native calls within a decomposed gate. Native future getters determine program
read values; the host does not substitute cached simulator outcomes. See [lazy measurements](runtime-qis-tracing.md#lazy-measurements).

At program completion, all Selene routes poll ready work, call the runtime end
hook exactly once, and execute its final operations before certifying completion.
The host adds no global barrier to flush work the runtime has chosen to defer.
Unobserved measurements may remain unemitted if the runtime discards them at end.
Required trace labels still fail if their physical operation was not emitted.

Post-end measurements receive feedback, and PECOS processes any work that feedback
releases before completion. Scheduled output keeps its batches, timestamps,
duration, order and runtime shot ID. The same noise model and simulator process
this output; the host does not construct a new schedule for it.

`QisRuntime::finish_operations` and `finish_scheduled_operations` expose final
delivery to host implementations. Direct `SeleneRuntime::shot_end` callers receive
an error if native end emits work they would otherwise discard. End-hook,
extraction, admission and custom-event failures remain sticky until reset; no
completed trace is emitted on failure. Native final extraction uses the existing
per-batch and aggregate transport retention bounds.

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

## Preparation and source metadata

The Selene hosting path submits the program's allocation and reset calls as
written. It does not insert a reset before first use or physical-slot reuse.
The runtime decides which physical operations those calls produce. Program
resets receive the simulator's normal preparation noise and the duration assigned
by the runtime. Guppy qubit construction and converted QIR supply explicit resets.
Hand-written QIS that requires initialization should supply one too.

PECOS's direct QIS lowering path retains its first-use preparation rule: a used
lifetime receives one reset before its first quantum operation unless that
operation is already a reset. This is separate from Selene runtime hosting.

Source traces retain the program's operations; lowered traces record native
emissions. Scoped metadata attaches to the next compatible source operation and
survives feedback continuations. Native emission order on each qubit determines
retirement: earlier unmatched records touching its qubits retire, while records
on disjoint qubits remain live. Labels never transfer from an absorbed operation
to a later pulse. `source_lowering_required=true` fails if its physical emission
is missing. Optional labels on discarded work expire at native completion.
Records clear at shot start and reset, without forcing runtime work.

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
by the program. Successful end allows instance reuse without requiring a forced
drain. A continuous runtime clock is preserved, including time advanced by queued
work the runtime later discards; the next trace uses the configured clock model.
This does not provide automatic mid-shot growth or native snapshots.
