# Decode a Guppy QEC experiment with idle noise

Use this workflow when you have a Guppy QEC experiment and want to estimate its
decoded logical error rate under circuit-level gate and idle noise. The example
is a hand-written three-qubit repetition-code memory, small enough to read in
full: everything here applies unchanged to larger hand-written programs.

The stages are:

1. Define the code in Guppy
2. Define detectors and observables
3. Generate the DEM with gate and idle noise
4. Sample — from the DEM, or by simulating the program
5. Decode the samples and compute logical error rates

Each stage builds on the previous one; the code blocks form a single script when
read in order.

## 1. Define the code in Guppy

The program prepares three data qubits in the logical `|0>` state, extracts the
two parity checks twice with fresh ancillas, and reads out the data qubits in
the Z basis. Every measurement that a detector will reference is tagged with
`result(...)`, so detectors can be written by name instead of by counting
positions.

Two constraints shape the program: quantum control flow must be static, so the
rounds are written out rather than looped, and the circuit must be Clifford.
Ancillas are freshly allocated each round because `measure()` consumes its
qubit. For larger codes, the built-in generators in
[QEC with Guppy](../user-guide/qec-guppy.md) produce this structure for you.

```python
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit


@guppy
def rep_code_memory() -> None:
    # Data qubits, prepared in the logical |0> state.
    d0, d1, d2 = qubit(), qubit(), qubit()

    # Round 0: the two Z-parity checks, (d0, d1) and (d1, d2).
    a0, a1 = qubit(), qubit()
    cx(d0, a0)
    cx(d1, a0)
    cx(d1, a1)
    cx(d2, a1)
    result("s0_r0", measure(a0).read())
    result("s1_r0", measure(a1).read())

    # Round 1: same checks, fresh ancillas (measure() consumes its qubit).
    b0, b1 = qubit(), qubit()
    cx(d0, b0)
    cx(d1, b0)
    cx(d1, b1)
    cx(d2, b1)
    result("s0_r1", measure(b0).read())
    result("s1_r1", measure(b1).read())

    # Final data readout in the Z basis.
    result("m0", measure(d0).read())
    result("m1", measure(d1).read())
    result("m2", measure(d2).read())
```

## 2. Define detectors and observables

A detector is the **parity** of the measurements it references, chosen so that
it is deterministic in the absence of noise:

- `D0`, `D1` — the first-round checks, deterministic because the data qubits
  start in `|000>`.
- `D2`, `D3` — each second-round check compared against the same check in the
  first round.
- `D4`, `D5` — each second-round check compared against the corresponding parity
  of the final data readout.

The observable is the logical Z value, which for this code is any single data
qubit measurement.

A bare string names a tagged measurement. `rec[-k]` refers to one by position
in the canonical Guppy measurement stream, as in Stim, and
`result_ref("tag", occurrence=...)` is the explicit form when you need its
extra selectors.

Detectors and observables are not named: each one's DEM label is its position in
the list, so `detectors[0]` is `D0` and `observables[0]` is `L0`. That is the
identity the decoders and the DEM text use. A tag that no `result()` call emits is a hard error, so
mistyped names fail loudly rather than silently dropping a detector term; in
larger programs you can also define each tag once as a module-level constant
and use it in both places, passing it to Guppy as `result(comptime(TAG), ...)`.

<!--continuation-->
```python
from pecos.qec import Detector, Observable

detectors = [
    Detector("s0_r0"),
    Detector("s1_r0"),
    Detector("s0_r0", "s0_r1"),
    Detector("s1_r0", "s1_r1"),
    Detector("s0_r1", "m0", "m1"),
    Detector("s1_r1", "m1", "m2"),
]
observables = [Observable("m0")]
```

## 3. Generate the DEM with gate and idle noise

`with_idle_after_2q(1.0)` inserts an idle of that duration on both qubits after
every two-qubit gate; traced `Idle` gates are stripped first by default, so
runtime-emitted idles are not double-counted.

That default matters because the trace need not be idle-free. `with_runtime(...)`
selects the Selene runtime plugin that lowers and unrolls the Guppy program into
the QIS trace this TickCircuit is built from — so the runtime does not decorate
the trace, it produces it. A runtime that models timing emits its own idle gates
as part of that lowering, reflecting real scheduling rather than the uniform
convention inserted here. Setting `with_idle_after_2q(...)` therefore
implies stripping first, so the two conventions cannot stack. To keep a runtime's
own idle placement instead, simply omit `with_idle_after_2q(...)` — stripping is
off unless insertion asked for it — and the idle-noise families apply to whatever
idles the runtime emitted. `with_strip_traced_idles(...)` overrides that pairing
in either direction when you want it stated explicitly.

The linear family uses a custom Z-biased distribution, keeping smaller X and Y
memory errors while making dephasing dominant; its weights are an additive
probability distribution and must sum to 1. The sine-squared family uses Z only,
because the sine-law dephasing remnant is Z by nature — its default is symmetric
across X, Y, and Z, so the single-axis choice is spelled out explicitly. See
[Idle Noise](../user-guide/dem-from-guppy.md#idle-noise) for the full family and
model semantics.

`DetectorErrorModel.builder()` configures the run through chained setters and
returns the DEM together with the audit trail and the result-column evaluator
used in stage 4b. A `NoiseParameters` instance carries the entire noise
configuration as one argument.

The one-call forms `DetectorErrorModel.from_guppy(...)` and
`build_dem_from_guppy(...)` remain available and run this same pipeline; they
take the noise settings as individual keyword arguments instead.

<!--continuation-->
```python
from pecos import NoiseParameters
from pecos.qec import DetectorErrorModel

noise = (
    NoiseParameters()
    .with_p1(0.002)
    .with_p2(0.02)
    .with_p_meas(0.02)
    .with_p_prep(0.02)
    .with_p_idle_linear(0.01, {"X": 0.25, "Y": 0.25, "Z": 0.5})
    .with_p_idle_sin_squared(0.03, {"Z": 1.0})
)

dem_build = (
    DetectorErrorModel.builder()
    .with_program(rep_code_memory)
    .with_qubits(7)
    .with_detectors(detectors)
    .with_observables(observables)
    .with_noise(noise)
    .with_idle_after_2q(1.0)
    .build()
)
dem = dem_build.dem

assert dem.num_detectors == 6
assert dem.num_observables == 1
print(f"detectors: {dem.num_detectors}, mechanisms: {dem.to_string().count('error(')}")
```

The DEM has several text forms, one per decoder appetite. `to_string()` returns
Stim-format text with raw hyperedges, which BP+OSD and Tesseract consume
directly. PyMatching requires a graph-like model, so it gets the terminal
projection from `to_string_terminal_graphlike_decomposed()`; the source-informed
`to_string_source_graphlike_decomposed()` form is used for Tesseract below.

<!--continuation-->
```python
raw_text = dem.to_string()
terminal_graphlike_text = dem.to_string_terminal_graphlike_decomposed()
source_graphlike_text = dem.to_string_source_graphlike_decomposed()

assert all("error(" in text for text in (raw_text, terminal_graphlike_text, source_graphlike_text))
```

## 4a. Sample the DEM

`to_sampler()` draws detector events and observable flips directly from the
error model, without simulating the circuit. `get_syndrome()` returns one shot's
detector bits and `get_observable_flips()` the actual logical flips that shot
incurred — the ground truth that decoder predictions are scored against.

<!--continuation-->
```python
sampler = dem.to_sampler()
batch = sampler.sample_batch(2000, seed=1)

assert batch.num_shots == 2000
for shot in range(2):
    syndrome = batch.get_syndrome(shot)
    observable_mask = batch.get_observable_flips(shot).mask
    assert len(syndrome) == dem.num_detectors
    print(f"shot {shot}: syndrome={syndrome}, observable_mask={observable_mask}")
```

## 4b. Or generate shots by simulating the program

Instead of sampling the error model, you can execute the Guppy program itself
under a noisy simulator and score those shots against the same DEM.
`dem_build.evaluate_result_columns()` maps the run's tagged result columns into
the same (detector events, observable flips) pairs a DEM sample carries, so
either source can feed the decoders.

The gate noise below mirrors stage 3, including the idle families. With the
default runtime, which emits no idle gates of its own, `with_idle_after_2q`
adds an idle site on each two-qubit gate operand, the same placement the DEM
pass uses.

The idle families take the same rates and the same model dictionaries on both
sides, so stage 3's settings carry over verbatim -- no unit conversion. Each
family is named for its own law, so there is no mode flag to set either.

The simulator still samples one linear event and then picks an axis, while the
DEM emits independent per-axis mechanisms; the DEM builder converts between the
two so both describe the same Pauli channel.

There are two ways to place idle sites after two-qubit gates, and they are
**alternatives, not steps**. Using both double-counts:

- `general_noise().with_idle_after_2q(d)` -- the noise model applies idle faults
  at each two-qubit gate's operands as it decorates the stream. This is what the
  run below uses.
- `TickCircuit.insert_idle_after_two_qubit_gates(d)` -- a circuit pass that
  inserts real `Idle` gates, which the noise model then treats like any other
  idle.

A caveat if you supply a runtime plugin via `with_runtime(...)`. Unlike the DEM
builder, a noise model cannot remove gates -- it only decorates a gate stream. So
a runtime that emits its own `Idle` gates gets idle noise applied to those *and*
at the after-2q sites, double-counting where the DEM counts once. Lowering the
program yourself lets you strip first, exactly as the DEM builder does:

<!--continuation-->
```python
from pecos.tracing import trace_program_to_tick_circuit

# The QIS trace lowers and unrolls the Guppy program; pass runtime=... to select
# a Selene runtime plugin, which may schedule idles of its own.
tick_circuit = trace_program_to_tick_circuit(rep_code_memory, 7)

# strip_idles() removes only duration-carrying Idle gates, clearing the
# runtime-emitted convention before insertion. strip_identities() is a separate
# unitary optimization for I and zero-angle rotations; it also removes their
# gate-noise locations, so call it only when that optimization is intended.
tick_circuit.strip_idles()
```

`sim()` does not yet accept a `TickCircuit` (PECOS #444), so today this is how to
inspect the lowered circuit rather than a path into the simulator. The run below
uses the default runtime, which emits no idles, so `with_idle_after_2q` is the
only convention in play and nothing is double-counted.

<!--continuation-->
```python
from pecos import general_noise, selene_engine, sim, stabilizer

# The same gate and idle noise the DEM was built with.
noise = (
    general_noise()
    .with_p1(0.002)
    .with_p2(0.02)
    .with_p_meas(0.02)
    .with_p_prep(0.02)
    .with_p_idle_linear(0.01, {"X": 0.25, "Y": 0.25, "Z": 0.5})
    .with_p_idle_sin_squared(0.03, {"Z": 1.0})
    .with_idle_after_2q(1.0)
)

results = sim(rep_code_memory).classical(selene_engine()).quantum(stabilizer()).qubits(7).noise(noise).seed(42).run(500)

columns = results.to_shot_map().to_dict()
sim_shots = dem_build.evaluate_result_columns(columns)

assert len(sim_shots) == 500
```

## 5. Decode the samples and compute logical error rates

Pass the same sampled batch to `batch.decode(...)` with a typed specification
for each decoder. A shot counts as a logical error when the predicted
observable flip disagrees with the flip the sample actually carried. The
returned `DecodeResult` supplies the aggregate count and rate directly.

<!--continuation-->
```python
from pecos.decoders import bp_osd, pymatching, tesseract

pymatching_result = batch.decode(
    terminal_graphlike_text,
    pymatching(correlated=True),
)
tesseract_result = batch.decode(
    source_graphlike_text,
    tesseract(preset="fast", pqlimit=50_000),
    workers=None,
)
bp_osd_result = batch.decode(
    raw_text,
    bp_osd(max_iter=10, osd_order=1),
    workers=None,
)

pymatching_errors = pymatching_result.num_errors
tesseract_errors = tesseract_result.num_errors
bp_osd_errors = bp_osd_result.num_errors

shots = batch.num_shots
assert 0 < pymatching_errors < shots
assert 0 < tesseract_errors < shots
assert 0 < bp_osd_errors < shots

print("DEM-sampled shots")
print(f"pymatching  {pymatching_errors:5}   {pymatching_errors / shots:.4%}")
print(f"tesseract   {tesseract_errors:5}   {tesseract_errors / shots:.4%}")
print(f"bp_osd      {bp_osd_errors:5}   {bp_osd_errors / shots:.4%}")
print(f"pymatching execution path: {pymatching_result.execution_path}")
```

With `workers=None`, PECOS automatically selects a native-batch, sequential, or
parallel path based on the decoder and batch size. Pass `workers=N` to request
an exact worker count. `result.execution_path` reports which path ran. Request
`predictions=True` when you also need each shot's arbitrary-precision
observable mask; the default avoids materializing them in Python.

`num_errors` counts shots where the predicted observable mask differs from the
true one anywhere, so with several observables it is the any-observable failure
count. For a per-observable rate, take `predictions=True` and compare bit `i` of
each predicted mask against bit `i` of the truth. With one observable the two
rates coincide — but say which one you mean.

The simulated shots decode the same way, against the same decoders:

<!--continuation-->
```python
from pecos_rslib.qec import SampleBatch

sim_batch = SampleBatch(
    [syndrome for syndrome, _ in sim_shots],
    [observable_mask for _, observable_mask in sim_shots],
)
sim_errors = sim_batch.decode(
    terminal_graphlike_text,
    pymatching(correlated=True),
).num_errors

print(f"simulated shots, pymatching: {sim_errors}/{len(sim_shots)}")
```

At this noise level the three decoders land within about a percentage point of
each other on this code; the gaps between decoders widen with code distance and
with genuinely hyperedge-like noise, which is where BP+OSD and Tesseract consume
the raw model rather than a graph-like projection.

## 6. Optional: per-shot confidence with an experimental decoder

!!! warning "Experimental API"

    The decoders below live in `exp/` and are reached through
    `pecos_rslib_exp`. They are under active development, are not part of the
    `pecos.decoders` surface, and may change without notice. They also do not
    participate in the unified execution planning used above — call them
    directly rather than through `batch.decode(...)`.

Every decoder in stage 5 answers "which observables flipped?". None of them
reports how close the call was. The experimental Frontier and BP-Trellis
decoders add a **complementary gap**: the log-probability margin between the
winning logical class and the runner-up. When the result's `status` is
`"exact"` (nothing was pruned), a large gap is a confident shot and a gap near
zero means the evidence barely favoured the answer -- the natural signal for
post-selection or for feeding a soft-output pipeline. On a pruned result the
gap describes only what the search retained, so treat it as a diagnostic
rather than a confidence; the gap is `None` whenever fewer than two logical
classes survive to the end, which can happen on a fully exact decode too, so a
missing gap is not a pruning signal.

<!--continuation-->
```python
from pecos_rslib_exp import FrontierDecoder

frontier = FrontierDecoder.from_dem(raw_text)
results = [frontier.decode_syndrome(batch.get_syndrome(shot)) for shot in range(200)]

assert all(result.status == "exact" for result in results)
gaps = [result.runner_up_gap for result in results if result.runner_up_gap is not None]

least_confident = min(gaps)
assert least_confident >= 0.0
print(f"least confident of {len(gaps)} shots: gap={least_confident:.3f}")
```

Because the gap is a per-shot quantity, a threshold on it partitions the run
into a confident majority and a tail worth treating differently:

<!--continuation-->
```python
confident = [gap for gap in gaps if gap >= 1.0]

print(f"{len(confident)}/{len(gaps)} shots decoded with gap >= 1.0")
```

Frontier consumes the raw model directly, so unlike the matching decoders it
needs no graph-like projection — `raw_text` rather than
`terminal_graphlike_text`. See [Experimental Decoders](../experimental/decoders.md)
for the full result surface, including what `status` and `dropped_log_mass` say
about whether pruning affected the answer.

## Where to go next

- [Detector Error Models from Guppy](../user-guide/dem-from-guppy.md) explains
  metadata references, idle-noise models, and DEM representations in detail.
- [QEC with Guppy](../user-guide/qec-guppy.md) covers the built-in QEC program
  generators for larger codes.
- [Decoders](../user-guide/decoders.md) describes the available decoder APIs.
- [Runtime QIS Tracing](../user-guide/runtime-qis-tracing.md) explains how PECOS
  captures the runtime-lowered gate stream used to build this DEM.
