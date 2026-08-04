# Decode a Guppy QEC experiment with idle noise

Use this workflow when you have a Guppy QEC experiment and want to estimate its
decoded logical error rate under circuit-level gate and idle noise. The example
is a hand-written three-qubit repetition-code memory, small enough to read in
full: everything here applies unchanged to larger hand-written programs.

## 1. Write the program

The program prepares three data qubits in the logical `|0>` state, extracts the
two parity checks twice with fresh ancillas, and reads out the data qubits in
the Z basis. Every measurement that a detector will reference is tagged with
`result(...)`, so detectors can be written by name instead of by counting
positions.

Two constraints shape the program. Quantum control flow must be static, so the
rounds are written out rather than looped, and the circuit must be Clifford.
Ancillas are freshly allocated each round because `measure()` consumes its
qubit. For larger codes, the built-in generators in
[QEC with Guppy](../user-guide/qec-guppy.md) produce this structure for you.

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

## 3. Attach idle gates and noise

`idle_after_2q_duration=1.0` inserts an idle of that duration on both qubits
after every two-qubit gate; traced identity-like gates are stripped first by
default, so runtime-emitted idles are not double-counted.

The linear family uses a custom Z-biased distribution, keeping smaller X and Y
memory errors while making dephasing dominant; its weights are an additive
probability distribution and must sum to 1. The sine-squared family uses Z only,
because the sine-law dephasing remnant is Z by nature — its default is symmetric
across X, Y, and Z, so the single-axis choice is spelled out explicitly. See
[Idle Noise](../user-guide/dem-from-guppy.md#idle-noise) for the full family and
model semantics.

## 4. Choose the DEM text for each decoder

`to_string()` returns Stim-format DEM text with raw hyperedges, which BP+OSD and
Tesseract consume directly. PyMatching requires a graph-like model, so it gets
the terminal projection from `to_string_terminal_graphlike_decomposed()`. The
source-informed `to_string_source_graphlike_decomposed()` form is used for
Tesseract below.

## 5. Sample detector events

`to_sampler()` draws detector events and observable flips directly from the DEM.
`get_syndrome()` returns one shot's detector bits and `get_observable_mask()` the
actual logical flips those shots incurred — the ground truth that decoder
predictions are scored against.

## 6. Decode the same batch three ways

Passing the same batch to each decoder makes the logical error rates directly
comparable, rather than mixing in sampling variation from separate experiments.

<!--test-name: guppy_dem_decoding_with_idle_noise-->
```python
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit

from pecos.qec import DetectorErrorModel


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
    result("s0_r0", measure(a0))
    result("s1_r0", measure(a1))

    # Round 1: same checks, fresh ancillas (measure() consumes its qubit).
    b0, b1 = qubit(), qubit()
    cx(d0, b0)
    cx(d1, b0)
    cx(d1, b1)
    cx(d2, b1)
    result("s0_r1", measure(b0))
    result("s1_r1", measure(b1))

    # Final data readout in the Z basis.
    result("m0", measure(d0))
    result("m1", measure(d1))
    result("m2", measure(d2))


# Each detector is a parity that is deterministic without noise.
detectors_json = """[
    {"id": "D0", "result_tags": ["s0_r0"]},
    {"id": "D1", "result_tags": ["s1_r0"]},
    {"id": "D2", "result_tags": ["s0_r0", "s0_r1"]},
    {"id": "D3", "result_tags": ["s1_r0", "s1_r1"]},
    {"id": "D4", "result_tags": ["s0_r1", "m0", "m1"]},
    {"id": "D5", "result_tags": ["s1_r1", "m1", "m2"]}
]"""
observables_json = '[{"id": "L0", "result_tags": ["m0"]}]'

dem = DetectorErrorModel.from_guppy(
    rep_code_memory,
    num_qubits=7,
    detectors_json=detectors_json,
    observables_json=observables_json,
    idle_after_2q_duration=1.0,
    p_idle_linear=0.01,
    p_idle_linear_model={"X": 0.25, "Y": 0.25, "Z": 0.5},
    p_idle_sin_squared=0.03,
    p_idle_sin_squared_model={"Z": 1.0},
    p1=0.002,
    p2=0.02,
    p_meas=0.02,
    p_prep=0.02,
)

# The decoder-specific text forms.
raw_text = dem.to_string()
terminal_graphlike_text = dem.to_string_terminal_graphlike_decomposed()
source_graphlike_text = dem.to_string_source_graphlike_decomposed()
num_mechanisms = raw_text.count("error(")

assert dem.num_detectors == 6
assert dem.num_observables == 1
assert num_mechanisms > 0
print(f"detectors: {dem.num_detectors}, error mechanisms: {num_mechanisms}")

# Draw one reproducible batch, then inspect a couple of shots explicitly.
sampler = dem.to_sampler()
batch = sampler.generate_samples(2000, 1)
assert batch.num_shots == 2000
for shot in range(2):
    syndrome = batch.get_syndrome(shot)
    observable_mask = batch.get_observable_mask(shot)
    assert len(syndrome) == dem.num_detectors
    print(f"shot {shot}: syndrome={syndrome}, observable_mask={observable_mask}")

# Decode identical detector events with all three decoders.
decoder_inputs = {
    "pymatching": terminal_graphlike_text,
    "tesseract": source_graphlike_text,
    "bp_osd": raw_text,
}
error_counts = {name: batch.decode_count(text, name) for name, text in decoder_inputs.items()}

assert all(0 < errors < batch.num_shots for errors in error_counts.values())
print("decoder     errors   logical error rate")
for name, errors in error_counts.items():
    print(f"{name:10}  {errors:6}   {errors / batch.num_shots:.4%}")
```

At this noise level the three decoders land within about a percentage point of
each other on this code; the gaps between decoders widen with code distance and
with genuinely hyperedge-like noise, which is where BP+OSD and Tesseract consume
the raw model rather than a graph-like projection.

## Where to go next

- [Detector Error Models from Guppy](../user-guide/dem-from-guppy.md) explains
  metadata references, idle-noise models, and DEM representations in detail.
- [QEC with Guppy](../user-guide/qec-guppy.md) covers the built-in QEC program
  generators for larger codes.
- [Decoders](../user-guide/decoders.md) describes the available decoder APIs.
- [Runtime QIS Tracing](../user-guide/runtime-qis-tracing.md) explains how PECOS
  captures the runtime-lowered gate stream used to build this DEM.
