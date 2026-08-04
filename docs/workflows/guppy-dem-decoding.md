# Decode a Guppy QEC experiment with idle noise

Use this workflow when you already have a Guppy QEC experiment and want to
estimate its decoded logical error rate with circuit-level gate and idle noise.

## 1. Build the program and its metadata

Create a distance-3 surface-code memory experiment with three syndrome rounds.
The abstract surface-code builder supplies detector and observable metadata in
the same measurement order as the generated Guppy program. Hand-written Guppy
works identically; see [Detector Error Models from Guppy](../user-guide/dem-from-guppy.md#referencing-measurements)
for metadata-authoring details.

## 2. Attach idle gates and noise

Insert an idle of duration `1.0` on both qubits after every two-qubit gate. This
option strips traced identity-like gates by default before inserting the uniform
idle convention, which prevents double-counting runtime-emitted idles.

The linear family uses a custom Z-biased distribution to retain smaller X/Y
memory errors while making dephasing dominant. Its weights are an additive
probability distribution and must sum to 1. The sine-squared family deliberately
uses only Z because the sine-law dephasing remnant is Z by nature. Its default
is symmetric across X, Y, and Z, so the single-axis choice is spelled out
explicitly. See [Idle Noise](../user-guide/dem-from-guppy.md#idle-noise) for the
full family and model semantics.

## 3. Choose the DEM text for each decoder

`to_string()` returns Stim-format DEM text with raw hyperedges, which BP+OSD and
Tesseract can consume. PyMatching requires a graph-like model, so use
`to_string_terminal_graphlike_decomposed()` for its terminal projection. The
source-informed `to_string_source_graphlike_decomposed()` form is used for the
Tesseract comparison below; Tesseract can also take the raw hyperedge form.

## 4. Sample detector events

Generate one batch and inspect individual shots with `get_syndrome()` and
`get_observable_mask()`. The observable mask contains the actual logical flips
against which decoder predictions are scored.

## 5. Decode the same batch three ways

Pass the appropriate text form and decoder name to `decode_count()`. Reusing the
same batch makes the logical error rates directly comparable rather than mixing
in sampling variation from separate experiments.

<!--test-name: guppy_dem_decoding_with_idle_noise-->
```python
from pecos.guppy import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch

distance = 3
num_rounds = 3
basis = "Z"

# Build the Guppy program and matching abstract-circuit metadata.
program = make_surface_code(
    distance=distance,
    num_rounds=num_rounds,
    basis=basis,
)
patch = SurfacePatch.create(distance=distance)
metadata_circuit = generate_tick_circuit_from_patch(
    patch,
    num_rounds=num_rounds,
    basis=basis,
)

# Trace the program, insert idle gates, and attach gate and memory noise.
dem = DetectorErrorModel.from_guppy(
    program,
    num_qubits=get_num_qubits(distance),
    detectors_json=metadata_circuit.get_meta("detectors"),
    observables_json=metadata_circuit.get_meta("observables"),
    num_measurements=int(metadata_circuit.get_meta("num_measurements")),
    idle_after_2q_duration=1.0,
    p_idle_linear=0.002,
    p_idle_linear_model={"X": 0.25, "Y": 0.25, "Z": 0.5},
    p_idle_sin_squared=0.01,
    p_idle_sin_squared_model={"Z": 1.0},
    p1=0.001,
    p2=0.005,
    p_meas=0.005,
    p_prep=0.005,
)

# Inspect the model and prepare the decoder-specific text forms.
raw_text = dem.to_string()
terminal_graphlike_text = dem.to_string_terminal_graphlike_decomposed()
source_graphlike_text = dem.to_string_source_graphlike_decomposed()
num_mechanisms = raw_text.count("error(")

assert dem.num_detectors > 0
assert dem.num_observables == 1
assert num_mechanisms > 0
assert all(
    "error(" in text
    for text in (raw_text, terminal_graphlike_text, source_graphlike_text)
)
print(f"detectors: {dem.num_detectors}")
print(f"error mechanisms: {num_mechanisms}")

# Draw one reproducible batch, then access a couple of shots explicitly.
sampler = dem.to_sampler()
batch = sampler.generate_samples(2000, 1)
assert batch.num_shots == 2000
for i in range(2):
    syndrome = batch.get_syndrome(i)
    observable_mask = batch.get_observable_mask(i)
    assert len(syndrome) == dem.num_detectors
    assert observable_mask in (0, 1)
    print(f"shot {i}: syndrome={syndrome}, observable_mask={observable_mask}")

# Decode identical detector events with all three decoders.
decoder_inputs = {
    "pymatching": terminal_graphlike_text,
    "tesseract": source_graphlike_text,
    "bp_osd": raw_text,
}
error_counts = {
    name: batch.decode_count(text, name)
    for name, text in decoder_inputs.items()
}
logical_error_rates = {
    name: errors / batch.num_shots
    for name, errors in error_counts.items()
}

assert all(0 < errors < batch.num_shots for errors in error_counts.values())
print("decoder     errors   logical error rate")
for name in decoder_inputs:
    print(f"{name:10}  {error_counts[name]:6}   {logical_error_rates[name]:.4%}")
```

## Where to go next

- [Detector Error Models from Guppy](../user-guide/dem-from-guppy.md) explains
  metadata references, idle-noise models, and DEM representations in detail.
- [QEC with Guppy](../user-guide/qec-guppy.md) covers the built-in QEC program
  generators and execution workflow.
- [Decoders](../user-guide/decoders.md) describes the available decoder APIs.
- [Runtime QIS Tracing](../user-guide/runtime-qis-tracing.md) explains how PECOS
  captures the runtime-lowered gate stream used to build this DEM.
