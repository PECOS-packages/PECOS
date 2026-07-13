# Noise Event Replay and Failure Diagnostics

This note records a future debugging path for understanding logical failures in
noisy circuit simulations. The goal is to make sampled noise events
reproducible and inspectable without turning normal simulation runs into large
trace dumps.

## Motivation

Detector error models tell us how physical error mechanisms can affect
detectors and observables, but they do not show which stochastic noise events
actually occurred in a sampled shot. When a decoder reports a logical failure,
we often want to inspect the concrete sampled events that produced that failure:

- which noise source sampled an event,
- which ideal gate or payload location it was attached to,
- which branch was selected,
- which qubits and measurement results were involved,
- and whether the resulting syndrome pattern looked decoder-ambiguous.

This is especially useful for local emulator studies where PECOS controls the
noise sampling. For hardware data, the physical events are not directly known,
but the same machinery can still be used to compare hardware syndromes against
failure patterns from calibrated local simulations.

## Decoder-Dependent Failure Selection

"Failed shots only" is a useful logging mode, but it is not intrinsic to the
simulator. Whether a shot is a logical failure depends on the analysis layer:

- the detector and observable metadata,
- the DEM used for decoding,
- the decoder backend,
- decoder options such as graph-like decomposition or correlated matching, and
- the logical-failure convention used by the experiment.

For that reason, the simulator should not decide by itself which shots are
interesting. Instead, the preferred long-term flow is:

1. Run or replay shots with deterministic per-shot seeds.
2. Decode the resulting detection events in the analysis layer.
3. Select failed or otherwise interesting shot ids.
4. Replay only those shot ids with noise-event tracing enabled.

This keeps logging sparse while preserving a clean separation between simulation
and decoder-specific analysis.

## Per-Shot Reproducibility

The key primitive is deterministic replay from stable seeds. A run should be
able to derive all randomness from a root seed and a shot identity:

```text
run_seed
  -> shot_seed(run_seed, shot_index)
    -> component_seed(shot_seed, "noise")
    -> component_seed(shot_seed, "runtime")
    -> component_seed(shot_seed, "decoder" or analysis-only randomness)
```

The exact derivation should be explicit, versioned, and independent of worker
count. Replaying `(program, noise_config, runtime_config, run_seed, shot_index)`
should reproduce the same sampled noise events even if the original run used a
different number of workers.

This is more important than logging everything during the first pass. Sparse
noise logs may be small at low physical error rates, but deterministic replay
lets us defer expensive diagnostics until we know which shots matter.

## Optional Event Trace Schema

When tracing is enabled, each sampled non-identity noise event should carry
enough source information to connect it back to the ideal program and DEM source
metadata. A compact JSONL-style record could contain:

```json
{
  "shot": 17,
  "shot_seed": "0x...",
  "event_index": 42,
  "tick": 19,
  "gate_index": 3,
  "gate_type": "SZZ",
  "gate_qubits": [4, 12],
  "source_family": "TwoQubitGate",
  "noise_parameter": "p2",
  "probability": 0.001,
  "branch": "IX",
  "random_draw": 0.00042
}
```

Some source families need additional payload fields:

- idle events should record duration and the axis-specific rate terms,
- replacement branches should record whether the ideal gate was omitted,
- measurement crosstalk should record payload gate type, candidate victim, and
  transition label,
- measurement errors should record the measurement result id when available.

Normal runs should default to no trace. Useful opt-in modes include:

- trace every sampled event for small debugging runs,
- trace the first `N` shots,
- replay and trace a caller-provided list of shot ids,
- summarize event counts by source family without writing per-event records.

## Data Volume

Full logs may be acceptable for small studies because physical error events are
sparse. They can still become large quickly when the number of gates, shots,
distances, or parameter scans increases. The implementation should therefore
avoid coupling ordinary simulation output to full event logs.

Recommended defaults:

- store shot seeds or enough metadata to reconstruct them,
- store detection events and logical outcomes as usual,
- write detailed event records only in explicit diagnostic/replay modes,
- support streaming event records so failed-shot replays do not require keeping
  all events in memory.

## Implementation Sketch

1. Add a worker-independent per-shot seed derivation API.
2. Ensure each simulator/noise layer receives deterministic component RNGs
   derived from the shot seed.
3. Add an optional `NoiseEventSink` trait or equivalent callback in Rust.
4. Emit event records only for sampled non-identity branches, with source
   metadata attached at the noise-model location.
5. Expose a replay API that accepts explicit shot ids or shot seeds.
6. Add Python helpers that decode first, select failed shots, then replay those
   shots with tracing enabled.
7. Add compact summaries that aggregate event counts by source family, gate
   type, branch, and qubit region.

## Open Questions

- How should runtime-generated events be traced when a runtime has its own RNGs?
- Should event traces include raw random draws, or only branch outcomes plus the
  seed needed to reproduce them?
- Which event identifiers should be stable across circuit recompilation, and
  which should be explicitly tied to the lowered/traced circuit version?
- How should traces link back to DEM contribution/source records when one sampled
  event corresponds to multiple detector/observable effects?
