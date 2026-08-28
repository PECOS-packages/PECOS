# Runtime QIS Tracing

This guide covers PECOS's public tracing APIs for capturing the QIS operation
stream produced during one program execution and replaying the runtime-lowered
gates into a `TickCircuit`.

## What You'll Learn

- Capture a structured operation trace from Guppy, HUGR, or QIS
- Inspect the source and runtime-lowered operation streams
- Store a trace as JSON and replay it without running the program again
- Convert a program directly to a `TickCircuit`
- Understand runtime-emitted `Idle` gates and single-path limitations

## Overview

The tracing workflow has two explicit stages:

1. `capture_qis_operation_trace(...)` executes one ideal shot through the
   selected Selene-compatible runtime and returns JSON-compatible trace chunks.
2. `qis_operation_trace_to_tick_circuit(...)` validates and replays the
   runtime-lowered gates into a PECOS `TickCircuit`.

`trace_program_to_tick_circuit(...)` combines both stages when the intermediate
trace is not needed.

The API names describe the shared execution boundary rather than the source
language. Today the input can be a Guppy definition, `pecos.Guppy`,
`pecos.Hugr`, `pecos.Qis`, or another input accepted by `pecos.sim` that lowers
through the QIS control path. Future language frontends can use the same API
when they target that path.

## Quick Start

The following example captures a Guppy program, persists its trace through a
JSON round trip, and replays the stored data:

<!--test-name: runtime_qis_trace_round_trip-->
```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit
from pecos import (
    capture_qis_operation_trace,
    qis_operation_trace_to_tick_circuit,
    trace_program_to_tick_circuit,
)
from pecos.quantum import TickCircuit


@guppy
def coin_flip() -> None:
    q = qubit()
    h(q)
    _ = measure(q).read()


trace = capture_qis_operation_trace(coin_flip, num_qubits=1, seed=0)
assert trace[-1]["stage"] == "trace_complete"

# The returned dictionaries can be stored with the standard JSON module.
with TemporaryDirectory() as trace_dir:
    trace_path = Path(trace_dir) / "runtime-qis-trace.json"
    trace_path.write_text(json.dumps(trace, indent=2))
    stored_trace = json.loads(trace_path.read_text())
    circuit = qis_operation_trace_to_tick_circuit(stored_trace)

assert isinstance(circuit, TickCircuit)
assert circuit.num_measurements() == 1

# Use the convenience function when the intermediate trace is unnecessary.
direct_circuit = trace_program_to_tick_circuit(coin_flip, num_qubits=1, seed=0)
assert direct_circuit.num_measurements() == circuit.num_measurements()
```

`num_qubits` is the capacity made available to the execution. It is required
even when the program allocates its qubits dynamically.

## Understanding the Trace

The result is a list of framed `pecos_qis_operation_trace_v1` chunks. A normal
completed trace contains one or more execution/result chunks followed by one
empty `trace_complete` chunk.

Important fields include:

- `operations`: source QIS operations such as allocation, gates, measurement,
  barriers, and result recording.
- `lowered_quantum_ops`: gates after the selected runtime has scheduled and
  lowered the source operations.
- `lowered_quantum_ops_complete`: the runtime's attestation that the lowered
  gate stream completely represents the lowerable source operations in that
  chunk.
- `named_result_traces`: provenance emitted for named `result(...)` records.
- `engine_trace_id`, `shot_index`, and `chunk_index`: framing identities used
  to reject mixed, missing, or reordered chunks.
- `stage`: the producer stage; exactly one final `trace_complete` marker is
  required for replay.

The runtime-lowered stream is authoritative for the resulting circuit. A
runtime may reorder independent gates, lower a source operation into different
native gates, or add explicit scheduling operations.

## Replaying a Stored Trace

`qis_operation_trace_to_tick_circuit(...)` does not execute the original
program. It validates the stored trace and constructs a new `TickCircuit` from
`lowered_quantum_ops`.

Validation rejects, among other problems:

- Empty, incomplete, mixed-shot, or non-contiguous trace streams
- Chunks that do not attest complete runtime lowering
- Unsupported gates and malformed gate parameters
- Missing, duplicated, or inconsistent measurement-result identities
- Runtime-lowered measurements without result-ID provenance

The circuit metadata records the source measurement IDs under
`qis_source_measurement_ids`. Measurement gates retain the corresponding
runtime-lowered `MeasId`s, even if runtime scheduling changes their execution
order.

## Runtime-Emitted Idle Gates

Some compatible runtimes emit explicit `Idle` operations as part of their
schedule. Replay preserves those operations as `Idle` gates. Runtime durations
are expressed in seconds and converted to integer PECOS `TimeUnits` using
`1 TimeUnit = 1 ns`.

PECOS does not infer or synthesize missing idles in this workflow. If the
selected runtime emits no explicit idle operations, the traced circuit has no
idle gates. This distinction matters when the circuit is later used with an
idle-, T1-, or T2-dependent noise model.

Runtime-specific measurement-crosstalk payload gates are also preserved. The
advanced `measurement_crosstalk_topology="global_from_measurements"` option can
synthesize global payload markers from measurement locations when that is the
intended model; the default uses only payloads present in the runtime trace.

## Choosing a Runtime

The `runtime=` argument is forwarded to `pecos.selene_engine(...)`. It accepts
the default runtime (`None`), a built runtime name, a compatible shared-library
path, or a Selene runtime plugin object. Different runtimes can legitimately
produce different traces and therefore different `TickCircuit`s.

Keep `seed` with the trace when reproducibility matters. It controls the ideal
measurement outcomes used during the captured execution.

## What the Result Does Not Include

The replayed `TickCircuit` contains gates and trace provenance, but it does not
automatically contain detector or logical-observable metadata. Use
[`DetectorErrorModel.from_guppy`](dem-from-guppy.md) or
`pecos.qec.build_dem_from_guppy` when the goal is an audited detector error
model rather than direct trace inspection.

The trace-to-circuit conversion preserves the runtime representation. It does
not automatically lower every parameterized Clifford rotation or prepare the
circuit for fault propagation. Fault-analysis entry points apply their own
required normalization.

## Single-Path Limitation

Tracing captures one execution path. It is appropriate for inspecting or
replaying that concrete execution, including a dynamic program.

Do not treat one trace from measurement-dependent control flow as a complete
static circuit model covering every possible branch. In particular, building
a fault model from one sampled branch would omit gates and faults present only
on other outcomes. The Guppy-to-DEM APIs therefore apply stricter static
schedule checks than these general tracing APIs.

## Next Steps

- **[HUGR & Guppy Simulation](hugr-simulation.md)** - Build and run programs
  accepted by the tracing entry points
- **[Circuit Representation](circuit-representation.md)** - Work with the
  resulting `TickCircuit`
- **[Detector Error Models from Guppy](dem-from-guppy.md)** - Build an audited
  fault model from a statically certifiable program
- **[Noise Model Builders](noise-model-builders.md)** - Apply gate and idle
  noise in subsequent simulation workflows
