<!-- Copyright 2026 The PECOS Developers -->
<!-- Licensed under the Apache License, Version 2.0 -->

# Surface golden provenance

Every golden file in `gadget_parity/` and `logical_builder/` was captured from
`dev` at `8568727d5` before the gadget implementation changed, except the post-change captures
described below. The four injection re-captures named in #757 are:

- `d3_sz_teleport_first_op`
- `d3_sz_teleport_memory_first`
- `d3_t_inject_first_op`
- `d3_t_inject_data_memory_first`

Those four builder shapes were re-captured after fixing injection preparation
and readout: a fresh ancilla receives its own product preparation even when data
memory comes first, and its random logical readout is separated from deterministic
observables. The SZ resource uses H then SZ; the T helper remains a Clifford
stand-in. The original outputs encoded the defects, so they could not serve as
correctness oracles for these fixes. Each exception covers `.stim`, `.noisy.stim`,
and `.tickmeta.json`.

The three repetition shapes `dx3dz1_mem_Z`, `dx1dz3_mem_Z`, and `dx1dz3_mem_X`
were re-captured after the second #753 review-fix round. Empty-register H and
measurement calls no longer emit operations or artificial ticks; comment-only
CX layers still emit one tick, except trailing empty groups at the end of the
merged stream, which are omitted. A wholly empty syndrome stream, such as
`create(1)` with rounds, contributes no syndrome ticks beyond preparation.
Their tick counts changed from 18, 18, and 20
on dev to 14, 18, and 20, respectively. Measurement records, detectors, and
observables are unchanged. Each exception covers all three builder suffixes.
All other captures, including balanced CX operations and distance-7 and
dx=5/dz=3 Guppy source, retain the base revision's output.

`capture.py` contains the explicit recipes for both directories and records the
base revision in `BASE_REVISION`. It uses the PECOS implementation installed in
the invoking environment; it does not switch revisions, build dependencies, or
change git state. Use an environment running the base revision for baseline
captures, or the corrected implementation with `--post-fix-only` for the seven
exceptions. Run from the repository root:

```bash
export UV_CACHE_DIR=/tmp/pecos-uv-cache-753
uv run --frozen --no-sync python python/quantum-pecos/tests/qec/surface/goldens/capture.py --output-dir /tmp/pecos-surface-captures
```

Existing files are never overwritten unless `--force` is supplied. Capture into
a separate output directory for review; do not regenerate protected goldens to
make parity tests pass. Serialization preserves the original whitespace, nested
metadata strings, nulls, empty operations, and absence of final newlines.
