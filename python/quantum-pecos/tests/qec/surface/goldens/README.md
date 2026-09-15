<!-- Copyright 2026 The PECOS Developers -->
<!-- Licensed under the Apache License, Version 2.0 -->

# Surface golden provenance

The baseline golden files in `gadget_parity/` and `logical_builder/` were captured
from `dev` at `8568727d5`. Later captures and re-captures are listed below,
including the PR #763 boundary-detector and PR #777 fold changes.
The four injection re-captures named in #757 are:

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
were first captured from the base revision during the #753 review-fix round and
then captured again from the corrected adapter. Empty-register H and
measurement calls no longer emit operations or artificial ticks; comment-only
CX layers still emit one tick, except trailing empty groups at the end of the
merged stream, which are omitted. A wholly empty syndrome stream, such as
`create(1)` with rounds, contributes no syndrome ticks beyond preparation.
Their tick counts changed from 18, 18, and 20
on dev to 14, 18, and 20, respectively. Measurement records, detectors, and
observables are unchanged, so `dx1dz3_mem_Z` and `dx1dz3_mem_X` differ from
the base revision only in `.stim` and `.noisy.stim`; their `.tickmeta.json` is
byte-identical to it.

The six builder shapes `d2_h_even`, `d3_cx_chain`, `d3_cxcx`,
`d3_hh_adjacent`, `d3_late_partner_cx`, and `d3_skip_segment` are PR #763
post-fix captures of boundary-detector derivation by backward propagation.
Each includes `.stim`, `.noisy.stim`, and `.tickmeta.json`. Their recipes in
`capture.py` reproduce all eighteen artifacts byte-identically.

The remaining captures, except the PR #777 captures below, retain the base
revision's output, including balanced CX operations and distance-7 and
dx=5/dz=3 Guppy source.

`capture.py` contains recipes for every current golden in both directories,
including the six PR #763 builder shapes, and records the baseline revision
in `BASE_REVISION`. It uses the PECOS implementation installed in
the invoking environment; it does not switch revisions, build dependencies, or
change git state. Use an environment running the base revision for baseline
captures, or the corrected implementation with `--post-fix-only` for the twenty
POST_FIX_SHAPES and the protocol drift guard. Run from the repository root:

```bash
export UV_CACHE_DIR=/tmp/pecos-uv-cache-753
uv run --frozen --no-sync python python/quantum-pecos/tests/qec/surface/goldens/capture.py --output-dir /tmp/pecos-surface-captures
```

Existing files are never overwritten unless `--force` is supplied. Capture into
a separate output directory for review; do not regenerate protected goldens to
make parity tests pass. Serialization preserves the original whitespace, nested
metadata strings, nulls, empty operations, and absence of final newlines.

## PR #777 captures

`d3_fold_s_mid`, `d3_fold_s_first`, `d3_fold_s_last`, `d3_fold_pair_x`, and
`d3_h_fold` are post-fix captures of the fold-aware builder, after independent
noiseless parity-space checks. Each has `.stim`, `.noisy.stim`, and
`.tickmeta.json` files. These are drift guards, not independent physics oracles.

`d3_mem_Z_zero_final` (`M(2,Z); M(0,Z)`) and `d3_h_zero_final`
(`M(2,Z); H; M(0,X)`) announce and pin a non-fold detector improvement in PR
#777. A zero-round final segment compares its data checks with the preceding
segment's last round. Both shapes have 16 detectors instead of the base
branch's 12; each retains one observable. These six new artifacts were captured
after the fix and certified by the deterministic-parity-basis oracle.
There are twenty entries in `POST_FIX_SHAPES`: three repetition shapes,
four injection shapes, six PR #763 shapes, five fold shapes, and these two zero-final shapes.

`gadget_parity/protocol_d3.py.txt` originated in PR #762 and is explicitly
re-captured during the PR #777 review fixes. The merged quantum import and
expanded factory docstring change the generated source, so the historical
source cannot remain the active exact-match guard. It now captures the entire
current protocol module, including the fold factories. `capture.py` produces
this file, also with `--post-fix-only` and `--fold-only`.

`protocol_d3.py.txt` is the only protocol-module drift guard. No other
pre-existing golden artifact is re-captured. `--fold-only` selects the fifteen
fold builder artifacts plus the current protocol guard; `--post-fix-only`
also includes the other fifteen post-fix builder shapes.
