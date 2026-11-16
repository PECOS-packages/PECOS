# Guppy runtime decoder admission probe

The supplied-model path and `RuntimeDecodeQisSite` work in live native QIS.
A production Guppy decoder primitive is still deferred. The diagnostic at
`scripts/diagnostics/runtime_decoder_guppy.py` establishes what already works
through the real Guppy compiler and exposes two separate admission gaps.
It is a development probe, not a frontend API or an adaptive DEM certificate.

## Reproducing the probe

Use the repository's development environment with matching native bindings,
QIS FFI and Selene runtime libraries. The script accepts `--runtime
selene_soft_rz_runtime` as well as the default simple runtime. Run it under a
process timeout: malformed compiler output could otherwise loop without ever
reaching a decoder guard. For example:

<!--skip: Standalone timed diagnostic; exercised by test_runtime_decode_guppy_admission.py with resolved package paths. -->
```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "scripts/diagnostics/runtime_decoder_guppy.py",
        "--case",
        "helper-loop",
        "--artifacts",
        "/tmp/pecos-guppy-probe",
    ],
    check=True,
    timeout=45,
)
```

The JSON report records package versions, hashes of the HUGR, compiler LLVM and
linked LLVM, provider instance sequences and live results. `--artifacts` saves
the three compiled artifacts for inspection. Exit code 0 means the probe's
specific output and instance contract passed; code 2 means rejection, with its
error included. A timeout is a failure, never evidence of successful admission.

| Case or option | Contract and observed result |
| --- | --- |
| `single` | An observed one-bit syndrome returns a frame that changes a later live measurement from one to zero. |
| `--no-correction` | The same supplied model's output projection suppresses the frame; the later measurement remains one. |
| `shared-measurement` | The synthetic read consumes one reference while Guppy's ordinary getter reads a retained copy of the same measurement. |
| `helper-loop` | Three calls to a Guppy helper pass instances 0, 1, 2 and correct all three outputs. This compiler can unroll the loop; it is not proof of dynamic loop provenance. |
| `--inject-failure` | A provider error aborts the first shot; subsequent runs reuse the runner with fresh shot state. |
| `branch-loop` | A loop carrying its instance counter across an external-result branch repeats instance zero on the affected stack. Native sequence checking rejects it before the second provider invocation. |
| `--control` | Compiler control capture rejects the generated module before provider construction. No batch-only fallback occurs. |

The successful cases execute two runs of two shots on two workers. Tests cover
both runtimes, correction enabled/disabled, failure recovery and the shared
measurement. They run the diagnostic in timed subprocesses in the core QEC
test shard. The branch-loop regression accepts either the precise sequence
rejection or fully correct execution after a compiler fix; it does not accept
other errors, partial results or timeouts as success.

## Evidence and limits

The branch-loop behavior was reproduced with Guppy 1.0.1, HUGR 0.18.8,
TKET 0.15.11 and `selene-hugr-qis-compiler` 0.4.3 on macOS arm64, matching the
repository's locked versions. Separately inspected LLVM at optimization levels
0 and 2 contains the counter reset; requesting Guppy's minimal optimization
also reproduced it. A plain counter loop without the external-result branch
compiles correctly. This localizes the failure to compilation, before PECOS
execution, but does not identify the responsible upstream pass. Preserve the
HUGR and LLVM artifacts when investigating; do not infer sound source control
flow from one realized native trace.

The diagnostic defines two synthetic external Guppy declarations and replaces
only their known LLVM declarations with shim definitions. It reuses
`RuntimeDecodeQisSite` for context, observed reads and parity/byte lowering,
and the existing DEM builder, window validation and decoder. The physical
program and branches remain compiler-generated. The one-bit model has the
independent oracle L0 = D0; deterministic X probes make both feedback outcomes
testable. This is not a fault-tolerance experiment.

The read shim delegates to the selected runtime's observed getter and decrements
the consumed future reference. The shared-measurement probe checks one retained
copy, but does not establish a general borrowed/owned Measurement API. Expected
measurement IDs are explicitly `2 * instance` for this fixed two-measurement
schedule; they are never derived from the observed data they validate. The
binding's synthetic QIS site label is a declaration, not a recovered HUGR node.
Batch requests remain `RuntimeDecodeBatchRequest`, with no compiler manifest.

## Next implementation boundary

Before exposing a production primitive, resolve the branch-loop compilation
failure and establish a checked lowering/admission path for actual compiler
output. Prefer bounded normalization into the existing single-function contract
over silently admitting arbitrary interprocedural LLVM. It must account for:

- Entry/helper definitions, setup/teardown, attributes and tail calls.
- Allocation failure paths (`panic` and `unreachable`) without deleting them.
- Resource-valued PHIs, distinct from scalar input dependencies; no resource
  handle may become a valid classical input through an arbitrary integer cast.
- Packed `print_bool` result tags, whose ABI differs from `print_bool_selene`.
- Measurement ownership/copies, semantic read placement, expected source
  mappings, and revision-specific QIS/HUGR identities.
- Branch outcomes, loop iterations and classical dependencies, including prior
  decoder frames, tested against source-level expectations.

Keep native control-capture rejection in force until those contracts are
implemented. Batch-only success does not fulfill compiler-owned provenance.
All existing restrictions on independent closed windows, fault ownership,
cross-boundary correlations, leakage persistence and unavailable future
operations remain unchanged. Full JIT generation and adaptive-model admission
remain separate follow-ups.
