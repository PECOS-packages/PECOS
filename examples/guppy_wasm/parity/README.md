# Guppy/WASM workload comparison

This bundle supplies public source programs, WAT modules, independent expected
answers and a common runner contract. It measures **conformance on these
workloads**, not complete equivalence between stacks. A successful comparison
never asserts full-stack parity. Broader coverage and performance acceptance
criteria must be agreed from the reference stack's actual capabilities and use
cases.

## Run the public baselines

Use a PECOS checkout built with its WASM extra and matching native libraries, as
in `docs/user-guide/guppy-wasm.md`. The standalone Selene route needs the public
`pecos-selene-wasm` utility, Guppy and Selene. From `examples/guppy_wasm` (or the
extracted `guppy_wasm` directory):

```bash
python parity/parity.py run --backend selene --stack-tag public-selene-1 --host-tag machine-a --output selene.json
python parity/parity.py run --backend pecos --stack-tag public-pecos-1 --host-tag machine-a --output pecos.json
python parity/parity.py compare selene.json pecos.json --output comparison.json
```

Defaults are eight shots per run, three measured repeats after the first run,
one worker, seed 42, one qubit, Helios and a state-vector simulator. Use
`--shots`, `--repeats` and `--workers` consistently across stacks. Workers must
not exceed shots, so neither backend silently reduces the requested parallelism. Each run must
start fresh shots; repeats reuse the prepared runner. Stateful cases check that reuse. The
trap case checks repeated traps on one prepared runner. The following case
prepares a fresh runner in the same harness process and checks process-level
recovery; successful recovery on the same prepared runner is not established. The shot dictionaries contain recorded
results, not the function's unrecorded return value.

Exit code 0 from `run` or `compare` means every required workload passed its
oracle. Exit code 1 means at least one did not conform. A malformed report or
incompatible settings/inputs exits with code 2, not a partial pass. Do not omit failing
cases or edit their observations to obtain a passing comparison.

## Create the handoff bundle

```bash
python parity/parity.py bundle /tmp/guppy-wasm-parity.zip
```

The deterministic archive contains an explicit allowlist of public sources,
modules, this harness, an unimplemented adapter template, this guide and the
workload manifest. `SHA256.json` lists hashes of every payload file. Neighboring
files, environments, adapters, reports and logs are not collected. The harness
and file hashes travel with the suite. The loader executes source directly and
bypasses timestamp-based Python bytecode caches. Keep the original archive intact and use
the same bundle on both sides. Archive hashes can additionally identify the
exact handoff artifact.

## Internal operator instructions

1. Extract the bundle into an existing environment that can execute the
   reference Guppy/WASM stack. Do not install the public runtime over that
   environment merely to make this test pass.
2. Copy `parity/adapter_template.py` **outside** the extracted bundle. Implement
   its `prepare(program, modules, settings, build_dir)` context manager using
   the reference stack's normal Guppy source compilation and execution path.
   Do not call the public PECOS lowering as the reference implementation.
3. Compile and construct the runner before yielding a zero-argument `run`
   callable. Every invocation must execute the requested fresh shots and return
   `{"shots": [{"label": numeric_value, ...}, ...]}`. Preserve shot order, reject duplicate result labels, and
   consume streamed results. Release resources when the context manager exits.
4. Only translate a recognized WASM `unreachable` trap into
   `{"error": "wasm_trap"}`. All compiler failures, unsupported operations,
   connection failures and unrelated runtime errors must remain exceptions.
   The harness records generic preparation/execution/cleanup errors without
   embedding exception text. Never return expected answers from the manifest
   in place of executing the workload.
5. Use an opaque stack tag and a non-sensitive host tag. Record the exact private
   compiler/runtime versions, adapter revision, hardware/OS details and runtime
   options in your **internal** run log, keyed by those tags. The public report
   deliberately does not inventory installed packages or record adapter paths.
   A host tag means the same physical machine and operating setup; it is not an
   automatically verified hardware identity.
6. Execute with the same settings as the public baselines:

```bash
python parity/parity.py run --adapter /private/location/adapter.py --stack-tag reference-1 --host-tag machine-a --output reference.json
python parity/parity.py compare reference.json pecos.json --output reference-vs-pecos.json
python parity/parity.py compare reference.json selene.json --output reference-vs-selene.json
```

The adapter is executable code supplied by the operator; run only a trusted
adapter. It may import private dependencies locally, but it is not copied into
the archive or report. Keep local stdout/stderr and diagnostic logs internal:
third-party compilers may print local paths or other details. Reports contain
only numeric workload observations, generic error categories, timings, hashes,
settings and operator-supplied tags. **Review the tags and report before sharing**;
this format is not an automatic declassification mechanism. Do not upload
private adapters, environments, implementations, raw compiler logs or HUGR/LLVM
artifacts.

Return the reference and public run reports and comparison reports, plus the original bundle
hash, through the agreed internal channel. No tool automatically sends them.
If a workload fails, preserve the failure and explain its category separately
without publishing private implementation details.

## Interpreting timings

Preparation includes importing the workload, reading/validating declarations,
source compilation and runner construction. First-run time is separate. Repeat
times include the complete requested shot batch, module instantiation, runtime
communication and result collection. They are not isolated WASM-call latency.

The comparison reports candidate/reference ratios of repeat medians only for
successful non-trap cases with matching host tags and settings. A ratio above 1
means the candidate took longer in that sample. Different host tags suppress
ratios; functional comparison still works. Raw preparation and per-run timings
remain in each report. Matching tags alone cannot prove identical hardware,
load, simulator implementation or compiler options.

Run stacks sequentially on an otherwise idle machine, alternate their order,
and repeat the experiment. Record resource usage and toolchain options in the
internal log. Do not infer a speed guarantee or statistical equivalence from
one small run; this harness sets no slowdown threshold. Different compiler
optimizations are part of the stack being measured and should be recorded.

## Coverage and the next comparison round

The manifest covers named/indexed scalar calls, mixed i64/f64 signatures,
state/context isolation, branches, nested and result-dependent loops, helper
ordering, quantum feedback, deliberate traps and subsequent recovery, plus
100- and 1,000-call dependent loops for timing call-count scaling. All
expected floating values are exactly representable; comparisons are exact.
The manifest marks measurements with boolean expected values; those may be
represented as bool or integer 0/1. Other scalar
types must match the expected type. Missing, extra, nonfinite or nonnumeric
observations are failures.

The internal operator should also return a capability inventory, using public
feature descriptions and `supported`, `unsupported` or `not tested` for each:

- Guppy/compiler versions and entry paths (keep exact private identifiers internal).
- i32/f32, aggregate and multiple results; memory/array transfer.
- WASI/host imports, initialization and resource-limit behavior.
- Asynchronous calls, concurrency and supported target platforms.
- Representative production call counts, module sizes, classical work per call,
  qubit counts and acceptable compilation/execution overhead.

If a capability exists only in the reference, provide a small publicly shareable
input/output specification or reproduction. Extend this suite and fix the
public stack against that evidence. Passing the current scalar cases does not
resolve these inventory items or establish full functional/performance parity.

Suite `guppy-wasm-scalar-v2` adds i32 boundaries, state and feedback cases.
The manifest records each case's `wasm_int_width` (default 64); the adapter
receives this value in its settings. The public adapters select strict native
width explicitly. Reports from v1 and v2 cannot be compared: rerun both stacks
from the same v2 archive.

Suite `guppy-wasm-scalar-v3` adds the independently written repetition-code
correction workload. Its manifest entry specifies `qubits: 5`; cases without
this field use the report's default one-qubit setting. Adapters receive each
case's effective qubit count in `settings["qubits"]`. The manifest is included
in the hashed payload, so all per-case resource settings are reproducible.
Rerun both stacks from the same v3 archive; older reports are not comparable.
The new case checks repeated measurement, WASM state, and quantum corrections,
with a known mathematical answer rather than a private decoder dependency.
