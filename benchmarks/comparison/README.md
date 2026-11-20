# Manual route comparison

Run with Python 3.11+ on a POSIX host. This command is **not scheduled in CI** and
has no pass/fail speed threshold. It runs four externally configured routes in
balanced Williams-square blocks, in isolated sequential process groups. No
simulator, workload, private runtime, calibration or experimental data is bundled.

```sh
python3.12 benchmarks/comparison/run.py plan --config /absolute/comparison.json
python3.12 benchmarks/comparison/run.py run --config /absolute/comparison.json --output /absolute/new-results
python3.12 benchmarks/comparison/run.py verify --output /absolute/new-results
```

The output directory must not exist. Failure stops the schedule; partial evidence
is retained and cannot verify as a successful result. Never tune or exclude runs
based on speed. Profiling belongs in separate executions. The verifier requires
unchanged input files and the same runner source.

## Configuration and external workers

JSON fields: `schema: 1`, positive integer `cycles`, `shots`, `warmup_shots`, and
`timeout_seconds`; `inputs` maps absolute input paths to SHA256 hashes; `routes`
contains exactly four objects with a distinct `name`, an `argv` string array and
an explicit `env` dictionary. No shell is used. The executable must be absolute.
The runner expands `{output}`, `{route}`, `{shots}`, `{warmup}` in argv/env; escape
literal braces as `{{` and `}}`. Input hashes should cover worker source, workload,
parameters, imported libraries and build/admission manifests. Workers must also
check their actual imported native libraries and external package identities.

One cycle is four balanced blocks (16 workers). Two cycles reverse block order
for the second cycle, matching an eight-block/32-worker comparison. Each worker
performs its warmup and timed run in one process with retained prepared programs.
Use `{output}` in cache paths for fresh per-worker compilation caches. The runner
records commands, clean environments, platform, loads, process receipts and all
output file hashes. Keep sensitive configurations and output folders private.

Workers write `report.json` in their provided, already-created directory:

- `route`, `shots`, `warmup_shots` must match the request.
- `correctness_passed` and `prepared_program_reused` must be true.
- `evidence`: nonempty relative paths to saved correctness/output evidence.
- `warmup` and `timed`: five ordered usage-point objects each, with `wall_ns`
  (monotonic integer nanoseconds), `self_user`, `self_system`, `children_user`,
  `children_system` (cumulative CPU seconds, including reaped descendants).

The boundaries delimit preparation, execution/materialization, result persistence,
and teardown. Primary API time is preparation+execution/materialization; report
all raw boundaries. Preparation must not redo lowering while another route reuses
its compiled program. Model/simulator/runtime, seed policy, approximation, worker
count and build profile must be matched and documented by the configuration owner.
Child CPU requires waiting for children before the boundary. Teardown semantics
can differ across APIs and must be documented rather than assumed comparable.

Correctness flags are **worker attestations**, not an independent proof of physics
or output equivalence. Workers must validate complete outputs and their relevant
runtime invariants outside timed intervals and save evidence. Route-specific saved
verification remains the adapter's responsibility. The generic verifier validates
schema, process completion, hashes, raw timing arithmetic and summaries; it cannot
interpret arbitrary simulator outputs. Hashes detect changes, not a malicious
rewrite of evidence and its manifest. Execute only trusted worker configurations.

The summary retains every observation and block-paired ratio, with route medians.
Ratios above one mean the numerator took longer. No confidence interval, general
speed guarantee or cross-route random-stream equality is implied. No peak-memory
measurement or arbitrary route-count scheduling is provided in this first version.

## Harness tests

```sh
python3.12 -m unittest discover -s benchmarks/comparison -p 'test_*.py'
```

These use synthetic workers and deterministic usage points. They launch no
simulator and make no timing assertions. CI may run these small harness tests;
actual benchmark configurations and performance runs remain manual.
