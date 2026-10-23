# Decoders

```hidden-rust
use pecos_decoders::{SparseMatrix, Decoder, BpMethod, OsdMethod, UfMethod, BpSchedule, BpOsdDecoder, BpLsdDecoder, BeliefFindDecoder, FlipDecoder, UnionFindDecoder, SoftInfoBpDecoder, LdpcError, InputVectorType};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows: Vec<u32> = vec![0, 0, 1, 1];
    let cols: Vec<u32> = vec![0, 1, 2, 3];
    let pcm = SparseMatrix::from_coo(2, 4, rows, cols)?;
    let error_rate = 0.05;
    let syndrome = array![1u8, 0];
    // CODE
    Ok(())
}
```

PECOS provides quantum error correction decoders through both Python and Rust APIs. The availability of specific decoders varies by language.

## Overview

The decoder system in PECOS is designed around modularity and performance:

- **Optional Components**: Decoders are not built by default to keep PECOS lightweight
- **External Integration**: LDPC decoders come from specialized external projects
- **Unified API**: Consistent interface across different decoder implementations
- **Cross-Language Support**: Some decoders available in both Python and Rust, others Rust-only

## DEM text grammar

PECOS reads flat Stim detector-error-model text using a strict subset of Stim's grammar:

- Instruction names and detector/observable target prefixes are case-insensitive; tags and inline `#` comments are supported.
- Targets and `^` separators require spacing; separators cannot be first, last, or adjacent.
- PECOS rejects a detector or observable repeated within one component of an error instruction, and an identical component repeated within one instruction, even though Stim accepts them, because it almost always indicates a mistake. Extension `TP` targets follow the same rule. Components are identical when they contain the same set of targets regardless of order: `D0 D1 ^ D1 D0` is rejected, while `D0 D1 L0 ^ D0 D1 L1` contains two distinct components. A target may appear in several components of a decomposed mechanism.
- Parenthesized arguments immediately follow the name or tag. `error` requires one probability in `[0, 1]`, with `error()` meaning zero; detector and logical-observable declarations require exactly one target of the appropriate kind.
- Grammar is validated per instruction; block balance is not checked by the tokenizer. PECOS readers requiring flat DEMs reject `repeat`, `shift_detectors`, and closing braces with a flattening error, including stray braces and unclosed repeat blocks.
- PECOS extension targets and metadata statements require an explicit opt-in: `ParsedDem` and `DetectorErrorModel::with_pecos_dem_metadata` enable the whole PECOS superset, including `TP` targets and JSON metadata statements.

Python constructors report tokenizer errors as `ValueError` with the `Invalid DEM syntax:` prefix. Stim-backed constructors such as `PyMatchingDecoder` and `TesseractDecoder` validate the original text before passing it to their backend, which supports loop expansion and retains its native errors for backend-specific constraints.

## Available Decoders

### Python Decoders

The following decoder APIs and supporting types are publicly re-exported from
`pecos.decoders`:

| API | Primary input | Description |
|-----|---------------|-------------|
| `MWPM2D` | QECC object | Legacy minimum-weight perfect matching for 2D codes. |
| `DummyDecoder` | None | No-op decoder for tests and interface benchmarks. |
| `PyMatchingDecoder` | Graph-like DEM text or `CheckMatrix` | PyMatching minimum-weight perfect matching, with optional correlated decoding. |
| `FusionBlossomDecoder` | Check matrix, standard-code parameters, or a manual graph | Pure-Rust minimum-weight perfect matching. |
| `TesseractDecoder` | DEM text | Search-based decoder that accepts raw hyperedges. |
| `TesseractTrellisDecoder` | DEM text | Sums probability mass over a beam of partial syndromes and reports a per-shot observable probability; accepts at most one observable and a 256-detector active frontier. The probability and `low_confidence` flag come only from this class; the `tesseract_trellis()` spec path yields the observable mask. |
| `DemAwareDecoder` | DEM text | Maps DEM mechanisms and observables onto BP-OSD and other check-matrix decoders. |
| `BpOsdBuilder` / `BpOsdDecoder` | `SparseMatrix` check matrix or DEM text | Belief propagation with ordered-statistics post-processing. |
| `BpLsdBuilder` / `BpLsdDecoder` | `SparseMatrix` check matrix or DEM text | Belief propagation with localized-statistics post-processing. |
| `MinSumBpBuilder` / `MinSumBpDecoder` | Dense check matrix and error priors, or DEM text | Min-sum belief propagation. |
| `RelayBpBuilder` / `RelayBpDecoder` | Dense check matrix and error priors, or DEM text | Relay belief propagation. |
| `UnionFindBuilder` / `UnionFindDecoder` | `SparseMatrix` check matrix or DEM text | Union-find decoding with inversion or peeling. |
| `CheckMatrix` / `SparseMatrix` | Dense or coordinate-form matrix data | Matrix containers used by matching and LDPC decoder constructors. |
| `MwpmResult` / `BpResult` / `TesseractResult` / `TesseractTrellisResult` | Decoder output | Result objects for matching, belief-propagation, and Tesseract decoders. |

The experimental `frontier()` and `bp_trellis()` factories are not part of this table: they import from `pecos.decoders` only when the optional `pecos-rslib-exp` package is installed, and are described in the [Rust-backed Frontier](#rust-backed-frontier-batch-decoding) and [Rust-backed BP-Trellis](#rust-backed-bp-trellis-batch-decoding) sections below.

Python decoder inputs name their encoding explicitly: use
`decode_syndrome(...)` for a dense detector vector and
`decode_from_defects(...)` for sparse detector indices. The BP/LDPC classes'
`from_dem(...)` constructors return a `DemAwareDecoder` wrapper so their results
include `observable_flips` and their instances retain the DEM dimensions.

### Rust Decoders

The Rust API provides access to a broader set of decoders:

**LDPC Decoders** (feature: `ldpc`):

- BP-OSD (Belief Propagation with Ordered Statistics Decoding)
- BP-LSD (Belief Propagation with Localized Statistics Decoding)
- MBP (Min-sum Belief Propagation)
- Belief Find decoder
- Flip decoder
- Union Find decoder
- SoftInfoBP decoder

**Other Decoders**:

- Fusion Blossom MWPM (feature: `fusion-blossom`)
- PyMatching MWPM (feature: `pymatching`)
- Tesseract (feature: `tesseract`)
- Chromobius color code decoder (feature: `chromobius`)

## Installation and Setup

=== ":fontawesome-brands-python: Python"

    Install PECOS with decoder support:

    ```bash
    pip install quantum-pecos
    ```

    The Python package exports the decoder APIs listed above.

=== ":fontawesome-brands-rust: Rust"

    Add decoder dependencies to your `Cargo.toml`:

    ```toml
    [dependencies]
    # Option 1: Use the meta-crate with specific features
    pecos-decoders = { version = "0.1.1", features = ["ldpc"] }

    # Option 2: Use individual decoder crate
    pecos-ldpc-decoders = "0.1.1"

    # Core types (always needed for custom decoders)
    pecos-decoder-core = "0.1.1"
    ```

    Build with LDPC decoders:

    ```bash
    # Build LDPC decoders
    cargo build --package pecos-decoders --features ldpc

    # Build all decoders
    cargo build --package pecos-decoders --all-features
    ```

## Python API

### MWPM2D Decoder

The `MWPM2D` decoder implements Minimum Weight Perfect Matching for 2D topological codes.

```python
import pecos as pc

# Create a surface code first
surface = pc.qeccs.Surface4444(distance=3)

# Create decoder with the QECC
decoder = pc.decoders.MWPM2D(surface)
```

### DummyDecoder

A no-op decoder useful for testing decoder interfaces without actual decoding.

```python
from pecos.decoders import DummyDecoder

decoder = DummyDecoder()
```

## Rust API

### Creating Error Correction Codes

Before using decoders, you need a quantum error correction code:

```rust
use pecos_decoders::SparseMatrix;

// Create a parity check matrix in COO (coordinate) format
let rows: Vec<u32> = vec![0, 0, 1, 1];
let cols: Vec<u32> = vec![0, 1, 2, 3];
let pcm = SparseMatrix::from_coo(2, 4, rows, cols)?;
```

### LDPC Decoders

#### BP-OSD Decoder

Combines belief propagation with ordered statistics decoding post-processing.

```rust
use pecos_decoders::{BpOsdDecoder, BpMethod, OsdMethod, BpSchedule};

// Build decoder with configuration
let mut decoder = BpOsdDecoder::builder(&pcm)
    .error_rate(error_rate)
    .max_iter(100)
    .bp_method(BpMethod::MinimumSum)
    .osd_method(OsdMethod::OsdE)
    .osd_order(10)
    .build()?;

// Decode syndrome
let result = decoder.decode(&syndrome.view())?;

println!("Decoding: {:?}", result.decoding);
println!("Converged: {}", result.converged);
```

#### BP-LSD Decoder

Localized version of OSD for better scaling with large codes.

```rust
use pecos_decoders::BpLsdDecoder;

let mut decoder = BpLsdDecoder::builder(&pcm)
    .error_rate(error_rate)
    .bits_per_step(1)
    .lsd_order(10)
    .build()?;

let result = decoder.decode(&syndrome.view())?;
```

#### Belief Find Decoder

Combines belief propagation with union-find algorithm.

```rust
use pecos_decoders::{BeliefFindDecoder, UfMethod};

let mut decoder = BeliefFindDecoder::builder(&pcm)
    .error_rate(error_rate)
    .uf_method(UfMethod::Inversion)
    .max_iter(10)
    .build()?;

let result = decoder.decode(&syndrome.view())?;
```

#### Flip Decoder

Fast bit-flipping decoder suitable for real-time applications.

```rust
use pecos_decoders::FlipDecoder;

let mut decoder = FlipDecoder::builder(&pcm)
    .max_iter(100)
    .build()?;

let result = decoder.decode(&syndrome.view())?;
```

#### Union Find Decoder

Graph-based decoder using union-find data structure.

```rust
use pecos_decoders::{UnionFindDecoder, UfMethod};

let mut decoder = UnionFindDecoder::builder(&pcm)
    .method(UfMethod::Inversion)
    .build()?;

// Union find decode takes syndrome, LLRs, and bits_per_step
let llrs = vec![0.1; 4];  // one LLR per bit column
let result = decoder.decode(&syndrome.view(), &llrs, 1)?;
```

### Advanced Features

#### Soft Information Decoding

Use log-likelihood ratios for improved decoding performance.

```rust
use pecos_decoders::SoftInfoBpDecoder;

let mut decoder = SoftInfoBpDecoder::builder(&pcm)
    .error_rate(error_rate)
    .max_iter(50)
    .build()?;

// Soft decode takes soft syndrome values, cutoff, and sigma
let soft_syndrome = vec![0.9, 0.1];
let result = decoder.decode(&soft_syndrome, 5.0, 0.5)?;
```

#### Batch Decoding

Decode multiple syndromes efficiently.

```rust
use pecos_decoders::BpOsdDecoder;
use ndarray::array;

let mut decoder = BpOsdDecoder::builder(&pcm).error_rate(error_rate).build()?;

let syndromes = vec![
    array![1u8, 0],
    array![0u8, 1],
];

for (i, syn) in syndromes.iter().enumerate() {
    let result = decoder.decode(&syn.view())?;
    println!("Syndrome {}: {:?}", i, result.decoding);
}
```

#### Performance Tuning

```rust
use pecos_decoders::{BpOsdDecoder, BpSchedule};

let mut decoder = BpOsdDecoder::builder(&pcm)
    .error_rate(error_rate)
    .bp_schedule(BpSchedule::Parallel)
    .omp_threads(4)
    .build()?;
```

### Error Handling

```rust
use pecos_decoders::{BpOsdDecoder, LdpcError};

let mut decoder = BpOsdDecoder::builder(&pcm).error_rate(error_rate).build()?;

match decoder.decode(&syndrome.view()) {
    Ok(result) => {
        println!("Decoding: {:?}", result.decoding);
        println!("Converged: {}", result.converged);
    }
    Err(e) => {
        eprintln!("Decoding error: {}", e);
    }
}
```

## Performance Considerations

1. **Algorithm Selection**:
   - BP-OSD: Best overall performance for most codes
   - BP-LSD: Better for very large codes
   - Flip: Fastest but lower performance
   - Union Find: Good for codes with specific structure

2. **Parameter Tuning**:
   - Start with default parameters
   - Increase `max_iterations` for better convergence
   - Adjust `osd_order` based on code size and error rate
   - Use parallel schedules for larger codes

3. **Hardware Optimization**:
   - Enable CPU-specific optimizations in release builds
   - Use multiple threads for batch decoding
   - Consider memory layout for cache efficiency

## Rust-backed Frontier batch decoding

Install the optional `pecos-rslib-exp` package for this section and BP-Trellis
below. Standard `pecos.decoders` imports do not load the experimental extension.
The explicit `from pecos.decoders import frontier, bp_trellis` convenience import
loads it lazily and raises an actionable `ImportError` if it is unavailable.
Experimental factories are excluded from wildcard imports. Their specifications
work with `SampleBatch.decode(...)` and `DemSampler.decode(...)`; the standard
`DecoderSpec.parse` strings and composite-spec factories do not load optional
providers. The experimental calls cross a Python adapter at each shot, with
native model construction and decoding releasing the GIL.

```python
from pecos_rslib_exp import frontier
from pecos_rslib.qec import SampleBatch

dem = "error(0.1) D0 D1 D2 L0\n"
batch = SampleBatch([[1, 1, 1], [0, 0, 0]], [1, 0])
result = batch.decode(dem, frontier(k=64), workers=2, predictions=True)
assert result.predictions == [1, 0]
assert result.num_errors == 0
```

Frontier accepts raw DEMs, including hyperedges. `workers=None` selects the
worker count automatically; `workers=1` runs sequentially. Parallel execution
releases the Python GIL and preserves shot order. At most one Rust decoder per
worker is alive at a time, so more workers and larger `k` increase memory use.
`SampleBatch.decode(...)` builds exactly one decoder per worker;
`DemSampler.decode(...)` builds one up front to check dimensions and then one per
scheduled group of sampling chunks, which can be several times the worker count
on a long run, so a model that is slow to construct pays that cost more than
once per worker there.

Options match `pecos_rslib_exp.FrontierDecoder.from_dem`: `k`, `delta`,
`score_alpha`, `bp_score_iterations`, `column_order`, `merge_indistinguishable`,
`metric_mode`, and `int_metric_scale`. The default ordering is
`"deadline_reorder"`; `"time_order"`, `"backward_deadline_reorder"`, and explicit
column permutations are also accepted. Frontier remains experimental, and
pruning can make predictions approximate. For per-shot logical masses, pruning
status, and complementary gaps, use the direct experimental binding.

### Strict, reporting, and always-answer Frontier calls

The direct `FrontierDecoder.decode_syndrome`, `decode_from_defects`, and
`decode_batch` methods share a Rust recovery implementation. They accept
`on_no_path="raise"` (default), `"report"`, or `"fallback"`, and an optional
`delta_max`. Invalid inputs, configuration errors, and engine faults still raise
in every mode. Always-answer means a prediction is returned for a normal
no-path outcome; it does not suppress programming errors.

```python
from pecos_rslib_exp import FrontierDecoder

decoder = FrontierDecoder.from_dem(
    "error(0.1) D0 D1 L0\n",
    k=256,
    delta=16,
    fallback="feasible_then_zero",
)

# Strict: raise if no configured attempt finds a path.
assert decoder.decode_syndrome([1, 1]).observable_flips.mask == 1

# Reporting: retain successful shots and explicit no-path reports in order.
outcomes = decoder.decode_batch(
    [[0, 0], [1, 0], [1, 1]],
    workers=2,
    on_no_path="report",
    delta_max=64,
)
assert [outcome.no_path for outcome in outcomes] == [False, True, False]
assert outcomes[1].cause == "infeasible"

# Always-answer: use the explicitly configured fallback after attempts fail.
answer = decoder.decode_syndrome([1, 0], on_no_path="fallback", delta_max=64)
assert answer.fallback_used
assert answer.status == "fallback:zero"
assert answer.syndrome_consistent is False
```

With `delta=16, delta_max=64`, attempts use 16, 32, then 64, stopping at the
first success. A non-power-of-two maximum is included as the final cap. A zero
base grows first to `min(1, delta_max)`. The maximum must be finite and at least
the base delta; omitting it disables retries. K stays fixed. Attempts restart
DP using one prepared model and one BP preparation per shot. Residual or
infeasible failures skip remaining attempts because widening cannot help.
`delta_max` bounds the pruning window, not wall-clock runtime.

The returned types are distinct:

- `FrontierResult`: a decoded prediction, `no_path=False`, `fallback_used=False`.
  `deltas_tried`, `delta_used`, and `escalation_rungs_used` record recovery.
  Transitions count all attempts; evidence and pruning diagnostics describe the
  successful attempt, whose prediction can still be approximate.
- `FrontierNoPath`: no prediction, `no_path=True`, `fallback_used=False`.
  `cause` is `"residual"`, `"infeasible"`, or `"exhausted"`. Residual includes a
  `detector` that cannot be changed; exhausted indicates failure after pruning,
  which alone does not prove incompatibility. Reports expose `deltas_tried`,
  total `transitions`, total `dropped_states`, and one-shot `bp_seconds`.
- `FrontierFallback`: an explicit prediction, `no_path=True`, `fallback_used=True`.
  `failure` retains the original no-path report. It has no likelihood or
  confidence estimate. Its `observable_flips` can be nonzero.

Configure `fallback="feasible_then_zero"` to prepare a reusable GF(2) solver for
an independent binary DEM. It chooses a probability-ordered column basis and
verifies the resulting fault explanation against the syndrome. A successful
fallback has `status="fallback:feasible"`, `syndrome_consistent=True`, and
`mechanism_indices` referring to the original DEM. This is a deterministic
heuristic, not a maximum-likelihood or minimum-weight decoder. Preprocessing
uses additional memory for the basis and witnesses, shared across batch workers;
`build_seconds` reports Frontier construction only, excluding that preparation.

If the DEM has no explanation, this policy returns zero observable flips with
`status="fallback:zero"` and `syndrome_consistent=False`. An explicitly selected
`fallback="zero"` always uses zero after no-path and reports consistency as
`None` (unverified). Neither zero prediction is presented as a successful decode.
Configuring a fallback does not activate it until `on_no_path="fallback"` is
requested. Feasibility handles p=0/p=1 and joint `^` components, but is rejected
for categorical `from_factors` models; those can use the explicit zero policy.

Syndrome consistency does not imply logical identity. For the model
`error(0.1) D0 L0`, a fired D0 requires logical mask 1; no identity-sector
explanation exists. A DEM mechanism explanation is also not automatically a
physical data-qubit Pauli correction. Record fallback and incompatible-shot
rates alongside logical error rates. Do not silently drop undecoded shots from
the denominator; always-answer predictions can be scored across all shots.

Rust callers use `pecos_frontier::recovery::FrontierRecoveryDecoder` with
`RecoveryOptions`, `NoPathAction`, and `FallbackPolicy`. The normal
`FrontierDecoder` remains the strict parity-port engine. The shared
`pecos_decoder_core::feasibility::DemFeasibility` utility can also be prepared
and queried independently of Frontier.

```rust
use pecos_frontier::{FrontierConfig, SparseDem};
use pecos_frontier::recovery::{
    FallbackPolicy, FrontierRecoveryDecoder, FrontierRecoveryOutcome,
    NoPathAction, RecoveryOptions,
};

let dem = SparseDem::from_dem_str("error(0.1) D0 D1 L0")?;
let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(
    &dem,
    FrontierConfig { k: 256, delta: 16.0, ..Default::default() },
    Some(FallbackPolicy::FeasibleThenZero),
)?;
let outcome = decoder.decode(&[1, 0], RecoveryOptions {
    delta_max: Some(64.0),
    on_no_path: NoPathAction::Fallback,
})?;
assert!(matches!(outcome, FrontierRecoveryOutcome::Fallback {
    syndrome_consistent: Some(false), ..
}));
```

Rust `decode_batch` preserves per-shot `Result` values in input order. Python
batch calls preserve normal no-path/fallback outcomes but still raise an indexed
exception for other errors. Rust callers can also bind options using
`decoder.with_options(options)?` to obtain an `ObservableDecoder` adapter for
a shared batch `DecoderFactory` (with native batch dispatch disabled).
`decode_outcome` preserves reports; prediction-only `decode_obs` returns an error
when reporting supplies no mask. `FrontierCommitteeDecoder` remains strict.

The `frontier(...)` specification accepts the same recovery controls for both
`SampleBatch.decode` and `DemSampler.decode`:

```python
from pecos_rslib_exp import frontier

spec = frontier(
    k=256,
    delta=16,
    delta_max=64,
    on_no_path="fallback",
    fallback="feasible_then_zero",
)
result = batch.decode(dem, spec, workers=4, predictions=True)
print(result.num_errors, result.num_fallback, result.num_undecoded)
fallback_errors = sum(r.logical_error for r in result.recovery if r.fallback_used)
```

Every supplied prediction, including fallback, is compared against the shot's
true logical observables. A syndrome-compatible fallback with the wrong logical
mask counts as an error. `recovery` contains shot indices, actual `deltas_tried`,
`fallback_used`, `syndrome_consistent`, `undecoded`, and `logical_error` for shots
requiring retries, reporting, or fallback. These records are retained even when
`predictions=False`, so fallback success can be measured separately.

With `on_no_path="report"`, retained `predictions` contains `None` for undecoded
shots. `num_scored` excludes those shots and `num_errors` counts mismatches among
supplied predictions. The full-batch `logical_error_rate` (including timing
stats) is NaN and `interval()` raises when any shot is undecoded; neither an
identity placeholder nor an artificially reduced error rate is reported. Rust
`DecodeResult` exposes `partial_predictions` in this case and leaves its
complete `predictions` absent. Strict defaults and version-1 providers retain
their existing behavior. Recovery specifications require both the standard and
experimental Python extensions to support provider protocol version 2.

## Rust-backed BP-Trellis batch decoding

```python
from pecos_rslib_exp import bp_trellis
from pecos_rslib.qec import SampleBatch

dem = "error(0.1) D0 D1 D2 L0\n"
batch = SampleBatch([[1, 1, 1], [0, 0, 0]], [1, 0])
spec = bp_trellis(
    k=8,
    delta=100.0,
    score_alpha=0.8,
    bp_score_iterations=5,
    merge_indistinguishable=True,
    ordering="deadline",
    escalation=[(64, 100.0)],
)
result = batch.decode(dem, spec, workers=2, predictions=True)
assert result.predictions == [1, 0]
assert result.num_errors == 0
```

The ladder uses one engine model and prepares each shot once, including BP.
`escalation=[(k, delta), ...]` chooses both pruning parameters per rung;
`escalation_ks=[k, ...]` is shorthand for rungs at the base `delta`. Pass only
one ladder keyword. The default ladder is empty. Rungs need not be monotone,
and an exact base cannot have a ladder. Rungs run only after a no-path attempt
that dropped states: never after a successful but wrong prediction, and never
after a residual or infeasible no-path.

### Recommended ladder

The default ladder is empty so that the per-shot cost of a configuration is
exactly what it says. When a no-path shot should be retried rather than
reported, the evidence-backed setting is one rung at `k = 64` with the base
`delta`, as in the example above. On the bivariate-bicycle 144 corpus at
physical error rate 0.003, a `4 -> 64` ladder reached 8 failures per 1000 shots
at 49 ms per shot, against 10 failures at 101 ms for a single `k = 16`; an
`8 -> 64` ladder reached 0 failures per 1000 at 72 ms per shot at rate 0.002.
On the rotated surface code at distance 5 no shot ever escalated, so the rung
costs nothing there. A ladder never rescues a confidently wrong prediction,
only a no-path, and a fully exact configuration (`k` unbounded, `delta`
infinite) exhausted memory on that corpus, so the last rung should stay
finite.

The direct `BpTrellisDecoder.decode_syndrome`, `decode_from_defects`, and
`decode_batch(shots, workers=1)` methods accept `on_no_path="raise"` (default)
or `"report"`. Report mode returns a `BpTrellisNoPath` in place for each failed
shot, preserving batch order. Its `cause` is `"residual"` when a detector's
residual cannot be changed, `"infeasible"` when an attempt proves there is no
path without pruning, or `"exhausted"` when all attempts fail after pruning.
Residual and infeasible outcomes skip remaining rungs. Other errors still raise.

Both outcome classes expose `no_path`, `transitions`, `bp_runs`, and
`bp_seconds`. Only a decoded result exposes `observable_flips`; a report exposes
`placeholder_flips` instead, the forced contribution of probability-one
mechanisms, including wide observables. The names differ on purpose, so code
written for a correction raises `AttributeError` on a report rather than
silently consuming a placeholder. The report also exposes `detector` (only for
residual) and `rungs_tried`.
BP time is counted once, while transitions sum all attempts.
The `bp_trellis(...)` spec route remains strict and accepts no `on_no_path`.

`ordering` also accepts `"backward_deadline"`, `"time_order"`, or an explicit
mechanism permutation. BP-Trellis uses floating-point coset masses and does not
expose Frontier's integer metric options.

Like Frontier, BP-Trellis accepts raw hyperedges and arbitrary-width observables,
releases the GIL during batch decoding, and supports automatic worker selection
and `DemSampler.decode(...)`. It remains experimental. Use
`pecos_rslib_exp.BpTrellisDecoder` for per-shot confidence, pruning status, and
retry telemetry; the unified batch result returns predictions and aggregate scores.

## Hyperedge models and matching decoders

Graph decoder specs (PyMatching, Fusion Blossom and its perturbed correlated
variant, PECOS UF, K-MWPM, non-hybrid belief matching, and `astar`) project raw
hyperedges themselves. Projection uses disjoint components already present as
standalone faults in the model, preserving probabilities and observable effects.
A DetectorErrorModel object decodes its standard text plus its `fault_sector` labels.
Those labels guide the choice of graphlike components.
Use `batch.decode(dem, spec)` or `dem.build_decoder(spec)` to preserve them.
Saving with `batch.save(path, dem=dem)` also preserves the object's labels in
corpus format v2. `SampleBatch.load(path).decode(None, spec)` uses the stored
text and labels. A labelled embedded model has object semantics, including
provider refusal; passing `loaded.dem` explicitly decodes unlabelled text.
All-null labels (including v1 corpora) retain text semantics and provider support.
`loaded.dem_fault_sectors` returns a fresh, dense list of labels or `None` entries;
batches without an embedded model return `None`. Corpus identity checks only text.
Resaving with equal embedded text preserves labels, including with
`allow_dem_mismatch=True`; different text writes all-null labels. An object
always supplies its own labels. Saving never changes the source batch.
Every save writes v2; new readers accept v1 and v2, while older installations
reject v2 files. `loaded.format_version` reports the version actually loaded.
Both results expose `.projection`. It is `None` when no projection was needed or
when a composition without a shared policy projects inside its members and has
no shared report. Text input has no fault-sector labels.

The default `irreducible="refuse"` rejects faults that cannot be decomposed and
reports their count and input indices. To remove those faults from the decoder's
model explicitly, use `pymatching(correlated=True, irreducible="drop")` or the
string `"pymatching:irreducible=drop"`. Sampling still uses the original model.
Shots that fire a dropped fault can make some graph decoders fail on detectors
left without edges ([#1266](https://github.com/PECOS-packages/PECOS/issues/1266)).
Alternatively, use a hyperedge-capable decoder such as `bp_osd()` or `tesseract()`.
Effects with more than 12 detectors or reducible only with overlapping components
are refused under either policy; `irreducible="drop"` does not apply to them.
Use a hyperedge-capable decoder such as `bp_osd()` or `tesseract()` for these effects.

Supplied `^` decompositions remain supported and are used verbatim when every
component touches at most two detectors. Mixing them with raw hyperedges is an
error. Hybrid belief matching still takes separate full and decomposed models.
Windowed decoding projects the full model using the inner spec's policy before
windowing; beam search still requires graphlike input in its phase-1 windows.
A graph decoder in beam phase 2 projects the full residual model under Refuse.
Drop is rejected in compositions without a shared projection report, including
beam search and mixed graph/hypergraph ensembles. Every graph member must share
the policy and the composition must resolve to a shared projection before Drop
is allowed. Phase 2 may use a hyperedge-capable decoder selected through Rust
`BeamSearchConfig.phase2` or `pecos_rslib.decoders.beamsearch(phase2=...)`;
the beamsearch string grammar has no `phase2` key.

See
[Experimental Decoders](../experimental/decoders.md) for the Frontier and
BP-Trellis decoders, which additionally report a per-shot complementary gap,
and for provenance-based decomposition of a hyperedge model into a graphlike
one. Frontier's exact default-float masses are true unnormalized posteriors.
Under its integer `maxlog_int` metric (accepted by Python `FrontierDecoder` as
`metric_mode="maxlog_int"`; the committee remains float-only), terminal masses
are per-label best-route (Viterbi) masses, the gap is a route-mass margin, and
`log_evidence` is the winner's mass rather than evidence.

## See Also

- [Getting Started Guide](getting-started.md) - Main installation guide
- [LLVM Setup Guide](llvm-setup.md) - For building with LLVM support
- [Experimental Decoders](../experimental/decoders.md) - Frontier, BP-Trellis, complementary gap
