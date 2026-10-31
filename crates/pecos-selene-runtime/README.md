# PECOS Selene runtime

An opt-in, deterministic virtual-Z runtime implementing Selene's existing
`RuntimeInterface`. It schedules operations; it does not simulate quantum states
or sample noise. It is inspired by Quantinuum's Apache-2.0 SoftRZ runtime at
revision `458d640dc4cf84a7a7988ea65ce5a9f7bcb8c511`.

The implementation is a normal Rust library. The thin `pecos-selene-runtime-plugin`
workspace package exports the standard Selene descriptor and supplies the Python
`PecosRuntimePlugin`. Both hosts run the same scheduler. There is no separate PECOS
runtime protocol, and the PECOS integration currently loads the shared library.

## Rust use

```rust
use pecos_selene_runtime::{PecosRuntime, RuntimeConfig};
use selene_core::runtime::RuntimeInterface;

let mut runtime = PecosRuntime::new(2, 0.into(), RuntimeConfig::default())?;
runtime.shot_start(0, 42)?;
let q = runtime.qalloc()?;
runtime.reset(q)?;
let result = runtime.measure_leaked(q)?;
runtime.force_result(result)?;
while let Some(batch) = runtime.get_next_operations()? {
    // Send the emitted operations to a simulator/noise consumer, then return
    // every measurement outcome through set_bool_result / set_u64_result.
}
# Ok::<(), anyhow::Error>(())
```

Rust callers use Selene's trait directly; a specialized direct PECOS dispatch
path can be added later without changing the scheduler or introducing another
interface.

## Contract

- RZ rotates a software phase; subsequent RXY axes incorporate that phase.
  RXY, RZ and RZZ are accepted. RPP is unsupported and must be decomposed by the
  caller. Noise applies to emitted operations, not eliminated virtual rotations.
  Phase tracking currently accumulates `f64` angles without range reduction:
  extreme magnitudes can lose smaller intervening rotations, for example
  `RZ(2^54); RZ(1); RZ(-2^54)`. Bounded-frame tracking needs a coordinated change
  to the runtime and PECOS source-provenance matching.
- Independent operations of the same kind can share a batch up to
  `max_batch_size`. Measurement emissions retain source order, including across
  ordinary and leakage measurements. This preserves reused PECOS result slots.
  Alternating measurement kinds can therefore produce more batches and longer
  modeled schedules than SoftRZ, which can move later measurements ahead of
  earlier ones. The two runtimes need not produce identical timing-dependent
  noise results.
- Measurements return unique futures. Submission does not force them. Forcing
  either measurement kind releases a queue prefix through that measurement;
  unrelated later work can remain queued. An empty drain means no work is ready,
  not that the program is finished.
- A force or barrier seals released batches against later insertions.
  Local barriers flush a prefix through the selected qubits' last operations.
  Their sleep applies to those qubits; global barriers and `simulate_delay` apply
  globally. Timing arithmetic is checked. This is a generic batch scheduler,
  not an Anduril transport model or a hardware timing prediction.
- The host drains after submissions. At `max_queued_batches` the queue becomes
  ready automatically; further undrained growth is rejected. PECOS drains between native submissions, including decomposed gates.
- Outcomes must arrive once, after emission, in the valid range (0/1 for ordinary
  measurements, 0/1/2 for leakage measurements). A leakage value of 2 cannot be
  silently read as a Boolean.
- Futures start with one reference. Retired futures cannot be read or forced;
  a discarded, unfinished future remains internally until feedback arrives.
  `max_live_results` optionally bounds retained futures; it defaults to no cap.
  Hosts forward reference-count changes from the program; feedback alone does
  not retire a referenced future. PECOS releases the temporary future after an
  eager measurement. The pinned Selene host does not, so an explicit result cap
  there also limits the number of eager measurements per shot. The default
  preserves compatibility without evicting still-readable lazy futures.
- Qubit release seals its earlier work; reallocation resets its virtual phase.
  Measurement itself does not free a qubit. Reset retains the software frame,
  matching SoftRZ's frame convention.
- `shot_end` releases shutdown work and keeps futures valid for feedback.
  `shot_start` requires the previous shot to be ended and drained, then resets
  allocation, phase and result state while preserving the continuous clock.
  Both Selene and PECOS can drain the shutdown work and return its feedback
  after the end hook.

## PECOS integration

Select the Python plugin object with `pecos.selene_engine(runtime=plugin)`.
It declares its native gates; PECOS's existing Selene adapter preserves lazy
measurement submission and forces the requested native future when read.
Rust callers configure the same adapter explicitly:

```rust,ignore
let runtime = pecos_qis::SeleneRuntime::with_plugin_config(
    library_path, vec!["--max-batch-size=2".into()], vec![],
)
.with_native_gate_set(pecos_qis::RuntimeNativeGateSet::RXY_RZ_RZZ);
```

The runtime's native future getters supply program-visible values after physical
measurement feedback. Both ordinary and leakage-aware futures support targeted
forcing. PECOS's flat and scheduled consumers use the same standard plugin.
Host calls, including metadata and qubit lifetime operations, can still affect
release boundaries, so different input streams need not produce identical schedules.

## Validation

```sh
cargo test -p pecos-selene-runtime
cargo test -p pecos-qis --lib pecos_runtime -- --test-threads=1
cargo build -p pecos-selene-runtime-plugin
```

The Python plugin tests exercise standalone Selene. The portable
`tests/fixtures/feedback.ll` is used by both hosts and includes ordinary and
leakage futures, conditional feedback, cached reads and unread shutdown work.
The Python integration suite also uses `tests/fixtures/leakage_feedback.ll` to
exercise actual leakage outcome `2` and conditional reset, with gate noise and
nonzero-duration idle noise on the flat and scheduled routes.
