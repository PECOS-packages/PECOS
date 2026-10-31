# PECOS runtime plugin for Selene

A normal Selene runtime plugin backed by the reusable `pecos-selene-runtime`
Rust crate. It implements lazy virtual-Z scheduling, configurable gate durations
and batching, and ordinary/leakage measurement futures. It does not model
Anduril's ion transport or reproduce its hardware schedule.

From a PECOS checkout:

```sh
uv pip install -e 'python/selene-plugins/pecos-selene-runtime-plugin[test]'
```

Use it like any other Selene runtime:

```python
from pecos_selene_runtime_plugin import PecosRuntimePlugin
from selene_sim import Stim, build

runtime = PecosRuntimePlugin(max_batch_size=2)
runner = build(program.compile())
shots = [dict(shot) for shot in runner.run_shots(Stim(), runtime=runtime, n_qubits=2, n_shots=10)]
```

The same plugin object works with an updated PECOS build:

```python
import pecos

shots = (
    pecos.sim(pecos.Guppy(program))
    .classical(pecos.selene_engine(runtime=runtime))
    .quantum(pecos.stabilizer())
    .qubits(2)
    .run(10)
)
```

Gate durations default to zero nanoseconds and `max_batch_size` defaults to one.
`max_queued_batches` defaults to 4096. `max_live_results` defaults to `None`
(no cap), matching SoftRZ's acceptance of long measurement streams. Set a
positive integer to bound retained futures when the program and host retire
unused handles. PECOS retires its temporary eager-measurement futures; the
pinned Selene host retains them until shot end, so an explicit cap also limits
eager measurements per shot there. Completed lazy futures remain readable
until their references are released; they are never evicted to meet a cap.

Hosts must drain ready batches. Results are deterministic for a given input
stream and configuration; randomness belongs to the simulator and error model.

For development builds, pass `library_path=Path(...)` to select the shared library
produced by `cargo build -p pecos-selene-runtime-plugin`. Set
`PECOS_RUNTIME_TEST_LIBRARY` to that absolute path when running the plugin tests
without installing a wheel.

The build hook requires the containing PECOS workspace. Distributed wheels
include the shared library and require only `selene-core~=0.3.0`; source-only
installation outside a PECOS checkout is not supported.

## Python integration checks

The core Python CI setup (`just python-ci-build-test`) installs this plugin and
builds the current PECOS extension. With that environment, run:

```sh
uv run --frozen pytest python/quantum-pecos/tests/pecos/test_pecos_runtime_plugin.py
```

These tests run one Guppy feedback program through both hosts, then exercise
the shared LLVM feedback fixture through PECOS's flat, scheduled and scheduled
event routes, with multiple workers and repeated runs. They cover batch sizes
one and two and load the packaged library by default.

The leakage cases inject gate noise on the flat route and idle noise on the
scheduled routes. They check that an exact outcome of `2` survives cached future
reads and triggers a conditional reset to `0`. Zero-noise and zero-duration
controls remain in `|1>`. The one-second idle used to make leakage deterministic
is an artificial test duration, not a hardware timing recommendation.

The lifetime regressions hold a lazy future across repeated eager measurements
with `max_live_results=2` on all three PECOS routes, and verify that standalone
Selene can perform 65,537 eager measurements with the default configuration.
