# PECOS General Noise Selene Plugin

This package exposes PECOS's general noise model as a device-neutral Selene
error-model plugin. It contains no calibrated hardware preset: every error
channel is disabled until the user configures it. Consequently,
`GeneralNoiseParameters()` is guaranteed to produce a noiseless model.

```python
from pecos_selene_general_noise import GeneralNoiseParameters, GeneralNoisePlugin

parameters = (
    GeneralNoiseParameters()
    .with_p_prep(1e-3)
    .with_p_meas_0(2e-3)
    .with_p_meas_1(3e-3)
    .with_average_p1(1e-4)
    .with_average_p2(1e-3)
    .with_p_idle_linear(2e-3, {"Z": 1.0})
)
noise = GeneralNoisePlugin(parameters=parameters, random_seed=7)
```

The same immutable parameter object can configure PECOS ``sim()`` directly, so
parity experiments do not need to duplicate a long setter chain:

```python
from pecos import sim

pecos_results = sim(program).noise(parameters).seed(7).run(1_000)
```

``local_groups`` is Selene-adapter topology rather than a
``GeneralNoiseModel`` parameter. Direct ``sim().noise(parameters)`` use rejects
parameters containing local groups rather than silently dropping topology.
Call ``parameters.to_pecos_builder(seed=...)`` when an explicit native builder
is useful; that lower-level conversion does not copy local groups.
End-to-end crosstalk comparisons must arrange for the two runtimes to emit the
same global and local crosstalk payloads. The channel probabilities and models
themselves are shared exactly.

For a first experiment, `GeneralNoiseParameters.uniform(1e-3)` applies a
common process-infidelity/error probability to preparation, measurement,
one-qubit, and two-qubit operations.

The fluent method names intentionally follow PECOS's Rust
`GeneralNoiseModelBuilder`. Parameter objects are immutable: every `with_*`
call returns a new locally validated configuration, making presets safe to
reuse. Cross-field scale combinations are validated when the plugin is created,
so compensating scales may be supplied in either fluent-call order.
`with_seed` is intentionally not a parameter method because Selene owns and
supplies the per-shot error-model seed; set `random_seed` on
`GeneralNoisePlugin` instead. PECOS's historical `auto()` demonstration preset
is also omitted so that this package remains explicitly device-neutral.

## Capabilities

- process infidelity or average infidelity for gate errors
- custom Pauli and spontaneous-emission distributions
- leakage, seepage, and continuous leakage-to-depolarization scaling
- asymmetric readout errors
- linear stochastic, sine-squared stochastic, and coherent idle noise
- angle-dependent two-qubit noise and post-two-qubit idle sites
- all-to-all and topology-defined local crosstalk
- per-family and global scaling, plus arbitrary PECOS noiseless gate names

Idle rates use seconds because the adapter converts Selene's nanosecond schedule
to the units expected by PECOS. Local crosstalk is described with neutral qubit
groups rather than a hard-coded device layout.

The adapter supports Selene's RXY, RZ, RZZ, RPP, reset, Boolean measurement, and
leakage-valued measurement operations. Selene `RPP` operations map to PECOS's
`RXYXY2Q` gate, so the noise model applies one two-qubit gate-error channel.
Device-specific `Custom` operations are outside this generic adapter.

## Development

The test suite includes a device-neutral statistical conformance framework. It
checks small analytic outcome distributions against samples from the native
plugin, rejects circuits that cannot distinguish a channel from their comparison
model, and runs against both Stim and PECOS StateVec. See
[`tests/README.md`](tests/README.md) for its coverage and extension rules.

From the PECOS repository root:

```bash
uv run --package pecos-selene-general-noise --extra test pytest \
  python/selene-plugins/pecos-selene-general-noise/tests
cargo test --release -p pecos-selene-general-noise-model
```


## Gate-hook channels

Configure ordered before/after hooks with `TransitionHook`, `PauliLeakageHook`,
and `IndependentHook`. Both families share one list per site, preserving insertion
order through Selene and the optional `sim()` bridge:

```python
from pecos_selene_general_noise import GeneralNoiseParameters, PauliLeakageHook, TransitionHook

parameters = (
    GeneralNoiseParameters()
    .add_p1_pauli_leakage_channel_before_gate(PauliLeakageHook(1.0, {"L": 1.0}))
    .add_p1_transition_channel_before_gate(TransitionHook(1.0, {"L->0": 1.0}))
)
```

Transition maps accept flat `"source->destination"` keys or nested rows.
Single-qubit sites require one-character labels. At two-qubit sites, one-qubit
hooks draw independently for each operand; two-character labels describe a joint
channel (including transition wildcards). `IndependentHook(first, second)` supplies
distinct one-qubit hooks of the same family for the two operands.

The `with_p1_*_channels_*_gate` and `with_p2_*_steps_*_gate` methods replace that
family's entries and append replacements after the retained other family.
The corresponding `add_*` methods append one hook. Outer probabilities use the
site scale and overall scale; independent legs are checked separately. Python
validates shape and probabilities; native channel constructors validate labels
and rows and report configuration errors at plugin initialization.

## Composing a Rust plugin

The noise model itself lives in the `pecos-selene-general-noise-model` crate
(`crates/pecos-selene-general-noise-model`); this plugin only exports it to
Selene. Downstream Rust plugins depend on that crate and export their own
factory. See its README for the composition contract.
