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

For a first experiment, `GeneralNoiseParameters.uniform(1e-3)` applies a
common process-infidelity/error probability to preparation, measurement,
one-qubit, and two-qubit operations.

The fluent method names intentionally follow PECOS's Rust
`GeneralNoiseModelBuilder`. Parameter objects are immutable: every `with_*`
call returns a new validated configuration, making presets safe to reuse.
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

## Development

From the PECOS repository root:

```bash
uv run --package pecos-selene-general-noise --extra test pytest \
  python/selene-plugins/pecos-selene-general-noise/tests
cargo test -p pecos-selene-general-noise
```
