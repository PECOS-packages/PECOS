# pecos-selene-general-noise-model

PECOS general noise as a Selene error model. `GeneralNoiseErrorModel` implements
Selene's `ErrorModelInterface`; the `pecos-selene-general-noise` plugin exports it
to Selene, and other Rust plugins can depend on this crate to reuse it.

## Composing a Rust plugin

Construct the model with `GeneralNoiseErrorModel::from_json(n_qubits, configuration)`.
The configuration is the JSON argument emitted by Python's
`GeneralNoisePlugin.get_init_args()`. Use the same Selene core revision as this
crate.

This crate exports no FFI entry points, so a downstream plugin exports its own
factory (`export_error_model_plugin!`) without duplicate symbols, including when
it is built together with `pecos-selene-general-noise`.

A downstream adapter owns any runtime-specific interpretation and shot state.
It must preserve batch boundaries, timing, operation ordering and measurement
identities when passing normalized batches to the model. Invoke `shot_start`,
`handle_operations` and `shot_end` in Selene lifecycle order, and discard the
instance after an execution error. The constructor does not install an adapter,
acknowledge metadata or enable custom operations: unhandled custom operations
still fail.
