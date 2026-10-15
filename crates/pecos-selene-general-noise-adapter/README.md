# pecos-selene-general-noise-adapter

PECOS general noise as a Selene error model. `GeneralNoiseErrorModel` implements
Selene's `ErrorModelInterface`; the `pecos-selene-general-noise` plugin exports it
to Selene, and other Rust plugins can depend on this crate to reuse it.

## Composing a Rust plugin

Construct the adapter with `GeneralNoiseErrorModel::new(n_qubits, &config, &local_groups)`,
where `config` is a `pecos_engines::GeneralNoiseConfig`. For the JSON argument emitted
by Python's `GeneralNoisePlugin.get_init_args()`, use
`GeneralNoiseErrorModel::from_plugin_json(n_qubits, configuration)`. Its envelope is
`{"model": {...}, "local_groups": [[0, 1]]}`; local groups default to empty.
Use the same Selene core revision as this crate.

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

Operations in a runtime batch may share qubits, including when the batch has a
positive duration. Each touched qubit accrues idle only before its first operation
in the batch, measured from its previous batch's end; all touched qubits advance
to the current batch's end after translation. Overlapping batches on a qubit are
rejected. A qubit's first operation in a shot accrues no idle; resets retain the
same idle timing rule as other operations.
