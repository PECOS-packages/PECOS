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

## Explicit measurement exposures

Runtime adapters that determine exposure placement can use
`ExplicitCrosstalkErrorModel::new(n_qubits, &config)` instead of the automatic
adapter. Supply a complete normalized `BatchOperation` and an ordered slice of
`CrosstalkPlacement` to `handle_operations`. Each placement inserts a
`CrosstalkExposure::Local(victims)` or `GlobalExcluding(exclusions)` before an
ordinary operation index; `operations.len()` means after the final operation.
Equal indices preserve slice order, including repeated exposures. Lists contain
distinct, in-capacity qubits; local victim order is preserved. Global effects
use the noise model's tracked prepared qubits and exclusion semantics.

The wrapper must explicitly supply an empty slice when a batch has no exposures.
No automatic measurement crosstalk is added in this mode. The type deliberately
does not implement `ErrorModelInterface`, so downstream plugins must implement
that interface and explicitly provide placements. The standard plugin and its
JSON envelope retain automatic crosstalk behavior; no Python mode switch is added.

Each complete batch uses one noise-controller lifecycle, including every
noise-to-simulator exchange needed for internal measurements. Only original user
measurement results return to Selene. Exposures do not advance qubit occupancy or
split deferred idle periods. An exposure before an ordinary operation also
precedes that operation's deferred idle noise; this is a placement contract, not
a continuous-time integration of noise through the gap. Resets keep the ordinary
adapter's pre-reset idle rule.

The whole batch is checked before execution, including timing overflow/overlap,
angles, capacity, duplicate result IDs and exposure placement. Invalid admission
can be corrected without losing the shot. Once execution starts, any error or
caught panic permanently poisons the instance. Recovery requires a new adapter
and a reset simulator; `shot_start` cannot clear a partial execution. Successful
shots use `shot_start` / `handle_operations` / `shot_end` in order.

Capacity is 1–16384 qubits. A batch admits at most 4096 ordinary operations plus
exposures and 16384 listed targets plus global candidate targets (capacity is
charged for every global record). Execution stops after 65536 controller
exchanges or before dispatching more than 262144 cumulative simulator operations.
These are failure limits, not a guarantee that all accepted noise configurations
fit, nor a bound on allocations made internally by the general noise model.
