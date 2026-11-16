# pecos-selene-core

Shared Rust adapter for exposing PECOS simulators as Selene plugins. The stabilizer,
state-vector, and Clifford-RZ plugins use this crate to share operation dispatch
and shot lifecycle handling.

Plugin implementations provide `SeleneSimBehavior` for simulator-specific
initialization, rotations, reset, and optional operations. `SeleneAdapter` wraps
that behavior and implements Selene's `SimulatorInterface`, including qubit bounds
checks, measurement, and per-shot setup.

See the [crate documentation](src/lib.rs) for the behavior trait and adapter.
