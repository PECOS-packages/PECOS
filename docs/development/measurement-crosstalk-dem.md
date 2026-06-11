# Measurement Crosstalk DEM Semantics

This note records the intended long-term semantics for adding measurement-crosstalk
sources to PECOS detector error model generation. The goal is to keep simulator
noise, traced-QIS circuits, raw hypergraph DEMs, and decomposed decoder inputs
consistent without silently substituting an unrelated scalar noise model.

## Runtime Source Model

The general noise model represents measurement crosstalk with payload gates:

- `MeasCrosstalkLocalPayload` identifies local victim qubits.
- `MeasCrosstalkGlobalPayload` identifies excluded active measurement qubits; the
  victims are the live prepared qubits not listed in the payload.

For each candidate victim, the simulator independently samples a crosstalk event
with the local or global payload probability. When an event occurs, the simulator
performs a hidden `MZ` on the victim and samples a transition from the configured
model:

- `0->0` and `1->1` leave the victim unchanged.
- `0->1` and `1->0` apply `X`.
- `0->L` and `1->L` leak the victim; with `leak2depolar`, leakage is replaced by
  an explicit depolarized Pauli/no-op branch in the simulator.

The transition is conditioned on the hidden measurement outcome, so the exact
channel is not always a fixed Pauli channel independent of circuit state.

## DEM Requirements

A crosstalk DEM implementation should satisfy these constraints:

- Preserve crosstalk as a first-class source family in source metadata.
- Use the actual payload placement from the traced/lowered circuit.
- Derive global-payload victims from the live prepared qubit set at that point in
  the circuit, not from static qubit count alone.
- For exact mode, replay each hidden-measurement/transition branch against the
  same detector and observable metadata used by the ideal circuit.
- Fail loudly if a crosstalk branch changes measurement dependencies in a way that
  cannot be represented as a detector/observable flip against the ideal record.
- Keep any Pauli-twirled or averaged treatment explicit and opt-in; it must not be
  used under a name that implies exact crosstalk DEM support.

## Implementation Plan

1. Extend `NoiseConfig` with measurement-crosstalk local/global probabilities,
   transition weights, and an explicit approximation mode.
2. Add crosstalk payload source extraction in the circuit-aware DEM path.
3. Reuse the exact branch replay machinery where possible: compute the ideal
   measurement parity expressions once, then evaluate branch effects by replaying
   hidden `MZ` plus the transition action at each payload victim.
4. Emit raw hypergraph DEM contributions with crosstalk source metadata.
5. Extend source-level decomposition so graph-like decoder inputs preserve the
   same crosstalk source identity and fail loudly on irreducible branch effects.
6. Thread the new options through Python bindings and surface helper APIs.
7. Keep coverage diagnostics reporting crosstalk as omitted until one of these DEM
   modes is explicitly enabled and tested.

## Minimal Tests

The first implementation tests should cover:

- Local payload with a single deterministic victim where `0->1` produces the same
  detector flip as an `X` at that spacetime point.
- Local payload with `0->0`/`1->1` only, producing no DEM contribution.
- Global payload victim selection excluding the active measured qubits.
- `leak2depolar` transition expansion into explicit Pauli/no-op branches.
- Fail-loud behavior when hidden measurement changes detector dependencies rather
  than only flipping deterministic parities.
