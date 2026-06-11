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

## Implemented Mode

`NoiseConfig` now exposes `MeasurementCrosstalkDemMode::ExactDeterministic`
for the local-payload subset. In this mode the circuit-aware DEM builder:

- Replays the ideal Clifford circuit up to each `MeasCrosstalkLocalPayload`.
- Synthesizes the hidden `MZ` on the payload victim.
- Requires that hidden result to be deterministic and state-independent.
- Emits an `X`-equivalent DEM source with `DirectSourceFamily::MeasurementCrosstalk`
  for `0->1` or `1->0` transitions.
- Emits no contribution for implicit `0->0` or `1->1` transitions.
- Fails loudly if global payloads, leakage transitions, missing circuit context,
  unsupported pre-payload gates, or nondeterministic hidden outcomes are present.

This mode is intentionally narrow: it is exact for the deterministic local cases
it accepts, and it rejects cases that still need a branch-level representation.

## Implementation Plan

1. Extend the exact crosstalk DEM path beyond deterministic local bit-flip
   transitions.
2. Add global-payload victim selection from the live prepared qubit set.
3. Reuse the exact branch replay machinery where possible: compute the ideal
   measurement parity expressions once, then evaluate branch effects by replaying
   hidden `MZ` plus the transition action at each payload victim.
4. Add `leak2depolar` transition expansion into explicit Pauli/no-op branches.
5. Emit raw hypergraph DEM contributions with crosstalk source metadata.
6. Extend source-level decomposition so graph-like decoder inputs preserve the
   same crosstalk source identity and fail loudly on irreducible branch effects.
7. Thread the new options through Python bindings and surface helper APIs.
8. Keep coverage diagnostics reporting unsupported crosstalk branches as omitted
   until the relevant DEM modes are explicitly enabled and tested.

## Minimal Tests

The first implementation tests should cover:

- Local payload with a single deterministic victim where `0->1` produces the same
  detector flip as an `X` at that spacetime point.
- Local payload with `0->0`/`1->1` only, producing no DEM contribution.
- Global payload victim selection excluding the active measured qubits.
- `leak2depolar` transition expansion into explicit Pauli/no-op branches.
- Fail-loud behavior when hidden measurement changes detector dependencies rather
  than only flipping deterministic parities.
