# Rust-only frame prototype

Historical executable probes were removed after separate production review.
Their source remains at commit `d4eaaa7b`; current regression coverage is in
`tests/runtime_frame_production.rs`. Descriptions of prototype tests below are
historical, not claims about current test targets.

Current milestone: [scoped production Rust execution](runtime-production-milestone.md).
This file describes the earlier test-only prototype; it remains historical evidence.

Status: implemented **only inside an integration test target**,
`crates/pecos-engines/tests/runtime_frame_slice.rs`. Nothing is exported by the
library or connected to Selene, `sim()`, Python, or a byte-message encoder.
The prototype itself is not production transport. This is executable evidence for the
[proposed contracts](runtime-effect-contracts.md), not their full implementation.

The two ordinary general-noise correctness fixes and their three regression
tests are isolated in [#828](https://github.com/PECOS-packages/PECOS/pull/828).
#827 is stacked on that branch so its incremental diff contains only design and
test-only work. Neither PR should be merged automatically.

## One logical execution across yields

The prototype uses GeneralNoiseModel's existing two phases. It samples the
**entire** original frame's start-phase noise before executing any simulator
commands. Calling `apply_noise_on_start` for each intact gate records the
expansion boundary without changing that phase's sampling order. It never calls
`ControlEngine::start`/`Complete` for successive segments. All original raw
measurement results are accumulated in source order and passed to the existing
`continue_processing` lifecycle only after the frame's simulator work finishes.
Any generated continuation work stays inside that same lifecycle.

`Yield(token)` pauses between complete noisy gate expansions. `resume(token)`
consumes a run/generation/position token exactly once. `Complete(outcomes)` is a
separate terminal step and occurs once. Metadata yields do not sample noise,
split idles, return partial results, or complete the input. Pauli events occur
after the preceding expansion and before the following expansion, bypassing
ordinary gate-noise sampling.

This deliberately retains legacy sampling/readout order, including its existing
leakage behavior across a measurement followed by preparation. It does not
silently replace that behavior with a new execution-order sampling profile.

## Enforced scope

Only a private factory can construct the prepared runner. It builds an owned
GeneralNoiseModel from configuration; it cannot accept arbitrary NoiseModel
implementations or a caller-provided support boolean. The prototype rejects a
request for another model. Model, simulator and execution state are private;
the tests directly initialize a leaked state solely for the metadata oracle.

For physical events, factory admission requires the existing builder's checked
`simple_probabilities()` profile. This excludes leakage, emission, crosstalk,
idle-noise channels, custom samplers and other unsupported mechanisms. The only
physical operations are independently invented, unconditional X and Z events.
There is no downstream callback. Metadata tests also exercise configurations
outside this physical subset, including leakage, nonlinear idle and crosstalk.
No physical handling is claimed for those configurations.

The input is a bounded, typed **in-process** envelope with version and record
validation; it is not an implementation of the proposed version-2 wire format.
The whole envelope is checked before a runner/model/simulator is constructed:
unknown records/tags, unsupported versions/models/physical profiles, invalid
targets/gates and over-limit record counts fail even after a valid gate prefix.
Noise configuration is validated before construction, including combined scales.
Gates receive structural validation; nonempty measurement IDs are rejected because
the prototype transport preserves positional outcomes only.
The maximum is 128 records and two qubit slots. Native batch timing, event bytes,
schema decoding and allocation-failure guarantees are outside this prototype.
There is no representation for a physical event inside a simultaneous gate
batch or halfway through an idle. Full production retention/streaming remains
unimplemented.

Runners cannot be cloned. Workers spawn independent state from the configuration
factory. Errors poison the runner and clear queued work/raw results. Reset leaves
it poisoned until both model and simulator resets succeed, then seeds a fresh
simulator and model stream using the existing `noise_model` and `quantum_engine`
seed domains. Old tokens remain invalid after reset. There is one
frame per prototype shot; carrying state across multiple runtime inputs in a
shot and allocating application-wide shot identities are not implemented.

## Nine executable checks

- Across 64 seeds, metadata insertion matches an ordinary whole-input
  GeneralNoiseModel execution: simulator gate trace, outcomes (including leakage
  readout), nonlinear idle behavior and eight subsequent model RNG words.
- Metadata around physical yields preserves the same observations and repeated
  seeded reset behavior across another 64-seed sweep.
- `H; phase; H; M` and `M; flip; M` verify placement. A p1=1 test checks that an
  event X receives no second gate fault or extra model RNG draw; ordinary
  measurement draws are compared against the matching event-free reference.
- Genuine general-noise crosstalk continuation emits its X exactly once, after
  all metadata yields, with the same trace/RNG as the unannotated frame.
- Unsupported envelopes/models/profiles fail before execution is constructible.
- Invalid combined noise scales reject before the builder can panic.
- Measurement IDs reject before structural panics or loss of result identity.
- Foreign/reused tokens fail; reset and four concurrent factory-created workers
  remain isolated.
- A simulator error with a raw outcome already buffered abandons that state;
  failed simulator reset keeps the runner poisoned, and successful reset recovers.

The original completion-fault and segmentation counterexamples remain regression
requirements. Unsupported models cannot enter this runner, and admitted metadata
does not create model completions. The proposed wire parser's unknown-mandatory-
record behavior still needs separate byte-level tests before production use.

## Review against #591 and remaining architectural decision

Reviewed #591 at `95eef838b6be7e6099e0226969456d581132ebda` and the existing
`EngineSystem::process_as_system` contract. The prototype preserves one original
completion lifecycle, explicit non-recursive generated effects, deterministic
ordering and strict capability rejection. It is a narrow legacy-phase experiment,
not the RFC's full event-driven engine or general-noise compatibility facade.

**Do not expand physical admission to leakage/crosstalk or outcome-dependent
events yet.** Legacy GeneralNoiseModel samples future gates before postprocessing
earlier measurements. This prototype exposes no completed outcomes at a yield;
its physical boundary is after the raw simulator measurement, while processed
readout remains at frame completion. Giving a physical handler post-noise
outcomes at that earlier boundary requires a decision about sampling/state
semantics, not another capability flag. Recommendation: retain this legacy path
as the parity oracle and review a separately named resumable sampling profile
under #591 before that expansion. Native batch preservation, full wire admission,
runtime/worker integration and bounded streaming also remain prerequisites.

No Python path or private event interpretation is included. These synthetic
tests do not establish PECOS/Selene parity or device-model equivalence.
