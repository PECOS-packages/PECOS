# Stabilizer Tensor-Network Simulators

The experimental `pecos_rslib_exp` package exposes three related tools:

- `StabMps` executes Clifford and arbitrary-rotation circuits using a stabilizer tableau plus an MPS of coefficients.
- `Mast` defers non-Clifford work through preallocated magic-state ancillas.
- `StabMpsCompile` replays a circuit without an MPS and estimates which execution strategy is practical.

All public bitstrings use qubit-index order: `bits[q]` is the bit for qubit `q`. Python bitstring inputs must contain actual `bool` items; integers such as `0` and `1` are rejected. Dense state-vector indices are little-endian, so a row maps to `sum(int(bits[q]) << q for q in range(len(bits)))`. Gate rotation angles are in radians.

## `StabMps` quickstart

Use `sample_bitstrings`, plural, for shot workloads. It shares each distinct measurement-prefix projection between shots; `sample_bitstring` clones and collapses the entire simulator once per shot. The two methods do not share an RNG stream, so seeded outputs should not be compared shot for shot across methods.

```python
import math

import pecos_rslib_exp as exp

sim = exp.StabMps(2, seed=7, lazy_measure=True)
sim.run_gate("H", {0})
sim.run_gate("CX", {(0, 1)})
sim.run_gate("RZ", {1}, angle=math.pi / 4)

shots = sim.sample_bitstrings(32)
assert len(shots) == 32
assert all(len(bits) == 2 for bits in shots)
assert all(bits in ([False, False], [True, True]) for bits in shots)

# bits[q] is qubit q, so these are |q1 q0> = |00> and |11>.
p00 = sim.prob_bitstring([False, False])
p11 = sim.prob_bitstring([True, True])
assert math.isclose(p00, 0.5, abs_tol=1e-12)
assert math.isclose(p11, 0.5, abs_tol=1e-12)

# Python reads auto-flush lazy operations and merged RZ rotations.
accuracy = {
    "state_exact": sim.is_state_exact(),
    "pragmatic_drift_count": sim.pragmatic_drift_count,
    "truncation_error": sim.truncation_error,
    "bond_cap_hits": sim.bond_cap_hits,
}
assert accuracy == {
    "state_exact": True,
    "pragmatic_drift_count": 0,
    "truncation_error": 0.0,
    "bond_cap_hits": 0,
}
```

The accuracy fields answer different questions:

- `is_state_exact()` detects pending work, an unmaterialized Pauli frame, and stored-state drift from eager measurement. It does not include MPS truncation in its definition.
- `pragmatic_drift_count` should remain zero when exact amplitudes are required after random measurements. Construct with `lazy_measure=True` to avoid that eager-measurement drift.
- `truncation_error` estimates accumulated discarded singular-value weight.
- `bond_cap_hits` counts SVDs at which `max_bond_dim` was binding.

Python state reads automatically flush lazy operations and merged rotations. When Pauli-frame tracking is enabled, call `flush_pauli_frame_to_state()` before a read that must include the physical frame.

`StabMps` defaults `merge_rz` to true for throughput. `Mast` defaults it to false so each RZ immediately exposes its injection and ancilla-capacity cost. Numerical flag redetection is opt-in. In `StabMps` it self-disables while lazy deferred operations are pending, because the stored tensors then differ from the effective MPS-frame state.

## `Mast` quickstart

`max_non_clifford` reserves one fresh ancilla for each deferred non-Clifford RZ. Exceeding it raises `PanicException`, so use compile advice to size the simulator and inspect `remaining_injections` while building a circuit. Prefer MAST for T-like gates whose injection corrections are Clifford and when the extra ancillas fit; prefer `StabMps` for direct arbitrary rotations, limited ancillary memory, or amplitude, probability, and bulk-sampling reads.

```python
import pecos_rslib_exp as exp

mast = exp.Mast(2, max_non_clifford=2, seed=11)
mast.run_gate("H", {0})
mast.run_gate("CX", {(0, 1)})
mast.run_gate("T", {1})

assert mast.num_ancillas_used == 1
assert mast.remaining_injections == 1
assert mast.truncation_error == 0.0
assert mast.bond_cap_hits == 0

# Complete all deferred injections and apply their corrections.
mast.project_all()
assert len(mast.projection_records()) == 1

# MZ would call project_all() automatically if work were still deferred.
outcome = mast.run_1q_gate("MZ", 0)
assert outcome in (0, 1)
```

`flush()` is not the MAST completion operation: it materializes pending merged rotations, but leaves already deferred injections alone. Finish with `project_all()` or measure a data qubit with MZ.

MAST predetermines each injection-gadget outcome from its exact half-probability distribution. That choice is exact for the untruncated state. If the coefficient MPS has been truncated, the predetermined branch can differ from the truncated representation's own outcome distribution; it still represents the exact, untruncated gadget protocol.

## Analyze first with `StabMpsCompile`

Replay the same gates through `StabMpsCompile`, then call `recommend()` or `advise()`. Advice is heuristic. Deferred capacity counts every non-Clifford RZ, even an arbitrary-angle rotation whose eventual correction is also non-Clifford.
When advice selects a simulator without an injection implementation (`state_vector`, `ch_form`, or `stab_vec`), its injection field is always `direct`, while capacity and gate counts are still reported. A zero ancilla budget emits the same insufficient-budget warning as any other insufficient budget.

```python
import pecos_rslib_exp as exp

analysis = exp.StabMpsCompile(20)
analysis.run_gate("H", {0, 1})
analysis.run_gate("CX", {(0, 2), (1, 3)})
analysis.run_gate("T", {0, 1})

recommendation = analysis.recommend()
assert recommendation["simulator"] == "stab_mps"

required = analysis.nonclifford_rz_total
assert required == 2

sufficient = analysis.advise(ancilla_budget=required)
assert sufficient["injection"] == "deferred"
assert sufficient["deferred_feasible"] is True
assert sufficient["simulator"] == "mast"

insufficient = analysis.advise(ancilla_budget=required - 1)
assert insufficient["injection"] == "immediate"
assert insufficient["deferred_feasible"] is False
assert insufficient["warnings"]

unspecified = analysis.advise()
assert unspecified["injection"] == "deferred"
assert unspecified["deferred_feasible"] is None
assert unspecified["warnings"]
```

`recommend()` uses ordered thresholds: pure Clifford circuits select CH form; otherwise `n <= 14` selects a dense state vector; otherwise nullity `<= 6` selects `StabMps`; otherwise non-Clifford count `<= 40` selects `StabVec`; remaining circuits select `StabMps` with adaptive bond growth suggested. `bond_dim_bound` returns `2**nullity` and saturates at the platform maximum integer if that power overflows.

## Rust quickstart

Rust reads do not auto-flush. Call `flush()` before state reads when lazy measurement or merged RZ is enabled, and materialize a tracked Pauli frame separately when required.

```rust
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;

fn main() {
    let mut sim = StabMps::builder(2)
        .seed(7)
        .lazy_measure(true)
        .build();
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(1)]);

    let outcome = sim.mz(&[QubitId(0)])[0].outcome;
    sim.flush();
    let bits = [outcome, outcome];
    assert!((sim.prob_bitstring(&bits) - 1.0).abs() < 1e-12);
}
```
