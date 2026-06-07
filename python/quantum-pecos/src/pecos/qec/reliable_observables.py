# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""PROTOTYPE: reliable logical-observable combinations from reset structure.

Port of lomatching's ``get_reliable_observables`` (Serra-Peralta et al.,
arXiv:2505.13599, "logical observable matching"). A logical observable is
*fragile* when its back-propagated observing region anticommutes with a reset
stabilizer -- decoding it directly is unreliable (a weight-2 space-time
stabilizer flips its decoded outcome without a logical fault). The paper's fix
is to decode only an **independent generating set of reliable observables** and
infer fragile ones as products.

This module computes that reliable set: build the anticommutation matrix
``A[reset, obs]`` (1 iff reset ``reset`` and observable ``obs`` anticommute) and
take its right null space over GF(2). Each null vector is a combination of raw
observables that commutes with *every* reset -- i.e. a reliable observable.

Status: prototype / proof-of-concept. Dependency-free (own GF(2) null space, no
``galois``). The eventual production path would compute the null space via
pecos-num's GF(2) and feed the reliable set into the logical-subgraph decoder
front-end. See ``pecos-docs/design/lomatching-paper-additional-learnings.md``.
"""

from __future__ import annotations

import numpy as np

try:
    import stim
except ImportError as exc:  # pragma: no cover - stim is an optional dep
    msg = "reliable_observables requires the `stim` package"
    raise ImportError(msg) from exc

# Stim reset instruction names (Pauli basis is the last character; default Z).
_RESET_INSTRS = frozenset(["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"])

# A Pauli region is {tick_index: stim.PauliString}.
PauliRegion = dict[int, "stim.PauliString"]


def reliable_observables(circuit: stim.Circuit) -> list[set[int]]:
    """Return a complete basis of reliable observable combinations.

    Args:
        circuit: A ``stim.Circuit`` whose observables are defined via
            ``OBSERVABLE_INCLUDE``. Qubits must be explicitly reset, and a reset
            must be the only operation on its qubit within its ``TICK`` (the
            lomatching precondition).

    Returns:
        One ``set[int]`` per basis element of the reliable space; each set is the
        raw-observable indices whose XOR is a reliable observable. An empty list
        means no nontrivial reliable combination exists.
    """
    if not isinstance(circuit, stim.Circuit):
        msg = f"`circuit` must be a stim.Circuit, got {type(circuit)}"
        raise TypeError(msg)

    resets = _reset_pauli_regions(circuit)
    num_obs = circuit.num_observables
    obs_regions = {o: _observing_region(circuit, o) for o in range(num_obs)}

    # A[reset, obs] = 1 iff the reset and the observable's region anticommute.
    a = np.zeros((len(resets), num_obs), dtype=np.uint8)
    for obs_id, region in obs_regions.items():
        for reset_id, reset_region in resets.items():
            if _anticommute(reset_region, region):
                a[reset_id, obs_id] = 1

    return [set(np.nonzero(vec)[0].tolist()) for vec in _gf2_right_null_space(a)]


def is_reliable(circuit: stim.Circuit, observable: set[int] | int) -> bool:
    """Whether a single observable (or XOR of observables) is reliable.

    A combination is reliable iff its region commutes with every reset.
    """
    obs = {observable} if isinstance(observable, int) else set(observable)
    resets = _reset_pauli_regions(circuit)
    # Combine the regions of the chosen observables by tick-wise Pauli product.
    combined: PauliRegion = {}
    for o in obs:
        for tick, ps in _observing_region(circuit, o).items():
            combined[tick] = combined[tick] * ps if tick in combined else ps
    return all(not _anticommute(r, combined) for r in resets.values())


# --------------------------------------------------------------------------- #
# Internals (faithful to lomatching's util.py)
# --------------------------------------------------------------------------- #


def _reset_pauli_regions(circuit: stim.Circuit) -> dict[int, PauliRegion]:
    """Per-reset single-tick Pauli region: the reset's Pauli on its qubit."""
    flat = circuit.flattened()
    n = flat.num_qubits
    resets: dict[int, PauliRegion] = {}
    reset_idx = 0
    tick = 0
    for instr in flat:
        if instr.name == "TICK":
            tick += 1
            continue
        if instr.name not in _RESET_INSTRS:
            continue
        pauli = "Z"
        if instr.name.endswith("X"):
            pauli = "X"
        elif instr.name.endswith("Y"):
            pauli = "Y"
        for target in instr.targets_copy():
            ps = stim.PauliString(n)
            ps[target.value] = pauli
            resets[reset_idx] = {tick: ps}
            reset_idx += 1
    return resets


def _observing_region(circuit: stim.Circuit, observable: int) -> PauliRegion:
    """Back-propagated observing region of one observable, via stim.

    Rewrites the circuit so only `observable` survives, renamed to L0, then uses
    `stim.Circuit.detecting_regions` to get its {tick: PauliString} region.
    """
    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue
        if instr.gate_args_copy()[0] != observable:
            continue
        new_circuit.append(
            stim.CircuitInstruction(
                name="OBSERVABLE_INCLUDE", gate_args=[0], targets=instr.targets_copy()
            )
        )
    target = stim.DemTarget("L0")
    regions = new_circuit.detecting_regions(
        targets=[target], ignore_anticommutation_errors=True
    )
    return regions.get(target, {})


def _anticommute(region_a: PauliRegion, region_b: PauliRegion) -> bool:
    """True iff two Pauli regions anticommute (odd number of anticommuting ticks)."""
    anti = 0
    for tick in set(region_a).intersection(region_b):
        if not region_a[tick].commutes(region_b[tick]):
            anti += 1
    return anti % 2 == 1


def _gf2_right_null_space(a: np.ndarray) -> list[np.ndarray]:
    """Basis of {x : a @ x == 0 (mod 2)} over GF(2), via row reduction.

    Returns a list of 0/1 vectors of length ``a.shape[1]``.
    """
    a = (np.asarray(a, dtype=np.uint8) % 2).copy()
    rows, cols = a.shape
    pivot_col_of_row: list[int] = []
    pivot_cols: set[int] = set()
    r = 0
    for c in range(cols):
        # find a pivot in column c at or below row r
        piv = next((i for i in range(r, rows) if a[i, c]), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivot_col_of_row.append(c)
        pivot_cols.add(c)
        r += 1
        if r == rows:
            break

    free_cols = [c for c in range(cols) if c not in pivot_cols]
    basis: list[np.ndarray] = []
    for f in free_cols:
        x = np.zeros(cols, dtype=np.uint8)
        x[f] = 1
        # back-substitute: pivot row i fixes its pivot col from the free col
        for i, pc in enumerate(pivot_col_of_row):
            if a[i, f]:
                x[pc] = 1
        basis.append(x)
    return basis
