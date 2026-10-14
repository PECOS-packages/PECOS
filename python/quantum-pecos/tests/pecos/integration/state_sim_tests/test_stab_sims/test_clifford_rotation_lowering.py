# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Clifford-angle rotation lowering across simulator entry points."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pecos.circuits import QuantumCircuit
from pecos.engines.hybrid_engine_old import HybridEngine
from pecos.exceptions import NotSupportedGateError
from pecos.simulators import SparseStab, SparseStabPy, Stabilizer, StabVec, StateVec
from pecos.simulators.gate_syms import alt_symbols
from pecos_rslib import StateVec as RustStateVec
from pecos_rslib import angle64, lower_clifford_rotation

SIMULATORS = (SparseStab, Stabilizer, SparseStabPy, StabVec)
CLIFFORD_ONLY_SIMULATORS = (SparseStab, Stabilizer, SparseStabPy)
ONE_QUBIT_GATES = {
    "RZ": ("SZ", "Z", "SZdg"),
    "RX": ("SX", "X", "SXdg"),
    "RY": ("SY", "Y", "SYdg"),
}
TWO_QUBIT_GATES = {
    "RZZ": ("SZZ", "Z", "SZZdg"),
    "RXX": ("SXX", "X", "SXXdg"),
    "RYY": ("SYY", "Y", "SYYdg"),
}
ROTATION_GATES = [*ONE_QUBIT_GATES.items(), *TWO_QUBIT_GATES.items()]
ANGLE_CASES = [
    (math.pi / 2, "sqrt"),
    (-math.pi / 2, "dg"),
    (1.5 * math.pi, "dg"),
    (math.pi, "pauli"),
    (2 * math.pi, "identity"),
    (2.5 * math.pi, "sqrt"),
    (3.5 * math.pi, "dg"),
    (4.71238898038469, "dg"),
    (1.5 * math.pi + 1e-10, "dg"),
]
PREPARATIONS = ("bell", "plus_zero")

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_PAULI_Z = np.diag([1.0, -1.0]).astype(complex)


def _rotation_matrix(generator: np.ndarray, angle: float) -> np.ndarray:
    identity = np.eye(generator.shape[0], dtype=complex)
    return math.cos(angle / 2) * identity - 1j * math.sin(angle / 2) * generator


def _rxy1q_matrix(theta: float, phi: float) -> np.ndarray:
    return _rotation_matrix(math.cos(phi) * _PAULI_X + math.sin(phi) * _PAULI_Y, theta)


def _u_matrix(theta: float, phi: float, lambda_: float) -> np.ndarray:
    cos_half = math.cos(theta / 2)
    sin_half = math.sin(theta / 2)
    return np.array(
        [
            [cos_half, -np.exp(1j * lambda_) * sin_half],
            [np.exp(1j * phi) * sin_half, np.exp(1j * (phi + lambda_)) * cos_half],
        ],
        dtype=complex,
    )


def _rxxryyrzz_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    return (
        _rotation_matrix(np.kron(_PAULI_X, _PAULI_X), alpha)
        @ _rotation_matrix(np.kron(_PAULI_Y, _PAULI_Y), beta)
        @ _rotation_matrix(np.kron(_PAULI_Z, _PAULI_Z), gamma)
    )


def _prepare_state(state: Any, preparation: str) -> None:
    state.bindings["H"](state, 0)
    if preparation == "bell":
        state.bindings["CX"](state, (0, 1))


def _snapshot(state: Any) -> Any:
    if isinstance(state, StateVec):
        return tuple((float(amplitude.real), float(amplitude.imag)) for amplitude in state.vector)
    if isinstance(state, StabVec):
        return state.state_vector()
    if hasattr(state, "stab_tableau"):
        return state.stab_tableau(), state.destab_tableau()
    return (
        state.stabs.print_tableau(verbose=False),
        state.destabs.print_tableau(verbose=False),
    )


def _states_are_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, StabVec):
        actual = [complex(*amplitude) for amplitude in left.state_vector()]
        expected = [complex(*amplitude) for amplitude in right.state_vector()]
        pivot = next(index for index, amplitude in enumerate(expected) if abs(amplitude) > 1e-12)
        if abs(actual[pivot]) <= 1e-12:
            return False
        global_phase = actual[pivot] / expected[pivot]
        assert abs(abs(global_phase) - 1) < 1e-9
        return actual == pytest.approx(
            [global_phase * amplitude for amplitude in expected],
            abs=1e-9,
        )
    return _snapshot(left) == _snapshot(right)


def _assert_equivalent_state(rotated: Any, reference: Any) -> None:
    assert _states_are_equivalent(rotated, reference)


def _apply_reference(
    state: Any,
    location: int | tuple[int, int],
    named_gates: tuple[str, str, str],
    expected: str,
) -> None:
    sqrt_gate, pauli_gate, dagger_gate = named_gates
    if expected == "identity":
        return
    if expected == "sqrt":
        state.bindings[sqrt_gate](state, location)
        return
    if expected == "dg":
        state.bindings[dagger_gate](state, location)
        return
    if isinstance(location, tuple):
        for qubit in location:
            state.bindings[pauli_gate](state, qubit)
    else:
        state.bindings[pauli_gate](state, location)


@pytest.mark.parametrize("simulator", SIMULATORS)
@pytest.mark.parametrize(("symbol", "named_gates"), ROTATION_GATES)
@pytest.mark.parametrize(("angle", "expected"), ANGLE_CASES)
@pytest.mark.parametrize("preparation", PREPARATIONS)
def test_rotation_matches_named_clifford(
    simulator: type,
    symbol: str,
    named_gates: tuple[str, str, str],
    angle: float,
    expected: str,
    preparation: str,
) -> None:
    """Python float rotations match the named-gate reference."""
    location = 0 if symbol in ONE_QUBIT_GATES else (0, 1)
    rotated = simulator(2)
    reference = simulator(2)
    _prepare_state(rotated, preparation)
    _prepare_state(reference, preparation)

    rotated.bindings[symbol](rotated, location, angle=angle)
    _apply_reference(reference, location, named_gates, expected)

    _assert_equivalent_state(rotated, reference)


@pytest.mark.parametrize("simulator", SIMULATORS)
@pytest.mark.parametrize(("symbol", "named_gates"), ROTATION_GATES)
def test_sqrt_and_dagger_references_are_distinguishable(
    simulator: type,
    symbol: str,
    named_gates: tuple[str, str, str],
) -> None:
    """At least one preparation distinguishes each sqrt gate from its dagger."""
    location = 0 if symbol in ONE_QUBIT_GATES else (0, 1)
    sqrt_gate, _, dagger_gate = named_gates
    distinguishable = []
    for preparation in PREPARATIONS:
        sqrt_state = simulator(2)
        dagger_state = simulator(2)
        _prepare_state(sqrt_state, preparation)
        _prepare_state(dagger_state, preparation)
        sqrt_state.bindings[sqrt_gate](sqrt_state, location)
        dagger_state.bindings[dagger_gate](dagger_state, location)
        distinguishable.append(not _states_are_equivalent(sqrt_state, dagger_state))
    assert any(distinguishable)


@pytest.mark.parametrize("simulator", CLIFFORD_ONLY_SIMULATORS)
@pytest.mark.parametrize("symbol", [*ONE_QUBIT_GATES, *TWO_QUBIT_GATES])
def test_non_clifford_rotation_fails(simulator: type, symbol: str) -> None:
    """Every Clifford-only stabilizer binding rejects a non-Clifford float angle."""
    state = simulator(2)
    location = 0 if symbol in ONE_QUBIT_GATES else (0, 1)
    with pytest.raises(ValueError, match="is not a Clifford rotation"):
        state.bindings[symbol](state, location, angle=0.5)


@pytest.mark.parametrize("symbol", [*ONE_QUBIT_GATES, *TWO_QUBIT_GATES])
def test_stab_vec_accepts_non_clifford_one_angle_rotations(symbol: str) -> None:
    """StabVec preserves arbitrary-angle support on all one-angle rotations."""
    state = StabVec(2)
    location = 0 if symbol in ONE_QUBIT_GATES else (0, 1)
    state.bindings[symbol](state, location, angle=0.5)


@pytest.mark.parametrize("simulator", [SparseStab, Stabilizer, StabVec])
def test_pyo3_rotation_parameters_are_required_and_exact(simulator: type) -> None:
    """Pyo3 rotation entry points reject absent, malformed, or extra angles."""
    state = simulator(2)
    with pytest.raises(ValueError, match="requires params with 'angle'"):
        state.bindings["RZ"](state, 0)
    with pytest.raises(ValueError, match="Expected a valid angle parameter"):
        state.bindings["RZ"](state, 0, angle="invalid")
    with pytest.raises(
        ValueError,
        match="Gate RXY1Q expected 2 angle parameters, got 3",
    ):
        state.bindings["RXY1Q"](state, 0, angles=(0.0, 0.0, 0.0))


@pytest.mark.parametrize("simulator", [SparseStab, Stabilizer])
def test_multi_angle_pyo3_rotation_arms(simulator: type) -> None:
    """The additional Clifford-only pyo3 arms reach CliffordRotation."""
    cases = (
        ("RXY1Q", 0, {"angles": (-math.pi / 2, 0.0)}, (("SXdg", 0),)),
        ("U", 0, {"angles": (0.0, 0.0, -math.pi / 2)}, (("SZdg", 0),)),
        (
            "RXXRYYRZZ",
            (0, 1),
            {"angles": (-math.pi / 2, 0.0, 0.0)},
            (("SXXdg", (0, 1)),),
        ),
        (
            "RZZRYYRXX",
            (0, 1),
            {"angles": (-math.pi / 2, 0.0, 0.0)},
            (("SXXdg", (0, 1)),),
        ),
        (
            "R2XXYYZZ",
            (0, 1),
            {"angles": (-math.pi / 2, 0.0, 0.0)},
            (("SXXdg", (0, 1)),),
        ),
        (
            "RXXYYZZ",
            (0, 1),
            {"angles": (-math.pi / 2, 0.0, 0.0)},
            (("SXXdg", (0, 1)),),
        ),
    )
    for symbol, location, params, reference_gates in cases:
        rotated = simulator(2)
        reference = simulator(2)
        _prepare_state(rotated, "plus_zero")
        _prepare_state(reference, "plus_zero")
        rotated.bindings[symbol](rotated, location, **params)
        for named, named_location in reference_gates:
            reference.bindings[named](reference, named_location)
        assert _snapshot(rotated) == _snapshot(reference)


@pytest.mark.parametrize(
    ("symbol", "location", "angles", "matrix"),
    [
        pytest.param("RXY1Q", 0, (math.pi / 2, 0.0), _rxy1q_matrix, id="rxy1q-clifford"),
        pytest.param("RXY1Q", 0, (0.5, 0.25), _rxy1q_matrix, id="rxy1q-non-clifford"),
        pytest.param("U", 0, (math.pi / 2, 0.0, 0.0), _u_matrix, id="u-clifford"),
        pytest.param("U", 0, (0.5, 0.25, 0.125), _u_matrix, id="u-non-clifford"),
        pytest.param(
            "RXXRYYRZZ",
            (0, 1),
            (math.pi / 2, 0.0, 0.0),
            _rxxryyrzz_matrix,
            id="rxxryyrzz-clifford",
        ),
        pytest.param(
            "RXXRYYRZZ",
            (0, 1),
            (0.5, 0.25, 0.125),
            _rxxryyrzz_matrix,
            id="rxxryyrzz-non-clifford",
        ),
    ],
)
def test_stab_vec_multi_angle_rotations_preserve_full_phase(
    symbol: str,
    location: int | tuple[int, int],
    angles: tuple[float, ...],
    matrix: Any,
) -> None:
    """StabVec multi-angle bindings match matrix references including global phase."""
    num_qubits = 2 if isinstance(location, tuple) else 1
    state = StabVec(num_qubits)

    state.bindings[symbol](state, location, angles=angles)

    initial = np.zeros(1 << num_qubits, dtype=complex)
    initial[0] = 1.0
    expected = matrix(*angles) @ initial
    actual = np.array([complex(*amplitude) for amplitude in state.state_vector()])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("simulator", [SparseStab, Stabilizer])
@pytest.mark.parametrize(
    ("alias", "canonical", "location", "params", "preparation"),
    [
        (
            "RXXYYZZ",
            "RXXRYYRZZ",
            (0, 1),
            {"angles": (-math.pi / 2, 0.0, 0.0)},
            "plus_zero",
        ),
        (
            "R2XXYYZZ",
            "RXXRYYRZZ",
            (0, 1),
            {"angles": (-math.pi / 2, 0.0, 0.0)},
            "plus_zero",
        ),
    ],
)
def test_pyo3_rotation_alias_matches_canonical(
    simulator: type,
    alias: str,
    canonical: str,
    location: int | tuple[int, int],
    params: dict[str, object],
    preparation: str,
) -> None:
    """Rust-backed bindings accept the RXXRYYRZZ spellings the state-vector bindings already accept."""
    alias_state = simulator(2)
    canonical_state = simulator(2)
    _prepare_state(alias_state, preparation)
    _prepare_state(canonical_state, preparation)

    alias_state.bindings[alias](alias_state, location, **params)
    canonical_state.bindings[canonical](canonical_state, location, **params)

    assert _snapshot(alias_state) == _snapshot(canonical_state)


@pytest.mark.parametrize("simulator", [SparseStab, Stabilizer, SparseStabPy])
def test_legacy_engine_rzz_matches_szz_dagger(simulator: type) -> None:
    """The reported legacy-engine RZZ scenario lowers end to end."""
    rotation = QuantumCircuit()
    rotation.append({"H": {0}})
    rotation.append({"RZZ": {(0, 1)}}, angle=1.5 * math.pi)
    named = QuantumCircuit()
    named.append({"H": {0}})
    named.append({"SZZdg": {(0, 1)}})
    rotated_state = simulator(2)
    named_state = simulator(2)

    HybridEngine().run(rotated_state, rotation, shot_id=0)
    HybridEngine().run(named_state, named, shot_id=0)

    assert _snapshot(rotated_state) == _snapshot(named_state)


def test_lowering_uses_per_instance_named_gate_override() -> None:
    """Lowered gates resolve through the simulator instance bindings."""
    state = SparseStabPy(1)
    calls = []
    state.bindings["SZ"] = lambda _state, location, **_params: calls.append(location)

    state.bindings["RZ"](state, 0, angle=math.pi / 2)

    assert calls == [0]


def test_state_vec_rz_requires_angle_and_matches_sz_at_quarter_turn() -> None:
    """State-vector bindings reject absent rotation parameters and apply valid ones."""
    state = RustStateVec(1)
    with pytest.raises(ValueError, match="RZ requires params with 'angle'"):
        state.bindings["RZ"](state, 0)

    rotated = RustStateVec(1)
    named = RustStateVec(1)
    rotated.bindings["H"](rotated, 0)
    named.bindings["H"](named, 0)
    rotated.bindings["RZ"](rotated, 0, angle=math.pi / 2)
    named.bindings["SZ"](named, 0)

    rotated_vector = list(rotated.vector)
    named_vector = list(named.vector)
    global_phase = rotated_vector[0] / named_vector[0]
    assert all(
        abs(actual - global_phase * expected) < 1e-12
        for actual, expected in zip(rotated_vector, named_vector, strict=True)
    )


def test_lowering_reports_named_gate_missing_from_instance() -> None:
    """A missing per-instance named gate uses the simulator's existing error."""
    state = SparseStabPy(2)
    del state.bindings["SZZ"]
    with pytest.raises(NotSupportedGateError, match='gate "SZZ" is not available'):
        state.bindings["RZZ"](state, (0, 1), angle=math.pi / 2)


def test_lower_clifford_rotation_examples() -> None:
    """The Python helper exposes the shared Rust lowering table."""
    assert lower_clifford_rotation("RZZ", [1.5 * math.pi]) == [("SZZdg", (0, 1))]
    assert lower_clifford_rotation("RZZ", [math.pi]) == [("Z", (0,)), ("Z", (1,))]
    assert lower_clifford_rotation("RZ", [0.0]) == [("I", (0,))]
    assert lower_clifford_rotation("RXY1Q", [math.pi / 2, 0.0]) == [("SX", (0,))]
    assert lower_clifford_rotation("R1XY", [math.pi / 2, 0.0]) == lower_clifford_rotation(
        "RXY1Q",
        [math.pi / 2, 0.0],
    )
    assert lower_clifford_rotation("RZ", [angle64.from_radians(math.pi / 2)]) == [
        ("SZ", (0,)),
    ]


def test_xy_plane_rotation_legacy_symbol_canonicalization() -> None:
    assert alt_symbols["R1XY"] == "RXY1Q"


@pytest.mark.parametrize("simulator", [SparseStab, Stabilizer, StabVec, StateVec])
def test_xy_plane_rotation_binding_alias_matches_canonical(simulator: type) -> None:
    legacy_state = simulator(1)
    canonical_state = simulator(1)

    legacy_state.bindings["R1XY"](legacy_state, 0, angles=(math.pi / 2, 0.0))
    canonical_state.bindings["RXY1Q"](canonical_state, 0, angles=(math.pi / 2, 0.0))

    assert _snapshot(legacy_state) == _snapshot(canonical_state)


@pytest.mark.parametrize("simulator", [SparseStab, StateVec])
def test_xy_plane_rotation_string_aliases_run_identically_through_legacy_engine(simulator: type) -> None:
    snapshots = []
    for symbol in ("RXY1Q", "R1XY", "U1q"):
        circuit = QuantumCircuit()
        circuit.append(symbol, {0}, angles=(math.pi / 2, 0.0))
        state = simulator(1)
        HybridEngine().run(state, circuit, shot_id=0)
        snapshots.append(_snapshot(state))

    assert snapshots[0] == snapshots[1] == snapshots[2]


@pytest.mark.parametrize("symbol", ["U", "RXXRYYRZZ"])
def test_lower_clifford_rotation_rejects_unsupported_symbols(symbol: str) -> None:
    """Decomposition-only symbols are outside the table helper."""
    with pytest.raises(ValueError, match=rf"^{symbol} is unsupported"):
        lower_clifford_rotation(symbol, [0.0])


def test_lower_clifford_rotation_rejects_non_clifford_angle() -> None:
    """The helper preserves the CliffordRotation error shape."""
    with pytest.raises(ValueError, match=r"RZZ.*is not a Clifford rotation"):
        lower_clifford_rotation("RZZ", [0.5])


def test_lower_clifford_rotation_rejects_wrong_angle_count() -> None:
    """The table helper requires the exact arity for its rotation symbol."""
    with pytest.raises(
        ValueError,
        match="Gate RXY1Q expected 2 angle parameters, got 1",
    ):
        lower_clifford_rotation("RXY1Q", [0.0])
