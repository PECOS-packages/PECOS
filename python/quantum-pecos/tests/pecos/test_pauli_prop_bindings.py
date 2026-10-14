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

"""End-to-end tests for the Rust-backed PauliProp gate binding surface."""

from __future__ import annotations

import itertools
import math
import pickle
from dataclasses import dataclass, field

import pytest
from pecos.circuits import QuantumCircuit
from pecos.engines.hybrid_engine_old import HybridEngine
from pecos.simulators import PauliProp
from pecos.simulators.pauliprop import bindings as legacy_bindings
from pecos.simulators.pauliprop import gates_meas, gates_one_qubit, gates_two_qubit
from pecos_rslib import CliffordRep, StabVec
from pecos_rslib import PauliProp as RustPauliProp
from pecos_rslib.simulators import SparseStab, Stabilizer


@dataclass
class LegacyOracle:
    """Minimal mutable state accepted by the legacy Pauli propagation functions."""

    track_sign: bool
    faults: dict[str, set[int]] = field(
        default_factory=lambda: {"X": set(), "Y": set(), "Z": set()},
    )
    sign: int = 0

    def flip_sign(self) -> None:
        """Toggle the tracked conjugation sign."""
        if self.track_sign:
            self.sign ^= 1


def _faults_for_labels(labels: tuple[str, ...]) -> dict[str, set[int]]:
    faults = {"X": set(), "Y": set(), "Z": set()}
    for qubit, label in enumerate(labels):
        if label != "I":
            faults[label].add(qubit)
    return faults


def _seeded_states(labels: tuple[str, ...], *, track_sign: bool) -> tuple[PauliProp, LegacyOracle]:
    faults = _faults_for_labels(labels)
    state = PauliProp(num_qubits=3, track_sign=track_sign)
    state.faults = faults
    oracle = LegacyOracle(track_sign=track_sign, faults={key: set(value) for key, value in faults.items()})
    return state, oracle


def _fault_circuit(symbol: str, qubit: int) -> QuantumCircuit:
    circuit = QuantumCircuit()
    circuit.append(symbol, {qubit})
    return circuit


def _sparse_stab_tableau_image(image: object, num_qubits: int) -> str:
    dense = image.to_dense_str(num_qubits)
    if dense.startswith(("+i", "-i")):
        prefix, paulis = dense[:2], dense[2:]
    else:
        prefix, paulis = dense[0], dense[1:]
    phase = {"+": 0, "+i": 1, "-": 2, "-i": 3}[prefix]
    phase = (phase + paulis.count("Y")) % 4
    return f"{['+', '+i', '-', '-i'][phase]}{paulis}"


def _stab_vec_amplitudes(state: StabVec) -> list[complex]:
    return [complex(real, imaginary) for real, imaginary in state.state_vector()]


def _state_vectors_are_equivalent(left: list[complex], right: list[complex]) -> bool:
    pivot = next(index for index, amplitude in enumerate(right) if abs(amplitude) > 1e-12)
    if abs(left[pivot]) <= 1e-12:
        return False
    phase = left[pivot] / right[pivot]
    return left == pytest.approx([phase * amplitude for amplitude in right], abs=1e-9)


def _pauli_expectation(amplitudes: list[complex], image: object, num_qubits: int) -> complex:
    dense = image.to_dense_str(num_qubits)
    if dense.startswith(("+i", "-i")):
        prefix, paulis = dense[:2], dense[2:]
    else:
        prefix, paulis = dense[0], dense[1:]
    phase = {"+": 1, "+i": 1j, "-": -1, "-i": -1j}[prefix]
    expectation = 0j
    for source, amplitude in enumerate(amplitudes):
        target = source
        coefficient = phase
        for qubit, pauli in enumerate(paulis):
            bit = (source >> qubit) & 1
            if pauli == "X":
                target ^= 1 << qubit
            elif pauli == "Y":
                coefficient *= -1j if bit else 1j
                target ^= 1 << qubit
            elif pauli == "Z" and bit:
                coefficient *= -1
        expectation += amplitudes[target].conjugate() * coefficient * amplitude
    return expectation


SUPPORTED_LEGACY_SYMBOLS = tuple(
    symbol for symbol in legacy_bindings.gate_dict if symbol not in {"force output", "check", "measure"}
)


@pytest.mark.parametrize("track_sign", [True, False])
@pytest.mark.parametrize("symbol", SUPPORTED_LEGACY_SYMBOLS)
def test_every_legacy_symbol_dispatches_like_the_fixed_reference(symbol: str, track_sign: bool) -> None:
    """Every formerly exposed legacy gate has matching frame, sign, and measurement behavior."""
    legacy_gate = legacy_bindings.gate_dict[symbol]
    is_two_qubit = legacy_gate.__module__.endswith("gates_two_qubit")
    configurations = itertools.product("IXYZ", repeat=2 if is_two_qubit else 1)
    location: int | tuple[int, int] = (0, 1) if is_two_qubit else 0

    for labels in configurations:
        state, oracle = _seeded_states(labels, track_sign=track_sign)
        rust_result = state.bindings[symbol](state, location)
        legacy_result = legacy_gate(oracle, location)

        assert state.faults == oracle.faults, (symbol, labels, track_sign)
        assert state.sign == oracle.sign, (symbol, labels, track_sign)
        if legacy_gate.__module__.endswith("gates_meas"):
            assert rust_result == legacy_result, (symbol, labels, track_sign)


def test_zero_measurement_output_is_consistent_across_dispatch_paths() -> None:
    """Batch and per-location measurement paths both return explicit zeros."""
    state = RustPauliProp(1)
    assert state.bindings["measure Z"](state, 0) == 0
    assert state.bindings["measure X"](state, 0) == 0


@pytest.mark.parametrize("symbol", ["force output", "check", "measure"])
def test_legacy_symbols_without_rust_meaning_are_rejected(symbol: str) -> None:
    """Legacy operations with no frame-propagation meaning never silently succeed."""
    state = PauliProp(num_qubits=2)
    assert symbol not in state.bindings
    with pytest.raises(KeyError, match=symbol):
        state.bindings[symbol]
    with pytest.raises(ValueError, match="Unsupported gate"):
        RustPauliProp(2).run_gate(symbol, {0})
    with pytest.raises(ValueError, match="Unsupported gate"):
        RustPauliProp(2).run_gate(symbol, {0}, simulate_gate=False)


def test_clifford_rotations_lower_through_bindings() -> None:
    """Parameterized bindings use the Rust Clifford-rotation lowering path."""
    cases = [
        ("RZ", {"angle": -math.pi / 2}, "SZdg", 0),
        ("RX", {"angle": math.pi}, "X", 0),
        ("RZZ", {"angle": math.pi / 2}, "SZZ", (0, 1)),
    ]
    for rotation, params, named_gate, location in cases:
        rotated, _ = _seeded_states(("Y", "X"), track_sign=True)
        named, _ = _seeded_states(("Y", "X"), track_sign=True)
        rotated.bindings[rotation](rotated, location, **params)
        named.bindings[named_gate](named, location)
        assert rotated.faults == named.faults
        assert rotated.sign == named.sign

    state, _ = _seeded_states(("Y",), track_sign=True)
    with pytest.raises(ValueError, match="is not a Clifford rotation"):
        state.bindings["RZ"](state, 0, angle=0.5)


def test_forced_outcome_parameters_do_not_change_frame_semantics() -> None:
    """Pauli-frame preparations clear and measurements report the frame regardless of forced outcomes."""
    state, _ = _seeded_states(("Y", "Z"), track_sign=True)
    state.flip_sign()
    state.bindings["init |0>"](state, 0, forced_outcome=1)
    assert state.faults == {"X": set(), "Y": set(), "Z": {1}}
    assert state.sign == 1

    state.faults = {"X": {0}, "Y": set(), "Z": set()}
    result = state.bindings["measure Z"](state, 0, forced_outcome=0)
    assert result == 1
    assert state.faults == {"X": {0}, "Y": set(), "Z": set()}


def test_hybrid_engine_user_scenario_matches_legacy_reference() -> None:
    """The legacy engine drives H, CX, SZdg, and measure Z through Rust bindings."""
    state, oracle = _seeded_states(("X", "I"), track_sign=True)
    circuit = QuantumCircuit(cvar_spec={"m": 1}, num_qubits=2)
    circuit.append("H", {0})
    circuit.append("CX", {(0, 1)})
    circuit.append("SZdg", {1})
    circuit.append("measure Z", {1}, var_output={1: ("m", 0)})

    output, _ = HybridEngine().run(state, circuit, shot_id=0)

    gates_one_qubit.H(oracle, 0)
    gates_two_qubit.CX(oracle, (0, 1))
    gates_one_qubit.SZdg(oracle, 1)
    expected_outcome = gates_meas.meas_z(oracle, 1)
    assert state.faults == oracle.faults
    assert state.sign == oracle.sign
    assert output["m"][0] == expected_outcome


def test_faults_property_is_a_snapshot_and_assignment_resets_phase() -> None:
    """Fault replacement is the mutation boundary and resets sign and imaginary phase."""
    state = PauliProp(num_qubits=2, track_sign=True)
    state.add_faults(_fault_circuit("X", 0))
    state.add_faults(_fault_circuit("Z", 0))
    assert (state.sign, state.img) == (1, 1)

    snapshot = state.faults
    snapshot["X"].add(1)
    assert state.faults == {"X": set(), "Y": {0}, "Z": set()}

    state.faults = {"Z": {1}}
    assert state.faults == {"X": set(), "Y": set(), "Z": {1}}
    assert (state.sign, state.img) == (0, 0)


def test_wrapper_pickle_round_trip_restores_bindings() -> None:
    """Wrapper state and its dynamic binding surface survive pickling."""
    state, _ = _seeded_states(("Y", "Z"), track_sign=True)
    state.flip_sign()
    restored = pickle.loads(pickle.dumps(state))

    assert restored.faults == state.faults
    assert (restored.sign, restored.img) == (state.sign, state.img)
    restored.bindings["H"](restored, 0)
    state.bindings["H"](state, 0)
    assert restored.faults == state.faults
    assert restored.sign == state.sign


@pytest.mark.parametrize(
    ("symbol", "rep"),
    [
        ("ISWAP", CliffordRep.iswap(0, 1)),
        ("ISWAPdg", CliffordRep.iswap(0, 1).inverse()),
        ("Gdg", CliffordRep.g(0, 1)),
    ],
)
@pytest.mark.parametrize("simulator", [SparseStab, Stabilizer])
def test_shared_dispatch_arms_match_clifford_rep_on_tableau_simulators(
    symbol: str,
    rep: CliffordRep,
    simulator: type,
) -> None:
    """Batch and parameterized paths produce the exact CliffordRep tableau images."""
    z_state = simulator(2)
    z_parameterized = simulator(2)
    z_state.bindings[symbol](z_state, (0, 1))
    z_parameterized.bindings[symbol](z_parameterized, (0, 1), simulate_gate=True)
    expected_z_images = "".join(f"{_sparse_stab_tableau_image(rep.z_image(q), 2)}\n" for q in range(2))
    assert z_state.stab_tableau() == expected_z_images
    assert z_parameterized.stab_tableau() == expected_z_images

    x_state = simulator(2)
    x_parameterized = simulator(2)
    x_state.bindings["H"](x_state, 0)
    x_state.bindings["H"](x_state, 1)
    x_parameterized.bindings["H"](x_parameterized, 0)
    x_parameterized.bindings["H"](x_parameterized, 1)
    x_state.bindings[symbol](x_state, (0, 1))
    x_parameterized.bindings[symbol](x_parameterized, (0, 1), simulate_gate=True)
    expected_x_images = "".join(f"{_sparse_stab_tableau_image(rep.x_image(q), 2)}\n" for q in range(2))
    assert x_state.stab_tableau() == expected_x_images
    assert x_parameterized.stab_tableau() == expected_x_images


@pytest.mark.parametrize(
    ("symbol", "rep"),
    [
        ("ISWAP", CliffordRep.iswap(0, 1)),
        ("ISWAPdg", CliffordRep.iswap(0, 1).inverse()),
        ("Gdg", CliffordRep.g(0, 1)),
    ],
)
def test_shared_dispatch_arms_match_clifford_rep_on_stab_vec(symbol: str, rep: CliffordRep) -> None:
    """StabVec batch and parameterized paths agree and satisfy transformed stabilizers."""
    batch = StabVec(2)
    parameterized = StabVec(2)
    batch.bindings["H"](batch, 0)
    parameterized.bindings["H"](parameterized, 0)
    batch.bindings[symbol](batch, (0, 1))
    parameterized.bindings[symbol](parameterized, (0, 1), simulate_gate=True)

    batch_amplitudes = _stab_vec_amplitudes(batch)
    parameterized_amplitudes = _stab_vec_amplitudes(parameterized)
    assert _state_vectors_are_equivalent(batch_amplitudes, parameterized_amplitudes)
    assert _pauli_expectation(batch_amplitudes, rep.x_image(0), 2) == pytest.approx(1, abs=1e-9)
    assert _pauli_expectation(batch_amplitudes, rep.z_image(1), 2) == pytest.approx(1, abs=1e-9)
