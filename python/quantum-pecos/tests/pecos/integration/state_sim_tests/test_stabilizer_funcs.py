"""Regression tests for signed stabilizer removal and symplectic commutation."""

from copy import deepcopy
from random import Random

from pecos.analysis.stabilizer_funcs import is_not_stabilizer, remove_stab
from pecos.circuits import QuantumCircuit
from pecos.simulators.sparsestab import SparseStabPy
from pecos.simulators.sparsestab.logical_sign import find_logical_signs


def test_remove_stab_preserves_signed_group() -> None:
    """Random Clifford states retain signed group membership after removal."""
    rng = Random(20261023)
    for trial in range(300):
        num_qubits = rng.randrange(3, 6)
        state = SparseStabPy(num_qubits)
        for _ in range(50):
            gate = rng.choice(["H", "Sd", "SZ", "CX", "CZ"])
            if gate in {"CX", "CZ"}:
                state.run_gate(gate, {tuple(rng.sample(range(num_qubits), 2))})
            else:
                state.run_gate(gate, {rng.randrange(num_qubits)})

        source_ids = set(rng.sample(range(num_qubits), rng.randrange(2, num_qubits + 1)))
        partner_id = rng.choice(sorted(source_ids))
        logical_xs, logical_zs = set(), set()
        for gen in source_ids:
            logical_xs ^= state.stabs.row_x[gen]
            logical_zs ^= state.stabs.row_z[gen]
        delogical_xs = set(state.destabs.row_x[partner_id])
        delogical_zs = set(state.destabs.row_z[partner_id])
        original = deepcopy(state)

        # Also exercise inputs that alias tableau rows changed during removal.
        remove_stab(
            state,
            logical_xs,
            logical_zs,
            state.destabs.row_x[partner_id] if trial % 2 else delogical_xs,
            state.destabs.row_z[partner_id] if trial % 2 else delogical_zs,
        )

        removed = []
        for gen, (xs, zs) in enumerate(zip(state.stabs.row_x, state.stabs.row_z, strict=True)):
            if not xs and not zs:
                removed.append(gen)
                assert gen not in state.stabs.signs_minus, (trial, gen)
                assert gen not in state.stabs.signs_i, (trial, gen)
                continue
            assert (len(xs & delogical_zs) + len(zs & delogical_xs)) % 2 == 0, (trial, gen)
            logical = QuantumCircuit(
                [{symbol: qubits for symbol, qubits in [("X", xs - zs), ("Y", xs & zs), ("Z", zs - xs)] if qubits}],
            )
            # W = XZ = -iY: i**(2*minus + imag) W-support has explicit-Y
            # phase i**(2*minus + imag - |X & Z|), which must be real.
            phase = (2 * (gen in state.stabs.signs_minus) + (gen in state.stabs.signs_i) - len(xs & zs)) % 4
            assert phase % 2 == 0, (trial, gen)
            assert find_logical_signs(original, logical) == phase // 2, (trial, gen)

        assert len(removed) == 1, trial
        assert removed[0] in source_ids, trial
        assert not state.destabs.row_x[removed[0]], trial
        assert not state.destabs.row_z[removed[0]], trial
        for gens in state.gens:
            for qubit in range(num_qubits):
                assert gens.col_x[qubit] == {gen for gen, xs in enumerate(gens.row_x) if qubit in xs}, trial
                assert gens.col_z[qubit] == {gen for gen, zs in enumerate(gens.row_z) if qubit in zs}, trial


def test_is_not_stabilizer_classifies_y_and_logicals() -> None:
    """Y commutes with itself; missing generators leave commuting logicals."""
    state = SparseStabPy(1)
    state.run_gate("H", {0})
    state.run_gate("SZ", {0})
    assert is_not_stabilizer(state, {0}, {0}) == 0
    assert is_not_stabilizer(state, {0}, set()) == 1

    state = SparseStabPy(2)
    remove_stab(state, set(), {0}, {0}, set())
    assert is_not_stabilizer(state, set(), {0}) == 2
    assert is_not_stabilizer(state, {1}, set()) == 1
    assert is_not_stabilizer(state, set(), {1}) == 0


def test_logical_sign_with_two_ys() -> None:
    """The graph-state product (XZ)(ZX) is +YY in explicit-Y notation."""
    state = SparseStabPy(2)
    state.run_gate("H", {0, 1})
    state.run_gate("CZ", {(0, 1)})
    assert find_logical_signs(state, QuantumCircuit([{"Y": {0, 1}}])) == 0
