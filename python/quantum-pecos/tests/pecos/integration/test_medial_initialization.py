"""Medial surface code initialization must prepare the declared logical state."""

from types import SimpleNamespace

import pecos as pc
import pytest


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("symbol", ["ideal init |0>", "ideal init |+>", "init |0>", "init |+>"])
def test_medial_initialization(distance, symbol) -> None:
    """Every init has zero subsequent syndrome and the expected logical sign."""
    qecc = pc.qeccs.SurfaceMedial4444(distance=distance)
    runner = pc.circuit_runners.Standard(seed=1)
    state = pc.simulators.SparseStabPy(qecc.num_qudits)
    init = pc.circuits.LogicalCircuit(suppress_warning=True)
    gate = qecc.gate(symbol, forced_outcome=0)
    init.append(gate)
    instruction = gate.final_instr()
    logical = instruction.final_logical_ops[0][instruction.logical_stabilizers[0]]
    runner.run(state, init)
    assert state.logical_sign(logical) == instruction.logical_signs[0] == 0
    extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
    extraction.append(qecc.gate("I", num_syn_extract=1))
    output, _ = runner.run(state, extraction)
    assert not output.simplified(last=True)
    assert state.logical_sign(logical) == 0


@pytest.mark.parametrize("symbol", ["instr_init_zero", "instr_init_plus"])
def test_color_initialization_metadata(symbol) -> None:
    """The analogous Color488 init copies preserve circuit metadata."""
    qecc = pc.qeccs.Color488(distance=3)
    instruction = qecc.instruction(symbol, error_free=True, forced_outcome=0)
    assert instruction.abstract_circuit.metadata["error_free"] is True
    assert instruction.abstract_circuit.metadata["forced_outcome"] == 0
    assert instruction.circuit.metadata["error_free"] is True


@pytest.mark.parametrize("distance", [3, 5])
def test_color_initialization_logical_operators(distance) -> None:
    """Initialization metadata names the operator stabilizing each declared state."""
    qecc = pc.qeccs.Color488(distance=distance)
    for basis, logical in [("|0>", "Z"), ("|+>", "X")]:
        runner = pc.circuit_runners.Standard(seed=1)
        state = pc.simulators.SparseStabPy(qecc.num_qudits)
        init = pc.circuits.LogicalCircuit(suppress_warning=True)
        gate = qecc.gate(f"ideal init {basis}")
        init.append(gate)
        runner.run(state, init)
        assert state.logical_sign(gate.final_instr().final_logical_ops[0][logical]) == 0


@pytest.mark.parametrize("distance", [3, 5])
def test_color_ideal_initialization(distance) -> None:
    """Ideal preparation projects onto positive checks; ordinary init stays unforced."""
    qecc = pc.qeccs.Color488(distance=distance)
    extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
    extraction.append(qecc.gate("I", num_syn_extract=1))
    for basis, logical in [("|0>", "Z"), ("|+>", "X")]:
        ordinary = qecc.gate(f"init {basis}")
        ordinary_measurements = [
            params for symbol, _, params in ordinary.final_instr().circuit.items() if symbol == "measure Z"
        ]
        assert ordinary_measurements
        assert all("forced_outcome" not in params for params in ordinary_measurements)
        init = pc.circuits.LogicalCircuit(suppress_warning=True)
        gate = qecc.gate(f"ideal init {basis}")
        init.append(gate)
        logical_op = gate.final_instr().final_logical_ops[0][logical]
        for seed in [1, 2, 3, 4]:
            runner = pc.circuit_runners.Standard(seed=seed)
            state = pc.simulators.SparseStabPy(qecc.num_qudits)
            output, _ = runner.run(state, init)
            assert not output, (distance, basis, seed)
            assert state.logical_sign(logical_op) == 0
            output, _ = runner.run(state, extraction)
            assert not output, (distance, basis, seed)
            assert state.logical_sign(logical_op) == 0


@pytest.mark.parametrize("distance", [3, 5])
def test_medial_initialization_tableau_shape(distance) -> None:
    """Both initializations append one complete logical stabilizer/destabilizer row."""
    qecc = pc.qeccs.SurfaceMedial4444(distance=distance)
    zero = qecc.instruction("instr_init_zero").stabs_destabs
    plus = qecc.instruction("instr_init_plus").stabs_destabs
    extraction = qecc.instruction("instr_syn_extract").stabs_destabs
    assert {len(rows) for rows in plus.values()} == {qecc.num_qudits}
    assert {len(rows) for rows in zero.values()} == {qecc.num_qudits}
    for name, rows in plus.items():
        assert rows[:-1] == zero[name][:-1] == extraction[name]
        dual = name[:-1] + ("z" if name.endswith("x") else "x")
        assert len(rows[-1]) == len(zero[dual][-1])
    assert plus["stabs_x"][-1] == zero["destabs_x"][-1] == set(qecc.sides["left"])
    assert plus["destabs_z"][-1] == zero["stabs_z"][-1] == set(qecc.sides["top"])


@pytest.mark.parametrize("mapped", [False, True])
def test_color_measurement_forcing_rule(mapped) -> None:
    """Gate parameters override metadata, with the surface compiler's truthiness rule."""
    qecc = pc.qeccs.Color488(distance=3)
    instruction = qecc.instruction("instr_syn_extract")
    mapping = {q: q + qecc.num_qudits for q in qecc.qudit_set} if mapped else None
    cases = [
        ({"forced_outcome": 0}, {"forced_outcome": True}, {"forced_outcome": 0}),
        ({"forced_outcome": False}, {}, {"forced_outcome": 0}),
        ({}, {"forced_outcome": 0}, {"forced_outcome": 0}),
        ({"forced_outcome": None}, {"forced_outcome": False}, {"forced_outcome": 0}),
        ({"forced_outcome": True}, {"forced_outcome": 0}, {}),
        ({"forced_outcome": 1}, {}, {}),
        ({"forced_outcome": None}, {}, {}),
        ({}, {}, {}),
    ]
    for gate_params, metadata, expected in cases:
        abstract = instruction.abstract_circuit.copy()
        abstract.metadata = metadata
        circuit = qecc.circuit_compiler.compile(SimpleNamespace(gate_params=gate_params), abstract, mapping=mapping)
        measurements = [(locations, params) for symbol, locations, params in circuit.items() if symbol == "measure Z"]
        assert measurements
        for locations, params in measurements:
            assert params == expected, (gate_params, metadata)
            ancillas = {mapping[q] for q in qecc.ancilla_qudit_set} if mapped else qecc.ancilla_qudit_set
            assert locations <= ancillas
