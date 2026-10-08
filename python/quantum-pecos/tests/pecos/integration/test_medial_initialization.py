"""Medial surface code initialization must prepare the declared logical state."""

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
