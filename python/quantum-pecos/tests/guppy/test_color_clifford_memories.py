"""Actual Guppy execution and audited DEMs for logical Clifford experiments."""

import pytest
from pecos import sim, stabilizer
from pecos.qec import build_dem_from_guppy
from pecos.qec.color import (
    ColorCliffordExperiment,
    ColorExtraction,
    LogicalState,
    color_code_implementation,
    color_code_spec,
    color_state_memory,
)


@pytest.mark.parametrize("state", list(LogicalState))
@pytest.mark.parametrize("choice", [ColorExtraction.SUPERDENSE, ColorExtraction.SWAP_SUPERDENSE])
def test_six_state_guppy_memory(state, choice):
    impl = color_code_implementation(color_code_spec("666", 3), choice)
    plan = color_state_memory(impl, 3, state, start_phase=impl.period - 1)
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(7).run(16).to_dict()
    for shot in range(16):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert not any(observables)


@pytest.mark.parametrize("state", [LogicalState.PLUS_I, LogicalState.MINUS_I])
def test_distance_five_y_guppy_memory_with_nonzero_byproduct_flow(state):
    impl = color_code_implementation(color_code_spec("666", 5), ColorExtraction.SWAP_SUPERDENSE)
    assert impl.cycles[0].flow.logical_z != 0
    plan = color_state_memory(impl, 4, state, start_phase=2)
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(8).run(16).to_dict()
    for shot in range(16):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert not any(observables)


@pytest.mark.parametrize("state", list(LogicalState))
def test_signed_readout_reports_logical_value_not_always_zero(state):
    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    experiment = ColorCliffordExperiment(impl)
    experiment.prepare("memory", state)
    experiment.extract("memory", 2)
    experiment.observable("logical value", experiment.readout("memory", state.value[1]))
    plan = experiment.compile()
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(9).run(8).to_dict()
    for shot in range(8):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert observables == (state.value[0] == "-",)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_two_block_guppy_bell_state(basis):
    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    experiment = ColorCliffordExperiment(impl, ("alice", "bob"))
    experiment.prepare("alice", "+X", start_phase=2)
    experiment.prepare("bob", "+Z", start_phase=1)
    experiment.extract("alice")
    experiment.extract("bob", 2)
    experiment.gate("CX", "alice", "bob")
    experiment.extract("alice", 2)
    experiment.extract("bob")
    experiment.observable("Bell", experiment.readout("alice", basis) ^ experiment.readout("bob", basis))
    plan = experiment.compile()
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(10).run(16).to_dict()
    for shot in range(16):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert not any(observables)


def test_negative_y_memory_audited_dem():
    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    plan = color_state_memory(impl, 2, "-Y", start_phase=1)
    program = plan.guppy_program()
    detectors, observables = plan.dem_spec()
    build = build_dem_from_guppy(
        program,
        num_qubits=plan.num_qubits,
        detectors=detectors,
        observables=observables,
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(11).run(8).to_shot_map().to_dict()
    assert all(not any(detectors) and logical == 0 for detectors, logical in build.evaluate_result_columns(columns))
    assert "error(" in build.dem.to_string()
