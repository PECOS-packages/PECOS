"""Executable Guppy memories and audited measurement identities for both papers."""

import pytest
from pecos import sim, stabilizer
from pecos.qec import build_dem_from_guppy
from pecos.qec.color import ColorExtraction, color_code_implementation, color_code_spec, color_memory


@pytest.mark.parametrize("choice", list(ColorExtraction))
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_functional_guppy_memory_and_raw_events(choice, basis):
    family = "488" if choice in {ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ} else "666"
    impl = color_code_implementation(color_code_spec(family, 3), choice)
    plan = color_memory(impl, 3, basis, start_phase=impl.period - 1)
    source = plan.guppy_source()
    assert impl.code.citation in source
    assert "swap(" not in source
    program = plan.guppy_program()
    columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(123).run(16).to_dict()
    assert len(columns) == plan.num_measurements
    for shot in range(16):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert not any(observables)


@pytest.mark.parametrize(
    ("family", "choice"),
    [("488", ColorExtraction.BRICKWORK_CZ), ("666", ColorExtraction.SWAP_SUPERDENSE)],
)
def test_audited_guppy_dem_uses_same_event_spec(family, choice):
    impl = color_code_implementation(color_code_spec(family, 3), choice)
    plan = color_memory(impl, 2)
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
    columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(234).run(8).to_shot_map().to_dict()
    evaluated = build.evaluate_result_columns(columns)
    assert len(evaluated) == 8
    assert all(not any(detectors) and observable == 0 for detectors, observable in evaluated)
    assert "error(" in build.dem.to_string()
