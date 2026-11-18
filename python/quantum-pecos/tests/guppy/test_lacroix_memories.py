"""Execute the released hardware memory construction through the Guppy backend."""

import pytest
from pecos import sim, stabilizer
from pecos.qec import build_dem_from_guppy
from pecos.qec.color import lacroix_memory
from pecos.qec.generic import SchedulePolicy


@pytest.mark.parametrize("patch", ["d3_west", "d3_east", "d3_north", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_lacroix_memory_guppy(patch, basis, schedule):
    plan = lacroix_memory(patch, 3, basis, schedule=schedule)
    program = plan.guppy_program()
    columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(24).run(16).to_dict()
    for shot in range(16):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert observables == (False,)


def test_lacroix_guppy_audited_dem():
    plan = lacroix_memory("d3_west", 3, "X")
    detectors, observables = plan.dem_spec()
    build = build_dem_from_guppy(
        plan.guppy_program(),
        num_qubits=plan.num_qubits,
        detectors=detectors,
        observables=observables,
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    assert build.dem.num_detectors == len(plan.detectors)
    assert "error(" in build.dem.to_string()
