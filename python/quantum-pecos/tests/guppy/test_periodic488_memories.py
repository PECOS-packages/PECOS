"""Execute four-logical-qubit periodic memories through both lower-level paths."""

import pytest
from pecos import sim, stabilizer
from pecos.qec import DetectorErrorModel, build_dem_from_guppy
from pecos.qec.color import periodic_488_code, periodic_488_implementation, periodic_488_memory
from pecos.qec.generic import SchedulePolicy


@pytest.mark.parametrize("distance", [4, 8])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_periodic_guppy_memory(distance, basis, schedule):
    implementation = periodic_488_implementation(periodic_488_code(distance))
    plan = periodic_488_memory(implementation, 2, basis, schedule=schedule)
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(37).run(16).to_dict()
    for shot in range(16):
        dets, obs = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(dets)
        assert obs == (False, False, False, False)


def test_four_observables_survive_both_dem_paths():
    impl = periodic_488_implementation(periodic_488_code(4))
    plan = periodic_488_memory(impl, 1, "X")
    detectors, observables = plan.dem_spec()
    noise = {"p1": 0.001, "p2": 0.002, "p_meas": 0.003, "p_prep": 0.004}
    guppy = build_dem_from_guppy(
        plan.guppy_program(),
        num_qubits=plan.num_qubits,
        detectors=detectors,
        observables=observables,
        **noise,
    ).dem
    native = DetectorErrorModel.from_circuit(plan.to_tick_circuit(), **noise)
    for dem in (guppy, native):
        assert dem.num_detectors == len(plan.detectors)
        assert dem.num_observables == 4
        for index in range(4):
            assert f"L{index}" in dem.to_string()
