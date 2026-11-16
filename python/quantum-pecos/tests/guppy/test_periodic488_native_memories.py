"""Actual native and Guppy runtime checks for signed 25-moment memories."""

import pytest
from pecos import sim, stabilizer
from pecos.qec import DetectorErrorModel, build_dem_from_guppy
from pecos.qec.color import periodic_488_code, periodic_488_implementation, periodic_488_memory
from pecos.qec.generic import SchedulePolicy, evaluate_tick_circuit


@pytest.mark.parametrize("distance", [4, 8])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_native_cz_guppy_memory(distance, basis, schedule):
    impl = periodic_488_implementation(periodic_488_code(distance), gate_set="native-cz")
    plan = periodic_488_memory(impl, 2, basis, schedule=schedule)
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(9).run(16).to_dict()
    for shot in range(16):
        assert plan.evaluate_named({tag: values[shot] for tag, values in columns.items()}) == (
            (False,) * len(plan.detectors),
            (False,) * 4,
        )


def test_native_and_guppy_signed_dem():

    impl = periodic_488_implementation(periodic_488_code(4), gate_set="native-cz")
    plan = periodic_488_memory(impl, 1, "Z")
    tick = plan.to_tick_circuit()
    detectors, observables = plan.dem_spec()
    program = plan.guppy_program()
    noise = {"p1": 0.001, "p2": 0.002, "p_prep": 0.003, "p_meas": 0.004}
    build = build_dem_from_guppy(
        program, num_qubits=plan.num_qubits, detectors=detectors, observables=observables, **noise,
    )
    native = DetectorErrorModel.from_circuit(tick, **noise)
    for dem in (native, build.dem):
        assert dem.num_observables == 4
        assert dem.num_detectors == len(plan.detectors)
    columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(33).run(16).to_shot_map().to_dict()
    assert build.evaluate_result_columns(columns) == [([0] * len(plan.detectors), 0)] * 16


def test_direct_native_tick_execution():
    runtime = pytest.importorskip("pecos_rslib_exp")
    impl = periodic_488_implementation(periodic_488_code(4), gate_set="native-cz")
    plan = periodic_488_memory(impl, 1, "Z")
    tick = plan.to_tick_circuit()
    noise = runtime.depolarizing().p1(0).p2(0).p_meas(0).p_prep(0)
    result = runtime.sim_neo(tick).quantum(runtime.stabilizer()).noise(noise).shots(16).seed(32).run()
    for shot in range(result.num_shots):
        assert evaluate_tick_circuit(tick, result[shot]) == ((False,) * len(plan.detectors), (False,) * 4)
