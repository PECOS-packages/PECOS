"""Decoder coordinates survive authoring and every lower-level representation."""

import json
from dataclasses import replace

import pytest
import stim
from pecos.qec import DetectorErrorModel, build_dem_from_guppy
from pecos.qec.generic import Gadget, GadgetProgram, Gate, MeasurementParity, SchedulePolicy
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim


def _program(coords):
    program = GadgetProgram(("data",))
    call = program.append(
        Gadget(
            "signed measurement",
            ("data",),
            (),
            (),
            ((Gate("R", ("data",)),), (Gate("X", ("data",)),), (Gate("M", ("data",), "result"),)),
        ),
        label="readout",
    )
    program.detector("check", call["result"] ^ 1, coords=coords)
    program.observable("logical", call["result"] ^ 1)
    return program


@pytest.mark.parametrize("coords", [None, (), (1.5,), (1, 2), (1, 2, 3), (1, 2, 3, -1), (1, 2, 3, 4, 5)])
@pytest.mark.parametrize("policy", list(SchedulePolicy))
def test_coordinates_across_schedules_and_native_backends(coords, policy):
    program = _program(coords)
    plan = program.compile(policy)
    assert program.detectors[0].coords == coords
    assert plan.detectors[0].coords == coords
    assert (plan.detectors[0] ^ 1).coords == coords
    assert plan.dem_spec()[0][0].coords == coords
    expected = {0: list(coords or ())}
    assert plan.to_stim().get_detector_coordinates() == expected
    tick = plan.to_tick_circuit()
    assert tick.circuit_definitions()["detectors"][0]["coords"] == (None if coords is None else list(coords))
    assert stim.Circuit(tick_circuit_to_stim(tick)).get_detector_coordinates() == expected
    dem = DetectorErrorModel.from_circuit(tick, p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    assert stim.DetectorErrorModel(str(dem)).get_detector_coordinates() == expected
    assert dem.detector_coordinates() == [(0, None if coords is None else list(coords))]
    assert plan.evaluate([1]) == ((False,), (False,))


def test_coordinates_are_snapshotted_and_not_implicit_schedule_times():
    coords = [0.5, -1, 7, 2]
    program = _program(coords)
    coords[2] = 999
    assert program.detectors[0].coords == (0.5, -1, 7, 2)
    assert "coords=(0.5, -1.0, 7.0, 2.0)" in program.describe()
    plan = program.compile().rescheduled(SchedulePolicy.ASAP_UNITARIES)
    assert plan.detectors[0].coords == (0.5, -1, 7, 2)
    with pytest.raises(ValueError, match="not observables"):
        replace(plan, observables=(MeasurementParity((0,), "logical", coords=(0,)),))


@pytest.mark.parametrize("coords", [(float("nan"),), (float("inf"),), (True,), ("2",)])
def test_non_numerical_or_non_finite_coordinates_fail_at_authoring(coords):
    with pytest.raises(ValueError, match="finite real"):
        _program(coords)


def test_guppy_dem_preserves_four_coordinates_and_signed_evaluation():
    plan = _program((1, 2, 3, 5)).compile()
    detectors, observables = plan.dem_spec()
    build = build_dem_from_guppy(
        plan.guppy_program(),
        num_qubits=1,
        detectors=detectors,
        observables=observables,
        p1=0.001,
        p2=0.002,
        p_meas=0.003,
        p_prep=0.004,
    )
    assert json.loads(build.detectors_json)[0]["coords"] == [1, 2, 3, 5]
    assert stim.DetectorErrorModel(str(build.dem)).get_detector_coordinates() == {0: [1, 2, 3, 5]}
    assert build.evaluate_measurements({0: 1}) == ([0], 0)
