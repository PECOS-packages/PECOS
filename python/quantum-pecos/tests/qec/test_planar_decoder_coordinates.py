"""Coordinate oracles copied from Shutty's unchanged published noisy circuits."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import stim
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    color_memory_program,
)
from pecos.qec.generic import SchedulePolicy
from pecos.qec.surface import tick_circuit_to_stim

ROOT = Path(__file__).parent / "fixtures"
CASES = json.loads((ROOT / "si1000/planar_coordinates.json").read_text())["cases"]


@lru_cache
def _implementation(distance, extraction):
    return color_code_implementation(color_code_spec("488", distance), extraction)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_published_planar_coordinates_survive_all_lowerings(case):
    _, _, distance, rounds, basis, phase, _ = case["file"].split("_")
    impl = _implementation(int(distance[1:]), ColorExtraction.BRICKWORK_CZ)
    program = color_memory_program(impl, int(rounds[1:]), basis, start_phase=int(phase[5:]))
    plan = program.compile()
    expected = dict(enumerate(case["coords"]))
    for schedule in SchedulePolicy:
        assert [d.coords for d in program.compile(schedule).detectors] == [d.coords for d in plan.detectors]
    for circuit in (plan.to_stim(), stim.Circuit(tick_circuit_to_stim(plan.to_tick_circuit()))):
        actual = circuit.get_detector_coordinates()
        assert actual.keys() == expected.keys()
        for key in actual:
            # Stim reference files print finite-precision decimal coordinates.
            assert actual[key] == pytest.approx(expected[key], abs=5e-5, rel=0)
    actual = plan.to_tick_circuit().circuit_definitions()["detectors"]
    assert [list(d.coords) for d in plan.detectors] == [d["coords"] for d in actual]


@pytest.mark.parametrize("path", sorted((ROOT / "paper_color_codes").glob("488_*_cnot_*.stim")), ids=lambda p: p.stem)
def test_cnot_preserves_original_index_color_coordinates(path):
    _, distance, _, basis, phase = path.stem.split("_")
    impl = _implementation(int(distance[1:]), ColorExtraction.BRICKWORK_CX)
    plan = color_memory(impl, 3, basis, start_phase=int(phase[1:]))
    assert plan.to_stim().get_detector_coordinates() == stim.Circuit(path.read_text()).get_detector_coordinates()
