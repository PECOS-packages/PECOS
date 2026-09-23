# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Program-bound certificates for gadget-rendered single-patch memory."""

import hashlib
import json
from itertools import zip_longest

import pecos
import pytest
from pecos._compilation import guppy_to_hugr
from pecos.guppy_gen import get_num_qubits, make_surface_code, make_surface_memory
from pecos.qec import (
    Detector,
    DetectorErrorModel,
    GuppyDemBuilder,
    build_dem_from_guppy,
    result_ref,
    surface_memory_dem_spec,
)
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch

CERTIFICATE = "__pecos_named_measurement_layout_v2__"
NOISE = {"p1": 0.001, "p2": 0.002, "p_meas": 0.003, "p_prep": 0.004}


def _assert_same_dem(actual: str, expected: str) -> None:
    for line, (got, want) in enumerate(zip_longest(actual.splitlines(), expected.splitlines()), start=1):
        assert got == want, f"First differing DEM line {line}: gadget={got!r}, legacy={want!r}"


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("num_rounds", [1, 2])
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("ancilla_budget", [None, 2])
def test_dem_matches_legacy(distance: int, num_rounds: int, basis: str, ancilla_budget: int | None) -> None:
    """The audited route also checks every certified slot against runtime outputs."""
    patch = SurfacePatch.create(distance=distance)
    circuit = generate_tick_circuit_from_patch(patch, num_rounds, basis, ancilla_budget=ancilla_budget)
    kwargs = {
        "num_qubits": get_num_qubits(distance, ancilla_budget=ancilla_budget),
        "detectors_json": circuit.get_meta("detectors"),
        "observables_json": circuit.get_meta("observables"),
        **NOISE,
    }
    gadget = make_surface_memory(distance, num_rounds, basis, ancilla_budget=ancilla_budget)
    legacy = make_surface_code(distance, num_rounds, basis, ancilla_budget=ancilla_budget)
    actual = DetectorErrorModel.from_guppy(gadget, **kwargs).to_string()
    expected = DetectorErrorModel.from_guppy(legacy, **kwargs).to_string()
    _assert_same_dem(actual, expected)


@pytest.mark.parametrize("ancilla_budget", [None, 2])
def test_certificate_is_program_bound(ancilla_budget: int | None) -> None:
    program = make_surface_memory(3, 2, "Z", ancilla_budget=ancilla_budget)
    digest, layout = getattr(program, CERTIFICATE)
    assert isinstance(layout, tuple)
    layout_json = json.dumps(layout, separators=(",", ":"))
    assert digest == hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
    # Exercise the original layout before tampering so a freshly hashed permutation also fails.
    build_dem_from_guppy(
        program,
        num_qubits=get_num_qubits(3, ancilla_budget=ancilla_budget),
        detectors=[Detector(result_ref("final:meas:0"))],
        **NOISE,
    )
    object.__setattr__(program, CERTIFICATE, (digest, (layout[1], layout[0], *layout[2:])))
    with pytest.raises(ValueError, match="measurement-layout certificate does not match the program and layout"):
        build_dem_from_guppy(
            program,
            num_qubits=get_num_qubits(3, ancilla_budget=ancilla_budget),
            detectors=[Detector(result_ref("final:meas:0"))],
            **NOISE,
        )


@pytest.mark.parametrize("route", ["from_guppy", "build_dem_from_guppy", "builder"])
def test_uncertified_gadget_is_rejected(route: str) -> None:
    program = make_surface_memory(3, 1, "Z")
    assert hasattr(program, CERTIFICATE)
    object.__delattr__(program, CERTIFICATE)
    calls = {
        "from_guppy": lambda: DetectorErrorModel.from_guppy(program, num_qubits=get_num_qubits(3), detectors_json="[]"),
        "build_dem_from_guppy": lambda: build_dem_from_guppy(program, num_qubits=get_num_qubits(3), detectors=[]),
        "builder": lambda: GuppyDemBuilder()
        .with_program(program)
        .with_qubits(get_num_qubits(3))
        .with_detectors([])
        .build(),
    }
    with pytest.raises(ValueError, match=r"requires a statically straight-line Guppy program.*trusted generator-owned"):
        calls[route]()


@pytest.mark.parametrize("ancilla_budget", [None, 2])
def test_named_columns_end_to_end(ancilla_budget: int | None) -> None:
    program = make_surface_memory(3, 2, "Z", ancilla_budget=ancilla_budget)
    num_qubits = get_num_qubits(3, ancilla_budget=ancilla_budget)
    detectors, observables = surface_memory_dem_spec(3, 2, "Z", ancilla_budget=ancilla_budget)
    detectors.append(Detector(result_ref("final:meas:0")))
    build = build_dem_from_guppy(
        program,
        num_qubits=num_qubits,
        detectors=detectors,
        observables=observables,
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )
    columns = (
        pecos.sim(program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(num_qubits)
        .seed(7)
        .run(3)
        .to_shot_map()
        .to_dict()
    )
    evaluated = build.evaluate_result_columns(columns)
    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"
    assert evaluated == [([0] * (build.dem.num_detectors - 1) + [int(bit)], 0) for bit in columns["final:meas:0"]]
    assert len(evaluated) == 3


@pytest.mark.parametrize("basis", ["z", "x"])
@pytest.mark.parametrize("ancilla_budget", [None, 2])
def test_patch_and_check_plan(basis: str, ancilla_budget: int | None) -> None:
    """Patch geometry and the plan must survive both rendering and certification."""
    patch = SurfacePatch.create(dx=3, dz=5)
    plan = "cx_balanced_data_v1"
    program = make_surface_memory(patch, 2, basis, ancilla_budget=ancilla_budget, check_plan=plan)
    circuit = generate_tick_circuit_from_patch(patch, 2, basis.upper(), ancilla_budget=ancilla_budget, check_plan=plan)
    build = (
        GuppyDemBuilder()
        .with_program(program)
        .with_qubits(
            get_num_qubits(patch=patch, ancilla_budget=ancilla_budget, check_plan=plan),
        )
    )
    build.with_detectors_json(circuit.get_meta("detectors"))
    build.with_observables_json(circuit.get_meta("observables"))
    assert build.build().audit["named_result_binding"] == "generator_layout_v2_program_bound"
