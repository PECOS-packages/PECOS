# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Program-bound certificates for gadget-rendered single-patch memory."""

import hashlib
import json
from itertools import zip_longest

import pecos
import pytest
from pecos._compilation import guppy_to_hugr
from pecos.guppy_gen import gadget_render, generate_guppy_source, get_num_qubits, make_surface_code, make_surface_memory
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
    gadget = make_surface_memory(patch, num_rounds, basis, ancilla_budget=ancilla_budget)
    legacy = make_surface_code(distance, num_rounds, basis, ancilla_budget=ancilla_budget)
    actual = DetectorErrorModel.from_guppy(gadget, **kwargs).to_string()
    expected = DetectorErrorModel.from_guppy(legacy, **kwargs).to_string()
    _assert_same_dem(actual, expected)


@pytest.mark.parametrize("ancilla_budget", [None, 2])
@pytest.mark.parametrize("rehash", [False, True])
def test_certificate_is_program_bound(ancilla_budget: int | None, rehash: bool) -> None:
    program = make_surface_memory(SurfacePatch.create(distance=3), 2, "Z", ancilla_budget=ancilla_budget)
    digest, layout = getattr(program, CERTIFICATE)
    assert isinstance(layout, tuple)
    layout_json = json.dumps(layout, separators=(",", ":"))
    assert digest == hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
    # The original binding must succeed before either integrity check is challenged.
    build_dem_from_guppy(
        program,
        num_qubits=get_num_qubits(3, ancilla_budget=ancilla_budget),
        detectors=[Detector(result_ref("final:meas:0"))],
        **NOISE,
    )
    # A stale digest checks integrity; an honest re-hash checks runtime measurement identity.
    permuted = (layout[1], layout[0], *layout[2:])
    error = "measurement-layout certificate does not match the program and layout"
    if rehash:
        layout_json = json.dumps(permuted, separators=(",", ":"))
        digest = hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
        error = (
            r"runtime result trace 'sx1:init:meas:1'\[0\] has measurement id 1, "
            r"but the generator-certified layout requires 0"
        )
    object.__setattr__(program, CERTIFICATE, (digest, permuted))
    with pytest.raises(ValueError, match=error):
        build_dem_from_guppy(
            program,
            num_qubits=get_num_qubits(3, ancilla_budget=ancilla_budget),
            detectors=[Detector(result_ref("final:meas:0"))],
            **NOISE,
        )


@pytest.mark.parametrize("route", ["from_guppy", "build_dem_from_guppy", "builder"])
def test_uncertified_gadget_is_rejected(route: str) -> None:
    program = make_surface_memory(SurfacePatch.create(distance=3), 1, "Z")
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
    program = make_surface_memory(SurfacePatch.create(distance=3), 2, "Z", ancilla_budget=ancilla_budget)
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
def test_patch_and_check_plan(basis: str) -> None:
    """Patch geometry and the plan must survive both rendering and certification."""
    patch = SurfacePatch.create(dx=3, dz=5)
    ancilla_budget = 2
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


@pytest.mark.parametrize(
    ("dx", "dz", "rotated"),
    [
        (dx, dz, rotated)
        for dx, dz in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]
        for rotated in (True, False)
    ]
    + [(2, 2, False), (2, 3, False), (2, 4, False), (3, 2, False), (4, 2, False)],
)
def test_empty_syndrome_geometry_rejected_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
    dx: int,
    dz: int,
    rotated: bool,
) -> None:
    """Pin the geometries found by compiling the rendered dx,dz=1..4 matrix."""
    patch = SurfacePatch.create(dx=dx, dz=dz, rotated=rotated)

    def unexpected_render(*_args, **_kwargs):
        pytest.fail("invalid geometry reached gadget rendering")

    monkeypatch.setattr(gadget_render, "render_gadget_function", unexpected_render)
    error = (
        f"surface gadget module for dx={dx}, dz={dz}, rotated={rotated} requires nonempty X and Z "
        "stabilizer families; Guppy cannot infer the type of empty syndrome arrays"
    )
    with pytest.raises(ValueError, match=error):
        gadget_render.render_surface_gadget_module(patch)
    legacy_error = error.replace("surface gadget module", "surface Guppy source")
    with pytest.raises(ValueError, match=legacy_error):
        generate_guppy_source(patch)


@pytest.mark.parametrize(("distance", "rotated"), [(2, True), (4, True), (3, False)])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_even_and_unrotated_memory_compile(distance: int, rotated: bool, basis: str) -> None:
    patch = SurfacePatch.create(distance=distance, rotated=rotated)
    program = make_surface_memory(patch, 1, basis)
    assert guppy_to_hugr(program)
    assert hasattr(program, CERTIFICATE)


@pytest.mark.parametrize("basis", ["y", "Y", ""])
def test_invalid_basis_preserves_caller_value(basis: str) -> None:
    with pytest.raises(ValueError, match=f"basis must be 'Z' or 'X', got {basis!r}"):
        make_surface_memory(SurfacePatch.create(distance=3), 1, basis)
