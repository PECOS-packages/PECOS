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
from pecos.qec.dem_spec import ResultRef
from pecos.qec.surface import SurfacePatch, circuit_builder
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

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
    detectors, observables = surface_memory_dem_spec(distance, num_rounds, basis, ancilla_budget=ancilla_budget)
    kwargs = {
        "num_qubits": get_num_qubits(distance, ancilla_budget=ancilla_budget),
        "detectors": detectors,
        "observables": observables,
        **NOISE,
    }
    gadget = make_surface_memory(patch, num_rounds, basis, ancilla_budget=ancilla_budget)
    legacy = make_surface_code(distance, num_rounds, basis, ancilla_budget=ancilla_budget)
    actual = build_dem_from_guppy(gadget, **kwargs).dem.to_string()
    expected = build_dem_from_guppy(legacy, **kwargs).dem.to_string()
    _assert_same_dem(actual, expected)


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("ancilla_budget", [None, 2])
def test_surface_spec_preserves_annotation_fields(
    monkeypatch: pytest.MonkeyPatch,
    basis: str,
    ancilla_budget: int | None,
) -> None:
    """Transport references without losing IDs, coordinates or opaque metadata."""
    circuit = generate_tick_circuit_from_patch(
        SurfacePatch.create(distance=3),
        2,
        basis,
        ancilla_budget=ancilla_budget,
        add_typed_annotations=False,
    )
    entries_by_kind = {}
    for kind in ("detectors", "observables"):
        entries = json.loads(circuit.get_meta(kind))
        assert entries
        for index, entry in enumerate(entries):
            entry["id"] = 10 + 3 * index
            entry["coords"] = [index, 0.5, -2]
            entry["custom"] = {"nested": [kind, index]}
        entries_by_kind[kind] = entries
        circuit.set_meta(kind, json.dumps(entries))
    monkeypatch.setattr(circuit_builder, "generate_tick_circuit_from_patch", lambda *_args, **_kwargs: circuit)
    detectors, observables = surface_memory_dem_spec(3, 2, basis, ancilla_budget=ancilla_budget)
    for specs, kind in ((detectors, "detectors"), (observables, "observables")):
        for spec, entry in zip(specs, entries_by_kind[kind], strict=True):
            assert spec.id == entry["id"]
            assert all(isinstance(ref, ResultRef) for ref in spec.refs)
            excluded = {"id", "records"}
            if kind == "detectors":
                assert spec.coords == tuple(entry["coords"])
                excluded.add("coords")
            # Observable coordinates, if supplied, belong to its metadata.
            assert spec.metadata == {key: value for key, value in entry.items() if key not in excluded}
    assert {spec.metadata["fault_sector"] for spec in detectors} == {"X", "Z"}


@pytest.mark.parametrize("ancilla_budget", [None, 2])
@pytest.mark.parametrize("rehash", [False, True])
def test_certificate_is_program_bound(ancilla_budget: int | None, rehash: bool) -> None:
    program = make_surface_memory(SurfacePatch.create(distance=3), 2, "Z", ancilla_budget=ancilla_budget)
    digest, layout = getattr(program, CERTIFICATE)
    assert isinstance(layout, tuple)
    layout_json = json.dumps(layout, separators=(",", ":"))
    assert digest == hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
    # The original binding must succeed before either integrity check is challenged.
    original = build_dem_from_guppy(
        program,
        num_qubits=get_num_qubits(3, ancilla_budget=ancilla_budget),
        detectors=[Detector(result_ref("final:meas:0"))],
        **NOISE,
    )
    # A stale digest fails integrity; a re-hashed inventory may have any order.
    permuted = (layout[1], layout[0], *layout[2:])
    error = "measurement-layout certificate does not match the program and layout"
    if rehash:
        layout_json = json.dumps(permuted, separators=(",", ":"))
        digest = hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
    object.__setattr__(program, CERTIFICATE, (digest, permuted))

    def build():
        return build_dem_from_guppy(
            program,
            num_qubits=get_num_qubits(3, ancilla_budget=ancilla_budget),
            detectors=[Detector(result_ref("final:meas:0"))],
            **NOISE,
        )

    if rehash:
        assert build().detectors_json == original.detectors_json
    else:
        with pytest.raises(ValueError, match=error):
            build()


@pytest.mark.parametrize("route", ["from_guppy", "build_dem_from_guppy", "builder"])
def test_uncertified_gadget_is_rejected(route: str) -> None:
    program = make_surface_memory(SurfacePatch.create(distance=3), 1, "Z")
    assert hasattr(program, CERTIFICATE)
    object.__delattr__(program, CERTIFICATE)
    calls = {
        "from_guppy": lambda: DetectorErrorModel.from_guppy(program, num_qubits=get_num_qubits(3), detectors_json="[]"),
        "build_dem_from_guppy": lambda: build_dem_from_guppy(program, num_qubits=get_num_qubits(3), detectors=[]),
        "builder": lambda: (
            GuppyDemBuilder().with_program(program).with_qubits(get_num_qubits(3)).with_detectors([]).build()
        ),
    }
    with pytest.raises(
        ValueError,
        match=r"requires a statically straight-line Guppy program.*generator-owned measurement-layout certificate",
    ):
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
    circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        2,
        basis.upper(),
        ancilla_budget=ancilla_budget,
        check_plan=plan,
        circuit_source="traced_qis",
    )
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


@pytest.mark.parametrize("num_rounds", [1, 2])
@pytest.mark.parametrize("generator", ["gadget", "legacy"])
def test_surface_slots_have_independent_physical_identity(num_rounds, generator):
    """Check surface (qubit, occurrence, basis), including repeated output tags.

    These Z-basis, unconstrained d=3 schedules share physical allocation with
    the trace. Helix allocation does not: its evidence is unique physical tag
    labels and end-to-end helix test_dem runs, not this qubit oracle.
    """
    from collections import Counter

    from pecos.qec.dem import _generator_certified_result_traces, _GuppyProgramAnalysis
    from pecos.qec.surface.decode import _surface_abstract_measurement_result_refs
    from pecos.tracing import _trace_program_to_tick_circuit_with_result_traces

    patch = SurfacePatch.create(distance=3)
    abstract = generate_tick_circuit_from_patch(patch, num_rounds, "Z")
    program = (
        make_surface_memory(patch, num_rounds, "Z") if generator == "gadget" else make_surface_code(3, num_rounds, "Z")
    )
    traced, traces = _trace_program_to_tick_circuit_with_result_traces(program, get_num_qubits(3))
    layout = getattr(program, CERTIFICATE)[1]
    bound = _generator_certified_result_traces(
        layout,
        _GuppyProgramAnalysis(guppy_to_hugr(program)),
        traced,
        traces,
        required_tags=[],
    )

    def physical_measurements(circuit):
        occurrences = Counter()
        measurements = []
        for tick in range(circuit.num_ticks()):
            for gate in circuit.get_tick(tick).gate_batches():
                if str(gate.gate_type).rsplit(".", 1)[-1] not in {"MZ", "MX", "MPZ", "MeasureFree"}:
                    continue
                for qubit in gate.qubits:
                    basis = "X" if str(gate.gate_type).endswith(".MX") else "Z"
                    measurements.append((qubit, occurrences[qubit], basis))
                    occurrences[qubit] += 1
        assert len(measurements) == len(circuit.measurement_emission())
        return measurements

    expected = {}
    occurrences = Counter()
    for ref, physical in zip(
        _surface_abstract_measurement_result_refs(abstract),
        physical_measurements(abstract),
        strict=True,
    ):
        if ref[0] == "scalar":
            slot = ref[1], occurrences[ref[1]]
            occurrences[ref[1]] += 1
        else:
            slot = f"{ref[1]}:meas:{ref[2]}", 0
        expected[slot] = physical
    actual_by_id = {
        mid: physical
        for (_, mid), physical in zip(
            traced.measurement_emission(),
            physical_measurements(traced),
            strict=True,
        )
    }
    assert {(item["name"], item["occurrence"]) for item in bound} == expected.keys()
    assert [item["result_ids"][0] for item in bound] != [mid for _, mid in traced.measurement_emission()]
    if num_rounds == 2:
        assert any(occurrence >= 1 for _, occurrence in expected)
    for item in bound:
        slot = item["name"], item["occurrence"]
        assert actual_by_id[item["result_ids"][0]] == expected[slot], slot
    # Shadow-backed final scalar prints always carry their own element ID.
    assert all(len(trace["result_ids"]) == 1 for trace in traces if trace["name"].startswith("final:meas:"))
