"""Named authoring, composition contracts, and provenance through scheduling."""

import pytest
from pecos.qec.color import ColorExtraction, color_code_implementation, color_code_spec, color_memory_program
from pecos.qec.generic import Gadget, GadgetFlow, GadgetProgram, Gate, NamedParity, SchedulePolicy


def zz_gadget():
    return Gadget(
        name="measure ZZ",
        roles=("left", "right", "ancilla"),
        inputs=("left", "right"),
        outputs=("left", "right"),
        moments=(
            (Gate("R", ("ancilla",)),),
            (Gate("CX", ("left", "ancilla")),),
            (Gate("CX", ("right", "ancilla")),),
            (Gate("M", ("ancilla",), outcome="syndrome"),),
        ),
        flows=(GadgetFlow("ZZ", NamedParity(("syndrome",))),),
    )


def zz_memory():
    program = GadgetProgram(("alice", "bob", "probe"))
    program.append(
        Gadget(
            "prepare",
            ("alice", "bob"),
            (),
            ("alice", "bob"),
            ((Gate("R", ("alice",)), Gate("R", ("bob",))), (Gate("H", ("alice",)),)),
        ),
        label="prepare",
    )
    # Binding order deliberately differs from physical allocation order.
    bindings = {"left": "bob", "right": "alice", "ancilla": "probe"}
    first = program.append(zz_gadget(), label="first", bindings=bindings)
    second = program.append(zz_gadget(), label="second", bindings=bindings)
    readout = program.append(
        Gadget(
            "readout",
            ("alice", "bob"),
            ("alice", "bob"),
            (),
            ((Gate("M", ("alice",), "alice"), Gate("M", ("bob",), "bob")),),
        ),
        label="readout",
    )
    program.detector("repeat", first["syndrome"] ^ second.flow("ZZ"))
    program.detector("final", second["syndrome"] ^ readout["alice"] ^ readout["bob"])
    program.observable("bob Z", readout["bob"])
    return program


def test_custom_gadget_lowers_without_backend_changes():
    stim = pytest.importorskip("stim")
    program = zz_memory()
    plan = program.compile()
    # Independent hand-authored physical sequence (no gadget/emitter helpers).
    expected = stim.Circuit("""
        R 0 1
        TICK
        H 0
        TICK
        R 2
        TICK
        CX 1 2
        TICK
        CX 0 2
        TICK
        M 2
        TICK
        R 2
        TICK
        CX 1 2
        TICK
        CX 0 2
        TICK
        M 2
        TICK
        M 0 1
        TICK
        DETECTOR rec[-4] rec[-3]
        DETECTOR rec[-3] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    assert plan.to_stim() == expected
    assert not expected.compile_detector_sampler(seed=5).sample(64, append_observables=True).any()
    assert str(program.detectors[0].parity) == "first/syndrome ^ second/syndrome"
    assert plan.explain_detector("repeat") == "repeat = first/syndrome ^ second/syndrome"
    assert plan.measurement_aliases["second/syndrome"] == "qec:m:1"
    assert "first/moment[1]/op[0]" in plan.guppy_source()
    assert "first/syndrome" in plan.guppy_source()


def test_named_xor_and_defensive_gadget_snapshot():
    assert NamedParity("syndrome") == NamedParity(("syndrome",))
    assert Gate("H", "data").roles == ("data",)
    assert NamedParity(("a", "b", "a")) == NamedParity(("b",))
    assert NamedParity(("a",), True) ^ NamedParity(("a",), True) == NamedParity()
    assert str(NamedParity()) == "0"
    roles = ["data"]
    moments = [[Gate("R", roles)], [Gate("M", roles, "result")]]
    gadget = Gadget("memory", roles, [], [], moments)
    roles.append("other")
    moments.clear()
    assert gadget.roles == ("data",)
    assert len(gadget.moments) == 2
    with pytest.raises(TypeError, match="bare string"):
        GadgetProgram("data")


def test_composition_rejects_mismatches_without_mutating_program():
    program = GadgetProgram(("a", "b", "scratch"))
    bindings = {"left": "a", "right": "b", "ancilla": "scratch"}
    with pytest.raises(ValueError, match="Live-port mismatch"):
        program.append(zz_gadget(), label="bad", bindings=bindings)
    assert not program.calls
    with pytest.raises(ValueError, match="alias"):
        program.append(zz_gadget(), label="bad", bindings=dict(bindings, right="a"))
    with pytest.raises(ValueError, match="map every"):
        program.append(zz_gadget(), label="bad", bindings={"left": "a"})
    with pytest.raises(ValueError, match="unknown measurement"):
        program.detector("unknown", NamedParity(("future/result",)))
    assert not program.detectors


def test_basis_frame_contract_requires_explicit_conversion():
    program = GadgetProgram(("data",))
    program.append(Gadget("prepare", ("data",), (), ("data",), ((Gate("R", ("data",)),),)), label="prepare")
    readout = Gadget(
        "readout H-frame",
        ("data",),
        ("data",),
        (),
        ((Gate("H", ("data",)),), (Gate("M", ("data",), "bit"),)),
        input_h_frame=("data",),
    )
    with pytest.raises(ValueError, match="Basis-frame mismatch"):
        program.append(readout, label="readout")
    with pytest.raises(ValueError, match="live roles"):
        program.compile()
    program.append(
        Gadget("enter H frame", ("data",), ("data",), ("data",), ((Gate("H", ("data",)),),), output_h_frame=("data",)),
        label="convert",
    )
    program.append(readout, label="readout")
    assert program.compile().num_measurements == 1
    with pytest.raises(ValueError, match="Duplicate gadget call"):
        program.append(readout, label="readout")


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Gate("M", ("data",)), "must name"),
        (lambda: Gate("H", ("data",), "invalid"), "must name"),
        (lambda: Gate("CX", ("data", "data")), "alias"),
        (lambda: Gadget("bad", ("data",), (), (), ((Gate("H", ("data",)),),)), "non-live"),
        (lambda: Gadget("bad", ("data",), (), (), ((Gate("R", ("data",)),),)), "output contract"),
        (
            lambda: Gadget("bad", ("data",), (), (), (), flows=(GadgetFlow("check", NamedParity(("unknown",))),)),
            "unknown outcome",
        ),
    ],
)
def test_invalid_gadget_contracts(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_asap_provenance_dependencies_and_separate_compilations():
    program = GadgetProgram(("a", "b"))
    program.append(
        Gadget("prepare", ("a", "b"), (), ("a", "b"), ((Gate("R", ("a",)), Gate("R", ("b",))),)),
        label="prepare",
    )
    for role in ("a", "b"):
        program.append(
            Gadget("H", ("wire",), ("wire",), ("wire",), ((Gate("H", ("wire",)),),)),
            label=f"H_{role}",
            bindings={"wire": role},
        )
    program.append(
        Gadget("readout", ("a", "b"), ("a", "b"), (), ((Gate("M", ("a",), "a"), Gate("M", ("b",), "b")),)),
        label="readout",
    )
    reference = program.compile()
    packed = program.compile(SchedulePolicy.ASAP_UNITARIES)
    assert len(reference.moments) == 4
    assert len(packed.moments) == 3
    before = {row.id: row for row in reference.inspect()}
    after = {row.id: row for row in packed.inspect()}
    assert before.keys() == after.keys()
    assert before["H_b/moment[0]/op[0]"].moment == 2
    assert after["H_b/moment[0]/op[0]"].moment == 1
    for row in after.values():
        assert all(after[dep].moment < row.moment for dep in row.dependencies)
        assert "ASAP" in row.placement
    assert packed.measurement_aliases == reference.measurement_aliases
    assert program.compile() == reference
    assert packed.resources.peak_live == 2
    assert packed.resources.physical_operations == 6
    assert "ASAP per-role order" in packed.explain_operation("H_b/moment[0]/op[0]")


@pytest.mark.parametrize("choice", list(ColorExtraction))
def test_paper_program_named_semantics_survive_schedule(choice):
    family = "488" if choice in {ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ} else "666"
    code = color_code_spec(family, 3)
    implementation = color_code_implementation(code, choice)
    program = color_memory_program(implementation, 3, "X")
    assert code.checks[0].name == "X[0]"
    assert all(role in code.data_roles for check in code.checks for role, _ in check.terms)
    assert "input:X[0]" in implementation.gadget().describe()
    assert "round[1]" in program.describe()
    for policy in SchedulePolicy:
        plan = program.compile(policy)
        assert len(plan.measurement_aliases) == plan.num_measurements
        assert "round[0]/" in plan.explain_detector("round:1:X:0")
        assert "round[1]/" in plan.explain_detector("round:1:X:0")
        rows = plan.inspect()
        assert all(row.source for row in rows)
        assert len({row.id for row in rows}) == len(rows)
        assert tuple(row.measurement for row in rows if row.measurement) == plan.measurement_names
        indexed = {row.id: row for row in rows}
        assert all(indexed[dep].moment < row.moment for row in rows for dep in row.dependencies)
        if choice == ColorExtraction.SWAP_SUPERDENSE:
            assert plan.resources.relabels > 0
            assert plan.resources.physical_moments < plan.resources.moments
            relabel = next(row for row in rows if row.gate == "RELABEL")
            role = relabel.roles[0]
            later = next(row for row in rows if row.moment > relabel.moment and role in row.roles)
            assert later.carriers[later.roles.index(role)] == relabel.carriers[1]


def test_custom_named_gadget_executes_through_guppy():
    from pecos import sim, stabilizer

    plan = zz_memory().compile(SchedulePolicy.ASAP_UNITARIES)
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(6).run(32).to_dict()
    for shot in range(32):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert not any(observables)
