"""Local idle channels through native scheduled extraction and production sim()."""

import math

import pytest


class PassThrough:
    def validate(self, batch):
        assert all(not isinstance(op, tuple) for op in batch.operations)

    def translate(self, batch):
        return batch.operations


def simulation(library, program, profile, events):
    import pecos
    import pecos_rslib as pr
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    classical = pr.qis_engine().selene_runtime_plugin(str(library), SimpleRuntimePlugin().get_init_args())
    if events:
        classical.scheduled_event_batches()
        profile = pr.scheduled_event_idle_noise(profile, lambda _: PassThrough())
    else:
        classical.scheduled_batches()
    return (
        pecos.sim(pecos.Qis(program) if isinstance(program, str) else program)
        .classical(classical.interface(pr.qis_helios_interface()))
        .qubits(1)
        .quantum(pr.state_vector())
        .noise(profile)
        .seed(42)
        .workers(2)
    )


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("sine", [False, True])
def test_forced_idle_leakage_feedback_reset_and_repeated_runs(tmp_path, scheduled_support, events, sine):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 1_000_000_000)
    program = scheduled_support.ramsey.replace(
        "declare void @setup",
        "declare i1 @___read_future_bool(i64)\ndeclare void @___reset(i64)\ndeclare void @setup",
    ).replace(
        "  call void @___qfree(i64 %q)",
        """  %value = call i1 @___read_future_bool(i64 %r)
  br i1 %value, label %reset, label %done
reset:
  call void @___reset(i64 %q)
  br label %done
done:
  %second = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)""",
    )
    assert "br i1 %value" in program
    profile = (
        pr.scheduled_idle_noise(1, sine=math.pi / 2, sine_model={"L": 1.0})
        if sine
        else pr.scheduled_idle_noise(1, linear=1.0, linear_model={"L": 1.0})
    )
    built = simulation(library, program, profile, events).build()
    for shots in (4, 6):
        result = built.run(shots).to_dict()
        assert result["measurement_0"] == [1] * shots
        assert result["measurement_1"] == [0] * shots


@pytest.mark.parametrize("events", [False, True])
def test_general_profile_z_defaults_preserve_existing_seeded_results(tmp_path, scheduled_support, events):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 1_000_000_000)
    kwargs = {"linear": 0.2, "sine": 0.3, "coherent": 0.4}
    general = simulation(library, scheduled_support.ramsey, pr.scheduled_idle_noise(1, **kwargs), events)
    narrow = simulation(library, scheduled_support.ramsey, pr.scheduled_idle_noise(1), events)
    narrow.noise(
        pr.scheduled_event_idle_z(1, lambda _: PassThrough(), **kwargs) if events else pr.scheduled_idle_z(1, **kwargs),
    )
    assert general.run(32).to_dict() == narrow.run(32).to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear": -1},
        {"linear_model": {}},
        {"linear_model": {"X": -0.5, "L": 1.5}},
        {"linear_model": {"L": 0.5}},
        {"linear_model": {"bad": 1.0}},
        {"sine_model": {"L": math.inf}},
        {"sine": math.nan},
        {"coherent_model": {"L": 1.0}},
        {"coherent": 1e308, "coherent_model": {"RX": 2.0}},
    ],
)
def test_invalid_profile_rejects_with_python_value_error(kwargs):
    import pecos_rslib as pr

    with pytest.raises(ValueError, match="scheduled"):
        pr.scheduled_idle_noise(1, **kwargs)


def test_event_factory_must_be_callable():
    import pecos_rslib as pr

    with pytest.raises(TypeError, match="adapter_factory must be callable"):
        pr.scheduled_event_idle_noise(pr.scheduled_idle_noise(1), None)


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("gap", [0, 1_000_000_000])
def test_compiled_guppy_responds_to_idle_leakage(tmp_path, scheduled_support, events, gap):
    import pecos_rslib as pr
    from guppylang import guppy
    from guppylang.std.angles import angle
    from guppylang.std.builtins import owned, result
    from guppylang.std.quantum import measure, qubit, ry

    @guppy.declare
    def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...

    @guppy
    def ramsey() -> None:
        q = qubit()
        ry(q, angle(0.5))
        q = pecos_qis_runtime_barrier_qubit_hugr(q)
        ry(q, angle(-0.5))
        result("outcome", measure(q).read())

    library, _ = scheduled_support.build_runtime(tmp_path, gap)
    profile = pr.scheduled_idle_noise(1, linear=1.0, linear_model={"L": 1.0})
    results = simulation(library, ramsey, profile, events).run(4).to_dict()
    assert results["outcome"] == [int(gap != 0)] * 4


@pytest.mark.parametrize("builder_name", ["qasm_engine", "phir_json_engine", "phir_engine"])
def test_other_engines_reject_general_event_profile(builder_name):
    import pecos_rslib as pr

    profile = pr.scheduled_event_idle_noise(pr.scheduled_idle_noise(1), lambda _: PassThrough())
    with pytest.raises(TypeError, match="requires QIS/HUGR"):
        getattr(pr, builder_name)().to_sim().noise(profile)
