"""Checked local faults through the native scheduled sim() route."""

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
        profile = pr.scheduled_event_local_noise(profile, lambda _: PassThrough())
    else:
        classical.scheduled_batches()
    return (
        pecos.sim(pecos.Qis(program))
        .classical(classical.interface(pr.qis_helios_interface()))
        .qubits(1)
        .quantum(pr.state_vector())
        .noise(profile)
        .seed(42)
        .workers(2)
    )


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize(
    ("faults", "expected"),
    [({"prep": 1.0}, 1), ({"meas0": 1.0}, 1), ({"prep": 1.0, "meas1": 1.0}, 0)],
)
def test_local_faults_and_repeated_worker_runs(tmp_path, scheduled_support, events, faults, expected):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 0)
    profile = pr.scheduled_local_noise(pr.scheduled_idle_noise(1), **faults)
    program = scheduled_support.ramsey.replace(
        "declare void @setup",
        "declare void @___reset(i64)\ndeclare void @setup",
    ).replace("  %q = call i64 @___qalloc()", "  %q = call i64 @___qalloc()\n  call void @___reset(i64 %q)")
    built = simulation(library, program, profile, events).build()
    for shots in (4, 6):
        assert built.run(shots).to_dict()["measurement_0"] == [expected] * shots


@pytest.mark.parametrize("events", [False, True])
def test_zero_local_faults_preserve_seeded_idle_output(tmp_path, scheduled_support, events):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 1_000_000_000)
    idle = pr.scheduled_idle_noise(1, linear=0.2, coherent=0.4)
    local = pr.scheduled_local_noise(idle)
    actual = simulation(library, scheduled_support.ramsey, local, events)
    expected = simulation(library, scheduled_support.ramsey, local, events).noise(
        pr.scheduled_event_idle_noise(idle, lambda _: PassThrough()) if events else idle,
    )
    assert actual.run(32).to_dict() == expected.run(32).to_dict()


@pytest.mark.parametrize("parameter", ["p1", "p2", "prep", "meas0", "meas1"])
@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, math.nan])
def test_invalid_probability(parameter, value):
    import pecos_rslib as pr

    with pytest.raises(ValueError, match="probabilities"):
        pr.scheduled_local_noise(pr.scheduled_idle_noise(1), **{parameter: value})


@pytest.mark.parametrize("parameter", ["emission", "crosstalk", "idle_after_2q", "p2_angle_params"])
def test_unsupported_noise_settings_are_not_silently_ignored(parameter):
    import pecos_rslib as pr

    with pytest.raises(TypeError):
        pr.scheduled_local_noise(pr.scheduled_idle_noise(1), **{parameter: 0.1})
