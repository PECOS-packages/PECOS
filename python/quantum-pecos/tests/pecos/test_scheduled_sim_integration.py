"""Production sim() transport with explicitly synthetic nonzero native timestamps."""

import math
from pathlib import Path

import pytest


@pytest.fixture(params=[0, 1_000_000_000])
def timed_runtime(request, tmp_path, scheduled_support):
    """Add explicitly synthetic timing to native callbacks."""
    library, gap = scheduled_support.build_runtime(tmp_path, request.param)
    return library, gap, scheduled_support.ramsey


def simulation(library: Path, program: str):
    import pecos
    import pecos_rslib
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    classical = (
        pecos_rslib.qis_engine()
        .selene_runtime_plugin(
            str(library),
            SimpleRuntimePlugin().get_init_args(),
            custom_event_policy="reject_unhandled",
        )
        .scheduled_batches()
        .interface(pecos_rslib.qis_helios_interface())
    )
    return (
        pecos.sim(pecos.Qis(program))
        .classical(classical)
        .qubits(1)
        .quantum(pecos_rslib.state_vector())
        .seed(42)
        .workers(1)
    )


def test_sim_ramsey_responds_to_synthetic_timing(timed_runtime: tuple[Path, int, str]) -> None:
    import pecos_rslib

    library, gap, program = timed_runtime
    result = simulation(library, program).noise(pecos_rslib.scheduled_idle_z(1, coherent=math.pi)).run(3).to_dict()
    assert result["measurement_0"] == [int(gap != 0)] * 3


def test_sim_rejects_scheduled_transport_without_capability(timed_runtime: tuple[Path, int, str]) -> None:
    with pytest.raises(RuntimeError, match="scheduled idle capability required"):
        simulation(timed_runtime[0], timed_runtime[2]).run(1)


def test_sim_feedback_and_worker_shots(timed_runtime: tuple[Path, int, str]) -> None:
    import pecos_rslib

    program = (
        timed_runtime[2]
        .replace(
            "declare void @setup",
            "declare i1 @___read_future_bool(i64)\ndeclare void @setup",
        )
        .replace(
            "  call void @___qfree(i64 %q)",
            """  %value = call i1 @___read_future_bool(i64 %r)
  br i1 %value, label %flip, label %done
flip:
  call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
  br label %done
done:
  %second = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)""",
        )
    )
    library, gap, _ = timed_runtime
    result = (
        simulation(library, program)
        .noise(pecos_rslib.scheduled_idle_z(1, coherent=math.pi))
        .workers(2)
        .run(6)
        .to_dict()
    )
    assert result["measurement_0"] == [int(gap != 0)] * 6
    assert result["measurement_1"] == [0] * 6


@pytest.mark.parametrize("runtime_name", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_public_native_route_with_zero_timing(runtime_name: str, scheduled_support) -> None:
    """Zero timing checks scheduling/terminal drain, not idle-noise sensitivity."""
    import pecos
    import pecos_rslib as pr

    classical = pr.qis_engine().selene_runtime(runtime_name).scheduled_batches().interface(pr.qis_helios_interface())
    result = (
        pecos.sim(pecos.Qis(scheduled_support.ramsey))
        .classical(classical)
        .qubits(1)
        .quantum(pr.state_vector())
        .noise(pr.scheduled_idle_z(1, coherent=math.pi))
        .seed(42)
        .workers(1)
        .run(3)
        .to_dict()
    )
    assert result["measurement_0"] == [0] * 3


def test_preparation_policy_through_both_python_routes(tmp_path, scheduled_support):
    """A discarded pre-reset noise draw still affects subsequent seeded noise."""
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 1_000_000_000, initial_nanos=1_000_000_000)

    class PassThrough:
        def validate(self, batch):
            assert all(not isinstance(op, tuple) for op in batch.operations)

        def translate(self, batch):
            return batch.operations

    # Allocation alone need not emit preparation in a compatible runtime.
    program = scheduled_support.ramsey.replace(
        "declare i64 @___qalloc()",
        "declare i64 @___qalloc()\ndeclare void @___reset(i64)",
    ).replace("%q = call i64 @___qalloc()", "%q = call i64 @___qalloc()\n  call void @___reset(i64 %q)")
    results = {}
    for events in (False, True):
        for include in (None, True, False):
            # Select the event route explicitly; it must forward the same policy
            # to its normalized schedule as the direct v3 route.
            builder = simulation(library, program)
            if events:
                from selene_simple_runtime_plugin import SimpleRuntimePlugin

                classical = (
                    pr.qis_engine()
                    .selene_runtime_plugin(str(library), SimpleRuntimePlugin().get_init_args())
                    .scheduled_event_batches()
                    .interface(pr.qis_helios_interface())
                )
                builder.classical(classical)
            options = {} if include is None else {"idle_before_preparation": include}
            profile = (
                pr.scheduled_event_idle_z(1, lambda _: PassThrough(), linear=0.2, **options)
                if events
                else pr.scheduled_idle_z(1, linear=0.2, **options)
            )
            results[events, include] = builder.noise(profile).run(64).to_dict()["measurement_0"]
        assert results[events, None] == results[events, True]
        assert results[events, False] != results[events, True]
    for include in (None, True, False):
        assert results[False, include] == results[True, include]
