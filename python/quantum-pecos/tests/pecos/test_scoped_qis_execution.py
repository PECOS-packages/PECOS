"""Explicit scoped QIS execution through Python sim() and live runtime feedback."""

import math
from types import SimpleNamespace

import pytest


def feedback_program(ramsey):
    """Force a result and flip the same qubit only when that result is one."""
    return (
        ramsey.replace(
            "declare void @setup",
            "declare void @___reset(i64)\ndeclare i1 @___read_future_bool(i64)\ndeclare void @setup",
        )
        .replace(
            "  %q = call i64 @___qalloc()",
            "  %q = call i64 @___qalloc()\n  call void @___reset(i64 %q)",
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


def classical_builder(wrapped, scoped, library=None):
    import pecos
    import pecos_rslib as pr
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    classical = pecos.qis_engine() if wrapped else pr.qis_engine()
    if scoped is not None:
        classical = classical.scoped_execution(enabled=True)
        if not scoped:
            classical = classical.scoped_execution(enabled=False)
    if library is None:
        classical = classical.selene_runtime()
    elif wrapped:
        classical = classical.selene_runtime(
            SimpleNamespace(library_file=library, get_init_args=SimpleRuntimePlugin().get_init_args),
        )
    else:
        classical = classical.selene_runtime_plugin(str(library), SimpleRuntimePlugin().get_init_args())
    return classical.interface(pr.qis_helios_interface())


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("backend", ["state_vector", "stabilizer"])
def test_scoped_feedback_reuse_and_workers(wrapped, backend, scheduled_support):
    """Both builder APIs preserve feedback, cloned configuration and seeded shots."""
    import pecos
    import pecos_rslib as pr

    # An H-equivalent rotation before the first read forces both branch outcomes.
    program = feedback_program(scheduled_support.ramsey).replace(
        "  call void @___rxy(i64 %q, double 0xBFF921FB54442D18, double 0x3FF921FB54442D18)\n",
        "",
    )
    runs = []
    for scoped in (None, False, True):
        built = (
            pecos.sim(pecos.Qis(program))
            .classical(classical_builder(wrapped, scoped))
            .quantum(getattr(pr, backend)())
            .qubits(1)
            .seed(42)
            .workers(2)
            .build()
        )
        results = []
        for workers in (2, 1):
            result = built.run_with_workers(32, workers).to_dict()
            assert set(result["measurement_0"]) == {0, 1}
            assert result["measurement_1"] == [0] * 32
            results.append(result)
        built.reset()
        results.append(built.run(8).to_dict())
        runs.append(results)
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("gap", [0, 1_000_000_000])
def test_scoped_preserves_runtime_idle_and_noise_feedback(wrapped, gap, tmp_path, scheduled_support):
    """Runtime timing causes a known phase flip before the forced read."""
    import pecos
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, gap)
    results = []
    for scoped in (False, True):
        classical = classical_builder(wrapped, scoped, library).scheduled_batches()
        result = (
            pecos.sim(pecos.Qis(feedback_program(scheduled_support.ramsey)))
            .classical(classical)
            .quantum(pr.state_vector())
            .noise(pr.scheduled_idle_z(1, coherent=math.pi))
            .qubits(1)
            .seed(42)
            .workers(2)
            .run(6)
            .to_dict()
        )
        assert result["measurement_0"] == [int(gap != 0)] * 6
        assert result["measurement_1"] == [0] * 6
        results.append(result)
    assert results[0] == results[1]
