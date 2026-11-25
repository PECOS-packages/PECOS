"""Native C adapters through Python sim(), runtime feedback and MonteCarlo."""

import ctypes
import math
import platform
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def native_plugin(tmp_path):
    if platform.system() == "Windows":
        pytest.skip("Synthetic shared-library fixture currently requires POSIX")
    compiler = shutil.which("cc")
    assert compiler is not None
    root = Path(__file__).resolve().parents[4]
    library = tmp_path / ("adapter.dylib" if platform.system() == "Darwin" else "adapter.so")
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "crates/pecos-foreign/include"),
            str(root / "crates/pecos-foreign/tests/fixtures/scheduled_adapter.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    monitor = ctypes.CDLL(str(library))
    monitor.probe_count.argtypes = [ctypes.c_uint32]
    monitor.probe_count.restype = ctypes.c_uint64
    monitor.probe_context.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    monitor.probe_context.restype = ctypes.c_uint64
    monitor.probe_fail_next_create.argtypes = []
    monitor.probe_fail_next_create.restype = None
    return library, monitor


@pytest.fixture
def runtime(tmp_path, scheduled_support):
    return scheduled_support.build_runtime(tmp_path, 0, events=True)[0], scheduled_support.ramsey


def builder(runtime, noise, program=None, backend="state_vector", *, scoped=False):
    import pecos
    import pecos_rslib as pr
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    library, default = runtime
    program = default if program is None else program
    classical = (
        pr.qis_engine()
        .selene_runtime_plugin(
            str(library),
            SimpleRuntimePlugin().get_init_args(),
            custom_event_policy="capture",
        )
        .scheduled_event_batches()
        .scoped_execution(scoped)
        .interface(pr.qis_helios_interface())
    )
    return (
        pecos.sim(pecos.Qis(program) if isinstance(program, str) else program)
        .classical(classical)
        .qubits(1)
        .quantum(getattr(pr, backend)())
        .noise(noise)
        .seed(42)
    )


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("backend", ["state_vector", "stabilizer"])
@pytest.mark.parametrize("worker_context", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
def test_repeated_runs_workers_and_contexts(runtime, native_plugin, workers, backend, worker_context, scoped):
    import pecos_rslib as pr

    library, monitor = native_plugin
    noise = pr.scheduled_native_adapter(pr.scheduled_idle_noise(1), library, config=b"\x12")
    assert isinstance(noise, pr.ScheduledEventIdleNoise)
    built = (
        builder(runtime, noise, backend=backend, scoped=scoped)
        .workers(workers)
        .python_worker_context(worker_context)
        .build()
    )
    assert type(built).__name__ == "QisControlSimulation"
    assert built.run(2).to_dict()["measurement_0"] == [1, 1]
    alternate_workers = 3 - workers
    assert built.run_with_workers(4, alternate_workers).to_dict()["measurement_0"] == [1] * 4
    built.reset()
    assert built.run_with_workers(2, workers).to_dict()["measurement_0"] == [1, 1]
    assert monitor.probe_count(0) == monitor.probe_count(1) == 8
    contexts = [tuple(monitor.probe_context(i, j) for j in range(3)) for i in range(8)]
    assert len(set(contexts)) == 8
    assert {c[1] for c in contexts[:2]} == set(range(workers))
    assert {c[1] for c in contexts[2:6]} == set(range(alternate_workers))
    assert {c[1] for c in contexts[6:]} == set(range(workers))
    assert len({c[0] for c in contexts}) == 3


@pytest.mark.parametrize("kind", ["z", "idle", "local", "crosstalk"])
def test_checked_profiles_are_preserved(runtime, native_plugin, kind):
    import pecos_rslib as pr

    library, _ = native_plugin
    profile = pr.scheduled_idle_noise(1)
    if kind == "z":
        profile = pr.scheduled_idle_z(1)
    elif kind == "local":
        profile = pr.scheduled_local_noise(profile, meas1=1.0)
    elif kind == "crosstalk":
        profile = pr.scheduled_measurement_crosstalk(profile, {"0->1": 1.0, "1->0": 1.0})
    noise = pr.scheduled_native_adapter(profile, str(library), config=b"\x12")
    assert builder(runtime, noise).run(2).to_dict()["measurement_0"] == [int(kind != "local")] * 2


def test_loader_argument_errors_precede_execution(tmp_path):
    import pecos_rslib as pr

    missing = tmp_path / "missing-library"
    with pytest.raises(TypeError, match="checked scheduled"):
        pr.scheduled_native_adapter(object(), missing)
    profile = pr.scheduled_idle_noise(1)
    with pytest.raises(ValueError, match="configuration limit"):
        pr.scheduled_native_adapter(profile, missing, config=b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="native adapter load"):
        pr.scheduled_native_adapter(profile, missing)
    with pytest.raises(TypeError):
        pr.scheduled_native_adapter(profile, missing, config=bytearray(b"x"))


@pytest.mark.parametrize("config", [None, b""])
def test_empty_config_is_forwarded_and_factory_errors_surface(runtime, native_plugin, config):
    import pecos_rslib as pr

    library, monitor = native_plugin
    noise = pr.scheduled_native_adapter(pr.scheduled_idle_noise(1), library, config=config)
    assert monitor.probe_count(0) == 0
    with pytest.raises(RuntimeError, match="native adapter create status 15"):
        builder(runtime, noise).run(1)


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [(mode, RuntimeError, "native adapter") for mode in [1, 2, 3, 7, 8, 9, 17]]
    + [(10, ValueError, "adapter changed measurement order, kind or targets")],
)
@pytest.mark.parametrize("scoped", [False, True])
def test_native_errors_reach_python(runtime, native_plugin, mode, error, message, scoped):
    import pecos_rslib as pr

    library, _ = native_plugin
    noise = pr.scheduled_native_adapter(pr.scheduled_idle_noise(1), library, config=bytes([mode]))
    with pytest.raises(error, match=message):
        builder(runtime, noise, scoped=scoped).run(1)


@pytest.mark.parametrize("scoped", [False, True])
def test_failed_run_uses_a_fresh_session(runtime, native_plugin, scoped):
    import pecos_rslib as pr

    library, monitor = native_plugin
    noise = pr.scheduled_native_adapter(pr.scheduled_idle_noise(1), library, config=b"\x13")
    built = builder(runtime, noise, scoped=scoped).workers(1).build()
    monitor.probe_fail_next_create()
    with pytest.raises(RuntimeError, match="translate status 62"):
        built.run(1)
    assert monitor.probe_count(0) == monitor.probe_count(1) == 1
    assert built.run(2).to_dict()["measurement_0"] == [1, 1]
    built.reset()
    assert built.run(1).to_dict()["measurement_0"] == [1]
    assert monitor.probe_count(0) == monitor.probe_count(1) == 4


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("backend", ["state_vector", "stabilizer"])
@pytest.mark.parametrize("scoped", [False, True])
def test_guppy_branch_receives_user_measurement_after_crosstalk(runtime, native_plugin, probability, backend, scoped):
    import pecos_rslib as pr
    from guppylang import guppy
    from guppylang.std.angles import angle
    from guppylang.std.builtins import owned, result
    from guppylang.std.quantum import measure, qubit, ry, x

    @guppy.declare
    def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...

    @guppy
    def feedback() -> None:
        q = qubit()
        ry(q, angle(0.5))
        q = pecos_qis_runtime_barrier_qubit_hugr(q)
        ry(q, angle(-0.5))
        first = measure(q).read()
        result("first", first)
        q2 = qubit()
        if first:
            x(q2)
        result("feedback", measure(q2).read())

    library, _ = native_plugin
    profile = pr.scheduled_measurement_crosstalk(
        pr.scheduled_idle_noise(1),
        {"0->L": 1.0, "1->L": 1.0},
        local_probability=probability,
    )
    noise = pr.scheduled_native_adapter(profile, library, config=b"\x0f")
    values = builder(runtime, noise, feedback, backend, scoped=scoped).workers(2).run(4).to_dict()
    assert values["first"] == values["feedback"] == [int(probability)] * 4


def test_unsupported_routes_reject_native_wrapper(runtime, native_plugin):
    import pecos
    import pecos_rslib as pr

    library, _ = native_plugin
    noise = pr.scheduled_native_adapter(pr.scheduled_idle_noise(1), library, config=b"\x12")
    with pytest.raises(TypeError, match="requires QIS/HUGR"):
        pecos.sim(pecos.Qasm("OPENQASM 2.0; qreg q[1];")).noise(noise).run(1)
    with pytest.raises(ValueError, match="neo"):
        builder(runtime, noise).stack("neo")
    with pytest.raises(TypeError, match="without operation tracing"):
        builder(runtime, noise).capture_operation_trace()


@pytest.mark.parametrize(("coherent", "expected"), [(0.0, 1), (math.pi, 0)])
def test_runtime_idle_interval_and_noise_profile_are_preserved(
    tmp_path,
    scheduled_support,
    native_plugin,
    coherent,
    expected,
):
    import pecos_rslib as pr

    library, _ = native_plugin
    runtime_library, _ = scheduled_support.build_runtime(tmp_path, 1_000_000_000, events=True)
    profile = pr.scheduled_idle_noise(1, coherent=coherent)
    noise = pr.scheduled_native_adapter(profile, library, config=b"\x12")
    result = builder((runtime_library, scheduled_support.ramsey), noise).run(3).to_dict()
    assert result["measurement_0"] == [expected] * 3
