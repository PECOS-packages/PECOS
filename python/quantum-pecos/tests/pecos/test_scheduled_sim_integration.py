"""Production sim() transport with explicitly synthetic nonzero native timestamps."""

import json
import math
import os
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

RAMSEY = """
declare i64 @___qalloc()
declare void @___qfree(i64)
declare i64 @___lazy_measure(i64)
declare void @___rxy(i64, double, double)
declare void @setup(i64)
declare i64 @teardown()
define i64 @qmain(i64 %arg) #0 {
  call void @setup(i64 %arg)
  %q = call i64 @___qalloc()
  call void @___rxy(i64 %q, double 0x3FF921FB54442D18, double 0x3FF921FB54442D18)
  call void @___rxy(i64 %q, double 0xBFF921FB54442D18, double 0x3FF921FB54442D18)
  %r = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)
  %end = call i64 @teardown()
  ret i64 %end
}
attributes #0 = { "EntryPoint" }
"""


@pytest.fixture(params=[0, 1_000_000_000])
def timed_runtime(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, int]:
    """Add explicitly synthetic timing to the public runtime's native callbacks."""
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    if platform.system() == "Windows":
        pytest.skip("Synthetic dlopen proxy is POSIX-only")
    assert shutil.which("cc") is not None, "Nonzero timing integration requires a C compiler"
    repo = Path(__file__).resolve().parents[4]
    lock = tomllib.loads((repo / "Cargo.lock").read_text())
    source = next(p["source"] for p in lock["package"] if p["name"] == "selene-core")
    revision = source.rsplit("#", 1)[1]
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    includes = list((cargo_home / "git/checkouts").glob(f"selene-*/{revision[:7]}/selene-core/c/include"))
    assert len(includes) == 1, f"Expected exactly one pinned Selene header directory, got {includes}"
    library = tmp_path / ("timing_proxy.dylib" if platform.system() == "Darwin" else "timing_proxy.so")
    args = [
        shutil.which("cc"),
        "-std=c11",
        "-shared",
        "-fPIC",
        "-I",
        str(includes[0]),
        "-DBASE_LIBRARY=" + json.dumps(str(SimpleRuntimePlugin().library_file)),
        f"-DGAP_NANOS={request.param}ULL",
        str(Path(__file__).with_name("fixtures") / "runtime_timing_proxy.c"),
        "-o",
        str(library),
    ]
    if platform.system() != "Darwin":
        args.append("-ldl")
    subprocess.run(args, check=True, capture_output=True, text=True)
    return library, request.param


def simulation(library: Path, program: str = RAMSEY):
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


def test_sim_ramsey_responds_to_synthetic_timing(timed_runtime: tuple[Path, int]) -> None:
    import pecos_rslib

    library, gap = timed_runtime
    result = simulation(library).noise(pecos_rslib.scheduled_idle_z(1, coherent=math.pi)).run(3).to_dict()
    assert result["measurement_0"] == [int(gap != 0)] * 3


def test_sim_rejects_scheduled_transport_without_capability(timed_runtime: tuple[Path, int]) -> None:
    with pytest.raises(RuntimeError, match="scheduled idle capability required"):
        simulation(timed_runtime[0]).run(1)


def test_sim_feedback_and_worker_shots(timed_runtime: tuple[Path, int]) -> None:
    import pecos_rslib

    program = RAMSEY.replace(
        "declare void @setup",
        "declare i1 @___read_future_bool(i64)\ndeclare void @setup",
    ).replace(
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
    library, gap = timed_runtime
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
def test_public_native_route_with_zero_timing(runtime_name: str) -> None:
    """Zero timing checks scheduling/terminal drain, not idle-noise sensitivity."""
    import pecos
    import pecos_rslib as pr

    classical = pr.qis_engine().selene_runtime(runtime_name).scheduled_batches().interface(pr.qis_helios_interface())
    result = (
        pecos.sim(pecos.Qis(RAMSEY))
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
