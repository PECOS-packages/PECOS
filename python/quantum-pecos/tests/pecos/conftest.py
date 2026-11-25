"""Opt-in native scheduled-runtime fixtures shared by integration tests."""

import json
import os
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

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


def runtime_compile_inputs() -> tuple[str, Path, str]:
    """Locate the compiler, pinned headers and public runtime for native fixtures."""
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
    assert includes, "Pinned Selene headers were not found"
    # SSH and HTTPS checkouts of the same pinned revision may both be cached.
    assert len({(p / "selene/runtime.h").read_bytes() for p in includes}) == 1
    includes.sort()
    return shutil.which("cc"), includes[0], str(SimpleRuntimePlugin().library_file)


def build_timed_runtime(
    tmp_path: Path,
    gap: int,
    *,
    events: bool = False,
    shot_relative_clock: bool = False,
    initial_nanos: int = 0,
    coalesce_queued: bool = False,
    prep_gap_nanos: int | None = None,
) -> tuple[Path, int]:
    """Compile synthetic timing, optionally isolating an initial prep before an idle gap."""
    compiler, include, base_library = runtime_compile_inputs()
    library = tmp_path / ("timing_proxy.dylib" if platform.system() == "Darwin" else "timing_proxy.so")
    args = [
        compiler,
        "-std=c11",
        "-shared",
        "-fPIC",
        "-pthread",
        "-I",
        str(include),
        "-DBASE_LIBRARY=" + json.dumps(base_library),
        f"-DGAP_NANOS={gap}ULL",
        f"-DINITIAL_NANOS={initial_nanos}ULL",
        str(Path(__file__).with_name("fixtures") / "runtime_timing_proxy.c"),
        "-o",
        str(library),
    ]
    if coalesce_queued:
        args.append("-DCOALESCE_QUEUED")
    if prep_gap_nanos is not None:
        args.append(f"-DPREP_GAP_NANOS={prep_gap_nanos}ULL")
    if events:
        args.append("-DSYNTHETIC_EVENT")
    if shot_relative_clock:
        args.append("-DSHOT_RELATIVE_CLOCK")
    if platform.system() != "Darwin":
        args.append("-ldl")
    subprocess.run(args, check=True, capture_output=True, text=True)
    return library, gap


def run_descriptor_probe(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Force publication/read interleaving in the actual fixture's descriptor getter."""
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    compiler, include, base_library = runtime_compile_inputs()
    fixtures = Path(__file__).with_name("fixtures")
    source = (fixtures / "runtime_timing_proxy.c").read_text()
    # Insert only a scheduling hook; exercise the fixture's actual getter and callbacks.
    assert source.count("proxy = original;") == 1
    (tmp_path / "timing_proxy_instrumented.c").write_text(
        source.replace("proxy = original;", "proxy = original; publication_pause();"),
    )
    probe = tmp_path / "descriptor_probe.c"
    probe.write_bytes((fixtures / "runtime_timing_descriptor_probe.c").read_bytes())
    executable = tmp_path / "descriptor_probe"
    args = [
        compiler,
        "-std=gnu11",
        "-O2",
        "-pthread",
        "-I",
        str(include),
        "-DBASE_LIBRARY=" + json.dumps(base_library),
        "-DGAP_NANOS=0ULL",
        "-DINITIAL_NANOS=0ULL",
        str(probe),
        "-o",
        str(executable),
    ]
    if platform.system() != "Darwin":
        args.append("-ldl")
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=30)
    return subprocess.run(
        [str(executable), *SimpleRuntimePlugin().get_init_args()],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def scheduled_support():
    """Keep shared data and compilation out of test-module imports."""
    return SimpleNamespace(ramsey=RAMSEY, build_runtime=build_timed_runtime, descriptor_probe=run_descriptor_probe)
