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


def build_timed_runtime(
    tmp_path: Path,
    gap: int,
    *,
    events: bool = False,
    initial_nanos: int = 0,
) -> tuple[Path, int]:
    """Compile the public timing proxy, optionally emitting an invented event."""
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
    library = tmp_path / ("timing_proxy.dylib" if platform.system() == "Darwin" else "timing_proxy.so")
    args = [
        shutil.which("cc"),
        "-std=c11",
        "-shared",
        "-fPIC",
        "-I",
        str(includes[0]),
        "-DBASE_LIBRARY=" + json.dumps(str(SimpleRuntimePlugin().library_file)),
        f"-DGAP_NANOS={gap}ULL",
        f"-DINITIAL_NANOS={initial_nanos}ULL",
        str(Path(__file__).with_name("fixtures") / "runtime_timing_proxy.c"),
        "-o",
        str(library),
    ]
    if events:
        args.append("-DSYNTHETIC_EVENT")
    if platform.system() != "Darwin":
        args.append("-ldl")
    subprocess.run(args, check=True, capture_output=True, text=True)
    return library, gap


@pytest.fixture
def scheduled_support():
    """Keep shared data and compilation out of test-module imports."""
    return SimpleNamespace(ramsey=RAMSEY, build_runtime=build_timed_runtime)
