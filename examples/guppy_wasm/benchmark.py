"""Measure public Guppy/WASM compilation and repeated shot execution.

Example: python examples/guppy_wasm/benchmark.py --backend both --calls 0 1 10 100

Calls are generated as straight-line Guppy to exclude loop-control overhead.
Timings include module instantiation, shot setup, result collection and backend
costs; amortized seconds per call are not isolated WASM boundary latency.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

WASM_PATH = Path(__file__).with_name("arithmetic.wat")


def program_source(calls: int) -> str:
    """Make a dependency chain whose expected result is independently known."""
    source = f"""
from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

@wasm_module({str(WASM_PATH)!r})
class Arithmetic:
    @wasm
    def add(self, left: int, right: int) -> int: ...

@guppy
def main() -> None:
    value = 0
"""
    if calls:
        source += "    arithmetic = Arithmetic(nat(0))\n"
        source += "    value = arithmetic.add(value, 1)\n" * calls
        source += "    arithmetic.discard()\n"
    return source + '    result("value", value)\n'


def measure(backend: str, calls: int, shots: int, repeats: int, directory: Path) -> dict:
    source = directory / f"benchmark_{backend}_{calls}.py"
    source.write_text(program_source(calls))
    spec = importlib.util.spec_from_file_location(source.stem, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    package = module.main.compile()
    guppy_seconds = time.perf_counter() - start
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}

    if backend == "selene":
        import selene_sim
        from pecos_selene_wasm import build
    else:
        import pecos as pc

    start = time.perf_counter()
    if backend == "selene":
        options = {"build_dir": directory / f"build_{calls}"}
        runner = build(package, modules, **options) if calls else selene_sim.build(package, **options)

        def run() -> list[int]:
            return [
                dict(shot)["value"]
                for shot in runner.run_shots(
                    simulator=selene_sim.Quest(),
                    n_qubits=1,
                    n_shots=shots,
                    n_processes=1,
                    random_seed=42,
                )
            ]

    else:
        program = pc.Hugr(package.to_bytes(), wasm_modules=modules if calls else None)
        runner = pc.sim(program).qubits(1).quantum(pc.state_vector()).seed(42).workers(1).build()

        def run() -> list[int]:
            return runner.run(shots).to_dict()["value"]

    prepare_seconds = time.perf_counter() - start

    def checked_run() -> None:
        values = run()
        if values != [calls] * shots:
            msg = f"{backend} returned incorrect results for {calls} calls: {values}"
            raise RuntimeError(msg)

    start = time.perf_counter()
    checked_run()
    first_run_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        checked_run()
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {
        "backend": backend,
        "calls_per_shot": calls,
        "shots": shots,
        "guppy_compile_seconds": guppy_seconds,
        "backend_prepare_seconds": prepare_seconds,
        "first_run_seconds": first_run_seconds,
        "repeat_seconds": samples,
        "median_seconds_per_shot": median / shots,
        "amortized_seconds_per_call": median / (shots * calls) if calls else None,
    }


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        message = "must be positive"
        raise argparse.ArgumentTypeError(message)
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        message = "must be nonnegative"
        raise argparse.ArgumentTypeError(message)
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos", "both"), default="selene")
    parser.add_argument("--calls", type=nonnegative, nargs="+", default=[0, 1, 10, 100])
    parser.add_argument("--shots", type=positive, default=10)
    parser.add_argument("--repeats", type=positive, default=3)
    args = parser.parse_args()
    versions = {}
    for name in ("guppylang", "selene-sim", "selene-hugr-qis-compiler", "hugr", "pecos-rslib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    backends = ("selene", "pecos") if args.backend == "both" else (args.backend,)
    with TemporaryDirectory(prefix="pecos-wasm-benchmark-") as temporary:
        records = [
            measure(backend, calls, args.shots, args.repeats, Path(temporary))
            for backend in backends
            for calls in args.calls
        ]
    print(
        json.dumps(
            {"platform": platform.platform(), "python": sys.version, "versions": versions, "results": records},
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
