"""Demonstrate a configurable timeout for a bounded, CPU-heavy WASM call.

    python examples/guppy_wasm/long_call.py --backend selene --timeout 10
    python examples/guppy_wasm/long_call.py --backend pecos --timeout 10

The duration depends on the machine; --timeout 1 can reproduce the default
execution limit. This is an intentional stress example, not a CI timing test.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from guppylang import guppy
from guppylang.std.builtins import comptime, nat, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_suffix(".wat")
ITERATIONS = 2_000_000_000


@wasm_module(str(WASM_PATH))
class Work:
    @wasm
    def work(self, iterations: int) -> int: ...


@guppy
def long_call() -> None:
    work = Work(nat(0))
    value = work.work(comptime(ITERATIONS))
    work.discard()
    result("value", value)


def expected_value(iterations: int) -> int:
    """Evaluate the affine recurrence by composition in O(log iterations)."""
    if iterations < 0:
        msg = "iterations must be nonnegative"
        raise ValueError(msg)
    multiplier, offset = 6_364_136_223_846_793_005, 1
    value = 0
    mask = (1 << 64) - 1
    while iterations:
        if iterations & 1:
            value = (multiplier * value + offset) & mask
        multiplier, offset = (multiplier * multiplier) & mask, (offset * (multiplier + 1)) & mask
        iterations >>= 1
    return value if value < 1 << 63 else value - (1 << 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import WasmOptions, build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(long_call, WasmOptions(modules, timeout_seconds=args.timeout), build_dir=Path(directory))
            shots = [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=1, timeout=60.0)]
            observed = shots[0]["value"]
    else:
        import pecos as pc

        observed = (
            pc.sim(pc.Guppy(long_call, wasm=pc.WasmOptions(modules, timeout_seconds=args.timeout)))
            .qubits(1)
            .quantum(pc.state_vector())
            .run(1)
            .to_dict()["value"][0]
        )
    if observed != expected_value(ITERATIONS):
        msg = "WASM result does not match the independent recurrence oracle"
        raise RuntimeError(msg)
    print({"value": observed})


if __name__ == "__main__":
    main()
