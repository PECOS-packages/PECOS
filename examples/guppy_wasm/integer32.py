"""Guppy's H2 WASM mode: i32 exports with explicit public runtime width selection."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module
from guppylang_internals.std._internal.wasm import WasmPlatform

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH), wasm_platform=WasmPlatform.H2)
class Arithmetic:
    @wasm
    def add(self, a: int, b: int) -> int: ...

    @wasm(1)
    def indexed_add(self, a: int, b: int) -> int: ...

    @wasm
    def set(self, value: int) -> None: ...

    @wasm
    def get(self) -> int: ...


@guppy
def boundaries() -> None:
    arithmetic = Arithmetic(nat(0))
    result("sum", arithmetic.add(2, 3))
    result("overflow", arithmetic.add(2147483647, 1))
    result("underflow", arithmetic.add(-2147483648, -1))
    result("wide", arithmetic.add(4294967301, 0))
    result("negative_wide", arithmetic.add(-2147483649, 0))
    result("indexed", arithmetic.indexed_add(-8, 3))
    arithmetic.discard()


@guppy
def state_and_feedback() -> None:
    arithmetic = Arithmetic(nat(0))
    result("initial", arithmetic.get())
    arithmetic.set(-7)
    arithmetic.discard()
    arithmetic = Arithmetic(nat(0))
    value = arithmetic.get()
    arithmetic.discard()
    result("retained", value)
    q = qubit()
    if value == -7:
        x(q)
    result("measured", measure(q).read())


def main() -> None:
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    args = parser.parse_args()
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import WasmOptions, build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(boundaries, WasmOptions(modules, int_width=32), build_dir=Path(directory))
            print([dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4)])
    else:
        from pecos import Guppy, WasmOptions, sim, state_vector

        print(
            sim(Guppy(boundaries, wasm=WasmOptions(modules, int_width=32)))
            .qubits(1)
            .quantum(state_vector())
            .run(4)
            .to_dict(),
        )


if __name__ == "__main__":
    main()
