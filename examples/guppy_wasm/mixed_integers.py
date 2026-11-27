"""Call i32 and i64 WASM exports in one program using automatic integer widths."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module
from guppylang_internals.std._internal.wasm import WasmPlatform

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH), wasm_platform=WasmPlatform.H2)
class Small:
    @wasm(0)
    def indexed_add32(self, a: int, b: int) -> int: ...
    @wasm
    def add32(self, a: int, b: int) -> int: ...
@wasm_module(str(WASM_PATH))
class Large:
    @wasm(1)
    def indexed_add64(self, a: int, b: int) -> int: ...
    @wasm
    def add64(self, a: int, b: int) -> int: ...
@guppy
def mixed() -> None:
    small = Small(nat(0))
    result("small", small.add32(2147483647, 1))
    result("small_indexed", small.indexed_add32(4294967301, -8))
    small.discard()
    large = Large(nat(1))
    result("large", large.add64(4294967301, 7))
    result("large_indexed", large.indexed_add64(4294967301, -8))
    large.discard()


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
            runner = build(mixed, WasmOptions(modules, int_width="auto"), build_dir=Path(directory))
            print([dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4)])
    else:
        from pecos import Guppy, WasmOptions, sim, state_vector

        print(
            sim(Guppy(mixed, wasm=WasmOptions(modules, int_width="auto")))
            .qubits(1)
            .quantum(state_vector())
            .run(4)
            .to_dict(),
        )


if __name__ == "__main__":
    main()
