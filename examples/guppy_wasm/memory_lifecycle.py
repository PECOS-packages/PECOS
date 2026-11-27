"""WASM startup, optional initialization, memory growth and indirect calls."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH))
class Memory:
    @wasm
    def init(self) -> None: ...
    @wasm
    def get(self) -> int: ...
    @wasm
    def increment(self) -> int: ...
    @wasm
    def grow(self) -> int: ...
    @wasm
    def pages(self) -> int: ...
    @wasm
    def read_new(self) -> int: ...
    @wasm
    def write_new(self, value: int) -> None: ...
    @wasm
    def dispatch(self, left: int, right: int) -> int: ...


@guppy
def memory_operations() -> None:
    module = Memory(nat(0))
    result("initialized", module.get())
    result("incremented", module.increment())
    result("old_pages", module.grow())
    result("pages", module.pages())
    result("zero", module.read_new())
    module.write_new(-7)
    result("stored", module.read_new())
    result("indirect", module.dispatch(5, 7))
    result("growth_refused", module.grow())
    module.discard()


@guppy
def explicit_initialization() -> None:
    module = Memory(nat(0))
    result("before", module.get())
    module.init()
    result("after", module.get())
    module.discard()


def main() -> None:
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    parser.add_argument("--auto-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explicit-init", action="store_true")
    args = parser.parse_args()
    program = explicit_initialization if args.explicit_init else memory_operations
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import WasmOptions, build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(program, WasmOptions(modules, auto_init=args.auto_init), build_dir=Path(directory))
            print([dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4)])
    else:
        import pecos as pc

        print(
            pc.sim(pc.Guppy(program, wasm=pc.WasmOptions(modules, auto_init=args.auto_init)))
            .qubits(1)
            .quantum(pc.state_vector())
            .run(4)
            .to_dict(),
        )


if __name__ == "__main__":
    main()
