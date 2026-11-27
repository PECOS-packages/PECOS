"""Retain WASM state across shots on one worker with an explicit scope option."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH))
class Counter:
    @wasm
    def increment(self) -> int: ...
@guppy
def counter() -> None:
    module = Counter(nat(0))
    result("count", module.increment())
    module.discard()


def main() -> None:
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    parser.add_argument("--scope", choices=("shot", "worker"), default="worker")
    args = parser.parse_args()
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import WasmOptions, build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(counter, WasmOptions(modules, state_scope=args.scope), build_dir=Path(directory))
            print([dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4)])
    else:
        import pecos as pc

        print(
            pc.sim(pc.Guppy(counter, wasm=pc.WasmOptions(modules, state_scope=args.scope)))
            .qubits(1)
            .quantum(pc.state_vector())
            .workers(1)
            .run(4)
            .to_dict(),
        )


if __name__ == "__main__":
    main()
