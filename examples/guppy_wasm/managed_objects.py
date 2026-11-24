"""Use WASM-managed structs and arrays behind ordinary scalar Guppy calls."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

# Guppy validates only this scalar interface. The actual runtime payload uses GC.
WASM_PATH = Path(__file__).with_suffix(".interface.wat")
RUNTIME_WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH))
class Managed:
    @wasm
    def get(self) -> int: ...

    @wasm
    def increment(self) -> int: ...

    @wasm
    def sum(self, left: int, right: int) -> int: ...


@guppy
def managed_state() -> None:
    module = Managed(nat(0))
    result("initial", module.get())
    result("next", module.increment())
    result("sum", module.sum(-7, 12))
    module.discard()
    module = Managed(nat(0))
    result("retained", module.get())
    module.discard()
    independent = Managed(nat(1))
    result("independent", independent.get())
    independent.discard()


def main() -> None:
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    args = parser.parse_args()
    modules = {str(WASM_PATH): RUNTIME_WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(managed_state, modules, build_dir=Path(directory))
            print([dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4)])
    else:
        import pecos as pc

        print(
            pc.sim(pc.Guppy(managed_state, wasm_modules=modules)).qubits(1).quantum(pc.state_vector()).run(4).to_dict(),
        )


if __name__ == "__main__":
    main()
