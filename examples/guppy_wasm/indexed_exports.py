"""Call WASM exports by their zero-based position on either public backend.

    python examples/guppy_wasm/indexed_exports.py --backend selene
    python examples/guppy_wasm/indexed_exports.py --backend pecos

Export positions include non-functions. See the explicit export order in the
WAT: memory is position 0, sum is 1, and product is 2, regardless of where the
function bodies are defined. Named lookup is preferable when exports may move.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH))
class Indexed:
    @wasm(1)
    def add(self, left: int, right: int) -> int: ...

    @wasm(2)
    def multiply(self, left: int, right: int) -> int: ...

    @wasm(3)
    def offset(self, value: float, amount: int) -> float: ...

    @wasm(4)
    def reset(self) -> None: ...

    @wasm(5)
    def next_value(self) -> int: ...

    @wasm
    def sum(self, left: int, right: int) -> int: ...


@guppy
def indexed_arithmetic() -> None:
    module = Indexed(nat(0))
    total = module.add(-3, 7)
    product = module.multiply(total, 5)
    named = module.sum(product, 2)
    module.discard()
    result("sum", total)
    result("product", product)
    result("named", named)


@guppy
def indexed_feedback() -> None:
    module = Indexed(nat(0))
    value = module.offset(-2.5, 4)
    module.discard()
    result("value", value)
    q = qubit()
    if value == 1.5:
        x(q)
    result("measured", measure(q).read())


@guppy
def indexed_state() -> None:
    module = Indexed(nat(0))
    value = 0
    for _ in range(3):
        value = module.next_value()
    module.reset()
    first = module.next_value()
    module.discard()
    result("three", value)
    result("reset", first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    args = parser.parse_args()
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    for name, function in (
        ("arithmetic", indexed_arithmetic),
        ("feedback", indexed_feedback),
        ("state", indexed_state),
    ):
        if args.backend == "pecos":
            import pecos as pc

            data = (
                pc.sim(pc.Guppy(function, wasm_modules=modules)).qubits(1).quantum(pc.state_vector()).run(4).to_dict()
            )
        else:
            from pecos_selene_wasm import build
            from selene_sim import Quest

            with TemporaryDirectory(prefix="pecos-indexed-wasm-") as directory:
                runner = build(function, modules, build_dir=Path(directory))
                data = [
                    dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=1)
                ]
        print(f"{args.backend}: {name}: {data}")


if __name__ == "__main__":
    main()
