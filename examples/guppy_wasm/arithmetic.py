"""Run Guppy WASM arithmetic and quantum feed-forward on either public backend.

    python examples/guppy_wasm/arithmetic.py --run pecos
    python examples/guppy_wasm/arithmetic.py --run selene

Install pecos-selene-wasm and build PECOS's QIS FFI for the PECOS path.
See docs/user-guide/guppy-wasm.md for setup and supported signatures.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_name("arithmetic.wat")


# These decorators are currently exposed through guppylang_internals.
# Guppy validates the declarations against the exports in this WAT module.
@wasm_module(str(WASM_PATH))
class Arithmetic:
    @wasm
    def add(self, a: int, b: int) -> int: ...

    @wasm
    def multiply(self, a: int, b: int) -> int: ...


@guppy
def addition() -> int:
    arithmetic = Arithmetic(nat(0))  # WASM context slot 0
    total = arithmetic.add(2, 3)
    arithmetic.discard()  # Release the linear context, even for stateless code.
    result("sum", total)
    return total


@guppy
def chained_arithmetic() -> int:
    arithmetic = Arithmetic(nat(0))
    total = arithmetic.add(2, 3)
    product = arithmetic.multiply(total, 4)
    arithmetic.discard()
    result("product", product)
    return product


@guppy
def conditional_x() -> bool:
    arithmetic = Arithmetic(nat(0))
    total = arithmetic.add(2, 3)
    arithmetic.discard()
    q = qubit()
    if total == 5:
        x(q)
    measured = measure(q).read()
    result("measured", measured)
    return measured


@guppy
def conditional_calls() -> int:
    """Allow the compiler to eliminate a provably unreachable WASM call."""
    arithmetic = Arithmetic(nat(0))
    total = arithmetic.add(2, 3)
    if total == 5:  # noqa: SIM108 - Keep the correlated branches visible in this optimizer example.
        value = arithmetic.multiply(2, 3) if total > 1 else arithmetic.multiply(8, 9)
    else:
        value = 0
    arithmetic.discard()
    result("value", value)
    return value


def run_pecos(function, shots: int = 4) -> dict:
    """Execute the Guppy program through PECOS sim()."""
    import pecos as pc

    return (
        pc.sim(pc.Guppy(function, wasm_modules={str(WASM_PATH): WASM_PATH.read_bytes()}))
        .qubits(1)
        .quantum(pc.state_vector())
        .seed(42)
        .run(shots)
        .to_dict()
    )


def run_selene(function, shots: int = 4) -> dict:
    """Execute through public Selene with the public WASM utility plugin."""
    import selene_sim
    from pecos_selene_wasm import build

    with TemporaryDirectory(prefix="pecos-guppy-wasm-") as build_dir:
        instance = build(function.compile(), {str(WASM_PATH): WASM_PATH.read_bytes()}, build_dir=Path(build_dir))
        results = {}
        for shot in instance.run_shots(
            simulator=selene_sim.Quest(),
            n_qubits=1,
            n_shots=shots,
            n_processes=1,
            random_seed=42,
        ):
            for label, value in shot:
                results.setdefault(label, []).append(value)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=("pecos", "selene"),
        default="selene",
        help="backend to execute all three examples (default: selene)",
    )
    args = parser.parse_args()
    examples = (
        ("addition", addition),
        ("chained_arithmetic", chained_arithmetic),
        ("conditional_x", conditional_x),
    )
    run = run_pecos if args.run == "pecos" else run_selene
    for name, function in examples:
        print(f"{args.run}: {name}: {run(function)}")


if __name__ == "__main__":
    main()
