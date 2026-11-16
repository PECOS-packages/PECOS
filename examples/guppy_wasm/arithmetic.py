"""Compile Guppy WASM calls and run the arithmetic module directly in PECOS.

Run from the repository root:
    uv run --no-sync python examples/guppy_wasm/arithmetic.py

The default run compiles Guppy to HUGR and separately executes the WASM module.
Use --run pecos or --run selene to probe hybrid execution through the public
compiler paths. See docs/user-guide/guppy-wasm.md for tested versions and the
remaining public compiler/runtime integration work.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pecos as pc
import selene_sim
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


def run_pecos(function, shots: int = 4) -> dict:
    """Probe Guppy WASM execution through PECOS (requires runtime support)."""
    return pc.sim(pc.Guppy(function)).qubits(1).quantum(pc.state_vector()).seed(42).run(shots).to_dict()


def run_selene(function, shots: int = 4) -> dict:
    """Probe standalone Selene execution (requires compiler/runtime support)."""
    with TemporaryDirectory(prefix="pecos-guppy-wasm-") as build_dir:
        instance = selene_sim.build(function.compile(), build_dir=Path(build_dir))
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
        help="probe Guppy execution through the public compiler/runtime paths",
    )
    args = parser.parse_args()
    examples = (
        ("addition", addition),
        ("chained_arithmetic", chained_arithmetic),
        ("conditional_x", conditional_x),
    )
    # Guppy(...) compiles and retains HUGR; it does not execute the program.
    for name, function in examples:
        program = pc.Guppy(function)
        print(f"Compiled {name} to HUGR ({len(program.hugr_bytes)} bytes)")

    arithmetic = pc.WasmForeignObject.from_file(WASM_PATH)
    arithmetic.init()
    try:
        total = arithmetic.exec("add", [2, 3])
        product = arithmetic.exec("multiply", [total, 4])
        print(f"Direct WASM execution: 2 + 3 = {total}; (2 + 3) * 4 = {product}")
    finally:
        arithmetic.teardown()
    print("Default checks cover HUGR compilation and direct WASM execution separately.")
    if args.run:
        run = run_pecos if args.run == "pecos" else run_selene
        for name, function in examples:
            print(f"{args.run}: {name}: {run(function)}")


if __name__ == "__main__":
    main()
