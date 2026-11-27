"""Correct one bit flip per round using quantum syndromes and a WASM lookup.

This small example uses the three-bit repetition code, not a general decoder.
Run from the checkout with either ``--backend selene`` or ``--backend pecos``.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import cx, measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH))
class Decoder:
    @wasm
    def decode(self, syndrome: int) -> int: ...

    @wasm
    def calls(self) -> int: ...


@guppy
def correct_each_position() -> None:
    decoder = Decoder(nat(0))
    q0, q1, q2 = qubit(), qubit(), qubit()
    correction_sum = 0
    syndrome_trace = 0
    correction_trace = 0
    for position in range(4):
        # Cover a flip at each position, followed by a round with no error.
        if position == 0:
            x(q0)
        elif position == 1:
            x(q1)
        elif position == 2:
            x(q2)
        a0, a1 = qubit(), qubit()
        cx(q0, a0)
        cx(q1, a0)
        cx(q1, a1)
        cx(q2, a1)
        syndrome = 0
        if measure(a0).read():
            syndrome += 1
        if measure(a1).read():
            syndrome += 2
        correction = decoder.decode(syndrome)
        correction_sum += correction
        syndrome_trace = 4 * syndrome_trace + syndrome
        correction_trace = 8 * correction_trace + correction
        if correction & 1 != 0:
            x(q0)
        if correction & 2 != 0:
            x(q1)
        if correction & 4 != 0:
            x(q2)
    calls = decoder.calls()
    decoder.discard()
    result("calls", calls)
    result("correction_sum", correction_sum)
    result("syndrome_trace", syndrome_trace)
    result("correction_trace", correction_trace)
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())
    result("q2", measure(q2).read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import WasmOptions, build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(correct_each_position, WasmOptions(modules), build_dir=Path(directory))
            shots = [
                dict(shot)
                for shot in runner.run_shots(
                    simulator=Quest(),
                    n_qubits=5,
                    n_shots=args.shots,
                    n_processes=args.workers,
                    random_seed=42,
                )
            ]
        print(shots)
    else:
        import pecos as pc

        results = (
            pc.sim(pc.Guppy(correct_each_position, wasm=pc.WasmOptions(modules)))
            .qubits(5)
            .quantum(pc.state_vector())
            .seed(42)
            .workers(args.workers)
            .run(args.shots)
            .to_dict()
        )
        print(results)


if __name__ == "__main__":
    main()
