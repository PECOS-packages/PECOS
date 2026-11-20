"""Mixed float/integer WASM signatures and floating-point quantum feedback."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_name("floating_point.wat")


@wasm_module(str(WASM_PATH))
class FloatingPoint:
    @wasm
    def affine(self, value: float, scale: int, offset: float) -> float: ...

    @wasm
    def remember(self, value: float) -> None: ...

    @wasm
    def recall(self) -> float: ...

    @wasm
    def echo(self, value: float) -> float: ...

    @wasm
    def truncate(self, value: float) -> int: ...


@guppy
def mixed_arithmetic() -> None:
    arithmetic = FloatingPoint(nat(0))
    value = arithmetic.affine(1.5, -3, 0.25)
    arithmetic.remember(value)
    result("value", arithmetic.recall())
    result("integer", arithmetic.truncate(value))
    arithmetic.discard()


@guppy
def float_feedback() -> None:
    arithmetic = FloatingPoint(nat(0))
    value = arithmetic.affine(1.5, 3, -0.25)
    arithmetic.discard()
    q = qubit()
    if value > 4.0:
        x(q)
    result("measured", measure(q).read())


@guppy
def float_state() -> None:
    arithmetic = FloatingPoint(nat(0))
    result("initial", arithmetic.recall())
    arithmetic.remember(-2.5)
    arithmetic.discard()
    arithmetic = FloatingPoint(nat(0))
    result("retained", arithmetic.recall())
    arithmetic.discard()


@guppy
def float_loop() -> None:
    """Carry a float through mixed-signature WASM calls."""
    arithmetic = FloatingPoint(nat(0))
    value = 0.25
    for _ in range(3):
        value = arithmetic.affine(value, 2, 0.25)
    arithmetic.discard()
    result("value", value)
    q = qubit()
    if value == 3.75:
        x(q)
    result("measured", measure(q).read())


def main() -> None:
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    backend = parser.parse_args().backend
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    for function in (mixed_arithmetic, float_feedback, float_state, float_loop):
        if backend == "pecos":
            import pecos as pc

            data = (
                pc.sim(pc.Guppy(function, wasm_modules=modules)).qubits(1).quantum(pc.state_vector()).run(4).to_dict()
            )
        else:
            from pecos_selene_wasm import build
            from selene_sim import Quest

            with TemporaryDirectory(prefix="pecos-wasm-float-") as directory:
                runner = build(function, modules, build_dir=Path(directory))
                data = [
                    dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=1)
                ]
        print(data)


if __name__ == "__main__":
    main()
