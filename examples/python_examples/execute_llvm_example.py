#!/usr/bin/env python3
"""Compile Guppy to LLVM IR through PECOS's Selene compiler boundary."""

from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit
from pecos.compilation_pipeline import compile_hugr_to_qis


def main() -> None:
    """Compile a quantum coin flip to QIS."""

    @guppy
    def coin_flip() -> bool:
        q = qubit()
        h(q)
        return measure(q).read()

    llvm_ir = compile_hugr_to_qis(coin_flip.compile().to_bytes())
    print(llvm_ir)


if __name__ == "__main__":
    main()
