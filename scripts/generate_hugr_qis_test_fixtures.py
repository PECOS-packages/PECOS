#!/usr/bin/env -S uv run python
"""Generate current-Guppy HUGR fixtures for pecos-hugr-qis tests."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import owned
from guppylang.std.quantum import cx, h, measure, qubit


@guppy.declare
def pecos_qis_runtime_barrier_qubits2_hugr(
    q0: qubit @ owned,
    q1: qubit @ owned,
) -> tuple[qubit, qubit]: ...


@guppy
def barrier_pair_probe() -> tuple[bool, bool]:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    q0, q1 = pecos_qis_runtime_barrier_qubits2_hugr(q0, q1)
    cx(q0, q1)
    return measure(q0).read(), measure(q1).read()


def main() -> None:
    output = Path("crates/pecos-hugr-qis/tests/fixtures/szz_barrier_probe.hugr")
    output.write_text(f"{barrier_pair_probe.compile().to_str()}\n")
    print(f"Created {output}")


if __name__ == "__main__":
    main()
