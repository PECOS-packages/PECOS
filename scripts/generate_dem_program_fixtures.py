# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regenerate native DEM program fixtures with the workspace's pinned Guppy."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import collect_measurements, cx, measure, measure_array, qubit, x
from pecos_rslib.hugr_lowering import compile_hugr_to_qis


@guppy
def parities() -> None:
    q0, q1, q2 = qubit(), qubit(), qubit()
    cx(q0, q1)
    m0 = measure(q0).read()
    m1 = measure(q1).read()
    m2 = measure(q2).read()
    output("DETECTOR", m0 ^ m2)
    output("raw measurements", array(m2, m0, m1))
    output("obs", m1)
    output("unused", m0 & m1)


@guppy
def changing_schedule() -> None:
    m0 = measure(qubit()).read()
    q1 = qubit()
    if m0:
        x(q1)
    m1 = measure(q1).read()
    output("raw measurements", m0)
    output("raw measurements", m1)
    output("DETECTOR", m0 ^ m1)
    output("obs", m1)


@guppy
def explicit_only() -> None:
    m0 = measure(qubit()).read()
    output("measurement", m0)


@guppy
def direct_outputs() -> None:
    bits = collect_measurements(measure_array(array(qubit(), qubit(), qubit())))
    m0, m1, m2 = bits[0], bits[1], bits[2]
    output("physical", bits)
    output("checks", m0 ^ m1)
    output("logical_z", m0)
    output("logical_x", array(m1, m2))


@guppy
def integer_outputs() -> None:
    m0 = measure(qubit()).read()
    m1 = measure(qubit()).read()
    m2 = measure(qubit()).read()
    output("raw measurements", m0)
    output("raw measurements", m1)
    output("raw measurements", m2)
    # Integer arrays carry no per-call trace; scalar 0/1 integers do. Element
    # positions must still follow the complete output stream.
    output("checks", array(int(m0)))
    output("checks", int(m1))
    output("bits", array(int(m1), int(m2)))
    output("counts", 2)
    output("counts", int(m2))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "crates/pecos-qec/tests/fixtures/dem_programs"
    directory.mkdir(parents=True, exist_ok=True)
    for name, program in [
        ("parities", parities),
        ("changing_schedule", changing_schedule),
        ("explicit_only", explicit_only),
        ("direct_outputs", direct_outputs),
        ("integer_outputs", integer_outputs),
    ]:
        hugr = program.compile().to_bytes()
        (directory / f"{name}.hugr").write_text(program.compile().to_str() + "\n")
        ir = compile_hugr_to_qis(hugr)
        ir = "\n".join(line for line in ir.splitlines() if not line.startswith(("target triple", "target datalayout")))
        (directory / f"{name}.ll").write_text(ir + "\n")


if __name__ == "__main__":
    main()
