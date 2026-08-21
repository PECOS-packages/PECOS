#!/usr/bin/env -S uv run python
"""Regenerate result-tag HUGR fixtures with the pinned Guppy version."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    qubit,
)


@guppy
def scrambled() -> None:
    qa = qubit()
    qb = qubit()
    qc = qubit()
    a = measure(qa).read()
    b = measure(qb).read()
    c = measure(qc).read()
    result("tag_c", c)
    result("tag_a", a)
    result("tag_b", b)


@guppy
def looped() -> None:
    for _ in range(comptime(3)):
        q = qubit()
        result("synx", measure(q).read())


@guppy
def computed() -> None:
    q0 = qubit()
    q1 = qubit()
    m0 = measure(q0).read()
    m1 = measure(q1).read()
    result("eq", m0 == m1)
    result("const", True)  # noqa: FBT003 - the fixture must capture a constant.


@guppy
def arr() -> None:
    qs = array(qubit() for _ in range(2))
    result("pair", collect_measurements(measure_array(qs)))


@guppy.declare
def mystery(b: bool) -> bool: ...


@guppy
def funcdecl() -> None:
    q = qubit()
    _ = mystery(measure(q).read())


@guppy
def helper() -> None:
    q = qubit()
    _ = measure(q).read()


@guppy
def indirect() -> None:
    g = helper
    g()


def main() -> None:
    fixtures = {
        "scrambled.hugr": scrambled,
        "looped.hugr": looped,
        "computed.hugr": computed,
        "arr.hugr": arr,
        "funcdecl.hugr": funcdecl,
        "indirect.hugr": indirect,
    }
    output_dir = Path("crates/pecos-hugr/tests/fixtures")
    for filename, program in fixtures.items():
        output_path = output_dir / filename
        output_path.write_text(f"{program.compile().to_str()}\n")
        print(f"Created {output_path}")


if __name__ == "__main__":
    main()
