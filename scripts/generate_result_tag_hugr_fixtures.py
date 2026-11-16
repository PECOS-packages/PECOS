#!/usr/bin/env -S uv run --frozen python
"""Regenerate result-tag HUGR fixtures with the pinned Guppy version."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.lang import owned
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    h,
    measure,
    measure_array,
    qubit,
    x,
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


@guppy
def measure_owned(q: qubit @ owned) -> bool:
    return measure(q).read()


@guppy
def inlined() -> None:
    # Issue #1001: the inlined helper's measurement runs first but gets a later HUGR node.
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()
    h(q1)
    cx(q1, q2)
    x(q2)
    x(q2)
    b = measure_owned(q1)
    a = measure(q0).read()
    c = measure(q2).read()
    result("a", a)
    result("b", b)
    result("c", c)


@guppy.declare
def mystery(b: bool) -> bool: ...


@guppy
def funcdecl() -> None:
    q = qubit()
    _ = mystery(measure(q).read())


def main() -> None:
    fixtures = {
        "scrambled.hugr": scrambled,
        "looped.hugr": looped,
        "computed.hugr": computed,
        "arr.hugr": arr,
        "funcdecl.hugr": funcdecl,
        "inlined.hugr": inlined,
    }
    output_dir = Path(__file__).resolve().parents[1] / "python/quantum-pecos/tests/qec/fixtures/result_tags"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, program in fixtures.items():
        output_path = output_dir / filename
        output_path.write_text(f"{program.compile().to_str()}\n")
        print(f"Created {output_path}")


if __name__ == "__main__":
    main()
