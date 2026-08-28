# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for :func:`pecos.guppy_gen.variant_scoped`.

``variant_scoped`` gives a factory-local ``@guppy`` program a variant-unique
name so that building several parameterizations in one process does not collide
in guppylang's compile cache / Selene's build cache (which key on the program's
module-qualified / entry-point name). See ``variant.py`` for the mechanism.
"""

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import h, measure, qubit
from pecos.guppy_gen import variant_scoped


def test_variant_scoped_suffixes_name_and_qualname() -> None:
    """The variant values are appended to both ``__name__`` and ``__qualname__``."""

    def prog() -> None:
        pass

    base_qualname = prog.__qualname__
    returned = variant_scoped(prog, 3)

    assert returned is prog
    assert prog.__name__ == "prog_3"
    assert prog.__qualname__ == f"{base_qualname}_3"


def test_variant_scoped_multiple_values() -> None:
    """Multiple distinguishing values are joined into the suffix."""

    def prog() -> None:
        pass

    variant_scoped(prog, 5, "Z")
    assert prog.__name__ == "prog_5_Z"


def test_variant_scoped_sanitizes_unsafe_characters() -> None:
    """Non-alphanumeric characters in a variant value are replaced, keeping the name valid."""

    def prog() -> None:
        pass

    variant_scoped(prog, "a-b.c")
    assert prog.__name__ == "prog_a_b_c"
    assert prog.__name__.replace("_", "").isalnum()


def test_variant_scoped_requires_a_value() -> None:
    """Calling with no distinguishing value fails loud."""

    def prog() -> None:
        pass

    with pytest.raises(ValueError, match="at least one distinguishing value"):
        variant_scoped(prog)


def test_variant_scoped_isolates_parameterizations() -> None:
    """Two round counts built in one process compile to distinct HUGR.

    Without the variant suffix both parameterizations share the factory-local
    ``@guppy`` name, so the second silently reuses the first's build; with it,
    each compiles to its own program.
    """
    from pecos.compilation_pipeline import compile_guppy_to_hugr

    def make(num_rounds: int) -> object:
        def memory() -> None:
            for _ in range(comptime(num_rounds)):
                q = qubit()
                result("synx", array(measure(q).read()))
            anc = qubit()
            h(anc)
            _ = measure(anc).read()

        return guppy(variant_scoped(memory, num_rounds))

    hugr_two = compile_guppy_to_hugr(make(2))
    hugr_five = compile_guppy_to_hugr(make(5))
    assert hugr_two != hugr_five
