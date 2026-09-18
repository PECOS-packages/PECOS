# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Grammar-verdict assertions for public Python DEM consumers."""

from __future__ import annotations

import re

INDEX_CONSUMERS = {"ParsedDem", "DemSampler", "DemAwareDecoder", "bp_osd", "pymatching"}
FLAT_CONSUMERS = INDEX_CONSUMERS


def assert_outcome(
    text: str,
    accepted: bool,
    consumer: str,
    counts: tuple[int, int] | None,
    expected: tuple[int, int] | None,
    error: str | None,
) -> None:
    """Assert a grammar verdict, dimensions, or an exact semantic rejection."""
    if error is None:
        assert accepted, (consumer, "accepted invalid grammar")
        if counts is not None and expected is not None:
            assert counts == expected, (consumer, counts, expected)
        return

    message = error.lower()
    overflow = re.search(
        r"(?:detector|observable) index (\d+) exceeds the supported maximum (\d+)",
        message,
    )
    semantic = (
        (consumer in INDEX_CONSUMERS and overflow is not None and int(overflow[1]) > int(overflow[2]))
        or (consumer in FLAT_CONSUMERS and "requires a flattened dem:" in message)
        or (consumer == "DemAwareDecoder" and message == "dem contains no error mechanisms")
        or (
            consumer == "pymatching"
            and text == "error(1) D0"
            and message
            == "decoder failed on shot 0: decoding failed: ffi error: maximum absolute edge weight of 16777215 exceeded."
        )
    )
    if accepted:
        assert semantic, (consumer, "unexpected rejection of accepted grammar", error)
        assert "Invalid DEM syntax: " not in error, (
            consumer,
            "grammar error disguised as semantic rejection",
            error,
        )
    else:
        assert not semantic, (
            consumer,
            "semantic error instead of grammar error",
            error,
        )
        if consumer != "PyMatchingDecoder":
            assert "Invalid DEM syntax: " in error, (consumer, error)
