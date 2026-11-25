"""Shared unknown-provenance check for consumers requiring measurement identity."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any


def known_provenance_traces(traces: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Yield records, rejecting unknown bytes with their tag and call occurrence.

    Unavailable legacy provenance and explicit constant/derived values retain
    each consumer's existing policy. Occurrences count calls, not array elements.
    """
    next_occurrence: dict[str, int] = {}
    for trace in traces:
        tag = trace.get("name")
        occurrence = next_occurrence.get(tag, 0) if isinstance(tag, str) else 0
        explicit = trace.get("occurrence")
        if type(explicit) is int and explicit >= 0:
            occurrence = explicit
        if isinstance(tag, str):
            next_occurrence[tag] = max(next_occurrence.get(tag, 0), occurrence + 1)
        provenance = trace.get("provenance")
        if isinstance(provenance, list):
            for element, value in enumerate(provenance):
                if isinstance(value, Mapping) and value.get("class") == "unknown":
                    msg = f"result tag {tag!r} occurrence {occurrence} has unknown provenance at element {element}"
                    raise ValueError(msg)
        yield trace
