# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Circuit diagrams for notebooks, terminals, and files.

Rendering uses PECOS's Rust renderer and needs no plotting package. It never
executes a program: QIS traces are validated and replayed as circuits first.
Ticks and dependency layers are structural positions, not physical time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pecos.quantum import DagCircuit, TickCircuit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike
    from typing import Any

DiagramFormat = Literal["ascii", "unicode", "svg", "tikz", "dot"]

__all__ = [
    "CircuitDiagram",
    "CircuitInspection",
    "CircuitInspector",
    "DiagramFormat",
    "circuit_diagram",
    "circuit_diagram_panels",
    "inspect_circuit",
]


class _PrettyPrinter(Protocol):
    def text(self, text: str) -> None: ...


@dataclass(frozen=True)
class CircuitDiagram:
    """An immutable rendered snapshot, displayable as SVG in notebooks.

    Use :func:`circuit_diagram` to create one. ``str(diagram)`` returns the
    rendered content; :meth:`save` writes exactly that content as UTF-8.
    Later edits to the source circuit do not change this snapshot.
    """

    content: str
    format: DiagramFormat

    def __str__(self) -> str:
        """Return the rendered content."""
        return self.content

    def __repr__(self) -> str:
        """Return a compact description without embedding the diagram."""
        return f"CircuitDiagram(format={self.format!r})"

    def _repr_svg_(self) -> str | None:
        """Return SVG only for an SVG diagram, following the notebook protocol."""
        return self.content if self.format == "svg" else None

    def _repr_pretty_(self, printer: _PrettyPrinter, cycle: bool) -> None:
        """Display text diagrams directly in IPython without quoted escapes."""
        printer.text(repr(self) if cycle or self.format == "svg" else self.content)

    def save(self, path: str | PathLike[str]) -> None:
        """Write the diagram, replacing an existing file if present.

        The format is chosen when rendering, not inferred from the filename.
        TikZ and DOT exports are source text; no external compiler is run.
        """
        Path(path).write_text(self.content, encoding="utf-8")


def circuit_diagram(
    source: TickCircuit | DagCircuit | list[dict[str, Any]],
    *,
    format: DiagramFormat = "svg",  # noqa: A002 - matches the native circuit render API
    color: bool = True,
    angle_unit: Literal["radians", "turns"] = "radians",
    start: int | None = None,
    stop: int | None = None,
    title: str | None = None,
    show_measurement_ids: bool = False,
) -> CircuitDiagram:
    """Render a circuit or a completed, single-shot QIS operation trace.

    Args:
        source: A ``TickCircuit``, ``DagCircuit``, or the JSON-compatible list
            returned by ``capture_qis_operation_trace``. Saved traces can be
            loaded with ``json.loads``. Programs and filenames are not accepted.
        format: ``svg`` (default), ``ascii``, ``unicode``, ``tikz``, or ``dot``.
            DOT renders the wire diagram, not the original DAG edge structure.
        color: Use color in graphical output. Text is always free of ANSI codes.
        angle_unit: Display rotation angles in ``radians`` or ``turns``.
        start: First tick or dependency layer to show, inclusive. Defaults to 0
            when ``stop`` is given. Original indices and all qubit rows remain.
        stop: Exclusive last tick or layer. Defaults to the circuit's end when
            ``start`` is given. Bounds must be non-negative and within the source.
        title: Optional panel title, prepended to the circuit/range description.
        show_measurement_ids: Label measurements with stored identities (e.g.
            ``MZ[m7]``). Missing result identities show ``m?``. IDs are not
            outcomes or emission ordinals, and rendering never assigns them.
            ``MeasureLeaked`` can return 0, 1, or 2 (leaked) at runtime; these
            labels identify measurements, not their returned values.

    Returns:
        A rendered snapshot. Leave it as the last expression in a notebook
        cell for SVG display, print it for text, or call ``save(path)``.

    Notes:
        Tick circuits retain explicit tick boundaries. DAG circuits use
        dependency layers. Traces use the validated runtime-lowered stream,
        replayed into ASAP-compacted ticks, and represent only one execution
        path. None of these axes represents physical time. Rendering does not
        attach detector metadata. Visualization retains ``MeasureLeaked`` as
        ``MZ_L`` with its result IDs. Labels describe operations and identities;
        individual measurement outcome values are not shown.

    Raises:
        TypeError: The source is not a supported circuit or trace list.
        ValueError: The format, angle unit, range, or trace is invalid. Trace validation
            may also raise ``TypeError`` for malformed field types.
    """
    if format not in ("ascii", "unicode", "svg", "tikz", "dot"):
        msg = "format must be 'ascii', 'unicode', 'svg', 'tikz', or 'dot'"
        raise ValueError(msg)
    if angle_unit not in ("radians", "turns"):
        msg = "angle_unit must be 'radians' or 'turns'"
        raise ValueError(msg)

    circuit = _as_circuit(source)

    return CircuitDiagram(
        content=circuit.render(
            format,
            color=color,
            angle_unit=angle_unit,
            start=start,
            stop=stop,
            title=title,
            show_measurement_ids=show_measurement_ids,
        ),
        format=format,
    )


def _as_circuit(
    source: TickCircuit | DagCircuit | list[dict[str, Any]],
) -> TickCircuit | DagCircuit:
    if isinstance(source, (TickCircuit, DagCircuit)):
        return source
    if isinstance(source, list):
        from pecos.tracing import _qis_operation_trace_to_visualization_circuit  # noqa: PLC0415

        return _qis_operation_trace_to_visualization_circuit(source)
    msg = "source must be a TickCircuit, DagCircuit, or a completed QIS trace list"
    raise TypeError(msg)


@dataclass(frozen=True)
class CircuitInspector:
    """Standalone Rust-generated viewer with an isolated notebook presentation.

    Create this with ``inspect_circuit(source).interactive()``. Saved HTML works
    offline. Notebook interactivity requires a client allowing scripted iframes.
    """

    content: str

    def __repr__(self) -> str:
        """Return a compact description without embedding HTML."""
        return "CircuitInspector()"

    def _repr_html_(self) -> str:
        """Embed in a sandbox without granting access to the notebook document."""
        return (
            '<iframe title="PECOS circuit inspector" sandbox="allow-scripts" '
            'style="width:100%;height:640px;border:1px solid #cdd6e1;border-radius:10px" '
            f'srcdoc="{escape(self.content, quote=True)}"></iframe>'
        )

    def save(self, path: str | PathLike[str]) -> None:
        """Save the standalone HTML document, replacing the destination."""
        Path(path).write_text(self.content, encoding="utf-8")


@dataclass(frozen=True)
class CircuitInspection:
    """Owned SVG and source records for inspection, independent of later edits.

    Each operation record represents a stored gate batch, not necessarily one
    gate instance. SVG components refer to its ID via ``data-pecos-operation``.
    IDs retain source locations across range selection of an unchanged circuit;
    they are not persistent identities across edits or different circuits.
    """

    _snapshot_json: str
    _html: str

    @property
    def svg(self) -> str:
        """Return the standalone annotated SVG."""
        return json.loads(self._snapshot_json)["svg"]

    @property
    def operations(self) -> list[dict[str, Any]]:
        """Return independent copies of the stored-batch records."""
        return json.loads(self._snapshot_json)["operations"]

    def __repr__(self) -> str:
        """Return a compact description."""
        return f"CircuitInspection(operations={len(self.operations)})"

    def _repr_svg_(self) -> str:
        """Display the annotated SVG in notebooks."""
        return self.svg

    def to_json(self) -> str:
        """Return the versioned snapshot as JSON, including SVG and metadata."""
        return self._snapshot_json

    def save(self, path: str | PathLike[str]) -> None:
        """Save snapshot JSON as UTF-8, replacing the destination if it exists."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def save_svg(self, path: str | PathLike[str]) -> None:
        """Save annotated SVG as UTF-8, replacing the destination if it exists."""
        Path(path).write_text(self.svg, encoding="utf-8")

    def to_html(self) -> str:
        """Return a standalone interactive document generated by Rust."""
        return self._html

    def save_html(self, path: str | PathLike[str]) -> None:
        """Save the standalone inspector HTML, replacing the destination."""
        Path(path).write_text(self.to_html(), encoding="utf-8")

    def interactive(self) -> CircuitInspector:
        """Return a viewer for explicit interactive display in a notebook."""
        return CircuitInspector(self.to_html())


def inspect_circuit(
    source: TickCircuit | DagCircuit | list[dict[str, Any]],
    *,
    color: bool = True,
    angle_unit: Literal["radians", "turns"] = "radians",
    start: int | None = None,
    stop: int | None = None,
    show_measurement_ids: bool = False,
) -> CircuitInspection:
    """Inspect a circuit or completed trace with Rust-generated source links.

    Trace metadata includes ``_pecos_trace_origin`` with the engine, shot,
    chunk and lowered-operation indices. It survives replay compaction and
    decomposition; it does not infer raw QIS or HUGR source nodes. Every emitted
    stage of a lowered operation shares its recorded trace origin.

    Snapshot JSON includes gate metadata; non-finite metadata/parameter floats
    are rejected rather than silently changed to JSON null. Channel presence
    is recorded, but the JSON does not serialize the channel payload itself.
    """
    snapshot = _as_circuit(source).inspect(
        color=color,
        angle_unit=angle_unit,
        start=start,
        stop=stop,
        show_measurement_ids=show_measurement_ids,
        include_html=True,
    )
    html = snapshot.pop("html")
    return CircuitInspection(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")), html)


def circuit_diagram_panels(
    source: TickCircuit | DagCircuit | list[dict[str, Any]],
    panels: Sequence[tuple[str, int, int]],
    *,
    format: DiagramFormat = "svg",  # noqa: A002 - matches the native render API
    color: bool = True,
    angle_unit: Literal["radians", "turns"] = "radians",
    show_measurement_ids: bool = False,
) -> list[CircuitDiagram]:
    """Render named half-open ranges, validating and replaying a trace once.

    Each panel is ``(title, start, stop)`` in source tick or dependency-layer
    coordinates. The result preserves input order; overlapping ranges, gaps,
    and repeated titles are allowed. Round boundaries are supplied explicitly,
    never inferred from gate patterns. Display or save each returned snapshot.
    The complete trace is validated even when a panel selects only a subset.
    Leakage-aware measurements retain their ``MZ_L`` symbols and result IDs.
    """
    # Check options even when no panels were requested.
    if format not in ("ascii", "unicode", "svg", "tikz", "dot"):
        msg = "format must be 'ascii', 'unicode', 'svg', 'tikz', or 'dot'"
        raise ValueError(msg)
    if angle_unit not in ("radians", "turns"):
        msg = "angle_unit must be 'radians' or 'turns'"
        raise ValueError(msg)
    circuit = _as_circuit(source)
    return [
        circuit_diagram(
            circuit,
            format=format,
            color=color,
            angle_unit=angle_unit,
            start=start,
            stop=stop,
            title=title,
            show_measurement_ids=show_measurement_ids,
        )
        for title, start, stop in panels
    ]
