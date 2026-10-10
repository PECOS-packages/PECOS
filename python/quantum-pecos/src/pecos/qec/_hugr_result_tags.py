"""Static measurement provenance and control-flow checks for Guppy DEMs."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from typing import TYPE_CHECKING

from hugr import ops
from hugr.cli import convert, validate
from hugr.envelope import EnvelopeFormat, EnvelopeHeader
from hugr.package import Package
from hugr.tys import StringArg
from tket_exts import tket_registry

if TYPE_CHECKING:
    from hugr import Hugr
    from hugr.hugr.node_port import Node


def load_hugr_from_bytes(data: bytes) -> Hugr:
    """Validate an envelope and return module zero, requiring at least one module.

    Validation and parsing failures raise ``ValueError`` with the prefix
    ``Failed to parse HUGR``. Standard and tket extensions are available even
    when the envelope does not embed them; unresolved operations are rejected.
    """
    if not data:
        msg = "Failed to parse HUGR: Empty HUGR input"
        raise ValueError(msg)
    try:
        # Use a fresh registry: Package.from_bytes adds embedded extensions to it.
        # Both loaders also include the standard HUGR extensions by default.
        registry = tket_registry()
        extension_data = files("tket_exts").joinpath("data")
        with ExitStack() as resources:
            extension_files = [
                extension_data.joinpath(extension.name.replace(".", "/") + ".json") for extension in registry.extensions
            ]
            extension_paths = [
                str(resources.enter_context(as_file(extension_file))) for extension_file in extension_files
            ]
            # HUGR 0.18.3's native validator still prints its success message to
            # stderr even with cli_with_io's --quiet flag. Errors must propagate.
            validate(data, extensions=extension_paths)
            # hugr-py 0.18.3 routes bare S-expressions to its binary model reader.
            # Normalize that format explicitly; never retry a failed parse.
            if EnvelopeHeader.from_bytes(data).format == EnvelopeFormat.S_EXPRESSION:
                data = convert(data, format="model", extensions=extension_paths)
        package = Package.from_bytes(data, registry)
    except Exception as error:
        msg = f"Failed to parse HUGR: {error}"
        raise ValueError(msg) from error
    if not package.modules:
        msg = "Failed to parse HUGR: Package contains no modules"
        raise ValueError(msg)
    unresolved = next(
        (module[node].op for module in package.modules for node in module if isinstance(module[node].op, ops.Custom)),
        None,
    )
    if unresolved is not None:
        msg = f"Failed to parse HUGR: unresolved operation {unresolved.extension}.{unresolved.op_name}"
        raise ValueError(msg)
    return package.modules[0]


def _extension_ids(op: ops.Op) -> tuple[str, str] | None:
    if isinstance(op, ops.ExtOp):
        custom = op.to_custom_op()
        return custom.extension, custom.op_name
    return None


def _is_measurement(op: ops.Op) -> bool:
    return _extension_ids(op) in {("tket.quantum", "Measure"), ("tket.quantum", "MeasureFree")}


def measurement_op_count(hugr: Hugr) -> int:
    """Count static measurements across the whole graph, including function bodies.

    Runtime loops count their measurement nodes once, regardless of iterations.
    """
    return sum(_is_measurement(hugr[node].op) for node in hugr)


def has_nontrivial_control_flow(hugr: Hugr) -> bool:
    """Detect branching, looping, bodyless declarations, and indirect calls.

    A DataflowBlock is nontrivial only if its sum has more than one row.
    Direct calls do not hide their function bodies from whole-graph traversal.
    """
    for node in hugr:
        op = hugr[node].op
        if isinstance(op, (ops.Conditional, ops.TailLoop, ops.FuncDecl, ops.CallIndirect)):
            return True
        if isinstance(op, ops.DataflowBlock) and len(op.sum_ty.variant_rows) > 1:
            return True
    return False


def _single_source(hugr: Hugr, node: Node) -> Node | None:
    sources = list(hugr.linked_ports(node.inp(0)))
    return sources[0].node if len(sources) == 1 else None


def extract_result_tag_measurements(hugr: Hugr) -> dict[str, list[int | None]]:
    """Bind scalar result tags to measurement ordinals in whole-graph node order.

    Accept only ``tket.result:result_bool <- tket.bool:read`` (or
    ``tket.measurement:Read``) ``<- tket.quantum:Measure|MeasureFree``.
    Every other tagged tket.result occurrence contributes None, preserving
    occurrence order and unsupported holes. Keys are returned in sorted order.
    """
    measurements = [node for node in hugr if _is_measurement(hugr[node].op)]
    ordinals = {node: ordinal for ordinal, node in enumerate(measurements)}
    occurrences: dict[str, list[int | None]] = {}
    for node in hugr:
        op = hugr[node].op
        if not isinstance(op, ops.ExtOp):
            continue
        custom = op.to_custom_op()
        if custom.extension != "tket.result":
            continue
        tag = next((arg.value for arg in custom.args if isinstance(arg, StringArg)), None)
        if tag is None:
            continue
        ordinal: int | None = None
        if custom.op_name == "result_bool":
            read = _single_source(hugr, node)
            if read is not None and _extension_ids(hugr[read].op) in {
                ("tket.bool", "read"),
                ("tket.measurement", "Read"),
            }:
                measurement = _single_source(hugr, read)
                if measurement is not None:
                    ordinal = ordinals.get(measurement)
        occurrences.setdefault(tag, []).append(ordinal)
    return dict(sorted(occurrences.items()))
