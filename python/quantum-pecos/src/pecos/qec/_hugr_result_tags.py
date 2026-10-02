"""Static measurement provenance and control-flow checks for Guppy DEMs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hugr import ops
from hugr.cli import validate
from hugr.package import Package
from hugr.tys import StringArg

if TYPE_CHECKING:
    from hugr import Hugr
    from hugr.hugr.node_port import Node


def load_hugr_from_bytes(data: bytes) -> Hugr:
    """Validate an envelope and return module zero, requiring at least one module.

    Validation and parsing failures raise ``ValueError`` with the prefix
    ``Failed to parse HUGR``.
    """
    if not data:
        msg = "Failed to parse HUGR: Empty HUGR input"
        raise ValueError(msg)
    try:
        # cli_with_io captures command output; native diagnostics (including
        # the validator's success message) go to stderr, never stdout.
        validate(data)
        package = Package.from_bytes(data)
    except Exception as error:
        msg = f"Failed to parse HUGR: {error}"
        raise ValueError(msg) from error
    if not package.modules:
        msg = "Failed to parse HUGR: Package contains no modules"
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
