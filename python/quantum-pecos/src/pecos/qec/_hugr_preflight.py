"""HUGR loading and control-flow checks for Guppy DEMs."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from typing import TYPE_CHECKING

from hugr import ops
from hugr.cli import convert, validate
from hugr.envelope import EnvelopeFormat, EnvelopeHeader
from hugr.package import Package
from hugr.std.int import int_t
from hugr.tys import Bool, FunctionType, Qubit
from tket_exts import tket_registry

if TYPE_CHECKING:
    from hugr import Hugr


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


def _is_static_pecos_barrier(op: ops.FuncDecl) -> bool:
    """Recognize only the two fixed native barrier ABIs, including their HUGR types.

    These helpers in pecos-qis-ffi queue one Barrier and return unchanged qubit
    handles. They cannot measure, allocate, branch on a result, or loop. The LLVM
    helper normalizer independently checks their lowered ABI before linking.
    A familiar name with a different signature is not a static certificate.
    """
    arity = {
        "pecos_qis_runtime_barrier_qubit_hugr": 1,
        "pecos_qis_runtime_barrier_qubits2_hugr": 2,
    }.get(op.f_name.rsplit(".", 1)[-1])
    return (
        arity is not None
        and not op.signature.params
        and op.signature.body == FunctionType([Qubit] * arity, [Qubit] * arity)
    )


def _is_static_pecos_ordering(op: ops.FuncDecl) -> bool:
    """Recognize opaque dependency helpers with fixed, measurement-free behavior.

    The allocation helper performs the ordinary allocate/reset sequence. The
    others return unchanged dependencies and optionally request a barrier; none
    branches on the measured value. Linking binds these names to the native ABI.
    """
    token = int_t(6)  # Guppy's 64-bit int, not an arbitrary integer extension.
    signature = {
        "pecos_qis_ordered_alloc_hugr": FunctionType([token], [Qubit]),
        "pecos_qis_ordered_alloc_tagged_hugr": FunctionType([token], [Qubit]),
        "pecos_qis_ordering_join_hugr": FunctionType([Qubit, token], [Qubit, token]),
        "pecos_qis_ordering_observe_hugr": FunctionType([token, Bool], [token]),
        "pecos_qis_ordering_barrier_hugr": FunctionType([token], [token]),
    }.get(op.f_name.rsplit(".", 1)[-1])
    return signature is not None and not op.signature.params and op.signature.body == signature


def has_nontrivial_control_flow(hugr: Hugr) -> bool:
    """Detect branching, looping, unknown bodyless declarations, and indirect calls.

    A DataflowBlock is nontrivial only if its sum has more than one row.
    Direct calls do not hide their function bodies from whole-graph traversal.
    Type-checked PECOS runtime barrier and ordering helpers are known straight-line
    operations, not a blanket exemption for external functions or helper names.
    """
    for node in hugr:
        op = hugr[node].op
        if isinstance(op, (ops.Conditional, ops.TailLoop, ops.CallIndirect)):
            return True
        if isinstance(op, ops.FuncDecl) and not (_is_static_pecos_barrier(op) or _is_static_pecos_ordering(op)):
            return True
        if isinstance(op, ops.DataflowBlock) and len(op.sum_ty.variant_rows) > 1:
            return True
    return False
