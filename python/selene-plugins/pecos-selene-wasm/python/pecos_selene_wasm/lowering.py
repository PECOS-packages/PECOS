"""Independent lowering of public tket.wasm to a versioned scalar WASM ABI.

The released public Selene compiler handles the remaining HUGR. Module
identifiers resolve only to explicitly supplied bytes, never implicit files.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import math
import re
import reprlib
import struct
from typing import TYPE_CHECKING

from hugr.hugr import Hugr
from hugr.package import Package
from selene_hugr_qis_compiler import compile_to_llvm_ir

if TYPE_CHECKING:
    from collections.abc import Mapping

    from guppylang.defs import GuppyFunctionDefinition

_I64 = {
    "t": "Opaque",
    "extension": "arithmetic.int.types",
    "extension_version": "0.1.0",
    "id": "int",
    "args": [{"tya": "BoundedNat", "n": 6}],
    "bound": "C",
}
_PREFIX = "pecos_wasm_adapter_"
_FRONTEND_METADATA = "pecos.wasm.frontend"
_SAFE_FRONTEND = "guppy-minimal-opt-v1"
type Json = dict[str, Json] | list[Json] | str | int | float | bool | None


def compile_guppy(guppy_function: object) -> Package:
    """Compile Guppy source without unsafe external-call constant folding.

    Pass the returned package directly to ``build`` or ``compile_wasm``, or
    serialize it for PECOS ``Hugr``. Do not run other optimizers on this package
    before WASM lowering: the metadata records how it was compiled, not a proof
    that arbitrary subsequent transformations preserved its semantics.
    """
    compiled = _compile_guppy_package(guppy_function)
    package = Package.from_bytes(compiled.to_bytes())
    for graph in package.modules:
        graph[graph.module_root].metadata[_FRONTEND_METADATA] = _SAFE_FRONTEND
    return package


def _compile_guppy_package(guppy_function: object) -> Package:
    """Return frontend output for read-only consumption; it may be cached."""
    from guppylang.defs import GuppyFunctionDefinition  # noqa: PLC0415

    if not isinstance(guppy_function, GuppyFunctionDefinition):
        msg = "Function must be decorated with @guppy"
        raise TypeError(msg)
    # A new optimizer instance leaves the original definition and normal PECOS
    # compile cache unchanged. Guppy compilation occurs before our HUGR pass.
    return guppy_function.with_minimal_opt().compile()


def _has_unresolved_versions(value: Json) -> bool:
    """Custom frontend operations may still need registry version resolution."""
    if isinstance(value, list):
        return any(_has_unresolved_versions(item) for item in value)
    if not isinstance(value, dict):
        return False
    if (value.get("op") == "Extension" or value.get("t") == "Opaque") and value.get(
        "extension_version",
    ) in (None, "0.0.0"):
        return True
    return any(_has_unresolved_versions(item) for item in value.values())


def _has_loops(document: dict) -> bool:
    nodes = document["nodes"]
    if any(node.get("op") == "TailLoop" for node in nodes):
        return True
    blocks = {index for index, node in enumerate(nodes) if node.get("op") == "DataflowBlock"}
    successors = {block: set() for block in blocks}
    indegree = dict.fromkeys(blocks, 0)
    for source, target in document["edges"]:
        a, b = source[0], target[0]
        if a in blocks and b in blocks and nodes[a]["parent"] == nodes[b]["parent"] and b not in successors[a]:
            successors[a].add(b)
            indegree[b] += 1
    ready = [block for block, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        visited += 1
        for successor in successors[ready.pop()]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return visited != len(blocks)


def _check_external_frontend(document: dict) -> None:
    """Reject visible mixed-extern loops whose frontend safety is unknown."""
    metadata = document.get("metadata")
    if metadata and metadata[0] and metadata[0].get(_FRONTEND_METADATA) == _SAFE_FRONTEND:
        return
    nodes = document["nodes"]
    declarations = {
        index
        for index, node in enumerate(nodes)
        if node.get("op") == "FuncDecl" and node["signature"]["body"]["output"]
    }
    calls_external = any(
        source[0] in declarations and nodes[target[0]].get("op") in {"Call", "LoadFunction"}
        for source, target in document["edges"]
    )
    if calls_external and _has_loops(document):
        msg = (
            "Loops with ordinary external declarations require the safe Guppy frontend: "
            "pass the Guppy function directly, or use pecos_selene_wasm.compile_guppy(function) "
            "before serializing. Default Guppy optimization may already have corrupted this package."
        )
        raise ValueError(msg)


def _order_calls(document: dict, acquisitions: set[int]) -> None:
    """Choose a stable dependency-respecting schedule for classical effects.

    Older Guppy releases leave context release/reacquisition unordered. Once
    WASM operations become external calls, choose a deterministic topological
    schedule, using serialized node order only to break ties. Add order edges
    between calls and enclosing dataflow regions, including calls to helpers.
    Never order mutually exclusive cases or function definitions themselves.
    """
    nodes, edges = document["nodes"], document["edges"]
    effects = set()
    for index, node in enumerate(nodes):
        if node.get("op") not in {"Call", "CallIndirect"}:
            continue
        current = index
        while nodes[current].get("op") not in {"FuncDefn", "Module"}:
            effects.add(current)
            current = nodes[current]["parent"]

    def child_of(node: int, parent: int) -> int | None:
        while node != parent:
            ancestor = nodes[node]["parent"]
            if ancestor == parent:
                return node
            if ancestor == node:
                break
            node = ancestor
        return None

    for parent, node in enumerate(nodes):
        if node.get("op") not in {"FuncDefn", "DFG", "DataflowBlock", "Case", "TailLoop"}:
            continue
        children = {index for index, child in enumerate(nodes) if child["parent"] == parent}
        successors = {child: set() for child in children}
        indegree = dict.fromkeys(children, 0)
        for source, target in edges:
            a, b = child_of(source[0], parent), child_of(target[0], parent)
            if a is not None and b is not None and a != b and b not in successors[a]:
                successors[a].add(b)
                indegree[b] += 1
        # Older Guppy orders the acquisition's success check, but can leave
        # get_context itself unordered (and hoist it before a preceding loop).
        # Restore those same-region predecessors when acquisition has no
        # explicit order. Never contradict an existing dependency path.
        for acquire in sorted(children & acquisitions):
            if any(target == [acquire, None] for _, target in edges):
                continue
            guards = {
                target[0]
                for source, target in edges
                if source == [acquire, 0]
                and target[1] == 0
                and target[0] in children
                and nodes[target[0]].get("op") == "Conditional"
                and not nodes[target[0]].get("other_inputs")
            }
            for before, after in edges.copy():
                predecessor = before[0]
                if after[0] not in guards or after[1] is not None or predecessor not in children:
                    continue
                reachable, pending = set(), [acquire]
                while pending:
                    current = pending.pop()
                    if current not in reachable:
                        reachable.add(current)
                        pending.extend(successors[current])
                if predecessor in reachable:
                    continue
                edges.append([[predecessor, None], [acquire, None]])
                if acquire not in successors[predecessor]:
                    successors[predecessor].add(acquire)
                    indegree[acquire] += 1

        ready = [child for child, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        schedule = []
        while ready:
            child = heapq.heappop(ready)
            schedule.append(child)
            for successor in successors[child]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, successor)
        if len(schedule) != len(children):
            msg = "Cannot schedule cyclic HUGR dataflow for WASM calls"
            raise ValueError(msg)
        previous = None
        for child in schedule:
            if child in effects:
                if previous is not None:
                    edge = [[previous, None], [child, None]]
                    if edge not in edges:
                        edges.append(edge)
                previous = child
        if previous is not None:
            # Indirect calls may be removed when their results are unused.
            # Anchor the final effect to the region's exit, retaining the chain.
            output = next(child for child in children if nodes[child].get("op") == "Output")
            edge = [[previous, None], [output, None]]
            if edge not in edges:
                edges.append(edge)


def _protect_external_calls(document: dict) -> None:
    """Keep external results unknown to released HUGR constant analysis.

    Compiler 0.4.x/0.5.2 treat a known external declaration's return values as
    impossible, which can discard loop-carried results. Load the function value
    through the public prelude Barrier before calling it indirectly. Barrier
    has no constant-fold rule and emits as an identity, so analysis uses the
    conservative unknown-target rule while execution calls the original export.
    Protect existing loads too, including declarations other than WASM adapters.
    """
    nodes, edges = document["nodes"], document["edges"]

    def append(node: dict) -> int:
        index = len(nodes)
        nodes.append(node)
        if document.get("metadata") is not None:
            document["metadata"].append(None)
        return index

    # Convert direct declaration calls to loads and indirect calls first.
    for index, node in enumerate(nodes.copy()):
        if node.get("op") != "Call":
            continue
        signature = node["instantiation"]
        # No result can be misfolded; older compilers may discard void indirect calls.
        if not signature["output"]:
            continue
        static_port = len(signature["input"])
        link = next(edge for edge in edges if edge[1] == [index, static_port])
        if nodes[link[0][0]].get("op") != "FuncDecl":
            continue
        load = append({**node, "op": "LoadFunction"})
        edges.remove(link)
        for _, target in edges:
            if target[0] == index and target[1] is not None:
                target[1] += 1
        edges.extend([[link[0], [load, 0]], [[load, 0], [index, 0]]])
        nodes[index] = {"op": "CallIndirect", "parent": node["parent"], "signature": signature}

    for index, node in enumerate(nodes.copy()):
        if node.get("op") != "LoadFunction":
            continue
        link = next(edge for edge in edges if edge[1] == [index, 0])
        if nodes[link[0][0]].get("op") != "FuncDecl":
            continue
        signature = node["instantiation"]
        if not signature["output"]:
            continue
        barrier = append(
            {
                "op": "Extension",
                "parent": node["parent"],
                "extension": "prelude",
                "name": "Barrier",
                "args": [{"tya": "List", "elems": [{"tya": "Type", "ty": signature}]}],
                "signature": {"t": "G", "input": [signature], "output": [signature]},
            },
        )
        for source, _ in edges:
            if source == [index, 0]:
                source[0] = barrier
        edges.append([[index, 0], [barrier, 0]])


def _replace_types(value: Json) -> Json:
    if isinstance(value, list):
        return [_replace_types(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("t") == "Opaque" and value.get("extension") == "tket.wasm":
        if value["id"] not in {"module", "context", "func", "result"}:
            msg = f"Unsupported WASM type: {value['id']}"
            raise ValueError(msg)
        return copy.deepcopy(_I64)
    return {key: _replace_types(item) for key, item in value.items()}


def _scalar_types(argument: dict) -> list[str]:
    if argument.get("tya") != "List":
        msg = "WASM signatures must have concrete type rows"
        raise ValueError(msg)
    types = []
    for entry in argument["elems"]:
        ty = entry.get("ty", {})
        if entry.get("tya") == "Type" and ty.get("t") == "Opaque":
            if (
                ty.get("extension") == "arithmetic.int.types"
                and ty.get("id") == "int"
                and ty.get("args") == [{"tya": "BoundedNat", "n": 6}]
            ):
                types.append("i64")
                continue
            if ty.get("extension") == "arithmetic.float.types" and ty.get("id") == "float64" and ty.get("args") == []:
                types.append("double")
                continue
        msg = "Public WASM lowering supports only scalar i64 and f64 signatures"
        raise ValueError(msg)
    return types


def _validate(operation: dict) -> tuple[int, int]:
    if operation.get("extension_version") != "0.5.0":
        msg = "Public WASM lowering requires tket.wasm 0.5.0"
        raise ValueError(msg)
    name, args = operation["name"], operation["args"]
    if name in {"lookup_by_name", "lookup_by_id"}:
        if name == "lookup_by_name" and args[0].get("tya") != "String":
            msg = "WASM export name must be a compile-time string"
            raise ValueError(msg)
        if name == "lookup_by_id" and (args[0].get("tya") != "BoundedNat" or not 0 <= args[0].get("n", -1) < 2**64):
            msg = "WASM export index must be a compile-time unsigned 64-bit integer"
            raise ValueError(msg)
        inputs, outputs = len(_scalar_types(args[1])), len(_scalar_types(args[2]))
    elif name == "call":
        inputs, outputs = len(_scalar_types(args[0])), len(_scalar_types(args[1]))
    elif name == "read_result":
        inputs, outputs = 0, len(_scalar_types(args[0]))
    elif name in {"get_context", "dispose_context"}:
        inputs, outputs = 0, 0
    else:
        msg = f"Unsupported public WASM operation: {name}"
        raise ValueError(msg)
    if outputs > 1:
        msg = "Public WASM lowering supports zero or one scalar result"
        raise ValueError(msg)
    return inputs, outputs


def _constant(name: str, data: bytes) -> str:
    literal = "".join(f"\\{byte:02X}" for byte in data)
    return f'@{name} = private constant [{len(data)} x i8] c"{literal}"\n'


_ABI = """
declare i32 @pecos_wasm_v1_reset()
declare i32 @pecos_wasm_v1_register(i64, ptr, i64)
declare i64 @pecos_wasm_v1_acquire(i64)
declare i32 @pecos_wasm_v1_release(i64)
declare i64 @pecos_wasm_v1_lookup(i64, ptr, i64, i64, i64)
declare i64 @pecos_wasm_v2_lookup(i64, ptr, i64, ptr, i64, ptr, i64)
declare i64 @pecos_wasm_v2_lookup_by_id(i64, i64, ptr, i64, ptr, i64)
declare i32 @pecos_wasm_v1_call(i64, i64, ptr, i64)
declare i32 @pecos_wasm_v1_read(i64, ptr, i64)
declare ptr @pecos_wasm_v1_error()

define private void @pecos_wasm_check(i1 %ok) {
entry:
  br i1 %ok, label %success, label %failure
failure:
  %message = call ptr @pecos_wasm_v1_error()
  call void @panic(i32 2001, ptr %message)
  unreachable
success:
  ret void
}
"""


def _check(status: str, bits: int = 32, *, nonzero: bool = False) -> str:
    predicate = "ne" if nonzero else "eq"
    return f"  %{status}_ok = icmp {predicate} i{bits} %{status}, 0\n  call void @pecos_wasm_check(i1 %{status}_ok)\n"


def _adapter(
    symbol: str,
    operation: dict,
    unique: str,
    wasm_int_width: int | str = 64,
    wasm_async_workers: int = 0,
) -> str:
    name, args = operation["name"], operation["args"]
    inputs, outputs = _validate(operation)
    if name == "get_context":
        return f"""define {{ i1, i64 }} {symbol}(i64 %index) {{
entry:
  %handle = call i64 @pecos_wasm_v1_acquire(i64 %index)
  %ok = icmp ne i64 %handle, 0
  %tag = insertvalue {{ i1, i64 }} poison, i1 %ok, 0
  %result = insertvalue {{ i1, i64 }} %tag, i64 %handle, 1
  ret {{ i1, i64 }} %result
}}
"""
    if name in {"lookup_by_name", "lookup_by_id"}:
        in_types, out_types = _scalar_types(args[1]), _scalar_types(args[2])
        constants = ""
        if name == "lookup_by_name":
            data = args[0]["arg"].encode("utf-8")
            constants = _constant(unique, data)
            lookup = f"@pecos_wasm_v1_lookup(i64 %module, ptr @{unique}, i64 {len(data)}, i64 {inputs}, i64 {outputs})"
        if name == "lookup_by_id" or "double" in in_types + out_types or wasm_int_width != 64:
            integer_tag = {64: 0, 32: 2, "auto": 3}[wasm_int_width]
            version = 3 if wasm_int_width == "auto" else 2
            constants += _constant(
                unique + "_inputs",
                bytes(1 if ty == "double" else integer_tag for ty in in_types),
            )
            constants += _constant(
                unique + "_outputs",
                bytes(1 if ty == "double" else integer_tag for ty in out_types),
            )
            target = (
                f"@pecos_wasm_v{version}_lookup_by_id(i64 %module, i64 {args[0]['n']}, "
                if name == "lookup_by_id"
                else f"@pecos_wasm_v{version}_lookup(i64 %module, ptr @{unique}, i64 {len(data)}, "
            )
            lookup = target + f"ptr @{unique}_inputs, i64 {inputs}, ptr @{unique}_outputs, i64 {outputs})"
        return constants + f"""define i64 {symbol}(i64 %module) {{
entry:
  %handle = call i64 {lookup}
{_check('handle', 64, nonzero=True)}  ret i64 %handle
}}
"""
    if name == "dispose_context":
        return f"""define void {symbol}(i64 %context) {{
entry:
  %status = call i32 @pecos_wasm_v1_release(i64 %context)
{_check('status')}  ret void
}}
"""
    if name == "call":
        in_types = _scalar_types(args[0])
        parameters = "".join(f", {ty} %arg{i}" for i, ty in enumerate(in_types))
        packing = f"  %args = alloca [{inputs} x i64]\n"
        for i, ty in enumerate(in_types):
            value = f"%arg{i}"
            if ty == "double":
                packing += f"  %bits{i} = bitcast double {value} to i64\n"
                value = f"%bits{i}"
            packing += (
                f"  %ptr{i} = getelementptr [{inputs} x i64], ptr %args, i64 0, i64 {i}\n"
                f"  store i64 {value}, ptr %ptr{i}\n"
            )
        target = (
            f"@pecos_wasm_v2_call(i64 %context, i64 %function, ptr %args, i64 {inputs}, i64 {wasm_async_workers})"
            if wasm_async_workers
            else f"@pecos_wasm_v1_call(i64 %context, i64 %function, ptr %args, i64 {inputs})"
        )
        return f"""define i64 {symbol}(i64 %context, i64 %function{parameters}) {{
entry:
{packing}  %status = call i32 {target}
{_check('status')}  ret i64 %context
}}
"""
    value_type = _scalar_types(args[0])[0] if outputs else None
    result_type = f"{{ i64, {value_type} }}" if outputs else "i64"
    result = "  ret i64 %context\n"
    if outputs:
        result = "  %bits = load i64, ptr %out\n"
        value = "%bits"
        if value_type == "double":
            result += "  %value = bitcast i64 %bits to double\n"
            value = "%value"
        result += (
            f"  %pair = insertvalue {result_type} poison, i64 %context, 0\n"
            f"  %result = insertvalue {result_type} %pair, {value_type} {value}, 1\n"
            f"  ret {result_type} %result\n"
        )
    return f"""define {result_type} {symbol}(i64 %context) {{
entry:
  %out = alloca i64
  %status = call i32 @pecos_wasm_v1_read(i64 %context, ptr %out, i64 {outputs})
{_check('status')}{result}}}
"""


def _describe_value(value: object) -> str:
    """Return a short repr of a rejected option for an error message; never raise.

    reprlib bounds the length. Formatting can still fail (an int longer than
    ``sys.get_int_max_str_digits()`` before Python 3.14, or a hostile object), and that
    must not replace the validation error, so any failure yields a fixed placeholder.
    """
    try:
        return reprlib.repr(value)
    except Exception:  # noqa: BLE001 - a diagnostic must never replace the error it reports
        return "<unprintable value>"


def compile_wasm(
    program: bytes | Package | GuppyFunctionDefinition,
    modules: Mapping[str, bytes],
    *,
    platform: str = "helios",
    opt_level: int = 2,
    emit_debug: bool = False,
    wasm_int_width: int | str = 64,
    wasm_timeout_seconds: float = 1.0,
    wasm_auto_init: bool = True,
    wasm_state_scope: str = "shot",
    wasm_async_workers: int = 0,
) -> str:
    """Compile with the public HUGR/QIS compiler and scalar WASM ABI.

    Guppy functions use the safe minimal-optimization frontend automatically.
    Precompiled input should come directly from ``compile_guppy`` without
    additional optimization.

    ``modules`` maps exact module identifiers to caller-supplied WAT or WASM
    bytes. ``wasm_int_width`` selects exact native integer signatures (64 by
    default; use 32 for Guppy H2 declarations). i32 arguments wrap to their low
    32 bits and results are sign-extended. Use "auto" to infer each integer lane
    from its actual WASM export signature, allowing mixed i32/i64 exports.
    ``wasm_timeout_seconds`` is a positive finite per-start/init/call limit,
    defaulting to one second, with approximately 10 ms clock resolution.
    The result requires the Selene WASM utility or WASM-enabled QIS FFI;
    nondefault timeouts require native ``pecos_wasm_v2_register`` support.
    ``wasm_state_scope="worker"`` retains mutable instances across clean shots
    on one execution thread. Custom hosts own that thread/session lifetime.
    ``wasm_auto_init=False`` skips optional exported init (not WASM start),
    requiring ``pecos_wasm_v3_register``. The default preserves automatic init.
    """
    if type(wasm_auto_init) is not bool:
        msg = f"wasm_auto_init must be True or False; got {_describe_value(wasm_auto_init)}"
        raise ValueError(msg)
    try:
        valid_timeout = (
            type(wasm_timeout_seconds) in (int, float)
            and math.isfinite(wasm_timeout_seconds)
            and wasm_timeout_seconds > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        msg = (
            "wasm_timeout_seconds must be a positive finite number of seconds; "
            f"got {_describe_value(wasm_timeout_seconds)}"
        )
        raise ValueError(msg)
    wasm_timeout_seconds = float(wasm_timeout_seconds)
    if not (
        (type(wasm_int_width) is int and wasm_int_width in (32, 64))
        or (type(wasm_int_width) is str and wasm_int_width == "auto")
    ):
        msg = f"wasm_int_width must be 32 or 64, or 'auto'; got {_describe_value(wasm_int_width)}"
        raise ValueError(msg)
    if type(wasm_state_scope) is not str or wasm_state_scope not in ("shot", "worker"):
        msg = f"wasm_state_scope must be 'shot' or 'worker'; got {_describe_value(wasm_state_scope)}"
        raise ValueError(msg)
    if type(wasm_async_workers) is not int or not 0 <= wasm_async_workers <= 64:
        msg = f"wasm_async_workers must be an integer between 0 and 64; got {_describe_value(wasm_async_workers)}"
        raise ValueError(msg)
    if struct.calcsize("P") != 8:
        msg = "Public WASM ABI v1 requires a 64-bit host"
        raise ValueError(msg)
    from_source = hasattr(program, "with_minimal_opt")
    if from_source:
        program = _compile_guppy_package(program)
    else:
        # Caller-built graphs may have sparse node IDs or unresolved versions.
        # Envelope decoding normalizes both; it is more than a defensive copy.
        program = Package.from_bytes(program if isinstance(program, bytes) else program.to_bytes())
    if len(program.modules) != 1:
        msg = "Public WASM lowering currently requires one HUGR module"
        raise ValueError(msg)
    graph = program.modules[0]
    if from_source and any(
        node.idx != index
        or ((parent := graph[node].parent) is not None and parent.idx >= index)
        or [child.idx for child in graph[node].children] != sorted(child.idx for child in graph[node].children)
        for index, node in enumerate(graph)
    ):
        # Custom compilers may delete/reuse nodes or reorder siblings. Decoding
        # restores dense parent-before-child numbering and child-list order.
        program = Package.from_bytes(program.to_bytes())
        graph = program.modules[0]
    # HUGR 0.18 exposes graph editing through its versioned serialization model.
    serialized = graph._to_serial()  # noqa: SLF001
    serial_type = type(serialized)
    document = serialized.model_dump(mode="json")
    if from_source and _has_unresolved_versions(document):
        program = Package.from_bytes(program.to_bytes())
        serialized = program.modules[0]._to_serial()  # noqa: SLF001
        document = serialized.model_dump(mode="json")
    # model_dump creates the independent document edited below, so built-in
    # Guppy output needs no envelope round trip solely to copy it.
    if not from_source:
        _check_external_frontend(document)
    operations = []
    module_ids = {}
    for index, node in enumerate(document["nodes"]):
        if node.get("op") in {"FuncDefn", "FuncDecl"} and node.get("name", "").startswith(_PREFIX):
            msg = "Program uses a reserved WASM adapter name"
            raise ValueError(msg)
        if node.get("op") == "Extension" and node.get("extension") == "tket.wasm":
            _validate(node)
            operations.append((index, copy.deepcopy(node)))
        if node.get("op") == "Const" and node["v"].get("value", {}).get("c") == "ConstWasmModule":
            identifier = node["v"]["value"]["v"]["module_filename"]
            if identifier not in modules:
                msg = f"Supply bytes for WASM module {identifier!r}"
                raise ValueError(msg)
            if not isinstance(modules[identifier], bytes):
                msg = "WASM modules must be explicit bytes, not paths"
                raise TypeError(msg)
            module_id = module_ids.setdefault(identifier, len(module_ids))
            node["v"] = {
                "v": "Extension",
                "typ": _I64,
                "value": {"c": "ConstInt", "v": {"log_width": 6, "value": module_id}},
            }
    if not operations:
        msg = "Program contains no tket.wasm operations"
        raise ValueError(msg)
    document = _replace_types(document)
    adapters = {}
    original_envelope = program.to_bytes()
    digest = hashlib.sha256(original_envelope).hexdigest()[:16]
    for index, operation in operations:
        signature = document["nodes"][index]["signature"]
        poly = {"params": [], "body": signature}
        name = f"{_PREFIX}{digest}_{index}"
        declaration = len(document["nodes"])
        document["nodes"].append(
            {"op": "FuncDecl", "parent": 0, "name": name, "signature": poly, "visibility": "Public"},
        )
        if document.get("metadata") is not None:
            document["metadata"].append(None)
        document["nodes"][index] = {
            "op": "Call",
            "parent": operation["parent"],
            "func_sig": poly,
            "type_args": [],
            "instantiation": signature,
        }
        document["edges"].append([[declaration, 0], [index, len(signature["input"])]])
        adapters[name] = operation
    _order_calls(document, {index for index, operation in operations if operation["name"] == "get_context"})
    _protect_external_calls(document)
    lowered = Hugr._from_serial(serial_type.model_validate(document))  # noqa: SLF001
    envelope = Package(modules=[lowered], extensions=program.extensions).to_bytes()
    llvm = compile_to_llvm_ir(envelope, platform=platform, opt_level=opt_level, emit_debug=emit_debug)
    declarations = re.compile(r"^declare [^\n]*? (@[^\s(]+)\([^\n]*$", re.MULTILINE)
    remaining = set(adapters)

    def replace(match: re.Match) -> str:
        symbol = match[1]
        for name, operation in adapters.items():
            if re.fullmatch(r'@"?__hugr__\.' + re.escape(name) + r'(?:\.\d+)?"?', symbol):
                remaining.discard(name)
                return _adapter(symbol, operation, name + "_name", wasm_int_width, wasm_async_workers)
        return match[0]

    llvm = declarations.sub(replace, llvm)
    # HUGR optimization may remove provably unreachable branches and their
    # declarations. Only adapters still mentioned in emitted LLVM need bodies.
    unresolved = {name for name in remaining if name in llvm}
    if unresolved:
        msg = f"Public compiler did not emit expected WASM adapters: {sorted(unresolved)}"
        raise ValueError(msg)
    if not re.search(r"^declare void @panic\(i32, ptr\)", llvm, re.MULTILINE):
        llvm += "\ndeclare void @panic(i32, ptr)\n"
    llvm += _ABI
    if wasm_async_workers:
        llvm += "\ndeclare i32 @pecos_wasm_v2_call(i64, i64, ptr, i64, i64)\n"
    if wasm_int_width == "auto":
        llvm += "\ndeclare i64 @pecos_wasm_v3_lookup(i64, ptr, i64, ptr, i64, ptr, i64)\n"
        llvm += "declare i64 @pecos_wasm_v3_lookup_by_id(i64, i64, ptr, i64, ptr, i64)\n"
    if not wasm_auto_init:
        llvm += "\ndeclare i32 @pecos_wasm_v3_register(i64, ptr, i64, double, i32)\n"
    elif wasm_timeout_seconds != 1.0:
        llvm += "\ndeclare i32 @pecos_wasm_v2_register(i64, ptr, i64, double)\n"
    if wasm_state_scope == "worker":
        # Length-prefix every field. Include module order because numeric module
        # identifiers address retained instances; a program hash is not a batch ID.
        key = hashlib.sha256()
        fields = [
            original_envelope,
            repr(
                (
                    platform,
                    opt_level,
                    emit_debug,
                    wasm_int_width,
                    wasm_timeout_seconds,
                    wasm_auto_init,
                    wasm_async_workers,
                ),
            ).encode(),
        ]
        for identifier in module_ids:
            fields.extend((identifier.encode(), modules[identifier]))
        for field in fields:
            key.update(struct.pack(">Q", len(field)))
            key.update(field)
        key_name = f"pecos_wasm_worker_key_{digest}"
        llvm += _constant(key_name, key.digest())
        llvm += "\ndeclare i32 @pecos_wasm_v2_begin_worker_shot(ptr)\n"
        llvm += "declare i32 @pecos_wasm_v2_finish_worker_shot()\n"
        setup = f"  %wasm_reset = call i32 @pecos_wasm_v2_begin_worker_shot(ptr @{key_name})\n"
        setup += _check("wasm_reset")
    else:
        setup = "  %wasm_reset = call i32 @pecos_wasm_v1_reset()\n" + _check("wasm_reset")
    for identifier, module_id in module_ids.items():
        data = modules[identifier]
        global_name = f"pecos_wasm_module_{digest}_{module_id}"
        llvm += _constant(global_name, data)
        status = f"wasm_module_{module_id}"
        if not wasm_auto_init:
            timeout_literal = "0x" + struct.pack(">d", wasm_timeout_seconds).hex().upper()
            setup += (
                f"  %{status} = call i32 @pecos_wasm_v3_register(i64 {module_id}, "
                f"ptr @{global_name}, i64 {len(data)}, double {timeout_literal}, i32 1)\n"
            )
        elif wasm_timeout_seconds == 1.0:
            setup += (
                f"  %{status} = call i32 @pecos_wasm_v1_register(i64 {module_id}, "
                f"ptr @{global_name}, i64 {len(data)})\n"
            )
        else:
            timeout_literal = "0x" + struct.pack(">d", wasm_timeout_seconds).hex().upper()
            setup += (
                f"  %{status} = call i32 @pecos_wasm_v2_register(i64 {module_id}, "
                f"ptr @{global_name}, i64 {len(data)}, double {timeout_literal})\n"
            )
        setup += _check(status)
    entry = re.compile(r"(define [^\n]* @qmain\([^\n]*\{\n[^\n]*:\n)")
    llvm, count = entry.subn(lambda m: m[0] + setup, llvm)
    if count != 1:
        msg = "Expected one public QIS qmain entry point"
        raise ValueError(msg)
    qmain = re.compile(r"(define [^\n]* @qmain\([^\n]*\{\n)(.*?)(^})", re.MULTILINE | re.DOTALL)

    def finish_body(match: re.Match) -> str:
        return_number = 0

        def finish_return(ret: re.Match) -> str:
            nonlocal return_number
            if wasm_state_scope == "shot":
                return "  call i32 @pecos_wasm_v1_reset()\n" + ret[0]
            status = f"wasm_finish_{return_number}"
            return_number += 1
            return f"  %{status} = call i32 @pecos_wasm_v2_finish_worker_shot()\n" + _check(status) + ret[0]

        return match[1] + re.sub(r"(?m)^(\s*ret )", finish_return, match[2]) + match[3]

    return qmain.sub(finish_body, llvm)
