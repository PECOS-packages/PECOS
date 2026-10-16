"""Certify result provenance and measurement order in the executed LLVM program.

Only blocks executed exactly once on every returning path may allocate results
or print tags. Their dominators give an order independent of LLVM block layout.
The reader deliberately accepts a limited LLVM subset, without a runtime LLVM
dependency; token and signature helpers are shared with HUGR lowering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pecos_rslib.hugr_lowering import (
    _LEXER,
    _functions,
    _parameter_type,
    _return_type,
    _split_parameters,
    _symbol,
)

_MEASUREMENTS = {"___lazy_measure", "___lazy_measure_leaked"}
_ALLOCATORS = {"__quantum__rt__result_allocate"}
# ffi.rs queues measurements through m__body; mz__body delegates to it.
# These use caller-selected result slots rather than the lazy ordinal stream.
_UNSUPPORTED_MEASUREMENTS = {"__quantum__qis__m__body", "__quantum__qis__mz__body"}
# All named_result_exports! variants in crates/pecos-qis-ffi/src/ffi.rs.
_PRINTS = {
    f"print_{kind}{suffix}": (prefix, scalar_type)
    for kind, prefix, scalar_type in (
        ("bool", "BOOL", "i1"),
        ("int", "INT", "i64"),
        ("uint", "UINT", "i64"),
        ("float", "FLOAT", "double"),
    )
    for suffix in ("", "_arr", "_selene", "_arr_selene")
}
_RELEVANT = _MEASUREMENTS | _ALLOCATORS | _UNSUPPORTED_MEASUREMENTS | _PRINTS.keys()
_RUNTIME_CALLS = _RELEVANT | {"___read_future_bool"}
_SUPPORTED_TERMINATORS = {"ret", "br", "switch", "unreachable"}
_TERMINATORS = _SUPPORTED_TERMINATORS | {
    "invoke",
    "callbr",
    "indirectbr",
    "resume",
    "catchswitch",
    "cleanupret",
    "catchret",
}


class _CertificationError(ValueError):
    """An unsupported lowered program, distinct from compiler/runtime failures."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Cannot certify measurement order from the lowered program: {reason}")


def _tokens(text: str) -> list[str]:
    return [match.group() for match in _LEXER.finditer(text) if not match.group().startswith(";")]


def _bytes(literal: str) -> bytes:
    """Decode LLVM strings as bytes, never as Python/unicode escape sequences."""
    result = bytearray()
    index = 0
    while index < len(literal):
        if literal[index] == "\\":
            if literal[index + 1 : index + 2] == "\\":
                result.append(0x5C)
                index += 2
                continue
            escape = literal[index + 1 : index + 3]
            if not re.fullmatch(r"[0-9a-fA-F]{2}", escape):
                msg = "invalid LLVM string escape"
                raise _CertificationError(msg)
            result.append(int(escape, 16))
            index += 3
        else:
            result.extend(literal[index].encode("utf-8"))
            index += 1
    return bytes(result)


@dataclass
class _Call:
    target: str
    result: str | None
    return_type: str
    args: list[str]

    def check(self, return_type: str, arg_types: list[str]) -> None:
        try:
            actual_types = [_parameter_type(arg) for arg in self.args]
        except ValueError as error:
            raise _CertificationError(str(error)) from error
        if self.return_type != return_type or actual_types != arg_types:
            msg = f"unsupported signature for @{self.target}"
            raise _CertificationError(msg)
        if any(len(_tokens(arg)) < 2 for arg in self.args):
            msg = f"missing operand for @{self.target}"
            raise _CertificationError(msg)


def _instruction_start(tokens: list[str]) -> int:
    start = 2 if len(tokens) > 1 and tokens[1] == "=" else 0
    if start >= len(tokens):
        msg = "incomplete instruction"
        raise _CertificationError(msg)
    return start


def _validate_opcode_position(instruction: str) -> None:
    """Reject hidden calls and terminators in the reader's one-instruction-per-line subset."""
    tokens = _tokens(instruction)
    opcode = _instruction_start(tokens)
    if tokens[opcode] in {"tail", "musttail", "notail"}:
        opcode += 1
    depth = 0
    for index, lexeme in enumerate(tokens):
        if (lexeme == "call" or lexeme in _TERMINATORS) and depth == 0 and index != opcode:
            msg = f"{lexeme} outside instruction opcode"
            raise _CertificationError(msg)
        depth += (lexeme in {"(", "[", "{", "<"}) - (lexeme in {")", "]", "}", ">"})


def _call(instruction: str) -> _Call | None:
    tokens = _tokens(instruction)
    start = _instruction_start(tokens)
    if tokens[start] in {"tail", "musttail", "notail"}:
        start += 1
        if start >= len(tokens):
            msg = "incomplete tail call"
            raise _CertificationError(msg)
    if tokens[start] != "call":
        return None
    # Skip attribute arguments and explicit function types. The callee itself
    # must be a global at nesting depth zero, immediately before its arguments.
    depth = 0
    opening = None
    for index in range(start + 1, len(tokens) - 1):
        lexeme = tokens[index]
        if depth == 0 and lexeme.startswith(("@", "%")) and tokens[index + 1] == "(":
            opening = index + 1
        depth += (lexeme in {"(", "[", "{", "<"}) - (lexeme in {")", "]", "}", ">"})
    if opening is None or not tokens[opening - 1].startswith("@"):
        msg = "indirect call"
        raise _CertificationError(msg)
    target = tokens[opening - 1]
    depth = 1
    closing = opening + 1
    while closing < len(tokens) and depth:
        depth += (tokens[closing] == "(") - (tokens[closing] == ")")
        closing += 1
    if depth:
        msg = "unterminated call"
        raise _CertificationError(msg)
    try:
        return_type = _return_type(" ".join(tokens[start + 1 : opening - 1])).partition(" (")[0]
    except ValueError as error:
        raise _CertificationError(str(error)) from error
    return _Call(
        _symbol(target),
        tokens[0] if start >= 2 and tokens[1] == "=" else None,
        return_type,
        _split_parameters(" ".join(tokens[opening + 1 : closing - 1])),
    )


def _blocks(body: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: list[str] = []
    label = "0"  # LLVM's implicit entry block, when no explicit label is emitted.
    pending = ""
    for line in body.splitlines():
        tokens = _tokens(line)
        if not tokens:
            continue
        if len(tokens) >= 2 and tokens[1] == ":":
            if pending:
                msg = "unterminated instruction before block label"
                raise _CertificationError(msg)
            if current:
                blocks[label] = current
            label, current = tokens[0].strip('"'), []
            if label in blocks:
                msg = f"duplicate block %{label}"
                raise _CertificationError(msg)
            blocks[label] = current
            tokens = tokens[2:]
            if not tokens:
                continue
        pending += " " + " ".join(tokens)
        # switch tables (and aggregate instructions) may span several lines.
        parts = _tokens(pending)
        if sum(parts.count(left) - parts.count(right) for left, right in (("[", "]"), ("(", ")"), ("{", "}"))):
            continue
        current.append(pending.strip())
        pending = ""
    if pending:
        msg = "unterminated instruction"
        raise _CertificationError(msg)
    if current:
        blocks[label] = current
    if not blocks:
        msg = "empty function body"
        raise _CertificationError(msg)
    # Validate block boundaries before CFG traversal, including dead blocks.
    # Unsupported terminators are structurally valid here; reachable ones are
    # rejected by _successors when their control flow would need analysis.
    for block, instructions in blocks.items():
        for instruction in instructions:
            _validate_opcode_position(instruction)
        for instruction in instructions[:-1]:
            opcode = _opcode(instruction)
            if opcode in _TERMINATORS:
                msg = f"terminator {opcode} is not last in block %{block}"
                raise _CertificationError(msg)
        if not instructions or _opcode(instructions[-1]) not in _TERMINATORS:
            msg = f"block %{block} has no terminator"
            raise _CertificationError(msg)
    return blocks


def _opcode(instruction: str) -> str:
    tokens = _tokens(instruction)
    return tokens[_instruction_start(tokens)]


def _successors(instruction: str) -> list[str]:
    opcode = _opcode(instruction)
    if opcode not in _SUPPORTED_TERMINATORS:
        msg = f"unsupported terminator {opcode}"
        raise _CertificationError(msg)
    if opcode in {"ret", "unreachable"}:
        return []
    tokens = _tokens(instruction)
    successors = [tokens[index + 1][1:].strip('"') for index, lexeme in enumerate(tokens[:-1]) if lexeme == "label"]
    if not successors or (opcode == "br" and len(successors) not in {1, 2}):
        msg = f"unsupported {opcode} targets"
        raise _CertificationError(msg)
    return successors


def _postorder(graph: dict[str, list[str]], entry: str) -> list[str]:
    """Finish-order DFS without consuming the Python call stack."""
    seen: set[str] = set()
    order: list[str] = []
    pending = [(entry, False)]
    while pending:
        block, finished = pending.pop()
        if finished:
            order.append(block)
        elif block not in seen:
            seen.add(block)
            pending.append((block, True))
            pending.extend((successor, False) for successor in reversed(graph[block]) if successor not in seen)
    return order


def _cycle_blocks(
    graph: dict[str, list[str]],
    predecessors: dict[str, list[str]],
    postorder: list[str],
) -> set[str]:
    """Kosaraju's second pass: find SCCs in the reversed CFG, iteratively."""
    seen: set[str] = set()
    cyclic: set[str] = set()
    for root in reversed(postorder):
        if root in seen:
            continue
        seen.add(root)
        component = []
        pending = [root]
        while pending:
            block = pending.pop()
            component.append(block)
            for predecessor in predecessors[block]:
                if predecessor not in seen:
                    seen.add(predecessor)
                    pending.append(predecessor)
        if len(component) > 1 or root in graph[root]:
            cyclic.update(component)
    return cyclic


def _common_dominator(first: str, second: str, idoms: dict[str, str], rank: dict[str, int]) -> str:
    while first != second:
        while rank[first] > rank[second]:
            first = idoms[first]
        while rank[second] > rank[first]:
            second = idoms[second]
    return first


def _immediate_dominators(predecessors: dict[str, list[str]], reverse_postorder: list[str]) -> dict[str, str]:
    # Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm" (2001),
    # Figure 3. Store only immediate dominators; intersect by walking parents.
    # https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom06.pdf
    # Our ranks use reverse postorder, so larger ranks move toward the root.
    entry = reverse_postorder[0]
    rank = {block: index for index, block in enumerate(reverse_postorder)}
    idoms = {entry: entry}
    changed = True
    while changed:
        changed = False
        for block in reverse_postorder[1:]:
            processed = [parent for parent in predecessors[block] if parent in idoms]
            parent = processed[0]  # Every non-entry block has a predecessor earlier in RPO.
            for other in processed[1:]:
                parent = _common_dominator(parent, other, idoms, rank)
            if idoms.get(block) != parent:
                idoms[block] = parent
                changed = True
    return idoms


@dataclass
class _FunctionBody:
    blocks: dict[str, list[str]]
    graph: dict[str, list[str]]
    calls: dict[str, list[_Call]]

    @classmethod
    def parse(cls, body: str) -> _FunctionBody:
        blocks = _blocks(body)
        graph: dict[str, list[str]] = {}
        calls: dict[str, list[_Call]] = {}
        pending = [next(iter(blocks))]
        while pending:
            block = pending.pop()
            if block in graph:
                continue
            if block not in blocks:
                msg = f"missing block %{block}"
                raise _CertificationError(msg)
            instructions = blocks[block]
            graph[block] = _successors(instructions[-1])
            pending.extend(graph[block])
            calls[block] = [call for instruction in instructions if (call := _call(instruction)) is not None]
        return cls(blocks, graph, calls)

    def ordered_blocks(self, relevant_functions: set[str]) -> list[str]:
        relevant = [
            block for block in self.graph if any(call.target in relevant_functions for call in self.calls[block])
        ]
        if not relevant:
            return []
        returns = {block for block in self.graph if self.blocks[block][-1].startswith("ret ")}
        if not returns:
            msg = "function has no reachable ret"
            raise _CertificationError(msg)
        entry = next(iter(self.blocks))
        predecessors: dict[str, list[str]] = {block: [] for block in self.graph}
        for block, successors in self.graph.items():
            for successor in successors:
                predecessors[successor].append(block)
        postorder = _postorder(self.graph, entry)
        cyclic = _cycle_blocks(self.graph, predecessors, postorder)
        reverse_postorder = postorder[::-1]
        idoms = _immediate_dominators(predecessors, reverse_postorder)
        depth = {entry: 0}
        for block in reverse_postorder[1:]:
            depth[block] = depth[idoms[block]] + 1
        # A block dominates all returns iff its dominator subtree contains all
        # return blocks. Accumulate once per tree edge, avoiding repeated walks
        # of shared ancestors when a function has many returning paths.
        returns_below = {block: int(block in returns) for block in self.graph}
        for block in reversed(reverse_postorder[1:]):
            returns_below[idoms[block]] += returns_below[block]
        for block in relevant:
            if block in cyclic:
                msg = f"order-relevant block %{block} is on a CFG cycle"
                raise _CertificationError(msg)
            if returns_below[block] != len(returns):
                msg = f"order-relevant block %{block} does not dominate every reachable ret"
                raise _CertificationError(msg)
        return sorted(relevant, key=depth.__getitem__)


def _tag(call: _Call, globals_: dict[str, bytes]) -> str:
    prefix, scalar_type = _PRINTS[call.target]
    array = "_arr" in call.target
    selene = call.target.endswith("_selene")
    types = ["ptr", "i64", "ptr" if array else scalar_type]
    if array and selene:
        types.append("i64")
    call.check("void", types)
    pointer = _tokens(call.args[0])
    if not pointer[-1].startswith("@"):
        msg = "result tag operand is not a module global"
        raise _CertificationError(msg)
    name = _symbol(pointer[-1])
    if name not in globals_:
        msg = f"missing tag global @{name}"
        raise _CertificationError(msg)
    length = _tokens(call.args[1])
    if len(length) != 2 or not re.fullmatch(r"[0-9]+", length[1]):
        msg = "result tag length is not a non-negative constant"
        raise _CertificationError(msg)
    try:
        size = int(length[1])
    except ValueError as error:
        msg = "result tag length exceeds the supported integer range"
        raise _CertificationError(msg) from error
    # Direct exports skip one byte; Selene exports already point at string data.
    # ffi.rs uses the explicit length, not the value of the skipped byte.
    offset = 0 if selene else 1
    data = globals_[name]
    if len(data) < offset + size:
        msg = "tag global is shorter than the runtime label length"
        raise _CertificationError(msg)
    try:
        label = data[offset : offset + size].decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "invalid UTF-8 tag label"
        raise _CertificationError(msg) from error
    for kind in (prefix, prefix + "ARR"):
        if label.startswith(f"USER:{kind}:"):
            return label[len(f"USER:{kind}:") :]
    msg = "tag label does not match USER:<TYPE>:<tag>"
    raise _CertificationError(msg)


class _Analysis:
    def __init__(self, text: str) -> None:
        tokens = [match for match in _LEXER.finditer(text) if not match.group().startswith(";")]
        try:
            functions = _functions(tokens)
        except ValueError as error:
            raise _CertificationError(str(error)) from error
        self.declarations = {_symbol(tokens[function.name].group()) for function in functions if function.body is None}
        self.bodies = {
            _symbol(tokens[function.name].group()): text[
                tokens[function.body[0]].end() : tokens[function.body[1]].start()
            ]
            for function in functions
            if function.body is not None
        }
        # Aliases and other global definitions can override runtime symbols too.
        global_definitions = {
            _symbol(token.group())
            for index, token in enumerate(tokens[:-1])
            if token.group().startswith("@") and tokens[index + 1].group() == "="
        }
        defined_runtime_calls = (self.bodies.keys() | global_definitions) & _RUNTIME_CALLS
        if defined_runtime_calls:
            msg = f"module defines runtime call @{min(defined_runtime_calls)}"
            raise _CertificationError(msg)
        self.globals = {}
        global_pattern = re.compile(
            r'(@(?:"[^"\n]*"|[\w.$-]+))\s*=\s*(?:[a-z_]+\s+)*constant\s+\[\s*\d+\s+x\s+i8\s*\]\s+c\s*"([^"\n]*)"',
        )
        for line in text.splitlines():
            # Tokenization removes comments without touching quoted bytes.
            match = global_pattern.match(" ".join(_tokens(line)))
            if match is not None:
                self.globals[_symbol(match[1])] = _bytes(match[2])
        self.functions: dict[str, _FunctionBody] = {}
        self.occurrences: dict[str, list[int | None]] = {}
        self.count = 0

    def discover(self, name: str, stack: tuple[str, ...] = ()) -> None:
        if name in stack:
            msg = f"recursive call to @{name}"
            raise _CertificationError(msg)
        if name in self.functions:
            return
        if name not in self.bodies:
            msg = f"missing definition of @{name}"
            raise _CertificationError(msg)
        function = _FunctionBody.parse(self.bodies[name])
        self.functions[name] = function
        for calls in function.calls.values():
            for call in calls:
                if call.target in self.bodies:
                    self.discover(call.target, (*stack, name))
                elif call.target not in self.declarations and call.target not in _RUNTIME_CALLS:
                    msg = f"callee @{call.target} is neither a function definition nor a declaration"
                    raise _CertificationError(msg)

    def walk(self, name: str, ordered: dict[str, list[str]]) -> None:
        function = self.functions[name]
        futures: dict[str, int] = {}
        # Reads are not order-relevant. SSA identifies their exact source even
        # when a read is in a separate measurement-free block.
        reads = {}
        for calls in function.calls.values():
            for call in calls:
                if call.target == "___read_future_bool":
                    call.check("i1", ["i64"])
                    reads[call.result] = _tokens(call.args[0])[-1]
        for block in ordered[name]:
            for call in function.calls[block]:
                if call.target in _ALLOCATORS:
                    msg = f"result-allocating call @{call.target}"
                    raise _CertificationError(msg)
                if call.target in _UNSUPPORTED_MEASUREMENTS:
                    msg = f"unsupported measurement call @{call.target}"
                    raise _CertificationError(msg)
                if call.target in _MEASUREMENTS:
                    call.check("i64", ["i64"])
                    if call.target == "___lazy_measure" and call.result is not None:
                        futures[call.result] = self.count
                    self.count += 1
                elif call.target in _PRINTS:
                    tag = _tag(call, self.globals)
                    ordinal = None
                    if call.target == "print_bool":
                        operand = _tokens(call.args[2])[-1]
                        ordinal = futures.get(reads.get(operand))
                    self.occurrences.setdefault(tag, []).append(ordinal)
                elif call.target in self.functions:
                    self.walk(call.target, ordered)


def extract_result_tag_measurements(qis: str) -> tuple[dict[str, list[int | None]], int]:
    """Return sorted tags with execution ordinals (or unsupported holes), and count.

    Raises _CertificationError (a ValueError) if the reachable program is
    outside the supported CFG/call subset. Dead code contributes no measurements
    or tags; block boundaries are validated throughout each discovered function.
    Only direct scalar ``print_bool <- ___read_future_bool <- ___lazy_measure``
    within one function carries provenance; all other prints contribute None.
    """
    analysis = _Analysis(qis)
    try:
        analysis.discover("qmain")
    except RecursionError as error:
        msg = "call graph exceeds Python's recursion limit"
        raise _CertificationError(msg) from error
    relevant = set(_RELEVANT)
    changed = True
    while changed:
        added = {
            name
            for name, function in analysis.functions.items()
            if any(call.target in relevant for calls in function.calls.values() for call in calls)
        }
        changed = not added <= relevant
        relevant.update(added)
    ordered = {name: function.ordered_blocks(relevant) for name, function in analysis.functions.items()}
    try:
        analysis.walk("qmain", ordered)
    except RecursionError as error:
        msg = "call graph exceeds Python's recursion limit"
        raise _CertificationError(msg) from error
    return dict(sorted(analysis.occurrences.items())), analysis.count
