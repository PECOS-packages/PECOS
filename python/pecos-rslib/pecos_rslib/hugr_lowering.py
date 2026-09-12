"""Lower HUGR with Selene and normalize PECOS runtime helper symbols.

Install ``pecos-rslib[hugr]`` to enable compilation. The base bindings wheel
remains usable without the optional ``selene-hugr-qis-compiler`` package.
Importing this module loads neither that compiler nor quantum-pecos; the
compiler is imported lazily when compile_hugr_to_qis is called.
"""

import re

# The Rust facade still has a matching table in pecos-hugr-qis/src/compiler.rs.
PECOS_HELPER_ABIS = {
    "pecos_qis_trace_metadata_hugr": "void (ptr, ptr)",
    "pecos_qis_trace_metadata_qubit_hugr": "i64 (i64, ptr, ptr)",
    "pecos_qis_runtime_barrier_qubit_hugr": "i64 (i64)",
    "pecos_qis_runtime_barrier_qubits2_hugr": "{ i64, i64 } (i64, i64)",
}

# Consume quoted strings and comments before considering symbol tokens. LLVM
# uses hexadecimal escapes inside strings; accepting escaped characters also
# keeps escaped quotes from prematurely ending a protected span.
_TOKEN = re.compile(r'@"(?:[^"\\]|\\.)*"|"(?:[^"\\]|\\.)*"|;[^\n]*|@[a-zA-Z0-9_.$-]+')
_HEADER = re.compile(r"^\s*(declare|define)\s+([^@\n]+)@([a-zA-Z0-9_.$-]+)\s*\(")


def _rename(match: re.Match) -> str:
    token = match.group()
    symbol = token[2:-1] if token.startswith('@"') else token[1:]
    if token.startswith('@"') and symbol in PECOS_HELPER_ABIS:
        return "@" + symbol
    if not token.startswith("@") or not symbol.startswith("__hugr__."):
        return token
    parts = symbol.removeprefix("__hugr__.").split(".")
    if parts[-1].isascii() and parts[-1].isdigit():
        parts.pop()
    if parts and parts[-1] in PECOS_HELPER_ABIS:
        return "@" + parts[-1]
    return token


def _mask_literal(match: re.Match) -> str:
    """Hide strings/comments when recognizing declarations, preserving offsets."""
    token = match.group()
    if token.startswith("@"):
        return token
    return "".join("\n" if char == "\n" else " " for char in token)


def _whitespace(text: str) -> str:
    """Canonicalize whitespace, including spacing around type punctuation."""
    return " ".join(re.findall(r'%?"(?:[^"\\]|\\.)*"|[^\s{},()<>\[\]*]+|[{},()<>\[\]*]', text))


def _split_parameters(text: str) -> list[str]:
    """Split only at commas outside aggregate types and attribute arguments."""
    parts = []
    depth = 0
    start = 0
    for index, char in enumerate(_TOKEN.sub(_mask_literal, text)):
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if text.strip():
        parts.append(text[start:].strip())
    return parts


def _parameter_type(parameter: str) -> str:
    """Read a parameter type, excluding following attributes and SSA names."""
    depth = 0
    for index, char in enumerate(parameter):
        if char in "[{<":
            depth += 1
        elif char in "]}>":
            depth -= 1
        elif char.isspace() and depth == 0:
            type_text = parameter[:index]
            # An opaque pointer's address space is part of its type, not an attribute.
            address_space = re.match(r"\s+addrspace\s*\(\s*\d+\s*\)", parameter[index:])
            if type_text == "ptr" and address_space:
                return type_text + address_space.group()
            return type_text
    return parameter


def _return_type(prefix: str) -> str:
    """Read the final type after linkage, visibility, and calling-convention words."""
    depth = 0
    start = 0
    previous_start = 0
    for index, char in enumerate(prefix):
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        elif char.isspace() and depth == 0 and index + 1 < len(prefix) and not prefix[index + 1].isspace():
            previous_start, start = start, index + 1
    if prefix[start:].startswith("addrspace") and prefix[previous_start:start].strip() == "ptr":
        start = previous_start
    return prefix[start:]


def _signature(line: str, header: re.Match) -> str:
    """Return the LLVM type, excluding declaration modifiers and SSA names."""
    masked = _TOKEN.sub(_mask_literal, line)
    start = header.end()
    depth = 1
    end = start
    while end < len(line) and depth:
        if masked[end] == "(":
            depth += 1
        elif masked[end] == ")":
            depth -= 1
        end += 1
    if depth:
        msg = f"PECOS helper '{header[3]}' has an unterminated signature: {line.strip()}"
        raise ValueError(msg)
    parameters = _split_parameters(line[start : end - 1])
    types = [_parameter_type(parameter) for parameter in parameters]
    return f"{_return_type(header[2].strip())} ({', '.join(types)})"


def normalize_pecos_helper_symbols(ir: str) -> str:
    """Rename helpers, merge equivalent declarations, and validate their ABIs.

    Raises:
        ValueError: For a helper definition, conflicting declarations, or a
            declaration that disagrees with PECOS_HELPER_ABIS.
    """
    renamed = _TOKEN.sub(_rename, ir)
    masked = _TOKEN.sub(_mask_literal, renamed)
    declarations = {}
    output = []
    for line, code in zip(renamed.split("\n"), masked.split("\n"), strict=True):
        header = _HEADER.match(code)
        if header is None or header[3] not in PECOS_HELPER_ABIS:
            output.append(line)
            continue
        helper = header[3]
        if header[1] == "define":
            msg = f"PECOS helper '{helper}' must be declared, not defined: {line.strip()}"
            raise ValueError(msg)
        signature = _signature(line, header)
        if helper in declarations:
            previous = declarations[helper]
            if _whitespace(previous) != _whitespace(signature):
                msg = f"PECOS helper '{helper}' has conflicting signatures: '{previous}' and '{signature}'"
                raise ValueError(msg)
        else:
            declarations[helper] = signature
            output.append(line)
    # Compare duplicates first so collisions report both signatures even if one
    # of them is also incompatible with the runtime ABI.
    for helper, actual in declarations.items():
        expected = PECOS_HELPER_ABIS[helper]
        if _whitespace(actual) != _whitespace(expected):
            msg = f"PECOS helper '{helper}' expects '{expected}', actual signature is '{actual}'"
            raise ValueError(msg)
    return "\n".join(output)


def compile_hugr_to_qis(
    hugr_bytes: bytes,
    *,
    platform: str = "helios",
    opt_level: int = 2,
    emit_debug: bool = False,
) -> str:
    """Lower a HUGR envelope with Selene and normalize PECOS runtime helpers.

    Args:
        hugr_bytes: HUGR package in the text or binary envelope format.
        platform: Target QSystem platform ("helios" or "sol").
        opt_level: LLVM optimization level.
        emit_debug: Whether to emit debug information.

    Returns:
        LLVM IR with PECOS's public helper symbols.

    Raises:
        ImportError: If selene-hugr-qis-compiler is not installed.
        ValueError: If a PECOS helper is defined or has conflicting/incorrect ABIs.

    Compiler exceptions, including HugrReadError and TypeError, propagate unchanged.
    """
    try:
        from selene_hugr_qis_compiler import compile_to_llvm_ir
    except ImportError as error:
        msg = "HUGR -> QIS lowering requires the selene-hugr-qis-compiler package"
        raise ImportError(msg) from error

    return normalize_pecos_helper_symbols(
        compile_to_llvm_ir(hugr_bytes, platform=platform, opt_level=opt_level, emit_debug=emit_debug),
    )
