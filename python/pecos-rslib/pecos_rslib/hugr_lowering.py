"""Lower HUGR with Selene and normalize PECOS runtime helper symbols.

Install ``pecos-rslib[hugr]`` to enable compilation. The base bindings wheel
remains usable without the optional ``selene-hugr-qis-compiler`` package.
Importing this module loads neither that compiler nor quantum-pecos; the
compiler is imported lazily when compile_hugr_to_qis is called.
"""

import re
from dataclasses import dataclass

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
# Identifiers are single tokens, so keywords in names cannot become headers.
_LEXER = re.compile(
    r'[@%!]"(?:[^"\\]|\\.)*"|"(?:[^"\\]|\\.)*"|;[^\n]*|' r"[@%$!#]?[a-zA-Z0-9_.$-]+|[^\s]",
)
_MODULE_KEYWORDS = {
    "declare",
    "define",
    "attributes",
    "source_filename",
    "target",
    "module",
    "uselistorder",
    "uselistorder_bb",
}
_INSTRUCTIONS = set(
    "ret br switch indirectbr invoke callbr resume catchswitch catchret cleanupret unreachable "
    "fneg add fadd sub fsub mul fmul udiv sdiv fdiv urem srem frem shl lshr ashr and or xor "
    "extractelement insertelement shufflevector extractvalue insertvalue alloca load store fence "
    "cmpxchg atomicrmw getelementptr trunc zext sext fptrunc fpext fptoui fptosi uitofp sitofp "
    "ptrtoint inttoptr bitcast addrspacecast icmp fcmp phi select freeze call va_arg landingpad "
    "catchpad cleanuppad".split(),
)


def _symbol(lexeme: str) -> str:
    """Decode LLVM hexadecimal byte escapes before classifying identifiers."""
    if lexeme.startswith('@"'):
        return re.sub(r"\\([0-9a-fA-F]{2})", lambda match: chr(int(match[1], 16)), lexeme[2:-1])
    return lexeme[1:]


def _rename(match: re.Match) -> str:
    lexeme = match.group()
    symbol = _symbol(lexeme)
    if lexeme.startswith('@"') and symbol in PECOS_HELPER_ABIS:
        return "@" + symbol
    if not lexeme.startswith("@") or not symbol.startswith("__hugr__."):
        return lexeme
    parts = symbol.removeprefix("__hugr__.").split(".")
    if parts[-1].isascii() and parts[-1].isdigit():
        parts.pop()
    if parts and parts[-1] in PECOS_HELPER_ABIS:
        return "@" + parts[-1]
    return lexeme


def _mask_literal(match: re.Match) -> str:
    """Hide strings/comments when recognizing declarations, preserving offsets."""
    lexeme = match.group()
    if lexeme.startswith("@"):
        return lexeme
    return "".join("\n" if char == "\n" else " " for char in lexeme)


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
    """Read the full type before attributes/names, including typed pointers."""
    tokens = list(_LEXER.finditer(parameter))
    if not tokens:
        return ""
    index = 1
    if tokens[0].group() in {"[", "{", "<"}:
        depth = 1
        while index < len(tokens) and depth:
            lexeme = tokens[index].group()
            if lexeme in {"[", "{", "<"}:
                depth += 1
            elif lexeme in {"]", "}", ">"}:
                depth -= 1
            index += 1
    while index < len(tokens):
        lexeme = tokens[index].group()
        if lexeme == "*":
            index += 1
        elif lexeme == "(":
            index = _matching_paren(tokens, index) + 1
        elif lexeme == "addrspace" and index + 1 < len(tokens) and tokens[index + 1].group() == "(":
            index = _matching_paren(tokens, index + 1) + 1
        else:
            break
    return parameter[: tokens[index - 1].end()]


def _return_type(prefix: str) -> str:
    """Find the complete return type after linkage and return attributes."""
    for lexeme in _LEXER.finditer(prefix):
        candidate = prefix[lexeme.start() :]
        if _parameter_type(candidate) == candidate:
            return candidate
    return prefix


@dataclass
class _Function:
    keyword: int
    name: int
    parameters_end: int
    end: int
    body: tuple[int, int] | None


def _module_start(tokens: list[re.Match], index: int) -> bool:
    lexeme = tokens[index].group()
    if lexeme == "target":
        return index + 1 < len(tokens) and tokens[index + 1].group() in {"triple", "datalayout"}
    return lexeme in _MODULE_KEYWORDS or (
        lexeme.startswith(("@", "%", "$", "!")) and index + 1 < len(tokens) and tokens[index + 1].group() == "="
    )


def _matching_paren(tokens: list[re.Match], start: int) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        lexeme = tokens[index].group()
        if lexeme == "(":
            depth += 1
        elif lexeme == ")":
            depth -= 1
            if depth == 0:
                return index
    msg = f"Unterminated function parameter list at offset {tokens[start].start()}"
    raise ValueError(msg)


def _entity_end(tokens: list[re.Match], start: int) -> tuple[int, tuple[int, int] | None]:
    """Find the next module entity, balancing function bodies and aggregate data.

    A definition's final outer brace pair is its body. Earlier brace pairs can
    belong to aggregate prefix/prologue constants in the function header.
    """
    depth = 0
    body_start = None
    body = None
    index = start
    while index < len(tokens):
        lexeme = tokens[index].group()
        if depth == 0 and _module_start(tokens, index):
            break
        if lexeme in ("(", "[", "{", "<"):
            if lexeme == "{" and depth == 0:
                body_start = index
            depth += 1
        elif lexeme in (")", "]", "}", ">"):
            depth -= 1
            if lexeme == "}" and depth == 0 and body_start is not None:
                body = body_start, index
        index += 1
    return index, body


def _functions(tokens: list[re.Match]) -> list[_Function]:
    functions = []
    index = 0
    while index < len(tokens):
        keyword = index
        if tokens[index].group() not in {"declare", "define"}:
            index += 1
            continue
        index += 1
        while index < len(tokens) and not tokens[index].group().startswith("@"):
            if _module_start(tokens, index):
                break
            index += 1
        if index + 1 >= len(tokens) or not tokens[index].group().startswith("@") or tokens[index + 1].group() != "(":
            continue
        name = index
        parameters_end = _matching_paren(tokens, name + 1)
        index, body = _entity_end(tokens, parameters_end + 1)
        functions.append(_Function(keyword, name, parameters_end, index, body))
    return functions


def _signature(ir: str, tokens: list[re.Match], function: _Function) -> str:
    """Return the LLVM type, excluding declaration modifiers and SSA names."""
    prefix = ir[tokens[function.keyword].end() : tokens[function.name].start()]
    parameters = ir[tokens[function.name + 1].end() : tokens[function.parameters_end].start()]
    prefix = _TOKEN.sub(lambda match: " " if match.group().startswith(";") else match.group(), prefix).strip()
    parameters = _TOKEN.sub(lambda match: " " if match.group().startswith(";") else match.group(), parameters)
    types = [_parameter_type(parameter) for parameter in _split_parameters(parameters)]
    return f"{_return_type(prefix)} ({', '.join(types)})"


def _instruction_uses(tokens: list[re.Match], functions: list[_Function]) -> set[int]:
    """Recognize global operands in instructions, excluding function headers."""
    uses = set()
    for function in functions:
        if tokens[function.keyword].group() != "define" or function.body is None:
            continue
        start, end = function.body
        instruction = False
        for index in range(start + 1, end):
            lexeme = tokens[index].group()
            if index + 1 < end and tokens[index + 1].group() in {":", "="}:
                instruction = False
            elif lexeme in _INSTRUCTIONS:
                instruction = True
            elif lexeme.startswith("@") and instruction:
                uses.add(index)
    return uses


def _remove_declarations(ir: str, spans: list[tuple[int, int]]) -> str:
    output = []
    previous = 0
    for start, end in spans:
        output.append(ir[previous:start])
        # Preserve comments, including the newline that keeps their following
        # tokens out of the comment. Trailing comments are outside this span.
        output.extend(match.group() + "\n" for match in _TOKEN.finditer(ir[start:end]) if match.group().startswith(";"))
        previous = end
    output.append(ir[previous:])
    return "".join(output)


def normalize_pecos_helper_symbols(ir: str) -> str:
    """Rename helper tokens, merge equivalent declarations, and validate ABIs.

    A private helper name must be the last dotted component before an optional
    purely numeric suffix. A bare ``__hugr__.<helper>`` is also renamed. Quoted
    identifiers are hex-decoded before matching and emitted as bare public names.
    Surviving private symbols containing a helper component are rejected.

    Headers and parameter lists may cross lines or share a line. Every public
    helper lexeme must be a validated declaration or an instruction operand.

    Raises:
        ValueError: For a definition, conflicting/incorrect ABI, unsupported
            placement, or unrecognized private helper mangling.
    """
    renamed = _TOKEN.sub(_rename, ir)
    tokens = [match for match in _LEXER.finditer(renamed) if not match.group().startswith(";")]
    functions = _functions(tokens)
    declarations = {}
    declared_names = set()
    duplicates = []
    for function in functions:
        helper = _symbol(tokens[function.name].group())
        if helper not in PECOS_HELPER_ABIS:
            continue
        if tokens[function.keyword].group() == "define":
            line = renamed.count("\n", 0, tokens[function.name].start()) + 1
            msg = f"PECOS helper '{helper}' must be declared, not defined (line {line})"
            raise ValueError(msg)
        signature = _signature(renamed, tokens, function)
        declared_names.add(function.name)
        if helper in declarations:
            previous = declarations[helper]
            if _whitespace(previous) != _whitespace(signature):
                msg = f"PECOS helper '{helper}' has conflicting signatures: '{previous}' and '{signature}'"
                raise ValueError(msg)
            duplicates.append((tokens[function.keyword].start(), tokens[function.end - 1].end()))
        else:
            declarations[helper] = signature
    # Compare duplicates first so collisions report both signatures even if one
    # of them is also incompatible with the runtime ABI.
    for helper, actual in declarations.items():
        expected = PECOS_HELPER_ABIS[helper]
        if _whitespace(actual) != _whitespace(expected):
            msg = f"PECOS helper '{helper}' expects '{expected}', actual signature is '{actual}'"
            raise ValueError(msg)
    uses = _instruction_uses(tokens, functions)
    for index, lexeme in enumerate(tokens):
        if not lexeme.group().startswith("@"):
            continue
        symbol = _symbol(lexeme.group())
        if symbol.startswith("__hugr__.") and PECOS_HELPER_ABIS.keys() & set(symbol.split(".")):
            line = renamed.count("\n", 0, lexeme.start()) + 1
            msg = f"Unrecognized PECOS helper symbol '@{symbol}' on line {line}"
            raise ValueError(msg)
        if symbol in PECOS_HELPER_ABIS and index not in declared_names and index not in uses:
            line = renamed.count("\n", 0, lexeme.start()) + 1
            msg = f"PECOS helper '{symbol}' has unsupported placement on line {line}"
            raise ValueError(msg)
    return _remove_declarations(renamed, duplicates)


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
        ValueError: If compiler options are invalid or helper validation fails.

    Compiler exceptions, including HugrReadError and TypeError, propagate unchanged.
    """
    if type(opt_level) is not int or opt_level not in (0, 1, 2, 3):
        msg = f"opt_level must be one of 0, 1, 2, 3; got {opt_level!r}"
        raise ValueError(msg)
    if platform not in ("helios", "sol"):
        msg = f"platform must be 'helios' or 'sol'; got {platform!r}"
        raise ValueError(msg)
    try:
        from selene_hugr_qis_compiler import compile_to_llvm_ir
    except ImportError as error:
        msg = "HUGR -> QIS lowering requires the selene-hugr-qis-compiler package"
        raise ImportError(msg) from error

    return normalize_pecos_helper_symbols(
        compile_to_llvm_ir(hugr_bytes, platform=platform, opt_level=opt_level, emit_debug=emit_debug),
    )
